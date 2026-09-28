/* llm_mlx_flash_decode_kernel.cpp — split-K (flash-decoding) SDPA decode per head_dim
 * che FORANO il fused SDPA di MLX (hd<=256 vettoriale / hd<=128 flash-full — gate
 * should_use_naive in libs/mlx-deps/mlx-src/.../scaled_dot_product_attention.cpp:600-640).
 * I layer FULL di gemma4 hanno head_dim=512: MLX ripiega sul percorso NAIVE che
 * MATERIALIZZA repeat_kv e i punteggi -> collo del decode (§44.1).
 *
 * v1 (2026-09-08) = kernel vettoriale single-pass: UN threadgroup per (batch, head).
 * CORRETTO (parita' verificata, maxdiff ~2e-5) ma PERDE ~21% vs il naive perche' ha
 * solo B*H=32 threadgroup: la GPU resta quasi vuota e ogni threadgroup itera
 * SERIALMENTE su visible_len.
 *
 * v2 (questa) = SPLIT-K / flash-decoding a 2 passi, lo stesso schema di mlxcel
 * (paged_attention.cpp) e di flash-decoding (Dao et al.):
 *   Pass 1 (partial): grid (H, B, S) — S split della sequenza. Ogni threadgroup calcola
 *     il parziale online-softmax (m_s, l_s, o_s) su un chunk di chiavi. Parallelismo
 *     B*H*S (es. 32*1*16 = 512 threadgroup) invece di B*H.
 *   Pass 2 (merge): grid (H, B) — combina gli S parziali:
 *     m = max_s m_s; l = sum_s l_s*exp(m_s-m); o = sum_s o_s*l_s*exp(m_s-m) / l.
 *   NIENTE materializzazione di repeat_kv ne' della matrice punteggi. La matematica e'
 *   l'online-softmax esatto (stessa forma del merge a due pezzi della KVSEG), quindi
 *   parita' col path fused che gli altri modelli usano.
 *
 * Plumbing via mlx_fast_metal_kernel (stesso meccanismo di llm_mlx_gdn_kernel.h /
 * llm_mlx_fused_rope_append.cpp), niente modifica di MLX. Adattato in spirito da MLX
 * sdpa_vector (Apple, BSD-3) e mlxcel paged-attention (Apache-2.0). */
#include "llm_mlx_flash_decode_kernel.h"

#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

constexpr const char* kHeader = R"(
#include <metal_stdlib>
using namespace metal;
)";

/* ── Pass 1: partial. Grid (H, B, S), threadgroup 32 lane (1 simdgroup).
 * Input: q,k,v + N (int32), S (int32), scale (float32). Output: pm,pl [B,H,S], po [B,H,S,D].
 * Template: InT (dtype), H, Hkv, D. */
constexpr const char* kPartial = R"(
    constexpr int L = 32;
    constexpr int DPT = D / L;
    const int lane = thread_position_in_threadgroup.x;
    const int h    = threadgroup_position_in_grid.x;
    const int b    = threadgroup_position_in_grid.y;
    const int s    = threadgroup_position_in_grid.z;
    const int gqa  = H / Hkv;
    const int kv   = h / gqa;
    const int d0   = lane * DPT;
    const int NN   = N;
    const int SS   = S;
    const float sc = scale;

    const int chunk = (NN + SS - 1) / SS;
    const int start = s * chunk;
    const int end   = min(start + chunk, NN);

    const device InT* qp = q + ((size_t)b * H + h) * D + d0;
    float qr[DPT];
    for (int j = 0; j < DPT; j++) qr[j] = (float)qp[j] * sc;

    const device InT* kp = k + (((size_t)b * Hkv + kv) * (size_t)NN + start) * D + d0;
    const device InT* vp = v + (((size_t)b * Hkv + kv) * (size_t)NN + start) * D + d0;

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float o[DPT];
    for (int j = 0; j < DPT; j++) o[j] = 0.0f;

    for (int i = start; i < end; i++) {
        float p = 0.0f;
        for (int j = 0; j < DPT; j++) p += qr[j] * (float)kp[j];
        float scv = simd_sum(p);
        float nm = scv > m ? scv : m;
        float f = fast::exp(m - nm);
        float e = fast::exp(scv - nm);
        l = l * f + e;
        m = nm;
        for (int j = 0; j < DPT; j++) o[j] = o[j] * f + e * (float)vp[j];
        kp += D;
        vp += D;
    }

    if (lane == 0) {
        pm[(b * H + h) * SS + s] = m;
        pl[(b * H + h) * SS + s] = l;
    }
    device InT* pos = po + (((size_t)b * H + h) * SS + s) * D + d0;
    for (int j = 0; j < DPT; j++) pos[j] = (InT)o[j];
)";

/* ── Pass 2: merge. Grid (H, B), threadgroup 32 lane.
 * Input: pm,pl [B,H,S], po [B,H,S,D], S (int32). Output: out [B,H,1,D].
 * Template: InT, H, D. */
constexpr const char* kMerge = R"(
    constexpr int L = 32;
    constexpr int DPT = D / L;
    const int lane = thread_position_in_threadgroup.x;
    const int h    = threadgroup_position_in_grid.x;
    const int b    = threadgroup_position_in_grid.y;
    const int d0   = lane * DPT;
    const int SS   = S;

    float m = -3.402823466e+38f;
    for (int s = 0; s < SS; s++) {
        float ms = pm[(b * H + h) * SS + s];
        if (ms > m) m = ms;
    }
    float l = 0.0f;
    float o[DPT];
    for (int j = 0; j < DPT; j++) o[j] = 0.0f;
    for (int s = 0; s < SS; s++) {
        float ms = pm[(b * H + h) * SS + s];
        float ls = pl[(b * H + h) * SS + s];
        float e = fast::exp(ms - m);
        l += ls * e;
        const device InT* pos = po + (((size_t)b * H + h) * SS + s) * D + d0;
        for (int j = 0; j < DPT; j++) o[j] += (float)pos[j] * e;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device InT* op = out + ((size_t)b * H + h) * D + d0;
    for (int j = 0; j < DPT; j++) op[j] = (InT)(o[j] * inv);
)";

/* ── Prefill (Lq>1): una query per threadgroup. Grid (Lq, H, B), threadgroup 32.
 * Input: q,k,v + N (int32 = cache_offset+Lq), COFF (int32), scale (float32).
 * Output: out [B,H,Lq,D]. Template: InT, H, Hkv, D, LQ. */
constexpr const char* kPrefill = R"(
    constexpr int L = 32;
    constexpr int DPT = D / L;
    const int lane = thread_position_in_threadgroup.x;
    const int qi = threadgroup_position_in_grid.x;
    const int h  = threadgroup_position_in_grid.y;
    const int b  = threadgroup_position_in_grid.z;
    const int gqa = H / Hkv;
    const int kv = h / gqa;
    const int d0 = lane * DPT;
    const int Nk = COFF + qi + 1;
    const float sc = scale;

    const device InT* qp = q + (((size_t)b * H + h) * LQ + qi) * D + d0;
    float qr[DPT];
    for (int j = 0; j < DPT; j++) qr[j] = (float)qp[j] * sc;

    const device InT* kp = k + (((size_t)b * Hkv + kv) * N) * D + d0;
    const device InT* vp = v + (((size_t)b * Hkv + kv) * N) * D + d0;

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float o[DPT];
    for (int j = 0; j < DPT; j++) o[j] = 0.0f;

    for (int i = 0; i < Nk; i++) {
        float p = 0.0f;
        for (int j = 0; j < DPT; j++) p += qr[j] * (float)kp[j];
        float scv = simd_sum(p);
        float nm = scv > m ? scv : m;
        float f = fast::exp(m - nm);
        float e = fast::exp(scv - nm);
        l = l * f + e;
        m = nm;
        for (int j = 0; j < DPT; j++) o[j] = o[j] * f + e * (float)vp[j];
        kp += D;
        vp += D;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device InT* op = out + (((size_t)b * H + h) * LQ + qi) * D + d0;
    for (int j = 0; j < DPT; j++) op[j] = (InT)(o[j] * inv);
)";

std::mutex g_kernel_mutex;
mlx_fast_metal_kernel g_kpartial = {nullptr};
mlx_fast_metal_kernel g_kmerge = {nullptr};
mlx_fast_metal_kernel g_kprefill = {nullptr};
int g_kernel_hd = -1;

bool ensure_kernels(int head_dim) {
    std::lock_guard<std::mutex> guard(g_kernel_mutex);
    if (g_kpartial.ctx != nullptr && g_kmerge.ctx != nullptr && g_kernel_hd == head_dim) return true;
    if (g_kpartial.ctx != nullptr) { mlx_fast_metal_kernel_free(g_kpartial); g_kpartial = {nullptr}; }
    if (g_kmerge.ctx != nullptr) { mlx_fast_metal_kernel_free(g_kmerge); g_kmerge = {nullptr}; }
    if (g_kprefill.ctx != nullptr) { mlx_fast_metal_kernel_free(g_kprefill); g_kprefill = {nullptr}; }

    const char* pin[] = {"q", "k", "v", "N", "S", "scale"};
    const char* pout[] = {"pm", "pl", "po"};
    const char* min[] = {"pm", "pl", "po", "S"};
    const char* mout[] = {"out"};
    auto mk = [&](const char* nm, const char** ins, int ni, const char** outs, int no, const char* src) {
        mlx_vector_string iv = mlx_vector_string_new(), ov = mlx_vector_string_new();
        for (int i = 0; i < ni; i++) mlx_vector_string_append_value(iv, ins[i]);
        for (int i = 0; i < no; i++) mlx_vector_string_append_value(ov, outs[i]);
        char name[64]; snprintf(name, sizeof(name), "%s_hd%d", nm, head_dim);
        mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new(
            name, iv, ov, src, kHeader, /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
        mlx_vector_string_free(iv); mlx_vector_string_free(ov);
        return k;
    };
    const char* prin[] = {"q", "k", "v", "N", "COFF", "LQ", "scale"};
    const char* prout[] = {"out"};
    g_kpartial = mk("llm_flash_decode_partial", pin, 6, pout, 3, kPartial);
    g_kmerge = mk("llm_flash_decode_merge", min, 4, mout, 1, kMerge);
    g_kprefill = mk("llm_flash_prefill", prin, 7, prout, 1, kPrefill);
    g_kernel_hd = head_dim;
    return g_kpartial.ctx != nullptr && g_kmerge.ctx != nullptr && g_kprefill.ctx != nullptr;
}

std::mutex g_reason_mutex;
const char* g_reject_reason = nullptr;
void reject_once(const char* why) {
    std::lock_guard<std::mutex> guard(g_reason_mutex);
    if (g_reject_reason != nullptr) return;
    g_reject_reason = why;
    fprintf(stderr, "[llm_flash_decode] rejected: %s\n", why);
}

void add_tmpl(mlx_fast_metal_kernel_config cfg, mlx_dtype dt, int H, int Hkv, int D) {
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", H);
    if (Hkv > 0) mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hkv", Hkv);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
}

}  // namespace

#define LLM_REJ(why) do { reject_once(why); return false; } while (0)

extern "C" bool llm_mlx_flash_decode_attention(
    mlx_array* out,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int visible_len,
    float scale,
    mlx_stream stream) {
    if (out == nullptr || q.ctx == nullptr || k.ctx == nullptr || v.ctx == nullptr)
        LLM_REJ("null handle");
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || visible_len <= 0)
        LLM_REJ("non-positive dims/len");
    if (num_heads % num_kv_heads != 0)
        LLM_REJ("num_heads not divisible by num_kv_heads (bad GQA)");
    if (head_dim % 32 != 0 || head_dim < 32)
        LLM_REJ("head_dim not a multiple of 32");

    mlx_dtype dt = mlx_array_dtype(q);
    if (dt != mlx_array_dtype(k) || dt != mlx_array_dtype(v))
        LLM_REJ("mixed q/k/v dtypes");
    if (dt != MLX_FLOAT32 && dt != MLX_BFLOAT16 && dt != MLX_FLOAT16)
        LLM_REJ("unsupported dtype");

    if (mlx_array_ndim(q) != 4 || mlx_array_ndim(k) != 4 || mlx_array_ndim(v) != 4)
        LLM_REJ("q/k/v not rank-4 [B,H,(1|N),D]");
    const int* qs = mlx_array_shape(q);
    const int* ks = mlx_array_shape(k);
    const int* vs = mlx_array_shape(v);
    if (qs == nullptr || ks == nullptr || vs == nullptr) LLM_REJ("null shape");
    if (qs[1] != num_heads || ks[1] != num_kv_heads || vs[1] != num_kv_heads)
        LLM_REJ("head count mismatch");
    if (qs[2] != 1) LLM_REJ("q sequence length must be 1 (decode)");
    if (qs[3] != head_dim || ks[3] != head_dim || vs[3] != head_dim)
        LLM_REJ("head_dim mismatch");
    if (ks[2] < visible_len || vs[2] < visible_len)
        LLM_REJ("k/v cache shorter than visible_len");

    if (!ensure_kernels(head_dim)) LLM_REJ("metal kernel registration failed");

    const int B = qs[0];
    const int N = visible_len;
    /* Split-K: abbastanza split da riempire la GPU (B*H*S threadgroup) senza chunk
     * troppo piccoli (overhead del merge). ~256 chiavi per split, clampato a 32. */
    int S = (N + 127) / 128;
    if (S < 1) S = 1;
    if (S > 64) S = 64;

    int Nval = N, Sval = S;
    float scal = scale;
    mlx_array Narr = mlx_array_new_data(&Nval, NULL, 0, MLX_INT32);
    mlx_array Sarr = mlx_array_new_data(&Sval, NULL, 0, MLX_INT32);
    mlx_array Scal = mlx_array_new_data(&scal, NULL, 0, MLX_FLOAT32);
    if (Narr.ctx == nullptr || Sarr.ctx == nullptr || Scal.ctx == nullptr) {
        mlx_array_free(Narr); mlx_array_free(Sarr); mlx_array_free(Scal);
        LLM_REJ("scalar alloc failed");
    }

    bool ok = true;
    mlx_array pm = mlx_array_new(), pl = mlx_array_new(), po = mlx_array_new();
    mlx_vector_array pinputs = mlx_vector_array_new();
    mlx_vector_array_append_value(pinputs, q);
    mlx_vector_array_append_value(pinputs, k);
    mlx_vector_array_append_value(pinputs, v);
    mlx_vector_array_append_value(pinputs, Narr);
    mlx_vector_array_append_value(pinputs, Sarr);
    mlx_vector_array_append_value(pinputs, Scal);

    mlx_fast_metal_kernel_config pcfg = mlx_fast_metal_kernel_config_new();
    add_tmpl(pcfg, dt, num_heads, num_kv_heads, head_dim);
    const int pmsh[3] = {B, num_heads, S};
    const int posh[4] = {B, num_heads, S, head_dim};
    mlx_fast_metal_kernel_config_add_output_arg(pcfg, pmsh, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(pcfg, pmsh, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(pcfg, posh, 4, dt);
    mlx_fast_metal_kernel_config_set_grid(pcfg, 32 * num_heads, B, S);
    mlx_fast_metal_kernel_config_set_thread_group(pcfg, 32, 1, 1);

    mlx_vector_array poutputs = mlx_vector_array_new();
    if (ok) ok = mlx_fast_metal_kernel_apply(&poutputs, g_kpartial, pinputs, pcfg, stream) == 0 &&
                 mlx_vector_array_size(poutputs) == 3;
    if (ok) {
        mlx_array t = mlx_array_new();
        ok = mlx_vector_array_get(&t, poutputs, 0) == 0 && mlx_array_set(&pm, t) == 0; mlx_array_free(t);
        t = mlx_array_new();
        ok = ok && mlx_vector_array_get(&t, poutputs, 1) == 0 && mlx_array_set(&pl, t) == 0; mlx_array_free(t);
        t = mlx_array_new();
        ok = ok && mlx_vector_array_get(&t, poutputs, 2) == 0 && mlx_array_set(&po, t) == 0; mlx_array_free(t);
    }
    if (!ok) reject_once("partial kernel apply failed");

    if (ok) {
        mlx_vector_array minputs = mlx_vector_array_new();
        mlx_vector_array_append_value(minputs, pm);
        mlx_vector_array_append_value(minputs, pl);
        mlx_vector_array_append_value(minputs, po);
        mlx_vector_array_append_value(minputs, Sarr);
        mlx_fast_metal_kernel_config mcfg = mlx_fast_metal_kernel_config_new();
        add_tmpl(mcfg, dt, num_heads, /*Hkv=*/0, head_dim);
        const int oshape[4] = {B, num_heads, 1, head_dim};
        mlx_fast_metal_kernel_config_add_output_arg(mcfg, oshape, 4, dt);
        mlx_fast_metal_kernel_config_set_grid(mcfg, 32 * num_heads, B, 1);
        mlx_fast_metal_kernel_config_set_thread_group(mcfg, 32, 1, 1);
        mlx_vector_array moutputs = mlx_vector_array_new();
        ok = mlx_fast_metal_kernel_apply(&moutputs, g_kmerge, minputs, mcfg, stream) == 0 &&
             mlx_vector_array_size(moutputs) == 1;
        if (ok) {
            mlx_array o0 = mlx_array_new();
            ok = mlx_vector_array_get(&o0, moutputs, 0) == 0 && mlx_array_set(out, o0) == 0;
            mlx_array_free(o0);
        }
        if (!ok) reject_once("merge kernel apply failed");
        mlx_fast_metal_kernel_config_free(mcfg);
        mlx_vector_array_free(minputs);
        mlx_vector_array_free(moutputs);
    }

    mlx_fast_metal_kernel_config_free(pcfg);
    mlx_vector_array_free(pinputs);
    mlx_vector_array_free(poutputs);
    mlx_array_free(pm); mlx_array_free(pl); mlx_array_free(po);
    mlx_array_free(Narr); mlx_array_free(Sarr); mlx_array_free(Scal);
    return ok;
}

extern "C" bool llm_mlx_flash_prefill_attention(
    mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
    int num_heads, int num_kv_heads, int head_dim, int lq, int cache_offset,
    float scale, mlx_stream stream) {
    if (out == nullptr || q.ctx == nullptr || k.ctx == nullptr || v.ctx == nullptr)
        LLM_REJ("null handle");
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || lq <= 0 || cache_offset < 0)
        LLM_REJ("bad dims");
    if (num_heads % num_kv_heads != 0) LLM_REJ("bad GQA");
    if (head_dim % 32 != 0 || head_dim < 32) LLM_REJ("head_dim not mult 32");
    mlx_dtype dt = mlx_array_dtype(q);
    if (dt != mlx_array_dtype(k) || dt != mlx_array_dtype(v)) LLM_REJ("mixed dtypes");
    if (dt != MLX_FLOAT32 && dt != MLX_BFLOAT16 && dt != MLX_FLOAT16) LLM_REJ("unsupported dtype");
    if (mlx_array_ndim(q) != 4 || mlx_array_ndim(k) != 4 || mlx_array_ndim(v) != 4)
        LLM_REJ("not rank-4");
    const int* qs = mlx_array_shape(q);
    const int* ks = mlx_array_shape(k);
    const int* vs = mlx_array_shape(v);
    if (qs == nullptr || ks == nullptr || vs == nullptr) LLM_REJ("null shape");
    if (qs[1] != num_heads || ks[1] != num_kv_heads || vs[1] != num_kv_heads)
        LLM_REJ("head count mismatch");
    if (qs[2] != lq) LLM_REJ("q seq != lq");
    if (qs[3] != head_dim || ks[3] != head_dim || vs[3] != head_dim) LLM_REJ("head_dim mismatch");
    const int N = cache_offset + lq;
    if (ks[2] < N || vs[2] < N) LLM_REJ("k/v shorter than N");
    if (!ensure_kernels(head_dim)) LLM_REJ("kernel registration failed");

    const int B = qs[0];
    int Nval = N, coff = cache_offset, lqv = lq;
    float scal = scale;
    mlx_array Narr = mlx_array_new_data(&Nval, NULL, 0, MLX_INT32);
    mlx_array Carr = mlx_array_new_data(&coff, NULL, 0, MLX_INT32);
    mlx_array Larr = mlx_array_new_data(&lqv, NULL, 0, MLX_INT32);
    mlx_array Scal = mlx_array_new_data(&scal, NULL, 0, MLX_FLOAT32);
    if (Narr.ctx == nullptr || Carr.ctx == nullptr || Larr.ctx == nullptr || Scal.ctx == nullptr) {
        mlx_array_free(Narr); mlx_array_free(Carr); mlx_array_free(Larr); mlx_array_free(Scal);
        LLM_REJ("scalar alloc failed");
    }
    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, q);
    mlx_vector_array_append_value(inputs, k);
    mlx_vector_array_append_value(inputs, v);
    mlx_vector_array_append_value(inputs, Narr);
    mlx_vector_array_append_value(inputs, Carr);
    mlx_vector_array_append_value(inputs, Larr);
    mlx_vector_array_append_value(inputs, Scal);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", num_heads);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hkv", num_kv_heads);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", head_dim);
    const int oshape[4] = {B, num_heads, lq, head_dim};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, oshape, 4, dt);
    /* grid in THREAD (come il decode kernel): x = 32*lq (32 lane per query token). */
    mlx_fast_metal_kernel_config_set_grid(cfg, 32 * lq, num_heads, B);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, g_kprefill, inputs, cfg, stream);
    bool ok = (rc == 0) && (mlx_vector_array_size(outputs) == 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        ok = (mlx_vector_array_get(&o0, outputs, 0) == 0);
        if (ok) ok = (mlx_array_set(out, o0) == 0);
        mlx_array_free(o0);
    }
    if (!ok) reject_once("prefill metal apply returned non-zero");
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(Narr); mlx_array_free(Carr); mlx_array_free(Larr); mlx_array_free(Scal);
    return ok;
}
