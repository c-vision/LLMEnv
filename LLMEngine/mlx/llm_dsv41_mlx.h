/* llm_dsv41_mlx.h — DeepSeek V4.1: forward in API MLX C (header-only).
 *
 * Traduzione fedele di `llm_dsv41_fwd_test.c` (plain-C, verificato vs il riferimento oMLX)
 * nelle API `mlx/c/mlx.h`. Copre il percorso del modello tiny (prefill, `compress_ratios==0`):
 *   embed -> repeat(hc) -> per ogni Block [ mHC(hc_mixes/hc_pre/hc_post) + attenzione
 *   (q-LoRA, RoPE, SWA + attention-sink, window KV packed fp8, wo_a a gruppi) + MoE
 *   (gate sqrtsoftplus, esperti SwiGLU, shared) ] -> hc_pre finale -> RMSNorm -> head.
 *
 * La spec numerica e' l'inference ufficiale (`deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py`);
 * il riferimento MLX e' oMLX `patches/deepseek_v41/language.py`. Il packing fp8 usa
 * `llm_dsv41_fp8.h` (byte-identico a MLX).
 *
 * SCOPE: prefill senza cache persistente (S<=window del tiny), niente compressor/indexer/engram
 * (richiedono i layer sorgente e i pesi reali: v. todo). NIENTE STUB: il percorso coperto e'
 * completo; i percorsi non coperti (ratio>0, cache decode) non sono dichiarati qui.
 *
 * NOTA RISORSE: gli array temporanei non vengono liberati esplicitamente (il test e' di breve
 * durata). L'integrazione nell'engine dovra' gestirne il refcount (mlx_array_free) o compilare
 * il grafo; tracciato nel todo Forward-Integration.
 *
 * Verifica: `llm_dsv41_mlx_test.c` vs `REF_FW_LOGITS`.
 */
#ifndef LLM_DSV41_MLX_H
#define LLM_DSV41_MLX_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mlx/c/mlx.h>

#include "llm_dsv41_metal.h"

#include "llm_dsv41_tensors.h"
#include "llm_dsv41_fp8.h"
#include "llm_dsv41_fp4.h"
#include "llm_dsv41_engram.h"
#include "llm_dsv41_container.h"

/* Layer weights: se in STREAMING (w->stream != NULL) carica on-demand il layer l nello slot
 * riusabile (idempotente); altrimenti &w->layers[l] (materializzato). */
static inline const LlmDsv41LayerWeights* llm_dsv41_mlx_L(const LlmDsv41Weights* w, int l) {
    if (w->stream) return llm_dsv41_stream_layer((LlmDsv41Stream*)w->stream, l);
    return &w->layers[l];
}


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LLM_DSV41_MLX_RET(_op, _res, ...) \
    do { if ((_op)(&(_res), __VA_ARGS__, s) != 0) { *ok = false; return mlx_array_new(); } } while (0)
#define LLM_DSV41_MLX_DO(_op, _res, ...) \
    do { if ((_op)(&(_res), __VA_ARGS__, s) != 0) { *ok = false; return; } } while (0)

static inline mlx_array llm_dsv41_mlx_scalar(float v) { return mlx_array_new_float(v); }

/* Stato condiviso cross-layer (reference `shared`): il KV compresso e l'index K calcolati dal
 * layer kv_source sono riusati dai layer index_source-only successivi (stesso ratio); i
 * candidate blocks sono pubblicati dal layer candidate_source. Un solo slot per forward. */
typedef struct {
    mlx_array kv;
    mlx_array index_k;
    mlx_array candidates;
    bool has_kv;
    bool has_index_k;
    bool has_candidates;
    mlx_array ci;
    bool has_ci;
} LlmDsv41Shared;

static inline void llm_dsv41_shared_init(LlmDsv41Shared* sh) {
    sh->kv = mlx_array_new(); sh->index_k = mlx_array_new(); sh->candidates = mlx_array_new();
    sh->ci = mlx_array_new();
    sh->has_kv = false; sh->has_index_k = false; sh->has_candidates = false; sh->has_ci = false;
}

/* y = x @ w^T  (w: [out,in], x: [...,in]) */
static inline mlx_array llm_dsv41_mlx_linear(mlx_array x, mlx_array w, mlx_stream s, bool* ok) {
    mlx_array wt = mlx_array_new(), y = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_transpose, wt, w);
    LLM_DSV41_MLX_RET(mlx_matmul, y, x, wt);
    return y;
}

/* slice su un asse: a[..., lo:hi, ...] */
static inline mlx_array llm_dsv41_mlx_slice_axis(mlx_array a, int axis, int lo, int hi,
                                                 mlx_stream s, bool* ok) {
    int nd = (int)mlx_array_ndim(a);
    const int* sh = mlx_array_shape(a);
    int* st = (int*)calloc((size_t)nd, sizeof(int));
    int* sp = (int*)calloc((size_t)nd, sizeof(int));
    int* sd = (int*)malloc((size_t)nd * sizeof(int));
    if (!st || !sp || !sd) { free(st); free(sp); free(sd); *ok = false; return mlx_array_new(); }
    for (int i = 0; i < nd; i++) { sp[i] = sh[i]; sd[i] = 1; }
    st[axis] = lo; sp[axis] = hi;
    mlx_array out = mlx_array_new();
    if (mlx_slice(&out, a, st, (size_t)nd, sp, (size_t)nd, sd, (size_t)nd, s) != 0) *ok = false;
    free(st); free(sp); free(sd);
    return out;
}

/* slice sull'ultimo asse: a[..., lo:hi] */
static inline mlx_array llm_dsv41_mlx_slice_last(mlx_array a, int lo, int hi,
                                                 mlx_stream s, bool* ok) {
    return llm_dsv41_mlx_slice_axis(a, (int)mlx_array_ndim(a) - 1, lo, hi, s, ok);
}

/* RMSNorm (language.py norm): out = x * rsqrt(mean(x^2)+eps) * weight, ultimo asse. */
static inline mlx_array llm_dsv41_mlx_rmsnorm(mlx_array x, mlx_array w, float eps,
                                              mlx_stream s, bool* ok) {
    int ax = (int)mlx_array_ndim(x) - 1;
    mlx_array xf = mlx_array_new(), sq = mlx_array_new(), mn = mlx_array_new();
    mlx_array rs = mlx_array_new(), y = mlx_array_new();
    mlx_array esc = llm_dsv41_mlx_scalar(eps);
    LLM_DSV41_MLX_RET(mlx_astype, xf, x, MLX_FLOAT32);
    LLM_DSV41_MLX_RET(mlx_square, sq, xf);
    LLM_DSV41_MLX_RET(mlx_mean_axis, mn, sq, ax, true);
    LLM_DSV41_MLX_RET(mlx_add, mn, mn, esc);
    LLM_DSV41_MLX_RET(mlx_rsqrt, rs, mn);
    LLM_DSV41_MLX_RET(mlx_multiply, y, xf, rs);
    LLM_DSV41_MLX_RET(mlx_multiply, y, y, w);
    return y;
}

/* sinkhorn (hyper_connection.py _sinkhorn_reference) su comb [..., n, n]. */
static inline mlx_array llm_dsv41_mlx_sinkhorn(mlx_array comb, float eps, int iters,
                                               mlx_stream s, bool* ok) {
    /* leva Metal opt-in (kill-switch `LLMMETAL_DSV41_METAL`): sinkhorn fuso per hc==4. */
    if (llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_metal_sinkhorn(&mo, comb, eps, iters, s)) return mo;
        mlx_array_free(mo);
    }
    int nd = (int)mlx_array_ndim(comb);
    int ax_row = nd - 2, ax_col = nd - 1;
    mlx_array sc = llm_dsv41_mlx_scalar(eps);
    mlx_array t = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_sum_axis, t, comb, ax_row, true);
    LLM_DSV41_MLX_RET(mlx_add, t, t, sc);
    LLM_DSV41_MLX_RET(mlx_divide, comb, comb, t);
    for (int it = 1; it < iters; it++) {
        LLM_DSV41_MLX_RET(mlx_sum_axis, t, comb, ax_col, true);
        LLM_DSV41_MLX_RET(mlx_add, t, t, sc);
        LLM_DSV41_MLX_RET(mlx_divide, comb, comb, t);
        LLM_DSV41_MLX_RET(mlx_sum_axis, t, comb, ax_row, true);
        LLM_DSV41_MLX_RET(mlx_add, t, t, sc);
        LLM_DSV41_MLX_RET(mlx_divide, comb, comb, t);
    }
    return comb;
}

/* hc_mixes (language.py _hc_mixes/_hc_mix_weights): h [1,S,hc,dim] -> pre/post [1,S,hc],
 * comb [1,S,hc,hc]. */
static inline void llm_dsv41_mlx_hc_mixes(mlx_array h, mlx_array fn, mlx_array scale,
                                          mlx_array base, const LlmDsv41Config* c,
                                          mlx_array* pre, mlx_array* post, mlx_array* comb,
                                          mlx_stream s, bool* ok) {
    const int hc = c->hc_mult;
    const int* sh = mlx_array_shape(h);
    int flat_sh[4] = { 1, sh[1], sh[2] * sh[3] };
    mlx_array flat = mlx_array_new(), sq = mlx_array_new(), mn = mlx_array_new();
    mlx_array rs = mlx_array_new(), mixes = mlx_array_new(), tmp = mlx_array_new();
    mlx_array esc = llm_dsv41_mlx_scalar(c->norm_eps);
    LLM_DSV41_MLX_DO(mlx_reshape, flat, h, flat_sh, 3);
    bool have_mix = false;
    if (llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_metal_hc_projection(&mo, flat, fn, c->norm_eps, s)) { mixes = mo; have_mix = true; }
        else mlx_array_free(mo);
    }
    if (!have_mix) {
        LLM_DSV41_MLX_DO(mlx_square, sq, flat);
        LLM_DSV41_MLX_DO(mlx_mean_axis, mn, sq, 2, true);
        LLM_DSV41_MLX_DO(mlx_add, mn, mn, esc);
        LLM_DSV41_MLX_DO(mlx_rsqrt, rs, mn);
        /* mixes = flat @ fn^T * rsqrt */
        mlx_array fnt = mlx_array_new();
        LLM_DSV41_MLX_DO(mlx_transpose, fnt, fn);
        LLM_DSV41_MLX_DO(mlx_matmul, mixes, flat, fnt);
        LLM_DSV41_MLX_DO(mlx_multiply, mixes, mixes, rs);
    }

    mlx_array sc0 = llm_dsv41_mlx_slice_last(scale, 0, 1, s, ok);
    mlx_array sc1 = llm_dsv41_mlx_slice_last(scale, 1, 2, s, ok);
    mlx_array sc2 = llm_dsv41_mlx_slice_last(scale, 2, 3, s, ok);
    mlx_array bpre = llm_dsv41_mlx_slice_last(base, 0, hc, s, ok);
    mlx_array bpost = llm_dsv41_mlx_slice_last(base, hc, 2 * hc, s, ok);
    mlx_array bcomb = llm_dsv41_mlx_slice_last(base, 2 * hc, 2 * hc + hc * hc, s, ok);
    mlx_array mpre = llm_dsv41_mlx_slice_last(mixes, 0, hc, s, ok);
    mlx_array mpost = llm_dsv41_mlx_slice_last(mixes, hc, 2 * hc, s, ok);
    mlx_array mcomb = llm_dsv41_mlx_slice_last(mixes, 2 * hc, 2 * hc + hc * hc, s, ok);

    mlx_array heps = llm_dsv41_mlx_scalar(c->hc_eps);
    /* pre = sigmoid(mpre*scale0 + bpre) + hc_eps */
    LLM_DSV41_MLX_DO(mlx_multiply, tmp, mpre, sc0);
    LLM_DSV41_MLX_DO(mlx_add, tmp, tmp, bpre);
    LLM_DSV41_MLX_DO(mlx_sigmoid, tmp, tmp);
    LLM_DSV41_MLX_DO(mlx_add, *pre, tmp, heps);
    /* post = 2*sigmoid(mpost*scale1 + bpost) */
    LLM_DSV41_MLX_DO(mlx_multiply, tmp, mpost, sc1);
    LLM_DSV41_MLX_DO(mlx_add, tmp, tmp, bpost);
    LLM_DSV41_MLX_DO(mlx_sigmoid, tmp, tmp);
    {
        mlx_array two = llm_dsv41_mlx_scalar(2.0f);
        LLM_DSV41_MLX_DO(mlx_multiply, *post, tmp, two);
    }
    /* comb = softmax(mcomb*scale2 + bcomb) + hc_eps; sinkhorn */
    LLM_DSV41_MLX_DO(mlx_multiply, tmp, mcomb, sc2);
    LLM_DSV41_MLX_DO(mlx_add, tmp, tmp, bcomb);
    {
        int comb_sh[4] = { sh[0], sh[1], hc, hc };
        LLM_DSV41_MLX_DO(mlx_reshape, tmp, tmp, comb_sh, 4);
    }
    LLM_DSV41_MLX_DO(mlx_softmax_axis, tmp, tmp, 3, false);
    LLM_DSV41_MLX_DO(mlx_add, tmp, tmp, heps);
    *comb = llm_dsv41_mlx_sinkhorn(tmp, c->hc_eps, c->hc_sinkhorn_iters, s, ok);
}

/* hc_pre: x [1,S,hc,dim] * pre [1,S,hc] -> [1,S,dim] */
static inline mlx_array llm_dsv41_mlx_hc_pre(mlx_array x, mlx_array pre,
                                             mlx_stream s, bool* ok) {
    int ax = (int)mlx_array_ndim(x) - 2;
    mlx_array pe = mlx_array_new(), prod = mlx_array_new(), out = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_expand_dims, pe, pre, -1);
    LLM_DSV41_MLX_RET(mlx_multiply, prod, x, pe);
    LLM_DSV41_MLX_RET(mlx_sum_axis, out, prod, ax, false);
    return out;
}

/* hc_pre + RMSNorm: leva Metal opt-in (kernel fuso) con fallback semantico. */
static inline mlx_array llm_dsv41_mlx_hc_pre_norm(mlx_array h, mlx_array pre, mlx_array w,
                                                  float eps, mlx_stream s, bool* ok) {
    if (llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_metal_hc_pre_norm(&mo, h, pre, w, eps, s)) return mo;
        mlx_array_free(mo);
    }
    mlx_array x = llm_dsv41_mlx_hc_pre(h, pre, s, ok);
    return llm_dsv41_mlx_rmsnorm(x, w, eps, s, ok);
}

/* hc_post: x [1,S,dim], residual [1,S,hc,dim], post [1,S,hc], comb [1,S,hc,hc]
 * -> out [1,S,hc,dim] = post[...,None]*x[...,None,:] + sum_j comb[...,j,k]*residual[...,j,:] */
static inline mlx_array llm_dsv41_mlx_hc_post(mlx_array x, mlx_array residual,
                                              mlx_array post, mlx_array comb,
                                              mlx_stream s, bool* ok) {
    if (llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_metal_hc_post(&mo, x, residual, post, comb, s)) return mo;
        mlx_array_free(mo);
    }
    int ax = (int)mlx_array_ndim(residual) - 2;
    mlx_array pe = mlx_array_new(), xe = mlx_array_new(), t1 = mlx_array_new();
    mlx_array ce = mlx_array_new(), re = mlx_array_new(), prod = mlx_array_new();
    mlx_array t2 = mlx_array_new(), out = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_expand_dims, pe, post, -1);
    LLM_DSV41_MLX_RET(mlx_expand_dims, xe, x, -2);
    LLM_DSV41_MLX_RET(mlx_multiply, t1, pe, xe);
    LLM_DSV41_MLX_RET(mlx_expand_dims, ce, comb, -1);
    LLM_DSV41_MLX_RET(mlx_expand_dims, re, residual, -2);
    LLM_DSV41_MLX_RET(mlx_multiply, prod, ce, re);
    LLM_DSV41_MLX_RET(mlx_sum_axis, t2, prod, ax, false);
    LLM_DSV41_MLX_RET(mlx_add, out, t1, t2);
    return out;
}

/* freq RoPE (language.py _rope) su host: freq[hd2], con correzione YaRN se compressed. */
static inline void llm_dsv41_mlx_rope_freq(float* freq, int hd2, int d, float base,
                                           const LlmDsv41Config* c, int compressed) {
    for (int j = 0; j < hd2; j++) freq[j] = 1.0f / powf(base, (2.0f * (float)j) / (float)d);
    if (compressed && c->original_seq_len > 0) {
        float lnb = logf(base);
        float cfast = (float)d * logf((float)c->original_seq_len / (c->beta_fast * 2.0f * (float)M_PI)) / (2.0f * lnb);
        float cslow = (float)d * logf((float)c->original_seq_len / (c->beta_slow * 2.0f * (float)M_PI)) / (2.0f * lnb);
        int low = (int)floorf(cfast); if (low < 0) low = 0;
        int high = (int)ceilf(cslow); if (high > d - 1) high = d - 1;
        float denom = (float)(high - low); if (denom < 1e-3f) denom = 1e-3f;
        for (int j = 0; j < hd2; j++) {
            float sm = 1.0f - fminf(fmaxf(((float)j - (float)low) / denom, 0.0f), 1.0f);
            freq[j] = freq[j] / c->rope_factor * (1.0f - sm) + freq[j] * sm;
        }
    }
}

/* RoPE su x [1,S,H,D]: ruota le ultime d=rope_head_dim dimensioni. positions [S]. */
static inline mlx_array llm_dsv41_mlx_rope(mlx_array x, mlx_array positions,
                                           const LlmDsv41Config* c, int compressed, int inverse,
                                           mlx_stream s, bool* ok) {
    const int d = c->rope_head_dim, hd2 = d / 2;
    int nd = (int)mlx_array_ndim(x);
    const int* sh = mlx_array_shape(x);
    const int D = sh[nd - 1];        /* ultima dim reale (head_dim o index_head_dim) */
    const int off = D - d;
    int S = sh[1];
    float base = compressed ? c->compress_rope_theta : c->rope_theta;
    float freq[512];
    llm_dsv41_mlx_rope_freq(freq, hd2, d, base, c, compressed);

    mlx_array freqv = llm_dsv41_array_f32(freq, 1, (const int[]){ hd2 });
    mlx_array pf = mlx_array_new(), pcol = mlx_array_new(), ang = mlx_array_new();
    mlx_array ang4 = mlx_array_new(), cs = mlx_array_new(), sn = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_astype, pf, positions, MLX_FLOAT32);
    LLM_DSV41_MLX_RET(mlx_reshape, pcol, pf, (const int[]){ S, 1 }, 2);
    LLM_DSV41_MLX_RET(mlx_multiply, ang, pcol, freqv);
    if (inverse) LLM_DSV41_MLX_RET(mlx_negative, ang, ang);
    /* angles: [1,S] + [1]*(nd-3) + [hd2]  (language.py _rope) */
    int ash[8];
    ash[0] = 1; ash[1] = S;
    for (int i = 2; i < nd - 1; i++) ash[i] = 1;
    ash[nd - 1] = hd2;
    LLM_DSV41_MLX_RET(mlx_reshape, ang4, ang, ash, (size_t)nd);
    LLM_DSV41_MLX_RET(mlx_cos, cs, ang4);
    LLM_DSV41_MLX_RET(mlx_sin, sn, ang4);

    mlx_array tail = llm_dsv41_mlx_slice_last(x, off, D, s, ok);
    mlx_array tail2 = mlx_array_new();
    int t2sh[8];
    for (int i = 0; i < nd - 1; i++) t2sh[i] = sh[i];
    t2sh[nd - 1] = hd2; t2sh[nd] = 2;
    LLM_DSV41_MLX_RET(mlx_reshape, tail2, tail, t2sh, (size_t)(nd + 1));
    mlx_array a = llm_dsv41_mlx_slice_last(tail2, 0, 1, s, ok);
    mlx_array b = llm_dsv41_mlx_slice_last(tail2, 1, 2, s, ok);
    LLM_DSV41_MLX_RET(mlx_squeeze_axis, a, a, nd);
    LLM_DSV41_MLX_RET(mlx_squeeze_axis, b, b, nd);
    mlx_array r0 = mlx_array_new(), r1 = mlx_array_new(), t = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_multiply, r0, a, cs);
    LLM_DSV41_MLX_RET(mlx_multiply, t, b, sn);
    LLM_DSV41_MLX_RET(mlx_subtract, r0, r0, t);
    LLM_DSV41_MLX_RET(mlx_multiply, r1, a, sn);
    LLM_DSV41_MLX_RET(mlx_multiply, t, b, cs);
    LLM_DSV41_MLX_RET(mlx_add, r1, r1, t);
    mlx_vector_array vecs = mlx_vector_array_new();
    mlx_vector_array_append_value(vecs, r0);
    mlx_vector_array_append_value(vecs, r1);
    mlx_array rot = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_stack_axis, rot, vecs, nd);
    int rsh[8];
    for (int i = 0; i < nd - 1; i++) rsh[i] = sh[i];
    rsh[nd - 1] = d;
    LLM_DSV41_MLX_RET(mlx_reshape, rot, rot, rsh, (size_t)nd);
    mlx_vector_array_free(vecs);
    mlx_array head = llm_dsv41_mlx_slice_last(x, 0, off, s, ok);
    mlx_vector_array vecs2 = mlx_vector_array_new();
    mlx_vector_array_append_value(vecs2, head);
    mlx_vector_array_append_value(vecs2, rot);
    mlx_array out = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_concatenate_axis, out, vecs2, nd - 1);
    mlx_vector_array_free(vecs2);
    return out;
}

/* pack/unpack fp8 (llm_dsv41_fp8.h) sull'ultima dimensione di kv [1,S,width]. */
static inline mlx_array llm_dsv41_mlx_fp8_roundtrip(mlx_array kv, int width,
                                                    mlx_stream s, bool* ok) {
    if (llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_metal_fp8_roundtrip(&mo, kv, s)) return mo;
        mlx_array_free(mo);
    }
    mlx_array_eval(kv);
    int nd = (int)mlx_array_ndim(kv);
    const int* sh = mlx_array_shape(kv);
    size_t nrows = 1;
    for (int i = 0; i < nd - 1; i++) nrows *= (size_t)sh[i];
    const float* src = mlx_array_data_float32(kv);
    if (!src) { *ok = false; return mlx_array_new(); }
    float* out = (float*)malloc(nrows * (size_t)width * sizeof(float));
    uint8_t* packed = (uint8_t*)malloc((size_t)(width + width / 32));
    if (!out || !packed) { free(out); free(packed); *ok = false; return mlx_array_new(); }
    for (size_t i = 0; i < nrows; i++) {
        llmcodec_pack_fp8(src + i * (size_t)width, width, packed);
        llmcodec_unpack_fp8(packed, width, out + i * (size_t)width);
    }
    int shp[8];
    for (int i = 0; i < nd; i++) shp[i] = sh[i];
    mlx_array res = llm_dsv41_array_f32(out, nd, shp);
    free(out); free(packed);
    return res;
}

/* pack/unpack FP4 (llm_dsv41_fp4.h) sull'ultima dimensione di kv [1,S,width]. */
static inline mlx_array llm_dsv41_mlx_fp4_roundtrip(mlx_array kv, int width, int group_size,
                                                    int e4m3_scale, mlx_stream s, bool* ok) {
    mlx_array_eval(kv);
    int nd = (int)mlx_array_ndim(kv);
    const int* sh = mlx_array_shape(kv);
    size_t nrows = 1;
    for (int i = 0; i < nd - 1; i++) nrows *= (size_t)sh[i];
    const float* src = mlx_array_data_float32(kv);
    if (!src) { *ok = false; return mlx_array_new(); }
    int nbytes = width * 4 / 8 + width / group_size;
    float* out = (float*)malloc(nrows * (size_t)width * sizeof(float));
    uint8_t* packed = (uint8_t*)malloc((size_t)nbytes);
    if (!out || !packed) { free(out); free(packed); *ok = false; return mlx_array_new(); }
    for (size_t i = 0; i < nrows; i++) {
        llmcodec_pack_activation(src + i * (size_t)width, width, 4, group_size, e4m3_scale, packed);
        llmcodec_unpack_activation(packed, width, 4, group_size, e4m3_scale, out + i * (size_t)width);
    }
    int shp[8];
    for (int i = 0; i < nd; i++) shp[i] = sh[i];
    mlx_array res = llm_dsv41_array_f32(out, nd, shp);
    free(out); free(packed);
    return res;
}

/* Compressor (language.py Compressor): ratio==1 -> norm(wkv(x)); ratio>1 -> pooling con wgate. */
static inline mlx_array llm_dsv41_mlx_compressor(mlx_array x, mlx_array wkv, mlx_array norm,
                                                 mlx_array wgate, int r, const LlmDsv41Config* c,
                                                 mlx_stream s, bool* ok) {
    const int D = c->head_dim;
    int S = (int)mlx_array_shape(x)[1];
    mlx_array kv = llm_dsv41_mlx_linear(x, wkv, s, ok);
    if (r <= 1) return llm_dsv41_mlx_rmsnorm(kv, norm, c->norm_eps, s, ok);
    mlx_array gate = llm_dsv41_mlx_linear(x, wgate, s, ok);
    int cutoff = (S / r) * r, m = cutoff / r;
    if (m <= 0) { *ok = false; return mlx_array_new(); }
    kv = llm_dsv41_mlx_slice_axis(kv, 1, 0, cutoff, s, ok);
    gate = llm_dsv41_mlx_slice_axis(gate, 1, 0, cutoff, s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, kv, kv, (const int[]){ 1, m, r, D }, 4);
    LLM_DSV41_MLX_RET(mlx_reshape, gate, gate, (const int[]){ 1, m, r, D }, 4);
    mlx_array probs = mlx_array_new(), prod = mlx_array_new(), pooled = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_softmax_axis, probs, gate, 2, false);
    LLM_DSV41_MLX_RET(mlx_multiply, prod, kv, probs);
    LLM_DSV41_MLX_RET(mlx_sum_axis, pooled, prod, 2, false);
    return llm_dsv41_mlx_rmsnorm(pooled, norm, c->norm_eps, s, ok);
}

/* gather righe: kv [1,m,D] con ids [N] (>=0) -> [N,D] */
static inline mlx_array llm_dsv41_mlx_gather_rows(mlx_array kv, mlx_array ids_flat, int m,
                                                  mlx_stream s, bool* ok) {
    int D = (int)mlx_array_shape(kv)[2];
    mlx_array kv2 = mlx_array_new(), out = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, kv2, kv, (const int[]){ m, D }, 2);
    LLM_DSV41_MLX_RET(mlx_take_axis, out, kv2, ids_flat, 0);
    return out;
}

/* top-k su [1,S,W] -> ids [1,S,count] (ordine cronologico crescente, -1 se invalido). */
static inline mlx_array llm_dsv41_mlx_topk_ids(mlx_array scores, int count, mlx_stream s, bool* ok) {
    if (count <= 0) {
        mlx_array z = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_zeros, z, (const int[]){ 1, (int)mlx_array_shape(scores)[1], 0 }, 3, MLX_INT32);
        return z;
    }
    mlx_array negs = mlx_array_new(), order = mlx_array_new(), taken = mlx_array_new();
    mlx_array valid = mlx_array_new(), ids = mlx_array_new();
    mlx_array neginf = mlx_array_new_float(-INFINITY), mone = mlx_array_new_int(-1);
    LLM_DSV41_MLX_RET(mlx_negative, negs, scores);
    LLM_DSV41_MLX_RET(mlx_argsort_axis, order, negs, 2);
    order = llm_dsv41_mlx_slice_last(order, 0, count, s, ok);
    LLM_DSV41_MLX_RET(mlx_take_along_axis, taken, scores, order, 2);
    LLM_DSV41_MLX_RET(mlx_greater, valid, taken, neginf);
    LLM_DSV41_MLX_RET(mlx_where, ids, valid, order, mone);
    LLM_DSV41_MLX_RET(mlx_sort_axis, ids, ids, 2);
    return ids;
}

/* top-k su [1,S,W] mappando ai VALORI in `vals` (stessa forma): ids = vals[topk]. */
static inline mlx_array llm_dsv41_mlx_topk_vals(mlx_array scores, mlx_array vals, int count,
                                                mlx_stream s, bool* ok) {
    if (count <= 0) {
        mlx_array z = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_zeros, z, (const int[]){ 1, (int)mlx_array_shape(scores)[1], 0 }, 3, MLX_INT32);
        return z;
    }
    mlx_array negs = mlx_array_new(), order = mlx_array_new(), taken = mlx_array_new();
    mlx_array sel = mlx_array_new(), valid = mlx_array_new(), ids = mlx_array_new();
    mlx_array neginf = mlx_array_new_float(-INFINITY), mone = mlx_array_new_int(-1);
    LLM_DSV41_MLX_RET(mlx_negative, negs, scores);
    LLM_DSV41_MLX_RET(mlx_argsort_axis, order, negs, 2);
    order = llm_dsv41_mlx_slice_last(order, 0, count, s, ok);
    LLM_DSV41_MLX_RET(mlx_take_along_axis, taken, scores, order, 2);
    LLM_DSV41_MLX_RET(mlx_take_along_axis, sel, vals, order, 2);
    LLM_DSV41_MLX_RET(mlx_greater, valid, taken, neginf);
    LLM_DSV41_MLX_RET(mlx_where, ids, valid, sel, mone);
    LLM_DSV41_MLX_RET(mlx_sort_axis, ids, ids, 2);
    return ids;
}

/* Selezione blocchi candidati (kernels.py _tile_topk con block_size/force_latest):
 * scores [1,S,Kc] (con -inf invert). Ritorna blocchi [1,S,nb] (crescente, -1 invalido). */
static inline mlx_array llm_dsv41_mlx_block_select(mlx_array scores, int Kc, int bs, int nb,
                                                   int start, int ratio, mlx_stream s, bool* ok) {
    int S = (int)mlx_array_shape(scores)[1];
    int nblocks = (Kc + bs - 1) / bs;
    int padn = nblocks * bs - Kc;
    mlx_array sc = scores;
    if (padn > 0) {
        mlx_array pv = mlx_array_new_float(-INFINITY), padded = mlx_array_new();
        int axes[1] = { 2 }, lo[1] = { 0 }, hi[1] = { padn };
        if (mlx_pad(&padded, scores, axes, 1, lo, 1, hi, 1, pv, "constant", s) != 0) *ok = false;
        sc = padded;
    }
    mlx_array sc4 = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, sc4, sc, (const int[]){ 1, S, nblocks, bs }, 4);
    mlx_array bmax = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_max_axis, bmax, sc4, 3, false);
    /* force_latest: il blocco dell'ultimo token visibile -> +inf */
    int* force = (int*)calloc((size_t)S * (size_t)nblocks, sizeof(int));
    if (!force) { *ok = false; return mlx_array_new(); }
    for (int i = 0; i < S; i++) {
        int vis = (start + i + 1) / ratio;
        if (vis > 0) { int lb = (vis - 1) / bs; if (lb < nblocks) force[i * nblocks + lb] = 1; }
    }
    mlx_array fmask = mlx_array_new_data(force, (const int[]){ 1, S, nblocks }, 3, MLX_BOOL);
    free(force);
    mlx_array inf = mlx_array_new_float(INFINITY), bmax2 = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_where, bmax2, fmask, inf, bmax);
    int cnt = nb < nblocks ? nb : nblocks;
    return llm_dsv41_mlx_topk_ids(bmax2, cnt, s, ok);
}

/* Coda dell'attenzione: RoPE inverso + wo_a (a gruppi) + wo_b. Condivisa tra path semantico e
 * path Metal packed -> il wiring e' un solo punto. */
static inline mlx_array llm_dsv41_mlx_attn_tail(const LlmDsv41Weights* w, int l, mlx_array out,
                                                mlx_array positions, int S, mlx_stream s, bool* ok) {
    const LlmDsv41Config* c = &w->cfg;
    const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
    const int OG = c->o_groups, OL = c->o_lora_rank;
    const int GRP = c->n_heads * c->head_dim / OG;
    out = llm_dsv41_mlx_rope(out, positions, c, c->compress_ratios[l] > 0, 1, s, ok);
    if (!*ok) return mlx_array_new();
    mlx_array grp = mlx_array_new(), woa = mlx_array_new(), y = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, grp, out, (const int[]){ 1, S, OG, GRP }, 4);
    LLM_DSV41_MLX_RET(mlx_reshape, woa, L->wo_a, (const int[]){ OG, OL, GRP }, 3);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, grp);
        mlx_vector_array_append_value(ops, woa);
        if (mlx_einsum(&y, "bsgd,grd->bsgr", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, y, y, (const int[]){ 1, S, OG * OL }, 3);
    return llm_dsv41_mlx_linear(y, L->wo_b, s, ok);
}

/* Path Metal packed (SWA, ratio==0, prefill): window fp8-packed + indici causali -> out [1,S,H,D].
 * Ritorna true solo se il kernel ha prodotto l'output (gated `LLMMETAL_DSV41_METAL`); altrimenti il
 * chiamante usa il path semantico. Il kernel gestisce sink e softmax online. */
static inline bool llm_dsv41_mlx_attention_packed_swa(mlx_array* out, mlx_array q, mlx_array keys,
                                                      mlx_array sink_v, const LlmDsv41Config* c,
                                                      int S, mlx_stream s) {
    if (!llm_dsv41_metal_enabled()) return false;
    const int H = c->n_heads, D = c->head_dim, W = c->window_size;
    if (W <= 0 || S <= 0 || W > S || S > 8192) return false;
    mlx_array wnd2 = mlx_array_new();
    if (!llm_dsv41_metal_pack8(&wnd2, keys, S, D, s)) return false;
    mlx_array wnd = mlx_array_new();
    if (mlx_reshape(&wnd, wnd2, (const int[]){ 1, S, D + D / 32 }, 3, s) != 0) { mlx_array_free(wnd2); return false; }
    mlx_array_free(wnd2);
    int32_t* wi_b = (int32_t*)malloc((size_t)S * (size_t)W * sizeof(int32_t));
    if (wi_b == NULL) { mlx_array_free(wnd); return false; }
    for (int i = 0; i < S; i++)
        for (int k = 0; k < W; k++) {
            int j = i - W + 1 + k;
            wi_b[(size_t)i * W + k] = (j >= 0 && j <= i) ? j : -1;
        }
    mlx_array wi = mlx_array_new_data(wi_b, (const int[]){ 1, S, W }, 3, MLX_INT32);
    /* Placeholder VALIDI (il wrapper richiede array non vuoti): pooled dummy [1,1,psz] e ci
     * [1,S,1] tutti -1 -> il kernel li salta (`row < 0` continua). Nel path SWA non ci sono
     * chiavi compresse, quindi NC/C non influenzano il risultato. */
    const int psz = D / 2 + D / 16;
    uint8_t* pz = (uint8_t*)calloc((size_t)psz, 1);
    int32_t* ci_b = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    if (pz == NULL || ci_b == NULL) { free(wi_b); free(pz); free(ci_b); mlx_array_free(wnd); mlx_array_free(wi); return false; }
    for (int i = 0; i < S; i++) ci_b[i] = -1;
    mlx_array pool_d = mlx_array_new_data(pz, (const int[]){ 1, 1, psz }, 3, MLX_UINT8);
    mlx_array ci_d = mlx_array_new_data(ci_b, (const int[]){ 1, S, 1 }, 3, MLX_INT32);
    mlx_array sink = mlx_array_new();
    if (mlx_reshape(&sink, sink_v, (const int[]){ H }, 1, s) != 0) {
        free(wi_b); free(pz); free(ci_b); mlx_array_free(wnd); mlx_array_free(wi);
        mlx_array_free(pool_d); mlx_array_free(ci_d); return false;
    }
    float scale = 1.0f / sqrtf((float)D);
    bool okp = llm_dsv41_metal_packed_attention(out, q, wnd, pool_d, wi, ci_d, sink, scale, s);
    free(wi_b); free(pz); free(ci_b);
    mlx_array_free(wnd); mlx_array_free(wi); mlx_array_free(pool_d); mlx_array_free(ci_d);
    return okp;
}

/* Path Metal packed COMPRESSO (ratio>0, prefill, no CED): window fp8-packed + KV compresso
 * fp4-packed (gs16 e4m3) + `ci` = indici top-k (gia' int32 [1,S,C], -1 = padding saltato dal
 * kernel). `att_kv` e' [1,m,D] (gia' fp4-roundtrippato dal path semantico -> ripacchettarlo
 * riproduce gli stessi byte). Ritorna true solo se il kernel ha prodotto l'output. */
static inline bool llm_dsv41_mlx_attention_packed_comp(mlx_array* out, mlx_array q, mlx_array keys,
                                                       mlx_array att_kv, int m, mlx_array ids, int C,
                                                       mlx_array sink_v, const LlmDsv41Config* c,
                                                       int S, mlx_stream s) {
    if (!llm_dsv41_metal_enabled()) return false;
    const int H = c->n_heads, D = c->head_dim, W = c->window_size;
    if (W <= 0 || S <= 0 || m <= 0 || C <= 0 || S > 8192 || C > 8192) return false;
    mlx_array wnd2 = mlx_array_new();
    if (!llm_dsv41_metal_pack8(&wnd2, keys, S, D, s)) return false;
    mlx_array wnd = mlx_array_new();
    if (mlx_reshape(&wnd, wnd2, (const int[]){ 1, S, D + D / 32 }, 3, s) != 0) { mlx_array_free(wnd2); return false; }
    mlx_array_free(wnd2);
    mlx_array pool2 = mlx_array_new();
    if (!llm_dsv41_metal_pack4(&pool2, att_kv, m, D, 16, 1, s)) { mlx_array_free(wnd); return false; }
    mlx_array pool = mlx_array_new();
    if (mlx_reshape(&pool, pool2, (const int[]){ 1, m, D / 2 + D / 16 }, 3, s) != 0) { mlx_array_free(wnd); mlx_array_free(pool2); return false; }
    mlx_array_free(pool2);
    mlx_array wi = mlx_array_new();
    {
        int32_t* wi_b = (int32_t*)malloc((size_t)S * (size_t)W * sizeof(int32_t));
        if (wi_b == NULL) { mlx_array_free(wnd); mlx_array_free(pool); return false; }
        for (int i = 0; i < S; i++)
            for (int k = 0; k < W; k++) {
                int j = i - W + 1 + k;
                wi_b[(size_t)i * W + k] = (j >= 0 && j <= i) ? j : -1;
            }
        mlx_array wi_t = mlx_array_new_data(wi_b, (const int[]){ 1, S, W }, 3, MLX_INT32);
        free(wi_b);
        if (mlx_reshape(&wi, wi_t, (const int[]){ 1, S, W }, 3, s) != 0) {
            mlx_array_free(wnd); mlx_array_free(pool); return false;
        }
    }
    mlx_array sink = mlx_array_new();
    if (mlx_reshape(&sink, sink_v, (const int[]){ H }, 1, s) != 0) {
        mlx_array_free(wnd); mlx_array_free(pool); mlx_array_free(wi); return false;
    }
    float scale = 1.0f / sqrtf((float)D);
    bool okp = llm_dsv41_metal_packed_attention(out, q, wnd, pool, wi, ids, sink, scale, s);
    mlx_array_free(wnd); mlx_array_free(pool); mlx_array_free(wi); mlx_array_free(sink);
    return okp;
}

/* Path Metal packed DECODE (offset>0): window dal buffer cache (`wbuf`, NW righe) + KV compresso
 * cache (`kbuf`, NC righe) + `ci` = top-k ids. `wi` = [S,W] (o [1,S,W]) di indici nel buffer
 * finestra (-1 = padding saltato dal kernel). NC/C == 0 -> dummy (path SWA puro). Ritorna true
 * solo se il kernel ha prodotto l'output. */
static inline bool llm_dsv41_mlx_attention_packed_dec(mlx_array* out, mlx_array q,
                                                      mlx_array wbuf, int NW,
                                                      mlx_array kbuf, int NC,
                                                      mlx_array wi, mlx_array ci, int C,
                                                      mlx_array sink_v, const LlmDsv41Config* c,
                                                      int S, mlx_stream s) {
    if (!llm_dsv41_metal_enabled()) return false;
    const int H = c->n_heads, D = c->head_dim;
    if (NW <= 0 || S <= 0 || S > 8192 || D % 32) return false;
    const int W = (int)mlx_array_shape(wi)[mlx_array_ndim(wi) - 1];
    mlx_array w2 = mlx_array_new();
    if (!llm_dsv41_metal_pack8(&w2, wbuf, NW, D, s)) return false;
    mlx_array wnd = mlx_array_new();
    if (mlx_reshape(&wnd, w2, (const int[]){ 1, NW, D + D / 32 }, 3, s) != 0) { mlx_array_free(w2); return false; }
    mlx_array_free(w2);
    mlx_array wi3 = mlx_array_new();
    if (mlx_reshape(&wi3, wi, (const int[]){ 1, S, W }, 3, s) != 0) { mlx_array_free(wnd); return false; }
    bool dummy = (C <= 0 || NC <= 0);
    mlx_array pool = mlx_array_new(), ci3 = mlx_array_new();
    uint8_t* pz = NULL; int32_t* cib = NULL;
    if (!dummy) {
        mlx_array p2 = mlx_array_new();
        if (!llm_dsv41_metal_pack4(&p2, kbuf, NC, D, 16, 1, s)) { mlx_array_free(wnd); mlx_array_free(wi3); return false; }
        if (mlx_reshape(&pool, p2, (const int[]){ 1, NC, D / 2 + D / 16 }, 3, s) != 0) {
            mlx_array_free(wnd); mlx_array_free(wi3); mlx_array_free(p2); return false;
        }
        mlx_array_free(p2);
        ci3 = ci;
    } else {
        const int psz = D / 2 + D / 16;
        pz = (uint8_t*)calloc((size_t)psz, 1);
        cib = (int32_t*)malloc((size_t)S * sizeof(int32_t));
        if (pz == NULL || cib == NULL) { free(pz); free(cib); mlx_array_free(wnd); mlx_array_free(wi3); return false; }
        for (int i = 0; i < S; i++) cib[i] = -1;
        pool = mlx_array_new_data(pz, (const int[]){ 1, 1, psz }, 3, MLX_UINT8);
        ci3 = mlx_array_new_data(cib, (const int[]){ 1, S, 1 }, 3, MLX_INT32);
    }
    mlx_array sink = mlx_array_new();
    if (mlx_reshape(&sink, sink_v, (const int[]){ H }, 1, s) != 0) {
        free(pz); free(cib); mlx_array_free(wnd); mlx_array_free(wi3); mlx_array_free(pool); mlx_array_free(ci3); return false;
    }
    float scale = 1.0f / sqrtf((float)D);
    bool okp = llm_dsv41_metal_packed_attention(out, q, wnd, pool, wi3, ci3, sink, scale, s);
    if (dummy) { free(pz); free(cib); mlx_array_free(pool); mlx_array_free(ci3); }
    mlx_array_free(wnd); mlx_array_free(wi3); mlx_array_free(sink);
    return okp;
}

/* Attenzione CSA2: q-LoRA, RoPE (base compressa se ratio>0), window fp8, KV compresso fp4,
 * Indexer (score = sum_h relu(dot(q_h,k)) * w_h) con visibilita' causale per query
 * (j < (query+1)/ratio), top-k (count=min(index_topk,m)) ordine cronologico, SWA + sink.
 *
 * LIMITE DICHIARATO: solo prefill (start==0) e layer che sono ANCHE kv_source (l'index key
 * e' calcolato localmente). I layer solo-index_source (reuse di `index_k` da un layer
 * precedente) e candidate_source_layer sono rifiutati con errore esplicito. */
static inline mlx_array llm_dsv41_mlx_attention(const LlmDsv41Weights* w, int l, mlx_array x,
                                                mlx_array positions, int S,
                                                LlmDsv41Shared* sh,
                                                int start_pos, int has_ced, mlx_array ced_kv, int ced_kv_start,
                                                mlx_stream s, bool* ok) {
    const LlmDsv41Config* c = &w->cfg;
    const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
    const int H = c->n_heads, D = c->head_dim, OG = c->o_groups, OL = c->o_lora_rank;
    const int GRP = H * D / OG;
    const int ratio = c->compress_ratios[l];
    const int comp = ratio > 0;

    mlx_array qr = llm_dsv41_mlx_linear(x, L->wq_a, s, ok);
    qr = llm_dsv41_mlx_rmsnorm(qr, L->q_norm, c->norm_eps, s, ok);
    mlx_array q = llm_dsv41_mlx_linear(qr, L->wq_b, s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, q, q, (const int[]){ 1, S, H, D }, 4);
    q = llm_dsv41_mlx_rope(q, positions, c, comp, 0, s, ok);

    mlx_array kv = llm_dsv41_mlx_linear(x, L->wkv, s, ok);
    kv = llm_dsv41_mlx_rmsnorm(kv, L->kv_norm, c->norm_eps, s, ok);
    kv = llm_dsv41_mlx_rope(kv, positions, c, comp, 0, s, ok);
    mlx_array keys = llm_dsv41_mlx_fp8_roundtrip(kv, D, s, ok);   /* window keys [1,S,D] */
    int N = S;

    /* Path Metal packed (gated): window fp8-packed + kernel `_ATTENTION` + coda. Solo ratio==0,
     * prefill (start_pos==0) e senza CED; altrimenti si prosegue col path semantico. */
    if (!comp && llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_mlx_attention_packed_swa(&mo, q, keys, L->attn_sink, c, S, s))
            return llm_dsv41_mlx_attn_tail(w, l, mo, positions, S, s, ok);
    }

    /* scores window [1,H,S,S] */
    mlx_array scores = mlx_array_new();
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, q);
        mlx_vector_array_append_value(ops, keys);
        if (mlx_einsum(&scores, "bihd,bjd->bhij", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    mlx_array qscale = llm_dsv41_mlx_scalar(1.0f / sqrtf((float)D));
    LLM_DSV41_MLX_RET(mlx_multiply, scores, scores, qscale);

    /* maschera window causale SWA: (j <= i) & (j >= i-(win-1)) */
    mlx_array ii = mlx_array_new(), jj = mlx_array_new(), thr = mlx_array_new();
    mlx_array ge = mlx_array_new(), le = mlx_array_new(), cond = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_arange, ii, 0.0, (double)S, 1.0, MLX_INT32);
    LLM_DSV41_MLX_RET(mlx_arange, jj, 0.0, (double)S, 1.0, MLX_INT32);
    LLM_DSV41_MLX_RET(mlx_reshape, ii, ii, (const int[]){ S, 1 }, 2);
    LLM_DSV41_MLX_RET(mlx_reshape, jj, jj, (const int[]){ 1, S }, 2);
    {
        mlx_array wm1 = mlx_array_new_int(c->window_size - 1);
        LLM_DSV41_MLX_RET(mlx_subtract, thr, ii, wm1);
    }
    LLM_DSV41_MLX_RET(mlx_greater_equal, ge, jj, thr);
    LLM_DSV41_MLX_RET(mlx_less_equal, le, jj, ii);
    LLM_DSV41_MLX_RET(mlx_logical_and, cond, ge, le);
    LLM_DSV41_MLX_RET(mlx_reshape, cond, cond, (const int[]){ 1, 1, S, S }, 4);

    mlx_array comp_sel = mlx_array_new();   /* [1,S,C,D] chiavi compresse selezionate */
    int C = 0;
    if (comp) {
        const int IH = c->index_head_dim, NH = c->index_n_heads;
        mlx_array att_kv = mlx_array_new(), ik = mlx_array_new();
        mlx_array ids = mlx_array_new();
        int m = 0;
        bool pure_reuse = (!c->in_kv_source[l] && !c->in_index_source[l]);
        if (pure_reuse) {
            /* ratio>0 ma né source: riusa KV compresso e top-k dal layer sorgente. */
            if (!sh->has_kv || !sh->has_ci) {
                fprintf(stderr, "llm_dsv41_mlx: layer %d ratio>0 senza shared kv/ci\n", l);
                *ok = false; return mlx_array_new();
            }
            att_kv = sh->kv; ids = sh->ci; m = (int)mlx_array_shape(att_kv)[1];
        } else {
        if (!c->in_index_source[l]) {
            fprintf(stderr, "llm_dsv41_mlx: layer %d kv_source senza index_source non supportato\n", l);
            *ok = false; return mlx_array_new();
        }
        if (c->in_kv_source[l]) {
            mlx_array kv_x = has_ced ? ced_kv : x;
            int kv_start = has_ced ? ced_kv_start : start_pos;
            int m = (int)mlx_array_shape(kv_x)[1] / ratio;
            if (m <= 0) { *ok = false; return mlx_array_new(); }
            mlx_array latent = llm_dsv41_mlx_compressor(kv_x, L->comp_wkv, L->comp_norm, L->comp_wgate,
                                                        ratio, c, s, ok);
            mlx_array cpos = mlx_array_new(), cpoff = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_arange, cpos, 0.0, (double)m, 1.0, MLX_INT32);
            {
                mlx_array rr = mlx_array_new_int(ratio);
                mlx_array off = mlx_array_new_int((kv_start / ratio) * ratio);
                LLM_DSV41_MLX_RET(mlx_multiply, cpoff, cpos, rr);
                LLM_DSV41_MLX_RET(mlx_add, cpoff, cpoff, off);
            }
            /* chiavi compresse per l'attenzione (fp4 gs16 e4m3) + index K (fp4 gs32) */
            att_kv = llm_dsv41_mlx_rope(latent, cpoff, c, 1, 0, s, ok);
            att_kv = llm_dsv41_mlx_fp4_roundtrip(att_kv, D, 16, 1, s, ok);
            ik = llm_dsv41_mlx_linear(latent, L->idx_wk, s, ok);
            ik = llm_dsv41_mlx_rmsnorm(ik, L->idx_k_norm, c->norm_eps, s, ok);
            ik = llm_dsv41_mlx_rope(ik, cpoff, c, 1, 0, s, ok);
            ik = llm_dsv41_mlx_fp4_roundtrip(ik, IH, 32, 0, s, ok);
            sh->kv = att_kv; sh->index_k = ik; sh->has_kv = true; sh->has_index_k = true;
        } else {
            /* index_source-only: riusa KV compresso e index K del kv_source precedente */
            if (!sh->has_kv || !sh->has_index_k) {
                fprintf(stderr, "llm_dsv41_mlx: layer %d index_source-only senza shared kv/index_k\n", l);
                *ok = false; return mlx_array_new();
            }
            att_kv = sh->kv; ik = sh->index_k;
        }
        m = (int)mlx_array_shape(att_kv)[1];
        /* Indexer */
        mlx_array iq = llm_dsv41_mlx_linear(qr, L->idx_wq_b, s, ok);
        LLM_DSV41_MLX_RET(mlx_reshape, iq, iq, (const int[]){ 1, S, NH, IH }, 4);
        iq = llm_dsv41_mlx_rope(iq, positions, c, 1, 0, s, ok);
        iq = llm_dsv41_mlx_fp4_roundtrip(iq, IH, 32, 0, s, ok);
        mlx_array wp = llm_dsv41_mlx_linear(x, L->idx_weights_proj, s, ok);  /* [1,S,NH] */
        {
            mlx_array ws = llm_dsv41_mlx_scalar(1.0f / sqrtf((float)IH) / sqrtf((float)NH));
            LLM_DSV41_MLX_RET(mlx_multiply, wp, wp, ws);
        }
        mlx_array sidx = mlx_array_new();
        bool sidx_kernel = false;
        /* visibilita' causale (bound per query): usata dal path semantico E dal ramo candidates. */
        int vis[4096];
        if (S > 4096) { *ok = false; return mlx_array_new(); }
        for (int i = 0; i < S; i++) vis[i] = (start_pos + i + 1) / ratio;
        mlx_array visv = mlx_array_new_data(vis, (const int[]){ S }, 1, MLX_INT32);
        mlx_array visc = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_reshape, visc, visv, (const int[]){ S, 1 }, 2);
        mlx_array neginf0 = mlx_array_new_float(-INFINITY);
        /* Path Metal packed (gated): score indexer + visibilita' causale direttamente dal kernel
         * `_INDEX` (chiavi index fp4-packed gs32). Sostituisce einsum+relu+pesi e la maschera. */
        if (ratio > 0 && llm_dsv41_metal_enabled()) {
            mlx_array packed = mlx_array_new();
            if (llm_dsv41_metal_pack4(&packed, ik, m, IH, 32, 0, s)) {
                mlx_array pk3 = mlx_array_new();
                if (mlx_reshape(&pk3, packed, (const int[]){ 1, m, IH / 2 + IH / 32 }, 3, s) == 0) {
                    mlx_array ec = mlx_array_new_int(0);
                    if (llm_dsv41_metal_index_scores(&sidx, iq, pk3, wp, start_pos, ratio, 0, ec, 0, s)) {
                        sidx_kernel = true;
                    }
                }
            }
        }
        if (!sidx_kernel) {
        mlx_array dots = mlx_array_new(), rel = mlx_array_new(), wpk = mlx_array_new();
        {
            mlx_vector_array ops = mlx_vector_array_new();
            mlx_vector_array_append_value(ops, iq);
            mlx_vector_array_append_value(ops, ik);
            if (mlx_einsum(&dots, "bshd,bkd->bshk", ops, s) != 0) *ok = false;
            mlx_vector_array_free(ops);
        }
        if (!*ok) return mlx_array_new();
        {
            mlx_array zero = llm_dsv41_mlx_scalar(0.0f);
            LLM_DSV41_MLX_RET(mlx_maximum, rel, dots, zero);
        }
        LLM_DSV41_MLX_RET(mlx_expand_dims, wpk, wp, -1);
        LLM_DSV41_MLX_RET(mlx_multiply, rel, rel, wpk);
        LLM_DSV41_MLX_RET(mlx_sum_axis, sidx, rel, 2, false);        /* [1,S,m] */
        /* visibilita' causale: chiave j visibile sse j < (query+1)/ratio */
        mlx_array jjc = mlx_array_new(), jarr = mlx_array_new();
        mlx_array cvis = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_arange, jarr, 0.0, (double)m, 1.0, MLX_INT32);
        LLM_DSV41_MLX_RET(mlx_reshape, jjc, jarr, (const int[]){ 1, m }, 2);
        LLM_DSV41_MLX_RET(mlx_less, cvis, jjc, visc);                 /* [S,m] */
        LLM_DSV41_MLX_RET(mlx_reshape, cvis, cvis, (const int[]){ 1, S, m }, 3);
        mlx_array sidx2 = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_where, sidx2, cvis, sidx, neginf0);
        sidx = sidx2;
        }
        /* selezione: candidate blocks (se previsti) oppure top-k su tutte le chiavi */
        const int cand_src = c->candidate_source_layer;
        if (cand_src >= 0 && l > cand_src && sh->has_candidates) {
            int nb = (int)mlx_array_shape(sh->candidates)[2];
            int bs = c->candidate_block_size;
            int Ncand = nb * bs;
            mlx_array ble = mlx_array_new(), bss = mlx_array_new_int(bs);
            mlx_array off = mlx_array_new(), ar = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_expand_dims, ble, sh->candidates, -1);
            LLM_DSV41_MLX_RET(mlx_multiply, ble, ble, bss);
            LLM_DSV41_MLX_RET(mlx_arange, ar, 0.0, (double)bs, 1.0, MLX_INT32);
            LLM_DSV41_MLX_RET(mlx_add, off, ble, ar);
            mlx_array zero_i0 = mlx_array_new_int(0), bmask = mlx_array_new(), cand = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_greater_equal, bmask, ble, zero_i0);
            {
                mlx_array mone = mlx_array_new_int(-1);
                LLM_DSV41_MLX_RET(mlx_where, cand, bmask, off, mone);
            }
            LLM_DSV41_MLX_RET(mlx_reshape, cand, cand, (const int[]){ 1, S, Ncand }, 3);
            mlx_array cf = mlx_array_new(), ccl = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_reshape, cf, cand, (const int[]){ S * Ncand }, 1);
            LLM_DSV41_MLX_RET(mlx_maximum, ccl, cf, zero_i0);
            mlx_array ksel = llm_dsv41_mlx_gather_rows(ik, ccl, m, s, ok);
            LLM_DSV41_MLX_RET(mlx_reshape, ksel, ksel, (const int[]){ 1, S, Ncand, IH }, 4);
            mlx_array dots2 = mlx_array_new(), rel2 = mlx_array_new(), wpk2 = mlx_array_new();
            {
                mlx_vector_array ops = mlx_vector_array_new();
                mlx_vector_array_append_value(ops, iq);
                mlx_vector_array_append_value(ops, ksel);
                if (mlx_einsum(&dots2, "bshd,bskd->bshk", ops, s) != 0) *ok = false;
                mlx_vector_array_free(ops);
            }
            if (!*ok) return mlx_array_new();
            {
                mlx_array zero = llm_dsv41_mlx_scalar(0.0f);
                LLM_DSV41_MLX_RET(mlx_maximum, rel2, dots2, zero);
            }
            LLM_DSV41_MLX_RET(mlx_expand_dims, wpk2, wp, -1);
            LLM_DSV41_MLX_RET(mlx_multiply, rel2, rel2, wpk2);
            mlx_array scnd = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_sum_axis, scnd, rel2, 2, false);
            /* maschera: cand >= 0 AND cand < vis */
            mlx_array nonsc = mlx_array_new(), lt = mlx_array_new(), cnd2 = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_greater_equal, nonsc, cand, zero_i0);
            LLM_DSV41_MLX_RET(mlx_less, lt, cand, visc);
            LLM_DSV41_MLX_RET(mlx_logical_and, cnd2, nonsc, lt);
            mlx_array scnd2 = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_where, scnd2, cnd2, scnd, neginf0);
            int ccnt = c->index_topk < Ncand ? c->index_topk : Ncand;
            ids = llm_dsv41_mlx_topk_vals(scnd2, cand, ccnt, s, ok);
        } else {
            int count = c->index_topk < m ? c->index_topk : m;
            ids = llm_dsv41_mlx_topk_ids(sidx, count, s, ok);
            if (cand_src >= 0 && l == cand_src) {
                sh->candidates = llm_dsv41_mlx_block_select(sidx, m, c->candidate_block_size,
                                                             c->candidate_topk_blocks, 0, ratio, s, ok);
                sh->has_candidates = true;
            }
        }
        sh->ci = ids; sh->has_ci = true;
        }
        C = (int)mlx_array_shape(ids)[2];
        /* Path Metal packed COMPRESSO (gated): window + KV compresso + top-k come `ci`, poi coda.
         * Solo prefill (start_pos==0) e senza CED; altrimenti path semantico. */
        if (llm_dsv41_metal_enabled()) {
            mlx_array mo = mlx_array_new();
            if (llm_dsv41_mlx_attention_packed_comp(&mo, q, keys, att_kv, m, ids, C, L->attn_sink, c, S, s))
                return llm_dsv41_mlx_attn_tail(w, l, mo, positions, S, s, ok);
        }
        mlx_array ids_flat = mlx_array_new(), ids_cl = mlx_array_new(), zero_i = mlx_array_new_int(0);
        LLM_DSV41_MLX_RET(mlx_reshape, ids_flat, ids, (const int[]){ S * C }, 1);
        LLM_DSV41_MLX_RET(mlx_maximum, ids_cl, ids_flat, zero_i);
        mlx_array sel = llm_dsv41_mlx_gather_rows(att_kv, ids_cl, m, s, ok);
        LLM_DSV41_MLX_RET(mlx_reshape, comp_sel, sel, (const int[]){ 1, S, C, D }, 4);
        mlx_array mask_c = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_greater_equal, mask_c, ids, zero_i);
        LLM_DSV41_MLX_RET(mlx_reshape, mask_c, mask_c, (const int[]){ 1, 1, S, C }, 4);
        /* scores compresse [1,H,S,C] */
        mlx_array scores_c = mlx_array_new();
        {
            mlx_vector_array ops = mlx_vector_array_new();
            mlx_vector_array_append_value(ops, q);
            mlx_vector_array_append_value(ops, comp_sel);
            if (mlx_einsum(&scores_c, "bihd,bikd->bhik", ops, s) != 0) *ok = false;
            mlx_vector_array_free(ops);
        }
        if (!*ok) return mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_multiply, scores_c, scores_c, qscale);
        mlx_array sc2 = mlx_array_new(), cond2 = mlx_array_new();
        mlx_vector_array vsc = mlx_vector_array_new_data((mlx_array[]){ scores, scores_c }, 2);
        mlx_vector_array vcd = mlx_vector_array_new_data((mlx_array[]){ cond, mask_c }, 2);
        if (mlx_concatenate_axis(&sc2, vsc, 3, s) != 0) *ok = false;
        if (mlx_concatenate_axis(&cond2, vcd, 3, s) != 0) *ok = false;
        mlx_vector_array_free(vsc); mlx_vector_array_free(vcd);
        if (!*ok) return mlx_array_new();
        scores = sc2; cond = cond2; N = S + C;
    }

    mlx_array neginf = mlx_array_new_float(-INFINITY);
    mlx_array masked = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_where, masked, cond, scores, neginf);

    /* sink: softmax su [scores | sink] poi scarta la colonna sink */
    mlx_array sink = mlx_array_new(), sinkb = mlx_array_new(), cat = mlx_array_new();
    mlx_array probs = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, sink, L->attn_sink, (const int[]){ 1, H, 1, 1 }, 4);
    LLM_DSV41_MLX_RET(mlx_broadcast_to, sinkb, sink, (const int[]){ 1, H, S, 1 }, 4);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, masked);
        mlx_vector_array_append_value(ops, sinkb);
        if (mlx_concatenate_axis(&cat, ops, 3, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_softmax_axis, probs, cat, 3, false);
    probs = llm_dsv41_mlx_slice_last(probs, 0, N, s, ok);

    /* out = sum probs_w * window_keys + sum probs_c * comp_sel */
    mlx_array out = mlx_array_new();
    mlx_array probs_w = llm_dsv41_mlx_slice_last(probs, 0, S, s, ok);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, probs_w);
        mlx_vector_array_append_value(ops, keys);
        if (mlx_einsum(&out, "bhij,bjd->bihd", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    if (C > 0) {
        mlx_array probs_c = llm_dsv41_mlx_slice_last(probs, S, N, s, ok);
        mlx_array outc = mlx_array_new();
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, probs_c);
        mlx_vector_array_append_value(ops, comp_sel);
        if (mlx_einsum(&outc, "bhik,bikd->bihd", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
        if (!*ok) return mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_add, out, out, outc);
    }
    return llm_dsv41_mlx_attn_tail(w, l, out, positions, S, s, ok);
}

/* MoE: gate sqrtsoftplus/softmax/sigmoid + esperti routed SwiGLU + shared. */
static inline mlx_array llm_dsv41_mlx_moe(const LlmDsv41Weights* w, int l, mlx_array x,
                                          int S, mlx_stream s, bool* ok) {
    const LlmDsv41Config* c = &w->cfg;
    const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
    const int ACT = c->n_activated_experts;
    const int DIM = c->dim, DI = c->moe_inter_dim;

    mlx_array raw = llm_dsv41_mlx_linear(x, L->gate_w, s, ok);
    mlx_array gt = llm_dsv41_mlx_scalar(c->gate_temp);
    LLM_DSV41_MLX_RET(mlx_divide, raw, raw, gt);
    mlx_array scores = mlx_array_new();
    if (c->score_func == LLM_DSV41_SCORE_SOFTMAX) {
        LLM_DSV41_MLX_RET(mlx_softmax_axis, scores, raw, 2, false);
    } else if (c->score_func == LLM_DSV41_SCORE_SIGMOID) {
        LLM_DSV41_MLX_RET(mlx_sigmoid, scores, raw);
    } else {
        mlx_array zero = llm_dsv41_mlx_scalar(0.0f), le = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_logaddexp, le, raw, zero);
        LLM_DSV41_MLX_RET(mlx_sqrt, scores, le);
    }
    mlx_array comb = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_add, comb, scores, L->gate_b);

    mlx_array neg = mlx_array_new(), order = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_negative, neg, comb);
    LLM_DSV41_MLX_RET(mlx_argsort_axis, order, neg, 2);
    mlx_array idx = llm_dsv41_mlx_slice_last(order, 0, ACT, s, ok);
    mlx_array wt = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_take_along_axis, wt, scores, idx, 2);
    if (c->norm_topk_prob && ACT > 1) {
        mlx_array sm = mlx_array_new(), eps20 = llm_dsv41_mlx_scalar(1e-20f);
        LLM_DSV41_MLX_RET(mlx_sum_axis, sm, wt, 2, true);
        LLM_DSV41_MLX_RET(mlx_add, sm, sm, eps20);
        LLM_DSV41_MLX_RET(mlx_divide, wt, wt, sm);
    }
    {
        mlx_array rs = llm_dsv41_mlx_scalar(c->route_scale);
        LLM_DSV41_MLX_RET(mlx_multiply, wt, wt, rs);
    }

    /* routed: per ogni k, gather esperti per-token + SwiGLU + w2, accumula in float32. */
    mlx_array acc = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_zeros, acc, (const int[]){ S, DIM }, 2, MLX_FLOAT32);
    mlx_array xflat = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, xflat, x, (const int[]){ S, 1, DIM }, 3);
    for (int k = 0; k < ACT; k++) {
        mlx_array idxk = llm_dsv41_mlx_slice_last(idx, k, k + 1, s, ok);      /* [1,S,1] */
        mlx_array wtk = llm_dsv41_mlx_slice_last(wt, k, k + 1, s, ok);        /* [1,S,1] */
        LLM_DSV41_MLX_RET(mlx_reshape, idxk, idxk, (const int[]){ S }, 1);
        mlx_array w1 = mlx_array_new(), w3 = mlx_array_new(), w2 = mlx_array_new();
        if (w->stream) {
            /* STREAMING: leggi i top-k esperti dal container per-token (niente pila in RAM). */
            mlx_array idx32 = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_astype, idx32, idxk, MLX_INT32);
            mlx_array_eval(idx32);
            const int32_t* ih = mlx_array_data_int32(idx32);
            if (!ih) { mlx_array_free(idx32); *ok = false; return mlx_array_new(); }
            size_t n1 = (size_t)S * DI * DIM, n2 = (size_t)S * DIM * DI;
            float* b1 = (float*)malloc(n1 * 4), *b3 = (float*)malloc(n1 * 4), *b2 = (float*)malloc(n2 * 4);
            if (!b1 || !b3 || !b2) { free(b1); free(b3); free(b2); mlx_array_free(idx32); *ok = false; return mlx_array_new(); }
            for (int t = 0; t < S && *ok; t++) {
                uint32_t ro = 0, co = 0;
                LlmEngineMoeStatus es = llm_dsv41_stream_expert((LlmDsv41Stream*)w->stream, l, (int)ih[t],
                    b1 + (size_t)t * DI * DIM, (size_t)DI * DIM,
                    b3 + (size_t)t * DI * DIM, (size_t)DI * DIM,
                    b2 + (size_t)t * DIM * DI, (size_t)DIM * DI, &ro, &co);
                if (es != LLMENGINE_MOE_OK) { fprintf(stderr, "dsv41 moe: expert %d: %s\n", (int)ih[t], llmengine_moe_status_str(es)); *ok = false; }
            }
            mlx_array_free(idx32);
            if (!*ok) { free(b1); free(b3); free(b2); return mlx_array_new(); }
            w1 = mlx_array_new_data(b1, (const int[]){ S, DI, DIM }, 3, MLX_FLOAT32);
            w3 = mlx_array_new_data(b3, (const int[]){ S, DI, DIM }, 3, MLX_FLOAT32);
            w2 = mlx_array_new_data(b2, (const int[]){ S, DIM, DI }, 3, MLX_FLOAT32);
            free(b1); free(b3); free(b2);   /* mlx_array_new_data COPIA i dati */
        } else {
            LLM_DSV41_MLX_RET(mlx_take_axis, w1, L->experts_w1, idxk, 0);
            LLM_DSV41_MLX_RET(mlx_take_axis, w3, L->experts_w3, idxk, 0);
            LLM_DSV41_MLX_RET(mlx_take_axis, w2, L->experts_w2, idxk, 0);
        }
        mlx_array w1t = mlx_array_new(), w3t = mlx_array_new(), w2t = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_transpose_axes, w1t, w1, (const int[]){ 0, 2, 1 }, 3);
        LLM_DSV41_MLX_RET(mlx_transpose_axes, w3t, w3, (const int[]){ 0, 2, 1 }, 3);
        LLM_DSV41_MLX_RET(mlx_transpose_axes, w2t, w2, (const int[]){ 0, 2, 1 }, 3);
        mlx_array g = mlx_array_new(), u = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_matmul, g, xflat, w1t);   /* [S,1,DI] */
        LLM_DSV41_MLX_RET(mlx_matmul, u, xflat, w3t);
        LLM_DSV41_MLX_RET(mlx_reshape, g, g, (const int[]){ S, DI }, 2);
        LLM_DSV41_MLX_RET(mlx_reshape, u, u, (const int[]){ S, DI }, 2);
        if (c->swiglu_limit > 0.0f) {
            mlx_array lim = llm_dsv41_mlx_scalar(c->swiglu_limit);
            mlx_array nlim = llm_dsv41_mlx_scalar(-c->swiglu_limit);
            LLM_DSV41_MLX_RET(mlx_minimum, g, g, lim);
            LLM_DSV41_MLX_RET(mlx_clip, u, u, nlim, lim);
        }
        mlx_array y = mlx_array_new(), sig = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_sigmoid, sig, g);
        LLM_DSV41_MLX_RET(mlx_multiply, y, g, sig);
        LLM_DSV41_MLX_RET(mlx_multiply, y, y, u);
        LLM_DSV41_MLX_RET(mlx_multiply, y, y, wtk);
        mlx_array y3 = mlx_array_new(), de = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_reshape, y3, y, (const int[]){ S, 1, DI }, 3);
        LLM_DSV41_MLX_RET(mlx_matmul, de, y3, w2t);      /* [S,1,DIM] */
        LLM_DSV41_MLX_RET(mlx_reshape, de, de, (const int[]){ S, DIM }, 2);
        LLM_DSV41_MLX_RET(mlx_add, acc, acc, de);
    }
    /* shared expert */
    mlx_array sg = llm_dsv41_mlx_linear(x, L->shared_w1, s, ok);
    mlx_array su = llm_dsv41_mlx_linear(x, L->shared_w3, s, ok);
    mlx_array ssig = mlx_array_new(), sy = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_sigmoid, ssig, sg);
    LLM_DSV41_MLX_RET(mlx_multiply, sy, sg, ssig);
    LLM_DSV41_MLX_RET(mlx_multiply, sy, sy, su);
    mlx_array shared = llm_dsv41_mlx_linear(sy, L->shared_w2, s, ok);
    mlx_array acc3 = mlx_array_new(), out = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, acc3, acc, (const int[]){ 1, S, DIM }, 3);
    LLM_DSV41_MLX_RET(mlx_add, out, acc3, shared);
    return out;
}

/* Iniezione Engram (engram.py Engram.__call__): h [1,S,hc,dim] -> h + gate*value.
 * `hist_in`/`hist_out` [depth] (history del NgramHash; -1 in prefill). ids raw [S]. */
static inline mlx_array llm_dsv41_mlx_engram(const LlmDsv41Weights* w, int l, mlx_array h,
                                             const int* ids, int S, const int64_t* hist_in,
                                             int64_t* hist_out, mlx_stream s, bool* ok) {
    const LlmDsv41Engram* G = &llm_dsv41_mlx_L(w, l)->engram;
    const LlmDsv41Config* c = &w->cfg;
    const int hc = c->hc_mult, dim = c->dim, depth = G->depth, nh = G->nheads, ncols = G->ncols, hd = G->hd;
    int64_t* tokens = (int64_t*)malloc((size_t)S * sizeof(int64_t));
    int64_t* hashes = (int64_t*)malloc((size_t)S * (size_t)ncols * sizeof(int64_t));
    int64_t* hin = (int64_t*)malloc((size_t)(depth > 0 ? depth : 1) * sizeof(int64_t));
    int64_t* hout = (int64_t*)malloc((size_t)(depth > 0 ? depth : 1) * sizeof(int64_t));
    int32_t* idx = (int32_t*)malloc((size_t)S * (size_t)ncols * sizeof(int32_t));
    if (!tokens || !hashes || !hin || !hout || !idx) {
        free(tokens); free(hashes); free(hin); free(hout); free(idx); *ok = false; return mlx_array_new();
    }
    for (int i = 0; i < S; i++) tokens[i] = G->token_map[ids[i]];
    for (int j = 0; j < depth; j++) hin[j] = hist_in ? hist_in[j] : -1;
    llm_dsv41_engram_hash(tokens, S, hin, depth, G->pad_id, G->mult, G->primes, nh, G->offsets,
                          hashes, hout);
    if (hist_out) for (int j = 0; j < depth; j++) hist_out[j] = hout[j];
    for (int i = 0; i < S * ncols; i++) idx[i] = (int32_t)hashes[i];
    free(tokens); free(hashes); free(hin); free(hout);

    mlx_array idsidx = mlx_array_new_data(idx, (const int[]){ S, ncols }, 2, MLX_INT32);
    free(idx);
    mlx_array rows = mlx_array_new(), flat = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_take_axis, rows, G->embed, idsidx, 0);              /* [S,ncols,hd] */
    LLM_DSV41_MLX_RET(mlx_reshape, rows, rows, (const int[]){ 1, S, ncols, hd }, 4);
    LLM_DSV41_MLX_RET(mlx_flatten, flat, rows, 2, 3);                        /* [1,S,ncols*hd] */
    mlx_array kv = llm_dsv41_mlx_linear(flat, G->wkv, s, ok);                /* [1,S,dim*(hc+1)] */
    mlx_array key = llm_dsv41_mlx_slice_last(kv, 0, dim * hc, s, ok);
    mlx_array value = llm_dsv41_mlx_slice_last(kv, dim * hc, dim * (hc + 1), s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, key, key, (const int[]){ 1, S, hc, dim }, 4);
    mlx_array x = h;   /* float32 */
    /* inv = rsqrt(mean(x^2)+eps) * rsqrt(mean(key^2)+eps) */
    mlx_array x2 = mlx_array_new(), k2 = mlx_array_new(), mx_ = mlx_array_new(), mk = mlx_array_new();
    mlx_array inv = mlx_array_new(), esc = llm_dsv41_mlx_scalar(c->norm_eps);
    LLM_DSV41_MLX_RET(mlx_square, x2, x);
    LLM_DSV41_MLX_RET(mlx_square, k2, key);
    LLM_DSV41_MLX_RET(mlx_mean_axis, mx_, x2, 3, false);
    LLM_DSV41_MLX_RET(mlx_mean_axis, mk, k2, 3, false);
    LLM_DSV41_MLX_RET(mlx_add, mx_, mx_, esc);
    LLM_DSV41_MLX_RET(mlx_add, mk, mk, esc);
    mlx_array r1 = mlx_array_new(), r2 = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_rsqrt, r1, mx_);
    LLM_DSV41_MLX_RET(mlx_rsqrt, r2, mk);
    LLM_DSV41_MLX_RET(mlx_multiply, inv, r1, r2);
    /* dot = sum(x*qw*kw*key, -1) * inv * dim^-0.5 */
    mlx_array prod = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_multiply, prod, x, G->qw);
    LLM_DSV41_MLX_RET(mlx_multiply, prod, prod, G->kw);
    LLM_DSV41_MLX_RET(mlx_multiply, prod, prod, key);
    mlx_array dot = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_sum_axis, dot, prod, 3, false);
    LLM_DSV41_MLX_RET(mlx_multiply, dot, dot, inv);
    {
        mlx_array ds = llm_dsv41_mlx_scalar(1.0f / sqrtf((float)dim));
        LLM_DSV41_MLX_RET(mlx_multiply, dot, dot, ds);
    }
    /* gate = sigmoid(sign(dot)*sqrt(max(|dot|,1e-6))); dot==0 -> sigmoid(0.001) */
    mlx_array sgn = mlx_array_new(), ab = mlx_array_new(), mx6 = mlx_array_new();
    mlx_array root = mlx_array_new(), arg = mlx_array_new(), gate = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_sign, sgn, dot);
    LLM_DSV41_MLX_RET(mlx_abs, ab, dot);
    {
        mlx_array m6 = llm_dsv41_mlx_scalar(1e-6f);
        LLM_DSV41_MLX_RET(mlx_maximum, mx6, ab, m6);
    }
    LLM_DSV41_MLX_RET(mlx_sqrt, root, mx6);
    LLM_DSV41_MLX_RET(mlx_multiply, arg, sgn, root);
    LLM_DSV41_MLX_RET(mlx_sigmoid, gate, arg);
    {
        mlx_array zero = llm_dsv41_mlx_scalar(0.0f), isz = mlx_array_new(), g0 = mlx_array_new();
        mlx_array small = llm_dsv41_mlx_scalar(0.001f);
        LLM_DSV41_MLX_RET(mlx_equal, isz, dot, zero);
        LLM_DSV41_MLX_RET(mlx_sigmoid, g0, small);
        mlx_array gate2 = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_where, gate2, isz, g0, gate);
        gate = gate2;
    }
    mlx_array ge = mlx_array_new(), ve = mlx_array_new(), t = mlx_array_new(), out = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_expand_dims, ge, gate, -1);       /* [1,S,hc,1] */
    LLM_DSV41_MLX_RET(mlx_expand_dims, ve, value, -2);      /* [1,S,1,dim] */
    LLM_DSV41_MLX_RET(mlx_multiply, t, ge, ve);
    LLM_DSV41_MLX_RET(mlx_add, out, x, t);
    return out;
}

/* Forward completo (prefill, ratio==0): ids [S] -> logits [1,S,vocab]. */
static inline bool llm_dsv41_mlx_forward(const LlmDsv41Weights* w, const int* ids, int S,
                                         mlx_array* out_logits, mlx_stream s) {
    bool ok = true;
    const LlmDsv41Config* c = &w->cfg;
    const int HC = c->hc_mult, DIM = c->dim;

    mlx_array ids_a = mlx_array_new_data(ids, (const int[]){ S }, 1, MLX_INT32);
    mlx_array h = mlx_array_new();
    if (mlx_take_axis(&h, w->embed, ids_a, 0, s) != 0) return false;   /* [S,DIM] */
    if (mlx_reshape(&h, h, (const int[]){ 1, S, 1, DIM }, 4, s) != 0) return false;
    if (mlx_repeat_axis(&h, h, HC, 2, s) != 0) return false;            /* [1,S,HC,DIM] */

    float pre_hc[LLM_DSV41_MAX_LAYERS];
    for (int i = 0; i < HC; i++) pre_hc[i] = (i == 0) ? 1.0f : 0.0f;
    mlx_array pre = llm_dsv41_array_f32(pre_hc, 1, (const int[]){ HC });
    if (mlx_broadcast_to(&pre, pre, (const int[]){ 1, S, HC }, 3, s) != 0) return false;

    mlx_array positions = mlx_array_new();
    if (mlx_arange(&positions, 0.0, (double)S, 1.0, MLX_INT32, s) != 0) return false;
    LlmDsv41Shared shared; llm_dsv41_shared_init(&shared);

    for (int l = 0; l < c->n_layers; l++) {
        const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
        if (L->engram.present) {
            int64_t hout[LLM_DSV41_MAX_LAYERS];
            h = llm_dsv41_mlx_engram(w, l, h, ids, S, NULL, hout, s, &ok);
            if (!ok) return false;
        }
        mlx_array ap = mlx_array_new(), ao = mlx_array_new(), ac = mlx_array_new();
        llm_dsv41_mlx_hc_mixes(h, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, c,
                               &ap, &ao, &ac, s, &ok);
        if (!ok) return false;
        mlx_array x = llm_dsv41_mlx_hc_pre_norm(h, pre, L->attn_norm, c->norm_eps, s, &ok);
        mlx_array attn_out = llm_dsv41_mlx_attention(w, l, x, positions, S,
                                                     &shared, 0, 0, mlx_array_new(), 0, s, &ok);
        h = llm_dsv41_mlx_hc_post(attn_out, h, ao, ac, s, &ok);

        mlx_array fp = mlx_array_new(), fo = mlx_array_new(), fc = mlx_array_new();
        llm_dsv41_mlx_hc_mixes(h, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, c,
                               &fp, &fo, &fc, s, &ok);
        if (!ok) return false;
        x = llm_dsv41_mlx_hc_pre_norm(h, ap, L->ffn_norm, c->norm_eps, s, &ok);
        mlx_array f = llm_dsv41_mlx_moe(w, l, x, S, s, &ok);
        h = llm_dsv41_mlx_hc_post(f, h, fo, fc, s, &ok);
        pre = fp;
        if (!ok) return false;
    }
    h = llm_dsv41_mlx_hc_pre_norm(h, pre, w->norm, c->norm_eps, s, &ok);
    *out_logits = llm_dsv41_mlx_linear(h, w->head, s, &ok);
    return ok;
}

/* Forward prefill con CED (prefill-skip, language.py `_ced_prefill`): la meta' decoder
 * (layer >= mid) forwarda solo gli ultimi `window_size` token; il layer mid (kv_source)
 * proietta il KV/index globale dagli hidden FINALI dell'encoder (full). Layout gia' validato da
 * `llm_dsv41_ced_layout_supported`. Restituisce i logits delle sole posizioni del tail. */
static inline bool llm_dsv41_mlx_forward_ced(const LlmDsv41Weights* w, const int* ids, int S,
                                             mlx_array* out_logits, mlx_stream s) {
    bool ok = true;
    const LlmDsv41Config* c = &w->cfg;
    const int HC = c->hc_mult, DIM = c->dim;
    const int mid = c->n_layers / 2, win = c->window_size;

    mlx_array ids_a = mlx_array_new_data(ids, (const int[]){ S }, 1, MLX_INT32);
    mlx_array h = mlx_array_new();
    if (mlx_take_axis(&h, w->embed, ids_a, 0, s) != 0) return false;
    if (mlx_reshape(&h, h, (const int[]){ 1, S, 1, DIM }, 4, s) != 0) return false;
    if (mlx_repeat_axis(&h, h, HC, 2, s) != 0) return false;
    float pre_hc[64];
    for (int i = 0; i < HC; i++) pre_hc[i] = (i == 0) ? 1.0f : 0.0f;
    mlx_array pre = llm_dsv41_array_f32(pre_hc, 1, (const int[]){ HC });
    if (mlx_broadcast_to(&pre, pre, (const int[]){ 1, S, HC }, 3, s) != 0) return false;
    LlmDsv41Shared shared; llm_dsv41_shared_init(&shared);

    int cur_len = S, cur_start = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
        if (L->engram.present) {
            int64_t hout[LLM_DSV41_MAX_LAYERS];
            h = llm_dsv41_mlx_engram(w, l, h, ids, cur_len, NULL, hout, s, &ok);
            if (!ok) return false;
        }
        if (l < mid) {
            mlx_array ap = mlx_array_new(), ao = mlx_array_new(), ac = mlx_array_new();
            llm_dsv41_mlx_hc_mixes(h, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, c,
                                   &ap, &ao, &ac, s, &ok);
            if (!ok) return false;
            mlx_array x = llm_dsv41_mlx_hc_pre_norm(h, pre, L->attn_norm, c->norm_eps, s, &ok);
            mlx_array positions = mlx_array_new();
            if (mlx_arange(&positions, (double)cur_start, (double)(cur_start + cur_len), 1.0, MLX_INT32, s) != 0) return false;
            mlx_array attn_out = llm_dsv41_mlx_attention(w, l, x, positions, cur_len, &shared,
                                                         0, 0, mlx_array_new(), 0, s, &ok);
            h = llm_dsv41_mlx_hc_post(attn_out, h, ao, ac, s, &ok);
            mlx_array fp = mlx_array_new(), fo = mlx_array_new(), fc = mlx_array_new();
            llm_dsv41_mlx_hc_mixes(h, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, c, &fp, &fo, &fc, s, &ok);
            if (!ok) return false;
            x = llm_dsv41_mlx_hc_pre_norm(h, ap, L->ffn_norm, c->norm_eps, s, &ok);
            mlx_array f = llm_dsv41_mlx_moe(w, l, x, cur_len, s, &ok);
            h = llm_dsv41_mlx_hc_post(f, h, fo, fc, s, &ok);
            pre = fp;
            if (!ok) return false;
        } else {
            int full_len = cur_len;
            int tail = win < full_len ? win : full_len;
            mlx_array h_tail = mlx_array_new(), pre_tail = mlx_array_new(), x_tail = mlx_array_new();
            mlx_array ap = mlx_array_new(), ao = mlx_array_new(), ac = mlx_array_new();
            mlx_array ced_kv = mlx_array_new();
            int has_ced = 0, ced_kv_start = cur_start;
            if (c->in_kv_source[l]) {
                llm_dsv41_mlx_hc_mixes(h, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, c,
                                       &ap, &ao, &ac, s, &ok);
                if (!ok) return false;
                mlx_array xf = llm_dsv41_mlx_hc_pre_norm(h, pre, L->attn_norm, c->norm_eps, s, &ok);
                ced_kv = xf; has_ced = 1; ced_kv_start = cur_start;
                h_tail = llm_dsv41_mlx_slice_axis(h, 1, full_len - tail, full_len, s, &ok);
                pre_tail = llm_dsv41_mlx_slice_axis(pre, 1, full_len - tail, full_len, s, &ok);
                x_tail = llm_dsv41_mlx_slice_axis(xf, 1, full_len - tail, full_len, s, &ok);
                ap = llm_dsv41_mlx_slice_axis(ap, 1, full_len - tail, full_len, s, &ok);
                ao = llm_dsv41_mlx_slice_axis(ao, 1, full_len - tail, full_len, s, &ok);
                ac = llm_dsv41_mlx_slice_axis(ac, 1, full_len - tail, full_len, s, &ok);
            } else {
                h_tail = llm_dsv41_mlx_slice_axis(h, 1, full_len - tail, full_len, s, &ok);
                pre_tail = llm_dsv41_mlx_slice_axis(pre, 1, full_len - tail, full_len, s, &ok);
                llm_dsv41_mlx_hc_mixes(h_tail, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, c,
                                       &ap, &ao, &ac, s, &ok);
                if (!ok) return false;
                x_tail = llm_dsv41_mlx_hc_pre(h_tail, pre_tail, s, &ok);
                x_tail = llm_dsv41_mlx_rmsnorm(x_tail, L->attn_norm, c->norm_eps, s, &ok);
            }
            int start_q = cur_start + full_len - tail;
            mlx_array positions = mlx_array_new();
            if (mlx_arange(&positions, (double)start_q, (double)(start_q + tail), 1.0, MLX_INT32, s) != 0) return false;
            mlx_array attn_out = llm_dsv41_mlx_attention(w, l, x_tail, positions, tail, &shared,
                                                         start_q, has_ced, ced_kv, ced_kv_start, s, &ok);
            h = llm_dsv41_mlx_hc_post(attn_out, h_tail, ao, ac, s, &ok);
            pre = pre_tail;
            mlx_array fp = mlx_array_new(), fo = mlx_array_new(), fc = mlx_array_new();
            llm_dsv41_mlx_hc_mixes(h, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, c, &fp, &fo, &fc, s, &ok);
            if (!ok) return false;
            mlx_array xf = llm_dsv41_mlx_hc_pre_norm(h, ap, L->ffn_norm, c->norm_eps, s, &ok);
            mlx_array f = llm_dsv41_mlx_moe(w, l, xf, tail, s, &ok);
            h = llm_dsv41_mlx_hc_post(f, h, fo, fc, s, &ok);
            pre = fp;
            cur_len = tail; cur_start = start_q;
            if (!ok) return false;
        }
    }
    h = llm_dsv41_mlx_hc_pre_norm(h, pre, w->norm, c->norm_eps, s, &ok);
    *out_logits = llm_dsv41_mlx_linear(h, w->head, s, &ok);
    return ok;
}

#endif /* LLM_DSV41_MLX_H */
