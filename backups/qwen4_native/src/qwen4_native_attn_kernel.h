/* qwen4_native_attn_kernel.h — P5a: prep QKV + attention gated nativa
 * (port semplificato di ds4 `qwen4_ref_attention`; testo, mRoPE con pos3 = pos).
 *
 * - prep_q: qg [T][H*2D] -> q (RMS gqn + RoPE) e gate (seconda meta'), un thread per (tok,h).
 * - prep_k: k [T][Hkv*D] -> RMS gkn + RoPE (out-of-place), un thread per (tok,h).
 * - attn:   q/gate [T][H*D], k/v [Nvis][Hkv*D], sel [Nsel] -> out [T][H*D] con
 *           softmax sui soli indici selezionati e gate sigmoid (1 thread per (tok,h)).
 * RoPE NeoX interleaved (mrope p3 = pos per il testo), coppie i / i+half.
 * L'indexer (select a blocchi) e' P5b; qui `sel` arriva dal chiamante.
 */
#ifndef LLM_QWEN4_NATIVE_ATTN_KERNEL_H
#define LLM_QWEN4_NATIVE_ATTN_KERNEL_H

#include "mlx/c/mlx.h"

static const char llm_q4n_attn_header[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n";

/* ---- prep_q ---- */
static const char llm_q4n_prep_q_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint H = uint(nhead), D = uint(hdim), T = uint(ntok), R = uint(nrot);\n"
    "if (i >= T*H || R > D || (R & 1u)) return;\n"
    "uint tok = i / H, h = i % H;\n"
    "ulong qb = ((ulong)tok*H + h)*2ul*D, ob = ((ulong)tok*H + h)*D;\n"
    "float ss = 0.f;\n"
    "for (uint d = 0; d < D; d++) ss += qg[qb+d]*qg[qb+d];\n"
    "float r = metal::rsqrt(ss/(float)D + eps);\n"
    "float tmp[256];\n"
    "for (uint d = 0; d < D; d++) tmp[d] = qg[qb+d]*r*gqn[d];\n"
    "uint nrh = R/2u;\n"
    "for (uint i = 0; i < nrh; i++) {\n"
    "  float th = (float)pos * freq[i];\n"
    "  float c = metal::cos(th)*mscale, sn = metal::sin(th)*mscale;\n"
    "  float x0 = tmp[i], x1 = tmp[i+nrh];\n"
    "  tmp[i] = x0*c - x1*sn; tmp[i+nrh] = x0*sn + x1*c;\n"
    "}\n"
    "for (uint d = 0; d < D; d++) { q[ob+d] = tmp[d]; gate[ob+d] = qg[qb+D+d]; }\n";

static mlx_fast_metal_kernel g_q4n_prep_q;
static int g_q4n_prep_q_built = 0;

static int llm_q4n_prep_q_build(void) {
    if (g_q4n_prep_q_built) return 0;
    const char* in_names[]  = {"qg","gqn","freq","ntok","nhead","hdim","nrot","pos","mscale","eps"};
    const char* out_names[] = {"q","gate"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 10; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_prep_q = mlx_fast_metal_kernel_new(
        "qwen4_attn_prep_q_native", iv, ov, llm_q4n_prep_q_src, llm_q4n_attn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_prep_q_built = (g_q4n_prep_q.ctx != NULL) ? 1 : 0;
    return g_q4n_prep_q_built ? 0 : -1;
}

static int llm_q4n_prep_q(mlx_array* q, mlx_array* gate, mlx_array qg, mlx_array gqn,
                          mlx_array freq, int n_rot, int pos, float mscale, float eps, mlx_stream s) {
    if (!q || !gate) return -1;
    if (mlx_array_ndim(qg) != 2 || mlx_array_ndim(gqn) != 1 || mlx_array_ndim(freq) != 1) return -1;
    const int* qs = mlx_array_shape(qg);
    const int* gs = mlx_array_shape(gqn);
    const int* fs = mlx_array_shape(freq);
    const int T = qs[0], hd2 = qs[1];
    /* hd2 = H*2*D: D ricavato da gqn; H = hd2/(2D) */
    const int D = gs[0];
    if (D <= 0 || hd2 % (2 * D) != 0) return -1;
    const int nh = hd2 / (2 * D);
    if (n_rot <= 0 || n_rot > D || (n_rot & 1) || fs[0] != n_rot / 2) return -1;
    if (mlx_array_dtype(qg) != MLX_FLOAT32 || mlx_array_dtype(gqn) != MLX_FLOAT32 ||
        mlx_array_dtype(freq) != MLX_FLOAT32) return -1;
    if (llm_q4n_prep_q_build() != 0) return -1;
    int ntok = T, nhead = nh, hdim = D, nrot = n_rot;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhead, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_array ar = mlx_array_new_data(&nrot, NULL, 0, MLX_INT32);
    int ipos = pos;
    mlx_array ap = mlx_array_new_data(&ipos, NULL, 0, MLX_INT32);
    mlx_array am = mlx_array_new_float32(mscale);
    mlx_array ae = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, qg);
    mlx_vector_array_append_value(in, gqn);
    mlx_vector_array_append_value(in, freq);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ad);
    mlx_vector_array_append_value(in, ar);
    mlx_vector_array_append_value(in, ap);
    mlx_vector_array_append_value(in, am);
    mlx_vector_array_append_value(in, ae);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {T, nh * D};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)nh, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_prep_q, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 2) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 && mlx_vector_array_get(&r1, outs, 1) == 0) {
            mlx_array_set(q, r0); mlx_array_set(gate, r1);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(ad); mlx_array_free(ar);
    mlx_array_free(ap); mlx_array_free(am); mlx_array_free(ae);
    return rc == 0 ? 0 : -1;
}

/* ---- prep_k ---- */
static const char llm_q4n_prep_k_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint Hkv = uint(nkv), D = uint(hdim), T = uint(ntok), R = uint(nrot);\n"
    "if (i >= T*Hkv || R > D || (R & 1u)) return;\n"
    "uint tok = i / Hkv, h = i % Hkv;\n"
    "ulong b = ((ulong)tok*Hkv + h)*D;\n"
    "float ss = 0.f;\n"
    "for (uint d = 0; d < D; d++) ss += k[b+d]*k[b+d];\n"
    "float r = metal::rsqrt(ss/(float)D + eps);\n"
    "float tmp[256];\n"
    "for (uint d = 0; d < D; d++) tmp[d] = k[b+d]*r*gkn[d];\n"
    "uint nrh = R/2u;\n"
    "for (uint i = 0; i < nrh; i++) {\n"
    "  float th = (float)pos * freq[i];\n"
    "  float c = metal::cos(th)*mscale, sn = metal::sin(th)*mscale;\n"
    "  float x0 = tmp[i], x1 = tmp[i+nrh];\n"
    "  tmp[i] = x0*c - x1*sn; tmp[i+nrh] = x0*sn + x1*c;\n"
    "}\n"
    "for (uint d = 0; d < D; d++) k_o[b+d] = tmp[d];\n";

static mlx_fast_metal_kernel g_q4n_prep_k;
static int g_q4n_prep_k_built = 0;

static int llm_q4n_prep_k_build(void) {
    if (g_q4n_prep_k_built) return 0;
    const char* in_names[]  = {"k","gkn","freq","ntok","nkv","hdim","nrot","pos","mscale","eps"};
    const char* out_names[] = {"k_o"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 10; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_prep_k = mlx_fast_metal_kernel_new(
        "qwen4_attn_prep_k_native", iv, ov, llm_q4n_prep_k_src, llm_q4n_attn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_prep_k_built = (g_q4n_prep_k.ctx != NULL) ? 1 : 0;
    return g_q4n_prep_k_built ? 0 : -1;
}

static int llm_q4n_prep_k(mlx_array* k_o, mlx_array k, mlx_array gkn, mlx_array freq,
                          int n_rot, int pos, float mscale, float eps, mlx_stream s) {
    if (!k_o || mlx_array_ndim(k) != 2 || mlx_array_ndim(gkn) != 1 || mlx_array_ndim(freq) != 1) return -1;
    const int* ks = mlx_array_shape(k);
    const int* gs = mlx_array_shape(gkn);
    const int* fs = mlx_array_shape(freq);
    const int T = ks[0], hdk = ks[1], D = gs[0];
    if (D <= 0 || hdk % D != 0) return -1;
    const int nkv = hdk / D;
    if (n_rot <= 0 || n_rot > D || (n_rot & 1) || fs[0] != n_rot / 2) return -1;
    if (mlx_array_dtype(k) != MLX_FLOAT32 || mlx_array_dtype(gkn) != MLX_FLOAT32 ||
        mlx_array_dtype(freq) != MLX_FLOAT32) return -1;
    if (llm_q4n_prep_k_build() != 0) return -1;
    int ntok = T, nkvh = nkv, hdim = D, nrot = n_rot;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nkvh, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_array ar = mlx_array_new_data(&nrot, NULL, 0, MLX_INT32);
    int ipos = pos;
    mlx_array ap = mlx_array_new_data(&ipos, NULL, 0, MLX_INT32);
    mlx_array am = mlx_array_new_float32(mscale);
    mlx_array ae = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, k);
    mlx_vector_array_append_value(in, gkn);
    mlx_vector_array_append_value(in, freq);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ad);
    mlx_vector_array_append_value(in, ar);
    mlx_vector_array_append_value(in, ap);
    mlx_vector_array_append_value(in, am);
    mlx_vector_array_append_value(in, ae);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {T, nkv * D};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)nkv, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_prep_k, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(k_o, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(ad); mlx_array_free(ar);
    mlx_array_free(ap); mlx_array_free(am); mlx_array_free(ae);
    return rc == 0 ? 0 : -1;
}

/* ---- attn ---- */
static const char llm_q4n_attn_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint H = uint(nhead), Hkv = uint(nkv), D = uint(hdim), T = uint(ntok), N = uint(nvis), S = uint(nsel);\n"
    "if (i >= T*H || H == 0u || H % Hkv != 0u) return;\n"
    "uint tok = i / H, h = i % H, kvh = h / (H/Hkv);\n"
    "ulong qb = ((ulong)tok*H + h)*D;\n"
    "float sc[2048];\n"
    "float mx = -3.0e38f;\n"
    "float scale = metal::rsqrt((float)D);\n"
    "for (uint j = 0; j < S; j++) {\n"
    "  uint idx = (uint)sel[j];\n"
    "  if (idx >= N) { sc[j] = -3.0e38f; continue; }\n"
    "  ulong kb = ((ulong)idx*Hkv + kvh)*D;\n"
    "  float dot = 0.f;\n"
    "  for (uint d = 0; d < D; d++) dot += q[qb+d]*k[kb+d];\n"
    "  sc[j] = dot*scale;\n"
    "  mx = metal::max(mx, sc[j]);\n"
    "}\n"
    "float sum = 0.f;\n"
    "for (uint j = 0; j < S; j++) { sc[j] = metal::exp(sc[j]-mx); sum += sc[j]; }\n"
    "float inv = sum > 0.f ? 1.f/sum : 0.f;\n"
    "for (uint d = 0; d < D; d++) {\n"
    "  float acc = 0.f;\n"
    "  for (uint j = 0; j < S; j++) {\n"
    "    uint idx = (uint)sel[j];\n"
    "    if (idx >= N) continue;\n"
    "    acc += sc[j]*v[((ulong)idx*Hkv + kvh)*D + d];\n"
    "  }\n"
    "  float g = gate[qb+d];\n"
    "  float sg = g >= 0.f ? 1.f/(1.f+metal::exp(-g)) : metal::exp(g)/(1.f+metal::exp(g));\n"
    "  out[qb+d] = acc*inv*sg;\n"
    "}\n";

static mlx_fast_metal_kernel g_q4n_attn;
static int g_q4n_attn_built = 0;

static int llm_q4n_attn_build(void) {
    if (g_q4n_attn_built) return 0;
    const char* in_names[]  = {"q","gate","k","v","sel","ntok","nhead","nkv","hdim","nvis","nsel"};
    const char* out_names[] = {"out"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 11; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_attn = mlx_fast_metal_kernel_new(
        "qwen4_attn_native", iv, ov, llm_q4n_attn_src, llm_q4n_attn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_attn_built = (g_q4n_attn.ctx != NULL) ? 1 : 0;
    return g_q4n_attn_built ? 0 : -1;
}

static int llm_q4n_attn(mlx_array* out, mlx_array q, mlx_array gate, mlx_array k, mlx_array v,
                        mlx_array sel, int n_head, int n_vis, mlx_stream s) {
    if (!out || mlx_array_ndim(q) != 2 || mlx_array_ndim(gate) != 2 || mlx_array_ndim(k) != 2 ||
        mlx_array_ndim(v) != 2 || mlx_array_ndim(sel) != 1) return -1;
    const int* qs = mlx_array_shape(q);
    const int* ks = mlx_array_shape(k);
    const int* ss = mlx_array_shape(sel);
    const int T = qs[0], H = n_head;
    if (H <= 0 || qs[1] % H != 0) return -1;
    const int D = qs[1] / H;
    const int* gs2 = mlx_array_shape(gate);
    if (gs2[0] != T || gs2[1] != qs[1]) return -1;
    if (ks[0] != n_vis || ks[1] % D != 0) return -1;
    const int Hkv = ks[1] / D;
    if (Hkv <= 0 || H % Hkv != 0 || ss[0] <= 0 || ss[0] > 2048) return -1;
    if (mlx_array_dtype(q) != MLX_FLOAT32 || mlx_array_dtype(k) != MLX_FLOAT32 ||
        mlx_array_dtype(v) != MLX_FLOAT32 || mlx_array_dtype(sel) != MLX_INT32) return -1;
    if (llm_q4n_attn_build() != 0) return -1;
    int ntok = T, nhead = H, nkv = Hkv, hdim = D, nvis = n_vis, nsel = ss[0];
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhead, NULL, 0, MLX_INT32);
    mlx_array akv = mlx_array_new_data(&nkv, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_array an = mlx_array_new_data(&nvis, NULL, 0, MLX_INT32);
    mlx_array as = mlx_array_new_data(&nsel, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, q);
    mlx_vector_array_append_value(in, gate);
    mlx_vector_array_append_value(in, k);
    mlx_vector_array_append_value(in, v);
    mlx_vector_array_append_value(in, sel);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, akv);
    mlx_vector_array_append_value(in, ad);
    mlx_vector_array_append_value(in, an);
    mlx_vector_array_append_value(in, as);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {T, H * D};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)H, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_attn, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(out, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(akv); mlx_array_free(ad);
    mlx_array_free(an); mlx_array_free(as);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_ATTN_KERNEL_H */
