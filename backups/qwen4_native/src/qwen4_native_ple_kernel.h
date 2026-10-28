/* qwen4_native_ple_kernel.h — P3: PLE (n-gram embedding + gate + conv) nativo
 * (port di ds4 `metal/qwen4.metal`: kernel_qwen4_ple_gate / kernel_qwen4_ple_conv).
 *
 * Convenzione `mlx_fast_metal_kernel` (come P2): input = array, scalari = array 0-dim,
 * output out-of-place, niente definizioni di funzioni nel body.
 *
 * PLE (config reale): hc = 4 gruppi, E = 2560, heads = (ngram-1)*heads_per_ngram = 16,
 * conv depthwise K = 4 dilatata di ngram = 3 (history = 9 righe), eps = rms_norm_eps.
 * Gli snapshot MTP del conv non sono portati (P6).
 */
#ifndef LLM_QWEN4_NATIVE_PLE_KERNEL_H
#define LLM_QWEN4_NATIVE_PLE_KERNEL_H

#include "mlx/c/mlx.h"

static const char llm_q4n_ple_header[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n";

/* ---- qwen4_ple_gate ----
 * Per (tok, gruppo s): normalizza key e R, gate = sigmoid(sign*sqrt(|dot|)), gated = gate*value,
 * poi normed = gated * rsqrt(mean(gated^2)+eps) * g_conv.  Un thread per (tok, s). */
static const char llm_q4n_ple_gate_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint T = uint(ntok), H = uint(nhc), E = uint(embd);\n"
    "if (i >= T*H) return;\n"
    "uint s = i % H, tok = i / H;\n"
    "ulong kb = ((ulong)tok*H + (ulong)s)*E;\n"
    "ulong vb = (ulong)tok*E;\n"
    "float sk = 0.f, sr = 0.f;\n"
    "for (uint d = 0; d < E; d++) { float k = key[kb+d]; float r = R[kb+d]; sk += k*k; sr += r*r; }\n"
    "float ik = metal::rsqrt(sk/(float)E + eps);\n"
    "float ir = metal::rsqrt(sr/(float)E + eps);\n"
    "float dot = 0.f;\n"
    "for (uint d = 0; d < E; d++)\n"
    "  dot += (key[kb+d]*ik*g_key[(ulong)s*E+d]) * (R[kb+d]*ir*g_query[(ulong)s*E+d]);\n"
    "float gsc = dot * metal::rsqrt((float)E);\n"
    "float mag = metal::sqrt(metal::max(metal::fabs(gsc), 1e-6f));\n"
    "float x = gsc > 0.f ? mag : (gsc < 0.f ? -mag : 0.f);\n"
    "float sg = x >= 0.f ? 1.f/(1.f+metal::exp(-x)) : metal::exp(x)/(1.f+metal::exp(x));\n"
    "float sv = 0.f;\n"
    "for (uint d = 0; d < E; d++) { float v = sg*value[vb+d]; gated[kb+d] = v; sv += v*v; }\n"
    "float ig = metal::rsqrt(sv/(float)E + eps);\n"
    "for (uint d = 0; d < E; d++) normed[kb+d] = gated[kb+d]*ig*g_conv[(ulong)s*E+d];\n";

static mlx_fast_metal_kernel g_q4n_ple_gate;
static int g_q4n_ple_gate_built = 0;

static int llm_q4n_ple_gate_build(void) {
    if (g_q4n_ple_gate_built) return 0;
    const char* in_names[]  = {"R","key","value","g_key","g_query","g_conv","ntok","nhc","embd","eps"};
    const char* out_names[] = {"gated","normed"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 10; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_ple_gate = mlx_fast_metal_kernel_new(
        "qwen4_ple_gate_native", iv, ov, llm_q4n_ple_gate_src, llm_q4n_ple_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_ple_gate_built = (g_q4n_ple_gate.ctx != NULL) ? 1 : 0;
    return g_q4n_ple_gate_built ? 0 : -1;
}

/* R/key/gated/normed:[T][hc*E] f32, value:[T][E], g_*:[hc*E]. */
static int llm_q4n_ple_gate(mlx_array* gated, mlx_array* normed,
                            mlx_array R, mlx_array key, mlx_array value,
                            mlx_array g_key, mlx_array g_query, mlx_array g_conv,
                            float eps, mlx_stream s) {
    if (!gated || !normed) return -1;
    if (mlx_array_ndim(R) != 2 || mlx_array_ndim(key) != 2 || mlx_array_ndim(value) != 2) return -1;
    if (mlx_array_ndim(g_key) != 1 || mlx_array_ndim(g_query) != 1 || mlx_array_ndim(g_conv) != 1) return -1;
    const int* rs = mlx_array_shape(R);
    const int* ks = mlx_array_shape(key);
    const int* vs = mlx_array_shape(value);
    const int T = rs[0], hcE = rs[1], E = vs[1];
    if (E <= 0 || hcE <= 0 || hcE % E != 0) return -1;
    const int hc = hcE / E;
    if (ks[0] != T || ks[1] != hcE || vs[0] != T) return -1;
    const int* gs = mlx_array_shape(g_key);
    if (gs[0] != hcE) return -1;
    if (mlx_array_dtype(R) != MLX_FLOAT32 || mlx_array_dtype(key) != MLX_FLOAT32 ||
        mlx_array_dtype(value) != MLX_FLOAT32) return -1;
    if (llm_q4n_ple_gate_build() != 0) return -1;
    int ntok = T, nhc = hc, embd = E;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhc, NULL, 0, MLX_INT32);
    mlx_array ae = mlx_array_new_data(&embd, NULL, 0, MLX_INT32);
    mlx_array aeps = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, R);
    mlx_vector_array_append_value(in, key);
    mlx_vector_array_append_value(in, value);
    mlx_vector_array_append_value(in, g_key);
    mlx_vector_array_append_value(in, g_query);
    mlx_vector_array_append_value(in, g_conv);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ae);
    mlx_vector_array_append_value(in, aeps);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int gshape[2] = {T, hcE};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, gshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, gshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)hc, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_ple_gate, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 2) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 && mlx_vector_array_get(&r1, outs, 1) == 0) {
            mlx_array_set(gated, r0); mlx_array_set(normed, r1);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(ae); mlx_array_free(aeps);
    return rc == 0 ? 0 : -1;
}

/* ---- qwen4_ple_conv ----
 * R += gated + silu(conv dilatata di normed); history [(K-1)*dil][C] oldest-first avanzata.
 * Un thread per canale, token sequenziali. Pesos taps f32. */
static const char llm_q4n_ple_conv_src[] =
    "\n"
    "uint c = uint(thread_position_in_grid.x);\n"
    "uint C = uint(nchan), K = uint(kconv), D = uint(dil), T = uint(ntok);\n"
    "if (c >= C || K < 1u || K > 4u) return;\n"
    "uint Hh = (K - 1u)*D;\n"
    "if (Hh > 9u) return;\n"
    "float hist[9];\n"
    "for (uint t = 0; t < Hh; t++) hist[t] = history[t*C + c];\n"
    "float taps[4];\n"
    "for (uint t = 0; t < K; t++) taps[t] = weight[(ulong)c*K + t];\n"
    "for (uint tok = 0; tok < T; tok++) {\n"
    "  float cur = normed[(ulong)tok*C + c];\n"
    "  float acc = taps[K-1u]*cur;\n"
    "  for (uint k = 0; k + 1u < K; k++) acc += taps[k]*hist[Hh - (K - 1u - k)*D];\n"
    "  for (uint t = 0; t + 1u < Hh; t++) hist[t] = hist[t+1u];\n"
    "  hist[Hh-1u] = cur;\n"
    "  float sg = acc >= 0.f ? 1.f/(1.f+metal::exp(-acc)) : metal::exp(acc)/(1.f+metal::exp(acc));\n"
    "  R_o[(ulong)tok*C + c] = R[(ulong)tok*C + c] + gated[(ulong)tok*C + c] + acc*sg;\n"
    "}\n"
    "for (uint t = 0; t < Hh; t++) history_o[t*C + c] = hist[t];\n";

static mlx_fast_metal_kernel g_q4n_ple_conv;
static int g_q4n_ple_conv_built = 0;

static int llm_q4n_ple_conv_build(void) {
    if (g_q4n_ple_conv_built) return 0;
    const char* in_names[]  = {"R","gated","normed","history","weight","ntok","nchan","kconv","dil"};
    const char* out_names[] = {"R_o","history_o"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 9; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_ple_conv = mlx_fast_metal_kernel_new(
        "qwen4_ple_conv_native", iv, ov, llm_q4n_ple_conv_src, llm_q4n_ple_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_ple_conv_built = (g_q4n_ple_conv.ctx != NULL) ? 1 : 0;
    return g_q4n_ple_conv_built ? 0 : -1;
}

/* R/gated/normed:[T][C], history:[H][C], weight:[C][K]. */
static int llm_q4n_ple_conv(mlx_array* R_o, mlx_array* history_o,
                            mlx_array R, mlx_array gated, mlx_array normed,
                            mlx_array history, mlx_array weight,
                            int K, int dilation, mlx_stream s) {
    if (!R_o || !history_o) return -1;
    if (mlx_array_ndim(R) != 2 || mlx_array_ndim(gated) != 2 || mlx_array_ndim(normed) != 2) return -1;
    if (mlx_array_ndim(history) != 2 || mlx_array_ndim(weight) != 2) return -1;
    const int* rs = mlx_array_shape(R);
    const int* hs = mlx_array_shape(history);
    const int* ws = mlx_array_shape(weight);
    const int T = rs[0], C = rs[1];
    if (K < 1 || K > 4 || dilation < 1) return -1;
    const int Hh = (K - 1) * dilation;
    if (Hh > 9 || hs[0] != Hh || hs[1] != C || ws[0] != C || ws[1] != K) return -1;
    if (mlx_array_dtype(R) != MLX_FLOAT32 || mlx_array_dtype(weight) != MLX_FLOAT32) return -1;
    if (llm_q4n_ple_conv_build() != 0) return -1;
    int ntok = T, nchan = C, kconv = K, dil = dilation;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ac = mlx_array_new_data(&nchan, NULL, 0, MLX_INT32);
    mlx_array ak = mlx_array_new_data(&kconv, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&dil, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, R);
    mlx_vector_array_append_value(in, gated);
    mlx_vector_array_append_value(in, normed);
    mlx_vector_array_append_value(in, history);
    mlx_vector_array_append_value(in, weight);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ac);
    mlx_vector_array_append_value(in, ak);
    mlx_vector_array_append_value(in, ad);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int rshape[2] = {T, C};
    int hshape[2] = {Hh, C};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, rshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, hshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)C, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_ple_conv, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 2) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 && mlx_vector_array_get(&r1, outs, 1) == 0) {
            mlx_array_set(R_o, r0); mlx_array_set(history_o, r1);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ac); mlx_array_free(ak); mlx_array_free(ad);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_PLE_KERNEL_H */
