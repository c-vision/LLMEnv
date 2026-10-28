/* qwen4_native_hc_kernel.h — P1: hyper-connection nativa (port di ds4 `metal/qwen4.metal`, MIT).
 * Convenzione `mlx_fast_metal_kernel` (come `llm_mlx_gdn_gate_kernel.h`): gli input sono ARRAY; gli
 * scalari (n_tokens/n_embd/n_hc) sono array 0-dim, referenziati per nome nel corpo del kernel.
 * Out-of-place (Rn = nuovo array), così nessun aliasing in/out.
 *
 * hc_combine: Rn[t][s*E+d] = R[t][s*E+d] + 2*sigmoid(inj[t*hc+s]/hc) * out[t*E+d]
 */
#ifndef LLM_QWEN4_NATIVE_HC_KERNEL_H
#define LLM_QWEN4_NATIVE_HC_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char llm_q4n_hc_header[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n";

static const char llm_q4n_hc_combine_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint E = uint(nembd), hc = uint(nhc), T = uint(ntok);\n"
    "if (i >= T*E) return;\n"
    "uint d = i % E, tok = i / E;\n"
    "float o = out[i];\n"
    "for (uint s = 0; s < hc; s++) { ulong k = ((ulong)tok*hc + s)*E + d;\n"
    "    float g = inj[(ulong)tok*hc + s]/(float)hc;\n"
    "    float sg = g >= 0.f ? 1.f/(1.f+metal::exp(-g)) : metal::exp(g)/(1.f+metal::exp(g));\n"
    "    Rn[k] = R[k] + 2.f*sg*o; }\n";

static mlx_fast_metal_kernel g_q4n_hc_combine;
static int g_q4n_hc_combine_built = 0;

static int llm_q4n_hc_combine_build(void) {
    if (g_q4n_hc_combine_built) return 0;
    const char* in_names[]  = {"R","out","inj","ntok","nembd","nhc"};
    const char* out_names[] = {"Rn"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 6; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_hc_combine = mlx_fast_metal_kernel_new(
        "qwen4_hc_combine_native", iv, ov, llm_q4n_hc_combine_src, llm_q4n_hc_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_hc_combine_built = (g_q4n_hc_combine.ctx != NULL) ? 1 : 0;
    return g_q4n_hc_combine_built ? 0 : -1;
}

/* R:[T][hc*E] f32, out:[T][E] f32, inj:[T][hc] f32 -> *Rn [T][hc*E] f32. 0 = ok. */
static int llm_q4n_hc_combine(mlx_array* Rn, mlx_array R, mlx_array out, mlx_array inj, mlx_stream s) {
    if (!Rn) return -1;
    if (mlx_array_ndim(R) != 2 || mlx_array_ndim(out) != 2 || mlx_array_ndim(inj) != 2) return -1;
    const int* rs = mlx_array_shape(R);
    const int* os = mlx_array_shape(out);
    const int* is = mlx_array_shape(inj);
    const int T = rs[0], hcE = rs[1];
    const int E = os[1];
    if (os[0] != T || is[0] != T) return -1;
    if (hcE % E != 0) return -1;
    const int hc = hcE / E;
    if (is[1] != hc) return -1;
    if (llm_q4n_hc_combine_build() != 0) return -1;
    int ntok = T, nembd = E, nhc = hc;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ae = mlx_array_new_data(&nembd, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhc, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, R);
    mlx_vector_array_append_value(in, out);
    mlx_vector_array_append_value(in, inj);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ae);
    mlx_vector_array_append_value(in, ah);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int oshape[2] = {T, hcE};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, oshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)E, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_hc_combine, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(Rn, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ae); mlx_array_free(ah);
    return rc == 0 ? 0 : -1;
}

/* ---- qwen4_hc_norm (zero-centered RMSNorm per stream, f32, n_inject=0) ----
 * xn[t][s*E+d] = R[t][s*E+d] * rsqrt(mean_d(R^2) + eps) * (1 + gamma[s*E+d]).
 * Un thread per stream (t,s): O(E) due passate. Parità con
 * `llm_mlx_zero_centered_rmsnorm` (group=hidden, calcolo in f32). */
static const char llm_q4n_hc_norm_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint E = uint(nembd), hc = uint(nhc), T = uint(ntok);\n"
    "if (i >= T*hc) return;\n"
    "uint s = i % hc;\n"
    "device const float* r = R + (ulong)i*E;\n"
    "device const float* g = gamma + (ulong)s*E;\n"
    "device float* o = Xn + (ulong)i*E;\n"
    "float ss = 0.f; for (uint d = 0; d < E; d++) ss += r[d]*r[d];\n"
    "float invs = metal::rsqrt(ss/(float)E + eps);\n"
    "for (uint d = 0; d < E; d++) o[d] = r[d]*invs*(1.f + g[d]);\n";

static mlx_fast_metal_kernel g_q4n_hc_norm;
static int g_q4n_hc_norm_built = 0;

static int llm_q4n_hc_norm_build(void) {
    if (g_q4n_hc_norm_built) return 0;
    const char* in_names[]  = {"R","gamma","ntok","nembd","nhc","eps"};
    const char* out_names[] = {"Xn"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 6; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_hc_norm = mlx_fast_metal_kernel_new(
        "qwen4_hc_norm_native", iv, ov, llm_q4n_hc_norm_src, llm_q4n_hc_header,
        true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_hc_norm_built = (g_q4n_hc_norm.ctx != NULL) ? 1 : 0;
    return g_q4n_hc_norm_built ? 0 : -1;
}

/* R:[T][hc*E] f32, gamma:[hc*E] f32, eps -> *Xn [T][hc*E] f32. 0 = ok. */
static int llm_q4n_hc_norm(mlx_array* Xn, mlx_array R, mlx_array gamma, int hc, int E, float eps, mlx_stream s) {
    if (!Xn || mlx_array_ndim(R) != 2 || mlx_array_ndim(gamma) != 1) return -1;
    const int* rs = mlx_array_shape(R);
    const int* gs = mlx_array_shape(gamma);
    const int T = rs[0], hcE = rs[1];
    if (hc <= 0 || E <= 0 || hc * E != hcE) return -1;
    if (gs[0] != hcE || mlx_array_dtype(R) != MLX_FLOAT32) return -1;
    if (llm_q4n_hc_norm_build() != 0) return -1;
    int ntok = T, nembd = E, nhc = hc;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ae = mlx_array_new_data(&nembd, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhc, NULL, 0, MLX_INT32);
    mlx_array aeps = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, R);
    mlx_vector_array_append_value(in, gamma);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ae);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, aeps);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int oshape[2] = {T, hcE};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, oshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)hc, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_hc_norm, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(Xn, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ae); mlx_array_free(ah); mlx_array_free(aeps);
    return rc == 0 ? 0 : -1;
}

/* ---- qwen4_hc_gate_mix (f32 weights) ----
 * mixed[t][d] = (1/hc) * sum_s sigmoid(w_up[s*E+d] . silu(lo[t]/hc)) * xn[t][s*E+d].
 * Parità con il ramo down/up di `qwen4_exp_gated_residual_q` (mix=mean_hc sigmoid(up)*normed),
 * con w_up grezzo f32 (i tipi quantizzati/f16 arrivano in P4/P7). */
static const char llm_q4n_hc_mix_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint E = uint(nembd), hc = uint(nhc), T = uint(ntok), R = uint(nrank);\n"
    "if (i >= T*E) return;\n"
    "uint d = i % E, tok = i / E;\n"
    "ulong lbase = (ulong)tok*R;\n"
    "float g = 0.f;\n"
    "for (uint s = 0; s < hc; s++) {\n"
    "    ulong wbase = ((ulong)(s*E + d))*R;\n"
    "    float acc = 0.f;\n"
    "    for (uint r = 0; r < R; r++) { float x = lo[lbase + r]/(float)hc; float sl = x/(1.f+metal::exp(-x)); acc += w_up[wbase + r]*sl; }\n"
    "    float sg = 1.f/(1.f+metal::exp(-acc));\n"
    "    g += sg * xn[(ulong)tok*hc*E + s*E + d];\n"
    "}\n"
    "mixed[(ulong)tok*E + d] = g/(float)hc;\n";

static mlx_fast_metal_kernel g_q4n_hc_mix;
static int g_q4n_hc_mix_built = 0;

static int llm_q4n_hc_mix_build(void) {
    if (g_q4n_hc_mix_built) return 0;
    const char* in_names[]  = {"xn","lo","w_up","ntok","nembd","nhc","nrank"};
    const char* out_names[] = {"mixed"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 7; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_hc_mix = mlx_fast_metal_kernel_new(
        "qwen4_hc_gate_mix_native", iv, ov, llm_q4n_hc_mix_src, llm_q4n_hc_header,
        true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_hc_mix_built = (g_q4n_hc_mix.ctx != NULL) ? 1 : 0;
    return g_q4n_hc_mix_built ? 0 : -1;
}

/* xn:[T][hc*E] f32, lo:[T][R] f32, w_up:[hc*E][R] f32 -> *mixed [T][E] f32. 0 = ok. */
static int llm_q4n_hc_gate_mix(mlx_array* mixed, mlx_array xn, mlx_array lo, mlx_array w_up,
                               int hc, int E, mlx_stream s) {
    if (!mixed || mlx_array_ndim(xn) != 2 || mlx_array_ndim(lo) != 2 || mlx_array_ndim(w_up) != 2) return -1;
    const int* xs = mlx_array_shape(xn);
    const int* ls = mlx_array_shape(lo);
    const int* ws = mlx_array_shape(w_up);
    const int T = xs[0], hcE = xs[1];
    const int R = ls[1];
    if (hc <= 0 || E <= 0 || hc*E != hcE || ls[0] != T || ws[0] != hcE || ws[1] != R) return -1;
    if (mlx_array_dtype(xn) != MLX_FLOAT32 || mlx_array_dtype(lo) != MLX_FLOAT32 ||
        mlx_array_dtype(w_up) != MLX_FLOAT32) return -1;
    if (llm_q4n_hc_mix_build() != 0) return -1;
    int ntok = T, nembd = E, nhc = hc, nrank = R;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ae = mlx_array_new_data(&nembd, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhc, NULL, 0, MLX_INT32);
    mlx_array ar = mlx_array_new_data(&nrank, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, xn);
    mlx_vector_array_append_value(in, lo);
    mlx_vector_array_append_value(in, w_up);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ae);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ar);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int oshape[2] = {T, E};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, oshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)E, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_hc_mix, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(mixed, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ae); mlx_array_free(ah); mlx_array_free(ar);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_HC_KERNEL_H */
