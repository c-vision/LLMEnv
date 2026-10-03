/* qwen4_native_gdn_kernel.h — P2: gated delta net nativo (port di ds4 `metal/qwen4.metal`, MIT).
 * Convenzione `mlx_fast_metal_kernel`: input = array, scalari = array 0-dim, output out-of-place,
 * niente definizioni di funzioni nel body (le formula sono inline). */
#ifndef LLM_QWEN4_NATIVE_GDN_KERNEL_H
#define LLM_QWEN4_NATIVE_GDN_KERNEL_H

#include "mlx/c/mlx.h"

static const char llm_q4n_gdn_header[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n";

/* ---- qwen4_gdn_prep ----
 * Dopo la conv: L2-normalizza q e k (q anche *1/sqrt(D)), e trasforma alpha/beta in
 * g = exp(ssm_a * softplus(a + dt_bias)) e beta = sigmoid(b).  qkv:[T][2*Hk*D+Hv*D],
 * a/b:[T][Hv] (in/out, out-of-place).  Un thread per (tok,k-head). */
static const char llm_q4n_gdn_prep_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint T = uint(ntok), Hk = uint(nkh), Hv = uint(nvh), D = uint(hdim);\n"
    "if (i >= T*Hk) return;\n"
    "uint h = i % Hk, tok = i / Hk;\n"
    "uint conv_dim = 2u*Hk*D + Hv*D;\n"
    "ulong q0 = (ulong)tok*conv_dim + (ulong)h*D;\n"
    "ulong k0 = q0 + (ulong)Hk*D;\n"
    "float sq = 0.f, sk = 0.f;\n"
    "for (uint d = 0; d < D; d++) { sq += qkv[q0+d]*qkv[q0+d]; sk += qkv[k0+d]*qkv[k0+d]; }\n"
    "float qs = metal::rsqrt(sq + 1e-6f) * metal::rsqrt((float)D);\n"
    "float ks = metal::rsqrt(sk + 1e-6f);\n"
    "for (uint d = 0; d < D; d++) { qkv_o[q0+d] = qkv[q0+d]*qs; qkv_o[k0+d] = qkv[k0+d]*ks; }\n"
    "if (h < Hv) { ulong v0 = (ulong)tok*conv_dim + (ulong)(2u*Hk + h)*D;\n"
    "  for (uint d = 0; d < D; d++) qkv_o[v0+d] = qkv[v0+d]; }\n"
    "if (h == 0u) {\n"
    "  for (uint j = 0; j < Hv; j++) {\n"
    "    float x = a[(ulong)tok*Hv + j] + dt_bias[j];\n"
    "    float sp = x > 20.f ? x : (x < -20.f ? metal::exp(x) : metal::log(1.f + metal::exp(x)));\n"
    "    a_o[(ulong)tok*Hv + j] = metal::exp(ssm_a[j]*sp);\n"
    "    float bv = b[(ulong)tok*Hv + j];\n"
    "    float sg = bv >= 0.f ? 1.f/(1.f+metal::exp(-bv)) : metal::exp(bv)/(1.f+metal::exp(bv));\n"
    "    b_o[(ulong)tok*Hv + j] = sg;\n"
    "  }\n"
    "}\n";

static mlx_fast_metal_kernel g_q4n_gdn_prep;
static int g_q4n_gdn_prep_built = 0;

static int llm_q4n_gdn_prep_build(void) {
    if (g_q4n_gdn_prep_built) return 0;
    const char* in_names[]  = {"qkv","a","b","ssm_a","dt_bias","ntok","nkh","nvh","hdim"};
    const char* out_names[] = {"qkv_o","a_o","b_o"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 9; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 3; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_gdn_prep = mlx_fast_metal_kernel_new(
        "qwen4_gdn_prep_native", iv, ov, llm_q4n_gdn_prep_src, llm_q4n_gdn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_gdn_prep_built = (g_q4n_gdn_prep.ctx != NULL) ? 1 : 0;
    return g_q4n_gdn_prep_built ? 0 : -1;
}

/* qkv:[T][cd] f32, a/b:[T][Hv] f32, ssm_a/dt_bias:[Hv] f32.
 * Out: *qkv_o [T][cd], *a_o/*b_o [T][Hv]. 0 = ok. */
static int llm_q4n_gdn_prep(mlx_array* qkv_o, mlx_array* a_o, mlx_array* b_o,
                            mlx_array qkv, mlx_array a, mlx_array b,
                            mlx_array ssm_a, mlx_array dt_bias,
                            int Hk, int Hv, int D, mlx_stream s) {
    if (!qkv_o || !a_o || !b_o) return -1;
    const int* qs = mlx_array_shape(qkv);
    const int* as = mlx_array_shape(a);
    if (mlx_array_ndim(qkv) != 2 || mlx_array_ndim(a) != 2 || mlx_array_ndim(b) != 2) return -1;
    if (mlx_array_ndim(ssm_a) != 1 || mlx_array_ndim(dt_bias) != 1) return -1;
    const int T = qs[0], cd = qs[1];
    if (Hk <= 0 || Hv <= 0 || D <= 0 || cd != 2*Hk*D + Hv*D) return -1;
    if (as[0] != T || as[1] != Hv) return -1;
    if (mlx_array_dtype(qkv) != MLX_FLOAT32 || mlx_array_dtype(a) != MLX_FLOAT32) return -1;
    if (llm_q4n_gdn_prep_build() != 0) return -1;
    int ntok = T, nkh = Hk, nvh = Hv, hdim = D;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array akh = mlx_array_new_data(&nkh, NULL, 0, MLX_INT32);
    mlx_array avh = mlx_array_new_data(&nvh, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, qkv);
    mlx_vector_array_append_value(in, a);
    mlx_vector_array_append_value(in, b);
    mlx_vector_array_append_value(in, ssm_a);
    mlx_vector_array_append_value(in, dt_bias);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, akh);
    mlx_vector_array_append_value(in, avh);
    mlx_vector_array_append_value(in, ad);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int qshape[2] = {T, cd};
    int ashape[2] = {T, Hv};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, qshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ashape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ashape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)Hk, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_gdn_prep, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 3) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new(), r2 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 &&
            mlx_vector_array_get(&r1, outs, 1) == 0 &&
            mlx_vector_array_get(&r2, outs, 2) == 0) {
            mlx_array_set(qkv_o, r0); mlx_array_set(a_o, r1); mlx_array_set(b_o, r2);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1); mlx_array_free(r2);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(akh); mlx_array_free(avh); mlx_array_free(ad);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_GDN_KERNEL_H */
