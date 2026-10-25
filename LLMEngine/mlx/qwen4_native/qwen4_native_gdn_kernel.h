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
    "for (uint vh = h; vh < Hv; vh += Hk) { ulong v0 = (ulong)tok*conv_dim + (ulong)(2u*Hk + vh)*D;\n"
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

/* ---- qwen4_conv_stream ----
 * Depthwise causal conv (K<=4) sull'asse token. x:[T][C] (out-of-place x_o), state:[K-1][C]
 * (out-of-place state_o, history oldest-first avanzata), weight:[C][K]. Un thread per canale. */
static const char llm_q4n_conv_stream_src[] =
    "\n"
    "uint c = uint(thread_position_in_grid.x);\n"
    "uint C = uint(nchan), K = uint(kconv), T = uint(ntok);\n"
    "if (c >= C) return;\n"
    "float win[3];\n"
    "for (uint t = 0; t + 1u < K; t++) win[t] = state[t*C + c];\n"
    "float taps[4];\n"
    "for (uint t = 0; t < K; t++) taps[t] = weight[c*K + t];\n"
    "for (uint tok = 0; tok < T; tok++) {\n"
    "  float raw = x[tok*C + c];\n"
    "  float acc = taps[K-1u]*raw;\n"
    "  for (uint t = 0; t + 1u < K; t++) acc += taps[t]*win[t];\n"
    "  for (uint t = 0; t + 2u < K; t++) win[t] = win[t+1u];\n"
    "  win[K-2u] = raw;\n"
    "  if (apply_silu != 0u) { float sg = acc >= 0.f ? 1.f/(1.f+metal::exp(-acc)) : metal::exp(acc)/(1.f+metal::exp(acc)); acc = acc*sg; }\n"
    "  x_o[tok*C + c] = acc;\n"
    "}\n"
    "for (uint t = 0; t + 1u < K; t++) state_o[t*C + c] = win[t];\n";

static mlx_fast_metal_kernel g_q4n_conv_stream;
static int g_q4n_conv_stream_built = 0;

static int llm_q4n_conv_stream_build(void) {
    if (g_q4n_conv_stream_built) return 0;
    const char* in_names[]  = {"x","state","weight","ntok","nchan","kconv","apply_silu"};
    const char* out_names[] = {"x_o","state_o"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 7; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_conv_stream = mlx_fast_metal_kernel_new(
        "qwen4_conv_stream_native", iv, ov, llm_q4n_conv_stream_src, llm_q4n_gdn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_conv_stream_built = (g_q4n_conv_stream.ctx != NULL) ? 1 : 0;
    return g_q4n_conv_stream_built ? 0 : -1;
}

/* x:[T][C], state:[K-1][C], weight:[C][K] -> *x_o [T][C], *state_o [K-1][C]. 0 = ok. */
static int llm_q4n_conv_stream(mlx_array* x_o, mlx_array* state_o, mlx_array x, mlx_array state,
                               mlx_array weight, int K, int apply_silu, mlx_stream s) {
    if (!x_o || !state_o || mlx_array_ndim(x) != 2 || mlx_array_ndim(state) != 2 || mlx_array_ndim(weight) != 2) return -1;
    const int* xs = mlx_array_shape(x);
    const int* ss = mlx_array_shape(state);
    const int* ws = mlx_array_shape(weight);
    const int T = xs[0], C = xs[1];
    if (K < 1 || K > 4 || ws[0] != C || ws[1] != K || ss[0] != K-1 || ss[1] != C) return -1;
    if (mlx_array_dtype(x) != MLX_FLOAT32) return -1;
    if (llm_q4n_conv_stream_build() != 0) return -1;
    int ntok = T, nchan = C, kconv = K, silu = apply_silu;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ac = mlx_array_new_data(&nchan, NULL, 0, MLX_INT32);
    mlx_array ak = mlx_array_new_data(&kconv, NULL, 0, MLX_INT32);
    mlx_array as = mlx_array_new_data(&silu, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, state);
    mlx_vector_array_append_value(in, weight);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ac);
    mlx_vector_array_append_value(in, ak);
    mlx_vector_array_append_value(in, as);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int xshape[2] = {T, C};
    int ssh[2] = {K-1, C};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, xshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ssh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)C, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_conv_stream, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 2) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 && mlx_vector_array_get(&r1, outs, 1) == 0) {
            mlx_array_set(x_o, r0); mlx_array_set(state_o, r1);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ac); mlx_array_free(ak); mlx_array_free(as);
    return rc == 0 ? 0 : -1;
}

/* ---- qwen4_gdn_scan ----
 * Scan gated-delta sequenziale. state:[Hv][D][D] (riga S[dv][dk]), qkv:[T][conv_dim],
 * ga/gb:[T][Hv] (g, beta), out:[T][Hv*D]. Un thread per (v-head h, dv); S tenuto in `state_o`.
 * Per token: S*=g; u=S·k; delta=(v-u)*beta; S+=k*delta; o=S·q. (snapshot MTP non portati) */
static const char llm_q4n_gdn_scan_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint Hk = uint(nkh), Hv = uint(nvh), D = uint(hdim), T = uint(ntok);\n"
    "if (i >= Hv*D) return;\n"
    "uint dv = i % D, h = i / D;\n"
    "uint kh = h % Hk;\n"
    "uint conv_dim = 2u*Hk*D + Hv*D;\n"
    "device float* so = state_o + ((ulong)h*D + dv)*D;\n"
    "for (uint dk = 0; dk < D; dk++) so[dk] = state[((ulong)h*D + dv)*D + dk];\n"
    "for (uint tok = 0; tok < T; tok++) {\n"
    "  float g = ga[(ulong)tok*Hv + h];\n"
    "  float beta = gb[(ulong)tok*Hv + h];\n"
    "  ulong qb = (ulong)tok*conv_dim + (ulong)kh*D;\n"
    "  ulong kb = qb + (ulong)Hk*D;\n"
    "  float u = 0.f;\n"
    "  for (uint dk = 0; dk < D; dk++) { so[dk] *= g; u += so[dk]*qkv[kb + dk]; }\n"
    "  float v = qkv[(ulong)tok*conv_dim + (ulong)(2u*Hk + h)*D + dv];\n"
    "  float delta = (v - u)*beta;\n"
    "  float o = 0.f;\n"
    "  for (uint dk = 0; dk < D; dk++) { so[dk] += qkv[kb + dk]*delta; o += so[dk]*qkv[qb + dk]; }\n"
    "  out[((ulong)tok*Hv + h)*D + dv] = o;\n"
    "}\n";

static mlx_fast_metal_kernel g_q4n_gdn_scan;
static int g_q4n_gdn_scan_built = 0;

static int llm_q4n_gdn_scan_build(void) {
    if (g_q4n_gdn_scan_built) return 0;
    const char* in_names[]  = {"qkv","ga","gb","state","ntok","nkh","nvh","hdim"};
    const char* out_names[] = {"state_o","out"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 8; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_gdn_scan = mlx_fast_metal_kernel_new(
        "qwen4_gdn_scan_native", iv, ov, llm_q4n_gdn_scan_src, llm_q4n_gdn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_gdn_scan_built = (g_q4n_gdn_scan.ctx != NULL) ? 1 : 0;
    return g_q4n_gdn_scan_built ? 0 : -1;
}

/* qkv:[T][cd], ga/gb:[T][Hv], state:[Hv][D][D] -> *state_o [Hv][D][D], *out [T][Hv*D]. 0 = ok. */
static int llm_q4n_gdn_scan(mlx_array* state_o, mlx_array* out, mlx_array qkv, mlx_array ga, mlx_array gb,
                            mlx_array state, int Hk, int Hv, int D, mlx_stream s) {
    if (!state_o || !out || mlx_array_ndim(qkv) != 2 || mlx_array_ndim(ga) != 2 ||
        mlx_array_ndim(gb) != 2 || mlx_array_ndim(state) != 3) return -1;
    const int* qs = mlx_array_shape(qkv);
    const int* ss = mlx_array_shape(state);
    const int T = qs[0], cd = qs[1];
    if (Hk <= 0 || Hv <= 0 || D <= 0 || cd != 2*Hk*D + Hv*D) return -1;
    if (ss[0] != Hv || ss[1] != D || ss[2] != D) return -1;
    if (mlx_array_dtype(qkv) != MLX_FLOAT32) return -1;
    if (llm_q4n_gdn_scan_build() != 0) return -1;
    int ntok = T, nkh = Hk, nvh = Hv, hdim = D;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array akh = mlx_array_new_data(&nkh, NULL, 0, MLX_INT32);
    mlx_array avh = mlx_array_new_data(&nvh, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, qkv);
    mlx_vector_array_append_value(in, ga);
    mlx_vector_array_append_value(in, gb);
    mlx_vector_array_append_value(in, state);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, akh);
    mlx_vector_array_append_value(in, avh);
    mlx_vector_array_append_value(in, ad);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int stsh[3] = {Hv, D, D};
    int osh[2] = {T, Hv*D};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, stsh, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)Hv * (size_t)D, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_gdn_scan, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 2) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 && mlx_vector_array_get(&r1, outs, 1) == 0) {
            mlx_array_set(state_o, r0); mlx_array_set(out, r1);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(akh); mlx_array_free(avh); mlx_array_free(ad);
    return rc == 0 ? 0 : -1;
}

/* ---- qwen4_gdn_out ----
 * Gated RMS-norm dell'output GDN: o[t][h*D+d] = o * rsqrt(Σo²/D + eps) * weight[d] * sigmoid(z).
 * o:[T][H*D], z:[T][H*D], weight:[D]. Un thread per (tok,head). Out-of-place o_o. */
static const char llm_q4n_gdn_out_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint H = uint(nhead), D = uint(hdim), T = uint(ntok);\n"
    "if (i >= T*H) return;\n"
    "uint h = i % H, tok = i / H;\n"
    "ulong base = ((ulong)tok*H + h)*D;\n"
    "float ss = 0.f; for (uint d = 0; d < D; d++) ss += o[base+d]*o[base+d];\n"
    "float r = metal::rsqrt(ss/(float)D + eps);\n"
    "for (uint d = 0; d < D; d++) {\n"
    "  float zz = z[base+d];\n"
    "  float sg = zz >= 0.f ? 1.f/(1.f+metal::exp(-zz)) : metal::exp(zz)/(1.f+metal::exp(zz));\n"
    "  o_o[base+d] = o[base+d]*r*weight[d]*sg;\n"
    "}\n";

static mlx_fast_metal_kernel g_q4n_gdn_out;
static int g_q4n_gdn_out_built = 0;

static int llm_q4n_gdn_out_build(void) {
    if (g_q4n_gdn_out_built) return 0;
    const char* in_names[]  = {"o","z","weight","ntok","nhead","hdim","eps"};
    const char* out_names[] = {"o_o"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 7; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_gdn_out = mlx_fast_metal_kernel_new(
        "qwen4_gdn_out_native", iv, ov, llm_q4n_gdn_out_src, llm_q4n_gdn_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_gdn_out_built = (g_q4n_gdn_out.ctx != NULL) ? 1 : 0;
    return g_q4n_gdn_out_built ? 0 : -1;
}

/* o/z:[T][H*D], weight:[D], eps -> *o_o [T][H*D]. 0 = ok. */
static int llm_q4n_gdn_out(mlx_array* o_o, mlx_array o, mlx_array z, mlx_array weight,
                           int H, int D, float eps, mlx_stream s) {
    if (!o_o || mlx_array_ndim(o) != 2 || mlx_array_ndim(z) != 2 || mlx_array_ndim(weight) != 1) return -1;
    const int* os_ = mlx_array_shape(o);
    const int* ws = mlx_array_shape(weight);
    const int T = os_[0], hd = os_[1];
    if (H <= 0 || D <= 0 || hd != H*D || ws[0] != D) return -1;
    if (mlx_array_dtype(o) != MLX_FLOAT32) return -1;
    if (llm_q4n_gdn_out_build() != 0) return -1;
    int ntok = T, nhead = H, hdim = D;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhead, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_array aeps = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, o);
    mlx_vector_array_append_value(in, z);
    mlx_vector_array_append_value(in, weight);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ad);
    mlx_vector_array_append_value(in, aeps);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {T, hd};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)H, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_gdn_out, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(o_o, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(ad); mlx_array_free(aeps);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_GDN_KERNEL_H */
