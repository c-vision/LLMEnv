/* llm_mlx_gdn_decode_fused.h — KERNEL GDN DECODE FUSO (S=1) come nodo lazy del grafo MLX.
 *
 * Un solo mlx_fast_metal_kernel al posto di ~45 op MLX per layer lineare: conv1d depthwise (kk=4) + silu, split q/k/v,
 * l2-norm di q/k, gating (beta, g), passo ricorrente gated-delta, RMSNormGated con silu/sigmoid(z). Un threadgroup da 1024
 * thread per testa v (Hv threadgroup); i 1024 thread = 128 righe dv x 8 segmenti da 16 dk (stesso layout del kernel prefill).
 * Vincoli: Dk = Dv = 128, kk = 4, B = 1, tutto f32. Fuori da questi il chiamante usa il percorso a op MLX. */
#ifndef LLM_MLX_GDN_DECODE_FUSED_H
#define LLM_MLX_GDN_DECODE_FUSED_H

#include "mlx/c/mlx.h"
#include <pthread.h>
#include <string.h>

static const char llm_gdn_dec_src[] =
"uint tid = thread_position_in_threadgroup.x; uint hv = threadgroup_position_in_grid.x;\n"
"uint lane = thread_index_in_simdgroup; uint sgid = simdgroup_index_in_threadgroup;\n"
"uint Hk = PR[0], Hv = PR[1]; uint key_dim = PR[2], conv_dim = PR[4]; uint use_silu = PR[5];\n"
"const float eps = FP[0], inv_scale = FP[1];\n"
"uint rep = Hv / Hk; uint hk = hv / rep;\n"
"threadgroup float qs[128]; threadgroup float ks[128]; threadgroup float vs[128]; threadgroup float yv[128]; threadgroup float part[32];\n"
"// A) conv1d depthwise (kernel 4, stride 1) + silu sui 384 canali q/k/v di questa testa; aggiorna conv_state (ultimi 3 ingressi)\n"
"if (tid < 384) {\n"
"    uint pt = tid / 128, i = tid % 128;\n"
"    uint c = pt == 0 ? hk * 128 + i : (pt == 1 ? key_dim + hk * 128 + i : 2 * key_dim + hv * 128 + i);\n"
"    float x3 = QKV[c]; float c0 = CS[c], c1 = CS[conv_dim + c], c2 = CS[2 * conv_dim + c];\n"
"    float acc = CW[c * 4 + 0] * c0 + CW[c * 4 + 1] * c1 + CW[c * 4 + 2] * c2 + CW[c * 4 + 3] * x3;\n"
"    float sv = acc / (1.0f + exp(-acc));\n"
"    if (pt == 0) qs[i] = sv; else if (pt == 1) ks[i] = sv; else vs[i] = sv;\n"
"    if (pt == 2 || (hv % rep) == 0) { CSO[c] = c1; CSO[conv_dim + c] = c2; CSO[2 * conv_dim + c] = x3; }\n"
"}\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"// B) l2-norm di q e k (somma dei quadrati + eps, come il percorso a op), q scalato per Dk^-0.5\n"
"float sq = 0.0f; if (tid < 256) { float v = tid < 128 ? qs[tid] : ks[tid - 128]; sq = v * v; }\n"
"float ssum = simd_sum(sq); if (lane == 0) part[sgid] = ssum;\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"float rq = rsqrt(part[0] + part[1] + part[2] + part[3] + 1e-6f) * inv_scale;\n"
"float rk = rsqrt(part[4] + part[5] + part[6] + part[7] + 1e-6f);\n"
"// C) gating: beta = sigmoid(b); g = exp(-exp(A_log) * softplus(a + dt_bias))\n"
"float bt = 1.0f / (1.0f + exp(-BV[hv]));\n"
"float xa = AV[hv] + DT[hv]; float sp = max(xa, 0.0f) + log(1.0f + exp(-fabs(xa)));\n"
"float gg = exp(-EA[hv] * sp);\n"
"// D) passo ricorrente gated-delta (T=1): riga dv x segmento di 16 dk\n"
"uint dv = tid / 8, seg = tid % 8, d0 = seg * 16;\n"
"float st[16]; float kf[16]; float qf[16];\n"
"const device float* sin_ = ST + ((ulong)hv * 128 + dv) * 128 + d0;\n"
"for (uint i = 0; i < 16; i++) { st[i] = sin_[i] * gg; kf[i] = ks[d0 + i] * rk; qf[i] = qs[d0 + i] * rq; }\n"
"float p = 0.0f; for (uint i = 0; i < 16; i++) p += st[i] * kf[i];\n"
"p += simd_shuffle_xor(p, 4); p += simd_shuffle_xor(p, 2); p += simd_shuffle_xor(p, 1);\n"
"float delta = (vs[dv] - p) * bt;\n"
"float o = 0.0f; for (uint i = 0; i < 16; i++) { st[i] += kf[i] * delta; o += st[i] * qf[i]; }\n"
"o += simd_shuffle_xor(o, 4); o += simd_shuffle_xor(o, 2); o += simd_shuffle_xor(o, 1);\n"
"device float* sout = STO + ((ulong)hv * 128 + dv) * 128 + d0;\n"
"for (uint i = 0; i < 16; i++) sout[i] = st[i];\n"
"if (seg == 0) yv[dv] = o;\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"// E) RMSNormGated: rms_norm(y, w, eps) * (silu|sigmoid)(z)\n"
"float yy = tid < 128 ? yv[tid] : 0.0f; float ys = simd_sum(yy * yy); if (lane == 0) part[sgid] = ys;\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"float rs = rsqrt((part[0] + part[1] + part[2] + part[3]) * (1.0f / 128.0f) + eps);\n"
"if (tid < 128) {\n"
"    float zz = Z[hv * 128 + tid]; float act = use_silu ? zz / (1.0f + exp(-zz)) : 1.0f / (1.0f + exp(-zz));\n"
"    Y[hv * 128 + tid] = yv[tid] * rs * NW[tid] * act;\n"
"}\n";

static mlx_fast_metal_kernel g_llm_gdn_dec_k;
static int g_llm_gdn_dec_ready;
static pthread_mutex_t g_llm_gdn_dec_mu = PTHREAD_MUTEX_INITIALIZER;

/* Tutti gli ingressi f32 e contigui: qkv[conv_dim] z[value_dim] a[Hv] b[Hv] cs[3*conv_dim] cw[conv_dim*4] ea[Hv] dt[Hv] st[Hv*128*128] nw[128].
 * Uscite: y[value_dim], st_out[Hv*128*128], cs_out[3*conv_dim]. 0 = ok. */
static inline int llm_mlx_gdn_decode_fused(mlx_array* y, mlx_array* st_out, mlx_array* cs_out,
                                           mlx_array qkv, mlx_array z, mlx_array a, mlx_array b, mlx_array cs, mlx_array cw,
                                           mlx_array ea, mlx_array dt, mlx_array st, mlx_array nw,
                                           int Hk, int Hv, int key_dim, int value_dim, int conv_dim, int use_silu, float eps, mlx_stream s) {
    pthread_mutex_lock(&g_llm_gdn_dec_mu);
    if (!g_llm_gdn_dec_ready) {
        static const char* const ins[] = { "QKV", "Z", "AV", "BV", "CS", "CW", "EA", "DT", "ST", "NW", "PR", "FP" };
        static const char* const outs[] = { "Y", "STO", "CSO" };
        mlx_vector_string vi = mlx_vector_string_new(), vo = mlx_vector_string_new();
        for (int i = 0; i < 12; i++) mlx_vector_string_append_value(vi, ins[i]);
        for (int i = 0; i < 3; i++) mlx_vector_string_append_value(vo, outs[i]);
        g_llm_gdn_dec_k = mlx_fast_metal_kernel_new("llm_gdn_decode_fused", vi, vo, llm_gdn_dec_src, "", true, false);
        mlx_vector_string_free(vi); mlx_vector_string_free(vo);
        g_llm_gdn_dec_ready = 1;
    }
    mlx_fast_metal_kernel k = g_llm_gdn_dec_k;
    pthread_mutex_unlock(&g_llm_gdn_dec_mu);

    int32_t pr[8] = { Hk, Hv, key_dim, value_dim, conv_dim, use_silu, 0, 0 };
    float fp[2] = { eps, 0.08838834764831845f /* 128^-0.5 */ };
    int psh[1] = { 8 }, fsh[1] = { 2 };
    mlx_array PR = mlx_array_new_data(pr, psh, 1, MLX_INT32), FP = mlx_array_new_data(fp, fsh, 1, MLX_FLOAT32);
    mlx_vector_array vi = mlx_vector_array_new();
    mlx_array ins[12] = { qkv, z, a, b, cs, cw, ea, dt, st, nw, PR, FP };
    for (int i = 0; i < 12; i++) mlx_vector_array_append_value(vi, ins[i]);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int ysh[1] = { value_dim }, ssh[4] = { 1, Hv, 128, 128 }, csh[3] = { 1, 3, conv_dim };
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ysh, 1, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ssh, 4, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, csh, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, Hv * 1024, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1);
    mlx_vector_array vo = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&vo, k, vi, cfg, s);
    if (rc == 0) { mlx_vector_array_get(y, vo, 0); mlx_vector_array_get(st_out, vo, 1); mlx_vector_array_get(cs_out, vo, 2); }
    mlx_vector_array_free(vo); mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(vi);
    mlx_array_free(PR); mlx_array_free(FP);
    return rc;
}

#endif
