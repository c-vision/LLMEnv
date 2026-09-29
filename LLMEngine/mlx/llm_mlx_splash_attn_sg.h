/* llm_mlx_splash_attn_sg.h — read KV int8 con sgmatrix MMA (register 8x8).
 *
 * Da sous#126: la via giusta (non MPP) e' il kernel simdgroup_matrix MMA via
 * mlx_fast_metal_kernel. Qui il QK e il PV dell'attenzione usano
 * sgmatrix::mma_acc (simdgroup_multiply_accumulate), fedele a
 * common/sgmatrix.h di Splash.
 *
 * Config: M=8 righe query, N=32 token, D=head_dim.
 * score[m][n] = sum_d Q[m][d] * (K[n][d] * kscale[n]) * attn_scale
 * out[m][d]   = sum_n softmax(score)[m][n] * (V[n][d] * vscale[n])
 *
 * Lane map (sgmatrix::lane_map): lane -> (fm, fn). MMA C[8x8] += A[8x8] x B[8x8]:
 *  a-fragment = A[fm][fn], A[fm][fn+1] ; b-fragment = B[fm][fn], B[fm][fn+1].
 * QK: A = Q [M x Dchunk], B = K^T [Dchunk x N] -> b = {K[n0+fn][k0+fm], K[n0+fn+1][k0+fm]}
 * PV: A = P [M x Nchunk], B = V [Nchunk x D]  -> b = {V[n0+fm][k0+fn], V[n0+fm][k0+fn+1]}
 */
#ifndef LLM_MLX_SPLASH_ATTN_SG_H
#define LLM_MLX_SPLASH_ATTN_SG_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdint.h>
#include "llm_mlx_splash_store.h"   /* usa i metodi di main (store/pool) */
#include "llm_mlx_splash_attn.h"    /* usa i metodi di main (read) */

static const char llm_splash_attn_sg_src[] =
    "  constexpr uint SR = 8u, PT = 32u;                 /* 8 righe, 32 token */\n"
    "  const uint D = Dhead;\n"
    "  const uint qh = threadgroup_position_in_grid.x;   /* un threadgroup per query head */\n"
    "  const uint sg = simdgroup_index_in_threadgroup;\n"
    "  const uint lane = thread_index_in_simdgroup;\n"
    "  const uint qid = lane >> 2;\n"
    "  const uint fm = (qid & 4u) | ((lane >> 1) & 3u);\n"
    "  const uint fn0 = ((qid & 2u) << 1) | ((lane & 1u) << 1);\n"
    "  const uint kh = 0u;\n"
    "  threadgroup float psh[SR * PT];\n"
    "  /* QK: 4 simdgroup coprono i 32 token; gli altri solo la PV */\n"
    "  const uint n0 = sg * 8u;\n"
    "  float2 accq = float2(0.0f);\n    if (sg < PT / 8u) {\n"
    "  for (uint k0 = 0; k0 + 8u <= D; k0 += 8u) {\n"
    "    float2 a; float2 b;\n"
    "    a.x = (float)q[(qh*SR + fm)*D + (k0 + fn0)]; a.y = (float)q[(qh*SR + fm)*D + (k0 + fn0 + 1u)];\n"
    "    b.x = (float)kpool[(kh*capi + n0 + fn0)*D + (k0 + fm)] * ks[kh*capi + n0 + fn0];\n"
    "    b.y = (float)kpool[(kh*capi + n0 + fn0 + 1u)*D + (k0 + fm)] * ks[kh*capi + n0 + fn0 + 1u];\n"
    "    sgmatrix::mma_acc<float>(accq, a, b);\n"
    "  }\n"
    "  /* accq = {C[fm][fn0], C[fm][fn0+1]} = score[fm][n0+fn0], [fm][n0+fn0+1] */\n"
    "    psh[(fm)*PT + (n0 + fn0)] = accq.x * attn_scale;\n"
    "    psh[(fm)*PT + (n0 + fn0 + 1u)] = accq.y * attn_scale;\n    }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* ---- softmax per riga (thread 0) ---- */\n"
    "  if (thread_index_in_threadgroup == 0) {\n"
    "    for (uint m = 0; m < SR; ++m) {\n"
    "      float mx = -INFINITY; for (uint n = 0; n < PT; ++n) mx = max(mx, psh[m*PT+n]);\n"
    "      float sum = 0.0f; for (uint n = 0; n < PT; ++n) { float e = exp(psh[m*PT+n]-mx); psh[m*PT+n]=e; sum+=e; }\n"
    "      float inv = sum > 0.0f ? 1.0f/sum : 0.0f; for (uint n = 0; n < PT; ++n) psh[m*PT+n] *= inv;\n"
    "    }\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* ---- PV: ogni simdgroup copre 8 dimensioni out ---- */\n"
    "  const uint d0 = sg * 8u;\n"
    "  float2 accv = float2(0.0f);\n"
    "  for (uint k0 = 0; k0 + 8u <= PT; k0 += 8u) {\n"
    "    float2 a; float2 b;\n"
    "    a.x = psh[fm*PT + (k0 + fn0)]; a.y = psh[fm*PT + (k0 + fn0 + 1u)];\n"
    "    b.x = (float)vpool[(kh*D + d0 + fn0)*capi + (k0 + fm)] * ks[kh*capi + k0 + fm];\n"
    "    b.y = (float)vpool[(kh*D + d0 + fn0 + 1u)*capi + (k0 + fm)] * ks[kh*capi + k0 + fm];\n"
    "    sgmatrix::mma_acc<float>(accv, a, b);\n"
    "  }\n"
    "  out[(qh*SR+fm)*D + d0 + fn0] = accv.x;\n"
    "  out[(qh*SR+fm)*D + d0 + fn0 + 1u] = accv.y;\n";
static mlx_fast_metal_kernel g_splash_attn_sg; static int g_splash_attn_sg_built = 0;
static inline int llm_splash_attn_sg_build(void) {
    if (g_splash_attn_sg_built) return g_splash_attn_sg.ctx ? 0 : -1;
    g_splash_attn_sg_built = 1;
    const char* hdr = "#include <metal_stdlib>\nusing namespace metal;\n"
        "namespace sgmatrix {\n"
        "inline ushort lfm(uint lane) { const uint qid = lane >> 2; return ushort((qid & 4u) | ((lane >> 1) & 3u)); }\n"
        "inline ushort lfn(uint lane) { const uint qid = lane >> 2; return ushort(((qid & 2u) << 1) | ((lane & 1u) << 1)); }\n"
        "template <typename T> __attribute__((always_inline)) inline thread vec<T,2> &te(thread simdgroup_matrix<T,8,8> &m) { return reinterpret_cast<thread vec<T,2> &>(m.thread_elements()); }\n"
        "template <typename T> __attribute__((always_inline)) inline void mma_acc(thread float2 &c, vec<T,2> a, vec<T,2> b) {\n"
        "  simdgroup_matrix<T,8,8> A,B; simdgroup_matrix<float,8,8> C,D; te(A)=a; te(B)=b; te(C)=c; simdgroup_multiply_accumulate(D,A,B,C); c = te(D); }\n"
        "}\n";
    mlx_vector_string iv = mlx_vector_string_new();
    const char* ins[] = {"q","kpool","vpool","ks","nc","capi","attn_scale"};
    for (int i = 0; i < 7; i++) mlx_vector_string_append_value(iv, ins[i]);
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, "out");
    g_splash_attn_sg = mlx_fast_metal_kernel_new("splash_attn_sg", iv, ov, llm_splash_attn_sg_src, hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    return g_splash_attn_sg.ctx ? 0 : -1;
}

static inline int llm_splash_attn_sg_run(mlx_array* out, mlx_array q, mlx_array kpool, mlx_array vpool,
                                         mlx_array ks, int nc, int cap, float attn_scale, mlx_stream s) {
    if (llm_splash_attn_sg_build() != 0) return -1;
    int nh = (int)mlx_array_shape(q)[0], hd = (int)mlx_array_shape(q)[1];
    mlx_array ncA = mlx_array_new_int(nc), capA = mlx_array_new_int(cap), asA = mlx_array_new_float(attn_scale);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, q); mlx_vector_array_append_value(in, kpool);
    mlx_vector_array_append_value(in, vpool); mlx_vector_array_append_value(in, ks);
    mlx_vector_array_append_value(in, ncA); mlx_vector_array_append_value(in, capA);
    mlx_vector_array_append_value(in, asA);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dhead", hd);
    int osh[2] = {nh, hd}; mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, nh * 256, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_splash_attn_sg, in, cfg, s);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) { mlx_array a0 = mlx_array_new(); ok = mlx_vector_array_get(&a0, outs, 0) == 0; if (ok) mlx_array_set(out, a0); mlx_array_free(a0); }
    mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(in); mlx_vector_array_free(outs);
    mlx_array_free(ncA); mlx_array_free(capA); mlx_array_free(asA);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_SPLASH_ATTN_SG_H */
