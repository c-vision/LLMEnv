/* llm_mlx_mpp_q4_kernel.h — GEMM q4 affine (4-bit, group 64) su MPP (tensor unit Apple).
 *
 * Kernel generato per (K, N, M), con M in {8,16,32,64}, TileN=128 e G=K/64 gruppi. Usa le tensor-op MPP
 * (`matmul2d`), abilitate da `-std=metal4.0` che `mlx_fast_metal_kernel` imposta su macOS 26+
 * (`mlx/backend/metal/device.cpp:get_metal_version()`). NIENTE metallib/bridge: MLX binda i
 * buffer GPU delle mlx_array da sé.
 *
 * Layout pesi = il NOSTRO: w [N, K/8] U32 (n-major, k contiguo, nibble low-first), scales/biases
 * [N, K/64] bf16. L'input x è [1,T,K] con T<=64 e viene paddato al minimo M valido >= T.
 * Lo scope cooperativo MPP richiede M multiplo di 8; M non multipli (es. 9) non compilano.
 *
 * Flag: LLMMETAL_MPP_Q4 (default OFF finché non validato end-to-end). */
#ifndef LLM_MLX_MPP_Q4_KERNEL_H
#define LLM_MLX_MPP_Q4_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LLM_MPP_Q4_MAX 32
typedef struct { mlx_fast_metal_kernel k; int K, N, M, GS; } llm_mpp_q4_cache;
static llm_mpp_q4_cache g_mpp_q4[LLM_MPP_Q4_MAX];
static int g_mpp_q4_n = 0;

static const char* llm_mpp_q4_hdr =
    "#include <metal_stdlib>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace metal;\n"
    "using namespace mpp::tensor_ops;\n";

static const char* llm_mpp_q4_body =
    "  constexpr uint TileN=128, GS=GSSZ, G=K/GS;\n"
    "  uint tid=thread_index_in_threadgroup; uint tile=threadgroup_position_in_grid.x;\n"
    "  threadgroup float sums_g[M];\n"
    "  auto a=tensor((device bfloat*)x, dextents<int,2>{int(K),int(M)}, array<int,2>{1,int(K)});\n"
    "  constexpr auto desc=matmul2d_descriptor(M,128,GS,false,true,false);\n"
    "  matmul2d<desc,execution_simdgroups<8>> op;\n"
    "  auto a0=a.slice<GS,M>(0,0);\n"
    "  tensor<device uint4b_format,dextents<int,2>,tensor_inline> b0((device uchar*)w+ulong(tile)*TileN*K/2, dextents<int,2>{GS,128}, array<int,2>{1,int(K)});\n"
    "  auto b00=b0.slice<GS,128>(0,0);\n"
    "  auto acc=op.get_destination_cooperative_tensor<decltype(a0),decltype(b00),float>();\n"
    "  for(uint i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) acc[i]=0.0f;\n"
    "  auto part=op.get_destination_cooperative_tensor<decltype(a0),decltype(b00),float>();\n"
    "  for(uint g=0;g<G;++g){\n"
    "    if(tid<M){ float ss=0.0f; for(uint k=0;k<GS;++k) ss+=float(x[tid*K+g*GS+k]); sums_g[tid]=ss; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    auto as=a.slice<GS,M>(g*GS,0);\n"
    "    tensor<device uint4b_format,dextents<int,2>,tensor_inline> bg((device uchar*)w+ulong(tile)*TileN*K/2+ulong(g)*GS/2, dextents<int,2>{GS,128}, array<int,2>{1,int(K)});\n"
    "    auto bs=bg.slice<GS,128>(0,0);\n"
    "    for(uint i=0;i<part.get_capacity();++i) if(part.is_valid_element(i)) part[i]=0.0f;\n"
    "    op.run(as,bs,part);\n"
    "    for(uint i=0;i<part.get_capacity();++i) if(part.is_valid_element(i)){ auto idx=part.get_multidimensional_index(i);\n"
    "      uint nl=idx[0], mm=idx[1]; uint n=tile*TileN+nl; acc[i]+=part[i]*float(sc[n*G+g])+sums_g[mm]*float(bi[n*G+g]); }\n    threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "  for(uint i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)){ auto idx=acc.get_multidimensional_index(i);\n"
    "    out[idx[1]*int(N)+int(tile)*TileN+idx[0]]=bfloat(acc[i]); }\n";

/* Ritorna il kernel per (K,N), compilandolo la prima volta. NULL su errore. */
static mlx_fast_metal_kernel llm_mpp_q4_get(int K, int N, int M, int GS) {
    for (int i = 0; i < g_mpp_q4_n; i++)
        if (g_mpp_q4[i].K == K && g_mpp_q4[i].N == N && g_mpp_q4[i].M == M && g_mpp_q4[i].GS == GS) return g_mpp_q4[i].k;
    size_t bl = strlen(llm_mpp_q4_body) + 64;
    char* body = (char*)malloc(bl);
    if (!body) return (mlx_fast_metal_kernel){0};
    snprintf(body, bl, "%s", llm_mpp_q4_body);
    char* src = (char*)malloc(bl + 64);
    if (!src) { free(body); return (mlx_fast_metal_kernel){0}; }
    snprintf(src, bl + 64, "  constexpr uint K=%d, N=%d, M=%d, GSSZ=%d;\n%s", K, N, M, GS, body);
    free(body);
    if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr, "=== MPP Q4 SRC (K=%d N=%d M=%d GS=%d) ===\n%s\n===\n", K, N, M, GS, src);
    mlx_vector_string iv = mlx_vector_string_new();
    const char* in[] = {"x", "w", "sc", "bi"};
    for (int i = 0; i < 4; i++) mlx_vector_string_append_value(iv, in[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, "out");
    char kname[64]; snprintf(kname, sizeof(kname), "llmmetal_mpp_q4_K%d_N%d_M%d_GS%d", K, N, M, GS);
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new(kname, iv, ov, src, llm_mpp_q4_hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    free(src);
    if (!k.ctx) return (mlx_fast_metal_kernel){0};
    if (g_mpp_q4_n < LLM_MPP_Q4_MAX) { g_mpp_q4[g_mpp_q4_n].k = k; g_mpp_q4[g_mpp_q4_n].K = K; g_mpp_q4[g_mpp_q4_n].N = N; g_mpp_q4[g_mpp_q4_n].M = M; g_mpp_q4[g_mpp_q4_n].GS = GS; g_mpp_q4_n++; }
    return k;
}

/* y = x @ dequant(w)^T, x [1,T,K] (T<=64), w [N,K/8] U32, sc/bi [N,K/64] bf16 -> out [1,T,N].
 * Ritorna false se non applicabile (il chiamante usa la qmm generica). */
static bool llm_mlx_mpp_q4_linear(mlx_array* out, mlx_array x, mlx_array w, mlx_array sc, mlx_array bi,
                                  int group_size, int bits, int K, int N, mlx_stream s) {
    if ((group_size != 64 && group_size != 32) || bits != 4) return false;
    if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr, "[mpp-dtype] sc=%d bi=%d w=%d x=%d (f32=%d bf16=%d u32=%d)\n", (int)mlx_array_dtype(sc), (int)mlx_array_dtype(bi), (int)mlx_array_dtype(w), (int)mlx_array_dtype(x), (int)MLX_FLOAT32, (int)MLX_BFLOAT16, (int)MLX_UINT32);
    if (mlx_array_dtype(sc) != MLX_BFLOAT16 || mlx_array_dtype(bi) != MLX_BFLOAT16) return false;
    if (K % group_size != 0 || N % 128 != 0 || K <= 0) return false;
    if (mlx_array_ndim(x) != 3) return false;
    int T = (int)mlx_array_shape(x)[1];
    if (T < 1 || T > 64) return false;
    int M = (T <= 8) ? 8 : (T <= 16) ? 16 : (T <= 32) ? 32 : 64;
    mlx_fast_metal_kernel k = llm_mpp_q4_get(K, N, M, group_size);
    if (!k.ctx) return false;
    /* Pad al minimo tile MPP valido, senza cambiare le T righe restituite. */
    mlx_array x2 = mlx_array_new();
    if (T < M) {
        int zsh[3] = {1, M - T, K};
        mlx_array z = mlx_array_new();
        if (mlx_zeros(&z, zsh, 3, mlx_array_dtype(x), s) != 0) { mlx_array_free(z); return false; }
        mlx_vector_array cv = mlx_vector_array_new_data((mlx_array[]){x, z}, 2);
        int rc = mlx_concatenate_axis(&x2, cv, 1, s);
        mlx_vector_array_free(cv); mlx_array_free(z);
        if (rc != 0) return false;
    } else { mlx_array_set(&x2, x); }
    mlx_array x3 = mlx_array_new(); int sh2[2] = {M, K};
    if (mlx_reshape(&x3, x2, sh2, 2, s) != 0) { mlx_array_free(x2); return false; }
    mlx_array_free(x2);
    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, x3);
    mlx_vector_array_append_value(inputs, w);
    mlx_vector_array_append_value(inputs, sc);
    mlx_vector_array_append_value(inputs, bi);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {M, N};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_BFLOAT16);
    /* FIX 2026-09-22: set_grid = numero di THREAD (non threadgroup). Passando N/128 il kernel
     * girava con UN SOLO threadgroup (tile 0) -> output ~0: e' la ragione per cui questa via
     * (mlx_fast_metal_kernel) dava "MPP ~0 silenzioso". Il numero di threadgroup in x e'
     * grid.x/threadgroup.x = N/128. */
    mlx_fast_metal_kernel_config_set_grid(cfg, (N / 128) * 256, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr, "[mpp-apply] K=%d N=%d M=%d GS=%d T=%d\n", K, N, M, group_size, T);
    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, k, inputs, cfg, s);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_array_free(x3);
    bool ok = (rc == 0 && mlx_vector_array_size(outputs) == 1);
    if (ok) {
        mlx_array y = mlx_array_new();
        if (mlx_vector_array_get(&y, outputs, 0) != 0) ok = false;
        else {
            /* [M,N] -> [1,M,N] -> slice [0,T) */
            mlx_array y3 = mlx_array_new(); int sh3[3] = {1, M, N};
            if (mlx_reshape(&y3, y, sh3, 3, s) != 0) ok = false;
            else {
                if (T < M) {
                    int st[3] = {0, 0, 0}, sp[3] = {1, T, N}, ss[3] = {1, 1, 1};
                    mlx_array yt = mlx_array_new();
                    if (mlx_slice(&yt, y3, st, 3, sp, 3, ss, 3, s) == 0) mlx_array_set(out, yt);
                    else ok = false;
                    mlx_array_free(yt);
                } else mlx_array_set(out, y3);
            }
            mlx_array_free(y3);
        }
        mlx_array_free(y);
    }
    mlx_vector_array_free(outputs);
    return ok;
}

#endif /* LLM_MLX_MPP_Q4_KERNEL_H */
