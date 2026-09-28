/* llm_mlx_qmm_smallm_kernel.h — P5 (Splash): quantized matmul AFFINE specializzato per M piccolo
 * (verify DFlash2, M=8). Il QMM di MLX a M=8 e' dequant-bound e gira a ~2,5 TFLOPs (probe
 * llm_mlx_qmm_smallm_probe.c): qui il peso viene dequantizzato una volta in threadgroup memory
 * (bf16) e moltiplicato con simdgroup_matrix MMA (8x8x8), con la x [8 x K] in shared.
 *
 * Semantica IDENTICA a `mlx_quantized_matmul(x, w, scales, biases, transpose=true, group, bits,
 * "affine")` per: bits=8, group=64, x [1,M,K] bf16 (M<=8), w u32 [N, K/4], scales/biases bf16
 * [N, K/64]. Output [1,M,N] bf16.
 *
 * Vincoli (fallback al chiamante se non soddisfatti): M<=8, K%64==0, N%32==0.
 * Layout: y[0, m, n] = sum_k dequant(w[n, k]) * x[0, m, k].
 */
#ifndef LLM_MLX_QMM_SMALLM_KERNEL_H
#define LLM_MLX_QMM_SMALLM_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char llm_qmm_sm_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "#include <metal_simdgroup_matrix>\n"
    "using namespace metal;\n"
;

/* Un threadgroup = 128 thread (4 simdgroup) = tile 8(M) x 32(N); loop K a gruppi di 64. */
static const char llm_qmm_sm_src[] =
    "\n"
    "    constexpr int GM = 8;          // M fisso (verify)\n"
    "    constexpr int GN = 64;         // N per threadgroup\n"
    "    constexpr int GS = 64;         // group size affine\n"
    "    constexpr int SGS = 8;         // simdgroup per threadgroup\n"
    "    constexpr int GNS = GN / SGS;  // 8 colonne per simdgroup\n"
    "    const int tid  = thread_position_in_threadgroup.x;   // 0..255\n"
    "    const int sg   = tid / 32;\n"
    "    const int lane = tid % 32;\n"
    "    const int tg   = threadgroup_position_in_grid.x;\n"
    "    const int n0   = tg * GN;\n"
    "    const int ngrp = K / GS;\n"
    "\n"
    "    threadgroup InT xs[GM][GS + 8];\n"
    "    threadgroup InT ws[GS][GN + 8];\n"
    "    threadgroup float yt[SGS][GM][GNS];\n"
    "\n"
    "    metal::simdgroup_matrix<float, GM, GNS> acc(0.0f);\n"
    "\n"
    "    for (int c = 0; c < ngrp; c++) {\n"
    "        // stage x [GM x GS]: xs[m][k] = x[m, c*GS+k]\n"
    "        for (int i = tid; i < GM * GS; i += 256) {\n"
    "            const int m = i / GS, k = i % GS;\n"
    "            xs[m][k] = x[(size_t)m * K + c * GS + k];\n"
    "        }\n"
    "        // stage w dequant [GS x GN]: ogni thread dequantizza 4 u32 (=16 pesi)\n"
    "        for (int i = tid; i < GN * (GS / 4); i += 256) {\n"
    "            const int n = i / (GS / 4), k4 = i % (GS / 4);\n"
    "            const uint32_t packed = w[(size_t)(n0 + n) * (K / 4) + c * (GS / 4) + k4];\n"
    "            const float sc = (float)scales[(size_t)(n0 + n) * ngrp + c];\n"
    "            const float bi = (float)biases[(size_t)(n0 + n) * ngrp + c];\n"
    "            #pragma unroll\n"
    "            for (int j = 0; j < 4; j++) {\n"
    "                const uint8_t q = (uint8_t)((packed >> (8 * j)) & 0xFFu);\n"
    "                ws[k4 * 4 + j][n] = (InT)((float)q * sc + bi);\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        // MMA: 8 passi k (GS/8), ogni simdgroup le sue GNS colonne\n"
    "        #pragma unroll\n"
    "        for (int kk = 0; kk < GS; kk += 8) {\n"
    "            metal::simdgroup_matrix<InT, GM, 8> a;\n"
    "            metal::simdgroup_matrix<InT, 8, GNS> b;\n"
    "            simdgroup_load(a, &xs[0][kk], GS + 8);\n"
    "            simdgroup_load(b, &ws[kk][sg * GNS], GN + 8);\n"
    "            simdgroup_multiply_accumulate(acc, a, b, acc);\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    // store: acc (float) -> threadgroup -> bf16 y [1,GM,N]\n"
    "    simdgroup_store(acc, &yt[sg][0][0], GNS);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (int i = tid; i < GM * GN; i += 256) {\n"
    "        const int s2 = i / (GM * GNS), r = (i % (GM * GNS)) / GNS, c = i % GNS;\n"
    "        y[(size_t)r * N + n0 + s2 * GNS + c] = (InT)yt[s2][r][c];\n"
    "    }\n"
;

typedef struct { mlx_fast_metal_kernel k; bool built; int rc; } llm_qmm_sm_cache;
static llm_qmm_sm_cache g_qmm_sm = { { NULL }, false, 0 };
static char g_qmm_sm_err[512] = { 0 };
static void g_qmm_sm_err_cb(const char* msg, void* data) {
    (void)data;
    if (msg != NULL) snprintf(g_qmm_sm_err, sizeof g_qmm_sm_err, "%s", msg);
}

static int llm_qmm_sm_build(void) {
    if (g_qmm_sm.built) return g_qmm_sm.rc;
    const char* in_names[]  = { "x", "w", "scales", "biases", "K", "N" };
    const char* out_names[] = { "y" };
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 6; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_qmm_sm_err[0] = '\0';
    mlx_set_error_handler(g_qmm_sm_err_cb, NULL, NULL);
    g_qmm_sm.k = mlx_fast_metal_kernel_new("llm_qmm_smallm", iv, ov, llm_qmm_sm_src, llm_qmm_sm_header,
                                           /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    g_qmm_sm.built = true;
    g_qmm_sm.rc = (g_qmm_sm.k.ctx == NULL) ? -1 : 0;
    return g_qmm_sm.rc;
}

/* y [1,M,N] bf16 = affine8 QMM(x [1,M,K], w [N,K/4] u32, s/b [N,K/64] bf16). 0 ok / -1. */
static int llm_qmm_smallm(mlx_array* y, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                          int M, int K, int N, mlx_stream stream) {
    /* Solo M==8: la MMA e lo store sono 8x8 (M<8 scriverebbe fuori dall'output). */
    if (M != 8 || (K % 64) != 0 || (N % 32) != 0) return -1;
    if (llm_qmm_sm_build() != 0) return -1;
    /* K e N sono input SCALARI (int32, ndim 0): il sorgente li usa come valori. */
    int k_i = K, n_i = N;
    mlx_array Karr = mlx_array_new_data(&k_i, NULL, 0, MLX_INT32);
    mlx_array Narr = mlx_array_new_data(&n_i, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, w);
    mlx_vector_array_append_value(in, scales);
    mlx_vector_array_append_value(in, biases);
    mlx_vector_array_append_value(in, Karr);
    mlx_vector_array_append_value(in, Narr);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx_array_dtype(x));
    const int osh[3] = {1, M, N};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 3, mlx_array_dtype(x));
    mlx_fast_metal_kernel_config_set_grid(cfg, (N / 64) * 256, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_qmm_sm.k, in, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        if (mlx_vector_array_get(&a0, outs, 0) == 0) { mlx_array_set(y, a0); }
        else { mlx_array_free(a0); ok = false; }
        mlx_array_free(a0);
    }
    if (!ok) fprintf(stderr, "[qmm-smallm] apply rc=%d err=%s\n", rc, g_qmm_sm_err);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(outs);
    mlx_vector_array_free(in);
    mlx_array_free(Karr); mlx_array_free(Narr);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_QMM_SMALLM_KERNEL_H */
