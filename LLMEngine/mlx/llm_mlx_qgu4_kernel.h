/* llm_mlx_qgu4_kernel.h — gate_proj + up_proj + SwiGLU in UN SOLO dispatch (affine 4-bit g64, M piccolo 2..8, MMA 8x8).
 *
 * Perche': nel verify a M=8 il MLP denso lancia gate (N=17408) e up (stesso x), poi sigmoid*mul*mul: 3 matmul/elementwise indipendenti,
 * ~54-58% del tempo GPU (profilo per primitiva). Qui i due matmul condividono i frammenti B (x^T, e la somma di x via MMA con la matrice di uni),
 * la dequantizzazione e' in registri (nibble esatti in bf16) e l'epilogo calcola direttamente act = silu(gate)*up (arrotondando gate e up a bf16
 * PRIMA, come i due output di MLX, e poi sigmoid/mul/mul con arrotondamento bf16 a ogni passo: stessa sequenza del path eager).
 * Niente scrittura/rilettura di gate_out/up_out, niente 3 dispatch elementwise.
 *
 * Mapping lane->frammento e permutazione di k: vedi llm_mlx_qgemv4_kernel.h (parcheggiato) — A = W (n=fm, k_virt=fn,fn+1) con 8 byte/lane/gruppo,
 * B = x^T con k = g*64 + (fm/2)*16 + 2j + fm%2.
 * Contratto: x [.., M, K] bf16, wg/wu u32 [N, K/8], sg/bg/su/bu bf16 [N, K/64]; K%64==0, N%16==0. Output act [1,M,N] bf16. Ritorna -1 se non applicabile. */
#ifndef LLM_MLX_QGU4_KERNEL_H
#define LLM_MLX_QGU4_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char llm_qgu4_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "#include <metal_simdgroup_matrix>\n"
    "using namespace metal;\n"
;

static const char llm_qgu4_src[] =
    "\n"
    "    const int sg = simdgroup_index_in_threadgroup;\n"
    "    const int lane = thread_index_in_simdgroup;\n"
    "    const int nbase = ((int)threadgroup_position_in_grid.x * SG + sg) * (8 * NT);\n"
    "    if (nbase >= N) return;\n"
    "    const short qid = lane / 4;\n"
    "    const short fm = (qid & 4) + ((lane / 2) % 4);\n"
    "    const short fn = (qid & 2) * 2 + (lane % 2) * 2;\n"
    "    const short p = fn / 2;\n"
    "    const short pp = fm / 2, ee = fm % 2;\n"
    "    const int KP = K / 8, NG = K / 64;\n"
    "    const bool v0 = fn < M, v1 = (fn + 1) < M;\n"
    "    const device bfloat* xr0 = x + (size_t)(v0 ? fn : 0) * K;\n"
    "    const device bfloat* xr1 = x + (size_t)(v1 ? fn + 1 : 0) * K;\n"
    "    const device uint* gro[NT]; const device uint* uro[NT]; size_t nrow[NT];\n"
    "    for (int t = 0; t < NT; t++) { nrow[t] = (size_t)(nbase + t * 8 + fm); gro[t] = wg + nrow[t] * KP; uro[t] = wu + nrow[t] * KP; }\n"
    "    simdgroup_matrix<float, 8, 8> totg[NT], totu[NT];\n"
    "    for (int t = 0; t < NT; t++) { totg[t] = simdgroup_matrix<float, 8, 8>(0.0f); totu[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n"
    "    simdgroup_matrix<bfloat, 8, 8> ones;\n"
    "    ones.thread_elements()[0] = (bfloat)1.0f; ones.thread_elements()[1] = (bfloat)1.0f;\n"
    "    for (int g = 0; g < NG; g++) {\n"
    "        const int kb = g * 64;\n"
    "        uint2 pg[NT], pu[NT];\n"
    "        for (int t = 0; t < NT; t++) {\n"
    "            pg[t] = *((const device uint2*)(gro[t] + (kb + p * 16) / 8));\n"
    "            pu[t] = *((const device uint2*)(uro[t] + (kb + p * 16) / 8));\n"
    "        }\n"
    "        simdgroup_matrix<float, 8, 8> aqg[NT], aqu[NT], ax(0.0f);\n"
    "        for (int t = 0; t < NT; t++) { aqg[t] = simdgroup_matrix<float, 8, 8>(0.0f); aqu[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n"
    "        #pragma unroll\n"
    "        for (int j = 0; j < 8; j++) {\n"
    "            const int sh = (j & 3) * 8;\n"
    "            simdgroup_matrix<bfloat, 8, 8> B;\n"
    "            const int kx = kb + pp * 16 + 2 * j + ee;\n"
    "            B.thread_elements()[0] = v0 ? xr0[kx] : (bfloat)0.0f;\n"
    "            B.thread_elements()[1] = v1 ? xr1[kx] : (bfloat)0.0f;\n"
    "            simdgroup_multiply_accumulate(ax, ones, B, ax);\n"
    "            for (int t = 0; t < NT; t++) {\n"
    "                const uint wgd = (j < 4) ? pg[t].x : pg[t].y;\n"
    "                const uint wud = (j < 4) ? pu[t].x : pu[t].y;\n"
    "                simdgroup_matrix<bfloat, 8, 8> A, U;\n"
    "                A.thread_elements()[0] = (bfloat)(float)((wgd >> sh) & 0xF);\n"
    "                A.thread_elements()[1] = (bfloat)(float)((wgd >> (sh + 4)) & 0xF);\n"
    "                U.thread_elements()[0] = (bfloat)(float)((wud >> sh) & 0xF);\n"
    "                U.thread_elements()[1] = (bfloat)(float)((wud >> (sh + 4)) & 0xF);\n"
    "                simdgroup_multiply_accumulate(aqg[t], A, B, aqg[t]);\n"
    "                simdgroup_multiply_accumulate(aqu[t], U, B, aqu[t]);\n"
    "            }\n"
    "        }\n"
    "        for (int t = 0; t < NT; t++) {\n"
    "            const float sgs = (float)sg_[nrow[t] * NG + g], bgs = (float)bg_[nrow[t] * NG + g];\n"
    "            const float sus = (float)su_[nrow[t] * NG + g], bus = (float)bu_[nrow[t] * NG + g];\n"
    "            totg[t].thread_elements()[0] += sgs * aqg[t].thread_elements()[0] + bgs * ax.thread_elements()[0];\n"
    "            totg[t].thread_elements()[1] += sgs * aqg[t].thread_elements()[1] + bgs * ax.thread_elements()[1];\n"
    "            totu[t].thread_elements()[0] += sus * aqu[t].thread_elements()[0] + bus * ax.thread_elements()[0];\n"
    "            totu[t].thread_elements()[1] += sus * aqu[t].thread_elements()[1] + bus * ax.thread_elements()[1];\n"
    "        }\n"
    "    }\n"
    "    for (int t = 0; t < NT; t++) {\n"
    "        for (int e = 0; e < 2; e++) {\n"
    "            if (e == 0 ? !v0 : !v1) continue;\n"
    "            const float gb = (float)(bfloat)totg[t].thread_elements()[e];\n"
    "            const float ub = (float)(bfloat)totu[t].thread_elements()[e];\n"
    "            const float sig = (float)(bfloat)(1.0f / (1.0f + exp(-gb)));\n"
    "            const float sil = (float)(bfloat)(gb * sig);\n"
    "            y[(size_t)(fn + e) * N + nbase + t * 8 + fm] = (bfloat)(sil * ub);\n"
    "        }\n"
    "    }\n"
;

typedef struct { mlx_fast_metal_kernel k; bool built; int rc; } llm_qgu4_cache;
static llm_qgu4_cache g_qgu4 = { { NULL }, false, 0 };

static int llm_qgu4_build(void) {
    if (g_qgu4.built) return g_qgu4.rc;
    const char* in_names[] = { "x", "wg", "sg_", "bg_", "wu", "su_", "bu_", "K", "N", "M" };
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 10; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, "y");
    g_qgu4.k = mlx_fast_metal_kernel_new("llm_qgu4_mma", iv, ov, llm_qgu4_src, llm_qgu4_header,
                                         /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    g_qgu4.built = true;
    g_qgu4.rc = (g_qgu4.k.ctx == NULL) ? -1 : 0;
    return g_qgu4.rc;
}

/* act [1,M,N] bf16 = silu(gate(x)) * up(x), gate/up affine4/g64. M in 2..8. 0 = ok, -1 = non applicabile/errore. */
static int llm_qgu4(mlx_array* act, mlx_array x, mlx_array wg, mlx_array sg, mlx_array bg,
                    mlx_array wu, mlx_array su, mlx_array bu, int M, int K, int N, mlx_stream stream) {
    const int SG = 4, NT = 2;
    if (M < 2 || M > 8 || (K % 64) != 0 || (N % (8 * NT)) != 0) return -1;
    if (llm_qgu4_build() != 0) return -1;
    int k_i = K, n_i = N, m_i = M;
    mlx_array Karr = mlx_array_new_data(&k_i, NULL, 0, MLX_INT32);
    mlx_array Narr = mlx_array_new_data(&n_i, NULL, 0, MLX_INT32);
    mlx_array Marr = mlx_array_new_data(&m_i, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, wg); mlx_vector_array_append_value(in, sg); mlx_vector_array_append_value(in, bg);
    mlx_vector_array_append_value(in, wu); mlx_vector_array_append_value(in, su); mlx_vector_array_append_value(in, bu);
    mlx_vector_array_append_value(in, Karr);
    mlx_vector_array_append_value(in, Narr);
    mlx_vector_array_append_value(in, Marr);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SG", SG);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NT", NT);
    const int osh[3] = {1, M, N};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 3, MLX_BFLOAT16);
    const int tgs = (N / (8 * NT) + SG - 1) / SG;
    mlx_fast_metal_kernel_config_set_grid(cfg, tgs * SG * 32, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, SG * 32, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_qgu4.k, in, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        if (mlx_vector_array_get(&a0, outs, 0) == 0) mlx_array_set(act, a0); else ok = false;
        mlx_array_free(a0);
    }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(outs);
    mlx_vector_array_free(in);
    mlx_array_free(Karr); mlx_array_free(Narr); mlx_array_free(Marr);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_QGU4_KERNEL_H */
