/* llm_mlx_tq_qmv_kernel.h — kernel TQ mat-vec/gather (port di turboquant-mlx/kernels/polar_qmv.py
 * + variante gather) via `mlx_fast_metal_kernel` (pattern llm_mlx_qmm_smallm_kernel.h).
 *
 * Due modalita' (stessa matematica, letta dai pesi PACKED senza materializzare):
 *  - mode 0 (qmv, un esperto):  y[row]  = sum_k dequant(W[ev,row,k])·x[k]
 *  - mode 1 (kg, gather):       y[kk,row] = sum_k dequant(W[indices[kk],row,k])·x[kk,k]
 *    → UN SOLO launch per proiezione (griglia tk*O*32), niente sort/tile, niente eval del host.
 * Pesi: packed [E,O,PW] U32, scales [E,O,NG] F16, codebook [NC] F16. 32 thread/riga, riduzione
 * ad albero in threadgroup. Supporta bit (2/3/4) e base-3 trit (20/U32). Il chiamante ruota
 * l'INPUT (como il riferimento); il kernel non ruota. */
#ifndef LLM_MLX_TQ_QMV_KERNEL_H
#define LLM_MLX_TQ_QMV_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { mlx_fast_metal_kernel k; int bits, group, trit, mode, built, rc; } llm_tq_qmv_cache;
#define LLM_TQ_QMV_NCACHE 16
static llm_tq_qmv_cache g_tq_qmv[LLM_TQ_QMV_NCACHE];
static int g_tq_qmv_ncache = 0;
static char g_tq_qmv_err[512] = { 0 };

static const char llm_tq_qmv_header[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n";

static char* llm_tq_qmv_build_src(int bits, int group, int trit, int mode) {
    int epu = trit ? 20 : (32 / bits);
    int nc = trit ? 3 : (1 << bits);
    const char* addr0 =
        "    const uint row  = threadgroup_position_in_grid.x;\n"
        "    if (row >= (uint)O) return;\n"
        "    const uint kk = 0u;\n"
        "    const uint e  = (uint)ev;\n";
    const char* addr1 =
        "    const uint g   = threadgroup_position_in_grid.x;\n"
        "    const uint kk  = g / (uint)O;\n"
        "    const uint row = g - kk * (uint)O;\n"
        "    const uint e   = (uint)indices[kk];\n";
    const char* xexpr = mode ? "(kk * (uint)Kv + col)" : "col";
    const char* oexpr = mode ? "(kk * (uint)O + row)" : "row";
    char* s = (char*)malloc(4608);
    if (!s) return NULL;
    snprintf(s, 4608,
        "#define LQ_EPU %d\n#define LQ_NC %d\n#define LQ_GS %d\n#define LQ_TRIT %d\n#define LQ_BITS %d\n"
        "    const uint lane = thread_position_in_threadgroup.x;\n"
        "%s"
        "    threadgroup float shared_sums[32];\n"
        "    float cb[LQ_NC];\n"
        "    for (uint i = 0; i < (uint)LQ_NC; i++) cb[i] = (float)codebook[i];\n"
        "    const uint ng = (uint)NG;\n"
        "    const uint pw = (uint)PW;\n"
        "    const uint K = (uint)Kv;\n"
        "    const uint pw_base = (e * (uint)O + row) * pw;\n"
        "    const uint sc_base = (e * (uint)O + row) * ng;\n"
        "    float accum = 0.0f;\n"
        "#if LQ_TRIT\n"
        "    const uint pow3[20] = { 1u,3u,9u,27u,81u,243u,729u,2187u,6561u,19683u,59049u,177147u,"
        "531441u,1594323u,4782969u,14348907u,43046721u,129140163u,387420489u,1162261467u };\n"
        "#endif\n"
        "    for (uint gi = lane; gi < ng; gi += 32u) {\n"
        "        const float scale = (float)scales[sc_base + gi];\n"
        "        const uint base_col = gi * (uint)LQ_GS;\n"
        "        float ga = 0.0f;\n"
        "        for (uint i = 0; i < (uint)LQ_GS; i++) {\n"
        "            const uint col = base_col + i;\n"
        "            if (col >= K) break;\n"
        "            const uint word = packed_weight[pw_base + col / (uint)LQ_EPU];\n"
        "#if LQ_TRIT\n"
        "            const uint code = (word / pow3[col %% (uint)LQ_EPU]) %% 3u;\n"
        "#else\n"
        "            const uint code = (word >> ((col %% (uint)LQ_EPU) * (uint)LQ_BITS)) & (uint)((1u << LQ_BITS) - 1u);\n"
        "#endif\n"
        "            ga += cb[code] * (float)x[%s];\n"
        "        }\n"
        "        accum += ga * scale;\n"
        "    }\n"
        "    shared_sums[lane] = accum;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (lane < 16u) shared_sums[lane] += shared_sums[lane + 16u];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (lane < 8u) shared_sums[lane] += shared_sums[lane + 8u];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (lane < 4u) shared_sums[lane] += shared_sums[lane + 4u];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (lane < 2u) shared_sums[lane] += shared_sums[lane + 2u];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (lane == 0u) out[%s] = T(shared_sums[0] + shared_sums[1]);\n",
        epu, nc, group, trit, bits, (mode ? addr1 : addr0), xexpr, oexpr);
    return s;
}

static void g_tq_qmv_err_cb(const char* msg, void* data) {
    (void)data;
    if (msg != NULL) snprintf(g_tq_qmv_err, sizeof g_tq_qmv_err, "%s", msg);
}

static llm_tq_qmv_cache* llm_tq_qmv_get(int bits, int group, int trit, int mode) {
    for (int i = 0; i < g_tq_qmv_ncache; i++) {
        llm_tq_qmv_cache* c = &g_tq_qmv[i];
        if (c->built && c->bits == bits && c->group == group && c->trit == trit && c->mode == mode) return c;
    }
    if (g_tq_qmv_ncache >= LLM_TQ_QMV_NCACHE) return NULL;
    llm_tq_qmv_cache* c = &g_tq_qmv[g_tq_qmv_ncache++];
    c->bits = bits; c->group = group; c->trit = trit; c->mode = mode;
    char* src = llm_tq_qmv_build_src(bits, group, trit, mode);
    if (!src) { c->built = 1; c->rc = -1; return c; }
    char name[64];
    snprintf(name, sizeof name, "llm_tq_qmv_%s_%d_%d_%d", trit ? "trit" : "b", bits, group, mode);
    const char* in0[] = { "packed_weight", "scales", "codebook", "x", "ev", "Kv", "O", "NG", "PW" };
    const char* in1[] = { "packed_weight", "scales", "codebook", "x", "indices", "Kv", "O", "NG", "PW" };
    const char* out_names[] = { "out" };
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 9; i++) mlx_vector_string_append_value(iv, mode ? in1[i] : in0[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_tq_qmv_err[0] = '\0';
    mlx_set_error_handler(g_tq_qmv_err_cb, NULL, NULL);
    c->k = mlx_fast_metal_kernel_new(name, iv, ov, src, llm_tq_qmv_header,
                                     /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    free(src);
    c->built = 1;
    c->rc = (c->k.ctx == NULL) ? -1 : 0;
    if (c->rc != 0) fprintf(stderr, "[tq-qmv] build '%s' fallita: %s\n", name, g_tq_qmv_err);
    return c;
}

/* mode 0: y [O] = sum dequant(W[ev])·x  (x [K]). */
static int llm_mlx_tq_qmv(mlx_array* y, mlx_array packed, mlx_array scales, mlx_array codebook,
                          mlx_array x, int e, int K, int O, int NG, int PW, int bits, int group, int trit,
                          mlx_stream stream) {
    llm_tq_qmv_cache* c = llm_tq_qmv_get(bits, group, trit, 0);
    if (c == NULL || c->rc != 0) return -1;
    int e_i = e, k_i = K, o_i = O, ng_i = NG, pw_i = PW;
    mlx_array earr  = mlx_array_new_data(&e_i, NULL, 0, MLX_INT32);
    mlx_array Karr  = mlx_array_new_data(&k_i, NULL, 0, MLX_INT32);
    mlx_array Oarr  = mlx_array_new_data(&o_i, NULL, 0, MLX_INT32);
    mlx_array NGarr = mlx_array_new_data(&ng_i, NULL, 0, MLX_INT32);
    mlx_array PWarr = mlx_array_new_data(&pw_i, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, packed);
    mlx_vector_array_append_value(in, scales);
    mlx_vector_array_append_value(in, codebook);
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, earr);
    mlx_vector_array_append_value(in, Karr);
    mlx_vector_array_append_value(in, Oarr);
    mlx_vector_array_append_value(in, NGarr);
    mlx_vector_array_append_value(in, PWarr);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx_array_dtype(x));
    const int osh[1] = { O };
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 1, mlx_array_dtype(x));
    mlx_fast_metal_kernel_config_set_grid(cfg, O * 32, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, c->k, in, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        if (mlx_vector_array_get(&a0, outs, 0) == 0) { mlx_array_set(y, a0); } else { mlx_array_free(a0); ok = false; }
        mlx_array_free(a0);
    }
    if (!ok) fprintf(stderr, "[tq-qmv] apply rc=%d err=%s\n", rc, g_tq_qmv_err);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(outs); mlx_vector_array_free(in);
    mlx_array_free(earr); mlx_array_free(Karr); mlx_array_free(Oarr); mlx_array_free(NGarr); mlx_array_free(PWarr);
    return ok ? 0 : -1;
}

/* mode 1 (gather): y [tk,O] = dequant(W[indices[kk]])·x[kk]; x [tk,K], indices [tk] u32. UN launch. */
static int llm_mlx_tq_qmv_kg(mlx_array* y, mlx_array packed, mlx_array scales, mlx_array codebook,
                             mlx_array x, mlx_array indices, int tk, int K, int O, int NG, int PW,
                             int bits, int group, int trit, mlx_stream stream) {
    llm_tq_qmv_cache* c = llm_tq_qmv_get(bits, group, trit, 1);
    if (c == NULL || c->rc != 0) return -1;
    int k_i = K, o_i = O, ng_i = NG, pw_i = PW;
    mlx_array Karr  = mlx_array_new_data(&k_i, NULL, 0, MLX_INT32);
    mlx_array Oarr  = mlx_array_new_data(&o_i, NULL, 0, MLX_INT32);
    mlx_array NGarr = mlx_array_new_data(&ng_i, NULL, 0, MLX_INT32);
    mlx_array PWarr = mlx_array_new_data(&pw_i, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, packed);
    mlx_vector_array_append_value(in, scales);
    mlx_vector_array_append_value(in, codebook);
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, indices);
    mlx_vector_array_append_value(in, Karr);
    mlx_vector_array_append_value(in, Oarr);
    mlx_vector_array_append_value(in, NGarr);
    mlx_vector_array_append_value(in, PWarr);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx_array_dtype(x));
    const int osh[2] = { tk, O };
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, mlx_array_dtype(x));
    mlx_fast_metal_kernel_config_set_grid(cfg, tk * O * 32, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, c->k, in, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        if (mlx_vector_array_get(&a0, outs, 0) == 0) { mlx_array_set(y, a0); } else { mlx_array_free(a0); ok = false; }
        mlx_array_free(a0);
    }
    if (!ok) fprintf(stderr, "[tq-qmv] kg apply rc=%d err=%s\n", rc, g_tq_qmv_err);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(outs); mlx_vector_array_free(in);
    mlx_array_free(Karr); mlx_array_free(Oarr); mlx_array_free(NGarr); mlx_array_free(PWarr);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_TQ_QMV_KERNEL_H */
