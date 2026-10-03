/* llm_mlx_tq_qmv_kernel.h — kernel TQ mat-vec/gather (port di turboquant-mlx/kernels/polar_qmv.py
 * + variante gather) via `mlx_fast_metal_kernel` (pattern llm_mlx_qmm_smallm_kernel.h).
 *
 * Due modalita' (stessa matematica, letta dai pesi PACKED senza materializzare):
 *  - mode 0 (qmv, un esperto):  y[row]  = sum_k dequant(W[ev,row,k])·x[k]
 *  - mode 1 (kg, gather):       y[kk,row] = sum_k dequant(W[indices[kk],row,k])·x[kk,k]
 *    → UN SOLO launch per proiezione (griglia tk*O*32), niente sort/tile, niente eval del host.
 * Pesi: packed [E,O,PW] U32, scales [E,O,NG] F16, codebook [NC] F16. 32 thread/riga, riduzione
 * ad albero in threadgroup. K/O/NG/PW sono letti dalle SHAPE built-in (`<name>_shape`), cosi' non
 * servono input scalari (meno alloc per call). Supporta bit (2/3/4) e base-3 trit (20/U32). Il
 * chiamante ruota l'INPUT (come il riferimento); il kernel non ruota. */
#ifndef LLM_MLX_TQ_QMV_KERNEL_H
#define LLM_MLX_TQ_QMV_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int O, rows, grid, K; mlx_dtype dt; mlx_fast_metal_kernel_config cfg; } llm_tq_qmv_cfg;
#define LLM_TQ_CFG_MAX 512
typedef struct {
    mlx_fast_metal_kernel k; int bits, group, trit, mode, K, built, rc;
    llm_tq_qmv_cfg cfgs[LLM_TQ_CFG_MAX]; int ncfg;   /* config riusabili per (O,rows,grid,dtype) */
} llm_tq_qmv_cache;
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
    /* TILED: 256 thread = 8 simdgroup = 8 righe per threadgroup. La x del routing (K valori) e'
     * caricata UNA volta in threadgroup memory e riusata dalle 8 righe (prima era riletta per
     * riga dalla global). `K` e' un template arg (dimensione compilata dell'array shared). */
    const char* expr_e = mode ? "(uint)indices[kk]" : "(uint)ev";
    const char* xload  = mode ? "x[kk * K + i]" : "x[i]";
    const char* oexpr  = mode ? "(kk * O + row)" : "row";
    char* s = (char*)malloc(4608);
    if (!s) return NULL;
    snprintf(s, 4608,
        "#define LQ_EPU %d\n#define LQ_NC %d\n#define LQ_GS %d\n#define LQ_TRIT %d\n#define LQ_BITS %d\n"
        "    const uint tid  = thread_position_in_threadgroup.x;\n"
        "    const uint sg   = tid >> 5u;\n"
        "    const uint lane = tid & 31u;\n"
        "    const uint O  = packed_weight_shape[packed_weight_ndim - 2];\n"
        "    const uint pw = packed_weight_shape[packed_weight_ndim - 1];\n"
        "    const uint ng = scales_shape[scales_ndim - 1];\n"
        "    const uint g0 = threadgroup_position_in_grid.x * 8u + sg;\n"
        "    const uint kk = g0 / O;\n"
        "    const uint row = g0 - kk * O;\n"
        "    const uint e = %s;\n"
        "    threadgroup float sx[K];\n"
        "    for (uint i = tid; i < K; i += 256u) sx[i] = (float)%s;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    float cb[LQ_NC];\n"
        "    for (uint i = 0; i < (uint)LQ_NC; i++) cb[i] = (float)codebook[i];\n"
        "    const uint pw_base = (e * O + row) * pw;\n"
        "    const uint sc_base = (e * O + row) * ng;\n"
        "    float accum = 0.0f;\n"
        "    const uint nwords = (K + (uint)LQ_EPU - 1u) / (uint)LQ_EPU;\n"
        "    for (uint w = lane; w < nwords; w += 32u) {\n"
        "        uint word = packed_weight[pw_base + w];\n"
        "        const uint base_col = w * (uint)LQ_EPU;\n"
        "        for (uint i = 0; i < (uint)LQ_EPU; i++) {\n"
        "            const uint col = base_col + i;\n"
        "            if (col >= K) break;\n"
        "#if LQ_TRIT\n"
        "            const uint code = word %% 3u; word /= 3u;\n"
        "#else\n"
        "            const uint code = word & (uint)((1u << LQ_BITS) - 1u); word >>= LQ_BITS;\n"
        "#endif\n"
        "            accum += cb[code] * (float)scales[sc_base + col / (uint)LQ_GS] * sx[col];\n"
        "        }\n"
        "    }\n"
        "    float v = accum;\n"
        "#pragma unroll\n"
        "    for (uint o = 16u; o > 0u; o >>= 1) v += simd_shuffle_xor(v, o);\n"
        "    if (lane == 0u && row < O) out[%s] = T(v);\n",
        epu, nc, group, trit, bits, expr_e, xload, oexpr);
    return s;
}

static void g_tq_qmv_err_cb(const char* msg, void* data) {
    (void)data;
    if (msg != NULL) snprintf(g_tq_qmv_err, sizeof g_tq_qmv_err, "%s", msg);
}

static llm_tq_qmv_cache* llm_tq_qmv_get(int bits, int group, int trit, int mode, int K) {
    for (int i = 0; i < g_tq_qmv_ncache; i++) {
        llm_tq_qmv_cache* c = &g_tq_qmv[i];
        if (c->built && c->bits == bits && c->group == group && c->trit == trit && c->mode == mode && c->K == K) return c;
    }
    if (g_tq_qmv_ncache >= LLM_TQ_QMV_NCACHE) return NULL;
    llm_tq_qmv_cache* c = &g_tq_qmv[g_tq_qmv_ncache++];
    c->bits = bits; c->group = group; c->trit = trit; c->mode = mode; c->K = K;
    char* src = llm_tq_qmv_build_src(bits, group, trit, mode);
    if (!src) { c->built = 1; c->rc = -1; return c; }
    char name[64];
    snprintf(name, sizeof name, "llm_tq_qmv_%s_%d_%d_%d_K%d", trit ? "trit" : "b", bits, group, mode, K);
    const char* in_names[] = { "packed_weight", "scales", "codebook", "x", mode ? "indices" : "ev" };
    const char* out_names[] = { "out" };
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 5; i++) mlx_vector_string_append_value(iv, in_names[i]);
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

/* Config del kernel per (O, grid, dtype): si RIPETE a ogni token -> cache (altrimenti config
 * new/add_output/set_grid/free a ogni chiamata = centinaia per token nel path denso). */
static mlx_fast_metal_kernel_config llm_tq_qmv_cfg_get(llm_tq_qmv_cache* c, int O, int rows, int grid, int K, mlx_dtype dt, int* is_temp) {
    for (int i = 0; i < c->ncfg; i++)
        if (c->cfgs[i].O == O && c->cfgs[i].rows == rows && c->cfgs[i].grid == grid && c->cfgs[i].K == K && c->cfgs[i].dt == dt) { *is_temp = 0; return c->cfgs[i].cfg; }
    mlx_fast_metal_kernel_config cf = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cf, "T", dt);
    mlx_fast_metal_kernel_config_add_template_arg_int(cf, "K", K);
    int osh[2]; int nd;
    if (c->mode) { osh[0] = rows; osh[1] = O; nd = 2; } else { osh[0] = O; nd = 1; }
    mlx_fast_metal_kernel_config_add_output_arg(cf, osh, nd, dt);
    mlx_fast_metal_kernel_config_set_grid(cf, grid, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cf, 256, 1, 1);
    if (c->ncfg < LLM_TQ_CFG_MAX) {
        c->cfgs[c->ncfg].O = O; c->cfgs[c->ncfg].rows = rows; c->cfgs[c->ncfg].grid = grid; c->cfgs[c->ncfg].K = K; c->cfgs[c->ncfg].dt = dt; c->cfgs[c->ncfg].cfg = cf; c->ncfg++;
        *is_temp = 0;
    } else *is_temp = 1;
    return cf;
}

static int llm_tq_qmv_apply(mlx_array* y, llm_tq_qmv_cache* c, mlx_array packed, mlx_array scales,
                            mlx_array codebook, mlx_array x, mlx_array arg5, int O, int rows, int grid_extra, int K,
                            mlx_stream stream) {
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, packed);
    mlx_vector_array_append_value(in, scales);
    mlx_vector_array_append_value(in, codebook);
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, arg5);
    int is_temp = 0;
    mlx_fast_metal_kernel_config cfg = llm_tq_qmv_cfg_get(c, O, rows, grid_extra, K, mlx_array_dtype(x), &is_temp);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, c->k, in, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        if (mlx_vector_array_get(&a0, outs, 0) == 0) { mlx_array_set(y, a0); } else { mlx_array_free(a0); ok = false; }
        mlx_array_free(a0);
    }
    if (!ok) fprintf(stderr, "[tq-qmv] apply rc=%d err=%s\n", rc, g_tq_qmv_err);
    if (is_temp) mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(outs); mlx_vector_array_free(in);
    return ok ? 0 : -1;
}

static mlx_array g_tq_zero;   /* scalare e=0 cachato (mode 0: quasi sempre 0) */
/* mode 0: y [O] = sum dequant(W[e])·x  (x [K]). K/O/NG/PW dalle shape. */
static int llm_mlx_tq_qmv(mlx_array* y, mlx_array packed, mlx_array scales, mlx_array codebook,
                          mlx_array x, int e, int K, int O, int NG, int PW, int bits, int group, int trit,
                          mlx_stream stream) {
    (void)NG; (void)PW;
    llm_tq_qmv_cache* c = llm_tq_qmv_get(bits, group, trit, 0, K);
    if (c == NULL || c->rc != 0) return -1;
    mlx_array earr; int own = 0;
    if (e == 0) {
        if (g_tq_zero.ctx == NULL) { int z = 0; g_tq_zero = mlx_array_new_data(&z, NULL, 0, MLX_INT32); }
        earr = g_tq_zero;
    } else { int e_i = e; earr = mlx_array_new_data(&e_i, NULL, 0, MLX_INT32); own = 1; }
    int rc = llm_tq_qmv_apply(y, c, packed, scales, codebook, x, earr, O, 1, O * 32, K, stream);
    if (own) mlx_array_free(earr);
    return rc;
}

/* mode 1 (gather): y [tk,O] = dequant(W[indices[kk]])·x[kk]; x [tk,K], indices [tk]. UN launch. */
static int llm_mlx_tq_qmv_kg(mlx_array* y, mlx_array packed, mlx_array scales, mlx_array codebook,
                             mlx_array x, mlx_array indices, int tk, int K, int O, int NG, int PW,
                             int bits, int group, int trit, mlx_stream stream) {
    (void)NG; (void)PW;
    llm_tq_qmv_cache* c = llm_tq_qmv_get(bits, group, trit, 1, K);
    if (c == NULL || c->rc != 0) return -1;
    return llm_tq_qmv_apply(y, c, packed, scales, codebook, x, indices, O, tk, tk * O * 32, K, stream);
}

#endif /* LLM_MLX_TQ_QMV_KERNEL_H */
