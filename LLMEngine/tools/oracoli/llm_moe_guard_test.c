/* llm_moe_guard_test.c — ORACOLO model-free: valida la GUARDIA ENGINE
 * `llm_mlx_moe_gather_mm` (llm_mlx_moe.c) contro `mlx_gather_qmm` diretto, sulle forme reali
 * degli esperti GLM-4.5-Air (affine g64 b4, E=128, N=2816, K=4096), x bf16 [M,1,K].
 *
 * Atteso (MLX 0.32.x, M5, bug #3856):
 *   M>32768 && M%64!=0 -> DIRECT corrotto (max|diff| ~1e3), GUARDED corretto;
 *   M<=32768 o M%64==0 -> entrambi corretti.
 * Riferimento: dequantize dell'esperto + matmul bf16 (ground truth), come
 * llm_gather_qmm_sorted_probe. Nessun modello, nessun peso reale.
 *
 * Uso: llm_moe_guard_test [M]      (default 32776)
 * Exit 0 = guarded PARI; 1 = guarded diverge.
 */
#include "llm_mlx_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define E 128
#define TOPK 8
#define GS 64
#define BITS 4
#define MODE "affine"
#define N 2816
#define K 4096

static mlx_stream g_st;
static uint32_t g_rng = 0x12345678u;
static uint32_t xr(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

static const uint32_t* g_ids_for_sort;
static int cmp_rows(const void* pa, const void* pb) {
    uint32_t a = *(const uint32_t*)pa, b = *(const uint32_t*)pb;
    uint32_t ia = g_ids_for_sort[a], ib = g_ids_for_sort[b];
    if (ia != ib) return ia < ib ? -1 : 1;
    return a < b ? -1 : (a > b ? 1 : 0);
}

/* max|a-b| in fp32 */
static double max_abs_diff(mlx_array a, mlx_array b, double* max_ref_out) {
    mlx_array a32 = mlx_array_new(), b32 = mlx_array_new(), d = mlx_array_new();
    mlx_array ad = mlx_array_new(), mx = mlx_array_new(), rmax = mlx_array_new(), rtb = mlx_array_new();
    mlx_astype(&a32, a, MLX_FLOAT32, g_st);
    mlx_astype(&b32, b, MLX_FLOAT32, g_st);
    mlx_subtract(&d, a32, b32, g_st);
    mlx_abs(&ad, d, g_st);
    mlx_max(&mx, ad, false, g_st);
    if (max_ref_out) {
        mlx_abs(&rtb, b32, g_st);
        mlx_max(&rmax, rtb, false, g_st);
        float rv = 0.0f;
        mlx_array_item_float32(&rv, rmax);
        *max_ref_out = (double)rv;
    }
    float v = 0.0f;
    mlx_array_item_float32(&v, mx);
    mlx_array_free(rtb); mlx_array_free(rmax); mlx_array_free(mx); mlx_array_free(ad);
    mlx_array_free(d); mlx_array_free(b32); mlx_array_free(a32);
    return (double)v;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 32776;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    const long packed_row = (long)K * BITS / 32, ngroups = K / GS;
    const long w_elems = (long)E * N * packed_row, s_elems = (long)E * N * ngroups;
    uint32_t* wpk = malloc((size_t)w_elems * 4);
    uint16_t* sc = malloc((size_t)s_elems * 2), *bi = malloc((size_t)s_elems * 2);
    uint16_t* xh = malloc((size_t)M * K * 2);
    uint32_t* idh = malloc((size_t)M * 4), *perm = malloc((size_t)M * 4);
    if (!wpk || !sc || !bi || !xh || !idh || !perm) { printf("OOM\n"); return 1; }
    for (long i = 0; i < w_elems; i++) wpk[i] = xr();
    for (long i = 0; i < s_elems; i++) {
        sc[i] = (uint16_t)(0x3C00 | (xr() & 0x00FF));
        bi[i] = (uint16_t)((xr() & 1 ? 0x8000 : 0) | 0x3F00 | (xr() & 0x007F));
    }
    for (long i = 0; i < (long)M * K; i++) xh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF));
    for (int i = 0; i < M; i++) { idh[i] = xr() % E; perm[i] = (uint32_t)i; }
    g_ids_for_sort = idh;
    qsort(perm, (size_t)M, sizeof(uint32_t), cmp_rows);
    /* righe/ids ORDINATI per esperto (come fa il motore prima del gather sorted) */
    uint16_t* xsh = malloc((size_t)M * K * 2);
    uint32_t* idsh = malloc((size_t)M * 4);
    if (!xsh || !idsh) { printf("OOM sorted\n"); return 1; }
    for (int r = 0; r < M; r++) {
        memcpy(&xsh[(long)r * K], &xh[(long)perm[r] * K], (size_t)K * 2);
        idsh[r] = idh[perm[r]];
    }

    const int ws[3] = {E, N, (int)packed_row}, ss[3] = {E, N, (int)ngroups};
    const int xs[3] = {M, 1, K}, is[1] = {M};
    mlx_array w = mlx_array_new_data(wpk, ws, 3, MLX_UINT32);
    mlx_array scales = mlx_array_new_data(sc, ss, 3, MLX_BFLOAT16);
    mlx_array biases = mlx_array_new_data(bi, ss, 3, MLX_BFLOAT16);
    mlx_array x = mlx_array_new_data(xh, xs, 3, MLX_BFLOAT16);
    mlx_array ids = mlx_array_new_data(idh, is, 1, MLX_UINT32);
    mlx_array xs_sorted = mlx_array_new_data(xsh, xs, 3, MLX_BFLOAT16);
    mlx_array ids_sorted = mlx_array_new_data(idsh, is, 1, MLX_UINT32);
    mlx_array no_lhs = mlx_array_new();
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};

    /* 1) riferimento per-esperto: dequantize + matmul (ground truth) */
    mlx_vector_array refs = mlx_vector_array_new();
    for (int e = 0; e < E; e++) {
        uint32_t* rows = malloc((size_t)M * 4); int cnt = 0;
        for (int r = 0; r < M; r++) if ((int)idh[r] == e) rows[cnt++] = (uint32_t)r;
        if (cnt == 0) { free(rows); continue; }
        const int rs[1] = {cnt};
        mlx_array rids = mlx_array_new_data(rows, rs, 1, MLX_UINT32);
        mlx_array xe = mlx_array_new(); mlx_take_axis(&xe, x, rids, 0, g_st);
        mlx_array es = mlx_array_new_int(e);
        mlx_array we = mlx_array_new(), sce = mlx_array_new(), bie = mlx_array_new();
        mlx_take_axis(&we, w, es, 0, g_st);
        mlx_take_axis(&sce, scales, es, 0, g_st);
        mlx_take_axis(&bie, biases, es, 0, g_st);
        mlx_array we_dec = mlx_array_new(), wt = mlx_array_new(), gs_null = mlx_array_new();
        mlx_optional_dtype dt = {MLX_BFLOAT16, true};
        mlx_dequantize(&we_dec, we, sce, bie, gsz, bts, MODE, gs_null, dt, g_st);
        const int pm[2] = {1, 0};
        mlx_transpose_axes(&wt, we_dec, pm, 2, g_st);
        mlx_array re = mlx_array_new();
        mlx_matmul(&re, xe, wt, g_st);
        mlx_vector_array_append_value(refs, re);
        mlx_array_free(re); mlx_array_free(wt); mlx_array_free(we_dec); mlx_array_free(gs_null);
        mlx_array_free(bie); mlx_array_free(sce); mlx_array_free(we); mlx_array_free(es);
        mlx_array_free(xe); mlx_array_free(rids); free(rows);
    }
    mlx_array refcat = mlx_array_new();
    mlx_concatenate_axis(&refcat, refs, 0, g_st);
    if (mlx_array_ndim(refcat) != 3) {
        mlx_array rc2 = mlx_array_new();
        const int sh[3] = {M, 1, N};
        if (mlx_reshape(&rc2, refcat, sh, 3, g_st) == 0) { mlx_array_free(refcat); refcat = rc2; }
    }

    /* 2) mlx_gather_qmm DIRETTO (path buggato) */
    mlx_array direct = mlx_array_new();
    int rc1 = mlx_gather_qmm(&direct, xs_sorted, w, scales, biases, no_lhs, ids_sorted, true, gsz, bts, MODE, true, g_st);
    /* 3) GUARDED via engine (la funzione spedita) */
    ModelConfig cfg; memset(&cfg, 0, sizeof cfg);
    snprintf(cfg.quant_mode, sizeof cfg.quant_mode, "%s", MODE);
    mlx_array guarded = mlx_array_new();
    int rc2 = llm_mlx_moe_gather_mm(&cfg, &guarded, xs_sorted, w, scales, biases, no_lhs, ids_sorted, true, gsz, bts, MODE, true, g_st);

    mlx_array_eval(refcat);
    { double rmv = 0.0; max_abs_diff(refcat, refcat, &rmv);
      printf("[dbg] refs=%zu refcat_ndim=%d max|refcat|=%.4g\n",
             (size_t)mlx_vector_array_size(refs), mlx_array_ndim(refcat), rmv); }
    if (rc1 == 0) mlx_array_eval(direct);
    if (rc2 == 0) mlx_array_eval(guarded);

    double ref_max = 0.0, d_direct = -1.0, d_guard = -1.0;
    if (rc1 == 0) d_direct = max_abs_diff(direct, refcat, &ref_max);
    if (rc2 == 0) d_guard = max_abs_diff(guarded, refcat, NULL);
    const double tol = 1e-2 + 5e-2 * ref_max;
    printf("llm_moe_guard_test — M=%d (M%%64=%d) E=%d N=%d K=%d g%d b%d\n",
           M, M % 64, E, N, K, GS, BITS);
    printf("  max|ref|=%.4g tol=%.4g\n", ref_max, tol);
    printf("  DIRECT  mlx_gather_qmm:        max|diff|=%.4g  %s\n", d_direct,
           rc1 != 0 ? "FAIL" : (d_direct <= tol ? "ok" : "CORROTTO (bug #3856)"));
    printf("  GUARDED llm_mlx_moe_gather_mm: max|diff|=%.4g  %s\n", d_guard,
           rc2 != 0 ? "FAIL" : (d_guard <= tol ? "PARI" : "DIVERGE"));
    int bad = (rc2 != 0) || (d_guard > tol);
    printf("ESITO: %s\n", bad ? "GUARDIA NON OK" : "guardia OK");

    mlx_array_free(guarded); mlx_array_free(direct); mlx_array_free(refcat);
    mlx_vector_array_free(refs);
    mlx_array_free(ids_sorted); mlx_array_free(xs_sorted);
    mlx_array_free(no_lhs); mlx_array_free(ids); mlx_array_free(x);
    free(idsh); free(xsh);
    mlx_array_free(biases); mlx_array_free(scales); mlx_array_free(w);
    free(perm); free(idh); free(xh); free(bi); free(sc); free(wpk);
    mlx_stream_free(g_st);
    mlx_device_free(dev);
    return bad ? 1 : 0;
}
