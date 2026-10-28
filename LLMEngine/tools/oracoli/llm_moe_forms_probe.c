/* llm_moe_forms_probe.c — ORACOLO model-free §2 GLM: confronta le FORME del blocco MoE decode
 * (3 gather separati vs gate_up fuso vs fused senza contiguous vs sorted) sulle forme reali
 * degli esperti GLM-4.5-Air (affine g64 b4, E=128, top-8, attivazioni bf16).
 *
 * Cosa misura (nessun modello, pesi sintetici):
 *   ms mediana per forma (rapporto A/B, mai tok/s) + picco memoria MLX per forma + parità vs A.
 * Forme:
 *   A sep_unsorted     : 3 gather unsorted (reference decode: M=8<64) + silu + weighted sum
 *   B fused_unsorted   : gate_up fuso (1 gather) + split + contiguous + down (DEFAULT motore)
 *   C fused_nocontig   : come B ma SENZA contiguous (riproduce la lezione -40x su down)
 *   D fused_sorted     : come B ma con gather sorted (forma prefill, M>=64)
 *   E sep_sorted       : 3 gather sorted (alternativa senza fusione)
 * Uso: llm_moe_forms_probe [M] [reps]      (default M=8 reps=5)
 */
#include <mlx/c/mlx.h>
#include <mlx/c/memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define E 128
#define H 4096
#define INTM 1408
#define GS 64
#define BITS 4
#define MODE "affine"

static mlx_stream g_st;
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static uint32_t g_rng = 0x9e3779b9u;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }

static const uint32_t* g_ids_for_sort;
static int cmp_rows(const void* pa, const void* pb) {
    uint32_t a = *(const uint32_t*)pa, b = *(const uint32_t*)pb;
    uint32_t ia = g_ids_for_sort[a], ib = g_ids_for_sort[b];
    if (ia != ib) return ia < ib ? -1 : 1;
    return a < b ? -1 : (a > b ? 1 : 0);
}
static double median(double* v, int n) {
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

/* pesi sintetici: [E, N, Kpacked] u32 + scales/biases [E, N, K/GS] bf16 */
typedef struct { int N, K; mlx_array w, sc, bi; uint32_t* wpk; uint16_t* sch; uint16_t* bih; } Wq;
static void wq_build(Wq* q, int N, int K) {
    q->N = N; q->K = K;
    const long pr = (long)K * BITS / 32, ng = K / GS;
    q->wpk = malloc((size_t)E * N * pr * 4);
    q->sch = malloc((size_t)E * N * ng * 2);
    q->bih = malloc((size_t)E * N * ng * 2);
    for (long i = 0; i < (long)E * N * pr; i++) q->wpk[i] = xr();
    for (long i = 0; i < (long)E * N * ng; i++) {
        q->sch[i] = (uint16_t)(0x3C00 | (xr() & 0x00FF));
        q->bih[i] = (uint16_t)((xr() & 1 ? 0x8000 : 0) | 0x3F00 | (xr() & 0x007F));
    }
    const int ws[3] = {E, N, (int)pr}, ss[3] = {E, N, (int)ng};
    q->w = mlx_array_new_data(q->wpk, ws, 3, MLX_UINT32);
    q->sc = mlx_array_new_data(q->sch, ss, 3, MLX_BFLOAT16);
    q->bi = mlx_array_new_data(q->bih, ss, 3, MLX_BFLOAT16);
}
static void wq_free(Wq* q) {
    mlx_array_free(q->w); mlx_array_free(q->sc); mlx_array_free(q->bi);
    free(q->wpk); free(q->sch); free(q->bih);
}
/* gate_up fuso: concatenazione su N dei due pesi (stessi K) */
static Wq wq_fuse(const Wq* g, const Wq* u) {
    Wq q; q.N = g->N + u->N; q.K = g->K;
    const long pr = (long)q.K * BITS / 32, ng = q.K / GS;
    q.wpk = malloc((size_t)E * q.N * pr * 4);
    q.sch = malloc((size_t)E * q.N * ng * 2);
    q.bih = malloc((size_t)E * q.N * ng * 2);
    for (int e = 0; e < E; e++) {
        memcpy(&q.wpk[(size_t)e * q.N * pr], &g->wpk[(size_t)e * g->N * pr], (size_t)g->N * pr * 4);
        memcpy(&q.wpk[(size_t)e * q.N * pr + (size_t)g->N * pr], &u->wpk[(size_t)e * u->N * pr], (size_t)u->N * pr * 4);
        memcpy(&q.sch[(size_t)e * q.N * ng], &g->sch[(size_t)e * g->N * ng], (size_t)g->N * ng * 2);
        memcpy(&q.sch[(size_t)e * q.N * ng + (size_t)g->N * ng], &u->sch[(size_t)e * u->N * ng], (size_t)u->N * ng * 2);
        memcpy(&q.bih[(size_t)e * q.N * ng], &g->bih[(size_t)e * g->N * ng], (size_t)g->N * ng * 2);
        memcpy(&q.bih[(size_t)e * q.N * ng + (size_t)g->N * ng], &u->bih[(size_t)e * u->N * ng], (size_t)u->N * ng * 2);
    }
    const int ws[3] = {E, q.N, (int)pr}, ss[3] = {E, q.N, (int)ng};
    q.w = mlx_array_new_data(q.wpk, ws, 3, MLX_UINT32);
    q.sc = mlx_array_new_data(q.sch, ss, 3, MLX_BFLOAT16);
    q.bi = mlx_array_new_data(q.bih, ss, 3, MLX_BFLOAT16);
    return q;
}

static int gather(mlx_array* out, mlx_array x, const Wq* q, mlx_array ids, bool sorted) {
    mlx_array no_lhs = mlx_array_new();
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    int rc = mlx_gather_qmm(out, x, q->w, q->sc, q->bi, no_lhs, ids, true, gsz, bts, MODE, sorted, g_st);
    mlx_array_free(no_lhs);
    return rc;
}
static int silu_mul(mlx_array* act, mlx_array g, mlx_array u) {
    mlx_array sg = mlx_array_new(), s = mlx_array_new();
    int rc = mlx_sigmoid(&sg, g, g_st);
    if (rc == 0) rc = mlx_multiply(&s, g, sg, g_st);
    if (rc == 0) rc = mlx_multiply(act, s, u, g_st);
    mlx_array_free(s); mlx_array_free(sg);
    return rc;
}
static int weighted_sum(mlx_array* out, mlx_array d, mlx_array scores) {
    const int m = mlx_array_shape(d)[0];
    const int sh[3] = {m, 1, 1};
    mlx_array sc4 = mlx_array_new(), m4 = mlx_array_new();
    int rc = mlx_reshape(&sc4, scores, sh, 3, g_st);
    if (rc == 0) rc = mlx_multiply(&m4, d, sc4, g_st);
    if (rc == 0) rc = mlx_sum_axis(out, m4, 0, true, g_st);
    mlx_array_free(m4); mlx_array_free(sc4);
    return rc;
}

/* Forme. Ritorna 0 e scrive out. */
static int form_A(mlx_array* out, mlx_array x, const Wq* g, const Wq* u, const Wq* d, mlx_array ids, mlx_array scores) {
    mlx_array go = mlx_array_new(), uo = mlx_array_new(), act = mlx_array_new(), dn = mlx_array_new();
    int rc = gather(&go, x, g, ids, false);
    if (rc == 0) rc = gather(&uo, x, u, ids, false);
    if (rc == 0) rc = silu_mul(&act, go, uo);
    if (rc == 0) rc = gather(&dn, act, d, ids, false);
    if (rc == 0) rc = weighted_sum(out, dn, scores);
    mlx_array_free(go); mlx_array_free(uo); mlx_array_free(act); mlx_array_free(dn);
    return rc;
}
static int form_B(mlx_array* out, mlx_array x, const Wq* gu, const Wq* d, mlx_array ids, mlx_array scores, bool contig, bool sorted) {
    mlx_array gau = mlx_array_new(), act = mlx_array_new(), dn = mlx_array_new();
    int rc = gather(&gau, x, gu, ids, sorted);
    if (rc == 0) {
        mlx_vector_array parts = mlx_vector_array_new();
        rc = mlx_split(&parts, gau, 2, -1, g_st);
        if (rc == 0) {
            mlx_array g = mlx_array_new(), u = mlx_array_new();
            mlx_vector_array_get(&g, parts, 0);
            mlx_vector_array_get(&u, parts, 1);
            if (contig) {
                mlx_array gc = mlx_array_new(), uc = mlx_array_new();
                if (mlx_contiguous(&gc, g, false, g_st) == 0 && mlx_contiguous(&uc, u, false, g_st) == 0) {
                    mlx_array_set(&g, gc); mlx_array_set(&u, uc);
                }
                mlx_array_free(gc); mlx_array_free(uc);
            }
            rc = silu_mul(&act, g, u);
            mlx_array_free(g); mlx_array_free(u);
        }
        mlx_vector_array_free(parts);
    }
    if (rc == 0) rc = gather(&dn, act, d, ids, sorted);
    if (rc == 0) rc = weighted_sum(out, dn, scores);
    mlx_array_free(gau); mlx_array_free(act); mlx_array_free(dn);
    return rc;
}

static double maxdiff(mlx_array a, mlx_array b, double* refmax) {
    mlx_array a32 = mlx_array_new(), b32 = mlx_array_new(), d = mlx_array_new(), ad = mlx_array_new(), mx = mlx_array_new();
    mlx_astype(&a32, a, MLX_FLOAT32, g_st); mlx_astype(&b32, b, MLX_FLOAT32, g_st);
    mlx_subtract(&d, a32, b32, g_st); mlx_abs(&ad, d, g_st); mlx_max(&mx, ad, false, g_st);
    float v = 0; mlx_array_item_float32(&v, mx);
    if (refmax) { mlx_array r = mlx_array_new(), rm = mlx_array_new();
        mlx_abs(&r, b32, g_st); mlx_max(&rm, r, false, g_st);
        float rv = 0; mlx_array_item_float32(&rv, rm); *refmax = rv; mlx_array_free(r); mlx_array_free(rm); }
    mlx_array_free(a32); mlx_array_free(b32); mlx_array_free(d); mlx_array_free(ad); mlx_array_free(mx);
    return v;
}

/* Esegue la forma f e ritorna l'output (chiamante: free). */
static int run_form(int f, mlx_array* out, mlx_array x, mlx_array xsorted, const Wq* wg, const Wq* wu,
                    const Wq* wgu, const Wq* wd, mlx_array ids, mlx_array idssorted, mlx_array scores) {
    switch (f) {
        case 0: return form_A(out, x, wg, wu, wd, ids, scores);
        case 1: return form_B(out, x, wgu, wd, ids, scores, true, false);
        case 2: return form_B(out, x, wgu, wd, ids, scores, false, false);
        case 3: return form_B(out, xsorted, wgu, wd, idssorted, scores, true, true);
        default: {
            mlx_array go = mlx_array_new(), uo = mlx_array_new(), act = mlx_array_new(), dn = mlx_array_new();
            int rc = gather(&go, xsorted, wg, idssorted, true);
            if (rc == 0) rc = gather(&uo, xsorted, wu, idssorted, true);
            if (rc == 0) rc = silu_mul(&act, go, uo);
            if (rc == 0) rc = gather(&dn, act, wd, idssorted, true);
            if (rc == 0) rc = weighted_sum(out, dn, scores);
            mlx_array_free(go); mlx_array_free(uo); mlx_array_free(act); mlx_array_free(dn);
            return rc;
        }
    }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int reps = argc > 2 ? atoi(argv[2]) : 9;
    if (reps < 1) reps = 1;
    { extern int mlx_set_qmv_split_k(int v);
      const char* sk = getenv("PROBE_SPLIT_K");
      if (sk) { int v = atoi(sk); mlx_set_qmv_split_k(v); printf("[probe] qmv_split_k=%d\n", v); } }
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    Wq wg, wu, wd;
    wq_build(&wg, I, H);
    wq_build(&wu, I, H);
    wq_build(&wd, H, I);
    Wq wgu = wq_fuse(&wg, &wu);

    uint16_t* xh = malloc((size_t)M * H * 2);
    uint16_t* xsh = malloc((size_t)M * H * 2);
    uint32_t* idh = malloc((size_t)M * 4), *idsh = malloc((size_t)M * 4), *perm = malloc((size_t)M * 4);
    uint16_t* sch = malloc((size_t)M * 2);
    for (long i = 0; i < (long)M * H; i++) xh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF));
    for (int i = 0; i < M; i++) { idh[i] = xr() % E; perm[i] = (uint32_t)i; sch[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF)); }
    g_ids_for_sort = idh;
    qsort(perm, (size_t)M, sizeof(uint32_t), cmp_rows);
    for (int r = 0; r < M; r++) {
        memcpy(&xsh[(long)r * H], &xh[(long)perm[r] * H], (size_t)H * 2);
        idsh[r] = idh[perm[r]];
    }
    const int xs[3] = {M, 1, H}, is[1] = {M};
    mlx_array x = mlx_array_new_data(xh, xs, 3, MLX_BFLOAT16);
    mlx_array xsorted = mlx_array_new_data(xsh, xs, 3, MLX_BFLOAT16);
    mlx_array ids = mlx_array_new_data(idh, is, 1, MLX_UINT32);
    mlx_array idssorted = mlx_array_new_data(idsh, is, 1, MLX_UINT32);
    mlx_array scores = mlx_array_new_data(sch, is, 1, MLX_BFLOAT16);
    mlx_array_eval(wgu.w); mlx_array_eval(wd.w); /* pesi residenti prima delle misure */

    /* riferimento A (unsorted) */
    mlx_array ref = mlx_array_new();
    if (form_A(&ref, x, &wg, &wu, &wd, ids, scores) != 0) { printf("form A FAIL\n"); return 1; }
    mlx_array_eval(ref);

    printf("llm_moe_forms_probe — E=%d H=%d I=%d g%d b%d | M=%d | %d reps (mediana)\n", E, H, INTM, GS, BITS, M, reps);
    printf("%-18s %-9s %-8s %-9s %-9s %s\n", "forma", "ms", "vs A", "min", "max", "parita_vs_A");

    const char* names[5] = {"A sep_unsorted", "B fused_unsorted", "C fused_nocontig", "D fused_sorted", "E sep_sorted"};
    double ts[5][64]; double mn[5], mx[5];
    mlx_array outs[5];
    for (int f = 0; f < 5; f++) { outs[f] = mlx_array_new(); mn[f] = 1e30; mx[f] = 0; }
    /* warmup 2 giri su tutte le forme, poi reps giri INTERLEAVED (metodo oracoli: il rapporto
     * A/B è robusto anche a macchina carica, il valore assoluto no) */
    for (int r = 0; r < reps + 2; r++) {
        for (int f = 0; f < 5; f++) {
            if (M > 4096 && f <= 2) continue;
            mlx_array o = mlx_array_new();
            double t0 = now_ms();
            int rc = run_form(f, &o, x, xsorted, &wg, &wu, &wgu, &wd, ids, idssorted, scores);
            if (rc == 0) mlx_array_eval(o);
            double t1 = now_ms();
            if (rc != 0) { printf("%s FAIL rc=%d\n", names[f], rc); mlx_array_free(o); return 1; }
            if (r >= 2) { ts[f][r - 2] = t1 - t0;
                if (ts[f][r - 2] < mn[f]) mn[f] = ts[f][r - 2];
                if (ts[f][r - 2] > mx[f]) mx[f] = ts[f][r - 2]; }
            mlx_array_set(&outs[f], o);
            mlx_array_free(o);
        }
    }
    double base = median(ts[0], reps);
    double rm = 0; maxdiff(ref, ref, &rm);
    double tol = 1e-2 + 5e-2 * rm;
    for (int f = 0; f < 5; f++) {
        if (M > 4096 && f <= 2) { printf("%-18s SKIP (unsorted espande a M grandi)\n", names[f]); continue; }
        double md = maxdiff(outs[f], ref, NULL);
        printf("%-18s %-9.3f %-8.2f %-9.3f %-9.3f %s (max|d|=%.3g)\n",
               names[f], median(ts[f], reps), median(ts[f], reps) / base, mn[f], mx[f],
               md <= tol ? "PARI" : "DIVERGE", md);
    }
    for (int f = 0; f < 5; f++) mlx_array_free(outs[f]);
    mlx_array_free(ref); mlx_array_free(scores); mlx_array_free(idssorted); mlx_array_free(ids);
    mlx_array_free(xsorted); mlx_array_free(x);
    free(sch); free(perm); free(idsh); free(idh); free(xsh); free(xh);
    wq_free(&wgu); wq_free(&wd); wq_free(&wu); wq_free(&wg);
    mlx_stream_free(g_st); mlx_device_free(dev);
    return 0;
}
