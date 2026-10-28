/* llm_gather_qmm_sorted_probe.c — ORACOLO model-free: parità e rapporto A/B del
 * gather_qmm(sorted_indices=true) sulle forme REALI degli esperti GLM-4.5-Air
 * (affine g64 b4, 128 esperti, top-8, attivazioni bf16).
 *
 * Motivazione (§7 TODO GLM-4.5): sospetto bug MLX/NAX con sorted_indices=true quando
 * righe M = n_tokens*top_k > 32768 (raggiungibile solo con prefill chunk >= 4097 token;
 * il default chunk 2048 -> M=16384). Il motore usa sorted per M>=64
 * (llm_mlx_glm_ds_ffn.c: do_sort = ... n_tokens*top_k >= 64).
 *
 * ORACOLO (nessun modello, pesi sintetici): il path sorted deve coincidere con il
 * riferimento per-esperto (mlx_quantized_matmul applicata agli stessi pesi/righe).
 * Qualsiasi max|diff| oltre il rumore a M>32768 = bug manifestato.
 * Il tempo è solo RAPPORTO A/B (mai tok/s, mai valore assoluto: la macchina può essere carica).
 *
 * Uso: llm_gather_qmm_sorted_probe [gu|down|both] [reps]
 *   gu   = gate_up  [E=128, N=2816, K=4096] (gather più grande)
 *   down = down_proj[E=128, N=4096, K=1408]
 * Exit 0 = parità; 1 = divergenza.
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define E 128
#define TOPK 8
#define GS 64
#define BITS 4
#define MODE "affine"

static mlx_stream g_st;
static mlx_array g_rtol, g_atol; /* tolleranza bf16: 5% rel, 1e-2 assoluta */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

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

static double median(double* v, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

static int g_dbg = 0;
static void pshape(const char* tag, mlx_array a) {
    if (!g_dbg) return;
    int nd = mlx_array_ndim(a);
    const int* sh = mlx_array_shape(a);
    fprintf(stderr, "    [dbg] %s ndim=%d shape=", tag, nd);
    for (int i = 0; i < nd; i++) fprintf(stderr, "%d%s", sh[i], i + 1 < nd ? "," : "");
    fprintf(stderr, "\n");
}

static int g_slice = 0; /* PROBE_SLICE=1: simula la guardia engine (slice <=32768) */

static int gather_sorted(mlx_array w, mlx_array sc, mlx_array bi,
                         mlx_array x, mlx_array idx, mlx_array* out) {
    mlx_array no_lhs = mlx_array_new();
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    int64_t rows = (int64_t)mlx_array_size(idx);
    if (g_slice && rows > 32768 && (rows % 64) != 0) {
        mlx_vector_array parts = mlx_vector_array_new();
        int rc = 0;
        for (int64_t off = 0; off < rows && rc == 0; off += 32768) {
            int64_t stop = off + 32768; if (stop > rows) stop = rows;
            int s0[8], s1[8], st1[8];
            int ndx = mlx_array_ndim(x);
            const int* shp = mlx_array_shape(x);
            for (int d = 0; d < ndx; d++) { s0[d] = 0; s1[d] = shp[d]; st1[d] = 1; }
            s0[0] = (int)off; s1[0] = (int)stop;
            const int r0[1] = {(int)off}, r1[1] = {(int)stop}, rst[1] = {1};
            mlx_array xs = mlx_array_new(), rs = mlx_array_new(), o = mlx_array_new();
            rc = mlx_slice(&xs, x, s0, ndx, s1, ndx, st1, ndx, g_st);
            if (rc == 0) rc = mlx_slice(&rs, idx, r0, 1, r1, 1, rst, 1, g_st);
            if (rc == 0) rc = mlx_gather_qmm(&o, xs, w, sc, bi, no_lhs, rs, true, gsz, bts, MODE, true, g_st);
            if (rc == 0) mlx_vector_array_append_value(parts, o);
            mlx_array_free(o); mlx_array_free(rs); mlx_array_free(xs);
        }
        if (rc == 0) rc = mlx_concatenate_axis(out, parts, 0, g_st);
        mlx_vector_array_free(parts);
        mlx_array_free(no_lhs);
        return rc;
    }
    int rc = mlx_gather_qmm(out, x, w, sc, bi, no_lhs, idx,
                            /*transpose=*/true, gsz, bts, MODE, /*sorted=*/true, g_st);
    mlx_array_free(no_lhs);
    return rc;
}

static int gather_unsorted(mlx_array w, mlx_array sc, mlx_array bi,
                           mlx_array x, mlx_array idx, mlx_array* out) {
    mlx_array no_lhs = mlx_array_new();
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    int rc = mlx_gather_qmm(out, x, w, sc, bi, no_lhs, idx,
                            /*transpose=*/true, gsz, bts, MODE, /*sorted=*/false, g_st);
    mlx_array_free(no_lhs);
    return rc;
}

/* Riferimento per-esperto: per ogni esperto e, ref_e = x_rows @ w_e^T (quantized_matmul).
 * Ritorna il vettore dei ref (in ordine di esperto) e, per ogni esperto, gli indici
 * host delle righe che lo usano (rows_of[e], count_of[e]). */
static int build_ref(mlx_array w, mlx_array sc, mlx_array bi, mlx_array x,
                     const uint32_t* ids, int M,
                     mlx_vector_array* refs, uint32_t*** rows_of, int** count_of, int** slot_of) {
    uint32_t* rows[E]; int cnt[E];
    for (int e = 0; e < E; e++) { rows[e] = malloc((size_t)M * 4); cnt[e] = 0; }
    for (int r = 0; r < M; r++) { rows[ids[r]][cnt[ids[r]]++] = (uint32_t)r; }
    *refs = mlx_vector_array_new();
    int* slot = malloc(sizeof(int) * E);
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    for (int e = 0; e < E; e++) {
        slot[e] = -1;
        if (cnt[e] == 0) continue;
        int rs[1] = {cnt[e]};
        mlx_array rids = mlx_array_new_data(rows[e], rs, 1, MLX_UINT32);
        mlx_array xe = mlx_array_new();
        mlx_take_axis(&xe, x, rids, 0, g_st);
        mlx_array we = mlx_array_new(), sce = mlx_array_new(), bie = mlx_array_new();
        mlx_array es = mlx_array_new_int(e);
        mlx_take_axis(&we, w, es, 0, g_st);
        mlx_take_axis(&sce, sc, es, 0, g_st);
        mlx_take_axis(&bie, bi, es, 0, g_st);
        /* ground truth: dequantizza l'esperto e fai la matmul (bf16), niente kernel quantizzato */
        mlx_array we_dec = mlx_array_new(), wt = mlx_array_new(), gs_null = mlx_array_new();
        mlx_optional_dtype dt = {MLX_BFLOAT16, true};
        mlx_dequantize(&we_dec, we, sce, bie, gsz, bts, MODE, gs_null, dt, g_st);
        const int perm[2] = {1, 0};
        mlx_transpose_axes(&wt, we_dec, perm, 2, g_st);
        mlx_array re = mlx_array_new();
        mlx_matmul(&re, xe, wt, g_st);
        pshape("xe", xe); pshape("we", we); pshape("re", re);
        slot[e] = (int)mlx_vector_array_size(*refs);
        mlx_vector_array_append_value(*refs, re);
        mlx_array_free(es); mlx_array_free(xe); mlx_array_free(we);
        mlx_array_free(sce); mlx_array_free(bie); mlx_array_free(re);
        mlx_array_free(we_dec); mlx_array_free(wt); mlx_array_free(gs_null);
        mlx_array_free(rids);
    }
    *rows_of = malloc(sizeof(uint32_t*) * E);
    *count_of = malloc(sizeof(int) * E);
    for (int e = 0; e < E; e++) { (*rows_of)[e] = rows[e]; (*count_of)[e] = cnt[e]; }
    *slot_of = slot;
    return 0;
}

typedef struct { double maxd; int neq; int nexact; int nrows; } CmpRes;

/* Confronto fp32 di una coppia (gather, riferimento) con tolleranza bf16:
 * |a-b| <= atol + rtol*|b|. Aggiorna l'accumulatore. */
static void check_pair(mlx_array ge, mlx_array re, int tag, CmpRes* r) {
    mlx_array diff = mlx_array_new(), ad = mlx_array_new(), mx = mlx_array_new();
    mlx_array ge32 = mlx_array_new(), re32 = mlx_array_new();
    mlx_astype(&ge32, ge, MLX_FLOAT32, g_st);
    mlx_astype(&re32, re, MLX_FLOAT32, g_st);
    mlx_subtract(&diff, ge32, re32, g_st);
    mlx_abs(&ad, diff, g_st);
    mlx_max(&mx, ad, false, g_st);
    /* criterio: max|a-b| <= atol + rtol * max|b| (scala dell'esperto, non elemento:
     * ai punti di cancellazione l'errore relativo non è significativo) */
    mlx_array rtb = mlx_array_new(), rmax = mlx_array_new(), zero = mlx_array_new_float32(0.0f);
    mlx_array neq_arr = mlx_array_new(), any_ne = mlx_array_new();
    mlx_abs(&rtb, re32, g_st);
    mlx_max(&rmax, rtb, false, g_st);
    mlx_not_equal(&neq_arr, ad, zero, g_st);
    mlx_any(&any_ne, neq_arr, false, g_st);
    float d = 0.0f, rm = 0.0f; bool ex = true;
    mlx_array_item_float32(&d, mx);
    mlx_array_item_float32(&rm, rmax);
    mlx_array_item_bool(&ex, any_ne);
    double tol = 1e-2 + 5e-2 * (double)rm;
    if ((double)d > r->maxd) r->maxd = (double)d;
    if (ex) r->nexact++;
    if ((double)d > tol) {
        r->neq++;
        if (g_dbg) fprintf(stderr, "    [cmp-fail] tag=%d maxd=%.6g max|ref|=%.6g tol=%.6g\n",
                           tag, (double)d, (double)rm, tol);
    }
    mlx_array_free(any_ne); mlx_array_free(neq_arr); mlx_array_free(zero);
    mlx_array_free(rmax); mlx_array_free(rtb);
    mlx_array_free(mx); mlx_array_free(ad); mlx_array_free(diff);
    mlx_array_free(re32); mlx_array_free(ge32);
}

/* Path sorted: out_s è in ordine (esperto, riga originale crescente); refs è già in
 * ordine di esperto → concatenare refs dà esattamente l'ordine di out_s. */
static void compare_sorted(mlx_array out_s, mlx_vector_array refs, int M, int N, CmpRes* r) {
    mlx_array refcat = mlx_array_new();
    if (mlx_concatenate_axis(&refcat, refs, 0, g_st) != 0) { r->neq++; return; }
    int nd = mlx_array_ndim(refcat);
    if (nd != 3) { /* fallback: il concat può appiattire -> reshape [M,1,N] */
        mlx_array rc2 = mlx_array_new();
        const int sh[3] = {M, 1, N};
        if (mlx_reshape(&rc2, refcat, sh, 3, g_st) == 0) { mlx_array_free(refcat); refcat = rc2; }
    }
    pshape("refcat", refcat);
    r->nrows += M;
    check_pair(out_s, refcat, -1, r);
    mlx_array_free(refcat);
}

/* Path unsorted: out è in ordine originale → per-esperto con gli indici riga. */
static void compare_rows(mlx_array out, mlx_vector_array refs, uint32_t** rows_of,
                         const int* count_of, const int* slot_of, CmpRes* r) {
    for (int e = 0; e < E; e++) {
        int n = count_of[e];
        if (n == 0) continue;
        r->nrows += n;
        int rs[1] = {n};
        mlx_array rids = mlx_array_new_data(rows_of[e], rs, 1, MLX_UINT32);
        mlx_array ge = mlx_array_new();
        mlx_take_axis(&ge, out, rids, 0, g_st);
        mlx_array re = mlx_array_new();
        mlx_vector_array_get(&re, refs, slot_of[e]);
        check_pair(ge, re, e, r);
        mlx_array_free(re); mlx_array_free(ge); mlx_array_free(rids);
    }
}

static int run_shape(const char* name, int N, int K, int reps, int* diverged) {
    const long packed_row = (long)K * BITS / 32;
    const long ngroups = K / GS;
    const long w_elems = (long)E * N * packed_row;
    const long s_elems = (long)E * N * ngroups;

    uint32_t* wpk = malloc((size_t)w_elems * 4);
    uint16_t* sc = malloc((size_t)s_elems * 2);
    uint16_t* bi = malloc((size_t)s_elems * 2);
    if (!wpk || !sc || !bi) { printf("OOM pesi\n"); return 1; }
    for (long i = 0; i < w_elems; i++) wpk[i] = xr();
    for (long i = 0; i < s_elems; i++) {
        sc[i] = (uint16_t)(0x3C00 | (xr() & 0x00FF));
        bi[i] = (uint16_t)((xr() & 1 ? 0x8000 : 0) | 0x3F00 | (xr() & 0x007F));
    }
    int ws[3] = {E, N, (int)packed_row};
    int ss[3] = {E, N, (int)ngroups};
    mlx_array w = mlx_array_new_data(wpk, ws, 3, MLX_UINT32);
    mlx_array scales = mlx_array_new_data(sc, ss, 3, MLX_BFLOAT16);
    mlx_array biases = mlx_array_new_data(bi, ss, 3, MLX_BFLOAT16);

    static const int m_def[] = {8, 32, 16384, 32768, 32776, 40000};
    int m_list[32]; int n_m = 6;
    for (int i = 0; i < 6; i++) m_list[i] = m_def[i];
    const char* ml = getenv("PROBE_M_LIST");
    if (ml) {
        n_m = 0; char* dup = strdup(ml); char* tok = strtok(dup, ",");
        while (tok && n_m < 32) { m_list[n_m++] = atoi(tok); tok = strtok(NULL, ","); }
        free(dup);
    }
    printf("\n== %s: E=%d N=%d K=%d g%d b%d %s | %d reps (mediana) ==\n",
           name, E, N, K, GS, BITS, MODE, reps);
    printf("%-8s %-8s %-11s %-11s %-7s %-12s %s\n",
           "n_tokens", "M(rows)", "sorted_ms", "ref_ms", "ratio", "max|diff|", "vs ref");

    for (int mi = 0; mi < n_m; mi++) {
        int M = m_list[mi];
        if (getenv("PROBE_ONLY_M") && atoi(getenv("PROBE_ONLY_M")) != M) continue;
        uint16_t* xh = malloc((size_t)M * K * 2);
        uint16_t* xsh = malloc((size_t)M * K * 2);
        uint32_t* idh = malloc((size_t)M * 4);
        uint32_t* idsh = malloc((size_t)M * 4);
        uint32_t* perm = malloc((size_t)M * 4);
        if (!xh || !xsh || !idh || !idsh || !perm) { printf("OOM M=%d\n", M); return 1; }
        for (long i = 0; i < (long)M * K; i++) xh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF));
        for (int i = 0; i < M; i++) { idh[i] = xr() % E; perm[i] = (uint32_t)i; }
        g_ids_for_sort = idh;
        qsort(perm, (size_t)M, sizeof(uint32_t), cmp_rows);
        for (int r = 0; r < M; r++) {
            memcpy(&xsh[(long)r * K], &xh[(long)perm[r] * K], (size_t)K * 2);
            idsh[r] = idh[perm[r]];
        }

        int xs3[3] = {M, 1, K}; /* forma ENGINE: xs [M,1,hidden] (llm_mlx_glm_ds_ffn.c:688) */
        mlx_array xa = mlx_array_new_data(xh, xs3, 3, MLX_BFLOAT16);
        mlx_array xsa = mlx_array_new_data(xsh, xs3, 3, MLX_BFLOAT16);
        int is1[1] = {M};
        mlx_array ida = mlx_array_new_data(idh, is1, 1, MLX_UINT32);
        mlx_array idsa = mlx_array_new_data(idsh, is1, 1, MLX_UINT32);
        mlx_array perma = mlx_array_new_data(perm, is1, 1, MLX_UINT32);

        /* parità: sorted gather vs riferimento per-esperto (stesso ordine di righe) */
        printf("  [M=%d] parità", M); fflush(stdout);
        mlx_array out_s = mlx_array_new();
        if (gather_sorted(w, scales, biases, xsa, idsa, &out_s) != 0) { printf(" gather sorted FAIL\n"); return 1; }
        pshape("xsa", xsa); pshape("idsa", idsa); pshape("out_s", out_s);
        mlx_vector_array refs; uint32_t** rows_of; int* count_of; int* slot;
        build_ref(w, scales, biases, xa, idh, M, &refs, &rows_of, &count_of, &slot);
        mlx_array_eval(out_s);
        mlx_eval(refs);
        CmpRes cr = {0.0, 0, 0, 0};
        compare_sorted(out_s, refs, M, N, &cr);
        double maxd = cr.maxd; int neq = cr.neq, nexact = cr.nexact, nrows = cr.nrows;
        if (neq > 0) *diverged = 1;
        printf(" -> %s (righe=%d, exact_fail=%d)\n",
               neq == 0 ? "PARI(bf16)" : "DIVERGE", nrows, nexact); fflush(stdout);

        /* tempi A/B (solo M<=16384: sopra è troppo costoso e serve solo la parità) */
        double ts[8], tr[8];
        if (reps > 8) reps = 8;
        int do_time = (M <= 16384);
        for (int r = 0; r < (do_time ? reps + 1 : 0); r++) {
            mlx_array o = mlx_array_new();
            double t0 = now_ms();
            gather_sorted(w, scales, biases, xsa, idsa, &o);
            mlx_array_eval(o);
            double t1 = now_ms();
            mlx_array_free(o);
            mlx_vector_array rf; uint32_t** ro; int* co; int* sl;
            build_ref(w, scales, biases, xa, idh, M, &rf, &ro, &co, &sl);
            mlx_eval(rf);
            double t2 = now_ms();
            mlx_vector_array_free(rf);
            for (int e = 0; e < E; e++) free(ro[e]);
            free(ro); free(co); free(sl);
            if (r >= 1) { ts[r - 1] = t1 - t0; tr[r - 1] = t2 - t1; }
        }
        if (do_time)
            printf("%-8d %-8d %-11.3f %-11.3f %-7.2f %-12.3g %s (righe=%d)\n",
                   M / TOPK, M, median(ts, reps), median(tr, reps),
                   median(ts, reps) / median(tr, reps), maxd,
                   neq == 0 ? "PARI(bf16)" : "DIVERGE", nrows);
        else
            printf("%-8d %-8d %-11s %-11s %-7s %-12.3g %s (righe=%d)\n",
                   M / TOPK, M, "-", "-", "-", maxd,
                   neq == 0 ? "PARI(bf16)" : "DIVERGE", nrows);
        fflush(stdout);

        /* decode: unsorted (engine lo usa per M<64) vs riferimento */
        if (M < 64) {
            mlx_array out_u = mlx_array_new();
            gather_unsorted(w, scales, biases, xa, ida, &out_u);
            mlx_array_eval(out_u);
            CmpRes cu = {0.0, 0, 0, 0};
            compare_rows(out_u, refs, rows_of, count_of, slot, &cu);
            printf("         decode unsorted: max|diff|=%.3g vs ref: %s\n",
                   cu.maxd, cu.neq == 0 ? "PARI(bf16)" : "DIVERGE");
            /* sorted vs unsorted DIRETTO: unsort di out_s con l'inversa di perm */
            uint32_t* invh = malloc((size_t)M * 4);
            for (int r = 0; r < M; r++) invh[perm[r]] = (uint32_t)r;
            mlx_array inva = mlx_array_new_data(invh, is1, 1, MLX_UINT32);
            mlx_array taken_s = mlx_array_new();
            mlx_take_axis(&taken_s, out_s, inva, 0, g_st);
            CmpRes cs = {0.0, 0, 0, 0};
            check_pair(taken_s, out_u, -2, &cs);
            printf("         sorted vs unsorted: max|diff|=%.3g -> %s\n",
                   cs.maxd, cs.neq == 0 ? "PARI(bf16)" : "DIVERGE");
            mlx_array_free(taken_s); mlx_array_free(inva); free(invh);
            mlx_array_free(out_u);
        }

        mlx_array_free(out_s);
        mlx_vector_array_free(refs);
        for (int e = 0; e < E; e++) free(rows_of[e]);
        free(rows_of); free(count_of); free(slot);
        mlx_array_free(xa); mlx_array_free(xsa);
        mlx_array_free(ida); mlx_array_free(idsa); mlx_array_free(perma);
        free(xh); free(xsh); free(idh); free(idsh); free(perm);
    }
    mlx_array_free(w); mlx_array_free(scales); mlx_array_free(biases);
    free(wpk); free(sc); free(bi);
    return 0;
}

int main(int argc, char** argv) {
    g_dbg = getenv("PROBE_DBG") != NULL;
    { const char* sd = getenv("PROBE_SEED"); if (sd) { g_rng = (uint32_t)strtoul(sd, NULL, 10) | 1u; } }
    g_slice = getenv("PROBE_SLICE") != NULL;
    const char* which = argc > 1 ? argv[1] : "both";
    int reps = argc > 2 ? atoi(argv[2]) : 1;
    if (reps < 1) reps = 1;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();
    g_rtol = mlx_array_new_float32(5e-2f);
    g_atol = mlx_array_new_float32(1e-2f);

    int diverged = 0;
    printf("llm_gather_qmm_sorted_probe — ORACOLO model-free (nessun peso reale, nessun modello)\n");
    printf("riferimento = quantized_matmul per-esperto sulle stesse righe/pesi; sorted deve coincidere\n");
    if (strcmp(which, "gu") == 0 || strcmp(which, "both") == 0) {
        if (run_shape("gate_up", 2816, 4096, reps, &diverged) != 0) return 1;
    }
    if (strcmp(which, "down") == 0 || strcmp(which, "both") == 0) {
        if (run_shape("down_proj", 4096, 1408, reps, &diverged) != 0) return 1;
    }
    printf("\nESITO: %s\n", diverged ? "DIVERGENZA (bug manifestato)" : "parità su tutte le forme");
    mlx_array_free(g_rtol); mlx_array_free(g_atol);
    mlx_stream_free(g_st);
    mlx_device_free(dev);
    return diverged ? 1 : 0;
}
