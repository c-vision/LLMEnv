/* llm_gather_qmm_form_test.c — UNIT BENCH (2026-09-17, nessun modello):
 * la stessa mlx_gather_qmm chiamata nelle convenzioni realmente usate:
 *   A) NOSTRA sorted/prefill: x=[M,1,C]        indices=[M]        sorted_indices=true
 *      (M = T*k, indici crescenti: e' il ramo `do_sort` di mixtral_ffn_block)
 *   B) mlx_lm UNSORTED:       x=[1,T,1,1,C]    indices=[1,T,k]    sorted_indices=false
 *      (SwitchGLU: expand_dims(x,(-2,-3)) -> 5-D! La 4-D [1,T,1,C] NON e' valida:
 *       MLX broadcasta lhs_indices=[1,T] con rhs=[1,T,k] e la dim k vs T e' incompatibile -> throw)
 *   C) NOSTRA unsorted/decode: x=[T,1,1,C]     indices=[T,k]      sorted_indices=false
 *      (stessa op di B, solo piegata: singletons impliciti)
 * ESITO 2026-09-17 (gate/up e down, Mixtral): B/C ~= 1.0 e max|B-C| = 0 => la forma
 * 5-D di mlx_lm NON costa nulla vs la nostra 4-D (port P2 "forme" REFUTATO);
 * C/A = 1.5-2.5x => il ramo UNSORTED e' PIU' LENTO del sorted (indici crescenti).
 * Uso: llm_gather_qmm_form_test [n_experts] [out] [in] [T] [k] [reps]
 */
#include <mlx/c/mlx.h>
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {
    const int NE = argc > 1 ? atoi(argv[1]) : 8;
    const int OUT = argc > 2 ? atoi(argv[2]) : 4096;
    const int IN = argc > 3 ? atoi(argv[3]) : 4096;
    const int T = argc > 4 ? atoi(argv[4]) : 64;
    const int K = argc > 5 ? atoi(argv[5]) : 2;
    const int REPS = argc > 6 ? atoi(argv[6]) : 20;
    const int GS = 64, BITS = 4;
    const int PIN = IN * BITS / 32;
    const int M = T * K;

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL) { printf("FAIL device\n"); return 1; }
    if (mlx_set_default_device(dev) != 0) { printf("FAIL set_default_device\n"); return 1; }
    mlx_stream s = mlx_stream_new();
    if (mlx_get_default_stream(&s, dev) != 0 || s.ctx == NULL) { printf("FAIL stream\n"); return 1; }

    /* pesi sintetici [NE, OUT, PIN] U32 + scales [NE, OUT, IN/GS] F16 (come Mixtral) */
    int wsh[3] = {NE, OUT, PIN}, ssh[3] = {NE, OUT, IN / GS};
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), bs = mlx_array_new(), no_b = mlx_array_new();
    /* sc/bs = 1 (non-zero): il peso dequantizzato (q-b)*s non e' 0 anche con q=0,
     * cosi' il confronto B vs C e' un test reale e non un banale 0==0. */
    if (mlx_zeros(&w, wsh, 3, MLX_UINT32, s) != 0 || mlx_ones(&sc, ssh, 3, MLX_FLOAT16, s) != 0 || mlx_ones(&bs, ssh, 3, MLX_FLOAT16, s) != 0) {
        printf("FAIL pesi\n"); return 1;
    }
    /* attivazioni + indici (5-D per B: la forma reale di mlx_lm SwitchGLU) */
    int xa_sh[3] = {M, 1, IN}, xb_sh[5] = {1, T, 1, 1, IN}, xc_sh[4] = {T, 1, 1, IN};
    int ib_sh[3] = {1, T, K}, ic_sh[2] = {T, K};
    mlx_array xa = mlx_array_new(), xb = mlx_array_new(), xc = mlx_array_new();
    mlx_array ia = mlx_array_new(), ib = mlx_array_new(), ic = mlx_array_new();
    mlx_array pa = mlx_array_new(), pb = mlx_array_new(), pc = mlx_array_new();
    if (mlx_ones(&xa, xa_sh, 3, MLX_FLOAT16, s) != 0 || mlx_ones(&xb, xb_sh, 5, MLX_FLOAT16, s) != 0 ||
        mlx_ones(&xc, xc_sh, 4, MLX_FLOAT16, s) != 0) {
        printf("FAIL input\n"); return 1;
    }
    /* Indici ESPERTI VARIATI e CRESCENTI (0..NE-1 ripetuti): cosi' il ramo sorted
     * raggruppa davvero NE gruppi (come con esperti reali) e non degenera. */
    {
        mlx_array ar = mlx_array_new(), ia32 = mlx_array_new();
        const int rows_per_expert = M / (NE > 0 ? NE : 1);
        mlx_array dv = mlx_array_new_int(rows_per_expert > 0 ? rows_per_expert : 1);
        bool iok = mlx_arange(&ar, 0, (double)M, 1.0, MLX_INT32, s) == 0;
        iok = iok && mlx_floor_divide(&ia32, ar, dv, s) == 0 && mlx_astype(&ia, ia32, MLX_UINT32, s) == 0;
        iok = iok && mlx_reshape(&ib, ia, ib_sh, 3, s) == 0 && mlx_reshape(&ic, ia, ic_sh, 2, s) == 0;
        if (!iok) { printf("FAIL idx\n"); return 1; }
        mlx_array_free(ar); mlx_array_free(dv); mlx_array_free(ia32);
    }
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};

    /* A) nostra sorted/prefill: x=[M,1,C], indices=[M], sorted_indices=true */
    double ms_a = -1.0;
    {
        for (int r = 0; r < 3; r++) { if (mlx_gather_qmm(&pa, xa, w, sc, bs, no_b, ia, true, gsz, bts, "affine", true, s) != 0) { printf("FAIL gather A\n"); return 1; } }
        if (mlx_array_eval(pa) != 0) { printf("FAIL eval A\n"); return 1; }
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < REPS; r++) if (mlx_gather_qmm(&pa, xa, w, sc, bs, no_b, ia, true, gsz, bts, "affine", true, s) != 0) { printf("FAIL gather A rep\n"); return 1; }
        if (mlx_array_eval(pa) != 0) { printf("FAIL eval A rep\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms_a = ((double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1e3 / REPS;
    }
    /* B) mlx_lm unsorted: x=[1,T,1,1,C] (5-D!), indices=[1,T,k], sorted_indices=false */
    double ms_b = -1.0;
    {
        for (int r = 0; r < 3; r++) { if (mlx_gather_qmm(&pb, xb, w, sc, bs, no_b, ib, true, gsz, bts, "affine", false, s) != 0) { printf("FAIL gather B\n"); return 1; } }
        if (mlx_array_eval(pb) != 0) { printf("FAIL eval B\n"); return 1; }
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < REPS; r++) if (mlx_gather_qmm(&pb, xb, w, sc, bs, no_b, ib, true, gsz, bts, "affine", false, s) != 0) { printf("FAIL gather B rep\n"); return 1; }
        if (mlx_array_eval(pb) != 0) { printf("FAIL eval B rep\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms_b = ((double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1e3 / REPS;
    }
    /* C) nostra unsorted/decode: x=[T,1,1,C], indices=[T,k], sorted_indices=false */
    double ms_c = -1.0;
    {
        for (int r = 0; r < 3; r++) { if (mlx_gather_qmm(&pc, xc, w, sc, bs, no_b, ic, true, gsz, bts, "affine", false, s) != 0) { printf("FAIL gather C\n"); return 1; } }
        if (mlx_array_eval(pc) != 0) { printf("FAIL eval C\n"); return 1; }
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < REPS; r++) if (mlx_gather_qmm(&pc, xc, w, sc, bs, no_b, ic, true, gsz, bts, "affine", false, s) != 0) { printf("FAIL gather C rep\n"); return 1; }
        if (mlx_array_eval(pc) != 0) { printf("FAIL eval C rep\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms_c = ((double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1e3 / REPS;
    }
    /* confronto forme output */
    int nd_a = mlx_array_ndim(pa), nd_b = mlx_array_ndim(pb), nd_c = mlx_array_ndim(pc);
    const int* sa = mlx_array_shape(pa); const int* sb = mlx_array_shape(pb); const int* sc2 = mlx_array_shape(pc);
    printf("gather_qmm NE=%d OUT=%d IN=%d T=%d k=%d => M=%d (PIN=%d)\n", NE, OUT, IN, T, K, M, PIN);
    printf("  A nostra sorted  x=[M,1,C]     idx=[M]    s=true : %.3f ms  out_ndim=%d shape=[%d,%d,%d]\n",
           ms_a, nd_a, nd_a>0?sa[0]:-1, nd_a>1?sa[1]:-1, nd_a>2?sa[2]:-1);
    printf("  B mlx_lm unsort  x=[1,T,1,1,C] idx=[1,T,k] s=false: %.3f ms  out_ndim=%d shape=[%d,%d,%d,%d,%d]\n",
           ms_b, nd_b, nd_b>0?sb[0]:-1, nd_b>1?sb[1]:-1, nd_b>2?sb[2]:-1, nd_b>3?sb[3]:-1, nd_b>4?sb[4]:-1);
    printf("  C nostra unsort  x=[T,1,1,C]   idx=[T,k]  s=false: %.3f ms  out_ndim=%d shape=[%d,%d,%d,%d]\n",
           ms_c, nd_c, nd_c>0?sc2[0]:-1, nd_c>1?sc2[1]:-1, nd_c>2?sc2[2]:-1, nd_c>3?sc2[3]:-1);
    if (ms_b > 0 && ms_c > 0)
        printf("  => B/C = %.2fx  (se ~1: la forma 5-D mlx_lm non costa nulla vs la nostra 4-D)\n", ms_b/ms_c);
    if (ms_a > 0 && ms_c > 0)
        printf("  => C/A = %.2fx  (se >1: il ramo UNSORTED e' %.2fx piu' lento del sorted)\n",
               ms_c/ms_a, ms_c/ms_a);
    /* B e C sono la stessa op a meno di un singleton: i valori devono combaciare. */
    {
        mlx_array pb3 = mlx_array_new(), pc2 = mlx_array_new();
        const int b3[3] = {T, K, OUT};
        bool okcmp = mlx_reshape(&pb3, pb, b3, 3, s) == 0 && mlx_reshape(&pc2, pc, b3, 3, s) == 0;
        mlx_array d = mlx_array_new(), ad = mlx_array_new(), mxv = mlx_array_new();
        double maxdiff = -1.0;
        if (okcmp) okcmp = mlx_array_eval(pb3) == 0 && mlx_array_eval(pc2) == 0 &&
                            mlx_subtract(&d, pb3, pc2, s) == 0 && mlx_abs(&ad, d, s) == 0 &&
                            mlx_max(&mxv, ad, false, s) == 0 && mlx_array_eval(mxv) == 0;
        if (okcmp) { const float* p = mlx_array_data_float32(mxv); if (p) maxdiff = (double)p[0]; }
        printf("  max|B-C| = %.6g  (B e C sono la stessa op, forma diversa) %s\n",
               maxdiff, (maxdiff >= 0.0 && maxdiff < 1e-3) ? "OK" : "FAIL");
        mlx_array_free(pb3); mlx_array_free(pc2); mlx_array_free(d); mlx_array_free(ad); mlx_array_free(mxv);
    }
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bs); mlx_array_free(no_b);
    mlx_array_free(xa); mlx_array_free(xb); mlx_array_free(xc);
    mlx_array_free(ia); mlx_array_free(ib); mlx_array_free(ic);
    mlx_array_free(pa); mlx_array_free(pb); mlx_array_free(pc);
    mlx_stream_free(s); mlx_device_free(dev);
    return 0;
}
