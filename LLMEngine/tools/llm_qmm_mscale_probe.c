/* llm_qmm_mscale_probe.c — M-scaling di mlx_quantized_matmul (MLX puro, model-free).
 * Branch-first per la verify multi-token: a quale M il matmul quantizzato PERDE banda?
 * (rapporto GB/s a M=1..32 su forme reali). Vedi Todo/(open) - Spec-Verify-Perf-e-MTP-Da-Strata.md.
 *
 * Build: bash LLMEngine/tools/build_qmm_mscale.sh   (o a mano, vedi commento in fondo).
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* innesti MLX del motore (patch nostre): qmv_wide per M>=2, split-K per qmv M=1.
 * Prototipi già in mlx/c/memory.h (incluso da mlx/c/mlx.h). */

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static double bench(int N, int K, int bits, int group, int M, int reps) {
    size_t Kw = (size_t)K * bits / 32, G = (size_t)K / group;
    uint32_t* w = (uint32_t*)calloc((size_t)N * Kw, 4);
    _Float16* sc = (_Float16*)malloc((size_t)N * G * 2), *bs = (_Float16*)malloc((size_t)N * G * 2);
    _Float16* x = (_Float16*)malloc((size_t)M * K * 2);
    for (size_t i = 0; i < (size_t)N * Kw; i++) w[i] = (uint32_t)rand();
    for (size_t i = 0; i < (size_t)N * G; i++) { sc[i] = (_Float16)0.01f; bs[i] = (_Float16)0.001f; }
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (_Float16)(((rand() % 2000) - 1000) * 0.001f);

    mlx_device d = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(d);
    mlx_stream s = mlx_stream_new();
    mlx_get_default_stream(&s, d);

    int wsh[2] = {N, (int)Kw}, gsh[2] = {N, (int)G}, xsh[2] = {M, K};
    mlx_array aw = mlx_array_new_data(w, wsh, 2, MLX_UINT32);
    mlx_array asc = mlx_array_new_data(sc, gsh, 2, MLX_FLOAT16);
    mlx_array abs_ = mlx_array_new_data(bs, gsh, 2, MLX_FLOAT16);
    mlx_array ax = mlx_array_new_data(x, xsh, 2, MLX_FLOAT16);
    mlx_optional_int gsz = {group, true}, bts = {bits, true};

    mlx_array* outs = (mlx_array*)calloc((size_t)reps, sizeof(mlx_array));
    for (int r = 0; r < 2; r++) { mlx_array o = mlx_array_new(); mlx_quantized_matmul(&o, ax, aw, asc, abs_, true, gsz, bts, "affine", s); mlx_array_eval(o); mlx_array_free(o); }
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) { outs[r] = mlx_array_new(); mlx_quantized_matmul(&outs[r], ax, aw, asc, abs_, true, gsz, bts, "affine", s); }
    for (int r = 0; r < reps; r++) mlx_array_eval(outs[r]);
    double ms = (now_ms() - t0) / reps;
    for (int r = 0; r < reps; r++) mlx_array_free(outs[r]);
    mlx_array_free(ax); mlx_array_free(aw); mlx_array_free(asc); mlx_array_free(abs_); free(outs);
    free(w); free(sc); free(bs); free(x);
    mlx_device_free(d); mlx_stream_free(s);
    return ms;
}

static int cmp_d(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

int main(int argc, char** argv) {
    int bits = argc > 1 ? atoi(argv[1]) : 8;
    int group = argc > 2 ? atoi(argv[2]) : 64;
    int reps = argc > 3 ? atoi(argv[3]) : 20;
    int trials = argc > 4 ? atoi(argv[4]) : 5;
    /* 2026-10-31: i numeri del probe erano STOCK MLX. L'engine imposta due nostri innesti:
     * `qmv_wide` (kernel per M>=2, cioe' il VERIFY) e `split_k` (qmv M=1). Qui si possono
     * forzare per misurare il path che gira davvero: arg5 = qmv_wide, arg6 = split_k (-1 = lascia). */
    int qw = argc > 5 ? atoi(argv[5]) : -1;
    int sk = argc > 6 ? atoi(argv[6]) : -1;
    if (qw >= 0) { mlx_set_qmv_wide(qw); printf("# qmv_wide=%d\n", qw); }
    if (sk >= 0) { mlx_set_qmv_split_k(sk); printf("# qmv_split_k=%d\n", sk); }
    if (trials < 1) trials = 1; if (trials > 25) trials = 25;
    int Mv[] = {1, 2, 4, 8, 16, 32};
    const int NM = (int)(sizeof(Mv) / sizeof(Mv[0]));
    int shapes[][2] = {{17408, 5120}, {5120, 17408}, {5120, 5120}};
    printf("# %dbit g%d  reps=%d trials=%d  (mediana di %d prove INDIPENDENTI, M in ordine ruotato)\n",
           bits, group, reps, trials, trials);
    printf("%-22s %-5s %-10s %-10s %-10s %-9s\n", "shape NxK", "M", "mediana ms", "min ms", "max ms", "GB/s(W) med");
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        int N = shapes[si][0], K = shapes[si][1];
        double samples[32][25];
        for (int mi = 0; mi < NM; mi++)
            for (int t = 0; t < trials; t++) samples[mi][t] = 0.0;
        /* M ruotati a ogni trial: un P-state che deriva nel tempo colpisce tutti gli M allo stesso modo */
        for (int t = 0; t < trials; t++)
            for (int k = 0; k < NM; k++) {
                int mi = (k + t) % NM;
                samples[mi][t] = bench(N, K, bits, group, Mv[mi], reps);
            }
        for (int mi = 0; mi < NM; mi++) {
            qsort(samples[mi], (size_t)trials, sizeof(double), cmp_d);
            double med = samples[mi][trials / 2], mn = samples[mi][0], mx = samples[mi][trials - 1];
            double wbytes = (double)N * K * bits / 8.0;
            char nm[32]; snprintf(nm, sizeof(nm), "%dx%d", N, K);
            printf("%-22s %-5d %-10.4f %-10.4f %-10.4f %-9.1f\n", nm, Mv[mi], med, mn, mx, wbytes / med / 1e6);
        }
    }
    return 0;
}
