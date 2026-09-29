/* llm_mlx_oq_a8_qmm_probe.c — P5(a): costo del QMM W8A8 (OQ-A8) a M piccolo, load-free.
 * Pesi sintetici Q4 (u32 packed) + scale/bias f16 [N,K/64], x f16 [M,K]; chiama il bridge
 * (llmmetal_oq_a8_stage_a + llmmetal_oq_a8_qmm, stessa strada dell'engine) e misura ms/call, TFLOPs e
 * GB/s di pesi letti. A/B varianti: LLMMETAL_OQ_A8_NOTM1=1 forza TM=2 (BM=32, vecchio default);
 * default = TM=1 (BM=16) per M<=16.
 *
 * Build (dal build/ gia' configurato; stesse include del target llm_ai_mlx):
 *   cd native/llmengine_mlx && /usr/bin/c++ -O3 -DNDEBUG -std=gnu++20 -arch arm64 -fPIC \
 *     -fobjc-arc -Ibuild/_deps/metal_cpp-src -I../../libs/mlx-deps/mlx-src -Ibuild/_deps/cjson-src \
 *     -I../llmmetal_oq_a8 -I. -I../../libs/mlx-deps/mlx-c-src \
 *     llm_mlx_oq_a8_qmm_probe.c ../llmmetal_oq_a8/llmmetal_oq_a8_bridge.mm \
 *     build/_deps/mlx_c-build/libmlxc.a build/_deps/mlx-build/libmlx.a build/_deps/cjson-build/libcjson.a \
 *     build/_deps/mlx-build/mlx/io/libgguflib.a build/jaccl/libjaccl.a \
 *     -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
 *     -o /tmp/llmmetal_oq_a8_qmm_probe
 * Uso: /tmp/llmmetal_oq_a8_qmm_probe [K] [N] [reps] [qmm.metallib] [quantize.metallib]
 */
#include "mlx/c/mlx.h"
#include "llm_engine_opts.h"
#include "llmmetal_oq_a8_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 4096;
    const int N = argc > 2 ? atoi(argv[2]) : 12288;
    const int reps = argc > 3 ? atoi(argv[3]) : 20;
    const char* qmm = argc > 4 ? argv[4] : "oq_a8_qmm_nax.metallib";
    const char* qz = argc > 5 ? argv[5] : "oq_a8_quantize.metallib";
    LlmMetalOqA8Ctx* ctx = llmmetal_oq_a8_create(qmm, qz);
    if (!ctx) { printf("FAIL: ctx (metallib: %s / %s)\n", qmm, qz); return 1; }
    mlx_stream st = mlx_default_gpu_stream_new();

    const int nw = N * (K / 8);            /* Q4 packed: K*4/32 = K/8 u32 per riga */
    uint32_t* w = (uint32_t*)malloc((size_t)nw * sizeof(uint32_t));
    for (int i = 0; i < nw; i++) w[i] = (uint32_t)(i * 2654435761u);
    int wsh[2] = {N, K / 8};
    mlx_array W = mlx_array_new_data(w, wsh, 2, MLX_UINT32);
    free(w);
    const int ns = N * (K / 64);
    uint16_t* s = (uint16_t*)malloc((size_t)ns * sizeof(uint16_t));
    for (int i = 0; i < ns; i++) s[i] = 0x3C00; /* f16 1.0 */
    int ssh[2] = {N, K / 64};
    mlx_array S = mlx_array_new_data(s, ssh, 2, MLX_FLOAT16);
    mlx_array B = mlx_array_new_data(s, ssh, 2, MLX_FLOAT16);
    free(s);

    const double wbytes = (double)nw * 4.0;
    printf("# OQ-A8 QMM W8A8 (int8 act x Q4 w) K=%d N=%d reps=%d  pesi=%.1f MB  NOTM1=%s\n",
           K, N, reps, wbytes / 1e6, llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_NOTM1") ? "1" : "0");
    printf("%6s | %10s %10s %8s\n", "M", "ms", "TFLOPs", "GB/s");
    for (int M = 1; M <= 64; M *= 2) {
        int ash[2] = {M, K};
        float* a = (float*)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) a[i] = 0.01f * (float)((i % 7) - 3);
        mlx_array A = mlx_array_new_data(a, ash, 2, MLX_FLOAT16);
        free(a);
        mlx_array qa = mlx_array_new(), sa = mlx_array_new(), ra = mlx_array_new();
        int rc = llmmetal_oq_a8_stage_a(A, &qa, &sa, &ra, st);
        double best = 1e9;
        int ok = 0;
        if (rc == 0) {
            for (int r = 0; r < reps; r++) {
                mlx_array y = llmmetal_oq_a8_qmm(qa, sa, ra, W, S, B, st, 4, 0);
                if (!y.ctx) break;
                double t0 = now();
                mlx_array_eval(y);
                double dt = now() - t0;
                mlx_array_free(y);
                if (dt < best) best = dt;
                ok = 1;
            }
        }
        if (ok)
            printf("%6d | %10.4f %10.2f %8.1f\n", M, best * 1e3,
                   2.0 * (double)M * (double)N * (double)K / best / 1e12, wbytes / best / 1e9);
        else
            printf("%6d | FAIL rc=%d\n", M, rc);
        mlx_array_free(qa); mlx_array_free(sa); mlx_array_free(ra); mlx_array_free(A);
    }
    /* Modalita' batch: R QMM indipendenti in UN grafo = un solo sync -> tempo kernel puro
     * (senza overhead CPU di eval/sync per call). R da env LLM_OQ_PROBE_BATCH (default 32). */
    {
        const char* be = getenv("LLM_OQ_PROBE_BATCH");
        const int R = be ? atoi(be) : 32;
        if (R > 0) {
            for (int M = 1; M <= 16; M *= 2) {
                int ash[2] = {M, K};
                float* a = (float*)malloc((size_t)M * K * sizeof(float));
                for (int i = 0; i < M * K; i++) a[i] = 0.01f * (float)((i % 7) - 3);
                mlx_array A = mlx_array_new_data(a, ash, 2, MLX_FLOAT16);
                free(a);
                mlx_array qa = mlx_array_new(), sa = mlx_array_new(), ra = mlx_array_new();
                if (llmmetal_oq_a8_stage_a(A, &qa, &sa, &ra, st) == 0) {
                    /* Input DISTINTI per output: con lo stesso qa MLX vede 32 output identici
                     * (stesso nodo) e il primo eval calcola tutto -> tempi 0. La Stage-A per r
                     * costa poco (M piccolo) e rende il grafo reale. */
                    mlx_array* ys = (mlx_array*)malloc(sizeof(mlx_array) * (size_t)R);
                    mlx_array* qs = (mlx_array*)malloc(sizeof(mlx_array) * (size_t)R);
                    mlx_array* ss = (mlx_array*)malloc(sizeof(mlx_array) * (size_t)R);
                    mlx_array* rs = (mlx_array*)malloc(sizeof(mlx_array) * (size_t)R);
                    int built = 0;
                    for (int r = 0; r < R; r++) {
                        qs[r] = mlx_array_new(); ss[r] = mlx_array_new(); rs[r] = mlx_array_new();
                        float* ar = (float*)malloc((size_t)M * K * sizeof(float));
                        for (int i = 0; i < M * K; i++) ar[i] = 0.01f * (float)((i % 7) - 3) + 0.001f * r;
                        mlx_array Ar = mlx_array_new_data(ar, ash, 2, MLX_FLOAT16);
                        free(ar);
                        if (llmmetal_oq_a8_stage_a(Ar, &qs[r], &ss[r], &rs[r], st) != 0) { mlx_array_free(Ar); break; }
                        mlx_array_free(Ar);
                        ys[r] = llmmetal_oq_a8_qmm(qs[r], ss[r], rs[r], W, S, B, st, 4, 0);
                        if (!ys[r].ctx) break;
                        built++;
                    }
                    /* UNA passata: ogni output e' un nodo distinto e il suo eval lo calcola;
                     * un secondo giro sarebbe no-op (cache MLX) -> min su piu' giri = 0. */
                    double t0 = now();
                    for (int r = 0; r < built; r++) mlx_array_eval(ys[r]);
                    double dt = now() - t0;
                    if (built > 0 && dt > 0)
                        printf("BATCH M=%d R=%d | %10.4f ms/call | %8.2f TFLOPs | %8.1f GB/s\n",
                               M, built, dt / built * 1e3,
                               2.0 * (double)M * (double)N * (double)K / (dt / built) / 1e12,
                               wbytes / (dt / built) / 1e9);
                    for (int r = 0; r < built; r++) {
                        mlx_array_free(ys[r]); mlx_array_free(qs[r]); mlx_array_free(ss[r]); mlx_array_free(rs[r]);
                    }
                    free(ys); free(qs); free(ss); free(rs);
                }
                mlx_array_free(qa); mlx_array_free(sa); mlx_array_free(ra); mlx_array_free(A);
            }
        }
    }
    /* A/B interleaved nello STESSO processo (LLM_OQ_PROBE_AB="1,4" = split da confrontare):
     * ogni ripetizione alterna i due env e prende il MIN per variante -> robusto al carico
     * della macchina condivisa (il min e' il run meno contendato, non la media). */
    if (const char* ab = getenv("LLM_OQ_PROBE_AB")) {
        const int R = getenv("LLM_OQ_PROBE_ABREPS") ? atoi(getenv("LLM_OQ_PROBE_ABREPS")) : 15;
        int splits[8], nsp = 0;
        char tmp[64]; snprintf(tmp, sizeof tmp, "%s", ab);
        for (char* tok = strtok(tmp, ","); tok && nsp < 8; tok = strtok(NULL, ",")) splits[nsp++] = atoi(tok);
        for (int M = 1; M <= 16; M *= 2) {
            int ash[2] = {M, K};
            float* a = (float*)malloc((size_t)M * K * sizeof(float));
            for (int i = 0; i < M * K; i++) a[i] = 0.01f * (float)((i % 7) - 3);
            mlx_array A = mlx_array_new_data(a, ash, 2, MLX_FLOAT16);
            free(a);
            mlx_array qa = mlx_array_new(), sa = mlx_array_new(), ra = mlx_array_new();
            if (llmmetal_oq_a8_stage_a(A, &qa, &sa, &ra, st) == 0) {
                double best[8];
                for (int i = 0; i < nsp; i++) best[i] = 1e9;
                for (int rep = 0; rep < R; rep++) {
                    for (int i = 0; i < nsp; i++) {
                        char sv[16]; snprintf(sv, sizeof sv, "%d", splits[i]);
                        setenv("LLMMETAL_OQ_A8_SPLITK", sv, 1);
                        mlx_array y = llmmetal_oq_a8_qmm(qa, sa, ra, W, S, B, st, 4, 0);
                        if (y.ctx) {
                            double t0 = now();
                            mlx_array_eval(y);
                            double dt = now() - t0;
                            if (dt < best[i]) best[i] = dt;
                            mlx_array_free(y);
                        }
                    }
                }
                unsetenv("LLMMETAL_OQ_A8_SPLITK");
                for (int i = 0; i < nsp; i++)
                    if (best[i] < 1e8)
                        printf("AB M=%d split=%d | %10.4f ms | %8.2f TFLOPs | %8.1f GB/s\n",
                               M, splits[i], best[i] * 1e3,
                               2.0 * (double)M * (double)N * (double)K / best[i] / 1e12,
                               wbytes / best[i] / 1e9);
            }
            mlx_array_free(qa); mlx_array_free(sa); mlx_array_free(ra); mlx_array_free(A);
        }
    }
    mlx_array_free(W); mlx_array_free(S); mlx_array_free(B);
    llmmetal_oq_a8_destroy(ctx);
    return 0;
}
