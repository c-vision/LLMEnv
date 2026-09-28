/* llm_mlx_qmm_smallm_probe.c — P5 (Splash): efficienza del quantized_matmul MLX a M piccolo
 * (verify DFlash2 = M 8) e confronto col kernel custom `llm_qmm_smallm`
 * (llm_mlx_qmm_smallm_kernel.h: dequant in threadgroup + simdgroup MMA).
 * Load-free: pesi sintetici 8-bit affine group 64, shape reali del 27B (5120->17408, ecc.).
 *
 * Build (dal build/ gia' configurato):
 *   cd native/llmengine_mlx && /usr/bin/c++ -std=c++20 -O2 -Wno-unused-function \
 *     -I. -I../../libs/mlx-deps/mlx-c-src -I../../libs/mlx-deps/mlx-src \
 *     -Ibuild/_deps/metal_cpp-src/include \
 *     llm_mlx_qmm_smallm_probe.c build/_deps/mlx_c-build/libmlxc.a \
 *     build/_deps/mlx-build/libmlx.a build/_deps/cjson-build/libcjson.a \
 *     build/_deps/mlx-build/mlx/io/libgguflib.a build/jaccl/libjaccl.a \
 *     -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
 *     -o /tmp/llm_qmm_smallm_probe
 * Uso: /tmp/llm_qmm_smallm_probe [K] [N] [reps]
 */
#include "mlx/c/mlx.h"
#include "llm_mlx_qmm_smallm_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 5120;
    const int N = argc > 2 ? atoi(argv[2]) : 17408;
    const int reps = argc > 3 ? atoi(argv[3]) : 20;
    const int G = 64;
    mlx_stream st = mlx_default_gpu_stream_new();

    /* pesi sintetici: u32 random, scales/biases bf16 (valori piccoli: 0.01/0.0) */
    const int nw = N * (K / 4);
    uint32_t* w = (uint32_t*)malloc((size_t)nw * sizeof(uint32_t));
    for (int i = 0; i < nw; i++) w[i] = (uint32_t)(i * 2654435761u);
    int wsh[2] = {N, K / 4};
    mlx_array W = mlx_array_new_data(w, wsh, 2, MLX_UINT32);
    free(w);
    const int ns = N * (K / G);
    uint16_t* s16 = (uint16_t*)malloc((size_t)ns * sizeof(uint16_t));
    for (int i = 0; i < ns; i++) s16[i] = 0x3F80; /* 1.0 bf16 */
    int ssh[2] = {N, K / G};
    mlx_array S = mlx_array_new_data(s16, ssh, 2, MLX_BFLOAT16);
    mlx_array B = mlx_array_new_data(s16, ssh, 2, MLX_BFLOAT16);
    free(s16);

    printf("# QMM affine 8-bit group=%d  K=%d N=%d  reps=%d\n", G, K, N, reps);
    printf("%6s | %10s %10s %8s | %10s %10s %8s | %10s\n",
           "M", "mlx_ms", "ms/tok", "TFLOPs", "cust_ms", "ms/tok", "TFLOPs", "maxdiff");
    for (int M = 1; M <= 16; M *= 2) {
        const int ash[3] = {1, M, K};
        float* a = (float*)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) a[i] = 0.01f * (float)((i % 7) - 3);
        mlx_array A = mlx_array_new_data(a, ash, 3, MLX_FLOAT32);
        mlx_array A16 = mlx_array_new();
        mlx_astype(&A16, A, MLX_BFLOAT16, st);
        free(a);

        mlx_optional_int gsz = {G, true}, bts = {8, true};
        /* MLX */
        mlx_array out = mlx_array_new();
        int rc = mlx_quantized_matmul(&out, A16, W, S, B, /*transpose=*/true, gsz, bts, "affine", st);
        double mlx_ms = 0.0;
        if (rc == 0) {
            mlx_array_eval(out);
            double best = 1e9;
            for (int r = 0; r < reps; r++) {
                double t0 = now();
                mlx_array o2 = mlx_array_new();
                mlx_quantized_matmul(&o2, A16, W, S, B, true, gsz, bts, "affine", st);
                mlx_array_eval(o2);
                double dt = now() - t0;
                mlx_array_free(o2);
                if (dt < best) best = dt;
            }
            mlx_ms = best;
        }
        const double flops = 2.0 * (double)M * (double)N * (double)K;

        /* custom (solo M<=8, K%64==0, N%32==0) */
        double cust_ms = -1.0, maxdiff = -1.0;
        if (M == 8 && (K % 64) == 0 && (N % 64) == 0) {
            mlx_array yout = mlx_array_new();
            int crc = llm_qmm_smallm(&yout, A16, W, S, B, M, K, N, st);
            if (crc == 0) {
                if (mlx_array_eval(yout) != 0)
                    fprintf(stderr, "[qmm-smallm] EVAL ERR: %s\n", g_qmm_sm_err);
                double best = 1e9;
                for (int r = 0; r < reps; r++) {
                    mlx_array y2 = mlx_array_new();
                    if (llm_qmm_smallm(&y2, A16, W, S, B, M, K, N, st) == 0) {
                        double t0 = now();
                        mlx_array_eval(y2);
                        double dt = now() - t0;
                        if (dt < best) best = dt;
                    }
                    mlx_array_free(y2);
                }
                cust_ms = best;
                /* maxdiff: astype f32 di entrambi */
                mlx_array f1 = mlx_array_new(), f2 = mlx_array_new();
                if (mlx_astype(&f1, out, MLX_FLOAT32, st) == 0 && mlx_astype(&f2, yout, MLX_FLOAT32, st) == 0 &&
                    mlx_array_eval(f1) == 0 && mlx_array_eval(f2) == 0) {
                    const float* d1 = mlx_array_data_float32(f1);
                    const float* d2 = mlx_array_data_float32(f2);
                    size_t n = mlx_array_size(f1);
                    if (d1 && d2 && n == mlx_array_size(f2)) {
                        double md = 0.0;
                        for (size_t i = 0; i < n; i++) { double df = (double)d1[i] - (double)d2[i]; if (df < 0) df = -df; if (df > md) md = df; }
                        maxdiff = md;
                    }
                }
                mlx_array_free(f1); mlx_array_free(f2);
            }
            mlx_array_free(yout);
        }
        if (cust_ms > 0)
            printf("%6d | %10.4f %10.4f %8.2f | %10.4f %10.4f %8.2f | %10.6f\n",
                   M, mlx_ms * 1e3, mlx_ms * 1e3 / M, flops / (mlx_ms * 1e12),
                   cust_ms * 1e3, cust_ms * 1e3 / M, flops / (cust_ms * 1e12), maxdiff);
        else
            printf("%6d | %10.4f %10.4f %8.2f | %10s %10s %8s | %10s\n",
                   M, mlx_ms * 1e3, mlx_ms * 1e3 / M, flops / (mlx_ms * 1e12), "-", "-", "-", "-");
        mlx_array_free(out); mlx_array_free(A); mlx_array_free(A16);
    }
    mlx_array_free(W); mlx_array_free(S); mlx_array_free(B);
    mlx_stream_free(st);
    return 0;
}
