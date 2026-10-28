/* llm_m5_tensor_path_probe.c — ORACOLO model-free (todo Rigel/M5): caratterizza il tensor path
 * dell'M5 Max e misura la fusione dell'epilogo SwiGLU nel regime PREFILL (compute-bound).
 *   T1: throughput GEMM bf16 a forme prefill (TFLOP/s) + qmm 4-bit a M grande;
 *   T2: epilogo SwiGLU eager (sigmoid+mul+mul) vs closure compilata (come
 *       llm_mlx_swiglu_compiled, default ON nel prefill) a M=2048;
 *   T3: bimodalita' dei kernel CORTI (spread min/max su burst brevi).
 * Uso: llm_m5_tensor_path_probe [reps]   (default 9)
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static mlx_stream g_st;
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static uint32_t g_rng = 0x3c6ef372u;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static double median(double* v, int n) {
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

/* T2: closure silu(gate)*up (come llm_mlx_swiglu_compiled) */
static int swiglu_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    (void)payload;
    mlx_array g = mlx_array_new(), u = mlx_array_new(), sg = mlx_array_new(), s = mlx_array_new(), act = mlx_array_new();
    if (mlx_vector_array_get(&g, in, 0) != 0 || mlx_vector_array_get(&u, in, 1) != 0) return 1;
    int rc = mlx_sigmoid(&sg, g, g_st);
    if (rc == 0) rc = mlx_multiply(&s, g, sg, g_st);
    if (rc == 0) rc = mlx_multiply(&act, s, u, g_st);
    if (rc == 0) { mlx_array o[1] = {act}; mlx_vector_array b = mlx_vector_array_new_data(o, 1); mlx_vector_array_set(out, b); mlx_vector_array_free(b); }
    mlx_array_free(g); mlx_array_free(u); mlx_array_free(sg); mlx_array_free(s); mlx_array_free(act);
    return rc;
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 9;
    if (reps > 50) reps = 50;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();
    mlx_array no_key = mlx_array_new();

    mlx_closure raw = mlx_closure_new_func_payload(swiglu_fn, NULL, NULL);
    mlx_closure swiglu = mlx_closure_new();
    mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
    int crc = mlx_compile(&swiglu, raw, /*shapeless=*/true);
    mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
    mlx_closure_free(raw);
    if (crc != 0) { printf("mlx_compile FALLITO\n"); return 1; }

    printf("llm_m5_tensor_path_probe — M5 Max, bf16 GEMM + epilogo SwiGLU (prefill)\n");
    /* T1: GEMM bf16 a forme prefill */
    printf("\n[T1] GEMM bf16 (matmul, mediana di %d)\n", reps);
    printf("%-22s %-10s %-10s\n", "shape MxKxN", "ms", "TFLOP/s");
    const int shapes[][3] = {{512, 4096, 4096}, {2048, 4096, 4096}, {4096, 4096, 4096}, {2048, 4096, 14336}};
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        int M = shapes[si][0], K = shapes[si][1], N = shapes[si][2];
        mlx_array a = mlx_array_new(), b = mlx_array_new();
        mlx_random_normal(&a, (const int[]){M, K}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_random_normal(&b, (const int[]){K, N}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_array_eval(a); mlx_array_eval(b);
        double ts[64];
        for (int r = 0; r < reps + 2; r++) {
            mlx_array c = mlx_array_new();
            double t0 = now_ms();
            mlx_matmul(&c, a, b, g_st);
            mlx_array_eval(c);
            double t1 = now_ms();
            if (r >= 2) ts[r - 2] = t1 - t0;
            mlx_array_free(c);
        }
        double med = median(ts, reps);
        double tflops = 2.0 * M * N * K / (med / 1000.0) / 1e12;
        printf("%dx%dx%d%*s %-10.3f %-10.2f\n", M, K, N, 22 - (int)snprintf(NULL, 0, "%dx%dx%d", M, K, N), " ", med, tflops);
        mlx_array_free(a); mlx_array_free(b);
    }

    /* T2: epilogo SwiGLU eager vs compilato a M=2048 (regime prefill) */
    printf("\n[T2] epilogo SwiGLU: eager vs closure compilata (M=2048, K=N=4096)\n");
    {
        const int M = 2048, K = 4096, N = 4096;
        mlx_array x = mlx_array_new(), wg = mlx_array_new(), wu = mlx_array_new();
        mlx_random_normal(&x, (const int[]){M, K}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_random_normal(&wg, (const int[]){K, N}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_random_normal(&wu, (const int[]){K, N}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_array_eval(x); mlx_array_eval(wg); mlx_array_eval(wu);
        double te[64], tc[64];
        for (int r = 0; r < reps + 2; r++) {
            mlx_array g = mlx_array_new(), u = mlx_array_new(), act = mlx_array_new();
            double t0 = now_ms();
            mlx_matmul(&g, x, wg, g_st);
            mlx_matmul(&u, x, wu, g_st);
            mlx_array sg = mlx_array_new(), s = mlx_array_new();
            mlx_sigmoid(&sg, g, g_st);
            mlx_multiply(&s, g, sg, g_st);
            mlx_multiply(&act, s, u, g_st);
            mlx_array_eval(act);
            double t1 = now_ms();
            mlx_array_free(sg); mlx_array_free(s); mlx_array_free(act);
            mlx_array_free(g); mlx_array_free(u);
            mlx_array g2 = mlx_array_new(), u2 = mlx_array_new();
            double t2 = now_ms();
            mlx_matmul(&g2, x, wg, g_st);
            mlx_matmul(&u2, x, wu, g_st);
            mlx_array ins[2] = {g2, u2};
            mlx_vector_array in = mlx_vector_array_new_data(ins, 2);
            mlx_vector_array res = mlx_vector_array_new();
            mlx_closure_apply(&res, swiglu, in);
            mlx_array act2 = mlx_array_new();
            mlx_vector_array_get(&act2, res, 0);
            mlx_array_eval(act2);
            double t3 = now_ms();
            mlx_array_free(act2); mlx_vector_array_free(res); mlx_vector_array_free(in);
            mlx_array_free(g2); mlx_array_free(u2);
            if (r >= 2) { te[r - 2] = t1 - t0; tc[r - 2] = t3 - t2; }
        }
        double me = median(te, reps), mc = median(tc, reps);
        printf("eager    ms=%.3f (%.2f TFLOP/s eff.)\n", me, 4.0 * M * N * K / (me / 1000.0) / 1e12);
        printf("compiled ms=%.3f (%.2f TFLOP/s eff.)  ratio=%.3f  %s\n", mc,
               4.0 * M * N * K / (mc / 1000.0) / 1e12, mc / me, mc < me ? "COMPILED PAGA" : "compiled NON paga");
        mlx_array_free(x); mlx_array_free(wg); mlx_array_free(wu);
    }

    /* T3: bimodalita' kernel CORTI (M=64): spread min/max in burst brevi */
    printf("\n[T3] kernel corto (GEMM 64x4096x4096): spread min/mediana/max (bimodalita' P-state)\n");
    {
        const int M = 64, K = 4096, N = 4096;
        mlx_array a = mlx_array_new(), b = mlx_array_new();
        mlx_random_normal(&a, (const int[]){M, K}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_random_normal(&b, (const int[]){K, N}, 2, MLX_BFLOAT16, 0.5f, 1.0f, no_key, g_st);
        mlx_array_eval(a); mlx_array_eval(b);
        double ts[64]; double mn = 1e30, mx = 0;
        for (int r = 0; r < reps + 2; r++) {
            mlx_array c = mlx_array_new();
            double t0 = now_ms();
            mlx_matmul(&c, a, b, g_st);
            mlx_array_eval(c);
            double t1 = now_ms();
            if (r >= 2) { ts[r - 2] = t1 - t0; if (ts[r - 2] < mn) mn = ts[r - 2]; if (ts[r - 2] > mx) mx = ts[r - 2]; }
            mlx_array_free(c);
        }
        printf("min=%.4f mediana=%.4f max=%.4f  max/min=%.2f\n", mn, median(ts, reps), mx, mn > 0 ? mx / mn : 0.0);
        mlx_array_free(a); mlx_array_free(b);
    }

    mlx_closure_free(swiglu);
    mlx_array_free(no_key);
    mlx_stream_free(g_st); mlx_device_free(dev);
    return 0;
}
