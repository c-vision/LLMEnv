/* oq_a8_prep_cache_selfcheck.c — regressione del bug MLX-A8-Prefill-MultiChunk-Degenera.
 *
 * Il primitive A8 chiamato DUE volte con GLI STESSI input (con mlx_clear_cache() fra le due,
 * pattern del prefill multi-chunk) deve dare lo STESSO output. La cache prep scT/biT del bridge
 * riallocava il buffer ad ogni eval_gpu ma lo riempiva solo la prima volta -> la 2a chiamata
 * leggeva memoria mai scritta (output degenere). Fix in llmmetal_oq_a8_bridge.mm (alloc solo alla
 * creazione della Prep).
 *
 * Build (dalla root del repo):
 *   clang -O2 -I libs/mlx-deps/mlx-c-src -I native/llmmetal_oq_a8 \
 *     native/llmmetal_oq_a8/tests/oq_a8_prep_cache_selfcheck.c \
 *     -L native/llmengine_mlx/build -lllmmetal_ai_mlx \
 *     -Wl,-rpath,$PWD/native/llmengine_mlx/build -o /tmp/oq_a8_prep_cache_selfcheck
 * Uso:
 *   /tmp/oq_a8_prep_cache_selfcheck [qmm.metallib] [quantize.metallib] [M] [N] [K]
 * Exit: 0 = call2 == call1 (PASS), 1 = DIVERGE (bug), 2 = setup fallito.
 */
#include <mlx/c/mlx.h>
#include "llmmetal_oq_a8_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double maxdiff_f32(mlx_array a, mlx_array b, double* mean_out) {
    mlx_array af = mlx_array_new(), bf = mlx_array_new();
    mlx_stream st = mlx_stream_new();
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_get_default_stream(&st, dev);
    mlx_astype(&af, a, MLX_FLOAT32, st);
    mlx_astype(&bf, b, MLX_FLOAT32, st);
    mlx_array_eval(af); mlx_array_eval(bf);
    const float* ad = mlx_array_data_float32(af);
    const float* bd = mlx_array_data_float32(bf);
    size_t n = mlx_array_size(af);
    double mx = 0, su = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)ad[i] - (double)bd[i]); if (d > mx) mx = d; su += d; }
    *mean_out = n ? su / (double)n : 0.0;
    mlx_array_free(af); mlx_array_free(bf);
    mlx_device_free(dev);
    return mx;
}

int main(int argc, char** argv) {
    const char* qmm = argc > 1 ? argv[1] : "native/llmengine_mlx/build/oq_a8_qmm_nax.metallib";
    const char* qnt = argc > 2 ? argv[2] : "native/llmengine_mlx/build/oq_a8_quantize.metallib";
    int M = argc > 3 ? atoi(argv[3]) : 256;
    int N = argc > 4 ? atoi(argv[4]) : 256;
    int K = argc > 5 ? atoi(argv[5]) : 256;
    if (K % 64 || N % 64 || M < 16) { fprintf(stderr, "shape non eleggibile\n"); return 2; }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(dev) != 0) { fprintf(stderr, "set_default_device fail\n"); return 2; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, dev) != 0) { fprintf(stderr, "get_default_stream fail\n"); return 2; }

    /* x bf16 [M,K] random (path use_bf16 -> nessun cast/copia nel C ABI: puntatori stabili,
     * necessari perche' la cache prep sia esercitata). w U32 random = Q4 valido; sc=1 bi=0. */
    float* xf = (float*)malloc((size_t)M * K * sizeof(float));
    uint32_t* wb = (uint32_t*)malloc((size_t)N * (K / 8) * sizeof(uint32_t));
    float* ones = (float*)malloc((size_t)N * (K / 64) * sizeof(float));
    float* zeros = (float*)calloc((size_t)N * (K / 64), sizeof(float));
    if (!xf || !wb || !ones || !zeros) { fprintf(stderr, "malloc fail\n"); return 2; }
    srand(7);
    for (size_t i = 0; i < (size_t)M * K; i++) xf[i] = (rand() % 2001 - 1000) / 500.0f;
    for (size_t i = 0; i < (size_t)N * (K / 8); i++) wb[i] = (uint32_t)rand();
    for (size_t i = 0; i < (size_t)N * (K / 64); i++) ones[i] = 1.0f;

    int xsh[2] = {M, K}, wsh[2] = {N, K / 8}, gsh[2] = {N, K / 64};
    mlx_array x32 = mlx_array_new_data(xf, xsh, 2, MLX_FLOAT32);
    mlx_array w = mlx_array_new_data(wb, wsh, 2, MLX_UINT32);
    mlx_array sc32 = mlx_array_new_data(ones, gsh, 2, MLX_FLOAT32);
    mlx_array bi32 = mlx_array_new_data(zeros, gsh, 2, MLX_FLOAT32);
    mlx_array x = mlx_array_new(), sc = mlx_array_new(), bi = mlx_array_new();
    mlx_astype(&x, x32, MLX_BFLOAT16, st);
    mlx_astype(&sc, sc32, MLX_BFLOAT16, st);
    mlx_astype(&bi, bi32, MLX_BFLOAT16, st);
    mlx_array_eval(x); mlx_array_eval(w); mlx_array_eval(sc); mlx_array_eval(bi);
    mlx_clear_cache();

    LlmMetalOqA8Ctx* ctx = llmmetal_oq_a8_create(qmm, qnt);
    if (!ctx) { fprintf(stderr, "ctx NULL (metallib?)\n"); return 2; }

    /* CALL 1 (crea la cache prep) */
    mlx_array r1 = llmmetal_oq_a8_prim(x, w, sc, bi, st, 4);
    if (r1.ctx == NULL) { fprintf(stderr, "prim call1 NULL\n"); return 2; }
    mlx_array_eval(r1);

    /* confine di chunk */
    mlx_clear_cache();

    /* CALL 2 (stessi input: deve dare lo STESSO output) */
    mlx_array r2 = llmmetal_oq_a8_prim(x, w, sc, bi, st, 4);
    if (r2.ctx == NULL) { fprintf(stderr, "prim call2 NULL\n"); return 2; }
    mlx_array_eval(r2);

    double mean12 = 0, mean1r = 0;
    double mx12 = maxdiff_f32(r1, r2, &mean12);

    /* sanity: r1 vs mlx_quantized_matmul (tolleranza int8) */
    mlx_array ref = mlx_array_new();
    mlx_optional_int gs = {64, true}, bt = {4, true};
    int rc = mlx_quantized_matmul(&ref, x, w, sc, bi, /*transpose=*/true, gs, bt, "affine", st);
    double mx1r = -1;
    if (rc == 0) { mlx_array_eval(ref); mx1r = maxdiff_f32(r1, ref, &mean1r); }

    printf("M=%d N=%d K=%d | call1-vs-call2 max=%.6g mean=%.6g | call1-vs-qmm max=%.6g mean=%.6g\n",
           M, N, K, mx12, mean12, mx1r, mean1r);
    printf("VERDETTO: %s\n", (mx12 == 0.0) ? "PASS (call2 == call1)" : "*** FAIL: call2 DIVERGE da call1 ***");

    mlx_array_free(ref); mlx_array_free(r2); mlx_array_free(r1);
    mlx_array_free(x); mlx_array_free(sc); mlx_array_free(bi);
    mlx_array_free(x32); mlx_array_free(w); mlx_array_free(sc32); mlx_array_free(bi32);
    llmmetal_oq_a8_destroy(ctx);
    free(xf); free(wb); free(ones); free(zeros);
    mlx_device_free(dev);
    return (mx12 == 0.0) ? 0 : 1;
}
