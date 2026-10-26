/* Parita' q8 wide: gemv Q8 su GPU vs riferimento double su dequant. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_gpu.h"
#include "qwen4_native/qwen4_native_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float frand(void) { return (float)((int)(rand() % 2001) - 1000) / 1000.0f; }

int main(void) {
    const int n_in = 2560, n_out = 512;
    const int nblk = n_in / 32;
    const size_t rb = (size_t)nblk * 34;
    uint8_t* W = malloc(rb * (size_t)n_out);
    float* x = malloc((size_t)n_in * sizeof(float));
    float* y = malloc((size_t)n_out * sizeof(float));
    float* ref = malloc((size_t)n_out * sizeof(float));
    if (!W || !x || !y || !ref) return 1;
    srand(7);
    for (int o = 0; o < n_out; o++) {
        uint8_t* row = W + (size_t)o * rb;
        for (int b = 0; b < nblk; b++) {
            uint8_t* blk = row + (size_t)b * 34;
            _Float16 d = (_Float16)(0.001f + 0.0001f * (float)(b % 7));
            __builtin_memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(int8_t)(rand() % 255 - 127);
        }
    }
    for (int i = 0; i < n_in; i++) x[i] = frand();
    for (int o = 0; o < n_out; o++) {
        const uint8_t* row = W + (size_t)o * rb;
        double acc = 0.0;
        for (int b = 0; b < nblk; b++) {
            const uint8_t* blk = row + (size_t)b * 34;
            _Float16 dh; __builtin_memcpy(&dh, blk, 2);
            double d = (double)dh, s = 0.0;
            for (int i = 0; i < 32; i++) s += (double)(int8_t)blk[2 + i] * (double)x[b * 32 + i];
            acc += d * s;
        }
        ref[o] = (float)acc;
    }
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (!dev.ctx || mlx_set_default_device(dev) != 0) return 1;
    Q4nGpu g; if (q4n_gpu_init(&g) != 0) return 1;
    Q4nMat mat = { W, 8, rb };
    if (q4n_mat_mv_gpu(&g, &mat, x, y, n_in, n_out) != 0) { printf("gemv FAIL\n"); return 1; }
    double maxe = 0.0, maxr = 0.0;
    for (int o = 0; o < n_out; o++) { double e = fabs((double)y[o] - (double)ref[o]); if (e > maxe) maxe = e; if (fabs((double)ref[o]) > maxr) maxr = fabs((double)ref[o]); }
    printf("max_abs_err=%.6g  max_ref=%.6g  rel=%.3g\n", maxe, maxr, maxe / (maxr > 0 ? maxr : 1.0));
    q4n_gpu_free(&g);
    free(W); free(x); free(y); free(ref);
    return (maxe / (maxr > 0 ? maxr : 1.0) < 1e-4) ? 0 : 1;
}
