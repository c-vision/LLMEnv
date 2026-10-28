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
    printf("q8: max_abs_err=%.6g  max_ref=%.6g  rel=%.3g\n", maxe, maxr, maxe / (maxr > 0 ? maxr : 1.0));
    /* Q4_0: blocco 18 B (d f16 + 16 byte nibble) */
    { const size_t rb4 = (size_t)(n_in / 32) * 18; uint8_t* W4 = malloc(rb4 * (size_t)n_out);
      float* y4 = malloc((size_t)n_out * sizeof(float)); float* ref4 = malloc((size_t)n_out * sizeof(float));
      if (!W4 || !y4 || !ref4) return 1;
      for (int o = 0; o < n_out; o++) { uint8_t* row = W4 + (size_t)o * rb4;
        for (int b = 0; b < n_in/32; b++) { uint8_t* blk = row + (size_t)b*18;
          _Float16 d = (_Float16)(0.002f + 0.0002f * (float)(b % 5)); __builtin_memcpy(blk, &d, 2);
          for (int i = 0; i < 16; i++) blk[2+i] = (uint8_t)((rand()%16) | ((rand()%16) << 4)); } }
      for (int o = 0; o < n_out; o++) { const uint8_t* row = W4 + (size_t)o*rb4; double acc = 0;
        for (int b = 0; b < n_in/32; b++) { const uint8_t* blk = row + (size_t)b*18; _Float16 dh; __builtin_memcpy(&dh, blk, 2);
          double d = (double)dh;
          for (int i = 0; i < 16; i++) { uint8_t q0 = blk[2+i];
            acc += d * (((int)(q0 & 0xF) - 8) * (double)x[b*32+i] + ((int)(q0 >> 4) - 8) * (double)x[b*32+16+i]); } }
        ref4[o] = (float)acc; }
      Q4nMat m4 = { W4, 2, rb4 };
      if (q4n_mat_mv_gpu(&g, &m4, x, y4, n_in, n_out) != 0) { printf("q4_0 FAIL\n"); return 1; }
      double e4 = 0, r4 = 0;
      for (int o = 0; o < n_out; o++) { double e = fabs((double)y4[o]-(double)ref4[o]); if (e > e4) e4 = e; if (fabs((double)ref4[o]) > r4) r4 = fabs((double)ref4[o]); }
      printf("q4_0: max_abs_err=%.6g rel=%.3g\n", e4, e4/(r4>0?r4:1.0));
      if (e4/(r4>0?r4:1.0) >= 1e-4) return 1;
      free(W4); free(y4); free(ref4); }
    q4n_gpu_free(&g);
    free(W); free(x); free(y); free(ref);
    return (maxe / (maxr > 0 ? maxr : 1.0) < 1e-4) ? 0 : 1;
}
