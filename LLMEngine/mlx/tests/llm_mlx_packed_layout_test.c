/* Verifica il LAYOUT affine MLX (M2, gate 1): GEMV Q4 del bridge (puntatori raw) contro
 * mlx_dequantize sul tensore reale. Se i valori coincidono, il packing del nostro kernel e'
 * compatibile con quello di MLX; altrimenti il test stampa i primi valori per correggerlo. */
#include "llm_mlx_wpacked.h"
#include "llm_gemvq4_bridge.h"
#include "mlx/c/mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

static float frand(void) { return (float)((rand() % 2001) - 1000) / 1000.0f; }

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "uso: %s <safetensors> <tensor_base> <bits> <group>\n", argv[0]); return 2; }
    const int bits = atoi(argv[3]), group = atoi(argv[4]);
    LlmMlxPackedW pw;
    if (llm_mlx_packed_open(argv[1], argv[2], bits, group, &pw) != 0) { printf("open FAIL\n"); return 1; }
    printf("tensor rows=%u cols=%u bits=%d group=%d w=%zu s=%zu b=%zu\n", pw.rows, pw.cols, pw.bits, pw.group, pw.w_bytes, pw.s_bytes, pw.b_bytes);
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream s = mlx_stream_new_device(dev);
    mlx_array deq = mlx_array_new();
    int wsh[2] = {(int)pw.rows, (int)(pw.cols * bits / 32)};
    int ssh[2] = {(int)pw.rows, (int)(pw.cols / group)};
    mlx_array aw = mlx_array_new_data((void*)pw.w, wsh, 2, MLX_UINT32);
    mlx_array asc = mlx_array_new_data((void*)pw.scales, ssh, 2, MLX_FLOAT16);
    mlx_array ab = mlx_array_new_data((void*)pw.biases, ssh, 2, MLX_FLOAT16);
    if (mlx_dequantize(&deq, aw, asc, ab, (mlx_optional_int){group, true}, (mlx_optional_int){bits, true},
                       "affine", mlx_array_new(), (mlx_optional_dtype){MLX_FLOAT32, true}, s) != 0) { printf("dequantize FAIL\n"); return 1; }
    { mlx_vector_array ev = mlx_vector_array_new(); mlx_vector_array_append_value(ev, deq);
      mlx_eval(ev); mlx_vector_array_free(ev); }
    const float* D = mlx_array_data_float32(deq);
    float* x = malloc((size_t)pw.cols * 4), *ref = malloc((size_t)pw.rows * 4), *got = malloc((size_t)pw.rows * 4);
    for (uint32_t c = 0; c < pw.cols; c++) x[c] = frand();
    for (uint32_t r = 0; r < pw.rows; r++) { double acc = 0; const float* row = D + (size_t)r * pw.cols;
        for (uint32_t c = 0; c < pw.cols; c++) acc += (double)row[c] * x[c]; ref[r] = (float)acc; }
    uint16_t* xh = malloc((size_t)pw.cols * 2); uint16_t* yh = malloc((size_t)pw.rows * 2);
    for (uint32_t c = 0; c < pw.cols; c++) { _Float16 h = (_Float16)x[c]; memcpy(&xh[c], &h, 2); memset(x+c, 0, 0); }
    LlmGemvQ4Ctx* g = llm_gemvq4_create();
    if (!g || llm_gemvq4_load_weights_wonly_nocopy(g, (const uint8_t*)pw.w, pw.scales, pw.biases, pw.cols, pw.rows, (uint32_t)group) != 0) { printf("load nocopy FAIL\n"); return 1; }
    if (llm_gemvq4_run(g, xh, yh) != 0) { printf("run FAIL\n"); return 1; }
    double maxe = 0, maxr = 0;
    for (uint32_t r = 0; r < pw.rows; r++) { _Float16 h; memcpy(&h, &yh[r], 2); double e = fabs((double)h - (double)ref[r]); if (e > maxe) maxe = e; if (fabs(ref[r]) > maxr) maxr = fabs(ref[r]); }
    printf("max_abs_err=%.6g max_ref=%.6g rel=%.3g\n", maxe, maxr, maxe/(maxr>0?maxr:1.0));
    { struct timespec t0,t1; timespec_get(&t0,TIME_UTC);
      const int IT = 200; for (int it = 0; it < IT; it++) llm_gemvq4_run(g, xh, yh);
      timespec_get(&t1,TIME_UTC);
      double dt = (t1.tv_sec-t0.tv_sec)+1e-9*(t1.tv_nsec-t0.tv_nsec);
      double bytes = (double)pw.rows * (double)(pw.cols/2) * (double)IT;
      printf("gemv nocopy: %.3f ms/run  %.1f GB/s (%.1f MB pesi)\n", dt*1000/IT, bytes/dt/1e9, bytes/IT/1e6); }
    printf("first3 ref=%.6f %.6f %.6f\n", ref[0], ref[1], ref[2]);
    { _Float16 h0,h1,h2; memcpy(&h0,&yh[0],2); memcpy(&h1,&yh[1],2); memcpy(&h2,&yh[2],2);
      printf("first3 gemv=%.6f %.6f %.6f\n", (double)h0, (double)h1, (double)h2); }
    return (maxe/(maxr>0?maxr:1.0) < 1e-2) ? 0 : 1;
}
