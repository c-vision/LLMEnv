/* llm_qwen4_native_router_test.c — gate model-free P4 (router top-k + shared gate).
 * Confronto con qwen4_ref_moe (softmax 512, top-10, pesi rinormalizzati). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_router_kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}

int main(void) {
    const int T = 2, NE = 512, K = 10, X = 8;
    float* logits = malloc((size_t)T * NE * sizeof(float));
    float* x = malloc((size_t)T * X * sizeof(float));
    float* wg = malloc((size_t)X * sizeof(float));
    uint32_t st = 777u;
    for (int i = 0; i < T * NE; i++) logits[i] = 4.0f * rnd(&st);
    for (int i = 0; i < T * X; i++) x[i] = rnd(&st);
    for (int i = 0; i < X; i++) wg[i] = rnd(&st);

    int* rsel = calloc((size_t)T * K, sizeof(int));
    double* rw = calloc((size_t)T * K, sizeof(double));
    double* rg = calloc((size_t)T, sizeof(double));
    for (int t = 0; t < T; t++) {
        double p[512], mx = -1e300, sum = 0.0;
        for (int e = 0; e < NE; e++) if (logits[t * NE + e] > mx) mx = logits[t * NE + e];
        for (int e = 0; e < NE; e++) { p[e] = exp((double)logits[t * NE + e] - mx); sum += p[e]; }
        for (int e = 0; e < NE; e++) p[e] /= sum;
        int sel[16]; double wsum = 0.0;
        for (int r = 0; r < K; r++) {
            int bi = -1; double bp = -1.0;
            for (int e = 0; e < NE; e++) {
                int used = 0;
                for (int j = 0; j < r; j++) used |= sel[j] == e;
                if (!used && p[e] > bp) { bp = p[e]; bi = e; }
            }
            sel[r] = bi; wsum += bp;
        }
        double sg = 0.0;
        for (int i = 0; i < X; i++) sg += (double)wg[i] * x[t * X + i];
        for (int r = 0; r < K; r++) { rsel[t * K + r] = sel[r]; rw[t * K + r] = p[sel[r]] / wsum; }
        rg[t] = sg;
    }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    if (s.ctx == NULL) { printf("FAIL stream\n"); return 1; }
    int lsh[2] = {T, NE}, xsh[2] = {T, X}, gsh[1] = {X};
    mlx_array aL = mlx_array_new_data(logits, lsh, 2, MLX_FLOAT32);
    mlx_array aX = mlx_array_new_data(x, xsh, 2, MLX_FLOAT32);
    mlx_array aW = mlx_array_new_data(wg, gsh, 1, MLX_FLOAT32);
    mlx_array aSel = mlx_array_new(), aWt = mlx_array_new(), aSG = mlx_array_new();
    if (llm_q4n_router_topk(&aSel, &aWt, &aSG, aL, aX, aW, K, s) != 0) { printf("FAIL apply\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aSel);
    mlx_vector_array_append_value(ev, aWt);
    mlx_vector_array_append_value(ev, aSG);
    mlx_eval(ev);
    mlx_vector_array_free(ev);

    const int* sel = mlx_array_data_int32(aSel);
    const float* wt = mlx_array_data_float32(aWt);
    const float* sg = mlx_array_data_float32(aSG);
    int bad = 0;
    for (int i = 0; i < T * K; i++) if (sel[i] != rsel[i]) { printf("sel[%d]=%d ref=%d\n", i, sel[i], rsel[i]); bad = 1; }
    double worst = 0.0;
    for (int i = 0; i < T * K; i++) { double d = fabs((double)wt[i] - rw[i]); if (d > worst) worst = d; if (d > 1e-6) bad = 1; }
    printf("router_sel  %s\n", bad ? "FAIL" : "ok");
    printf("router_wt   max|diff| = %.3e %s\n", worst, bad ? "FAIL" : "ok");
    double gw = 0.0;
    for (int t = 0; t < T; t++) { double d = fabs((double)sg[t] - rg[t]); if (d > gw) gw = d; if (d > 1e-4) bad = 1; }
    printf("shared_gate max|diff| = %.3e %s\n", gw, bad ? "FAIL" : "ok");
    printf("%s\n", bad ? "P4 ROUTER: FALLITO" : "P4 ROUTER: OK");
    return bad ? 1 : 0;
}
