/* llm_qwen4_native_ple_test.c — gate model-free P3 (PLE nativo) vs riferimento CPU.
 * Verifica qwen4_ple_gate + qwen4_ple_conv (formule ds4 qwen4_ref_ple) su dati random.
 * Nessun modello: solo kernel + parita'. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_ple_kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}

static double dsig(double x) {
    return x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

static int check(const char* name, const float* got, const double* ref, int n, double tol) {
    double worst = 0.0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - ref[i]);
        if (d > worst) worst = d;
        if (d > tol) bad = 1;
    }
    printf("%-12s max|diff| = %.3e %s\n", name, worst, bad ? "FAIL" : "ok");
    return bad;
}

int main(void) {
    const int T = 3, E = 8, hc = 4, C = E * hc;
    const int K = 4, dil = 3, Hh = (K - 1) * dil;
    const double eps = 1e-6;
    const size_t nC = (size_t)T * C, nE = (size_t)T * E;

    float* R = malloc(nC * sizeof(float));
    float* key = malloc(nC * sizeof(float));
    float* value = malloc(nE * sizeof(float));
    float* g_key = malloc((size_t)C * sizeof(float));
    float* g_query = malloc((size_t)C * sizeof(float));
    float* g_conv = malloc((size_t)C * sizeof(float));
    float* hist = malloc((size_t)Hh * C * sizeof(float));
    float* taps = malloc((size_t)C * K * sizeof(float));
    uint32_t st = 12345u;
    for (size_t i = 0; i < nC; i++) { R[i] = rnd(&st); key[i] = rnd(&st); }
    for (size_t i = 0; i < nE; i++) value[i] = rnd(&st);
    for (size_t i = 0; i < (size_t)C; i++) { g_key[i] = rnd(&st); g_query[i] = rnd(&st); g_conv[i] = rnd(&st); }
    for (size_t i = 0; i < (size_t)Hh * C; i++) hist[i] = rnd(&st);
    for (size_t i = 0; i < (size_t)C * K; i++) taps[i] = rnd(&st);

    /* riferimento CPU (double) */
    double* rg = calloc(nC, sizeof(double));
    double* rn = calloc(nC, sizeof(double));
    for (int tok = 0; tok < T; tok++) {
        for (int s = 0; s < hc; s++) {
            size_t base = (size_t)tok * hc * E + (size_t)s * E, vb = (size_t)tok * E;
            double sk = 0.0, sr = 0.0;
            for (int d = 0; d < E; d++) {
                sk += (double)key[base + d] * key[base + d];
                sr += (double)R[base + d] * R[base + d];
            }
            double ik = 1.0 / sqrt(sk / E + eps), ir = 1.0 / sqrt(sr / E + eps), dot = 0.0;
            for (int d = 0; d < E; d++)
                dot += ((double)key[base + d] * ik * g_key[s * E + d]) *
                       ((double)R[base + d] * ir * g_query[s * E + d]);
            double gsc = dot / sqrt((double)E);
            double mag = sqrt(fmax(fabs(gsc), 1e-6));
            double x = gsc > 0 ? mag : (gsc < 0 ? -mag : 0.0);
            double gate = dsig(x), sv = 0.0;
            for (int d = 0; d < E; d++) {
                double v = gate * (double)value[vb + d];
                rg[base + d] = v;
                sv += v * v;
            }
            double ig = 1.0 / sqrt(sv / E + eps);
            for (int d = 0; d < E; d++)
                rn[base + d] = rg[base + d] * ig * (double)g_conv[s * E + d];
        }
    }
    double* rR = calloc(nC, sizeof(double));
    double* rH = calloc((size_t)Hh * C, sizeof(double));
    for (int c = 0; c < C; c++) {
        double win[9];
        for (int t = 0; t < Hh; t++) win[t] = hist[(size_t)t * C + c];
        for (int tok = 0; tok < T; tok++) {
            double cur = rn[(size_t)tok * C + c], acc = 0.0;
            double tp[4];
            for (int k = 0; k < K; k++) tp[k] = taps[(size_t)c * K + k];
            acc = tp[K - 1] * cur;
            for (int k = 0; k + 1 < K; k++) acc += tp[k] * win[Hh - (K - 1 - k) * dil];
            for (int t = 0; t + 1 < Hh; t++) win[t] = win[t + 1];
            win[Hh - 1] = cur;
            double sg = dsig(acc);
            rR[(size_t)tok * C + c] = (double)R[(size_t)tok * C + c] + rg[(size_t)tok * C + c] + acc * sg;
        }
        for (int t = 0; t < Hh; t++) rH[(size_t)t * C + c] = win[t];
    }

    /* GPU */
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL) { printf("FAIL device\n"); return 1; }
    if (mlx_set_default_device(dev) != 0) { printf("FAIL set_default_device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    if (s.ctx == NULL) { printf("FAIL stream\n"); return 1; }
    int shR[2] = {T, C}, shE[2] = {T, E}, sh1[1] = {C}, shH[2] = {Hh, C}, shW[2] = {C, K};
    mlx_array aR = mlx_array_new_data(R, shR, 2, MLX_FLOAT32);
    mlx_array aKey = mlx_array_new_data(key, shR, 2, MLX_FLOAT32);
    mlx_array aVal = mlx_array_new_data(value, shE, 2, MLX_FLOAT32);
    mlx_array aGK = mlx_array_new_data(g_key, sh1, 1, MLX_FLOAT32);
    mlx_array aGQ = mlx_array_new_data(g_query, sh1, 1, MLX_FLOAT32);
    mlx_array aGC = mlx_array_new_data(g_conv, sh1, 1, MLX_FLOAT32);
    mlx_array aHist = mlx_array_new_data(hist, shH, 2, MLX_FLOAT32);
    mlx_array aW = mlx_array_new_data(taps, shW, 2, MLX_FLOAT32);
    mlx_array aGated = mlx_array_new(), aNormed = mlx_array_new();
    int rc = llm_q4n_ple_gate(&aGated, &aNormed, aR, aKey, aVal, aGK, aGQ, aGC, (float)eps, s);
    if (rc != 0) { printf("gate: apply FAIL\n"); return 1; }
    mlx_array aRo = mlx_array_new(), aHo = mlx_array_new();
    rc = llm_q4n_ple_conv(&aRo, &aHo, aR, aGated, aNormed, aHist, aW, K, dil, s);
    if (rc != 0) { printf("conv: apply FAIL\n"); return 1; }

    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aGated);
    mlx_vector_array_append_value(ev, aNormed);
    mlx_vector_array_append_value(ev, aRo);
    mlx_vector_array_append_value(ev, aHo);
    mlx_eval(ev);
    mlx_vector_array_free(ev);
    int bad = 0;
    bad |= check("ple_gate", mlx_array_data_float32(aGated), rg, (int)nC, 3e-4);
    bad |= check("ple_norm", mlx_array_data_float32(aNormed), rn, (int)nC, 3e-4);
    bad |= check("ple_convR", mlx_array_data_float32(aRo), rR, (int)nC, 3e-4);
    bad |= check("ple_hist", mlx_array_data_float32(aHo), rH, Hh * C, 3e-4);
    printf("%s\n", bad ? "P3 PLE: FALLITO" : "P3 PLE: OK");
    return bad ? 1 : 0;
}
