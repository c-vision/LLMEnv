/* llm_qwen4_native_hc_test.c — gate model-free P1 (HC norm + combine + gate mix).
 * Convenzione: HF/MLX usa (1+gamma); il packer ds4 (qwen4_pack.py
 * "zero_centered_rms_norm": "add-one-bf16") scrive gamma+1 nel GGUF e ds4.c usa
 * gamma secco -> stessa matematica, il loader sceglie in base all'artefatto. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_hc_kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}
static double dsig(double x) { return 1.0 / (1.0 + exp(-x)); }

int main(void) {
    const int T = 2, E = 4, hc = 4, R = 3, CE = hc * E;
    float* Rb = malloc((size_t)T * CE * sizeof(float));
    float* out = malloc((size_t)T * E * sizeof(float));
    float* inj = malloc((size_t)T * hc * sizeof(float));
    float* xn = malloc((size_t)T * CE * sizeof(float));
    float* lo = malloc((size_t)T * R * sizeof(float));
    float* wup = malloc((size_t)CE * R * sizeof(float));
    uint32_t st = 2026u;
    for (int i = 0; i < T * CE; i++) { Rb[i] = rnd(&st); xn[i] = rnd(&st); }
    for (int i = 0; i < T * E; i++) out[i] = rnd(&st);
    for (int i = 0; i < T * hc; i++) inj[i] = rnd(&st);
    for (int i = 0; i < T * R; i++) lo[i] = rnd(&st);
    for (int i = 0; i < CE * R; i++) wup[i] = rnd(&st);

    double* rc = calloc((size_t)T * CE, sizeof(double));
    for (int t = 0; t < T; t++)
        for (int s = 0; s < hc; s++) {
            double ig = 2.0 * dsig((double)inj[t * hc + s] / hc);
            for (int d = 0; d < E; d++) {
                int k = t * CE + s * E + d;
                rc[k] = (double)Rb[k] + ig * out[t * E + d];
            }
        }
    float* gamma = malloc((size_t)CE * sizeof(float));
    for (int i = 0; i < CE; i++) gamma[i] = rnd(&st);
    double* rn = calloc((size_t)T * CE, sizeof(double));
    for (int t = 0; t < T; t++)
        for (int s = 0; s < hc; s++) {
            double ss = 0.0;
            for (int d = 0; d < E; d++) { double v = xn[t * CE + s * E + d]; ss += v * v; }
            double inv = 1.0 / sqrt(ss / E + 1e-6);
            for (int d = 0; d < E; d++) {
                int k = t * CE + s * E + d;
                rn[k] = (double)xn[k] * inv * (1.0 + (double)gamma[s * E + d]);
            }
        }
    double* rm = calloc((size_t)T * E, sizeof(double));
    for (int t = 0; t < T; t++)
        for (int d = 0; d < E; d++) {
            double acc = 0.0;
            for (int s = 0; s < hc; s++) {
                double g = 0.0;
                for (int r = 0; r < R; r++) {
                    double x = (double)lo[t * R + r] / hc;
                    g += (double)wup[(s * E + d) * R + r] * (x * dsig(x));
                }
                acc += dsig(g) * (double)xn[t * CE + s * E + d];
            }
            rm[t * E + d] = acc / hc;
        }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    int rsh[2] = {T, CE}, osh[2] = {T, E}, ish[2] = {T, hc}, lsh[2] = {T, R}, wsh[2] = {CE, R};
    int gsh1[1] = {CE};
    mlx_array aR = mlx_array_new_data(Rb, rsh, 2, MLX_FLOAT32);
    mlx_array aO = mlx_array_new_data(out, osh, 2, MLX_FLOAT32);
    mlx_array aI = mlx_array_new_data(inj, ish, 2, MLX_FLOAT32);
    mlx_array aXn = mlx_array_new_data(xn, rsh, 2, MLX_FLOAT32);
    mlx_array aLo = mlx_array_new_data(lo, lsh, 2, MLX_FLOAT32);
    mlx_array aW = mlx_array_new_data(wup, wsh, 2, MLX_FLOAT32);
    mlx_array aG = mlx_array_new_data(gamma, gsh1, 1, MLX_FLOAT32);
    mlx_array aC = mlx_array_new(), aM = mlx_array_new(), aN = mlx_array_new();
    if (llm_q4n_hc_norm(&aN, aXn, aG, hc, E, 1e-6f, s) != 0) { printf("FAIL norm\n"); return 1; }
    if (llm_q4n_hc_combine(&aC, aR, aO, aI, s) != 0) { printf("FAIL combine\n"); return 1; }
    if (llm_q4n_hc_gate_mix(&aM, aXn, aLo, aW, hc, E, s) != 0) { printf("FAIL gate_mix\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aN);
    mlx_vector_array_append_value(ev, aC);
    mlx_vector_array_append_value(ev, aM);
    mlx_eval(ev);
    mlx_vector_array_free(ev);

    const float* gn = mlx_array_data_float32(aN);
    const float* gc = mlx_array_data_float32(aC);
    const float* gm = mlx_array_data_float32(aM);
    double w1 = 0, w2 = 0, w0 = 0;
    for (int i = 0; i < T * CE; i++) { double d = fabs((double)gn[i] - rn[i]); if (d > w0) w0 = d; }
    for (int i = 0; i < T * CE; i++) { double d = fabs((double)gc[i] - rc[i]); if (d > w1) w1 = d; }
    for (int i = 0; i < T * E; i++) { double d = fabs((double)gm[i] - rm[i]); if (d > w2) w2 = d; }
    int bad = (w0 > 2e-5 || w1 > 2e-5 || w2 > 2e-5);
    printf("hc_norm    max|diff| = %.3e\nhc_combine max|diff| = %.3e\nhc_mix     max|diff| = %.3e\n", w0, w1, w2);
    printf("%s\n", bad ? "P1 HC: FALLITO" : "P1 HC: OK");
    return bad ? 1 : 0;
}
