/* llm_qwen4_native_moe_layer_test.c — gate model-free P7c: wiring MoE
 * (router + esperti routed + shared) vs qwen4_ref_moe. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_moe_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}
static double dsig(double x) { return 1.0 / (1.0 + exp(-x)); }
static void mvd(const float* W, const double* x, double* y, int ni, int no) {
    for (int o = 0; o < no; o++) {
        double a = 0.0;
        for (int i = 0; i < ni; i++) a += (double)W[(size_t)o * ni + i] * x[i];
        y[o] = a;
    }
}
static void mv(const float* W, const float* x, double* y, int ni, int no) {
    for (int o = 0; o < no; o++) {
        double a = 0.0;
        for (int i = 0; i < ni; i++) a += (double)W[(size_t)o * ni + i] * x[i];
        y[o] = a;
    }
}

int main(void) {
    const int E = 8, NE = 4, K = 2, F = 4;
    Q4nMoeW w; w.E = E; w.NE = NE; w.K = K; w.F = F;
    float* gi = malloc((size_t)NE * E * sizeof(float));
    float* ge = malloc((size_t)NE * F * E * sizeof(float));
    float* ue = malloc((size_t)NE * F * E * sizeof(float));
    float* de = malloc((size_t)NE * E * F * sizeof(float));
    float* sge = malloc((size_t)F * E * sizeof(float));
    float* sue = malloc((size_t)F * E * sizeof(float));
    float* sde = malloc((size_t)E * F * sizeof(float));
    float* gsh = malloc((size_t)E * sizeof(float));
    float* x = malloc((size_t)E * sizeof(float));
    uint32_t st = 31337u;
    for (int i = 0; i < NE * E; i++) gi[i] = 0.5f * rnd(&st);
    for (int i = 0; i < NE * F * E; i++) { ge[i] = 0.3f * rnd(&st); ue[i] = 0.3f * rnd(&st); }
    for (int i = 0; i < NE * E * F; i++) de[i] = 0.3f * rnd(&st);
    for (int i = 0; i < F * E; i++) { sge[i] = 0.3f * rnd(&st); sue[i] = 0.3f * rnd(&st); }
    for (int i = 0; i < E * F; i++) sde[i] = 0.3f * rnd(&st);
    for (int i = 0; i < E; i++) { gsh[i] = rnd(&st); x[i] = rnd(&st); }
    w.gate_inp = gi; w.gate_exps = ge; w.up_exps = ue; w.down_exps = de;
    w.shexp_gate = sge; w.shexp_up = sue; w.shexp_down = sde; w.gate_inp_sh = gsh;

    double logits[4], p[4], mx = -1e300, sum = 0.0;
    mv(gi, x, logits, E, NE);
    for (int e = 0; e < NE; e++) if (logits[e] > mx) mx = logits[e];
    for (int e = 0; e < NE; e++) { p[e] = exp(logits[e] - mx); sum += p[e]; }
    for (int e = 0; e < NE; e++) p[e] /= sum;
    int sel[16]; double ws = 0.0;
    for (int i = 0; i < K; i++) {
        int b = -1;
        for (int e = 0; e < NE; e++) {
            int used = 0;
            for (int j = 0; j < i; j++) used |= sel[j] == e;
            if (!used && (b < 0 || p[e] > p[b])) b = e;
        }
        sel[i] = b; ws += p[b];
    }
    double ref[8] = {0};
    for (int i = 0; i < K; i++) {
        int e = sel[i];
        double g[4], u[4], h[4], y[8];
        mv(ge + (size_t)e * F * E, x, g, E, F);
        mv(ue + (size_t)e * F * E, x, u, E, F);
        for (int f = 0; f < F; f++) h[f] = g[f] * dsig(g[f]) * u[f];
        mvd(de + (size_t)e * E * F, h, y, F, E);
        for (int d = 0; d < E; d++) ref[d] += (p[e] / ws) * y[d];
    }
    {
        double g[4], u[4], h[4], y[8], sg;
        mv(sge, x, g, E, F); mv(sue, x, u, E, F);
        for (int f = 0; f < F; f++) h[f] = g[f] * dsig(g[f]) * u[f];
        mvd(sde, h, y, F, E);
        mv(gsh, x, &sg, E, 1);
        for (int d = 0; d < E; d++) ref[d] += dsig(sg) * y[d];
    }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    int xsh[2] = {1, E};
    mlx_array ax = mlx_array_new_data(x, xsh, 2, MLX_FLOAT32);
    mlx_array ao = mlx_array_new();
    if (qwen4_native_moe_layer(&ao, ax, &w, q4n_default_mv, NULL, s) != 0) { printf("FAIL layer\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, ao);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const float* go = mlx_array_data_float32(ao);
    double worst = 0.0;
    for (int d = 0; d < E; d++) { double dd = fabs((double)go[d] - ref[d]); if (dd > worst) worst = dd; }
    printf("moe max|diff| = %.3e\n", worst);
    int bad = worst > 5e-5;
    printf("%s\n", bad ? "P7c MOE LAYER: FALLITO" : "P7c MOE LAYER: OK");
    return bad ? 1 : 0;
}
