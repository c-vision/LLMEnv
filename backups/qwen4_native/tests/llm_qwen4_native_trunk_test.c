/* llm_qwen4_native_trunk_test.c — gate model-free P7d: hc_mix + ordine dei
 * combine nel trunk. I due block sono identita', cosi' si isola HC/ordine
 * (GDN/attention/MoE sono gia' gated separatamente). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_trunk.h"

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
static int identity_block(void* ctx, mlx_array x, mlx_array* out, mlx_stream s) {
    (void)ctx; (void)s;
    *out = mlx_array_new();
    return mlx_array_set(out, x);
}

static void hc_mix_ref(const double* R, const Q4nHcW* w, int E, int hc, int rank,
                       double* mixed, double* inj) {
    double xn[64];
    for (int s = 0; s < hc; s++) {
        double ss = 0.0;
        for (int d = 0; d < E; d++) ss += R[s * E + d] * R[s * E + d];
        double r = 1.0 / sqrt(ss / E + 1e-6);
        for (int d = 0; d < E; d++) xn[s * E + d] = R[s * E + d] * r * (1.0 + (double)w->norm[s * E + d]);
    }
    double lo[16];
    mvd(w->down, xn, lo, hc * E, rank);
    for (int r = 0; r < rank; r++) lo[r] = lo[r] / hc;
    for (int r = 0; r < rank; r++) lo[r] = lo[r] * dsig(lo[r]);
    double gate[64];
    mvd(w->up, lo, gate, rank, hc * E);
    for (int d = 0; d < E; d++) {
        double acc = 0.0;
        for (int s = 0; s < hc; s++) acc += dsig(gate[s * E + d]) * xn[s * E + d];
        mixed[d] = acc / hc;
    }
    mvd(w->inject, xn, inj, hc * E, hc);
}
static void combine_ref(const double* R, const double* out, const double* inj, int E, int hc, double* Rn) {
    for (int s = 0; s < hc; s++)
        for (int d = 0; d < E; d++) Rn[s * E + d] = R[s * E + d] + 2.0 * dsig(inj[s] / hc) * out[d];
}

int main(void) {
    const int E = 8, hc = 2, rank = 3;
    Q4nHcW wa, wf;
    wa.rank = rank; wf.rank = rank;
    float *an = malloc((size_t)hc * E * sizeof(float)), *ad = malloc((size_t)rank * hc * E * sizeof(float));
    float *au = malloc((size_t)hc * E * rank * sizeof(float)), *ai = malloc((size_t)hc * hc * E * sizeof(float));
    float *fn = malloc((size_t)hc * E * sizeof(float)), *fd = malloc((size_t)rank * hc * E * sizeof(float));
    float *fu = malloc((size_t)hc * E * rank * sizeof(float)), *fi = malloc((size_t)hc * hc * E * sizeof(float));
    float* R = malloc((size_t)hc * E * sizeof(float));
    uint32_t st = 1234u;
    for (int i = 0; i < hc * E; i++) { an[i] = 1.0f + 0.1f * rnd(&st); fn[i] = 1.0f + 0.1f * rnd(&st); R[i] = rnd(&st); }
    for (int i = 0; i < rank * hc * E; i++) { ad[i] = 0.2f * rnd(&st); fd[i] = 0.2f * rnd(&st); }
    for (int i = 0; i < hc * E * rank; i++) { au[i] = 0.2f * rnd(&st); fu[i] = 0.2f * rnd(&st); }
    for (int i = 0; i < hc * hc * E; i++) { ai[i] = 0.2f * rnd(&st); fi[i] = 0.2f * rnd(&st); }
    wa.norm = an; wa.down = ad; wa.up = au; wa.inject = ai;
    wf.norm = fn; wf.down = fd; wf.up = fu; wf.inject = fi;

    double R0[16], mixed[8], inj[2], R1[16], mixed2[8], inj2[2], R2[16];
    for (int i = 0; i < hc * E; i++) R0[i] = R[i];
    hc_mix_ref(R0, &wa, E, hc, rank, mixed, inj);
    combine_ref(R0, mixed, inj, E, hc, R1);
    hc_mix_ref(R1, &wf, E, hc, rank, mixed2, inj2);
    combine_ref(R1, mixed2, inj2, E, hc, R2);

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    int rsh[2] = {1, hc * E};
    mlx_array aR = mlx_array_new_data(R, rsh, 2, MLX_FLOAT32);
    mlx_array oR = mlx_array_new();
    if (qwen4_native_trunk_layer(&oR, aR, &wa, &wf, identity_block, NULL, identity_block, NULL,
                                 E, hc, 1e-6f, q4n_default_mv, NULL, s) != 0) {
        printf("FAIL trunk\n"); return 1;
    }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, oR);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const float* go = mlx_array_data_float32(oR);
    double worst = 0.0;
    for (int i = 0; i < hc * E; i++) { double d = fabs((double)go[i] - R2[i]); if (d > worst) worst = d; }
    printf("trunk max|diff| = %.3e\n", worst);
    int bad = worst > 3e-4;
    printf("%s\n", bad ? "P7d TRUNK: FALLITO" : "P7d TRUNK: OK");
    return bad ? 1 : 0;
}
