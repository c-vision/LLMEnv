/* llm_qwen4_native_attn_layer_test.c — gate model-free P7b: wiring full-attention
 * (prep Q/K, cache KV/idx, select, attention gated, out proj) vs
 * qwen4_ref_attention su 4 token consecutivi (sotto soglia: tutti i token). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_attn_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}
static double dsig(double x) { return 1.0 / (1.0 + exp(-x)); }
static void mv(const float* W, const float* x, double* y, int ni, int no) {
    for (int o = 0; o < no; o++) {
        double a = 0.0;
        for (int i = 0; i < ni; i++) a += (double)W[(size_t)o * ni + i] * x[i];
        y[o] = a;
    }
}
static void mvd(const float* W, const double* x, double* y, int ni, int no) {
    for (int o = 0; o < no; o++) {
        double a = 0.0;
        for (int i = 0; i < ni; i++) a += (double)W[(size_t)o * ni + i] * x[i];
        y[o] = a;
    }
}
static void rope(double* x, int n_rot, int pos, const float* f) {
    int h = n_rot / 2;
    for (int i = 0; i < h; i++) {
        double th = (double)pos * f[i], c = cos(th), s = sin(th);
        double x0 = x[i], x1 = x[i + h];
        x[i] = x0 * c - x1 * s; x[i + h] = x0 * s + x1 * c;
    }
}
static double dotd(const double* a, const double* b, int n) {
    double s = 0.0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}

int main(void) {
    const int E = 8, H = 2, Hkv = 1, D = 4, Hi = 1, Di = 4;
    const int qd = H * D, kvd = Hkv * D, nrot = 4, cap = 8, nsteps = 4;
    Q4nAttnW w; w.H = H; w.Hkv = Hkv; w.D = D; w.Hi = Hi; w.Di = Di;
    float *Wq = malloc((size_t)2 * qd * E * sizeof(float));
    float *Wk = malloc((size_t)kvd * E * sizeof(float));
    float *Wv = malloc((size_t)kvd * E * sizeof(float));
    float *Wqn = malloc((size_t)D * sizeof(float));
    float *Wkn = malloc((size_t)D * sizeof(float));
    float *Wo = malloc((size_t)E * qd * sizeof(float));
    float *Wi_q = malloc((size_t)Hi * Di * E * sizeof(float));
    float *Wi_k = malloc((size_t)Di * E * sizeof(float));
    float *Wi_qn = malloc((size_t)Di * sizeof(float));
    float *Wi_kn = malloc((size_t)Di * sizeof(float));
    float *freq = malloc((size_t)(nrot / 2) * sizeof(float));
    uint32_t st = 909u;
    for (int i = 0; i < 2 * qd * E; i++) Wq[i] = 0.2f * rnd(&st);
    for (int i = 0; i < kvd * E; i++) { Wk[i] = 0.2f * rnd(&st); Wv[i] = 0.2f * rnd(&st); }
    for (int i = 0; i < D; i++) { Wqn[i] = 1.0f + 0.1f * rnd(&st); Wkn[i] = 1.0f + 0.1f * rnd(&st); }
    for (int i = 0; i < E * qd; i++) Wo[i] = 0.2f * rnd(&st);
    for (int i = 0; i < Hi * Di * E; i++) Wi_q[i] = 0.2f * rnd(&st);
    for (int i = 0; i < Di * E; i++) Wi_k[i] = 0.2f * rnd(&st);
    for (int i = 0; i < Di; i++) { Wi_qn[i] = 1.0f + 0.1f * rnd(&st); Wi_kn[i] = 1.0f + 0.1f * rnd(&st); }
    for (int i = 0; i < nrot / 2; i++) freq[i] = (float)pow(1e7, -2.0 * i / nrot);
    w.attn_q = Wq; w.attn_k = Wk; w.attn_v = Wv; w.attn_q_norm = Wqn; w.attn_k_norm = Wkn;
    w.attn_out = Wo; w.idx_q = Wi_q; w.idx_k = Wi_k; w.idx_q_norm = Wi_qn; w.idx_k_norm = Wi_kn;
    float* xs = malloc((size_t)nsteps * E * sizeof(float));
    for (int i = 0; i < nsteps * E; i++) xs[i] = rnd(&st);

    double* rkc = calloc((size_t)cap * kvd, sizeof(double));
    double* rvc = calloc((size_t)cap * kvd, sizeof(double));
    double* rout = calloc((size_t)nsteps * E, sizeof(double));
    double qg[64], k[16], v[16], qi[16], ik[16], q[16], gate[16], o[16], tmp[64];
    for (int pos = 0; pos < nsteps; pos++) {
        const float* x = xs + pos * E;
        mv(Wq, x, qg, E, 2 * qd); mv(Wk, x, k, E, kvd); mv(Wv, x, v, E, kvd);
        mv(Wi_q, x, qi, E, Hi * Di); mv(Wi_k, x, ik, E, Di);
        for (int h = 0; h < H; h++) {
            double ss = 0.0;
            for (int d = 0; d < D; d++) ss += qg[h * 2 * D + d] * qg[h * 2 * D + d];
            double r = 1.0 / sqrt(ss / D + 1e-6);
            for (int d = 0; d < D; d++) { q[h * D + d] = qg[h * 2 * D + d] * r * Wqn[d]; gate[h * D + d] = qg[h * 2 * D + D + d]; }
            rope(q + h * D, nrot, pos, freq);
        }
        for (int h = 0; h < Hkv; h++) {
            double ss = 0.0;
            for (int d = 0; d < D; d++) ss += k[h * D + d] * k[h * D + d];
            double r = 1.0 / sqrt(ss / D + 1e-6);
            for (int d = 0; d < D; d++) tmp[d] = k[h * D + d] * r * Wkn[d];
            rope(tmp, nrot, pos, freq);
            for (int d = 0; d < D; d++) rkc[(size_t)pos * kvd + h * D + d] = tmp[d];
        }
        for (int d = 0; d < kvd; d++) rvc[(size_t)pos * kvd + d] = v[d];
        (void)qi; (void)ik; (void)Hi;
        const int nvis = pos + 1;
        for (int h = 0; h < H; h++) {
            int kvh = h / (H / Hkv);
            double sc[64], mx = -1e300, sum = 0.0;
            for (int i = 0; i < nvis; i++) {
                sc[i] = dotd(q + h * D, rkc + (size_t)i * kvd + kvh * D, D) / sqrt((double)D);
                if (sc[i] > mx) mx = sc[i];
            }
            for (int i = 0; i < nvis; i++) { sc[i] = exp(sc[i] - mx); sum += sc[i]; }
            for (int d = 0; d < D; d++) {
                double acc = 0.0;
                for (int i = 0; i < nvis; i++) acc += sc[i] * rvc[(size_t)i * kvd + kvh * D + d];
                o[h * D + d] = acc / sum * dsig(gate[h * D + d]);
            }
        }
        mvd(Wo, o, tmp, qd, E);
        for (int e = 0; e < E; e++) rout[pos * E + e] = tmp[e];
    }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    float* kc = calloc((size_t)cap * kvd, sizeof(float));
    float* vc = calloc((size_t)cap * kvd, sizeof(float));
    float* ic = calloc((size_t)cap * Di, sizeof(float));
    int bad = 0;
    for (int pos = 0; pos < nsteps; pos++) {
        int xsh[2] = {1, E};
        mlx_array ax = mlx_array_new_data(xs + pos * E, xsh, 2, MLX_FLOAT32);
        mlx_array ao = mlx_array_new();
        if (qwen4_native_attn_layer(&ao, ax, kc, vc, ic, cap, pos, &w, freq, nrot, 1e-6f,
                                    q4n_default_mv, NULL, s) != 0) {
            printf("FAIL layer pos %d\n", pos); return 1;
        }
        mlx_vector_array ev = mlx_vector_array_new();
        mlx_vector_array_append_value(ev, ao);
        mlx_eval(ev); mlx_vector_array_free(ev);
        const float* go = mlx_array_data_float32(ao);
        double worst = 0.0;
        for (int e = 0; e < E; e++) { double d = fabs((double)go[e] - rout[pos * E + e]); if (d > worst) worst = d; }
        printf("pos %d max|diff| = %.3e\n", pos, worst);
        if (worst > 3e-4) bad = 1;
        mlx_array_free(ao); mlx_array_free(ax);
    }
    printf("%s\n", bad ? "P7b ATTN LAYER: FALLITO" : "P7b ATTN LAYER: OK");
    return bad ? 1 : 0;
}
