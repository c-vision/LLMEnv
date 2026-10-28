/* llm_qwen4_native_layer_test.c — gate model-free P7a: wiring di un layer
 * linear/GDN contro il riferimento ds4 `qwen4_ref_linear` (2 token consecutivi,
 * stato conv + ricorrente persistiti). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}
static double dsig(double x) { return 1.0 / (1.0 + exp(-x)); }
static double dsp(double x) { return x > 20.0 ? x : (x < -20.0 ? exp(x) : log1p(exp(x))); }
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

int main(void) {
    const int E = 8, Hk = 2, Hv = 3, D = 4, K = 4;
    const int kd = Hk * D, vd = Hv * D, cd = 2 * kd + vd;
    const int nsteps = 2;
    Q4nGdnW w;
    w.conv_dim = cd; w.v_dim = vd; w.Hk = Hk; w.Hv = Hv; w.D = D;
    float* Wqkv = malloc((size_t)cd * E * sizeof(float));
    float* Wgt = malloc((size_t)vd * E * sizeof(float));
    float* Wbe = malloc((size_t)Hv * E * sizeof(float));
    float* Wal = malloc((size_t)Hv * E * sizeof(float));
    float* Wcv = malloc((size_t)cd * K * sizeof(float));
    float* Wa = malloc((size_t)Hv * sizeof(float));
    float* Wdt = malloc((size_t)Hv * sizeof(float));
    float* Wn = malloc((size_t)D * sizeof(float));
    float* Wo = malloc((size_t)E * vd * sizeof(float));
    uint32_t st = 555u;
    for (int i = 0; i < cd * E; i++) Wqkv[i] = 0.2f * rnd(&st);
    for (int i = 0; i < vd * E; i++) Wgt[i] = 0.2f * rnd(&st);
    for (int i = 0; i < Hv * E; i++) { Wbe[i] = 0.2f * rnd(&st); Wal[i] = 0.2f * rnd(&st); }
    for (int i = 0; i < cd * K; i++) Wcv[i] = 0.3f * rnd(&st);
    for (int i = 0; i < Hv; i++) { Wa[i] = rnd(&st); Wdt[i] = 0.2f * rnd(&st); }
    for (int i = 0; i < D; i++) Wn[i] = 1.0f + 0.1f * rnd(&st);
    for (int i = 0; i < E * vd; i++) Wo[i] = 0.2f * rnd(&st);
    w.lin_qkv = Wqkv; w.lin_gate = Wgt; w.lin_beta = Wbe; w.lin_alpha = Wal;
    w.lin_conv = Wcv; w.lin_a = Wa; w.lin_dt = Wdt; w.lin_norm = Wn; w.lin_out = Wo;
    float* xs = malloc((size_t)nsteps * E * sizeof(float));
    for (int i = 0; i < nsteps * E; i++) xs[i] = rnd(&st);

    /* riferimento CPU (qwen4_ref_linear) */
    double hist[3 * 28], S[3 * 4 * 4];
    memset(hist, 0, sizeof(hist)); memset(S, 0, sizeof(S));
    double* rout = calloc((size_t)nsteps * E, sizeof(double));
    double qkv[64], z[32], a[8], b[8], conv[64], o[32], tmp[64];
    for (int step = 0; step < nsteps; step++) {
        const float* x = xs + step * E;
        mv(Wqkv, x, qkv, E, cd); mv(Wgt, x, z, E, vd);
        mv(Wbe, x, b, E, Hv); mv(Wal, x, a, E, Hv);
        for (int c = 0; c < cd; c++) {
            double acc = (double)Wcv[(size_t)c * K + (K - 1)] * qkv[c];
            for (int k = 0; k + 1 < K; k++) acc += (double)Wcv[(size_t)c * K + k] * hist[(size_t)k * cd + c];
            conv[c] = acc * dsig(acc);
        }
        memmove(hist, hist + cd, (size_t)(K - 2) * cd * sizeof(double));
        for (int c = 0; c < cd; c++) hist[(size_t)(K - 2) * cd + c] = qkv[c];
        double* q = conv, *kk = conv + kd, *vv = conv + 2 * kd;
        for (int h = 0; h < Hk; h++) {
            double sq = 0.0, sk = 0.0;
            for (int d = 0; d < D; d++) { sq += q[h * D + d] * q[h * D + d]; sk += kk[h * D + d] * kk[h * D + d]; }
            double rq = 1.0 / sqrt(sq + 1e-6) / sqrt((double)D), rk = 1.0 / sqrt(sk + 1e-6);
            for (int d = 0; d < D; d++) { q[h * D + d] *= rq; kk[h * D + d] *= rk; }
        }
        for (int j = 0; j < Hv; j++) {
            int kh = j % Hk;
            double g = exp((double)Wa[j] * dsp(a[j] + Wdt[j])), beta = dsig(b[j]);
            double* Sj = S + (size_t)j * D * D;
            const double* qj = q + kh * D, *kj = kk + kh * D, *vj = vv + j * D;
            double* oj = o + j * D;
            for (int dv = 0; dv < D; dv++) {
                double kv = 0.0;
                for (int dk = 0; dk < D; dk++) { Sj[dk * D + dv] *= g; kv += Sj[dk * D + dv] * kj[dk]; }
                double delta = (vj[dv] - kv) * beta, acc = 0.0;
                for (int dk = 0; dk < D; dk++) { Sj[dk * D + dv] += kj[dk] * delta; acc += Sj[dk * D + dv] * qj[dk]; }
                oj[dv] = acc;
            }
            double ss = 0.0;
            for (int d = 0; d < D; d++) ss += oj[d] * oj[d];
            double inv = 1.0 / sqrt(ss / D + 1e-6);
            for (int d = 0; d < D; d++) oj[d] = oj[d] * inv * Wn[d] * dsig(z[j * D + d]);
        }
        mvd(Wo, o, tmp, vd, E);
        for (int e = 0; e < E; e++) rout[step * E + e] = tmp[e];
    }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    int hsh[2] = {K - 1, cd}, s3[3] = {Hv, D, D};
    float hist0[3 * 28]; memset(hist0, 0, sizeof(hist0));
    float state0[3 * 16]; memset(state0, 0, sizeof(state0));
    mlx_array ahist = mlx_array_new_data(hist0, hsh, 2, MLX_FLOAT32);
    mlx_array astate = mlx_array_new_data(state0, s3, 3, MLX_FLOAT32);
    int bad = 0;
    for (int step = 0; step < nsteps; step++) {
        int xsh[2] = {1, E};
        mlx_array ax = mlx_array_new_data(xs + step * E, xsh, 2, MLX_FLOAT32);
        mlx_array ao = mlx_array_new();
        if (qwen4_native_gdn_layer(&ao, ax, &ahist, &astate, &w, q4n_default_mv, NULL, s) != 0) {
            printf("FAIL layer step %d\n", step); return 1;
        }
        mlx_vector_array ev = mlx_vector_array_new();
        mlx_vector_array_append_value(ev, ao);
        mlx_eval(ev);
        mlx_vector_array_free(ev);
        const float* go = mlx_array_data_float32(ao);
        double worst = 0.0;
        for (int e = 0; e < E; e++) { double d = fabs((double)go[e] - rout[step * E + e]); if (d > worst) worst = d; }
        printf("step %d max|diff| = %.3e\n", step, worst);
        if (worst > 2e-4) bad = 1;
        mlx_array_free(ao); mlx_array_free(ax);
    }
    printf("%s\n", bad ? "P7a GDN LAYER: FALLITO" : "P7a GDN LAYER: OK");
    return bad ? 1 : 0;
}
