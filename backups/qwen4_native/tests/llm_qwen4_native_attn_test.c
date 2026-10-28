/* llm_qwen4_native_attn_test.c — gate model-free P5a (prep QKV + attention gated).
 * Confronto con qwen4_ref_attention (testo: mrope p3 = pos). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_attn_kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rnd(uint32_t* st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)((*st >> 8) & 0xffffu) / 32768.0f) - 1.0f;
}

static void rope(double* x, int n_rot, int pos, const float* freq) {
    int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        double th = (double)pos * (double)freq[i], c = cos(th), s = sin(th);
        double x0 = x[i], x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

int main(void) {
    const int T = 2, H = 4, Hkv = 2, D = 8, R = 4, Nvis = 6, Nsel = 4, pos = 3;
    const double eps = 1e-6;
    const int sel[4] = {0, 2, 3, 5};
    const int qw = H * D, kw = Hkv * D;
    float *qg = malloc((size_t)T * H * 2 * D * sizeof(float));
    float *gqn = malloc((size_t)D * sizeof(float));
    float *gkn = malloc((size_t)D * sizeof(float));
    float *freq = malloc((size_t)(R / 2) * sizeof(float));
    float *K = malloc((size_t)Nvis * kw * sizeof(float));
    float *V = malloc((size_t)Nvis * kw * sizeof(float));
    uint32_t st = 4242u;
    for (int i = 0; i < T * H * 2 * D; i++) qg[i] = rnd(&st);
    for (int i = 0; i < D; i++) { gqn[i] = rnd(&st); gkn[i] = rnd(&st); }
    for (int i = 0; i < R / 2; i++) freq[i] = (float)pow(1e7, -2.0 * i / R);
    for (int i = 0; i < Nvis * kw; i++) { K[i] = rnd(&st); V[i] = rnd(&st); }

    /* riferimento CPU */
    double *rq = calloc((size_t)T * qw, sizeof(double));
    double *rg = calloc((size_t)T * qw, sizeof(double));
    double *rk = calloc((size_t)Nvis * kw, sizeof(double));
    double *rout = calloc((size_t)T * qw, sizeof(double));
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++) {
            int qb = (t * H + h) * 2 * D, ob = (t * H + h) * D;
            double ss = 0.0;
            for (int d = 0; d < D; d++) ss += (double)qg[qb + d] * qg[qb + d];
            double r = 1.0 / sqrt(ss / D + eps), tmp[256];
            for (int d = 0; d < D; d++) tmp[d] = (double)qg[qb + d] * r * gqn[d];
            rope(tmp, R, pos, freq);
            for (int d = 0; d < D; d++) { rq[ob + d] = tmp[d]; rg[ob + d] = qg[qb + D + d]; }
        }
    for (int t = 0; t < Nvis; t++)
        for (int h = 0; h < Hkv; h++) {
            int b = (t * Hkv + h) * D;
            double ss = 0.0;
            for (int d = 0; d < D; d++) ss += (double)K[b + d] * K[b + d];
            double r = 1.0 / sqrt(ss / D + eps), tmp[256];
            for (int d = 0; d < D; d++) tmp[d] = (double)K[b + d] * r * gkn[d];
            rope(tmp, R, pos, freq);
            for (int d = 0; d < D; d++) rk[b + d] = tmp[d];
        }
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++) {
            int kvh = h / (H / Hkv), ob = (t * H + h) * D;
            double sc[2048], mx = -1e300, sum = 0.0;
            for (int j = 0; j < Nsel; j++) {
                int idx = sel[j], kb = (idx * Hkv + kvh) * D;
                double dot = 0.0;
                for (int d = 0; d < D; d++) dot += rq[ob + d] * rk[kb + d];
                sc[j] = dot / sqrt((double)D);
                if (sc[j] > mx) mx = sc[j];
            }
            for (int j = 0; j < Nsel; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int d = 0; d < D; d++) {
                double acc = 0.0;
                for (int j = 0; j < Nsel; j++) {
                    int idx = sel[j], kb = (idx * Hkv + kvh) * D;
                    acc += sc[j] * (double)V[kb + d];
                }
                double g = rg[ob + d];
                double sg = 1.0 / (1.0 + exp(-g));
                rout[ob + d] = acc / sum * sg;
            }
        }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    int qgsh[2] = {T, H * 2 * D}, dsh[1] = {D}, fsh[1] = {R / 2};
    int ksh[2] = {Nvis, kw}, ssh[1] = {Nsel};
    mlx_array aQ = mlx_array_new_data(qg, qgsh, 2, MLX_FLOAT32);
    mlx_array aGQ = mlx_array_new_data(gqn, dsh, 1, MLX_FLOAT32);
    mlx_array aGK = mlx_array_new_data(gkn, dsh, 1, MLX_FLOAT32);
    mlx_array aF = mlx_array_new_data(freq, fsh, 1, MLX_FLOAT32);
    mlx_array aK = mlx_array_new_data(K, ksh, 2, MLX_FLOAT32);
    mlx_array aV = mlx_array_new_data(V, ksh, 2, MLX_FLOAT32);
    mlx_array aSel = mlx_array_new_data((void*)sel, ssh, 1, MLX_INT32);
    mlx_array aq = mlx_array_new(), ag = mlx_array_new(), ak = mlx_array_new(), ao = mlx_array_new();
    if (llm_q4n_prep_q(&aq, &ag, aQ, aGQ, aF, R, pos, 1.0f, (float)eps, s) != 0) { printf("FAIL prep_q\n"); return 1; }
    if (llm_q4n_prep_k(&ak, aK, aGK, aF, R, pos, 1.0f, (float)eps, s) != 0) { printf("FAIL prep_k\n"); return 1; }
    if (llm_q4n_attn(&ao, aq, ag, ak, aV, aSel, H, Nvis, s) != 0) { printf("FAIL attn\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aq);
    mlx_vector_array_append_value(ev, ak);
    mlx_vector_array_append_value(ev, ao);
    mlx_eval(ev);
    mlx_vector_array_free(ev);

    int bad = 0;
    const float* gq1 = mlx_array_data_float32(aq);
    const float* gk1 = mlx_array_data_float32(ak);
    const float* go1 = mlx_array_data_float32(ao);
    double w1 = 0, w2 = 0, w3 = 0;
    for (int i = 0; i < T * qw; i++) { double d = fabs((double)gq1[i] - rq[i]); if (d > w1) w1 = d; }
    for (int i = 0; i < Nvis * kw; i++) { double d = fabs((double)gk1[i] - rk[i]); if (d > w2) w2 = d; }
    for (int i = 0; i < T * qw; i++) { double d = fabs((double)go1[i] - rout[i]); if (d > w3) w3 = d; }
    if (w1 > 2e-5 || w2 > 2e-5 || w3 > 2e-4) bad = 1;
    printf("prep_q  max|diff| = %.3e\nprep_k  max|diff| = %.3e\nattn    max|diff| = %.3e\n", w1, w2, w3);
    printf("%s\n", bad ? "P5a ATTN: FALLITO" : "P5a ATTN: OK");
    return bad ? 1 : 0;
}
