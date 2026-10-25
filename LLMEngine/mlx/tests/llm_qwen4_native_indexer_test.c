/* llm_qwen4_native_indexer_test.c — gate model-free P5b (indexer QSA).
 * Confronto con qwen4_ref_select (blocchi ratio 4, top-k, coda). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_indexer_kernel.h"

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
        x[i] = x0 * c - x1 * s; x[i + half] = x0 * s + x1 * c;
    }
}

static int run_case(int T, int Hi, int Di, int R, int nvis, int ratio, int kblocks, int pos, uint32_t seed) {
    const double eps = 1e-6;
    const int qw = Hi * Di;
    float* qi = malloc((size_t)T * qw * sizeof(float));
    float* gq = malloc((size_t)Di * sizeof(float));
    float* gk = malloc((size_t)Di * sizeof(float));
    float* freq = malloc((size_t)(R / 2) * sizeof(float));
    float* ik = malloc((size_t)nvis * Di * sizeof(float));
    uint32_t st = seed;
    for (int i = 0; i < T * qw; i++) qi[i] = rnd(&st);
    for (int i = 0; i < Di; i++) { gq[i] = rnd(&st); gk[i] = rnd(&st); }
    for (int i = 0; i < R / 2; i++) freq[i] = (float)pow(1e7, -2.0 * i / R);
    for (int i = 0; i < nvis * Di; i++) ik[i] = rnd(&st);

    /* riferimento */
    int nb = nvis / ratio;
    double* rq = calloc((size_t)T * qw, sizeof(double));
    for (int t = 0; t < T; t++)
        for (int h = 0; h < Hi; h++) {
            int b = (t * Hi + h) * Di;
            double ss = 0.0, tmp[256];
            for (int d = 0; d < Di; d++) ss += (double)qi[b + d] * qi[b + d];
            double r = 1.0 / sqrt(ss / Di + eps);
            for (int d = 0; d < Di; d++) tmp[d] = (double)qi[b + d] * r * gq[d];
            rope(tmp, R, pos, freq);
            for (int d = 0; d < Di; d++) rq[b + d] = tmp[d];
        }
    double* sc = calloc((size_t)T * nb, sizeof(double));
    double key[256];
    for (int t = 0; t < T; t++)
        for (int blk = 0; blk < nb; blk++) {
            double pooled[256], ss = 0.0;
            for (int d = 0; d < Di; d++) {
                double acc = 0.0;
                for (int j = 0; j < ratio; j++) acc += ik[(blk * ratio + j) * Di + d];
                pooled[d] = acc / ratio;
            }
            for (int d = 0; d < Di; d++) ss += pooled[d] * pooled[d];
            double r = 1.0 / sqrt(ss / Di + eps);
            for (int d = 0; d < Di; d++) key[d] = pooled[d] * r * gk[d];
            rope(key, R, blk * ratio, freq);
            double sum = 0.0;
            for (int h = 0; h < Hi; h++) {
                double dot = 0.0;
                for (int d = 0; d < Di; d++) dot += rq[(t * Hi + h) * Di + d] * key[d];
                if (dot > 0.0) sum += dot;
            }
            sc[t * nb + blk] = sum;
        }
    int* rsel = malloc((size_t)T * nvis * sizeof(int));
    int* rn = calloc(T, sizeof(int));
    for (int t = 0; t < T; t++) {
        int n = 0;
        if (nb <= kblocks) {
            for (int i = 0; i < nvis; i++) rsel[t * nvis + n++] = i;
        } else {
            char taken[2048]; memset(taken, 0, (size_t)nb);
            for (int i = 0; i < kblocks; i++) {
                int best = -1; double bv = -1e300;
                for (int b = 0; b < nb; b++) {
                    if (taken[b]) continue;
                    if (best < 0 || sc[t * nb + b] > bv) { bv = sc[t * nb + b]; best = b; }
                }
                taken[best] = 1;
            }
            for (int b = 0; b < nb; b++)
                if (taken[b]) for (int j = 0; j < ratio; j++) rsel[t * nvis + n++] = b * ratio + j;
            for (int i = nb * ratio; i < nvis; i++) rsel[t * nvis + n++] = i;
        }
        rn[t] = n;
    }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    int qsh[2] = {T, qw}, dsh[1] = {Di}, fsh[1] = {R / 2}, ksh[2] = {nvis, Di};
    mlx_array aQ = mlx_array_new_data(qi, qsh, 2, MLX_FLOAT32);
    mlx_array aGQ = mlx_array_new_data(gq, dsh, 1, MLX_FLOAT32);
    mlx_array aGK = mlx_array_new_data(gk, dsh, 1, MLX_FLOAT32);
    mlx_array aF = mlx_array_new_data(freq, fsh, 1, MLX_FLOAT32);
    mlx_array aK = mlx_array_new_data(ik, ksh, 2, MLX_FLOAT32);
    mlx_array aQp = mlx_array_new(), aSc = mlx_array_new(), aSel = mlx_array_new(), aN = mlx_array_new();
    if (llm_q4n_idx_q(&aQp, aQ, aGQ, aF, R, pos, 1.0f, (float)eps, s) != 0) { printf("FAIL idx_q\n"); return 1; }
    if (llm_q4n_idx_scr(&aSc, aQp, aK, aGK, aF, ratio, R, 1.0f, (float)eps, s) != 0) { printf("FAIL idx_scr\n"); return 1; }
    if (llm_q4n_idx_sel(&aSel, &aN, aSc, nvis, ratio, kblocks, s) != 0) { printf("FAIL idx_sel\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aQp);
    mlx_vector_array_append_value(ev, aSc);
    mlx_vector_array_append_value(ev, aSel);
    mlx_vector_array_append_value(ev, aN);
    mlx_eval(ev);
    mlx_vector_array_free(ev);

    int bad = 0;
    const float* gp = mlx_array_data_float32(aQp);
    const float* gsc = mlx_array_data_float32(aSc);
    const int* gsel = mlx_array_data_int32(aSel);
    const int* gn = mlx_array_data_int32(aN);
    double w1 = 0, w2 = 0;
    for (int i = 0; i < T * qw; i++) { double d = fabs((double)gp[i] - rq[i]); if (d > w1) w1 = d; }
    for (int i = 0; i < T * nb; i++) { double d = fabs((double)gsc[i] - sc[i]); if (d > w2) w2 = d; }
    if (w1 > 2e-5 || w2 > 2e-4) bad = 1;
    for (int t = 0; t < T; t++) {
        if (gn[t] != rn[t]) { printf("nsel[%d] %d != %d\n", t, gn[t], rn[t]); bad = 1; continue; }
        for (int i = 0; i < gn[t]; i++)
            if (gsel[t * nvis + i] != rsel[t * nvis + i]) { printf("sel[%d][%d] %d != %d\n", t, i, gsel[t * nvis + i], rsel[t * nvis + i]); bad = 1; break; }
    }
    printf("T=%d Hi=%d Di=%d nvis=%d nb=%d | qi %.2e score %.2e sel %s\n", T, Hi, Di, nvis, nb, w1, w2, bad ? "FAIL" : "ok");
    return bad;
}

int main(void) {
    int bad = 0;
    bad |= run_case(2, 2, 8, 4, 20, 4, 2, 5, 99u);
    bad |= run_case(1, 2, 8, 4, 8, 4, 2, 3, 123u);
    bad |= run_case(1, 2, 8, 4, 23, 4, 2, 7, 321u);
    printf("%s\n", bad ? "P5b INDEXER: FALLITO" : "P5b INDEXER: OK");
    return bad ? 1 : 0;
}
