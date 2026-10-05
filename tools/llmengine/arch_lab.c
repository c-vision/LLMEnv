/* arch_lab.c — OpenArch A3 ("architecture lab"): test per-modulo con GOLDEN.
 *
 * Per ogni primitiva del NOSTRO engine (funzioni reali, header C `ds4-slow`) confronta l'output
 * float col **riferimento CPU in double** e riporta max/mean/rel error vs tolleranza dichiarata.
 * Deterministico, offline, nessuna GPU/modello → **nessun impatto sulle prestazioni**.
 *
 * Moduli coperti (primitives condivise fra architetture):
 *   linear (matvec) · rms_norm_weight · attention_rows (sink, K=V) · rope_tail_ext (neox tail)
 *
 * Build/uso: clang -O2 -I LLMEngine/mlx/ds4-slow tools/llmengine/arch_lab.c -lm -o /tmp/arch_lab && /tmp/arch_lab
 */
#include "llm_dsv4_attn_fwd.h"   /* llm_dsv4_linear, rms_norm_weight, attention_rows, rope_tail_ext_inplace */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static unsigned long long RS = 0x123456789abcdefULL;
static double rnd(void) { RS ^= RS << 13; RS ^= RS >> 7; RS ^= RS << 17; return (double)((RS >> 11) & 0xFFFFFF) / 8388608.0 - 1.0; }

#define TOL 2e-4

static double g_max, g_sum; static long g_n;
static void acc(float got, double ref) {
    double d = fabs((double)got - ref); if (d > g_max) g_max = d; g_sum += d; g_n++;
}

/* ---- riferimenti in DOUBLE (port fedele ma in double) ---- */
static double ref_rms_w(const float* x, const float* w, int n, float eps, int i) {
    double ss = 0; for (int k = 0; k < n; k++) ss += (double)x[k] * x[k];
    double sc = 1.0 / sqrt(ss / n + eps); return (double)x[i] * sc * (double)w[i];
}
static double ref_dot(const float* a, const float* b, int n) { double s = 0; for (int k = 0; k < n; k++) s += (double)a[k] * (double)b[k]; return s; }

static int test_linear(void) {
    int out = 16, in = 32; float *W = malloc(out * in * 4), *x = malloc(in * 4), *y = malloc(out * 4);
    for (int i = 0; i < out * in; i++) W[i] = (float)rnd(); for (int i = 0; i < in; i++) x[i] = (float)rnd();
    llm_dsv4_linear(y, W, out, in, x);
    g_max = 0; g_sum = 0; g_n = 0;
    for (int o = 0; o < out; o++) acc(y[o], ref_dot(W + (long)o * in, x, in));
    double mean = g_sum / g_n; printf("  linear            max=%.2e mean=%.2e  %s\n", g_max, mean, g_max < TOL ? "OK" : "FAIL");
    free(W); free(x); free(y); return g_max < TOL;
}
static int test_rms(void) {
    int n = 64; float *x = malloc(n * 4), *w = malloc(n * 4), *o = malloc(n * 4);
    for (int i = 0; i < n; i++) { x[i] = (float)rnd(); w[i] = (float)(1.0 + 0.1 * rnd()); }
    llm_dsv4_rms_norm_weight(o, x, w, n, 1e-5f);
    g_max = 0; g_sum = 0; g_n = 0;
    for (int i = 0; i < n; i++) acc(o[i], ref_rms_w(x, w, n, 1e-5f, i));
    double mean = g_sum / g_n; printf("  rms_norm_weight   max=%.2e mean=%.2e  %s\n", g_max, mean, g_max < TOL ? "OK" : "FAIL");
    free(x); free(w); free(o); return g_max < TOL;
}
static int test_attention_rows(void) {
    int nh = 4, hd = 16, nkv = 8; float *q = malloc(nh * hd * 4), *kv = malloc(nkv * hd * 4), *snk = malloc(nh * 4), *o = malloc(nh * hd * 4);
    for (int i = 0; i < nh * hd; i++) q[i] = (float)rnd();
    for (int i = 0; i < nkv * hd; i++) kv[i] = (float)rnd();
    for (int i = 0; i < nh; i++) snk[i] = (float)(rnd() * 2.0);
    llm_dsv4_attention_rows(o, q, kv, nkv, snk, nh, hd);
    g_max = 0; g_sum = 0; g_n = 0;
    for (int h = 0; h < nh; h++) {
        const float* qh = q + h * hd; double mx = snk[h];
        for (int r = 0; r < nkv; r++) { double s = ref_dot(qh, kv + r * hd, hd) / sqrt((double)hd); if (s > mx) mx = s; }
        double den = exp((double)snk[h] - mx);
        for (int r = 0; r < nkv; r++) den += exp(ref_dot(qh, kv + r * hd, hd) / sqrt((double)hd) - mx);
        for (int e = 0; e < hd; e++) {
            double num = 0; for (int r = 0; r < nkv; r++) num += exp(ref_dot(qh, kv + r * hd, hd) / sqrt((double)hd) - mx) * (double)kv[r * hd + e];
            acc(o[h * hd + e], num / den);
        }
    }
    double mean = g_sum / g_n; printf("  attention_rows    max=%.2e mean=%.2e  %s\n", g_max, mean, g_max < 1e-3 ? "OK" : "FAIL");
    free(q); free(kv); free(snk); free(o); return g_max < 1e-3;
}
static int test_rope(void) {
    int nh = 2, hd = 16, nrot = 8, pos = 7; float* x = malloc(nh * hd * 4); float* xr = malloc(nh * hd * 4);
    for (int i = 0; i < nh * hd; i++) { x[i] = (float)rnd(); xr[i] = x[i]; }
    llm_dsv4_rope_tail_ext_inplace(x, nh, hd, nrot, pos, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 0);
    g_max = 0; g_sum = 0; g_n = 0;
    /* riferimento neox (tail), double: theta_i = pos * base^(-2i/nrot), rotazione a coppie (2i,2i+1) */
    int nnope = hd - nrot;
    for (int h = 0; h < nh; h++) for (int i = 0; i < nrot; i += 2) {
        double th = (double)pos * pow(10000.0, -(double)i / (double)nrot);
        double c = cos(th), s = sin(th);
        double x0 = xr[h * hd + nnope + i], x1 = xr[h * hd + nnope + i + 1];
        acc(x[h * hd + nnope + i],     x0 * c - x1 * s);
        acc(x[h * hd + nnope + i + 1], x0 * s + x1 * c);
    }
    double mean = g_sum / g_n; printf("  rope_tail_ext     max=%.2e mean=%.2e  %s\n", g_max, mean, g_max < 1e-3 ? "OK" : "FAIL");
    free(x); free(xr); return g_max < 1e-3;
}

int main(void) {
    printf("[arch_lab] tol=%.1e (float vs riferimento double)\n", (double)TOL);
    int ok = 1;
    ok &= test_linear(); ok &= test_rms(); ok &= test_rope(); ok &= test_attention_rows();
    printf("[arch_lab] %s\n", ok ? "TUTTI OK" : "QUALCHE FAIL");
    return ok ? 0 : 1;
}
