/* gmb_asr_kernels.c — kernel numerici host per Qwen3-ASR.
 * Semantica allineata a antirez/qwen-asr `qwen_asr_kernels*.c` (MIT); implementazione propria
 * (formule canoniche). Include un path NEON per il matvec BF16 (point 4) + generic (parità). */
#include "gmb_asr.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Pool di thread leggero: le matvec grandi (lm_head/MLP/encoder) sono parallelizzate su (seq×out)
 * righe. Sotto soglia (op piccola) resta scalare — lo spawn costerebbe piu' del lavoro. */
static int g_asr_ncpu = 0;
static int asr_ncpu(void) {
    if (g_asr_ncpu == 0) { long n = sysconf(_SC_NPROCESSORS_ONLN); g_asr_ncpu = (n > 1 && n < 32) ? (int)n : 1; }
    return g_asr_ncpu;
}

/* bf16 -> f32 (shift dei 16 bit alti). */
static inline float gmb_bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

void gmb_asr_matmul_t(float* C, const float* A, const float* B, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            const float* a = A + (size_t)i * K; const float* b = B + (size_t)j * K; float s = 0;
            for (int k = 0; k < K; k++) s += a[k] * b[k];
            C[(size_t)i * N + j] = s;
        }
}

typedef struct { const float* x; const float* W; const float* b; float* y; int in, out, i0, i1; } LinJob;
static void* lin_worker(void* p) {
    LinJob* j = (LinJob*)p;
    for (int i = j->i0; i < j->i1; i++) {
        int s = i / j->out, o = i % j->out;
        const float* xr = j->x + (size_t)s * j->in; const float* wr = j->W + (size_t)o * j->in;
        float sum = j->b ? j->b[o] : 0.0f;
        for (int k = 0; k < j->in; k++) sum += wr[k] * xr[k];
        j->y[i] = sum;
    }
    return NULL;
}

void gmb_asr_linear(float* y, const float* x, const float* W, const float* b, int seq, int in, int out) {
    long total = (long)seq * out;
    int nth = asr_ncpu();
    if (nth <= 1 || total * in < 4000000L) {
        LinJob j = { x, W, b, y, in, out, 0, (int)total };
        lin_worker(&j);
        return;
    }
    pthread_t th[31]; LinJob jobs[31]; int nt = 0, per = (int)((total + nth - 1) / nth);
    for (int t = 0; t < nth; t++) {
        int i0 = t * per, i1 = i0 + per; if (i1 > total) i1 = (int)total;
        if (i0 >= i1) break;
        jobs[nt] = (LinJob){ x, W, b, y, in, out, i0, i1 };
        pthread_create(&th[nt], NULL, lin_worker, &jobs[nt]); nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

void gmb_asr_add_inplace(float* a, const float* b, int n) { for (int i = 0; i < n; i++) a[i] += b[i]; }

/* una riga: y[o] = b[o] + dot(W[o,:], x). */
static inline float lin_bf_dot(const float* xr, const uint16_t* wr, int in) {
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= in; k += 4) {
        uint16x4_t w16 = vld1_u16(wr + k);
        uint32x4_t w32 = vshll_n_u16(w16, 16);      /* bf16 -> bit alti f32 */
        acc = vfmaq_f32(acc, vreinterpretq_f32_u32(w32), vld1q_f32(xr + k));
    }
    float s = vaddvq_f32(acc);
    for (; k < in; k++) s += gmb_bf16_to_f32(wr[k]) * xr[k];
    return s;
#else
    float s = 0.0f;
    for (int k = 0; k < in; k++) s += gmb_bf16_to_f32(wr[k]) * xr[k];
    return s;
#endif
}

typedef struct { const float* x; const uint16_t* W; const float* b; float* y; int in, out, i0, i1; } LinBfJob;
static void* lin_bf_worker(void* p) {
    LinBfJob* j = (LinBfJob*)p;
    for (int i = j->i0; i < j->i1; i++) {
        int s = i / j->out, o = i % j->out;
        float sum = (j->b ? j->b[o] : 0.0f) + lin_bf_dot(j->x + (size_t)s * j->in, j->W + (size_t)o * j->in, j->in);
        j->y[i] = sum;
    }
    return NULL;
}

void gmb_asr_linear_bf16(float* y, const float* x, const uint16_t* W, const float* b, int seq, int in, int out) {
    long total = (long)seq * out;
    int nth = asr_ncpu();
    if (nth <= 1 || total * in < 4000000L) {
        LinBfJob j = { x, W, b, y, in, out, 0, (int)total };
        lin_bf_worker(&j);
        return;
    }
    pthread_t th[31]; LinBfJob jobs[31]; int nt = 0, per = (int)((total + nth - 1) / nth);
    for (int t = 0; t < nth; t++) {
        int i0 = t * per, i1 = i0 + per; if (i1 > total) i1 = (int)total;
        if (i0 >= i1) break;
        jobs[nt] = (LinBfJob){ x, W, b, y, in, out, i0, i1 };
        pthread_create(&th[nt], NULL, lin_bf_worker, &jobs[nt]); nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

typedef struct {
    float* out; const float* in; const float* w; const float* b;
    int cin, cout, hin, win, kh, kw, stride, pad, hout, wout, co0, co1;
} ConvJob;
static void* conv_worker(void* p) {
    ConvJob* j = (ConvJob*)p;
    for (int co = j->co0; co < j->co1; co++) {
        for (int ho = 0; ho < j->hout; ho++)
            for (int wo = 0; wo < j->wout; wo++) {
                float acc = j->b ? j->b[co] : 0.0f;
                for (int ci = 0; ci < j->cin; ci++)
                    for (int r = 0; r < j->kh; r++)
                        for (int c = 0; c < j->kw; c++) {
                            int hi = ho * j->stride - j->pad + r, wi = wo * j->stride - j->pad + c;
                            if (hi < 0 || hi >= j->hin || wi < 0 || wi >= j->win) continue;
                            acc += j->in[((size_t)ci * j->hin + hi) * j->win + wi] * j->w[((size_t)co * j->cin + ci) * j->kh * j->kw + (size_t)r * j->kw + c];
                        }
                j->out[((size_t)co * j->hout + ho) * j->wout + wo] = acc;
            }
    }
    return NULL;
}

void gmb_asr_conv2d(float* out, const float* in, const float* w, const float* b,
                    int cin, int cout, int hin, int win, int kh, int kw, int stride, int pad) {
    int hout = (hin + 2 * pad - kh) / stride + 1;
    int wout = (win + 2 * pad - kw) / stride + 1;
    /* Il conv stem e' il ~75% del costo dell'ASR: lo si parallelizza su `co` (canali di USCITA).
     * Il calcolo di ciascun output resta IDENTICO al seriale (stesso ordine di accumulo ci,r,c),
     * solo su thread diversi -> risultato BIT-EXACT. Sotto soglia resta scalare. */
    long ops = (long)cout * hout * wout * cin * kh * kw;
    int nth = asr_ncpu();
    if (nth <= 1 || ops < 4000000L || cout < 2) {
        ConvJob j = { out, in, w, b, cin, cout, hin, win, kh, kw, stride, pad, hout, wout, 0, cout };
        conv_worker(&j);
        return;
    }
    pthread_t th[31]; ConvJob jobs[31]; int nt = 0, per = (cout + nth - 1) / nth;
    for (int t = 0; t < nth; t++) {
        int c0 = t * per, c1 = c0 + per; if (c1 > cout) c1 = cout;
        if (c0 >= c1) break;
        jobs[nt] = (ConvJob){ out, in, w, b, cin, cout, hin, win, kh, kw, stride, pad, hout, wout, c0, c1 };
        pthread_create(&th[nt], NULL, conv_worker, &jobs[nt]); nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

void gmb_asr_layer_norm(float* out, const float* x, const float* w, const float* b, int seq, int hid, float eps) {
    for (int s = 0; s < seq; s++) {
        const float* xr = x + (size_t)s * hid; float mean = 0, var = 0;
        for (int i = 0; i < hid; i++) mean += xr[i]; mean /= hid;
        for (int i = 0; i < hid; i++) { float d = xr[i] - mean; var += d * d; } var /= hid;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < hid; i++) out[(size_t)s * hid + i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
    }
}

void gmb_asr_rms_norm(float* out, const float* x, const float* w, int seq, int hid, float eps) {
    for (int s = 0; s < seq; s++) {
        const float* xr = x + (size_t)s * hid; float ss = 0;
        for (int i = 0; i < hid; i++) ss += xr[i] * xr[i];
        float inv = 1.0f / sqrtf(ss / hid + eps);
        for (int i = 0; i < hid; i++) out[(size_t)s * hid + i] = xr[i] * inv * w[i];
    }
}

void gmb_asr_rms_norm_per_head(float* x, const float* w, int seq, int heads, int hd, float eps) {
    for (int s = 0; s < seq; s++)
        for (int h = 0; h < heads; h++) {
            float* xr = x + ((size_t)s * heads + h) * hd; float ss = 0;
            for (int i = 0; i < hd; i++) ss += xr[i] * xr[i];
            float inv = 1.0f / sqrtf(ss / hd + eps);
            for (int i = 0; i < hd; i++) xr[i] = xr[i] * inv * w[i];
        }
}

void gmb_asr_silu(float* x, int n) { for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i])); }
void gmb_asr_gelu(float* x, int n) { for (int i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678f)); }

void gmb_asr_softmax(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* xr = x + (size_t)r * cols; float mx = xr[0];
        for (int i = 1; i < cols; i++) if (xr[i] > mx) mx = xr[i];
        float sum = 0; for (int i = 0; i < cols; i++) { xr[i] = expf(xr[i] - mx); sum += xr[i]; }
        float inv = sum > 0 ? 1.0f / sum : 0.0f; for (int i = 0; i < cols; i++) xr[i] *= inv;
    }
}

void gmb_asr_swiglu(float* out, const float* gate_up, int seq, int inter) {
    for (int s = 0; s < seq; s++) {
        const float* g = gate_up + (size_t)s * 2 * inter;
        for (int j = 0; j < inter; j++) { float gate = g[2 * j]; out[(size_t)s * inter + j] = (gate / (1.0f + expf(-gate))) * g[2 * j + 1]; }
    }
}

void gmb_asr_attn_bidirectional(float* out, const float* Q, const float* K, const float* V,
                                int seq, int heads, int hd, float scale, const int* ws, int nwin) {
    for (int w = 0; w < nwin; w++) {
        int a = ws[w], b = ws[w + 1];
        for (int i = a; i < b; i++)
            for (int h = 0; h < heads; h++) {
                const float* q = Q + ((size_t)i * heads + h) * hd; float* o = out + ((size_t)i * heads + h) * hd;
                float mx = -1e30f; int len = b - a;
                float* sc = (float*)malloc((size_t)len * sizeof(float));
                for (int j = a; j < b; j++) { const float* k = K + ((size_t)j * heads + h) * hd; float s = 0; for (int d = 0; d < hd; d++) s += q[d] * k[d]; s *= scale; sc[j - a] = s; if (s > mx) mx = s; }
                float sum = 0; for (int j = 0; j < len; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                float inv = sum > 0 ? 1.0f / sum : 0.0f;
                for (int d = 0; d < hd; d++) o[d] = 0.0f;
                for (int j = a; j < b; j++) { float p = sc[j - a] * inv; const float* v = V + ((size_t)j * heads + h) * hd; for (int d = 0; d < hd; d++) o[d] += p * v[d]; }
                free(sc);
            }
    }
}

void gmb_asr_attn_causal_gqa(float* out, const float* Q, const float* K, const float* V,
                             int sq, int sk, int heads, int kv_heads, int hd, float scale, int q_offset) {
    int rep = heads / kv_heads;
    for (int i = 0; i < sq; i++) {
        int qpos = q_offset + i;
        for (int h = 0; h < heads; h++) {
            int kvh = h / rep;
            const float* q = Q + ((size_t)i * heads + h) * hd; float* o = out + ((size_t)i * heads + h) * hd;
            int len = qpos + 1; if (len > sk) len = sk;
            float* sc = (float*)malloc((size_t)len * sizeof(float)); float mx = -1e30f;
            for (int j = 0; j < len; j++) { const float* k = K + ((size_t)j * kv_heads + kvh) * hd; float s = 0; for (int d = 0; d < hd; d++) s += q[d] * k[d]; s *= scale; sc[j] = s; if (s > mx) mx = s; }
            float sum = 0; for (int j = 0; j < len; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; d++) o[d] = 0.0f;
            for (int j = 0; j < len; j++) { float p = sc[j] * inv; const float* v = V + ((size_t)j * kv_heads + kvh) * hd; for (int d = 0; d < hd; d++) o[d] += p * v[d]; }
            free(sc);
        }
    }
}

void gmb_asr_sinusoidal_pe(float* pe, int n_pos, int d_model) {
    int half = d_model / 2;
    for (int p = 0; p < n_pos; p++)
        for (int i = 0; i < half; i++) {
            double freq = exp(-(log(10000.0) * (double)i) / (double)half);
            double a = (double)p * freq;
            pe[(size_t)p * d_model + i] = (float)sin(a);
            pe[(size_t)p * d_model + half + i] = (float)cos(a);
        }
}

void gmb_asr_rope_neox_prep(float* cos_out, float* sin_out, const int* pos, int seq, int hd, float theta) {
    int half = hd / 2;
    for (int s = 0; s < seq; s++)
        for (int i = 0; i < half; i++) {
            double freq = 1.0 / pow((double)theta, (2.0 * (double)i) / (double)hd);
            double a = (double)pos[s] * freq;
            float c = (float)cos(a), si = (float)sin(a);
            cos_out[(size_t)s * hd + i] = c; cos_out[(size_t)s * hd + half + i] = c;
            sin_out[(size_t)s * hd + i] = si; sin_out[(size_t)s * hd + half + i] = si;
        }
}

void gmb_asr_rope_neox_apply(float* x, const float* cos_v, const float* sin_v, int seq, int heads, int hd) {
    int half = hd / 2;
    for (int s = 0; s < seq; s++)
        for (int h = 0; h < heads; h++) {
            float* xr = x + ((size_t)s * heads + h) * hd;
            const float* c = cos_v + (size_t)s * hd; const float* si = sin_v + (size_t)s * hd;
            for (int i = 0; i < half; i++) {
                float x1 = xr[i], x2 = xr[half + i];
                xr[i] = x1 * c[i] - x2 * si[i];
                xr[half + i] = x2 * c[half + i] + x1 * si[half + i];
            }
        }
}

int gmb_asr_argmax_bf16(const float* x, const uint16_t* W, int in, int out) {
    int best = 0; float bestv = -1e30f;
    for (int o = 0; o < out; o++) {
        const uint16_t* wr = W + (size_t)o * in; float s = 0;
        for (int k = 0; k < in; k++) s += gmb_bf16_to_f32(wr[k]) * x[k];
        if (s > bestv) { bestv = s; best = o; }
    }
    return best;
}

/* ===== Q8 runtime (point 5) ===== */
void gmb_asr_q8_quantize(const float* x, int8_t* q, float* scales, int n) {
    int nb = n / GMB_ASR_Q8_BLK;
    for (int b = 0; b < nb; b++) {
        const float* xb = x + (size_t)b * GMB_ASR_Q8_BLK; float amax = 0;
        for (int i = 0; i < GMB_ASR_Q8_BLK; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        float sc = amax > 0 ? amax / 127.0f : 1.0f; scales[b] = sc;
        for (int i = 0; i < GMB_ASR_Q8_BLK; i++) {
            float v = xb[i] / sc; long r = (long)(v < 0 ? v - 0.5f : v + 0.5f);
            if (r > 127) r = 127; if (r < -127) r = -127; q[(size_t)b * GMB_ASR_Q8_BLK + i] = (int8_t)r;
        }
    }
}

void gmb_asr_q8_dequantize(const int8_t* q, const float* scales, float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = (float)q[i] * scales[i / GMB_ASR_Q8_BLK];
}
