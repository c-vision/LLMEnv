/* llmcodec_moe_quant.c — quantizzatore affine RTN (formato MLX). Vedi include/llmcodec_moe_quant.h. */
#include "llmcodec_moe_quant.h"
#include "llmcodec_fp16.h"   /* atomi f16/bf16 UNA SORGENTE (§0bis) */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* f32->f16 variante MLX (TRONCAMENTO normali; NON round-to-even come llmcodec_f16_from_f32):
 * tenuta identica per non alterare la parita' bit-exact del converter MLX. */


/* f16_to_f32/bf16_to_f32: atomi del CODEC (llmcodec_fp16.h, §0bis). */

size_t llmcodec_moe_affine_w_bytes(uint32_t rows, uint32_t cols, uint32_t bits) {
    /* Packing a FLUSSO CONTIGUO di bit (vedi llmcodec_moe_affine_quantize): la larghezza
     * impacchettata e' cols*bits/32, valida per bits 1..8 (contiguo, LSB-first).
     * Per bits potenza di 2 coincide con il vecchio schema a gruppi di 32/bits. */
    if (bits < 1 || bits > 8) return 0;
    if ((cols * bits) % 32 != 0) return 0;
    return (size_t)rows * (cols * bits / 32) * 4;
}
size_t llmcodec_moe_affine_sc_bytes(uint32_t rows, uint32_t cols, uint32_t group) {
    if (group == 0) return 0;
    return (size_t)rows * (cols / group) * 2;
}

int llmcodec_moe_affine_quantize(const float* x, uint32_t rows, uint32_t cols,
                            uint32_t group, uint32_t bits,
                            uint32_t* w_out, uint16_t* scales_out, uint16_t* biases_out) {
    if (x == NULL || w_out == NULL || scales_out == NULL || biases_out == NULL) return -1;
    if (bits < 1 || bits > 8) return -1;   /* 1..8 bit (flusso contiguo LSB-first) */
    if (group == 0 || (cols % group) != 0) return -1;
    if ((cols * bits) % 32 != 0) return -1;
    uint32_t packed_cols = cols * bits / 32;
    uint32_t gcols = cols / group;
    uint32_t qmax = (1u << bits) - 1u;
    memset(w_out, 0, (size_t)rows * packed_cols * 4);

    for (uint32_t o = 0; o < rows; ++o) {
        const float* xr = x + (size_t)o * cols;
        uint32_t* wr = w_out + (size_t)o * packed_cols;
        uint16_t* sr = scales_out + (size_t)o * gcols;
        uint16_t* br = biases_out + (size_t)o * gcols;
        for (uint32_t gi = 0; gi < gcols; ++gi) {
            float mn = xr[gi * group], mx = mn;
            for (uint32_t k = 1; k < group; ++k) {
                float v = xr[gi * group + k];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float scale = (mx - mn) / (float)qmax;
            float bias = mn;
            if (scale <= 0.0f) scale = 1e-8f;   /* gruppo costante: evita /0 (q tutti 0) */
            sr[gi] = llmcodec_f16_from_f32(scale);
            br[gi] = llmcodec_f16_from_f32(bias);
            /* dequant usa la scale F16 arrotondata: quantizza con QUELLA per coerenza di round-trip */
            float s16 = llmcodec_f16_to_f32(sr[gi]), b16 = llmcodec_f16_to_f32(br[gi]);
            for (uint32_t k = 0; k < group; ++k) {
                float qf = (xr[gi * group + k] - b16) / s16;
                long q = lroundf(qf);
                if (q < 0) q = 0;
                if (q > (long)qmax) q = (long)qmax;
                /* FLUSSO CONTIGUO LSB-first (come MLX pack_and_quantize, ramo non potenza di 2):
                 * i `bits` bit di ogni valore vanno dal LSB al MSB, in sequenza, nei word
                 * little-endian. Per bits potenza di 2 e' IDENTICO al vecchio schema. */
                size_t bitpos = (size_t)(gi * group + k) * bits;
                uint32_t uq = (uint32_t)q;
                for (uint32_t b = 0; b < bits; ++b) {
                    if ((uq >> b) & 1u) wr[bitpos >> 5] |= (1u << (bitpos & 31u));
                    bitpos++;
                }
            }
        }
    }
    return 0;
}

void llmcodec_moe_affine_dequantize(const uint32_t* w, const uint16_t* scales, const uint16_t* biases,
                               uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                               int sc_f16, float* out) {
    if (w == NULL || scales == NULL || biases == NULL || out == NULL) return;
    uint32_t packed_cols = cols * bits / 32, gcols = cols / group;
    for (uint32_t o = 0; o < rows; ++o) {
        const uint32_t* wr = w + (size_t)o * packed_cols;
        const uint16_t* sr = scales + (size_t)o * gcols;
        const uint16_t* br = biases + (size_t)o * gcols;
        for (uint32_t k = 0; k < cols; ++k) {
            /* lettura del flusso contiguo LSB-first (inverso di llmcodec_moe_affine_quantize) */
            size_t bitpos = (size_t)k * bits;
            uint32_t q = 0;
            for (uint32_t b = 0; b < bits; ++b) {
                q |= ((wr[bitpos >> 5] >> (bitpos & 31u)) & 1u) << b;
                bitpos++;
            }
            float s = sc_f16 ? llmcodec_f16_to_f32(sr[k / group]) : llmcodec_bf16_to_f32(sr[k / group]);
            float b = sc_f16 ? llmcodec_f16_to_f32(br[k / group]) : llmcodec_bf16_to_f32(br[k / group]);
            out[(size_t)o * cols + k] = s * (float)q + b;
        }
    }
}

/* ── Quantizzatore affine PESATO dall'IMATRIX (DS4-Q1.b §3.5), tensor-level. ──
 * Per ogni riga e gruppo, raffinamento stile Lloyd attorno allo scale/bias "naive" (edge-anchored,
 * = formula MLX affine_quantize): grid search log-spaziata su 61 scale + bias ottimo in forma
 * chiusa (alternanza codici<->bias). Peso per-colonna w_i = imp[i]*sqrt(var_blocco + x_i^2):
 * l'errore sui valori ad ALTA importanza pesa di piu'. imp==NULL -> fallback IDENTICO al RTN. */
static void llmcodec_w_naive_affine_group(const float* x, int n, int bits, float* out_scale, float* out_bias) {
    const float n_bins = (float)((1 << bits) - 1);
    float wmin = x[0], wmax = x[0];
    for (int i = 1; i < n; i++) { if (x[i] < wmin) wmin = x[i]; if (x[i] > wmax) wmax = x[i]; }
    float scale = (wmax - wmin) / n_bins;
    if (scale < 1e-7f) scale = 1e-7f;
    int side = fabsf(wmin) > fabsf(wmax);
    scale = side ? scale : -scale;
    float edge = side ? wmin : wmax;
    float q0 = roundf(edge / scale);
    int at_zero = (q0 == 0.0f);
    scale = at_zero ? scale : edge / q0;
    float bias = at_zero ? 0.0f : edge;
    *out_scale = scale; *out_bias = bias;
}
static float llmcodec_w_affine_group_error(const float* x, const float* w, int n,
                                      float scale, float bias, float n_bins, uint8_t* codes_out) {
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
        float c = roundf((x[i] - bias) / scale);
        if (c < 0) c = 0; else if (c > n_bins) c = n_bins;
        if (codes_out) codes_out[i] = (uint8_t)c;
        float d = x[i] - (c * scale + bias);
        err += (w ? w[i] : 1.0f) * d * d;
    }
    return err;
}
static void llmcodec_w_group_quantize(const float* x, const float* imp, int n, int bits,
                                 float* out_scale, float* out_bias, uint8_t* codes_out) {
    const float n_bins = (float)((1 << bits) - 1);
    float scale0, bias0;
    llmcodec_w_naive_affine_group(x, n, bits, &scale0, &bias0);
    if (!imp) { llmcodec_w_affine_group_error(x, NULL, n, scale0, bias0, n_bins, codes_out); *out_scale = scale0; *out_bias = bias0; return; }
    double mean = 0.0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    double var = 0.0; for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; } var /= n;
    float* w = (float*)malloc(sizeof(float) * (size_t)n);
    if (!w) { llmcodec_w_affine_group_error(x, NULL, n, scale0, bias0, n_bins, codes_out); *out_scale = scale0; *out_bias = bias0; return; }
    for (int i = 0; i < n; i++) w[i] = imp[i] * sqrtf((float)var + x[i] * x[i]);
    double wn = 0.0, wd = 0.0;
    for (int i = 0; i < n; i++) { wn += (double)w[i] * x[i]; wd += w[i]; }
    const float wmean = wd > 1e-12 ? (float)(wn / wd) : bias0;
    float best_scale = scale0, best_bias = bias0;
    float best_err = llmcodec_w_affine_group_error(x, w, n, scale0, bias0, n_bins, NULL);
    const float log_lo = logf(1.0f / 50.0f), log_hi = logf(1.3f);
    const int n_cand = 60;
    for (int gi = 0; gi <= n_cand; gi++) {
        float t = (float)gi / (float)n_cand;
        float cand = scale0 * expf(log_lo + t * (log_hi - log_lo));
        if (fabsf(cand) < 1e-7f) continue;
        float bias = wmean;
        for (int it = 0; it < 12; it++) {
            double nm = 0.0, dm = 0.0;
            for (int i = 0; i < n; i++) {
                float c = roundf((x[i] - bias) / cand);
                if (c < 0) c = 0; else if (c > n_bins) c = n_bins;
                nm += (double)w[i] * (double)(x[i] - c * cand);
                dm += (double)w[i];
            }
            if (dm > 1e-12) bias = (float)(nm / dm);
        }
        float e = llmcodec_w_affine_group_error(x, w, n, cand, bias, n_bins, NULL);
        if (e < best_err) { best_err = e; best_scale = cand; best_bias = bias; }
    }
    llmcodec_w_affine_group_error(x, w, n, best_scale, best_bias, n_bins, codes_out);
    *out_scale = best_scale; *out_bias = best_bias;
    free(w);
}

void llmcodec_moe_weighted_affine_group(const float* x, const float* imp, uint32_t n, uint32_t bits,
                                   float* out_scale, float* out_bias, uint8_t* codes_out) {
    if (!x || !out_scale || !out_bias) return;
    llmcodec_w_group_quantize(x, imp, (int)n, (int)bits, out_scale, out_bias, codes_out);
}

int llmcodec_moe_affine_quantize_weighted(const float* x, const float* imp, uint32_t rows, uint32_t cols,
                                      uint32_t group, uint32_t bits,
                                      uint32_t* w_out, uint16_t* scales_out, uint16_t* biases_out) {
    if (x == NULL || w_out == NULL || scales_out == NULL || biases_out == NULL) return -1;
    if (bits < 1 || bits > 8) return -1;   /* 1..8 bit (flusso contiguo LSB-first) */
    if (group == 0 || (cols % group) != 0) return -1;
    if ((cols * bits) % 32 != 0) return -1;
    uint32_t packed_cols = cols * bits / 32, gcols = cols / group;
    uint32_t qmax = (1u << bits) - 1u;
    memset(w_out, 0, (size_t)rows * packed_cols * 4);
    uint8_t* codes = (uint8_t*)malloc(group);
    if (!codes) return -1;
    for (uint32_t o = 0; o < rows; ++o) {
        const float* xr = x + (size_t)o * cols;
        uint32_t* wr = w_out + (size_t)o * packed_cols;
        uint16_t* sr = scales_out + (size_t)o * gcols;
        uint16_t* br = biases_out + (size_t)o * gcols;
        for (uint32_t gi = 0; gi < gcols; ++gi) {
            const float* xg = xr + (size_t)gi * group;
            const float* ig = imp ? (imp + (size_t)gi * group) : NULL;
            float scale, bias;
            llmcodec_w_group_quantize(xg, ig, (int)group, (int)bits, &scale, &bias, codes);
            sr[gi] = llmcodec_f16_from_f32(scale);
            br[gi] = llmcodec_f16_from_f32(bias);
            float s16 = llmcodec_f16_to_f32(sr[gi]), b16 = llmcodec_f16_to_f32(br[gi]);
            for (uint32_t k = 0; k < group; ++k) {   /* ri-arrotonda con lo scale F16 (round-trip) */
                float qf = (xg[k] - b16) / s16;
                long q = lroundf(qf);
                if (q < 0) q = 0;
                if (q > (long)qmax) q = (long)qmax;
                /* flusso contiguo LSB-first (vedi llmcodec_moe_affine_quantize) */
                size_t bitpos = (size_t)(gi * group + k) * bits;
                uint32_t uq = (uint32_t)q;
                for (uint32_t b = 0; b < bits; ++b) { if ((uq >> b) & 1u) wr[bitpos >> 5] |= (1u << (bitpos & 31u)); bitpos++; }
            }
        }
    }
    free(codes);
    return 0;
}
