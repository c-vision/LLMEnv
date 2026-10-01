/* llm_dsv4_actquant.h — quantizzazione attivazioni + dot interi (parità col path CPU ds4):
 *  - Q8_0 attivazione e dot Q8_0(w) x Q8_0(x);
 *  - Q8_K attivazione e dot IQ2_XXS/Q2_K(w) x Q8_K(x) (usato dal path esperti prequant).
 * Header-only. Test: llm_dsv4_actquant_test.c. */
#ifndef LLM_DSV4_ACTQUANT_H
#define LLM_DSV4_ACTQUANT_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "llm_ggml.h"

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } LlmDsv4Q8K;

static inline void llm_dsv4_q8_0_quantize_row(const float* x, int n, int8_t* xq, float* xscale) {
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        int i0 = b * 32, bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (int i = 0; i < bn; i++) { float ax = fabsf(x[i0 + i]); if (ax > amax) amax = ax; }
        float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        xscale[b] = d;
        for (int i = 0; i < bn; i++) {
            int v = llmcodec_nearest_int(x[i0 + i] * id);   /* regola del CODEC (§0bis) */
            if (v > 127) v = 127; if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (int i = bn; i < 32; i++) xq[i0 + i] = 0;
    }
}
static inline float llm_dsv4_dot_q8_0_prequant(const uint8_t* w_row, int n, const int8_t* xq, const float* xscale) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((n & 31) == 0) {   /* port VERBATIM di ds4 dot_q8_0_row: 2 accumulatori + vfmaq_n_f32 */
        int blocks = n / 32;
        float32x4_t accv0 = vdupq_n_f32(0.0f), accv1 = vdupq_n_f32(0.0f);
        int b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t s0, s1; memcpy(&s0, w_row + b * 34, 2); memcpy(&s1, w_row + (b + 1) * 34, 2);
            const int8_t* qs0 = (const int8_t*)(w_row + b * 34 + 2);
            const int8_t* qs1 = (const int8_t*)(w_row + (b + 1) * 34 + 2);
            const int8_t* x0 = xq + b * 32; const int8_t* x1 = xq + (b + 1) * 32;
            int32x4_t d0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(qs0), vld1q_s8(x0)), vld1q_s8(qs0 + 16), vld1q_s8(x0 + 16));
            int32x4_t d1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(qs1), vld1q_s8(x1)), vld1q_s8(qs1 + 16), vld1q_s8(x1 + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(d0), llmcodec_f16_to_f32(s0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(d1), llmcodec_f16_to_f32(s1) * xscale[b + 1]);
        }
        if (b < blocks) {
            uint16_t sb; memcpy(&sb, w_row + b * 34, 2);
            const int8_t* qs = (const int8_t*)(w_row + b * 34 + 2); const int8_t* xb = xq + b * 32;
            int32x4_t d = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(qs), vld1q_s8(xb)), vld1q_s8(qs + 16), vld1q_s8(xb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(d), llmcodec_f16_to_f32(sb) * xscale[b]);
        }
        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif
    float acc = 0.0f;
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        uint16_t sb; memcpy(&sb, w_row + b * 34, 2);
        const int8_t* qs = (const int8_t*)(w_row + b * 34 + 2);
        int i0 = b * 32, bnn = n - i0 < 32 ? n - i0 : 32;
        int dot = 0; for (int j = 0; j < bnn; j++) dot += (int)qs[j] * (int)xq[i0 + j];
        acc += llmcodec_f16_to_f32(sb) * xscale[b] * (float)dot;
    }
    return acc;
}

/* Q8_K = formato ggml: delega al CODEC (§0bis). LlmDsv4Q8K ha lo STESSO layout di blk_q8_K. */
static inline void llm_dsv4_q8_K_quantize_row(const float* x, int n, LlmDsv4Q8K* y) {
    llmcodec_ggml_q8_K_encode(x, y, n);
}
/* Port VERBATIM (scalar) di ds4_vec_dot_q2_K_q8_K: accumulo intero `isum`/`summs`. */
static inline int llm_dsv4_dot_q2_16(const uint8_t* q2, const int8_t* q8, int shift) {
#if defined(__ARM_NEON)
    const uint8x16_t packed = vld1q_u8(q2);
    uint8x16_t sh;
    switch (shift) {
        case 0: sh = packed; break;
        case 2: sh = vshrq_n_u8(packed, 2); break;
        case 4: sh = vshrq_n_u8(packed, 4); break;
        default: sh = vshrq_n_u8(packed, 6); break;
    }
    const uint8x16_t vals = vandq_u8(sh, vdupq_n_u8(3));
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vals), vld1q_s8(q8)));
#else
    int s = 0;
    for (int j = 0; j < 16; j++) s += (int)((q2[j] >> shift) & 3u) * (int)q8[j];
    return s;
#endif
}
static inline float llm_dsv4_dot_q2_K_q8_K(const uint8_t* x, int n, const LlmDsv4Q8K* y) {
    int nb = n / 256; float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t* blk = x + i * 84;
        const uint8_t* sc = blk;            /* scales[16] */
        const uint8_t* qs = blk + 16;       /* qs[64] */
        const int8_t* q8 = y[i].qs;
        uint16_t dh, mh; memcpy(&dh, blk + 80, 2); memcpy(&mh, blk + 82, 2);
        int summs = 0;
        for (int j = 0; j < 16; j++) summs += (int)y[i].bsums[j] * (int)(sc[j] >> 4);
        const float dall = y[i].d * llmcodec_f16_to_f32(dh);
        const float dmin = y[i].d * llmcodec_f16_to_f32(mh);
        int isum = 0, is = 0;
        const uint8_t* q2p = qs;
        const int8_t* q8p = q8;
        for (int k = 0; k < 2; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f; isum += d * llm_dsv4_dot_q2_16(q2p, q8p, shift);
                d = sc[is++] & 0x0f; isum += d * llm_dsv4_dot_q2_16(q2p + 16, q8p + 16, shift);
                shift += 2; q8p += 32;
            }
            q2p += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}
/* Port VERBATIM (scalar) di ds4_vec_dot_iq2_xxs_q8_K (griglia firmata calcolata on-the-fly). */
static inline const uint8_t* llm_dsv4_iq2_grid_entry(uint32_t g) { return (const uint8_t*)(llm_iq2_grid + g); }
static inline int llm_dsv4_iq2_grid_signed(const uint8_t* grid, uint32_t sidx, int j) {
    int v = (int)grid[j];
    return (llm_iq2_ksigns[sidx] & llm_iq2_kmask[j]) ? -v : v;
}
static inline int llm_dsv4_dot_iq2_pair_16(const uint8_t* g0, uint32_t s0, const uint8_t* g1, uint32_t s1,
                                           const int8_t* q8) {
    int s = 0;
    for (int j = 0; j < 8; j++) s += llm_dsv4_iq2_grid_signed(g0, s0, j) * (int)q8[j];
    for (int j = 0; j < 8; j++) s += llm_dsv4_iq2_grid_signed(g1, s1, j) * (int)q8[8 + j];
    return s;
}
/* Port VERBATIM di ds4 `dot_f32`: 2 accumulatori a 4 lane, 8 per iterazione, FMA, poi
 * somma orizzontale; tail scalare. Usato per i dot q.k dell'attenzione (512 elementi). */
static inline float llm_dsv4_dot_f32(const float* a, const float* b, int n) {
#if defined(__ARM_NEON)
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
#endif
}

#if defined(__ARM_NEON)
static int llm_dsv4_iq2_signs_ready = 0;
static int8_t llm_dsv4_iq2_signs[128][8];
static inline void llm_dsv4_iq2_signs_ensure(void) {
    if (llm_dsv4_iq2_signs_ready) return;
    for (int s = 0; s < 128; s++)
        for (int j = 0; j < 8; j++)
            llm_dsv4_iq2_signs[s][j] = (int8_t)((llm_iq2_ksigns[s] & llm_iq2_kmask[j]) ? -1 : 1);
    llm_dsv4_iq2_signs_ready = 1;
}
#endif

static inline float llm_dsv4_dot_iq2_xxs_q8_K(const uint8_t* x, int n, const LlmDsv4Q8K* y) {
#if defined(__ARM_NEON)
    llm_dsv4_iq2_signs_ensure();
    int nb = n / 256; float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        uint16_t dh; memcpy(&dh, x + i * 66, 2);
        float d = llmcodec_f16_to_f32(dh) * y[i].d;
        const uint16_t* q2 = (const uint16_t*)(x + i * 66 + 2);
        const int8_t* q8 = y[i].qs;
        float sumf1 = 0.0f, sumf2 = 0.0f;
        for (int ib = 0; ib < 8; ib += 2) {
            const int8x16_t q8b0 = vld1q_s8(q8), q8b1 = vld1q_s8(q8 + 16);
            const int8x16_t q8b2 = vld1q_s8(q8 + 32), q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            uint32_t aux32[4]; memcpy(aux32, q2, sizeof(aux32)); q2 += 8;
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const int8_t* G = (const int8_t*)llm_iq2_grid;
            int8x16_t q2u0 = vcombine_s8(vld1_s8(G + 8 * aux8[0]), vld1_s8(G + 8 * aux8[1]));
            int8x16_t q2u1 = vcombine_s8(vld1_s8(G + 8 * aux8[2]), vld1_s8(G + 8 * aux8[3]));
            int8x16_t q2u2 = vcombine_s8(vld1_s8(G + 8 * aux8[8]), vld1_s8(G + 8 * aux8[9]));
            int8x16_t q2u3 = vcombine_s8(vld1_s8(G + 8 * aux8[10]), vld1_s8(G + 8 * aux8[11]));
            const int8x16_t q2s0 = vcombine_s8(vld1_s8(llm_dsv4_iq2_signs[(aux32[1] >> 0) & 127]),
                                               vld1_s8(llm_dsv4_iq2_signs[(aux32[1] >> 7) & 127]));
            const int8x16_t q2s1 = vcombine_s8(vld1_s8(llm_dsv4_iq2_signs[(aux32[1] >> 14) & 127]),
                                               vld1_s8(llm_dsv4_iq2_signs[(aux32[1] >> 21) & 127]));
            const int8x16_t q2s2 = vcombine_s8(vld1_s8(llm_dsv4_iq2_signs[(aux32[3] >> 0) & 127]),
                                               vld1_s8(llm_dsv4_iq2_signs[(aux32[3] >> 7) & 127]));
            const int8x16_t q2s3 = vcombine_s8(vld1_s8(llm_dsv4_iq2_signs[(aux32[3] >> 14) & 127]),
                                               vld1_s8(llm_dsv4_iq2_signs[(aux32[3] >> 21) & 127]));
            q2u0 = vmulq_s8(q2u0, q2s0); q2u1 = vmulq_s8(q2u1, q2s1);
            q2u2 = vmulq_s8(q2u2, q2s2); q2u3 = vmulq_s8(q2u3, q2s3);
            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u0, q8b0), q2u1, q8b1);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u2, q8b2), q2u3, q8b3);
            sumf1 += (float)vaddvq_s32(p1) * (0.5f + (float)(aux32[1] >> 28));
            sumf2 += (float)vaddvq_s32(p2) * (0.5f + (float)(aux32[3] >> 28));
        }
        sumf += d * (sumf1 + sumf2);
    }
    return 0.25f * sumf;
#else
    int nb = n / 256; float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        uint16_t dh; memcpy(&dh, x + i * 66, 2);
        float d = llmcodec_f16_to_f32(dh) * y[i].d;
        const uint16_t* q2 = (const uint16_t*)(x + i * 66 + 2);
        const int8_t* q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux32[2]; memcpy(aux32, q2, 8); q2 += 4;
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const uint32_t ls = 2u * (aux32[1] >> 28) + 1u;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                uint32_t s0 = (aux32[1] >> (7 * l)) & 127u;
                uint32_t s1 = (aux32[1] >> (7 * (l + 1))) & 127u;
                sumi += llm_dsv4_dot_iq2_pair_16(llm_dsv4_iq2_grid_entry(aux8[l]), s0,
                                                 llm_dsv4_iq2_grid_entry(aux8[l + 1]), s1, q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
#endif
}
/* Port VERBATIM (scalar) di ds4_vec_dot_q4_K_q8_K: accumulo intero `isum`/`summs`. */
static inline float llm_dsv4_dot_q4_K_q8_K(const uint8_t* x, int n, const LlmDsv4Q8K* y) {
    int nb = n / 256; float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t* xb = x + i * 144;
        uint16_t dh, mh; memcpy(&dh, xb, 2); memcpy(&mh, xb + 2, 2);
        const uint8_t* sc = xb + 4; const uint8_t* qs = xb + 16;
        const int8_t* q8 = y[i].qs;
        const float dall = y[i].d * llmcodec_f16_to_f32(dh);
        const float dmin = -y[i].d * llmcodec_f16_to_f32(mh);
        int summs = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t scv, mv; llmcodec_ggml_q4_K_scale_min((uint32_t)j, sc, &scv, &mv);
            summs += (int)mv * ((int)y[i].bsums[j * 2] + (int)y[i].bsums[j * 2 + 1]);
        }
        int isum = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t scv, mv; llmcodec_ggml_q4_K_scale_min((uint32_t)j, sc, &scv, &mv);
            int bo = (j >> 1) * 32, sh = (j & 1) * 4;
            for (int l = 0; l < 32; l++)
                isum += (int)((qs[bo + l] >> sh) & 0xF) * (int)q8[j * 32 + l] * scv;
        }
        sumf += dall * (float)isum + dmin * (float)summs;
    }
    return sumf;
}
static inline float llm_dsv4_dot_w_q8_K(int type, const uint8_t* x, int n, const LlmDsv4Q8K* y) {
    if (type == LLM_GGML_TYPE_Q2_K) return llm_dsv4_dot_q2_K_q8_K(x, n, y);
    if (type == LLM_GGML_TYPE_IQ2_XXS) return llm_dsv4_dot_iq2_xxs_q8_K(x, n, y);
    if (type == LLM_GGML_TYPE_Q4_K) return llm_dsv4_dot_q4_K_q8_K(x, n, y);
    return 0.0f;
}

#endif /* LLM_DSV4_ACTQUANT_H */
