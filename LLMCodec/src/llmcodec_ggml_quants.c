/* llmcodec_ggml_quants.c — codec GGML (blocco<->f32). Vedi include/llmcodec_ggml_quants.h.
 * REIMPLEMENTAZIONE PROPRIA del formato dei blocchi ggml (layout pubblico): non è codice copiato
 * da terzi. La dimensione dei blocchi è allineata a `llmcodec_ggml_block()` (llmcodec_gguf_writer.c). */
#include "llmcodec_ggml_quants.h"
#include "llmcodec_fp8_fp4.h"
#include <stdio.h>
#include "llmcodec_fp16.h"
#include "llmcodec_ggml_iq_grids.h"
#include "llmcodec_iq2_xxs.h"   /* llmcodec_make_qp_quants (scala del blocco nella variante imatrix) */

#include <string.h>
#include <math.h>

/* ── conversioni f16/bf16 (round-to-nearest) ── */

static float q_f16_to_f32(uint16_t h) { return llmcodec_f16_to_f32(h); }
static uint16_t q_f32_to_bf16(float f) { return llmcodec_bf16_from_f32(f); }   /* atomo CORE (era locale, ripristinato) */
static float q_bf16_to_f32(uint16_t h) { return llmcodec_bf16_to_f32(h); }

#define QMAX(a, b) ((a) > (b) ? (a) : (b))
#define QMIN(a, b) ((a) < (b) ? (a) : (b))

/* arrotondamento a intero piu' vicino (trucco del magic number, senza libm). */
int llmcodec_nearest_int(float f) {
    float v = f + 12582912.0f;   /* 1.5 * 2^23 */
    int i; memcpy(&i, &v, 4);
    return (i & 0x007FFFFF) - 0x00400000;
}

/* Ricerca scala+min (asimmetrica) per un gruppo. Riscrittura propria. Riempie L (0..nmax) e
 * *the_min; ritorna lo scale. `use_mad` = errore L1 invece di L2. */
/* Ricerca scala (segno, senza min) per un gruppo. Riscrittura propria. */

/* Ricerca dello scale ottimo per un gruppo (rmse_type: 1 = peso x^2). Riempie L[0..n) e ritorna lo scale. */

/* layout blocchi (ggml-common.h) — natural packing, size = type_size */
typedef struct { uint16_t d; uint8_t qs[16]; } blk_q4_0;                 /* 18 B / 32 */
typedef struct { uint16_t d, m; uint8_t qs[16]; } blk_q4_1;             /* 20 B / 32 */
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } blk_q5_0; /* 22 B / 32 */
typedef struct { uint16_t d, m; uint8_t qh[4]; uint8_t qs[16]; } blk_q5_1; /* 24 B / 32 */
typedef struct { uint16_t d; int8_t qs[32]; } blk_q8_0;                 /* 34 B / 32 */

/* K-quant (QK_K = 256), layout pubblico ggml */
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; } blk_q2_K;            /* 84 B / 256 */
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } blk_q3_K; /* 110 B / 256 */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } blk_q4_K;           /* 144 B / 256 */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } blk_q5_K; /* 176 B / 256 */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } blk_q6_K;  /* 210 B / 256 */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } blk_q8_K;                       /* 292 B / 256 */

/* coppia scala/min a 6 bit j-esima di un blocco K (4-bit K). */
static void k4_scale_min(int j, const uint8_t* q, uint8_t* sd, uint8_t* sm) {
    if (j < 4) { *sd = (uint8_t)(q[j] & 63u); *sm = (uint8_t)(q[j + 4] & 63u); }
    else { *sd = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
           *sm = (uint8_t)((q[j + 4] >> 4)    | ((q[j]     >> 6) << 4)); }
}

/* float 4-bit (OCP MX/NV): E2M1 con scala condivisa E8M0 (MX) o UE4M3 per sub-blocco (NV). */
typedef struct { uint8_t e; uint8_t qs[16]; } blk_mxfp4;       /* 17 B / 32 */
typedef struct { uint8_t d[4]; uint8_t qs[32]; } blk_nvfp4;    /* 36 B / 64 */

/* e2m1/e8m0/ue4m3: atomi del CODEC (llmcodec_fp8_fp4.h / llmcodec_fp16.h, §0bis) — usati direttamente. */
/* Indice E2M1 (0..15) piu' vicino a x con scala d. */

/* 1/2-bit e ternari (TQ). */
typedef struct { uint16_t d; uint8_t qs[16]; } blk_q1_0;       /* 18 B / 128 (1 bit/val) */
typedef struct { uint16_t d; uint8_t qs[16]; } blk_q2_0;       /* 18 B / 64  (2 bit/val) */
typedef struct { uint8_t qs[48]; uint8_t qh[4]; uint16_t d; } blk_tq1_0; /* 54 B / 256 */
typedef struct { uint8_t qs[64]; uint16_t d; } blk_tq2_0;      /* 66 B / 256 */

/* IQ4 (codebook iq4nl, non uniforme). */
static const int8_t k_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
typedef struct { uint16_t d; uint8_t qs[16]; } blk_iq4_nl;                       /* 18 B / 32 */
typedef struct { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } blk_iq4_xs; /* 136 B / 256 */

/* IQ (grid-based). */
typedef struct { uint16_t d; uint16_t qs[32]; } blk_iq2_xxs;                                       /* 66 B / 256 */
typedef struct { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; } blk_iq2_xs;                     /* 74 B / 256 */
typedef struct { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; } blk_iq2_s;        /* 82 B / 256 */
typedef struct { uint16_t d; uint8_t qs[96]; } blk_iq3_xxs;                                        /* 98 B / 256 */
typedef struct { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; } blk_iq3_s; /* 110 B / 256 */
typedef struct { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; } blk_iq1_s;                          /* 50 B / 256 */
typedef struct { uint8_t qs[32]; uint8_t qh[16]; uint8_t scales[8]; } blk_iq1_m;                   /* 56 B / 256 */

#define IQ1S_DELTA_V 0.125f

int64_t llmcodec_ggml_block_vals(int t) {
    switch (t) {
        case LLMCODEC_GGML_T_F32: case LLMCODEC_GGML_T_F16: case LLMCODEC_GGML_T_BF16: return 1;
        case LLMCODEC_GGML_T_Q4_0: case LLMCODEC_GGML_T_Q4_1: case LLMCODEC_GGML_T_Q5_0:
        case LLMCODEC_GGML_T_Q5_1: case LLMCODEC_GGML_T_Q8_0: return 32;
        case LLMCODEC_GGML_T_Q2_K: case LLMCODEC_GGML_T_Q3_K: case LLMCODEC_GGML_T_Q4_K:
        case LLMCODEC_GGML_T_Q5_K: case LLMCODEC_GGML_T_Q6_K: case LLMCODEC_GGML_T_Q8_K: return 256;
        case LLMCODEC_GGML_T_MXFP4: return 32;
        case LLMCODEC_GGML_T_NVFP4: return 64;
        case LLMCODEC_GGML_T_Q1_0: return 128;
        case LLMCODEC_GGML_T_Q2_0: return 64;
        case LLMCODEC_GGML_T_TQ1_0: case LLMCODEC_GGML_T_TQ2_0: return 256;
        case LLMCODEC_GGML_T_IQ4_NL: return 32;
        case LLMCODEC_GGML_T_IQ4_XS: return 256;
        case LLMCODEC_GGML_T_IQ1_S: case LLMCODEC_GGML_T_IQ1_M: case LLMCODEC_GGML_T_IQ2_XXS: case LLMCODEC_GGML_T_IQ2_XS:
        case LLMCODEC_GGML_T_IQ2_S: case LLMCODEC_GGML_T_IQ3_XXS: case LLMCODEC_GGML_T_IQ3_S: return 256;
        default: return 0;
    }
}

int llmcodec_ggml_dequant_row(int t, const void* src, float* dst, int64_t n) {
    if (src == NULL || dst == NULL || n <= 0) return -1;
    switch (t) {
        case LLMCODEC_GGML_T_F32: memcpy(dst, src, (size_t)n * 4u); return 0;
        case LLMCODEC_GGML_T_F16: { const uint16_t* s = (const uint16_t*)src; for (int64_t i = 0; i < n; ++i) dst[i] = q_f16_to_f32(s[i]); return 0; }
        case LLMCODEC_GGML_T_BF16: { const uint16_t* s = (const uint16_t*)src; for (int64_t i = 0; i < n; ++i) dst[i] = q_bf16_to_f32(s[i]); return 0; }
        /* INTERI (non quantizzati): riportati dall'engine DS4 (I8/I16/I32 in ds4flash.gguf). */
        case LLMCODEC_GGML_T_I8:  { const int8_t*  s=(const int8_t*)src;  for (int64_t i=0;i<n;++i) dst[i]=(float)s[i]; return 0; }
        case LLMCODEC_GGML_T_I16: { const int16_t* s=(const int16_t*)src; for (int64_t i=0;i<n;++i) dst[i]=(float)s[i]; return 0; }
        case LLMCODEC_GGML_T_I32: { const int32_t* s=(const int32_t*)src; for (int64_t i=0;i<n;++i) dst[i]=(float)s[i]; return 0; }
        default: break;
    }
    int64_t qk = llmcodec_ggml_block_vals(t);
    if (qk == 0 || (n % qk) != 0) return -1;
    const int64_t nb = n / qk;
    if (t == LLMCODEC_GGML_T_Q8_0) {
        const blk_q8_0* x = (const blk_q8_0*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            for (int j = 0; j < 32; ++j) dst[i*32 + j] = (float)x[i].qs[j] * d;
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q8_1) {   /* {f16 d; f16 s; int8 qs[32]} — dequant = d*qs (s = somma) */
        const uint8_t* p = (const uint8_t*)src;
        for (int64_t i = 0; i < nb; ++i) {
            uint16_t dh; memcpy(&dh, p, 2); float d = q_f16_to_f32(dh);
            const int8_t* qs = (const int8_t*)(p + 4);
            for (int j = 0; j < 32; ++j) dst[i*32 + j] = (float)qs[j] * d;
            p += 36;
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q4_0) {
        const blk_q4_0* x = (const blk_q4_0*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            for (int j = 0; j < 16; ++j) {
                dst[i*32 + j     ] = (float)((x[i].qs[j] & 0x0F) - 8) * d;
                dst[i*32 + j + 16] = (float)((x[i].qs[j] >>   4) - 8) * d;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q4_1) {
        const blk_q4_1* x = (const blk_q4_1*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d), m = q_f16_to_f32(x[i].m);
            for (int j = 0; j < 16; ++j) {
                dst[i*32 + j     ] = (float)(x[i].qs[j] & 0x0F) * d + m;
                dst[i*32 + j + 16] = (float)(x[i].qs[j] >>   4) * d + m;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q5_0 || t == LLMCODEC_GGML_T_Q5_1) {
        for (int64_t i = 0; i < nb; ++i) {
            uint32_t qh; const uint8_t* qh_p = (t == LLMCODEC_GGML_T_Q5_0) ? ((const blk_q5_0*)src)[i].qh : ((const blk_q5_1*)src)[i].qh;
            const uint8_t* qs = (t == LLMCODEC_GGML_T_Q5_0) ? ((const blk_q5_0*)src)[i].qs : ((const blk_q5_1*)src)[i].qs;
            float d, m = 0.0f;
            if (t == LLMCODEC_GGML_T_Q5_0) d = q_f16_to_f32(((const blk_q5_0*)src)[i].d);
            else { d = q_f16_to_f32(((const blk_q5_1*)src)[i].d); m = q_f16_to_f32(((const blk_q5_1*)src)[i].m); }
            memcpy(&qh, qh_p, sizeof(qh));
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh0 = (uint8_t)(((qh >> (j +  0)) << 4) & 0x10);
                const uint8_t xh1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
                if (t == LLMCODEC_GGML_T_Q5_0) {
                    dst[i*32 + j     ] = (float)(((qs[j] & 0x0F) | xh0) - 16) * d;
                    dst[i*32 + j + 16] = (float)(((qs[j] >>   4) | xh1) - 16) * d;
                } else {
                    dst[i*32 + j     ] = (float)((qs[j] & 0x0F) | xh0) * d + m;
                    dst[i*32 + j + 16] = (float)((qs[j] >>   4) | xh1) * d + m;
                }
            }
        }
        return 0;
    }
    /* ── K-quant (QK_K = 256), 8/16 sotto-blocchi ── */
    if (t == LLMCODEC_GGML_T_Q8_K) {
        const blk_q8_K* x = (const blk_q8_K*)src;
        for (int64_t i = 0; i < nb; ++i)
            for (int j = 0; j < 256; ++j) dst[i*256 + j] = x[i].d * (float)x[i].qs[j];
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q2_K) {
        const blk_q2_K* x = (const blk_q2_K*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d), mn = q_f16_to_f32(x[i].dmin);
            const uint8_t* q = x[i].qs; float* yp = dst + i*256; int is = 0;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc = x[i].scales[is++];
                    float dl = d * (float)(sc & 0x0F), ml = mn * (float)(sc >> 4);
                    for (int l = 0; l < 16; ++l) *yp++ = dl * (float)((q[l]      >> shift) & 3u) - ml;
                    sc = x[i].scales[is++];
                    dl = d * (float)(sc & 0x0F); ml = mn * (float)(sc >> 4);
                    for (int l = 0; l < 16; ++l) *yp++ = dl * (float)((q[l + 16] >> shift) & 3u) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q3_K) {
        const blk_q3_K* x = (const blk_q3_K*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d_all = q_f16_to_f32(x[i].d);
            const uint8_t* q = x[i].qs; const uint8_t* hm = x[i].hmask;
            uint32_t aux[4]; memset(aux, 0, sizeof aux); memcpy(aux, x[i].scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = ( aux[0]       & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = ( aux[1]       & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
            const int8_t* sc = (const int8_t*)aux;
            float* yp = dst + i*256; int is = 0; uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = d_all * (float)(sc[is++] - 32);
                    for (int l = 0; l < 16; ++l) *yp++ = dl * (float)((int)((q[l]      >> shift) & 3u) - ((hm[l]      & m) ? 0 : 4));
                    dl = d_all * (float)(sc[is++] - 32);
                    for (int l = 0; l < 16; ++l) *yp++ = dl * (float)((int)((q[l + 16] >> shift) & 3u) - ((hm[l + 16] & m) ? 0 : 4));
                    shift += 2; m = (uint8_t)(m << 1);
                }
                q += 32;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q4_K) {
        const blk_q4_K* x = (const blk_q4_K*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d), mn = q_f16_to_f32(x[i].dmin);
            const uint8_t* q = x[i].qs; float* yp = dst + i*256; int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                k4_scale_min(is + 0, x[i].scales, &sc, &m);
                float d1 = d * (float)sc, m1 = mn * (float)m;
                k4_scale_min(is + 1, x[i].scales, &sc, &m);
                float d2 = d * (float)sc, m2 = mn * (float)m;
                for (int l = 0; l < 32; ++l) yp[j + l]      = d1 * (float)(q[l] & 0x0F) - m1;
                for (int l = 0; l < 32; ++l) yp[j + 32 + l] = d2 * (float)(q[l] >>   4) - m2;
                q += 32; is += 2;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q5_K) {
        const blk_q5_K* x = (const blk_q5_K*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d), mn = q_f16_to_f32(x[i].dmin);
            const uint8_t* ql = x[i].qs; const uint8_t* qh = x[i].qh;
            float* yp = dst + i*256; int is = 0; uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                k4_scale_min(is + 0, x[i].scales, &sc, &m); float d1 = d * (float)sc, m1 = mn * (float)m;
                k4_scale_min(is + 1, x[i].scales, &sc, &m); float d2 = d * (float)sc, m2 = mn * (float)m;
                for (int l = 0; l < 32; ++l) yp[j + l]      = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) yp[j + 32 + l] = d2 * (float)((ql[l] >>   4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q6_K) {
        const blk_q6_K* x = (const blk_q6_K*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            const uint8_t* ql = x[i].ql; const uint8_t* qh = x[i].qh; const int8_t* sc = x[i].scales;
            float* yp = dst + i*256;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int iss = l / 16;
                    int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                    int q3 = (int)((ql[l]      >> 4)   | (((qh[l] >> 4) & 3u) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3u) << 4)) - 32;
                    yp[n + l +  0] = d * (float)sc[iss + 0] * (float)q1;
                    yp[n + l + 32] = d * (float)sc[iss + 2] * (float)q2;
                    yp[n + l + 64] = d * (float)sc[iss + 4] * (float)q3;
                    yp[n + l + 96] = d * (float)sc[iss + 6] * (float)q4;
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_MXFP4) {
        const blk_mxfp4* x = (const blk_mxfp4*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = llmcodec_e8m0_to_f32(x[i].e);
            for (int j = 0; j < 16; ++j) {
                dst[i*32 + j     ] = llmcodec_fp4_nibble_to_f32(x[i].qs[j] & 0x0F) * d;
                dst[i*32 + j + 16] = llmcodec_fp4_nibble_to_f32(x[i].qs[j] >>   4) * d;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_NVFP4) {
        const blk_nvfp4* x = (const blk_nvfp4*)src;
        for (int64_t i = 0; i < nb; ++i) {
            for (int s = 0; s < 4; ++s) {
                const float d = llmcodec_ue4m3_to_f32(x[i].d[s]);
                float* yb = dst + i*64 + s*16;
                for (int j = 0; j < 8; ++j) {
                    yb[j]     = llmcodec_fp4_nibble_to_f32(x[i].qs[s*8 + j] & 0x0F) * d;
                    yb[j + 8] = llmcodec_fp4_nibble_to_f32(x[i].qs[s*8 + j] >>   4) * d;
                }
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q1_0) {
        const blk_q1_0* x = (const blk_q1_0*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d), nd = -d;
            for (int j = 0; j < 128; ++j)
                dst[i*128 + j] = ((x[i].qs[j >> 3] >> (j & 7)) & 1u) ? d : nd;
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_Q2_0) {
        const blk_q2_0* x = (const blk_q2_0*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            for (int j = 0; j < 64; ++j)
                dst[i*64 + j] = (float)((int)((x[i].qs[j >> 2] >> ((j & 3) * 2)) & 3u) - 1) * d;
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_TQ2_0) {
        const blk_tq2_0* x = (const blk_tq2_0*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            float* yp = dst + i*256;
            for (int j = 0; j < 64; j += 32)
                for (int l = 0; l < 4; ++l)
                    for (int m = 0; m < 32; ++m)
                        *yp++ = (float)((int)((x[i].qs[j + m] >> (l*2)) & 3u) - 1) * d;
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_TQ1_0) {
        const blk_tq1_0* x = (const blk_tq1_0*)src;
        const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            float* yp = dst + i*256;
            for (int j = 0; j < 32; j += 32)
                for (int n = 0; n < 5; ++n)
                    for (int m = 0; m < 32; ++m) {
                        int xi = ((int)((uint16_t)((uint8_t)(x[i].qs[j + m] * pow3[n])) * 3)) >> 8;
                        *yp++ = (float)(xi - 1) * d;
                    }
            for (int j = 32; j < 48; j += 16)
                for (int n = 0; n < 5; ++n)
                    for (int m = 0; m < 16; ++m) {
                        int xi = ((int)((uint16_t)((uint8_t)(x[i].qs[j + m] * pow3[n])) * 3)) >> 8;
                        *yp++ = (float)(xi - 1) * d;
                    }
            for (int n = 0; n < 4; ++n)
                for (int j = 0; j < 4; ++j) {
                    int xi = ((int)((uint16_t)((uint8_t)(x[i].qh[j] * pow3[n])) * 3)) >> 8;
                    *yp++ = (float)(xi - 1) * d;
                }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ4_NL) {
        const blk_iq4_nl* x = (const blk_iq4_nl*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            for (int j = 0; j < 16; ++j) {
                dst[i*32 + j     ] = d * (float)k_iq4nl[x[i].qs[j] & 0x0F];
                dst[i*32 + j + 16] = d * (float)k_iq4nl[x[i].qs[j] >>   4];
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ4_XS) {
        const blk_iq4_xs* x = (const blk_iq4_xs*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            float* yp = dst + i*256; const uint8_t* qs = x[i].qs;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((x[i].scales_l[ib/2] >> (4 * (ib % 2))) & 0x0F) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d * (float)(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    yp[j]     = dl * (float)k_iq4nl[qs[j] & 0x0F];
                    yp[j + 16] = dl * (float)k_iq4nl[qs[j] >>   4];
                }
                yp += 32; qs += 16;
            }
        }
        return 0;
    }
    /* ── IQ (grid-based): codebook + segni ── */
    if (t == LLMCODEC_GGML_T_IQ2_XXS) {
        const blk_iq2_xxs* x = (const blk_iq2_xxs*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux[2]; memcpy(aux, (const uint8_t*)x[i].qs + 8*ib, 8);
                const uint8_t* a8 = (const uint8_t*)aux;
                const float db = d * (0.5f + (float)(aux[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + a8[l]);
                    const uint8_t sg = ksigns_iq2xs[(aux[1] >> (7*l)) & 127];
                    for (int j = 0; j < 8; ++j) yp[j] = db * (float)grid[j] * ((sg & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    yp += 8;
                }
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ2_XS) {
        const blk_iq2_xs* x = (const blk_iq2_xs*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ++ib) {
                const float db0 = d * (0.5f + (float)(x[i].scales[ib] & 0x0F)) * 0.25f;
                const float db1 = d * (0.5f + (float)(x[i].scales[ib] >>   4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint16_t v = x[i].qs[4*ib + l];
                    const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (v & 511));
                    const uint8_t sg = ksigns_iq2xs[v >> 9];
                    const float db = (l < 2) ? db0 : db1;
                    for (int j = 0; j < 8; ++j) yp[j] = db * (float)grid[j] * ((sg & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    yp += 8;
                }
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ2_S) {
        const blk_iq2_s* x = (const blk_iq2_s*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            const uint8_t* qs = x[i].qs; const uint8_t* qh = x[i].qh; const uint8_t* sg = x[i].qs + 32;
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ++ib) {
                const float db0 = d * (0.5f + (float)(x[i].scales[ib] & 0x0F)) * 0.25f;
                const float db1 = d * (0.5f + (float)(x[i].scales[ib] >>   4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | ((qh[ib] << (8 - 2*l)) & 0x300)));
                    const float dl = (l < 2) ? db0 : db1;
                    for (int j = 0; j < 8; ++j) yp[j] = dl * (float)grid[j] * ((sg[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    yp += 8;
                }
                qs += 4; sg += 4;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ3_XXS) {
        const blk_iq3_xxs* x = (const blk_iq3_xxs*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            const uint8_t* qs = x[i].qs; const uint8_t* ss = x[i].qs + 64;
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; memcpy(&aux, ss + 4*ib, 4);
                const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sg = ksigns_iq2xs[(aux >> (7*l)) & 127];
                    const uint8_t* g1 = (const uint8_t*)(iq3xxs_grid + qs[2*l + 0]);
                    const uint8_t* g2 = (const uint8_t*)(iq3xxs_grid + qs[2*l + 1]);
                    for (int j = 0; j < 4; ++j) {
                        yp[j]     = db * (float)g1[j] * ((sg & kmask_iq2xs[j])     ? -1.0f : 1.0f);
                        yp[j + 4] = db * (float)g2[j] * ((sg & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    }
                    yp += 8;
                }
                qs += 8;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ3_S) {
        const blk_iq3_s* x = (const blk_iq3_s*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            const uint8_t* qs = x[i].qs; const uint8_t* qh = x[i].qh; const uint8_t* sg = x[i].signs;
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ib += 2) {
                const float db1 = d * (float)(1 + 2*(x[i].scales[ib/2] & 0x0F));
                const float db2 = d * (float)(1 + 2*(x[i].scales[ib/2] >>   4));
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2*l + 0] | ((qh[0] << (8 - 2*l)) & 256)));
                    const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2*l + 1] | ((qh[0] << (7 - 2*l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        yp[j]     = db1 * (float)g1[j] * ((sg[l] & kmask_iq2xs[j])     ? -1.0f : 1.0f);
                        yp[j + 4] = db1 * (float)g2[j] * ((sg[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    }
                    yp += 8;
                }
                qs += 8; sg += 4;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2*l + 0] | ((qh[1] << (8 - 2*l)) & 256)));
                    const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2*l + 1] | ((qh[1] << (7 - 2*l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        yp[j]     = db2 * (float)g1[j] * ((sg[l] & kmask_iq2xs[j])     ? -1.0f : 1.0f);
                        yp[j + 4] = db2 * (float)g2[j] * ((sg[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    }
                    yp += 8;
                }
                qh += 2; qs += 8; sg += 4;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ1_S) {
        const blk_iq1_s* x = (const blk_iq1_s*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = q_f16_to_f32(x[i].d);
            const uint8_t* qs = x[i].qs; const uint16_t* qh = x[i].qh;
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ++ib) {
                const float dl = d * (float)(2*((qh[ib] >> 12) & 7) + 1);
                const float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA_V : IQ1S_DELTA_V;
                for (int l = 0; l < 4; ++l) {
                    const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> (3*l)) & 7) << 8)));
                    for (int j = 0; j < 8; ++j) yp[j] = dl * ((float)grid[j] + delta);
                    yp += 8;
                }
                qs += 4;
            }
        }
        return 0;
    }
    if (t == LLMCODEC_GGML_T_IQ1_M) {
        const blk_iq1_m* x = (const blk_iq1_m*)src;
        for (int64_t i = 0; i < nb; ++i) {
            const uint16_t* sc = (const uint16_t*)x[i].scales;
            const uint16_t su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
            const float d = q_f16_to_f32(su);
            const uint8_t* qs = x[i].qs; const uint8_t* qh = x[i].qh;
            float* yp = dst + i*256;
            for (int ib = 0; ib < 8; ++ib) {
                const float dl1 = d * (float)(2*((sc[ib/2] >> (6*(ib%2) + 0)) & 7) + 1);
                const float dl2 = d * (float)(2*((sc[ib/2] >> (6*(ib%2) + 3)) & 7) + 1);
                uint16_t idx[4];
                idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
                idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
                idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
                idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
                float dlt[4];
                dlt[0] = (qh[0] & 0x08) ? -IQ1S_DELTA_V : IQ1S_DELTA_V;
                dlt[1] = (qh[0] & 0x80) ? -IQ1S_DELTA_V : IQ1S_DELTA_V;
                dlt[2] = (qh[1] & 0x08) ? -IQ1S_DELTA_V : IQ1S_DELTA_V;
                dlt[3] = (qh[1] & 0x80) ? -IQ1S_DELTA_V : IQ1S_DELTA_V;
                for (int l = 0; l < 2; ++l) {
                    const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                    for (int j = 0; j < 8; ++j) yp[j] = dl1 * ((float)grid[j] + dlt[l]);
                    yp += 8;
                }
                for (int l = 2; l < 4; ++l) {
                    const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                    for (int j = 0; j < 8; ++j) yp[j] = dl2 * ((float)grid[j] + dlt[l]);
                    yp += 8;
                }
                qs += 4; qh += 2;
            }
        }
        return 0;
    }
    return -1;
}

/* ---- Variante IMATRIX dei K-quant (port fedele da ggml: quantize_row_q4_K_impl): la scala/min
 * per-sotto-blocco è pesata con `quant_weights` (imatrix). Una sola sorgente (§0bis): prima
 * viveva in llm_runtime/llmmodelio.c. ---- */

/* APACHE: encode Q8_K (attivazioni, actquant). L encode dei formati di MODELLO e PUL (gmbllm). */
int llmcodec_ggml_q8_K_encode(const float* src, void* dst, int64_t n) {
    if (src == NULL || dst == NULL || n <= 0 || (n % 256) != 0) return -1;
    const int64_t nb = n / 256;
    {
        blk_q8_K* y = (blk_q8_K*)dst;
        for (int64_t i = 0; i < nb; ++i) {
            const float* x = src + i*256;
            float mx = 0.0f, amax = 0.0f;
            for (int j = 0; j < 256; ++j) { float ax = fabsf(x[j]); if (ax > amax) { amax = ax; mx = x[j]; } }
            if (!amax) { y[i].d = 0.0f; memset(y[i].qs, 0, 256); memset(y[i].bsums, 0, sizeof y[i].bsums); continue; }
            const float iscale = -127.0f / mx;   /* -127 (non -128): compat IQ2_XXS */
            y[i].d = 1.0f / iscale;
            for (int j = 0; j < 16; ++j) {
                int sum = 0;
                for (int k = 0; k < 16; ++k) {
                    int q = llmcodec_nearest_int(x[16*j + k] * iscale);
                    if (q > 127) q = 127; else if (q < -128) q = -128;
                    y[i].qs[16*j + k] = (int8_t)q; sum += q;
                }
                y[i].bsums[j] = (int16_t)sum;
            }
        }
        return 0;
    }
}

/* ── helper PER-VALORE (esposti: anche l'engine DS4 li usa per il compute) ── */
void llmcodec_ggml_q4_K_scale_min(uint32_t j, const uint8_t* q, uint8_t* sc, uint8_t* m) {
    k4_scale_min((int)j, q, sc, m);
}
float llmcodec_ggml_q2_K_value(const uint8_t* xb, uint32_t idx) {
    uint32_t group = idx / 16u, l = idx - group * 16u;
    uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
    uint32_t shift = ((group / 2u) & 3u) * 2u;
    uint32_t q = ((uint32_t)xb[16 + q_base + l] >> shift) & 3u;
    uint32_t sc = xb[group];
    uint16_t dh, mh; memcpy(&dh, xb + 80, 2); memcpy(&mh, xb + 82, 2);
    return q_f16_to_f32(dh) * (float)(sc & 0x0fu) * (float)q - q_f16_to_f32(mh) * (float)(sc >> 4u);
}
float llmcodec_ggml_iq2_xxs_value(const uint8_t* blk, uint32_t idx) {
    uint16_t dh; memcpy(&dh, blk, 2); float d = q_f16_to_f32(dh);
    const uint16_t* q2 = (const uint16_t*)(blk + 2);
    uint32_t ib = idx / 32u, r = idx - ib * 32u, li = r / 8u, j = r - li * 8u;
    uint32_t aux32[2]; memcpy(aux32, q2 + 4 * ib, 8);
    const uint8_t* aux8 = (const uint8_t*)aux32;
    uint32_t ls = 2u * (aux32[1] >> 28) + 1u;
    uint32_t sidx = (aux32[1] >> (7 * li)) & 127u;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[li]);
    float v = (float)grid[j];
    if (ksigns_iq2xs[sidx] & kmask_iq2xs[j]) v = -v;
    return 0.125f * d * (float)ls * v;
}

/* Nome-fmt canonico di un tipo GGML (era DICHIARATO ma NON definito → gap latente). Stessi nomi
 * storici del runtime MoE (`q4_K`, `iq2_xxs`, …) per non cambiarne l'output. 0 ok / -1 ignoto. */
int llmcodec_ggml_type_name(int ggml_type, char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    const char* n = NULL;
    switch (ggml_type) {
        case LLMCODEC_GGML_T_F32:  n = "f32"; break;
        case LLMCODEC_GGML_T_F16:  n = "f16"; break;
        case LLMCODEC_GGML_T_BF16: n = "bf16"; break;
        case LLMCODEC_GGML_T_Q4_0: n = "q4_0"; break;
        case LLMCODEC_GGML_T_Q4_1: n = "q4_1"; break;
        case LLMCODEC_GGML_T_Q5_0: n = "q5_0"; break;
        case LLMCODEC_GGML_T_Q5_1: n = "q5_1"; break;
        case LLMCODEC_GGML_T_Q8_0: n = "q8_0"; break;
        case LLMCODEC_GGML_T_Q2_K: n = "q2_K"; break;
        case LLMCODEC_GGML_T_Q3_K: n = "q3_K"; break;
        case LLMCODEC_GGML_T_Q4_K: n = "q4_K"; break;
        case LLMCODEC_GGML_T_Q5_K: n = "q5_K"; break;
        case LLMCODEC_GGML_T_Q6_K: n = "q6_K"; break;
        case LLMCODEC_GGML_T_Q8_K: n = "q8_K"; break;
        case LLMCODEC_GGML_T_IQ2_XXS: n = "iq2_xxs"; break;
        case LLMCODEC_GGML_T_IQ2_XS:  n = "iq2_xs"; break;
        case LLMCODEC_GGML_T_IQ3_XXS: n = "iq3_xxs"; break;
        case LLMCODEC_GGML_T_IQ1_S:   n = "iq1_s"; break;
        case LLMCODEC_GGML_T_IQ4_NL:  n = "iq4_nl"; break;
        case LLMCODEC_GGML_T_IQ3_S:   n = "iq3_s"; break;
        case LLMCODEC_GGML_T_IQ2_S:   n = "iq2_s"; break;
        case LLMCODEC_GGML_T_IQ4_XS:  n = "iq4_xs"; break;
        case LLMCODEC_GGML_T_IQ1_M:   n = "iq1_m"; break;
        case LLMCODEC_GGML_T_TQ1_0:   n = "tq1_0"; break;
        case LLMCODEC_GGML_T_TQ2_0:   n = "tq2_0"; break;
        case LLMCODEC_GGML_T_MXFP4:   n = "mxfp4"; break;
        case LLMCODEC_GGML_T_NVFP4:   n = "nvfp4"; break;
        default: return -1;
    }
    snprintf(out, cap, "%s", n);
    return 0;
}

int llmcodec_ggml_type_of_fmt(const char* fmt) {
    if (fmt == NULL) return -1;
    char b[24]; size_t n = 0;
    for (; fmt[n] && n + 1 < sizeof b; ++n) { char c = fmt[n]; b[n] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c; }
    b[n] = 0;
    if (!strcmp(b,"f32")) return LLMCODEC_GGML_T_F32;
    if (!strcmp(b,"f16")) return LLMCODEC_GGML_T_F16;
    if (!strcmp(b,"bf16")) return LLMCODEC_GGML_T_BF16;
    if (!strcmp(b,"q4_0")) return LLMCODEC_GGML_T_Q4_0;
    if (!strcmp(b,"q4_1")) return LLMCODEC_GGML_T_Q4_1;
    if (!strcmp(b,"q5_0")) return LLMCODEC_GGML_T_Q5_0;
    if (!strcmp(b,"q5_1")) return LLMCODEC_GGML_T_Q5_1;
    if (!strcmp(b,"q8_0")) return LLMCODEC_GGML_T_Q8_0;
    if (!strcmp(b,"q2_k")) return LLMCODEC_GGML_T_Q2_K;
    if (!strcmp(b,"q3_k")) return LLMCODEC_GGML_T_Q3_K;
    if (!strcmp(b,"q4_k")) return LLMCODEC_GGML_T_Q4_K;
    if (!strcmp(b,"q5_k")) return LLMCODEC_GGML_T_Q5_K;
    if (!strcmp(b,"q6_k")) return LLMCODEC_GGML_T_Q6_K;
    if (!strcmp(b,"q8_k")) return LLMCODEC_GGML_T_Q8_K;
    if (!strcmp(b,"iq2_xxs")) return LLMCODEC_GGML_T_IQ2_XXS;
    if (!strcmp(b,"iq2_xs"))  return LLMCODEC_GGML_T_IQ2_XS;
    if (!strcmp(b,"iq3_xxs")) return LLMCODEC_GGML_T_IQ3_XXS;
    if (!strcmp(b,"iq1_s"))   return LLMCODEC_GGML_T_IQ1_S;
    if (!strcmp(b,"iq4_nl"))  return LLMCODEC_GGML_T_IQ4_NL;
    if (!strcmp(b,"iq3_s"))   return LLMCODEC_GGML_T_IQ3_S;
    if (!strcmp(b,"iq2_s"))   return LLMCODEC_GGML_T_IQ2_S;
    if (!strcmp(b,"iq4_xs"))  return LLMCODEC_GGML_T_IQ4_XS;
    if (!strcmp(b,"iq1_m"))   return LLMCODEC_GGML_T_IQ1_M;
    if (!strcmp(b,"tq1_0"))   return LLMCODEC_GGML_T_TQ1_0;
    if (!strcmp(b,"tq2_0"))   return LLMCODEC_GGML_T_TQ2_0;
    if (!strcmp(b,"mxfp4"))   return LLMCODEC_GGML_T_MXFP4;
    if (!strcmp(b,"nvfp4"))   return LLMCODEC_GGML_T_NVFP4;
    if (!strcmp(b,"q1_0"))    return LLMCODEC_GGML_T_Q1_0;
    if (!strcmp(b,"q2_0"))    return LLMCODEC_GGML_T_Q2_0;
    return -1;
}

/* Tabella byte-per-tipo: FORMATO del codec (vive nel plugin, non nel contenitore). */
static int llmcodec_ggml_block(int t, uint64_t* bb, uint64_t* bv) {
    switch (t) {
        case 0: *bb=4;   *bv=1;   return 0;  case 1: *bb=2;   *bv=1;   return 0;
        case 2: *bb=18;  *bv=32;  return 0;  case 3: *bb=20;  *bv=32;  return 0;
        /* 4 (Q4_2) e 5 (Q4_3) erano qui con {18,32} e {20,32}: 0,56 e 0,63 bit/valore, cioè
         * UN OTTAVO del tasso storico reale (4,55 e 5,03 bpv). Tipi ELIMINATI da ggml upstream
         * nel 2024 — `ds4/gguf-tools/quants.h:20-30` infatti salta da Q4_1=3 a Q5_0=6 — quindi
         * non c'è riferimento con cui validarli e nessun .gguf locale li contiene (header probe:
         * tipi {0,1,8,10,12,14,16,26,30}). Un numero non verificabile per un tipo inesistente è
         * peggio di un errore: rimossi dal `switch`, ricadono nel `default` → -1. */
        case 6: *bb=22;  *bv=32;  return 0;  case 7: *bb=24;  *bv=32;  return 0;
        case 8: *bb=34;  *bv=32;  return 0;  case 9: *bb=36;  *bv=32;  return 0;
        case 10: *bb=84;  *bv=256; return 0; case 11: *bb=110; *bv=256; return 0;
        case 12: *bb=144; *bv=256; return 0; case 13: *bb=176; *bv=256; return 0;
        case 14: *bb=210; *bv=256; return 0; case 15: *bb=292; *bv=256; return 0;
        case 16: *bb=66;  *bv=256; return 0; case 17: *bb=74;  *bv=256; return 0;
        case 18: *bb=98;  *bv=256; return 0;  /* IQ3_XXS: era 34 (-65%) */
        case 19: *bb=50;  *bv=256; return 0;  /* IQ1_S:   era 144 (+188%) */
        case 20: *bb=18;  *bv=32;  return 0;  /* IQ4_NL */
        case 21: *bb=110; *bv=256; return 0;  /* IQ3_S */
        case 22: *bb=82;  *bv=256; return 0;  /* IQ2_S */
        case 23: *bb=136; *bv=256; return 0;  /* IQ4_XS */
        case 24: *bb=1;   *bv=1;   return 0;  /* I8  (non dequantizzabili: sono dati interi) */
        case 25: *bb=2;   *bv=1;   return 0;  /* I16 */
        case 26: *bb=4;   *bv=1;   return 0;  /* I32 — presente in ds4flash.gguf */
        case 27: *bb=8;   *bv=1;   return 0;  /* I64 */
        case 28: *bb=8;   *bv=1;   return 0;  /* F64 */
        case 29: *bb=56;  *bv=256; return 0;  /* IQ1_M */
        case 30: *bb=2;   *bv=1;   return 0;  /* BF16: 2 byte / 1 valore */
        case 34: *bb=54;  *bv=256; return 0;  /* TQ1_0 (ternario) */
        case 35: *bb=66;  *bv=256; return 0;  /* TQ2_0 (ternario) */
        case 39: *bb=17;  *bv=32;  return 0;  /* MXFP4: {e:E8M0, qs[16]} / 32 */
        case 40: *bb=36;  *bv=64;  return 0;  /* NVFP4: {d[4]:UE4M3, qs[32]} / 64 (dim. indicativa) */
        case 41: *bb=18;  *bv=128; return 0;  /* Q1_0: {d:f16, qs[16]} / 128 (1-bit binario) */
        default: return -1;
    }
}
/* Ritorna 0 e scrive in `*out` i byte necessari per `numel` valori di `ggml_type`, oppure -1.
 * Il -1 copre DUE cause, ed è il motivo per cui questa funzione non ritorna un uint64_t: il
 * valore 0 era il numero di un tensore vuoto, quindi nessun chiamatore poteva distinguerlo dal
 * "tipo che non conosco" (era la firma del bug: pesi letti come 0 byte, senza messaggio).
 * L'altra causa è `numel` non multiplo del blocco: `(numel / bv) * bb` TRONCA, e un blocco
 * mancante diventava un tensore da 0 byte — nei file GGUF `dims[0]` è multiplo per definizione,
 * quindi un non-multiplo viene da uno slicing fatto a monte ed è un danno, non un arrotondamento. */
int llmcodec_ggml_type_bytes(int ggml_type, uint64_t numel, uint64_t* out) {
    uint64_t bb = 0, bv = 0;
    if (out == NULL) return -1;
    *out = 0;
    if (llmcodec_ggml_block(ggml_type, &bb, &bv) != 0 || bv == 0) return -1;
    if (numel % bv != 0) return -1;
    *out = (numel / bv) * bb;
    return 0;
}
