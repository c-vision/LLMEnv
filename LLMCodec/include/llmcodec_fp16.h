/* llmcodec_fp16.h — atomi di FORMATO fp16/bf16/ue4m3 — HEADER-ONLY (§0bis: UNA SOLA SORGENTE).
 * Prima erano reimplementati in llm_runtime/llmmodelio.c, nel plugin ggml, nel converter e
 * nell'engine; ora tutti includono questo header (static inline → nessun simbolo/link). */
#ifndef LLMCODEC_FP16_H
#define LLMCODEC_FP16_H
#include <stdint.h>
#include <string.h>
#include <math.h>

/* f32 -> f16 IEEE 754 (subnormali + round-to-nearest-even, inf/NaN preservati). */
static inline uint16_t llmcodec_f16_from_f32(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xff) - 127;
    uint32_t m = x & 0x7fffffu;
    if ((x & 0x7fffffffu) >= 0x7f800000u)          /* inf / NaN */
        return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e < -14) {                                 /* subnormale o zero */
        if (e < -24) return (uint16_t)sign;
        m |= 0x800000u;
        int shift = -e - 1;
        uint32_t s = m >> shift, rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (s & 1u))) s++;
        return (uint16_t)(sign | s);
    }
    {                                              /* normale */
        int32_t ex = e + 15;
        uint32_t s = m >> 13, rem = m & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (s & 1u))) { s++; if (s == 0x400u) { s = 0; ex++; } }
        if (ex >= 31) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | ((uint32_t)ex << 10) | s);
    }
}
/* f16 -> f32. */
static inline float llmcodec_f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, u;
    if (e == 0) { if (m == 0) u = s; else { int sh = 0; while (!(m & 0x400u)) { m <<= 1; sh++; } m &= 0x3FFu; u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13); } }
    else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
/* bf16 <-> f32. */
static inline float llmcodec_bf16_to_f32(uint16_t h) { uint32_t u = ((uint32_t)h) << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t llmcodec_bf16_from_f32(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    if ((x & 0x7F800000u) == 0x7F800000u)                  /* Inf/NaN: preserva */
        return (uint16_t)((x >> 16) | ((x & 0xFFFFu) ? 1u : 0u));
    uint32_t lsb = (x >> 16) & 1u, bias = 0x7fffu + lsb;   /* round-to-nearest-even */
    x += bias;
    return (uint16_t)(x >> 16);
}
/* ue4m3 (E4M3FN senza segno) encode/decode — NVFP4 scale e varianti. */
static inline float llmcodec_ue4m3_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF, man = x & 0x7;
    return exp == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
}
static inline uint8_t llmcodec_ue4m3_from_f32(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits; memcpy(&bits, &x, 4);
    int fp32_exp = (int)(((bits >> 23) & 0xFF) - 127);
    int fp32_man = (int)((bits >> 20) & 0x7);
    int ue_exp = fp32_exp + 7;
    if (ue_exp <= 0) { int man = (int)(x * 512.0f + 0.5f); if (man > 7) man = 7; if (man < 1) return 0; return (uint8_t)man; }
    if (ue_exp >= 15) return 0x7E;
    int round_bit = (int)((bits >> 19) & 1);
    int ue_man = fp32_man + round_bit;
    if (ue_man > 7) { ue_man = 0; ue_exp++; if (ue_exp >= 15) return 0x7E; }
    return (uint8_t)((ue_exp << 3) | ue_man);
}

#endif
