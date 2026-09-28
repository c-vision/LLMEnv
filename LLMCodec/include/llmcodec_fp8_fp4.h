/* llmcodec_fp8_fp4.h — dequant/encode dei formati SORGENTE DeepSeek V4.1:
 *   - pesi matmul fp8 e4m3 [out,in] + scala ue8m0 [ceil(out/32), ceil(in/32)]  (block [32,32])
 *   - esperti routed fp4 e2m1 packed [out, in/2] + scala ue8m0 [out, in/32]
 *   - tabella engram fp8 e4m3 [rows,cols] + ue8m0 [rows,cols/32] (row-block)
 *
 * ATOMI piccoli = header-only static inline (UNA SOLA SORGENTE, §0bis): inclusi e INLINATI da
 * codec, motore e runtime → nessun simbolo, nessuna chiamata out-of-line. Le routine di BLOCCO
 * (loop pesanti) restano extern nel .c.
 */
#ifndef LLMCODEC_FP8_FP4_H
#define LLMCODEC_FP8_FP4_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ---- ATOMI di FORMATO (static inline) ---- */

/* E4M3FN byte -> f32 (0x80/0x00 = +-0). */
static inline float llmcodec_fp8_e4m3_to_f32(uint8_t b) {
    int sign = (b >> 7) & 1, exp_byte = (b >> 3) & 15, mant = b & 7;
    if (exp_byte == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float v = (exp_byte == 0) ? ((float)mant / 8.0f) * ldexpf(1.0f, -6)
                              : (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp_byte - 7);
    return sign ? -v : v;
}
/* ue8m0 (esponente puro): byte b -> 2^(b-127). */
static inline float llmcodec_e8m0_to_f32(uint8_t b) {
    /* 2^(b-127) via bit-trick (molto piu' veloce di ldexpf: chiamata per-elemento nei dequant). */
    if (b == 0) return ldexpf(1.0f, -127);
    uint32_t u = (uint32_t)b << 23; float f; memcpy(&f, &u, 4); return f;
}
/* E5M2 (OCP): 1 segno + 5 esponente (bias 15) + 2 mantissa. */
static inline float llmcodec_fp8_e5m2_to_f32(uint8_t b) {
    int s = (b >> 7) & 1, e = (b >> 2) & 0x1F, m = b & 0x3;
    float v;
    if (e == 0) { v = ldexpf((float)m, -16); }
    else if (e == 31) { v = (m == 0) ? (float)INFINITY : (float)NAN; }
    else { v = ldexpf(1.0f + (float)m/4.0f, e - 15); }
    return s ? -v : v;
}
/* E4M3 (per le scale NVFP4). */
static inline float llmcodec_fp8_e4m3_scale_to_f32(uint8_t b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    float v;
    if (e == 0) v = ldexpf((float)m, -9);
    else if (e == 15) v = (m == 7) ? 0.0f : ldexpf(1.0f + (float)m/8.0f, 8);
    else v = ldexpf(1.0f + (float)m/8.0f, e - 7);
    return s ? -v : v;
}
/* E2M1: livello del code 0..7 {0,.5,1,1.5,2,3,4,6}. */
static inline float llmcodec_fp4_e2m1_level(int code) {
    static const float t[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    return t[code & 7];
}
/* E2M1 encode (a=|valore normalizzato| -> code 0..7), tie-to-even (pack_activation). */
static inline int llmcodec_fp4_e2m1_code(float a) {
    if (a > 5.0f) return 7;
    if (a >= 3.5f) return 6;
    if (a > 2.5f) return 5;
    if (a >= 1.75f) return 4;
    if (a > 1.25f) return 3;
    if (a >= 0.75f) return 2;
    if (a > 0.25f) return 1;
    return 0;
}
/* E2M1 round-to-grid (firma signed): x -> livello piu' vicino (tie-to-even). */
static inline float llmcodec_fp4_e2m1_round(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0; float bd = fabsf(ax - llmcodec_fp4_e2m1_level(0));
    for (int i = 1; i < 8; i++) {
        float d = fabsf(ax - llmcodec_fp4_e2m1_level(i));
        if (d < bd || (d == bd && (i & 1) == 0 && (best & 1) != 0)) { best = i; bd = d; }
    }
    return sign * llmcodec_fp4_e2m1_level(best);
}
/* E4M3FN round-to-nearest-even con saturazione a 448 (round_fp8). */
static inline float llmcodec_fp8_round_e4m3(float x) {
    float a = fabsf(x);
    if (a > 448.0f) a = 448.0f;
    float am = a < 1.953125e-3f ? 1.953125e-3f : a;
    int e = (int)floorf(log2f(am));
    int ste = e - 3; if (ste < -9) ste = -9;
    float step = ldexpf(1.0f, ste);
    float q = rintf(a / step) * step;
    if (q > 448.0f) q = 448.0f;
    return x < 0.0f ? -q : q;
}
/* Codifica un float gia' sulla griglia e4m3fn nel byte. */
static inline uint8_t llmcodec_fp8_to_e4m3_byte(float v) {
    int sign = (v < 0.0f) ? 1 : 0;
    float a = fabsf(v);
    if (a == 0.0f) return (uint8_t)(sign << 7);
    int e; float f = frexpf(a, &e);
    int exp_byte, mant, E = e - 1;
    if (E < -6) {
        mant = (int)lroundf(a / ldexpf(1.0f, -9)); exp_byte = 0;
        if (mant > 7) { exp_byte = 1; mant = 0; }
    } else {
        mant = (int)lroundf((2.0f * f - 1.0f) * 8.0f); exp_byte = E + 7;
        if (mant == 8) { mant = 0; exp_byte++; }
        if (exp_byte > 15) { exp_byte = 15; mant = 6; }
    }
    return (uint8_t)((sign << 7) | ((exp_byte & 15) << 3) | (mant & 7));
}
static inline float llmcodec_fp4_nibble_to_f32(uint8_t nibble) {
    float l = llmcodec_fp4_e2m1_level(nibble & 0xF);
    return (nibble & 0x8) ? -l : l;
}

/* ---- Alias coi NOMI storici del runtime (static inline: una sola sorgente qui). ---- */
static inline float llmcodec_f8_e4m3_to_f32(uint8_t x) { return llmcodec_fp8_e4m3_to_f32(x); }
static inline float llmcodec_f8_e8m0_to_f32(uint8_t e) { return llmcodec_e8m0_to_f32(e); }

/* ---- PACK/UNPACK di ATTIVAZIONI (static inline): layout di formato, non calcolo ----
 * `pack_activation`: float[width] -> [width*bits/8 valori][width/group_size scale].
 * Scala = potenza-di-2 normale (ue8m0) oppure E4M3 (e4m3_scale). E' l'INVERSA di unpack. */
static inline void llmcodec_pack_activation(const float* x, int width, int bits,
                                       int group_size, int e4m3_scale, uint8_t* out) {
    if (width <= 0 || group_size <= 0 || width % group_size != 0) return;
    if (bits != 4 && bits != 8) return;
    int G = width / group_size;
    int nbytes_val = width * bits / 8;
    float limit = (bits == 8) ? 448.0f : 6.0f;
    float minimum = limit * (e4m3_scale ? ldexpf(1.0f, -9) : ldexpf(1.0f, -126));
    for (int g = 0; g < G; g++) {
        const float* grp = x + (size_t)g * group_size;
        float amax = 0.0f;
        for (int i = 0; i < group_size; i++) { float a = fabsf(grp[i]); if (a > amax) amax = a; }
        if (amax < minimum) amax = minimum;
        float scale;
        if (e4m3_scale) {
            scale = llmcodec_fp8_round_e4m3(amax / limit);
            out[nbytes_val + g] = llmcodec_fp8_to_e4m3_byte(scale);
        } else {
            float ex = ceilf(log2f(amax / limit));
            if (ex < -126.0f) ex = -126.0f;
            int ei = (int)ex;
            scale = ldexpf(1.0f, ei);
            out[nbytes_val + g] = (uint8_t)(ei + 127);
        }
        for (int i = 0; i < group_size; i++) {
            float s = grp[i] / scale;
            if (s > limit) s = limit; else if (s < -limit) s = -limit;
            int idx = g * group_size + i;
            if (bits == 8) {
                out[idx] = llmcodec_fp8_to_e4m3_byte(llmcodec_fp8_round_e4m3(s));
            } else {
                int code = llmcodec_fp4_e2m1_code(fabsf(s));
                if (s < 0.0f) code |= 8;
                uint8_t* byte = &out[idx / 2];
                if ((i & 1) == 0) *byte = (uint8_t)((*byte & 0xF0) | (code & 15));
                else *byte = (uint8_t)((*byte & 0x0F) | ((code & 15) << 4));
            }
        }
    }
}
static inline void llmcodec_unpack_activation(const uint8_t* packed, int width, int bits,
                                         int group_size, int e4m3_scale, float* out) {
    if (width <= 0 || group_size <= 0 || width % group_size != 0) return;
    if (bits != 4 && bits != 8) return;
    int G = width / group_size;
    int nbytes_val = width * bits / 8;
    for (int g = 0; g < G; g++) {
        float scale = e4m3_scale ? llmcodec_fp8_e4m3_to_f32(packed[nbytes_val + g])
                                 : llmcodec_e8m0_to_f32(packed[nbytes_val + g]);
        for (int i = 0; i < group_size; i++) {
            int idx = g * group_size + i;
            float v;
            if (bits == 8) {
                v = llmcodec_fp8_e4m3_to_f32(packed[idx]);
            } else {
                uint8_t byte = packed[idx / 2];
                int code = (i & 1) ? (byte >> 4) : (byte & 15);
                float lvl = llmcodec_fp4_e2m1_level(code & 7);
                v = (code & 8) ? -lvl : lvl;
            }
            out[idx] = v * scale;
        }
    }
}
/* Specializzazione fp8/e4m3, group 32, scala ue8m0 (bits=8, group_size=32, e4m3_scale=0). */
static inline void llmcodec_pack_fp8(const float* x, int width, uint8_t* out) {
    llmcodec_pack_activation(x, width, 8, 32, 0, out);
}
static inline void llmcodec_unpack_fp8(const uint8_t* packed, int width, float* out) {
    llmcodec_unpack_activation(packed, width, 8, 32, 0, out);
}

/* ---- ROUTINE DI BLOCCO (extern, loop pesanti nel .c) ---- */

/* fp8 e4m3 [out,in] + ue8m0 [ceil(out/32),ceil(in/32)] -> f32 [out,in]. */
void llmcodec_dsv41_dequant_fp8_block(const uint8_t* w, const uint8_t* sc, int out, int in, float* dst);
/* PARAMETRICA: blocco di `br`x`bc` (0 => dim intera); scala dims [ceil(R/br), ceil(C/bc)]. */
void llmcodec_dsv41_dequant_fp8_block_sz(const uint8_t* w, const uint8_t* sc, int R, int C, int br, int bc, float* dst);
void llmcodec_dsv41_dequant_fp4_block_sz(const uint8_t* w, const uint8_t* sc, int R, int C, int br, int bc, float* dst);
/* NVFP4: e2m1 (8 nibble/U32) + scala E4M3 (u8) per gruppo di `g` (tip. 16). */
void llmcodec_dsv41_dequant_nvfp4(const uint8_t* w, const uint8_t* sc, int R, int C, int g, float* dst);
/* NVFP4 OCP/NVIDIA con GLOBAL f32 per-tensore (2o livello): value = e2m1*e4m3(blocco)*global. */
void llmcodec_dsv41_dequant_nvfp4_g(const uint8_t* w, const uint8_t* sc, int R, int C, int g, float global, float* dst);
/* fp4 e2m1 packed [out,in/2] + ue8m0 [out,in/32] -> f32 [out,in]. */
void llmcodec_dsv41_dequant_fp4_expert(const uint8_t* w, const uint8_t* sc, int out, int in, float* dst);
/* fp8 e4m3 [rows,cols] + ue8m0 [rows,cols/32] (row-block, tabella ENGRAM) -> f32. */
void llmcodec_dsv41_dequant_fp8_row(const uint8_t* w, const uint8_t* sc, int rows, int cols, float* dst);

/* INVERSE (encoder) del dequant block-scaled DS4.1. */
void llmcodec_dsv41_quant_fp8_block(const float* f, uint8_t* w, uint8_t* sc, int out, int in);
void llmcodec_dsv41_quant_fp8_row(const float* f, uint8_t* w, uint8_t* sc, int rows, int cols);

/* Alias storici (routine di blocco). */
void llmcodec_fp4_dequant(const uint8_t* src, int64_t n_packed, float scale, float* dst);
void llmcodec_fp8_e4m3_blocked(const uint8_t* w, int R, int C, const uint8_t* sc, int SR, int SC, float* dst);
void llmcodec_fp4_blocked(const uint8_t* w, int R, int C, const uint8_t* sc, int SR, int SC, float* dst);

#endif
