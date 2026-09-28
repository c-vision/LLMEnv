/* llmcodec_fp8_fp4.c — port dei dequant sorgente DS4.1 (engine: llmcodec_dsv41_loader.h + llmcodec_dsv41_fp8.h). */
#include "llmcodec_fp8_fp4.h"
#include <math.h>

/* PARAMETRICA (spostata da llm_runtime/src/llmcodec_fp8.c, una sola sorgente). */
void llmcodec_dsv41_dequant_fp8_block_sz(const uint8_t* w, const uint8_t* sc, int R, int C, int br, int bc, float* dst) {
    if (br <= 0) br = R; if (bc <= 0) bc = C;
    int SR = (R + br - 1) / br, SC = (C + bc - 1) / bc;
    for (int r = 0; r < R; ++r)
        for (int cb = 0, c = 0; c < C; cb++, c += bc) {
            float s = llmcodec_e8m0_to_f32(sc[(r / br) * SC + cb]);
            for (int k = 0; k < bc && c + k < C; ++k)
                dst[(size_t)r * C + c + k] = llmcodec_fp8_e4m3_to_f32(w[(size_t)r * C + c + k]) * s;
        }
}
void llmcodec_dsv41_dequant_fp4_block_sz(const uint8_t* w, const uint8_t* sc, int R, int C, int br, int bc, float* dst) {
    if (br <= 0) br = R; if (bc <= 0) bc = 32;
    int C2 = C / 2, SC = (C + bc - 1) / bc;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C2; ++c) {
            uint8_t b = w[(size_t)r * C2 + c];
            float s0 = llmcodec_e8m0_to_f32(sc[(r / br) * SC + (2 * c) / bc]);
            float s1 = llmcodec_e8m0_to_f32(sc[(r / br) * SC + (2 * c + 1) / bc]);
            dst[(size_t)r * C + 2 * c]     = llmcodec_fp4_e2m1_level(b & 0xF) * ((b & 0x8) ? -1.0f : 1.0f) * s0;
            dst[(size_t)r * C + 2 * c + 1] = llmcodec_fp4_e2m1_level(b >> 4)  * ((b & 0x80) ? -1.0f : 1.0f) * s1;
        }
}
/* wrapper storico [32,32]: delega alla parametrica. */
void llmcodec_dsv41_dequant_fp8_block(const uint8_t* w, const uint8_t* sc, int out, int in, float* dst) {
    llmcodec_dsv41_dequant_fp8_block_sz(w, sc, out, in, 32, 32, dst);
}

void llmcodec_dsv41_dequant_fp4_expert(const uint8_t* w, const uint8_t* sc, int out, int in, float* dst) {
    int in2 = in / 2, scn = in / 32;
    for (int r = 0; r < out; ++r)
        for (int b = 0; b < in2; ++b) {
            uint8_t byte = w[(size_t)r * in2 + b];
            for (int j = 0; j < 2; ++j) {
                int code = j ? (byte >> 4) : (byte & 15);
                int c = b * 2 + j;
                float lvl = llmcodec_fp4_e2m1_level(code & 7);
                float v = (code & 8) ? -lvl : lvl;
                dst[(size_t)r * in + c] = v * llmcodec_e8m0_to_f32(sc[(size_t)r * scn + c / 32]);
            }
        }
}

/* fp8 e4m3 row-block (tabella engram): scn = cols/32, una scala per 32 colonne di ogni riga. */
void llmcodec_dsv41_dequant_fp8_row(const uint8_t* w, const uint8_t* sc, int rows, int cols, float* dst) {
    int scn = cols / 32;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* wr = w + (size_t)r * cols; const uint8_t* sr = sc + (size_t)r * scn;
        float* dr = dst + (size_t)r * cols;
        int c = 0;
        for (int b = 0; b < scn; ++b) {           /* scala calcolata UNA volta per blocco da 32 */
            float s = llmcodec_e8m0_to_f32(sr[b]);
            for (int k = 0; k < 32; ++k, ++c) dr[c] = llmcodec_fp8_e4m3_to_f32(wr[c]) * s;
        }
    }
}

/* Scala ue8m0 potenza-di-due per un blocco di ampiezza `amax` (limite 448). */
static uint8_t llmcodec_dsv41_e8m0_for_amax(float amax, float limit) {
    float minv = limit * ldexpf(1.0f, -126);
    if (amax < minv) amax = minv;
    float ex = ceilf(log2f(amax / limit));
    if (ex < -126.0f) ex = -126.0f;
    return (uint8_t)((int)ex + 127);
}

void llmcodec_dsv41_quant_fp8_block(const float* f, uint8_t* w, uint8_t* sc, int out, int in) {
    int scn = (in + 31) / 32;
    for (int br = 0; br < out; br += 32)
        for (int bc = 0; bc < in; bc += 32) {
            float amax = 0.0f;
            for (int r = br; r < br + 32 && r < out; ++r)
                for (int c = bc; c < bc + 32 && c < in; ++c) { float a = fabsf(f[(size_t)r*in+c]); if (a > amax) amax = a; }
            uint8_t sb = llmcodec_dsv41_e8m0_for_amax(amax, 448.0f);
            sc[(size_t)(br/32)*scn + bc/32] = sb;
            float scale = llmcodec_e8m0_to_f32(sb);
            for (int r = br; r < br + 32 && r < out; ++r)
                for (int c = bc; c < bc + 32 && c < in; ++c) {
                    float s = f[(size_t)r*in+c] / scale;
                    if (s > 448.0f) s = 448.0f; else if (s < -448.0f) s = -448.0f;
                    w[(size_t)r*in+c] = llmcodec_fp8_to_e4m3_byte(llmcodec_fp8_round_e4m3(s));
                }
        }
}

void llmcodec_dsv41_quant_fp8_row(const float* f, uint8_t* w, uint8_t* sc, int rows, int cols) {
    int scn = cols / 32;
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < scn; ++g) {
            const float* grp = f + (size_t)r*cols + g*32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) { float a = fabsf(grp[i]); if (a > amax) amax = a; }
            uint8_t sb = llmcodec_dsv41_e8m0_for_amax(amax, 448.0f);
            sc[(size_t)r*scn + g] = sb;
            float scale = llmcodec_e8m0_to_f32(sb);
            for (int i = 0; i < 32; ++i) {
                float s = grp[i] / scale;
                if (s > 448.0f) s = 448.0f; else if (s < -448.0f) s = -448.0f;
                w[(size_t)r*cols + g*32 + i] = llmcodec_fp8_to_e4m3_byte(llmcodec_fp8_round_e4m3(s));
            }
        }
}

void llmcodec_fp4_dequant(const uint8_t* src, int64_t n_packed, float scale, float* dst) {
    for (int64_t i = 0; i < n_packed; ++i) {
        float lo = llmcodec_fp4_e2m1_level(src[i] & 0xF); if (src[i] & 0x8) lo = -lo;
        float hi = llmcodec_fp4_e2m1_level(src[i] >> 4);  if (src[i] & 0x80) hi = -hi;
        dst[2*i]   = lo * scale;
        dst[2*i+1] = hi * scale;
    }
}
void llmcodec_fp8_e4m3_blocked(const uint8_t* w, int R, int C, const uint8_t* sc, int SR, int SC, float* dst) {
    int br = SR > 0 ? R/SR : R; int bc = SC > 0 ? C/SC : C;
    llmcodec_dsv41_dequant_fp8_block_sz(w, sc, R, C, br, bc, dst);
}
void llmcodec_fp4_blocked(const uint8_t* w, int R, int C, const uint8_t* sc, int SR, int SC, float* dst) {
    int br = SR > 0 ? R/SR : R; int bc = SC > 0 ? C/SC : C;
    llmcodec_dsv41_dequant_fp4_block_sz(w, sc, R, C, br, bc, dst);
}
/* NVFP4: peso e2m1 packed 8-nibble/U32, scala E4M3 (u8) per gruppo di `g` (su colonne). */
void llmcodec_dsv41_dequant_nvfp4_g(const uint8_t* w, const uint8_t* sc, int R, int C, int g, float global, float* dst) {
    llmcodec_dsv41_dequant_nvfp4(w, sc, R, C, g, dst);
    if (global != 1.0f) for (size_t i = 0; i < (size_t)R*C; ++i) dst[i] *= global;
}
void llmcodec_dsv41_dequant_nvfp4(const uint8_t* w, const uint8_t* sc, int R, int C, int g, float* dst) {
    if (g <= 0) g = 16;
    uint8_t* v = (uint8_t*)w;              /* w come byte (2 nibble/byte) */
    int Cy = (C + 1) / 2;                  /* byte per riga? no: nibble 2/byte -> C/2 byte */
    Cy = C / 2;
    int scn = (C + g - 1) / g;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            uint8_t b = v[(size_t)r*Cy + (size_t)(c/2)];
            int code = (c & 1) ? (b >> 4) : (b & 0x0F);
            float lv = llmcodec_fp4_e2m1_level(code & 7); if (code & 8) lv = -lv;
            float s = llmcodec_fp8_e4m3_scale_to_f32(sc[(size_t)r*scn + c/g]);
            dst[(size_t)r*C + c] = lv * s;
        }
}
