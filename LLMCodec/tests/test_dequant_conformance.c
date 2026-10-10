/* test_dequant_conformance.c — conformance INDIPENDENTE del dequant GGML.
 *
 * Confronta `llmcodec_ggml_dequant_row` (GMBasic/LLMCodec) con un SECONDO implementatore
 * indipendente dei bit-layout GGUF: le funzioni `gguf_q*_to_float` di antirez/gguf-tools
 * (`gguflib.c`, licenza MIT — © Salvatore Sanfilippo), qui trascritte VERBATIM come riferimento.
 * Su blob pseudo-casuali deterministici i due devono produrre valori IDENTICI (stesso formato
 * pubblico): una divergenza = errore di layout/decodifica in una delle due.
 *
 * Copre: Q8_0, Q4_K, Q6_K, Q2_K, Q4_0, Q4_1 (le famiglie che gguflib implementa).
 * Uscita 0 = tutte identiche; 1 = almeno una divergenza.
 */
#include "llmcodec_ggml_quants.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================= riferimento INDIPENDENTE (port di antirez/gguf-tools, MIT) ================ */

/* from_half di gguflib (fp16.c), IEEE: bit-exact, nessun rounding. */
static float ref_from_half(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { int e = -1; do { e++; man <<= 1; } while (!(man & 0x400)); man &= 0x3ff;
               f = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13); }
    } else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
    else f = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    float o; memcpy(&o, &f, 4); return o;
}

static void ref_q8_0(const void* wd, float* f, uint64_t count) {
    int8_t* block = (int8_t*)wd; uint64_t i = 0;
    while (i < count) {
        float scale = ref_from_half(*((uint16_t*)block));
        for (uint32_t j = 0; j < 32; j++) {
            float weight = block[j + 2] * scale;
            f[i] = weight; if (++i == count) break;
        }
        block += 34;
    }
}

static void ref_q4_0(const void* wd, float* f, uint64_t count) {
    uint8_t* block = (uint8_t*)wd; uint64_t i = 0;
    while (i < count) {
        float scale = ref_from_half(*((uint16_t*)block));
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j + 2] & 0xf; f[i] = ((int8_t)value - 8) * scale; if (++i == count) break;
        }
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j + 2] >> 4; f[i] = ((int8_t)value - 8) * scale; if (++i == count) break;
        }
        block += 18;
    }
}

static void ref_q4_1(const void* wd, float* f, uint64_t count) {
    uint8_t* block = (uint8_t*)wd; uint64_t i = 0;
    while (i < count) {
        float scale = ref_from_half(*((uint16_t*)block));
        float bias = ref_from_half(*((uint16_t*)block + 1));
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j + 4] & 0xf; f[i] = value * scale + bias; if (++i == count) break;
        }
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j + 4] >> 4; f[i] = value * scale + bias; if (++i == count) break;
        }
        block += 20;
    }
}

static void ref_q2_k(const void* wd, float* f, uint64_t count) {
    uint8_t* block = (uint8_t*)wd; uint64_t i = 0;
    while (i < count) {
        float scale_of_scales = ref_from_half(*((uint16_t*)(block + 16 + 64)));
        float scale_of_mins = ref_from_half(*((uint16_t*)(block + 16 + 64 + 2)));
        float scale = 0, min = 0; int bn = 0;
        for (uint64_t cluster = 0; cluster < 2; cluster++) {
            for (uint64_t j = 0; j < 128; j++) {
                if (j % 16 == 0) { scale = scale_of_scales * (block[bn] & 0xf); min = scale_of_mins * (block[bn] >> 4); bn++; }
                uint8_t q = (block[16 + j % 32 + cluster * 32] >> (j / 32 * 2)) & 3;
                f[i] = q * scale - min; if (++i == count) return;
            }
        }
        block += 16 + 64 + 4;
    }
}

static void ref_q4_k(const void* wd, float* f, uint64_t count) {
    uint8_t* block = (uint8_t*)wd; uint64_t i = 0;
    while (i < count) {
        float scales_scale = ref_from_half(*((uint16_t*)block));
        float mins_scale = ref_from_half(*((uint16_t*)(block + 2)));
        block += 4;
        float scales[8], mins[8];
        for (int j = 0; j < 8; j++) {
            uint8_t d, m;
            if (j < 4) { d = block[j] & 63; m = block[j + 4] & 63; }
            else { d = (block[j + 4] & 0xF) | ((block[j - 4] >> 6) << 4); m = (block[j + 4] >> 4) | ((block[j - 0] >> 6) << 4); }
            scales[j] = d * scales_scale; mins[j] = m * mins_scale;
        }
        block += 12;
        /* NOTA (bug gguflib): la versione di antirez usa scales[b]/mins[b] per ENTRAMBI i nibble
         * della coppia (b,b+1). Da spec ggml i due nibble sono DUE sub-blocchi distinti con scale
         * diverse: nibble basso -> scales[b], nibble alto -> scales[b+1] (ggml get_scale_min_k4
         * chiamato con +0 e +1). Qui si usa la forma CORRETTA (come llmcodec_ggml_dequant_row):
         * la "divergenza" trovata su q4_k era il bug del riferimento, non di GMBasic. */
        for (uint32_t b = 0; b < 8; b += 2) {
            for (uint32_t half = 0; half < 2; half++) {
                float scale = scales[b + half], min = mins[b + half];
                for (uint32_t j = 0; j < 32; j++) {
                    uint8_t w = (half == 0) ? (block[j] & 0xf) : (block[j] >> 4);
                    f[i] = w * scale - min; if (++i == count) return;
                }
            }
            block += 32;
        }
    }
}

static void ref_q6_k(const void* wd, float* f, uint64_t count) {
    uint8_t* block = (uint8_t*)wd; uint64_t i = 0;
    while (i < count) {
        float super_scale = ref_from_half(*((uint16_t*)(block + 128 + 64 + 16)));
        uint8_t* L = block; uint8_t* H = block + 128; int8_t* scales = (int8_t*)block + 128 + 64;
        for (int cluster = 0; cluster < 2; cluster++) {
            for (uint64_t j = 0; j < 128; j++) {
                float weight = (super_scale * scales[j / 16]) *
                    ((int8_t)((((L[j % 64] >> (j / 64 * 4)) & 0xF) | (((H[j % 32] >> (j / 32 * 2)) & 3) << 4))) - 32);
                f[i] = weight; if (++i == count) return;
            }
            L += 64; H += 32; scales += 8;
        }
        block += 128 + 64 + 16 + 2;
    }
}

/* ============================ generatore deterministico  ==================================== */
static uint32_t g_lcg;
static uint32_t lcg(void) { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg; }

/* ============== confronto: riferimento vs llmcodec_ggml_dequant_row ========================= */
static int check(const char* name, int type, uint64_t items_per_blk, uint64_t bytes_per_blk,
                 void (*ref)(const void*, float*, uint64_t))
{
    const uint64_t elems = items_per_blk * 8;   /* 8 blocchi */
    const size_t nblk = (size_t)(elems / items_per_blk);
    const size_t bytes = nblk * (size_t)bytes_per_blk;
    uint8_t* raw = (uint8_t*)malloc(bytes);
    float* r = (float*)malloc(elems * sizeof(float));
    float* g = (float*)malloc(elems * sizeof(float));
    if (!raw || !r || !g) { printf("[dequant] %-6s ALLOC FAIL\n", name); free(raw); free(r); free(g); return 1; }

    g_lcg = 0xC0FFEEu ^ (uint32_t)type;
    for (size_t i = 0; i < bytes; i++) raw[i] = (uint8_t)(lcg() >> 13);

    ref(raw, r, elems);
    int rc = llmcodec_ggml_dequant_row(type, raw, g, (int64_t)elems);
    int bad = 0; uint64_t firstbad = 0; double maxd = 0.0;
    if (rc != 0) bad = 1;
    else for (uint64_t k = 0; k < elems; k++) {
        double a = r[k], b = g[k];
        int eq = (a == b) || (isnan(a) && isnan(b)) || (isinf(a) && isinf(b) && ((a > 0) == (b > 0)));
        if (!eq && !bad) { bad = 1; firstbad = k; }
        if (isfinite(a) && isfinite(b)) { double d = fabs(a - b); if (d > maxd) maxd = d; }
    }
    printf("[dequant] %-6s n=%llu bytes=%zu : %s%s\n", name,
           (unsigned long long)elems, bytes,
           bad ? "MISMATCH" : "IDENTICO",
           bad ? "" : "");
    if (bad) printf("           ref[%llu]=%.9g gmb=%.9g maxd=%.3g rc=%d\n",
                    (unsigned long long)firstbad, (double)r[firstbad], (double)g[firstbad], maxd, rc);
    free(raw); free(r); free(g);
    return bad;
}

int main(void) {
    int bad = 0;
    printf("=== conformance dequant GGML: llmcodec vs gguflib (antirez/gguf-tools, MIT) ===\n");
    bad |= check("q8_0", LLMCODEC_GGML_T_Q8_0, 32, 34, ref_q8_0);
    bad |= check("q4_0", LLMCODEC_GGML_T_Q4_0, 32, 18, ref_q4_0);
    bad |= check("q4_1", LLMCODEC_GGML_T_Q4_1, 32, 20, ref_q4_1);
    bad |= check("q2_k", LLMCODEC_GGML_T_Q2_K, 256, 84, ref_q2_k);
    bad |= check("q4_k", LLMCODEC_GGML_T_Q4_K, 256, 144, ref_q4_k);
    bad |= check("q6_k", LLMCODEC_GGML_T_Q6_K, 256, 210, ref_q6_k);
    printf("=== %s ===\n", bad ? "DIVERGENZE" : "TUTTO IDENTICO");
    return bad ? 1 : 0;
}
