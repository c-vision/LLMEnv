/* test_moe_quant.c — quantizzatore affine RTN: errore di round-trip per bits + dequant di verifica.
 * Verifica anche l'anti-simmetria col dequant esistente (formula q*scale+bias, scales F16). */
#include "llmcodec_moe_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static float rnd(unsigned* s) { *s = *s * 1103515245u + 12345u; return (float)((*s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

int main(void) {
#define R 8u
#define C 128u
#define G 64u
    float x[R * C];
    unsigned s = 99;
    for (uint32_t i = 0; i < R * C; ++i) x[i] = rnd(&s);   /* ~[-1,1] */
    const float range = 2.0f;

    uint32_t* w = (uint32_t*)malloc((size_t)R * (C / 4) * 4);      /* bits>=4 packed */
    uint16_t* sc = (uint16_t*)malloc((size_t)R * (C / G) * 2);
    uint16_t* bi = (uint16_t*)malloc((size_t)R * (C / G) * 2);
    float* out = (float*)malloc((size_t)R * C * sizeof(float));
    CHECK(w && sc && bi && out);

    for (uint32_t bits = 1; bits <= 8; ++bits) {
        memset(w, 0, (size_t)R * (C * bits / 32u) * 4u);
        CHECK(llmcodec_moe_affine_quantize(x, R, C, G, bits, w, sc, bi) == 0);
        llmcodec_moe_affine_dequantize(w, sc, bi, R, C, G, bits, /*sc_f16=*/1, out);
        double md = 0;
        for (uint32_t i = 0; i < R * C; ++i) md = fmax(md, fabs((double)out[i] - (double)x[i]));
        double rel = md / range;
        printf("[quant] bits=%u maxerr=%.4e rel=%.4e\n", bits, md, rel);
        /* step atteso = range/(2^bits-1); l'errore max ~ step/2 → rel ~ 1/(2*(2^bits-1)) con margine */
        double bound = 1.2 / (2.0 * ((double)(1u << bits) - 1.0));
        CHECK(rel < bound);
    }

    /* idempotenza: quantizzare il dequantizzato non cambia nulla */
    {
        uint32_t bits = 4, packed = C / (32 / bits), gcols = C / G;
        CHECK(llmcodec_moe_affine_quantize(x, R, C, G, bits, w, sc, bi) == 0);
        llmcodec_moe_affine_dequantize(w, sc, bi, R, C, G, bits, 1, out);
        float* out2 = (float*)malloc((size_t)R * C * sizeof(float));
        uint32_t* w2 = (uint32_t*)malloc((size_t)R * packed * 4);
        uint16_t* sc2 = (uint16_t*)malloc((size_t)R * gcols * 2);
        uint16_t* bi2 = (uint16_t*)malloc((size_t)R * gcols * 2);
        CHECK(llmcodec_moe_affine_quantize(out, R, C, G, bits, w2, sc2, bi2) == 0);
        llmcodec_moe_affine_dequantize(w2, sc2, bi2, R, C, G, bits, 1, out2);
        double md = 0; for (uint32_t i = 0; i < R * C; ++i) md = fmax(md, fabs((double)out2[i] - out[i]));
        printf("[quant] idempotenza maxdiff=%.4e\n", md);
        CHECK(md < 1e-3);
        free(out2); free(w2); free(sc2); free(bi2);
    }

    /* parametri invalidi */
    CHECK(llmcodec_moe_affine_quantize(x, R, C, G, 0, w, sc, bi) < 0);     /* bits fuori da 1..8 */
    CHECK(llmcodec_moe_affine_quantize(x, R, C, G, 9, w, sc, bi) < 0);     /* bits fuori da 1..8 */
    CHECK(llmcodec_moe_affine_quantize(x, R, 48, 16, 5, w, sc, bi) < 0);   /* (cols*bits)%32 != 0 */
    CHECK(llmcodec_moe_affine_quantize(x, R, C, 48, 4, w, sc, bi) < 0);  /* C%group != 0 */
    CHECK(llmcodec_moe_affine_quantize(NULL, R, C, G, 4, w, sc, bi) < 0);

    /* §E2/DS4-Q1.b: quantizzatore PESATO dall'imatrix — l'errore sul SEGNALE IMPORTANTE migliora
     * (scenario DS4: outlier a bassa importanza che allargano lo scale naive). */
    {
        const uint32_t C64 = 64, G64 = 64;   /* un gruppo */
        float xw[64], imp[64];
        for (int i = 0; i < 8; i++) { xw[i] = (i % 2 == 0) ? 97.0f : -101.0f; imp[i] = 0.0002f; }
        for (int i = 8; i < 64; i++) { xw[i] = 0.5f * sinf((float)i); imp[i] = 200.0f; }
        uint32_t wq[64 / 8]; uint16_t scq[1], biq[1]; float dq[64];
        CHECK(llmcodec_moe_affine_quantize_weighted(xw, NULL, 1, C64, G64, 4, wq, scq, biq) == 0);
        llmcodec_moe_affine_dequantize(wq, scq, biq, 1, C64, G64, 4, 1, dq);
        double e_naive = 0; for (int i = 8; i < 64; i++) { double d = xw[i] - dq[i]; e_naive += d * d; }
        CHECK(llmcodec_moe_affine_quantize_weighted(xw, imp, 1, C64, G64, 4, wq, scq, biq) == 0);
        llmcodec_moe_affine_dequantize(wq, scq, biq, 1, C64, G64, 4, 1, dq);
        double e_w = 0; for (int i = 8; i < 64; i++) { double d = xw[i] - dq[i]; e_w += d * d; }
        printf("[quant-weighted] err segnale importante: naive=%.6g pesata=%.6g\n", e_naive, e_w);
        CHECK(e_w < e_naive);
        /* fallback: imp=NULL == RTN (stessi codici) */
        uint32_t wq2[64 / 8]; uint16_t scq2[1], biq2[1];
        CHECK(llmcodec_moe_affine_quantize(xw, 1, C64, G64, 4, wq2, scq2, biq2) == 0);
        CHECK(llmcodec_moe_affine_quantize_weighted(xw, NULL, 1, C64, G64, 4, wq, scq, biq) == 0);
        CHECK(memcmp(wq, wq2, sizeof(wq)) == 0);
        /* group-level wrapper: gira e produce scale/bias finiti */
        float s1 = 0, b1 = 0; uint8_t codes[64];
        llmcodec_moe_weighted_affine_group(xw, imp, C64, 4, &s1, &b1, codes);
        CHECK(isfinite(s1) && isfinite(b1) && s1 != 0.0f);
    }

    free(w); free(sc); free(bi); free(out);
    if (fails == 0) printf("test_moe_quant: OK\n");
    else fprintf(stderr, "test_moe_quant: %d FAIL\n", fails);
    return fails ? 1 : 0;
}
