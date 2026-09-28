/* llm_dsv4_quant.h — quantizzazioni DS4 del path attenzione/cache:
 *  - fp8 (E4M3FN) round-trip della parte non-RoPE del KV (grezzo e compresso);
 *  - QAT indexer: Hadamard 128 + FP4 (E2M1FN) round-trip (Q e KV compresso indexer).
 * Port fedele del riferimento CPU ds4.c. Header-only. Test: llm_dsv4_quant_test.c.
 *
 * Gli ATOMI di formato (e4m3/fp4) NON sono piu' qui: vivono nel CODEC (§0bis) e sono chiamati
 * direttamente (llmcodec_fp8_round_e4m3, llmcodec_fp4_e2m1_round). */
#ifndef LLM_DSV4_QUANT_H
#define LLM_DSV4_QUANT_H

#include <math.h>
#include <stdint.h>
#include "llmcodec_fp8_fp4.h"   /* atomi di FORMATO del CODEC: llmcodec_fp8_round_e4m3, llmcodec_fp4_e2m1_round */

/* Round-trip E4M3 della parte non-RoPE (head_dim-n_rot), blocchi da 64. */
static inline void llm_dsv4_fp8_kv_quantize_row(float* x, int head_dim, int n_rot) {
    int n_nope = head_dim - n_rot;
    if (n_nope <= 0 || (n_nope % 64) != 0) return;   /* il modello reale ha n_nope multiplo di 64 */
    for (int off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (int i = 0; i < 64; i++) { float av = fabsf(x[off + i]); if (av > amax) amax = av; }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (int i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = llmcodec_fp8_round_e4m3(v) * scale;
        }
    }
}
static inline void llm_dsv4_hadamard128_inplace(float* x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1)
        for (uint32_t base = 0; base < 128; base += 2u * stride)
            for (uint32_t i = 0; i < stride; i++) {
                float a = x[base + i], b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
    const float scale = 0.08838834764831845f;
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}
static inline void llm_dsv4_fp4_act_quantize_row(float* x, int n) {
    for (int off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) { float av = fabsf(x[off + i]); if (av > amax) amax = av; }
        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (int i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = llmcodec_fp4_e2m1_round(v) * scale;
        }
    }
}
/* QAT indexer: Hadamard128 + FP4 round-trip (head_dim deve essere 128). */
static inline void llm_dsv4_indexer_qat_row(float* x, int head_dim) {
    if (head_dim != 128) return;
    llm_dsv4_hadamard128_inplace(x);
    llm_dsv4_fp4_act_quantize_row(x, head_dim);
}
static inline void llm_dsv4_indexer_qat_rows(float* x, int rows, int head_dim) {
    for (int r = 0; r < rows; r++) llm_dsv4_indexer_qat_row(x + (long)r * head_dim, head_dim);
}

#endif /* LLM_DSV4_QUANT_H */
