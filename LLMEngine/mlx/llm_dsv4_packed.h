/* llm_dsv4_packed.h — dot di un tensore quantizzato GGUF (packed) contro un vettore f32,
 * senza dequantizzare in RAM. Tipi: F32/F16/BF16/Q8_0/Q2_K/IQ2_XXS. Header-only.
 * Test: llm_dsv4_packed_test.c (confronto col dequant). */
#ifndef LLM_DSV4_PACKED_H
#define LLM_DSV4_PACKED_H

#include <math.h>
#include "llm_ggml.h"

/* out = sum_k weight(k) * y[k]; ritorna false se il tipo non e' supportato. */
static inline bool llm_dsv4_dot_tensor(int t, const uint8_t* p, int64_t n, const float* y, float* out) {
    switch (t) {
        case LLM_GGML_TYPE_F32: {
            double s = 0.0; const float* w = (const float*)p;
            for (int64_t k = 0; k < n; k++) s += (double)w[k] * y[k];
            *out = (float)s; return true;
        }
        case LLM_GGML_TYPE_F16: {
            double s = 0.0;
            for (int64_t k = 0; k < n; k++) { uint16_t h; memcpy(&h, p + 2*k, 2); s += (double)llmcodec_f16_to_f32(h) * y[k]; }
            *out = (float)s; return true;
        }
        case LLM_GGML_TYPE_BF16: {
            double s = 0.0;
            for (int64_t k = 0; k < n; k++) { uint16_t b; memcpy(&b, p + 2*k, 2); s += (double)llmcodec_bf16_to_f32(b) * y[k]; }
            *out = (float)s; return true;
        }
        case LLM_GGML_TYPE_Q8_0: {
            double s = 0.0;
            for (int64_t b = 0; b < n / 32; b++) {
                const uint8_t* blk = p + b * 34;
                uint16_t dh; memcpy(&dh, blk, 2); float d = llmcodec_f16_to_f32(dh);
                double acc = 0.0;
                for (int j = 0; j < 32; j++) acc += (double)(int8_t)blk[2 + j] * y[b*32 + j];
                s += (double)d * acc;
            }
            *out = (float)s; return true;
        }
        case LLM_GGML_TYPE_Q2_K: {
            double s = 0.0;
            for (int64_t k = 0; k < n; k++) s += (double)llmcodec_ggml_q2_K_value(p + (size_t)(k/256)*84, (uint32_t)(k%256)) * y[k];
            *out = (float)s; return true;
        }
        case LLM_GGML_TYPE_IQ2_XXS: {
            double s = 0.0;
            for (int64_t k = 0; k < n; k++) s += (double)llmcodec_ggml_iq2_xxs_value(p + (size_t)(k/256)*66, (uint32_t)(k%256)) * y[k];
            *out = (float)s; return true;
        }
        default: return false;
    }
}

#endif /* LLM_DSV4_PACKED_H */
