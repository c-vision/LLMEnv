/* llmengine_moe_gguf_expert.c — SwiGLU di un esperto GGUF (dequant-on-read), singolo e batched. Vedi .h. */
#include "llmengine_moe_gguf_expert.h"
#include "llmengine_moe_ggml.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* dequantizza gate/up/down dell'esperto `e` nei tre buffer (layout già [righe=out] come da .h). */
static LlmEngineMoeStatus dequant_expert(FILE* f, const LlmEngineMoeGguf* g,
                                   const LlmCodecGgufTensor* gate, const LlmCodecGgufTensor* up,
                                   const LlmCodecGgufTensor* down, uint32_t expert,
                                   uint32_t hidden, uint32_t inter,
                                   float* wg, float* wu, float* wd) {
    LlmEngineMoeStatus s = llmengine_moe_gguf_expert_f32(f, g, gate, expert, wg, (size_t)hidden * inter);
    if (s == LLMENGINE_MOE_OK) s = llmengine_moe_gguf_expert_f32(f, g, up, expert, wu, (size_t)hidden * inter);
    if (s == LLMENGINE_MOE_OK) s = llmengine_moe_gguf_expert_f32(f, g, down, expert, wd, (size_t)inter * hidden);
    return s;
}

/* dequantizza gate/up/down dell'esperto `e` dai byte GREZZI in memoria (store mmap/LRU DS4). */
static LlmEngineMoeStatus dequant_expert_mem(uint32_t t_gate, uint32_t t_up, uint32_t t_down,
                                       const void* bg, const void* bu, const void* bd,
                                       uint32_t hidden, uint32_t inter,
                                       float* wg, float* wu, float* wd) {
    if (llmengine_moe_ggml_dequant_row(t_gate, bg, wg, (size_t)hidden * inter) != 0)
        return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    if (llmengine_moe_ggml_dequant_row(t_up, bu, wu, (size_t)hidden * inter) != 0)
        return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    if (llmengine_moe_ggml_dequant_row(t_down, bd, wd, (size_t)inter * hidden) != 0)
        return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    return LLMENGINE_MOE_OK;
}

/* out[hidden] = down( silu(gate·x) * (up·x) ) con pesi GIA' dequantizzati.
 * `t` è scratch di `inter` elementi (riusabile tra le posizioni nella variante batch). */
static void swiglu_apply(const float* wg, const float* wu, const float* wd,
                         uint32_t hidden, uint32_t inter, float* t,
                         const float* x, float* out) {
    for (uint32_t o = 0; o < inter; ++o) {
        double gsum = 0.0, usum = 0.0;
        for (uint32_t i = 0; i < hidden; ++i) {
            gsum += (double)wg[(size_t)o * hidden + i] * (double)x[i];
            usum += (double)wu[(size_t)o * hidden + i] * (double)x[i];
        }
        double silu = gsum / (1.0 + exp(-gsum));   /* silu(z) = z*sigmoid(z) */
        t[o] = (float)(silu * usum);
    }
    for (uint32_t h = 0; h < hidden; ++h) {
        double acc = 0.0;
        for (uint32_t o = 0; o < inter; ++o) acc += (double)wd[(size_t)h * inter + o] * (double)t[o];
        out[h] = (float)acc;
    }
}

LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu(FILE* f, const LlmEngineMoeGguf* g,
                                        const LlmCodecGgufTensor* gate, const LlmCodecGgufTensor* up,
                                        const LlmCodecGgufTensor* down, uint32_t expert,
                                        uint32_t hidden, uint32_t inter,
                                        const float* x, float* out) {
    if (f == NULL || g == NULL || gate == NULL || up == NULL || down == NULL || x == NULL || out == NULL)
        return LLMENGINE_MOE_ERR_INVALID;
    if (hidden == 0 || inter == 0) return LLMENGINE_MOE_ERR_INVALID;

    float* wg = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wu = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wd = (float*)malloc((size_t)inter * hidden * sizeof(float));
    float* t  = (float*)malloc((size_t)inter * sizeof(float));
    if (wg == NULL || wu == NULL || wd == NULL || t == NULL) {
        free(wg); free(wu); free(wd); free(t); return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    LlmEngineMoeStatus s = dequant_expert(f, g, gate, up, down, expert, hidden, inter, wg, wu, wd);
    if (s == LLMENGINE_MOE_OK) swiglu_apply(wg, wu, wd, hidden, inter, t, x, out);
    free(wg); free(wu); free(wd); free(t);
    return s;
}

LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu_mem(uint32_t t_gate, uint32_t t_up, uint32_t t_down,
                                            const void* bg, const void* bu, const void* bd,
                                            uint32_t hidden, uint32_t inter,
                                            const float* x, float* out) {
    if (bg == NULL || bu == NULL || bd == NULL || x == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (hidden == 0 || inter == 0) return LLMENGINE_MOE_ERR_INVALID;
    float* wg = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wu = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wd = (float*)malloc((size_t)inter * hidden * sizeof(float));
    float* t  = (float*)malloc((size_t)inter * sizeof(float));
    if (wg == NULL || wu == NULL || wd == NULL || t == NULL) {
        free(wg); free(wu); free(wd); free(t); return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    LlmEngineMoeStatus s = dequant_expert_mem(t_gate, t_up, t_down, bg, bu, bd, hidden, inter, wg, wu, wd);
    if (s == LLMENGINE_MOE_OK) swiglu_apply(wg, wu, wd, hidden, inter, t, x, out);
    free(wg); free(wu); free(wd); free(t);
    return s;
}

LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu_batch_mem(uint32_t t_gate, uint32_t t_up, uint32_t t_down,
                                                  const void* bg, const void* bu, const void* bd,
                                                  uint32_t hidden, uint32_t inter,
                                                  const float* X, uint32_t n, float* out) {
    if (bg == NULL || bu == NULL || bd == NULL || X == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (hidden == 0 || inter == 0 || n == 0) return LLMENGINE_MOE_ERR_INVALID;
    float* wg = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wu = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wd = (float*)malloc((size_t)inter * hidden * sizeof(float));
    float* t  = (float*)malloc((size_t)inter * sizeof(float));
    if (wg == NULL || wu == NULL || wd == NULL || t == NULL) {
        free(wg); free(wu); free(wd); free(t); return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    LlmEngineMoeStatus s = dequant_expert_mem(t_gate, t_up, t_down, bg, bu, bd, hidden, inter, wg, wu, wd);
    if (s == LLMENGINE_MOE_OK)
        for (uint32_t i = 0; i < n; ++i)
            swiglu_apply(wg, wu, wd, hidden, inter, t, X + (size_t)i * hidden, out + (size_t)i * hidden);
    free(wg); free(wu); free(wd); free(t);
    return s;
}

LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu_batch(FILE* f, const LlmEngineMoeGguf* g,
                                              const LlmCodecGgufTensor* gate, const LlmCodecGgufTensor* up,
                                              const LlmCodecGgufTensor* down, uint32_t expert,
                                              uint32_t hidden, uint32_t inter,
                                              const float* X, uint32_t n, float* out) {
    if (f == NULL || g == NULL || gate == NULL || up == NULL || down == NULL || X == NULL || out == NULL)
        return LLMENGINE_MOE_ERR_INVALID;
    if (hidden == 0 || inter == 0 || n == 0) return LLMENGINE_MOE_ERR_INVALID;

    float* wg = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wu = (float*)malloc((size_t)hidden * inter * sizeof(float));
    float* wd = (float*)malloc((size_t)inter * hidden * sizeof(float));
    float* t  = (float*)malloc((size_t)inter * sizeof(float));
    if (wg == NULL || wu == NULL || wd == NULL || t == NULL) {
        free(wg); free(wu); free(wd); free(t); return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    LlmEngineMoeStatus s = dequant_expert(f, g, gate, up, down, expert, hidden, inter, wg, wu, wd);
    if (s == LLMENGINE_MOE_OK)
        for (uint32_t i = 0; i < n; ++i)
            swiglu_apply(wg, wu, wd, hidden, inter, t, X + (size_t)i * hidden, out + (size_t)i * hidden);
    free(wg); free(wu); free(wd); free(t);
    return s;
}
