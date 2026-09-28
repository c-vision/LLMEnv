/* llmengine_moe_gguf_expert.h — SwiGLU di UN esperto di un GGUF (dequant-on-read). */
#ifndef LLMENGINE_MOE_GGUF_EXPERT_H
#define LLMENGINE_MOE_GGUF_EXPERT_H

#include "llmengine_moe_gguf.h"
#include "llmengine_moe_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* out[hidden] = down( silu(gate·x) * (up·x) )
 * Layout ggml ne0-fastest: gate/up [hidden,inter] → riga o = `o*hidden+i`;
 * down [inter,hidden] → riga h = `h*inter+o`. L'esperto è il dim più lento → blocco contiguo
 * (dequant-on-read via llmengine_moe_gguf_expert_f32). `f` è il file GGUF riaperto dal chiamante.
 * Ritorna LLMENGINE_MOE_OK o un errore esplicito. */
LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu(FILE* f, const LlmEngineMoeGguf* g,
                                        const LlmCodecGgufTensor* gate, const LlmCodecGgufTensor* up,
                                        const LlmCodecGgufTensor* down, uint32_t expert,
                                        uint32_t hidden, uint32_t inter,
                                        const float* x, float* out);

/* Come sopra ma BATCHED (prefill M17): dequantizza l'esperto UNA volta e lo applica alle
 * `n` posizioni X[n][hidden] → out[n][hidden]. Evita di rileggere il blocco da disco per
 * ogni posizione del prompt. */
LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu_batch(FILE* f, const LlmEngineMoeGguf* g,
                                              const LlmCodecGgufTensor* gate, const LlmCodecGgufTensor* up,
                                              const LlmCodecGgufTensor* down, uint32_t expert,
                                              uint32_t hidden, uint32_t inter,
                                              const float* X, uint32_t n, float* out);

/* Variante MEMORIA (store mmap/LRU, stile DS4): i byte grezzi (ggml, packed) degli esperti
 * gate/up/down sono gia' disponibili → niente read da file. L'UNICA sorgente della matematica
 * SwiGLU (nessuna duplicazione): stesso swiglu_apply di llmengine_moe_gguf_expert_swiglu. */
LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu_mem(uint32_t t_gate, uint32_t t_up, uint32_t t_down,
                                            const void* bg, const void* bu, const void* bd,
                                            uint32_t hidden, uint32_t inter,
                                            const float* x, float* out);
LlmEngineMoeStatus llmengine_moe_gguf_expert_swiglu_batch_mem(uint32_t t_gate, uint32_t t_up, uint32_t t_down,
                                                  const void* bg, const void* bu, const void* bd,
                                                  uint32_t hidden, uint32_t inter,
                                                  const float* X, uint32_t n, float* out);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_GGUF_EXPERT_H */
