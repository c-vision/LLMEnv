/* llm_kvmetal_bridge.h — ABI C pura per llmmetal_kv.h (native/llmmetal_kv/), che e'
 * Objective-C (usa direttamente <Metal/Metal.h>) e quindi non includibile da
 * llmengine_mlx.c (C puro). L'implementazione (llm_kvmetal_bridge.m) e' compilata come
 * Objective-C separatamente; questo header non espone nessun tipo Objective-C.
 *
 * Wraps il modulo standalone llmmetal_kv.h (KV page-pool con buffer Metal reali, decode
 * attention paged online-softmax split-K) per l'uso sperimentale in llmengine_mlx.c —
 * vedi LLMMETAL_KVMETAL in llama_attention_block. Piano:
 * ~/Documents/AiWiki/01-Projects/LLMEnv/Piani/Proposta-614GBs-Banda-Metal-DeepDive.md §4 L4.
 */
#ifndef LLM_KVMETAL_BRIDGE_H
#define LLM_KVMETAL_BRIDGE_H

#include <stdint.h>

typedef struct LlmKvMetalCtx LlmKvMetalCtx; /* opaco lato C */

/* Crea il contesto (device Metal, kernel compilati, pool K/V allocato). Ritorna NULL su
 * errore (stampa il motivo su stderr). head_dim deve essere multiplo di 32, n_q_heads
 * multiplo di n_kv_heads (stessi vincoli di llmmetal_kv.h). */
LlmKvMetalCtx* llm_kvmetal_create(int n_pages, int page_tokens, int n_q_heads, int n_kv_heads, int head_dim);
void llm_kvmetal_destroy(LlmKvMetalCtx* ctx);

/* Scrive UN token (k_tok/v_tok: n_kv_heads*head_dim half ciascuno, bit fp16 in uint16_t)
 * nella pagina fisica page_id, slot offset. Ritorna 0 su successo. */
int llm_kvmetal_write(LlmKvMetalCtx* ctx, int page_id, int offset,
                       const uint16_t* k_tok, const uint16_t* v_tok);

/* Decode attention paged online-softmax. q/out: n_q_heads*head_dim half (bit fp16 in
 * uint16_t). page_table: n_pages_used id di pagina fisica in ordine logico. Ritorna 0 su
 * successo. */
int llm_kvmetal_decode_attention(LlmKvMetalCtx* ctx, const uint16_t* q,
                                  const int* page_table, int n_pages_used, int last_page_len,
                                  float scale, uint16_t* out);

#endif /* LLM_KVMETAL_BRIDGE_H */
