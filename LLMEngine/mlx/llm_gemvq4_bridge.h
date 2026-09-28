/* llm_gemvq4_bridge.h — ABI C pura per llmmetal_gemv_q4.h (native/llmmetal_gemv/), che e'
 * Objective-C (usa direttamente <Metal/Metal.h>) e quindi non includibile da
 * llmengine_mlx.c (C puro) — stesso motivo e stesso schema di llm_kvmetal_bridge.h.
 *
 * Formato Q4 affine verificato byte-per-byte contro un tensore REALE (GLM-4.5-Air, layer 1
 * expert 0 gate_proj, 1408x4096 q4 g64): il layout uint32-packed little-endian di MLX,
 * letto come stream di byte grezzi, coincide esattamente col layout [M,H/2] uint8 (nibble
 * basso = colonna pari) assunto dal kernel — 0/1408 righe fuori tolleranza, max_abs_diff
 * 3.8e-4 vs mlx.quantized_matmul sullo stesso tensore. Vedi
 * ~/Documents/AiWiki/01-Projects/LLMEnv/Piani/Proposta-614GBs-Banda-Metal-DeepDive.md §D16
 * e /tmp/glm_q4val/ (harness di validazione, riusabile per regressione).
 */
#ifndef LLM_GEMVQ4_BRIDGE_H
#define LLM_GEMVQ4_BRIDGE_H

#include <stdint.h>

typedef struct LlmGemvQ4Ctx LlmGemvQ4Ctx; /* opaco lato C */

LlmGemvQ4Ctx* llm_gemvq4_create(void);
void llm_gemvq4_destroy(LlmGemvQ4Ctx* ctx);

/* Carica i pesi quantizzati UNA VOLTA (Wq: [M,H/2] uint8; scales/biases: [M,H/group_size]
 * half). Ritorna 0 su successo. */
int llm_gemvq4_load_weights(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                             const uint16_t* scales, const uint16_t* biases,
                             uint32_t H, uint32_t M, uint32_t group_size);

/* Un passo di decode: x [H] half, out [M] half. Ritorna 0 su successo. */
int llm_gemvq4_run(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out);

#endif /* LLM_GEMVQ4_BRIDGE_H */
