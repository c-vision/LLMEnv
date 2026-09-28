/* llm_gemvmx4_bridge.h — ABI C pura per llmmetal_gemv_mxfp4.h (native/llmmetal_gemv/), che
 * è Objective-C (usa <Metal/Metal.h>) e quindi non includibile da llmengine_mlx.c (C puro) —
 * stesso motivo e schema di llm_gemvq4_bridge.h.
 *
 * Differenze vs il bridge affine (Q4): le scale sono FP8-E8M0 (uint8) non half, e NON esiste
 * bias (il dequant MXFP4 è valore = scala * e2m1(nibble), senza addendo). Verificato contro i
 * sorgenti MLX (fp4.h / fp8.h / fp_quantized.h), vedi il commento di testa del kernel.
 */
#ifndef LLM_GEMVMX4_BRIDGE_H
#define LLM_GEMVMX4_BRIDGE_H

#include <stdint.h>

typedef struct LlmGemvMx4Ctx LlmGemvMx4Ctx; /* opaco lato C */

LlmGemvMx4Ctx* llm_gemvmx4_create(void);
void llm_gemvmx4_destroy(LlmGemvMx4Ctx* ctx);

/* Carica i pesi MXFP4 UNA VOLTA (Wq: [M,H/2] uint8 nibble-packed; scales: [M,H/group_size]
 * uint8 e8m0). group_size = 32 (MXFP4). Ritorna 0 su successo. */
int llm_gemvmx4_load_weights(LlmGemvMx4Ctx* ctx, const uint8_t* Wq,
                             const uint8_t* scales,
                             uint32_t H, uint32_t M, uint32_t group_size);

/* Un passo di decode: x [H] half, out [M] half. Ritorna 0 su successo. */
int llm_gemvmx4_run(LlmGemvMx4Ctx* ctx, const uint16_t* x, uint16_t* out);

#endif /* LLM_GEMVMX4_BRIDGE_H */
