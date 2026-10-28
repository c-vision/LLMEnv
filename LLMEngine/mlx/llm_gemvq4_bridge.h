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
#include <stddef.h>

typedef struct LlmGemvQ4Ctx LlmGemvQ4Ctx; /* opaco lato C */

LlmGemvQ4Ctx* llm_gemvq4_create(void);
void llm_gemvq4_destroy(LlmGemvQ4Ctx* ctx);

/* Carica i pesi quantizzati UNA VOLTA (Wq: [M,H/2] uint8; scales/biases: [M,H/group_size]
 * half). Ritorna 0 su successo. */
/* Zero-copy: view Metal sui puntatori (mmap), nessuna copia dei pesi. */
int llm_gemvq4_load_weights_wonly_nocopy(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                                        const uint16_t* scales, const uint16_t* biases,
                                        uint32_t H, uint32_t M, uint32_t group_size);
int llm_gemvq4_load_weights_nocopy(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                                   const uint16_t* scales, const uint16_t* biases,
                                   uint32_t H, uint32_t M, uint32_t group_size);
int llm_gemvq4_load_weights(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                             const uint16_t* scales, const uint16_t* biases,
                             uint32_t H, uint32_t M, uint32_t group_size);

/* Un passo di decode: x [H] half, out [M] half. Ritorna 0 su successo. */
/* 1 = scales/biases BF16 (checkpoint MLX in bf16); 0 = half (default). Da chiamare dopo il load, prima di run. */
void llm_gemvq4_set_scale_bf16(LlmGemvQ4Ctx* ctx, int bf16);
int llm_gemvq4_run(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out);

/* Vista 3D zero-copy [E][M][H] (esperti): un contesto per tensore, esperto scelto a run-time. */
int llm_gemvq4_load_weights_3d_nocopy(LlmGemvQ4Ctx* ctx, const uint8_t* Wq, const uint16_t* scales, const uint16_t* biases,
                                      uint32_t H, uint32_t M, uint32_t group_size, uint32_t E, size_t w_stride, size_t s_stride);
int llm_gemvq4_run_expert(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out, uint32_t expert);

/* Batch di gemv in un solo command buffer. expert < 0 = lineare 2D del contesto. */
/* Router + top-K + mid + down + combine pesata in UN command buffer. W: [NE+1][H] f32 (ultima riga = gate shared expert).
 * upload una volta sul contesto gate (cg); run: x [H] f32 -> out [H] f32. */
int llm_gemvq4_router_upload(LlmGemvQ4Ctx* cg, const float* W, int rows, int H);
int llm_gemvq4_run_moe_gpu(LlmGemvQ4Ctx* cg, LlmGemvQ4Ctx* cu, LlmGemvQ4Ctx* sg, LlmGemvQ4Ctx* su,
                           LlmGemvQ4Ctx* cd, LlmGemvQ4Ctx* sd, const float* x, float* out, int K);

typedef struct { LlmGemvQ4Ctx* ctx; int expert; const uint16_t* x; uint16_t* out; } LlmGemvQ4Job;
int llm_gemvq4_run_batch(const LlmGemvQ4Job* jobs, int n);
/* Esperto-major (come ds4): K esperti selezionati in UN solo dispatch. out_all = K*rows. */
/* FUSI stile ds4: mid = gate+up+silu, down = proiezione*peso router. Due dispatch per layer. */
int llm_gemvq4_run_moe_mid(LlmGemvQ4Ctx* cg, LlmGemvQ4Ctx* cu, LlmGemvQ4Ctx* sg, LlmGemvQ4Ctx* su,
                           const uint16_t* x, float* act, const int32_t* experts, int K);
int llm_gemvq4_run_moe(LlmGemvQ4Ctx* cg, LlmGemvQ4Ctx* cu, LlmGemvQ4Ctx* sg, LlmGemvQ4Ctx* su,
                       LlmGemvQ4Ctx* cd, LlmGemvQ4Ctx* sd, const uint16_t* x, float* act, float* out,
                       const int32_t* experts, int K, uint32_t x_stride);
int llm_gemvq4_run_moe_down(LlmGemvQ4Ctx* cd, LlmGemvQ4Ctx* sd, const float* act,
                            float* out, const int32_t* experts, int K, uint32_t x_stride);
int llm_gemvq4_run_experts(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out_all, const int32_t* experts, int K, uint32_t x_stride);

#endif /* LLM_GEMVQ4_BRIDGE_H */
