/* llmengine_moe_qwen3.h — M15.3: primitive dell'adattatore Qwen3-MoE (dequant MLX + SwiGLU esperto).
 *
 * Il blob esperto (prodotto da (converter rimosso)) preserva il quant MLX affine
 * (group_size/bits) e viene DEQUANTIZZATO ON READ: nessuna materializzazione F16 del peso.
 * Layout blob (little-endian):
 *   [header 32B] magic'MLX' version group bits hidden inter ntensors rsvd
 *   gate {weight U32 (inter, hidden/per), scales BF16 (inter, hidden/group), biases BF16}
 *   up   {idem}
 *   down {weight U32 (hidden, inter/per), scales BF16 (hidden, inter/group), biases BF16}
 *   con per = 32/bits. */
#ifndef LLMENGINE_MOE_QWEN3_H
#define LLMENGINE_MOE_QWEN3_H

#include "llmengine_moe_types.h"
#include <stddef.h>
#include <stdint.h>

/* Dimensione minima valida + dims dal blob. */
LlmEngineMoeStatus llmengine_moe_q3_blob_dims(const void* blob, size_t size,
                                  uint32_t* hidden, uint32_t* inter,
                                  uint32_t* group, uint32_t* bits);

/* GEMV dequant-on-read da un sotto-tensore MLX: out[rows] = W·x (W = rows×cols, quantizzato).
 * `w/scales/biases` puntano al sotto-tensore dentro il blob. */
void llmengine_moe_q3_gemv(const void* w, const void* scales, const void* biases,
                     uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                     const float* x, float* out);

/* Calcolo di UN esperto SwiGLU: out[hidden] = down( silu(gate·x) * (up·x) ). */
LlmEngineMoeStatus llmengine_moe_q3_expert_swiglu(const void* blob, size_t size,
                                      const float* x, float* out);

/* Come sopra ma BATCHED (M17): X[n][hidden] → out[n][hidden], un solo passaggio sul peso
 * quantizzato (decodificato una volta e riusato per le n posizioni). */
LlmEngineMoeStatus llmengine_moe_q3_expert_swiglu_batch(const void* blob, size_t size,
                                            const float* X, uint32_t n, float* out);

#endif /* LLMENGINE_MOE_QWEN3_H */
