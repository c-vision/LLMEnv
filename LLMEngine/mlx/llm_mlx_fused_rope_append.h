/* llm_mlx_fused_rope_append.h — kernel Metal fuso (decode-only) che dispone Q/K/V
 * proiettati ([B,L,H*D] -> [B,H,L,D]) e applica il RoPE in un solo lancio, con
 * frequenze custom (llama3/yarn) o formula a base. API: llm_mlx_fused_rope_append. */
#ifndef LLM_MLX_FUSED_ROPE_APPEND_H
#define LLM_MLX_FUSED_ROPE_APPEND_H

#include "mlx/c/mlx.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode-only fused layout + RoPE for independent Q/K/V projections.
 * Inputs are [B,L,H*D]; outputs are [B,H,L,D]. Returns false without
 * modifying the caller-visible outputs when the geometry is unsupported or
 * the MLX custom-kernel launch fails.
 *
 * `freqs`: tabella float32 [rope_dims/2] di frequenze CUSTOM (convenzione MLX:
 * inv_freq[i] = 1/freqs[i], vedi rope_single_freqs in mlx rope.metal). Serve per
 * i rope scaling che NON sono la formula a base singola (llama3, yarn). Handle
 * vuoto (freqs.ctx == NULL) = usa la formula a base con rope_base/rope_scale. */
bool llm_mlx_fused_rope_append(
    mlx_array* q_out,
    mlx_array* k_out,
    mlx_array* v_out,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rope_dims,
    float rope_base,
    float rope_scale,
    bool traditional,
    mlx_array freqs,
    int positions_base,
    mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif
