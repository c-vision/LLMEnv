/* llm_mlx_flash_decode_kernel.h — kernel Metal di flash-decoding SDPA causale decode-only
 * (split-K/online-softmax, adattato da mlxcel paged_attention, Apache-2.0) e la sua API.
 * Vedi sotto per la forma dei tensori q/k/v/out e l'indicizzazione K/V contigua. */
#ifndef LLM_MLX_FLASH_DECODE_KERNEL_H
#define LLM_MLX_FLASH_DECODE_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode-only (n_tokens==1) flash-decoding SDPA causale, adattata da mlxcel
 * (src/lib/mlx-cpp/turbo/paged_attention.cpp, PAGED_ATTENTION_DECODE_SOURCE,
 * Apache-2.0). L'originale legge K/V tramite una page-table multi-tenant
 * (pool+CSR); qui il buffer K/V e' quello GIA' contiguo che LLM_KVSLICE
 * produce (mlx_slice a `needed`, letto in §10 come zero-copy), quindi
 * l'indicizzazione a pagine e' sostituita da un indice diretto -- la sola
 * parte portata e' l'algoritmo split-K (online-softmax parallelizzato su
 * NumSplits SIMD-group), che e' il motivo per cui questo kernel dovrebbe
 * battere `mlx_fast_scaled_dot_product_attention` a contesto lungo (token-
 * parallelismo nel decode, dove SDPA generica ha un solo query token da
 * spalmare sulle unita' di calcolo).
 *
 * q: [1, num_heads, 1, head_dim]. k/v: [1, num_kv_heads, visible_len, head_dim]
 * (row-contiguous). out: [1, num_heads, 1, head_dim]. Causale per costruzione
 * (attende SOLO ai primi `visible_len` token, tutta la sequenza e' visibile
 * dato che chi chiama ha gia' tagliato K/V a needed = cache_offset+1). */
bool llm_mlx_flash_decode_attention(
    mlx_array* out,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int visible_len,
    float scale,
    mlx_stream stream);

/* Prefill (Lq>1) flash SDPA causale per head_dim che forano il fused di MLX. Una query
 * token per threadgroup (grid (Lq,H,B)), loop causale sulle chiavi 0..cache_offset+qi.
 * q: [B,H,Lq,D]; k/v: [B,Hkv,cache_offset+Lq,D]; out: [B,H,Lq,D]. Stessa matematica
 * (online-softmax esatto) del decode kernel. */
bool llm_mlx_flash_prefill_attention(
    mlx_array* out,
    mlx_array q,
    mlx_array k,
    mlx_array v,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int lq,
    int cache_offset,
    float scale,
    mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif
