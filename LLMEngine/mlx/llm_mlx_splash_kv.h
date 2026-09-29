/* llm_mlx_splash_kv.h — TRASCRIZIONE FEDELE del KV di Splash (github.com/incoai/splash).
 *
 * REGOLA DI QUESTA COPIA: si trascrive, non si inventa. Ogni blocco cita il file
 * e le righe di origine. Le deviazioni sono SOLO dove l'host differisce e sono
 * elencate tutte in fondo (DEVIATIONS). Se una parte non e' copiabile, si lascia
 * segnalata come GAP, non si sostituisce in silenzio.
 *
 * Fonti (clone locale del repo Splash):
 *   runtime/metal/abi/PagedAttention.h            -> ABI dei parametri
 *   runtime/ops/PagedKv.hpp                       -> Format / Layout / LayoutGuard
 *   runtime/metal/kernels/common/q8_paging.h      -> indici Q8
 *   runtime/metal/kernels/common/paged_store_row.h-> store/quant in scrittura
 *   runtime/metal/kernels/common/paged_attention_tile.h -> attenzione paged
 *   runtime/ops/PageStorage.{hpp,mm}              -> storage fisico
 *   runtime/engine/KvPool.{hpp,cpp}, KvCache.*    -> policy/cache
 */
#ifndef LLM_MLX_SPLASH_KV_H
#define LLM_MLX_SPLASH_KV_H

#include <stdint.h>
#include <stdbool.h>

/* ── 1) ABI dei parametri — VERBATIM da abi/PagedAttention.h ─────────────── */
/* (i struct Splash* usano uint32_t; qui identici per layout ABI) */
struct SplashChunkedPrefillParams {      /* 32 B */
    uint32_t committed_tokens;
    uint32_t chunk_tokens;
    uint32_t chunk_stride;
    uint32_t page_table_entries;
    uint32_t physical_page_count;
    uint32_t reserved0, reserved1, reserved2;
};
struct SplashQ8PrefillAttentionParams {  /* 32 B */
    uint32_t committed_tokens;
    uint32_t rows;
    uint32_t chunk_stride;
    uint32_t page_table_entries;
    uint32_t physical_page_count;
    uint32_t split_count;
    uint32_t reserved0, reserved1;
};
struct SplashQ8VerifyAttentionParams {   /* 32 B */
    uint32_t committed_tokens;
    uint32_t active_rows;
    uint32_t chunk_stride;
    uint32_t page_table_entries;
    uint32_t physical_page_count;
    uint32_t split_count;
    uint32_t slot_splits;
    uint32_t reserved2;
};

/* ── 2) Format / Layout — VERBATIM da ops/PagedKv.hpp ────────────────────── */
/* enum Format { Int8=1, BFloat16=2 };  symmetric int8 min/max = -127/127.
 * storageFormatName(Int8) = "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major"
 * elementsPerScale = headDimension ; una scala f32 per (token, kv-head).
 * K token-major ; V dimension-major ; page = SPLASH_TARGET_KV_BLOCK_TOKENS (32). */
#define SPLASH_KV_FORMAT_INT8      1u
#define SPLASH_KV_FORMAT_BF16      2u
#define SPLASH_KV_QUANT_MIN      (-127)
#define SPLASH_KV_QUANT_MAX      (127)
#define SPLASH_KV_PAGE_TOKENS      32u

/* ── 3) Indici Q8 — VERBATIM da kernels/common/q8_paging.h ───────────────── */
/* SplashQ8HeadDimension e' 256 nel reference (li' head_dim=256). */
static inline unsigned long splash_q8_key_index(unsigned page, unsigned head, unsigned token,
                                                unsigned dimension, unsigned pageTokens, unsigned headDim) {
    const unsigned long elementsPerPage = (unsigned long)pageTokens * headDim; /* per-head slice */
    return (unsigned long)page * (unsigned long)pageTokens * headDim
         + (unsigned long)head * pageTokens * headDim
         + (unsigned long)token * headDim + dimension;                         /* K token-major */
}
static inline unsigned long splash_q8_value_index(unsigned page, unsigned head, unsigned token,
                                                  unsigned dimension, unsigned pageTokens, unsigned headDim) {
    return (unsigned long)page * (unsigned long)pageTokens * headDim
         + (unsigned long)head * headDim * pageTokens
         + (unsigned long)dimension * pageTokens + token;                      /* V dimension-major */
}
static inline unsigned long splash_q8_scale_index(unsigned page, unsigned head, unsigned token,
                                                  unsigned pageTokens) {
    return (unsigned long)page * (unsigned long)pageTokens
         + (unsigned long)head * pageTokens + token;                           /* 1 scala per (head,token) */
}

/* ── 4) Store / quant in scrittura — TRASCRITTO da paged_store_row.h ──────── */
/* Blocco Metal (da montare via mlx_fast_metal_kernel; in Splash gira nativamente).
 * UN threadgroup per (token, kv-head); i thread sono le DIMENSIONI; la scala e'
 * max|v|/127 ridotta sul head_dim; clamp int8 simmetrico. */
static const char splash_store_row_metal[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    /* splash_store_kv_row (INT8 branch), fedele: */
    "kernel void splash_store(device const T* chunk_keys [[buffer(0)]],\n"
    "                         device const T* chunk_values [[buffer(1)]],\n"
    "                         device char* cache_keys [[buffer(2)]],\n"
    "                         device float* key_scales [[buffer(3)]],\n"
    "                         device char* cache_values [[buffer(4)]],\n"
    "                         device float* value_scales [[buffer(5)]],\n"
    "                         device const uint* page_table [[buffer(6)]],\n"
    "                         constant SplashChunkedPrefillParams& params [[buffer(7)]],\n"
    "                         threadgroup float* maxima [[threadgroup(0)]],\n"
    "                         uint3 tgid [[threadgroup_position_in_grid]],\n"
    "                         uint lane [[thread_position_in_threadgroup]],\n"
    "                         uint simd_group [[simdgroup_index_in_threadgroup]],\n"
    "                         uint simd_lane [[thread_index_in_simdgroup]])\n"
    "{\n"
    "    const uint head = tgid.y; const uint chunk_token = tgid.z; const uint dimension = lane;\n"
    "    const uint logical_token = params.committed_tokens + chunk_token;\n"
    "    const uint physical_page = page_table[logical_token / SPLASH_KV_PAGE_TOKENS];\n"
    "    if (physical_page >= params.physical_page_count) return;\n"
    "    const uint page_token = logical_token % SPLASH_KV_PAGE_TOKENS;\n"
    "    /* value_tensor = false */\n"
    "    const ulong kidx = splash_current_key_index(params.chunk_stride, head, chunk_token, dimension);\n"
    "    const float value = float(chunk_keys[kidx]);\n"
    "    float local_maximum = simd_max(abs(value));\n"
    "    if (simd_lane == 0) maxima[simd_group] = local_maximum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (simd_group == 0) { for (uint o = 4; o != 0; o >>= 1) { if (simd_lane < o) maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + o]); simdgroup_barrier(mem_flags::mem_threadgroup); } }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    const float maximum = maxima[0];\n"
    "    const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;\n"
    "    const int quantized = maximum == 0.0f ? 0 : clamp(int(rint(value * 127.0f / maximum)), -127, 127);\n"
    "    cache_keys[splash_q8_key_index_t(physical_page, head, page_token, dimension)] = char(quantized);\n"
    "    if (dimension == 0) key_scales[splash_q8_scale_index_t(physical_page, head, page_token)] = scale;\n"
    "}\n";

/* NOTA: nel reference la funzione splash_current_key_index/value_index sono
 * device-side (definite in q8_paging.h); qui vanno aggiunte al sorgente Metal
 * come template (analogo). Non ancora fatto: vedi GAP-1. */

/* ── DEVIATIONS (ognuna da approvare) ────────────────────────────────────────
 * (vuoto) — finche' non ce ne sono, la copia e' fedele.
 *
 * GAP (parti NON copiabili/ancora non copiate, NON sostituite):
 * GAP-1: helper device-side `splash_current_key_index/value_index` e i template
 *        (KVHeads) dai kernel Splash vanno ancora trascritti nel sorgente Metal.
 * GAP-2: il read usa MPP (MetalPerformancePrimitives) `matmul2d` in
 *        paged_attention_tile.h: MPP non e' disponibile nel nostro host MLX.
 *        Da decidere COME con il committente (non sostituito da me).
 * GAP-3: PageStorage/KvPool/KvCache (storage fisico, sparse heap, policy) e
 *        l'ABI `PagedAttention.h` completo: da trascrivere integralmente.
 * GAP-4: page_table e la pianificazione (PageStorage/KvPool) da portare.
 * ──────────────────────────────────────────────────────────────────────────── */

#endif /* LLM_MLX_SPLASH_KV_H */
