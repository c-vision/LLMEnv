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
#include <stddef.h>

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
    /* device-side helpers — VERBATIM da kernels/common/paged_store_row.h (splash_current_key_index/value_index) */
    "inline ulong splash_current_key_index(uint stride, uint head, uint token, uint dimension) {\n"
    "  return (ulong(head) * stride + token) * 256ul /*SplashQ8HeadDimension*/ + dimension; }\n"
    "inline ulong splash_current_value_index(uint stride, uint head, uint token, uint dimension) {\n"
    "  return (ulong(head) * 256ul /*SplashQ8HeadDimension*/ + dimension) * stride + token; }\n"
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

/* ── 5) Read paged — TRASCRITTO da kernels/common/paged_attention_tile.h ─────
 * dequant-folded: le KEY scales moltiplicano le SCORE, le VALUE scales le
 * probabilita' (int8 resta int8 in lettura). Usa MPP matmul2d, importato cosi'
 * com'e' (nessun rimpiazzo). */
static const char splash_paged_attention_metal[] =
    "#include <metal_stdlib>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace metal;\n"
    "using namespace mpp::tensor_ops;\n"
    /* splash_attention_page_softmax — fedele (TokensPerLane=8, 4 lane/riga): */
    "template <uint QueryHeadsPerKVHead, uint RowsPerTile, bool ScaleInSoftmax, bool Quantized>\n"
    "inline void splash_attention_page_softmax(\n"
    "    threadgroup const float *scores, threadgroup bfloat *probabilities,\n"
    "    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,\n"
    "    threadgroup atomic_uint *rescale, device const float4 *key_scales,\n"
    "    device const float4 *value_scales, uint token_start, uint visible_tokens,\n"
    "    uint committed_tokens, uint active_rows, uint thread_index) {\n"
    "  constexpr uint N = 32u; constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;\n"
    "  constexpr uint TokensPerLane = 8u; constexpr uint LanesPerRow = N / TokensPerLane;\n"
    "  if (thread_index >= LanesPerRow * FusedRows) return;\n"
    "  const uint fused_row = thread_index / LanesPerRow;\n"
    "  const uint column = thread_index % LanesPerRow * TokensPerLane;\n"
    "  const uint query_row = fused_row / QueryHeadsPerKVHead;\n"
    "  const uint causal_end = committed_tokens + min(query_row, active_rows - 1u) + 1u;\n"
    "  const uint limit = min(visible_tokens, causal_end);\n"
    "  const uint token = token_start + column;\n"
    "  threadgroup const float4 *scores4 = reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N + column);\n"
    "  const uint vector = column / 4u;\n"
    "  float score[TokensPerLane];\n"
    "  { float4 low = scores4[0], high = scores4[1];\n"
    "    if (Quantized && ScaleInSoftmax) { low *= key_scales[vector]; high *= key_scales[vector + 1u]; }\n"
    "    low *= 0.0625f; high *= 0.0625f;\n"
    "    score[0]=low.x; score[1]=low.y; score[2]=low.z; score[3]=low.w;\n"
    "    score[4]=high.x; score[5]=high.y; score[6]=high.z; score[7]=high.w; }\n"
    "  float local_max = -INFINITY;\n"
    "  for (uint j = 0; j < TokensPerLane; ++j) { score[j] = token + j < limit ? score[j] : -INFINITY; local_max = max(local_max, score[j]); }\n"
    "  local_max = max(local_max, simd_shuffle_xor(local_max, 1u));\n"
    "  local_max = max(local_max, simd_shuffle_xor(local_max, 2u));\n"
    "  const float previous_max = row_max[fused_row]; const float next_max = max(previous_max, local_max);\n"
    "  float probability[TokensPerLane]; float local_sum = 0.0f;\n"
    "  for (uint j = 0; j < TokensPerLane; ++j) { probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f; local_sum += probability[j]; }\n"
    "  local_sum += simd_shuffle_xor(local_sum, 1u); local_sum += simd_shuffle_xor(local_sum, 2u);\n"
    "  if (column == 0u) { const float scale = (next_max == -INFINITY || next_max == previous_max) ? 1.0f : fast::exp(previous_max - next_max);\n"
    "    previous_scale[fused_row] = scale; row_sum[fused_row] = row_sum[fused_row] * scale + local_sum; row_max[fused_row] = next_max;\n"
    "    if (scale != 1.0f) atomic_store_explicit(rescale, 1u, memory_order_relaxed); }\n"
    "  float4 low_scales(1.0f), high_scales(1.0f);\n"
    "  if (Quantized) { low_scales = value_scales[vector]; high_scales = value_scales[vector + 1u]; }\n"
    "  const float4 low(token+0u<limit?probability[0]*low_scales.x:0.0f, token+1u<limit?probability[1]*low_scales.y:0.0f,\n"
    "                   token+2u<limit?probability[2]*low_scales.z:0.0f, token+3u<limit?probability[3]*low_scales.w:0.0f);\n"
    "  const float4 high(token+4u<limit?probability[4]*high_scales.x:0.0f, token+5u<limit?probability[5]*high_scales.y:0.0f,\n"
    "                    token+6u<limit?probability[6]*high_scales.z:0.0f, token+7u<limit?probability[7]*high_scales.w:0.0f);\n"
    "  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(probabilities + fused_row * N + column);\n"
    "  probabilities4[0] = bfloat4(low); probabilities4[1] = bfloat4(high); }\n";

/* splash_paged_attention_tile — fedele da paged_attention_tile.h (righe 185-310). */
static const char splash_paged_attention_tile_metal[] =
    "template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, bool ScaleInSoftmax, typename CacheElement>\n"
    "inline void splash_paged_attention_tile(\n"
    "    device bfloat *tile_queries, device CacheElement *cache_keys, device const float *key_scales_buffer,\n"
    "    device CacheElement *cache_values, device const float *value_scales_buffer, device const uint *page_table,\n"
    "    uint kv_head, uint committed_tokens, uint active_rows, uint splits, uint split,\n"
    "    device float *partials, device float *statistics, ulong slot, threadgroup float *scores,\n"
    "    threadgroup bfloat *probabilities, threadgroup float *row_max, threadgroup float *row_sum,\n"
    "    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint thread_index) {\n"
    "  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead; constexpr bool Quantized = is_same<CacheElement, int8_t>::value;\n"
    "  constexpr ushort N = 32; constexpr ushort D = 256;\n"
    "  uint visible_tokens = committed_tokens + active_rows;\n"
    "  uint pages = (visible_tokens + N - 1) / N; uint per_split = (pages + splits - 1) / splits;\n"
    "  uint page_begin = split * per_split; if (page_begin >= pages) return;\n"
    "  uint page_end = min(pages, page_begin + per_split);\n"
    "  if (thread_index < M) { row_max[thread_index] = -INFINITY; row_sum[thread_index] = 0.0f; }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  auto qt = tensor(tile_queries, dextents<int,2>{D,M}, array<int,2>{1,D});\n"
    "  auto st = tensor(scores, dextents<int,2>{N,M}, array<int,2>{1,N});\n"
    "  auto pt = tensor(probabilities, dextents<int,2>{N,M}, array<int,2>{1,N}); auto p0 = pt.slice<N,M>(0,0);\n"
    "  auto key_type = tensor(cache_keys, dextents<int,2>{D,N}, array<int,2>{1,D});\n"
    "  auto value_type = tensor(cache_values, dextents<int,2>{N,D}, array<int,2>{1,N});\n"
    "  auto q0 = qt.slice<D,M>(0,0); auto k0 = key_type.template slice<D,N>(0,0); auto v0 = value_type.template slice<N,D>(0,0);\n"
    "  constexpr auto qk_descriptor = matmul2d_descriptor(M,N,D,false,true,false, matmul2d_descriptor::mode::multiply);\n"
    "  constexpr auto pv_descriptor = matmul2d_descriptor(M,D,N,false,true,true, matmul2d_descriptor::mode::multiply_accumulate);\n"
    "  matmul2d<qk_descriptor, execution_simdgroups<8>> qk; matmul2d<pv_descriptor, execution_simdgroups<8>> pv;\n"
    "  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();\n"
    "  const bool running_full = uint(running.get_capacity()) * (8u*32u) == uint(M) * D;\n"
    "  for (ushort i = 0; i < running.get_capacity(); ++i) if (running_full || running.is_valid_element(i)) running[i] = 0.0f;\n"
    "  for (uint page = page_begin; page < page_end; ++page) {\n"
    "    uint physical = page_table[page]; uint token_start = page * N;\n"
    "    auto kt = tensor(cache_keys + splash_q8_key_index<KVHeads>(physical, kv_head, 0, 0), dextents<int,2>{D,N}, array<int,2>{1,D});\n"
    "    auto vt = tensor(cache_values + splash_q8_value_index<KVHeads>(physical, kv_head, 0, 0), dextents<int,2>{N,D}, array<int,2>{1,N});\n"
    "    device const float *key_scales = nullptr; device const float *value_scales = nullptr;\n"
    "    if constexpr (Quantized) { ulong si = splash_q8_scale_index<KVHeads>(physical, kv_head, 0); key_scales = key_scales_buffer + si; value_scales = value_scales_buffer + si; }\n"
    "    auto page_scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();\n"
    "    auto ks = kt.template slice<D,N>(0,0); qk.run(q0, ks, page_scores);\n"
    "    if constexpr (Quantized && !ScaleInSoftmax) { const bool sf = uint(page_scores.get_capacity())*(8u*32u) == uint(M)*N;\n"
    "      for (ushort i = 0; i < page_scores.get_capacity(); ++i) { if (!sf && !page_scores.is_valid_element(i)) continue;\n"
    "        const auto co = page_scores.get_multidimensional_index(i); page_scores[i] *= key_scales[co[0]]; } }\n"
    "    page_scores.store(st.slice<N,M>(0,0));\n"
    "    if (thread_index == 0) atomic_store_explicit(rescale, 0u, memory_order_relaxed);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, ScaleInSoftmax, Quantized>(\n"
    "      scores, probabilities, row_max, row_sum, previous_scale, rescale,\n"
    "      reinterpret_cast<device const float4 *>(key_scales), reinterpret_cast<device const float4 *>(value_scales),\n"
    "      token_start, visible_tokens, committed_tokens, active_rows, thread_index);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (atomic_load_explicit(rescale, memory_order_relaxed)) {\n"
    "      for (ushort i = 0; i < running.get_capacity(); ++i) { if (!running_full && !running.is_valid_element(i)) continue;\n"
    "        auto co = running.get_multidimensional_index(i); running[i] *= previous_scale[co[1]]; } }\n"
    "    auto vs = vt.template slice<N,D>(0,0); pv.run(p0, vs, running);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "  auto target = tensor(partials + slot * M * D, dextents<int,2>{D,M}, array<int,2>{1,D});\n"
    "  running.store(target.slice<D,M>(0,0));\n"
    "  if (thread_index < M) { statistics[(slot*M + thread_index)*2] = row_max[thread_index]; statistics[(slot*M + thread_index)*2 + 1] = row_sum[thread_index]; }\n"
    "}\n";

/* splash_q8_attention_reduce_row(_shared) — fedele da paged_attention_tile.h (righe 317-389). */
static const char splash_paged_attention_reduce_metal[] =
    "template <uint QueryHeadsPerKVHead, uint RowsPerTile>\n"
    "inline void splash_q8_attention_reduce_row(device const float *partials, device const float *statistics,\n"
    "    device bfloat *tile_output, uint committed_tokens, uint active_rows, uint splits, ulong head_slot,\n"
    "    uint fused_row, uint thread_index) {\n"
    "  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead; constexpr ushort D = 256;\n"
    "  uint query_row = fused_row / QueryHeadsPerKVHead; uint visible_tokens = committed_tokens + active_rows;\n"
    "  uint pages = (visible_tokens + 32u - 1u) / 32u; uint per_split = (pages + splits - 1u) / splits;\n"
    "  uint written = (pages + per_split - 1u) / per_split; float value = 0.0f;\n"
    "  if (query_row < active_rows) {\n"
    "    float maximum = -INFINITY;\n"
    "    for (uint split = 0; split < written; ++split) maximum = max(maximum, statistics[((head_slot + split)*M + fused_row)*2]);\n"
    "    float numerator = 0.0f; float denominator = 0.0f;\n"
    "    for (uint split = 0; split < written; ++split) { ulong stat = ((head_slot + split)*M + fused_row)*2;\n"
    "      float weight = fast::exp(statistics[stat] - maximum);\n"
    "      numerator += weight * partials[((head_slot + split)*M + fused_row)*D + thread_index];\n"
    "      denominator += weight * statistics[stat + 1]; }\n"
    "    value = denominator > 0.0f ? numerator / denominator : 0.0f; }\n"
    "  tile_output[fused_row * D + thread_index] = bfloat(value);\n"
    "}\n"
    "template <uint QueryHeadsPerKVHead, uint RowsPerTile>\n"
    "inline void splash_q8_attention_reduce_row_shared(device const float *partials, device const float *statistics,\n"
    "    device bfloat *tile_output, uint committed_tokens, uint active_rows, uint splits, ulong head_slot,\n"
    "    uint fused_row, uint thread_index, threadgroup float *weights, threadgroup float *group_values) {\n"
    "  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead; constexpr uint D = 256;\n"
    "  if (fused_row / QueryHeadsPerKVHead >= active_rows) { tile_output[fused_row*D + thread_index] = bfloat(0.0f); return; }\n"
    "  const uint pages = (committed_tokens + active_rows + 32u - 1u) / 32u; const uint per_split = (pages + splits - 1u) / splits;\n"
    "  const uint written = (pages + per_split - 1u) / per_split;\n"
    "  const uint lane = thread_index % 32u, sg = thread_index / 32u;\n"
    "  const ulong stat = ((head_slot + thread_index)*M + fused_row)*2;\n"
    "  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;\n"
    "  const float group_maximum = simd_max(maximum); if (lane == 0) group_values[sg] = group_maximum;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  const float row_maximum = simd_max(lane < 8 ? group_values[lane] : -INFINITY);\n"
    "  if (thread_index < written) weights[thread_index] = fast::exp(maximum - row_maximum);\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float numerator = 0.0f, denominator = 0.0f;\n"
    "  for (uint split = 0; split < written; ++split) { const float weight = weights[split];\n"
    "    const ulong st2 = ((head_slot + split)*M + fused_row)*2;\n"
    "    numerator += weight * partials[((head_slot + split)*M + fused_row)*D + thread_index];\n"
    "    denominator += weight * statistics[st2 + 1]; }\n"
    "  tile_output[fused_row * D + thread_index] = bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);\n"
    "}\n";

/* restano le ENTRY/dispatch dei kernel [[kernel]] (Splash: PagedAttention.mm le
 * lancia; qui vanno montate via mlx_fast_metal_kernel) — GAP-2b/entry. */

/* entry PREFILL — fedeli da kernels/prefill/attention_q8.metal */
static const char splash_prefill_attention_entries_metal[] =
    "template <uint KVHeads, uint QueryHeadsPerKVHead, bool ScaleInSoftmax, typename CacheElement>\n"
    "inline void splash_prefill_attention_split_phase(device bfloat *queries, device CacheElement *cache_keys,\n"
    "    device const float *key_scales_buffer, device CacheElement *cache_values, device const float *value_scales_buffer,\n"
    "    device float *partials, device float *statistics, device const uint *page_table,\n"
    "    constant SplashQ8PrefillAttentionParams &params, threadgroup float *scores, threadgroup bfloat *probabilities,\n"
    "    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,\n"
    "    threadgroup atomic_uint *rescale, uint3 group, uint thread_index) {\n"
    "  constexpr uint D = SPLASH_Q8_HEAD_DIM; uint kv_head = group.x; uint split = group.z; uint tile = group.y;\n"
    "  uint tile_start = tile * SPLASH_PREFILL_TILE_ROWS;\n"
    "  if (!splash_q8_prefill_attention_contract_valid(params) || kv_head >= KVHeads || split >= params.split_count || tile_start >= params.rows) return;\n"
    "  uint active_rows = min(SPLASH_PREFILL_TILE_ROWS, params.rows - tile_start);\n"
    "  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) * QueryHeadsPerKVHead * D;\n"
    "  ulong slot = (ulong(tile) * KVHeads + kv_head) * params.split_count + split;\n"
    "  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead, SPLASH_PREFILL_TILE_ROWS, ScaleInSoftmax>(\n"
    "    queries + tile_offset, cache_keys, key_scales_buffer, cache_values, value_scales_buffer, page_table, kv_head,\n"
    "    params.committed_tokens + tile_start, active_rows, params.split_count, split, partials, statistics, slot,\n"
    "    scores, probabilities, row_max, row_sum, previous_scale, rescale, thread_index);\n"
    "}\n"
    "template <uint KVHeads, uint QueryHeadsPerKVHead>\n"
    "inline void splash_q8_prefill_attention_reduce_phase(device const float *partials, device const float *statistics,\n"
    "    device bfloat *output, constant SplashQ8PrefillAttentionParams &params, uint3 group, uint thread_index) {\n"
    "  constexpr ushort M = SPLASH_PREFILL_TILE_ROWS * QueryHeadsPerKVHead; constexpr ushort D = SPLASH_Q8_HEAD_DIM;\n"
    "  uint kv_head = group.x; uint fused_row = group.y; uint tile = group.z; uint tile_start = tile * SPLASH_PREFILL_TILE_ROWS;\n"
    "  if (!splash_q8_prefill_attention_contract_valid(params) || kv_head >= KVHeads || fused_row >= M || thread_index >= D || tile_start >= params.rows) return;\n"
    "  uint active_rows = min(SPLASH_PREFILL_TILE_ROWS, params.rows - tile_start);\n"
    "  if (fused_row / QueryHeadsPerKVHead >= active_rows) return;\n"
    "  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) * QueryHeadsPerKVHead * D;\n"
    "  splash_q8_attention_reduce_row<QueryHeadsPerKVHead, SPLASH_PREFILL_TILE_ROWS>(\n"
    "    partials, statistics, output + tile_offset, params.committed_tokens + tile_start, active_rows, params.split_count,\n"
    "    (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row, thread_index);\n"
    "}\n"
    "#define Q8_PREFILL_SPLIT(Name, Heads, Group, ScaleInSoftmax) \\\n"
    "  kernel void Name(device bfloat *queries [[buffer(0)]], device int8_t *cache_keys [[buffer(1)]], \\\n"
    "    device const float *key_scales_buffer [[buffer(2)]], device int8_t *cache_values [[buffer(3)]], \\\n"
    "    device const float *value_scales_buffer [[buffer(4)]], device float *partials [[buffer(5)]], \\\n"
    "    device float *statistics [[buffer(6)]], device const uint *page_table [[buffer(7)]], \\\n"
    "    constant SplashQ8PrefillAttentionParams &params [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]], \\\n"
    "    uint thread_index [[thread_index_in_threadgroup]]) { \\\n"
    "    constexpr uint M = Group * SPLASH_PREFILL_TILE_ROWS; constexpr uint N = SPLASH_KV_PAGE_TOKENS; \\\n"
    "    alignas(16) threadgroup float scores[M*N]; alignas(16) threadgroup bfloat probabilities[M*N]; \\\n"
    "    threadgroup float row_max[M]; threadgroup float row_sum[M]; threadgroup float previous_scale[M]; \\\n"
    "    threadgroup atomic_uint rescale; \\\n"
    "    splash_prefill_attention_split_phase<Heads, Group, ScaleInSoftmax>(queries, cache_keys, key_scales_buffer, \\\n"
    "      cache_values, value_scales_buffer, partials, statistics, page_table, params, scores, probabilities, \\\n"
    "      row_max, row_sum, previous_scale, &rescale, group, thread_index); }\n"
    "Q8_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, true)\n"
    "Q8_PREFILL_SPLIT(prefill_attention_q8_split_cooperative_scale, 4, 6, false)\n"
    "Q8_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, true)\n"
    "Q8_PREFILL_SPLIT(prefill_attention_q8_split_cooperative_scale_kv2_g8, 2, 8, false)\n"
    "#undef Q8_PREFILL_SPLIT\n";

/* ancora da trascrivere: le entry del DECODE/VERIFY (kernels/decode/attention_q8.metal)
 * e il contract `splash_q8_prefill_attention_contract_valid` (dai .h importati) — GAP-2b. */

/* entry VERIFY — fedeli da kernels/decode/attention_q8.metal */
static const char splash_verify_attention_entries_metal[] =
    "struct SplashQ8VerifyTile { device bfloat *queries; device const uint *page_table; ulong slot;\n"
    "  uint kv_head; uint split; uint splits; uint committed_tokens; uint active_rows; bool active; };\n"
    "template <uint KVHeads, uint QueryHeadsPerKVHead>\n"
    "inline SplashQ8VerifyTile splash_q8_verify_attention_tile_at(device bfloat *queries,\n"
    "  device const uint *page_table0, device const uint *page_table1, device const uint *page_table2,\n"
    "  device const uint *page_table3, constant SplashQ8VerifyAttentionParams *params, uint3 group) {\n"
    "  constexpr uint D = SPLASH_Q8_HEAD_DIM; SplashQ8VerifyTile tile{};\n"
    "  uint kv_head = group.x; uint split = group.y; uint batch = group.z;\n"
    "  constant SplashQ8VerifyAttentionParams &lp = params[batch];\n"
    "  if (!splash_q8_verify_attention_contract_valid(lp) || kv_head >= KVHeads || split >= lp.split_count) return tile;\n"
    "  ulong group_stride = ulong(lp.chunk_stride) * QueryHeadsPerKVHead * D;\n"
    "  tile.queries = queries + (ulong(batch) * KVHeads + kv_head) * group_stride;\n"
    "  tile.page_table = batch == 0 ? page_table0 : (batch == 1 ? page_table1 : (batch == 2 ? page_table2 : page_table3));\n"
    "  tile.slot = (ulong(batch) * KVHeads + kv_head) * lp.slot_splits + split;\n"
    "  tile.kv_head = kv_head; tile.split = split; tile.splits = lp.split_count;\n"
    "  tile.committed_tokens = lp.committed_tokens; tile.active_rows = lp.active_rows; tile.active = true; return tile; }\n"
    "#define Q8_VERIFY_SPLIT_SIGNATURE(Name) \\\n"
    "  kernel void Name(device bfloat *queries [[buffer(0)]], device int8_t *cache_keys [[buffer(1)]], \\\n"
    "    device const float *key_scales_buffer [[buffer(2)]], device int8_t *cache_values [[buffer(3)]], \\\n"
    "    device const float *value_scales_buffer [[buffer(4)]], device float *partials [[buffer(5)]], \\\n"
    "    device float *statistics [[buffer(6)]], device const uint *page_table0 [[buffer(7)]], \\\n"
    "    device const uint *page_table1 [[buffer(8)]], device const uint *page_table2 [[buffer(9)]], \\\n"
    "    device const uint *page_table3 [[buffer(10)]], constant SplashQ8VerifyAttentionParams *params [[buffer(11)]], \\\n"
    "    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]])\n"
    "#define Q8_VERIFY_SCRATCH(Group) \\\n"
    "  constexpr uint M = Group * SPLASH_VERIFY_TILE_ROWS; constexpr uint N = SPLASH_KV_PAGE_TOKENS; \\\n"
    "  alignas(16) threadgroup float scores[M*N]; alignas(16) threadgroup bfloat probabilities[M*N]; \\\n"
    "  threadgroup float row_max[M]; threadgroup float row_sum[M]; threadgroup float previous_scale[M]; threadgroup atomic_uint rescale;\n"
    "#define Q8_VERIFY_TILE_AT(Heads, Group) \\\n"
    "  const SplashQ8VerifyTile tile = splash_q8_verify_attention_tile_at<Heads, Group>(queries, page_table0, page_table1, page_table2, page_table3, params, group); \\\n"
    "  if (!tile.active) return;\n"
    "#define Q8_VERIFY_SPLIT(Name, Heads, Group, ScaleInSoftmax) \\\n"
    "  Q8_VERIFY_SPLIT_SIGNATURE(Name) { Q8_VERIFY_SCRATCH(Group) Q8_VERIFY_TILE_AT(Heads, Group) \\\n"
    "    splash_paged_attention_tile<Heads, Group, SPLASH_VERIFY_TILE_ROWS, ScaleInSoftmax>(tile.queries, cache_keys, \\\n"
    "      key_scales_buffer, cache_values, value_scales_buffer, tile.page_table, tile.kv_head, tile.committed_tokens, \\\n"
    "      tile.active_rows, tile.splits, tile.split, partials, statistics, tile.slot, scores, probabilities, \\\n"
    "      row_max, row_sum, previous_scale, &rescale, thread_index); }\n"
    "Q8_VERIFY_SPLIT(verify_attention_q8_split, 4, 6, true)\n"
    "Q8_VERIFY_SPLIT(verify_attention_q8_split_cooperative_scale, 4, 6, false)\n"
    "Q8_VERIFY_SPLIT(verify_attention_q8_split_kv2_g8, 2, 8, true)\n"
    "Q8_VERIFY_SPLIT(verify_attention_q8_split_cooperative_scale_kv2_g8, 2, 8, false)\n"
    "#undef Q8_VERIFY_SPLIT\n";

/* ── 6b) Costanti di geometria — VERBATIM da metal/abi/ExecutionGeometry.h ── */
#define SPLASH_DRAFT_QUERY_ROWS 8u
#define SPLASH_DRAFT_PROPOSAL_TOKENS 7u
#define SPLASH_TARGET_VERIFY_ROWS 8u
#define SPLASH_MAXIMUM_CONTEXT_TOKENS 262144u
#define SPLASH_SPECULATIVE_SCRATCH_TOKENS (SPLASH_TARGET_VERIFY_ROWS - 1u)
#define SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS (SPLASH_MAXIMUM_CONTEXT_TOKENS + SPLASH_SPECULATIVE_SCRATCH_TOKENS)
#define SPLASH_MAXIMUM_BATCH_WIDTH 4u
#define SPLASH_PREFILL_TOKEN_BUDGET 2048u
#define SPLASH_DRAFT_SLIDING_WINDOW 2048u
#define SPLASH_TARGET_KV_BLOCK_TOKENS 32u          /* == SPLASH_KV_PAGE_TOKENS */
#define SPLASH_PREFILL_ATTENTION_TILE_ROWS 8u
#define SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
#define SPLASH_VERIFY_ATTENTION_SPLITS 32u
#define SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT 16u
#define SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#define SPLASH_ALLOCATION_EXTENT_TARGET_BYTES (128ull * 1024ull * 1024ull)
/* da kernels/common/q8_paging.h: */
#define SPLASH_Q8_HEAD_DIM 256u
#define SPLASH_Q8_PAGE_TOKENS SPLASH_TARGET_KV_BLOCK_TOKENS

/* ── 6) Contract — TRASCRITTI fedeli da paged_attention_tile.h / paged_store_row.h ── */
static inline bool splash_q8_prefill_attention_contract_valid(const struct SplashQ8PrefillAttentionParams* p) {
    return p->rows > 0 && p->rows <= SPLASH_PREFILL_TOKEN_BUDGET &&
           p->chunk_stride >= p->rows && p->chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET &&
           p->chunk_stride % SPLASH_KV_PAGE_TOKENS == 0 &&
           p->page_table_entries >= (p->committed_tokens + p->rows + SPLASH_KV_PAGE_TOKENS - 1) / SPLASH_KV_PAGE_TOKENS &&
           p->physical_page_count > 0 && p->split_count > 0 &&
           p->split_count <= SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS &&
           (unsigned long)p->committed_tokens + p->rows <= (unsigned long)SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS &&
           p->reserved0 == 0 && p->reserved1 == 0;
}
static inline bool splash_q8_verify_attention_contract_valid(const struct SplashQ8VerifyAttentionParams* p) {
    return p->active_rows > 0 && p->active_rows <= SPLASH_TARGET_VERIFY_ROWS &&
           p->chunk_stride % SPLASH_KV_PAGE_TOKENS == 0 && p->chunk_stride >= p->active_rows &&
           p->page_table_entries >= (p->committed_tokens + p->active_rows + SPLASH_KV_PAGE_TOKENS - 1) / SPLASH_KV_PAGE_TOKENS &&
           p->physical_page_count > 0 &&
           (unsigned long)p->committed_tokens + p->active_rows <= (unsigned long)SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS &&
           p->split_count > 0 && p->split_count <= SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS &&
           p->slot_splits >= p->split_count && p->slot_splits <= SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS &&
           p->reserved2 == 0;
}
static inline bool splash_chunk_contract_valid(const struct SplashChunkedPrefillParams* p) {
    unsigned required_pages = (p->committed_tokens + p->chunk_tokens + SPLASH_KV_PAGE_TOKENS - 1) / SPLASH_KV_PAGE_TOKENS;
    return p->committed_tokens <= SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS &&
           p->chunk_tokens > 0 && p->chunk_tokens <= SPLASH_PREFILL_TOKEN_BUDGET &&
           p->committed_tokens + p->chunk_tokens <= SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS &&
           p->chunk_stride >= p->chunk_tokens && p->chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET &&
           p->chunk_stride % SPLASH_KV_PAGE_TOKENS == 0 && p->page_table_entries >= required_pages &&
           p->physical_page_count > 0 && p->reserved0 == 0 && p->reserved1 == 0 && p->reserved2 == 0;
}
static inline unsigned splash_attention_pages(unsigned visible_tokens) {
    return (visible_tokens + SPLASH_KV_PAGE_TOKENS - 1) / SPLASH_KV_PAGE_TOKENS;
}
static inline unsigned splash_attention_pages_per_split(unsigned pages, unsigned splits) {
    return (pages + splits - 1) / splits;
}

/* (costanti di geometria spostate sopra i contract) */

/* ── 7) Pianificazione host — TRASCRITTA da ops/PagedAttention.hpp ──────────── */
#define kQ8VerifyMaximumRows    SPLASH_TARGET_VERIFY_ROWS
#define kQ8VerifySplits         SPLASH_VERIFY_ATTENTION_SPLITS
#define kQ8VerifyMaximumSplits  SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS
#define kQ8VerifyPagesPerSplit  SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT
#define kChunkedPrefillMaximumRows SPLASH_PREFILL_TOKEN_BUDGET
#define kQ8PrefillAttentionTileRows SPLASH_PREFILL_ATTENTION_TILE_ROWS
#define kPageTokens SPLASH_KV_PAGE_TOKENS
#define kMaximumPhysicalTokens SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS

static inline unsigned long splash_min_u64(unsigned long a, unsigned long b){ return a < b ? a : b; }
static inline unsigned long splash_max_u64(unsigned long a, unsigned long b){ return a > b ? a : b; }

/* q8VerifyAttentionSplits — fedele */
static inline uint32_t q8VerifyAttentionSplits(uint32_t baseSplits, uint32_t committedTokens, uint32_t activeRows) {
    const unsigned long visible = (unsigned long)committedTokens + activeRows;
    const unsigned long pages = (visible + kPageTokens - 1) / kPageTokens;
    const unsigned long scaled = (pages + kQ8VerifyPagesPerSplit - 1) / kQ8VerifyPagesPerSplit;
    return (uint32_t)splash_min_u64(splash_max_u64(baseSplits, scaled), kQ8VerifyMaximumSplits);
}
/* q8VerifyAttentionParams — fedele (default; il plan imposta i due split per lane) */
static inline struct SplashQ8VerifyAttentionParams q8VerifyAttentionParams(
    uint32_t committedTokens, uint32_t activeRows, uint32_t chunkStride,
    uint32_t pageTableEntries, uint32_t physicalPageCount) {
    struct SplashQ8VerifyAttentionParams p = { committedTokens, activeRows, chunkStride,
        pageTableEntries, physicalPageCount, kQ8VerifySplits, kQ8VerifySplits, 0 };
    return p;
}
/* validazione — fedele (ritorna NULL se ok, altrimenti la stringa d'errore come in Splash) */
static inline const char* q8VerifyAttentionValidationError(const struct SplashQ8VerifyAttentionParams* params) {
    if (!params->active_rows || params->active_rows > kQ8VerifyMaximumRows) return "active_rows_out_of_range";
    if (params->chunk_stride < params->active_rows || params->chunk_stride % kPageTokens) return "chunk_stride_invalid";
    const unsigned long visible = (unsigned long)params->committed_tokens + params->active_rows;
    if (visible > kMaximumPhysicalTokens) return "context_out_of_range";
    const unsigned long requiredPages = (visible + kPageTokens - 1) / kPageTokens;
    if (params->page_table_entries < requiredPages) return "page_table_too_short";
    if (!params->physical_page_count) return "physical_page_pool_empty";
    if (!params->split_count || params->split_count > kQ8VerifyMaximumSplits) return "split_count_invalid";
    if (params->slot_splits < params->split_count || params->slot_splits > kQ8VerifyMaximumSplits) return "slot_splits_invalid";
    if (params->reserved2) return "reserved_fields_nonzero";
    return NULL;
}
static inline uint32_t prefillAttentionTiles(uint32_t rows) {
    return (rows + kQ8PrefillAttentionTileRows - 1) / kQ8PrefillAttentionTileRows;
}

/* ── 8) Storage — TRASCRITTO fedele da ops/PagedKv.hpp (Layout) e PageStorage.hpp ── */
/* LayerStorage: device keyData/keyScales/valueData/valueScales + format. */
struct SplashLayerStorage { void* keyData; void* keyScales; void* valueData; void* valueScales; uint32_t format; };

/* Layout: byte-count geometry — fedele da PagedKv.hpp (Layout::*) */
static inline uint32_t splash_elements_per_scale(uint32_t fmt, uint32_t headDim) {
    return fmt == SPLASH_KV_FORMAT_INT8 ? headDim : 0u;   /* INT8: head_dim; BF16: 0 */
}
static inline uint64_t splash_elements_per_layer_page(uint32_t kvHeads, uint32_t headDim) {
    return (uint64_t)SPLASH_KV_PAGE_TOKENS * kvHeads * headDim;
}
static inline uint64_t splash_scales_per_tensor_layer_page(uint32_t fmt, uint32_t kvHeads) {
    return fmt == SPLASH_KV_FORMAT_INT8 ? (uint64_t)SPLASH_KV_PAGE_TOKENS * kvHeads : 0u;
}
static inline uint64_t splash_data_bytes_per_layer_page(uint32_t fmt, uint32_t kvHeads, uint32_t headDim) {
    return splash_elements_per_layer_page(kvHeads, headDim) * (fmt == SPLASH_KV_FORMAT_INT8 ? 1u : 2u);
}
static inline uint64_t splash_scale_bytes_per_layer_page(uint32_t fmt, uint32_t kvHeads) {
    return splash_scales_per_tensor_layer_page(fmt, kvHeads) * sizeof(float);
}
static inline uint64_t splash_bytes_per_layer_page(uint32_t fmt, uint32_t kvHeads, uint32_t headDim) {
    return 2ull * (splash_data_bytes_per_layer_page(fmt, kvHeads, headDim) + splash_scale_bytes_per_layer_page(fmt, kvHeads));
}
static inline uint64_t splash_bytes_per_model_page(uint32_t fmt, uint32_t layers, uint32_t kvHeads, uint32_t headDim) {
    return (uint64_t)layers * splash_bytes_per_layer_page(fmt, kvHeads, headDim);
}

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
