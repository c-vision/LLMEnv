/* llm_mlx_kvq_splash.h — KV-quant port dalla reference Splash (github.com/incoai/splash).
 *
 * Portato fedelmente da:
 *   runtime/ops/PagedKv.hpp              (Format, Layout, LayoutGuard)
 *   runtime/metal/kernels/common/q8_paging.h        (indici Q8)
 *   runtime/metal/kernels/common/paged_store_row.h  (store: quantizzazione in scrittura)
 *   runtime/metal/kernels/common/paged_attention_tile.h (attenzione paged int8)
 *
 * DIFERENZA CHIAVE vs il path precedente (eliminato, `659db7d`): Splash NON usa
 * `mlx_quantize`. La quantizzazione avviene ONDA-SCRITTURA in un buffer int8
 * preallocato, con UNA scala f32 SIMMETRICA per (token, head) calcolata dal
 * massimo sul head_dim (simd_max). K = token-major, V = dimension-major.
 * L'attenzione applica la scala dopo la matmul (folded nella softmax), quindi
 * il cold resta int8 anche in lettura: niente materializzazione fp, niente
 * intermedi per-layer, niente concat.
 *
 * Formato (Splash `storageFormatName`):
 *   int8  -> "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major"
 *   bf16  -> "bf16_k_token_major_v_dimension_major"
 * Scala: elementsPerScale = head_dim (una f32 per (token,kv-head)); min/max = -127/127.
 */
#ifndef LLM_MLX_KVQ_SPLASH_H
#define LLM_MLX_KVQ_SPLASH_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Formato (port di PagedKv.hpp) ─────────────────────────────────────────── */
typedef enum { LLM_KVQ_FMT_INT8 = 1, LLM_KVQ_FMT_BF16 = 2 } llm_kvq_fmt;

#define LLM_KVQ_QUANT_MIN (-127)
#define LLM_KVQ_QUANT_MAX (127)

/* Page = 32 token (Splash SPLASH_TARGET_KV_BLOCK_TOKENS). */
#define LLM_KVQ_PAGE_TOKENS 32

/* Identita' di formato (port semplificato di LayoutGuard): entra nelle chiavi di
 * policy/cache. `quantization`: 1=int8, 2=bf16. */
typedef struct {
    int quantization;      /* 1 int8 / 2 bf16 */
    int scale_type;        /* 0 none / 2 f32 */
    int key_layout;        /* 1 token-major */
    int value_layout;      /* 1 dimension-major */
    int page_tokens;
    int elements_per_scale; /* = head_dim per int8, 0 per bf16 */
    int attention_layers;
    int kv_heads;
    int head_dimension;
    int quantized_minimum;
    int quantized_maximum;
} llm_kvq_layout;

static inline llm_kvq_layout llm_kvq_make_layout(int layers, int kv_heads, int hd, llm_kvq_fmt fmt) {
    llm_kvq_layout g; memset(&g, 0, sizeof g);
    const bool q = (fmt == LLM_KVQ_FMT_INT8);
    g.quantization = (int)fmt;
    g.scale_type = q ? 2 : 0;
    g.key_layout = 1;
    g.value_layout = 1;
    g.page_tokens = LLM_KVQ_PAGE_TOKENS;
    g.elements_per_scale = q ? hd : 0;
    g.attention_layers = layers;
    g.kv_heads = kv_heads;
    g.head_dimension = hd;
    g.quantized_minimum = q ? LLM_KVQ_QUANT_MIN : 0;
    g.quantized_maximum = q ? LLM_KVQ_QUANT_MAX : 0;
    return g;
}

/* Indici Q8 (port VERBATIM di q8_paging.h, con hd/headTokens parametrizzati). */
static inline uint64_t llm_kvq_key_index(uint32_t page, uint32_t head, uint32_t token,
                                         uint32_t dim, uint32_t kvh, uint32_t page_tokens, uint32_t hd) {
    const uint64_t per_page = (uint64_t)kvh * page_tokens * hd;
    return (uint64_t)page * per_page + ((uint64_t)head * page_tokens + token) * hd + dim;   /* token-major */
}
static inline uint64_t llm_kvq_value_index(uint32_t page, uint32_t head, uint32_t token,
                                           uint32_t dim, uint32_t kvh, uint32_t page_tokens, uint32_t hd) {
    const uint64_t per_page = (uint64_t)kvh * page_tokens * hd;
    return (uint64_t)page * per_page + ((uint64_t)head * hd + dim) * page_tokens + token;   /* dimension-major */
}
static inline uint64_t llm_kvq_scale_index(uint32_t page, uint32_t head, uint32_t token,
                                           uint32_t kvh, uint32_t page_tokens) {
    const uint64_t per_page = (uint64_t)kvh * page_tokens;
    return (uint64_t)page * per_page + (uint64_t)head * page_tokens + token;
}

/* ── Store kernel (port di splash_store_kv_row) ────────────────────────────────
 * UN threadgroup per (token, kv-head): i thread sono le DIMENSIONI. La scala
 * simmetrica e' max|v|/127 ridotta sul head_dim (in Splash, per-simdgroup con
 * hd=256; qui generico: threads = hd, riduzione threadgroup ad albero).
 * Scrive DIRETTAMENTE nei buffer int8 preallocati (K token-major, V dim-major).
 * Input: krow [hd] fp32, vrow [hd] fp32. Output: l'int8 + la scala sono SCRITTI
 * nei buffer passati come input (device) -> serve un kernel "in-place": usiamo
 * un output-writer con l'indice calcolato dall'host via `dst_off`.
 * Args: keys [nkv*n*pagetokens*hd] , kscales [nkv*n*pagetokens], ... (buffer globali). */
static const char llm_kvq_store_src[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void llm_kvq_store(\n"
    "    device const float* krow [[buffer(0)]],\n"
    "    device const float* vrow [[buffer(1)]],\n"
    "    device char*  kdata [[buffer(2)]],\n"
    "    device float* kscale[[buffer(3)]],\n"
    "    device char*  vdata [[buffer(4)]],\n"
    "    device float* vscale[[buffer(5)]],\n"
    "    constant uint& hd [[buffer(6)]],\n"
    "    constant uint& kbase [[buffer(7)]],\n"
    "    constant uint& vbase [[buffer(8)]],\n"
    "    constant uint& sbase [[buffer(9)]],\n"
    "    uint gid [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]],\n"
    "    uint tgsz [[threads_per_threadgroup]],\n"
    "    threadgroup float* sh [[threadgroup(0)]])\n"
    "{\n"
    "    float k = krow[lane]; float v = vrow[lane];\n"
    "    sh[lane] = fmax(fabs(k), fabs(v));\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = tgsz/2; s > 0; s >>= 1) {\n"
    "        if (lane < s) sh[lane] = fmax(sh[lane], sh[lane+s]);\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    float mx = sh[0];\n"
    "    float ks = mx == 0.0f ? 0.0f : mx / 127.0f;\n"
    "    float vs = ks;\n"
    "    int qk = mx == 0.0f ? 0 : clamp((int)rint((double)k * 127.0 / (double)mx), -127, 127);\n"
    "    int qv = mx == 0.0f ? 0 : clamp((int)rint((double)v * 127.0 / (double)mx), -127, 127);\n"
    "    kdata[kbase + lane] = (char)qk;\n"
    "    vdata[vbase + lane] = (char)qv;\n"
    "    if (lane == 0) { kscale[sbase] = ks; vscale[sbase] = vs; }\n"
    "}\n";

#endif /* LLM_MLX_KVQ_SPLASH_H */
