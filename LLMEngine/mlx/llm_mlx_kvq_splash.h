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

/* ── Quant Splash-style via op MLX (fallback dello store kernel) ── */
static int llm_mlx_kvq_splash_quant(mlx_array* kq, mlx_array* vq, mlx_array* sc,
                                    mlx_array k, mlx_array v, mlx_stream s) {
    mlx_array ka = mlx_array_new(), va = mlx_array_new();
    mlx_array km = mlx_array_new(), vm = mlx_array_new(), mx = mlx_array_new();
    mlx_array sca = mlx_array_new(), ksc = mlx_array_new(), vsc = mlx_array_new();
    mlx_array kf = mlx_array_new(), vf = mlx_array_new();
    mlx_array kqf = mlx_array_new(), vqf = mlx_array_new();
    bool ok = mlx_abs(&ka, k, s) == 0 && mlx_abs(&va, v, s) == 0;
    if (ok) ok = mlx_max_axis(&km, ka, -1, /*keepdims=*/true, s) == 0 &&
                 mlx_max_axis(&vm, va, -1, true, s) == 0;
    if (ok) ok = mlx_maximum(&mx, km, vm, s) == 0;
    if (ok) ok = mlx_divide(&sca, mx, mlx_array_new_float(127.0f), s) == 0;
    if (ok) ok = mlx_astype(&kf, k, MLX_FLOAT32, s) == 0 && mlx_astype(&vf, v, MLX_FLOAT32, s) == 0;
    if (ok) ok = mlx_multiply(&ksc, kf, mlx_array_new_float(127.0f), s) == 0;
    if (ok) ok = mlx_divide(&kqf, ksc, mx, s) == 0;         /* k*127/max */
    if (ok) ok = mlx_multiply(&vsc, vf, mlx_array_new_float(127.0f), s) == 0;
    if (ok) ok = mlx_divide(&vqf, vsc, mx, s) == 0;
    mlx_array kr = mlx_array_new(), vr = mlx_array_new();
    mlx_array kr2 = mlx_array_new(), vr2 = mlx_array_new();
    mlx_array lo = mlx_array_new_float(-127.0f), hi = mlx_array_new_float(127.0f);
    if (ok) ok = mlx_round(&kr, kqf, 0, s) == 0 && mlx_round(&vr, vqf, 0, s) == 0;
    if (ok) ok = mlx_clip(&kr2, kr, lo, hi, s) == 0 && mlx_clip(&vr2, vr, lo, hi, s) == 0;
    if (ok) ok = mlx_astype(kq, kr2, MLX_INT8, s) == 0 && mlx_astype(vq, vr2, MLX_INT8, s) == 0;
    if (ok) mlx_array_set(sc, sca);
    mlx_array_free(kr); mlx_array_free(vr); mlx_array_free(kr2); mlx_array_free(vr2);
    mlx_array_free(lo); mlx_array_free(hi);
    mlx_array_free(ka); mlx_array_free(va); mlx_array_free(km); mlx_array_free(vm); mlx_array_free(mx);
    mlx_array_free(sca); mlx_array_free(ksc); mlx_array_free(vsc);
    mlx_array_free(kf); mlx_array_free(vf); mlx_array_free(kqf); mlx_array_free(vqf);
    return ok ? 0 : -1;
}

/* Dequant Splash-style: x = q * scale (sc broadcast su hd, [1,nkv,n,1]). */
static int llm_mlx_kvq_splash_dequant(mlx_array* x, mlx_array q, mlx_array sc, mlx_stream s) {
    mlx_array qf = mlx_array_new();
    bool ok = mlx_astype(&qf, q, MLX_FLOAT32, s) == 0;
    if (ok) ok = mlx_multiply(x, qf, sc, s) == 0;
    mlx_array_free(qf);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_KVQ_SPLASH_H */
