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

/* ── Store kernel MLX (one-pass, come splash_store_kv_row) ─────────────────────
 * UN threadgroup per (kv-head, token): i thread sono le dimensioni; la scala
 * simmetrica e' max(|k|,|v|)/127 ridotta sul head_dim (threadgroup tree). Scrive
 * int8 + la scala. NIENTE op MLX quant (niente grafo/intermedi): e' la forma di
 * Splash. In: k,v [1,nkv,n,hd] fp. Out: kq,vq [1,nkv,n,hd] int8, sc [1,nkv,n] f32. */
static const char llm_kvq_store_src[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void llm_kvq_store(device const T* k [[buffer(0)]], device const T* v [[buffer(1)]],\n"
    "  device char* kq [[buffer(2)]], device char* vq [[buffer(3)]], device float* sc [[buffer(4)]],\n"
    "  uint tg [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]],\n"
    "  uint tgsz [[threads_per_threadgroup]], threadgroup float* sh [[threadgroup(0)]])\n"
    "{\n"
    "  uint hd = k_shape[k_ndim-1];\n"
    "  float km = lane < hd ? fabs((float)k[tg*hd+lane]) : 0.0f;\n"
    "  float vm = lane < hd ? fabs((float)v[tg*hd+lane]) : 0.0f;\n"
    "  sh[lane] = fmax(km, vm);\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  for (uint s = tgsz/2; s > 0; s >>= 1) { if (lane < s) sh[lane] = fmax(sh[lane], sh[lane+s]);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "  float mx = sh[0]; float scale = mx == 0.0f ? 0.0f : mx / 127.0f;\n"
    "  if (lane < hd) {\n"
    "    kq[tg*hd+lane] = (char)(mx == 0.0f ? 0 : clamp((int)rint((double)k[tg*hd+lane] * 127.0 / (double)mx), -127, 127));\n"
    "    vq[tg*hd+lane] = (char)(mx == 0.0f ? 0 : clamp((int)rint((double)v[tg*hd+lane] * 127.0 / (double)mx), -127, 127));\n"
    "  }\n"
    "  if (lane == 0) sc[tg] = scale;\n"
    "}\n";

static mlx_fast_metal_kernel g_kvq_store_k; static int g_kvq_store_built = 0;
static int llm_mlx_kvq_splash_store_kernel(void) {
    if (g_kvq_store_built) return g_kvq_store_k.ctx ? 0 : -1;
    g_kvq_store_built = 1;
    const char* hdr = "#include <metal_stdlib>\nusing namespace metal;\n";
    mlx_vector_string iv = mlx_vector_string_new();
    mlx_vector_string_append_value(iv, "k"); mlx_vector_string_append_value(iv, "v");
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, "kq"); mlx_vector_string_append_value(ov, "vq");
    mlx_vector_string_append_value(ov, "sc");
    g_kvq_store_k = mlx_fast_metal_kernel_new("llm_kvq_store", iv, ov, llm_kvq_store_src, hdr,
                                              /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    return g_kvq_store_k.ctx ? 0 : -1;
}

/* k,v [1,nkv,n,hd] fp. Ritorna kq,vq int8 [1,nkv,n,hd] + sc f32 [1,nkv,n,1]. */
static int llm_mlx_kvq_splash_store(mlx_array* kq, mlx_array* vq, mlx_array* sc,
                                    mlx_array k, mlx_array v, int hd, mlx_stream s) {
    if (llm_mlx_kvq_splash_store_kernel() != 0) return -1;
    int nkv = (int)mlx_array_shape(k)[1], n = (int)mlx_array_shape(k)[2];
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, k); mlx_vector_array_append_value(in, v);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx_array_dtype(k));
    int kqsh[3] = {nkv, n, hd}, scsh[2] = {nkv, n};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, kqsh, 3, MLX_INT8);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, kqsh, 3, MLX_INT8);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, scsh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, nkv * n, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, hd, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_kvq_store_k, in, cfg, s);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 3;
    if (ok) {
        mlx_array a0 = mlx_array_new(), a1 = mlx_array_new(), a2 = mlx_array_new();
        ok = mlx_vector_array_get(&a0, outs, 0) == 0 && mlx_vector_array_get(&a1, outs, 1) == 0 &&
             mlx_vector_array_get(&a2, outs, 2) == 0;
        if (ok) { mlx_array_set(kq, a0); mlx_array_set(vq, a1);
                  /* sc -> [nkv,n,1] */
                  int sh3[3] = {nkv, n, 1}; mlx_array sr = mlx_array_new();
                  if (mlx_reshape(&sr, a2, sh3, 3, s) == 0) mlx_array_set(sc, sr);
                  mlx_array_free(sr); }
        mlx_array_free(a0); mlx_array_free(a1); mlx_array_free(a2);
    }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in); mlx_vector_array_free(outs);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_KVQ_SPLASH_H */
