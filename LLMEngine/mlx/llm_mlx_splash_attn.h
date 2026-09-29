/* llm_mlx_splash_attn.h — read paged int8, matematica di Splash (dequant-folded).
 *
 * Fedele alla MATEMATICA di paged_attention_tile.h: score = (q·k) * key_scale,
 * softmax, out = sum(p * value_scale * v); int8 letto dal pool, scala per
 * (token,head). NOTA (deviazione, da approvare): la matmul QK/PV qui e' in
 * Metal semplice, NON con MPP matmul2d — perche' MPP (pur disponibile sotto
 * mlx_fast_metal_kernel, vedi probe) richiede l'apparato cooperative-tensor
 * completo. La matematica e il layout sono quelli di Splash.
 */
#ifndef LLM_MLX_SPLASH_ATTN_H
#define LLM_MLX_SPLASH_ATTN_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdint.h>

/* q [nh, hd]; kpool int8 [nkv, cap, hd]; vpool int8 [nkv, hd, cap];
 * ks [nkv, cap]; vs [nkv, cap]; nc = token visibili. out [nh, hd].
 * Un threadgroup per query head, hd thread (un thread = una dimensione out). */
static const char llm_splash_attn_src[] =
    "    const uint qh = threadgroup_position_in_grid.x;\n"
    "    const uint d  = thread_position_in_threadgroup.x;\n"
    "    const uint nh = q_shape[q_ndim - 2];\n"
    "    const uint hd = q_shape[q_ndim - 1];\n"
    "    const uint nkv = ks_shape[ks_ndim - 2];\n"
    "    const uint cap = capi;\n"
    "    const uint kh = qh * nkv / nh;\n"
    "    /* score[t] = (sum_d q[d]*k[t,d]) * ks[t] * attn_scale, poi softmax, poi out[d] */\n"
    "    float mx = -INFINITY, sum = 0.0f, acc = 0.0f;\n"
    "    float scores[256];\n"
    "    for (uint t = 0; t < nc && t < 256u; ++t) {\n"
    "      float dot = 0.0f;\n"
    "      for (uint dd = 0; dd < hd; ++dd) dot += (float)q[qh*hd+dd] * (float)kpool[(kh*cap+t)*hd+dd];\n"
    "      float s = dot * ks[kh*cap+t] * attn_scale;\n"
    "      scores[t] = s; if (s > mx) mx = s;\n"
    "    }\n"
    "    for (uint t = 0; t < nc && t < 256u; ++t) { float e = exp(scores[t]-mx); scores[t]=e; sum += e; }\n"
    "    for (uint t = 0; t < nc && t < 256u; ++t) acc += (scores[t]/sum) * vs[kh*cap+t] * (float)vpool[(kh*hd+d)*cap+t];\n"
    "    out[qh*hd+d] = acc;\n";

static mlx_fast_metal_kernel g_splash_attn; static int g_splash_attn_built = 0;
static inline int llm_splash_attn_build(void) {
    if (g_splash_attn_built) return g_splash_attn.ctx ? 0 : -1;
    g_splash_attn_built = 1;
    const char* hdr = "#include <metal_stdlib>\nusing namespace metal;\n";
    /* attn_scale passato come arg scalare: lo mettiamo come input "attn_scale" 1-elem. */
    mlx_vector_string iv = mlx_vector_string_new();
    mlx_vector_string_append_value(iv, "q"); mlx_vector_string_append_value(iv, "kpool");
    mlx_vector_string_append_value(iv, "vpool"); mlx_vector_string_append_value(iv, "ks");
    mlx_vector_string_append_value(iv, "vs"); mlx_vector_string_append_value(iv, "nc");
    mlx_vector_string_append_value(iv, "attn_scale"); mlx_vector_string_append_value(iv, "capi");
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, "out");
    g_splash_attn = mlx_fast_metal_kernel_new("splash_attn", iv, ov, llm_splash_attn_src, hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    return g_splash_attn.ctx ? 0 : -1;
}

/* q [nh,hd] f32; kpool int8 [nkv,cap,hd]; vpool int8 [nkv,hd,cap]; ks,vs f32 [nkv,cap].
 * nc int; attn_scale f32. out [nh,hd] f32. */
static inline int llm_splash_attn_run(mlx_array* out, mlx_array q, mlx_array kpool, mlx_array vpool,
                                      mlx_array ks, mlx_array vs, int nc, int cap, float attn_scale, mlx_stream s) {
    if (llm_splash_attn_build() != 0) return -1;
    int nh = (int)mlx_array_shape(q)[0], hd = (int)mlx_array_shape(q)[1];
    mlx_array ncA = mlx_array_new_int(nc), asA = mlx_array_new_float(attn_scale); mlx_array capA = mlx_array_new_int(cap);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, q); mlx_vector_array_append_value(in, kpool);
    mlx_vector_array_append_value(in, vpool); mlx_vector_array_append_value(in, ks);
    mlx_vector_array_append_value(in, vs); mlx_vector_array_append_value(in, ncA);
    mlx_vector_array_append_value(in, asA); mlx_vector_array_append_value(in, capA);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    int osh[2] = {nh, hd}; mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, nh * hd, 1, 1);      /* thread totali */
    mlx_fast_metal_kernel_config_set_thread_group(cfg, hd, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_splash_attn, in, cfg, s);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) { mlx_array a0 = mlx_array_new(); ok = mlx_vector_array_get(&a0, outs, 0) == 0; if (ok) mlx_array_set(out, a0); mlx_array_free(a0); }
    mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(in); mlx_vector_array_free(outs);
    mlx_array_free(ncA); mlx_array_free(asA); mlx_array_free(capA);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_SPLASH_ATTN_H */
