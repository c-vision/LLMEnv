/* llmengine_moe_mla.h — attenzione MLA (Multi-head Latent Attention) di DeepSeek-V2/V3, NON assorbita.
 *
 * Riferimento: mlx_lm/models/deepseek_v2.py — verificato contro l'engine MLX di LLMEnv
 * (`deepseek_v2_attention_block`) che è l'ORACOLO locale per questa implementazione.
 * Schema per token:
 *   xn = rmsnorm(x, attn_norm)
 *   q  = q_proj(xn) -> [H*(qn+qr)]  (o q_a_proj->q_a_layernorm->q_b_proj se q_lora>0)
 *   ckv = kv_a_proj_with_mqa(xn) -> [kvl+qr];  compressed=ckv[:kvl]; kpe=ckv[kvl:]
 *   ckn = rmsnorm(compressed, kv_a_layernorm)
 *   kv = kv_b_proj(ckn) -> [H*(qn+vhd)]; k_nope=kv[:qn] per head; values=kv[qn:] per head
 *   rope(q_pe, kpe) (yarn, interleaved) + mscale; keys=concat(k_nope, kpe_bcast); queries=concat(q_nope,q_pe)
 *   attn = softmax(q·k / sqrt(qn+qr)); ctx[h] = sum_t attn*values[h][t]
 *   out = o_proj(concat(ctx))
 * Cache: keys [n_layers][H][ctx][qn+qr], values [n_layers][H][ctx][vhd]. */
#ifndef LLMENGINE_MOE_MLA_H
#define LLMENGINE_MOE_MLA_H

#include "llmengine_moe_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t hidden, n_heads, q_lora, kv_lora, qk_nope, qk_rope, v_head;
    float eps;
    float rope_theta;
    float yarn_factor;            /* <=1 = nessun yarn */
    int   yarn_orig_max;
    float yarn_beta_fast, yarn_beta_slow, yarn_mscale, yarn_mscale_all_dim;
    uint32_t rope_variant;        /* 0 = half-split, 1 = interleaved (DeepSeek: interleaved) */
} MlaCfg;

typedef struct {
    const float* attn_norm;   /* [hidden] */
    const float* wq;          /* [H*(qn+qr), hidden] (q_lora==0) */
    const float* wq_a;        /* [q_lora, hidden] */
    const float* q_a_norm;    /* [q_lora] */
    const float* wq_b;        /* [H*(qn+qr), q_lora] */
    const float* wkv_a_mqa;   /* [kvl+qr, hidden] */
    const float* kv_a_norm;   /* [kvl] */
    const float* wkv_b;       /* [H*(qn+vhd), kvl] */
    const float* wo;          /* [hidden, H*vhd] */
} MlaLayer;

/* Cache MLA: keys [n_layers][H][ctx][qn+qr], values [n_layers][H][ctx][vhd]. */
typedef struct { float* k; float* v; uint32_t n_layers, n_heads, ctx, qh, vhd; } MlaKv;

/* Un token di UN layer: x[hidden] → out[hidden]. Appende pos nella cache (per layer). */
LlmEngineMoeStatus llmengine_moe_mla_decode(const MlaCfg* cfg, const MlaLayer* ly, uint32_t layer, uint32_t pos,
                                const float* x, float* out, MlaKv* kv);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_MLA_H */
