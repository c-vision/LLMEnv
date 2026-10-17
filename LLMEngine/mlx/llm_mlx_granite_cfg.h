/* llm_mlx_granite_cfg.h — config SPECIFICA della famiglia `granite`
 * (granite / granitemoehybrid / granite_hybrid). Refactor Fase B: campi per-famiglia fuori dal
 * monolite cfg (LLMEngine/mlx/llm_mlx_internal.h). Accesso: `m->cfg.granite.<campo>`. */
#ifndef LLM_MLX_GRANITE_CFG_H
#define LLM_MLX_GRANITE_CFG_H

typedef struct {
    float residual_multiplier;  /* scala del residuo su mixer e mlp (es. 0.22) */
    float attn_multiplier;      /* scale SDPA (es. 1/64 = 0.015625) */
    int   ssm_n_heads;          /* teste del mixer Mamba2 (mamba_n_heads, distinte da n_heads attn) */
    int   ssm_head_dim;         /* head_dim del mixer Mamba2 (mamba_d_head) */
    int   attn_no_rope;         /* 1 = attention SENZA RoPE (position_embedding_type "nope") */
} LlmCfgGranite;

#endif /* LLM_MLX_GRANITE_CFG_H */
