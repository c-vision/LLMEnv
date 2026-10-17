/* llm_mlx_hybrid_cfg.h — config SPECIFICA delle famiglie ibride non-Mamba
 * (jamba: dispatch layer attn via periodo/offset; olmo_hybrid: GDN con beta in [0,2]).
 * Refactor Fase B: campi fuori dal monolite. Accesso: `m->cfg.hybrid.<campo>`. */
#ifndef LLM_MLX_HYBRID_CFG_H
#define LLM_MLX_HYBRID_CFG_H

typedef struct {
    int attn_layer_period;   /* jamba: ogni N layer e' attenzione (0 = tutti attn) */
    int attn_layer_offset;   /* jamba: offset della prima attn nel periodo */
    int allow_neg_eigval;    /* olmo_hybrid: beta in [0,2] (beta *= 2) invece di [0,1] */
} LlmCfgHybrid;

#endif /* LLM_MLX_HYBRID_CFG_H */
