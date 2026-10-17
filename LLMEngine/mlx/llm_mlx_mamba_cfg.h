/* llm_mlx_mamba_cfg.h — config SPECIFICA della famiglia SSM/mamba
 * (mamba / mamba2 / jamba / falcon_h1 / granite_hybrid usano il mixer Mamba2).
 * Refactor Fase B: campi fuori dal monolite cfg. Accesso: `m->cfg.mamba.<campo>`. */
#ifndef LLM_MLX_MAMBA_CFG_H
#define LLM_MLX_MAMBA_CFG_H

typedef struct {
    int   state_size;    /* Mamba: d_state (es. 16) */
    int   bcdt_rms;      /* falcon_mamba: mixer_norm RMSNorm su delta/B/C */
    float ts_lo, ts_hi;  /* Mamba2: time_step_limit [lo,hi] */
    int   fh1_heads;     /* falcon_h1: mamba_n_heads (distinto da n_heads attn) */
    int   fh1_head_dim;  /* falcon_h1: mamba_d_head */
    int   dt_rank;       /* Mamba: dt_rank = ceil(d_model/16) */
    float ln_eps;        /* Mamba2 gated RMSNorm (o_norm) eps; default 1e-5 */
} LlmCfgMamba;

#endif /* LLM_MLX_MAMBA_CFG_H */
