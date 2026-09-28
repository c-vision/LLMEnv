/* llm_mlx_dpo.h — §D R7: preference optimization (DPO/ORPO) — matematica PURA.
 *
 * Con i log-prob (sommati sul completamento) di chosen/rejected sotto policy e reference:
 *   DPO:  z = beta * ((lp_w - ref_w) - (lp_l - ref_l));   loss = -log sigmoid(z)
 *   ORPO: loss = sft_loss_w + lambda * (-log sigmoid(lp_w - lp_l))   (nessun reference)
 * Espone loss + i gradienti rispetto ai log-prob (per la backward via chain rule): il motore
 * fornisce i log-prob (4 passi: policy/ref su chosen/rejected) e usa questi gradienti.
 * PURO. */
#ifndef LLM_MLX_DPO_H
#define LLM_MLX_DPO_H

#include <math.h>
#include <stdint.h>

static inline double llm_sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
static inline double llm_softplus(double z) { return (z > 30.0) ? z : log1p(exp(z)); }  /* log(1+e^z) stabile */

/* DPO: loss e gradienti (d/d lp_w, d/d lp_l, d/d beta). */
static inline void llm_dpo_loss_grad(double lp_w, double lp_l, double ref_w, double ref_l, double beta,
                                      double* out_loss, double* out_d_lpw, double* out_d_lpl, double* out_d_beta) {
    double a = (lp_w - ref_w) - (lp_l - ref_l);
    double z = beta * a;
    double loss = llm_softplus(-z);              /* -log sigmoid(z) */
    double m = llm_sigmoid(-z);                  /* 1 - sigmoid(z) */
    if (out_loss) *out_loss = loss;
    if (out_d_lpw) *out_d_lpw = -beta * m;       /* d loss/d lp_w */
    if (out_d_lpl) *out_d_lpl = +beta * m;       /* d loss/d lp_l */
    if (out_d_beta) *out_d_beta = -a * m;        /* d loss/d beta */
}

/* ORPO: loss = sft_w + lambda * (-log sigmoid(lp_w - lp_l)). */
static inline void llm_orpo_loss_grad(double sft_w, double lp_w, double lp_l, double lambda,
                                       double* out_loss, double* out_d_lpw, double* out_d_lpl, double* out_d_sft) {
    double z = lp_w - lp_l;
    double pref = llm_softplus(-z);
    double m = llm_sigmoid(-z);
    if (out_loss) *out_loss = sft_w + lambda * pref;
    if (out_d_lpw) *out_d_lpw = -lambda * m;
    if (out_d_lpl) *out_d_lpl = +lambda * m;
    if (out_d_sft) *out_d_sft = 1.0;
}

#endif /* LLM_MLX_DPO_H */
