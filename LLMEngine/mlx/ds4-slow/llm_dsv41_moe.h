/* llm_dsv41_moe.h — DeepSeek V4.1: MoE semantics (Level-3 port, step 4), header-only.
 *
 * Port C delle formule VERIFICABILI di `omlx/patches/deepseek_v41/language.py`:
 *   - Gate.__call__ : raw = x@W.T/gate_temp; score {softmax|sigmoid|sqrtsoftplus};
 *                     +bias; top-k; norm_topk_prob; route_scale.
 *   - Expert.__call__ : SwiGLU con swiglu_limit (gate=min(g,lim); up=clip(u,-lim,lim));
 *                       y = silu(g)*u (*weight)  [poi w2 lineare].
 *   - MoE.__call__ : somma dei K esperti instradati + shared (i pesi sono gia' dentro l'Expert).
 *
 * NON include i pesi (w1/w2/w3) ne' i kernel gather_qmm: quelli richiedono il modello (v. todo).
 * Verifica: `llm_dsv41_moe_test.c` vs riferimento (`tools/gen_dsv41_moe_reference.py`).
 */
#ifndef LLM_DSV41_MOE_H
#define LLM_DSV41_MOE_H

#include <math.h>

#define LLM_DSV41_MOE_MAXE 256

/* score_func (config.py): 0 softmax, 1 sigmoid, 2 sqrtsoftplus (default di riferimento). */
#define LLM_DSV41_SCORE_SOFTMAX 0
#define LLM_DSV41_SCORE_SIGMOID 1
#define LLM_DSV41_SCORE_SQRTSOFTPLUS 2

/* logaddexp(a,0) = softplus(a) = log(e^a + 1). */
static inline float llm_dsv41_softplus(float a) {
    float m = a > 0.0f ? a : 0.0f;
    return m + logf(expf(a - m) + expf(-m));
}

/* Gate: x[D], weight[n_experts*D], bias[n_experts] -> idx[n_act], weights[n_act]. */
static inline void llm_dsv41_gate(const float* x, int D, const float* weight, int n_experts,
                                  const float* bias, float gate_temp, int score_func,
                                  int n_act, int norm_topk_prob, float route_scale,
                                  int* idx_out, float* weights_out) {
    float raw[LLM_DSV41_MOE_MAXE], scores[LLM_DSV41_MOE_MAXE], comb[LLM_DSV41_MOE_MAXE];
    for (int e = 0; e < n_experts; e++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += x[d] * weight[e * D + d];
        raw[e] = dot / gate_temp;
    }
    if (score_func == LLM_DSV41_SCORE_SOFTMAX) {
        float mxv = raw[0]; for (int e = 1; e < n_experts; e++) if (raw[e] > mxv) mxv = raw[e];
        float s = 0.0f;
        for (int e = 0; e < n_experts; e++) { scores[e] = expf(raw[e] - mxv); s += scores[e]; }
        for (int e = 0; e < n_experts; e++) scores[e] /= s;
    } else if (score_func == LLM_DSV41_SCORE_SIGMOID) {
        for (int e = 0; e < n_experts; e++) scores[e] = 1.0f / (1.0f + expf(-raw[e]));
    } else {
        for (int e = 0; e < n_experts; e++) scores[e] = sqrtf(llm_dsv41_softplus(raw[e]));
    }
    for (int e = 0; e < n_experts; e++) comb[e] = scores[e] + (bias ? bias[e] : 0.0f);
    /* top-n_act per (combined desc, indice asc) */
    char used[LLM_DSV41_MOE_MAXE]; for (int e = 0; e < n_experts; e++) used[e] = 0;
    for (int t = 0; t < n_act; t++) {
        int best = -1;
        for (int e = 0; e < n_experts; e++) {
            if (used[e]) continue;
            if (best < 0 || comb[e] > comb[best]) best = e;
        }
        used[best] = 1; idx_out[t] = best; weights_out[t] = scores[best];
    }
    if (norm_topk_prob && n_act > 1) {
        float s = 0.0f; for (int t = 0; t < n_act; t++) s += weights_out[t];
        s += 1e-20f;
        for (int t = 0; t < n_act; t++) weights_out[t] /= s;
    }
    for (int t = 0; t < n_act; t++) weights_out[t] *= route_scale;
}

/* Expert.__call__ (attivazione): y[i] = silu(clamp(gate_i)) * clamp(up_i) * weight. w2 e' poi
 * una lineare (non inclusa: richiede i pesi). limit<=0 -> nessun clamp. */
static inline void llm_dsv41_expert_swiglu(const float* gate, const float* up, int dff,
                                           float limit, float weight, float* y) {
    for (int i = 0; i < dff; i++) {
        float g = gate[i], u = up[i];
        if (limit > 0.0f) {
            if (g > limit) g = limit;
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
        }
        y[i] = (g * (1.0f / (1.0f + expf(-g)))) * u * weight;
    }
}

/* MoE.__call__ (combine): out[d] = sum_e routed[e*D+d] + shared[d]. */
static inline void llm_dsv41_moe_combine(const float* routed, const float* shared, int n_act, int D,
                                         float* out) {
    for (int d = 0; d < D; d++) {
        float s = 0.0f;
        for (int e = 0; e < n_act; e++) s += routed[e * D + d];
        out[d] = s + (shared ? shared[d] : 0.0f);
    }
}

#endif /* LLM_DSV41_MOE_H */
