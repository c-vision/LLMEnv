/* llm_dsv4_moe_fwd.h — MoE DS4 (V4 Flash): router sqrt(softplus(logits)), selezione top-k
 * (con bias exp_probs_b) o hash (ffn_gate_tid2eid), pesi normalizzati * expert_weight_scale,
 * esperti SwiGLU gate/up/down + shared expert. Port fedele del riferimento CPU ds4.c.
 * Header-only. Test: llm_dsv4_moe_fwd_test.c. */
#ifndef LLM_DSV4_MOE_FWD_H
#define LLM_DSV4_MOE_FWD_H

#include <math.h>
#include <stdlib.h>
#include "llm_dsv4.h"
#include "llm_dsv4_attn_fwd.h"   /* llm_dsv4_linear, rms */
#include "llm_dsv4_packed.h"     /* llm_dsv4_dot_tensor */
#include "llm_dsv4_actquant.h"   /* Q8_0/Q8_K attivazioni + dot interi */

#define LLM_DSV4_MAX_EXPERT 1024
#define LLM_DSV4_MAX_EXPERT_USED 64

static inline float llm_dsv4_sigmoid_stable(float x) {
    if (x >= 0.0f) { float e = expf(-x); return 1.0f / (1.0f + e); }
    float e = expf(x); return e / (1.0f + e);
}
static inline float llm_dsv4_silu(float x) { return x * llm_dsv4_sigmoid_stable(x); }
static inline float llm_dsv4_softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}
static inline void llm_dsv4_swiglu(float* out, const float* gate, const float* up, int n, float clamp) {
    for (int i = 0; i < n; i++) {
        float g = gate[i], u = up[i];
        if (clamp > 1.0e-6f) { if (g > clamp) g = clamp; if (u > clamp) u = clamp; if (u < -clamp) u = -clamp; }
        out[i] = llm_dsv4_silu(g) * u;
    }
}
static inline void llm_dsv4_router_probs(float* probs, const float* Wgate, int n_expert, int embd, const float* x) {
    float logits[LLM_DSV4_MAX_EXPERT];
    llm_dsv4_linear(logits, Wgate, n_expert, embd, x);
    for (int i = 0; i < n_expert; i++) probs[i] = sqrtf(llm_dsv4_softplus_stable(logits[i]));
}
static inline void llm_dsv4_topk_desc(const float* score, int n, int k, int* idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i; break;
            }
}

/* Routing (hash o top-k): riempie selected[] e weight[]. */
static inline void llm_dsv4_moe_route(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* x, int il, int token, int* selected, float* weight) {
    int n_expert = c->n_expert, used = c->n_expert_used;
    float probs[LLM_DSV4_MAX_EXPERT];
    llm_dsv4_router_probs(probs, L->ffn_gate_inp, n_expert, c->n_embd, x);
    if (il < c->n_hash_layer && L->ffn_gate_tid2eid) {
        const float* row = L->ffn_gate_tid2eid + (long)token * used;
        for (int i = 0; i < used; i++) selected[i] = (int)row[i];
    } else {
        float selection[LLM_DSV4_MAX_EXPERT];
        for (int i = 0; i < n_expert; i++) selection[i] = probs[i] + (L->exp_probs_b ? L->exp_probs_b[i] : 0.0f);
        llm_dsv4_topk_desc(selection, n_expert, used, selected);
    }
    float sum = 0.0f;
    for (int i = 0; i < used; i++) { weight[i] = probs[selected[i]]; sum += weight[i]; }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < used; i++) weight[i] = weight[i] / sum * c->expert_weight_scale;
}

/* MoE completo con esperti f32. `tid2eid` (hash) usato se il<cfg->n_hash_layer. */
static inline void llm_dsv4_moe_forward(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* x, int il, int token, float* out) {
    int used = c->n_expert_used, ff = c->n_ff_exp, embd = c->n_embd;
    int selected[LLM_DSV4_MAX_EXPERT_USED];
    float weight[LLM_DSV4_MAX_EXPERT_USED];
    llm_dsv4_moe_route(c, L, x, il, token, selected, weight);

    float *gate = (float*)malloc(sizeof(float) * (size_t)ff);
    float *up = (float*)malloc(sizeof(float) * (size_t)ff);
    float *mid = (float*)malloc(sizeof(float) * (size_t)ff);
    for (int o = 0; o < embd; o++) out[o] = 0.0f;
    for (int i = 0; i < used; i++) {
        int e = selected[i];
        llm_dsv4_linear(gate, L->ffn_gate_exps + (long)e * ff * embd, ff, embd, x);
        llm_dsv4_linear(up,   L->ffn_up_exps   + (long)e * ff * embd, ff, embd, x);
        llm_dsv4_swiglu(mid, gate, up, ff, c->swiglu_clamp_exp);
        float* de = L->ffn_down_exps + (long)e * embd * ff;
        for (int o = 0; o < embd; o++) {
            const float* wr = de + (long)o * ff;
            double acc = 0.0;
            for (int j = 0; j < ff; j++) acc += (double)wr[j] * mid[j];
            out[o] += weight[i] * (float)acc;
        }
    }
    /* shared expert */
    llm_dsv4_linear(gate, L->ffn_gate_shexp, ff, embd, x);
    llm_dsv4_linear(up,   L->ffn_up_shexp,   ff, embd, x);
    llm_dsv4_swiglu(mid, gate, up, ff, c->swiglu_clamp_exp);
    for (int o = 0; o < embd; o++) {
        const float* wr = L->ffn_down_shexp + (long)o * ff;
        double acc = 0.0;
        for (int j = 0; j < ff; j++) acc += (double)wr[j] * mid[j];
        out[o] += (float)acc;
    }
    free(mid); free(up); free(gate);
}

/* MoE con esperti PACKED (IQ2_XXS/Q2_K) via dot interi Q8_K (parità col path CPU ds4).
 * Attivazioni quantizzate: Q8_K per gli esperti, Q8_0 per lo shared expert. */
static inline void llm_dsv4_moe_forward_packed(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, const float* x, int il, int token, float* out) {
    int used = c->n_expert_used, ff = c->n_ff_exp, embd = c->n_embd;
    int selected[LLM_DSV4_MAX_EXPERT_USED];
    float weight[LLM_DSV4_MAX_EXPERT_USED];
    llm_dsv4_moe_route(c, L, x, il, token, selected, weight);

    int64_t nb_in = llm_ggml_nbytes(pe->gate_type, pe->in_dim);    /* gate/up ne0 = in_dim */
    int64_t nb_out = llm_ggml_nbytes(pe->down_type, pe->ff_dim);   /* down ne0 = ff_dim */
    LlmDsv4Q8K* xqk = (LlmDsv4Q8K*)malloc(sizeof(LlmDsv4Q8K) * (size_t)(pe->in_dim / 256));
    LlmDsv4Q8K* mqk = (LlmDsv4Q8K*)malloc(sizeof(LlmDsv4Q8K) * (size_t)(pe->ff_dim / 256));
    llm_dsv4_q8_K_quantize_row(x, pe->in_dim, xqk);

    float *gate = (float*)malloc(sizeof(float) * (size_t)ff);
    float *up = (float*)malloc(sizeof(float) * (size_t)ff);
    float *mid = (float*)malloc(sizeof(float) * (size_t)ff);
    for (int o = 0; o < embd; o++) out[o] = 0.0f;
    for (int i = 0; i < used; i++) {
        int e = selected[i];
        const uint8_t *gb, *ub, *db;
        if (pe->store && pe->store_get) {
            pe->store_get(pe->store, il, e, &gb, &ub, &db);
        } else {
            gb = pe->gate + (size_t)e * (size_t)ff * (size_t)nb_in;
            ub = pe->up   + (size_t)e * (size_t)ff * (size_t)nb_in;
            db = pe->down + (size_t)e * (size_t)embd * (size_t)nb_out;
        }
        for (int o = 0; o < ff; o++) {
            gate[o] = llm_dsv4_dot_w_q8_K(pe->gate_type, gb + (int64_t)o * nb_in, pe->in_dim, xqk);
            up[o]   = llm_dsv4_dot_w_q8_K(pe->up_type,   ub + (int64_t)o * nb_in, pe->in_dim, xqk);
        }
        llm_dsv4_swiglu(mid, gate, up, ff, c->swiglu_clamp_exp);
        /* ds4: il peso del router e' applicato al mid PRIMA di quantizzarlo a Q8_K
         * (layer_routed_moe_one), poi il down accumula senza ulteriore peso. */
        for (int j = 0; j < ff; j++) mid[j] *= weight[i];
        /* #1025: sotto -ffast-math un prodotto non-finito puo' avvelenare residual/KV. */
        for (int j = 0; j < ff; j++) if (!isfinite(mid[j])) mid[j] = 0.0f;
        llm_dsv4_q8_K_quantize_row(mid, pe->ff_dim, mqk);
        for (int o = 0; o < embd; o++)
            { float v = llm_dsv4_dot_w_q8_K(pe->down_type, db + (int64_t)o * nb_out, pe->ff_dim, mqk);
              if (isfinite(v)) out[o] += v; }   /* #1025 */
    }
    free(mqk); free(xqk);
    /* shared expert: Q8_0 packed se disponibile, altrimenti f32 */
    if (pe->shexp_gate && pe->shexp_down) {
        int64_t nb_g = llm_ggml_nbytes(pe->shexp_gate_type, embd);
        int64_t nb_d = llm_ggml_nbytes(pe->shexp_down_type, ff);
        int8_t* xq8 = (int8_t*)malloc((size_t)embd);
        float* xs8 = (float*)malloc(sizeof(float) * (size_t)((embd + 31) / 32));
        llm_dsv4_q8_0_quantize_row(x, embd, xq8, xs8);
        for (int o = 0; o < ff; o++) {
            gate[o] = llm_dsv4_dot_q8_0_prequant(pe->shexp_gate + (int64_t)o * nb_g, embd, xq8, xs8);
            up[o]   = llm_dsv4_dot_q8_0_prequant(pe->shexp_up   + (int64_t)o * nb_g, embd, xq8, xs8);
        }
        llm_dsv4_swiglu(mid, gate, up, ff, c->swiglu_clamp_exp);
        int8_t* mq8 = (int8_t*)malloc((size_t)ff);
        float* ms8 = (float*)malloc(sizeof(float) * (size_t)((ff + 31) / 32));
        llm_dsv4_q8_0_quantize_row(mid, ff, mq8, ms8);
        for (int o = 0; o < embd; o++)
            { float v = llm_dsv4_dot_q8_0_prequant(pe->shexp_down + (int64_t)o * nb_d, ff, mq8, ms8);
              if (isfinite(v)) out[o] += v; }   /* #1025 */
        free(ms8); free(mq8); free(xs8); free(xq8);
    } else {
        llm_dsv4_linear(gate, L->ffn_gate_shexp, ff, embd, x);
        llm_dsv4_linear(up,   L->ffn_up_shexp,   ff, embd, x);
        llm_dsv4_swiglu(mid, gate, up, ff, c->swiglu_clamp_exp);
        for (int o = 0; o < embd; o++) {
            const float* wr = L->ffn_down_shexp + (long)o * ff;
            double acc = 0.0;
            for (int j = 0; j < ff; j++) acc += (double)wr[j] * mid[j];
            out[o] += (float)acc;
        }
    }
    free(mid); free(up); free(gate);
}

#endif /* LLM_DSV4_MOE_FWD_H */
