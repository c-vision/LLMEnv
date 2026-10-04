/* llm_dsv4_head.h — testa finale DS4: collapse mHC (output_hc_fn/scale/base) -> RMSNorm ->
 * proiezione vocab (output). Port fedele del riferimento CPU ds4.c. Header-only.
 * Test: llm_dsv4_head_test.c. */
#ifndef LLM_DSV4_HEAD_H
#define LLM_DSV4_HEAD_H

#include <math.h>
#include "llm_dsv4_moe_fwd.h"
#include "llm_dsv4_weights.h"

static inline void llm_dsv4_rms_norm_no_weight(float* out, const float* x, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

/* out[d] = sum_h w[h]*x[h*embd+d] */
static inline void llm_dsv4_hc_weighted_sum(float* out, const float* x, const float* w, int embd, int n_hc) {
    for (int d = 0; d < embd; d++) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; h++) acc += w[h] * x[(long)h * embd + d];
        out[d] = acc;
    }
}

/* Collapse mHC -> vettore embd. */
static inline void llm_dsv4_output_hc_head(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        const float* inp_hc, float* out) {
    int HC = c->n_hc, D = c->n_embd, hc_dim = HC * D;
    float* flat = (float*)malloc(sizeof(float) * (size_t)hc_dim);
    float* pre = (float*)malloc(sizeof(float) * (size_t)HC);
    float* wt = (float*)malloc(sizeof(float) * (size_t)HC);
    llm_dsv4_rms_norm_no_weight(flat, inp_hc, hc_dim, c->rms_eps);
    llm_dsv4_linear(pre, w->output_hc_fn, HC, hc_dim, flat);
    for (int i = 0; i < HC; i++)
        wt[i] = llm_dsv4_sigmoid_stable(pre[i] * w->output_hc_scale[0] + w->output_hc_base[i]) + c->hc_eps;
    llm_dsv4_hc_weighted_sum(out, inp_hc, wt, D, HC);
    free(wt); free(pre); free(flat);
}

/* logits [n_vocab]. */
static inline void llm_dsv4_output_logits(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        const float* inp_hc, float* logits) {
    float* embd = (float*)malloc(sizeof(float) * (size_t)c->n_embd);
    float* norm = (float*)malloc(sizeof(float) * (size_t)c->n_embd);
    llm_dsv4_output_hc_head(c, w, inp_hc, embd);
    llm_dsv4_rms_norm_weight(norm, embd, w->output_norm, c->n_embd, c->rms_eps);
    if (w->output_p) {
        int8_t* xq = (int8_t*)malloc((size_t)c->n_embd);
        float* xs = (float*)malloc(sizeof(float) * (size_t)((c->n_embd + 31) / 32));
        llm_dsv4_q8_0_quantize_row(norm, c->n_embd, xq, xs);
        int64_t nb = llm_ggml_nbytes(w->output_type, c->n_embd);
        for (int o = 0; o < c->n_vocab; o++)
            logits[o] = llm_dsv4_dot_q8_0_prequant(w->output_p + (int64_t)o * nb, c->n_embd, xq, xs);
        free(xs); free(xq);
    } else {
        llm_dsv4_linear(logits, w->output, c->n_vocab, c->n_embd, norm);
    }
    free(norm); free(embd);
}

static inline int llm_dsv4_argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

#endif /* LLM_DSV4_HEAD_H */
