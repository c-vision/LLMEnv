/* llm_dsv41_dspark.h — DeepSeek V4.1: DSpark/MTP, header-only.
 *   - get_dspark_topk_idxs (official model.py)
 *   - MarkovHead: bias = embed_row @ head.weight^T
 *   - ConfidenceHead: dot(concat(h, markov), proj)
 *   - proposta: argmax(proposal_logits + markov_bias) iterata
 *   - DSparkStage (Block con DSparkAttention + MoE draft), proposal_forward, forward_spec
 * Verifica: llm_dsv41_dspark_test.c / llm_dsv41_dspark_attn_test.c. */
#ifndef LLM_DSV41_DSPARK_H
#define LLM_DSV41_DSPARK_H
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static inline void llm_dsv41_dspark_topk_idxs(int window, int bsz, int block, int start, int* out) {
    int n1 = window < start + 1 ? window : start + 1;
    int width = n1 + block;
    for (int b = 0; b < bsz; b++)
        for (int k = 0; k < block; k++)
            for (int j = 0; j < width; j++)
                out[(b * block + k) * width + j] = (j < n1) ? j : (window + (j - n1));
}

/* bias[v] = sum_r embed_row[r] * head_w[v*rank+r] */
static inline void llm_dsv41_dspark_markov_bias(const float* embed_row, const float* head_w,
                                                int rank, int vocab, float* out) {
    for (int v = 0; v < vocab; v++) {
        float s = 0.0f;
        for (int r = 0; r < rank; r++) s += embed_row[r] * head_w[(size_t)v * rank + r];
        out[v] = s;
    }
}

static inline float llm_dsv41_dspark_confidence(const float* h, const float* markov,
                                                int dim, int rank, const float* proj) {
    float s = 0.0f;
    for (int d = 0; d < dim; d++) s += h[d] * proj[d];
    for (int r = 0; r < rank; r++) s += markov[r] * proj[dim + r];
    return s;
}

/* proposta iterata: token = argmax(proposal_logits[i] + markov_bias(prev)) (prop_logits puo' essere NULL). */
static inline void llm_dsv41_dspark_proposal(const float* prop_logits, const float* embed_table,
                                             const float* head_w, int rank, int vocab, int block,
                                             int first_id, int32_t* out_ids) {
    int prev = first_id;
    float* bias = (float*)malloc((size_t)vocab * sizeof(float));
    if (!bias) return;
    for (int i = 0; i < block; i++) {
        llm_dsv41_dspark_markov_bias(embed_table + (size_t)prev * rank, head_w, rank, vocab, bias);
        int best = 0; float bv = -INFINITY;
        for (int v = 0; v < vocab; v++) {
            float s = (prop_logits ? prop_logits[(size_t)i * vocab + v] : 0.0f) + bias[v];
            if (s > bv) { bv = s; best = v; }
        }
        out_ids[i] = best; prev = best;
    }
    free(bias);
}

#endif /* LLM_DSV41_DSPARK_H */
