/* llm_dsv4_attn.h — attenzione DS4 (V4 Flash): KV single-head (K=V), query multi-head, sink,
 * selezione di righe (raw window + compressed) via `indices` (-1 = non selezionata).
 * Port del kernel DS4 `flash_attn` (sinks + raw/compressed). Header-only. Test: llm_dsv4_attn_test2.c. */
#ifndef LLM_DSV4_ATTN_H
#define LLM_DSV4_ATTN_H

#include <math.h>

#define LLM_DSV4_ATTN_MAXLEN 8192

/* q [n_head*head_dim]; kv [M*head_dim] (single head, K=V); indices[M] (>=0 seleziona, -1 salta);
 * sink [n_head]; scale. out [n_head*head_dim]. */
static inline void llm_dsv4_attn_sink(const float* q, int n_head, int head_dim,
                                      const float* kv, const int* indices, int M,
                                      const float* sink, float scale, float* out) {
    float sc[LLM_DSV4_ATTN_MAXLEN], w[LLM_DSV4_ATTN_MAXLEN];
    for (int h = 0; h < n_head; h++) {
        const float* qh = q + (long)h * head_dim;
        float mx = sink[h];
        for (int m = 0; m < M; m++) {
            if (indices[m] < 0) { sc[m] = -INFINITY; continue; }
            float dot = 0.0f;
            for (int e = 0; e < head_dim; e++) dot += qh[e] * kv[(long)m * head_dim + e];
            sc[m] = dot * scale;
            if (sc[m] > mx) mx = sc[m];
        }
        float ssum = expf(sink[h] - mx);
        for (int m = 0; m < M; m++) { w[m] = (indices[m] < 0) ? 0.0f : expf(sc[m] - mx); ssum += w[m]; }
        float inv = 1.0f / ssum;
        for (int e = 0; e < head_dim; e++) out[(long)h * head_dim + e] = 0.0f;
        for (int m = 0; m < M; m++) {
            if (indices[m] < 0) continue;
            float wm = w[m] * inv;
            const float* vm = kv + (long)m * head_dim;
            for (int e = 0; e < head_dim; e++) out[(long)h * head_dim + e] += wm * vm[e];
        }
    }
}

#endif /* LLM_DSV4_ATTN_H */
