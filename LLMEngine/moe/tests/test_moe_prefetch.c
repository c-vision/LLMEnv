/* test_moe_prefetch.c — M6 gate: coupling cross-layer, recall post-warm-up, bound, reset, metriche. */
#include "llmengine_moe_prefetch.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L 6u
#define E 16u
#define TOPN 2u
#define KK 4u

static void step_token(LlmEngineMoePredictor* p, uint32_t t) {
    uint32_t ex[L];
    ex[0] = t % E;
    for (uint32_t l = 1; l < L; ++l) ex[l] = (ex[l - 1] * 3u + 1u) % E;
    for (uint32_t l = 0; l < L; ++l)
        assert(llmengine_moe_predictor_notify(p, l, &ex[l], 1) == LLMENGINE_MOE_OK);
}

int main(void) {
    LlmEngineMoePredictor* p = NULL;
    assert(llmengine_moe_predictor_create(&p, L, E, TOPN, KK) == LLMENGINE_MOE_OK);

    /* 1. warm-up: impara il coupling e[l+1] = (e[l]*3+1) % E */
    for (uint32_t t = 0; t < 40; ++t) step_token(p, t);

    /* 2. azzera le metriche (NON il coupling) e misura la recall sulla fase calda */
    llmengine_moe_predictor_reset(p, 0);
    LlmEngineMoePrefetchMetrics z; llmengine_moe_predictor_metrics(p, &z);
    assert(z.predictions == 0 && z.hits == 0);
    for (uint32_t t = 100; t < 130; ++t) step_token(p, t);
    LlmEngineMoePrefetchMetrics m; llmengine_moe_predictor_metrics(p, &m);
    assert(m.predictions > 0);
    double recall = (double)m.hits / (double)m.actual_total;
    printf("[m6] recall topK post-warmup=%.3f  top1=%llu/%llu  wasted=%llu\n",
           recall, (unsigned long long)m.top1_hits, (unsigned long long)m.predictions,
           (unsigned long long)m.wasted);
    assert(recall > 0.95); /* coupling deterministico: dopo il warm-up e' quasi esatto */
    assert((double)m.top1_hits / (double)m.predictions > 0.95);

    /* 3. bound: mai esperti inesistenti; determinismo */
    uint32_t out[TOPN], n = 0, out2[TOPN], n2 = 0;
    assert(llmengine_moe_predictor_predict(p, 3, out, TOPN, &n) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_predictor_predict(p, 3, out2, TOPN, &n2) == LLMENGINE_MOE_OK);
    assert(n == n2 && n <= TOPN);
    for (uint32_t i = 0; i < n; ++i) assert(out[i] < E);
    assert(memcmp(out, out2, n * sizeof(uint32_t)) == 0);

    /* 4. metriche per-layer coerenti */
    uint64_t h = 0, a = 0;
    assert(llmengine_moe_predictor_layer_metrics(p, 3, &h, &a) == LLMENGINE_MOE_OK);
    assert(h <= a);

    /* 5. reset FULL (azzera il coupling): la recall crolla all'inizio */
    llmengine_moe_predictor_reset(p, 1);
    step_token(p, 200);            /* 1o token dopo il reset: nessuna evidenza */
    step_token(p, 201);
    LlmEngineMoePrefetchMetrics m2; llmengine_moe_predictor_metrics(p, &m2);
    assert(m2.predictions >= 1);
    double cold = (double)m2.hits / (double)m2.actual_total;
    assert(cold < recall);         /* a freddo si predice peggio */

    llmengine_moe_predictor_destroy(p);
    printf("test_moe_prefetch: PASS\n");
    return 0;
}
