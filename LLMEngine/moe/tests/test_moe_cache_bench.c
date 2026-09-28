/* test_moe_cache_bench.c — M7: LRU vs routing-aware sullo STESSO trace (hit rate + bytes). */
#include "llmengine_moe_cache.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define E 64u
#define NACC 4000u

typedef struct { uint64_t bytes; } LCtx;
static LlmEngineMoeStatus load(void* ctx, LlmEngineMoeExpertKey key, void* buf, size_t cap, size_t* got) {
    LCtx* lc = (LCtx*)ctx; (void)key;
    if (cap < 64) return LLMENGINE_MOE_ERR_INVALID;
    memset(buf, 0, 64); lc->bytes += 64; *got = 64; return LLMENGINE_MOE_OK;
}

static uint64_t lcg(uint64_t* s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 33; }

static void run_trace(LlmEngineMoeExpertCache* c, const uint32_t* tr, float* rp, LCtx* lc) {
    for (uint32_t i = 0; i < NACC; ++i) {
        LlmEngineMoeExpertKey k = {0, tr[i]};
        const void* d = NULL; size_t sz = 0;
        assert(llmengine_moe_cache_get(c, k, 64, load, lc, &d, &sz) == LLMENGINE_MOE_OK);
        llmengine_moe_cache_set_hint(c, k, rp[tr[i]], 0.0f);   /* hint di routing (M7) */
    }
}

int main(void) {
    /* trace skewato: 80% dagli 8 HOT, 20% dai COLD; route_prob alta per gli HOT */
    uint32_t* tr = (uint32_t*)malloc(NACC * sizeof(uint32_t));
    float* rp = (float*)malloc(E * sizeof(float));
    for (uint32_t e = 0; e < E; ++e) rp[e] = (e < 8) ? 0.9f : 0.01f;
    uint64_t seed = 12345;
    for (uint32_t i = 0; i < NACC; ++i) {
        uint64_t r = lcg(&seed);
        tr[i] = (r % 100 < 80) ? (uint32_t)(lcg(&seed) % 8) : (uint32_t)(8 + lcg(&seed) % (E - 8));
    }
    /* budget: ~8 esperti da 64B (tiene il working set HOT) */
    const uint64_t budget = 8 * 64;

    LCtx l1 = {0}, l2 = {0};
    LlmEngineMoeExpertCache* clru = NULL; assert(llmengine_moe_cache_create(&clru, budget) == LLMENGINE_MOE_OK);
    LlmEngineMoeExpertCache* crt = NULL;  assert(llmengine_moe_cache_create(&crt, budget) == LLMENGINE_MOE_OK);
    llmengine_moe_cache_set_policy(clru, LLMENGINE_MOE_CACHE_LRU, 1.0f, 0, 0, 0);
    llmengine_moe_cache_set_policy(crt, LLMENGINE_MOE_CACHE_ROUTING, 0.0f, 0, 1.0f, 0); /* evince per route_prob */

    run_trace(clru, tr, rp, &l1);
    run_trace(crt, tr, rp, &l2);

    LlmEngineMoeCacheStats s1, s2;
    llmengine_moe_cache_stats(clru, &s1); llmengine_moe_cache_stats(crt, &s2);
    double hr1 = (double)s1.hits / (double)(s1.hits + s1.misses);
    double hr2 = (double)s2.hits / (double)(s2.hits + s2.misses);
    printf("[m7] LRU:      hit=%.3f  bytes=%llu\n", hr1, (unsigned long long)s1.bytes_loaded);
    printf("[m7] ROUTING:  hit=%.3f  bytes=%llu\n", hr2, (unsigned long long)s2.bytes_loaded);
    assert(hr2 > hr1);                     /* routing batte LRU sullo stesso trace */
    assert(s2.bytes_loaded < s1.bytes_loaded);

    llmengine_moe_cache_destroy(clru); llmengine_moe_cache_destroy(crt);
    free(tr); free(rp);
    printf("test_moe_cache_bench: PASS\n");
    return 0;
}
