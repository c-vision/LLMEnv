/* test_moe_sim.c — K1: il replay offline deve riprodurre ESATTAMENTE la cache del runtime
 * (stesso trace, stessa policy) e i contatori resident (K2). */
#include "llmengine_moe_sim.h"
#include "llmengine_moe_cache.h"
#include "llmengine_moe_prof.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define TRACE_PATH "/tmp/llmengine_moe_sim_test.moetrace"
#define EXPERT_SIZE 1024ull
#define BUDGET (2ull * EXPERT_SIZE)   /* 2 slot */

typedef struct { uint64_t size; } LCtx;
static void evict_hook(void* ctx, LlmEngineMoeExpertKey key) {
    llmengine_moe_prof_event((LlmEngineMoeProfiler*)ctx, LLMENGINE_MOE_TRACE_EVICT, key.layer, key.expert, 0, 1);
}
static LlmEngineMoeStatus load_cb(void* ctx, LlmEngineMoeExpertKey key, void* buf, size_t cap, size_t* got) {
    LCtx* lc = (LCtx*)ctx;
    (void)key;
    if (cap < lc->size) { *got = 0; return LLMENGINE_MOE_ERR_INVALID; }
    memset(buf, 0, lc->size);
    *got = lc->size;
    return LLMENGINE_MOE_OK;
}

int main(void) {
    /* Sequenza: A B A C A (cap=2): hits=2, misses=3, evictions=1 (B evitto), resident=2. */
    const uint32_t seq[5] = { 0, 1, 0, 2, 0 };

    /* 1. Replay DIRETTO sulla cache reale: produce la verità di riferimento e gli eventi
     *    HIT/MISS che un run reale scriverebbe nel trace. */
    LlmEngineMoeProfiler* p = llmengine_moe_prof_create();
    assert(p != NULL);
    assert(llmengine_moe_prof_trace_open_ex(p, TRACE_PATH, BUDGET, EXPERT_SIZE) == LLMENGINE_MOE_OK);

    LlmEngineMoeExpertCache* c = NULL;
    assert(llmengine_moe_cache_create(&c, BUDGET) == LLMENGINE_MOE_OK);
    /* K1: gli EVICT reali finiscono nel trace via hook (stessa numerazione del summary). */
    llmengine_moe_cache_set_evict_hook(c, evict_hook, p);

    LCtx lc = { EXPERT_SIZE };
    for (int i = 0; i < 5; i++) {
        llmengine_moe_cache_step_begin(c);
        llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_ROUTER, 0, seq[i], 0, 1);
        LlmEngineMoeExpertKey k = { 0, seq[i] };
        const void* data = NULL; size_t size = 0;
        int was_hit = 0;
        assert(llmengine_moe_cache_get_ex(c, k, EXPERT_SIZE, load_cb, &lc, &data, &size, &was_hit) == LLMENGINE_MOE_OK);
        llmengine_moe_prof_event(p, was_hit ? LLMENGINE_MOE_TRACE_CACHE_HIT : LLMENGINE_MOE_TRACE_CACHE_MISS,
                                 k.layer, k.expert, 0, (uint64_t)size);
    }
    llmengine_moe_prof_trace_close(p);
    llmengine_moe_prof_destroy(p);

    LlmEngineMoeCacheStats direct;
    llmengine_moe_cache_stats(c, &direct);
    assert(direct.hits == 2 && direct.misses == 3 && direct.evictions == 1 && direct.resident_hits == 2);
    llmengine_moe_cache_destroy(c);

    /* 2. Meta del trace (budget/expert size registrati) */
    uint64_t mb = 0, me = 0;
    assert(llmengine_moe_trace_meta(TRACE_PATH, &mb, &me) == LLMENGINE_MOE_OK);
    assert(mb == BUDGET && me == EXPERT_SIZE);

    /* 3. Simulatore: stessi numeri del replay diretto (policy = quella del runtime) */
    LlmEngineMoeSimResult r;
    assert(llmengine_moe_sim_replay(TRACE_PATH, BUDGET, EXPERT_SIZE, &r) == LLMENGINE_MOE_OK);
    assert(r.requests == 5 && r.steps == 5);
    assert(r.hits == direct.hits && r.misses == direct.misses && r.evictions == direct.evictions);
    assert(r.resident_hits == direct.resident_hits);
    assert(r.unique_experts == 3);
    assert(r.bytes_loaded == direct.misses * EXPERT_SIZE);
    assert(r.hit_rate > 0.39 && r.hit_rate < 0.41);
    assert(r.resident_rate > 0.99);   /* 2/2: entrambi gli hit sono residenti */

    /* 4. Capacità 1 slot: peggiore (nessun hit su A ripetuto) */
    LlmEngineMoeSimResult r1;
    assert(llmengine_moe_sim_replay(TRACE_PATH, EXPERT_SIZE, EXPERT_SIZE, &r1) == LLMENGINE_MOE_OK);
    assert(r1.hits == 0 && r1.misses == 5);

    /* 5. Capacità ampia: tutti hit dopo il primo (A,B,C unici) */
    LlmEngineMoeSimResult rbig;
    assert(llmengine_moe_sim_replay(TRACE_PATH, 8 * EXPERT_SIZE, EXPERT_SIZE, &rbig) == LLMENGINE_MOE_OK);
    assert(rbig.hits == 2 && rbig.misses == 3 && rbig.evictions == 0);

    /* 6. Verifica col summary del trace (stesso gate di `moe_sim --verify`) */
    LlmEngineMoeTraceSummary sum;
    assert(llmengine_moe_trace_summary(TRACE_PATH, &sum) == LLMENGINE_MOE_OK);
    assert(sum.cache_hits == r.hits && sum.cache_misses == r.misses && sum.evictions == r.evictions);

    printf("test_moe_sim: OK (hits=%llu misses=%llu evict=%llu resident=%llu unique=%llu)\n",
           (unsigned long long)r.hits, (unsigned long long)r.misses,
           (unsigned long long)r.evictions, (unsigned long long)r.resident_hits,
           (unsigned long long)r.unique_experts);
    return 0;
}
