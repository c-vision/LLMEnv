/* test_moe_cache.c — M4 gate: LRU, dedup single-flight, metriche.
 * Exit 0 = PASS. Pensato per girare anche sotto TSan (LLMENGINE_MOE_TSAN). */
#include "llmengine_moe_cache.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static LlmEngineMoeExpertKey K(uint32_t l, uint32_t e) { LlmEngineMoeExpertKey k = {l, e}; return k; }
static LlmEngineMoeCacheStats stats(LlmEngineMoeExpertCache* c) { LlmEngineMoeCacheStats s; llmengine_moe_cache_stats(c, &s); return s; }

/* ── dedup concorrente ── */
typedef struct { int loads; pthread_mutex_t mu; } LoadCtx;
static LlmEngineMoeStatus test_load(void* ctx, LlmEngineMoeExpertKey key, void* buf, size_t cap, size_t* got) {
    LoadCtx* lc = (LoadCtx*)ctx;
    pthread_mutex_lock(&lc->mu); lc->loads++; pthread_mutex_unlock(&lc->mu);
    usleep(50000); /* allarga la finestra di race */
    if (cap < 64) return LLMENGINE_MOE_ERR_INVALID;
    for (int i = 0; i < 64; ++i) ((uint8_t*)buf)[i] = (uint8_t)(key.layer * 7u + key.expert + (uint32_t)i);
    *got = 64;
    return LLMENGINE_MOE_OK;
}
typedef struct { LlmEngineMoeExpertCache* c; LoadCtx* lc; int ok; } Worker;
static void* worker_fn(void* arg) {
    Worker* w = (Worker*)arg;
    const void* data = NULL; size_t size = 0;
    LlmEngineMoeStatus s = llmengine_moe_cache_get(w->c, K(3, 42), 64, test_load, w->lc, &data, &size);
    if (s == LLMENGINE_MOE_OK && data != NULL && size == 64) {
        int good = 1;
        for (int i = 0; i < 64; ++i) if (((const uint8_t*)data)[i] != (uint8_t)(3u * 7u + 42u + (uint32_t)i)) good = 0;
        w->ok = good;
    }
    return NULL;
}

int main(void) {
    LlmEngineMoeCacheStats s;

    /* 1. LRU + byte limit (budget 1024) */
    LlmEngineMoeExpertCache* c = NULL;
    assert(llmengine_moe_cache_create(&c, 1024) == LLMENGINE_MOE_OK);
    uint8_t a[500], b[500], d[500];
    memset(a, 1, sizeof(a)); memset(b, 2, sizeof(b)); memset(d, 3, sizeof(d));
    assert(llmengine_moe_cache_insert(c, K(0, 0), a, 500) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_insert(c, K(0, 1), b, 500) == LLMENGINE_MOE_OK);
    const void* p = NULL; size_t n = 0;
    assert(llmengine_moe_cache_lookup(c, K(0, 0), &p, &n) == LLMENGINE_MOE_OK && n == 500); /* touch A */
    assert(llmengine_moe_cache_insert(c, K(0, 2), d, 500) == LLMENGINE_MOE_OK);            /* -> evict LRU (B) */
    assert(llmengine_moe_cache_lookup(c, K(0, 0), &p, &n) == LLMENGINE_MOE_OK && p != NULL); /* A presente */
    assert(llmengine_moe_cache_lookup(c, K(0, 1), &p, &n) == LLMENGINE_MOE_OK && p == NULL); /* B evitto */
    assert(llmengine_moe_cache_lookup(c, K(0, 2), &p, &n) == LLMENGINE_MOE_OK && p != NULL); /* C presente */
    s = stats(c);
    assert(s.evictions >= 1);

    /* 2. remove + clear */
    assert(llmengine_moe_cache_remove(c, K(0, 0)) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_lookup(c, K(0, 0), &p, &n) == LLMENGINE_MOE_OK && p == NULL);
    llmengine_moe_cache_clear(c);
    assert(llmengine_moe_cache_lookup(c, K(0, 2), &p, &n) == LLMENGINE_MOE_OK && p == NULL);

    /* 3. 10000 op senza errori + budget rispettato */
    uint8_t blob[64];
    for (uint32_t i = 0; i < 10000; ++i) {
        memset(blob, (int)(i & 0xFF), sizeof(blob));
        LlmEngineMoeExpertKey k = K(i % 8, i % 16);
        assert(llmengine_moe_cache_insert(c, k, blob, sizeof(blob)) == LLMENGINE_MOE_OK);
        assert(llmengine_moe_cache_lookup(c, k, &p, &n) == LLMENGINE_MOE_OK);
    }
    llmengine_moe_cache_destroy(c);
    c = NULL;

    /* 4. dedup single-flight: 8 thread, stesso key -> UNA sola load */
    assert(llmengine_moe_cache_create(&c, 1u << 20) == LLMENGINE_MOE_OK);
    LoadCtx lc; lc.loads = 0; pthread_mutex_init(&lc.mu, NULL);
    enum { T = 8 };
    pthread_t th[T]; Worker w[T];
    for (int i = 0; i < T; ++i) { w[i].c = c; w[i].lc = &lc; w[i].ok = 0; }
    for (int i = 0; i < T; ++i) assert(pthread_create(&th[i], NULL, worker_fn, &w[i]) == 0);
    for (int i = 0; i < T; ++i) pthread_join(th[i], NULL);
    for (int i = 0; i < T; ++i) assert(w[i].ok == 1);
    assert(lc.loads == 1);                 /* dedup: una sola lettura */
    s = stats(c);
    assert(s.hits + s.misses == (uint64_t)T); /* HIT + MISS = LOOKUPS */
    assert(s.misses == 1 && s.hits == (uint64_t)(T - 1));
    /* il blob e' memorizzato: una get successiva e' un hit senza load */
    const void* q = NULL; size_t qn = 0;
    assert(llmengine_moe_cache_get(c, K(3, 42), 64, test_load, &lc, &q, &qn) == LLMENGINE_MOE_OK);
    assert(q != NULL && lc.loads == 1);
    pthread_mutex_destroy(&lc.mu);
    llmengine_moe_cache_destroy(c);

    /* 5. budget 0: crea/usa/distruggi senza crash (nessuna retention) */
    LlmEngineMoeExpertCache* z = NULL;
    assert(llmengine_moe_cache_create(&z, 0) == LLMENGINE_MOE_OK);
    const void* zp = NULL; size_t zn = 0;
    assert(llmengine_moe_cache_get(z, K(1, 1), 64, test_load, &lc, &zp, &zn) == LLMENGINE_MOE_OK);
    assert(zn == 64);
    llmengine_moe_cache_destroy(z);

    printf("test_moe_cache: PASS\n");
    return 0;
}
