/* test_moe_async.c — M5 gate: pool bounded, REQUIRED/PREFETCH, dedup, shutdown pulito.
 * Pensato anche per TSan. Exit 0 = PASS. */
#include "llmengine_moe_async.h"
#include "llmengine_moe_cache.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { int loads; pthread_mutex_t mu; } LCtx;
static LlmEngineMoeStatus slow_load(void* ctx, LlmEngineMoeExpertKey key, void* buf, size_t cap, size_t* got) {
    LCtx* lc = (LCtx*)ctx;
    pthread_mutex_lock(&lc->mu); lc->loads++; pthread_mutex_unlock(&lc->mu);
    usleep(20000); /* 20ms: simula I/O SSD */
    if (cap < 64) return LLMENGINE_MOE_ERR_INVALID;
    for (int i = 0; i < 64; ++i) ((uint8_t*)buf)[i] = (uint8_t)(key.layer * 7u + key.expert + (uint32_t)i);
    *got = 64;
    return LLMENGINE_MOE_OK;
}
static int loads(LCtx* lc) { pthread_mutex_lock(&lc->mu); int v = lc->loads; pthread_mutex_unlock(&lc->mu); return v; }
static LlmEngineMoeExpertKey K(uint32_t l, uint32_t e) { LlmEngineMoeExpertKey k = {l, e}; return k; }

static void check_blob(const void* data, size_t size, LlmEngineMoeExpertKey key) {
    assert(data != NULL && size == 64);
    for (int i = 0; i < 64; ++i)
        assert(((const uint8_t*)data)[i] == (uint8_t)(key.layer * 7u + key.expert + (uint32_t)i));
}

int main(void) {
    LCtx lc; lc.loads = 0; pthread_mutex_init(&lc.mu, NULL);
    LlmEngineMoeExpertCache* cache = NULL;
    assert(llmengine_moe_cache_create(&cache, 1u << 20) == LLMENGINE_MOE_OK);
    LlmEngineMoeAsync* a = NULL;
    assert(llmengine_moe_async_create(&a, 2, slow_load, &lc, cache, 8) == LLMENGINE_MOE_OK);

    /* 1. prefetch di 5 chiavi; poi le get sono hit (nessuna load extra) */
    for (uint32_t e = 0; e < 5; ++e) assert(llmengine_moe_async_prefetch(a, K(0, e), 64) == LLMENGINE_MOE_OK);
    usleep(200000); /* lascia completare le prefetch */
    for (uint32_t e = 0; e < 5; ++e) {
        const void* d = NULL; size_t n = 0;
        assert(llmengine_moe_async_get(a, K(0, e), 64, &d, &n) == LLMENGINE_MOE_OK);
        check_blob(d, n, K(0, e));
    }
    assert(loads(&lc) == 5); /* 5 chiavi distinte -> 5 load, la prefetch ha nascosto l'I/O */

    /* 2. REQUIRED su chiave nuova + dedup: piu' get dello stesso key = 1 load */
    for (int r = 0; r < 3; ++r) {
        const void* d = NULL; size_t n = 0;
        assert(llmengine_moe_async_get(a, K(1, 1), 64, &d, &n) == LLMENGINE_MOE_OK);
        check_blob(d, n, K(1, 1));
    }
    assert(loads(&lc) == 6);

    /* 3. cancel deterministico: 1 worker occupato su una load lunga, le altre restano in coda */
    LlmEngineMoeAsync* q = NULL;
    assert(llmengine_moe_async_create(&q, 1, slow_load, &lc, cache, 64) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_async_prefetch(q, K(9, 0), 64) == LLMENGINE_MOE_OK); /* il worker la prende (20ms) */
    usleep(5000);                                                 /* worker occupato */
    for (uint32_t e = 1; e <= 5; ++e) assert(llmengine_moe_async_prefetch(q, K(9, e), 64) == LLMENGINE_MOE_OK);
    uint32_t cx = llmengine_moe_async_cancel_prefetch(q);
    assert(cx == 5);
    LlmEngineMoeAsyncStats sq; llmengine_moe_async_stats(q, &sq);
    assert(sq.cancelled == 5);
    llmengine_moe_async_destroy(q);
    LlmEngineMoeAsyncStats st; llmengine_moe_async_stats(a, &st);
    assert(st.completed >= 6);

    /* 4. coda piena -> prefetch droppate (mai bloccante) */
    LlmEngineMoeAsync* b = NULL;
    assert(llmengine_moe_async_create(&b, 1, slow_load, &lc, cache, 2) == LLMENGINE_MOE_OK);
    uint64_t drop_before = 0; { LlmEngineMoeAsyncStats s2; llmengine_moe_async_stats(b, &s2); drop_before = s2.dropped; }
    for (uint32_t e = 0; e < 10; ++e) assert(llmengine_moe_async_prefetch(b, K(3, e), 64) == LLMENGINE_MOE_OK);
    LlmEngineMoeAsyncStats s3; llmengine_moe_async_stats(b, &s3);
    assert(s3.dropped > drop_before); /* cap 2 -> le altre droppate */
    llmengine_moe_async_destroy(b);          /* shutdown pulito anche con coda */

    /* 5. shutdown pulito: destroy con richieste in volo (join di tutti i worker) */
    for (uint32_t e = 0; e < 8; ++e) llmengine_moe_async_prefetch(a, K(4, e), 64);
    llmengine_moe_async_destroy(a);          /* se un worker restasse attivo -> hang/TSan lo becca */

    llmengine_moe_cache_destroy(cache);
    pthread_mutex_destroy(&lc.mu);
    printf("test_moe_async: PASS  load totali=%d\n", loads(&lc));
    return 0;
}
