/* test_moe_kv.c — M11 gate: KV cache separata, budget dedicato, LRU, e INDIPENDENZA dalla expert
 * cache (evincere esperti non invalida la KV; pressione KV non corrompe gli esperti). */
#include "llmengine_moe_kv.h"
#include "llmengine_moe_cache.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static LlmEngineMoeExpertKey ekey(uint32_t e) { LlmEngineMoeExpertKey k = {0u, e}; return k; }
static LlmEngineMoeKvKey kvkey(uint32_t l, uint32_t p) { LlmEngineMoeKvKey k = {l, p}; return k; }

int main(void) {
    /* 1. budget KV 8192, pagine da 2048 -> 4 residenti */
    LlmEngineMoeKvCache* kv = NULL;
    assert(llmengine_moe_kv_create(&kv, 8192) == LLMENGINE_MOE_OK);
    uint8_t page[2048];
    for (int p = 0; p < 4; ++p) {
        memset(page, p + 1, sizeof(page));
        assert(llmengine_moe_kv_insert(kv, kvkey(0, (uint32_t)p), page, sizeof(page)) == LLMENGINE_MOE_OK);
    }
    assert(llmengine_moe_kv_bytes_used(kv) == 8192);
    const void* d = NULL; size_t n = 0;
    assert(llmengine_moe_kv_lookup(kv, kvkey(0, 2), &d, &n) == LLMENGINE_MOE_OK);
    assert(d != NULL && n == 2048 && ((const uint8_t*)d)[0] == 3);

    /* 2. una quinta pagina evince la LRU: la 0 (mai toccata) sparisce */
    memset(page, 9, sizeof(page));
    assert(llmengine_moe_kv_insert(kv, kvkey(0, 4), page, sizeof(page)) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_kv_bytes_used(kv) == 8192); /* budget rispettato */
    assert(llmengine_moe_kv_lookup(kv, kvkey(0, 0), &d, &n) == LLMENGINE_MOE_OK);
    assert(d == NULL); /* la 0 e' stata evinta */

    /* 3. pagina piu' grande del budget -> NO_MEMORY */
    assert(llmengine_moe_kv_insert(kv, kvkey(0, 99), page, 100000) == LLMENGINE_MOE_ERR_NO_MEMORY);

    /* 4. sostituzione in place (stessa chiave, nuova dimensione) */
    uint8_t small[64]; memset(small, 7, sizeof(small));
    assert(llmengine_moe_kv_insert(kv, kvkey(0, 4), small, sizeof(small)) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_kv_lookup(kv, kvkey(0, 4), &d, &n) == LLMENGINE_MOE_OK);
    assert(d != NULL && n == 64 && ((const uint8_t*)d)[0] == 7);

    /* 5. INDIPENDENZA: expert cache budget 4096 (8 esperti da 1024 -> meta' evinti) */
    LlmEngineMoeExpertCache* ec = NULL;
    assert(llmengine_moe_cache_create(&ec, 4096) == LLMENGINE_MOE_OK);
    uint8_t blob[1024];
    for (int e = 0; e < 8; ++e) {
        memset(blob, e + 1, sizeof(blob));
        assert(llmengine_moe_cache_insert(ec, ekey((uint32_t)e), blob, sizeof(blob)) == LLMENGINE_MOE_OK);
    }
    /* gli ultimi 4 esperti (4..7) sono residenti */
    const void* ed = NULL; size_t en = 0;
    assert(llmengine_moe_cache_lookup(ec, ekey(7), &ed, &en) == LLMENGINE_MOE_OK);
    assert(ed != NULL);

    /* 5a. le pagine KV di (0,4) sono ancora intatte dopo l'eviction degli esperti */
    assert(llmengine_moe_kv_lookup(kv, kvkey(0, 4), &d, &n) == LLMENGINE_MOE_OK);
    assert(d != NULL && n == 64);

    /* 6. ora stresso la KV: gli esperti residenti NON devono sparire */
    memset(page, 5, sizeof(page));
    for (int p = 10; p < 20; ++p)
        assert(llmengine_moe_kv_insert(kv, kvkey(0, (uint32_t)p), page, sizeof(page)) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_kv_bytes_used(kv) <= llmengine_moe_kv_budget(kv));
    assert(llmengine_moe_cache_lookup(ec, ekey(7), &ed, &en) == LLMENGINE_MOE_OK);
    assert(ed != NULL); /* l'esperto 7 e' ancora residente */
    /* e la KV ha la pagina piu' recente */
    assert(llmengine_moe_kv_lookup(kv, kvkey(0, 19), &d, &n) == LLMENGINE_MOE_OK);
    assert(d != NULL);

    /* 7. metriche coherenti */
    LlmEngineMoeKvStats ks; llmengine_moe_kv_stats(kv, &ks);
    assert(ks.hits > 0 && ks.misses > 0 && ks.evictions > 0);

    /* 8. remove/clear */
    assert(llmengine_moe_kv_remove(kv, kvkey(0, 19)) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_kv_lookup(kv, kvkey(0, 19), &d, &n) == LLMENGINE_MOE_OK && d == NULL);
    llmengine_moe_kv_clear(kv);
    assert(llmengine_moe_kv_bytes_used(kv) == 0);

    /* 9. argomenti invalidi */
    LlmEngineMoeKvCache* bad = NULL;
    assert(llmengine_moe_kv_create(&bad, 0) == LLMENGINE_MOE_ERR_INVALID);

    llmengine_moe_kv_destroy(kv);
    llmengine_moe_cache_destroy(ec);
    printf("test_moe_kv: PASS\n");
    return 0;
}
