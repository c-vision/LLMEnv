/* test_moe_cache_adaptive.c — D19 gate: hotness appresa (freq per (layer,expert)) + profilo.
 * Verifica: (a) set_dims/note_route aggiornano freq/fetches; (b) hotness ON protegge dall'eviction
 * un esperto caldo (dove l'LRU lo evincerebbe); (c) profile save/load round-trip; (d) senza dims la
 * hotness e' inerte. Exit 0 = PASS. */
#include "llmengine_moe_cache.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static LlmEngineMoeExpertKey K(uint32_t l, uint32_t e) { LlmEngineMoeExpertKey k = {l, e}; return k; }

int main(void) {
    /* (a) dims + note_route */
    LlmEngineMoeExpertCache* c = NULL;
    assert(llmengine_moe_cache_create(&c, 1u << 20) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_set_dims(c, 4, 8) == LLMENGINE_MOE_OK);
    llmengine_moe_cache_note_route(c, K(1, 3), 5);
    llmengine_moe_cache_note_route(c, K(1, 3), 2);
    assert(llmengine_moe_cache_freq(c, K(1, 3)) == 7);
    assert(llmengine_moe_cache_freq(c, K(0, 0)) == 0);
    assert(llmengine_moe_cache_fetches(c) == 7);
    llmengine_moe_cache_destroy(c);

    /* (d) senza dims: note_route conta i fetch ma freq resta 0 (hotness inerte) */
    c = NULL;
    assert(llmengine_moe_cache_create(&c, 1u << 20) == LLMENGINE_MOE_OK);
    llmengine_moe_cache_set_hotness(c, 1);
    llmengine_moe_cache_note_route(c, K(0, 0), 9);
    assert(llmengine_moe_cache_freq(c, K(0, 0)) == 0);
    assert(llmengine_moe_cache_fetches(c) == 9);
    llmengine_moe_cache_destroy(c);

    /* (b) LRU vs hotness: budget = 3 x 64 byte; H e' l'esperto "caldo". */
    uint8_t blob[64];
    const void* p = NULL; size_t n = 0;

    /* LRU (default): H inserito per primo -> evitto per primo */
    c = NULL;
    assert(llmengine_moe_cache_create(&c, 3 * 64) == LLMENGINE_MOE_OK);
    memset(blob, 0xAA, sizeof(blob));
    assert(llmengine_moe_cache_insert(c, K(0, 0), blob, 64) == LLMENGINE_MOE_OK);   /* H */
    assert(llmengine_moe_cache_insert(c, K(0, 1), blob, 64) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_insert(c, K(0, 2), blob, 64) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_insert(c, K(0, 3), blob, 64) == LLMENGINE_MOE_OK);   /* evict LRU = H */
    assert(llmengine_moe_cache_lookup(c, K(0, 0), &p, &n) == LLMENGINE_MOE_OK && p == NULL);
    llmengine_moe_cache_destroy(c);

    /* hotness ON: H ha freq alta -> sopravvive alle ondate di esperti freddi */
    c = NULL;
    assert(llmengine_moe_cache_create(&c, 3 * 64) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_set_dims(c, 2, 8) == LLMENGINE_MOE_OK);
    llmengine_moe_cache_note_route(c, K(0, 0), 100);   /* H = caldo */
    llmengine_moe_cache_set_hotness(c, 1);
    assert(llmengine_moe_cache_insert(c, K(0, 0), blob, 64) == LLMENGINE_MOE_OK);   /* H */
    for (uint32_t e = 1; e < 8; ++e)
        assert(llmengine_moe_cache_insert(c, K(0, e), blob, 64) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_lookup(c, K(0, 0), &p, &n) == LLMENGINE_MOE_OK && p != NULL); /* H presente */

    /* (c) profile save/load round-trip */
    const char* path = "/tmp/test_moe_cache_adaptive.stats";
    assert(llmengine_moe_cache_profile_save(c, path) == LLMENGINE_MOE_OK);
    llmengine_moe_cache_destroy(c);

    LlmEngineMoeExpertCache* c2 = NULL;
    assert(llmengine_moe_cache_create(&c2, 1u << 20) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_set_dims(c2, 2, 8) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_profile_load(c2, path) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_cache_freq(c2, K(0, 0)) == 100);
    llmengine_moe_cache_destroy(c2);
    remove(path);

    printf("test_moe_cache_adaptive: PASS\n");
    return 0;
}
