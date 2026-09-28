/* test_moe_slot.c — M8 gate (slot pool): allocazione/riuso/eviction + remap logico->fisico. */
#include "llmengine_moe_slot.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static LlmEngineMoeExpertKey K(uint32_t l, uint32_t e) { LlmEngineMoeExpertKey k = {l, e}; return k; }

int main(void) {
    LlmEngineMoeSlotPool* p = NULL;
    assert(llmengine_moe_slot_pool_create(&p, 4, 256) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_slot_pool_count(p) == 4 && llmengine_moe_slot_pool_used(p) == 0);

    /* 4 acquire -> 4 slot distinti, poi READY */
    uint32_t slots[4];
    for (uint32_t e = 0; e < 4; ++e) {
        uint32_t s = 0xFFFFFFFF; LlmEngineMoeExpertKey ev = {0, 0}; uint32_t evs = 0xFFFFFFFF;
        assert(llmengine_moe_slot_pool_acquire(p, K(0, e), &s, &ev, &evs) == LLMENGINE_MOE_OK);
        assert(evs == 0xFFFFFFFF);      /* nessuna eviction */
        slots[e] = s;
        static uint8_t buf[256];
        assert(llmengine_moe_slot_pool_set_ready(p, K(0, e), buf, 128) == LLMENGINE_MOE_OK);
    }
    assert(llmengine_moe_slot_pool_used(p) == 4);
    /* lookup: chiave -> slot fisico corretto */
    for (uint32_t e = 0; e < 4; ++e) assert(llmengine_moe_slot_pool_lookup(p, K(0, e)) == slots[e]);

    /* acquire di un 5o esperto -> evince il meno recente (slot di e=0) e RIMMAPPA */
    uint32_t s5 = 0xFFFFFFFF; LlmEngineMoeExpertKey ev = {0, 0}; uint32_t evs = 0xFFFFFFFF;
    assert(llmengine_moe_slot_pool_acquire(p, K(0, 100), &s5, &ev, &evs) == LLMENGINE_MOE_OK);
    assert(evs == slots[0]);
    assert(ev.expert == 0);
    assert(llmengine_moe_slot_pool_lookup(p, K(0, 0)) == 0xFFFFFFFF); /* il vecchio ID logico decade */
    assert(llmengine_moe_slot_pool_lookup(p, K(0, 100)) == slots[0]); /* il nuovo usa lo slot liberato */
    LlmEngineMoeSlot desc;
    assert(llmengine_moe_slot_pool_get(p, slots[0], &desc) == LLMENGINE_MOE_OK);
    assert(desc.key.expert == 100 && desc.state == LLMENGINE_MOE_SLOT_LOADING);

    /* chiavi di LAYER diversi non collidono (spec §11) */
    assert(llmengine_moe_slot_pool_used(p) == 4);
    uint32_t s6 = 0xFFFFFFFF; uint32_t e6 = 0xFFFFFFFF;
    assert(llmengine_moe_slot_pool_acquire(p, K(1, 1), &s6, &ev, &e6) == LLMENGINE_MOE_OK); /* evince il piu' vecchio restante */
    assert(llmengine_moe_slot_pool_lookup(p, K(1, 1)) == s6);
    assert(llmengine_moe_slot_pool_lookup(p, K(0, 1)) != s6); /* layer diverso -> chiave diversa */

    /* release -> slot libero */
    assert(llmengine_moe_slot_pool_release(p, K(1, 1)) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_slot_pool_lookup(p, K(1, 1)) == 0xFFFFFFFF);
    assert(llmengine_moe_slot_pool_used(p) == 3);

    llmengine_moe_slot_pool_destroy(p);
    printf("test_moe_slot: PASS\n");
    return 0;
}
