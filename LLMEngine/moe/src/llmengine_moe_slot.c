/* llmengine_moe_slot.c — pool di slot fisici + remap logico->fisico (M8, tier device). */
#include "llmengine_moe_slot.h"

#include <stdlib.h>
#include <string.h>

#define LLMENGINE_MOE_SLOT_EMPTY 0xFFFFFFFFu

static int key_eq(LlmEngineMoeExpertKey a, LlmEngineMoeExpertKey b) { return a.layer == b.layer && a.expert == b.expert; }

struct LlmEngineMoeSlotPool {
    LlmEngineMoeSlot* slots;
    uint32_t n_slots;
    size_t slot_bytes;
    uint64_t tick;
};

LlmEngineMoeStatus llmengine_moe_slot_pool_create(LlmEngineMoeSlotPool** out, uint32_t n_slots, size_t slot_bytes) {
    if (out == NULL || n_slots == 0 || slot_bytes == 0) return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    LlmEngineMoeSlotPool* p = (LlmEngineMoeSlotPool*)calloc(1, sizeof(*p));
    if (p == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    p->slots = (LlmEngineMoeSlot*)calloc(n_slots, sizeof(LlmEngineMoeSlot));
    if (p->slots == NULL) { free(p); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    p->n_slots = n_slots; p->slot_bytes = slot_bytes;
    for (uint32_t i = 0; i < n_slots; ++i) { p->slots[i].used = 0; p->slots[i].state = LLMENGINE_MOE_SLOT_FREE; }
    *out = p;
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_slot_pool_destroy(LlmEngineMoeSlotPool* p) {
    if (p == NULL) return;
    free(p->slots);
    free(p);
}

uint32_t llmengine_moe_slot_pool_lookup(const LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key) {
    if (p == NULL) return LLMENGINE_MOE_SLOT_EMPTY;
    for (uint32_t i = 0; i < p->n_slots; ++i)
        if (p->slots[i].used && key_eq(p->slots[i].key, key))
            return i;
    return LLMENGINE_MOE_SLOT_EMPTY;
}

LlmEngineMoeStatus llmengine_moe_slot_pool_get(const LlmEngineMoeSlotPool* p, uint32_t slot_id, LlmEngineMoeSlot* out) {
    if (p == NULL || out == NULL || slot_id >= p->n_slots) return LLMENGINE_MOE_ERR_INVALID;
    *out = p->slots[slot_id];
    return LLMENGINE_MOE_OK;
}

uint32_t llmengine_moe_slot_pool_count(const LlmEngineMoeSlotPool* p) { return (p == NULL) ? 0 : p->n_slots; }
uint32_t llmengine_moe_slot_pool_used(const LlmEngineMoeSlotPool* p) {
    if (p == NULL) return 0;
    uint32_t u = 0;
    for (uint32_t i = 0; i < p->n_slots; ++i) if (p->slots[i].used) u++;
    return u;
}

LlmEngineMoeStatus llmengine_moe_slot_pool_acquire(LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key,
                                       uint32_t* slot_id_out,
                                       LlmEngineMoeExpertKey* evicted_key_out,
                                       uint32_t* evicted_slot_out) {
    if (p == NULL || slot_id_out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *slot_id_out = LLMENGINE_MOE_SLOT_EMPTY;
    if (evicted_slot_out) *evicted_slot_out = LLMENGINE_MOE_SLOT_EMPTY;

    /* gia' presente? (per expert_id; il layer e' gestito dal chiamante come chiave piena) */
    for (uint32_t i = 0; i < p->n_slots; ++i) {
        if (p->slots[i].used && key_eq(p->slots[i].key, key)) {
            p->slots[i].last_used = ++p->tick;
            *slot_id_out = i;
            return LLMENGINE_MOE_OK;
        }
    }
    /* slot libero? */
    for (uint32_t i = 0; i < p->n_slots; ++i) {
        if (!p->slots[i].used) {
            p->slots[i].key = key;
            p->slots[i].used = 1;
            p->slots[i].state = LLMENGINE_MOE_SLOT_LOADING;
            p->slots[i].size = 0; p->slots[i].address = NULL;
            p->slots[i].last_used = ++p->tick;
            *slot_id_out = i;
            return LLMENGINE_MOE_OK;
        }
    }
    /* eviction: il meno recente (LRU tra gli slot non LOADING) */
    uint32_t victim = LLMENGINE_MOE_SLOT_EMPTY;
    uint64_t best = 0;
    for (uint32_t i = 0; i < p->n_slots; ++i) {
        if (p->slots[i].state == LLMENGINE_MOE_SLOT_LOADING) continue; /* in uso, non si evince */
        if (victim == LLMENGINE_MOE_SLOT_EMPTY || p->slots[i].last_used < best) { victim = i; best = p->slots[i].last_used; }
    }
    if (victim == LLMENGINE_MOE_SLOT_EMPTY) return LLMENGINE_MOE_ERR_NO_MEMORY; /* tutti in LOADING */
    if (evicted_key_out) *evicted_key_out = p->slots[victim].key;
    if (evicted_slot_out) *evicted_slot_out = victim;
    /* remap: il vecchio ID logico decade, il nuovo lo sostituisce */
    p->slots[victim].key = key;
    p->slots[victim].state = LLMENGINE_MOE_SLOT_LOADING;
    p->slots[victim].size = 0; p->slots[victim].address = NULL;
    p->slots[victim].last_used = ++p->tick;
    *slot_id_out = victim;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_slot_pool_set_ready(LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key,
                                         const void* address, size_t size) {
    if (p == NULL) return LLMENGINE_MOE_ERR_INVALID;
    uint32_t s = llmengine_moe_slot_pool_lookup(p, key);
    if (s == LLMENGINE_MOE_SLOT_EMPTY) return LLMENGINE_MOE_ERR_INVALID;
    if (size > p->slot_bytes) return LLMENGINE_MOE_ERR_NO_MEMORY; /* blob piu' grande dello slot */
    p->slots[s].address = (void*)address;
    p->slots[s].size = size;
    p->slots[s].state = LLMENGINE_MOE_SLOT_READY;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_slot_pool_release(LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key) {
    if (p == NULL) return LLMENGINE_MOE_ERR_INVALID;
    uint32_t s = llmengine_moe_slot_pool_lookup(p, key);
    if (s == LLMENGINE_MOE_SLOT_EMPTY) return LLMENGINE_MOE_ERR_INVALID;
    p->slots[s].used = 0;
    p->slots[s].state = LLMENGINE_MOE_SLOT_FREE;
    p->slots[s].size = 0;
    p->slots[s].address = NULL;
    return LLMENGINE_MOE_OK;
}
