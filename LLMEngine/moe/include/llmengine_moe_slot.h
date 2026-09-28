/* llmengine_moe_slot.h — pool di slot fisici per esperti (M8, spec §33-§38).
 *
 * Separazione Model/Runtime/Cache: qui c'e' il DEVICE tier (GPU). Slot fisici di dimensione
 * fissa `{slot_id, expert_id, state, address, size}`; l'ID logico (layer,expert) e' rimappato a
 * uno slot fisico, e il remap e' aggiornato a ogni eviction. Il compute usa SEMPRE lo slot attuale.
 * Indipendente dalla cache RAM (M4): sono due tier distinti. */
#ifndef LLMENGINE_MOE_SLOT_H
#define LLMENGINE_MOE_SLOT_H

#include "llmengine_moe_types.h"

typedef enum { LLMENGINE_MOE_SLOT_FREE = 0, LLMENGINE_MOE_SLOT_LOADING = 1, LLMENGINE_MOE_SLOT_READY = 2 } LlmEngineMoeSlotState;

typedef struct {
    LlmEngineMoeExpertKey key;      /* chiave PIENA (layer, expert); spec §11: mai solo expert_id */
    uint8_t used;             /* 0 = libero */
    uint8_t state;
    size_t size;              /* byte del blob caricato */
    void* address;            /* handle device (backend-specifico) */
    uint64_t last_used;
} LlmEngineMoeSlot;

typedef struct LlmEngineMoeSlotPool LlmEngineMoeSlotPool;

LlmEngineMoeStatus llmengine_moe_slot_pool_create(LlmEngineMoeSlotPool** out, uint32_t n_slots, size_t slot_bytes);
void llmengine_moe_slot_pool_destroy(LlmEngineMoeSlotPool* p);

/* Acquire: se (layer,expert) e' presente ritorna lo stesso slot; altrimenti ne assegna uno
 * (evincendo il meno recente tra READY/FREE). Se evince, `evicted_key_out`/`evicted_slot_out`
 * dicono al backend cosa rilasciare (evicted_slot_out = UINT32_MAX se nessuna eviction). */
LlmEngineMoeStatus llmengine_moe_slot_pool_acquire(LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key,
                                       uint32_t* slot_id_out,
                                       LlmEngineMoeExpertKey* evicted_key_out,
                                       uint32_t* evicted_slot_out);

/* Marca lo slot di key come READY (upload completato). */
LlmEngineMoeStatus llmengine_moe_slot_pool_set_ready(LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key,
                                         const void* address, size_t size);

/* Rilascia la chiave (slot -> FREE, address=NULL). */
LlmEngineMoeStatus llmengine_moe_slot_pool_release(LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key);

/* Slot fisico attuale per una chiave (UINT32_MAX se assente). */
uint32_t llmengine_moe_slot_pool_lookup(const LlmEngineMoeSlotPool* p, LlmEngineMoeExpertKey key);

/* Copia della descrizione dello slot fisico (per il backend). */
LlmEngineMoeStatus llmengine_moe_slot_pool_get(const LlmEngineMoeSlotPool* p, uint32_t slot_id, LlmEngineMoeSlot* out);

uint32_t llmengine_moe_slot_pool_count(const LlmEngineMoeSlotPool* p);
uint32_t llmengine_moe_slot_pool_used(const LlmEngineMoeSlotPool* p);

#endif /* LLMENGINE_MOE_SLOT_H */
