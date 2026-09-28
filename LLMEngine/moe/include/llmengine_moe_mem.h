/* llmengine_moe_mem.h — memory manager (M10, spec §39-§40).
 *
 * Ogni allocazione e' CLASSIFICATA per categoria; il manager traccia il totale per categoria e
 * la pressione (NORMAL/WARNING/CRITICAL in base a frazioni del budget). Policy: a WARNING il
 * prefetch e' ridotto; a CRITICAL e' sospeso e si evincono le allocazioni PREFETCH **prima** di
 * quelle attive (EXPERT_RAM/EXPERT_DEVICE). Mai thread/asincrono qui: pura contabilita'. */
#ifndef LLMENGINE_MOE_MEM_H
#define LLMENGINE_MOE_MEM_H

#include "llmengine_moe_types.h"

typedef enum {
    LLMENGINE_MOE_MEM_DENSE = 0,
    LLMENGINE_MOE_MEM_EXPERT_RAM,
    LLMENGINE_MOE_MEM_EXPERT_DEVICE,
    LLMENGINE_MOE_MEM_KV,
    LLMENGINE_MOE_MEM_PREFETCH,
    LLMENGINE_MOE_MEM_TEMP,
    LLMENGINE_MOE_MEM_CAT_COUNT
} LlmEngineMoeMemCat;

typedef enum { LLMENGINE_MOE_PRESS_NORMAL = 0, LLMENGINE_MOE_PRESS_WARNING = 1, LLMENGINE_MOE_PRESS_CRITICAL = 2 } LlmEngineMoePressure;

const char* llmengine_moe_mem_cat_name(LlmEngineMoeMemCat c);

typedef struct {
    uint64_t total_bytes;
    uint64_t used_bytes;
    uint64_t per_cat[LLMENGINE_MOE_MEM_CAT_COUNT];
} LlmEngineMoeMemStats;

typedef struct LlmEngineMoeMemMgr LlmEngineMoeMemMgr;

/* `warn_frac`/`crit_frac` in (0,1], warn<crit. */
LlmEngineMoeStatus llmengine_moe_mem_create(LlmEngineMoeMemMgr** out, uint64_t total_bytes,
                                double warn_frac, double crit_frac);
void llmengine_moe_mem_destroy(LlmEngineMoeMemMgr* m);

/* Prenota `bytes` per la categoria. Ritorna NO_MEMORY se sfora il budget totale. */
LlmEngineMoeStatus llmengine_moe_mem_reserve(LlmEngineMoeMemMgr* m, LlmEngineMoeMemCat cat, uint64_t bytes);
void llmengine_moe_mem_release(LlmEngineMoeMemMgr* m, LlmEngineMoeMemCat cat, uint64_t bytes);

LlmEngineMoePressure llmengine_moe_mem_pressure(const LlmEngineMoeMemMgr* m);
uint64_t llmengine_moe_mem_cat_used(const LlmEngineMoeMemMgr* m, LlmEngineMoeMemCat cat);
void llmengine_moe_mem_stats(const LlmEngineMoeMemMgr* m, LlmEngineMoeMemStats* out);

/* Policy prefetch (§40): fattore di budget [0,1]. NORMAL=1 (pieno), WARNING=0.5 (ridotto),
 * CRITICAL=0 (sospeso). Il chiamante scala la quota prefetch con questo fattore. */
double llmengine_moe_mem_prefetch_factor(const LlmEngineMoeMemMgr* m);

/* Policy prefetch binaria: 1 se il fattore > 0. */
int llmengine_moe_mem_allow_prefetch(const LlmEngineMoeMemMgr* m);

/* Ordine di eviction a CRITICAL: priorita' piu' BASSA = da evincere PRIMA. A pari pressione le
 * PREFETCH (priorita' 0) vengono evinte prima degli expert attivi (1) e delle KV (2). */
int llmengine_moe_mem_evict_priority(LlmEngineMoeMemCat cat);

#endif /* LLMENGINE_MOE_MEM_H */
