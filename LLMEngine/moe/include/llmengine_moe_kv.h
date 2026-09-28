/* llmengine_moe_kv.h — KV cache separata (M11, spec §41).
 *
 * Indipendente dall'expert cache: budget in BYTE dedicato, struttura dati dedicata. Evincere
 * esperti NON tocca le pagine KV e pressione KV NON tocca gli esperti. Chiave = (layer, page).
 * Uso dal thread GPM (§0ter): non thread-safe di proposito (nessuna sincronizzazione esterna). */
#ifndef LLMENGINE_MOE_KV_H
#define LLMENGINE_MOE_KV_H

#include "llmengine_moe_types.h"

typedef struct LlmEngineMoeKvCache LlmEngineMoeKvCache;

typedef struct {
    uint32_t layer;
    uint32_t page;
} LlmEngineMoeKvKey;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_stored;
} LlmEngineMoeKvStats;

/* Crea una KV cache con budget `budget_bytes` (>0). L'ownership e' del chiamante. */
LlmEngineMoeStatus llmengine_moe_kv_create(LlmEngineMoeKvCache** out, uint64_t budget_bytes);
void llmengine_moe_kv_destroy(LlmEngineMoeKvCache* kv);

/* Lookup (conta hits/misses, aggiorna la recency). Se presente *data_out punta al blob interno
 * (valido fino alla prossima mutazione); altrimenti *data_out=NULL e ritorna OK (miss). */
LlmEngineMoeStatus llmengine_moe_kv_lookup(LlmEngineMoeKvCache* kv, LlmEngineMoeKvKey key,
                               const void** data_out, size_t* size_out);

/* Inserisce/sostituisce una pagina (copia). Evince le LRU per rientrare nel budget.
 * Ritorna NO_MEMORY se la singola pagina eccede l'intero budget. */
LlmEngineMoeStatus llmengine_moe_kv_insert(LlmEngineMoeKvCache* kv, LlmEngineMoeKvKey key,
                               const void* data, size_t size);

LlmEngineMoeStatus llmengine_moe_kv_remove(LlmEngineMoeKvCache* kv, LlmEngineMoeKvKey key);
void llmengine_moe_kv_clear(LlmEngineMoeKvCache* kv);

/* Byte attualmente residenti e budget configurato. */
uint64_t llmengine_moe_kv_bytes_used(const LlmEngineMoeKvCache* kv);
uint64_t llmengine_moe_kv_budget(const LlmEngineMoeKvCache* kv);
void llmengine_moe_kv_stats(const LlmEngineMoeKvCache* kv, LlmEngineMoeKvStats* out);

#endif /* LLMENGINE_MOE_KV_H */
