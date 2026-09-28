/* llmengine_moe_async.h — motore di I/O asincrono (M5, spec §23-§26).
 *
 * Coda a due priorita': REQUIRED (necessario al token corrente, prioritario) e PREFETCH
 * (speculativo, limitato e droppabile). Pool di worker **bounded** (mai thread illimitati),
 * shutdown pulito (join di tutti i thread). L'I/O avviene fuori dal thread chiamante, cosi'
 * letture e compute si sovrappongono. Integra la cache con single-flight (un esperto = una load). */
#ifndef LLMENGINE_MOE_ASYNC_H
#define LLMENGINE_MOE_ASYNC_H

#include "llmengine_moe_types.h"
#include "llmengine_moe_cache.h"

typedef struct LlmEngineMoeAsync LlmEngineMoeAsync;

typedef struct {
    uint64_t required;    /* richieste REQUIRED servite */
    uint64_t prefetch;    /* prefetch eseguite */
    uint64_t completed;   /* load completate (required+prefetch) */
    uint64_t dropped;     /* prefetch scartate (coda piena) */
    uint64_t cancelled;   /* prefetch cancellate */
} LlmEngineMoeAsyncStats;

/* Crea il pool. `n_workers` >= 1 (bounded). `cache` puo' essere NULL. `prefetch_cap` limita
 * la coda speculativa (0 => nessun prefetch). La `load` viene eseguita dai worker. */
LlmEngineMoeStatus llmengine_moe_async_create(LlmEngineMoeAsync** out, uint32_t n_workers,
                                  LlmEngineMoeCacheLoadFn load, void* ctx,
                                  LlmEngineMoeExpertCache* cache, uint32_t prefetch_cap);
/* Shutdown pulito: sveglia i worker, join di tutti, libera le code. */
void llmengine_moe_async_destroy(LlmEngineMoeAsync* a);

/* Richiede un esperto e ATTENDE il blob (via cache, single-flight). */
LlmEngineMoeStatus llmengine_moe_async_get(LlmEngineMoeAsync* a, LlmEngineMoeExpertKey key, size_t expected_size,
                               const void** data, size_t* size);

/* Prefetch speculativo: non blocca; se la coda e' piena la scarta (dropped++). */
LlmEngineMoeStatus llmengine_moe_async_prefetch(LlmEngineMoeAsync* a, LlmEngineMoeExpertKey key, size_t expected_size);

/* Cancella TUTTE le prefetch in coda (mai le REQUIRED). Ritorna quante ne ha cancellate. */
uint32_t llmengine_moe_async_cancel_prefetch(LlmEngineMoeAsync* a);

void llmengine_moe_async_stats(LlmEngineMoeAsync* a, LlmEngineMoeAsyncStats* out);

#endif /* LLMENGINE_MOE_ASYNC_H */
