/* llmengine_moe_cache.h — Expert cache (spec §16-§22).
 *
 * Contratto: chiave (layer, expert); budget in BYTE; LRU che evince il meno recente ma MAI
 * il piu' recente; dedup single-flight (due richieste dello stesso esperto => UNA lettura SSD);
 * metriche hits/misses/evictions/bytes_loaded. Thread-safe (mutex + condvar). */
#ifndef LLMENGINE_MOE_CACHE_H
#define LLMENGINE_MOE_CACHE_H

#include "llmengine_moe_types.h"

typedef struct LlmEngineMoeExpertCache LlmEngineMoeExpertCache;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_loaded;
    /* K2 (metrica "TRUE resident hit rate"): hit su un esperto GIA' residente all'inizio
     * dello step corrente (non caricato dal prefetch/load di questo step). `step` avanza
     * con `llmengine_moe_cache_step_begin`; senza chiamate resta 0 e `resident_hits`=0. */
    uint64_t resident_hits;
} LlmEngineMoeCacheStats;

/* Politica di eviction (M7). LRU e' il caso particolare di ROUTING con α=1, β=γ=δ=0. */
typedef enum { LLMENGINE_MOE_CACHE_LRU = 0, LLMENGINE_MOE_CACHE_ROUTING = 1 } LlmEngineMoeCachePolicy;

/* Callback di caricamento (tipicamente uno `pread` dal container). Scrive fino a `cap` byte. */
typedef LlmEngineMoeStatus (*LlmEngineMoeCacheLoadFn)(void* ctx, LlmEngineMoeExpertKey key,
                                          void* buf, size_t cap, size_t* got);

/* Crea una cache con budget `budget_bytes` (>0). `budget_bytes==0` => cache disabilitata
 * (nessuna allocazione; `get` carica senza memorizzare). */
LlmEngineMoeStatus llmengine_moe_cache_create(LlmEngineMoeExpertCache** out, uint64_t budget_bytes);
void llmengine_moe_cache_destroy(LlmEngineMoeExpertCache* c);

/* Lookup raw (NON conta hits/misses): se presente ritorna il blob (valido fino alla prossima
 * mutazione), altrimenti *data_out=NULL e ritorna OK (miss). */
LlmEngineMoeStatus llmengine_moe_cache_lookup(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                  const void** data_out, size_t* size_out);

/* Inserisce/sostituisce un blob (copia). Evince gli LRU per rientrare nel budget. */
LlmEngineMoeStatus llmengine_moe_cache_insert(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                  const void* data, size_t size);

/* Inserisce con hint di routing (M7): `route_prob` = probabilita' che il router lo scelga;
 * `future` = probabilita' che servira' a breve (dal prefetcher). Usati nello scoring. */
LlmEngineMoeStatus llmengine_moe_cache_insert_hint(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                       const void* data, size_t size,
                                       float route_prob, float future);

/* Aggiorna l'hint di routing di una chiave GIA' presente (no-op se assente). */
void llmengine_moe_cache_set_hint(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key, float route_prob, float future);

/* Imposta la politica di eviction e i pesi (M7). Deterministico (tie-break su (layer,expert)). */
void llmengine_moe_cache_set_policy(LlmEngineMoeExpertCache* c, LlmEngineMoeCachePolicy pol,
                              float alpha, float beta, float gamma, float delta);

/* Rimuove una chiave (kill dell'in-flight la marca fallita). */
LlmEngineMoeStatus llmengine_moe_cache_remove(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key);
void llmengine_moe_cache_clear(LlmEngineMoeExpertCache* c);

/* Get con SINGLE-FLIGHT: se piu' thread chiedono lo stesso key, la `load` viene invocata UNA
 * sola volta; gli altri attendono e ricevono lo stesso blob. Conta hits/misses/bytes_loaded.
 * `get_ex` riporta anche `was_hit` (1 = servito dalla cache, 0 = load eseguita): usato dal
 * layer async per emettere gli eventi CACHE_HIT/CACHE_MISS nel trace (replay K1). */
LlmEngineMoeStatus llmengine_moe_cache_get(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key, size_t expected_size,
                               LlmEngineMoeCacheLoadFn load, void* ctx,
                               const void** data_out, size_t* size_out);
LlmEngineMoeStatus llmengine_moe_cache_get_ex(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key, size_t expected_size,
                               LlmEngineMoeCacheLoadFn load, void* ctx,
                               const void** data_out, size_t* size_out, int* was_hit);

/* Avanza il "clock di step" (una unita' = una fase di routing/decodifica): un hit su una
 * entry residente da uno step PRECEDENTE conta in `resident_hits` (K2). Chiamare una volta
 * per step prima delle richieste dello step. */
void llmengine_moe_cache_step_begin(LlmEngineMoeExpertCache* c);

/* Hook di EVICTION (K1): chiamato (con lock tenuto, deve essere veloce e non bloccante) per
 * ogni esperto evitto — il runtime lo usa per emettere l'evento EVICT nel trace, cosi' il
 * summary e il replay offline contano le stesse eviction. */
typedef void (*LlmEngineMoeCacheEvictFn)(void* ctx, LlmEngineMoeExpertKey key);
void llmengine_moe_cache_set_evict_hook(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEvictFn fn, void* ctx);

void llmengine_moe_cache_stats(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheStats* out);

/* Byte attualmente residenti e budget configurato (M14: AI.STREAMSTATS). */
uint64_t llmengine_moe_cache_bytes_used(const LlmEngineMoeExpertCache* c);
uint64_t llmengine_moe_cache_budget(const LlmEngineMoeExpertCache* c);

/* ── D19: hotness appresa (frequenza per (layer,expert) dai routing osservati) + profilo ──
 * La cache puo' possedere una tabella `freq` alimentata da `note_route`; la policy ROUTING puo'
 * cosi' preferire gli esperti storicamente piu' caldi (che sopravvivono all'eviction) e la tabella
 * si persiste in un profilo (formato `GMSTATS 1`). Default: nessuna tabella, policy LRU invariata. */
LlmEngineMoeStatus llmengine_moe_cache_set_dims(LlmEngineMoeExpertCache* c, uint32_t layers, uint32_t experts);
void llmengine_moe_cache_note_route(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key, uint32_t count);
uint64_t llmengine_moe_cache_freq(const LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key);
uint64_t llmengine_moe_cache_fetches(const LlmEngineMoeExpertCache* c);
void llmengine_moe_cache_set_hotness(LlmEngineMoeExpertCache* c, int on);
LlmEngineMoeStatus llmengine_moe_cache_profile_save(const LlmEngineMoeExpertCache* c, const char* path);
LlmEngineMoeStatus llmengine_moe_cache_profile_load(LlmEngineMoeExpertCache* c, const char* path);

#endif /* LLMENGINE_MOE_CACHE_H */
