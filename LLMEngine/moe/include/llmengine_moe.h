/* llmengine_moe.h — MoERuntime: API pubblica del runtime MoE out-of-core di LLMEnv.
 *
 * Separa quattro decisioni (spec §2, MAI fuse): Model (architettura), Runtime (dove stanno
 * gli esperti), Cache (cosa resta residente), Prefetcher (cosa anticipare). Questa API e' il
 * contratto "MoE virtual memory": il backend riceve esperti attivi + handle e NON deve sapere
 * che il modello e' enorme.
 *
 * M1: lifecycle + adapter registry + load + forward/generate. Il container `GGUF` e la cache
 * arrivano in M2/M4 (API dichiarate, mai risultati finti). */
#ifndef LLMENGINE_MOE_RUNTIME_H
#define LLMENGINE_MOE_RUNTIME_H

#include "llmengine_moe_types.h"
#include "llmengine_moe_backend.h"
#include "llmengine_moe_prefetch.h"
#include "llmengine_moe_mem.h"
#include "llmengine_moe_cache.h"
#include "llmengine_moe_kv.h"
#include "llmengine_moe_prof.h"

typedef struct LlmEngineMoeRuntime LlmEngineMoeRuntime;

/* Crea il runtime (adapter non registrato, nessun modello). NULL su OOM. */
LlmEngineMoeRuntime* llmengine_moe_create(void);

/* Distrugge: invoca adapter.destroy dello stato attivo, chiude lo storage, libera. NULL-safe. */
void llmengine_moe_destroy(LlmEngineMoeRuntime* rt);

/* Registra (copia) un adapter. Il primo registrato che sa caricare il modello vince. */
LlmEngineMoeStatus llmengine_moe_register_adapter(LlmEngineMoeRuntime* rt, const LlmEngineMoeModelAdapter* adapter);

/* Carica il modello da `path`: MODEL_NOT_FOUND se il file non esiste; poi prova gli adapter
 * registrati. Su successo riempie i metadati. */
LlmEngineMoeStatus llmengine_moe_load_model(LlmEngineMoeRuntime* rt, const char* path);

int llmengine_moe_is_loaded(const LlmEngineMoeRuntime* rt);

/* Metadati del modello caricato (NULL se non caricato). */
const LlmEngineMoeModelMeta* llmengine_moe_meta(const LlmEngineMoeRuntime* rt);

/* ── Configurazione di load (M14) — multi-modello, swap/resident, device ────── */
/* Modalita' di memoria: SWAP (default) = streaming out-of-core (esperti on-demand da disco);
 * RESIDENT = precarica TUTTI gli esperti in RAM (nessuno swap durante l'inferenza). */
typedef enum { LLMENGINE_MOE_MODE_SWAP = 0, LLMENGINE_MOE_MODE_RESIDENT = 1 } LlmEngineMoeLoadMode;
typedef enum { LLMENGINE_MOE_DEVICE_AUTO = 0, LLMENGINE_MOE_DEVICE_CPU = 1, LLMENGINE_MOE_DEVICE_METAL = 2 } LlmEngineMoeDevice;

/* M21 — gerarchia di residenza del modello (stati di memoria). Oggi il runtime copre WARM
 * (dense residente + esperti on-demand) e ACTIVE (dense + tutti gli esperti residenti); DISK =
 * non caricato. COLD (struttura pronta, pesi fuori) è previsto ma non ancora distinto. */
typedef enum {
    LLMENGINE_MOE_RESIDENCY_DISK = 0,    /* non caricato (solo file) */
    LLMENGINE_MOE_RESIDENCY_COLD = 1,    /* struttura nota, pesi non residenti (futuro) */
    LLMENGINE_MOE_RESIDENCY_WARM = 2,    /* dense residente, esperti on-demand (SWAP) */
    LLMENGINE_MOE_RESIDENCY_ACTIVE = 3   /* dense + tutti gli esperti residenti (RESIDENT) */
} LlmEngineMoeResidency;

typedef struct {
    LlmEngineMoeLoadMode mode;         /* SWAP (default) | RESIDENT */
    LlmEngineMoeDevice device;         /* AUTO | CPU | METAL (METAL fallisce se assente) */
    uint64_t cache_mb;           /* expert cache in MB (0 = disabilitata; RESIDENT la forza) */
    uint32_t async_workers;      /* worker di I/O asincrono (0 = disabilitato) */
    uint32_t prefetch_topn;      /* esperti predetti per layer (0 = nessun prefetch) */
    uint32_t prefetch_cap;       /* bound della coda prefetch */
    int hotness;                 /* D19: expert cache ADATTIVA (hotness appresa + profilo) */
} LlmEngineMoeRuntimeConfig;

/* Config di default: SWAP, AUTO, e i valori dalle env LLMENGINE_MOE_* (compatibilita'). */
void llmengine_moe_config_defaults(LlmEngineMoeRuntimeConfig* out);

/* Carica con config ESPLICITA (i comandi BASIC usano questa). `llmengine_moe_load_model` = defaults. */
LlmEngineMoeStatus llmengine_moe_load_model_ex(LlmEngineMoeRuntime* rt, const char* path, const LlmEngineMoeRuntimeConfig* cfg);

/* Statistiche di streaming/inferenza del runtime (M14, per AI.STREAMSTATS). */
typedef struct {
    int loaded;
    LlmEngineMoeLoadMode mode;
    LlmEngineMoeDevice device;
    uint32_t layers;
    uint32_t experts;
    uint64_t bytes_read;        /* byte letti da disco nella sessione */
    uint64_t expert_bytes;      /* byte totali degli esperti routed */
    uint64_t resident_bytes;    /* byte residenti nella expert cache */
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
    uint64_t cache_resident_hits;   /* K2: hit su esperto residente da >=1 step (non prefetch) */
    uint64_t prefetch_predictions;
    uint64_t prefetch_hits;
    uint64_t prefetch_wasted;
    uint64_t unresolvable_experts;  /* K5: esperti richiesti ma assenti/corrotti (output non affidabile) */
} LlmEngineMoeStreamStats;

LlmEngineMoeStatus llmengine_moe_stream_stats(const LlmEngineMoeRuntime* rt, LlmEngineMoeStreamStats* out);

/* Pubblica le metriche dello store di un adapter (GGUF stile DS4). Usate da llmengine_moe_stream_stats
 * quando rt->cache e' assente. Chiamato dall'adapter a fine forward. */
void llmengine_moe_set_stream_metrics(LlmEngineMoeRuntime* rt, uint64_t bytes_read, uint64_t hits,
                                uint64_t misses, uint64_t evictions, uint64_t resident_hits,
                                uint64_t resident_bytes, uint64_t budget_bytes,
                                uint64_t unresolvable_experts);

/* Modalita' di memoria corrente del runtime (SWAP/RESIDENT). */
LlmEngineMoeLoadMode llmengine_moe_mode(const LlmEngineMoeRuntime* rt);
/* cache_mb configurata al load (0 = non specificata). Usata dallo store GGUF dell'adapter. */
uint64_t llmengine_moe_cache_mb(const LlmEngineMoeRuntime* rt);

/* ── M21: gerarchia di residenza — stato + transizioni ──
 * `llmengine_moe_promote` → ACTIVE (tutti gli esperti residenti, dalla RAM); `llmengine_moe_demote` → WARM
 * (on-demand). `llmengine_moe_residency` ritorna lo stato corrente. */
LlmEngineMoeResidency llmengine_moe_residency(const LlmEngineMoeRuntime* rt);
LlmEngineMoeStatus llmengine_moe_promote(LlmEngineMoeRuntime* rt);
LlmEngineMoeStatus llmengine_moe_demote(LlmEngineMoeRuntime* rt);

/* Lettura di un tensore DENSE del container per nome (usata dall'adattatore/provider). */
LlmEngineMoeStatus llmengine_moe_get_tensor(const LlmEngineMoeRuntime* rt, const char* name, void* buf, size_t cap, size_t* got);

/* ── Riconfigurazione a caldo (M14): policy realmente cambiante ──────────────
 * Cambiano il componente e hanno effetto immediato. In RESIDENT la cache e' fissa (INVALID). */
LlmEngineMoeStatus llmengine_moe_set_device(LlmEngineMoeRuntime* rt, LlmEngineMoeDevice device);
LlmEngineMoeStatus llmengine_moe_set_cache_mb(LlmEngineMoeRuntime* rt, uint64_t cache_mb); /* 0 = disabilita */
LlmEngineMoeStatus llmengine_moe_set_prefetch(LlmEngineMoeRuntime* rt, uint32_t topn, uint32_t coupling_k); /* 0 = off */
LlmEngineMoeStatus llmengine_moe_set_async(LlmEngineMoeRuntime* rt, uint32_t workers, uint32_t cap); /* 0 = off */

/* Forward generico: `in` [seq*hidden] -> `out` [seq*hidden]. */
LlmEngineMoeStatus llmengine_moe_forward(LlmEngineMoeRuntime* rt, const float* in, uint32_t seq, float* out);

/* Fetch on-demand di un esperto dal container `GGUF` (cache in M4). Ritorna in `blob_out`
 * un puntatore al blob, valido fino alla prossima chiamata. `bytes_read` conta i byte presi
 * da disco (miss) — metrica: SSD bytes/token (§8). */
LlmEngineMoeStatus llmengine_moe_get_expert(LlmEngineMoeRuntime* rt, LlmEngineMoeExpertKey key,
                                const void** blob_out, size_t* size_out);

/* Byte totali letti da disco per gli esperti in questa sessione. */
uint64_t llmengine_moe_bytes_read(const LlmEngineMoeRuntime* rt);

/* ── M17: controller ADATTIVO (prefetch depth + cache budget), opt-in LLMENGINE_MOE_ADAPTIVE=1 ── */
int llmengine_moe_adaptive_enabled(const LlmEngineMoeRuntime* rt);
uint32_t llmengine_moe_adaptive_topn(const LlmEngineMoeRuntime* rt);

/* ── D19: attiva/disattiva la expert cache ADATTIVA (hotness appresa + profilo `<container>.stats`).
 * Richiede una cache (cache_mb>0). Default OFF (policy LRU, comportamento invariato). ── */
LlmEngineMoeStatus llmengine_moe_set_hotness(LlmEngineMoeRuntime* rt, int on);

/* ── M18: statistiche d'uso per-esperto, persistite in `<container>.stats` — ADVISORY ──
 * File mancante/corrotto NON blocca il load (il chiamante ignora l'errore). */
uint64_t llmengine_moe_expert_hits(const LlmEngineMoeRuntime* rt, LlmEngineMoeExpertKey key);
uint64_t llmengine_moe_expert_fetches(const LlmEngineMoeRuntime* rt);
LlmEngineMoeStatus llmengine_moe_stats_save(const LlmEngineMoeRuntime* rt, const char* path);
LlmEngineMoeStatus llmengine_moe_stats_load(LlmEngineMoeRuntime* rt, const char* path);

/* Prefetch speculativo (M5): non blocca; no-op se l'async I/O non e' attivo. */
LlmEngineMoeStatus llmengine_moe_prefetch_expert(LlmEngineMoeRuntime* rt, LlmEngineMoeExpertKey key);

/* M6: segnala il routing effettivo del layer -> aggiorna il predittore cross-layer e prefetcha
 * (async) gli esperti predetti per il layer successivo. No-op se il predittore non e' attivo. */
LlmEngineMoeStatus llmengine_moe_notify_routing(LlmEngineMoeRuntime* rt, uint32_t layer,
                                    const uint32_t* experts, uint32_t n);
LlmEngineMoeStatus llmengine_moe_prefetch_metrics(LlmEngineMoeRuntime* rt, LlmEngineMoePrefetchMetrics* out);
void llmengine_moe_reset_prefetch(LlmEngineMoeRuntime* rt, int full);

/* Genera fino a `n_out` token greedy a partire dal prompt. Ritorna il numero generato o
 * un codice negativo (LlmEngineMoeStatus). Toy-orchestrazione: il `forward` fornisce lo stato, la
 * scelta del token e' argmax (M1; l'adapter reale fornira' lm_head/logits). */
int llmengine_moe_generate(LlmEngineMoeRuntime* rt, const int32_t* prompt, uint32_t n_prompt,
                     int32_t* out, uint32_t n_out);

/* M10: budget di memoria unificato (categorie + soglie NORMAL/WARNING/CRITICAL). Il manager e'
 * di proprieta' del runtime; chiama con total_bytes=0 per rimuoverlo. Soglie in (0,1], warn<crit. */
LlmEngineMoeStatus llmengine_moe_set_memory_budget(LlmEngineMoeRuntime* rt, uint64_t total_bytes,
                                       double warn_frac, double crit_frac);
const LlmEngineMoeMemMgr* llmengine_moe_mem(const LlmEngineMoeRuntime* rt);

/* M11: budget SEPARATI per expert cache e KV cache (0 = disabilita quella cache). Le due cache
 * sono indipendenti: l'eviction di una NON tocca l'altra. Sostituisce eventuali cache esistenti. */
LlmEngineMoeStatus llmengine_moe_set_cache_budgets(LlmEngineMoeRuntime* rt, uint64_t expert_bytes, uint64_t kv_bytes);
const LlmEngineMoeExpertCache* llmengine_moe_expert_cache(const LlmEngineMoeRuntime* rt);
const LlmEngineMoeKvCache* llmengine_moe_kv(const LlmEngineMoeRuntime* rt);

/* M12: profiler opzionale (di proprieta' del chiamante). Default NULL => zero overhead.
 * Se impostato, il runtime misura DISK (read esperti) e COMPUTE (forward) ed emette eventi. */
void llmengine_moe_set_profiler(LlmEngineMoeRuntime* rt, LlmEngineMoeProfiler* prof);

/* D19: hotness opt-in corrente (0/1) — letto dall'adapter per la policy dello store reale. */
int llmengine_moe_hotness(const LlmEngineMoeRuntime* rt);
LlmEngineMoeProfiler* llmengine_moe_profiler(const LlmEngineMoeRuntime* rt);

#endif /* LLMENGINE_MOE_RUNTIME_H */
