/* llmengine_moe_prof.h — Profiler del runtime MoE (M12, spec §42-§43).
 *
 * Due livelli indipendenti:
 *  1. timing in-process per fase (ROUTER/CACHE/DISK/DECOMPRESS/UPLOAD/COMPUTE/COMBINE), con
 *     ogni fase abilitabile/disabilitabile per il benchmark; somma delle fasi = totale.
 *  2. trace binario `.moetrace` (eventi + timestamp monotoni), registrabile e RILEGGIBILE
 *     offline; due run sono confrontabili via summary/diff.
 *
 * Uso dal thread GPM (§0ter): non thread-safe di proposito, nessuna sincronizzazione esterna. */
#ifndef LLMENGINE_MOE_PROF_H
#define LLMENGINE_MOE_PROF_H

#include "llmengine_moe_types.h"

/* ── fasi di timing ────────────────────────────────────────────────────────── */
typedef enum {
    LLMENGINE_MOE_PH_ROUTER = 0,
    LLMENGINE_MOE_PH_CACHE,
    LLMENGINE_MOE_PH_DISK,
    LLMENGINE_MOE_PH_DECOMP,
    LLMENGINE_MOE_PH_UPLOAD,
    LLMENGINE_MOE_PH_COMPUTE,
    LLMENGINE_MOE_PH_COMBINE,
    LLMENGINE_MOE_PH_COUNT
} LlmEngineMoeProfPhase;

const char* llmengine_moe_prof_phase_name(LlmEngineMoeProfPhase ph);

typedef struct {
    uint64_t total_ns;
    uint64_t count;
    uint64_t min_ns;
    uint64_t max_ns;
} LlmEngineMoeProfPhaseStat;

typedef struct {
    LlmEngineMoeProfPhaseStat phase[LLMENGINE_MOE_PH_COUNT];
    uint64_t total_ns; /* somma delle fasi (§42: somma ≈ totale) */
} LlmEngineMoeProfStats;

/* ── eventi di trace ───────────────────────────────────────────────────────── */
typedef enum {
    LLMENGINE_MOE_TRACE_ROUTER = 1,
    LLMENGINE_MOE_TRACE_CACHE_HIT,
    LLMENGINE_MOE_TRACE_CACHE_MISS,
    LLMENGINE_MOE_TRACE_LOAD_BEGIN,
    LLMENGINE_MOE_TRACE_LOAD_END,
    LLMENGINE_MOE_TRACE_PREFETCH_BEGIN,
    LLMENGINE_MOE_TRACE_PREFETCH_END,
    LLMENGINE_MOE_TRACE_COMPUTE_BEGIN,
    LLMENGINE_MOE_TRACE_COMPUTE_END,
    LLMENGINE_MOE_TRACE_EVICT,
    /* estensione: durata di una fase misurata dal profiler (value = ns) */
    LLMENGINE_MOE_TRACE_PHASE_TIME
} LlmEngineMoeTraceEvent;

const char* llmengine_moe_trace_event_name(LlmEngineMoeTraceEvent ev);

typedef struct {
    uint32_t type;    /* LlmEngineMoeTraceEvent */
    uint32_t layer;
    uint32_t expert;
    uint32_t phase;   /* LlmEngineMoeProfPhase per PHASE_TIME, altrimenti 0 */
    uint64_t ts_ns;   /* clock monotono */
    uint64_t value;   /* byte letti / durata / ecc. */
} LlmEngineMoeTraceRecord;

/* ── profiler in-process ───────────────────────────────────────────────────── */
typedef struct LlmEngineMoeProfiler LlmEngineMoeProfiler;

LlmEngineMoeProfiler* llmengine_moe_prof_create(void);
void llmengine_moe_prof_destroy(LlmEngineMoeProfiler* p);

void llmengine_moe_prof_enable(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph, int on);
void llmengine_moe_prof_enable_all(LlmEngineMoeProfiler* p, int on);
int llmengine_moe_prof_enabled(const LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph);

/* Misura wall-clock attorno a una fase. `end` ritorna i ns misurati (0 se disabilitata). */
void llmengine_moe_prof_begin(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph);
uint64_t llmengine_moe_prof_end(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph);
/* Accumula una durata gia' misurata (per timing esterni). */
void llmengine_moe_prof_add(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph, uint64_t ns);

void llmengine_moe_prof_stats(const LlmEngineMoeProfiler* p, LlmEngineMoeProfStats* out);
void llmengine_moe_prof_reset(LlmEngineMoeProfiler* p);

/* ── trace su file ─────────────────────────────────────────────────────────── */
LlmEngineMoeStatus llmengine_moe_prof_trace_open(LlmEngineMoeProfiler* p, const char* path);
/* Come sopra, ma registra nell'header i parametri del run utili al replay offline (K1):
 * `cache_budget` (byte della cache esperti) e `expert_size` (byte per esperto, per il
 * dimensionamento a slot). */
LlmEngineMoeStatus llmengine_moe_prof_trace_open_ex(LlmEngineMoeProfiler* p, const char* path,
                                          uint64_t cache_budget, uint64_t expert_size);
void llmengine_moe_prof_trace_close(LlmEngineMoeProfiler* p);
int llmengine_moe_prof_trace_active(const LlmEngineMoeProfiler* p);
/* Emette un evento (no-op se il trace non e' attivo). `end` emette anche PHASE_TIME. */
void llmengine_moe_prof_event(LlmEngineMoeProfiler* p, LlmEngineMoeTraceEvent ev,
                        uint32_t layer, uint32_t expert, uint32_t phase, uint64_t value);

/* ── lettura offline ───────────────────────────────────────────────────────── */
typedef struct LlmEngineMoeTraceReader LlmEngineMoeTraceReader;

LlmEngineMoeStatus llmengine_moe_trace_open(const char* path, LlmEngineMoeTraceReader** out);
/* 1 = record letto; 0 = EOF; <0 = errore (header invalido/troncato). */
int llmengine_moe_trace_next(LlmEngineMoeTraceReader* r, LlmEngineMoeTraceRecord* out);
void llmengine_moe_trace_close(LlmEngineMoeTraceReader* r);

/* Riassunto di un trace: timing per fase (da PHASE_TIME) + conteggi eventi. */
typedef struct {
    LlmEngineMoeProfStats timing;
    uint64_t router;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t loads;
    uint64_t prefetches;
    uint64_t evictions;
    uint64_t records;
} LlmEngineMoeTraceSummary;

LlmEngineMoeStatus llmengine_moe_trace_summary(const char* path, LlmEngineMoeTraceSummary* out);

/* Meta del trace (header): budget cache del run e byte-per-esperto (0 se non registrati).
 * Usata dal replay offline K1 (`tools/moe_sim`). */
LlmEngineMoeStatus llmengine_moe_trace_meta(const char* path, uint64_t* cache_budget, uint64_t* expert_size);

/* Confronto tra due run (delta per fase e per conteggio): a - b. */
typedef struct {
    int64_t phase_delta_ns[LLMENGINE_MOE_PH_COUNT];
    int64_t total_delta_ns;
    int64_t router_delta;
    int64_t cache_hits_delta;
    int64_t cache_misses_delta;
    int64_t loads_delta;
    int64_t prefetches_delta;
    int64_t evictions_delta;
} LlmEngineMoeTraceDiff;

void llmengine_moe_trace_diff(const LlmEngineMoeTraceSummary* a, const LlmEngineMoeTraceSummary* b,
                        LlmEngineMoeTraceDiff* out);

#endif /* LLMENGINE_MOE_PROF_H */
