/* llmengine_moe_sim.h — Replay offline della cache esperti da un trace `.moetrace` (K1).
 *
 * Perche': il trace registra la sequenza di richieste (eventi CACHE_HIT/CACHE_MISS emessi
 * dal layer async); ri-giocarla su una cache a capacita' diverse permette il sizing
 * DETERMINISTICO e LOAD-FREE del budget (AGENTS §0sexies) senza rifare il run.
 *
 * La policy e' QUELLA DEL RUNTIME: il replay usa `llmengine_moe_cache` vera (stessa
 * funzione di score/eviction) — nessuna copia divergente della policy.
 *
 * Modello: esperti a dimensione UNIFORME (`expert_size`); la simulazione usa blob unitari
 * con capacita' in SLOT (`budget/expert_size`): la sequenza di hit/miss/eviction e' identica
 * a quella con i byte reali (l'eviction dipende solo dal confronto total>budget), senza
 * allocare la capacita' reale (un budget da 128 GB non entra in RAM). */
#ifndef LLMENGINE_MOE_SIM_H
#define LLMENGINE_MOE_SIM_H

#include "llmengine_moe_types.h"

typedef struct {
    uint64_t budget_bytes;      /* capacita' simulata (byte reali) */
    uint64_t slots;             /* capacita' in slot (budget/expert_size) */
    uint64_t requests;          /* richieste ri-giocate (HIT+MISS nel trace) */
    uint64_t steps;             /* step di routing nel trace */
    uint64_t hits;
    uint64_t resident_hits;     /* K2: hit su esperto residente da >=1 step */
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_loaded;      /* misses * expert_size (stima byte letti) */
    uint64_t unique_experts;    /* (layer,expert) distinti richiesti */
    double hit_rate;            /* hits / requests */
    double resident_rate;       /* resident_hits / hits (0 se hits=0) */
} LlmEngineMoeSimResult;

/* Ri-gioca il trace a `budget_bytes` (0 => usa il budget registrato nell'header del trace).
 * `expert_size_bytes` = byte per esperto (0 => usa quello dell'header; se anche l'header e'
 * a 0 la funzione ritorna LLMENGINE_MOE_ERR_INVALID). */
LlmEngineMoeStatus llmengine_moe_sim_replay(const char* trace_path, uint64_t budget_bytes,
                                            uint64_t expert_size_bytes, LlmEngineMoeSimResult* out);

#endif /* LLMENGINE_MOE_SIM_H */
