/* llmengine_moe_prefetch.h — prefetch cross-layer (M6, spec §29-§32, §76).
 *
 * History di routing per layer (bounded, azzerabile) + **coupling table** deterministica
 * top-K: cosa tende a seguire un esperto al layer precedente. Predice top-N esperti del layer
 * successivo; il chiamante li prefetcha (via I/O async, M5) PRIMA del layer. Mai esperti
 * inesistenti (bound a `experts`). Metriche globali e per-layer: recall top1/topK, useful, wasted. */
#ifndef LLMENGINE_MOE_PREFETCH_H
#define LLMENGINE_MOE_PREFETCH_H

#include "llmengine_moe_types.h"

typedef struct LlmEngineMoePredictor LlmEngineMoePredictor;

typedef struct {
    uint64_t predictions;    /* layer su cui e' stata fatta una predizione */
    uint64_t actual_total;   /* somma degli esperti realmente usati (layer predetti) */
    uint64_t hits;           /* predetti ∩ actual */
    uint64_t wasted;         /* predetti \ actual */
    uint64_t top1_hits;      /* il 1o predetto era tra gli actual */
} LlmEngineMoePrefetchMetrics;

/* Crea il predittore: `layers` layer, `experts` per layer, `topn` esperti predetti per layer,
 * `coupling_k` voci per (layer, expert). */
LlmEngineMoeStatus llmengine_moe_predictor_create(LlmEngineMoePredictor** out, uint32_t layers, uint32_t experts,
                                      uint32_t topn, uint32_t coupling_k);
void llmengine_moe_predictor_destroy(LlmEngineMoePredictor* p);

/* Azzera history + metriche (il coupling table resta: e' conoscenza appresa). `full=1` azzera
 * anche il coupling. */
void llmengine_moe_predictor_reset(LlmEngineMoePredictor* p, int full);

/* Segnala il routing effettivo del layer `layer` (aggiorna coupling + metriche + predizione
 * del layer successivo). Ritorna il numero di esperti predetti per `layer+1` (0 se ultimo layer). */
LlmEngineMoeStatus llmengine_moe_predictor_notify(LlmEngineMoePredictor* p, uint32_t layer,
                                      const uint32_t* experts, uint32_t n);

/* Esperti predetti per `layer` (top-N, deterministico). `out` deve avere cap >= topn. */
LlmEngineMoeStatus llmengine_moe_predictor_predict(LlmEngineMoePredictor* p, uint32_t layer,
                                       uint32_t* out, uint32_t cap, uint32_t* n_out);

void llmengine_moe_predictor_metrics(LlmEngineMoePredictor* p, LlmEngineMoePrefetchMetrics* out);
/* M17: cambia la profondità (topn) senza perdere la coupling table; clamp a [1, experts]. */
LlmEngineMoeStatus llmengine_moe_predictor_set_topn(LlmEngineMoePredictor* p, uint32_t topn);
uint32_t llmengine_moe_predictor_topn(const LlmEngineMoePredictor* p);/* metriche del singolo layer: hits e actual (topK recall = hits/actual) */
LlmEngineMoeStatus llmengine_moe_predictor_layer_metrics(LlmEngineMoePredictor* p, uint32_t layer,
                                             uint64_t* hits, uint64_t* actual);

#endif /* LLMENGINE_MOE_PREFETCH_H */
