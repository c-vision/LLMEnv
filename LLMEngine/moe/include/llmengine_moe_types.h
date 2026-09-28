/* llmengine_moe_types.h — tipi, codici di stato e metadati del MoE runtime LLMEnv.
 *
 * Piano: Todo/(open) - MoE-Expert-Streaming-OutOfCore-Plan.md §2 (spec §1-§7).
 * Regola §80/§11: nessuna assunzione semantica su un esperto; la chiave e' SEMPRE
 * (layer, expert) — mai il solo expert_id (esperti di layer diversi = oggetti diversi). */
#ifndef LLMENGINE_MOE_TYPES_H
#define LLMENGINE_MOE_TYPES_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    LLMENGINE_MOE_OK                  = 0,
    LLMENGINE_MOE_ERR_INVALID         = -1,
    LLMENGINE_MOE_ERR_NO_MEMORY       = -2,
    LLMENGINE_MOE_ERR_MODEL_NOT_FOUND = -3,
    LLMENGINE_MOE_ERR_IO              = -4,
    LLMENGINE_MOE_ERR_NOT_IMPLEMENTED = -5,
    LLMENGINE_MOE_ERR_ADAPTER         = -6,
    LLMENGINE_MOE_ERR_FORMAT          = -7,
    LLMENGINE_MOE_ERR_DEVICE          = -8
} LlmEngineMoeStatus;

/* Descrizione leggibile di un codice di stato (mai NULL). */
const char* llmengine_moe_status_str(LlmEngineMoeStatus s);

/* Chiave canonica di un esperto: MAI solo expert_id (spec §11). */
typedef struct {
    uint32_t layer;
    uint32_t expert;
} LlmEngineMoeExpertKey;

static inline int llmengine_moe_key_eq(LlmEngineMoeExpertKey a, LlmEngineMoeExpertKey b) {
    return a.layer == b.layer && a.expert == b.expert;
}

/* Metadati del modello (riempiti dall'adapter al load; spec §5-§7). */
typedef struct {
    uint64_t total_bytes;        /* dimensione totale del container */
    uint64_t non_routed_bytes;   /* pesi sempre residenti */
    uint64_t routed_bytes;       /* somma di tutti gli esperti routed */
    uint32_t layer_count;
    uint32_t expert_count;       /* esperti per layer */
    uint32_t experts_per_tok;    /* top-k del router */
    uint32_t hidden_size;
} LlmEngineMoeModelMeta;

#endif /* LLMENGINE_MOE_TYPES_H */
