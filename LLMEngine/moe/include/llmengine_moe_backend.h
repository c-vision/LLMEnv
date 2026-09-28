/* llmengine_moe_backend.h — MoEModelAdapter: interfaccia tra il runtime generico e l'architettura
 * del modello (spec §2, §8-§10, §80).
 *
 * Principio: il runtime NON conosce Qwen/DeepSeek/GLM. Sa solo di esperti = tensor+metadata.
 * L'adapter e' l'unico punto che sa interpretare l'architettura. `self` e' il suo stato. */
#ifndef LLMENGINE_MOE_BACKEND_H
#define LLMENGINE_MOE_BACKEND_H

#include "llmengine_moe_types.h"

struct LlmEngineMoeRuntime; /* forward declaration (evita include circolare con llmengine_moe.h) */

typedef struct LlmEngineMoeModelAdapter {
    const char* name;
    void* self;

    /* Legge i metadati del modello. Ritorna LLMENGINE_MOE_ERR_MODEL_NOT_FOUND / _IO / _FORMAT. */
    LlmEngineMoeStatus (*load)(void* self, const char* path, LlmEngineMoeModelMeta* meta);

    /* Routing del layer: riempie `experts_out[0..k)` e `weights_out[0..k)`. */
    LlmEngineMoeStatus (*router)(void* self, uint32_t layer, const float* hidden, uint32_t hidden_size,
                           uint32_t* experts_out, float* weights_out, uint32_t k);

    /* Calcolo di UN esperto (gate/up/down); `in`->`out`. */
    LlmEngineMoeStatus (*expert_compute)(void* self, LlmEngineMoeExpertKey key, const float* in, uint32_t in_size,
                                   float* out, uint32_t out_size);

    /* Combinazione pesata degli output degli esperti selezionati. */
    LlmEngineMoeStatus (*combine)(void* self, uint32_t layer, const float* const* expert_out,
                            const float* weights, uint32_t n, uint32_t size, float* out);

    /* Forward completo del modello. Riceve il runtime per FETCHARE gli esperti on-demand
     * (`llmengine_moe_get_expert`): l'adapter non sa se l'esperto arriva da RAM o da SSD. */
    LlmEngineMoeStatus (*forward)(void* self, struct LlmEngineMoeRuntime* rt, const float* in, uint32_t seq,
                            uint32_t hidden_size, float* out);

    /* Libera `self` (gia' registrato; il runtime lo invoca in destroy). */
    void (*destroy)(void* self);
} LlmEngineMoeModelAdapter;

#endif /* LLMENGINE_MOE_BACKEND_H */
