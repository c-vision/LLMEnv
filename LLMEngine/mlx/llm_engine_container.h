/* llm_engine_container.h — lettura ENGINE-side di un GGUF verso f32 (denso + esperti stacked).
 * CONFINE (§0bis): il READER GGUF (`llm_gguf.h`) dà tabella+byte; il CODEC i dequant di formato
 * (`llmcodec_ggml_dequant_row`); qui si ORCHESTRA per il forward (quale tensore/esperto, layout).
 * Il container `GGUF` è DEPRECATO e NON è più usato. */
#ifndef LLM_ENGINE_CONTAINER_H
#define LLM_ENGINE_CONTAINER_H
#include <stdint.h>
#include <stddef.h>
#include "llm_gguf.h"
#include "llmengine_moe_types.h"   /* LlmEngineMoeStatus */

/* Legge un tensore DENSO del GGUF per nome canonico e lo dequanta in f32 (out cap>=n elementi). */
LlmEngineMoeStatus llm_engine_read_dense_f32(const LlmEngineMoeGguf* g, const char* name,
                                       float* out, size_t cap, size_t* n_out);

/* Legge l'esperto `expert` del layer `layer` dai 3 tensori stacked
 * `blk.<L>.ffn_{gate,up,down}_exps.weight` e ne dequanta gate/up/down in f32. */
LlmEngineMoeStatus llm_engine_read_expert_f32(const LlmEngineMoeGguf* g, int layer, uint32_t expert,
                                        float* gate, size_t gate_cap,
                                        float* up, size_t up_cap,
                                        float* down, size_t down_cap,
                                        uint32_t* rows_out, uint32_t* cols_out);

#endif /* LLM_ENGINE_CONTAINER_H */
