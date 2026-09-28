/* llmengine_moe_expert.h — esperto = tensor data + metadata (spec §80).
 * Nessuna semantica: il runtime lo tratta come blob; la chiave e' (layer, expert). */
#ifndef LLMENGINE_MOE_EXPERT_H
#define LLMENGINE_MOE_EXPERT_H

#include "llmengine_moe_types.h"

typedef struct {
    LlmEngineMoeExpertKey key;
    uint64_t storage_offset; /* offset assoluto nel container */
    uint64_t byte_size;      /* byte del blob (uncompressed) */
    uint32_t tensor_count;   /* gate/up/down = 3 tipicamente */
} LlmEngineMoeExpertDesc;

#endif /* LLMENGINE_MOE_EXPERT_H */
