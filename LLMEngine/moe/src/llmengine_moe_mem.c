/* llmengine_moe_mem.c — contabilita' memoria per categoria + pressione (M10). */
#include "llmengine_moe_mem.h"

#include <stdlib.h>

struct LlmEngineMoeMemMgr {
    uint64_t total;
    uint64_t used;
    uint64_t per_cat[LLMENGINE_MOE_MEM_CAT_COUNT];
    double warn_frac, crit_frac;
};

const char* llmengine_moe_mem_cat_name(LlmEngineMoeMemCat c) {
    switch (c) {
        case LLMENGINE_MOE_MEM_DENSE:         return "DENSE";
        case LLMENGINE_MOE_MEM_EXPERT_RAM:    return "EXPERT_RAM";
        case LLMENGINE_MOE_MEM_EXPERT_DEVICE: return "EXPERT_DEVICE";
        case LLMENGINE_MOE_MEM_KV:            return "KV";
        case LLMENGINE_MOE_MEM_PREFETCH:      return "PREFETCH";
        case LLMENGINE_MOE_MEM_TEMP:          return "TEMP";
        default:                        return "?";
    }
}

LlmEngineMoeStatus llmengine_moe_mem_create(LlmEngineMoeMemMgr** out, uint64_t total_bytes,
                                double warn_frac, double crit_frac) {
    if (out == NULL || total_bytes == 0 || warn_frac <= 0.0 || crit_frac <= warn_frac ||
        crit_frac > 1.0)
        return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    LlmEngineMoeMemMgr* m = (LlmEngineMoeMemMgr*)calloc(1, sizeof(*m));
    if (m == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    m->total = total_bytes; m->warn_frac = warn_frac; m->crit_frac = crit_frac;
    *out = m;
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_mem_destroy(LlmEngineMoeMemMgr* m) { free(m); }

LlmEngineMoeStatus llmengine_moe_mem_reserve(LlmEngineMoeMemMgr* m, LlmEngineMoeMemCat cat, uint64_t bytes) {
    if (m == NULL || cat >= LLMENGINE_MOE_MEM_CAT_COUNT) return LLMENGINE_MOE_ERR_INVALID;
    if (bytes > m->total - m->used) return LLMENGINE_MOE_ERR_NO_MEMORY; /* sforerebbe il budget */
    m->used += bytes;
    m->per_cat[cat] += bytes;
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_mem_release(LlmEngineMoeMemMgr* m, LlmEngineMoeMemCat cat, uint64_t bytes) {
    if (m == NULL || cat >= LLMENGINE_MOE_MEM_CAT_COUNT) return;
    if (bytes > m->per_cat[cat]) bytes = m->per_cat[cat];
    m->per_cat[cat] -= bytes;
    if (bytes > m->used) m->used = 0; else m->used -= bytes;
}

LlmEngineMoePressure llmengine_moe_mem_pressure(const LlmEngineMoeMemMgr* m) {
    if (m == NULL) return LLMENGINE_MOE_PRESS_NORMAL;
    double f = (double)m->used / (double)m->total;
    if (f >= m->crit_frac) return LLMENGINE_MOE_PRESS_CRITICAL;
    if (f >= m->warn_frac) return LLMENGINE_MOE_PRESS_WARNING;
    return LLMENGINE_MOE_PRESS_NORMAL;
}

uint64_t llmengine_moe_mem_cat_used(const LlmEngineMoeMemMgr* m, LlmEngineMoeMemCat cat) {
    return (m == NULL || cat >= LLMENGINE_MOE_MEM_CAT_COUNT) ? 0 : m->per_cat[cat];
}

void llmengine_moe_mem_stats(const LlmEngineMoeMemMgr* m, LlmEngineMoeMemStats* out) {
    if (m == NULL || out == NULL) return;
    out->total_bytes = m->total;
    out->used_bytes = m->used;
    for (int i = 0; i < LLMENGINE_MOE_MEM_CAT_COUNT; ++i) out->per_cat[i] = m->per_cat[i];
}

double llmengine_moe_mem_prefetch_factor(const LlmEngineMoeMemMgr* m) {
    switch (llmengine_moe_mem_pressure(m)) {
        case LLMENGINE_MOE_PRESS_NORMAL:   return 1.0;
        case LLMENGINE_MOE_PRESS_WARNING:  return 0.5;
        default:                     return 0.0;
    }
}

int llmengine_moe_mem_allow_prefetch(const LlmEngineMoeMemMgr* m) {
    return llmengine_moe_mem_prefetch_factor(m) > 0.0;
}

int llmengine_moe_mem_evict_priority(LlmEngineMoeMemCat cat) {
    if (cat == LLMENGINE_MOE_MEM_PREFETCH) return 0;      /* da evincere PRIMA */
    if (cat == LLMENGINE_MOE_MEM_KV) return 2;
    if (cat == LLMENGINE_MOE_MEM_DENSE) return 3;     /* K3: trunk/dense pinnato — per ULTIMO */
    return 1;                                        /* expert attivi in mezzo */
}
