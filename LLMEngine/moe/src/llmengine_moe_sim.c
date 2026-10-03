/* llmengine_moe_sim.c — Replay offline della cache esperti (K1). Vedi header. */
#include "llmengine_moe_sim.h"
#include "llmengine_moe_cache.h"
#include "llmengine_moe_prof.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint32_t step;      /* step di routing (ROUTER delimita gli step) */
} SimReq;

typedef struct {
    uint64_t size;      /* byte per esperto (unitari nella simulazione) */
} SimLoadCtx;

static LlmEngineMoeStatus sim_load(void* ctx, LlmEngineMoeExpertKey key, void* buf, size_t cap, size_t* got) {
    SimLoadCtx* lc = (SimLoadCtx*)ctx;
    (void)key;
    if (lc == NULL || buf == NULL || got == NULL || cap < lc->size) { if (got) *got = 0; return LLMENGINE_MOE_ERR_INVALID; }
    memset(buf, 0, lc->size);
    *got = lc->size;
    return LLMENGINE_MOE_OK;
}

static int key64_less(const void* a, const void* b) {
    uint64_t ka = *(const uint64_t*)a, kb = *(const uint64_t*)b;
    return ka < kb ? -1 : (ka > kb ? 1 : 0);
}

LlmEngineMoeStatus llmengine_moe_sim_replay(const char* trace_path, uint64_t budget_bytes,
                                            uint64_t expert_size_bytes, LlmEngineMoeSimResult* out) {
    if (trace_path == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    memset(out, 0, sizeof(*out));

    uint64_t hdr_budget = 0, hdr_expert = 0;
    if (llmengine_moe_trace_meta(trace_path, &hdr_budget, &hdr_expert) != LLMENGINE_MOE_OK)
        return LLMENGINE_MOE_ERR_FORMAT;
    if (budget_bytes == 0) budget_bytes = hdr_budget;
    if (expert_size_bytes == 0) expert_size_bytes = hdr_expert;
    if (budget_bytes == 0 || expert_size_bytes == 0) return LLMENGINE_MOE_ERR_INVALID;

    LlmEngineMoeTraceReader* r = NULL;
    if (llmengine_moe_trace_open(trace_path, &r) != LLMENGINE_MOE_OK) return LLMENGINE_MOE_ERR_IO;

    SimReq* reqs = NULL;
    size_t n = 0, cap = 0;
    uint64_t* uniq = NULL;
    size_t n_uniq = 0, cap_uniq = 0;
    uint32_t step = 0;
    int have_requests = 0;

    LlmEngineMoeTraceRecord rec;
    int rc;
    while ((rc = llmengine_moe_trace_next(r, &rec)) == 1) {
        if (rec.type == LLMENGINE_MOE_TRACE_ROUTER) { step++; continue; }
        if (rec.type != LLMENGINE_MOE_TRACE_CACHE_HIT && rec.type != LLMENGINE_MOE_TRACE_CACHE_MISS) continue;
        have_requests = 1;
        if (n == cap) {
            cap = cap ? cap * 2 : 1024;
            SimReq* nr = (SimReq*)realloc(reqs, cap * sizeof(*reqs));
            if (nr == NULL) { free(reqs); llmengine_moe_trace_close(r); return LLMENGINE_MOE_ERR_NO_MEMORY; }
            reqs = nr;
        }
        reqs[n].layer = rec.layer; reqs[n].expert = rec.expert; reqs[n].step = step ? step - 1 : 0;
        n++;
        if (n_uniq == cap_uniq) {
            cap_uniq = cap_uniq ? cap_uniq * 2 : 1024;
            uint64_t* nu = (uint64_t*)realloc(uniq, cap_uniq * sizeof(*uniq));
            if (nu == NULL) { free(reqs); free(uniq); llmengine_moe_trace_close(r); return LLMENGINE_MOE_ERR_NO_MEMORY; }
            uniq = nu;
        }
        uniq[n_uniq++] = ((uint64_t)rec.layer << 32) | rec.expert;
    }
    llmengine_moe_trace_close(r);
    if (rc < 0) { free(reqs); free(uniq); return LLMENGINE_MOE_ERR_FORMAT; }
    if (!have_requests) { free(reqs); free(uniq); return LLMENGINE_MOE_ERR_FORMAT; }

    /* unique experts */
    if (n_uniq > 1) {
        qsort(uniq, n_uniq, sizeof(*uniq), key64_less);
        size_t w = 1;
        for (size_t i = 1; i < n_uniq; i++) if (uniq[i] != uniq[w - 1]) uniq[w++] = uniq[i];
        n_uniq = w;
    }

    /* Simulazione in SLOT con blob unitari: stessa sequenza di hit/miss/eviction dei byte
     * reali (l'eviction dipende solo da total > budget), senza allocare la capacita'. */
    uint64_t slots = budget_bytes / expert_size_bytes;
    if (slots == 0) slots = 1;

    LlmEngineMoeExpertCache* c = NULL;
    if (llmengine_moe_cache_create(&c, slots) != LLMENGINE_MOE_OK) { free(reqs); free(uniq); return LLMENGINE_MOE_ERR_NO_MEMORY; }

    SimLoadCtx lc = { 1 };
    uint32_t cur_step = 0;
    uint64_t resident = 0;
    for (size_t i = 0; i < n; i++) {
        while (cur_step <= reqs[i].step) { llmengine_moe_cache_step_begin(c); cur_step++; }
        LlmEngineMoeExpertKey k = { reqs[i].layer, reqs[i].expert };
        const void* data = NULL; size_t size = 0;
        if (llmengine_moe_cache_get(c, k, 1, sim_load, &lc, &data, &size) != LLMENGINE_MOE_OK) {
            llmengine_moe_cache_destroy(c); free(reqs); free(uniq);
            return LLMENGINE_MOE_ERR_IO;
        }
    }

    LlmEngineMoeCacheStats st;
    llmengine_moe_cache_stats(c, &st);
    resident = st.resident_hits;

    out->budget_bytes = budget_bytes;
    out->slots = slots;
    out->requests = n;
    out->steps = step;
    out->hits = st.hits;
    out->resident_hits = resident;
    out->misses = st.misses;
    out->evictions = st.evictions;
    out->bytes_loaded = st.misses * expert_size_bytes;
    out->unique_experts = n_uniq;
    out->hit_rate = n ? (double)st.hits / (double)n : 0.0;
    out->resident_rate = st.hits ? (double)resident / (double)st.hits : 0.0;

    llmengine_moe_cache_destroy(c);
    free(reqs);
    free(uniq);
    return LLMENGINE_MOE_OK;
}
