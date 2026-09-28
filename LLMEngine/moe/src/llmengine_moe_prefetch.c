/* llmengine_moe_prefetch.c — predittore cross-layer per MoE (M6).
 *
 * Coupling: per ogni (layer, expert_prev) una top-K di (expert_next, count). Predizione del layer
 * L = top-N, aggregando i coupling degli esperti usati a L-1. Fallback: popolarita' per-layer.
 * Deterministico: ordinamento per (count desc, expert asc). */
#include "llmengine_moe_prefetch.h"

#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t expert, count; } Couple;
typedef struct { Couple* v; uint32_t n; } CplEntry;

struct LlmEngineMoePredictor {
    uint32_t layers, experts, topn, k;
    CplEntry* table;          /* layers*experts */
    uint32_t* pop;            /* layers*experts: popolarita' (fallback) */
    uint32_t* cur;            /* layers*(k+1): esperti effettivi dell'ultimo notify */
    uint32_t* cur_n;          /* per layer */
    uint32_t* pred;           /* layers*topn: ultima predizione per layer */
    uint32_t* pred_n;
    uint64_t* layer_hits;
    uint64_t* layer_actual;
    int* layer_predicted;
    LlmEngineMoePrefetchMetrics m;
};

LlmEngineMoeStatus llmengine_moe_predictor_create(LlmEngineMoePredictor** out, uint32_t layers, uint32_t experts,
                                      uint32_t topn, uint32_t coupling_k) {
    if (out == NULL || layers == 0 || experts == 0 || topn == 0 || coupling_k == 0)
        return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    LlmEngineMoePredictor* p = (LlmEngineMoePredictor*)calloc(1, sizeof(*p));
    if (p == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    p->layers = layers; p->experts = experts; p->topn = topn; p->k = coupling_k;
    p->table = (CplEntry*)calloc((size_t)layers * experts, sizeof(CplEntry));
    p->pop = (uint32_t*)calloc((size_t)layers * experts, sizeof(uint32_t));
    p->cur = (uint32_t*)calloc((size_t)layers * (coupling_k + 1), sizeof(uint32_t));
    p->cur_n = (uint32_t*)calloc(layers, sizeof(uint32_t));
    p->pred = (uint32_t*)calloc((size_t)layers * topn, sizeof(uint32_t));
    p->pred_n = (uint32_t*)calloc(layers, sizeof(uint32_t));
    p->layer_hits = (uint64_t*)calloc(layers, sizeof(uint64_t));
    p->layer_actual = (uint64_t*)calloc(layers, sizeof(uint64_t));
    p->layer_predicted = (int*)calloc(layers, sizeof(int));
    if (!p->table || !p->pop || !p->cur || !p->cur_n || !p->pred || !p->pred_n ||
        !p->layer_hits || !p->layer_actual || !p->layer_predicted) {
        llmengine_moe_predictor_destroy(p);
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    *out = p;
    return LLMENGINE_MOE_OK;
}

/* M17: cambia la profondità (topn) SENZA perdere la coupling table appresa (adaptive prefetch). */
LlmEngineMoeStatus llmengine_moe_predictor_set_topn(LlmEngineMoePredictor* p, uint32_t topn) {
    if (p == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (topn == 0) topn = 1;
    if (topn > p->experts) topn = p->experts;
    if (topn == p->topn) return LLMENGINE_MOE_OK;
    uint32_t* np = (uint32_t*)realloc(p->pred, (size_t)p->layers * topn * sizeof(uint32_t));
    if (np == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    p->pred = np;
    p->topn = topn;
    memset(p->pred_n, 0, (size_t)p->layers * sizeof(uint32_t));
    return LLMENGINE_MOE_OK;
}
uint32_t llmengine_moe_predictor_topn(const LlmEngineMoePredictor* p) { return p == NULL ? 0 : p->topn; }

void llmengine_moe_predictor_destroy(LlmEngineMoePredictor* p) {    if (p == NULL) return;
    if (p->table)
        for (size_t i = 0; i < (size_t)p->layers * p->experts; ++i) free(p->table[i].v);
    free(p->table); free(p->pop); free(p->cur); free(p->cur_n);
    free(p->pred); free(p->pred_n); free(p->layer_hits); free(p->layer_actual);
    free(p->layer_predicted);
    free(p);
}

void llmengine_moe_predictor_reset(LlmEngineMoePredictor* p, int full) {
    if (p == NULL) return;
    memset(p->cur_n, 0, p->layers * sizeof(uint32_t));
    memset(p->pred_n, 0, p->layers * sizeof(uint32_t));
    memset(p->layer_hits, 0, p->layers * sizeof(uint64_t));
    memset(p->layer_actual, 0, p->layers * sizeof(uint64_t));
    memset(p->layer_predicted, 0, p->layers * sizeof(int));
    memset(&p->m, 0, sizeof(p->m));
    if (full) {
        for (size_t i = 0; i < (size_t)p->layers * p->experts; ++i) {
            free(p->table[i].v); p->table[i].v = NULL; p->table[i].n = 0;
        }
        memset(p->pop, 0, (size_t)p->layers * p->experts * sizeof(uint32_t));
    }
}

/* aggiorna la top-K di (layer, e_prev) con e_next */
static void coupling_bump(LlmEngineMoePredictor* p, uint32_t layer, uint32_t e_prev, uint32_t e_next) {
    CplEntry* ce = &p->table[(size_t)layer * p->experts + e_prev];
    if (ce->v == NULL) {
        ce->v = (Couple*)calloc(p->k, sizeof(Couple));
        ce->n = 0;
        if (ce->v == NULL) return;
    }
    /* cerca e_next */
    for (uint32_t i = 0; i < ce->n; ++i)
        if (ce->v[i].expert == e_next) { ce->v[i].count++; return; }
    if (ce->n < p->k) { ce->v[ce->n].expert = e_next; ce->v[ce->n].count = 1; ce->n++; return; }
    /* sostituisci la voce con count minore se questa ha gia' piu' evidenza (non c'e': count=1) */
    uint32_t mi = 0;
    for (uint32_t i = 1; i < ce->n; ++i) if (ce->v[i].count < ce->v[mi].count) mi = i;
    if (ce->v[mi].count <= 1) { ce->v[mi].expert = e_next; ce->v[mi].count = 1; }
}

LlmEngineMoeStatus llmengine_moe_predictor_predict(LlmEngineMoePredictor* p, uint32_t layer,
                                       uint32_t* out, uint32_t cap, uint32_t* n_out) {
    if (p == NULL || out == NULL || n_out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *n_out = 0;
    if (layer >= p->layers) return LLMENGINE_MOE_ERR_INVALID;
    uint32_t want = (p->topn < cap) ? p->topn : cap;

    /* accumulatori temporanei: count per esperto del layer */
    uint32_t* acc = (uint32_t*)calloc(p->experts, sizeof(uint32_t));
    if (acc == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    int have = 0;
    if (layer > 0 && p->cur_n[layer - 1] > 0) {
        for (uint32_t i = 0; i < p->cur_n[layer - 1]; ++i) {
            uint32_t e_prev = p->cur[(size_t)(layer - 1) * (p->k + 1) + i];
            CplEntry* ce = &p->table[(size_t)(layer - 1) * p->experts + e_prev];
            for (uint32_t j = 0; j < ce->n; ++j) {
                if (ce->v[j].expert < p->experts) { acc[ce->v[j].expert] += ce->v[j].count; have = 1; }
            }
        }
    }
    if (!have) { /* fallback: popolarita' del layer */
        for (uint32_t e = 0; e < p->experts; ++e) { acc[e] = p->pop[(size_t)layer * p->experts + e]; if (acc[e]) have = 1; }
    }
    /* top-`want` deterministico: count desc, expert asc (selezione semplice) */
    for (uint32_t s = 0; s < want; ++s) {
        int best = -1;
        for (uint32_t e = 0; e < p->experts; ++e) {
            if (acc[e] == 0) continue;
            if (best < 0 || acc[e] > acc[best] || (acc[e] == acc[best] && e < (uint32_t)best)) best = (int)e;
        }
        if (best < 0) break;
        out[(*n_out)++] = (uint32_t)best;
        acc[best] = 0; /* consuma */
    }
    free(acc);
    memcpy(&p->pred[(size_t)layer * p->topn], out, (size_t)(*n_out) * sizeof(uint32_t));
    p->pred_n[layer] = *n_out;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_predictor_notify(LlmEngineMoePredictor* p, uint32_t layer,
                                      const uint32_t* experts, uint32_t n) {
    if (p == NULL || (experts == NULL && n > 0)) return LLMENGINE_MOE_ERR_INVALID;
    if (layer >= p->layers) return LLMENGINE_MOE_ERR_INVALID;

    /* 1. metrica della predizione fatta per QUESTO layer (se c'era) */
    if (p->layer_predicted[layer]) {
        uint32_t hits = 0;
        for (uint32_t i = 0; i < p->pred_n[layer]; ++i) {
            uint32_t pr = p->pred[(size_t)layer * p->topn + i];
            for (uint32_t j = 0; j < n; ++j) if (experts[j] == pr) { hits++; break; }
        }
        int top1 = 0;
        if (p->pred_n[layer] > 0) {
            uint32_t p0 = p->pred[(size_t)layer * p->topn];
            for (uint32_t j = 0; j < n; ++j) if (experts[j] == p0) { top1 = 1; break; }
        }
        p->m.predictions++;
        p->m.actual_total += n;
        p->m.hits += hits;
        p->m.wasted += (p->pred_n[layer] > hits) ? (p->pred_n[layer] - hits) : 0;
        p->m.top1_hits += (uint64_t)top1;
        p->layer_hits[layer] += hits;
        p->layer_actual[layer] += n;
    }

    /* 2. registra il routing effettivo (history bounded) */
    uint32_t cap = (n < p->k + 1) ? n : (p->k + 1);
    for (uint32_t i = 0; i < cap; ++i) p->cur[(size_t)layer * (p->k + 1) + i] = experts[i];
    p->cur_n[layer] = cap;
    for (uint32_t i = 0; i < n; ++i)
        if (experts[i] < p->experts) p->pop[(size_t)layer * p->experts + experts[i]]++;

    /* 3. aggiorna il coupling (layer-1 -> layer) */
    if (layer > 0 && p->cur_n[layer - 1] > 0) {
        for (uint32_t i = 0; i < p->cur_n[layer - 1]; ++i) {
            uint32_t e_prev = p->cur[(size_t)(layer - 1) * (p->k + 1) + i];
            if (e_prev >= p->experts) continue;
            for (uint32_t j = 0; j < n; ++j)
                if (experts[j] < p->experts) coupling_bump(p, layer - 1, e_prev, experts[j]);
        }
    }

    /* 4. predici il layer successivo */
    if (layer + 1 < p->layers) {
        uint32_t no = 0;
        uint32_t tmp[64];
        uint32_t cap2 = (p->topn < 64) ? p->topn : 64;
        llmengine_moe_predictor_predict(p, layer + 1, tmp, cap2, &no);
        p->layer_predicted[layer + 1] = 1;
    }
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_predictor_metrics(LlmEngineMoePredictor* p, LlmEngineMoePrefetchMetrics* out) {
    if (p == NULL || out == NULL) return;
    *out = p->m;
}

LlmEngineMoeStatus llmengine_moe_predictor_layer_metrics(LlmEngineMoePredictor* p, uint32_t layer,
                                             uint64_t* hits, uint64_t* actual) {
    if (p == NULL || hits == NULL || actual == NULL || layer >= p->layers) return LLMENGINE_MOE_ERR_INVALID;
    *hits = p->layer_hits[layer];
    *actual = p->layer_actual[layer];
    return LLMENGINE_MOE_OK;
}
