/* llm_dsv4_generate.h — driver di generazione DS4: cache per-layer, embedding, layer cached,
 * testa finale, argmax. Header-only. Test: llm_dsv4_generate_test.c. */
#ifndef LLM_DSV4_GENERATE_H
#define LLM_DSV4_GENERATE_H

#include <stdlib.h>
#include "llm_dsv4_layer.h"
#include "llm_dsv4_head.h"

typedef struct {
    LlmDsv4LayerCache* layers;   /* [n_layer] */
    int n_layer;
} LlmDsv4Caches;

/* `ctx` = contesto totale; la capacita' delle righe compresse e' per-layer
 * `ctx/ratio + 2` (ds4 #1010), non un valore unico. */
static inline void llm_dsv4_caches_init(LlmDsv4Caches* cs, const LlmDsv4Config* c, int raw_cap, int ctx) {
    cs->n_layer = c->n_layer;
    cs->layers = (LlmDsv4LayerCache*)calloc((size_t)c->n_layer, sizeof(LlmDsv4LayerCache));
    for (int il = 0; il < c->n_layer; il++) {
        int ratio = llm_dsv4_layer_compress_ratio(c, il);
        int comp_cap = ratio > 0 ? ctx / ratio + 2 : 0;
        llm_dsv4_cache_init(&cs->layers[il], c, il, raw_cap, comp_cap);
    }
}
static inline void llm_dsv4_caches_free(LlmDsv4Caches* cs) {
    for (int il = 0; il < cs->n_layer; il++) llm_dsv4_cache_free(&cs->layers[il]);
    free(cs->layers); cs->layers = NULL; cs->n_layer = 0;
}

/* Un passo: token -> out_hc (stato mHC finale). Usa w->exps[il] se presente (esperti packed). */
static inline void llm_dsv4_forward_step(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        LlmDsv4Caches* cs, int token, int pos, float* out_hc) {
    int HC = c->n_hc, D = c->n_embd;
    float* cur = (float*)malloc(sizeof(float) * (size_t)HC * D);
    float* nxt = (float*)malloc(sizeof(float) * (size_t)HC * D);
    llm_dsv4_hc_from_embedding(c, w->token_embd + (long)token * D, cur);
    for (int il = 0; il < c->n_layer; il++) {
        const LlmDsv4PackedExperts* pe = w->exps ? &w->exps[il] : NULL;
        llm_dsv4_layer_forward_cached_pe(c, &w->layers[il], pe, &cs->layers[il], cur, token, pos, il, nxt);
        float* t = cur; cur = nxt; nxt = t;
    }
    memcpy(out_hc, cur, sizeof(float) * (size_t)HC * D);
    free(nxt); free(cur);
}

/* Genera n_new token da un prompt di token ids (greedy). out_tokens [pos0+n_new].
 * Ritorna i logits dell'ultimo passo in logits_out [n_vocab] se non NULL. */
static inline void llm_dsv4_generate(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        LlmDsv4Caches* cs, int* tokens, int n_prompt, int n_new, float* logits_out) {
    int HC = c->n_hc, D = c->n_embd;
    float* hc = (float*)malloc(sizeof(float) * (size_t)HC * D);
    float* logits = (float*)malloc(sizeof(float) * (size_t)c->n_vocab);
    int pos = 0;
    for (int i = 0; i < n_prompt; i++, pos++) llm_dsv4_forward_step(c, w, cs, tokens[i], pos, hc);
    for (int i = 0; i < n_new; i++) {
        llm_dsv4_output_logits(c, w, hc, logits);
        int nt = llm_dsv4_argmax(logits, c->n_vocab);
        tokens[n_prompt + i] = nt;
        llm_dsv4_forward_step(c, w, cs, nt, pos, hc); pos++;
    }
    if (logits_out) llm_dsv4_output_logits(c, w, hc, logits_out);
    free(logits); free(hc);
}

#endif /* LLM_DSV4_GENERATE_H */
