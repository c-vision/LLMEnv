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

/* Prefill a CHUNK layer-major: N token consecutivi (pos0..) attraversano il layer l prima del layer l+1.
 * Equivalente a N forward_step (le cache di ogni layer vedono i token nello stesso ordine). Se la pipeline
 * dichiarata non e' la canonica ricade sul percorso token-per-token. `out_hc_last` = stato mHC dell'ultimo
 * token (per i logit). */
static inline void llm_dsv4_forward_batch(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        LlmDsv4Caches* cs, const int* tokens, int pos0, int N, float* out_hc_last) {
    const int HC = c->n_hc, D = c->n_embd; const size_t hcn = (size_t)HC * D;
    if (N <= 0) return;
    if (g_dsv4_layer_nops > 0 || N == 1) {
        for (int t = 0; t < N; t++) llm_dsv4_forward_step(c, w, cs, tokens[t], pos0 + t, out_hc_last);
        return;
    }
    float* cur = (float*)malloc(sizeof(float) * hcn * (size_t)N);
    float* nxt = (float*)malloc(sizeof(float) * hcn * (size_t)N);
    for (int t = 0; t < N; t++) llm_dsv4_hc_from_embedding(c, w->token_embd + (long)tokens[t] * D, cur + hcn * t);
    int pf_hot = 0; { int v; if (llm_ai_mlx_opt_get_int("dsv4_prefetch_hot", &v) == 0) pf_hot = v; }
    for (int il = 0; il < c->n_layer; il++) {
        const LlmDsv4PackedExperts* pe = w->exps ? &w->exps[il] : NULL;
        /* ITEM B (opt-in, default OFF): prefetch ASINCRONO (madvise WILLNEED) dei K esperti piu'
         * frequenti del layer SUCCESSIVO mentre QUESTO calcola -> l'I/O si sovrappone alla GPU.
         * Vedi llm_dsv4_store_madvise_layer (misura: whole-layer NOCIVO). */
        if (pf_hot > 0 && w->exps && il + 1 < c->n_layer && w->exps[il + 1].store_madvise)
            w->exps[il + 1].store_madvise(w->exps[il + 1].store, il + 1);
        llm_dsv4_layer_forward_batch(c, &w->layers[il], pe, &cs->layers[il], cur, tokens, pos0, N, il, nxt);
        float* tmp = cur; cur = nxt; nxt = tmp;
    }
    llm_dsv4_pf_dump("prefill");   /* TEMP: profiler per-fase (no-op se dsv4_profile=0) */
    memcpy(out_hc_last, cur + hcn * (size_t)(N - 1), sizeof(float) * hcn);
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
