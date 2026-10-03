/* llm_dsv4_engine.h — engine DS4 (DeepSeek V4 Flash): carica un GGUF (mmap), costruisce i pesi,
 * inizializza le cache per-layer e genera greedy. C puro, self-contained.
 * Header-only. Uso: LlmDsv4Engine* e = llm_dsv4_engine_load(path, ctx); ... */
#ifndef LLM_DSV4_ENGINE_H
#define LLM_DSV4_ENGINE_H

#include <stdlib.h>
#include <string.h>
#include "llm_dsv4_generate.h"
#include "llm_dsv4_store.h"
#include "llm_dsv4_gpu_ctx.h"
#include "llm_gguf.h"
#include "llm_engine_opts.h"   /* opzioni dsv4_prefill_chunk / dsv4_prefill_mm (registro, no env) */

#define LLM_DSV4_PREFILL_CHUNK 2048   /* default (item C: chunk grandi = meno riletture esperti; 512→2048 su synth 8192: 67.9→105.7 t/s). 4096 via opzione dsv4_prefill_chunk (fino a 150 t/s a 4096 tok; piu' memoria) */

typedef struct {
    LlmDsv4Weights w;
    LlmDsv4Caches caches;
    LlmEngineMoeGguf gguf;
    void* map; size_t maplen;
    int n_ctx;
    char path[1024];
    LlmDsv4Store* store;   /* NULL = residente (mmap) */
    int pos;               /* posizione corrente (per append/generate incrementali) */
    int has_pending, pending;
    float* logits;         /* [n_vocab] cache dei logit del prossimo token */
} LlmDsv4Engine;

/* Callback per il MoE: legge l'esperto dallo store (residente o swap). */
static inline const uint8_t* llm_dsv4_engine_store_cb(void* s, int layer, int e,
        const uint8_t** g, const uint8_t** u, const uint8_t** d) {
    return llm_dsv4_store_expert((LlmDsv4Store*)s, layer, e, g, u, d);
}

static inline void llm_dsv4_engine_store_prefetch_cb(void* s, int layer, const int* experts, int n) {
    llm_dsv4_store_prefetch((LlmDsv4Store*)s, layer, experts, n);
}
static inline void llm_dsv4_engine_store_madvise_cb(void* s, int layer) {
    llm_dsv4_store_madvise_layer((LlmDsv4Store*)s, layer);
}

static inline int llm_dsv4_engine_q4_stacks_cb(void* s, int layer, const uint8_t** g, const uint8_t** u, const uint8_t** d) {
    return llm_dsv4_store_q4_stacks((LlmDsv4Store*)s, layer, g, u, d);
}
static inline int llm_dsv4_engine_q4_ensure_cb(void* s, int layer, const int* experts, int n) {
    return llm_dsv4_store_q4_ensure((LlmDsv4Store*)s, layer, experts, n);
}

/* Abilita lo SWAP degli esperti: store con cache LRU (cap_bundles bundle gate+up+down).
 * Va chiamata dopo il load (usa path+gguf+cfg). Ritorna false su errore. */
static inline bool llm_dsv4_engine_enable_swap(LlmDsv4Engine* e, int cap_bundles) {
    if (!e || !e->w.exps) return false;
    if (!e->store) e->store = llm_dsv4_store_open(e->path, &e->gguf, &e->w.cfg, LLM_DSV4_STORE_SWAP, cap_bundles);
    if (!e->store) return false;
    for (int il = 0; il < e->w.cfg.n_layer; il++) {
        e->w.exps[il].store = e->store;
        e->w.exps[il].store_get = llm_dsv4_engine_store_cb;
        e->w.exps[il].store_prefetch = llm_dsv4_engine_store_prefetch_cb;
        e->w.exps[il].store_madvise = llm_dsv4_engine_store_madvise_cb;
        e->w.exps[il].q4_stacks = llm_dsv4_engine_q4_stacks_cb;
        e->w.exps[il].q4_ensure = llm_dsv4_engine_q4_ensure_cb;
    }
    return true;
}

/* `packed=true`: esperti tenuti packed (mmap), nessun dequant a f32 (richiesto per il modello reale). */
static inline LlmDsv4Engine* llm_dsv4_engine_load_ex(const char* path, int n_ctx, bool packed) {
    LlmDsv4Engine* e = (LlmDsv4Engine*)calloc(1, sizeof(LlmDsv4Engine));
    if (!e) return NULL;
    snprintf(e->path, sizeof e->path, "%s", path);
    if (!llm_gguf_open_file(path, &e->gguf, &e->map, &e->maplen)) {
        /* Il "no" senza motivo era il danno: un MoE con piu' tensori del cap NON e' un file
         * corrotto, e chi chiama riceveva solo NULL — impossibile dire all'utente cosa cambiare. */
        char why[160];
        fprintf(stderr, "DSV4: %s non aperto: %s\n", path, llm_gguf_errmsg(&e->gguf, why, sizeof why));
        free(e); return NULL;
    }
    if (!llm_dsv4_load_impl(&e->w, &e->gguf, !packed)) {
        llm_gguf_close_file(&e->gguf, e->map, e->maplen); free(e); return NULL;
    }
    e->n_ctx = n_ctx > 0 ? n_ctx : 4096;
    int raw_cap = e->w.cfg.n_swa > 0 ? e->w.cfg.n_swa : 128;
    llm_dsv4_caches_init(&e->caches, &e->w.cfg, raw_cap, e->n_ctx);
    /* GPU (F4): attiva il contesto Metal (esperti + attn) — di default. */
    if (packed && e->w.exps) llm_dsv4_gpu_ctx_init(&e->w.cfg, e->w.exps);
    return e;
}
static inline void llm_dsv4_engine_free(LlmDsv4Engine* e) {
    if (!e) return;
    llm_dsv4_gpu_ctx_close();
    free(e->logits);
    llm_dsv4_caches_free(&e->caches);
    llm_dsv4_store_free(e->store);
    llm_dsv4_weights_free(&e->w);
    llm_gguf_close_file(&e->gguf, e->map, e->maplen);
    free(e);
}
/* Load + abilita lo SWAP esperti (cache di cap_bundles bundle). */
static inline LlmDsv4Engine* llm_dsv4_engine_load_swap(const char* path, int n_ctx, int cap_bundles) {
    LlmDsv4Engine* e = llm_dsv4_engine_load_ex(path, n_ctx, true);
    if (e && !llm_dsv4_engine_enable_swap(e, cap_bundles)) { llm_dsv4_engine_free(e); return NULL; }
    return e;
}
static inline void llm_dsv4_engine_reset(LlmDsv4Engine* e) {
    LlmDsv4Config c = e->w.cfg;
    llm_dsv4_caches_free(&e->caches);
    int raw_cap = c.n_swa > 0 ? c.n_swa : 128;
    llm_dsv4_caches_init(&e->caches, &c, raw_cap, e->n_ctx);
}
static inline void llm_dsv4_engine_generate(LlmDsv4Engine* e, int* tokens, int n_prompt, int n_new) {
    llm_dsv4_generate(&e->w.cfg, &e->w, &e->caches, tokens, n_prompt, n_new, NULL);
}

/* ── API incrementale (per l'host LLMEnv: append di token, poi generate) ── */
/* Prefilla `n` token (avanza pos) e aggiorna `pending` dall'ultimo. */
static inline void llm_dsv4_engine_append(LlmDsv4Engine* e, const int* ids, int n) {
    int HC = e->w.cfg.n_hc, D = e->w.cfg.n_embd;
    float* hc = (float*)malloc(sizeof(float) * (size_t)HC * D);
    for (int i = 0; i < n; ) {   /* chunk layer-major (LLM_DSV4_PREFILL_CHUNK token) */
        int chunk = LLM_DSV4_PREFILL_CHUNK; { int v; if (llm_ai_mlx_opt_get_int("dsv4_prefill_chunk", &v) == 0 && v >= 1 && v <= 4096) chunk = v; }
        int m = n - i; if (m > chunk) m = chunk;
        llm_dsv4_forward_batch(&e->w.cfg, &e->w, &e->caches, ids + i, e->pos, m, hc);
        e->pos += m; i += m;
    }
    if (!e->logits) e->logits = (float*)malloc(sizeof(float) * (size_t)e->w.cfg.n_vocab);
    llm_dsv4_output_logits(&e->w.cfg, &e->w, hc, e->logits);
    e->pending = llm_dsv4_argmax(e->logits, e->w.cfg.n_vocab);
    e->has_pending = 1;
    free(hc);
}
/* Genera greedy fino a max_new (o eos). Ritorna il numero generato. */
static inline int llm_dsv4_engine_generate_step(LlmDsv4Engine* e, int* out, int max_new, int eos) {
    int HC = e->w.cfg.n_hc, D = e->w.cfg.n_embd;
    if (!e->has_pending) return 0;
    float* hc = (float*)malloc(sizeof(float) * (size_t)HC * D);
    int n = 0;
    while (n < max_new) {
        int t = e->pending;
        if (eos >= 0 && t == eos) break;
        out[n++] = t;
        llm_dsv4_forward_step(&e->w.cfg, &e->w, &e->caches, t, e->pos++, hc);
        llm_dsv4_output_logits(&e->w.cfg, &e->w, hc, e->logits);
        e->pending = llm_dsv4_argmax(e->logits, e->w.cfg.n_vocab);
    }
    free(hc);
    return n;
}
/* Default per il modello reale: esperti packed. */
static inline LlmDsv4Engine* llm_dsv4_engine_load(const char* path, int n_ctx) {
    return llm_dsv4_engine_load_ex(path, n_ctx, true);
}

#endif /* LLM_DSV4_ENGINE_H */
