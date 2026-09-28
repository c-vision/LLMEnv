/* llm_dsv4_store.h — store degli esperti DS4 con due modalità:
 *  - RESIDENT: puntatori nel mmap del GGUF (tutti gli esperti "virtualmente residenti");
 *  - SWAP: record-esperto letti on-demand (pread, `llmengine_moe_storage`) in una cache LRU a bundle
 *    (gate+up+down per esperto). Metriche: bytes_read / hit / miss / eviction.
 * I/O POSIX self-contained (open/pread; specchia llmengine_moe_storage del runtime).
 * Header-only. Test: tools/llm_dsv4_store_test.sh. */
#ifndef LLM_DSV4_STORE_H
#define LLM_DSV4_STORE_H

#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "llm_dsv4.h"
#include "llm_dsv4_packed.h"
#include "llm_gguf.h"

typedef enum { LLM_DSV4_STORE_RESIDENT = 0, LLM_DSV4_STORE_SWAP = 1 } LlmDsv4StoreMode;

typedef struct {
    uint64_t off_g, off_u, off_d;   /* offset assoluti nel file */
    size_t nb_g, nb_u, nb_d;        /* byte per record-esperto */
    int t_g, t_u, t_d;
    size_t stride_g, stride_u, stride_d; /* byte per esperto (per risalire dal base) */
} LlmDsv4StoreLayer;

typedef struct { int layer, e, valid; uint64_t age; uint8_t* buf; size_t len, cap; } LlmDsv4StoreEnt;

typedef struct {
    LlmDsv4StoreMode mode;
    const uint8_t* resident_base;   /* g->buf (RESIDENT) */
    int fd;                         /* SWAP (pread) */
    int n_layer;
    LlmDsv4StoreLayer* layers;
    LlmDsv4StoreEnt* ents;
    int cap, n, tick;
    uint64_t bytes_read, hits, misses, evictions;
} LlmDsv4Store;

static inline LlmDsv4Store* llm_dsv4_store_open(const char* path, const LlmEngineMoeGguf* g,
        const LlmDsv4Config* c, LlmDsv4StoreMode mode, int cap_bundles) {
    LlmDsv4Store* s = (LlmDsv4Store*)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->mode = mode;
    s->resident_base = g->buf;
    s->n_layer = c->n_layer;
    s->cap = cap_bundles > 0 ? cap_bundles : 64;
    s->layers = (LlmDsv4StoreLayer*)calloc((size_t)c->n_layer, sizeof(LlmDsv4StoreLayer));
    s->ents = (LlmDsv4StoreEnt*)calloc((size_t)s->cap, sizeof(LlmDsv4StoreEnt));
    if (!s->layers || !s->ents) { free(s->layers); free(s->ents); free(s); return NULL; }
    s->fd = -1;
    if (mode == LLM_DSV4_STORE_SWAP) {
        s->fd = open(path, O_RDONLY);
        if (s->fd < 0) { free(s->layers); free(s->ents); free(s); return NULL; }
    }
    for (int il = 0; il < c->n_layer; il++) {
        char b[160];
        snprintf(b, sizeof b, "blk.%d.ffn_gate_exps.weight", il);
        int ig = llm_gguf_find_tensor(g, b);
        snprintf(b, sizeof b, "blk.%d.ffn_up_exps.weight", il);
        int iu = llm_gguf_find_tensor(g, b);
        snprintf(b, sizeof b, "blk.%d.ffn_down_exps.weight", il);
        int id = llm_gguf_find_tensor(g, b);
        if (ig < 0 || iu < 0 || id < 0) { if (s->fd >= 0) close(s->fd); free(s->layers); free(s->ents); free(s); return NULL; }
        LlmDsv4StoreLayer* L = &s->layers[il];
        const uint8_t* dg = llm_gguf_tensor_data(g, ig, NULL);
        L->off_g = (uint64_t)(dg - g->buf); L->t_g = g->ti[ig].type;
        L->nb_g = (size_t)(c->n_ff_exp * llm_ggml_nbytes((uint32_t)L->t_g, c->n_embd));
        L->stride_g = L->nb_g;
        const uint8_t* du = llm_gguf_tensor_data(g, iu, NULL);
        L->off_u = (uint64_t)(du - g->buf); L->t_u = g->ti[iu].type;
        L->nb_u = (size_t)(c->n_ff_exp * llm_ggml_nbytes((uint32_t)L->t_u, c->n_embd));
        L->stride_u = L->nb_u;
        const uint8_t* dd = llm_gguf_tensor_data(g, id, NULL);
        L->off_d = (uint64_t)(dd - g->buf); L->t_d = g->ti[id].type;
        L->nb_d = (size_t)(c->n_embd * llm_ggml_nbytes((uint32_t)L->t_d, c->n_ff_exp));
        L->stride_d = L->nb_d;
    }
    return s;
}
static inline void llm_dsv4_store_free(LlmDsv4Store* s) {
    if (!s) return;
    if (s->ents) { for (int i = 0; i < s->cap; i++) free(s->ents[i].buf); free(s->ents); }
    if (s->fd >= 0) close(s->fd);
    free(s->layers); free(s);
}
static inline int llm_dsv4_store_find(LlmDsv4Store* s, int layer, int e) {
    for (int i = 0; i < s->n; i++) if (s->ents[i].valid && s->ents[i].layer == layer && s->ents[i].e == e) return i;
    return -1;
}
/* Ritorna i puntatori ai tre record-esperto (validi fino alla prossima chiamata). */
static inline const uint8_t* llm_dsv4_store_expert(LlmDsv4Store* s, int layer, int e,
        const uint8_t** gate, const uint8_t** up, const uint8_t** down) {
    LlmDsv4StoreLayer* L = &s->layers[layer];
    /* fallback sempre valido: puntatori nel mmap (il file e' gia' mappato dall'engine) */
    const uint8_t* r_g = s->resident_base + L->off_g + (uint64_t)e * L->stride_g;
    const uint8_t* r_u = s->resident_base + L->off_u + (uint64_t)e * L->stride_u;
    const uint8_t* r_d = s->resident_base + L->off_d + (uint64_t)e * L->stride_d;
    *gate = r_g; *up = r_u; *down = r_d;
    if (s->mode == LLM_DSV4_STORE_RESIDENT) return r_g;
    int idx = llm_dsv4_store_find(s, layer, e);
    if (idx >= 0) { s->hits++; s->ents[idx].age = ++s->tick; }
    else {
        s->misses++;
        idx = -1;
        for (int i = 0; i < s->cap; i++) if (!s->ents[i].valid) { idx = i; break; }
        if (idx < 0) {  /* evict LRU */
            uint64_t oldest = ~0ull; for (int i = 0; i < s->cap; i++) if (s->ents[i].age < oldest) { oldest = s->ents[i].age; idx = i; }
            s->evictions++;
        }
        LlmDsv4StoreEnt* en = &s->ents[idx];
        size_t len = L->nb_g + L->nb_u + L->nb_d;
        if (en->cap < len) { free(en->buf); en->buf = NULL; en->cap = 0; }
        if (!en->buf) en->buf = (uint8_t*)malloc(len);
        if (!en->buf) { *gate = r_g; *up = r_u; *down = r_d; return r_g; }
        en->cap = len; en->len = len;
        if (pread(s->fd, en->buf, L->nb_g, (off_t)(L->off_g + (uint64_t)e * L->stride_g)) != (ssize_t)L->nb_g ||
            pread(s->fd, en->buf + L->nb_g, L->nb_u, (off_t)(L->off_u + (uint64_t)e * L->stride_u)) != (ssize_t)L->nb_u ||
            pread(s->fd, en->buf + L->nb_g + L->nb_u, L->nb_d, (off_t)(L->off_d + (uint64_t)e * L->stride_d)) != (ssize_t)L->nb_d) {
            en->valid = 0; fprintf(stderr, "llm_dsv4_store: read fallita layer=%d e=%d (fallback mmap)\n", layer, e);
            return r_g;   /* output gia' settati ai puntatori resident */
        }
        s->bytes_read += len;
        en->valid = 1; en->layer = layer; en->e = e; en->age = ++s->tick;
        if (idx >= s->n) s->n = idx + 1;
    }
    LlmDsv4StoreEnt* en = &s->ents[idx];
    *gate = en->buf; *up = en->buf + L->nb_g; *down = en->buf + L->nb_g + L->nb_u;
    return en->buf;
}

#endif /* LLM_DSV4_STORE_H */
