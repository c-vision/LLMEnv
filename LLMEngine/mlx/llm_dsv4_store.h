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
#include <dispatch/dispatch.h>
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
    int* ht_keys; int* ht_vals; int ht_cap;   /* hash (layer,e) -> idx entry, open addressing */
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
    int hcap = 16; while (hcap < s->cap * 2) hcap <<= 1;
    s->ht_cap = hcap;
    s->ht_keys = (int*)malloc(sizeof(int) * (size_t)hcap);
    s->ht_vals = (int*)malloc(sizeof(int) * (size_t)hcap);
    if (!s->ht_keys || !s->ht_vals) { free(s->ht_keys); free(s->ht_vals); free(s->layers); free(s->ents); free(s); return NULL; }
    for (int i = 0; i < hcap; i++) s->ht_keys[i] = -1;
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
    free(s->ht_keys); free(s->ht_vals);
    if (s->fd >= 0) close(s->fd);
    free(s->layers); free(s);
}
static inline int llm_dsv4_store_find(LlmDsv4Store* s, int layer, int e) {
    int key = layer * 512 + e, mask = s->ht_cap - 1;
    uint32_t h = (uint32_t)((uint32_t)key * 2654435761u) & (uint32_t)mask;
    for (int p = 0; p < s->ht_cap; p++) {
        int i = ((int)h + p) & mask, k = s->ht_keys[i];
        if (k == -1) return -1;
        if (k == key) return s->ht_vals[i];
    }
    return -1;
}
static inline void llm_dsv4_store_ht_put(LlmDsv4Store* s, int layer, int e, int val) {
    int key = layer * 512 + e, mask = s->ht_cap - 1;
    uint32_t h = (uint32_t)((uint32_t)key * 2654435761u) & (uint32_t)mask;
    int tomb = -1;
    for (int p = 0; p < s->ht_cap; p++) {
        int i = ((int)h + p) & mask, k = s->ht_keys[i];
        if (k == key) { s->ht_vals[i] = val; return; }
        if (k == -1) { if (tomb >= 0) { s->ht_keys[tomb] = key; s->ht_vals[tomb] = val; } else { s->ht_keys[i] = key; s->ht_vals[i] = val; } return; }
        if (k == -2 && tomb < 0) tomb = i;
    }
}
static inline void llm_dsv4_store_ht_del(LlmDsv4Store* s, int layer, int e) {
    int key = layer * 512 + e, mask = s->ht_cap - 1;
    uint32_t h = (uint32_t)((uint32_t)key * 2654435761u) & (uint32_t)mask;
    for (int p = 0; p < s->ht_cap; p++) {
        int i = ((int)h + p) & mask, k = s->ht_keys[i];
        if (k == -1) return;
        if (k == key) { s->ht_keys[i] = -2; return; }
    }
}
/* Assegna lo slot della cache a (layer, e): hit -> idx e *miss=0; miss -> sceglie/ sfratta uno slot, prepara il buffer
 * (allineato a pagina + padding, per le view Metal no-copy) e ritorna idx con *miss=1 (I RECORD NON SONO ANCORA LETTI:
 * li legge chi chiama, vedi llm_dsv4_store_read_ent). Ritorna -1 se non c'e' memoria (fallback mmap). */
static inline int llm_dsv4_store_claim(LlmDsv4Store* s, int layer, int e, int* miss) {
    LlmDsv4StoreLayer* L = &s->layers[layer];
    *miss = 0;
    int idx = llm_dsv4_store_find(s, layer, e);
    if (idx >= 0) { s->hits++; s->ents[idx].age = ++s->tick; return idx; }
    s->misses++;
    for (int i = 0; i < s->cap; i++) if (!s->ents[i].valid) { idx = i; break; }
    if (idx < 0) {  /* evict LRU */
        uint64_t oldest = ~0ull; for (int i = 0; i < s->cap; i++) if (s->ents[i].age < oldest) { oldest = s->ents[i].age; idx = i; }
        s->evictions++;
        llm_dsv4_store_ht_del(s, s->ents[idx].layer, s->ents[idx].e);
    }
    LlmDsv4StoreEnt* en = &s->ents[idx];
    size_t len = L->nb_g + L->nb_u + L->nb_d;
    /* Buffer ALLINEATO a pagina + paddato: le view Metal zero-copy (newBufferWithBytesNoCopy) mappano pagine intere
     * [align_down, align_up) -> il buffer deve coprirle, altrimenti la view esce dall'allocazione. */
    const size_t page = 16384;
    size_t padded = ((len + page - 1) / page) * page + page;
    if (en->cap < padded) { free(en->buf); en->buf = NULL; en->cap = 0; }
    if (!en->buf) {
        void* p = NULL;
        if (posix_memalign(&p, page, padded) == 0) { en->buf = (uint8_t*)p; en->cap = padded; }
    }
    if (!en->buf) { en->valid = 0; return -1; }
    en->len = len;
    en->valid = 1; en->layer = layer; en->e = e; en->age = ++s->tick;
    llm_dsv4_store_ht_put(s, layer, e, idx);
    if (idx >= s->n) s->n = idx + 1;
    *miss = 1;
    return idx;
}
/* Legge i 3 record (gate/up/down) dell'esperto nello slot. 0 = ok. */
static inline int llm_dsv4_store_read_ent(LlmDsv4Store* s, int layer, int e, int idx, int part) {
    LlmDsv4StoreLayer* L = &s->layers[layer]; LlmDsv4StoreEnt* en = &s->ents[idx];
    if (part == 0) return pread(s->fd, en->buf, L->nb_g, (off_t)(L->off_g + (uint64_t)e * L->stride_g)) == (ssize_t)L->nb_g ? 0 : -1;
    if (part == 1) return pread(s->fd, en->buf + L->nb_g, L->nb_u, (off_t)(L->off_u + (uint64_t)e * L->stride_u)) == (ssize_t)L->nb_u ? 0 : -1;
    return pread(s->fd, en->buf + L->nb_g + L->nb_u, L->nb_d, (off_t)(L->off_d + (uint64_t)e * L->stride_d)) == (ssize_t)L->nb_d ? 0 : -1;
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
    int miss = 0, idx = llm_dsv4_store_claim(s, layer, e, &miss);
    if (idx < 0) return r_g;
    LlmDsv4StoreEnt* en = &s->ents[idx];
    if (miss) {
        if (llm_dsv4_store_read_ent(s, layer, e, idx, 0) || llm_dsv4_store_read_ent(s, layer, e, idx, 1) ||
            llm_dsv4_store_read_ent(s, layer, e, idx, 2)) {
            en->valid = 0; llm_dsv4_store_ht_del(s, layer, e);
            fprintf(stderr, "llm_dsv4_store: read fallita layer=%d e=%d (fallback mmap)\n", layer, e);
            return r_g;   /* output gia' settati ai puntatori resident */
        }
        s->bytes_read += en->len;
    }
    *gate = en->buf; *up = en->buf + L->nb_g; *down = en->buf + L->nb_g + L->nb_u;
    return en->buf;
}

/* PREFETCH parallelo di un insieme di esperti di un layer (es. i 6 esperti di ogni token di un chunk): gli slot sono
 * assegnati in serie (logica LRU), le letture dei record mancanti (3 pread per esperto) girano in parallelo su GCD
 * (l'NVMe/page cache scalano con piu' richieste). Dopo, llm_dsv4_store_expert() e' tutto hit. `n` fino a 4096. */
static inline void llm_dsv4_store_prefetch(LlmDsv4Store* s, int layer, const int* experts, int n) {
    if (!s || s->mode != LLM_DSV4_STORE_SWAP || n <= 0) return;
    typedef struct { int e, idx; } Job;
    Job* jobs = (Job*)malloc(sizeof(Job) * (size_t)n);   /* heap: letto dai worker GCD (un __thread non sarebbe condiviso) */
    if (!jobs) return;
    int nj = 0;
    for (int i = 0; i < n; i++) {
        int miss = 0, idx = llm_dsv4_store_claim(s, layer, experts[i], &miss);
        if (idx >= 0 && miss) { jobs[nj].e = experts[i]; jobs[nj].idx = idx; nj++; }
    }
    if (nj == 0) { free(jobs); return; }
    __block int bad = 0;
    dispatch_apply((size_t)nj * 3, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
        if (llm_dsv4_store_read_ent(s, layer, jobs[k / 3].e, jobs[k / 3].idx, (int)(k % 3))) bad = 1;
    });
    LlmDsv4StoreLayer* L = &s->layers[layer];
    for (int j = 0; j < nj; j++) {
        LlmDsv4StoreEnt* en = &s->ents[jobs[j].idx];
        if (bad) { en->valid = 0; llm_dsv4_store_ht_del(s, layer, jobs[j].e); }   /* rilegge (serie) al primo uso */
        else s->bytes_read += L->nb_g + L->nb_u + L->nb_d;
    }
    free(jobs);
}

#endif /* LLM_DSV4_STORE_H */
