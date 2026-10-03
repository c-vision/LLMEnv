/* llmengine_moe_gguf_store.h — store degli esperti di un modello MoE GGUF, in stile DS4
 * (native/llmengine_mlx/llm_dsv4_store.h generalizzato al runtime condiviso):
 *
 *   - RESIDENT: i puntatori degli esperti cadono nel mmap del file (zero-copy, tutti
 *     "virtualmente residenti" — il SO pagina on-demand);
 *   - SWAP:     bundle-esperto (gate+up+down) letti on-demand con `pread` in una cache
 *     LRU a `cap` bundle. Metriche bytes_read/hits/misses/evictions.
 *
 * Header-only (nessun costo di build nuovo), pensato per l'adapter GGUF del runtime MoE.
 * Config: mode dal config di `LLM MODEL LOAD` (SWAP default / RESIDENT), capienza dal
 * `cache_mb`. Il container si assume mmap-abile (GGUF). */
#ifndef LLMENGINE_MOE_GGUF_STORE_H
#define LLMENGINE_MOE_GGUF_STORE_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "llmengine_moe_gguf.h"
#include "llmengine_moe_ggml.h"
#include "llmengine_moe_prof.h"   /* K1/K2: eventi trace HIT/MISS/EVICT dal cache reale */

typedef struct {
    uint64_t off;        /* offset assoluto nel file del primo esperto */
    size_t   nb;         /* byte per record-esperto */
    uint32_t type;       /* ggml type */
    uint64_t n_experts;
} GgufStoreTensor;

typedef struct {
    GgufStoreTensor g, u, d;   /* gate/up/down; n_experts coincide */
} GgufStoreLayer;

typedef struct { int layer, e, valid; uint64_t age; uint64_t born_step; uint8_t* buf; size_t len, cap; } GgufStoreEnt;

typedef struct {
    int swap;                    /* 0 RESIDENT (mmap), 1 SWAP (pread+LRU) */
    const uint8_t* base;         /* mmap */
    size_t map_len;
    int fd;
    uint32_t n_layer;
    GgufStoreLayer* layers;
    GgufStoreEnt* ents;          /* cache bundle (SWAP) */
    int cap, n;
    uint64_t tick;
    uint64_t bytes_read, hits, misses, evictions;
    /* K2: hit su un bundle residente da uno step PRECEDENTE (non caricato in questo step);
     * lo step avanza col route-callback per layer (stesso confine dei ROUTER nel trace). */
    uint64_t resident_hits, step;
    /* K5: richieste di esperti non risolvibili (indice fuori dal tensor / read fallita):
     * l'output non e' affidabile → lo store ritorna NULL e il chiamante propaga lo stato. */
    uint64_t unresolvable;
    /* K1: profiler opzionale (NULL = nessun trace): emette CACHE_HIT/MISS/EVICT dal cache reale. */
    LlmEngineMoeProfiler* prof;
} GgufStore;

/* Riempe un GgufStoreTensor dal tensore `name` (3D stacked). 0 = ok. */
static inline int gguf_store_tensor(const LlmEngineMoeGguf* g, const char* name, GgufStoreTensor* out) {
    const LlmCodecGgufTensor* t = llmengine_moe_gguf_tensor(g, name);
    if (t == NULL || t->n_dims < 3) return 0;
    uint64_t per = t->dims[0] * t->dims[1];
    size_t nb = llmengine_moe_ggml_row_bytes(t->type, (size_t)per);
    if (nb == 0) return 0;
    out->off = g->data_offset + t->offset;
    out->nb = nb;
    out->type = t->type;
    out->n_experts = t->dims[2];
    return 1;
}

/* Apre lo store. `swap`: 0 RESIDENT (mmap pointer), !=0 SWAP (pread+LRU). `cache_mb` = budget
 * della cache bundle in SWAP (0 = default 256 MB). NULL su errore. */
static inline GgufStore* llmengine_moe_gguf_store_open(const char* path, const LlmEngineMoeGguf* g,
                                             uint32_t n_layers, int swap, uint64_t cache_mb) {
    if (path == NULL || g == NULL || n_layers == 0) return NULL;
    GgufStore* s = (GgufStore*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->n_layer = n_layers;
    s->swap = swap ? 1 : 0;
    s->fd = -1;
    s->layers = (GgufStoreLayer*)calloc(n_layers, sizeof(GgufStoreLayer));
    if (s->layers == NULL) { free(s); return NULL; }

    /* mmap sempre (RESIDENT: zero-copy; SWAP: fallback pointer). */
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0) { free(s->layers); free(s); return NULL; }
    struct stat st;
    if (fstat(s->fd, &st) != 0 || st.st_size <= 0) { close(s->fd); free(s->layers); free(s); return NULL; }
    s->map_len = (size_t)st.st_size;
    void* m = mmap(NULL, s->map_len, PROT_READ, MAP_PRIVATE, s->fd, 0);
    if (m == MAP_FAILED) { close(s->fd); free(s->layers); free(s); return NULL; }
    s->base = (const uint8_t*)m;

    for (uint32_t l = 0; l < n_layers; ++l) {
        char b[160];
        snprintf(b, sizeof(b), "blk.%u.ffn_gate_exps.weight", l);
        int ok_g = gguf_store_tensor(g, b, &s->layers[l].g);
        snprintf(b, sizeof(b), "blk.%u.ffn_up_exps.weight", l);
        int ok_u = gguf_store_tensor(g, b, &s->layers[l].u);
        snprintf(b, sizeof(b), "blk.%u.ffn_down_exps.weight", l);
        int ok_d = gguf_store_tensor(g, b, &s->layers[l].d);
        if (!ok_g || !ok_u || !ok_d) {
            munmap((void*)s->base, s->map_len); close(s->fd); free(s->layers); free(s); return NULL;
        }
    }
    if (s->swap) {
        if (cache_mb == 0) cache_mb = 256;
        /* cap = budget/size-bundle (min 8): il bundle varia per layer, si usa il primo come stima. */
        size_t bundle = s->layers[0].g.nb + s->layers[0].u.nb + s->layers[0].d.nb;
        if (bundle == 0) bundle = 1;
        uint64_t cap = (cache_mb * 1024ull * 1024ull) / bundle;
        if (cap < 8) cap = 8;
        if (cap > 65536) cap = 65536;
        s->cap = (int)cap;
        s->ents = (GgufStoreEnt*)calloc((size_t)s->cap, sizeof(GgufStoreEnt));
        if (s->ents == NULL) { munmap((void*)s->base, s->map_len); close(s->fd); free(s->layers); free(s); return NULL; }
    }
    return s;
}

static inline void llmengine_moe_gguf_store_close(GgufStore* s) {
    if (s == NULL) return;
    if (s->ents) { for (int i = 0; i < s->cap; i++) free(s->ents[i].buf); free(s->ents); }
    if (s->base) munmap((void*)s->base, s->map_len);
    if (s->fd >= 0) close(s->fd);
    free(s->layers);
    free(s);
}

static inline int gguf_store_find(GgufStore* s, int layer, int e) {
    for (int i = 0; i < s->n; i++)
        if (s->ents[i].valid && s->ents[i].layer == layer && s->ents[i].e == e) return i;
    return -1;
}

/* Ritorna i puntatori ai tre record-esperto (raw ggml bytes), validi fino alla prossima chiamata.
 * RESIDENT: puntatori nel mmap. SWAP: bundle in cache LRU (pread su miss). NULL su errore. */
static inline const uint8_t* llmengine_moe_gguf_store_expert(GgufStore* s, uint32_t layer, uint32_t e,
                                                   const uint8_t** up, const uint8_t** down) {
    if (s == NULL || layer >= s->n_layer) return NULL;
    GgufStoreLayer* L = &s->layers[layer];
    /* K5: bounds-check PRIMA di calcolare i puntatori (un indice fuori dal tensor darebbe
     * letture fuori dal file, silenziose). Mai output silenzioso. */
    if (e >= L->g.n_experts || e >= L->u.n_experts || e >= L->d.n_experts) {
        s->unresolvable++;
        if (up) *up = NULL;
        if (down) *down = NULL;
        return NULL;
    }
    const uint8_t* rg = s->base + L->g.off + (uint64_t)e * L->g.nb;
    const uint8_t* ru = s->base + L->u.off + (uint64_t)e * L->u.nb;
    const uint8_t* rd = s->base + L->d.off + (uint64_t)e * L->d.nb;
    if (!s->swap) { *up = ru; *down = rd; return rg; }
    int idx = gguf_store_find(s, (int)layer, (int)e);
    if (idx >= 0) {
        s->hits++;
        if (s->ents[idx].born_step < s->step) s->resident_hits++;   /* K2 */
        s->ents[idx].age = ++s->tick;
        if (s->prof != NULL)
            llmengine_moe_prof_event(s->prof, LLMENGINE_MOE_TRACE_CACHE_HIT, layer, e, 0, (uint64_t)s->ents[idx].len);
    } else {
        s->misses++;
        idx = -1;
        for (int i = 0; i < s->cap; i++) if (!s->ents[i].valid) { idx = i; break; }
        if (idx < 0) {
            uint64_t oldest = ~0ull;
            for (int i = 0; i < s->cap; i++)
                if (s->ents[i].age < oldest) { oldest = s->ents[i].age; idx = i; }
            s->evictions++;
            if (s->prof != NULL)   /* K1: EVICT reale (chiave evitta) prima di sovrascrivere */
                llmengine_moe_prof_event(s->prof, LLMENGINE_MOE_TRACE_EVICT,
                                   (uint32_t)s->ents[idx].layer, (uint32_t)s->ents[idx].e, 0, 1);
        }
        GgufStoreEnt* en = &s->ents[idx];
        size_t len = L->g.nb + L->u.nb + L->d.nb;
        if (en->cap < len) { free(en->buf); en->buf = NULL; en->cap = 0; }
        if (en->buf == NULL) en->buf = (uint8_t*)malloc(len);
        if (en->buf == NULL) { *up = ru; *down = rd; return rg; }   /* fallback mmap */
        en->cap = len; en->len = len;
        if (pread(s->fd, en->buf, L->g.nb, (off_t)(L->g.off + (uint64_t)e * L->g.nb)) != (ssize_t)L->g.nb ||
            pread(s->fd, en->buf + L->g.nb, L->u.nb, (off_t)(L->u.off + (uint64_t)e * L->u.nb)) != (ssize_t)L->u.nb ||
            pread(s->fd, en->buf + L->g.nb + L->u.nb, L->d.nb, (off_t)(L->d.off + (uint64_t)e * L->d.nb)) != (ssize_t)L->d.nb) {
            en->valid = 0; *up = ru; *down = rd; return rg;
        }
        s->bytes_read += len;
        en->valid = 1; en->layer = (int)layer; en->e = (int)e; en->age = ++s->tick;
        en->born_step = s->step;   /* K2: residente da questo step in poi */
        if ((int)idx >= s->n) s->n = idx + 1;
        if (s->prof != NULL)
            llmengine_moe_prof_event(s->prof, LLMENGINE_MOE_TRACE_CACHE_MISS, layer, e, 0, (uint64_t)len);
    }
    GgufStoreEnt* en = &s->ents[idx];
    *up = en->buf + L->g.nb;
    *down = en->buf + L->g.nb + L->u.nb;
    return en->buf;
}

/* Byte di bundle risiedenti (SWAP) per AI.STREAMSTATS. */
static inline uint64_t llmengine_moe_gguf_store_resident_bytes(const GgufStore* s) {
    if (s == NULL || !s->swap) return 0;
    uint64_t b = 0;
    for (int i = 0; i < s->cap; i++) if (s->ents[i].valid) b += s->ents[i].len;
    return b;
}

#endif /* LLMENGINE_MOE_GGUF_STORE_H */
