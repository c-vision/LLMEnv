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
#include <stdio.h>
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
    /* D19: hotness appresa (freq per (layer,expert) dai routing osservati). Default OFF → LRU. */
    uint32_t* freq;
    uint32_t f_layers, f_experts;
    int hotness;
} GgufStore;

/* ── D19: hotness appresa + policy di eviction per frequenza (opt-in) ─────────────────── */
static inline void llmengine_moe_gguf_store_set_dims(GgufStore* s, uint32_t layers, uint32_t experts) {
    if (s == NULL || layers == 0 || experts == 0) return;
    if (s->freq != NULL && s->f_layers == layers && s->f_experts == experts) return;
    uint32_t* nf = (uint32_t*)calloc((size_t)layers * experts, sizeof(uint32_t));
    if (nf == NULL) return;
    free(s->freq);
    s->freq = nf; s->f_layers = layers; s->f_experts = experts;
}

static inline void llmengine_moe_gguf_store_note_route(GgufStore* s, uint32_t layer,
                                                       const uint32_t* experts, uint32_t n) {
    if (s == NULL || s->freq == NULL || layer >= s->f_layers) return;
    for (uint32_t i = 0; i < n; ++i)
        if (experts != NULL && experts[i] < s->f_experts)
            s->freq[(size_t)layer * s->f_experts + experts[i]]++;
}

/* D19/M7: hotness PESATA per route_prob — accumula massa di probabilita' (fixed-point) per
 * (layer,expert); un esperto scelto con probabilita' alta pesa piu' di uno marginale. Con
 * `probs==NULL` ricade sul conteggio (come note_route). */
static inline void llmengine_moe_gguf_store_note_route_prob(GgufStore* s, uint32_t layer,
                                                            const uint32_t* experts, const float* probs, uint32_t n) {
    if (s == NULL || s->freq == NULL || layer >= s->f_layers) return;
    for (uint32_t i = 0; i < n; ++i)
        if (experts != NULL && experts[i] < s->f_experts) {
            uint32_t bump = (probs != NULL) ? (uint32_t)(probs[i] * 1024.0f) + 1u : 1u;
            s->freq[(size_t)layer * s->f_experts + experts[i]] += bump;
        }
}

static inline void llmengine_moe_gguf_store_set_hotness(GgufStore* s, int on) {
    if (s != NULL) s->hotness = on ? 1 : 0;
}

static inline uint32_t llmengine_moe_gguf_store_freq(const GgufStore* s, uint32_t layer, uint32_t e) {
    if (s == NULL || s->freq == NULL || layer >= s->f_layers || e >= s->f_experts) return 0;
    return s->freq[(size_t)layer * s->f_experts + e];
}

/* Default del preload warm-up (K esperti più caldi dal profilo). */
#define LLMENGINE_MOE_GGUF_PRELOAD_TOPK 64

/* Forward: definita sotto (il preload la usa). */
static inline const uint8_t* llmengine_moe_gguf_store_expert(GgufStore* s, uint32_t layer, uint32_t e,
                                                             const uint8_t** up, const uint8_t** down);

/* D19-follow-up B: preload dei K esperti più caldi del profilo (warm-up senza attendere il router).
 * Chiama `_expert` (che legge e inserisce in cache) per i top-K; ritorna quanti ne ha precaricati.
 * No-op se la tabella freq è assente (nessun profilo) o k<=0. Deterministico (tie-break layer,e). */
static inline int llmengine_moe_gguf_store_preload_topk(GgufStore* s, int k) {
    if (s == NULL || s->freq == NULL || k <= 0) return 0;
    if (k > 4096) k = 4096;
    uint32_t* sel_l = (uint32_t*)malloc((size_t)k * sizeof(uint32_t));
    uint32_t* sel_e = (uint32_t*)malloc((size_t)k * sizeof(uint32_t));
    if (sel_l == NULL || sel_e == NULL) { free(sel_l); free(sel_e); return 0; }
    int loaded = 0;
    for (int rank = 0; rank < k; ++rank) {
        uint32_t best_f = 0; int best_l = -1, best_e = -1;
        for (uint32_t l = 0; l < s->f_layers; ++l)
            for (uint32_t e = 0; e < s->f_experts; ++e) {
                uint32_t f = s->freq[(size_t)l * s->f_experts + e];
                if (f == 0) continue;
                int skip = 0;
                for (int j = 0; j < loaded; ++j)
                    if (sel_l[j] == l && sel_e[j] == e) { skip = 1; break; }
                if (skip) continue;
                if (best_l < 0 || f > best_f ||
                    (f == best_f && (l < (uint32_t)best_l ||
                                     (l == (uint32_t)best_l && e < (uint32_t)best_e)))) {
                    best_f = f; best_l = (int)l; best_e = (int)e;
                }
            }
        if (best_l < 0) break;   /* nessun candidato con freq > 0 */
        const uint8_t *u, *d;
        if (llmengine_moe_gguf_store_expert(s, (uint32_t)best_l, (uint32_t)best_e, &u, &d) == NULL) break;
        sel_l[loaded] = (uint32_t)best_l; sel_e[loaded] = (uint32_t)best_e; loaded++;
    }
    free(sel_l); free(sel_e);
    return loaded;
}

/* Profilo `GMSTATS 1` (stesso formato del profilo della cache: righe `layer expert freq`). */
static inline int llmengine_moe_gguf_store_profile_save(const GgufStore* s, const char* path) {
    if (s == NULL || s->freq == NULL || path == NULL || *path == 0) return -1;
    FILE* f = fopen(path, "wb");
    if (f == NULL) return -1;
    fprintf(f, "GMSTATS 1\n");
    for (uint32_t l = 0; l < s->f_layers; ++l)
        for (uint32_t e = 0; e < s->f_experts; ++e) {
            uint32_t v = s->freq[(size_t)l * s->f_experts + e];
            if (v != 0) fprintf(f, "%u %u %u\n", l, e, v);
        }
    fclose(f);
    return 0;
}

static inline int llmengine_moe_gguf_store_profile_load(GgufStore* s, const char* path) {
    if (s == NULL || s->freq == NULL || path == NULL || *path == 0) return -1;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    char magic[16] = {0}; int ver = 0;
    if (fscanf(f, "%15s", magic) != 1 || strcmp(magic, "GMSTATS") != 0) { fclose(f); return -1; }
    if (fscanf(f, "%d", &ver) != 1 || ver != 1) { fclose(f); return -1; }
    unsigned l = 0, e = 0, v = 0; int nread = 0;
    while (fscanf(f, "%u %u %u", &l, &e, &v) == 3) {
        if (l < s->f_layers && e < s->f_experts) s->freq[(size_t)l * s->f_experts + e] += v;
        nread++;
    }
    fclose(f);
    return nread > 0 ? 0 : -1;
}

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
    free(s->freq);
    free(s->layers);
    free(s);
}

static inline int gguf_store_find(GgufStore* s, int layer, int e) {
    for (int i = 0; i < s->n; i++)
        if (s->ents[i].valid && s->ents[i].layer == layer && s->ents[i].e == e) return i;
    return -1;
}

/* M6: PREFETCH REALE — carica l'esperto nello store (warm cache) per il layer successivo, cosi'
 * l'accesso al forward diventa un HIT. Usato dalla callback di prefetch (route_prob/future).
 * Ritorna 0 se il record esiste (ora in cache), -1 altrimenti. */
static inline int llmengine_moe_gguf_store_prefetch(GgufStore* s, uint32_t layer, uint32_t e) {
    const uint8_t* u; const uint8_t* d;
    return llmengine_moe_gguf_store_expert(s, layer, e, &u, &d) != NULL ? 0 : -1;
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
            /* D19: hotness ON → evince il MINOR freq osservato (tie-break: il più vecchio);
             * default (OFF o freq assente) → LRU puro. */
            int best = -1; uint32_t best_freq = 0; uint64_t best_age = ~0ull;
            for (int i = 0; i < s->cap; i++) {
                uint32_t f = 0;
                if (s->hotness && s->freq != NULL) {
                    int L = s->ents[i].layer, E = s->ents[i].e;
                    if (L >= 0 && (uint32_t)L < s->f_layers && (uint32_t)E < s->f_experts)
                        f = s->freq[(size_t)(uint32_t)L * s->f_experts + (uint32_t)E];
                }
                if (best < 0 || f < best_freq || (f == best_freq && s->ents[i].age < best_age)) {
                    best = i; best_freq = f; best_age = s->ents[i].age;
                }
            }
            idx = best;
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
