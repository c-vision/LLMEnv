/* llm_dsv4_store.h — store degli esperti DS4 con due modalità:
 *  - RESIDENT: puntatori nel mmap del GGUF (tutti gli esperti "virtualmente residenti");
 *  - SWAP: record-esperto letti on-demand (pread, `llmengine_moe_storage`) in una cache LRU a bundle
 *    (gate+up+down per esperto). Metriche: bytes_read / hit / miss / eviction.
 * I/O POSIX self-contained (open/pread; specchia llmengine_moe_storage del runtime).
 * Header-only. Test policy (D19-A): tools/llm_dsv4_store_policy_test.sh. */
#ifndef LLM_DSV4_STORE_H
#define LLM_DSV4_STORE_H

#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include "llm_dsv4.h"
#include "llm_dsv4_packed.h"
#include "llm_gguf.h"
#include "llm_engine_opts.h"   /* opzione dsv4_q4_arena (registro, no env) */

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
    int n_expert;                   /* esperti per layer (per il prefetch WILLNEED di layer interi) */
    LlmDsv4StoreLayer* layers;
    LlmDsv4StoreEnt* ents;
    int cap, n, tick;
    int* ht_keys; int* ht_vals; int ht_cap;   /* hash (layer,e) -> idx entry, open addressing */
    uint64_t bytes_read, hits, misses, evictions;
    /* D19-follow-up A: hotness appresa (freq per (layer,expert) dalle claim osservate).
     * Default OFF → LRU invariata. La feed avviene in `claim` (ogni accesso è una route osservata). */
    uint32_t* freq;
    int f_layers, f_experts;
    int hotness;
    int mmap_only;   /* opzione di registro `dsv4_experts_mmap`: esperti via puntatori nel mmap del GGUF
                      * (resident_base, UNA copia, come il vendored) invece della cache bundle anon (2ª copia).
                      * Nel prefill a chunk grande la cache LRU e' inutile (tutti gli esperti toccati) -> qui evita solo I/O+copia. */
    /* ARENA per i layer con esperti Q4_K (gate/up/down tutti tipo 12): i tensori di layer con gli esperti IMPILATI
     * (stessa disposizione del GGUF) vivono in un'unica regione anonima, riempita a richiesta con pread. Cosi' il kernel
     * mul_mm_id (src0 = tensore impilato, stride per esperto) li legge senza modifiche; niente LRU per questi layer. */
    uint8_t* q4_base; size_t q4_size; int q4_ne;
    size_t* q4_off;        /* [n_layer*3]: offset gate/up/down nell'arena (size_t(-1) = layer non in arena) */
    uint8_t* q4_present;   /* [n_layer*q4_ne] */
} LlmDsv4Store;

/* ── D19-follow-up A: hotness appresa (policy per frequenza, opt-in; default LRU) ── */
static inline void llm_dsv4_store_set_dims(LlmDsv4Store* s, int layers, int experts) {
    if (s == NULL || layers <= 0 || experts <= 0) return;
    if (s->freq != NULL && s->f_layers == layers && s->f_experts == experts) return;
    uint32_t* nf = (uint32_t*)calloc((size_t)layers * (size_t)experts, sizeof(uint32_t));
    if (nf == NULL) return;
    free(s->freq);
    s->freq = nf; s->f_layers = layers; s->f_experts = experts;
}

static inline void llm_dsv4_store_set_hotness(LlmDsv4Store* s, int on) {
    if (s != NULL) s->hotness = on ? 1 : 0;
}

static inline uint32_t llm_dsv4_store_freq(const LlmDsv4Store* s, int layer, int e) {
    if (s == NULL || s->freq == NULL || layer < 0 || e < 0 || layer >= s->f_layers || e >= s->f_experts) return 0;
    return s->freq[(size_t)layer * s->f_experts + e];
}

/* Sceglie lo slot da evincere: hotness ON → MINOR freq osservata (tie-break LRU); OFF → LRU pura.
 * Pura sul contenuto dello store → testabile offline (tests/llm_dsv4_store_policy_test.c). */
static inline int llm_dsv4_store_evict_choice(const LlmDsv4Store* s) {
    int best = -1; uint32_t bf = 0; uint64_t ba = ~0ull;
    for (int i = 0; i < s->cap; ++i) {
        uint32_t f = 0;
        if (s->hotness && s->freq != NULL) {
            int L = s->ents[i].layer, E = s->ents[i].e;
            if (L >= 0 && L < s->f_layers && E >= 0 && E < s->f_experts)
                f = s->freq[(size_t)L * s->f_experts + E];
        }
        if (best < 0 || f < bf || (f == bf && s->ents[i].age < ba)) { best = i; bf = f; ba = s->ents[i].age; }
    }
    return best;
}

/* Crea l'arena per i layer Q4_K (vedi sopra). Ritorna il numero di layer in arena (0 = nessuno / fallito). */
static inline int llm_dsv4_store_enable_q4(LlmDsv4Store* s, int n_expert) {
    if (!s || n_expert <= 0 || n_expert > 1024) return 0;
    const size_t page = 16384;
    size_t total = 0; int nl = 0;
    s->q4_off = (size_t*)malloc(sizeof(size_t) * 3 * (size_t)s->n_layer);
    if (!s->q4_off) return 0;
    for (int il = 0; il < s->n_layer; il++) {
        LlmDsv4StoreLayer* L = &s->layers[il];
        size_t* o = &s->q4_off[il * 3];
        o[0] = o[1] = o[2] = (size_t)-1;
        if (L->t_g == 12 && L->t_u == 12 && L->t_d == 12) {
            size_t sg = (size_t)n_expert * L->nb_g, su = (size_t)n_expert * L->nb_u, sd = (size_t)n_expert * L->nb_d;
            sg = (sg + page - 1) / page * page; su = (su + page - 1) / page * page; sd = (sd + page - 1) / page * page;
            o[0] = total; o[1] = total + sg; o[2] = total + sg + su; total += sg + su + sd; nl++;
        }
    }
    if (nl == 0) { free(s->q4_off); s->q4_off = NULL; return 0; }
    void* m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);   /* pagine allineate, lazy */
    s->q4_present = (uint8_t*)calloc((size_t)s->n_layer * n_expert, 1);
    if (m == MAP_FAILED || !s->q4_present) { if (m != MAP_FAILED) munmap(m, total); free(s->q4_present); s->q4_present = NULL; free(s->q4_off); s->q4_off = NULL; return 0; }
    s->q4_base = (uint8_t*)m; s->q4_size = total; s->q4_ne = n_expert;
    return nl;
}
/* 1 se il layer e' servito dall'arena; riempie i tre tensori impilati. */
static inline int llm_dsv4_store_q4_stacks(LlmDsv4Store* s, int layer, const uint8_t** g, const uint8_t** u, const uint8_t** d) {
    if (!s || !s->q4_off || s->q4_off[layer * 3] == (size_t)-1) return 0;
    if (g) *g = s->q4_base + s->q4_off[layer * 3]; if (u) *u = s->q4_base + s->q4_off[layer * 3 + 1]; if (d) *d = s->q4_base + s->q4_off[layer * 3 + 2];
    return 1;
}
/* Legge (se mancanti) gli esperti indicati nell'arena, in parallelo. Ritorna 0 = ok. */
static inline int llm_dsv4_store_q4_ensure(LlmDsv4Store* s, int layer, const int* experts, int n) {
    if (!llm_dsv4_store_q4_stacks(s, layer, NULL, NULL, NULL)) return -1;
    LlmDsv4StoreLayer* L = &s->layers[layer];
    uint8_t* pres = s->q4_present + (size_t)layer * s->q4_ne;
    int* todo = (int*)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1)); int nt = 0;
    if (!todo) return -2;
    for (int i = 0; i < n; i++) { int e = experts[i]; if (e < 0 || e >= s->q4_ne) { free(todo); return -3; }
        if (!pres[e]) { pres[e] = 2; todo[nt++] = e; s->misses++; } else if (pres[e] == 1) s->hits++; }   /* 2 = in lettura (dedup nella stessa chiamata) */
    if (nt == 0) { free(todo); return 0; }
    __block int bad = 0;
    uint8_t* const base = s->q4_base; const size_t* off = &s->q4_off[layer * 3]; const int fd = s->fd;
    const LlmDsv4StoreLayer LL = *L;
    int* td = todo;
    dispatch_apply((size_t)nt * 3, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
        int e = td[k / 3]; int part = (int)(k % 3);
        size_t nb = part == 0 ? LL.nb_g : part == 1 ? LL.nb_u : LL.nb_d;
        uint64_t fo = (part == 0 ? LL.off_g + (uint64_t)e * LL.stride_g : part == 1 ? LL.off_u + (uint64_t)e * LL.stride_u : LL.off_d + (uint64_t)e * LL.stride_d);
        if (pread(fd, base + off[part] + (size_t)e * nb, nb, (off_t)fo) != (ssize_t)nb) bad = 1;
    });
    for (int i = 0; i < nt; i++) { pres[todo[i]] = bad ? 0 : 1; if (!bad) s->bytes_read += LL.nb_g + LL.nb_u + LL.nb_d; }
    free(todo);
    return bad ? -4 : 0;
}

/* forward decl: hotlist I/O (definite in fondo a questo header) */
static inline int llm_dsv4_store_save_hot(const LlmDsv4Store* s, const char* path);
static inline int llm_dsv4_store_load_hot(LlmDsv4Store* s, const char* path);
static inline void llm_dsv4_store_seed_hot(LlmDsv4Store* s, int top_k);

static inline LlmDsv4Store* llm_dsv4_store_open(const char* path, const LlmEngineMoeGguf* g,
        const LlmDsv4Config* c, LlmDsv4StoreMode mode, int cap_bundles) {
    LlmDsv4Store* s = (LlmDsv4Store*)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->mode = mode;
    s->resident_base = g->buf;
    s->n_layer = c->n_layer;
    s->n_expert = c->n_expert;
    { int v = 0; if (llm_ai_mlx_opt_get_bool("dsv4_experts_mmap", &v) == 0) s->mmap_only = v; }   /* default OFF */
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
    /* Arena Q4_K: SPERIMENTALE, DISATTIVA di default (alloca ~21 GB e li rende residenti: sopra il budget di una macchina da 128 GB
     * con la cache esperti da 86 GB). Si attiva solo con l'opzione di registro `dsv4_q4_arena=1` (mai env). */
    { int v = 0; if (mode == LLM_DSV4_STORE_SWAP && !s->mmap_only && llm_ai_mlx_opt_get_bool("dsv4_q4_arena", &v) == 0 && v) llm_dsv4_store_enable_q4(s, c->n_expert); }
    /* item B: se il prefetch mirato e' attivo serve la mappa di hotness (alimentata in claim). */
    { int v = 0; if (mode == LLM_DSV4_STORE_SWAP && llm_ai_mlx_opt_get_int("dsv4_prefetch_hot", &v) == 0 && v > 0) llm_dsv4_store_set_dims(s, c->n_layer, c->n_expert); }
    /* HOTLIST I/O (item B/c): carica la hotness persistita e pre-carica i top-K esperti per layer. */
    if (mode == LLM_DSV4_STORE_SWAP) {
        char hin[1024], hout[1024];
        int has_in = llm_ai_mlx_opt_get_str("dsv4_hotlist_in", hin, sizeof hin) == 0 && hin[0];
        int has_out = llm_ai_mlx_opt_get_str("dsv4_hotlist_out", hout, sizeof hout) == 0 && hout[0];
        if (has_in || has_out) {
            llm_dsv4_store_set_dims(s, c->n_layer, c->n_expert);
            if (has_in && llm_dsv4_store_load_hot(s, hin) == 0) {
                int K = 64; { int v; if (llm_ai_mlx_opt_get_int("dsv4_hotlist_topk", &v) == 0 && v > 0) K = v; }
                llm_dsv4_store_seed_hot(s, K);
            }
        }
    }
    return s;
}
static inline void llm_dsv4_store_free(LlmDsv4Store* s) {
    if (!s) return;
    /* TEMP (gated ds4_profile): statistiche I/O deterministiche (validita' anche a macchina carica). */
    { int pf = 0; if (llm_ai_mlx_opt_get_bool("dsv4_profile", &pf) == 0 && pf)
        fprintf(stderr, "[dsv4-store] mode=%s cap=%d | bytes_read=%.2fGB hit=%llu miss=%llu evict=%llu\n",
                s->mode == LLM_DSV4_STORE_SWAP ? "swap" : "resident", s->cap,
                (double)s->bytes_read / 1073741824.0,
                (unsigned long long)s->hits, (unsigned long long)s->misses, (unsigned long long)s->evictions); }
    /* HOTLIST I/O: se richiesto, salva la hotness osservata (profilo) per il seed di run successivi. */
    { char p[1024]; if (s->freq && llm_ai_mlx_opt_get_str("dsv4_hotlist_out", p, sizeof p) == 0 && p[0]) llm_dsv4_store_save_hot(s, p); }
    if (s->ents) { for (int i = 0; i < s->cap; i++) free(s->ents[i].buf); free(s->ents); }
    free(s->ht_keys); free(s->ht_vals);
    if (s->fd >= 0) close(s->fd);
    if (s->q4_base) munmap(s->q4_base, s->q4_size);
    free(s->q4_off); free(s->q4_present);
    free(s->freq);
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
    if (s->freq != NULL && layer >= 0 && layer < s->f_layers && e >= 0 && e < s->f_experts)
        s->freq[(size_t)layer * s->f_experts + e]++;   /* D19-A: hotness osservata */
    int idx = llm_dsv4_store_find(s, layer, e);
    if (idx >= 0) { s->hits++; s->ents[idx].age = ++s->tick; return idx; }
    s->misses++;
    for (int i = 0; i < s->cap; i++) if (!s->ents[i].valid) { idx = i; break; }
    if (idx < 0) {  /* D19-A: hotness ON → min-freq (tie-break LRU); OFF → LRU pura */
        idx = llm_dsv4_store_evict_choice(s);
        s->evictions++;
        llm_dsv4_store_ht_del(s, s->ents[idx].layer, s->ents[idx].e);
    }
    LlmDsv4StoreEnt* en = &s->ents[idx];
    size_t len = L->nb_g + L->nb_u + L->nb_d;
    /* Buffer ALLINEATO a pagina + paddato: le view Metal zero-copy (newBufferWithBytesNoCopy) mappano pagine intere
     * [align_down, align_up) -> il buffer deve coprirle, altrimenti la view esce dall'allocazione. */
    const size_t page = 16384;
    /* Ogni slot ha SEMPRE la dimensione massima fra i layer: mai free()+riallocazione al cambio layer. Le view no-copy Metal sono
     * cached per indirizzo e legano le pagine FISICHE alla creazione: se l'indirizzo viene liberato e riassegnato ad altre pagine, la
     * GPU leggerebbe dati vecchi (bug: prefill a chunk errato appena la cache esperti inizia a evincere). */
    size_t maxlen = len;
    for (int il = 0; il < s->n_layer; il++) { size_t l2 = s->layers[il].nb_g + s->layers[il].nb_u + s->layers[il].nb_d; if (l2 > maxlen) maxlen = l2; }
    size_t padded = ((maxlen + page - 1) / page) * page + page;
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
    if (s->mode == LLM_DSV4_STORE_RESIDENT || s->mmap_only) return r_g;   /* UNA copia: puntatori nel mmap del GGUF */
    if (llm_dsv4_store_q4_stacks(s, layer, NULL, NULL, NULL)) {   /* layer Q4_K: puntatori nell'arena */
        if (llm_dsv4_store_q4_ensure(s, layer, &e, 1) == 0) {
            *gate = s->q4_base + s->q4_off[layer * 3] + (size_t)e * L->nb_g;
            *up   = s->q4_base + s->q4_off[layer * 3 + 1] + (size_t)e * L->nb_u;
            *down = s->q4_base + s->q4_off[layer * 3 + 2] + (size_t)e * L->nb_d;
            return *gate;
        }
        return r_g;
    }
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
    if (!s || s->mode != LLM_DSV4_STORE_SWAP || s->mmap_only || n <= 0) return;
    if (llm_dsv4_store_q4_stacks(s, layer, NULL, NULL, NULL)) { llm_dsv4_store_q4_ensure(s, layer, experts, n); return; }
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

/* PREFETCH ASINCRONO MIRATO di un layer (item B): `posix_madvise(WILLNEED)` sui K ESPERTI piu'
 * frequenti osservati (hotness `freq`, D19-A) del layer, non su tutto il layer. Il kernel legge in
 * BACKGROUND (ritorna subito) → l'I/O si sovrappone al calcolo GPU del layer precedente. Opzione di
 * registro `dsv4_prefetch_hot=K` (K esperti; default 0 = OFF). Il GGUF e' gia' mmap (`resident_base`).
 *
 * MISURA (2026-10-03): il prefetch dell'INTERO layer e' NOCIVO (8192 tok sintetici: 67.9 → 17.7 t/s,
 * 3.8x piu' lento: legge tutti gli esperti di ogni layer e satura l'I/O). Quindi QUI si limita ai
 * top-K per frequenza; senza localita' (token sintetici casuali) freq e' ~uniforme → nessun guadagno
 * atteso (serve testo reale, come la hotlist ds4). Se `freq` non e' allocata → no-op. */
static inline void llm_dsv4_store_madvise_layer(LlmDsv4Store* s, int layer) {
    if (!s || s->mode != LLM_DSV4_STORE_SWAP || s->mmap_only || !s->resident_base || s->freq == NULL) return;
    if (layer < 0 || layer >= s->f_layers) return;
    if (s->q4_off && s->q4_off[layer * 3] != (size_t)-1) return;   /* arena Q4_K: gestita da q4_ensure */
    int K = 0; { int v; if (llm_ai_mlx_opt_get_int("dsv4_prefetch_hot", &v) == 0) K = v; }
    if (K <= 0 || s->f_experts <= 0 || s->f_experts > 512) return;
    if (K > s->f_experts) K = s->f_experts;
    const LlmDsv4StoreLayer* L = &s->layers[layer];
    if (L->stride_g == 0 || L->stride_u == 0 || L->stride_d == 0) return;
    uint32_t tmp[512];
    memcpy(tmp, &s->freq[(size_t)layer * s->f_experts], sizeof(uint32_t) * (size_t)s->f_experts);
    for (int k = 0; k < K; k++) {
        int best = -1; uint32_t bf = 0;
        for (int e = 0; e < s->f_experts; e++) if (tmp[e] > bf) { bf = tmp[e]; best = e; }
        if (best < 0) break;   /* nessuna osservazione (>0) per questo layer */
        tmp[best] = 0;
#if defined(POSIX_MADV_WILLNEED)
        (void)posix_madvise((void*)(s->resident_base + L->off_g + (uint64_t)best * L->stride_g), L->stride_g, POSIX_MADV_WILLNEED);
        (void)posix_madvise((void*)(s->resident_base + L->off_u + (uint64_t)best * L->stride_u), L->stride_u, POSIX_MADV_WILLNEED);
        (void)posix_madvise((void*)(s->resident_base + L->off_d + (uint64_t)best * L->stride_d), L->stride_d, POSIX_MADV_WILLNEED);
#endif
    }
}

/* ── HOTLIST I/O (item B/c, come `ds4 --ssd-streaming`): persisti la hotness (freq) su file e
 * pre-carica i top-K esperti per layer nella cache. Attacca l'I/O (~29%) senza speculazione cieca:
 * il seed usa le frequenze OSSERVATE (profilo), non una predizione arbitraria. ── */
static inline int llm_dsv4_store_save_hot(const LlmDsv4Store* s, const char* path) {
    if (!s || !path || !path[0] || !s->freq || s->f_layers <= 0 || s->f_experts <= 0) return -1;
    FILE* f = fopen(path, "wb"); if (!f) return -1;
    int hdr[2] = { s->f_layers, s->f_experts };
    int ok = (fwrite(hdr, sizeof(int), 2, f) == 2) &&
             (fwrite(s->freq, sizeof(uint32_t), (size_t)s->f_layers * s->f_experts, f) ==
              (size_t)s->f_layers * s->f_experts);
    fclose(f); return ok ? 0 : -1;
}
static inline int llm_dsv4_store_load_hot(LlmDsv4Store* s, const char* path) {
    if (!s || !path || !path[0]) return -1;
    FILE* f = fopen(path, "rb"); if (!f) return -1;
    int hdr[2] = { 0, 0 };
    if (fread(hdr, sizeof(int), 2, f) != 2 || hdr[0] <= 0 || hdr[1] <= 0 || hdr[1] > 65536) { fclose(f); return -1; }
    llm_dsv4_store_set_dims(s, hdr[0], hdr[1]);
    if (!s->freq) { fclose(f); return -1; }
    size_t n = (size_t)hdr[0] * hdr[1];
    size_t got = fread(s->freq, sizeof(uint32_t), n, f);
    fclose(f); return got == n ? 0 : -1;
}
/* Pre-carica i top-K (per freq>0) esperti di OGNI layer nella cache (claim+pread via prefetch). */
static inline void llm_dsv4_store_seed_hot(LlmDsv4Store* s, int top_k) {
    if (!s || !s->freq || top_k <= 0 || s->f_experts <= 0 || s->f_experts > 4096) return;
    int* sel = (int*)malloc(sizeof(int) * (size_t)top_k);
    if (!sel) return;
    for (int L = 0; L < s->f_layers; L++) {
        const uint32_t* f = &s->freq[(size_t)L * s->f_experts];
        int n = 0;
        for (int k = 0; k < top_k; k++) {
            int best = -1; uint32_t bf = 0;
            for (int e = 0; e < s->f_experts; e++) {
                int chosen = 0; for (int j = 0; j < n; j++) if (sel[j] == e) { chosen = 1; break; }
                if (!chosen && f[e] > bf) { bf = f[e]; best = e; }
            }
            if (best < 0) break;   /* nessuna osservazione (>0) rimasta */
            sel[n++] = best;
        }
        if (n > 0) llm_dsv4_store_prefetch(s, L, sel, n);
    }
    free(sel);
}

#endif /* LLM_DSV4_STORE_H */
