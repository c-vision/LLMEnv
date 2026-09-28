/* llm_mlx_kvpage.h — KV-CACHE PAGING (roadmap 8.11 fase 3 Safe Surgery).
 *
 * Gestione della memoria KV a BLOCCHI (vLLM-style block paging): invece di una
 * cache contigua che cresce (concatenate O(N)/step, allocazione contigua), la
 * KV-cache e' organizzata in pagine di `block_size` token. Il pool alloca/libera
 * blocchi fisici e fa EVICTION LRU quando e' pieno.
 *
 * Questo modulo e' PURO (nessuna dipendenza MLX): gestisce la MAPPA di blocchi
 * (quale pagina fisica e' usata, da chi, quanto recente). L'integrazione col
 * layout reale della cache MLX (array per-blocco, indirizzamento
 * seq_id -> [page, offset]) e' il passo successivo nel motore.
 *
 * CONTENT-ADDRESSED (2026-09-01): oltre al paging fisico, il pool e' content-
 * addressed — il blocco e' la chiave, non l'offset. Ogni pagina porta un
 * `content_hash` (FNV-1a dei token del blocco); `llm_kv_page_alloc_by_hash`
 * RIUUSA una pagina con lo stesso contenuto (stesso prefisso tra richieste),
 * `llm_kv_find_by_hash` fa lookup per hash, e `pin`/`unpin` rendono i blocchi
 * stabili NON evictabili (prefissi stabili del contesto lungo). Verificato:
 * dedup hash, riuso, lookup, eviction rispetta i pinned, pool tutto-pinnato -> -1.
 *
 * Convenzione (come llm_mlx_ops.h): funzioni `static`, pura, nessuno stato globale.
 * Testabile in isolamento senza MLX.
 */
#ifndef LLM_MLX_KVPAGE_H
#define LLM_MLX_KVPAGE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int id;            /* indice della pagina fisica */
    bool used;         /* true se allocata */
    bool pinned;       /* pinnata: mai evictabile (prefissi stabili) */
    uint64_t last_used;/* timestamp LRU (dal clock del pool) */
    uint64_t content_hash; /* content-addressed: hash del contenuto del blocco */
    int64_t owner;     /* seq_id (o -1 se libera) -- per l'integrazione futura */
} KvPage;

typedef struct {
    KvPage* pages;     /* array di n_pages pagine fisiche */
    int n_pages;       /* numero totale di pagine fisiche */
    int block_size;    /* token per pagina */
    uint64_t clock;    /* contatore monotono per LRU */
    long allocs;       /* statistiche: allocazioni totali */
    long evictions;    /* statistiche: eviction LRU */
} KvPagePool;

static inline KvPagePool llm_kv_page_pool_new(int n_pages, int block_size) {
    KvPagePool p;
    p.n_pages = n_pages > 0 ? n_pages : 1024;
    p.block_size = block_size > 0 ? block_size : 16;
    p.clock = 0;
    p.allocs = 0;
    p.evictions = 0;
    p.pages = (KvPage*)calloc((size_t)p.n_pages, sizeof(KvPage));
    for (int i = 0; i < p.n_pages; i++) { p.pages[i].id = i; p.pages[i].owner = -1; }
    return p;
}

static inline void llm_kv_page_pool_free(KvPagePool* p) {
    if (!p || !p->pages) return;
    free(p->pages);
    p->pages = NULL;
}

/* Segna la pagina come recentemente usata (LRU touch). */
static inline void llm_kv_page_touch(KvPagePool* p, int id) {
    if (!p || id < 0 || id >= p->n_pages) return;
    p->pages[id].last_used = ++p->clock;
}

/* FNV-1a 64-bit: hash deterministico del contenuto del blocco (token ids). */
static inline uint64_t llm_kv_block_hash(const int32_t* tokens, int len) {
    uint64_t h = 1469598103934665603ULL;
    if (!tokens || len <= 0) return h;
    for (int i = 0; i < len; i++) {
        uint64_t b = (uint64_t)(uint32_t)tokens[i];
        h ^= b & 0xFF;            h *= 1099511628211ULL;
        h ^= (b >> 8) & 0xFF;     h *= 1099511628211ULL;
        h ^= (b >> 16) & 0xFF;    h *= 1099511628211ULL;
        h ^= (b >> 24) & 0xFF;    h *= 1099511628211ULL;
    }
    return h;
}

/* Pin: rende la pagina NON evictabile (prefissi stabili del contesto lungo). */
static inline void llm_kv_page_pin(KvPagePool* p, int id) {
    if (!p || id < 0 || id >= p->n_pages) return;
    p->pages[id].pinned = true;
    p->pages[id].last_used = ++p->clock;
}

/* Lookup content-addressed: trova una pagina USATA col dato hash (stesso contenuto), o -1. */
static inline int llm_kv_find_by_hash(const KvPagePool* p, uint64_t hash) {
    if (!p || !p->pages) return -1;
    for (int i = 0; i < p->n_pages; i++) {
        if (p->pages[i].used && p->pages[i].content_hash == hash) return i;
    }
    return -1;
}

/* Alloca una pagina: ritorna l'id fisico, o -1 se il pool e' vuoto. Se non ci sono
 * pagine libere, EVICT la meno usata di recente (LRU) e la riassegna. */
static inline int llm_kv_page_alloc(KvPagePool* p, int64_t owner) {
    if (!p || !p->pages) return -1;
    /* cerca una pagina libera */
    for (int i = 0; i < p->n_pages; i++) {
        if (!p->pages[i].used) {
            p->pages[i].used = true;
            p->pages[i].owner = owner;
            p->pages[i].last_used = ++p->clock;
            p->allocs++;
            return i;
        }
    }
    /* pool pieno: evict LRU (salta le pagine PINNATE -- mai evictabile) */
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < p->n_pages; i++) {
        if (p->pages[i].pinned) continue;
        if (p->pages[i].last_used < oldest) { oldest = p->pages[i].last_used; victim = i; }
    }
    if (victim < 0) return -1; /* tutte pinnate: nessuna evictabile */
    p->pages[victim].owner = owner;
    p->pages[victim].last_used = ++p->clock;
    p->allocs++;
    p->evictions++;
    return victim;
}

/* Alloca (o RIUUSA content-addressed) una pagina: se esiste gia' una pagina usata con lo
 * stesso content_hash, la riusa (restituisce il suo id, senza nuova allocazione); altrimenti
 * alloca una pagina nuova e le assegna l'hash. Ritorna l'id, o -1 se il pool e' pieno e
 * nessuna pagina evictabile (tutte pinnate). */
static inline int llm_kv_page_alloc_by_hash(KvPagePool* p, int64_t owner, uint64_t hash) {
    if (!p || !p->pages) return -1;
    /* content-addressed: riuso del blocco con lo stesso contenuto */
    int existing = llm_kv_find_by_hash(p, hash);
    if (existing >= 0) {
        p->pages[existing].last_used = ++p->clock; /* touch */
        return existing;
    }
    int id = llm_kv_page_alloc(p, owner);
    if (id < 0) return -1;
    p->pages[id].content_hash = hash;
    return id;
}

/* Libera una pagina (ritorna al pool). */
static inline void llm_kv_page_free(KvPagePool* p, int id) {
    if (!p || id < 0 || id >= p->n_pages) return;
    p->pages[id].used = false;
    p->pages[id].owner = -1;
}

#endif /* LLM_MLX_KVPAGE_H */
