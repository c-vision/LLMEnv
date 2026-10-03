/* llm_mlx_kvpage_store.h — PAGE TABLE + ALLOCATOR del KV paged (port dell'idea
 * `PageStorage`/`PagedKv.hpp` di Splash).
 *
 * Da `splash`:
 *   - blocchi logici di prefisso da **32 token** (`SPLASH_TARGET_KV_BLOCK_TOKENS`);
 *   - storage fisico a **extent** con **allineamento 64 KiB** (extent ~128 MiB);
 *   - una **page table logical->physical** unica (alloc/admission/identita' cache/pianificazione
 *     attenzione): i blocchi logici contigui possono vivere in pagine fisiche NON contigue.
 *
 * Questo modulo e' PURO (nessuna MLX): gestisce la MAPPA e l'allocatore di pagine fisiche,
 * testabile in isolamento. L'array MLX che materializza le pagine vive nel pool int8
 * (`llm_mlx_kvq_pool.h`), che usa questa page table per calcolare gli indici fisici.
 *
 * Convenzione (come llm_mlx_kvpage.h / llm_mlx_ops.h): `static inline`, pura.
 */
#ifndef LLM_MLX_KVPAGE_STORE_H
#define LLM_MLX_KVPAGE_STORE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Layout (Splash) ────────────────────────────────────────────────────────── */
#define LLM_KV_PAGE_TOKENS   32u                       /* token per blocco logico */
#define LLM_KV_PAGE_ALIGN    (64u * 1024u)             /* allineamento di una pagina (byte) */
#define LLM_KV_EXTENT_BYTES  (128u * 1024u * 1024u)    /* extent fisico ~128 MiB */

/* Byte occupati da UNA pagina fisica di un layer: K + V int8 (nkv*32*hd ciascuno) +
 * scale f32 (nkv*32 ciascuno, K e V). bf16 raddoppia i dati. */
static inline size_t llm_kv_page_bytes(int fmt_int8, uint32_t nkv, uint32_t hd) {
    const uint32_t d = fmt_int8 ? 1u : 2u;
    return 2u * (size_t)LLM_KV_PAGE_TOKENS * nkv * hd * d        /* K + V data */
         + 2u * (size_t)LLM_KV_PAGE_TOKENS * nkv * sizeof(float);/* scale K + V */
}

/* Pagine per extent (dal budget di byte): almeno 1. */
static inline int llm_kv_pages_per_extent(size_t page_bytes) {
    if (page_bytes == 0) return 1;
    int n = (int)(LLM_KV_EXTENT_BYTES / page_bytes);
    return n > 0 ? n : 1;
}

/* Numero di pagine fisiche necessarie per `bytes` con allineamento a 64 KiB (>=1). */
static inline int llm_kv_pages_for_bytes(size_t bytes) {
    size_t a = (bytes + LLM_KV_PAGE_ALIGN - 1) / LLM_KV_PAGE_ALIGN * LLM_KV_PAGE_ALIGN;
    if (a == 0) a = LLM_KV_PAGE_ALIGN;
    int n = (int)(a / LLM_KV_PAGE_ALIGN);
    return n > 0 ? n : 1;
}

/* ── Page table logical -> physical ─────────────────────────────────────────── */
typedef struct {
    int      phys_capacity;   /* numero di pagine fisiche disponibili */
    int      used;            /* pagine fisiche in uso */
    int      free_head;       /* testa della free-list (-1 se vuota) */
    int32_t* next_free;       /* [phys_capacity] catena free-list */
    int32_t* table;           /* [logical_capacity] logical -> physical (-1 se non mappata) */
    int      logical_capacity;
    int      n_logical;       /* pagine logiche mappate (prefisso contiguo) */
} LlmKvPageTable;

static inline LlmKvPageTable llm_kv_pt_new(int phys_pages, int logical_capacity) {
    LlmKvPageTable t;
    memset(&t, 0, sizeof t);
    if (phys_pages < 1) phys_pages = 1;
    if (logical_capacity < 1) logical_capacity = 1;
    t.phys_capacity = phys_pages;
    t.logical_capacity = logical_capacity;
    t.used = 0;
    t.n_logical = 0;
    t.next_free = (int32_t*)malloc((size_t)phys_pages * sizeof(int32_t));
    t.table = (int32_t*)malloc((size_t)logical_capacity * sizeof(int32_t));
    if (!t.next_free || !t.table) { free(t.next_free); free(t.table); t.next_free = NULL; t.table = NULL; return t; }
    for (int i = 0; i + 1 < phys_pages; i++) t.next_free[i] = i + 1;
    t.next_free[phys_pages - 1] = -1;
    t.free_head = 0;
    for (int i = 0; i < logical_capacity; i++) t.table[i] = -1;
    return t;
}

static inline void llm_kv_pt_free(LlmKvPageTable* t) {
    if (!t) return;
    free(t->next_free); free(t->table);
    t->next_free = NULL; t->table = NULL;
}

/* Alloca una pagina fisica (dalla free-list). Ritorna l'id o -1 se esaurite. */
static inline int llm_kv_pt_alloc_phys(LlmKvPageTable* t) {
    if (!t || t->free_head < 0) return -1;
    int id = t->free_head;
    t->free_head = t->next_free[id];
    t->used++;
    return id;
}

/* Rilascia una pagina fisica alla free-list (per eviction/riuso). */
static inline void llm_kv_pt_release_phys(LlmKvPageTable* t, int id) {
    if (!t || id < 0 || id >= t->phys_capacity) return;
    t->next_free[id] = (int32_t)t->free_head;
    t->free_head = id;
    if (t->used > 0) t->used--;
}

/* Mappa (o ritorna) la pagina fisica per la pagina logica `lp`. -1 se non allocabile. */
static inline int llm_kv_pt_map(LlmKvPageTable* t, int lp) {
    if (!t || lp < 0 || lp >= t->logical_capacity) return -1;
    if (t->table[lp] >= 0) return t->table[lp];
    int phys = llm_kv_pt_alloc_phys(t);
    if (phys < 0) return -1;
    t->table[lp] = (int32_t)phys;
    if (lp + 1 > t->n_logical) t->n_logical = lp + 1;
    return phys;
}

static inline int llm_kv_pt_phys(const LlmKvPageTable* t, int lp) {
    if (!t || lp < 0 || lp >= t->logical_capacity) return -1;
    return t->table[lp];
}

/* Estende la capacita' fisica (e logica) preservando le mappature e la free-list. */
static inline bool llm_kv_pt_grow(LlmKvPageTable* t, int new_phys, int new_logical) {
    if (!t || !t->next_free || !t->table) return false;
    if (new_phys < t->phys_capacity) new_phys = t->phys_capacity;
    if (new_logical < new_phys) new_logical = new_phys;
    int32_t* nf = (int32_t*)realloc(t->next_free, (size_t)new_phys * sizeof(int32_t));
    int32_t* nt = (int32_t*)realloc(t->table, (size_t)new_logical * sizeof(int32_t));
    if (!nf || !nt) { if (nf) t->next_free = nf; if (nt) t->table = nt; return false; }
    t->next_free = nf; t->table = nt;
    /* FIX (2026-10-03): le nuove pagine fisiche vanno in CODA alla free-list, NON in testa.
     * In testa, dopo >=2 grow l'ordine di allocazione non era piu' monotono -> `map(lp) != lp`
     * (mapping NON identita'), mentre il `llm_kvq_pool_grow` copia il prefisso fisico [0..committed)
     * ASSUMENDO identita' -> mescolava i token -> pool int8 corrotto (misurato da 2400 token). In
     * coda, l'allocazione resta monotona (0,1,2,...) = identita' -> la copia del grow e' valida. */
    if (t->free_head < 0) {
        t->free_head = t->phys_capacity;
        for (int i = t->phys_capacity; i < new_phys; i++)
            t->next_free[i] = (i + 1 < new_phys) ? (int32_t)(i + 1) : -1;
    } else {
        int cur = t->free_head;
        while (t->next_free[cur] != -1) cur = t->next_free[cur];
        t->next_free[cur] = (int32_t)t->phys_capacity;
        for (int i = t->phys_capacity; i < new_phys; i++)
            t->next_free[i] = (i + 1 < new_phys) ? (int32_t)(i + 1) : -1;
    }
    for (int i = t->phys_capacity; i < new_logical; i++) t->table[i] = -1;
    t->phys_capacity = new_phys;
    t->logical_capacity = new_logical;
    return true;
}

/* Riempie `out` con la mappa logical->physical per [0, n_logical): usata per l'array
 * int32 da passare al kernel. `out` deve avere >= n_logical elementi. */
static inline void llm_kv_pt_export(const LlmKvPageTable* t, int32_t* out, int n) {
    if (!t || !out) return;
    for (int i = 0; i < n; i++) out[i] = (i < t->logical_capacity) ? t->table[i] : -1;
}

#endif /* LLM_MLX_KVPAGE_STORE_H */
