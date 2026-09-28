/* llmengine_moe_cache.c — expert cache thread-safe con LRU + single-flight (M4).
 *
 * Struttura: hash table (chiave layer/expert) + lista LRU doppia. `total` = byte dei blob.
 * Single-flight: un placeholder `in_flight` per chiave; chi arriva secondo attende sul condvar.
 * L'eviction non tocca MAI il record piu' recente (ne' gli in-flight). */
#include "llmengine_moe_cache.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define LLMENGINE_MOE_CACHE_BUCKETS 4096u

typedef struct LlmEngineMoeCacheEntry {
    LlmEngineMoeExpertKey key;
    uint8_t* data;
    size_t size;
    int in_flight;
    uint64_t use_count;   /* frequenza (M7) */
    uint64_t last_used;   /* tick di recency (M7) */
    float route_prob;     /* probabilita' di routing (M7) */
    float future;         /* probabilita' d'uso futuro (M7, dal prefetcher) */
    struct LlmEngineMoeCacheEntry* hnext;              /* catena hash */
    struct LlmEngineMoeCacheEntry* lru_prev, *lru_next;/* lista LRU (head=MRU) */
} LlmEngineMoeCacheEntry;

struct LlmEngineMoeExpertCache {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t budget;
    uint64_t total;
    LlmEngineMoeCacheEntry** buckets;
    size_t nbuckets;
    LlmEngineMoeCacheEntry* head, *tail;
    LlmEngineMoeCacheStats st;
    uint8_t* scratch; size_t scratch_cap; /* solo per budget==0 (carica senza memorizzare) */
    LlmEngineMoeCachePolicy policy;             /* M7 */
    float alpha, beta, gamma, delta;      /* pesi scoring (M7) */
    uint64_t tick;                        /* orologio logico per la recency */
};

static size_t cache_hash(LlmEngineMoeExpertKey k) {
    uint64_t h = (uint64_t)k.layer * 1000003ull ^ (uint64_t)k.expert;
    return (size_t)(h % LLMENGINE_MOE_CACHE_BUCKETS);
}

static void lru_unlink(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* e) {
    if (e->lru_prev) e->lru_prev->lru_next = e->lru_next; else c->head = e->lru_next;
    if (e->lru_next) e->lru_next->lru_prev = e->lru_prev; else c->tail = e->lru_prev;
    e->lru_prev = e->lru_next = NULL;
}
static void lru_push_front(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* e) {
    e->lru_prev = NULL; e->lru_next = c->head;
    if (c->head) c->head->lru_prev = e;
    c->head = e;
    if (c->tail == NULL) c->tail = e;
}
static void lru_touch(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* e) {
    e->use_count++;
    e->last_used = ++c->tick;
    if (c->head == e) return;
    lru_unlink(c, e);
    lru_push_front(c, e);
}

static LlmEngineMoeCacheEntry* hash_find(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey k) {
    for (LlmEngineMoeCacheEntry* e = c->buckets[cache_hash(k)]; e; e = e->hnext)
        if (e->key.layer == k.layer && e->key.expert == k.expert) return e;
    return NULL;
}
static void hash_link(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* e) {
    size_t b = cache_hash(e->key);
    e->hnext = c->buckets[b];
    c->buckets[b] = e;
}
static void hash_unlink(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* e) {
    size_t b = cache_hash(e->key);
    LlmEngineMoeCacheEntry** pp = &c->buckets[b];
    while (*pp) { if (*pp == e) { *pp = e->hnext; return; } pp = &(*pp)->hnext; }
}

/* rimuove e libera (chiamare con lock tenuto) */
static void entry_free(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* e) {
    hash_unlink(c, e);
    lru_unlink(c, e);
    if (e->data) { c->total -= e->size; free(e->data); }
    free(e);
}

/* score di eviction (M7): piu' basso = evincibile prima. LRU = caso con alpha=1, beta=gamma=delta=0. */
static double entry_score(const LlmEngineMoeExpertCache* c, const LlmEngineMoeCacheEntry* e) {
    return (double)c->alpha * (double)e->last_used
         + (double)c->beta  * (double)e->use_count
         + (double)c->gamma * (double)e->route_prob
         + (double)c->delta * (double)e->future;
}
static int key_less(LlmEngineMoeExpertKey a, LlmEngineMoeExpertKey b) {
    if (a.layer != b.layer) return a.layer < b.layer;
    return a.expert < b.expert;
}
/* evince per score (deterministico) finche' total <= budget; mai `keep` ne' in-flight. */
static void evict_over_budget(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheEntry* keep) {
    if (c->budget == 0) return;
    while (c->total > c->budget) {
        LlmEngineMoeCacheEntry* best = NULL;
        double bs = 0.0;
        for (size_t b = 0; b < c->nbuckets; ++b) {
            for (LlmEngineMoeCacheEntry* e = c->buckets[b]; e; e = e->hnext) {
                if (e == keep || e->in_flight) continue;
                double sc = entry_score(c, e);
                if (best == NULL || sc < bs || (sc == bs && key_less(e->key, best->key))) {
                    best = e; bs = sc;
                }
            }
        }
        if (best == NULL) break; /* restano solo keep/in-flight: non si evince oltre */
        entry_free(c, best);
        c->st.evictions++;
    }
}

LlmEngineMoeStatus llmengine_moe_cache_create(LlmEngineMoeExpertCache** out, uint64_t budget_bytes) {
    if (out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    LlmEngineMoeExpertCache* c = (LlmEngineMoeExpertCache*)calloc(1, sizeof(*c));
    if (c == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    c->buckets = (LlmEngineMoeCacheEntry**)calloc(LLMENGINE_MOE_CACHE_BUCKETS, sizeof(*c->buckets));
    if (c->buckets == NULL) { free(c); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    c->nbuckets = LLMENGINE_MOE_CACHE_BUCKETS;
    c->budget = budget_bytes;
    c->policy = LLMENGINE_MOE_CACHE_LRU; c->alpha = 1.0f; c->beta = 0.f; c->gamma = 0.f; c->delta = 0.f;
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->cv, NULL);
    *out = c;
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_cache_destroy(LlmEngineMoeExpertCache* c) {
    if (c == NULL) return;
    for (size_t i = 0; i < c->nbuckets; ++i) {
        LlmEngineMoeCacheEntry* e = c->buckets[i];
        while (e) { LlmEngineMoeCacheEntry* n = e->hnext; if (e->data) free(e->data); free(e); e = n; }
    }
    free(c->buckets);
    free(c->scratch);
    pthread_mutex_destroy(&c->mu);
    pthread_cond_destroy(&c->cv);
    free(c);
}

LlmEngineMoeStatus llmengine_moe_cache_lookup(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                  const void** data_out, size_t* size_out) {
    if (c == NULL || data_out == NULL || size_out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *data_out = NULL; *size_out = 0;
    pthread_mutex_lock(&c->mu);
    LlmEngineMoeCacheEntry* e = hash_find(c, key);
    if (e != NULL && !e->in_flight) { lru_touch(c, e); *data_out = e->data; *size_out = e->size; }
    pthread_mutex_unlock(&c->mu);
    return LLMENGINE_MOE_OK;
}

static LlmEngineMoeStatus cache_insert_impl(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                      const void* data, size_t size, int has_hint,
                                      float route_prob, float future) {
    if (c == NULL || (data == NULL && size > 0)) return LLMENGINE_MOE_ERR_INVALID;
    uint8_t* copy = NULL;
    if (size > 0) {
        copy = (uint8_t*)malloc(size);
        if (copy == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
        memcpy(copy, data, size);
    }
    pthread_mutex_lock(&c->mu);
    LlmEngineMoeCacheEntry* e = hash_find(c, key);
    if (e != NULL) {
        if (e->data) { c->total -= e->size; free(e->data); }
        e->data = copy; e->size = size; e->in_flight = 0;
        c->total += size;
        c->tick++; e->last_used = c->tick;
    } else {
        e = (LlmEngineMoeCacheEntry*)calloc(1, sizeof(*e));
        if (e == NULL) { pthread_mutex_unlock(&c->mu); free(copy); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        e->key = key; e->data = copy; e->size = size; e->in_flight = 0;
        hash_link(c, e);
        lru_push_front(c, e);
        c->tick++; e->last_used = c->tick;
        c->total += size;
    }
    if (has_hint) { e->route_prob = route_prob; e->future = future; }
    evict_over_budget(c, e);
    pthread_mutex_unlock(&c->mu);
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_cache_insert(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                  const void* data, size_t size) {
    return cache_insert_impl(c, key, data, size, 0, 0.0f, 0.0f);
}

LlmEngineMoeStatus llmengine_moe_cache_insert_hint(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key,
                                       const void* data, size_t size,
                                       float route_prob, float future) {
    return cache_insert_impl(c, key, data, size, 1, route_prob, future);
}

void llmengine_moe_cache_set_hint(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key, float route_prob, float future) {
    if (c == NULL) return;
    pthread_mutex_lock(&c->mu);
    LlmEngineMoeCacheEntry* e = hash_find(c, key);
    if (e != NULL) { e->route_prob = route_prob; e->future = future; }
    pthread_mutex_unlock(&c->mu);
}

void llmengine_moe_cache_set_policy(LlmEngineMoeExpertCache* c, LlmEngineMoeCachePolicy pol,
                              float alpha, float beta, float gamma, float delta) {
    if (c == NULL) return;
    pthread_mutex_lock(&c->mu);
    c->policy = pol; c->alpha = alpha; c->beta = beta; c->gamma = gamma; c->delta = delta;
    pthread_mutex_unlock(&c->mu);
}

LlmEngineMoeStatus llmengine_moe_cache_remove(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key) {
    if (c == NULL) return LLMENGINE_MOE_ERR_INVALID;
    pthread_mutex_lock(&c->mu);
    LlmEngineMoeCacheEntry* e = hash_find(c, key);
    if (e != NULL) entry_free(c, e);
    pthread_mutex_unlock(&c->mu);
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_cache_clear(LlmEngineMoeExpertCache* c) {
    if (c == NULL) return;
    pthread_mutex_lock(&c->mu);
    LlmEngineMoeCacheEntry* e = c->head;
    while (e) { LlmEngineMoeCacheEntry* n = e->lru_next; entry_free(c, e); e = n; }
    c->total = 0;
    pthread_mutex_unlock(&c->mu);
}

LlmEngineMoeStatus llmengine_moe_cache_get(LlmEngineMoeExpertCache* c, LlmEngineMoeExpertKey key, size_t expected_size,
                               LlmEngineMoeCacheLoadFn load, void* ctx,
                               const void** data_out, size_t* size_out) {
    if (c == NULL || load == NULL || data_out == NULL || size_out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *data_out = NULL; *size_out = 0;
    if (c->budget == 0) { /* cache disabilitata: carica nello scratch (nessuna retention) */
        pthread_mutex_lock(&c->mu);
        if (expected_size > c->scratch_cap) {
            uint8_t* nb = (uint8_t*)realloc(c->scratch, expected_size ? expected_size : 1);
            if (nb == NULL) { pthread_mutex_unlock(&c->mu); return LLMENGINE_MOE_ERR_NO_MEMORY; }
            c->scratch = nb; c->scratch_cap = expected_size;
        }
        uint8_t* buf = c->scratch;
        pthread_mutex_unlock(&c->mu);
        size_t got = 0;
        LlmEngineMoeStatus s = load(ctx, key, buf, expected_size, &got);
        if (s != LLMENGINE_MOE_OK) return s;
        *data_out = buf; *size_out = got;
        return LLMENGINE_MOE_OK;
    }

    LlmEngineMoeCacheEntry* e = NULL;
    pthread_mutex_lock(&c->mu);
    for (;;) {
        e = hash_find(c, key);
        if (e != NULL && !e->in_flight) {
            c->st.hits++;
            lru_touch(c, e);
            *data_out = e->data; *size_out = e->size;
            pthread_mutex_unlock(&c->mu);
            return LLMENGINE_MOE_OK;
        }
        if (e != NULL && e->in_flight) { pthread_cond_wait(&c->cv, &c->mu); continue; }
        /* miss: placeholder in-flight */
        c->st.misses++;
        e = (LlmEngineMoeCacheEntry*)calloc(1, sizeof(*e));
        if (e == NULL) { pthread_mutex_unlock(&c->mu); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        e->key = key; e->in_flight = 1;
        hash_link(c, e);
        lru_push_front(c, e);
        break;
    }
    pthread_mutex_unlock(&c->mu);

    /* load fuori dal lock (I/O) */
    uint8_t* buf = (uint8_t*)malloc(expected_size ? expected_size : 1);
    size_t got = 0;
    LlmEngineMoeStatus s = (buf == NULL) ? LLMENGINE_MOE_ERR_NO_MEMORY : load(ctx, key, buf, expected_size, &got);

    pthread_mutex_lock(&c->mu);
    if (s == LLMENGINE_MOE_OK) {
        e->data = buf; e->size = got; e->in_flight = 0;
        c->total += got; c->st.bytes_loaded += got;
        lru_touch(c, e);
        evict_over_budget(c, e);
        *data_out = e->data; *size_out = e->size;
    } else {
        free(buf);
        entry_free(c, e);
    }
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
    return s;
}

void llmengine_moe_cache_stats(LlmEngineMoeExpertCache* c, LlmEngineMoeCacheStats* out) {
    if (c == NULL || out == NULL) return;
    pthread_mutex_lock(&c->mu);
    *out = c->st;
    pthread_mutex_unlock(&c->mu);
}

uint64_t llmengine_moe_cache_bytes_used(const LlmEngineMoeExpertCache* c) { return c == NULL ? 0 : c->total; }
uint64_t llmengine_moe_cache_budget(const LlmEngineMoeExpertCache* c) { return c == NULL ? 0 : c->budget; }
