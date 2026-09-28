/* llmengine_moe_kv.c — KV cache separata (M11): LRU con hash chain, budget in byte dedicato. */
#include "llmengine_moe_kv.h"

#include <stdlib.h>
#include <string.h>

#define LLMENGINE_MOE_KV_BUCKETS 64u

typedef struct LlmEngineMoeKvEntry {
    LlmEngineMoeKvKey key;
    uint8_t* data;
    size_t size;
    struct LlmEngineMoeKvEntry* prev;  /* LRU: piu' recente */
    struct LlmEngineMoeKvEntry* next;  /* LRU: meno recente */
    struct LlmEngineMoeKvEntry* hnext; /* catena hash */
} LlmEngineMoeKvEntry;

struct LlmEngineMoeKvCache {
    LlmEngineMoeKvEntry** buckets;
    size_t nbuckets;
    LlmEngineMoeKvEntry* head; /* MRU */
    LlmEngineMoeKvEntry* tail; /* LRU */
    uint64_t budget;
    uint64_t used;
    uint32_t entries;
    LlmEngineMoeKvStats stats;
};

static size_t kv_hash(LlmEngineMoeKvKey k, size_t nb) {
    uint64_t h = ((uint64_t)k.layer << 32) ^ (uint64_t)k.page;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return (size_t)(h % nb);
}

static int kv_eq(LlmEngineMoeKvKey a, LlmEngineMoeKvKey b) {
    return a.layer == b.layer && a.page == b.page;
}

LlmEngineMoeStatus llmengine_moe_kv_create(LlmEngineMoeKvCache** out, uint64_t budget_bytes) {
    if (out == NULL || budget_bytes == 0) return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    LlmEngineMoeKvCache* kv = (LlmEngineMoeKvCache*)calloc(1, sizeof(*kv));
    if (kv == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    kv->nbuckets = LLMENGINE_MOE_KV_BUCKETS;
    kv->buckets = (LlmEngineMoeKvEntry**)calloc(kv->nbuckets, sizeof(*kv->buckets));
    if (kv->buckets == NULL) { free(kv); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    kv->budget = budget_bytes;
    *out = kv;
    return LLMENGINE_MOE_OK;
}

static void kv_free_entry(LlmEngineMoeKvCache* kv, LlmEngineMoeKvEntry* e) {
    size_t b = kv_hash(e->key, kv->nbuckets);
    LlmEngineMoeKvEntry** pp = &kv->buckets[b];
    while (*pp != NULL && *pp != e) pp = &(*pp)->hnext;
    if (*pp == e) *pp = e->hnext;
    /* unlink dalla LRU */
    if (e->prev != NULL) e->prev->next = e->next; else kv->head = e->next;
    if (e->next != NULL) e->next->prev = e->prev; else kv->tail = e->prev;
    kv->used -= e->size;
    kv->entries--;
    free(e->data);
    free(e);
}

static void kv_touch(LlmEngineMoeKvCache* kv, LlmEngineMoeKvEntry* e) {
    if (kv->head == e) return;
    if (e->prev != NULL) e->prev->next = e->next; else kv->head = e->next;
    if (e->next != NULL) e->next->prev = e->prev; else kv->tail = e->prev;
    e->prev = NULL;
    e->next = kv->head;
    if (kv->head != NULL) kv->head->prev = e;
    kv->head = e;
    if (kv->tail == NULL) kv->tail = e;
}

void llmengine_moe_kv_destroy(LlmEngineMoeKvCache* kv) {
    if (kv == NULL) return;
    LlmEngineMoeKvEntry* e = kv->head;
    while (e != NULL) { LlmEngineMoeKvEntry* n = e->next; free(e->data); free(e); e = n; }
    free(kv->buckets);
    free(kv);
}

LlmEngineMoeStatus llmengine_moe_kv_lookup(LlmEngineMoeKvCache* kv, LlmEngineMoeKvKey key,
                               const void** data_out, size_t* size_out) {
    if (kv == NULL || data_out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *data_out = NULL;
    if (size_out != NULL) *size_out = 0;
    size_t b = kv_hash(key, kv->nbuckets);
    for (LlmEngineMoeKvEntry* e = kv->buckets[b]; e != NULL; e = e->hnext) {
        if (kv_eq(e->key, key)) {
            kv_touch(kv, e);
            kv->stats.hits++;
            *data_out = e->data;
            if (size_out != NULL) *size_out = e->size;
            return LLMENGINE_MOE_OK;
        }
    }
    kv->stats.misses++;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_kv_insert(LlmEngineMoeKvCache* kv, LlmEngineMoeKvKey key,
                               const void* data, size_t size) {
    if (kv == NULL || data == NULL || size == 0) return LLMENGINE_MOE_ERR_INVALID;
    if (size > kv->budget) return LLMENGINE_MOE_ERR_NO_MEMORY; /* non entrerebbe mai */
    size_t b = kv_hash(key, kv->nbuckets);
    /* sostituzione in place */
    for (LlmEngineMoeKvEntry* e = kv->buckets[b]; e != NULL; e = e->hnext) {
        if (kv_eq(e->key, key)) {
            uint8_t* nd = (uint8_t*)malloc(size);
            if (nd == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
            memcpy(nd, data, size);
            free(e->data);
            kv->used += (uint64_t)size - e->size;
            e->data = nd;
            e->size = size;
            kv->stats.bytes_stored += size;
            kv_touch(kv, e);
            return LLMENGINE_MOE_OK;
        }
    }
    /* evince finche' non c'e' spazio */
    while (kv->used + size > kv->budget && kv->tail != NULL) {
        kv_free_entry(kv, kv->tail);
        kv->stats.evictions++;
    }
    LlmEngineMoeKvEntry* e = (LlmEngineMoeKvEntry*)calloc(1, sizeof(*e));
    if (e == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    e->data = (uint8_t*)malloc(size);
    if (e->data == NULL) { free(e); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    memcpy(e->data, data, size);
    e->key = key;
    e->size = size;
    e->hnext = kv->buckets[b];
    kv->buckets[b] = e;
    e->prev = NULL;
    e->next = kv->head;
    if (kv->head != NULL) kv->head->prev = e;
    kv->head = e;
    if (kv->tail == NULL) kv->tail = e;
    kv->used += size;
    kv->entries++;
    kv->stats.bytes_stored += size;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_kv_remove(LlmEngineMoeKvCache* kv, LlmEngineMoeKvKey key) {
    if (kv == NULL) return LLMENGINE_MOE_ERR_INVALID;
    size_t b = kv_hash(key, kv->nbuckets);
    for (LlmEngineMoeKvEntry* e = kv->buckets[b]; e != NULL; e = e->hnext) {
        if (kv_eq(e->key, key)) { kv_free_entry(kv, e); return LLMENGINE_MOE_OK; }
    }
    return LLMENGINE_MOE_OK; /* assente: no-op */
}

void llmengine_moe_kv_clear(LlmEngineMoeKvCache* kv) {
    if (kv == NULL) return;
    LlmEngineMoeKvEntry* e = kv->head;
    while (e != NULL) {
        LlmEngineMoeKvEntry* n = e->next;
        free(e->data);
        free(e);
        e = n;
    }
    kv->head = kv->tail = NULL;
    kv->used = 0;
    kv->entries = 0;
    memset(kv->buckets, 0, kv->nbuckets * sizeof(*kv->buckets));
}

uint64_t llmengine_moe_kv_bytes_used(const LlmEngineMoeKvCache* kv) { return kv == NULL ? 0 : kv->used; }
uint64_t llmengine_moe_kv_budget(const LlmEngineMoeKvCache* kv) { return kv == NULL ? 0 : kv->budget; }

void llmengine_moe_kv_stats(const LlmEngineMoeKvCache* kv, LlmEngineMoeKvStats* out) {
    if (kv == NULL || out == NULL) return;
    *out = kv->stats;
}
