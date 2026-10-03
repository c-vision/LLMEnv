/* llmengine_moe_async.c — worker pool bounded + coda (REQUIRED prioritaria / PREFETCH droppabile) (M5).
 *
 * Ogni worker serve prima la coda REQUIRED (prioritaria), poi la PREFETCH (droppabile). Le load
 * passano dalla cache (single-flight): un esperto richiesto da piu' parti = UNA lettura SSD.
 * Shutdown pulito: flag + broadcast + join di tutti i worker. */
#include "llmengine_moe_async.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef enum { REQ_REQUIRED = 0, REQ_PREFETCH = 1 } ReqType;

typedef struct Req {
    LlmEngineMoeExpertKey key;
    ReqType type;
    size_t expected;
    const void* data;
    size_t size;
    LlmEngineMoeStatus status;
    int handled;
    int cancelled;
    struct Req* next;
} Req;

struct LlmEngineMoeAsync {
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    pthread_t* threads;
    uint32_t n_workers;
    LlmEngineMoeCacheLoadFn load;
    void* ctx;
    LlmEngineMoeExpertCache* cache;
    LlmEngineMoeProfiler* prof;   /* K1: trace CACHE_HIT/MISS per il replay offline */
    uint32_t prefetch_cap, prefetch_n;
    Req* req_head, *req_tail;   /* REQUIRED (FIFO) */
    Req* pre_head, *pre_tail;   /* PREFETCH (FIFO) */
    int shutdown;
    LlmEngineMoeAsyncStats st;
};

static void* worker_main(void* arg);

LlmEngineMoeStatus llmengine_moe_async_create(LlmEngineMoeAsync** out, uint32_t n_workers,
                                  LlmEngineMoeCacheLoadFn load, void* ctx,
                                  LlmEngineMoeExpertCache* cache, uint32_t prefetch_cap) {
    if (out == NULL || load == NULL || cache == NULL || n_workers == 0) return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    LlmEngineMoeAsync* a = (LlmEngineMoeAsync*)calloc(1, sizeof(*a));
    if (a == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    a->n_workers = n_workers; a->load = load; a->ctx = ctx; a->cache = cache;
    a->prefetch_cap = prefetch_cap;
    pthread_mutex_init(&a->mu, NULL);
    pthread_cond_init(&a->cv_work, NULL);
    pthread_cond_init(&a->cv_done, NULL);
    a->threads = (pthread_t*)calloc(n_workers, sizeof(pthread_t));
    if (a->threads == NULL) { llmengine_moe_async_destroy(a); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    for (uint32_t i = 0; i < n_workers; ++i) {
        if (pthread_create(&a->threads[i], NULL, worker_main, a) != 0) {
            a->n_workers = i; /* solo quelli creati vanno joinati */
            llmengine_moe_async_destroy(a);
            return LLMENGINE_MOE_ERR_IO;
        }
    }
    *out = a;
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_async_destroy(LlmEngineMoeAsync* a) {
    if (a == NULL) return;
    pthread_mutex_lock(&a->mu);
    a->shutdown = 1;
    pthread_cond_broadcast(&a->cv_work);
    pthread_mutex_unlock(&a->mu);
    if (a->threads != NULL)
        for (uint32_t i = 0; i < a->n_workers; ++i) pthread_join(a->threads[i], NULL);
    free(a->threads);
    Req* r = a->req_head; while (r) { Req* n = r->next; free(r); r = n; }
    r = a->pre_head; while (r) { Req* n = r->next; free(r); r = n; }
    pthread_mutex_destroy(&a->mu);
    pthread_cond_destroy(&a->cv_work);
    pthread_cond_destroy(&a->cv_done);
    free(a);
}

LlmEngineMoeStatus llmengine_moe_async_get(LlmEngineMoeAsync* a, LlmEngineMoeExpertKey key, size_t expected_size,
                               const void** data, size_t* size) {
    if (a == NULL || data == NULL || size == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *data = NULL; *size = 0;
    Req* r = (Req*)calloc(1, sizeof(*r));
    if (r == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    r->key = key; r->type = REQ_REQUIRED; r->expected = expected_size;
    pthread_mutex_lock(&a->mu);
    if (a->req_tail) a->req_tail->next = r; else a->req_head = r;
    a->req_tail = r;
    a->st.required++;
    pthread_cond_signal(&a->cv_work);
    while (!r->handled) pthread_cond_wait(&a->cv_done, &a->mu);
    LlmEngineMoeStatus s = r->status;
    if (s == LLMENGINE_MOE_OK) { *data = r->data; *size = r->size; }
    pthread_mutex_unlock(&a->mu);
    free(r);
    return s;
}

LlmEngineMoeStatus llmengine_moe_async_prefetch(LlmEngineMoeAsync* a, LlmEngineMoeExpertKey key, size_t expected_size) {
    if (a == NULL) return LLMENGINE_MOE_ERR_INVALID;
    pthread_mutex_lock(&a->mu);
    if (a->shutdown || a->prefetch_cap == 0 || a->prefetch_n >= a->prefetch_cap) {
        a->st.dropped++;
        pthread_mutex_unlock(&a->mu);
        return LLMENGINE_MOE_OK; /* droppata: non e' un errore */
    }
    Req* r = (Req*)calloc(1, sizeof(*r));
    if (r == NULL) { pthread_mutex_unlock(&a->mu); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    r->key = key; r->type = REQ_PREFETCH; r->expected = expected_size;
    if (a->pre_tail) a->pre_tail->next = r; else a->pre_head = r;
    a->pre_tail = r;
    a->prefetch_n++;
    a->st.prefetch++;
    pthread_cond_signal(&a->cv_work);
    pthread_mutex_unlock(&a->mu);
    return LLMENGINE_MOE_OK;
}

uint32_t llmengine_moe_async_cancel_prefetch(LlmEngineMoeAsync* a) {
    if (a == NULL) return 0;
    uint32_t n = 0;
    pthread_mutex_lock(&a->mu);
    for (Req* r = a->pre_head; r; r = r->next) if (!r->cancelled) { r->cancelled = 1; n++; }
    a->st.cancelled += n;
    pthread_mutex_unlock(&a->mu);
    return n;
}

void llmengine_moe_async_set_profiler(LlmEngineMoeAsync* a, LlmEngineMoeProfiler* p) {
    if (a != NULL) a->prof = p;
}

void llmengine_moe_async_stats(LlmEngineMoeAsync* a, LlmEngineMoeAsyncStats* out) {
    if (a == NULL || out == NULL) return;
    pthread_mutex_lock(&a->mu);
    *out = a->st;
    pthread_mutex_unlock(&a->mu);
}

static void* worker_main(void* arg) {
    LlmEngineMoeAsync* a = (LlmEngineMoeAsync*)arg;
    for (;;) {
        pthread_mutex_lock(&a->mu);
        while (!a->shutdown && a->req_head == NULL && a->pre_head == NULL)
            pthread_cond_wait(&a->cv_work, &a->mu);
        if (a->shutdown && a->req_head == NULL && a->pre_head == NULL) { pthread_mutex_unlock(&a->mu); break; }
        Req* r = NULL;
        if (a->req_head != NULL) {                          /* REQUIRED prioritaria */
            r = a->req_head; a->req_head = r->next; if (a->req_head == NULL) a->req_tail = NULL;
        } else {
            r = a->pre_head; a->pre_head = r->next; if (a->pre_head == NULL) a->pre_tail = NULL;
            if (a->prefetch_n > 0) a->prefetch_n--;
        }
        int cancelled = r->cancelled;
        pthread_mutex_unlock(&a->mu);
        if (cancelled) { free(r); continue; }

        const void* data = NULL; size_t size = 0;
        int was_hit = 0;
        LlmEngineMoeStatus s = llmengine_moe_cache_get_ex(a->cache, r->key, r->expected, a->load, a->ctx, &data, &size, &was_hit);
        if (a->prof != NULL) {
            llmengine_moe_prof_event(a->prof,
                was_hit ? LLMENGINE_MOE_TRACE_CACHE_HIT : LLMENGINE_MOE_TRACE_CACHE_MISS,
                r->key.layer, r->key.expert, 0, (uint64_t)size);
        }
        pthread_mutex_lock(&a->mu);
        r->data = data; r->size = size; r->status = s; r->handled = 1;
        a->st.completed++;
        pthread_cond_broadcast(&a->cv_done);
        if (r->type == REQ_PREFETCH) free(r);               /* nessun lettore */
        pthread_mutex_unlock(&a->mu);
    }
    return NULL;
}
