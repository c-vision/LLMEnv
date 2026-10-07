/* llm_mlx_xs_pool.h — POOL di buffer riusabili per lo streaming esperti MoE (MLX).
 *
 * Il path streamed costruiva un `malloc` per OGNI (layer,proj,kind) ad OGNI token e lo cedeva a MLX
 * (array managed): ~9 alloc/free per layer per token nel hot path (overhead puro, §0septies). Qui i
 * buffer vivono in un POOL e vengono RICICLATI dal dtor di MLX (vedi
 * `mlx_array_new_data_managed_payload`): zero churn; e il buffer — una volta caldo — resta `mlock`-ato
 * così Metal non lo rifaulta (era il commento "buffer RESIDENTI (mlock)" di stream_build, prima non
 * mantenuto). I buffer oltre il tetto di wire-ata restano malloc normali (sempre corretti).
 *
 * Il dtor di MLX può scattare al release del buffer anche fuori dal forward thread -> `inuse` è
 * atomico (acquire/release). Il forward MoE è single-thread (come le cache `g_tq_full`).
 *
 * Pura libc (malloc/mlock/atomics): testabile senza MLX (tests/test_xs_pool.c). */
#ifndef LLM_MLX_XS_POOL_H
#define LLM_MLX_XS_POOL_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/mman.h>   /* mlock/munlock */

#define LLM_XS_POOL_MAX 64
#define LLM_XS_MLOCK_MAX_BYTES (256ull * 1024ull * 1024ull)

typedef struct { uint8_t* buf; size_t cap; int inuse; int locked; } LlmXsSlot;

static LlmXsSlot g_xs_pool[LLM_XS_POOL_MAX];
static uint64_t g_xs_locked_bytes = 0;

/* dtor per `mlx_array_new_data_managed_payload(buf, ..., slot, llm_xs_slot_dtor)`. */
static inline void llm_xs_slot_dtor(void* payload) {
    if (payload != NULL) __atomic_store_n(&((LlmXsSlot*)payload)->inuse, 0, __ATOMIC_RELEASE);
}

/* Prende uno slot libero con capacità >= `need` (o ne alloca uno), lo marca in uso e ritorna il
 * buffer; `*out_slot` = slot (payload per il dtor) o NULL se si è ricaduti su... no: ritorna il
 * buffer del pool, oppure NULL se il pool è ESAURITO (il chiamante ricade sul malloc singolo). */
static inline uint8_t* llm_xs_pool_acquire(size_t need, LlmXsSlot** out_slot) {
    for (int i = 0; i < LLM_XS_POOL_MAX; i++) {
        LlmXsSlot* sl = &g_xs_pool[i];
        if (__atomic_load_n(&sl->inuse, __ATOMIC_ACQUIRE)) continue;
        if (__atomic_exchange_n(&sl->inuse, 1, __ATOMIC_ACQ_REL)) continue;   /* perso la corsa */
        if (sl->buf != NULL && sl->cap < need) {   /* slot troppo piccolo: scartalo e rialloca */
            if (sl->locked) { munlock(sl->buf, sl->cap); g_xs_locked_bytes -= sl->cap; sl->locked = 0; }
            free(sl->buf); sl->buf = NULL; sl->cap = 0;
        }
        if (sl->buf == NULL) {
            sl->buf = (uint8_t*)malloc(need);
            if (sl->buf == NULL) { __atomic_store_n(&sl->inuse, 0, __ATOMIC_RELEASE); if (out_slot) *out_slot = NULL; return NULL; }
            sl->cap = need;
            if (g_xs_locked_bytes + need <= LLM_XS_MLOCK_MAX_BYTES && mlock(sl->buf, need) == 0) {
                sl->locked = 1; g_xs_locked_bytes += need;
            }
        }
        if (out_slot) *out_slot = sl;
        return sl->buf;
    }
    if (out_slot) *out_slot = NULL;
    return NULL;   /* pool esaurito */
}

#endif /* LLM_MLX_XS_POOL_H */
