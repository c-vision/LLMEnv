/* llm_mlx_mobj.h — §9.A M10: MEMORY OBJECT runtime manager (unificato).
 *
 * Un solo ente per TUTTI gli oggetti-memoria del runtime (pesi/esperti/KV/prefix/temp): ogni
 * oggetto ha categoria, residenza (RESIDENT/SWAP/DISK), byte e lifetime; il manager somma le
 * quote per categoria, applica un BUDGET globale e una PRIORITÀ di eviction per categoria (K3:
 * temp/prefetch per primi, il trunk/pesi per ultimi). I tier esistenti (KV di sessione, prefix
 * cache, cache esperti) REGISTRANO qui i loro oggetti → vista unificata + policy.
 */
#ifndef LLM_MLX_MOBJ_H
#define LLM_MLX_MOBJ_H

#include <stddef.h>

typedef enum {
    LLM_MOBJ_WEIGHTS = 0, LLM_MOBJ_EXPERT_RAM, LLM_MOBJ_EXPERT_DEVICE,
    LLM_MOBJ_KV_RUNTIME, LLM_MOBJ_KV_PERSISTENT, LLM_MOBJ_PREFIX, LLM_MOBJ_TEMP, LLM_MOBJ_NCAT
} LlmMobjCat;

typedef enum { LLM_MOBJ_RESIDENT = 0, LLM_MOBJ_SWAP = 1, LLM_MOBJ_DISK = 2 } LlmMobjRes;

void llm_mlx_mobj_reset(void);
/* Registra/aggiorna/rimuovi un oggetto per tag (idempotente). 0 / -1. */
int  llm_mlx_mobj_register(const char* tag, int cat, int res, unsigned long long bytes);
int  llm_mlx_mobj_update(const char* tag, unsigned long long bytes);
int  llm_mlx_mobj_unregister(const char* tag);
int  llm_mlx_mobj_count(void);

void llm_mlx_mobj_stats(unsigned long long* total_bytes, int* n_objects);
void llm_mlx_mobj_stats_cat(int cat, unsigned long long* bytes, int* n);

/* Budget globale: 1 = superato. */
void llm_mlx_mobj_set_budget(unsigned long long bytes);
int  llm_mlx_mobj_over_budget(void);

/* Priorità di eviction per categoria (più BASSO = evincibile prima; K3). */
int  llm_mlx_mobj_evict_priority(int cat);
const char* llm_mlx_mobj_cat_name(int cat);

#endif /* LLM_MLX_MOBJ_H */
