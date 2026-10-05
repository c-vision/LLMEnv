/* llm_mlx_kvdag.h — §9.A M10b: il KV come DAG (KV_BASE · KV_DERIVED · KV_INDEX).
 *
 * In CSA2 un layer può RIUSARE il KV di un altro (FULL/REINDEX/REUSE): un oggetto memoria non è
 * "il KV del layer X" ma un NODO con producer/consumers/dipendenze. Il PREFETCH di un oggetto si
 * ESPANDE in modo topologico: READ (base) | REUSE (derivato condiviso) | RECONSTRUCT (reindex) |
 * COMPUTE. Questo modulo modella il grafo e l'espansione (declarativo, nessun modello).
 */
#ifndef LLM_MLX_KVDAG_H
#define LLM_MLX_KVDAG_H

#include <stddef.h>

#define LLM_KVDAG_MAX 64

typedef enum { LLM_KVNODE_BASE = 0, LLM_KVNODE_DERIVED = 1, LLM_KVNODE_INDEX = 2 } LlmKvNodeKind;

void llm_mlx_kvdag_reset(void);
/* Aggiunge un nodo (id univoco). kind: BASE/DERIVED/INDEX. `reconstructable`: ricostruibile. 0/-1. */
int  llm_mlx_kvdag_add(const char* id, int kind, int reconstructable);
/* Dipendenza `id` dipende da `dep` (id già aggiunti). 0/-1. */
int  llm_mlx_kvdag_dep(const char* id, const char* dep);
int  llm_mlx_kvdag_count(void);

/* Espande il PREFETCH di `id`: ordine TOPOLOGICO delle op (dipendenze prima), come
 * "READ a,REUSE b,RECONSTRUCT c,..." in `out`. Ritorna il n. di op, -1 se nodo assente. */
int  llm_mlx_kvdag_prefetch(const char* id, char* out, size_t cap);

/* Costruisce un grafo CSA2 rappresentativo (7 nodi: base/derived/index/... ) e ritorna
 * l'espansione PREFETCH della radice. Deterministico (per il test/superficie). */
int  llm_mlx_kvdag_demo_csa2(char* out, size_t cap);

#endif /* LLM_MLX_KVDAG_H */
