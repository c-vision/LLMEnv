/* llm_mlx_kvclasses.h — §9.A M10d (M10 MEMORY OBJECT): classi KV esplicite.
 *
 * Separa il KV in due CLASSI con policy indipendenti (il paper DeepSeek-V4.1 distingue "always in
 * HBM" da "always on SSD"):
 *   - KV_RUNTIME    : KV vivo in RAM (attivo per il decode corrente);
 *   - KV_PERSISTENT : KV persistito su disco (kvdisk/prefix) — RICOSTRUIBILE/ricaricabile.
 * Manager runtime: i tier KV (prefix cache, kvdisk, pool) REGISTRANO qui i loro byte per classe;
 * `stats` fornisce le due quote; `set_limits`/`over_limit` una policy per-classe (budget).
 * Nessuna conversione silenziosa: la classe e' dichiarata dal produttore del KV.
 */
#ifndef LLM_MLX_KVCLASSES_H
#define LLM_MLX_KVCLASSES_H

#include <stdint.h>

typedef enum { LLM_KVCLASS_RUNTIME = 0, LLM_KVCLASS_PERSISTENT = 1 } LlmKvClass;

/* Registra/aggiorna/rimuovi un oggetto KV per tag (idempotente sul tag). 0 ok, -1 errore. */
int  llm_mlx_kvclasses_register(const char* tag, int cls, uint64_t bytes);
int  llm_mlx_kvclasses_update(const char* tag, uint64_t bytes);
int  llm_mlx_kvclasses_unregister(const char* tag);

/* Quote per classe (byte sommati) + numero di oggetti. */
void llm_mlx_kvclasses_stats(uint64_t* runtime_bytes, uint64_t* persistent_bytes,
                             int* n_runtime, int* n_persistent);

/* Policy per-classe: limiti in byte (<0 = illimitato). over_limit ritorna 1 se superato. */
void llm_mlx_kvclasses_set_limits(int64_t runtime_bytes, int64_t persistent_bytes);
int  llm_mlx_kvclasses_over_limit(int cls);

void llm_mlx_kvclasses_reset(void);

#endif /* LLM_MLX_KVCLASSES_H */
