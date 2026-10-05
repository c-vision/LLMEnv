/* llm_mlx_expolicy.h — §9.A EXPERT_RESIDENCY_POLICY (applicazione LIVE, model-aware).
 *
 * Oggetto POLICY per la residenza degli ESPERTI MoE, ingerito a runtime da JSON (dato PUL,
 * es. dal compile-time `LLM TENSOR EXPOLICY`) e APPLICATO alle opzioni del registro engine
 * (`LLMENGINE_MOE_*` + `offload`), che `llmengine_moe_config_defaults` legge alla creazione
 * dei componenti. Cosi' la policy dichiarata diventa comportamento live (cache budget, modo
 * OFFLOAD/RESIDENT, hotness ROUTING, prefetch distance) invece di restare un report submit-time.
 *
 * Formato JSON:
 *   {"mode":"OFFLOAD"|"RESIDENT","cache_mb":N,"hotness":true|false,
 *    "prefetch_topn":K,"prefetch_cap":C,"async_workers":W}
 * Tutti i campi opzionali; i non presenti non vengono toccati.
 */
#ifndef LLM_MLX_EXPOLICY_H
#define LLM_MLX_EXPOLICY_H

#include <stddef.h>

typedef enum { LLM_MLX_RESIDENCY_OFFLOAD = 0, LLM_MLX_RESIDENCY_RESIDENT = 1 } LlmMlxResidencyMode;

/* Carica la policy da file JSON / stringa. 0 ok, -1 errore. Sostituisce la precedente. */
int  llm_mlx_expolicy_load(const char* path);
int  llm_mlx_expolicy_load_str(const char* json);
void llm_mlx_expolicy_reset(void);
int  llm_mlx_expolicy_active(void);

/* Lettura dei campi (default 0/"" se assenti). */
const char* llm_mlx_expolicy_mode_name(void);   /* "OFFLOAD" | "RESIDENT" | "" */
long llm_mlx_expolicy_cache_mb(void);
int  llm_mlx_expolicy_hotness(void);
int  llm_mlx_expolicy_prefetch_topn(void);

/* Deriva una policy MODEL-AWARE dal budget (byte) + geometria MoE: setta cache_mb (esperti
 * RESIDENT entro il budget), hotness (budget < tutti gli esperti) e prefetch_topn=top_k.
 * Ritorna 0 ok, -1 argomenti invalidi. */
int  llm_mlx_expolicy_derive(long long n_experts, long long top_k, long long expert_bytes, long long budget_bytes);

/* Applica la policy corrente alle opzioni del registro engine (LLMENGINE_MOE_* + offload).
 * Ritorna il numero di opzioni impostate. */
int  llm_mlx_expolicy_apply(void);

#endif /* LLM_MLX_EXPOLICY_H */
