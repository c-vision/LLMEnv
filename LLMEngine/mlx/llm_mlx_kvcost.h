/* llm_mlx_kvcost.h — §9.A M10a: reconstruction_cost + policy RECOMPUTE/RECONSTRUCT.
 *
 * Un oggetto KV può essere TRATTENUTO (KEEP), PERSISTITO (su disco) o SCARTATO e RICOSTRUITO
 * (ri-prefill). Il paper DeepSeek-V4.1 mostra che conviene "buttare e ricostruire" (KV persistente
 * 1/8 con replay di ~n_win token). Modello di costo, in unità omogenee:
 *   COST_RECOMPUTE = n_tokens * prefill_cost_per_token
 *   COST_PERSIST   = kv_bytes / reload_bw   (costo di ricaricare dal disco)
 *   COST_KEEP      = keep_cost_per_unit * kv_bytes   (costo di tenere residente)
 * decisione = argmin(COST_KEEP, COST_PERSIST, COST_RECOMPUTE).
 */
#ifndef LLM_MLX_KVCOST_H
#define LLM_MLX_KVCOST_H

typedef enum { LLM_KVKEEP = 0, LLM_KVPERSIST = 1, LLM_KVRECOMPUTE = 2 } LlmKvAction;

typedef struct {
    int    action;            /* LLM_KVKEEP | LLM_KVPERSIST | LLM_KVRECOMPUTE */
    double cost_keep;
    double cost_persist;
    double cost_recompute;
} LlmKvCostResult;

/* Decide per un oggetto KV (n_tokens da ri-prefillare, byte del KV, costo/token di prefill,
 * banda di ricarica byte/unità, costo per byte di residenza). `reload_bw<=0` = persist non
 * praticabile (COST_PERSIST=inf). Ritorna l'azione (== out->action). */
int llm_mlx_kvcost_decide(long long n_tokens, double prefill_cost_per_token,
                          double kv_bytes, double reload_bw, double keep_cost_per_unit,
                          LlmKvCostResult* out);

#endif /* LLM_MLX_KVCOST_H */
