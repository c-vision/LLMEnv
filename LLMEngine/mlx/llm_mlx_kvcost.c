/* llm_mlx_kvcost.c — §9.A M10a: modello di costo reconstruction_cost. Vedi llm_mlx_kvcost.h. */
#include "llm_mlx_kvcost.h"

#include <math.h>
#include <stddef.h>

int llm_mlx_kvcost_decide(long long n_tokens, double prefill_cost_per_token,
                          double kv_bytes, double reload_bw, double keep_cost_per_unit,
                          LlmKvCostResult* out) {
    double c_recompute = (double)(n_tokens > 0 ? n_tokens : 0) * (prefill_cost_per_token >= 0 ? prefill_cost_per_token : 0.0);
    double c_persist = (reload_bw > 0.0 && kv_bytes >= 0.0) ? kv_bytes / reload_bw : INFINITY;
    double c_keep = (keep_cost_per_unit >= 0.0 && kv_bytes >= 0.0) ? keep_cost_per_unit * kv_bytes : INFINITY;

    int action = LLM_KVRECOMPUTE;
    double best = c_recompute;
    if (c_persist < best) { best = c_persist; action = LLM_KVPERSIST; }
    if (c_keep < best)    { best = c_keep;    action = LLM_KVKEEP; }

    if (out != NULL) {
        out->action = action;
        out->cost_keep = c_keep;
        out->cost_persist = c_persist;
        out->cost_recompute = c_recompute;
    }
    return action;
}
