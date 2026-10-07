/* llm_mlx_dsv4_layer_ops.h — pipeline dei layer DS4 dichiarata dal `.llmarch` (structure `layer`).
 *
 * Stato di COMPOSIZIONE del forward DS4: vive nel motore (non in ds4) cosi' che il wiring
 * del `.llmarch` in llm_mlx_model_io.c NON dipenda dalla cartella dell'engine. Il forward DS4
 * (ds4/llm_dsv4_layer.h) LEGGE questo stato: una struttura nuova = una lista di primitive,
 * senza biforcazioni. `static` per-TU, come prima (comportamento invariato). */
#ifndef LLM_MLX_DSV4_LAYER_OPS_H
#define LLM_MLX_DSV4_LAYER_OPS_H

#include <stdio.h>

#define LLM_DSV4_MAX_OPS 16
static char g_dsv4_layer_ops_buf[LLM_DSV4_MAX_OPS][24];   /* copia: la LlmDesc e' su stack */
static const char* g_dsv4_layer_ops[LLM_DSV4_MAX_OPS];
static int g_dsv4_layer_nops = 0;   /* 0 = usa la sequenza canonica */

/* Il motore imposta la pipeline dichiarata dal `.llmarch` (structure `layer`). */
static inline void llm_dsv4_set_layer_ops(const char* const* ops, int n) {
    if (n < 0) n = 0; if (n > LLM_DSV4_MAX_OPS) n = LLM_DSV4_MAX_OPS;
    for (int i = 0; i < n; i++) {
        snprintf(g_dsv4_layer_ops_buf[i], sizeof g_dsv4_layer_ops_buf[i], "%s", ops[i] ? ops[i] : "");
        g_dsv4_layer_ops[i] = g_dsv4_layer_ops_buf[i];
    }
    g_dsv4_layer_nops = n;
}

#endif /* LLM_MLX_DSV4_LAYER_OPS_H */
