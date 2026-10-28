/* llm_mlx_weights_backend.h — selettore del percorso pesi del motore MLX.
 *
 * "mlx"    (default): i pesi sono array MLX (comportamento storico).
 * "packed" (sperimentale, branch feat/mlx-packed-weights): i pesi quantizzati sono letti
 *          zero-copy dal mmap delle safetensors (weight u32 + scales/biases) e consumati dai
 *          kernel affine-packed, senza materializzare array MLX.
 *
 * Opzione nel REGISTRO (non env): `weights_backend`. Default "mlx": nessun cambiamento di
 * comportamento finche' il percorso packed non passa i gate di parita'/memoria. */
#ifndef LLM_MLX_WEIGHTS_BACKEND_H
#define LLM_MLX_WEIGHTS_BACKEND_H

#include <string.h>
#include "../Options/llm_ai_mlx_options.h"

static inline int llm_mlx_weights_backend_packed(void) {
    char b[16];
    if (llm_ai_mlx_opt_get_str("weights_backend", b, (int)sizeof b) != 0) return 0;
    return strcmp(b, "packed") == 0;
}

#endif
