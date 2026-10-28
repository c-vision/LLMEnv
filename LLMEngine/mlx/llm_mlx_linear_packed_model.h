/* llm_mlx_linear_packed_model.h — M3.2: linea packed agganciata al MODELLO (model_dir + opzione).
 *
 * Se `weights_backend == "packed"` e il tensore e' affine 4-bit, esegue la proiezione sui pesi
 * zero-copy (registry -> descrittore -> kernel Q4). Altrimenti ritorna 1 (= "non gestito"):
 * il chiamante prosegue con il percorso MLX normale. Nessun fallback silenzioso. */
#ifndef LLM_MLX_LINEAR_PACKED_MODEL_H
#define LLM_MLX_LINEAR_PACKED_MODEL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_weights_backend.h"
#include "llm_mlx_wpacked_registry.h"
#include "llm_mlx_linear_packed.h"

typedef struct { LlmMlxPackedReg reg; } LlmMlxPackedCtx;

static inline void llm_mlx_packed_ctx_init(LlmMlxPackedCtx* c) { llm_mlx_packed_reg_init(&c->reg); }
static inline void llm_mlx_packed_ctx_free(LlmMlxPackedCtx* c) { llm_mlx_packed_reg_free(&c->reg); }

/* 0 = eseguito (out pronto); 1 = non gestito (usa MLX); <0 errore. */
static inline int llm_mlx_linear_packed_model(LlmMlxPackedCtx* c, const char* model_dir,
                                              const char* tensor_base, int bits, int group,
                                              mlx_array x, mlx_array* out, mlx_stream s) {
    if (!c || !model_dir || !tensor_base || !out) return -1;
    if (!llm_mlx_weights_backend_packed()) return 1;
    const LlmMlxPackedW* w = llm_mlx_packed_reg_get(&c->reg, model_dir, tensor_base, bits, group);
    if (!w) return 1;
    if (w->bits != 4) return 1;   /* v1: solo Q4 */
    LlmMlxPackedLinear L; L.g = NULL; L.w = *w;   /* descrittore gia' mappato dal registry */
    L.g = llm_gemvq4_create();
    if (!L.g) return -1;
    if (llm_gemvq4_load_weights_wonly_nocopy(L.g, (const uint8_t*)w->w, w->scales, w->biases,
                                             w->cols, w->rows, (uint32_t)w->group) != 0) {
        llm_gemvq4_destroy(L.g); return -1;
    }
    int rc = llm_mlx_linear_packed_run(&L, x, out, s);
    llm_gemvq4_destroy(L.g);
    return rc == 0 ? 0 : -1;
}

#endif
