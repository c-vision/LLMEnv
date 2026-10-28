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
    /* prefill: x 3D [B,S,in] -> 2D [B*S,in]; l'uscita viene riportata a [B,S,out]. */
    const int nd = mlx_array_ndim(x);
    mlx_array xr = mlx_array_new(), orr = mlx_array_new();
    int rc = -1;
    if (nd == 2) rc = llm_mlx_linear_packed_run(&L, x, out, s);
    else if (nd == 3) {
        const int* xs = mlx_array_shape(x);
        int sh2[2] = { xs[0]*xs[1], xs[2] };
        if (mlx_reshape(&xr, x, sh2, 2, s) == 0 && llm_mlx_linear_packed_run(&L, xr, &orr, s) == 0) {
            int osh[3] = { xs[0], xs[1], (int)w->rows };
            rc = mlx_reshape(out, orr, osh, 3, s);
        }
    }
    mlx_array_free(xr); mlx_array_free(orr);
    llm_gemvq4_destroy(L.g);
    return rc == 0 ? 0 : -1;
}

#endif
