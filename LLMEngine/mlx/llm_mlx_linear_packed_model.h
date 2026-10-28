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
int llm_mlx_qp_name_get(const char* name, int* group, int* bits);

typedef struct { LlmMlxPackedReg reg; } LlmMlxPackedCtx;

static inline void llm_mlx_packed_ctx_init(LlmMlxPackedCtx* c) { llm_mlx_packed_reg_init(&c->reg); }
static inline void llm_mlx_packed_ctx_free(LlmMlxPackedCtx* c) { llm_mlx_packed_reg_free(&c->reg); }

/* 0 = eseguito (out pronto); 1 = non gestito (usa MLX); <0 errore. */
static inline int llm_mlx_linear_packed_model(LlmMlxPackedCtx* c, const char* model_dir,
                                              const char* tensor_base, int bits, int group,
                                              mlx_array x, mlx_array* out, mlx_stream s) {
    if (!c || !model_dir || !tensor_base || !out) return -1;
    if (!llm_mlx_weights_backend_packed()) return 1;
    { int qg=0,qb=0; if (llm_mlx_qp_name_get(tensor_base,&qg,&qb) && qb>0) { bits=qb; if(qg>0) group=qg; } }
    LlmMlxPackedEntry* e = llm_mlx_packed_reg_entry(&c->reg, model_dir, tensor_base, bits, group);
    if (!e) return 1;
    const LlmMlxPackedW* w = &e->w;
    if (w->bits != 4) return 1;   /* v1: solo Q4 */
    if (llm_mlx_lazy_build(&e->lazy, w) != 0) return 1;
    /* nodo lazy del grafo: nessun eval, nessuna attesa (il gemv e' un mlx_fast_metal_kernel su viste zero-copy) */
    return llm_mlx_lazy_linear(&e->lazy, w, x, out, s) == 0 ? 0 : -1;
}

int llm_mlx_qp_name_get(const char* name, int* group, int* bits);
#endif
