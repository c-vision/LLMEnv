/* llm_dsv4_gpu_ctx.h — contesto GPU DS4 (F4): singleton che lega il backend Metal ai path
 * layer/attenzione/engine. GPU attiva di default. ABI C (implementato in llm_dsv4_gpu_ctx.mm).
 * Gli hook ritornano 1 se hanno eseguito su GPU (0 => il chiamante usa il path C). */
#ifndef LLM_DSV4_GPU_CTX_H
#define LLM_DSV4_GPU_CTX_H

#include "llm_dsv4.h"
#include "llm_dsv4_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Crea backend Metal + view moe/attn. 1 = attivo (refcount++), 0 = fallback. */
int  llm_dsv4_gpu_ctx_init(const LlmDsv4Config* c, const LlmDsv4PackedExperts* exps);
void llm_dsv4_gpu_ctx_reinit(const LlmDsv4Config* c, const LlmDsv4PackedExperts* exps);
/* Rilascia un riferimento (distrugge quando scende a 0). */
void llm_dsv4_gpu_ctx_close(void);
/* 1 se il contesto GPU e' attivo. */
int  llm_dsv4_gpu_ctx_active(void);

/* Hook layer/attenzione (0 => esegui su CPU). */
int llm_dsv4_gpu_moe_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                         const float* x, int token, float* out);
int llm_dsv4_gpu_attn_project_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                  const float* xn, int pos, float* q, float* kv, float* qrn);
int llm_dsv4_gpu_attn_finish_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                 float* heads, int pos, float* out);

#ifdef __cplusplus
}
#endif

#endif /* LLM_DSV4_GPU_CTX_H */
