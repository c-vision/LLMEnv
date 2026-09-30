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
/* Routing di un chunk su GPU (1 = eseguito): sel [nt*used], wt [nt*used]. */
int llm_dsv4_gpu_route_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il, const float* x, const int* tokens, int nt, int* sel, float* wt);
/* mHC batched su GPU (1 = eseguito): res [nt*HC*D] -> normed [nt*D], sp [nt*size] (pre|post|comb); post: bo,res,sp -> out. */
int llm_dsv4_gpu_hc_pre_batch_try(const LlmDsv4Config* c, const float* fn, const float* scale, const float* base, const float* norm_w,
                                  const float* res, int nt, float* normed, float* sp);
int llm_dsv4_gpu_hc_post_batch_try(const LlmDsv4Config* c, const float* bo, const float* res, const float* sp, int nt, float* out);
/* MoE di un chunk di `nt` token in UNA command buffer (1 = eseguito; 0 = usare il per-token). */
int llm_dsv4_gpu_moe_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                               const float* x, const int* tokens, int nt, float* out);
int llm_dsv4_gpu_attn_project_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                  const float* xn, int pos, float* q, float* kv, float* qrn);
int llm_dsv4_gpu_attn_finish_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                 float* heads, int pos, float* out);
/* Middle attenzione (compressor+indexer+attn) su GPU con KV/stato su device (1 = eseguito).
 * `q`/`kv`/`qrn` sono l'uscita della proiezione; `ratio`/`raw_cap`/`comp_cap` dalla cache. */
int llm_dsv4_gpu_attn_mid_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap,
        const float* q, const float* kv, const float* qrn, const float* xn, int pos, float* heads);
/* Attenzione di un chunk di `nt` token consecutivi in una command buffer per fase (tutto-o-niente, 1 = eseguito). */
int llm_dsv4_gpu_attn_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap, const float* xn, int pos0, int nt, float* out);
/* Tail fuso del layer (fin+mhc+route+moe+mhc) in UNA command buffer (1 = eseguito). */
int llm_dsv4_gpu_attn_tail_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, int il, const float* heads, const float* saved_attn,
        const float* post_attn, const float* comb_attn, int pos, float* out_hc);

#ifdef __cplusplus
}
#endif

#endif /* LLM_DSV4_GPU_CTX_H */
