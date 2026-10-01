/* ============================================================================
 * llm_mlx_train_abi.c — ABI Apache-2.0 dei PRIMITIVI di training (vedi llmengine_mlx_train.h).
 * L'orchestrazione (loss/grad/DPO/optim) vive nel PUL `gmbllm/train/`.
 * ============================================================================ */
#include "llmengine_mlx_train.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdlib.h>

/* primitivi dal dominio training (non-statici, in llm_mlx_train.c) */
mlx_array llm_mlx_train_ce_impl(LlmMlxModel* m, const int32_t* toks, const int32_t* tgt,
                                const uint8_t* mask, int T, bool sum_mode, bool* ok_out);
bool llm_mlx_train_sdpa_eager_impl(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                   float scale, void* ctx, mlx_stream st);

int llm_ai_mlx_train_setup(llm_ai_mlx_model_handle mh) {
    if (mh == NULL) return -1;
    /* R3.3: l'autograd MLX non ha il VJP per i CustomKernel fusi ne' per mlx_compile. */
    setenv("LLM_BF16_DENSE", "fp32", 1);
    { const char* z[] = { "LLMMETAL_FUSE_SWIGLU_METAL","LLM_FUSE_SWIGLU_COMPILE","LLM_COMPILE_FUSE" };
      for (size_t i = 0; i < sizeof(z)/sizeof(z[0]); i++) setenv(z[i], "0", 1); }
    { const char* off[] = { "LLM_FUSE_ROPE","LLM_FUSE_ROPE_VERIFY","LLM_FUSE_SWIGLU_PREFILL",
        "LLMMETAL_FUSE_GEGLU_METAL","LLM_FUSE_GEGLU","LLM_FUSE_GATEUP","LLM_GU_FUSE",
        "LLMENGINE_MOE_FUSE_MOE_DECODE","LLMENGINE_MOE_FUSE_MOE_DECODE_VERIFY","LLMENGINE_MOE_FUSE_MOE_MXFP4","LLM_GEMVQ4",
        "LLM_GEMVQ4_VERIFY","LLMMETAL_KVMETAL","LLMMETAL_KVMETAL_AUTH","LLMMETAL_KVMETAL_VERIFY",
        "LLMMETAL_GPTOSS_SWIGLU_METAL","LLMENGINE_MOE_GATEUP_FUSE","LLM_PACKED_QKV" };
      for (size_t i = 0; i < sizeof(off)/sizeof(off[0]); i++) unsetenv(off[i]); }
    return 0;
}

int llm_ai_mlx_train_set_lora_mlx(llm_ai_mlx_model_handle mh, int layer, const char* proj,
                                  mlx_array A, mlx_array B, int* out_li) {
    if (mh == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    int li = llm_mlx_model_lora_get(m, layer, proj ? proj : "v_proj");
    if (li < 0) return -1;
    mlx_array_set(&m->lora_A[li], A);
    mlx_array_set(&m->lora_B[li], B);
    m->lora_enabled = true;
    if (out_li) *out_li = li;
    return 0;
}

mlx_array llm_ai_mlx_train_forward_ce(llm_ai_mlx_model_handle mh,
                                      const int32_t* toks, const int32_t* tgt, const uint8_t* mask,
                                      int T, bool sum_mode, bool* ok_out) {
    if (mh == NULL || toks == NULL || tgt == NULL || T <= 0) { if (ok_out) *ok_out = false; return mlx_array_new(); }
    return llm_mlx_train_ce_impl((LlmMlxModel*)mh, toks, tgt, mask, T, sum_mode, ok_out);
}

void llm_ai_mlx_train_set_eager_sdpa(int on) {
    if (on) { llm_train_sdpa_hook.fn = llm_mlx_train_sdpa_eager_impl; llm_train_sdpa_hook.ctx = NULL; }
    else    { llm_train_sdpa_hook.fn = NULL; llm_train_sdpa_hook.ctx = NULL; }
}
