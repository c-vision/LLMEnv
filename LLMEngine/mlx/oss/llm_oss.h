/* ============================================================================
 * oss/llm_oss.h — CONTRATTO del dominio GPT-OSS (motore `oss/`).
 *
 * Tutto ciò che è specifico di GPT-OSS vive in `oss/` e passa da qui. I file condivisi
 * dell'engine NON devono contenere logica gpt-oss: chiamano solo questi simboli (thin-hook),
 * come per DS4 (`ds4/`, ABI `llmengine_llm_engine_ds4`).
 *
 * Contiene: struct di memo pesi per-layer + prototipi del dominio gpt-oss (getter/attn/ffn/
 * flag). Il forward generico (layer_forward) usa `gpt_oss_attention_block`/`gpt_oss_ffn_block`.
 * ========================================================================== */
#ifndef LLM_OSS_H
#define LLM_OSS_H

#include <stdbool.h>
#include "mlx/c/mlx.h"

struct LlmMlxModel;   /* fwd: definito in llm_mlx_internal.h */
typedef struct LlmMlxModel LlmMlxModel;

/* ── Memo nome->array per-layer dei pesi (oss/llm_mlx_gptoss.c) ─────────────────────────────── */
typedef struct { const char* sfx; int sub; mlx_array a; } GptossWE;  /* sub: -1 singolo, 0/1/2 w/s/b */
typedef struct { GptossWE* e; int n; int cap; } GptossLayerWE;

bool llm_mlx_gptoss_bias_prep(LlmMlxModel* m, int layer_idx, int slot, mlx_array src, int dim, mlx_array* out);
bool llm_mlx_gptoss_get_sfx(LlmMlxModel* m, int layer_idx, const char* sfx, mlx_array* out);
bool llm_mlx_gptoss_get_triple_sfx(LlmMlxModel* m, int layer_idx, const char* sfx,
                                   mlx_array* w, mlx_array* s, mlx_array* b);

/* ── Attn/FFN + buffer MoE (oss/llm_mlx_gptoss_ffn.c) ──────────────────────────────────────── */
bool gpt_oss_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool gpt_oss_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
void llm_mlx_pad512_reset(void);
void llm_mlx_gateup_reset(void);

/* ── Flag/opzioni specifiche GPT-OSS (oss/llm_oss_config.c) ────────────────────────────────── */
bool llm_mlx_kv_f16_gptoss_enabled(void);

/* ── Config + policy GPT-OSS (oss/llm_oss_config.c) ────────────────────────────────────────── */
/* true se il modello è GPT-OSS (model_type "gpt_oss"). */
bool llm_oss_is_gptoss(const LlmMlxModel* m);
/* bf16-promotion gpt-oss (mxfp4 / per-tensor). */
bool llm_oss_want_bf16(const LlmMlxModel* m);
/* prefill chunk di default per gpt-oss. */
int  llm_oss_default_prefill_chunk(void);

#endif /* LLM_OSS_H */
