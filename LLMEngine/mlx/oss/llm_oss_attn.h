/* oss/llm_oss_attn.h — helper di ATTENZIONE specifici GPT-OSS (sinks, kv-f16 per-layer,
 * SDPA finestra+sinks, o_proj bias). Usati SIA dal path per-sessione
 * (oss/llm_mlx_gptoss_ffn.c) SIA dal path batch (llm_mlx_attn_batch.c): la logica gpt-oss
 * resta UNICA e in `oss/` (i file condivisi fanno solo thin-call).
 *
 * Richiede il contesto condiviso (LlmMlxModel completo, tname/multishard_get/mlx_*,
 * llm_mlx_gqa_sdpa_sinks): va incluso DOPO llm_mlx_internal.h. */
#ifndef LLM_OSS_ATTN_H
#define LLM_OSS_ATTN_H

#include "oss/llm_oss.h"
#include <string.h>
#include <math.h>

/* Sinks appresi per-head (self_attn.sinks). Il multishard può restituire una VIEW a stride ->
 * copia contigua. Ritorna false se il tensore è assente. */
static inline bool llm_oss_load_sinks(LlmMlxModel* m, int layer_idx, mlx_stream st, mlx_array* sinks) {
    char snb[256];
    mlx_array sk = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.sinks", snb, sizeof(snb));
    bool ok = multishard_get(&m->weights, snb, &sk);
    if (ok) {
        mlx_array sc = mlx_array_new();
        ok = mlx_contiguous(&sc, sk, /*allow_col_major=*/false, st) == 0;
        if (ok) mlx_array_set(sinks, sc);
        mlx_array_free(sc);
    }
    mlx_array_free(sk);
    return ok;
}

/* kv-f16 SOLO per i layer FULL (i layer sliding restano nel dtype di attivazione). */
static inline bool llm_oss_kv_f16_layer(const LlmMlxModel* m, int layer_idx) {
    return llm_mlx_kv_f16_gptoss_enabled()
        && !(m->cfg.is_sliding_layer[layer_idx] && m->cfg.sliding_window > 0);
}

/* SDPA gpt-oss: finestra sliding (se layer sliding) + sinks appresi. */
static inline bool llm_oss_attn_sdpa(mlx_array* out, const LlmMlxModel* m, int layer_idx,
                                     mlx_array qr, mlx_array k, mlx_array v, int hd,
                                     mlx_array sinks, mlx_stream st) {
    int win = (m->cfg.is_sliding_layer[layer_idx] && m->cfg.sliding_window > 0) ? m->cfg.sliding_window : 0;
    return llm_mlx_gqa_sdpa_sinks(out, qr, k, v, 1.0f / sqrtf((float)hd), win, sinks, st);
}

/* o_proj gpt-oss: bias lineare per-layer (self_attn.o_proj.bias) sommato all'uscita. */
static inline bool llm_oss_oproj_bias(LlmMlxModel* m, int layer_idx, mlx_array attn_prj,
                                      mlx_array* attn_out, mlx_stream st) {
    char obb[256];
    mlx_array o_lb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj.bias", obb, sizeof(obb));
    bool ok = multishard_get(&m->weights, obb, &o_lb);
    if (ok) ok = mlx_add(attn_out, attn_prj, o_lb, st) == 0;
    mlx_array_free(o_lb);
    return ok;
}

#endif /* LLM_OSS_ATTN_H */
