/* oss/llm_oss_config.c — CONFIG + POLICY del dominio GPT-OSS (tutto in `oss/`).
 *
 * Estratto dai file condivisi (load_config, model_io, session_lifecycle, engine_mlx): la logica
 * gpt-oss non deve più vivere lì — quelli chiamano solo questi thin-hook (come per DS4). */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* llm_ai_mlx_opt_env (registro) */
#include "llm_mlx_internal.h"  /* ModelConfig, LlmMlxModel */
#include "oss/llm_oss.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cJSON.h>

bool llm_oss_is_gptoss(const LlmMlxModel* m) {
    return m != NULL && strncmp(m->cfg.model_type, "gpt_oss", 7) == 0;
}

/* LLM_KVF16_GPTOSS: cache KV fp16 SOLO per i layer FULL di gpt-oss (default OFF).
 * (Estratto da llmengine_mlx.c.) */
static int g_kvf16_gptoss_state = -1;
bool llm_mlx_kv_f16_gptoss_enabled(void) {
    if (g_kvf16_gptoss_state < 0)
        g_kvf16_gptoss_state = (llm_ai_mlx_opt_env("LLM_KVF16_GPTOSS") != NULL) ? 1 : 0;
    return g_kvf16_gptoss_state == 1;
}

/* bf16 PROMOSSO a default per gpt-oss mxfp4 / per-tensor (kill-switch LLM_GPTOSS_BF16=0).
 * (Estratto da llm_mlx_model_io.c.) */
bool llm_oss_want_bf16(const LlmMlxModel* m) {
    const char* e = llm_ai_mlx_opt_env("LLM_GPTOSS_BF16");
    const bool off = (e != NULL && e[0] == '0');
    return m->cfg.per_tensor_quant || (strcmp(m->cfg.quant_mode, "mxfp4") == 0 && !off);
}

/* Prefill chunk di default per gpt-oss (misurato: picco a 2048). (da session_lifecycle.c) */
int llm_oss_default_prefill_chunk(void) { return 2048; }

/* Parsing del branch config `model_type=="gpt_oss"` (sliding_window + layer_types esplicito).
 * (Estratto da llm_mlx_load_config.c; il chiamante fa la cleanup del cJSON su false.) */
bool llm_oss_parse_config(ModelConfig* cfg, cJSON* src) {
    cJSON* sw = cJSON_GetObjectItemCaseSensitive(src, "sliding_window");
    cfg->sliding_window = cJSON_IsNumber(sw) ? (int)sw->valuedouble : 0;
    cJSON* layer_types = cJSON_GetObjectItemCaseSensitive(src, "layer_types");
    if (!cJSON_IsArray(layer_types) || cJSON_GetArraySize(layer_types) != cfg->n_layers) {
        fprintf(stderr, "load_config: layer_types assente o di lunghezza diversa da num_hidden_layers\n");
        return false;
    }
    cfg->is_sliding_layer = (bool*)malloc((size_t)cfg->n_layers * sizeof(bool));
    if (!cfg->is_sliding_layer) return false;
    for (int i = 0; i < cfg->n_layers; i++) {
        cJSON* lt = cJSON_GetArrayItem(layer_types, i);
        if (!cJSON_IsString(lt)) { free(cfg->is_sliding_layer); cfg->is_sliding_layer = NULL; return false; }
        if (strcmp(lt->valuestring, "sliding_attention") == 0 || strcmp(lt->valuestring, "linear_attention") == 0)
            cfg->is_sliding_layer[i] = true;
        else if (strcmp(lt->valuestring, "full_attention") == 0)
            cfg->is_sliding_layer[i] = false;
        else { free(cfg->is_sliding_layer); cfg->is_sliding_layer = NULL; return false; }
    }
    return true;
}
