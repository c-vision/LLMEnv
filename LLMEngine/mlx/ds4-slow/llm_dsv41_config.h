/* llm_dsv41_config.h — parsing di `config.json` (DeepSeek V4.1) in `LlmDsv41Config`, header-only.
 *
 * Gestisce sia il config di inferenza (chiavi dirette: `dim`, `n_layers`, `original_seq_len`, ...)
 * sia il config HF root (con `text_config` annidato e alias transformers: `hidden_size`,
 * `num_hidden_layers`, `qk_rope_head_dim`, `kv_source_layer_ids`, `candidate_source_layer_id`,
 * `engram_pad_token_id`, `num_nextn_predict_layers`, ...). Gli alias sono quelli di
 * oMLX `config.py::from_dict`; verificato contro i due config reali del release.
 *
 * `llm_dsv41_config_validate()` deriva il `LlmDsv41Layout` e chiama `llm_dsv41_validate()`.
 */
#ifndef LLM_DSV41_CONFIG_H
#define LLM_DSV41_CONFIG_H

#include <string.h>
#include <math.h>

#include <cJSON.h>

#include "llm_dsv41_tensors.h"
#include "llm_level3_planner.h"

static inline int llm_dsv41_jnum(cJSON* o, const char* k, double* out) {
    cJSON* it = cJSON_GetObjectItemCaseSensitive(o, k);
    if (cJSON_IsNumber(it)) { *out = it->valuedouble; return 1; }
    if (cJSON_IsBool(it)) { *out = cJSON_IsTrue(it) ? 1.0 : 0.0; return 1; }
    return 0;
}
/* prova `a`, poi `b` (alias). 1 se trovato. */
static inline int llm_dsv41_jfirst(cJSON* o, double* out, const char* a, const char* b) {
    return llm_dsv41_jnum(o, a, out) || (b && llm_dsv41_jnum(o, b, out));
}
static inline void llm_dsv41_seti(int* d, cJSON* o, const char* a, const char* b) {
    double v; if (llm_dsv41_jfirst(o, &v, a, b)) *d = (int)llround(v);
}
static inline void llm_dsv41_setf(float* d, cJSON* o, const char* a, const char* b) {
    double v; if (llm_dsv41_jfirst(o, &v, a, b)) *d = (float)v;
}
static inline void llm_dsv41_setarr(int* d, int* n, int max, cJSON* o, const char* a, const char* b) {
    cJSON* it = cJSON_GetObjectItemCaseSensitive(o, a);
    if (!cJSON_IsArray(it) && b) it = cJSON_GetObjectItemCaseSensitive(o, b);
    if (!cJSON_IsArray(it)) return;
    int k = 0; cJSON* e;
    cJSON_ArrayForEach(e, it) if (k < max) d[k++] = (int)llround(e->valuedouble);
    *n = k;
}
static inline int llm_dsv41_score_from_str(const char* s) {
    if (!s) return LLM_DSV41_SCORE_SQRTSOFTPLUS;
    if (!strcmp(s, "softmax")) return LLM_DSV41_SCORE_SOFTMAX;
    if (!strcmp(s, "sigmoid")) return LLM_DSV41_SCORE_SIGMOID;
    return LLM_DSV41_SCORE_SQRTSOFTPLUS;
}

/* Parsa `json` in `c`. Ritorna true se il JSON e' valido e contiene `dim`/`n_layers`. */
static inline bool llm_dsv41_config_from_json(const char* json, LlmDsv41Config* c) {
    memset(c, 0, sizeof *c);
    /* default scala-indipendenti (config.py) */
    c->gate_temp = 1.0f; c->norm_topk_prob = 1; c->hc_mult = 4; c->hc_sinkhorn_iters = 20;
    c->norm_eps = 1e-20f; c->hc_eps = 1e-6f; c->beta_fast = 32.0f; c->beta_slow = 1.0f;
    c->rope_factor = 1.0f; c->compress_rope_theta = 40000.0f; c->window_size = 128;
    c->score_func = LLM_DSV41_SCORE_SQRTSOFTPLUS; c->candidate_source_layer = -1;
    cJSON* root = cJSON_Parse(json);
    if (!root) return false;
    cJSON* t = cJSON_GetObjectItemCaseSensitive(root, "text_config");
    cJSON* o = cJSON_IsObject(t) ? t : root;

    llm_dsv41_seti(&c->dim, o, "dim", "hidden_size");
    llm_dsv41_seti(&c->moe_inter_dim, o, "moe_inter_dim", "moe_intermediate_size");
    llm_dsv41_seti(&c->n_layers, o, "n_layers", "num_hidden_layers");
    llm_dsv41_seti(&c->n_heads, o, "n_heads", "num_attention_heads");
    llm_dsv41_seti(&c->head_dim, o, "head_dim", NULL);
    llm_dsv41_seti(&c->rope_head_dim, o, "rope_head_dim", "qk_rope_head_dim");
    llm_dsv41_seti(&c->q_lora_rank, o, "q_lora_rank", NULL);
    llm_dsv41_seti(&c->o_lora_rank, o, "o_lora_rank", NULL);
    llm_dsv41_seti(&c->o_groups, o, "o_groups", NULL);
    llm_dsv41_seti(&c->n_routed_experts, o, "n_routed_experts", NULL);
    llm_dsv41_seti(&c->n_shared_experts, o, "n_shared_experts", NULL);
    llm_dsv41_seti(&c->n_activated_experts, o, "n_activated_experts", "num_experts_per_tok");
    llm_dsv41_setf(&c->route_scale, o, "route_scale", "routed_scaling_factor");
    llm_dsv41_setf(&c->swiglu_limit, o, "swiglu_limit", NULL);
    llm_dsv41_setf(&c->norm_eps, o, "norm_eps", "rms_norm_eps");
    llm_dsv41_seti(&c->window_size, o, "window_size", "sliding_window");
    llm_dsv41_seti(&c->vocab_size, o, "vocab_size", NULL);
    llm_dsv41_seti(&c->hc_mult, o, "hc_mult", NULL);
    llm_dsv41_seti(&c->hc_sinkhorn_iters, o, "hc_sinkhorn_iters", NULL);
    llm_dsv41_setf(&c->hc_eps, o, "hc_eps", NULL);
    llm_dsv41_setf(&c->rope_theta, o, "rope_theta", NULL);
    llm_dsv41_setf(&c->compress_rope_theta, o, "compress_rope_theta", NULL);
    llm_dsv41_seti(&c->index_n_heads, o, "index_n_heads", NULL);
    llm_dsv41_seti(&c->index_head_dim, o, "index_head_dim", NULL);
    llm_dsv41_seti(&c->index_topk, o, "index_topk", NULL);
    llm_dsv41_seti(&c->candidate_source_layer, o, "candidate_source_layer", "candidate_source_layer_id");
    llm_dsv41_seti(&c->candidate_topk_blocks, o, "candidate_topk_blocks", NULL);
    llm_dsv41_seti(&c->candidate_block_size, o, "candidate_block_size", NULL);
    llm_dsv41_seti(&c->max_seq_len, o, "max_seq_len", "max_position_embeddings");
    llm_dsv41_seti(&c->n_mtp_layers, o, "n_mtp_layers", "num_nextn_predict_layers");
    llm_dsv41_seti(&c->engram_max_ngram_size, o, "engram_max_ngram_size", NULL);
    llm_dsv41_seti(&c->engram_n_heads, o, "engram_n_heads", NULL);
    llm_dsv41_seti(&c->engram_head_dim, o, "engram_head_dim", NULL);
    llm_dsv41_seti(&c->engram_vocab_size, o, "engram_vocab_size", NULL);
    llm_dsv41_seti(&c->engram_compressed_vocab_size, o, "engram_compressed_vocab_size", NULL);
    llm_dsv41_seti(&c->engram_pad_id, o, "engram_pad_id", "engram_pad_token_id");
    llm_dsv41_seti(&c->dspark_block_size, o, "dspark_block_size", NULL);
    llm_dsv41_seti(&c->dspark_n_routed_experts, o, "dspark_n_routed_experts", NULL);
    llm_dsv41_seti(&c->dspark_n_activated_experts, o, "dspark_n_activated_experts", "dspark_num_experts_per_tok");
    llm_dsv41_seti(&c->dspark_noise_token_id, o, "dspark_noise_token_id", NULL);
    llm_dsv41_seti(&c->dspark_markov_rank, o, "dspark_markov_rank", NULL);
    llm_dsv41_setarr(c->dspark_target_layer_ids, &c->n_dspark_targets, LLM_DSV41_MAX_LAYERS,
                     o, "dspark_target_layer_ids", "mtp_target_layer_ids");
    cJSON* sf = cJSON_GetObjectItemCaseSensitive(o, "score_func");
    if (!cJSON_IsString(sf)) sf = cJSON_GetObjectItemCaseSensitive(o, "scoring_func");
    if (cJSON_IsString(sf)) c->score_func = llm_dsv41_score_from_str(sf->valuestring);

    int dummy;
    llm_dsv41_setarr(c->compress_ratios, &dummy, LLM_DSV41_MAX_LAYERS, o, "compress_ratios", NULL);
    llm_dsv41_setarr(c->engram_layer_ids, &c->n_engram_layers, LLM_DSV41_MAX_LAYERS, o, "engram_layer_ids", NULL);
    int en;
    llm_dsv41_setarr(c->engram_num_embeddings, &en, LLM_DSV41_MAX_LAYERS, o, "engram_num_embeddings", NULL);
    int kvn, ixn;
    int kv[LLM_DSV41_MAX_LAYERS], ix[LLM_DSV41_MAX_LAYERS];
    llm_dsv41_setarr(kv, &kvn, LLM_DSV41_MAX_LAYERS, o, "kv_source_layers", "kv_source_layer_ids");
    llm_dsv41_setarr(ix, &ixn, LLM_DSV41_MAX_LAYERS, o, "index_source_layers", "index_source_layer_ids");
    for (int i = 0; i < kvn; i++) if (kv[i] < LLM_DSV41_MAX_LAYERS) c->in_kv_source[kv[i]] = 1;
    for (int i = 0; i < ixn; i++) if (ix[i] < LLM_DSV41_MAX_LAYERS) c->in_index_source[ix[i]] = 1;

    /* RoPE: YaRN da `rope_scaling` (root) o campi diretti (inference). */
    cJSON* rs = cJSON_GetObjectItemCaseSensitive(o, "rope_scaling");
    if (cJSON_IsObject(rs)) {
        llm_dsv41_setf(&c->rope_factor, rs, "factor", NULL);
        llm_dsv41_seti(&c->original_seq_len, rs, "original_max_position_embeddings", NULL);
        llm_dsv41_setf(&c->beta_fast, rs, "beta_fast", NULL);
        llm_dsv41_setf(&c->beta_slow, rs, "beta_slow", NULL);
    } else {
        llm_dsv41_seti(&c->original_seq_len, o, "original_seq_len", NULL);
        llm_dsv41_setf(&c->rope_factor, o, "rope_factor", NULL);
        llm_dsv41_setf(&c->beta_fast, o, "beta_fast", NULL);
        llm_dsv41_setf(&c->beta_slow, o, "beta_slow", NULL);
    }
    /* vision: `vision_config.num_hidden_layers` (root) o `vision_n_layers` diretto. */
    cJSON* vc = cJSON_GetObjectItemCaseSensitive(root, "vision_config");
    if (cJSON_IsObject(vc)) llm_dsv41_seti(&c->vision_n_layers, vc, "num_hidden_layers", NULL);
    else llm_dsv41_seti(&c->vision_n_layers, o, "vision_n_layers", NULL);
    cJSON* im = cJSON_GetObjectItemCaseSensitive(root, "image_token_id");
    if (!cJSON_IsNumber(im)) im = cJSON_GetObjectItemCaseSensitive(o, "image_token_id");
    if (cJSON_IsNumber(im)) c->image_token_id = (int)llround(im->valuedouble);

    bool ok = (c->dim > 0 && c->n_layers > 0);
    cJSON_Delete(root);
    return ok;
}

/* Deriva il `LlmDsv41Layout` (per validate()/CED). */
static inline void llm_dsv41_config_layout(const LlmDsv41Config* c, LlmDsv41Layout* L) {
    memset(L, 0, sizeof *L);
    L->model_type = "deepseek_v41";
    L->n_layers = c->n_layers; L->n_heads = c->n_heads; L->head_dim = c->head_dim;
    L->o_groups = c->o_groups; L->rope_head_dim = c->rope_head_dim;
    L->index_head_dim = c->index_head_dim; L->n_shared_experts = c->n_shared_experts;
    L->n_routed_experts = c->n_routed_experts; L->n_activated_experts = c->n_activated_experts;
    L->window_size = c->window_size; L->ced_prefill = c->ced_prefill;
    L->compress_ratio_n = c->n_layers;
    for (int i = 0; i < c->n_layers && i < 64; i++) L->compress_ratios[i] = c->compress_ratios[i];
    for (int i = 0; i < c->n_layers; i++) {
        if (c->in_kv_source[i] && L->kv_source_n < 16) L->kv_source_layers[L->kv_source_n++] = i;
        if (c->in_index_source[i] && L->index_source_n < 16) L->index_source_layers[L->index_source_n++] = i;
    }
    L->engram_layer_n = c->n_engram_layers;
    L->engram_num_embeddings_n = c->n_engram_layers;
    for (int i = 0; i < c->n_engram_layers && i < 16; i++) {
        L->engram_layer_ids[i] = c->engram_layer_ids[i];
        L->engram_num_embeddings[i] = c->engram_num_embeddings[i];
    }
}

/* Ritorna LLM_DSV41_OK o il codice d'errore di `llm_dsv41_validate()`. */
static inline int llm_dsv41_config_validate(const LlmDsv41Config* c) {
    LlmDsv41Layout L;
    llm_dsv41_config_layout(c, &L);
    return llm_dsv41_validate(&L);
}

#endif /* LLM_DSV41_CONFIG_H */
