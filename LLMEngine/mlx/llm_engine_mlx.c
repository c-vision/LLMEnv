/* llm_engine_mlx.c — adapter del motore MLX all'ABI comune `llmengine`.
 * Delega 1:1 alle funzioni native `llm_ai_mlx_*` (nessuna logica duplicata). MLX lavora a
 * token-id: caps = IDS (la tokenizzazione sta a monte, in C#). */
#include "llmengine.h"
#include "llmengine_mlx.h"

static LlmEngineLlmModel mlx_load(const char* path) { return (LlmEngineLlmModel)llm_ai_mlx_load_model(path); }
static void        mlx_free(LlmEngineLlmModel m)    { llm_ai_mlx_free_model((llm_ai_mlx_model_handle)m); }
static LlmEngineLlmSession mlx_sess(LlmEngineLlmModel m)  { return (LlmEngineLlmSession)llm_ai_mlx_create_session((llm_ai_mlx_model_handle)m); }
static void        mlx_dsess(LlmEngineLlmSession s) { llm_ai_mlx_destroy_session((llm_ai_mlx_session_handle)s); }

static int mlx_append_ids(LlmEngineLlmSession s, const int32_t* ids, int n) {
    return llm_ai_mlx_session_append((llm_ai_mlx_session_handle)s, ids, n);
}
static int mlx_generate_ids(LlmEngineLlmSession s, int max_new, int32_t eos, int32_t* out) {
    return llm_ai_mlx_session_generate((llm_ai_mlx_session_handle)s, max_new, eos, out);
}
static int mlx_set_sampling(LlmEngineLlmSession s, const LlmEngineLlmSampling* sp) {
    if (sp == NULL) return -1;
    llm_ai_mlx_session_handle h = (llm_ai_mlx_session_handle)s;
    llm_ai_mlx_session_set_sampling(h, sp->temperature, sp->top_p, sp->top_k, sp->seed);
    llm_ai_mlx_session_set_repetition_penalty(h, sp->repetition_penalty);
    llm_ai_mlx_session_set_frequency_penalty(h, sp->frequency_penalty);
    llm_ai_mlx_session_set_presence_penalty(h, sp->presence_penalty);
    return 0;
}
static void mlx_free_str(char* p) { llm_ai_mlx_free_str(p); }

/* ABI v2 — LOGPROBS: delega 1:1 ai path nativi (distribuzione alla posizione pending). */
static int mlx_token_logprob(LlmEngineLlmSession s, int32_t token_id, float* out_logprob) {
    return llm_ai_mlx_session_logprob_of((llm_ai_mlx_session_handle)s, token_id, out_logprob);
}
static int mlx_last_logprob(LlmEngineLlmSession s, float* out_logprob) {
    return llm_ai_mlx_session_last_logprob((llm_ai_mlx_session_handle)s, out_logprob);
}
static int mlx_generate_ids_probs(LlmEngineLlmSession s, int max_new, int32_t eos,
                                  int32_t* out_ids, float* out_logprobs) {
    return llm_ai_mlx_session_generate_probs((llm_ai_mlx_session_handle)s, max_new, eos,
                                             out_ids, out_logprobs);
}

/* ABI v2 — CONSTRAINED: un token con argmax ristretto (nativo). */
static int mlx_generate_id_constrained(LlmEngineLlmSession s, const int32_t* allowed_ids, int n_allowed,
                                       int32_t* out_token_id) {
    return llm_ai_mlx_session_generate_constrained((llm_ai_mlx_session_handle)s, allowed_ids,
                                                   n_allowed, out_token_id);
}

/* ABI v3 — TOPK: distribuzione top-k alla posizione pending (nativo). */
static int mlx_token_topk(LlmEngineLlmSession s, int k, int32_t* out_ids, float* out_logprobs) {
    return llm_ai_mlx_session_topk((llm_ai_mlx_session_handle)s, k, out_ids, out_logprobs);
}

/* ABI v4 — EMBED IDS: mean-pool IDS-based (nativo, senza sessione del chiamante). */
static int mlx_embed_dim(LlmEngineLlmModel m) {
    return llm_ai_mlx_embed_dim((llm_ai_mlx_model_handle)m);
}
static int mlx_embed_ids(LlmEngineLlmModel m, const int32_t* ids, int n, float* out, int cap) {
    return llm_ai_mlx_embed_ids((llm_ai_mlx_model_handle)m, ids, n, out, cap);
}

/* ABI v5 — EXPERT SWAP: offload esperti da disco (scelta del consumer, nessun env). */
static void mlx_set_expert_swap(const char* dir) { llm_ai_mlx_set_moe_swap_dir(dir); }

/* ABI v2 — KV: reset in-place + stats. */
static int mlx_kv_clear(LlmEngineLlmSession s) {
    return llm_ai_mlx_session_reset((llm_ai_mlx_session_handle)s);
}
static int mlx_kv_stats(LlmEngineLlmSession s, int* out_used, int* out_cap) {
    return llm_ai_mlx_session_kv_stats((llm_ai_mlx_session_handle)s, out_used, out_cap);
}

/* Estensioni opzionali (capacità MLX-specifiche): prefill a chunk, prefix cache radix, batching. */
static int mlx_set_prefill_chunk(LlmEngineLlmSession s, int chunk) {
    return llm_ai_mlx_session_set_prefill_chunk((llm_ai_mlx_session_handle)s, chunk);
}
static int mlx_append_ids_prefix(LlmEngineLlmModel m, LlmEngineLlmSession s, const int32_t* ids, int n, int min_match) {
    return llm_ai_mlx_session_append_auto_prefix((llm_ai_mlx_session_handle)s, (llm_ai_mlx_model_handle)m, ids, n, min_match);
}
static int mlx_save_prefix(LlmEngineLlmModel m, const char* name, LlmEngineLlmSession s) {
    return llm_ai_mlx_model_save_prefix((llm_ai_mlx_model_handle)m, name, (llm_ai_mlx_session_handle)s);
}
static int mlx_generate_batch(LlmEngineLlmSession* sess, int n, int max_new, int32_t eos, int32_t* out) {
    return llm_ai_mlx_session_generate_batch((llm_ai_mlx_session_handle*)sess, n, max_new, eos, out);
}


/* CAP_KV_MAX: imposta la capienza della KV-cache (contesto) a runtime, per qualunque modello. */
static int mlx_kv_set_max(LlmEngineLlmModel m, int n) {
    return llm_ai_mlx_model_set_kv_max((llm_ai_mlx_model_handle)m, n);
}

static const LlmEngineLlmEngine LLM_MLX_ENGINE = {
    .name = "mlx",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_PREFIX | LLMMODELIO_CAP_BATCH |
            LLMMODELIO_CAP_CHUNK | LLMMODELIO_CAP_LORA | LLMMODELIO_CAP_KV |
            LLMMODELIO_CAP_LOGPROBS | LLMMODELIO_CAP_CONSTRAINED | LLMMODELIO_CAP_EMBED |
            LLMMODELIO_CAP_KV_MAX,
    .load_model = mlx_load, .free_model = mlx_free,
    .create_session = mlx_sess, .destroy_session = mlx_dsess,
    .append_ids = mlx_append_ids, .append_text = NULL,
    .generate_ids = mlx_generate_ids, .generate_text = NULL,
    .set_sampling = mlx_set_sampling,
    .kv_clear = mlx_kv_clear, .kv_stats = mlx_kv_stats,
    .tokenize = NULL, .detokenize = NULL,
    .set_prefill_chunk = mlx_set_prefill_chunk,
    .append_ids_prefix = mlx_append_ids_prefix,
    .save_prefix = mlx_save_prefix,
    .generate_batch = mlx_generate_batch,
    /* CAP_KV_MAX: contesto/KV a runtime (generico, tutti i modelli MLX). */
    .kv_set_max = mlx_kv_set_max,
    .emb_dim = mlx_embed_dim, .embed_text = NULL,
    .vision_load = NULL, .vision_read_image = NULL, .vision_read_audio = NULL, .vision_unload = NULL,
    .free_str = mlx_free_str,
    /* ABI v2 (in coda): logprobs + constrained. */
    .token_logprob = mlx_token_logprob,
    .last_logprob = mlx_last_logprob,
    .generate_ids_probs = mlx_generate_ids_probs,
    .generate_id_constrained = mlx_generate_id_constrained,
    /* ABI v3 (in coda): top-k della distribuzione pending. */
    .token_topk = mlx_token_topk,
    /* ABI v4 (in coda): embedding IDS-based. */
    .embed_ids = mlx_embed_ids,
    /* ABI v5 (in coda): offload esperti da disco. */
    .set_expert_swap = mlx_set_expert_swap,
};

const LlmEngineLlmEngine* llmengine_llm_engine_mlx(void) { return &LLM_MLX_ENGINE; }

unsigned llmengine_abi_version(void) { return LLMENGINE_ABI_VERSION; }
