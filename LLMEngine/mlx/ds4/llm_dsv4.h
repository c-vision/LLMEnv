/* llm_dsv4.h — DeepSeek V4 Flash (DS4/DwarfStar) — config del modello e policy per-layer.
 * Port delle costanti `g_ds4_shape` + `ds4_expected_layer_compress_ratio` di `/Users/gaetano/ds4/ds4.c`.
 * Fase 2 del porting DS4. Header-only, C puro (nessuna dipendenza MLX qui).
 * Verifica: llm_dsv4_test.c. */
#ifndef LLM_DSV4_H
#define LLM_DSV4_H

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "llm_gguf.h"

#define LLM_DSV4_MAX_LAYER 64

#define LLM_DSV4_VARIANT_FLASH 0
#define LLM_DSV4_VARIANT_PRO   1

typedef struct {
    const char* name;
    int variant;                 /* LLM_DSV4_VARIANT_* */
    int n_layer, n_embd, n_vocab;
    int n_head, n_head_kv, n_head_dim, n_value_dim;
    int n_rot;                   /* RoPE tail (64) */
    int n_out_group, n_lora_q, n_lora_o;
    int n_expert, n_expert_used, n_expert_shared, n_ff_exp;
    int n_hash_layer;            /* primi N layer con routing hash su token-id (3) */
    int n_swa;                   /* raw sliding window (128) */
    int n_indexer_head, n_indexer_head_dim, n_indexer_top_k;
    int n_hc, n_hc_sinkhorn_iter;
    float rms_eps, hc_eps, expert_weight_scale, swiglu_clamp_exp;
    float rope_freq_base, rope_scale_factor;
    float rope_yarn_beta_fast, rope_yarn_beta_slow, compress_rope_freq_base;
    int64_t rope_orig_ctx;
} LlmDsv4Config;

/* DeepSeek V4 Flash (ds4.c g_ds4_shape, DS4_VARIANT_FLASH). */
static inline LlmDsv4Config llm_dsv4_config_flash(void) {
    LlmDsv4Config c;
    memset(&c, 0, sizeof c);
    c.name = "DeepSeek V4 Flash";
    c.variant = LLM_DSV4_VARIANT_FLASH;
    c.n_layer = 43; c.n_embd = 4096; c.n_vocab = 129280;
    c.n_head = 64; c.n_head_kv = 1; c.n_head_dim = 512; c.n_value_dim = 512;
    c.n_rot = 64;
    c.n_out_group = 8; c.n_lora_q = 1024; c.n_lora_o = 1024;
    c.n_expert = 256; c.n_expert_used = 6; c.n_expert_shared = 1; c.n_ff_exp = 2048;
    c.n_hash_layer = 3;
    c.n_swa = 128;
    c.n_indexer_head = 64; c.n_indexer_head_dim = 128; c.n_indexer_top_k = 512;
    c.n_hc = 4; c.n_hc_sinkhorn_iter = 20;
    c.rms_eps = 1.0e-6f; c.hc_eps = 1.0e-6f;
    c.expert_weight_scale = 1.5f; c.swiglu_clamp_exp = 10.0f;
    c.rope_freq_base = 10000.0f; c.rope_scale_factor = 16.0f;
    c.rope_yarn_beta_fast = 32.0f; c.rope_yarn_beta_slow = 1.0f;
    c.compress_rope_freq_base = 160000.0f;
    c.rope_orig_ctx = 65536;
    return c;
}

/* ds4_expected_layer_compress_ratio: ratio di compressione CSA/HCA per layer.
 * Flash: il<2 -> 0; pari -> 4 (CSA); dispari -> 128 (HCA). Pro: il<2 -> 128. */
static inline int llm_dsv4_layer_compress_ratio(const LlmDsv4Config* c, int il) {
    if (il < 0 || il >= c->n_layer) return -1;
    if (c->variant == LLM_DSV4_VARIANT_PRO) return (il < 2) ? 128 : (((il & 1) == 0) ? 4 : 128);
    return (il < 2) ? 0 : (((il & 1) == 0) ? 4 : 128);
}

/* Validazione: invarianti di shape del V4 Flash. Ritorna 0 (OK) o codice errore. */
#define LLM_DSV4_E_LAYER   -1
#define LLM_DSV4_E_HEADS   -2
#define LLM_DSV4_E_DIM     -3
#define LLM_DSV4_E_ROT     -4
#define LLM_DSV4_E_EXPERTS -5
#define LLM_DSV4_E_HC      -6
static inline int llm_dsv4_validate(const LlmDsv4Config* c) {
    if (c->n_layer <= 0 || c->n_layer > LLM_DSV4_MAX_LAYER) return LLM_DSV4_E_LAYER;
    if (c->n_head <= 0 || c->n_out_group <= 0 || c->n_head % c->n_out_group) return LLM_DSV4_E_HEADS;
    if (c->n_head_dim <= 0 || c->n_head_dim % 32) return LLM_DSV4_E_DIM;   /* packing fp8/fp4 grp32 */
    if (c->n_value_dim <= 0 || c->n_value_dim % 32) return LLM_DSV4_E_DIM;
    if (c->n_rot <= 0 || c->n_rot > c->n_head_dim || (c->n_rot & 1)) return LLM_DSV4_E_ROT;
    if (c->n_expert <= 0 || c->n_expert_used <= 0 || c->n_expert_used > c->n_expert) return LLM_DSV4_E_EXPERTS;
    if (c->n_hc < 2) return LLM_DSV4_E_HC;
    if (c->n_hash_layer < 0 || c->n_hash_layer > c->n_layer) return LLM_DSV4_E_LAYER;
    if (c->n_swa <= 0) return LLM_DSV4_E_LAYER;
    return 0;
}


/* ── Mappa nomi tensori GGUF (ds4 weights_bind_layer) ─────────────────────── */
typedef int (*LlmDsv4NameSink)(void* user, const char* name);

static inline int llm_dsv4_expected_names(const LlmDsv4Config* c, LlmDsv4NameSink sink, void* user) {
    char b[160];
    const char* glob[] = { "token_embd.weight", "output_norm.weight", "output.weight",
                           "output_hc_base.weight", "output_hc_fn.weight", "output_hc_scale.weight" };
    for (size_t i = 0; i < sizeof glob / sizeof glob[0]; i++) if (sink(user, glob[i])) return 1;
    const char* base[] = {
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm", "attn_q_a", "attn_q_a_norm",
        "attn_q_b", "attn_kv", "attn_kv_a_norm", "attn_sinks", "attn_output_a", "attn_output_b",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm", "ffn_gate_inp", "ffn_gate_exps",
        "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
    const char* comp[] = { "attn_compressor_ape", "attn_compressor_kv", "attn_compressor_gate",
                           "attn_compressor_norm" };
    const char* idx[] = { "indexer.attn_q_b", "indexer.proj", "indexer_compressor_ape",
                          "indexer_compressor_kv", "indexer_compressor_gate", "indexer_compressor_norm" };
    for (int il = 0; il < c->n_layer; il++) {
        int ratio = llm_dsv4_layer_compress_ratio(c, il);
        for (size_t i = 0; i < sizeof base / sizeof base[0]; i++) {
            snprintf(b, sizeof b, "blk.%d.%s.weight", il, base[i]);
            if (sink(user, b)) return 1;
        }
        /* il bias esiste solo per i layer non-hash (hash routing usa solo la tabella) */
        if (il >= c->n_hash_layer) {
            snprintf(b, sizeof b, "blk.%d.exp_probs_b.bias", il);
            if (sink(user, b)) return 1;
        }
        if (ratio != 0)
            for (size_t i = 0; i < sizeof comp / sizeof comp[0]; i++) {
                snprintf(b, sizeof b, "blk.%d.%s.weight", il, comp[i]);
                if (sink(user, b)) return 1;
            }
        if (ratio == 4)
            for (size_t i = 0; i < sizeof idx / sizeof idx[0]; i++) {
                snprintf(b, sizeof b, "blk.%d.%s.weight", il, idx[i]);
                if (sink(user, b)) return 1;
            }
        if (il < c->n_hash_layer) {
            snprintf(b, sizeof b, "blk.%d.ffn_gate_tid2eid.weight", il);
            if (sink(user, b)) return 1;
        }
    }
    return 0;
}


/* ── Config dal GGUF (metadati `deepseek4.*` + alias generici) ─────────────── */
static inline bool llm_dsv4_config_from_gguf(const LlmEngineMoeGguf* g, LlmDsv4Config* c) {
    *c = llm_dsv4_config_flash();
#define D4U(field, k, k2) c->field = (int)llm_gguf_u(g, k, (uint64_t)c->field, k2)
    D4U(n_layer, "deepseek4.block_count", "block_count");
    D4U(n_embd, "deepseek4.embedding_length", "embedding_length");
    D4U(n_vocab, "deepseek4.vocab_size", NULL);
    D4U(n_head, "deepseek4.attention.head_count", "attention.head_count");
    D4U(n_head_kv, "deepseek4.attention.head_count_kv", "attention.head_count_kv");
    D4U(n_head_dim, "deepseek4.attention.key_length", "attention.key_length");
    D4U(n_value_dim, "deepseek4.attention.value_length", "attention.value_length");
    D4U(n_rot, "deepseek4.rope.dimension_count", NULL);
    D4U(n_out_group, "deepseek4.attention.output_group_count", NULL);
    D4U(n_lora_q, "deepseek4.attention.q_lora_rank", NULL);
    D4U(n_lora_o, "deepseek4.attention.output_lora_rank", NULL);
    D4U(n_expert, "deepseek4.expert_count", "expert_count");
    D4U(n_expert_used, "deepseek4.expert_used_count", "expert_used_count");
    D4U(n_expert_shared, "deepseek4.expert_shared_count", "expert_shared_count");
    D4U(n_ff_exp, "deepseek4.expert_feed_forward_length", "expert_feed_forward_length");
    D4U(n_hash_layer, "deepseek4.hash_layer_count", NULL);
    D4U(n_swa, "deepseek4.attention.sliding_window", "attention.sliding_window");
    D4U(n_indexer_head, "deepseek4.attention.indexer.head_count", NULL);
    D4U(n_indexer_head_dim, "deepseek4.attention.indexer.key_length", NULL);
    D4U(n_indexer_top_k, "deepseek4.attention.indexer.top_k", NULL);
    D4U(n_hc, "deepseek4.hyper_connection.count", "hyper_connection.count");
    D4U(n_hc_sinkhorn_iter, "deepseek4.hyper_connection.sinkhorn_iterations",
        "hyper_connection.sinkhorn_iterations");
#undef D4U
    c->rms_eps = (float)llm_gguf_f(g, "deepseek4.attention.layer_norm_rms_epsilon",
                                   c->rms_eps, "attention.layer_norm_rms_epsilon");
    c->hc_eps = (float)llm_gguf_f(g, "deepseek4.hyper_connection.epsilon", c->hc_eps,
                                  "hyper_connection.epsilon");
    c->expert_weight_scale = (float)llm_gguf_f(g, "deepseek4.expert_weights_scale",
                                               c->expert_weight_scale, "expert_weights_scale");
    c->compress_rope_freq_base = (float)llm_gguf_f(g, "deepseek4.attention.compress_rope_freq_base",
                                                   c->compress_rope_freq_base, NULL);
    c->rope_freq_base = (float)llm_gguf_f(g, "deepseek4.rope.freq_base", c->rope_freq_base, NULL);
    c->rope_orig_ctx = (int64_t)llm_gguf_u(g, "context_length", (uint64_t)c->rope_orig_ctx, NULL);
    c->variant = LLM_DSV4_VARIANT_FLASH;
    return llm_dsv4_validate(c) == 0;
}

#endif /* LLM_DSV4_H */
