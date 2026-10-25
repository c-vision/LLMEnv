/* qwen4_native_gguf.h — P7e: adapter qwen4exp sopra il loader GGUF esistente.
 * NON c'e' un nuovo parser: si usano llm_gguf.h (parse/mmap/dequant) e
 * ds4/llm_dsv4_weights.h (accesso per nome + packed). Qui solo:
 *  - lettura hparam `qwen4exp.*` (chiavi gia' emesse dal packer ds4);
 *  - role map dei suffissi per layer (lineare/GDN, full+indexer, PLE, MoE).
 */
#ifndef LLM_QWEN4_NATIVE_GGUF_H
#define LLM_QWEN4_NATIVE_GGUF_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm_gguf.h"
#include "ds4/llm_dsv4_weights.h"

typedef struct {
    int E, L, H, Hkv, D, Hi, Di, topk, NE, K, F, Fsh, hc, rank;
    int lin_conv, lin_Hk, lin_Hv, lin_D;
    int ple_ng, ple_heads, ple_conv, ple_eos, n_ple_layers;
    int n_rot, full_interval, nextn;
    float eps, rope_base;
} Q4nHp;

static int q4n_hparams(const LlmEngineMoeGguf* g, Q4nHp* hp, char* err, size_t errn) {
    if (!g || !hp) return -1;
    const LlmGgufVal* a = llm_gguf_find_val(g, "general.architecture");
    if (!a || !a->s || strcmp(a->s, "qwen4exp") != 0) {
        snprintf(err, errn, "arch non qwen4exp: %s", (a && a->s) ? a->s : "?");
        return -1;
    }
    memset(hp, 0, sizeof *hp);
    hp->E = (int)llm_gguf_u(g, "qwen4exp.embedding_length", 0, NULL);
    hp->L = (int)llm_gguf_u(g, "qwen4exp.block_count", 0, NULL);
    hp->H = (int)llm_gguf_u(g, "qwen4exp.attention.head_count", 0, NULL);
    hp->Hkv = (int)llm_gguf_u(g, "qwen4exp.attention.head_count_kv", 0, NULL);
    hp->D = (int)llm_gguf_u(g, "qwen4exp.attention.key_length", 0, NULL);
    hp->Hi = (int)llm_gguf_u(g, "qwen4exp.attention.indexer.head_count", 0, NULL);
    hp->Di = (int)llm_gguf_u(g, "qwen4exp.attention.indexer.key_length", 0, NULL);
    hp->topk = (int)llm_gguf_u(g, "qwen4exp.attention.indexer.top_k", 0, NULL);
    hp->NE = (int)llm_gguf_u(g, "qwen4exp.expert_count", 0, NULL);
    hp->K = (int)llm_gguf_u(g, "qwen4exp.expert_used_count", 0, NULL);
    hp->F = (int)llm_gguf_u(g, "qwen4exp.expert_feed_forward_length", 0, NULL);
    hp->Fsh = (int)llm_gguf_u(g, "qwen4exp.expert_shared_feed_forward_length", 0, NULL);
    hp->hc = (int)llm_gguf_u(g, "qwen4exp.hyper_connection.count", 0, NULL);
    hp->rank = (int)llm_gguf_u(g, "qwen4exp.hyper_connection.low_rank", 0, NULL);
    hp->lin_conv = (int)llm_gguf_u(g, "qwen4exp.ssm.conv_kernel", 0, NULL);
    hp->lin_Hv = (int)llm_gguf_u(g, "qwen4exp.ssm.group_count", 0, NULL);
    hp->lin_D = (int)llm_gguf_u(g, "qwen4exp.ssm.state_size", 0, NULL);
    hp->lin_Hk = (int)llm_gguf_u(g, "qwen4exp.ssm.inner_size", 0, NULL);   /* n. key-head */
    hp->ple_ng = (int)llm_gguf_u(g, "qwen4exp.ple.ngram_size", 0, NULL);
    hp->ple_heads = (int)llm_gguf_u(g, "qwen4exp.ple.heads_per_ngram", 0, NULL);
    hp->ple_conv = (int)llm_gguf_u(g, "qwen4exp.ple.conv_kernel", 0, NULL);
    hp->ple_eos = (int)llm_gguf_u(g, "qwen4exp.ple.eos_token_id", 0, NULL);
    hp->n_ple_layers = (int)llm_gguf_u(g, "qwen4exp.ple.layers", 0, NULL);
    hp->n_rot = (int)llm_gguf_u(g, "qwen4exp.rope.dimension_count", 0, NULL);
    hp->full_interval = (int)llm_gguf_u(g, "qwen4exp.full_attention_interval", 0, NULL);
    hp->nextn = (int)llm_gguf_u(g, "qwen4exp.nextn_predict_layers", 0, NULL);
    hp->eps = (float)llm_gguf_f(g, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-6, NULL);
    hp->rope_base = (float)llm_gguf_f(g, "qwen4exp.rope.freq_base", 1e7, NULL);
    if (hp->E <= 0 || hp->L <= 0 || hp->H <= 0 || hp->D <= 0 || hp->NE <= 0 || hp->K <= 0) {
        snprintf(err, errn, "hparam qwen4exp incompleti (E=%d L=%d H=%d D=%d NE=%d K=%d)",
                 hp->E, hp->L, hp->H, hp->D, hp->NE, hp->K);
        return -1;
    }
    return 0;
}

/* `suf` = suffisso COMPLETO dopo "blk.<il>." (es. "hc_attn_norm.weight", "ssm_a"). */
typedef struct {
    float *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
    float *hc_ffn_norm, *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
    float *attn_qkv, *attn_gate, *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_beta, *ssm_alpha, *ssm_norm, *ssm_out;
    float *attn_q, *attn_k, *attn_v, *attn_out, *attn_q_norm, *attn_k_norm;
    float *idx_q_proj, *idx_k_proj, *idx_q_norm, *idx_k_norm;
    float *ple_key, *ple_value, *ple_norm_key, *ple_norm_query, *ple_norm_conv, *ple_conv;
    float *gate_inp, *gate_inp_sh, *gate_sh, *up_sh, *down_sh;
    const uint8_t *gate_exps, *up_exps, *down_exps;
    int type_gate_exps, type_up_exps, type_down_exps;
    bool is_linear, is_ple;
} Q4nLayerRefs;

static float* q4n_f32(const LlmEngineMoeGguf* g, int il, const char* suf, bool required, bool* ok) {
    char b[200];
    snprintf(b, sizeof b, "blk.%d.%s", il, suf);
    float* x = llm_gguf_load_f32(g, b, NULL);
    if (!x && required) { fprintf(stderr, "qwen4: manca %s\n", b); *ok = false; }
    return x;
}
static const uint8_t* q4n_packed(const LlmEngineMoeGguf* g, int il, const char* suf, int* type, bool* ok) {
    char b[200];
    snprintf(b, sizeof b, "blk.%d.%s", il, suf);
    int i = llm_gguf_find_tensor(g, b);
    if (i < 0) { fprintf(stderr, "qwen4: manca %s\n", b); *ok = false; return NULL; }
    if (type) *type = g->ti[i].type;
    return llm_gguf_tensor_data(g, i, NULL);
}

static int q4n_layer_load(const LlmEngineMoeGguf* g, int il, bool is_linear, bool is_ple,
                          Q4nLayerRefs* r, char* err, size_t errn) {
    if (!g || !r) return -1;
    bool ok = true;
    memset(r, 0, sizeof *r);
    r->is_linear = is_linear; r->is_ple = is_ple;
    r->hc_attn_norm = q4n_f32(g, il, "hc_attn_norm.weight", true, &ok);
    r->hc_attn_down = q4n_f32(g, il, "hc_attn_down.weight", true, &ok);
    r->hc_attn_up = q4n_f32(g, il, "hc_attn_up.weight", true, &ok);
    r->hc_attn_inject = q4n_f32(g, il, "hc_attn_inject.weight", true, &ok);
    r->hc_ffn_norm = q4n_f32(g, il, "hc_ffn_norm.weight", true, &ok);
    r->hc_ffn_down = q4n_f32(g, il, "hc_ffn_down.weight", true, &ok);
    r->hc_ffn_up = q4n_f32(g, il, "hc_ffn_up.weight", true, &ok);
    r->hc_ffn_inject = q4n_f32(g, il, "hc_ffn_inject.weight", true, &ok);
    if (is_linear) {
        r->attn_qkv = q4n_f32(g, il, "attn_qkv.weight", true, &ok);
        r->attn_gate = q4n_f32(g, il, "attn_gate.weight", true, &ok);
        r->ssm_conv1d = q4n_f32(g, il, "ssm_conv1d.weight", true, &ok);
        r->ssm_dt = q4n_f32(g, il, "ssm_dt.bias", true, &ok);
        r->ssm_a = q4n_f32(g, il, "ssm_a", true, &ok);
        r->ssm_beta = q4n_f32(g, il, "ssm_beta.weight", true, &ok);
        r->ssm_alpha = q4n_f32(g, il, "ssm_alpha.weight", true, &ok);
        r->ssm_norm = q4n_f32(g, il, "ssm_norm.weight", true, &ok);
        r->ssm_out = q4n_f32(g, il, "ssm_out.weight", true, &ok);
    } else {
        r->attn_q = q4n_f32(g, il, "attn_q.weight", true, &ok);
        r->attn_k = q4n_f32(g, il, "attn_k.weight", true, &ok);
        r->attn_v = q4n_f32(g, il, "attn_v.weight", true, &ok);
        r->attn_out = q4n_f32(g, il, "attn_output.weight", true, &ok);
        r->attn_q_norm = q4n_f32(g, il, "attn_q_norm.weight", true, &ok);
        r->attn_k_norm = q4n_f32(g, il, "attn_k_norm.weight", true, &ok);
        r->idx_q_proj = q4n_f32(g, il, "indexer.q_proj.weight", true, &ok);
        r->idx_k_proj = q4n_f32(g, il, "indexer.k_proj.weight", true, &ok);
        r->idx_q_norm = q4n_f32(g, il, "indexer.q_norm.weight", true, &ok);
        r->idx_k_norm = q4n_f32(g, il, "indexer.k_norm.weight", true, &ok);
    }
    if (is_ple) {
        r->ple_key = q4n_f32(g, il, "ple_key.weight", true, &ok);
        r->ple_value = q4n_f32(g, il, "ple_value.weight", true, &ok);
        r->ple_norm_key = q4n_f32(g, il, "ple_norm_key.weight", true, &ok);
        r->ple_norm_query = q4n_f32(g, il, "ple_norm_query.weight", true, &ok);
        r->ple_norm_conv = q4n_f32(g, il, "ple_norm_conv.weight", true, &ok);
        r->ple_conv = q4n_f32(g, il, "ple_conv1d.weight", true, &ok);
    }
    r->gate_inp = q4n_f32(g, il, "ffn_gate_inp.weight", true, &ok);
    r->gate_inp_sh = q4n_f32(g, il, "ffn_gate_inp_shexp.weight", true, &ok);
    r->gate_sh = q4n_f32(g, il, "ffn_gate_shexp.weight", true, &ok);
    r->up_sh = q4n_f32(g, il, "ffn_up_shexp.weight", true, &ok);
    r->down_sh = q4n_f32(g, il, "ffn_down_shexp.weight", true, &ok);
    r->gate_exps = q4n_packed(g, il, "ffn_gate_exps.weight", &r->type_gate_exps, &ok);
    r->up_exps = q4n_packed(g, il, "ffn_up_exps.weight", &r->type_up_exps, &ok);
    r->down_exps = q4n_packed(g, il, "ffn_down_exps.weight", &r->type_down_exps, &ok);
    if (!ok) { snprintf(err, errn, "layer %d: tensori mancanti", il); return -1; }
    return 0;
}

#endif /* LLM_QWEN4_NATIVE_GGUF_H */
