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
    if (!a || !a->s || (strcmp(a->s, "qwen4exp") != 0 && strcmp(a->s, "qwen4_exp") != 0)) {
        snprintf(err, errn, "arch non qwen4exp/qwen4_exp: %s", (a && a->s) ? a->s : "?");
        return -1;
    }
    /* Le chiavi standard sono emesse con il nome ABI (qwen4_exp.*), quelle della sezione
     * gguf_meta con il nome ds4 (qwen4exp.*): si accettano entrambe. */
#define HP_U(suf, def) llm_gguf_u(g, "qwen4exp." suf, (uint64_t)(def), "qwen4_exp." suf)
#define HP_F(suf, def) llm_gguf_f(g, "qwen4exp." suf, (double)(def), "qwen4_exp." suf)
    memset(hp, 0, sizeof *hp);
    hp->E = (int)HP_U("embedding_length", 0);
    hp->L = (int)HP_U("block_count", 0);
    hp->H = (int)HP_U("attention.head_count", 0);
    hp->Hkv = (int)HP_U("attention.head_count_kv", 0);
    hp->D = (int)HP_U("attention.key_length", 0);
    hp->Hi = (int)HP_U("attention.indexer.head_count", 0);
    hp->Di = (int)HP_U("attention.indexer.key_length", 0);
    hp->topk = (int)HP_U("attention.indexer.top_k", 0);
    hp->NE = (int)HP_U("expert_count", 0);
    hp->K = (int)HP_U("expert_used_count", 0);
    hp->F = (int)HP_U("expert_feed_forward_length", 0);
    hp->Fsh = (int)HP_U("expert_shared_feed_forward_length", 0);
    hp->hc = (int)HP_U("hyper_connection.count", 0);
    hp->rank = (int)HP_U("hyper_connection.low_rank", 0);
    hp->lin_conv = (int)HP_U("ssm.conv_kernel", 0);
    hp->lin_Hk = (int)HP_U("ssm.group_count", 0);          /* ds4: group_count = n. key-head */
    hp->lin_Hv = (int)HP_U("ssm.time_step_rank", 0);       /* ds4: time_step_rank = n. value-head */
    hp->lin_D = (int)HP_U("ssm.state_size", 0);
    hp->ple_ng = (int)HP_U("ple.ngram_size", 0);
    hp->ple_heads = (int)HP_U("ple.heads_per_ngram", 0);
    hp->ple_conv = (int)HP_U("ple.conv_kernel", 0);
    hp->ple_eos = (int)HP_U("ple.eos_token_id", 0);
    hp->n_ple_layers = 1;                                  /* PLE presente per tensori (P7) */
    hp->n_rot = (int)HP_U("rope.dimension_count", 0);
    hp->full_interval = (int)HP_U("full_attention_interval", 0);
    hp->nextn = (int)HP_U("nextn_predict_layers", 0);
    if (hp->nextn > 0 && hp->L > hp->nextn) hp->L -= hp->nextn;   /* blk.48=MTP: non fa parte del forward */
    hp->eps = (float)HP_F("attention.layer_norm_rms_epsilon", 1e-6);
    hp->rope_base = (float)HP_F("rope.freq_base", 1e7);
#undef HP_U
#undef HP_F
    if (hp->E <= 0 || hp->L <= 0 || hp->H <= 0 || hp->D <= 0 || hp->NE <= 0 || hp->K <= 0) {
        snprintf(err, errn, "hparam qwen4exp incompleti (E=%d L=%d H=%d D=%d NE=%d K=%d)",
                 hp->E, hp->L, hp->H, hp->D, hp->NE, hp->K);
        return -1;
    }
    if (hp->n_rot <= 0) hp->n_rot = 64;   /* qwen4exp: mrope [11,11,10] -> 64 dim */
    return 0;
}

/* `suf` = suffisso COMPLETO dopo "blk.<il>." (es. "hc_attn_norm.weight", "ssm_a"). */
typedef struct {
    float *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
    float *hc_ffn_norm, *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
    float *attn_qkv, *attn_gate, *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_beta, *ssm_alpha, *ssm_norm, *ssm_out;
    float *attn_q, *attn_k, *attn_v, *attn_out, *attn_q_norm, *attn_k_norm;
    float *idx_q_proj, *idx_k_proj, *idx_q_norm, *idx_k_norm, *idx_qk;
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
static float* q4n_f32_global(const LlmEngineMoeGguf* g, const char* name, bool required, bool* ok) {
    float* x = llm_gguf_load_f32(g, name, NULL);
    if (!x && required) { fprintf(stderr, "qwen4: manca %s\n", name); *ok = false; }
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

static void q4n_layer_free(Q4nLayerRefs* r) {
    if (!r) return;
#define F(x) free(r->x)
    F(hc_attn_norm); F(hc_attn_down); F(hc_attn_up); F(hc_attn_inject);
    F(hc_ffn_norm); F(hc_ffn_down); F(hc_ffn_up); F(hc_ffn_inject);
    F(attn_qkv); F(attn_gate); F(ssm_conv1d); F(ssm_dt); F(ssm_a); F(ssm_beta);
    F(ssm_alpha); F(ssm_norm); F(ssm_out);
    F(attn_q); F(attn_k); F(attn_v); F(attn_out); F(attn_q_norm); F(attn_k_norm);
    F(idx_q_proj); F(idx_k_proj); F(idx_q_norm); F(idx_k_norm); F(idx_qk);
    F(ple_key); F(ple_value); F(ple_norm_key); F(ple_norm_query); F(ple_norm_conv); F(ple_conv);
    F(gate_inp); F(gate_inp_sh); F(gate_sh); F(up_sh); F(down_sh);
#undef F
    memset(r, 0, sizeof *r);
}

static int q4n_t_quantized(int t) { return t == 2 || t == 8 || t == 10 || t == 12 || t == 16; }

static int q4n_tensor_is_quant(const LlmEngineMoeGguf* g, int il, const char* suf) {
    char b[200]; snprintf(b, sizeof b, "blk.%d.%s", il, suf);
    int i = llm_gguf_find_tensor(g, b);
    return (i >= 0) ? q4n_t_quantized(g->ti[i].type) : 0;
}

static int q4n_layer_load_ex(const LlmEngineMoeGguf* g, int il, bool is_linear, bool is_ple,
                             int skip_big, Q4nLayerRefs* r, char* err, size_t errn) {
    if (!g || !r) return -1;
    bool ok = true;
    memset(r, 0, sizeof *r);
    r->is_linear = is_linear; r->is_ple = is_ple;
    r->hc_attn_norm = q4n_f32(g, il, "hc_attn_norm.weight", true, &ok);
    r->hc_attn_down = (skip_big && q4n_tensor_is_quant(g, il, "hc_attn_down.weight")) ? NULL : q4n_f32(g, il, "hc_attn_down.weight", true, &ok);
    r->hc_attn_up = (skip_big && q4n_tensor_is_quant(g, il, "hc_attn_up.weight")) ? NULL : q4n_f32(g, il, "hc_attn_up.weight", true, &ok);
    r->hc_attn_inject = q4n_f32(g, il, "hc_attn_inject.weight", true, &ok);
    r->hc_ffn_norm = q4n_f32(g, il, "hc_ffn_norm.weight", true, &ok);
    r->hc_ffn_down = (skip_big && q4n_tensor_is_quant(g, il, "hc_ffn_down.weight")) ? NULL : q4n_f32(g, il, "hc_ffn_down.weight", true, &ok);
    r->hc_ffn_up = (skip_big && q4n_tensor_is_quant(g, il, "hc_ffn_up.weight")) ? NULL : q4n_f32(g, il, "hc_ffn_up.weight", true, &ok);
    r->hc_ffn_inject = q4n_f32(g, il, "hc_ffn_inject.weight", true, &ok);
    if (is_linear) {
        r->attn_qkv = skip_big ? NULL : q4n_f32(g, il, "attn_qkv.weight", true, &ok);
        r->attn_gate = skip_big ? NULL : q4n_f32(g, il, "attn_gate.weight", true, &ok);
        r->ssm_conv1d = q4n_f32(g, il, "ssm_conv1d.weight", true, &ok);
        r->ssm_dt = q4n_f32(g, il, "ssm_dt.bias", true, &ok);
        r->ssm_a = q4n_f32(g, il, "ssm_a", true, &ok);
        r->ssm_beta = skip_big ? NULL : q4n_f32(g, il, "ssm_beta.weight", true, &ok);
        r->ssm_alpha = skip_big ? NULL : q4n_f32(g, il, "ssm_alpha.weight", true, &ok);
        r->ssm_norm = q4n_f32(g, il, "ssm_norm.weight", true, &ok);
        r->ssm_out = skip_big ? NULL : q4n_f32(g, il, "ssm_out.weight", true, &ok);
    } else {
        r->attn_q = skip_big ? NULL : q4n_f32(g, il, "attn_q.weight", true, &ok);
        r->attn_k = skip_big ? NULL : q4n_f32(g, il, "attn_k.weight", true, &ok);
        r->attn_v = skip_big ? NULL : q4n_f32(g, il, "attn_v.weight", true, &ok);
        r->attn_out = skip_big ? NULL : q4n_f32(g, il, "attn_output.weight", true, &ok);
        r->attn_q_norm = q4n_f32(g, il, "attn_q_norm.weight", true, &ok);
        r->attn_k_norm = q4n_f32(g, il, "attn_k_norm.weight", true, &ok);
        r->idx_q_proj = (!skip_big) ? q4n_f32(g, il, "indexer.q_proj.weight", false, &ok) : NULL;
        r->idx_k_proj = (!skip_big) ? q4n_f32(g, il, "indexer.k_proj.weight", false, &ok) : NULL;
        r->idx_qk = (!skip_big) ? q4n_f32(g, il, "indexer.qk_proj.weight", false, &ok) : NULL;
        r->idx_q_norm = q4n_f32(g, il, "indexer.q_norm.weight", true, &ok);
        r->idx_k_norm = q4n_f32(g, il, "indexer.k_norm.weight", true, &ok);
        if (!skip_big && !r->idx_qk && (!r->idx_q_proj || !r->idx_k_proj)) { fprintf(stderr, "qwen4: indexer q/k mancanti (layer %d)\n", il); ok = false; }
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
static int q4n_layer_load(const LlmEngineMoeGguf* g, int il, bool is_linear, bool is_ple,
                          Q4nLayerRefs* r, char* err, size_t errn) {
    return q4n_layer_load_ex(g, il, is_linear, is_ple, 0, r, err, errn);
}

#endif /* LLM_QWEN4_NATIVE_GGUF_H */
