/* llm_dsv4_weights.h — pesi DeepSeek V4 Flash (DS4) e loader dal GGUF.
 * Binding nome->tensore secondo ds4 weights_bind_layer; dequant a f32 via llm_gguf_load_f32.
 * NOTA RAM: per il modello reale gli esperti (IQ2_XXS/Q2_K) vanno tenuti PACKED + dot interi;
 * qui si dequantizza a f32 (verifica/test e modelli tiny). Header-only, C puro. Test: llm_dsv4_weights_test.c. */
#ifndef LLM_DSV4_WEIGHTS_H
#define LLM_DSV4_WEIGHTS_H

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "llm_dsv4.h"

typedef struct {
    float *hc_attn_fn, *hc_attn_scale, *hc_attn_base;
    float *attn_norm, *attn_q_a, *attn_q_a_norm, *attn_q_b, *attn_kv, *attn_kv_a_norm;
    float *attn_sinks, *attn_output_a, *attn_output_b;
    float *comp_ape, *comp_kv, *comp_gate, *comp_norm;                 /* ratio != 0 */
    float *idx_q_b, *idx_proj, *idx_c_ape, *idx_c_kv, *idx_c_gate, *idx_c_norm; /* ratio == 4 */
    float *hc_ffn_fn, *hc_ffn_scale, *hc_ffn_base;
    float *ffn_norm, *ffn_gate_inp, *exp_probs_b;
    float *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
    float *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp;
    float *ffn_gate_tid2eid;                                            /* il < n_hash_layer */
} LlmDsv4LayerWeights;

/* Esperti tenuti PACKED (puntatori nel mmap GGUF), senza dequant a f32: per il modello reale
 * gate/up sono IQ2_XXS e down Q2_K (~2e9 elementi). */
typedef struct {
    const uint8_t *gate, *up, *down;
    int gate_type, up_type, down_type;
    int in_dim, ff_dim;   /* gate/up: [in_dim, ff_dim]; down: [ff_dim, n_embd] */
    const uint8_t *shexp_gate, *shexp_up, *shexp_down;   /* shared expert packed (Q8_0) */
    int shexp_gate_type, shexp_up_type, shexp_down_type;
    const uint8_t *q_a, *q_b, *kv, *out_a, *out_b;       /* attn packed (Q8_0) */
    int q_a_type, q_b_type, kv_type, out_a_type, out_b_type;
    void* store;   /* LlmDsv4Store* opzionale (swap/residency esperti); NULL = mmap residente */
    const uint8_t* (*store_get)(void* store, int layer, int e, const uint8_t** g, const uint8_t** u, const uint8_t** d);
    void (*store_prefetch)(void* store, int layer, const int* experts, int n);   /* letture parallele dei mancanti (opz.) */
    void (*store_madvise)(void* store, int layer);   /* prefetch asincrono WILLNEED di un intero layer (item B, opz.) */
    /* Layer Q4_K in ARENA (tensori impilati): 1 se servito, riempie i tre tensori di layer; q4_ensure legge i mancanti. */
    int (*q4_stacks)(void* store, int layer, const uint8_t** g, const uint8_t** u, const uint8_t** d);
    int (*q4_ensure)(void* store, int layer, const int* experts, int n);
} LlmDsv4PackedExperts;

typedef struct {
    LlmDsv4Config cfg;
    float *token_embd, *output_norm, *output;
    float *output_hc_base, *output_hc_fn, *output_hc_scale;
    LlmDsv4LayerWeights* layers;   /* [cfg.n_layer] */
    LlmDsv4PackedExperts* exps;    /* [cfg.n_layer], opzionale (se esperti non f32) */
    const uint8_t* output_p;       /* lm_head packed (Q8_0) nel modo packed */
    int output_type;
} LlmDsv4Weights;

static inline void llm_dsv4_layer_free(LlmDsv4LayerWeights* l) {
    float** p = (float**)l;
    for (size_t i = 0; i < sizeof(LlmDsv4LayerWeights) / sizeof(float*); i++) free(p[i]);
    memset(l, 0, sizeof *l);
}
static inline void llm_dsv4_weights_free(LlmDsv4Weights* w) {
    free(w->token_embd); free(w->output_norm); free(w->output);
    free(w->output_hc_base); free(w->output_hc_fn); free(w->output_hc_scale);
    if (w->layers) { for (int i = 0; i < w->cfg.n_layer; i++) llm_dsv4_layer_free(&w->layers[i]); free(w->layers); }
    free(w->exps);
    memset(w, 0, sizeof *w);
}

/* Bind packed (puntatori nel mmap) di un tensore `blk.<il>.<suf>.weight`. */
static inline const uint8_t* llm_dsv4_bind_packed(const LlmEngineMoeGguf* g, int il, const char* suf, int* type, bool* ok) {
    char b[200];
    snprintf(b, sizeof b, "blk.%d.%s.weight", il, suf);
    int i = llm_gguf_find_tensor(g, b);
    if (i < 0) { fprintf(stderr, "llm_dsv4: manca %s\n", b); *ok = false; return NULL; }
    if (type) *type = g->ti[i].type;
    return llm_gguf_tensor_data(g, i, NULL);
}

/* Carica un campo `blk.<il>.<suf>.weight`; `required` -> errore se assente. */
static inline float* llm_dsv4_lw(const LlmEngineMoeGguf* g, int il, const char* suf, bool required, bool* ok) {
    char b[200];
    snprintf(b, sizeof b, "blk.%d.%s.weight", il, suf);
    float* x = llm_gguf_load_f32(g, b, NULL);
    if (!x && required) { fprintf(stderr, "llm_dsv4: manca %s\n", b); *ok = false; }
    return x;
}

/* Come llm_dsv4_lw, ma SOLO per il modo con esperti f32 (test/tiny). Nel modo PACKED (modello reale) i tensori attn/shexp/head vivono
 * come puntatori Q8_0 nel mmap (pe->q_a/q_b/kv/out_a/out_b/shexp_*, output_p) e i rami f32 dei percorsi CPU (`else` di
 * `if (pe && pe->q_a ...)` ecc.) non girano mai: la copia dequantizzata a f32 occuperebbe ~25 GB di RAM inutili (misurato con vmmap/
 * malloc_history: 3x43 regioni da 128 MB + 3x43 da 32 MB ... nel prefill RSS 62 GB contro 35 del ds4 originale). */
static inline float* llm_dsv4_lw_cpu(const LlmEngineMoeGguf* g, int il, const char* suf, bool required, bool* ok, bool experts_f32) {
    return experts_f32 ? llm_dsv4_lw(g, il, suf, required, ok) : NULL;
}

static inline bool llm_dsv4_load_impl(LlmDsv4Weights* w, const LlmEngineMoeGguf* g, bool experts_f32) {
    memset(w, 0, sizeof *w);
    if (!llm_dsv4_config_from_gguf(g, &w->cfg)) return false;
    const LlmDsv4Config* c = &w->cfg;
    bool ok = true;
    w->token_embd = llm_gguf_load_f32(g, "token_embd.weight", NULL);
    w->output_norm = llm_gguf_load_f32(g, "output_norm.weight", NULL);
    /* packed: il lm_head e' output_p (Q8_0 nel mmap): niente copia f32 da 2 GB (llm_dsv4_output_logits usa output_p se presente) */
    w->output = (experts_f32 || llm_gguf_find_tensor(g, "output.weight") < 0) ? llm_gguf_load_f32(g, "output.weight", NULL) : NULL;
    w->output_hc_base = llm_gguf_load_f32(g, "output_hc_base.weight", NULL);
    w->output_hc_fn = llm_gguf_load_f32(g, "output_hc_fn.weight", NULL);
    w->output_hc_scale = llm_gguf_load_f32(g, "output_hc_scale.weight", NULL);
    w->layers = (LlmDsv4LayerWeights*)calloc((size_t)c->n_layer, sizeof(LlmDsv4LayerWeights));
    if (!w->layers) return false;
    if (!experts_f32) {
        w->exps = (LlmDsv4PackedExperts*)calloc((size_t)c->n_layer, sizeof(LlmDsv4PackedExperts));
        if (!w->exps) { free(w->layers); return false; }
        int oi = llm_gguf_find_tensor(g, "output.weight");
        if (oi >= 0) { w->output_p = llm_gguf_tensor_data(g, oi, NULL); w->output_type = g->ti[oi].type; }
    }
    for (int il = 0; il < c->n_layer && ok; il++) {
        LlmDsv4LayerWeights* L = &w->layers[il];
        int ratio = llm_dsv4_layer_compress_ratio(c, il);
        L->hc_attn_fn = llm_dsv4_lw(g, il, "hc_attn_fn", true, &ok);
        L->hc_attn_scale = llm_dsv4_lw(g, il, "hc_attn_scale", true, &ok);
        L->hc_attn_base = llm_dsv4_lw(g, il, "hc_attn_base", true, &ok);
        L->attn_norm = llm_dsv4_lw(g, il, "attn_norm", true, &ok);
        L->attn_q_a = llm_dsv4_lw_cpu(g, il, "attn_q_a", true, &ok, experts_f32);
        L->attn_q_a_norm = llm_dsv4_lw(g, il, "attn_q_a_norm", true, &ok);
        L->attn_q_b = llm_dsv4_lw_cpu(g, il, "attn_q_b", true, &ok, experts_f32);
        L->attn_kv = llm_dsv4_lw_cpu(g, il, "attn_kv", true, &ok, experts_f32);
        L->attn_kv_a_norm = llm_dsv4_lw(g, il, "attn_kv_a_norm", true, &ok);
        L->attn_sinks = llm_dsv4_lw(g, il, "attn_sinks", true, &ok);
        L->attn_output_a = llm_dsv4_lw_cpu(g, il, "attn_output_a", true, &ok, experts_f32);
        L->attn_output_b = llm_dsv4_lw_cpu(g, il, "attn_output_b", true, &ok, experts_f32);
        if (ratio != 0) {
            L->comp_ape = llm_dsv4_lw(g, il, "attn_compressor_ape", false, &ok);
            L->comp_kv = llm_dsv4_lw(g, il, "attn_compressor_kv", false, &ok);
            L->comp_gate = llm_dsv4_lw(g, il, "attn_compressor_gate", false, &ok);
            L->comp_norm = llm_dsv4_lw(g, il, "attn_compressor_norm", false, &ok);
        }
        if (ratio == 4) {
            L->idx_q_b = llm_dsv4_lw(g, il, "indexer.attn_q_b", false, &ok);
            L->idx_proj = llm_dsv4_lw(g, il, "indexer.proj", false, &ok);
            L->idx_c_ape = llm_dsv4_lw(g, il, "indexer_compressor_ape", false, &ok);
            L->idx_c_kv = llm_dsv4_lw(g, il, "indexer_compressor_kv", false, &ok);
            L->idx_c_gate = llm_dsv4_lw(g, il, "indexer_compressor_gate", false, &ok);
            L->idx_c_norm = llm_dsv4_lw(g, il, "indexer_compressor_norm", false, &ok);
        }
        L->hc_ffn_fn = llm_dsv4_lw(g, il, "hc_ffn_fn", true, &ok);
        L->hc_ffn_scale = llm_dsv4_lw(g, il, "hc_ffn_scale", true, &ok);
        L->hc_ffn_base = llm_dsv4_lw(g, il, "hc_ffn_base", true, &ok);
        L->ffn_norm = llm_dsv4_lw(g, il, "ffn_norm", true, &ok);
        L->ffn_gate_inp = llm_dsv4_lw(g, il, "ffn_gate_inp", true, &ok);
        { char b[200]; snprintf(b, sizeof b, "blk.%d.exp_probs_b.bias", il); L->exp_probs_b = llm_gguf_load_f32(g, b, NULL); }
        if (experts_f32) {
            L->ffn_gate_exps = llm_dsv4_lw(g, il, "ffn_gate_exps", true, &ok);
            L->ffn_up_exps = llm_dsv4_lw(g, il, "ffn_up_exps", true, &ok);
            L->ffn_down_exps = llm_dsv4_lw(g, il, "ffn_down_exps", true, &ok);
        } else {
            LlmDsv4PackedExperts* pe = &w->exps[il];
            pe->gate = llm_dsv4_bind_packed(g, il, "ffn_gate_exps", &pe->gate_type, &ok);
            pe->up = llm_dsv4_bind_packed(g, il, "ffn_up_exps", &pe->up_type, &ok);
            pe->down = llm_dsv4_bind_packed(g, il, "ffn_down_exps", &pe->down_type, &ok);
            pe->in_dim = c->n_embd; pe->ff_dim = c->n_ff_exp;
            pe->shexp_gate = llm_dsv4_bind_packed(g, il, "ffn_gate_shexp", &pe->shexp_gate_type, &ok);
            pe->shexp_up = llm_dsv4_bind_packed(g, il, "ffn_up_shexp", &pe->shexp_up_type, &ok);
            pe->shexp_down = llm_dsv4_bind_packed(g, il, "ffn_down_shexp", &pe->shexp_down_type, &ok);
            pe->q_a   = llm_dsv4_bind_packed(g, il, "attn_q_a",       &pe->q_a_type, &ok);
            pe->q_b   = llm_dsv4_bind_packed(g, il, "attn_q_b",       &pe->q_b_type, &ok);
            pe->kv    = llm_dsv4_bind_packed(g, il, "attn_kv",        &pe->kv_type, &ok);
            pe->out_a = llm_dsv4_bind_packed(g, il, "attn_output_a",  &pe->out_a_type, &ok);
            pe->out_b = llm_dsv4_bind_packed(g, il, "attn_output_b",  &pe->out_b_type, &ok);
        }
        L->ffn_gate_shexp = llm_dsv4_lw_cpu(g, il, "ffn_gate_shexp", true, &ok, experts_f32);
        L->ffn_up_shexp = llm_dsv4_lw_cpu(g, il, "ffn_up_shexp", true, &ok, experts_f32);
        L->ffn_down_shexp = llm_dsv4_lw_cpu(g, il, "ffn_down_shexp", true, &ok, experts_f32);
        if (il < c->n_hash_layer) L->ffn_gate_tid2eid = llm_dsv4_lw(g, il, "ffn_gate_tid2eid", true, &ok);
    }
    if (!ok) { llm_dsv4_weights_free(w); return false; }
    return true;
}
static inline bool llm_dsv4_load(LlmDsv4Weights* w, const LlmEngineMoeGguf* g) {
    return llm_dsv4_load_impl(w, g, true);   /* test/tiny: esperti dequant a f32 */
}

#endif /* LLM_DSV4_WEIGHTS_H */
