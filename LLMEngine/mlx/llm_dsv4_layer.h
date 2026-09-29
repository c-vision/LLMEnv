/* llm_dsv4_layer.h — layer loop DS4: mHC pre -> attenzione -> mHC post -> mHC pre ->
 * MoE -> mHC post, ripetuto sugli n_layer. Port fedele del riferimento CPU ds4.c.
 * Header-only. Test: llm_dsv4_layer_test.c. */
#ifndef LLM_DSV4_LAYER_H
#define LLM_DSV4_LAYER_H

#include <stdlib.h>
#include <stdio.h>
#include "llm_dsv4_block.h"
#include "llm_dsv4_moe_fwd.h"
#include "llm_dsv4_cache.h"

/* Un layer su un token. hc [n_hc*embd] in ingresso, out_hc [n_hc*embd] in uscita.
 * kv = righe KV (raw+compresse) se presenti, altrimenti NULL (solo kv corrente). */
static inline void llm_dsv4_layer_forward(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* hc, int token, int pos, int il, const float* kv, int n_kv, float* out_hc) {
    int HC = c->n_hc, D = c->n_embd;
    float* attn_x = (float*)malloc(sizeof(float) * (size_t)D);
    float* attn_out = (float*)malloc(sizeof(float) * (size_t)D);
    float* after = (float*)malloc(sizeof(float) * (size_t)HC * D);
    float* ffn_x = (float*)malloc(sizeof(float) * (size_t)D);
    float* ffn_out = (float*)malloc(sizeof(float) * (size_t)D);
    float* post = (float*)malloc(sizeof(float) * (size_t)HC);
    float* comb = (float*)malloc(sizeof(float) * (size_t)HC * HC);

    /* attenzione */
    llm_dsv4_block_pre(c, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, L->attn_norm,
                       hc, attn_x, post, comb);
    llm_dsv4_attn_from_normed(c, L, attn_x, pos, il, kv, n_kv, attn_out);
    llm_dsv4_block_post(c, attn_out, hc, post, comb, after);

    /* MoE */
    llm_dsv4_block_pre(c, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, L->ffn_norm,
                       after, ffn_x, post, comb);
    llm_dsv4_moe_forward(c, L, ffn_x, il, token, ffn_out);
    llm_dsv4_block_post(c, ffn_out, after, post, comb, out_hc);

    free(comb); free(post); free(ffn_out); free(ffn_x); free(after); free(attn_out); free(attn_x);
}

/* ── COMPOSIZIONE (livello 3): il layer DS4 come PIPELINE di passi dichiarati (`.llmarch`,
 * structure `layer`). Ogni `op` -> la primitiva DS4 corrispondente. Nessuna biforcazione:
 * `llm_dsv4_layer_forward_cached_pe` esegue QUESTA pipeline (ops==NULL -> sequenza canonica). ── */
#define LLM_DSV4_MAX_OPS 16
static char g_dsv4_layer_ops_buf[LLM_DSV4_MAX_OPS][24];   /* copia: la LlmDesc e' su stack */
static const char* g_dsv4_layer_ops[LLM_DSV4_MAX_OPS];
static int g_dsv4_layer_nops = 0;   /* 0 = usa la sequenza canonica */
/* Il motore imposta la pipeline dichiarata dal `.llmarch` (structure `layer`). */
static inline void llm_dsv4_set_layer_ops(const char* const* ops, int n) {
    if (n < 0) n = 0; if (n > LLM_DSV4_MAX_OPS) n = LLM_DSV4_MAX_OPS;
    for (int i = 0; i < n; i++) {
        snprintf(g_dsv4_layer_ops_buf[i], sizeof g_dsv4_layer_ops_buf[i], "%s", ops[i] ? ops[i] : "");
        g_dsv4_layer_ops[i] = g_dsv4_layer_ops_buf[i];
    }
    g_dsv4_layer_nops = n;
}

/* Un layer su un token eseguendo la PIPELINE `ops` (default: canonica mhc/attn/mhc/mhc/moe/mhc). */
static inline void llm_dsv4_layer_forward_cached_pe_pipe(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* cache, const float* hc, int token, int pos, int il,
        float* out_hc, const char* const* ops, int n_ops) {
    int HC = c->n_hc, D = c->n_embd;
    static const char* def_ops[] = { "mhc_pre_attn", "attention", "mhc_post_attn",
                                     "mhc_pre_ffn", "moe", "mhc_post_ffn" };
    if (ops == NULL || n_ops <= 0) { ops = def_ops; n_ops = 6; }
    float* cur    = (float*)malloc(sizeof(float) * (size_t)HC * D); memcpy(cur, hc, sizeof(float) * (size_t)HC * D);
    float* normed = (float*)malloc(sizeof(float) * (size_t)D);
    float* proj   = (float*)malloc(sizeof(float) * (size_t)D);
    float* saved  = (float*)malloc(sizeof(float) * (size_t)HC * D);
    float* nxt    = (float*)malloc(sizeof(float) * (size_t)HC * D);
    float* post   = (float*)malloc(sizeof(float) * (size_t)HC);
    float* comb   = (float*)malloc(sizeof(float) * (size_t)HC * HC);
    for (int k = 0; k < n_ops; k++) {
        const char* op = ops[k];
        if (strcmp(op, "mhc_pre_attn") == 0) {
            memcpy(saved, cur, sizeof(float) * (size_t)HC * D);
            llm_dsv4_block_pre(c, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, L->attn_norm, cur, normed, post, comb);
            { const char* dn = getenv("LLM_DSV4_DUMP_ANORM");
              if (dn && dn[0] && il == 0) { FILE* f = fopen(dn, "wb"); if (f) { fwrite(normed, sizeof(float), (size_t)D, f); fclose(f); } } }
        } else if (strcmp(op, "mhc_pre_ffn") == 0) {
            memcpy(saved, cur, sizeof(float) * (size_t)HC * D);
            llm_dsv4_block_pre(c, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, L->ffn_norm, cur, normed, post, comb);
        } else if (strcmp(op, "attention") == 0) {
            llm_dsv4_attn_cached(c, L, pe, cache, normed, pos, il, proj);
            { const char* da = getenv("LLM_DSV4_DUMP_AOUT");
              if (da && da[0] && il == 0) { FILE* f = fopen(da, "wb"); if (f) { fwrite(proj, sizeof(float), (size_t)D, f); fclose(f); } } }
        } else if (strcmp(op, "moe") == 0) {
            if (!llm_dsv4_gpu_moe_try(c, L, il, normed, token, proj)) {
                if (pe && pe->gate) llm_dsv4_moe_forward_packed(c, L, pe, normed, il, token, proj);
                else                llm_dsv4_moe_forward(c, L, normed, il, token, proj);
            }
        } else if (strcmp(op, "mhc_post_attn") == 0 || strcmp(op, "mhc_post_ffn") == 0) {
            llm_dsv4_block_post(c, proj, saved, post, comb, nxt);
            memcpy(cur, nxt, sizeof(float) * (size_t)HC * D);
            if (strcmp(op, "mhc_post_attn") == 0) {
                const char* da = getenv("LLM_DSV4_DUMP_ATTN"); const char* dt = getenv("LLM_DSV4_DUMP_TOKEN");
                int want = dt ? atoi(dt) : -1;
                if (da && da[0] && token == want) { FILE* f = fopen(da, il == 0 ? "wb" : "ab");
                    if (f) { fwrite(cur, sizeof(float), (size_t)HC * D, f); fclose(f); } }
            }
        }
        /* op sconosciuto: ignorato (il gap e' visibile nel log della pipeline) */
    }
    if (out_hc) memcpy(out_hc, cur, sizeof(float) * (size_t)HC * D);
    free(comb); free(post); free(nxt); free(saved); free(proj); free(normed); free(cur);
}

/* Un layer su un token con cache KV. `pe` = esperti packed (NULL -> usa gli esperti f32).
 * Esegue la pipeline dichiarata (g_dsv4_layer_ops) o quella canonica se non impostata. */
static inline void llm_dsv4_layer_forward_cached_pe(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* cache, const float* hc, int token, int pos, int il,
        float* out_hc) {
    llm_dsv4_layer_forward_cached_pe_pipe(c, L, pe, cache, hc, token, pos, il, out_hc,
                                          g_dsv4_layer_nops > 0 ? g_dsv4_layer_ops : NULL,
                                          g_dsv4_layer_nops);
}
static inline void llm_dsv4_layer_forward_cached(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        LlmDsv4LayerCache* cache, const float* hc, int token, int pos, int il, float* out_hc) {
    llm_dsv4_layer_forward_cached_pe(c, L, NULL, cache, hc, token, pos, il, out_hc);
}

/* Tutti i layer su un token (nessuna cache: attenzione solo sul token corrente). */
static inline void llm_dsv4_forward_token(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        const float* token_emb, int token, int pos, float* out_hc) {
    int HC = c->n_hc, D = c->n_embd;
    float* cur = (float*)malloc(sizeof(float) * (size_t)HC * D);
    float* nxt = (float*)malloc(sizeof(float) * (size_t)HC * D);
    llm_dsv4_hc_from_embedding(c, token_emb, cur);
    for (int il = 0; il < c->n_layer; il++) {
        llm_dsv4_layer_forward(c, &w->layers[il], cur, token, pos, il, NULL, 0, nxt);
        float* t = cur; cur = nxt; nxt = t;
    }
    if (out_hc) memcpy(out_hc, cur, sizeof(float) * (size_t)HC * D);
    free(nxt); free(cur);
}

#endif /* LLM_DSV4_LAYER_H */
