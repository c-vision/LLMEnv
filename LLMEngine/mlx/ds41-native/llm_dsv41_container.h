/* llm_dsv41_container.h — DeepSeek V4.1 dal file GGUF (header-only, C + MLX).
 *
 * È ORCHESTRAZIONE dell'ENGINE (§0bis): il reader GGUF dà tabella+byte, il codec i dequant di
 * formato (via llm_engine_read_*), qui si mappa ogni campo del forward al tensore giusto.
 *   - dense: nome canonico DS4.1 (layers.N.<suffix>) -> nome canonico GGUF (blk.N.<suffix>);
 *   - esperti routed: (layer, expert) -> slice del 3D blk.N.ffn_{gate,up,down}_exps.weight.
 *
 * DUE modalità:
 *   - `llm_dsv41_load_container` / `_dir`: materializza TUTTI i layer (correttezza/parità);
 *   - `LlmDsv41Stream`: UN layer alla volta (dense) + esperti on demand → RAM bounded. */
#ifndef LLM_DSV41_CONTAINER_H
#define LLM_DSV41_CONTAINER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm_dsv41_tensors.h"
#include "llm_dsv41_config.h"
#include "llm_engine_container.h"   /* lettura ENGINE del GGUF (dense + esperti stacked) */
#include "llm_gguf.h"

/* mapping nome canonico DS4.1 -> nome canonico GGUF (inverso della mappa del converter). */
static inline void llm_dsv41_gguf_name(const char* in, char* out, size_t cap) {
    if (strcmp(in, "embed.weight") == 0) { snprintf(out, cap, "token_embd.weight"); return; }
    if (strcmp(in, "norm.weight") == 0)  { snprintf(out, cap, "output_norm.weight"); return; }
    if (strcmp(in, "head.weight") == 0)  { snprintf(out, cap, "output.weight"); return; }
    if (strncmp(in, "layers.", 7) == 0) {
        long L = strtol(in + 7, NULL, 10);
        const char* s = strchr(in + 7, '.');
        if (s != NULL) {
            ++s;
            const char* t = NULL;
            if      (!strcmp(s, "attn_norm.weight"))            t = "attn_norm.weight";
            else if (!strcmp(s, "ffn_norm.weight"))             t = "ffn_norm.weight";
            else if (!strcmp(s, "attn.wq_a.weight"))            t = "attn_q_a.weight";
            else if (!strcmp(s, "attn.wq_b.weight"))            t = "attn_q_b.weight";
            else if (!strcmp(s, "attn.wkv.weight"))             t = "attn_kv.weight";
            else if (!strcmp(s, "attn.wo_a.weight"))            t = "attn_output_a.weight";
            else if (!strcmp(s, "attn.wo_b.weight"))            t = "attn_output_b.weight";
            else if (!strcmp(s, "attn.q_norm.weight"))          t = "attn_q_a_norm.weight";
            else if (!strcmp(s, "attn.kv_norm.weight"))         t = "attn_kv_a_norm.weight";
            else if (!strcmp(s, "attn.attn_sink"))              t = "attn_sinks.weight";
            else if (!strcmp(s, "ffn.gate.weight"))             t = "ffn_gate_inp.weight";
            else if (!strcmp(s, "ffn.gate.bias"))               t = "ffn_gate_inp.bias";
            else if (!strcmp(s, "ffn.shared_experts.w1.weight"))t = "ffn_gate_shexp.weight";
            else if (!strcmp(s, "ffn.shared_experts.w3.weight"))t = "ffn_up_shexp.weight";
            else if (!strcmp(s, "ffn.shared_experts.w2.weight"))t = "ffn_down_shexp.weight";
            else if (!strcmp(s, "hc_attn_fn"))                  t = "hc_attn_fn.weight";
            else if (!strcmp(s, "hc_attn_scale"))               t = "hc_attn_scale.weight";
            else if (!strcmp(s, "hc_attn_base"))                t = "hc_attn_base.weight";
            else if (!strcmp(s, "hc_ffn_fn"))                   t = "hc_ffn_fn.weight";
            else if (!strcmp(s, "hc_ffn_scale"))                t = "hc_ffn_scale.weight";
            else if (!strcmp(s, "hc_ffn_base"))                 t = "hc_ffn_base.weight";
            if (t != NULL) { snprintf(out, cap, "blk.%ld.%s", L, t); return; }
        }
    }
    /* non mappato (compressor/indexer/engram/...): nome sorgente verbatim (come scritto dal converter) */
    snprintf(out, cap, "%s", in);
}

/* ── helper: tensore denso per nome canonico -> mlx_array f32 (shape dal GGUF) ── */
static inline mlx_array llm_dsv41_cont_dense(const LlmEngineMoeGguf* g, const char* name, bool* ok) {
    char gn[160]; llm_dsv41_gguf_name(name, gn, sizeof gn);
    int i = llm_gguf_find_tensor(g, gn);
    if (i < 0) { fprintf(stderr, "llm_dsv41_container: manca %s (%s)\n", name, gn); *ok = false; return mlx_array_new(); }
    const LlmCodecGgufTensor* t = &g->ti[i];
    size_t n = 1; for (int d = 0; d < t->ndim; ++d) n *= (size_t)t->dims[d];
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) { *ok = false; return mlx_array_new(); }
    size_t got = 0;
    LlmEngineMoeStatus s = llm_engine_read_dense_f32(g, gn, buf, n, &got);
    if (s != LLMENGINE_MOE_OK || got != n) {
        fprintf(stderr, "llm_dsv41_container: %s read %d (got=%zu n=%zu)\n", gn, (int)s, got, n);
        free(buf); *ok = false; return mlx_array_new();
    }
    int dims[4] = { 0, 0, 0, 0 };
    for (int d = 0; d < t->ndim && d < 4; ++d) dims[d] = (int)t->dims[d];
    mlx_array r = llm_dsv41_array_f32(buf, t->ndim, dims);
    free(buf);
    return r;
}

/* DENSE di UN layer -> L (nessun esperto). Elenco campi allineato a llm_dsv41_load_checkpoint. */
static inline void llm_dsv41_cont_layer_dense(const LlmEngineMoeGguf* g, const LlmDsv41Config* cfg,
                                              int l, LlmDsv41LayerWeights* L, bool* ok) {
    char nm[1400];
#define DC(_suf, _field) do { llm_dsv41_name_layer(nm, sizeof nm, l, _suf); L->_field = llm_dsv41_cont_dense(g, nm, ok); } while (0)
    DC("hc_attn_fn", hc_attn_fn); DC("hc_attn_scale", hc_attn_scale); DC("hc_attn_base", hc_attn_base);
    DC("hc_ffn_fn", hc_ffn_fn);   DC("hc_ffn_scale", hc_ffn_scale);   DC("hc_ffn_base", hc_ffn_base);
    DC("attn_norm.weight", attn_norm); DC("ffn_norm.weight", ffn_norm);
    DC("attn.wq_a.weight", wq_a); DC("attn.q_norm.weight", q_norm); DC("attn.wq_b.weight", wq_b);
    DC("attn.wkv.weight", wkv);   DC("attn.kv_norm.weight", kv_norm);
    DC("attn.wo_a.weight", wo_a); DC("attn.wo_b.weight", wo_b); DC("attn.attn_sink", attn_sink);
    DC("ffn.gate.weight", gate_w); DC("ffn.gate.bias", gate_b);
    DC("ffn.shared_experts.w1.weight", shared_w1);
    DC("ffn.shared_experts.w3.weight", shared_w3);
    DC("ffn.shared_experts.w2.weight", shared_w2);
    if (cfg->in_kv_source[l]) {
        DC("attn.compressor.wkv.weight", comp_wkv);
        DC("attn.compressor.norm.weight", comp_norm);
        if (cfg->compress_ratios[l] > 1) DC("attn.compressor.wgate.weight", comp_wgate);
    }
    if (cfg->in_index_source[l]) {
        DC("attn.indexer.weights_proj.weight", idx_weights_proj);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.indexer.wq_b.weight");
        L->idx_wq_b = llm_dsv41_cont_dense(g, nm, ok);
        if (cfg->in_kv_source[l]) {
            DC("attn.indexer.wk.weight", idx_wk);
            DC("attn.indexer.k_norm.weight", idx_k_norm);
        }
    }
#undef DC
}

/* ESPERTI routed di UN layer -> L (gate/up [DI,DIM], down [DIM,DI], stacked [E,...]). */
static inline void llm_dsv41_cont_layer_experts(const LlmEngineMoeGguf* g, const LlmDsv41Config* cfg,
                                                int l, LlmDsv41LayerWeights* L, bool* ok) {
    const int E = cfg->n_routed_experts, DI = cfg->moe_inter_dim, DIM = cfg->dim;
    float* e1 = (float*)malloc((size_t)E * DI * DIM * sizeof(float));
    float* e3 = (float*)malloc((size_t)E * DI * DIM * sizeof(float));
    float* e2 = (float*)malloc((size_t)E * DIM * DI * sizeof(float));
    if (!e1 || !e3 || !e2) { free(e1); free(e3); free(e2); *ok = false; return; }
    for (int e = 0; e < E && *ok; e++) {
        uint32_t ro = 0, co = 0;
        LlmEngineMoeStatus es = llm_engine_read_expert_f32(g, l, (uint32_t)e,
            e1 + (size_t)e * DI * DIM, (size_t)DI * DIM,
            e3 + (size_t)e * DI * DIM, (size_t)DI * DIM,
            e2 + (size_t)e * DIM * DI, (size_t)DIM * DI, &ro, &co);
        if (es != LLMENGINE_MOE_OK) { fprintf(stderr, "llm_dsv41_container: esperto %d.%d: %d\n", l, e, (int)es); *ok = false; }
    }
    L->experts_w1 = llm_dsv41_array_f32(e1, 3, (const int[]){ E, DI, DIM });
    L->experts_w3 = llm_dsv41_array_f32(e3, 3, (const int[]){ E, DI, DIM });
    L->experts_w2 = llm_dsv41_array_f32(e2, 3, (const int[]){ E, DIM, DI });
    free(e1); free(e3); free(e2);
}

/* ── Load COMPLETO (tutti i layer, materializzato). ── */
static inline bool llm_dsv41_load_container(LlmDsv41Weights* w, const LlmDsv41Config* cfg,
                                            const char* gguf_path, mlx_stream s, int* unloaded) {
    (void)s;
    if (w == NULL || cfg == NULL || gguf_path == NULL) return false;
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(gguf_path, &g, &map, &maplen)) { fprintf(stderr, "llm_dsv41_container: inapribile: %s\n", gguf_path); return false; }
    if (!llm_dsv41_weights_alloc(w, cfg)) { llm_gguf_close_file(&g, map, maplen); return false; }
    bool ok = true;
    w->embed = llm_dsv41_cont_dense(&g, "embed.weight", &ok);
    w->norm  = llm_dsv41_cont_dense(&g, "norm.weight", &ok);
    w->head  = llm_dsv41_cont_dense(&g, "head.weight", &ok);
    for (int l = 0; l < cfg->n_layers && ok; l++) {
        LlmDsv41LayerWeights* L = &w->layers[l];
        llm_dsv41_cont_layer_dense(&g, cfg, l, L, &ok);
        if (ok) llm_dsv41_cont_layer_experts(&g, cfg, l, L, &ok);
    }
    if (unloaded) {
        int n = 0;
        for (int l = 0; l < cfg->n_layers; l++)
            for (int e = 0; e < cfg->n_engram_layers; e++)
                if (cfg->engram_layer_ids[e] == l) n += 5;
        *unloaded = n;
    }
    llm_gguf_close_file(&g, map, maplen);
    if (!ok) llm_dsv41_weights_free(w);
    return ok;
}

/* config da <cfg_dir>/config.json (il GGUF non contiene il config runtime dell'engine). */
static inline bool llm_dsv41_container_read_cfg(const char* cfg_dir, LlmDsv41Config* out) {
    char path[1200];
    snprintf(path, sizeof path, "%s/config.json", cfg_dir);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "llm_dsv41_container: manca %s\n", path); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return false; }
    size_t rd = fread(buf, 1, (size_t)n, f); fclose(f); buf[rd] = '\0';
    bool ok = llm_dsv41_config_from_json(buf, out);
    free(buf);
    if (!ok) { fprintf(stderr, "llm_dsv41_container: config.json non valido in %s\n", cfg_dir); return false; }
    if (llm_dsv41_config_validate(out) != LLM_DSV41_OK) {
        fprintf(stderr, "llm_dsv41_container: config non valido\n"); return false;
    }
    return true;
}

static inline bool llm_dsv41_load_container_dir(LlmDsv41Weights* w, const char* cfg_dir,
                                                const char* gguf_path, mlx_stream s, int* unloaded) {
    LlmDsv41Config cfg;
    if (!llm_dsv41_container_read_cfg(cfg_dir, &cfg)) return false;
    return llm_dsv41_load_container(w, &cfg, gguf_path, s, unloaded);
}

/* ── STREAMING: UN layer alla volta (dense) + esperti per-chiave → RAM bounded ─────────────── */
typedef struct {
    LlmEngineMoeGguf g;
    void*   map; size_t maplen;
    LlmDsv41Config cfg;
    LlmDsv41LayerWeights slot;   /* dense del SOLO layer corrente */
    int loaded;                  /* indice del layer caricato, -1 = nessuno */
    bool ok;
} LlmDsv41Stream;

static inline bool llm_dsv41_stream_open(LlmDsv41Stream* s, const char* cfg_dir, const char* gguf_path) {
    memset(s, 0, sizeof *s);
    s->loaded = -1;
    if (!llm_dsv41_container_read_cfg(cfg_dir, &s->cfg)) return false;
    if (!llm_gguf_open_file(gguf_path, &s->g, &s->map, &s->maplen)) {
        fprintf(stderr, "llm_dsv41_stream: inapribile: %s\n", gguf_path); return false;
    }
    return true;
}

/* Carica il dense del layer `l` nello slot (libera il precedente). Ritorna &slot o NULL. */
static inline LlmDsv41LayerWeights* llm_dsv41_stream_layer(LlmDsv41Stream* s, int l) {
    if (l < 0 || l >= s->cfg.n_layers) return NULL;
    if (s->loaded == l) return &s->slot;   /* idempotente */
    if (s->loaded >= 0) llm_dsv41_layer_free(&s->slot);
    s->ok = true;
    llm_dsv41_cont_layer_dense(&s->g, &s->cfg, l, &s->slot, &s->ok);
    s->loaded = l;
    return s->ok ? &s->slot : NULL;
}

/* Esperti del layer `l` per la chiave (l, e): gate/up [DI,DIM], down [DIM,DI] in f32 (top-k). */
static inline LlmEngineMoeStatus llm_dsv41_stream_expert(LlmDsv41Stream* s, int l, int e,
                                                   float* gate, size_t gate_cap,
                                                   float* up, size_t up_cap,
                                                   float* down, size_t down_cap,
                                                   uint32_t* rows_out, uint32_t* cols_out) {
    return llm_engine_read_expert_f32(&s->g, l, (uint32_t)e, gate, gate_cap, up, up_cap, down, down_cap,
                                      rows_out, cols_out);
}

static inline void llm_dsv41_stream_close(LlmDsv41Stream* s) {
    if (s->loaded >= 0) llm_dsv41_layer_free(&s->slot);
    llm_gguf_close_file(&s->g, s->map, s->maplen);
    s->loaded = -1;
}

#endif /* LLM_DSV41_CONTAINER_H */
