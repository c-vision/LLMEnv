/* llm_dsv41_loader.h — carica un checkpoint DeepSeek V4.1 (raw hub release) in `LlmDsv41Weights`
 * con array MLX float32. Header-only, C puro.
 *
 * Formati (da Rapid-MLX `deepseek_v41_native/dequant.py`, confermati dal root `config.json`:
 * `quant_method=fp8, weight_block_size=[32,32], scale_fmt=ue8m0, expert_dtype=fp4`):
 *   - pesi matmul fp8 e4m3 [out,in] + scala ue8m0 [ceil(out/32),ceil(in/32)]
 *   - esperti routed fp4 e2m1 packed [out,in//2] + scala ue8m0 [out,in//32]
 *   - il resto (norm, hc_*, gate.bias, attn_sink, compressor, indexer) bf16/f32
 *   - ue8m0: byte b -> 2^(b-127)
 * Nomi canonici senza prefisso `model.` (v. `llm_dsv41_tensors.h`).
 *
 * SCOPE: carica i campi usati dal forward (`LlmDsv41LayerWeights`) + i pesi DSpark/MTP
 * (`mtp.N.*`, v. `llm_dsv41_load_dspark`). Indexer/engram/vision NON sono materializzati qui
 * (enumerati da `llm_dsv41_expected_names` per la copertura, v. docs).
 * I tensori sono dequantizzati a float32: per il modello reale questo NON e' sostenibile in RAM
 * (gli esperti vanno tenuti packed + gather_qmm); il loader serve alla correttezza/mappatura.
 *
 * Verifica: `llm_dsv41_loader_test.c` su un checkpoint tiny sintetico (tools/gen_dsv41_st_ref.py)
 * + copertura nomi esatta vs `model.safetensors.index.json` reale.
 */
#ifndef LLM_DSV41_LOADER_H
#define LLM_DSV41_LOADER_H

#include "llm_dsv41_tensors.h"
#include "llm_dsv41_ckpt.h"
#include "llm_dsv41_fp4.h"
#include "llm_dsv41_config.h"
#include "llm_dsv41_dspark_spec.h"
/* DEQUANT di FORMATO (pesi fp8 block / fp4 esperti): UNA SOLA SORGENTE nel CODEC (§0bis).
 * Prima erano reimplementati qui (static inline) — duplicazione rimossa 2026-09-24. */
#include "llmcodec_fp8_fp4.h"

/* tensore non quantizzato (bf16/f16/f32) -> float32. */
static inline mlx_array llm_dsv41_load_plain(LlmDsv41Ckpt* ck, const char* name,
                                             mlx_stream s, bool* ok) {
    mlx_array a = mlx_array_new(), r = mlx_array_new();
    if (!llm_dsv41_ckpt_get(ck, name, &a, s) || mlx_astype(&r, a, MLX_FLOAT32, s) != 0) {
        fprintf(stderr, "llm_dsv41_loader: manca %s\n", name);
        *ok = false; return mlx_array_new();
    }
    return r;
}

/* nome tensore quantizzato: il release usa `X.weight` (<-> `X.scale`), NON `X.weight.scale`. */
static inline void llm_dsv41_scale_name(char* out, size_t n, const char* name) {
    snprintf(out, n, "%s", name);
    char* w = strstr(out, ".weight");
    if (w) { *w = '\0'; strncat(out, ".scale", n - strlen(out) - 1); }
    else snprintf(out + strlen(out), n - strlen(out), ".scale");
}

/* nome.scale esiste? -> fp4/fp8; altrimenti bf16/f32. fp4 = esperti routed. */
static inline mlx_array llm_dsv41_load_weight(LlmDsv41Ckpt* ck, const char* name,
                                              int out, int in, int fp4,
                                              mlx_stream s, bool* ok) {
    char sn[1400];
    llm_dsv41_scale_name(sn, sizeof sn, name);
    if (llm_dsv41_ckpt_has(ck, sn)) {
        mlx_array w = mlx_array_new(), sc = mlx_array_new();
        if (!llm_dsv41_ckpt_get(ck, name, &w, s) || !llm_dsv41_ckpt_get(ck, sn, &sc, s)) {
            fprintf(stderr, "llm_dsv41_loader: manca %s o %s\n", name, sn);
            *ok = false; return mlx_array_new();
        }
        mlx_array_eval(w); mlx_array_eval(sc);
        const uint8_t* wd = mlx_array_data_uint8(w);
        const uint8_t* sd = mlx_array_data_uint8(sc);
        if (!wd || !sd) { *ok = false; return mlx_array_new(); }
        float* d = (float*)malloc((size_t)out * (size_t)in * sizeof(float));
        if (!d) { *ok = false; return mlx_array_new(); }
        if (fp4) llmcodec_dsv41_dequant_fp4_expert(wd, sd, out, in, d);
        else     llmcodec_dsv41_dequant_fp8_block(wd, sd, out, in, d);
        mlx_array res = llm_dsv41_array_f32(d, 2, (const int[]){ out, in });
        free(d);
        return res;
    }
    return llm_dsv41_load_plain(ck, name, s, ok);
}

static inline bool llm_dsv41_copy_f32(mlx_array a, float* dst, int n) {
    mlx_array_eval(a);
    const float* p = mlx_array_data_float32(a);
    if (!p) return false;
    memcpy(dst, p, (size_t)n * sizeof(float));
    return true;
}

/* Carica il backbone in `w` (alloca secondo `cfg`). Ritorna false + messaggio su tensore
 * obbligatorio mancante. `unloaded` (opzionale) riceve il numero di nomi attesi non
 * materializzati (indexer/engram, se presenti). */
static inline bool llm_dsv41_load_checkpoint(LlmDsv41Weights* w, const LlmDsv41Config* cfg,
                                             const char* path, mlx_stream s, int* unloaded) {
    LlmDsv41Ckpt ck;
    if (!llm_dsv41_ckpt_open(&ck, path)) {
        fprintf(stderr, "llm_dsv41_loader: checkpoint inapribile: %s\n", path);
        return false;
    }
    if (!llm_dsv41_weights_alloc(w, cfg)) { llm_dsv41_ckpt_close(&ck); return false; }
    bool ok = true;
    const int DIM = cfg->dim, H = cfg->n_heads, D = cfg->head_dim;
    const int QD = cfg->q_lora_rank, OG = cfg->o_groups, OL = cfg->o_lora_rank;
    const int E = cfg->n_routed_experts, DI = cfg->moe_inter_dim, GRP = H * D / OG;

    char nm[1400];
#define D41_W(_name, _out, _in, _fp4) llm_dsv41_load_weight(&ck, _name, _out, _in, _fp4, s, &ok)
#define D41_P(_name) llm_dsv41_load_plain(&ck, _name, s, &ok)

    w->embed = D41_P("embed.weight");
    w->norm  = D41_P("norm.weight");
    w->head  = D41_P("head.weight");

    for (int l = 0; l < cfg->n_layers && ok; l++) {
        LlmDsv41LayerWeights* L = &w->layers[l];
        llm_dsv41_name_layer(nm, sizeof nm, l, "hc_attn_fn");    L->hc_attn_fn = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "hc_attn_scale"); L->hc_attn_scale = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "hc_attn_base");  L->hc_attn_base = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "hc_ffn_fn");     L->hc_ffn_fn = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "hc_ffn_scale");  L->hc_ffn_scale = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "hc_ffn_base");   L->hc_ffn_base = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn_norm.weight"); L->attn_norm = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "ffn_norm.weight");  L->ffn_norm = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.wq_a.weight"); L->wq_a = D41_W(nm, QD, DIM, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.q_norm.weight"); L->q_norm = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.wq_b.weight"); L->wq_b = D41_W(nm, H * D, QD, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.wkv.weight");  L->wkv = D41_W(nm, D, DIM, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.kv_norm.weight"); L->kv_norm = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.wo_a.weight"); L->wo_a = D41_W(nm, OG * OL, GRP, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.wo_b.weight"); L->wo_b = D41_W(nm, DIM, OG * OL, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "attn.attn_sink");   L->attn_sink = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "ffn.gate.weight");  L->gate_w = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "ffn.gate.bias");    L->gate_b = D41_P(nm);
        llm_dsv41_name_layer(nm, sizeof nm, l, "ffn.shared_experts.w1.weight"); L->shared_w1 = D41_W(nm, DI, DIM, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "ffn.shared_experts.w3.weight"); L->shared_w3 = D41_W(nm, DI, DIM, 0);
        llm_dsv41_name_layer(nm, sizeof nm, l, "ffn.shared_experts.w2.weight"); L->shared_w2 = D41_W(nm, DIM, DI, 0);
        if (cfg->in_kv_source[l]) {
            llm_dsv41_name_layer(nm, sizeof nm, l, "attn.compressor.wkv.weight");   L->comp_wkv = D41_P(nm);
            llm_dsv41_name_layer(nm, sizeof nm, l, "attn.compressor.norm.weight");  L->comp_norm = D41_P(nm);
            if (cfg->compress_ratios[l] > 1) {
                llm_dsv41_name_layer(nm, sizeof nm, l, "attn.compressor.wgate.weight"); L->comp_wgate = D41_P(nm);
            }
        }
        if (cfg->in_index_source[l]) {
            const int IH = cfg->index_head_dim, NH = cfg->index_n_heads;
            llm_dsv41_name_layer(nm, sizeof nm, l, "attn.indexer.wq_b.weight");
            L->idx_wq_b = D41_W(nm, NH * IH, QD, 0);
            llm_dsv41_name_layer(nm, sizeof nm, l, "attn.indexer.weights_proj.weight");
            L->idx_weights_proj = D41_P(nm);
            if (cfg->in_kv_source[l]) {
                llm_dsv41_name_layer(nm, sizeof nm, l, "attn.indexer.wk.weight");
                L->idx_wk = D41_P(nm);
                llm_dsv41_name_layer(nm, sizeof nm, l, "attn.indexer.k_norm.weight");
                L->idx_k_norm = D41_P(nm);
            }
        }
        /* esperti routed (stack) */
        float* e1 = (float*)malloc((size_t)E * DI * DIM * sizeof(float));
        float* e3 = (float*)malloc((size_t)E * DI * DIM * sizeof(float));
        float* e2 = (float*)malloc((size_t)E * DIM * DI * sizeof(float));
        if (!e1 || !e3 || !e2) { free(e1); free(e3); free(e2); ok = false; break; }
        for (int e = 0; e < E && ok; e++) {
            llm_dsv41_name_expert(nm, sizeof nm, l, e, "w1");
            mlx_array a = D41_W(nm, DI, DIM, 1);
            if (ok && !llm_dsv41_copy_f32(a, e1 + (size_t)e * DI * DIM, DI * DIM)) ok = false;
            llm_dsv41_name_expert(nm, sizeof nm, l, e, "w3");
            mlx_array b = D41_W(nm, DI, DIM, 1);
            if (ok && !llm_dsv41_copy_f32(b, e3 + (size_t)e * DI * DIM, DI * DIM)) ok = false;
            llm_dsv41_name_expert(nm, sizeof nm, l, e, "w2");
            mlx_array c = D41_W(nm, DIM, DI, 1);
            if (ok && !llm_dsv41_copy_f32(c, e2 + (size_t)e * DIM * DI, DIM * DI)) ok = false;
        }
        L->experts_w1 = llm_dsv41_array_f32(e1, 3, (const int[]){ E, DI, DIM });
        L->experts_w3 = llm_dsv41_array_f32(e3, 3, (const int[]){ E, DI, DIM });
        L->experts_w2 = llm_dsv41_array_f32(e2, 3, (const int[]){ E, DIM, DI });
        free(e1); free(e3); free(e2);
    }
    if (unloaded) {
        /* nomi attesi non materializzati: solo le tabelle Engram (enumerati ma non caricate). */
        int n = 0;
        for (int l = 0; l < cfg->n_layers; l++)
            for (int e = 0; e < cfg->n_engram_layers; e++)
                if (cfg->engram_layer_ids[e] == l) n += 5;
        *unloaded = n;
    }
#undef D41_W
#undef D41_P
    llm_dsv41_ckpt_close(&ck);
    if (!ok) llm_dsv41_weights_free(w);
    return ok;
}

/* ── DSpark/MTP: carica i pesi `mtp.N.*` in un `LlmDsv41DSpark` ─────────────────
 * Nomi reali (release HF): `mtp.N.{hc_*,attn.*,attn_norm,ffn_norm,ffn.gate.*,
 * ffn.shared_experts.*,ffn.experts.E.*}`; stage 0 ha `main_proj.{weight,scale}` e
 * `main_norm.weight`; l'ultimo stage ha `norm.weight`, `markov_head.{embed,head}.weight`,
 * `confidence_head.proj.weight`. Gli esperti MTP usano `dspark_n_routed_experts`. */
static inline void llm_dsv41_dspark_name(char* o, size_t n, int st, const char* suf) {
    snprintf(o, n, "mtp.%d.%s", st, suf);
}
static inline void llm_dsv41_dspark_expert_name(char* o, size_t n, int st, int e, const char* w) {
    snprintf(o, n, "mtp.%d.ffn.experts.%d.%s.weight", st, e, w);
}

static inline bool llm_dsv41_load_dspark(LlmDsv41DSpark* d, const LlmDsv41Config* cfg,
                                         const char* path, mlx_stream s) {
    memset(d, 0, sizeof *d);
    LlmDsv41Ckpt ck;
    if (!llm_dsv41_ckpt_open(&ck, path)) {
        fprintf(stderr, "llm_dsv41_loader: checkpoint DSpark inapribile: %s\n", path);
        return false;
    }
    bool ok = true;
    d->cfg = *cfg;
    LlmDsv41Config* dc = &d->draft_cfg;
    *dc = *cfg;
    dc->n_routed_experts = cfg->dspark_n_routed_experts ? cfg->dspark_n_routed_experts : cfg->n_routed_experts;
    dc->n_activated_experts = cfg->dspark_n_activated_experts ? cfg->dspark_n_activated_experts
                                                              : cfg->n_activated_experts;
    dc->n_engram_layers = 0;
    const int N = cfg->n_mtp_layers;
    const int DIM = cfg->dim, H = cfg->n_heads, D = cfg->head_dim;
    const int QD = cfg->q_lora_rank, OG = cfg->o_groups, OL = cfg->o_lora_rank;
    const int E = dc->n_routed_experts, DI = cfg->moe_inter_dim, GRP = H * D / OG;
    const int M = cfg->n_dspark_targets > 0 ? cfg->n_dspark_targets : 1;
    char nm[1400];
#define DS41_W(_name, _out, _in, _fp4) llm_dsv41_load_weight(&ck, _name, _out, _in, _fp4, s, &ok)
#define DS41_P(_name) llm_dsv41_load_plain(&ck, _name, s, &ok)
    d->embed = DS41_P("embed.weight");
    d->head  = DS41_P("head.weight");
    d->n_stages = N;
    d->stages = (LlmDsv41DSparkStage*)calloc((size_t)(N > 0 ? N : 1), sizeof(LlmDsv41DSparkStage));
    if (!d->stages) { llm_dsv41_ckpt_close(&ck); return false; }
    for (int st = 0; st < N && ok; st++) {
        LlmDsv41LayerWeights* L = &d->stages[st].block;
        llm_dsv41_dspark_name(nm, sizeof nm, st, "hc_attn_fn");    L->hc_attn_fn = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "hc_attn_scale"); L->hc_attn_scale = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "hc_attn_base");  L->hc_attn_base = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "hc_ffn_fn");     L->hc_ffn_fn = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "hc_ffn_scale");  L->hc_ffn_scale = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "hc_ffn_base");   L->hc_ffn_base = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn_norm.weight"); L->attn_norm = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "ffn_norm.weight");  L->ffn_norm = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.wq_a.weight"); L->wq_a = DS41_W(nm, QD, DIM, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.q_norm.weight"); L->q_norm = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.wq_b.weight");  L->wq_b = DS41_W(nm, H * D, QD, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.wkv.weight");   L->wkv = DS41_W(nm, D, DIM, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.kv_norm.weight"); L->kv_norm = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.wo_a.weight");  L->wo_a = DS41_W(nm, OG * OL, GRP, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.wo_b.weight");  L->wo_b = DS41_W(nm, DIM, OG * OL, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "attn.attn_sink");    L->attn_sink = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "ffn.gate.weight");   L->gate_w = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "ffn.gate.bias");     L->gate_b = DS41_P(nm);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "ffn.shared_experts.w1.weight"); L->shared_w1 = DS41_W(nm, DI, DIM, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "ffn.shared_experts.w3.weight"); L->shared_w3 = DS41_W(nm, DI, DIM, 0);
        llm_dsv41_dspark_name(nm, sizeof nm, st, "ffn.shared_experts.w2.weight"); L->shared_w2 = DS41_W(nm, DIM, DI, 0);
        float* e1 = (float*)malloc((size_t)E * DI * DIM * sizeof(float));
        float* e3 = (float*)malloc((size_t)E * DI * DIM * sizeof(float));
        float* e2 = (float*)malloc((size_t)E * DIM * DI * sizeof(float));
        if (!e1 || !e3 || !e2) { free(e1); free(e3); free(e2); ok = false; break; }
        for (int e = 0; e < E && ok; e++) {
            llm_dsv41_dspark_expert_name(nm, sizeof nm, st, e, "w1");
            mlx_array a = DS41_W(nm, DI, DIM, 1);
            if (ok && !llm_dsv41_copy_f32(a, e1 + (size_t)e * DI * DIM, DI * DIM)) ok = false;
            llm_dsv41_dspark_expert_name(nm, sizeof nm, st, e, "w3");
            mlx_array b = DS41_W(nm, DI, DIM, 1);
            if (ok && !llm_dsv41_copy_f32(b, e3 + (size_t)e * DI * DIM, DI * DIM)) ok = false;
            llm_dsv41_dspark_expert_name(nm, sizeof nm, st, e, "w2");
            mlx_array c = DS41_W(nm, DIM, DI, 1);
            if (ok && !llm_dsv41_copy_f32(c, e2 + (size_t)e * DIM * DI, DIM * DI)) ok = false;
        }
        L->experts_w1 = llm_dsv41_array_f32(e1, 3, (const int[]){ E, DI, DIM });
        L->experts_w3 = llm_dsv41_array_f32(e3, 3, (const int[]){ E, DI, DIM });
        L->experts_w2 = llm_dsv41_array_f32(e2, 3, (const int[]){ E, DIM, DI });
        free(e1); free(e3); free(e2);
        if (st == 0) {
            llm_dsv41_dspark_name(nm, sizeof nm, st, "main_proj.weight");
            d->stages[st].main_proj = DS41_W(nm, DIM, DIM * M, 0);
            llm_dsv41_dspark_name(nm, sizeof nm, st, "main_norm.weight");
            d->stages[st].main_norm = DS41_P(nm);
        }
        if (st == N - 1) {
            llm_dsv41_dspark_name(nm, sizeof nm, st, "norm.weight");
            d->stages[st].out_norm = DS41_P(nm);
            llm_dsv41_dspark_name(nm, sizeof nm, st, "markov_head.embed.weight");
            d->stages[st].markov_embed = DS41_P(nm);
            llm_dsv41_dspark_name(nm, sizeof nm, st, "markov_head.head.weight");
            d->stages[st].markov_head = DS41_P(nm);
            llm_dsv41_dspark_name(nm, sizeof nm, st, "confidence_head.proj.weight");
            d->stages[st].conf_proj = DS41_P(nm);
        }
    }
#undef DS41_W
#undef DS41_P
    llm_dsv41_ckpt_close(&ck);
    if (!ok) { llm_dsv41_dspark_free(d); return false; }
    return true;
}

/* Step 1 completo: legge `<dir>/config.json`, valida e carica il checkpoint. */
static inline bool llm_dsv41_load_model_dir(LlmDsv41Weights* w, const char* dir,
                                            mlx_stream s, int* unloaded) {
    char path[1200];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "llm_dsv41_loader: manca %s\n", path); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return false; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    fclose(f); buf[rd] = '\0';
    LlmDsv41Config cfg;
    bool ok = llm_dsv41_config_from_json(buf, &cfg);
    free(buf);
    if (!ok) { fprintf(stderr, "llm_dsv41_loader: config.json non valido in %s\n", dir); return false; }
    int v = llm_dsv41_config_validate(&cfg);
    if (v != LLM_DSV41_OK) {
        fprintf(stderr, "llm_dsv41_loader: config non valido (llm_dsv41_validate=%d)\n", v);
        return false;
    }
    return llm_dsv41_load_checkpoint(w, &cfg, dir, s, unloaded);
}

#endif /* LLM_DSV41_LOADER_H */
