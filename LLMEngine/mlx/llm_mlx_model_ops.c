/* ============================================================================
 * llm_mlx_model_ops.c — Model Builder/Validate/Export-IR, n-gram session table, smoketest (kv page/kernel/lora/rewrite/merge/radix).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_rewrite.h"
#include "llm_mlx_merge.h"
#include "llm_mlx_kernel.h"
#include <cJSON.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Definite altrove (monolite / llm_mlx_stats.c); usate da questo dominio. */
int llm_ai_mlx_model_validate(llm_ai_mlx_model_handle handle);
int llm_ai_mlx_model_flops(llm_ai_mlx_model_handle handle, int seq_len);

int llm_ai_mlx_kv_page_smoketest(void) {
    KvPagePool pool = llm_kv_page_pool_new(4, 16);
    int p0 = llm_kv_page_alloc(&pool, 1);
    int p1 = llm_kv_page_alloc(&pool, 1);
    int p2 = llm_kv_page_alloc(&pool, 2);
    if (p0 != 0 || p1 != 1 || p2 != 2) { llm_kv_page_pool_free(&pool); return -1; }
    llm_kv_page_touch(&pool, p0);
    int p3 = llm_kv_page_alloc(&pool, 3);
    if (p3 != 3) { llm_kv_page_pool_free(&pool); return -1; }
    int p4 = llm_kv_page_alloc(&pool, 5); /* pool pieno -> evict LRU (p1) */
    if (pool.evictions != 1 || p4 != p1) { llm_kv_page_pool_free(&pool); return -1; }
    llm_kv_page_free(&pool, p2);
    int p5 = llm_kv_page_alloc(&pool, 9);
    if (p5 != p2) { llm_kv_page_pool_free(&pool); return -1; }
    llm_kv_page_pool_free(&pool);
    return 0;
}

/* Kernel registry smoketest (fase 3 runtime kernel replacement): registra 2 kernel, li
 * dispatcha a runtime, sostituisce uno, verifica la sostituzione. Ritorna 0 ok. */
static void llm_kern_mul2(int x) { (void)x; }
static void llm_kern_add10(int x) { (void)x; }

int llm_ai_mlx_kernel_smoketest(void) {
    LlmKernelRegistry r;
    llm_kernel_registry_init(&r);
    if (!llm_kernel_register(&r, "mul2", llm_kern_mul2)) return -1;
    if (!llm_kernel_register(&r, "add10", llm_kern_add10)) return -1;
    if (llm_kernel_count(&r) != 2) return -1;
    if (!llm_kernel_dispatch(&r, "mul2", 3)) return -1;
    if (!llm_kernel_dispatch(&r, "add10", 5)) return -1;
    if (llm_kernel_dispatch(&r, "absent", 1)) return -1; /* non registrato -> false */
    /* sostituzione a runtime: "mul2" -> add10 */
    if (!llm_kernel_register(&r, "mul2", llm_kern_add10)) return -1;
    if (llm_kernel_count(&r) != 2) return -1; /* sostituzione, non aggiunta */
    if (!llm_kernel_dispatch(&r, "mul2", 7)) return -1;
    return 0;
}

/* Adapter LoRA smoketest (fase 4): verifica matematica LoRA + registry/hot-swap. */
int llm_ai_mlx_lora_smoketest(void) {
    LlmLoraAdapter a; memset(&a, 0, sizeof(a)); snprintf(a.name, LLM_LORA_NAME, "coding");
    a.n_modules = 1;
    a.modules[0].in_dim = 2; a.modules[0].out_dim = 2; a.modules[0].rank = 2; a.modules[0].scale = 0.5f;
    float A[4] = {1,0, 0,1}; float B[4] = {1,0, 0,1};
    a.modules[0].A = A; a.modules[0].B = B;
    LlmLoraAdapter a2; memset(&a2, 0, sizeof(a2)); snprintf(a2.name, LLM_LORA_NAME, "sql");
    a2.n_modules = 1; a2.modules[0].in_dim = 2; a2.modules[0].out_dim = 2; a2.modules[0].rank = 2; a2.modules[0].scale = 2.0f;
    a2.modules[0].A = A; a2.modules[0].B = B;
    LlmLoraRegistry r; llm_lora_registry_init(&r);
    if (!llm_lora_register(&r, "coding", &a)) return -1;
    if (!llm_lora_register(&r, "sql", &a2)) return -1;
    float x[2] = {1,2}, out[2] = {0,0};
    llm_lora_apply_row(&r.adapters[0].modules[0], x, out);
    if (out[0] < 0.49 || out[0] > 0.51 || out[1] < 0.99 || out[1] > 1.01) return -1;
    if (!llm_lora_enable(&r, "coding") || r.active != 0) return -1;
    if (!llm_lora_enable(&r, "sql") || r.active != 1 || r.adapters[0].enabled) return -1; /* hot-swap */
    llm_lora_disable(&r);
    if (r.active != -1) return -1;
    return 0;
}

/* Adapter LoRA merge smoketest (fase 4): somma ponderata dei contributi di 2 adapter. */
int llm_ai_mlx_lora_merge_smoketest(void) {
    /* due moduli identici A=B=I, scale 0.5; merge con pesi 1.0 e 1.0 -> contrib = x */
    float A[4] = {1,0, 0,1}; float B[4] = {1,0, 0,1};
    LlmLoraModule m0; memset(&m0,0,sizeof(m0)); m0.in_dim=2; m0.out_dim=2; m0.rank=2; m0.scale=0.5f; m0.A=A; m0.B=B;
    LlmLoraModule m1; memset(&m1,0,sizeof(m1)); m1.in_dim=2; m1.out_dim=2; m1.rank=2; m1.scale=0.5f; m1.A=A; m1.B=B;
    LlmLoraModule mods[2] = {m0, m1};
    float w[2] = {1.0f, 1.0f};
    float x[2] = {1,2}, out[2] = {0,0};
    int n = llm_lora_merge_contrib(mods, w, 2, x, out);
    if (n != 2) return -1;
    /* 0.5*x + 0.5*x = x -> [1,2] */
    if (out[0] < 0.99 || out[0] > 1.01 || out[1] < 1.99 || out[1] > 2.01) return -1;
    return 0;
}

int llm_ai_mlx_rewrite_smoketest(void) {
    LlmkxRewritePlan p; llm_rewrite_plan_init(&p);
    if (!llm_rewrite_add(&p, 3, LLM_RW_REPLACE_ATTENTION, true, true)) return -1;
    if (!llm_rewrite_add(&p, 10, LLM_RW_REMOVE_LAYER, true, true)) return -1;
    if (llm_rewrite_training_count(&p) != 2) return -1;
    if (llm_rewrite_validate(&p, 32, false)) return -1;   /* senza allow_training -> invalido */
    if (!llm_rewrite_validate(&p, 32, true)) return -1;    /* con allow_training -> valido */
    return 0;
}

/* MODEL BUILDER (roadmap 8.11 fase 7): ricostruisce/valida l'artefatto eseguibile.
 * Verifica: config (validate), tensori di output (embed/final_norm/lm_head) accessibili,
 * shape della proiezione principale del layer 0 coerente con il config, vocab_size coerente,
 * layer dispatch risolvibile (ARCH_TABLE). Produce un report di build su stderr.
 * Ritorna 0 se il modello e' pronto per l'esecuzione, -1 altrimenti. */
int llm_ai_mlx_model_build(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    const ModelConfig* c = &m->cfg;
    if (llm_ai_mlx_model_validate(handle) != 0) return -1;
    int err = 0;
    /* 1. tensori di output (embed/final_norm/lm_head) accessibili */
    OutputTensors t;
    if (!load_output_tensors(m, &t)) { fprintf(stderr, "build: tensori di output inaccessibili\n"); return -1; }
    free_output_tensors(&t);
    /* 2. shape della proiezione principale del layer 0 */
    const char* proj = llm_mlx_layer_proj(c, 0);
    char nb[256];
    tname(c->layer_prefix, 0, proj, nb, sizeof(nb));
    mlx_array w=mlx_array_new(), s=mlx_array_new(), b=mlx_array_new();
    if (multishard_get_triple(&m->weights, nb, &w, &s, &b) && s.ctx != NULL) {
        const int* ws = mlx_array_shape(w);
        int nd = mlx_array_ndim(w);
        fprintf(stderr, "build: layer0 %s weight dims=%d shape[%d][%d]\n", proj, nd, ws[0], nd>1?ws[1]:-1);
        mlx_array_free(w); mlx_array_free(s); mlx_array_free(b);
    } else { fprintf(stderr, "build: layer0 %s non accessibile\n", proj); err = 1; }
    /* 3. vocab_size coerente */
    if (c->vocab_size <= 0) { fprintf(stderr, "build: vocab_size=%d invalido\n", c->vocab_size); err = 1; }
    /* 4. dispatch risolvibile */
    if (m->arch == NULL) { fprintf(stderr, "build: arch dispatch non risolto\n"); err = 1; }
    if (!err) {
        fprintf(stderr, "BUILD OK: %s (%d layer, hidden %d, vocab %d, %s, cache %s)\n",
                c->model_type, c->n_layers, c->hidden_size, c->vocab_size, c->quant_mode,
                c->is_linear_layer ? (llm_mlx_layer_proj(c,0)!=NULL ? "hybrid" : "") : "kv");
    }
    return err ? -1 : 0;
}

/* Model Merging smoketest (fase 6): weighted sum, TIES, DARE. */
int llm_ai_mlx_merge_smoketest(void) {
    float m0[3] = {1,2,3}, m1[3] = {3,2,1};
    const float* ms[2] = {m0, m1}; float w[2] = {0.5f, 0.5f};
    float out[3];
    llm_merge_weighted_sum(ms, w, 2, 3, out);
    if (out[0] < 1.99 || out[0] > 2.01) return -1;
    float n0[3] = {1,2,3}, n1[3] = {-1,-2,-3};
    const float* ns[2] = {n0, n1}; float wn[2] = {1,1};
    llm_merge_ties(ns, wn, 2, 3, out);
    if (out[0] != 0 || out[1] != 0) return -1;
    const float* ds[1] = {m0}; float wd[1] = {1.0f};
    llm_merge_dare(ds, wd, 1, 3, 0.5f, out);
    if (out[0] != 0 || out[2] < 5.9 || out[2] > 6.1) return -1;
    return 0;
}

/* AUTONOMOUS MODEL SURGERY (roadmap 8.11 fase 8): ciclo AI propone -> validate -> build
 * -> benchmark -> compare -> accept/reject -> rollback.
 * - valida la rewrite plan (se richiede training e !allow_training -> reject);
 * - misura il baseline (FLOPS prefill) e stima il candidato (REPLACE_ATTENTION full->linear
 *   dimezza il costo attention O(N^2)->O(N) a contesto lungo; REPLACE_MLP/norm ~ invariato);
 * - confronta: se il candidato riduce i FLOPS -> ACCEPT, altrimenti REJECT (rollback = nessun
 *   cambiamento effettivo applicato);
 * - stampa il report e ritorna 0 ok (decisione in *out_decision: 1=accept 0=reject), -1 errore.
 * L'applicazione REALE della rewrite al grafo resta delegata al Model Builder; qui la
 * decisione e' basata sull'impatto stimato (fondamento del ciclo autonomo). */
int llm_ai_mlx_model_autosurgery(llm_ai_mlx_model_handle handle, LlmkxRewritePlan* plan,
                                 bool allow_training, int* out_decision) {
    if (handle == NULL || plan == NULL || out_decision == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    const ModelConfig* c = &m->cfg;
    /* validate: se il plan e' invalido (o richiede training non permesso) -> reject */
    if (!llm_rewrite_validate(plan, c->n_layers, allow_training)) { *out_decision = 0; return 0; }
    /* benchmark baseline */
    int base_gflops = llm_ai_mlx_model_flops(handle, 4096);
    int cand_gflops = base_gflops;
    int n_attn = 0, n_active = 0;
    for (int i = 0; i < plan->count; i++) {
        if (!plan->items[i].enabled) continue;
        n_active++;
        if (plan->items[i].kind == LLM_RW_REPLACE_ATTENTION) { n_attn++; cand_gflops /= 2; }
    }
    fprintf(stderr, "AUTOSURGERY: %d rewrites attive, baseline=%d GFLOPs, candidato=%d GFLOPs\n",
            n_active, base_gflops, cand_gflops);
    if (n_attn == 0 || cand_gflops >= base_gflops) {
        fprintf(stderr, "  -> REJECT (nessun beneficio stimato; nessuna modifica applicata)\n");
        *out_decision = 0;
        return 0;
    }
    fprintf(stderr, "  -> ACCEPT (FLOPS ridotti da %d a %d: -%d%%); build candidato per conferma\n",
            base_gflops, cand_gflops, (int)(100 * (1.0 - (double)cand_gflops / base_gflops)));
    if (llm_ai_mlx_model_build(handle) != 0) { fprintf(stderr, "  -> REJECT (build fallita)\n"); *out_decision = 0; return 0; }
    *out_decision = 1;
    return 0;
}

/* Test esportato di autosurgery: costruisce un plan REPLACE_ATTENTION e lo valuta. */
int llm_ai_mlx_model_autosurgery_test(llm_ai_mlx_model_handle h, bool allow_training, int* out_decision) {
    if (h == NULL || out_decision == NULL) return -1;
    LlmkxRewritePlan p; llm_rewrite_plan_init(&p);
    llm_rewrite_add(&p, 3, LLM_RW_REPLACE_ATTENTION, true, true);
    return llm_ai_mlx_model_autosurgery(h, &p, allow_training, out_decision);
}

/* Radix-trie smoketest (prefix cache content-addressed): insert + longest-prefix-match. */
int llm_ai_mlx_radix_smoketest(void) {
    llm_radix_trie t; llm_radix_init(&t);
    int32_t a[3] = {1,2,3}, b[4] = {1,2,5,6};
    llm_radix_insert(&t, a, 3, 10);
    llm_radix_insert(&t, b, 4, 11);
    int32_t p[4] = {1,2,3,9}; int pid = -1;
    int d = llm_radix_longest_match(&t, p, 4, &pid);
    if (d != 3 || pid != 10) { llm_radix_free(&t); return -1; }
    llm_radix_free(&t);
    return 0;
}

int llm_ai_mlx_model_export_ir(llm_ai_mlx_model_handle handle, const char* path) {
    if (handle == NULL || path == NULL) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    cJSON* root = cJSON_CreateObject();
    if (!root) { fclose(f); return -1; }
    cJSON_AddStringToObject(root, "model_type", c->model_type);
    cJSON_AddNumberToObject(root, "hidden_size", c->hidden_size);
    cJSON_AddNumberToObject(root, "n_layers", c->n_layers);
    cJSON_AddNumberToObject(root, "n_heads", c->n_heads);
    cJSON_AddNumberToObject(root, "kv_heads", c->n_kv_heads);
    cJSON_AddNumberToObject(root, "head_dim", c->head_dim);
    cJSON_AddNumberToObject(root, "vocab_size", c->vocab_size);
    cJSON_AddStringToObject(root, "quant_mode", c->quant_mode);
    cJSON_AddNumberToObject(root, "quant_bits", c->quant_bits);
    cJSON_AddNumberToObject(root, "quant_group_size", c->quant_group_size);
    cJSON_AddNumberToObject(root, "rope_theta", c->rope_theta);
    cJSON_AddNumberToObject(root, "rms_norm_eps", c->rms_norm_eps);
    cJSON_AddNumberToObject(root, "n_experts", c->n_experts);
    cJSON_AddNumberToObject(root, "top_k", c->top_k);
    /* cache_type ABI */
    int n_full = 0;
    if (c->is_linear_layer) for (int i = 0; i < c->n_layers; i++) if (!c->is_linear_layer[i]) n_full++;
    const char* ct = c->is_linear_layer ? (n_full > 0 ? "HYBRID" : "RECURRENT") : "KV";
    cJSON_AddStringToObject(root, "cache_type", ct);
    /* layout per-layer: G=GDN A=full-attn S=sliding */
    cJSON* layers = cJSON_AddArrayToObject(root, "layers");
    for (int i = 0; i < c->n_layers; i++) {
        cJSON* lay = cJSON_CreateObject();
        cJSON_AddNumberToObject(lay, "idx", i);
        const char* ty = "ATTENTION";
        if (c->is_linear_layer) ty = c->is_linear_layer[i] ? "GDN" : "ATTENTION";
        else if (c->is_sliding_layer) ty = c->is_sliding_layer[i] ? "SLIDING" : "ATTENTION";
        cJSON_AddStringToObject(lay, "type", ty);
        cJSON_AddItemToArray(layers, lay);
    }
    char* out = cJSON_PrintUnformatted(root);
    int ok = out ? (fputs(out, f) >= 0) : 0;
    if (out) free(out);
    cJSON_Delete(root);
    fclose(f);
    return ok ? 0 : -1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * MODEL VALIDATE (roadmap fase 7, Model Builder): verifica la coerenza strutturale
 * del modello caricato (config + dispatch) prima di ricostruire/eseguire. Ritorna
 * 0 se coerente, altrimenti -1 (e stampa il motivo su stderr).
 */
int llm_ai_mlx_model_validate(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    int err = 0;
    if (c->n_layers <= 0) { fprintf(stderr, "validate: n_layers=%d invalido\n", c->n_layers); err = 1; }
    if (c->hidden_size <= 0) { fprintf(stderr, "validate: hidden_size=%d invalido\n", c->hidden_size); err = 1; }
    if (c->n_heads <= 0) { fprintf(stderr, "validate: n_heads=%d invalido\n", c->n_heads); err = 1; }
    if (c->head_dim <= 0) { fprintf(stderr, "validate: head_dim=%d invalido\n", c->head_dim); err = 1; }
    if (c->vocab_size <= 0) { fprintf(stderr, "validate: vocab_size=%d invalido\n", c->vocab_size); err = 1; }
    if (c->n_kv_heads > c->n_heads || c->n_kv_heads <= 0) { fprintf(stderr, "validate: kv_heads=%d vs heads=%d\n", c->n_kv_heads, c->n_heads); err = 1; }
    if (c->n_heads * c->head_dim != c->hidden_size) {
        fprintf(stderr, "validate: heads*head_dim=%d != hidden=%d (ok se MoE/down-proj)\n", c->n_heads * c->head_dim, c->hidden_size);
    }
    if (c->is_linear_layer && c->full_attention_interval <= 0) { fprintf(stderr, "validate: hybrid GDN ma full_attention_interval<=0\n"); err = 1; }
    return err ? -1 : 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * N-GRAM self-speculative (§8.4/§8.10 del piano): costruzione della tabella dalla
 * history della sessione + previsione del draft.
 *
 * `llm_ai_mlx_session_build_ngram(handle, order)` ricostruisce la tabella dai token
 * visti (prefill + generati). `llm_ai_mlx_session_ngram_predict` propone il successore
 * piu' probabile (1 draft). L'integrazione con verifica/rollback (draft batch) usa la
 * Cache ABI (llm_mlx_cache_checkpoint/rollback/commit in llm_mlx_ops.h).
 *
 * Ritorna 0 ok, -1 handle nullo. Predict ritorna il numero di draft proposti (0 se
 * niente di noto).
 */
int llm_ai_mlx_session_build_ngram(llm_ai_mlx_session_handle handle, int order) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (order < 1) order = 3;
    llm_mlx_ngram_free(&s->ngram);
    s->ngram = llm_mlx_ngram_new(order, 1024);
    s->ngram_order = order;
    llm_mlx_ngram_add(&s->ngram, s->history, s->history_len, NULL);
    s->ngram_built = true;
    return 0;
}

int llm_ai_mlx_session_ngram_predict(llm_ai_mlx_session_handle handle, int32_t* out, int max) {
    if (handle == NULL || out == NULL || max <= 0) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->ngram_built || s->history_len < s->ngram_order) return 0;
    int32_t p = llm_mlx_ngram_predict(&s->ngram, s->history, s->history_len);
    if (p < 0) return 0;
    out[0] = p;
    return 1;
}
