/* ============================================================================
 * Dominio model lifecycle + ARCH_TABLE: ARCH_TABLE (model_type -> attn/ffn), find_arch_ops, llm_mlx_model_init, llm_mlx_model_destroy (free cache pesi/attn/ffn/packed/lora/bonsai/qwen4exp).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_ngram.h"
#include "llm_mlx_qwen4exp.h"
#include "llm_dsv4_engine.h"
#include "llm_dsv41_engine.h"
#include <cJSON.h>
#include "llm_mlx_bonsai_hadamard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
/* Fase 1b.0: "qwen3_moe" (comportamento invariato). Fase 1b.1: "qwen3" (Qwen3
 * non-MoE) aggiunta qui sotto, riusa qwen3_moe_attention_block invariata
 * (QK-norm/RoPE identici in mlx_lm/models/qwen3.py vs qwen3_moe.py) e
 * cambia solo l'FFN. "qwen2" (es. DeepSeek-R1-Distill-Qwen-32B): niente
 * QK-norm, bias reale su q/k/v_proj -- attenzione dedicata
 * (qwen2_attention_block), FFN denso riusato invariato (stessa formula
 * SwiGLU/naming di dense_swiglu_ffn, verificato contro mlx_lm/models/
 * qwen2.py). "gemma4": attenzione dedicata (gemma4_attention_block, Fase
 * 1b.4.3 -- scala fissa, K=V condiviso sui layer full, dimensioni/RoPE
 * per-tipo di layer) + FFN GeGLU dedicata (dense_geglu_ffn, Fase 1b.4.1).
 * "mixtral" (Fase 1b.3): attenzione GQA pura (mixtral_attention_block --
 * nessun QK-norm, nessun bias reale, scala standard 1/sqrt(head_dim)) + FFN
 * MoE dedicata (mixtral_ffn_block -- router non quantizzato, softmax SOLO sui
 * top-k selezionati, pesi degli esperti impilati a caricamento da
 * stack_mixtral_moe_experts perché il modello reale li tiene separati per
 * indice invece che in un unico tensore stacked come Qwen3-MoE).
 * "llama" (Fase 1b.2): attenzione GQA pura (llama_attention_block -- stessa
 * forma di mixtral_attention_block: nessun QK-norm, nessun bias reale, scala
 * standard) ma con RoPE "llama3" a 3 fasce di frequenza quando
 * cfg.use_llama3_rope (llm_mlx_llama3_rope_freqs), altrimenti RoPE default
 * (stesso branch pattern di gemma4_attention_block per il custom-freq RoPE) +
 * FFN denso SwiGLU riusato invariato (dense_swiglu_ffn, stesso naming tensori
 * di Qwen2/Qwen3 denso).
 * Le fasi successive aggiungono altre entry, mai toccano le esistenti. */
static const LlmMlxArchOps ARCH_TABLE[] = {
    { "qwen3_moe", qwen3_moe_attention_block, qwen3_moe_ffn_block },
    { "qwen3",     qwen3_moe_attention_block, dense_swiglu_ffn },
    { "qwen3_next", qwen3_next_attention_dispatch, qwen3_5_moe_ffn_block },
    { "qwen4_exp",  qwen4_exp_attention_dispatch, qwen4_exp_ffn_dispatch },
    { "qwen3_5",   qwen3_5_attention_block,  dense_swiglu_ffn }, /* FASE 1b.5: Qwen3.8-27B hybrid GDN (linear_attn 3:1) */
    { "prism_hadamard_qwen35", qwen3_5_attention_block, dense_swiglu_ffn }, /* Ternary-Bonsai 2 27B (PrismML):
      base qwen3_5 + pesi FOLDED in basis ruotata -> la de-rotazione delle attivazioni e' negli hook
      llm_bonsai_rotate (proiezioni per-proiezione + uscita embed). Vedi i commenti in fondo. */
    { "qwen3_5_moe", qwen3_5_attention_block, qwen3_5_moe_ffn_block }, /* FASE 1b.6: Qwen3.6 hybrid GDN + MoE + shared_expert */
    { "qwen2",     qwen2_attention_block,     dense_swiglu_ffn },
    { "qwen2vl",   qwen2_attention_block,     dense_swiglu_ffn }, /* Qwen2.5-Omni: parte TESTO = qwen2 (vision via llm_host_ai) */
    { "olmoe",     qwen3_moe_attention_block, qwen3_moe_ffn_block }, /* OLMoE-1B-7B: GQA + QK-norm + MoE (gate_inp + *_exps) */
    { "gemma4",    gemma4_attention_block,    dense_geglu_ffn },
    { "mixtral",   mixtral_attention_block,   mixtral_ffn_block },
    { "gpt_oss",   gpt_oss_attention_block,   gpt_oss_ffn_block }, /* GPT-OSS MoE (sinks + sliding 128 + SwiGLU speciale) */
    { "glm4_moe",  glm4_moe_attention_block,  glm4_moe_ffn_block }, /* GLM-4.5 MoE (128 routed + shared, gate sigmoid, primi layer densi) */
    { "deepseek_v2", deepseek_v2_attention_block, deepseek_v2_ffn_block }, /* DeepSeek-V2-Lite (MLA + MoE group-limited-greedy) */
    { "deepseek_v3", deepseek_v2_attention_block, deepseek_v2_ffn_block }, /* DeepSeek-V3 (MLA, q_lora) */
    { "kimi_k2", deepseek_v2_attention_block, deepseek_v2_ffn_block },     /* Kimi K2 = deepseek_v3 */
    { "joyai_llm_flash", deepseek_v2_attention_block, deepseek_v2_ffn_block }, /* = deepseek_v3 */
    { "kimi_linear", kimi_linear_attention_dispatch, kimi_linear_ffn_dispatch }, /* MLA assorbita + KDA */
    { "llama",     llama_attention_block,     dense_swiglu_ffn },
    { "mistral3",  llama_attention_block,     dense_swiglu_ffn }, /* Ministral-3-14B: Mistral denso (GQA + SwiGLU) */
    /* DeepSeek V4.1: forward whole-model (mHC + CSA2 + Engram) con cache 7 slot; attn/ffn NULL,
     * dispatchato per model_type in load/session_append/session_generate. */
    { "deepseek_v41", NULL, NULL, (void*)1 },
    /* DS4 (DeepSeek V4 Flash): forward whole-model (CSA/HCA + mHC + MoE + cache KV) da GGUF
     * singolo; attn/ffn NULL, dispatchato per model_type in load/session_append/session_generate. */
    { "deepseek_v4", NULL, NULL, (void*)1 },
};
#define ARCH_TABLE_COUNT (sizeof(ARCH_TABLE) / sizeof(ARCH_TABLE[0]))

const LlmMlxArchOps* find_arch_ops(const char* model_type) {
    for (size_t i = 0; i < ARCH_TABLE_COUNT; i++)
        if (strcmp(ARCH_TABLE[i].model_type, model_type) == 0) return &ARCH_TABLE[i];
    return NULL;
}

/* Prefix cache (paragrafo 8.4/8.10 del piano): stato di prefill salvato di un prefisso

 * riusabile tra sessioni. Le cache sono alias di array MLX (refcount): la sessione che le

 * carica non le possiede, il prefisso le libera. */

/* Imatrix (activation importance) collector — port del concetto DS4/DwarfStar
 * (gguf-tools/imatrix): accumula per (layer, routed expert, colonna) l'energia di
 * attivazione REALE del grafo. gate = input FFN-normato (colonne = hidden), down =
 * riga SwiGLU dopo route weighting (colonne = moe_intermediate). Serve a pesare la
 * quantizzazione aggressiva dei routed experts: le colonne poco attive si possono
 * quantizzare male senza costo. Offline/calibration: attivo solo via API dedicata. */

/* Cache pesi+quant del FFN MoE per-layer (bug P1/P3 2026-09-03: i pesi venivano risolti per
 * nome e la quant derivata a OGNI forward). Compilata lazy al primo forward del layer: i campi
 * sono alias (refcount) delle shard; la struttura NON viene liberata nel forward (solo al destroy). */

/* Cache pesi dell'ATTENZIONE per-layer (P1/P3): norm + proiezioni q/k/v/o (+ QK-norm best-effort)
 * risolte una volta al load per i blocchi con naming standard self_attn.*. Compilata lazy. */

/* Cache pesi FFN DENSO (swiglu/geglu): post_attention_layernorm + mlp.gate/up/down_proj.
 * Prima erano 4 tname+lookup a OGNI layer a OGNI token (su Llama-1B: 64 snprintf + ~192
 * refcount per token), senza cache: l'attenzione ha attn_quant, il MoE ha ffn_quant, il
 * denso no. Lazy per-layer, trattenuta dal modello, liberata in llm_mlx_model_destroy.
 * LLM_FFN_WCACHE=0 -> kill-switch (lookup per chiamata = comportamento pre-fix). */

/* Memo NOME->array per-layer per gpt-oss (attenzione + ffn): 20 lookup/layer/token
 * (tname + multishard_get*), misurato ~2 us ciascuno (gemma-4: 240 lookup = -0,50 ms).
 * Chiave = NOME COMPLETO (anche per i triple, che usano .weight/.scales/.biases): un solo
 * store per tutto. Costruita al primo forward (i pesi sono immutabili dopo il load),
 * trattenuta dal modello, liberata in destroy. LLM_GPTOSS_WCACHE=0 -> kill-switch. */


void llm_mlx_model_init(LlmMlxModel* m) {
    memset(m, 0, sizeof(*m));
    m->device = mlx_device_new_type(MLX_CPU, 0);
    m->stream = mlx_stream_new();
    m->prefix_max = 16; /* default: eviction LRU oltre 16 prefissi salvati */
    llm_radix_init(&m->radix);
    m->kvpool = llm_kv_page_pool_new(/*n_pages=*/1024, /*block_size=*/16);
    m->kvdisk_state = -1; /* §G: tier SSD KV non inizializzato (lazy su LLM_KVDISK) */
    m->kvckpt_block = 256; /* §G-ter: checkpoint del rewind ogni 256 token (LLM_KVCHECKPOINT_BLOCK) */
    { const char* kb = llm_ai_mlx_opt_env("LLM_KVCHECKPOINT_BLOCK"); if (kb) m->kvckpt_block = atoi(kb); }
    /* §G-bis leva 1 — sparse page-level attention (default OFF) */
    m->pagesel_on = (llm_ai_mlx_opt_env("LLM_PAGESEL") != NULL);
    m->ps_page = 64; m->ps_topk = 8; m->ps_window = 2; m->ps_min = 1024; m->ps_cap = 512; m->ps_verify = 0;
    { const char* e;
      if ((e = llm_ai_mlx_opt_env("LLM_PAGESEL_PAGE"))) m->ps_page = atoi(e);
      if ((e = llm_ai_mlx_opt_env("LLM_PAGESEL_TOPK"))) m->ps_topk = atoi(e);
      if ((e = llm_ai_mlx_opt_env("LLM_PAGESEL_WINDOW"))) m->ps_window = atoi(e);
      if ((e = llm_ai_mlx_opt_env("LLM_PAGESEL_MIN"))) m->ps_min = atoi(e);
      if ((e = llm_ai_mlx_opt_env("LLM_PAGESEL_CAP"))) m->ps_cap = atoi(e);
      if ((e = llm_ai_mlx_opt_env("LLM_PAGESEL_VERIFY"))) m->ps_verify = 1;
      if (m->ps_page < 1) m->ps_page = 1;
      if (m->ps_cap < 4) m->ps_cap = 4;
    }
    m->ngram_fd = -1; /* 0 (da memset) sarebbe stdin -- deve restare "non caricato" finche' non aperto */
    for (int q = 0; q < 16; q++) m->ngram_file[q].fd = -1; /* idem: 0 sarebbe stdin */
}

void llm_mlx_model_destroy(LlmMlxModel* m) {
    /* Cache pesi FFN MoE: alias delle shard — liberare PRIMA delle shard (i handle tengono vivi
     * gli array finche' la cache esiste). */
    if (m->dsv41) { llm_dsv41_engine_free((LlmDsv41Engine*)m->dsv41); m->dsv41 = NULL; }
    if (m->bonsai_had) { llm_bonsai_hadamard_free(m->bonsai_had); free(m->bonsai_had); m->bonsai_had = NULL; }
    if (m->dsv4) { llm_dsv4_engine_free((LlmDsv4Engine*)m->dsv4); m->dsv4 = NULL; }
    if (m->ffn_quant) {
        for (int l = 0; l < m->ffn_quant_n; l++) {
            GmoeFfnQuant* c = &m->ffn_quant[l];
            if (!c->valid) continue;
            if (llm_env_get("LLM_WDEBUG")) fprintf(stderr, "[wcache] destroy free layer %d\n", l);
            mlx_array_free(c->ln2); mlx_array_free(c->rw); mlx_array_free(c->rs); mlx_array_free(c->rb);
            mlx_array_free(c->gw); mlx_array_free(c->gs); mlx_array_free(c->gb);
            mlx_array_free(c->uw); mlx_array_free(c->us); mlx_array_free(c->ub);
            mlx_array_free(c->dw); mlx_array_free(c->ds); mlx_array_free(c->db);
            if (c->has_shared) {
                mlx_array_free(c->se_w); mlx_array_free(c->se_s); mlx_array_free(c->se_b);
                mlx_array_free(c->su_w); mlx_array_free(c->su_s); mlx_array_free(c->su_b);
                mlx_array_free(c->sd_w); mlx_array_free(c->sd_s); mlx_array_free(c->sd_b);
                mlx_array_free(c->sg_w); mlx_array_free(c->sg_s); mlx_array_free(c->sg_b);
            }
        }
        free(m->ffn_quant);
        m->ffn_quant = NULL;
    }
    multishard_free_all(&m->weights);
    if (m->attn_quant) {
        for (int l = 0; l < m->attn_quant_n; l++) {
            GattnQuant* c = &m->attn_quant[l];
            llm_mlx_qkv_destroy(&c->packed_qkv);
            if (!c->valid) continue;
            mlx_array_free(c->ln1);
            mlx_array_free(c->qw); mlx_array_free(c->qs); mlx_array_free(c->qb);
            mlx_array_free(c->kw); mlx_array_free(c->ks); mlx_array_free(c->kb);
            mlx_array_free(c->vw); mlx_array_free(c->vs); mlx_array_free(c->vb);
            mlx_array_free(c->ow); mlx_array_free(c->os); mlx_array_free(c->ob);
            if (c->has_qn) { mlx_array_free(c->qn_w); mlx_array_free(c->kn_w); }
        }
        free(m->attn_quant);
        m->attn_quant = NULL;
    }
    if (m->gdn_cw) {
        for (int l = 0; l < m->gdn_cw_n; l++) mlx_array_free(m->gdn_cw[l]);
        free(m->gdn_cw);
        m->gdn_cw = NULL;
    }
    if (m->gdn_ae) {
        for (int l = 0; l < m->gdn_ae_n; l++) mlx_array_free(m->gdn_ae[l]);
        free(m->gdn_ae);
        m->gdn_ae = NULL;
    }
    if (m->gdn_dt) {
        for (int l = 0; l < m->gdn_dt_n; l++) mlx_array_free(m->gdn_dt[l]);
        free(m->gdn_dt);
        m->gdn_dt = NULL;
    }
    if (m->qsa_indexer_cache) {
        for (int l = 0; l < m->qsa_indexer_cache_n; l++) llm_mlx_qwen4exp_qsa_cache_free(&m->qsa_indexer_cache[l]);
        free(m->qsa_indexer_cache);
        m->qsa_indexer_cache = NULL;
    }
    if (m->ngram_table_loaded && m->ngram_fd >= 0) {
        close(m->ngram_fd);
        m->ngram_fd = -1;
        m->ngram_table_loaded = false;
    }
    for (int q = 0; q < m->ngram_nfiles; q++) {
        if (m->ngram_file[q].fd >= 0) { close(m->ngram_file[q].fd); m->ngram_file[q].fd = -1; }
    }
    m->ngram_nfiles = 0;
    m->ngram_sharded = 0;
    if (m->ple_conv_cache) {
        for (int l = 0; l < m->ple_conv_cache_n; l++) mlx_array_free(m->ple_conv_cache[l]);
        free(m->ple_conv_cache);
        m->ple_conv_cache = NULL;
    }
    mlx_stream_free(m->stream);
    mlx_device_free(m->device);
    free(m->cfg.is_sliding_layer); /* FASE 1b.4.0: NULL per architetture non-Gemma4, free(NULL) è un no-op */
    free(m->cfg.is_linear_layer);  /* FASE 1b.5: NULL per architetture senza GDN */
    free(m->cfg.ple_layer_ids);    /* qwen4_exp: NULL se assente/vuoto */
    /* Prefix cache: libera i prefissi salvati (cache alias, token, history). */
    for (int i = 0; i < m->n_prefixes; i++) {
        SavedPrefix* p = &m->prefixes[i];
        for (int l = 0; l < p->n_layers; l++) { mlx_array_free(p->k_cache[l]); mlx_array_free(p->v_cache[l]); }
        free(p->k_cache); free(p->v_cache); free(p->has_cache);
        free(p->tokens); free(p->history);
    }
    free(m->prefixes);
    llm_radix_free(&m->radix);
    llm_kv_page_pool_free(&m->kvpool);
    if (m->kvdisk_state == 1) llm_kv_disk_close(&m->kvdisk); /* §G: chiude/rimuove lo scratch SSD */
    /* LoRA moduli */
    for (int i = 0; i < m->lora_n; i++) { mlx_array_free(m->lora_A[i]); mlx_array_free(m->lora_B[i]); }
    free(m->lora_A); free(m->lora_B); free(m->lora_layer); free(m->lora_proj); free(m->lora_scale); free(m->lora_expert);
    /* Imatrix collector */
    if (m->im.active) {
        free(m->im.gate); free(m->im.down); free(m->im.cnt);
        free(m->im.attn_qkv); free(m->im.attn_o); free(m->im.attn_cnt);
        m->im.gate = m->im.down = NULL; m->im.cnt = NULL;
        m->im.attn_qkv = NULL; m->im.attn_o = NULL; m->im.attn_cnt = NULL;
        m->im.active = false;
    }
    if (m->gptoss_we != NULL) {
        for (int l = 0; l < m->gptoss_we_n; l++) {
            for (int i = 0; i < m->gptoss_we[l].n; i++) {
                if (m->gptoss_we[l].e[i].a.ctx != NULL) mlx_array_free(m->gptoss_we[l].e[i].a);
            }
            free(m->gptoss_we[l].e);
        }
        free(m->gptoss_we);
        m->gptoss_we = NULL;
        m->gptoss_we_n = 0;
    }
    if (m->llama3_freqs.ctx != NULL) mlx_array_free(m->llama3_freqs);
    m->llama3_freqs = mlx_array_new();
    if (m->dense_ffn != NULL) {
        for (int l = 0; l < m->dense_ffn_n; l++) {
            GdenseFfnQuant* c = &m->dense_ffn[l];
            mlx_array_free(c->ln2);
            mlx_array_free(c->gate_w); mlx_array_free(c->gate_s); mlx_array_free(c->gate_b);
            mlx_array_free(c->up_w); mlx_array_free(c->up_s); mlx_array_free(c->up_b);
            mlx_array_free(c->down_w); mlx_array_free(c->down_s); mlx_array_free(c->down_b);
        }
        free(m->dense_ffn);
        m->dense_ffn = NULL;
        m->dense_ffn_n = 0;
    }
}
