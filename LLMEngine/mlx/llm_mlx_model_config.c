/* ============================================================================
 * llm_mlx_model_config.c — generate one-shot, setter runtime (kv_max/kv_quantize/bf16), rms_norm_cfg, inspector (hidden/manifest/cache_type/kv_pool/layer_type).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llmengine_plugin.h"
#include "llm_dsv4_engine.h"   /* CAP_KV_MAX: DS4 ha il contesto nel SUO engine */

#include <stdio.h>
#include <string.h>

/* Handle del plugin primitiva MLX (rms_norm_cfg); definiti nel monolite. */
extern const LlmEnginePluginPrimMLX* g_primmlx_norm;
extern void* g_primmlx_norm_h;

/* Kept for callers that don't need a persistent session (a single one-shot
 * completion) -- thin sugar over create/append/generate/destroy, so there is
 * exactly ONE code path implementing the actual math, not two to keep in
 * sync (same principle llmengine_llama.cpp's Complete() already follows
 * on top of its own session primitives). */
int llm_ai_mlx_generate(llm_ai_mlx_model_handle handle, const int32_t* prompt_ids, int n_prompt_ids,
                         int32_t* out_ids, int max_new_tokens, int32_t eos_token_id) {
    if (handle == NULL || prompt_ids == NULL || n_prompt_ids <= 0 || out_ids == NULL || max_new_tokens <= 0) return -1;

    llm_ai_mlx_session_handle session = llm_ai_mlx_create_session(handle);
    if (session == NULL) return -1;

    int result = -1;
    if (llm_ai_mlx_session_append(session, prompt_ids, n_prompt_ids) == 0) {
        result = llm_ai_mlx_session_generate(session, max_new_tokens, eos_token_id, out_ids);
    }
    llm_ai_mlx_destroy_session(session);
    return result;
}

/* ── FASE 5 (RecursiveMAS R5 — latent nativo): hidden states reali ─────────────
 * pending_h è l'hidden [1,1,hidden_size] dell'ultimo token (pre-lm_head), mantenuto
 * vivo per la FASE 3 constrained decode. Si rilegge con cast esplicito a float32:
 * i tensori f16 letti via mlx_array_data_float32 reinterpretano i bit (bug reale
 * documentato in plain_linear) → mlx_astype PRIMA della copia. */

/* Setta a runtime il limite rolling-window KV (max token nella KV-cache full). 0=off. */
int llm_ai_mlx_model_set_kv_max(llm_ai_mlx_model_handle handle, int n) {
    if (handle == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    m->cfg.kv_max_tokens = n;
    /* DS4 (GGUF single-file): il contesto è allocato nel SUO engine -> rialloca le cache di
     * conseguenza (il KV si azzera: il prossimo turno ri-prefilla la cronologia). */
    if (m->dsv4 && n > 0) {
        LlmDsv4Engine* e = (LlmDsv4Engine*)m->dsv4;
        llm_dsv4_caches_free(&e->caches);
        int raw_cap = e->w.cfg.n_swa > 0 ? e->w.cfg.n_swa : 128;
        llm_dsv4_caches_init(&e->caches, &e->w.cfg, raw_cap, n);
        e->n_ctx = n;
        e->pos = 0;
    }
    return 0;
}

/* KV quant (E3): abilita/disabilita la quantizzazione della parte COLD della cache a contesto lungo.
 * threshold = token oltre cui quantizzare la parte piu' vecchia; bits = 4|8. Attiva la cache
 * quantizzata persistente (kvq_* per-layer) gestita da layer_forward. Default OFF. */
int llm_ai_mlx_model_set_kv_quantize(llm_ai_mlx_model_handle handle, bool enabled, int threshold, int bits) {
    if (handle == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    m->cfg.kv_quantize = enabled;
    m->cfg.kv_quantize_threshold = enabled ? (threshold > 0 ? threshold : 1024) : 0;
    m->cfg.kv_quantize_bits = (bits == 4 || bits == 8) ? bits : 8;
    return 0;
}

/* Matcha mlx_lm: bfloat16 per embed (riduce la differenza di precisione che il MoE amplifica). */
int llm_ai_mlx_model_set_bfloat16(llm_ai_mlx_model_handle handle, bool enabled) {
    if (handle == NULL) return -1;
    ((LlmMlxModel*)handle)->cfg.bfloat16_input = enabled;
    g_bf16_all = enabled; /* sincronizza il refactor bf16 dei layer (matmul/rms_norm) */
    /* La firma promette BFLOAT16, e g_act16 e' un globale che sopravvive al prossimo load nello
     * stesso processo: senza questo pin un modello caricato dopo un "f16" erediterebbe float16. */
    g_act16 = MLX_BFLOAT16;
    return 0;
}

/* Scelta esplicita del dtype di calcolo, per modello: "both" (embed+matmul), "embed",
 * "matmul", "bf16"/"f16"/"fp16" (both + il flavor a 16 bit), "off"/"fp32". E' la versione
 * stabile della diagnostica LLM_BF16_DENSE -- stessa
 * regola di ammissione (llm_mlx_apply_bf16_mode), ma invocata dall'host invece che da una
 * variabile d'ambiente, quindi l'opzione e' raggiungibile da un programma BASIC.
 * Va chiamata DOPO llm_ai_mlx_load_model: i due bit sono letti SOLO a forward-time
 * (siti: embed/rms_norm/matmul), nessun weight-conversion di load li consulta, quindi
 * applicarli dopo il load e' equivalente -- e vince sul default architetturale e
 * sull'ambiente proprio perche' arriva per ultimo.
 * Ritorno: 0 applicato, -1 rifiuto del motore (quant_mode non ammesso, o model_type=llama:
 * il perche' e' sempre su stderr), -2 handle/mode non validi. Mai un no-op silenzioso. */
int llm_ai_mlx_model_set_bf16_mode(llm_ai_mlx_model_handle handle, const char* mode) {
    if (handle == NULL) return -2;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    /* Convalida PRIMA della guardia architetturale: nell'ordine inverso un typo su llama
     * (misurato: bf16=banana) rispondeva "model_type=llama" e il chiamante non poteva
     * distinguere "argomento sbagliato" da "questa architettura e' vietata". */
    if (!llm_mlx_bf16_mode_known(mode)) {
        fprintf(stderr, "[bf16:config] mode \"%s\" non valida (attese: both|embed|matmul|bf16|f16|fp16|off|fp32)\n",
                mode ? mode : "(null)");
        return -2;
    }
    /* Guardia SOLO per la richiesta esplicita: l'A/B bf16 del 2026-09-08 ha MISURATO che la
     * combinazione dei due bit da' spazzatura silenziosa su llama (head_dim 64) e token identici
     * su qwen3_5 (head_dim 256) -> il difetto e' locale a llama_attention_block e non e' ancora
     * corretto. Offrire al chiamante quella combinazione significherebbe vendergli un'opzione che
     * corrompe l'output senza dire niente. LLM_BF16_DENSE resta libera: e' lo strumento con cui
     * quel bug si deve isolare, e passa dall'helper condiviso, non da qui. La guardia scatta solo
     * se la richiesta CAMBIA qualcosa: con LLM_BF16_DENSE gia' attivo, diniegare "both" vorrebbe
     * dire negare uno stato che il processo ha gia' -- e il messaggio punterebbe al flag che e'
     * proprio la causa dello stato. */
    const bool wants_bf16 = strcmp(mode, "off") != 0 && strcmp(mode, "fp32") != 0;
    if (wants_bf16 && !llm_mlx_bf16_bits_on(m, mode) && strcmp(m->cfg.model_type, "llama") == 0) {
        fprintf(stderr, "[bf16:config] rifiutato: model_type=llama + bf16 produce output errato in "
                        "modo silenzioso (misurato 2026-09-08, difetto locale a "
                        "llama_attention_block). Usa LLM_BF16_DENSE per indagare.\n");
        return -1;
    }
    return llm_mlx_apply_bf16_mode(m, mode, "config");
}

/* Diagnostica: attiva SOLO il matmul/rms_norm bf16 (g_bf16_all) senza toccare embed/router
 * (bfloat16_input). Per isolare la causa della divergenza bf16 sul MoE 35B (OptiQ 4bit). */
int llm_ai_mlx_model_set_bfloat16_matmul(llm_ai_mlx_model_handle handle, bool enabled) {
    if (handle == NULL) return -1;
    g_bf16_all = enabled;
    g_act16 = MLX_BFLOAT16; /* firma bf16: mai ereditare un flavor f16 attivo */
    return 0;
}

/* rms_norm + (bfloat16 refactor) cast di INPUT e PESO a bf16, come mlx_lm. Lezione §9:
 * mlx_fast_rms_norm richiede x e weight dello STESSO dtype; il peso del safetensors e' float32,
 * quindi serve convertire anche il peso a bf16 quando l'input e' bf16.
 * NB: niente mlx_array_eval qui (un eval per-norm = una sync GPU per norma che serializza il
 * batch): il grafo resta lazy, materializzato solo all'eval finale. */
bool rms_norm_cfg(mlx_array* res, mlx_array x, mlx_array w, float eps, mlx_stream stream) {
    /* Primitiva dalla .llmarch: se il plugin MLX e' presente, la usa (handle opachi, zero copie). */
    if (g_primmlx_norm != NULL && g_primmlx_norm->forward != NULL) {
        LlmEnginePluginPrimMLXIO io; memset(&io, 0, sizeof io);
        io.op = LLMENGINE_PLUGIN_PRIMMLX_OP_NORM;
        io.x = (void*)&x; io.w = (void*)&w; io.out = (void*)res; io.eps = eps; io.stream = (void*)&stream;
        if (g_primmlx_norm->forward(g_primmlx_norm_h, &io) == 0) return true;
    }
    if (!g_bf16_all) return mlx_fast_rms_norm(res, x, w, eps, stream) == 0;
    mlx_array xb = {0};
    mlx_array xuse = x;
    /* Nessun alias se `x` e' gia' nel dtype di attivazione (evita un `new mlx::core::array`
     * per norma): niente handle temp quando il cast non serve. */
    bool need_cast = (mlx_array_dtype(x) != g_act16);
    bool ok = true;
    if (need_cast) {
        ok = mlx_astype(&xb, x, g_act16, stream) == 0;
        if (ok) xuse = xb;
    }
    /* Chiamata DIRETTA su `res`: nessun handle temp `n` + set + free (erano 3 allocazioni
     * per norma). Il fix ownership d7984060d resta onorato (nessun handle sorgente di set). */
    if (ok) ok = mlx_fast_rms_norm(res, xuse, w, eps, stream) == 0;
    if (xb.ctx != NULL) mlx_array_free(xb);
    return ok;
}

int llm_ai_mlx_hidden_size(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    return m->cfg.hidden_size;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * MODEL INSPECTOR / MANIFEST (§8.1/§8.7 del piano long-context).
 *
 * Il modello non e' un oggetto opaco: stampa la descrizione strutturale per-layer
 * (Model Manifest), che i plugin consultano per dire «quali layer posso modificare»
 * anziché tentare di modificare tutto. Il tipo di cache (KV / RECURRENT / HYBRID)
 * e' l'informazione chiave della Cache ABI (§8.4): un ibrido GDN+attention ha
 * KV state + recurrent state, e un reject nel speculative deve ripristinare entrambi.
 *
 * Ritorna 0 ok, -1 handle nullo. Stampa su stderr.
 */
int llm_ai_mlx_model_manifest(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    fprintf(stderr, "MODEL manifest\n");
    fprintf(stderr, "  type=%s hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
            c->model_type, c->hidden_size, c->n_layers, c->n_heads, c->n_kv_heads,
            c->head_dim, c->vocab_size);
    fprintf(stderr, "  quant_mode=%s bits=%d group=%d rope_theta=%g rms_eps=%g tie_emb=%d\n",
            c->quant_mode, c->quant_bits, c->quant_group_size, c->rope_theta,
            c->rms_norm_eps, c->tie_word_embeddings);
    if (c->n_experts > 0)
        fprintf(stderr, "  MoE experts=%d top_k=%d moe_intermediate=%d norm_topk=%d\n",
                c->n_experts, c->top_k, c->moe_intermediate_size, c->norm_topk_prob);

    int n_gdn = 0, n_full = 0, n_sliding = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_linear_layer) { if (c->is_linear_layer[i]) n_gdn++; else n_full++; }
        else if (c->is_sliding_layer) { if (c->is_sliding_layer[i]) n_sliding++; else n_full++; }
        else n_full++;
    }
    if (c->is_linear_layer)
        fprintf(stderr, "  layers: %d GDN + %d full (full_attention_interval=%d)\n", n_gdn, n_full, c->full_attention_interval);
    else if (c->is_sliding_layer)
        fprintf(stderr, "  layers: %d sliding + %d full\n", n_sliding, n_full);
    else
        fprintf(stderr, "  layers: %d full attention\n", n_full);

    /* Cache ABI type: KV / RECURRENT / HYBRID (§8.4). */
    const char* ct = c->is_linear_layer ? ((n_full > 0) ? "HYBRID" : "RECURRENT") : "KV";
    fprintf(stderr, "  cache_type=%s\n", ct);

    /* Layout compatto per-layer: G=GDN A=full-attn S=sliding. */
    fprintf(stderr, "  layout: ");
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_linear_layer) fprintf(stderr, "%c", c->is_linear_layer[i] ? 'G' : 'A');
        else if (c->is_sliding_layer) fprintf(stderr, "%c", c->is_sliding_layer[i] ? 'S' : 'A');
        else fprintf(stderr, "A");
    }
    fprintf(stderr, "\n");
    return 0;
}

/* Ritorna la stringa del tipo di cache ABI ("KV" / "RECURRENT" / "HYBRID") in *out. */
int llm_ai_mlx_model_cache_type(llm_ai_mlx_model_handle handle, char* out, int cap) {
    if (handle == NULL || out == NULL || cap <= 0) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    int n_full = 0;
    if (c->is_linear_layer)
        for (int i = 0; i < c->n_layers; i++) if (!c->is_linear_layer[i]) n_full++;
    const char* s = c->is_linear_layer ? (n_full > 0 ? "HYBRID" : "RECURRENT") : "KV";
    llm_mlx_mode_set(out, (size_t)cap, s);
    return 0;
}

/* Statistiche della paged cache CONTENT-ADDRESSED (blocchi KV del prefisso). */
int llm_ai_mlx_model_kv_pool_stats(llm_ai_mlx_model_handle handle,
                                   long* out_allocs, long* out_evictions, long* out_reused) {
    if (handle == NULL) return -1;
    KvPagePool* p = &((LlmMlxModel*)handle)->kvpool;
    if (out_allocs) *out_allocs = p->allocs;
    if (out_evictions) *out_evictions = p->evictions;
    if (out_reused) {
        /* riusi content-addressed: pagine usate piu' di una volta come proprietario? misura
         * quanti blocchi sono stati riusati = allocs senza evictions di blocchi vivi */
        *out_reused = (p->allocs - p->evictions) < p->n_pages ? (p->allocs - p->evictions) : p->n_pages;
    }
    return 0;
}

/* Pinna i blocchi KV di un prefisso nel pool (mai evictabili — prefisso stabile). */
int llm_ai_mlx_model_kv_pin_prefix(llm_ai_mlx_model_handle handle, const char* name) {
    if (handle == NULL || name == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    int idx = -1;
    for (int i = 0; i < m->n_prefixes; i++) if (strcmp(m->prefixes[i].name, name) == 0) { idx = i; break; }
    if (idx < 0) return -1;
    for (int i = 0; i < m->kvpool.n_pages; i++) if (m->kvpool.pages[i].owner == idx) llm_kv_page_pin(&m->kvpool, i);
    return 0;
}

/* Ritorna il tipo del layer: 1=GDN/recurrent, 0=full attention, -1=err/out of range. */
int llm_ai_mlx_model_layer_type(llm_ai_mlx_model_handle handle, int layer) {
    if (handle == NULL) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    if (layer < 0 || layer >= c->n_layers) return -1;
    if (c->is_linear_layer) return c->is_linear_layer[layer] ? 1 : 0;
    return 0;
}
