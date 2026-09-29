/* ============================================================================
 * Dominio session lifecycle + embed/forward: embed_token, forward_one_token, load_output_tensors, free_output_tensors, llm_mlx_session_sampling, llm_mlx_session_create, llm_mlx_session_destroy, session_history_append.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_ngram.h"
#include "llm_mlx_qwen4exp.h"
#include "llm_dsv4_engine.h"
#include "llm_dsv41_engine.h"
#include "llm_pflash.h"
#include "llm_kvmetal_bridge.h"
#include "llm_gemvq4_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Guardie multi-sessione LLMMETAL_KVMETAL/LLM_GEMVQ4 (usate solo in session_create/_destroy;
 * lo stato ombra condiviso vive nel monolite ed e' extern in llm_mlx_internal.h). */
static int g_kvmetal_act = 0;
static int g_gemvq4_act = 0;
static bool embed_token(LlmMlxModel* m, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                         int32_t token_id, mlx_array* out_x) {
    const bool quantized = (embed_s.ctx != NULL);
    int e_group = m->cfg.quant_group_size, e_bits = m->cfg.quant_bits;
    char e_mode[16];
    llm_mlx_mode_set(e_mode, sizeof(e_mode), m->cfg.quant_mode);
    if (quantized)
        llm_mlx_quant_derive(embed_w, embed_s, e_group, e_bits, e_mode, &e_group, &e_bits, e_mode, sizeof(e_mode));
    const int idx_shape[1] = {1};
    mlx_array indices = mlx_array_new_data(&token_id, idx_shape, 1, MLX_INT32);
    mlx_array w_row = mlx_array_new(), s_row = mlx_array_new(), b_row = mlx_array_new();
    bool g = mlx_take_axis(&w_row, embed_w, indices, 0, m->stream) == 0;
    if (g && quantized) {
        g = mlx_take_axis(&s_row, embed_s, indices, 0, m->stream) == 0;
        if (g && embed_b.ctx != NULL && mlx_array_size(embed_b) > 0)
            g = mlx_take_axis(&b_row, embed_b, indices, 0, m->stream) == 0;
    }
    mlx_array x_flat = mlx_array_new();
    bool ok2 = g;
    if (ok2) {
        if (quantized) {
            mlx_optional_int gsz = {e_group, true};
            mlx_optional_int bts = {e_bits, true};
            mlx_optional_dtype dt = {llm_mlx_act_dt(m->cfg.bfloat16_input), true};
            ok2 = mlx_dequantize(&x_flat, w_row, s_row, b_row, gsz, bts, e_mode, g_mlx_empty_arr, dt, m->stream) == 0;
        } else {
            /* Bug reale trovato dal vivo (Mixtral): embed_tokens.weight non quantizzato
             * è salvato in F16 nel safetensors reale -- senza questo cast esplicito a
             * float32, ogni lettura host successiva (mlx_array_data_float32) reinterpreta
             * i bit float16 come float32 (stride sbagliato), producendo valori quasi-zero
             * plausibili ma completamente sbagliati (osservato dal vivo: tutti gli 8
             * valori letti ~0.000000 invece dei veri pesi di embedding). */
            ok2 = mlx_astype(&x_flat, w_row, llm_mlx_act_dt(m->cfg.bfloat16_input), m->stream) == 0;
        }
    }
    if (ok2) {
        const int x_shape[3] = {1, 1, m->cfg.hidden_size};
        ok2 = mlx_reshape(out_x, x_flat, x_shape, 3, m->stream) == 0;
        
        /* Ternary-Bonsai: l'embed e' in inverse_weight_names -> si ruota l'USCITA (invert=true).
         * La direzione (R vs R^T) e' ambigua dal naming: la decide il test empirico (NLL). */
        const int emb_inv = (llm_ai_mlx_opt_env("LLM_BONSAI_EMBED_INV") != NULL) ? (atoi(llm_ai_mlx_opt_env("LLM_BONSAI_EMBED_INV")) != 0 ? 1 : 0) : 1;
        ok2 = ok2 && llm_bonsai_rotate(out_x, m, -1, "embed_tokens.weight", emb_inv == 1);
    }
    /* FASE 1b.4.5 (Gemma4): embed_scale = sqrt(hidden_size), sempre (gemma4_text.py:402).
     * NOTA lifetime: mlx_array_set() ASSEGNA sull'array esistente (*out_x = scaled, refcount
     * C++ condiviso) -- NON chiamare mlx_array_free(*out_x) prima: mlx_array_free_ cancella
     * il ctx ma NON azzera l'handle, quindi un set successivo scriverebbe in memoria liberata
     * (use-after-free osservato). scaled va liberato DOPO il set (il set copia il valore). */
    if (ok2 && m->cfg.embed_scale > 0.0f) {
        mlx_array scaled = mlx_array_new();
        mlx_array scl = mlx_array_new_float32(m->cfg.embed_scale);
        ok2 = mlx_multiply(&scaled, *out_x, scl, m->stream) == 0;
        mlx_array_free(scl);
        if (ok2) mlx_array_set(out_x, scaled);
        mlx_array_free(scaled);
    }
    /* Qwen4-Exp: l'intero modello lavora su stati "tileati" hc_count volte
     * (Qwen4ExpTextModel.__call__: `mx.tile(embed, (1,1,hc_count))`) -- fatto qui, una sola
     * volta, prima del primo layer; ogni attn/ffn di qwen4_exp riceve e ritorna hc_width. */
    if (ok2 && strcmp(m->cfg.model_type, "qwen4_exp") == 0 && m->cfg.hc_count > 1) {
        mlx_array tiled = mlx_array_new();
        const int reps[3] = {1, 1, m->cfg.hc_count};
        ok2 = mlx_tile(&tiled, *out_x, reps, 3, m->stream) == 0;
        if (ok2) mlx_array_set(out_x, tiled);
        mlx_array_free(tiled);
    }
    mlx_array_free(indices);
    mlx_array_free(w_row);
    mlx_array_free(s_row);
    mlx_array_free(b_row);
    mlx_array_free(x_flat);
    return ok2;
}

/* Passa `token_id` attraverso tutti i layer, aggiornando k_cache/v_cache (di
 * proprieta' del chiamante -- una LlmMlxSession per le chiamate FASE 2b, o
 * array locali per llm_ai_mlx_generate). Ritorna l'hidden state finale
 * [1,1,hidden] in *out_h. k_cache/v_cache/has_cache: array di lunghezza
 * m->cfg.n_layers. */
bool forward_one_token(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                               int32_t token_id, int cache_offset,
                               mlx_array* k_cache, mlx_array* v_cache, bool* has_cache, mlx_array* out_h) {
    mlx_array x = mlx_array_new();
    if (!embed_token(m, embed_w, embed_s, embed_b, token_id, &x)) {
        mlx_array_free(x);
        return false;
    }
    mlx_array h = x;
    bool step_ok = true;
    for (int layer = 0; step_ok && layer < m->cfg.n_layers; layer++) {
        mlx_array layer_out = mlx_array_new();
        step_ok = layer_forward(m, kvq_sess, layer, cache_offset, /*n_tokens=*/1, h, &k_cache[layer], &v_cache[layer], has_cache[layer], &layer_out);
        has_cache[layer] = true;
        mlx_array_free(h);
        h = layer_out;
    }
    if (step_ok) mlx_array_set(out_h, h);
    mlx_array_free(h);
    return step_ok;
}

/* Fase 1 continuous batching (2026-09-11): forward di N SESSIONI (un token ciascuna) in un
 * colpo. Attention PER-SESSIONE (loop, KV per-sessione: ogni token attende solo alla propria
 * cache) + FFN/proiezioni BATCHED (N token -> 1 GEMM, ammortizza la lettura dei pesi).
 * x/out: [1, N, hidden]. LLM_BATCH_DECODE=1 (default OFF). */

/* Le tre triple quantizzate necessarie per andare da hidden state a logits:
 * embedding (sempre), e out_w/s/b che ALIASANO embed_w/s/b se il modello usa
 * tie_word_embeddings (nessun lm_head separato). */

bool load_output_tensors(LlmMlxModel* m, OutputTensors* t) {
    memset(t, 0, sizeof(*t));
    /* FASE 1b.4 (Gemma4): modello multimodale, il sotto-modello testuale vive sotto
     * "language_model.model.*" invece di "model.*" -- verificato sull'elenco tensori
     * reale (gemma-4-31b-it-4bit). "lm_head" resta senza prefisso in entrambi i casi
     * (quando esiste -- per Gemma4, tie_word_embeddings=true, non esiste affatto). */
    const char* prefix = (strncmp(m->cfg.model_type, "gemma4", 6) == 0
                          || strcmp(m->cfg.model_type, "qwen3_5") == 0
                          || strcmp(m->cfg.model_type, "qwen3_5_moe") == 0
                          || strcmp(m->cfg.model_type, "qwen4_exp") == 0
                          || strcmp(m->cfg.model_type, "prism_hadamard_qwen35") == 0) ? "language_model.model." : "model.";
    char embed_name[128], norm_name[128];
    snprintf(embed_name, sizeof(embed_name), "%sembed_tokens", prefix);
    snprintf(norm_name, sizeof(norm_name), "%snorm.weight", prefix);

    t->embed_w = mlx_array_new(); t->embed_s = mlx_array_new(); t->embed_b = mlx_array_new();
    bool ok = multishard_get_triple(&m->weights, embed_name, &t->embed_w, &t->embed_s, &t->embed_b);
    if (!ok) {
        /* FASE 1b.3 (Mixtral): embed_tokens può non essere quantizzato (verificato:
         * Mixtral-8x22B-4bit ha solo ".weight" per embed_tokens, nessuna ".scales"/
         * ".biases" accanto -- a differenza di ogni altro tensore del layer e di
         * embed_tokens in ogni altro modello reale verificato finora, tutti
         * quantizzati). t->embed_s/t->embed_b restano ctx==NULL (mai riassegnati
         * qui sotto) -- embed_token/embed_tokens_batch usano questo per saltare la
         * dequantizzazione, stesso schema "may be null" già documentato altrove in
         * questo file per mlx_array opzionali. */
        char plain_name[160];
        snprintf(plain_name, sizeof(plain_name), "%s.weight", embed_name);
        ok = multishard_get(&m->weights, plain_name, &t->embed_w);
    }
    if (!ok) fprintf(stderr, "load_output_tensors: %s lookup fallito (né quantizzato né float)\n", embed_name);

    t->final_norm_w = mlx_array_new();
    ok = ok && multishard_get(&m->weights, norm_name, &t->final_norm_w);
    if (!ok && strcmp(m->cfg.model_type, "qwen4_exp") == 0) {
        /* Flash-Next: la RMSNorm finale del modello vive nel top-level hyper_connection_mixer. */
        ok = multishard_get(&m->weights, "language_model.model.hyper_connection_mixer.hc_norm.weight", &t->final_norm_w);
    }
    if (!ok) fprintf(stderr, "load_output_tensors: %s lookup fallito\n", norm_name);

    mlx_array lm_w = mlx_array_new(), lm_s = mlx_array_new(), lm_b = mlx_array_new();
    /* Qwen3.8-27B: lm_head vive sotto language_model.lm_head (verificato). */
    char lm_name[64];
    snprintf(lm_name, sizeof(lm_name), "%s",
             (strcmp(m->cfg.model_type, "qwen3_5") == 0 || strcmp(m->cfg.model_type, "qwen3_5_moe") == 0
              || strcmp(m->cfg.model_type, "qwen4_exp") == 0
              || strcmp(m->cfg.model_type, "prism_hadamard_qwen35") == 0)
                 ? "language_model.lm_head" : "lm_head");
    /* Fallback a catena per lm_head, specchio del caso embed (alcuni checkpoint NON hanno la
     * tripletta quantizzata completa):
     *   1. quantizzato completo (weight+scales+biases)  [gpt_oss-20b, qwen3_5-moe quantizzati]
     *   2. quantizzato senza bias (weight+scales)        [Qwen3.5-122B-A10B-mxfp4]
     *   3. plain (solo weight)                            [gpt-oss-120b-mxfp4-bf16: lm_head bf16]
     * out_s/out_b vuoti (ctx==NULL) => quantized_linear_cfg instrada a plain_linear (fix sopra). */
    t->have_lm_head = multishard_get_triple(&m->weights, lm_name, &lm_w, &lm_s, &lm_b);
    if (!t->have_lm_head) {
        char n1[160], n2[160];
        snprintf(n1, sizeof(n1), "%s.weight", lm_name);
        snprintf(n2, sizeof(n2), "%s.scales", lm_name);
        bool has_w = multishard_get(&m->weights, n1, &lm_w);
        bool has_s = multishard_get(&m->weights, n2, &lm_s);
        t->have_lm_head = has_w && (has_s || !m->cfg.tie_word_embeddings);
        if (!has_s) mlx_array_free(lm_s); /* ctx==NULL vuoto -> plain */
        if (!has_w) mlx_array_free(lm_w);
    }
    if (!t->have_lm_head && !m->cfg.tie_word_embeddings) {
        fprintf(stderr, "load_output_tensors: lm_head.* assente e tie_word_embeddings=false -- impossibile procedere\n");
        ok = false;
    }
    t->out_w = t->have_lm_head ? lm_w : t->embed_w;
    t->out_s = t->have_lm_head ? lm_s : t->embed_s;
    t->out_b = t->have_lm_head ? lm_b : t->embed_b;
    return ok;
}

void free_output_tensors(OutputTensors* t) {
    mlx_array_free(t->embed_w);
    mlx_array_free(t->embed_s);
    mlx_array_free(t->embed_b);
    mlx_array_free(t->final_norm_w);
    if (t->have_lm_head) {
        mlx_array_free(t->out_w);
        mlx_array_free(t->out_s);
        mlx_array_free(t->out_b);
    }
}


/* Deriva i parametri di sampling dalla sessione (per final_step). Deve vivere
 * DOPO la definizione di LlmMlxSession (typedef anonimo, niente forward). */
llm_mlx_sampling_t llm_mlx_session_sampling(const LlmMlxSession* s) {
    llm_mlx_sampling_t sm = {0};
    if (s != NULL && s->sampling_set) {
        sm.active = true;
        sm.temperature = s->sampling_temperature;
        sm.top_p = s->sampling_top_p;
        sm.top_k = s->sampling_top_k;
        sm.seed = s->sampling_seed;
        sm.repetition_penalty = s->sampling_repetition_penalty;
        sm.frequency_penalty = s->sampling_frequency_penalty;
        sm.presence_penalty = s->sampling_presence_penalty;
        sm.hist = s->history;
        sm.hist_n = s->history_len;
    }
    return sm;
}

LlmMlxSession* llm_mlx_session_create(LlmMlxModel* m) {
    LlmMlxSession* s = (LlmMlxSession*)calloc(1, sizeof(LlmMlxSession));
    s->model = m;
    s->n_layers = m->cfg.n_layers;
    s->k_cache = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
    s->v_cache = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
    s->aux_cache = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
    s->has_cache = (bool*)calloc((size_t)m->cfg.n_layers, sizeof(bool));
    /* Ogni elemento dalla propria chiamata (stessa disciplina gia' documentata nell'originale:
     * un ciclo di assegnazioni con mlx_array_new() ripetuto, MAI un fill con lo stesso valore
     * copiato -- un bug reale di doppio-rilascio/uso-dopo-libero fu trovato con lldb durante
     * la verifica di Fase 1 quando due slot condividevano lo stesso mlx_array). */
    for (int i = 0; i < m->cfg.n_layers; i++) {
        s->k_cache[i] = mlx_array_new();
        s->v_cache[i] = mlx_array_new();
        s->aux_cache[i] = mlx_array_new();
    }
    /* KV quant: alloco per-layer SOLO se kv_quantize attivo (default OFF -> nessun overhead).
     * KV-quant PER-SESSIONE: lo stato kvq_* vive sulla sessione e viene passato esplicitamente
     * a layer_forward (param kvq_sess): piu' sessioni KV-quant possono coesistere senza
     * collisioni (rimossi la globale g_kvq e la guardia multi-sessione/fallback FP16). */
    /* NB: lo stato kvq va allocato anche se la KV-quant si attivera' DOPO via auto-policy
     * (LLM_KVQUANT_AUTO, latch nel prefill): altrimenti kvq_bits resta 0 e mlx_quantize
     * fallisce con "bits 0 is not supported" al primo step di decode (bug 2026-09-19). */
    const char* kvq_auto_env = llm_ai_mlx_opt_env("LLM_KVQUANT_AUTO");
    if (m->cfg.kv_quantize || (kvq_auto_env != NULL && atoi(kvq_auto_env) > 0)) {
        s->kvq_len = (int*)calloc((size_t)m->cfg.n_layers, sizeof(int));
        s->kvq_w = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
        s->kvq_sc = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
        s->kvq_bi = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
        s->kvq_bits = m->cfg.kv_quantize_bits > 0 ? m->cfg.kv_quantize_bits : 8;
        s->kvq_v_bits = s->kvq_bits;
        s->kvq_v_len = NULL; s->kvq_v_w = NULL; s->kvq_v_sc = NULL; s->kvq_v_bi = NULL;
        /* HETERO-KV: LLM_KV_HETERO=K<bits>V<bits> (es. K8V4). K piu' preciso di V. */
        { const char* het = llm_ai_mlx_opt_env("LLM_KV_HETERO");
          if (het != NULL) {
            int kb = 0, vb = 0;
            if (sscanf(het, "K%dV%d", &kb, &vb) == 2 && kb >= 2 && kb <= 16 && vb >= 2 && vb <= 16 && vb < kb) {
                s->kvq_bits = kb; s->kvq_v_bits = vb;
            } else {
                fprintf(stderr, "[kvq-hetero] formato non valido '%s' (atteso K<bits>V<bits>, bits 2..16, V<K) -> ignorato\n", het);
            }
          }
        }
        for (int i = 0; i < m->cfg.n_layers; i++) {
            s->kvq_w[i] = mlx_array_new(); s->kvq_sc[i] = mlx_array_new(); s->kvq_bi[i] = mlx_array_new();
        }
        /* kvq_v_* servono SEMPRE al percorso dequant-on-read, non solo a HETERO-KV: la
         * condizione kvq_read_used (vedi layer_forward, bool kvq_read) richiede kvq_v_w
         * non-NULL, e da questa sessione in poi kv_quantize=1 basta da solo ad abilitare quel
         * percorso (fix 2026-09-17) -- prima l'allocazione scattava SOLO con bit K/V diversi
         * (HETERO-KV) o LLM_KV_DEQUANT_READ esplicito, lasciando kvq_v_w NULL nel caso comune
         * (stessi bit, nessun flag): kvq_read_used restava sempre falso e il kernel
         * dequant-on-read non veniva mai raggiunto, anche dopo aver reso kvq_read automatico
         * (era la causa PRINCIPALE del gap 5-8x della fp16, vedi AiWiki
         * "MLX-KVQuant-Contesto-Lungo-5x-Peggiore": senza kvq_v_w si cade nel percorso vecchio
         * che dequantizza+concatena+riquantizza l'INTERO cold a OGNI step di decode). */
        s->kvq_v_len = (int*)calloc((size_t)m->cfg.n_layers, sizeof(int));
        s->kvq_v_w = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
        s->kvq_v_sc = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
        s->kvq_v_bi = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
        for (int i = 0; i < m->cfg.n_layers; i++) {
            s->kvq_v_w[i] = mlx_array_new(); s->kvq_v_sc[i] = mlx_array_new(); s->kvq_v_bi[i] = mlx_array_new();
        }
        if (s->kvq_v_bits != s->kvq_bits)
            fprintf(stderr, "[kvq-hetero] K%dV%d attivo (LLM_KV_HETERO)\n", s->kvq_bits, s->kvq_v_bits);
    }
    s->pending_h = mlx_array_new();
    /* Default chunk: per-famiglia (2026-09-03, matrice misurata). Dense: 8192 (misurato a 8K su
     * Qwen3.8-27B: 8192 -> 68.8 tok/s vs 41.9 a 2048); denso piccolo (Llama-1B) ottimo ~4096-8192.
     * Famiglie GDN-hybrid DENSE (qwen3_5/qwen3_next): chunk PICCOLO vince dopo il fix no-eval
     * GDN (Qwen3.6-27B prefill8192: chunk512=357 vs 8192=295, +15-18%) -> default 1024.
     * qwen3_5_moe (MoE GDN, es. Qwen3.6-35B-A3B) NON segue il denso: picco a 4096 (vedi sotto).
     * Override: LLM_PREFILL_CHUNK=N (bug §7). */
    {
        static int chunk_env = -1;
        if (chunk_env < 0) {
            int p = 0;   /* parametro `prefill` del registro (niente env) */
            if (llm_ai_mlx_opt_get_int("prefill", &p) == 0 && p > 0) chunk_env = p; else chunk_env = 0;
        }
        int defc = 8192;
        if (strcmp(m->cfg.model_type, "qwen3_5") == 0 ||
            strcmp(m->cfg.model_type, "qwen3_next") == 0)
            defc = 1024;
        else if (strcmp(m->cfg.model_type, "qwen3_5_moe") == 0)
            /* qwen3_5_moe (misurato 2026-09-10, sweep prefill N=8192, 3 rep ordine alternato):
             * il vecchio 1024 era una generalizzazione errata dal DENSE GDN (Qwen3.6-27B). La
             * variante MoE ha picco a 4096: 1024->2750, 4096->3564 tok/s mediana (+30%), e
             * 4096 vince anche a N=4096 (3791 vs 2957, +28%). PARITA' greedy BIT-IDENTICA tra
             * chunk 1024 e 4096 (decodectx N=3000, IDS 24/24 uguali). Split dal denso: qwen3_5
             * e qwen3_next restano 1024. Override LLM_PREFILL_CHUNK. */
            defc = 4096;
        else if (strcmp(m->cfg.model_type, "glm4_moe") == 0)
            defc = 2048; /* GLM prefill 4096: chunk2048 ~100 vs 8192 ~80 tok/s (2026-09-03) */
        else if (strncmp(m->cfg.model_type, "gemma4", 6) == 0)
            /* gemma4: i layer FULL hanno head_dim 512 -> eccede il limite del kernel flash MLX ->
             * fast_sdpa ripiega sul percorso GENERALE che MATERIALIZZA i punteggi (tensore
             * Lq*Lk). La materializzazione cresce col chunk (Lq): chunk PIU' PICCOLO vince.
             * PROMOSSO 2026-09-11 a 128 (era 1024): sweep prefill N=4096, 3 rep interleaved,
             * mediana 128->~320 vs 1024->~283 tok/s (+12-13%; ranking 128>256>512>1024
             * consistente; 64 peggiore). Parita' greedy BIT-IDENTICA chunk 128 vs 1024
             * (decodectx N=2048, IDS 16/16). Gli sliding (head_dim 256, 50 layer) restano in
             * flash: non-touchati. Override LLM_PREFILL_CHUNK. */
            defc = llm_mlx_flash_prefill_enabled() ? 2048 : 128;
        else if (strcmp(m->cfg.model_type, "gpt_oss") == 0)
            /* gpt_oss (misurato §V-quater, sweep prefill N=8192): picco a 2048 = +15% vs il
             * default 8192. ATTENZIONE: la parita' qui e' DIVERSA da qwen3_5_moe/gemma4 (che
             * erano bit-identici fra chunk). La SDPA flash fusa
             * (mlx_fast_scaled_dot_product_attention in llm_mlx_gqa_sdpa_sinks) tileggia la
             * riduzione sul dim-key IN FUNZIONE DELLA FORMA del chunk -> ordine di somma fp32
             * diverso -> flip di knife-edge (~1 ULP) su una quasi-parita' di argmax a ~N=3000
             * (decodectx: token 11 = id11 a 8192 vs id8202 a 2048), mentre a N=1500 e N=5000 gli
             * IDS restano identici. NON e' un bug della finestra (divergerebbe a tutte le
             * lunghezze). Per scelta del mandante (opzione A) la parita' adottata e' DETERMINISMO
             * A CONFIG FISSA (stesso prompt -> stesso output al chunk fissato qui), non identicita'
             * bit fra chunk diversi: renderlo chunk-invariante a bit richiederebbe un kernel
             * attention custom a ordine fisso = la via gia' dimostrata lenta/inconveniente (#60).
             * Decode n_tokens=1 non toccato. Override LLM_PREFILL_CHUNK. */
            defc = 2048;
        s->prefill_chunk_size = chunk_env > 0 ? chunk_env : defc;
    }
    /* N-Gram: history vuota all'avvio; tabella pre-allocata (ordine default 3, riempita
     * da llm_ai_mlx_session_build_ngram). */
    s->history = NULL; s->history_len = 0; s->history_cap = 0;
    s->ngram = llm_mlx_ngram_new(3, 1024);
    s->ngram_built = false;
    s->ngram_order = 3;
    s->compiled_decode = mlx_closure_new();
    s->compiled_decode_valid = false;
    /* LLMMETAL_KVMETAL: alloco per-layer SOLO se il flag (o _VERIFY) e' attivo -- default OFF,
     * nessun overhead. Stessa guardia multi-sessione di kvq (g_kvmetal_id globale).
     * g_kvmetal_ctx/_n_layers (stato ombra, dichiarato in cima al file) sincronizzati qui
     * insieme a g_kvmetal_id -- vedi commento su g_kvmetal_id per il perche'. */
    s->kvmetal_ctx = NULL;
    if (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled() || llm_mlx_kvmetal_auth_enabled()) {
        if (llm_mlx_kvmetal_auth_enabled() && m->cfg.kv_max_tokens <= 0)
            fprintf(stderr, "[kvmetal-auth] WARNING: LLMMETAL_KVMETAL_AUTH attivo ma kv_max_tokens<=0 -> "
                            "il pool O(1) non ha capacita' fissa, auth NON si attivera' (serve "
                            "LLM_KV_MAX_TOKENS >= lunghezza sessione). Fallback eager invariato.\n");
        s->kvmetal_ctx = (void**)calloc((size_t)m->cfg.n_layers, sizeof(void*));
        if (g_kvmetal_act >= 1) {
            g_kvmetal_id = NULL; g_kvmetal_ctx = NULL; g_kvmetal_n_layers = 0; g_kvmetal_pop_ok = NULL;
            if (!g_kvmetal_disabled) {
                g_kvmetal_disabled = 1;
                fprintf(stderr, "[kvmetal] WARNING: sessioni LLMMETAL_KVMETAL concorrenti non supportate "
                                "(g_kvmetal_id globale) -> disabilitato su tutte (fallback normale)\n");
            }
        } else {
            g_kvmetal_id = s; g_kvmetal_ctx = s->kvmetal_ctx; g_kvmetal_n_layers = s->n_layers;
            g_kvmetal_act = 1;
            g_kvmetal_pop_ok = (char*)calloc((size_t)s->n_layers, sizeof(char));
        }
    }
    /* LLM_GEMVQ4: stessa disciplina di kvmetal -- alloco le cache (ctx + chiavi) SOLO se il
     * flag (o _VERIFY) e' attivo, cap = n_layers*MAX_PER_LAYER + margine per lm_head/shared
     * extrasessione. La creazione dei singoli context Metal resta lazy al primo uso (vedi
     * llm_mlx_gemvq4_try); qui si alloca solo l'array di slot. */
    s->gemvq4_ctx = NULL; s->gemvq4_key = NULL; s->gemvq4_cap = 0;
    if (llm_mlx_gemvq4_enabled() || llm_mlx_gemvq4_verify_enabled()) {
        int cap = m->cfg.n_layers * LLM_GEMVQ4_MAX_PER_LAYER + 16;
        s->gemvq4_ctx = (void**)calloc((size_t)cap, sizeof(void*));
        s->gemvq4_key = (const void**)calloc((size_t)cap, sizeof(const void*));
        if (s->gemvq4_ctx != NULL && s->gemvq4_key != NULL) {
            s->gemvq4_cap = cap;
            if (g_gemvq4_act >= 1) {
                g_gemvq4_id = NULL; g_gemvq4_ctx = NULL; g_gemvq4_key = NULL; g_gemvq4_cap = 0;
                if (!g_gemvq4_disabled) {
                    g_gemvq4_disabled = 1;
                    fprintf(stderr, "[gemvq4] WARNING: sessioni LLM_GEMVQ4 concorrenti non supportate "
                                    "(g_gemvq4_id globale) -> disabilitato su tutte (fallback normale)\n");
                }
            } else {
                g_gemvq4_id = s; g_gemvq4_ctx = s->gemvq4_ctx;
                g_gemvq4_key = s->gemvq4_key; g_gemvq4_cap = s->gemvq4_cap;
                g_gemvq4_act = 1;
            }
        }
    }
    return s;
}

void llm_mlx_session_destroy(LlmMlxSession* s) {
    llm_mlx_session_spec_free(s);   /* spec streaming: libera i checkpoint K/V (alias) */
    if (s->dsv41) { llm_dsv41_cache_free((LlmDsv41Cache*)s->dsv41); free(s->dsv41); s->dsv41 = NULL; }
    if (s->kvmetal_ctx != NULL) {
        if (g_kvmetal_act > 0) g_kvmetal_act--;
        if (g_kvmetal_act == 0) g_kvmetal_disabled = 0;
    }
    if (g_kvmetal_id == s) { g_kvmetal_id = NULL; g_kvmetal_ctx = NULL; g_kvmetal_n_layers = 0;
                             free(g_kvmetal_pop_ok); g_kvmetal_pop_ok = NULL; }
    if (s->kvmetal_ctx) {
        for (int i = 0; i < s->n_layers; i++)
            if (s->kvmetal_ctx[i]) llm_kvmetal_destroy((LlmKvMetalCtx*)s->kvmetal_ctx[i]);
        free(s->kvmetal_ctx);
    }
    if (s->gemvq4_ctx != NULL) {
        if (g_gemvq4_act > 0) g_gemvq4_act--;
        if (g_gemvq4_act == 0) g_gemvq4_disabled = 0;
    }
    if (g_gemvq4_id == s) { g_gemvq4_id = NULL; g_gemvq4_ctx = NULL; g_gemvq4_key = NULL; g_gemvq4_cap = 0; }
    if (s->gemvq4_ctx) {
        for (int i = 0; i < s->gemvq4_cap; i++)
            if (s->gemvq4_ctx[i]) llm_gemvq4_destroy((LlmGemvQ4Ctx*)s->gemvq4_ctx[i]);
        free(s->gemvq4_ctx);
    }
    free(s->gemvq4_key);
    for (int i = 0; i < s->n_layers; i++) {
        mlx_array_free(s->k_cache[i]);
        mlx_array_free(s->v_cache[i]);
        if (s->aux_cache) mlx_array_free(s->aux_cache[i]);
        if (s->ps_mean) mlx_array_free(s->ps_mean[i]);   /* §G-bis pagesel: summary per pagina */
    }
    free(s->ps_mean); s->ps_mean = NULL;
    free(s->ps_done); s->ps_done = NULL;
    free(s->k_cache);
    free(s->v_cache);
    free(s->aux_cache);
    free(s->has_cache);
    if (s->kvq_w) { for (int i = 0; i < s->n_layers; i++) { mlx_array_free(s->kvq_w[i]); mlx_array_free(s->kvq_sc[i]); mlx_array_free(s->kvq_bi[i]); } }
    free(s->kvq_len); free(s->kvq_w); free(s->kvq_sc); free(s->kvq_bi);
    if (s->kvq_v_w) { for (int i = 0; i < s->n_layers; i++) { mlx_array_free(s->kvq_v_w[i]); mlx_array_free(s->kvq_v_sc[i]); mlx_array_free(s->kvq_v_bi[i]); } }
    free(s->kvq_v_len); free(s->kvq_v_w); free(s->kvq_v_sc); free(s->kvq_v_bi);
    mlx_array_free(s->pending_h);
    llm_mlx_ngram_free(&s->ngram);
    free(s->history);
    if (s->compiled_decode_valid) mlx_closure_free(s->compiled_decode);
    free(s);
}

/* ABI v2 — kv_clear: azzera la KV-cache IN-PLACE (stesso handle), riportando la sessione allo
 * stato di una appena creata. Il modello NON viene toccato. Rispetta la disciplina di create:
 * ogni slot e' liberato e ricreato con una PROPRIA mlx_array_new() (mai un fill con lo stesso
 * valore). Gestisce i casi parked (KV su disco: si scarta, parked=false) e dsv41 (cache liberata
 * e ricreata lazy). history/n-gram/pagesel resettati perche' descrivono la sequenza cancellata. */
int llm_ai_mlx_session_reset(llm_ai_mlx_session_handle handle) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    /* DeepSeek V4.1 whole-model: libera la cache (ricreata lazy al prossimo append/prefill). */
    if (s->dsv41) { llm_dsv41_cache_free((LlmDsv41Cache*)s->dsv41); free(s->dsv41); s->dsv41 = NULL; }
    /* KV quant: azzera i contatori cold (i buffer quantizzati restano allocati, riusati). */
    if (s->kvq_len)   for (int l = 0; l < s->n_layers; l++) s->kvq_len[l] = 0;
    if (s->kvq_v_len) for (int l = 0; l < s->n_layers; l++) s->kvq_v_len[l] = 0;
    /* KV principale: libera e ricrea vuota per layer. */
    for (int l = 0; l < s->n_layers; l++) {
        mlx_array_free(s->k_cache[l]); s->k_cache[l] = mlx_array_new();
        mlx_array_free(s->v_cache[l]); s->v_cache[l] = mlx_array_new();
        if (s->aux_cache) { mlx_array_free(s->aux_cache[l]); s->aux_cache[l] = mlx_array_new(); }
        s->has_cache[l] = false;
    }
    /* Pagesel summary: libera se allocato (indice di una sequenza cancellata). */
    if (s->ps_mean) {
        for (int l = 0; l < s->n_layers; l++) mlx_array_free(s->ps_mean[l]);
        free(s->ps_mean); s->ps_mean = NULL;
    }
    free(s->ps_done); s->ps_done = NULL; s->ps_cap = 0;
    mlx_array_free(s->pending_h); s->pending_h = mlx_array_new();
    s->cache_offset = 0;
    s->has_pending = false;
    s->pending_next_id = 0;
    s->history_len = 0;
    s->ngram_built = false;
    s->pinned_tokens = 0;
    s->parked = false;
    return 0;
}

/* ABI v2 — kv_stats: token usati e capacita' della KV. used = cache_offset (nativo). cap = il
 * limite configurato (cfg.kv_max_tokens) se > 0, altrimenti la lunghezza ALLOCATA della cache
 * del layer 0 (per la KV dinamica corrente coincide con l'uso: nessun headroom configurato).
 * Non inventa numeri: se la cache non e' ancora stata allocata, cap = 0. */
int llm_ai_mlx_session_kv_stats(llm_ai_mlx_session_handle handle, int* out_used, int* out_cap) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    LlmMlxModel* m = s->model;
    if (out_used) *out_used = s->cache_offset;
    int cap = 0;
    if (m && m->cfg.kv_max_tokens > 0) {
        cap = m->cfg.kv_max_tokens;
    } else if (s->has_cache && s->k_cache && s->k_cache[0].ctx != NULL) {
        const int* shp = mlx_array_shape(s->k_cache[0]);
        if (shp != NULL && mlx_array_ndim(s->k_cache[0]) == 4) cap = shp[2];
    }
    if (out_cap) *out_cap = cap;
    return 0;
}

/* Appende token alla history della sessione (per la tabella n-gram). */
void session_history_append(LlmMlxSession* s, const int32_t* tok, int n) {
    if (!s || n <= 0 || !tok) return;
    if (s->history_len + n > s->history_cap) {
        int ncap = s->history_cap ? s->history_cap * 2 : 1024;
        while (ncap < s->history_len + n) ncap *= 2;
        int32_t* nh = (int32_t*)realloc(s->history, (size_t)ncap * sizeof(int32_t));
        if (!nh) return;
        s->history = nh;
        s->history_cap = ncap;
    }
    memcpy(s->history + s->history_len, tok, (size_t)n * sizeof(int32_t));
    const int hist_before = s->history_len;
    s->history_len += n;
    /* N-GRAM self-speculative INCREMENTALE: aggiorna la tabella coi token appena aggiunti, cosi'
     * il copy-draft impara i pattern che si ripetono NELL'OUTPUT (non solo quelli del prompt).
     * last_suffix = i `order` token che precedono `tok` (dall'history). No-op se la tabella non e'
     * ancora costruita (build_ngram la costruira' dall'intera history). */
    if (s->ngram_built && s->ngram.buckets != NULL) {
        const int ord = s->ngram_order;
        const int32_t* suffix = (hist_before >= ord) ? (s->history + hist_before - ord) : NULL;
        llm_mlx_ngram_add(&s->ngram, tok, n, suffix);
    }
}
