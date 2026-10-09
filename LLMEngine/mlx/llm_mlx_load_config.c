/* ============================================================================
 * Dominio load_config: lettura/parsing di config.json -> ModelConfig (model_type, MLA/GDN/gemma4/sliding, rope_scaling, MoE, dflash, quant).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include <cJSON.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
bool load_config(const char* model_dir, ModelConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));

    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_config: impossibile aprire %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 0) { fclose(f); return false; }
    char* buf = (char*)malloc((size_t)fsize + 1);
    size_t nread = fread(buf, 1, (size_t)fsize, f);
    fclose(f);
    buf[nread] = '\0';

    cJSON* j = cJSON_ParseWithLength(buf, nread);
    free(buf);
    if (!j) {
        fprintf(stderr, "load_config: JSON malformato in %s\n", path);
        return false;
    }

    /* model_type: chiave verificata presente nel config.json reale (Qwen3-Coder-30B-A3B,
     * valore "qwen3_moe") -- vedi Fase1b-MLX-Multi-Architecture-Plan.md. Richiesta come le
     * altre chiavi sotto: un config.json senza questa chiave non è un formato che questo
     * motore sa interpretare, meglio fallire qui che dispatchare su un valore vuoto. Letta
     * PRIMA delle altre chiavi (a differenza delle fasi precedenti) perché serve a decidere
     * se il resto dei campi va letto dalla radice o da un oggetto annidato (Gemma4, sotto). */
    cJSON* model_type = cJSON_GetObjectItemCaseSensitive(j, "model_type");
    if (!cJSON_IsString(model_type)) {
        fprintf(stderr, "load_config: chiave attesa mancante: model_type\n");
        cJSON_Delete(j);
        return false;
    }
    llm_mlx_mode_set(cfg->model_type, sizeof(cfg->model_type), model_type->valuestring);
    /* Ternary-Bonsai 2 27B (PrismML): il pack si dichiara `prism_hadamard_qwen35` ma la base e'
     * qwen3_5. Normalizzo QUI, prima di ogni ramo model_type-dipendente (layer_types, rope, MoE,
     * chunk, ...): cosi' la config viene letta esattamente come un qwen3_5. La presenza del
     * contratto Hadamard NON dipende dal nome: si rileva dal file hadamard.json (vedi la load). */
    if (strcmp(cfg->model_type, "prism_hadamard_qwen35") == 0)
        snprintf(cfg->model_type, sizeof(cfg->model_type), "qwen3_5");

    /* FASE 1b.4.0 (Gemma4): i campi rilevanti vivono sotto config.json["text_config"], non
     * alla radice come in tutte le altre architetture -- verificato sul config.json reale di
     * gemma-4-31b-it-4bit. `src` è l'oggetto cJSON da cui leggere tutti i campi sotto: la
     * radice per default, `text_config` se presente (e solo se model_type inizia con "gemma4",
     * per evitare di raccogliere per sbaglio un `text_config` di un'architettura diversa che
     * dovesse averne uno per un motivo non correlato). */
    cJSON* src = j;
    bool is_gemma4 = strncmp(cfg->model_type, "gemma4", 6) == 0;
    /* FASE 1b.5.0 (Qwen3.5/3.8): anche il config reale del 27B (Qwen3_5ForConditionalGeneration)
     * ha i campi sotto text_config (verificato: qwen3_5_text, 64 layer, head_dim 256). */
    bool has_nested_text_config = is_gemma4 || strcmp(cfg->model_type, "qwen3_5") == 0 || strcmp(cfg->model_type, "qwen3_5_moe") == 0
                                   || strcmp(cfg->model_type, "qwen4_exp") == 0   /* Flash-Next: campi in text_config */
                                   || strcmp(cfg->model_type, "prism_hadamard_qwen35") == 0; /* Ternary-Bonsai 2 27B
                                       (PrismML): i parametri del modello stanno in text_config (model_type
                                       annidato "qwen3_5_text"), come per gli altri qui sopra. */
    /* qwen4_exp_text (model_type del text_config) NON e' il model_type di dispatch (root
     * qwen4_exp); se un save avesse root = qwen4_exp_text, normalizziamo anche quello. */
    bool is_qwen4_exp = strcmp(cfg->model_type, "qwen4_exp") == 0 || strcmp(cfg->model_type, "qwen4_exp_text") == 0;
    if (has_nested_text_config) {
        cJSON* text_config = cJSON_GetObjectItemCaseSensitive(j, "text_config");
        if (!cJSON_IsObject(text_config)) {
            fprintf(stderr, "load_config: model_type=%s ma \"text_config\" assente/non-oggetto\n", cfg->model_type);
            cJSON_Delete(j);
            return false;
        }
        src = text_config;
    }
    /* Fallback generico: se la radice NON ha i campi del text-encoder ma esiste un
     * `text_config` annidato (es. Ministral-3-14B-mxfp8, wrapper multimodale), usa quello. */
    if (!has_nested_text_config) {
        cJSON* root_hs = cJSON_GetObjectItemCaseSensitive(j, "hidden_size");
        cJSON* tc = cJSON_GetObjectItemCaseSensitive(j, "text_config");
        if (!cJSON_IsNumber(root_hs) && cJSON_IsObject(tc)) src = tc;
    }

#define REQ_INT(field, key) do { \
        cJSON* it = cJSON_GetObjectItemCaseSensitive(src, key); \
        if (!cJSON_IsNumber(it)) { fprintf(stderr, "load_config: chiave attesa mancante: %s\n", key); cJSON_Delete(j); return false; } \
        cfg->field = (int)it->valuedouble; \
    } while (0)
#define REQ_FLOAT(field, key) do { \
        cJSON* it = cJSON_GetObjectItemCaseSensitive(src, key); \
        if (!cJSON_IsNumber(it)) { fprintf(stderr, "load_config: chiave attesa mancante: %s\n", key); cJSON_Delete(j); return false; } \
        cfg->field = (float)it->valuedouble; \
    } while (0)
/* FASE 1b.1: campi specifici del MoE -- assenti in un config.json di un modello
 * denso (es. Qwen3 non-MoE, Llama, Mistral). Opzionali: 0 se la chiave manca,
 * non un errore di caricamento. dense_swiglu_ffn non li usa comunque. */
#define OPT_INT(field, key) do { \
        cJSON* it = cJSON_GetObjectItemCaseSensitive(src, key); \
        cfg->field = cJSON_IsNumber(it) ? (int)it->valuedouble : 0; \
    } while (0)
#define OPT_FLOAT(field, key, deflt) do { \
        cJSON* it = cJSON_GetObjectItemCaseSensitive(src, key); \
        cfg->field = cJSON_IsNumber(it) ? (float)it->valuedouble : (deflt); \
    } while (0)
#define OPT_BOOL(field, key) do { \
        cJSON* it = cJSON_GetObjectItemCaseSensitive(src, key); \
        cfg->field = cJSON_IsBool(it) ? cJSON_IsTrue(it) : false; \
    } while (0)

    REQ_INT(hidden_size, "hidden_size");
    REQ_INT(n_layers, "num_hidden_layers");
    REQ_INT(n_heads, "num_attention_heads");
    REQ_INT(n_kv_heads, "num_key_value_heads");
    /* FASE 1b (Qwen2): head_dim assente nel config.json reale (verificato su
     * DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit, model_type "qwen2") -- mlx_lm lo
     * deriva come hidden_size/num_attention_heads quando la chiave manca
     * (stesso comportamento di ModelArgs.head_dim in qwen2.py). Qwen3 lo
     * esplicita sempre nel proprio config.json (verificato), quindi per
     * quell'architettura questo ramo non cambia nulla. */
    {
        cJSON* hd = cJSON_GetObjectItemCaseSensitive(src, "head_dim");
        if (cJSON_IsNumber(hd)) cfg->head_dim = (int)hd->valuedouble;
        else if (cfg->n_heads > 0) cfg->head_dim = cfg->hidden_size / cfg->n_heads;
        else { fprintf(stderr, "load_config: head_dim assente e num_attention_heads invalido\n"); cJSON_Delete(j); return false; }
    }
    OPT_INT(n_experts, "num_experts");
    /* FASE 1b.3 (Mixtral): la stessa cosa si chiama "num_local_experts" invece di
     * "num_experts" -- verificato sul config.json reale di Mixtral-8x22B-4bit
     * (che non ha affatto la chiave "num_experts"). Nomi diversi per lo stesso
     * concetto fra famiglie MoE, non un'omissione: leggere sempre entrambe le
     * chiavi possibili invece di assumerne una sola. */
    if (cfg->n_experts == 0) OPT_INT(n_experts, "num_local_experts");
    OPT_INT(top_k, "num_experts_per_tok");
    OPT_INT(moe_intermediate_size, "moe_intermediate_size");
    /* Alcune famiglie MoE chiamano "intermediate_size" l'intermedio PER-ESPERTO invece di
     * "moe_intermediate_size": Mixtral-8x22B (16384) e gpt-oss (2880) non hanno affatto la
     * chiave "moe_intermediate_size" e, avendo tutti i layer MoE, la loro "intermediate_size"
     * E' la dimensione switch_mlp. Stesso dual-name del blocco num_experts/num_local_experts
     * sopra. Il fallback e' guardato da n_experts>1 per NON toccare le famiglie dense (che
     * usano "intermediate_size" per il FFN denso, letto a parte dalla forma del peso). Senza
     * di qui cfg resta 0: il gate dff>0 del kernel MoE fuso scarta silenziosamente la famiglia
     * e la stima di memoria (~:13621) sottostima i pesi esperti. */
    if (cfg->moe_intermediate_size == 0 && cfg->n_experts > 1)
        OPT_INT(moe_intermediate_size, "intermediate_size");
    /* Q2: "intermediate_size" della radice, SEMPRE (anche densi). Per le famiglie MoE coincide col
     * per-esperto letto sopra; per i densi e' la larghezza del FFN (es. Llama-3.2-1B 8192) e serve
     * al collector imatrix dei layer densi (allocazione di down_cols). Nessun fallback: se manca
     * resta 0 e il collector denso si disattiva da solo (vedi llm_mlx_imatrix_begin). */
    OPT_INT(intermediate_size, "intermediate_size");
    REQ_INT(vocab_size, "vocab_size");
    /* rms_norm_eps: alcune conversioni MLX non la scrivono (OLMoE-1B-7B-0924: chiave
     * assente) -> fallback a layer_norm_eps e infine al default 1e-6. */
    OPT_FLOAT(rms_norm_eps, "rms_norm_eps", 0.0f);
    if (cfg->rms_norm_eps == 0.0f) OPT_FLOAT(rms_norm_eps, "layer_norm_eps", 0.0f);
    if (cfg->rms_norm_eps == 0.0f) cfg->rms_norm_eps = 1e-6f;
    /* FIX 2026-09-01: parse generico di rope_theta dalla radice (es. gpt_oss: 150000).
     * Prima era letto SOLO nei branch gemma4/qwen3_5 (rope_parameters) -> per gpt_oss
     * restava 0 -> RoPE base 0 -> frequenze zero -> attention garbage (divergenza
     * attribuita erroneamente al bf16). I branch sotto sovrascrivono con rope_parameters
     * quando presente (gemma4/qwen3_5), quindi questo e' il default per le altre arch. */
    OPT_FLOAT(rope_theta, "rope_theta", 10000.0f);

    /* FASE 1b.4.0 (Gemma4): rope_theta/parametri RoPE per-tipo di layer vivono sotto
     * text_config.rope_parameters.{full_attention,sliding_attention}, non una singola
     * chiave rope_theta piatta -- letti SOLO se is_gemma4, altrimenti rope_theta resta la
     * chiave piatta di sempre (architetture non-Gemma4). */
    if (is_gemma4) {
        cJSON* rp = cJSON_GetObjectItemCaseSensitive(src, "rope_parameters");
        cJSON* rp_full = rp ? cJSON_GetObjectItemCaseSensitive(rp, "full_attention") : NULL;
        cJSON* rp_sliding = rp ? cJSON_GetObjectItemCaseSensitive(rp, "sliding_attention") : NULL;
        if (!cJSON_IsObject(rp_full) || !cJSON_IsObject(rp_sliding)) {
            fprintf(stderr, "load_config: rope_parameters.{full,sliding}_attention assenti per model_type=%s\n", cfg->model_type);
            cJSON_Delete(j);
            return false;
        }
        cJSON* rt_full = cJSON_GetObjectItemCaseSensitive(rp_full, "rope_theta");
        cJSON* rt_sliding = cJSON_GetObjectItemCaseSensitive(rp_sliding, "rope_theta");
        if (!cJSON_IsNumber(rt_full) || !cJSON_IsNumber(rt_sliding)) {
            fprintf(stderr, "load_config: rope_theta assente in full/sliding_attention\n");
            cJSON_Delete(j);
            return false;
        }
        cfg->rope_theta_full = (float)rt_full->valuedouble;
        cfg->rope_theta_sliding = (float)rt_sliding->valuedouble;
        cJSON* prf = cJSON_GetObjectItemCaseSensitive(rp_full, "partial_rotary_factor");
        cfg->partial_rotary_factor = cJSON_IsNumber(prf) ? (float)prf->valuedouble : 1.0f;
        cfg->rope_theta = cfg->rope_theta_sliding; /* fallback per eventuale codice generico che legga ancora rope_theta piatto */

        OPT_INT(global_head_dim, "global_head_dim");
        OPT_INT(num_global_kv_heads, "num_global_key_value_heads");
        OPT_BOOL(attention_k_eq_v, "attention_k_eq_v");
        OPT_FLOAT(final_logit_softcap, "final_logit_softcapping", 0.0f);
        OPT_INT(sliding_window, "sliding_window");
        /* FASE 1b.4.5 (Gemma4): l'embedding in ingresso è scalato per sqrt(hidden_size)
         * (Gemma4TextModel.__init__: self.embed_scale = config.hidden_size**0.5, SEMPRE,
         * anche quando embed_scale è assente dal config). Verificato in gemma4_text.py:402. */
        cfg->embed_scale = sqrtf((float)cfg->hidden_size);

        /* layer_types: array di stringhe "sliding_attention"/"full_attention", un valore per
         * layer -- letto per intero, MAI assunto un periodo fisso anche se il modello reale
         * osservato ne ha uno regolare. */
        cJSON* layer_types = cJSON_GetObjectItemCaseSensitive(src, "layer_types");
        if (!cJSON_IsArray(layer_types) || cJSON_GetArraySize(layer_types) != cfg->n_layers) {
            fprintf(stderr, "load_config: layer_types assente o di lunghezza diversa da num_hidden_layers\n");
            cJSON_Delete(j);
            return false;
        }
        cfg->is_sliding_layer = (bool*)malloc((size_t)cfg->n_layers * sizeof(bool));
        for (int i = 0; i < cfg->n_layers; i++) {
            cJSON* lt = cJSON_GetArrayItem(layer_types, i);
            if (!cJSON_IsString(lt)) { fprintf(stderr, "load_config: layer_types[%d] non è una stringa\n", i); cJSON_Delete(j); free(cfg->is_sliding_layer); cfg->is_sliding_layer = NULL; return false; }
            if (strcmp(lt->valuestring, "sliding_attention") == 0 || strcmp(lt->valuestring, "linear_attention") == 0) cfg->is_sliding_layer[i] = true;
            else if (strcmp(lt->valuestring, "full_attention") == 0) cfg->is_sliding_layer[i] = false;
            else { fprintf(stderr, "load_config: layer_types[%d]=\"%s\" non riconosciuto\n", i, lt->valuestring); cJSON_Delete(j); free(cfg->is_sliding_layer); cfg->is_sliding_layer = NULL; return false; }
        }
    } else if (strcmp(cfg->model_type, "lfm2") == 0) {
        /* LFM2 (Liquid): ibrido. Layer a ShortConv (is_linear_layer=true) alternati a layer a
         * piena attenzione (indici in full_attn_idxs). conv_L_cache = kernel del conv1d. */
        cfg->is_linear_layer = (bool*)calloc((size_t)cfg->n_layers, sizeof(bool));
        if (cfg->is_linear_layer) for (int i = 0; i < cfg->n_layers; i++) cfg->is_linear_layer[i] = true;
        cJSON* fa = cJSON_GetObjectItemCaseSensitive(src, "full_attn_idxs");
        if (cJSON_IsArray(fa) && cfg->is_linear_layer)
            for (int k = 0; k < cJSON_GetArraySize(fa); k++) {
                cJSON* it = cJSON_GetArrayItem(fa, k);
                if (cJSON_IsNumber(it)) { int li = (int)it->valuedouble; if (li >= 0 && li < cfg->n_layers) cfg->is_linear_layer[li] = false; }
            }
        OPT_INT(linear_conv_kernel_dim, "conv_L_cache");
        llm_mlx_mode_set(cfg->layer_prefix, sizeof(cfg->layer_prefix), "model.layers.");
    } else if (strcmp(cfg->model_type, "glm4_moe") == 0) {
        /* GLM-4.5-Air (glm4_moe): MoE con esperti condivisi, primi layer densi,
         * attenzione con bias + partial rotary 0.5, router BF16 piatto + sigmoid. */
        OPT_INT(n_experts, "n_routed_experts");          /* 128 */
        OPT_INT(n_shared_experts, "n_shared_experts");   /* 1 */
        OPT_INT(first_k_dense_replace, "first_k_dense_replace"); /* 1: layer 0 denso */
        OPT_INT(n_group, "n_group");
        OPT_INT(topk_group, "topk_group");
        OPT_FLOAT(routed_scaling_factor, "routed_scaling_factor", 1.0f);
        cJSON* prf = cJSON_GetObjectItemCaseSensitive(src, "partial_rotary_factor");
        if (cJSON_IsNumber(prf)) cfg->partial_rotary_factor = (float)prf->valuedouble;
    } else if (strcmp(cfg->model_type, "deepseek_v2") == 0 || strcmp(cfg->model_type, "deepseek_v3") == 0 ||
               strcmp(cfg->model_type, "kimi_k2") == 0 || strcmp(cfg->model_type, "joyai_llm_flash") == 0) {
        /* Famiglia MLA (Multi-head Latent Attention): DeepSeek V2/V3, Kimi K2, JoyAI (mlx_lm
         * remappa kimi_k2/joyai_llm_flash -> deepseek_v3: stesso codice). KV compresso in latente
         * + riproiezione; RoPE (yarn) solo sui qk_rope_head_dim; MoE group-limited-greedy. */
        OPT_INT(kv_lora_rank, "kv_lora_rank");
        OPT_INT(q_lora_rank, "q_lora_rank");             /* assente in V2-Lite (=0) */
        OPT_INT(qk_nope_head_dim, "qk_nope_head_dim");
        OPT_INT(qk_rope_head_dim, "qk_rope_head_dim");
        OPT_INT(v_head_dim, "v_head_dim");
        OPT_INT(n_experts, "n_routed_experts");
        OPT_INT(n_shared_experts, "n_shared_experts");
        OPT_INT(first_k_dense_replace, "first_k_dense_replace");
        OPT_INT(n_group, "n_group");
        OPT_INT(topk_group, "topk_group");
        OPT_FLOAT(routed_scaling_factor, "routed_scaling_factor", 1.0f);
    } else if (strcmp(cfg->model_type, "kimi_linear") == 0) {
        /* Kimi Linear: attenzione ibrida MLA-assorbita (full_attn_layers) + KDA/Gated DeltaNet
         * (kda_layers); MoE con router sigmoid|softmax, grouped top-k, shared experts. */
        OPT_INT(kv_lora_rank, "kv_lora_rank");
        OPT_INT(qk_nope_head_dim, "qk_nope_head_dim");
        OPT_INT(qk_rope_head_dim, "qk_rope_head_dim");
        OPT_INT(v_head_dim, "v_head_dim");
        OPT_INT(n_experts, "num_experts");
        OPT_INT(top_k, "num_experts_per_token");
        OPT_INT(n_shared_experts, "num_shared_experts");
        OPT_INT(moe_intermediate_size, "moe_intermediate_size");
        OPT_INT(first_k_dense_replace, "first_k_dense_replace");
        OPT_INT(n_group, "num_expert_group");
        OPT_INT(topk_group, "topk_group");
        OPT_FLOAT(routed_scaling_factor, "routed_scaling_factor", 1.0f);
        {
            cJSON* ra = cJSON_GetObjectItemCaseSensitive(src, "moe_router_activation_func");
            if (cJSON_IsString(ra)) llm_mlx_mode_set(cfg->moe_router_activation, sizeof cfg->moe_router_activation, ra->valuestring);
            else snprintf(cfg->moe_router_activation, sizeof cfg->moe_router_activation, "sigmoid");
            cJSON* rn = cJSON_GetObjectItemCaseSensitive(src, "moe_renormalize");
            cfg->moe_renormalize = cJSON_IsBool(rn) ? cJSON_IsTrue(rn) : true;
            /* glm4_moe_ffn_block usa norm_topk_prob per la rinormalizzazione dei top-k:
             * la semantica Kimi `moe_renormalize` e' la stessa. */
            cfg->norm_topk_prob = cfg->moe_renormalize;
            cJSON* mlf = cJSON_GetObjectItemCaseSensitive(src, "moe_layer_freq");
            cfg->moe_layer_freq = cJSON_IsNumber(mlf) ? (int)mlf->valuedouble : 1;
            cJSON* lac = cJSON_GetObjectItemCaseSensitive(src, "linear_attn_config");
            if (cJSON_IsObject(lac)) {
                cJSON* hd = cJSON_GetObjectItemCaseSensitive(lac, "head_dim");
                if (cJSON_IsNumber(hd)) cfg->linear_attn_head_dim = (int)hd->valuedouble;
                cJSON* nh = cJSON_GetObjectItemCaseSensitive(lac, "num_heads");
                if (cJSON_IsNumber(nh)) cfg->linear_attn_num_heads = (int)nh->valuedouble;
                cJSON* ck = cJSON_GetObjectItemCaseSensitive(lac, "short_conv_kernel_size");
                cfg->linear_attn_conv_kernel = cJSON_IsNumber(ck) ? (int)ck->valuedouble : 4;
                cJSON* kda = cJSON_GetObjectItemCaseSensitive(lac, "kda_layers");
                if (cJSON_IsArray(kda)) {
                    cfg->is_linear_layer = (bool*)calloc((size_t)cfg->n_layers, sizeof(bool));
                    for (int k = 0; k < cJSON_GetArraySize(kda); k++) {
                        cJSON* it = cJSON_GetArrayItem(kda, k);
                        if (cJSON_IsNumber(it)) { int li = (int)it->valuedouble - 1; if (li >= 0 && li < cfg->n_layers) cfg->is_linear_layer[li] = true; }
                    }
                }
            }
        }
    } else if (strcmp(cfg->model_type, "gpt_oss") == 0) {
        /* GPT-OSS: config/policy in oss/ (llm_oss_config.c). Thin-hook: nessuna logica qui. */
        if (!llm_oss_parse_config(cfg, src)) { cJSON_Delete(j); return false; }
    } else if (strcmp(cfg->model_type, "qwen3_5") == 0 || strcmp(cfg->model_type, "qwen3_5_moe") == 0) {
        /* FASE 1b.5 (Qwen3.8-27B): rope_theta vive in text_config.rope_parameters (piatto,
         * non per-tipo come gemma4). Es. reale: rope_parameters.rope_theta=10000000,
         * partial_rotary_factor=0.25 (hybrid GDN/full). qwen3_5_moe (Qwen3.6-35B) usa lo
         * stesso schema di parsing GDN. */
        cJSON* rp = cJSON_GetObjectItemCaseSensitive(src, "rope_parameters");
        cJSON* rt = rp ? cJSON_GetObjectItemCaseSensitive(rp, "rope_theta") : NULL;
        if (!cJSON_IsNumber(rt)) {
            fprintf(stderr, "load_config: rope_parameters.rope_theta assente per model_type=%s\n", cfg->model_type);
            cJSON_Delete(j);
            return false;
        }
        cfg->rope_theta = (float)rt->valuedouble;
        cJSON* prf = rp ? cJSON_GetObjectItemCaseSensitive(rp, "partial_rotary_factor") : NULL;
        cfg->partial_rotary_factor = cJSON_IsNumber(prf) ? (float)prf->valuedouble : 1.0f;
        /* Qwen3.8-27B usa il prefisso language_model.model.layers. (come gemma4) per i layer. */
        llm_mlx_mode_set(cfg->layer_prefix, sizeof(cfg->layer_prefix), "language_model.model.layers.");

        /* FASE 1b.5 (hybrid GDN): parametri linear_attn + dispatch per-layer. */
        cJSON* fai = cJSON_GetObjectItemCaseSensitive(src, "full_attention_interval");
        if (cJSON_IsNumber(fai)) cfg->full_attention_interval = (int)fai->valuedouble;
        cfg->linear_num_key_heads   = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_num_key_heads"));
        cfg->linear_num_value_heads = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_num_value_heads"));
        cfg->linear_key_head_dim    = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_key_head_dim"));
        cfg->linear_value_head_dim  = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_value_head_dim"));
        cfg->linear_conv_kernel_dim = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_conv_kernel_dim"));
        cJSON* aog = cJSON_GetObjectItemCaseSensitive(src, "attn_output_gate");
        cfg->attn_output_gate = cJSON_IsTrue(aog);
        /* E3 KV quant (opzionale, default OFF): {kvQuantize:true, kvQuantizeBits:8, kvQuantizeThreshold:N} */
        cJSON* kvq = cJSON_GetObjectItemCaseSensitive(src, "kvQuantize");
        cfg->kv_quantize = cJSON_IsTrue(kvq);
        cJSON* kvqb = cJSON_GetObjectItemCaseSensitive(src, "kvQuantizeBits");
        cfg->kv_quantize_bits = cJSON_IsNumber(kvqb) ? (int)kvqb->valuedouble : 0;
        cJSON* kvqt = cJSON_GetObjectItemCaseSensitive(src, "kvQuantizeThreshold");
        cfg->kv_quantize_threshold = cJSON_IsNumber(kvqt) ? (int)kvqt->valuedouble : 0;
        /* Rolling-Window KV (fase 3 paging): {kvMaxTokens:N} -- trim della KV-cache full ai piu' recenti N */
        cJSON* kvm = cJSON_GetObjectItemCaseSensitive(src, "kvMaxTokens");
        cfg->kv_max_tokens = cJSON_IsNumber(kvm) ? (int)kvm->valuedouble : 0;
        /* NSA sparse (opzionale, default OFF): {nsaSparse:true, nsaTopk:N, nsaWindow:M} */
        cJSON* nsa = cJSON_GetObjectItemCaseSensitive(src, "nsaSparse");
        cfg->nsa_sparse = cJSON_IsTrue(nsa);
        cJSON* nsat = cJSON_GetObjectItemCaseSensitive(src, "nsaTopk");
        cfg->nsa_topk = cJSON_IsNumber(nsat) ? (int)nsat->valuedouble : 0;
        cJSON* nsaw = cJSON_GetObjectItemCaseSensitive(src, "nsaWindow");
        cfg->nsa_window = cJSON_IsNumber(nsaw) ? (int)nsaw->valuedouble : 0;
        /* Fallback ENV per la CAMPAGNA MISURATA (senza toccare la config del provider):
         * LLM_MLX_NSA_SPARSE=1 [+ LLM_MLX_NSA_TOPK=N, LLM_MLX_NSA_WINDOW=M]. Solo se la config
         * non l'ha gia' deciso: default invariato (OFF) per gli altri path. */
        if (!cfg->nsa_sparse && llm_ai_mlx_opt_env("LLM_MLX_NSA_SPARSE") != NULL) {
            cfg->nsa_sparse = true;
            const char* t = llm_ai_mlx_opt_env("LLM_MLX_NSA_TOPK");
            if (cfg->nsa_topk <= 0 && t != NULL) cfg->nsa_topk = atoi(t);
            const char* w = llm_ai_mlx_opt_env("LLM_MLX_NSA_WINDOW");
            if (cfg->nsa_window <= 0 && w != NULL) cfg->nsa_window = atoi(w);
        }
        if (cfg->full_attention_interval > 0) {
            cfg->is_linear_layer = (bool*)malloc((size_t)cfg->n_layers * sizeof(bool));
            for (int i = 0; i < cfg->n_layers; i++)
                cfg->is_linear_layer[i] = ((i + 1) % cfg->full_attention_interval) != 0;
        }
    } else if (is_qwen4_exp) {
        /* Qwen3.8-Flash-Next (qwen4_exp): rope_theta/partial_rotary_factor in rope_parameters
         * (piatto, come qwen3_5), stesso schema linear_attn.* di qwen3_5/qwen3_next per i layer
         * GDN, PIU' i campi propri di Qwen4-Exp (hyper-connection a 4 stream, indexer QSA) --
         * nomi presi 1:1 da TextModelArgs del reference rapidMLX (nessuna invenzione, vedi
         * Todo C17). layer_types usa "linear_attention"/"qwen_sparse_attention" (non
         * "full_attention" come gemma4/gpt_oss) -- letto per intero se presente, altrimenti
         * derivato da full_attention_interval con la STESSA formula di qwen3_5/qwen3_next
         * (il post_init del reference fa esattamente questo fallback). */
        cJSON* rp = cJSON_GetObjectItemCaseSensitive(src, "rope_parameters");
        cJSON* rt = rp ? cJSON_GetObjectItemCaseSensitive(rp, "rope_theta") : NULL;
        cfg->rope_theta = cJSON_IsNumber(rt) ? (float)rt->valuedouble : 10000000.0f;
        cJSON* prf = rp ? cJSON_GetObjectItemCaseSensitive(rp, "partial_rotary_factor") : NULL;
        cfg->partial_rotary_factor = cJSON_IsNumber(prf) ? (float)prf->valuedouble : 0.25f;

        cfg->linear_num_key_heads   = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_num_key_heads"));
        cfg->linear_num_value_heads = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_num_value_heads"));
        cfg->linear_key_head_dim    = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_key_head_dim"));
        cfg->linear_value_head_dim  = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_value_head_dim"));
        cfg->linear_conv_kernel_dim = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_conv_kernel_dim"));

        OPT_INT(hc_count, "hc_count");
        OPT_INT(hc_lowrank, "hc_lowrank");
        OPT_INT(indexer_n_heads, "indexer_n_heads");
        OPT_INT(indexer_kv_heads, "indexer_kv_heads");
        OPT_INT(indexer_head_dim, "indexer_head_dim");
        OPT_INT(indexer_budget, "indexer_budget");
        OPT_INT(indexer_compress_ratio, "indexer_compress_ratio");

        cJSON* fai = cJSON_GetObjectItemCaseSensitive(src, "full_attention_interval");
        cfg->full_attention_interval = cJSON_IsNumber(fai) ? (int)fai->valuedouble : 4;
        cJSON* layer_types = cJSON_GetObjectItemCaseSensitive(src, "layer_types");
        if (cJSON_IsArray(layer_types) && cJSON_GetArraySize(layer_types) == cfg->n_layers) {
            cfg->is_linear_layer = (bool*)malloc((size_t)cfg->n_layers * sizeof(bool));
            for (int i = 0; i < cfg->n_layers; i++) {
                cJSON* lt = cJSON_GetArrayItem(layer_types, i);
                if (!cJSON_IsString(lt)) { fprintf(stderr, "load_config: layer_types[%d] non e' una stringa\n", i); cJSON_Delete(j); free(cfg->is_linear_layer); cfg->is_linear_layer = NULL; return false; }
                if (strcmp(lt->valuestring, "linear_attention") == 0) cfg->is_linear_layer[i] = true;
                else if (strcmp(lt->valuestring, "qwen_sparse_attention") == 0 || strcmp(lt->valuestring, "full_attention") == 0) cfg->is_linear_layer[i] = false;
                else { fprintf(stderr, "load_config: layer_types[%d]=\"%s\" non riconosciuto (qwen4_exp)\n", i, lt->valuestring); cJSON_Delete(j); free(cfg->is_linear_layer); cfg->is_linear_layer = NULL; return false; }
            }
        } else if (cfg->full_attention_interval > 0) {
            cfg->is_linear_layer = (bool*)malloc((size_t)cfg->n_layers * sizeof(bool));
            for (int i = 0; i < cfg->n_layers; i++)
                cfg->is_linear_layer[i] = ((i + 1) % cfg->full_attention_interval) != 0;
        }
        cJSON* plids = cJSON_GetObjectItemCaseSensitive(src, "ple_layer_ids");
        if (cJSON_IsArray(plids)) {
            int n = cJSON_GetArraySize(plids);
            if (n > 0) {
                cfg->ple_layer_ids = (int*)malloc((size_t)n * sizeof(int));
                cfg->ple_layer_ids_n = n;
                for (int i = 0; i < n; i++) {
                    cJSON* it = cJSON_GetArrayItem(plids, i);
                    cfg->ple_layer_ids[i] = cJSON_IsNumber(it) ? (int)it->valuedouble : -1;
                }
            }
        }
        {
            cJSON* v;
            v = cJSON_GetObjectItemCaseSensitive(src, "ple_embed_dim");
            cfg->ple_embed_dim = cJSON_IsNumber(v) ? (int)v->valuedouble : cfg->hidden_size;
            v = cJSON_GetObjectItemCaseSensitive(src, "ple_conv_kernel_size");
            cfg->ple_conv_kernel_size = cJSON_IsNumber(v) ? (int)v->valuedouble : 4;
            v = cJSON_GetObjectItemCaseSensitive(src, "ngram_size");
            cfg->ngram_size = cJSON_IsNumber(v) ? (int)v->valuedouble : 3;
            v = cJSON_GetObjectItemCaseSensitive(src, "heads_per_ngram");
            cfg->heads_per_ngram = cJSON_IsNumber(v) ? (int)v->valuedouble : 8;
            v = cJSON_GetObjectItemCaseSensitive(src, "ngram_vocab_size_base");
            cfg->ngram_vocab_size_base = cJSON_IsNumber(v) ? (int64_t)v->valuedouble : 20000000;
            v = cJSON_GetObjectItemCaseSensitive(src, "make_ngram_vocab_size_divisible_by");
            cfg->make_ngram_vocab_size_divisible_by = cJSON_IsNumber(v) ? (int)v->valuedouble : 128;
            v = cJSON_GetObjectItemCaseSensitive(src, "split_ngram_parts");
            cfg->split_ngram_parts = cJSON_IsNumber(v) ? (int)v->valuedouble : 128;
            v = cJSON_GetObjectItemCaseSensitive(src, "seed");
            cfg->ngram_seed = cJSON_IsNumber(v) ? (int64_t)v->valuedouble : 1234;
            v = cJSON_GetObjectItemCaseSensitive(src, "eos_token_id");
            if (cJSON_IsArray(v) && cJSON_GetArraySize(v) > 0) v = cJSON_GetArrayItem(v, 0);
            cfg->eos_token_id = cJSON_IsNumber(v) ? (int64_t)v->valuedouble : -1;
        }
    } else {
        /* Qwen3.5/3.8 e DFlash2 mettono rope_theta in rope_parameters (piatto). Fallback generico. */
        cJSON* rp = cJSON_GetObjectItemCaseSensitive(src, "rope_parameters");
        cJSON* rpt = rp ? cJSON_GetObjectItemCaseSensitive(rp, "rope_theta") : NULL;
        if (cJSON_IsNumber(rpt)) {
            cfg->rope_theta = (float)rpt->valuedouble;
            cJSON* prf = rp ? cJSON_GetObjectItemCaseSensitive(rp, "partial_rotary_factor") : NULL;
            cfg->partial_rotary_factor = cJSON_IsNumber(prf) ? (float)prf->valuedouble : 1.0f;
        } else {
            REQ_FLOAT(rope_theta, "rope_theta");
        }
    }
    /* Qwen3-Coder-Next (qwen3_next, Qwen3NextForCausalLM): hybrid GDN-conv/full, config PIATTO
     * (root, non text_config). full_attention_interval=4 => 36 layer lineari + 12 full; i layer
     * lineari usano linear_attn.{A_log,conv1d,dt_bias,in_proj_qkvz,in_proj_ba,norm,out_proj}
     * (proiezioni MERGED, non separate come qwen3_5). I MoE params (num_experts/per_tok, moe_size)
     * sono gia' letti dal blocco comune qui sopra. */
    if (strcmp(cfg->model_type, "qwen3_next") == 0) {
        cJSON* fai = cJSON_GetObjectItemCaseSensitive(src, "full_attention_interval");
        if (cJSON_IsNumber(fai)) cfg->full_attention_interval = (int)fai->valuedouble;
        cfg->linear_num_key_heads   = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_num_key_heads"));
        cfg->linear_num_value_heads = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_num_value_heads"));
        cfg->linear_key_head_dim    = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_key_head_dim"));
        cfg->linear_value_head_dim  = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_value_head_dim"));
        cfg->linear_conv_kernel_dim = (int)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(src, "linear_conv_kernel_dim"));
        if (cfg->full_attention_interval > 0) {
            cfg->is_linear_layer = (bool*)malloc((size_t)cfg->n_layers * sizeof(bool));
            for (int i = 0; i < cfg->n_layers; i++)
                cfg->is_linear_layer[i] = ((i + 1) % cfg->full_attention_interval) != 0;
        }
    }
#undef REQ_INT
#undef REQ_FLOAT
#undef OPT_INT
#undef OPT_FLOAT
#undef OPT_BOOL

    /* E3 KV quant + Rolling-Window KV (fase 3 paging): {kvQuantize/.../kvMaxTokens} erano
     * letti SOLO nel branch qwen3_5/qwen3_5_moe sopra -- per ogni altra architettura (llama,
     * qwen2, mixtral, gemma4, ...) il campo restava sempre 0/false qualunque cosa contenesse
     * config.json, rendendo LLMMETAL_KVMETAL (che richiede kv_max_tokens>0, wired in
     * llama_attention_block) impossibile da verificare end-to-end -- scoperto 2026-09-07
     * mentre si eseguiva la verifica dichiarata "prossimo passo" dal commit 8097e8aa. Qui si
     * applica lo stesso parsing genericamente (skip se gia' fatto sopra), nessun default
     * cambiato per chi non imposta questi campi. */
    if (!(strcmp(cfg->model_type, "qwen3_5") == 0 || strcmp(cfg->model_type, "qwen3_5_moe") == 0)) {
        cJSON* kvq = cJSON_GetObjectItemCaseSensitive(src, "kvQuantize");
        cfg->kv_quantize = cJSON_IsTrue(kvq);
        cJSON* kvqb = cJSON_GetObjectItemCaseSensitive(src, "kvQuantizeBits");
        cfg->kv_quantize_bits = cJSON_IsNumber(kvqb) ? (int)kvqb->valuedouble : 0;
        cJSON* kvqt = cJSON_GetObjectItemCaseSensitive(src, "kvQuantizeThreshold");
        cfg->kv_quantize_threshold = cJSON_IsNumber(kvqt) ? (int)kvqt->valuedouble : 0;
        cJSON* kvm = cJSON_GetObjectItemCaseSensitive(src, "kvMaxTokens");
        cfg->kv_max_tokens = cJSON_IsNumber(kvm) ? (int)kvm->valuedouble : 0;
    }
    /* LLMMETAL_KVMETAL richiede kv_max_tokens>0 (pool a capacita' fissa). Molte arch non lo
     * impostano in config.json -> il path paging restava inverificabile end-to-end. Knob
     * diagnostico: se impostato, forza la capacita' KV (una cap >= lunghezza contestuale
     * NON tronca nulla, quindi e' corretto per il full-attention e attiva il paging).
     * Nessun effetto se la variabile manca (comportamento identico a prima). */
    const char* kve = llm_ai_mlx_opt_env("LLM_KV_MAX_TOKENS");
    if (kve) {
        int v = atoi(kve);
        if (v > 0) cfg->kv_max_tokens = v;
    }

    /* norm_topk_prob: default TRUE per qwen3_5/qwen3_5_moe (il dataclass mlx_lm
     * TextModelArgs ha norm_topk_prob: bool = True; config None -> True). Il parsing
     * usa src (text_config per i nested), NON j: per qwen3_5_moe il campo vive in
     * text_config. Prima leggeva da j e default false -> il MoE non normalizzava gli
     * scores mentre il reference sì (divergeva il layer 0). */
    bool moe_def_norm_topk = has_nested_text_config;
    cJSON* norm_topk = cJSON_GetObjectItemCaseSensitive(src, "norm_topk_prob");
    cfg->norm_topk_prob = cJSON_IsBool(norm_topk) ? cJSON_IsTrue(norm_topk) : moe_def_norm_topk;
    /* kimi_linear: la rinormalizzazione dei top-k e' `moe_renormalize` (reference
     * _group_expert_select), il config NON ha `norm_topk_prob` -> senza questo il parse
     * generico qui sopra lo azzera e il MoE non rinormalizza (routed ~3x troppo grande). */
    if (strcmp(cfg->model_type, "kimi_linear") == 0 && !cJSON_IsBool(norm_topk))
        cfg->norm_topk_prob = cfg->moe_renormalize;
    cJSON* tie = cJSON_GetObjectItemCaseSensitive(j, "tie_word_embeddings");
    cfg->tie_word_embeddings = cJSON_IsBool(tie) ? cJSON_IsTrue(tie) : false;

    /* FASE 1b.2 (Llama 3.x): rope_scaling.rope_type == "llama3" -- letto dalla radice
     * (mai annidato per questa famiglia), assente/altro valore per ogni altra architettura
     * verificata finora (nessuna collisione nota con rope_parameters di Gemma4, chiave
     * diversa). mlx_lm accetta sia "rope_type" sia "type" per la stessa chiave -- letti
     * entrambi, il primo trovato vince (stesso ordine di initialize_rope in rope_utils.py). */
    cJSON* rope_scaling = cJSON_GetObjectItemCaseSensitive(j, "rope_scaling");
    cJSON* rope_type_item = rope_scaling ? cJSON_GetObjectItemCaseSensitive(rope_scaling, "rope_type") : NULL;
    if (!rope_type_item && rope_scaling) rope_type_item = cJSON_GetObjectItemCaseSensitive(rope_scaling, "type");
    cfg->use_llama3_rope = rope_scaling && cJSON_IsString(rope_type_item) && strcmp(rope_type_item->valuestring, "llama3") == 0;
    if (cfg->use_llama3_rope) {
        cJSON* factor = cJSON_GetObjectItemCaseSensitive(rope_scaling, "factor");
        cJSON* low = cJSON_GetObjectItemCaseSensitive(rope_scaling, "low_freq_factor");
        cJSON* high = cJSON_GetObjectItemCaseSensitive(rope_scaling, "high_freq_factor");
        cJSON* old_ctx = cJSON_GetObjectItemCaseSensitive(rope_scaling, "original_max_position_embeddings");
        if (!cJSON_IsNumber(factor)) {
            fprintf(stderr, "load_config: rope_scaling.factor mancante per rope_type=llama3\n");
            cJSON_Delete(j);
            return false;
        }
        cfg->llama3_rope_factor = (float)factor->valuedouble;
        cfg->llama3_low_freq_factor = cJSON_IsNumber(low) ? (float)low->valuedouble : 1.0f;
        cfg->llama3_high_freq_factor = cJSON_IsNumber(high) ? (float)high->valuedouble : 4.0f;
        cfg->llama3_old_context_len = cJSON_IsNumber(old_ctx) ? (float)old_ctx->valuedouble : 8192.0f;
    }
    /* YaRN (FASE 1b.7): rope_scaling.rope_type == "yarn" (o "deepseek_yarn"/"telechat3-yarn",
     * stessi parametri -- mlx_lm initialize_rope li tratta allo stesso modo). Stessa chiave
     * radice `rope_scaling` di llama3. Default: use_yarn_rope=false. */
    cfg->use_yarn_rope = rope_scaling && cJSON_IsString(rope_type_item) &&
        (strcmp(rope_type_item->valuestring, "yarn") == 0 ||
         strcmp(rope_type_item->valuestring, "deepseek_yarn") == 0 ||
         strcmp(rope_type_item->valuestring, "telechat3-yarn") == 0);
    if (cfg->use_yarn_rope) {
        cJSON* factor = cJSON_GetObjectItemCaseSensitive(rope_scaling, "factor");
        cJSON* orig = cJSON_GetObjectItemCaseSensitive(rope_scaling, "original_max_position_embeddings");
        cJSON* bf = cJSON_GetObjectItemCaseSensitive(rope_scaling, "beta_fast");
        cJSON* bs = cJSON_GetObjectItemCaseSensitive(rope_scaling, "beta_slow");
        cJSON* ms = cJSON_GetObjectItemCaseSensitive(rope_scaling, "mscale");
        if (!cJSON_IsNumber(factor)) {
            fprintf(stderr, "load_config: rope_scaling.factor mancante per rope_type=%s\n", rope_type_item->valuestring);
            cJSON_Delete(j);
            return false;
        }
        cfg->yarn_factor = (float)factor->valuedouble;
        cfg->yarn_orig_max = cJSON_IsNumber(orig) ? (int)orig->valuedouble : 4096;
        cfg->yarn_beta_fast = cJSON_IsNumber(bf) ? (float)bf->valuedouble : 32.0f;
        cfg->yarn_beta_slow = cJSON_IsNumber(bs) ? (float)bs->valuedouble : 1.0f;
        cfg->yarn_mscale = cJSON_IsNumber(ms) ? (float)ms->valuedouble : 1.0f;
        cJSON* mad = cJSON_GetObjectItemCaseSensitive(rope_scaling, "mscale_all_dim");
        cfg->yarn_mscale_all_dim = cJSON_IsNumber(mad) ? (float)mad->valuedouble : 0.0f;
    }

    cJSON* quant = cJSON_GetObjectItemCaseSensitive(j, "quantization");
    cJSON* qgs = quant ? cJSON_GetObjectItemCaseSensitive(quant, "group_size") : NULL;
    cJSON* qbits = quant ? cJSON_GetObjectItemCaseSensitive(quant, "bits") : NULL;
    /* §13.6: modello NON quantizzato (draft BF16/FP32 pieni, es. Qwen3.8-27B-DFlash2 che non ha
     * la chiave quantization). quant_mode="none" -> quantized_linear_cfg delega a plain_linear;
     * group/bits restano 0 (non usati). I modelli quantizzati (affine/mxfp4/...) hanno sempre
     * quantization.group_size/bits. */
    if (!cJSON_IsNumber(qgs) || !cJSON_IsNumber(qbits)) {
        llm_mlx_mode_set(cfg->quant_mode, sizeof(cfg->quant_mode), "none");
        cfg->quant_group_size = 0;
        cfg->quant_bits = 0;
    } else {
        cfg->quant_group_size = (int)qgs->valuedouble;
        cfg->quant_bits = (int)qbits->valuedouble;
        /* modalita' di quantizzazione (mlx quantized_matmul mode): default affine, ma
         * supporta mxfp4/mxfp8/nvfp4 (es. modello DFlash2). Backward-compat: assente -> affine. */
        cJSON* qmode = quant ? cJSON_GetObjectItemCaseSensitive(quant, "mode") : NULL;
        snprintf(cfg->quant_mode, sizeof(cfg->quant_mode), "%s",
                 (cJSON_IsString(qmode) && qmode->valuestring) ? qmode->valuestring : "affine");
        /* TurboQuant (PolarQuant-MLX): NON è un modo di MLX quantized_matmul. I pesi vengono
         * dequantizzati a load da turboquant_synthesize_weights in `.weight` F16 plain ->
         * il percorso caldo deve essere plain_linear, quindi quant_mode = "none". */
        if (strcmp(cfg->quant_mode, "turboquant") == 0) {
            llm_mlx_mode_set(cfg->quant_mode, sizeof(cfg->quant_mode), "none");
            cfg->quant_group_size = 0;
            cfg->quant_bits = 0;
            /* AUTO-SWAP (config-driven, nessun flag/env): gli esperti tq sono packed (ternary
             * {'-c,0,+c}) e non stanno in RAM materializzati. Se il modello è MoE (n_experts>0) si
             * abilita lo streaming degli esperti dalla dir del modello: il synth salta `switch_mlp`
             * (`turboquant_synthesize_weights`) e il path esperto li converte on-the-fly in affine
             * 2-bit (gather_qmm GPU). Un modello DENSE tq non passa da llm_mlx_moe_gather_streamed,
             * quindi non si abilita. Se l'ingest non apre, `llm_mlx_moe_swap_active()` resta falso e
             * il synth materializza comunque (resident F16): mai spazzatura. */
            if (cfg->n_experts > 0) llm_ai_mlx_set_moe_swap_dir(model_dir);
            /* Codebook GLOBALE TurboQuant (`quantization.codebook`, identico in tutti i tensori tq):
             * serve al path TQ dense (opzione 2). Config-driven. */
            cJSON* tqcb = quant ? cJSON_GetObjectItemCaseSensitive(quant, "codebook") : NULL;
            if (cJSON_IsArray(tqcb)) {
                float cb[16]; int n = 0; cJSON* it = NULL;
                cJSON_ArrayForEach(it, tqcb) { if (n < 16 && cJSON_IsNumber(it)) cb[n++] = (float)it->valuedouble; }
                llm_mlx_tq_set_cb(cb, n);
            }
        }
        /* override per-tensore presenti? (chiavi figlie di "quantization" contenenti un punto,
         * es. "model.embed_tokens", "model.layers.0.mlp.experts...") */
        cfg->per_tensor_quant = 0;
        if (quant && cJSON_IsObject(quant)) {
            cJSON* child = NULL;
            cJSON_ArrayForEach(child, quant) {
                if (child->string && strchr(child->string, '.') != NULL) { cfg->per_tensor_quant = 1; break; }
            }
        }
    }
    if (getenv("LLM_BONSAI_DBG")) fprintf(stderr, "[cfg] quant mode=%s group=%d bits=%d model_type=%s prefix=%s\n",
                                          cfg->quant_mode, cfg->quant_group_size, cfg->quant_bits, cfg->model_type, cfg->layer_prefix);
    if (getenv("LLM_BONSAI_DBG")) fprintf(stderr, "[cfg] gdn Hk=%d Hv=%d Dk=%d Dv=%d convk=%d convkdim=%d attn_hd=%d\n",
                                          cfg->linear_num_key_heads, cfg->linear_num_value_heads, cfg->linear_key_head_dim,
                                          cfg->linear_value_head_dim, cfg->linear_attn_conv_kernel, cfg->linear_conv_kernel_dim, cfg->linear_attn_head_dim);
    /* qwen4_exp: output gate = sigmoid SE (output_gate_type or hidden_act)=="sigmoid", altrimenti silu. */
    {
        cJSON* ogt = cJSON_GetObjectItemCaseSensitive(j, "output_gate_type");
        cJSON* ha = cJSON_GetObjectItemCaseSensitive(j, "hidden_act");
        const char* og = NULL;
        if (cJSON_IsObject(src)) { cJSON* t = cJSON_GetObjectItemCaseSensitive(src, "output_gate_type"); if (cJSON_IsString(t)) og = t->valuestring; }
        if (og == NULL && cJSON_IsString(ogt)) og = ogt->valuestring;
        if (og == NULL) { cJSON* h2 = cJSON_IsObject(src) ? cJSON_GetObjectItemCaseSensitive(src, "hidden_act") : NULL; if (cJSON_IsString(h2)) og = h2->valuestring; }
        if (og == NULL && cJSON_IsString(ha)) og = ha->valuestring;
        cfg->q4_gate_silu = !(og != NULL && strcmp(og, "sigmoid") == 0);
    }
    /* Prefisso dei layer: default "model.layers." (qwen/llama/mistral...). DFlash2 usa
     * "layers." (niente "model."). Qwen3.8-27B (qwen3_5) usa "language_model.model.layers."
     * (come gemma4). L'override QUI, DOPO il default (che altrimenti lo sovrascriverebbe). */
    llm_mlx_mode_set(cfg->layer_prefix, sizeof(cfg->layer_prefix), "model.layers.");
    if (strcmp(cfg->model_type, "qwen3_5") == 0 || strcmp(cfg->model_type, "qwen3_5_moe") == 0
        || strcmp(cfg->model_type, "qwen4_exp") == 0)
        llm_mlx_mode_set(cfg->layer_prefix, sizeof(cfg->layer_prefix), "language_model.model.layers.");
    /* DFlash2 DRAFT (config.json ha dflash_config + arch DFlash2DraftModel): prefisso "layers.". */
    if (cJSON_GetObjectItemCaseSensitive(j, "dflash_config") != NULL)
        llm_mlx_mode_set(cfg->layer_prefix, sizeof(cfg->layer_prefix), "layers.");
    /* DFlash2 DRAFT: attn_dim (640) derivato dal config se presente, altrimenti 0 = hidden. */
    cJSON* ad = cJSON_GetObjectItemCaseSensitive(src, "attn_dim");
    cfg->attn_dim = cJSON_IsNumber(ad) ? (int)ad->valuedouble : 0;
    /* DFlash2 DRAFT: campi dflash_config (block-diffusion drafter). Se assenti restano 0
     * (qualsiasi altra architettura ignora il blocco). */
    cJSON* dc = cJSON_GetObjectItemCaseSensitive(j, "dflash_config");
    if (cJSON_IsObject(dc)) {
        cfg->dflash_block_size = (int)(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(dc, "block_size")) ? cJSON_GetObjectItemCaseSensitive(dc, "block_size")->valuedouble : 0);
        cfg->dflash_conv_kernel = (int)(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(dc, "conv_kernel_size")) ? cJSON_GetObjectItemCaseSensitive(dc, "conv_kernel_size")->valuedouble : 0);
        cfg->dflash_conv_group = (int)(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(dc, "conv_group_size")) ? cJSON_GetObjectItemCaseSensitive(dc, "conv_group_size")->valuedouble : 0);
        cfg->dflash_selector_rank = (int)(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(dc, "selector_rank")) ? cJSON_GetObjectItemCaseSensitive(dc, "selector_rank")->valuedouble : 0);
        cfg->dflash_selector_top_k = (int)(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(dc, "selector_top_k")) ? cJSON_GetObjectItemCaseSensitive(dc, "selector_top_k")->valuedouble : 0);
        cfg->dflash_mask_token = (int)(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(dc, "mask_token_id")) ? cJSON_GetObjectItemCaseSensitive(dc, "mask_token_id")->valuedouble : 0);
        cfg->dflash_n_ctx_layers = 0;
        cJSON* tl = cJSON_GetObjectItemCaseSensitive(dc, "target_layer_ids");
        if (cJSON_IsArray(tl)) {
            int n = cJSON_GetArraySize(tl);
            if (n > 8) n = 8;
            for (int i = 0; i < n; i++) {
                cJSON* it = cJSON_GetArrayItem(tl, i);
                if (cJSON_IsNumber(it)) cfg->dflash_ctx_layers[cfg->dflash_n_ctx_layers++] = (int)it->valuedouble;
            }
        }
        /* sliding_window dei layer sliding del draft (tutti i 5 layer): letto dal config
         * root per il draft dflash (altre architetture non lo leggono qui). */
        cJSON* sw = cJSON_GetObjectItemCaseSensitive(j, "sliding_window");
        if (cJSON_IsNumber(sw)) cfg->sliding_window = (int)sw->valuedouble;
    }

    cJSON_Delete(j);
    return true;
}
