/* llmengine.h — ABI COMUNE dei runner di inferenza LLM (MLX, llama.cpp, MoE).
 *
 * PERCHÉ. I motori espongono API native diverse: MLX lavora a sessioni di token-id (la
 * tokenizzazione sta a monte), llama.cpp a chat di testo (possiede tokenizer e template),
 * MoE a forward/runtime. Il runtime e il C# NON devono conoscere l'uno o l'altro: parlano
 * questa ABI. Ogni runner esporta la PROPRIA vtable (`llmmodelio_engine_<nome>()`), delegando
 * alle sue funzioni native (una sola implementazione per motore, §0bis). Le voci non
 * supportate da un motore sono NULL, dichiarate da `caps`.
 *
 * CONVENZIONI. Handles opachi (LlmEngineLlmModel/Session). Le stringhe ritornate da
 * generate_text/detokenize sono malloc e vanno liberate con `free_str`. I metodi ritornano
 * 0/-1 (o il conteggio reale, dove indicato).
 */
#ifndef LLMENGINE_H
#define LLMENGINE_H

#include <stdint.h>

/* Versione dell'ABI. Incrementata quando la vtable cambia in modo osservabile dal consumatore
 * (nuovi metodi/capabilities in coda, semantica di quelli esistenti). I consumatori possono
 * confrontarla a runtime via `llmengine_abi_version()` per decidere se una voce opzionale e'
 * disponibile, PRIMA di dereferenziarla. v2 = logprobs + constrained + kv_stats/kv_clear (MLX).
 * v3 = token_topk (distribuzione top-k alla posizione pending, MLX).
 * v4 = embed_ids (embedding IDS-based, MLX).
 * v5 = set_expert_swap (offload esperti MoE da disco, MLX). */
#define LLMENGINE_ABI_VERSION 6u

#ifdef __cplusplus
extern "C" {
#endif

/* Capacità: quali gruppi di metodi sono validi (gli altri sono NULL nella vtable). */
enum {
    LLMMODELIO_CAP_IDS      = 1 << 0,   /* append_ids / generate_ids */
    LLMMODELIO_CAP_TEXT     = 1 << 1,   /* append_text / generate_text */
    LLMMODELIO_CAP_TOKENIZE = 1 << 2,   /* tokenize / detokenize */
    LLMMODELIO_CAP_KV       = 1 << 3,   /* kv_clear / kv_stats */
    LLMMODELIO_CAP_PREFIX   = 1 << 4,   /* append_ids_prefix / save_prefix (prefix cache) */
    LLMMODELIO_CAP_BATCH    = 1 << 5,   /* generate_batch (continuous batching) */
    LLMMODELIO_CAP_CHUNK    = 1 << 6,   /* set_prefill_chunk (chunked prefill) */
    LLMMODELIO_CAP_EMBED    = 1 << 7,   /* emb_dim / embed_text */
    LLMMODELIO_CAP_VISION   = 1 << 8,   /* vision_load / vision_read_image / vision_read_audio / vision_unload */
    LLMMODELIO_CAP_LORA     = 1 << 9,   /* lora_* / capture_hidden (training; oggi solo MLX) */
    LLMMODELIO_CAP_LOGPROBS = 1 << 10,  /* token_logprob / last_logprob / generate_ids_probs */
    LLMMODELIO_CAP_CONSTRAINED = 1 << 11, /* generate_id_constrained (argmax ristretto) */
    LLMMODELIO_CAP_KV_MAX    = 1 << 12,  /* kv_set_max: dimensione contesto/KV a runtime */
};

/* Parametri di sampling (sottoinsieme comune). I campi non supportati sono ignorati. */
typedef struct {
    float   temperature, top_p;
    int     top_k;
    float   repetition_penalty;     /* 1.0 = off */
    float   frequency_penalty;      /* 0.0 = off */
    float   presence_penalty;       /* 0.0 = off */
    int64_t seed;
} LlmEngineLlmSampling;

typedef void* LlmEngineLlmModel;
typedef void* LlmEngineLlmSession;
typedef void* LlmEngineLlmVision;    /* handle OCR/visione (mmproj) */

typedef struct LlmEngineLlmEngine {
    const char* name;
    unsigned    caps;

    LlmEngineLlmModel   (*load_model)(const char* path);
    LlmEngineLlmModel   (*load_model_opts)(const char* path, int n_gpu_layers);   /* opz.; NULL -> load_model */
    void          (*free_model)(LlmEngineLlmModel m);

    LlmEngineLlmSession (*create_session)(LlmEngineLlmModel m);
    void          (*destroy_session)(LlmEngineLlmSession s);

    int   (*append_ids)(LlmEngineLlmSession s, const int32_t* ids, int n);          /* CAP_IDS  */
    int   (*append_text)(LlmEngineLlmSession s, const char* role, const char* text);/* CAP_TEXT */
    int   (*generate_ids)(LlmEngineLlmSession s, int max_new, int32_t eos, int32_t* out); /* CAP_IDS; ritorna il conteggio reale */
    char* (*generate_text)(LlmEngineLlmSession s, int max_new);                     /* CAP_TEXT; malloc, libera con free_str */

    int   (*set_sampling)(LlmEngineLlmSession s, const LlmEngineLlmSampling* sp);
    int   (*kv_clear)(LlmEngineLlmSession s);                                       /* CAP_KV */
    int   (*kv_stats)(LlmEngineLlmSession s, int* out_used, int* out_cap);          /* CAP_KV */

    int   (*tokenize)(LlmEngineLlmModel m, const char* text, int32_t* out_ids, int cap); /* CAP_TOKENIZE */
    char* (*detokenize)(LlmEngineLlmModel m, const int32_t* ids, int n);            /* CAP_TOKENIZE; malloc */

    /* Estensioni OPZIONALI (NULL se il motore non le ha). */
    int   (*set_prefill_chunk)(LlmEngineLlmSession s, int chunk);                    /* CAP_CHUNK */
    int   (*append_ids_prefix)(LlmEngineLlmModel m, LlmEngineLlmSession s, const int32_t* ids, int n, int min_match); /* CAP_PREFIX */
    int   (*save_prefix)(LlmEngineLlmModel m, const char* name, LlmEngineLlmSession s);    /* CAP_PREFIX */
    int   (*generate_batch)(LlmEngineLlmSession* sess, int n, int max_new, int32_t eos, int32_t* out); /* CAP_BATCH */
    /* Contesto/KV a runtime per TUTTI i modelli del motore (CAP_KV_MAX): imposta la capienza della
     * KV-cache in token (0 = default del modello). Ritorna 0 se accettato, !=0 se non supportato. */
    int   (*kv_set_max)(LlmEngineLlmModel m, int n);

    /* EMBED (CAP_EMBED): embedding testuale del modello. */
    int   (*emb_dim)(LlmEngineLlmModel m);
    int   (*embed_text)(LlmEngineLlmModel m, const char* text, float* out, int cap);

    /* VISION/OCR (CAP_VISION): encoder visione/audio (mmproj). Le stringhe vanno liberate con free_str. */
    LlmEngineLlmVision (*vision_load)(const char* model_path, const char* mmproj_path, int n_ctx, int n_gpu_layers);
    char* (*vision_read_image)(LlmEngineLlmVision v, const char* image_path, const char* prompt, int max_tokens);
    char* (*vision_read_audio)(LlmEngineLlmVision v, const char* audio_path, const char* prompt, int max_tokens);
    void  (*vision_unload)(LlmEngineLlmVision v);

    /* TRAIN/LoRA (CAP_LORA): training/adapter (oggi solo MLX). A/B sono i fattori LoRA [rank x dim]. */
    void  (*lora_clear)(LlmEngineLlmModel m);
    int   (*lora_add)(LlmEngineLlmModel m, int layer, const char* proj, int rank, float scale,
                      const float* a, const float* b, int in_dim, int out_dim);
    int   (*lora_set_enabled)(LlmEngineLlmModel m, int enabled);
    int   (*lora_save)(LlmEngineLlmModel m, const char* path);
    int   (*lora_load)(LlmEngineLlmModel m, const char* path);
    int   (*capture_hidden)(LlmEngineLlmSession s, const int32_t* ids, int n, float* out, int cap);

    void  (*free_str)(char* p);

    /* ── ABI v2 — campi NUOVI in coda: NULL/0 se il motore non li implementa. ──────────────────
     * Compatibilita': i campi esistenti sopra restano nella stessa posizione; questi si
     * appendono DOPO `free_str`, quindi una vtable costruita con un header piu' vecchio resta
     * leggibile (i campi nuovi sono NULL). Un motore v1 e' comunque utilizzabile da un consumer
     * v2: basta controllare `caps` / `llmengine_abi_version()` prima di usare un metodo. */

    /* LOGPROBS (CAP_LOGPROBS). Tutte operano sulla posizione "pending" della sessione (quella
     * che il prossimo step emetterebbe):
     *   token_logprob(s, id, out)       logprob naturale di `id` in quella distribuzione;
     *   last_logprob(s, out)            logprob del token greedy pending (pending_next_id);
     *   generate_ids_probs(s, n, eos, out_ids, out_logprobs)
     *                                   genera come generate_ids ma scrive per OGNI token emesso
     *                                   la sua logprob, catturata PRIMA di avanzare (esatta anche
     *                                   con sampling, perche' la distribuzione e' quella del
     *                                   modello alla posizione di partenza). Ritorna il conteggio
     *                                   reale (~generate_ids) o -1.                                        */
    int   (*token_logprob)(LlmEngineLlmSession s, int32_t token_id, float* out_logprob);
    int   (*last_logprob)(LlmEngineLlmSession s, float* out_logprob);
    int   (*generate_ids_probs)(LlmEngineLlmSession s, int max_new, int32_t eos,
                                int32_t* out_ids, float* out_logprobs);

    /* CONSTRAINED (CAP_CONSTRAINED): decide UN token alla posizione pending con argmax ristretto
     * ad `allowed_ids` (n_allowed id), poi avanza. L'insieme ammesso e' responsabilita' del
     * chiamante, che lo ricalcola a ogni passo (grammatica/JSON). NON usa parentesi-sintassi:
     * e' la controparte ABI di `llm_ai_mlx_session_generate_constrained`. Ritorna 0 / -1. */
    int   (*generate_id_constrained)(LlmEngineLlmSession s, const int32_t* allowed_ids, int n_allowed,
                                     int32_t* out_token_id);

    /* ── ABI v3 — campo NUOVO in coda (NULL se il motore non lo implementa). ────────────────────
     * TOPK (CAP_LOGPROBS): i k token piu' probabili alla posizione "pending" della sessione, con
     * le loro logprob naturali, ordinati per logprob DECRESCENTE (indice 0 = argmax). E' la
     * distribuzione top-k che OpenAI espone in `logprobs.content[].top_logprobs[]`, complementare
     * a `generate_ids_probs` (che da' la logprob del SOLO token scelto). Non modifica lo stato:
     * puo' essere chiamata PRIMA di un passo di generate per fotografare la distribuzione che lo
     * step emettera'. Scrive fino a k voci (k ridotto al vocabolario) e ritorna il numero scritto,
     * oppure -1 su argomenti invalidi / sessione senza posizione pending. out_ids/out_logprobs
     * hanno almeno k elementi. */
    int   (*token_topk)(LlmEngineLlmSession s, int k, int32_t* out_ids, float* out_logprobs);

    /* ── ABI v4 — campo NUOVO in coda (NULL se il motore non lo implementa). ────────────────────
     * EMBED IDS (CAP_EMBED): embedding testuale IDS-based. Il runner MLX non possiede la
     * tokenizzazione (sta a monte), quindi l'embedding si chiede sui token-id gia' prodotti dal
     * consumer con il proprio tokenizer. Ritorna la dimensione `dim` del vettore (e ne scrive
     * `dim` float in `out`, cap >= dim) oppure -1. `emb_dim` da' la dimensione senza calcolare.
     * Non usa una sessione del chiamante: e' una lettura senza stato. */
    int   (*embed_ids)(LlmEngineLlmModel m, const int32_t* ids, int n, float* out, int cap);

    /* ── ABI v5 — campo NUOVO in coda (NULL se il motore non lo implementa). ────────────────────
     * EXPERT SWAP/OFFLOAD (MLX): abilita lo streaming degli esperti MoE da DISCO invece di
     * materializzarli in RAM. `dir` è la directory del modello (di norma la stessa di load_model)
     * da cui leggere solo i k esperti attivi per step. Va chiamata PRIMA di `load_model`; `NULL`
     * o "" disattiva. E' la controparte API dell'opzione di offload degli altri engine (niente
     * variabili d'ambiente). Per modelli senza esperti (dense) è un no-op. */
    void  (*set_expert_swap)(const char* dir);

    /* ── ABI v6 — campo NUOVO in coda (NULL se il motore non lo implementa). ────────────────────
     * MASKED (grammatica con BITMASK, vedi llm_grammar.h): come `generate_id_constrained` ma l'insieme ammesso e'
     * una bitmask sul vocabolario (1 bit per token, bit i della parola i/32), 19 KB per 150k token invece di un
     * array di id per passo. Decide UN token alla posizione pending con argmax ristretto ai bit accesi, poi avanza.
     * Greedy (la temperatura non si applica). `n_words` = (n_vocab+31)/32. Ritorna 0 / -1 (nessun bit valido,
     * sessione senza pending, modello delegato). */
    int   (*generate_id_masked)(LlmEngineLlmSession s, const uint32_t* mask_words, int n_words, int32_t* out_token_id);
} LlmEngineLlmEngine;

/* Versione dell'ABI esportata dal runner (== LLMENGINE_ABI_VERSION dell'header con cui e' stato
 * compilato). Confrontala con la macro dell'header del consumatore per rilevare disallineamenti. */
unsigned llmengine_abi_version(void);

/* Accessor per-runner: una vtable per motore (implementata nel progetto del motore). */
const LlmEngineLlmEngine* llmengine_llm_engine_mlx(void);      /* llmengine_mlx */
const LlmEngineLlmEngine* llmengine_llm_engine_ds4(void);
/* Selezione automatica: directory/MLX -> motore MLX; file GGUF -> motore ds4 (se linkato). */
const LlmEngineLlmEngine* llmengine_engine_for_model(const char* model_path);      /* ds4 (motore DS4: vendorizzato in LLMEngine/ds4; il nativo e' ritirato) */
const LlmEngineLlmEngine* llmengine_llm_engine_ds41(void);     /* ds41-native (nostra impl. DS4.1 nativa) */
const LlmEngineLlmEngine* llmengine_llm_engine_llamacpp(void); /* llm_host_ai */
const LlmEngineLlmEngine* llmengine_llm_engine_moe(void);      /* llmengine_moe */

#ifdef __cplusplus
}
#endif
#endif /* LLMENGINE_H */
