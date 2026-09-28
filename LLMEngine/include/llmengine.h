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
 * disponibile, PRIMA di dereferenziarla. v2 = logprobs + constrained + kv_stats/kv_clear (MLX). */
#define LLMENGINE_ABI_VERSION 2u

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
} LlmEngineLlmEngine;

/* Versione dell'ABI esportata dal runner (== LLMENGINE_ABI_VERSION dell'header con cui e' stato
 * compilato). Confrontala con la macro dell'header del consumatore per rilevare disallineamenti. */
unsigned llmengine_abi_version(void);

/* Accessor per-runner: una vtable per motore (implementata nel progetto del motore). */
const LlmEngineLlmEngine* llmengine_llm_engine_mlx(void);      /* llmengine_mlx */
const LlmEngineLlmEngine* llmengine_llm_engine_llamacpp(void); /* llm_host_ai */
const LlmEngineLlmEngine* llmengine_llm_engine_moe(void);      /* llmengine_moe */

#ifdef __cplusplus
}
#endif
#endif /* LLMENGINE_H */
