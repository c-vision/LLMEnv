/* llmcodec_gguf_meta.h — costruzione dei METADATA GGUF (hyperparametri + tokenizer) da una dir sorgente
 * (config.json + tokenizer.json) per una data arch. Usato dal converter MLX/safetensors -> GGUF:
 * senza questi KV un GGUF è tensori-only e llama.cpp/LLMEnv non lo caricano. */
#ifndef LLMCODEC_GGUF_META_H
#define LLMCODEC_GGUF_META_H
#include "llmcodec_gguf_read.h"

/* Tokenizer pre-digerito dal PROGETTO tokenizer (unica sorgente di tokenizer.json).
 * Il codec NON ri-parsa il file: riceve i dati e li emette. Le stringhe sono
 * possedute dal chiamante e devono restare valide fino a llmcodec_gguf_write. */
typedef struct {
    const char*        model;      /* "gpt2" | "llama" */
    const char* const* tokens;     /* id -> token (denso, n_tokens voci) */
    int                n_tokens;
    const char* const* merges;     /* merge in ordine di rango ("left right") */
    int                n_merges;
    const int32_t*     token_types;/* id -> tipo GGUF (1 normal,3 control,4 user_defined); opzionale */
    int                n_token_types;
} LlmCodecTokenizerMeta;

/* Riempie *out_kv (malloc, *out_n elementi) coi KV: general.architecture, <arch>.<hyperparam>,
 * tokenizer.ggml.*. `tm` puo' essere NULL (nessun KV tokenizer). 0 ok / -1.
 * Liberare con llmcodec_gguf_meta_free_built. */
int llmcodec_gguf_meta_build(const char* model_dir, const char* arch, const LlmCodecTokenizerMeta* tm,
                        const char* tokenizer_pre,
                        LlmCodecGgufMetaKV** out_kv, int* out_n);
void llmcodec_gguf_meta_free_built(LlmCodecGgufMetaKV* kv, int n);

#endif
