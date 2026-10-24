/* FASE 5B: OCR/visione embedded via llama.cpp mtmd. Vedi llmengine_llama_ocr.h.
 *
 * Riscrittura in C puro (2026-08-24) -- vedi Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md.
 *
 * Design invariato: il modello testo (GGUF) è caricato con llama.cpp esattamente
 * come in llmengine_llama.c; il vision encoder (mmproj) è inizializzato con
 * mtmd (tools/mtmd, vendored). Ogni chiamata a llm_ai_ocr_read_image:
 *   1. carica l'immagine da path (mtmd_helper_bitmap_init_from_file, stb_image
 *      vendored -- nessuna decodifica da fare in C#),
 *   2. tokenizza testo + immagine in chunks (mtmd_tokenize),
 *   3. eval dei chunk (mtmd_helper_eval_chunk_single) per il prefill,
 *   4. genera i token risposta con llama_decode + sampler, fermandosi su EOG.
 * Il KV-cache è usa-e-getta per immagine (niente sessione persistente): coerente
 * con il caso d'uso OCR, dove ogni richiesta è un'immagine nuova.
 */

#include "llmengine_llama_ocr.h"

#include <llama.h>
#include "mtmd.h"
#include "mtmd-helper.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* llama.h/mtmd.h dichiarano alcuni tipi come "struct nome;" nudi (senza
 * typedef) -- funziona in C++ ma non in C puro. mtmd.h fornisce già i
 * typedef per i propri tipi opachi (mtmd_context, mtmd_bitmap,
 * mtmd_input_chunk(s), mtmd_input_text) -- solo quelli di llama.h e
 * mtmd_context_params (struct nudo anche in mtmd.h) mancano qui. */
typedef struct llama_model llama_model;
typedef struct llama_vocab llama_vocab;
typedef struct llama_context llama_context;
typedef struct llama_sampler llama_sampler;
typedef struct llama_model_params llama_model_params;
typedef struct llama_context_params llama_context_params;
typedef struct mtmd_context_params mtmd_context_params;

/* Il nome deve combaciare con "struct llm_ai_ocr_ctx*" dichiarato (opaco) in
 * llmengine_llama_ocr.h -- cosi' l'handle e' un puntatore diretto, nessun cast
 * necessario (l'originale C++ usava reinterpret_cast tra un nome interno
 * diverso, OcRContext, e il tipo opaco -- qui non serve). */
struct llm_ai_ocr_ctx {
    llama_model* model;
    const llama_vocab* vocab;
    llama_context* lctx;
    mtmd_context* mtmd_ctx;
    llama_sampler* sampler;
    int32_t n_ctx;
    int32_t n_batch;
    /* true per i modelli Qwen (qwen2vl / Qwen2.5-Omni): il loro chat template è
     * ChatML con system prompt, non il framing "User:/Assistant:" di PaddleOCR. */
    bool chatml;
};

static bool init_backend_done = false;

llm_ai_ocr_handle llm_ai_ocr_load(const char* model_path, const char* mmproj_path,
                                  int n_ctx, int n_gpu_layers) {
    if (model_path == NULL || mmproj_path == NULL) return NULL;

    if (!init_backend_done) {
        llama_backend_init();
        init_backend_done = true;
    }

    struct llm_ai_ocr_ctx* ctx = (struct llm_ai_ocr_ctx*)calloc(1, sizeof(struct llm_ai_ocr_ctx));
    ctx->n_ctx = n_ctx > 0 ? n_ctx : 4096;
    ctx->n_batch = ctx->n_ctx > 512 ? 512 : ctx->n_ctx;

    /* 1. Modello testo (GGUF) -- stessi params del wrapper chat. */
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    ctx->model = llama_model_load_from_file(model_path, model_params);
    if (ctx->model == NULL) { free(ctx); return NULL; }
    ctx->vocab = llama_model_get_vocab(ctx->model);

    /* Rileva la famiglia del modello dal GGUF (general.architecture): i Qwen
     * (qwen2vl / Qwen2.5-Omni) richiedono ChatML, PaddleOCR il framing
     * "User:/Assistant:". */
    {
        char arch[64] = {0};
        if (llama_model_meta_val_str(ctx->model, "general.architecture", arch, sizeof(arch)) > 0) {
            ctx->chatml = strstr(arch, "qwen") != NULL;
        }
    }

    /* 2. Context llama per il testo. */
    llama_context_params lctx_params = llama_context_default_params();
    lctx_params.n_ctx = (uint32_t)ctx->n_ctx;
    lctx_params.n_batch = (uint32_t)ctx->n_batch;
    ctx->lctx = llama_init_from_model(ctx->model, lctx_params);
    if (ctx->lctx == NULL) {
        llama_model_free(ctx->model);
        free(ctx);
        return NULL;
    }

    /* 3. Encoder visione (mmproj) via mtmd. */
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    ctx->mtmd_ctx = mtmd_init_from_file(mmproj_path, ctx->model, mtmd_params);
    if (ctx->mtmd_ctx == NULL) {
        llama_free(ctx->lctx);
        llama_model_free(ctx->model);
        free(ctx);
        return NULL;
    }

    /* 3b. Sampler semplice (greedy con temp bassa -- OCR deve essere determinista). */
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    ctx->sampler = sampler;

    return ctx;
}

/* Legge un campo stringa da un descriptor .gmmod (JSON minimale). 0/-1. */
static int gmmod_field(const char* path, const char* key, char* out, size_t cap) {
    if (out && cap) out[0] = 0;
    FILE* f = fopen(path, "rb"); if (f == NULL) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (1 << 20)) { fclose(f); return -1; }
    char* b = (char*)malloc((size_t)n + 1);
    if (b == NULL) { fclose(f); return -1; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return -1; }
    b[n] = 0; fclose(f);
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(b, pat);
    int ok = 0;
    if (p != NULL) {
        p += strlen(pat); while (*p == ' ' || *p == '\t') p++;
        if (*p == ':') { p++; while (*p == ' ' || *p == '\t') p++;
            if (*p == '"') { p++; size_t i = 0; while (*p && *p != '"' && i < cap - 1) out[i++] = *p++; out[i] = 0; ok = i > 0; } }
    }
    free(b);
    return ok ? 0 : -1;
}

/* Loader guidato dal descriptor .gmmod: base + mmproj dalla DICHIARAZIONE (non dal naming). */
llm_ai_ocr_handle llm_ai_ocr_load_gmmod(const char* gmmod_path, int n_ctx, int n_gpu_layers) {
    if (gmmod_path == NULL) return NULL;
    char base[2048] = {0}, mmproj[2048] = {0};
    if (gmmod_field(gmmod_path, "base", base, sizeof base) != 0) return NULL;
    if (gmmod_field(gmmod_path, "mmproj", mmproj, sizeof mmproj) != 0) return NULL;
    return llm_ai_ocr_load(base, mmproj, n_ctx, n_gpu_layers);
}


/* run_media_pipeline: percorso comune per immagine e audio -- carica il media da
 * file (auto-detect magic bytes via mtmd helper), tokenizza testo+media, eval dei
 * chunk (prefill), genera fino a max_tokens o EOG. Ritorna stringa heap-allocata
 * o NULL su errore. */
static char* run_media_pipeline(struct llm_ai_ocr_ctx* ctx, const char* media_path,
                                 const char* prompt, size_t prompt_len, int max_tokens) {
    char* result = NULL;
    mtmd_bitmap* bitmap = NULL;
    mtmd_input_chunks* chunks = NULL;

    /* 1. Carica il media (immagine via stb_image, audio via miniaudio -- auto-detect). */
    struct mtmd_helper_bitmap_wrapper bitmap_wrap = mtmd_helper_bitmap_init_from_file(ctx->mtmd_ctx, media_path, false, mtmd_helper_init_opt_default());
    if (!bitmap_wrap.bitmap) {
        fprintf(stderr, "[llm_ai_ocr] unable to load media: %s\n", media_path);
        return NULL;
    }
    bitmap = bitmap_wrap.bitmap;

    mtmd_input_text text;
    text.text = prompt;
    text.text_len = prompt_len;
    text.add_special = true;
    text.parse_special = true;

    /* 2. Tokenizza testo + media in chunks. */
    chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = {bitmap};
    int32_t res = mtmd_tokenize(ctx->mtmd_ctx, chunks, &text, bitmaps, 1);
    if (res != 0) {
        fprintf(stderr, "[llm_ai_ocr] tokenize failed: %d\n", res);
        goto cleanup;
    }

    /* 3. Eval dei chunk (prefill) -- stesso pattern di mtmd-cli.cpp eval_message. */
    {
        size_t n_chunks = mtmd_input_chunks_size(chunks);
        llama_pos n_past = 0;
        for (size_t i = 0; i < n_chunks; i++) {
            const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
            llama_pos new_n_past = n_past;
            res = mtmd_helper_eval_chunk_single(ctx->mtmd_ctx, ctx->lctx, chunk,
                                                 n_past, 0, ctx->n_batch,
                                                 i == n_chunks - 1, &new_n_past);
            if (res != 0) {
                fprintf(stderr, "[llm_ai_ocr] eval chunk %zu failed: %d\n", i, res);
                goto cleanup;
            }
            n_past = new_n_past;
        }

        /* 4. Genera fino a max_tokens o EOG. Buffer dimensionato al caso peggiore
         * (ogni token al massimo 7 byte in piece_buf, come nell'originale) --
         * evita una crescita con realloc per un limite comunque bounded dal
         * chiamante. */
        int limit = max_tokens > 0 ? max_tokens : 256;
        size_t out_cap = (size_t)limit * 8 + 1;
        char* output = (char*)malloc(out_cap);
        size_t output_len = 0;
        for (int i = 0; i < limit; i++) {
            llama_token token_id = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);
            if (llama_vocab_is_eog(ctx->vocab, token_id)) break;

            char buf[8];
            int n = llama_token_to_piece(ctx->vocab, token_id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                memcpy(output + output_len, buf, (size_t)n);
                output_len += (size_t)n;
            }

            /* Decode del token scelto (stesso pattern di llm_ai_session_generate). */
            llama_batch batch = llama_batch_get_one(&token_id, 1);
            if (llama_decode(ctx->lctx, batch)) break;
            n_past++;
        }

        if (output_len == 0) { free(output); goto cleanup; }
        output[output_len] = '\0';
        result = output;
    }

cleanup:
    if (chunks) mtmd_input_chunks_free(chunks);
    if (bitmap) mtmd_bitmap_free(bitmap);
    return result;
}

char* llm_ai_ocr_read_image(llm_ai_ocr_handle handle, const char* image_path,
                            const char* image_prompt, int max_tokens) {
    if (handle == NULL || image_path == NULL || image_prompt == NULL) return NULL;
    struct llm_ai_ocr_ctx* ctx = handle;

    /* Framing del modello: ChatML per Qwen (qwen2vl/omni), template
     * "User: <marker>\n<prompt>\nAssistant:" per PaddleOCR. Verificato
     * end-to-end che senza il framing corretto il modello genera vuoto
     * (PaddleOCR) o entra in loop di ripetizione (Qwen). */
    const char* marker = mtmd_default_marker();
    size_t marker_len = strlen(marker);
    size_t prompt_arg_len = strlen(image_prompt);
    bool has_marker = strstr(image_prompt, marker) != NULL;

    size_t cap = marker_len + prompt_arg_len + 256;
    char* prompt = (char*)malloc(cap);
    int len;
    const char* sys = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n";
    if (ctx->chatml) {
        if (has_marker) {
            len = snprintf(prompt, cap, "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", sys, image_prompt);
        } else {
            len = snprintf(prompt, cap, "%s<|im_start|>user\n%s%s<|im_end|>\n<|im_start|>assistant\n", sys, marker, image_prompt);
        }
    } else if (has_marker) {
        len = snprintf(prompt, cap, "User: %s\nAssistant:\n", image_prompt);
    } else {
        len = snprintf(prompt, cap, "User: %s\n%s\nAssistant:\n", marker, image_prompt);
    }

    char* result = run_media_pipeline(ctx, image_path, prompt, (size_t)len, max_tokens);
    free(prompt);
    return result;
}

char* llm_ai_ocr_read_audio(llm_ai_ocr_handle handle, const char* audio_path,
                            const char* audio_prompt, int max_tokens) {
    if (handle == NULL || audio_path == NULL || audio_prompt == NULL) return NULL;
    struct llm_ai_ocr_ctx* ctx = handle;

    /* Framing ChatML di Qwen2.5-Omni. Il system prompt ("You are a helpful
     * assistant") e l'assenza di newline tra marker audio e prompt sono
     * richiesti dal template del modello: senza di essi il modello emette
     * subito EOG e la trascrizione risulta vuota. */
    const char* marker = mtmd_default_marker();
    size_t marker_len = strlen(marker);
    size_t prompt_arg_len = strlen(audio_prompt);
    bool has_marker = strstr(audio_prompt, marker) != NULL;

    size_t cap = marker_len + prompt_arg_len + 256;
    char* prompt = (char*)malloc(cap);
    int len;
    const char* sys = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n";
    if (has_marker) {
        len = snprintf(prompt, cap, "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", sys, audio_prompt);
    } else {
        len = snprintf(prompt, cap, "%s<|im_start|>user\n%s%s<|im_end|>\n<|im_start|>assistant\n", sys, marker, audio_prompt);
    }

    char* result = run_media_pipeline(ctx, audio_path, prompt, (size_t)len, max_tokens);
    free(prompt);
    return result;
}

void llm_ai_ocr_free_string(char* s) {
    if (s != NULL) free(s);
}

void llm_ai_ocr_unload(llm_ai_ocr_handle handle) {
    if (handle == NULL) return;
    struct llm_ai_ocr_ctx* ctx = handle;
    if (ctx->sampler != NULL) llama_sampler_free(ctx->sampler);
    if (ctx->mtmd_ctx != NULL) mtmd_free(ctx->mtmd_ctx);
    if (ctx->lctx != NULL) llama_free(ctx->lctx);
    if (ctx->model != NULL) llama_model_free(ctx->model);
    free(ctx);
}
