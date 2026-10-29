/* ============================================================================
 * llmengine_llama.c
 * Implementation -- see llmengine_llama.h for the public C surface and
 * Native-LLM-Provider-Plan.md for the integration plan this backs (FASE 2:
 * model/session split for persistent, incrementally-decoded conversations).
 *
 * Riscrittura in C puro (2026-08-24, ULTIMO modulo del piano di conversione,
 * per istruzione esplicita dell'utente -- vedi
 * Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md per il piano completo e per
 * llmtoken/llmengine_mlx, gia' convertiti prima di questo).
 *
 * llama.cpp stesso resta C++ (codice upstream vendorizzato, esposto qui via
 * la sua API C pura llama.h) -- non e' in discussione, questo file riscrive
 * SOLO il wrapper LLMENV attorno ad essa.
 *
 * Sostituzioni: std::map<int, std::unique_ptr<T>> (registry modelli/sessioni)
 * -> array C indicizzato direttamente dall'handle (gli handle sono interi
 * sequenziali 0,1,2,... generati da g_next_model_handle/g_next_session_handle,
 * quindi un array a crescita geometrica con lookup O(1) per indice e' piu'
 * semplice ed efficiente di un albero bilanciato generico); std::string ->
 * buffer C posseduti (malloc/strdup); std::vector<std::pair<string,string>>
 * (messages) -> array C di {char* role; char* content;}; std::vector<llama_token>
 * (tokens) -> array C di int32_t; std::mutex -> pthread_mutex_t con
 * lock/unlock espliciti al posto di std::lock_guard (RAII) -- ogni percorso di
 * uscita anticipata di una funzione deve sbloccare esplicitamente prima del
 * return, replicando esattamente lo scoping a blocchi dell'originale.
 * L'incremental-decode-del-delta (diff del transcript formattato dal chat
 * template rispetto a quanto gia' decodificato, tokenize+decode solo del
 * delta) ricalca examples/simple-chat/simple-chat.cpp di llama.cpp stesso
 * (verificato 2026-08-16) -- non inventato da zero, invariato in questa
 * riscrittura.
 * ============================================================================ */
#include "llmengine_llama.h"

#include <llama.h>

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/* llama.h dichiara questi tipi opachi come "struct nome;" nudi, senza
 * typedef -- funziona in C++ (dove il nome del tag e' gia' un nome di tipo)
 * ma non in C puro, dove serve scrivere "struct nome" per esteso ad ogni
 * uso. Piu' semplice fornire qui i typedef mancanti una volta sola che
 * riscrivere "struct" davanti a ogni occorrenza nel resto del file. */
typedef struct llama_model llama_model;
typedef struct llama_vocab llama_vocab;
typedef struct llama_context llama_context;
typedef struct llama_sampler llama_sampler;
typedef struct llama_model_params llama_model_params;
typedef struct llama_context_params llama_context_params;

/* --------------------------------------------------------------------------
 * Strutture di stato (erano struct C++ con distruttore/RAII) e registry
 * indicizzato direttamente dall'handle (gli handle crescono sequenzialmente
 * da 0, mai riusati -- un array che cresce con realloc e non si restringe
 * mai e' la struttura piu' semplice e diretta, senza bisogno di un albero o
 * di una hash table generica).
 * -------------------------------------------------------------------------- */

typedef struct {
    llama_model* model;
    const llama_vocab* vocab;
    char* model_name; /* owned, strdup'd */
    /* Context di EMBEDDING PERSISTENTE per modello (RAG/bert): riusato tra le chiamate invece
     * di fare llama_init_from_model+llama_free a ogni testo. Su Metal l'init/free ripetuto del
     * device faceva scattare 'GGML_ASSERT([rsets->data count] == 0)' alla distruzione (bug di
     * cleanup ggml-metal, PR #17869). Lazy-creato al primo uso (emb_ctx_size = 0 = non ancora). */
    llama_context* emb_ctx;
    int emb_ctx_need;        /* n_ctx con cui emb_ctx è stato creato (0 = non creato) */
    pthread_mutex_t emb_lock;
} LlamaModel;

typedef struct {
    char* role;    /* owned, strdup'd */
    char* content; /* owned, strdup'd */
} ChatMessage;

typedef struct {
    int model_handle;
    int ctx_size; /* n_ctx this session's llama_context was created with -- needed to recreate an identically-sized context on swap-in */
    int kv_type_k; /* GGML_TYPE_F16 -- llama.cpp's own unquantized default; persisted so swap-in recreates an identically-configured context */
    int kv_type_v;
    llama_context* ctx;
    /* Owning storage for chat turns -- llama_chat_message holds non-owning
     * const char* pointers, so the view array handed to
     * llama_chat_apply_template is rebuilt fresh from this array on every
     * call, never stored long-term. */
    ChatMessage* messages;
    size_t messages_len, messages_cap;
    char* prev_formatted; /* owned, malloc'd, NUL-terminated; NULL == empty */
    size_t prev_formatted_len;
    bool decoded_anything;
    /* Full token history decoded into ctx so far -- llama_state_save_file
     * requires this explicitly (it does not track it internally); appended
     * to in decode_delta_locked, never truncated. */
    llama_token* tokens;
    size_t tokens_len, tokens_cap;
    /* Guards this ONE session's actual work (decode/generate/swap I/O) so two
     * threads never touch the same llama_context concurrently -- llama_context
     * is not safe for concurrent use. Deliberately NOT g_registry_mutex: that
     * one only protects the handle tables (model/session arrays), held briefly;
     * holding it across a whole decode/generate call would serialize every
     * session against every other session process-wide, defeating the point
     * of LLMENV's own real (GOMAXPROCS) goroutine parallelism -- see
     * AgentServer.cs's locking discipline, which this mirrors at the native
     * layer: one lock per session, not one lock for the whole registry. */
    pthread_mutex_t session_mutex;
    /* Per-session sampling for the NEXT generate. sampling_set==false means
     * "never configured": use the historical default chain (min_p 0.05 +
     * temperature 0.8 + entropy seed). Persisted in the swap meta-file so a
     * swapped-out/in session restores the same sampling behavior. Guarded by
     * session_mutex (read in generate, written by llm_ai_session_set_sampling). */
    bool sampling_set;
    float sampling_temperature;
    float sampling_top_p;
    int sampling_top_k;
    int64_t sampling_seed;
} LlamaChatSession;

static pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;
static LlamaModel** g_models = NULL;
static size_t g_models_cap = 0;
static LlamaChatSession** g_sessions = NULL;
static size_t g_sessions_cap = 0;
static int g_next_model_handle = 0;
static int g_next_session_handle = 0;
static bool g_backend_initialized = false;

static void ensure_backend_initialized(void) {
    /* llama_backend_init() sets up ggml's compute backends (Metal/CUDA/
     * ROCm/Vulkan/CPU, whichever this build was compiled with) -- must run
     * exactly once before the first model load. */
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }
}

/* Cleanup a exit del processo: libera TUTTI i modelli e le sessioni RESIDUI (chi non ha chiamato
 * llm_ai_unload_model/llm_ai_destroy_session prima dell'exit, es. un host .NET che termina senza
 * Dispose) PRIMA che ggml distrugga i device Metal. Senza questo, modelli con pesi offloadati
 * (gpuLayers) e llama_context ancora vivi mantengono residency set sul device Metal, e il
 * teardown del device a dlclose fa scattare 'GGML_ASSERT([rsets->data count] == 0)' in
 * ggml-metal-device.m:657. Ordine corretto: prima i ctx di sessione, poi emb_ctx, poi i modelli,
 * infine llama_backend_free(). */
static void llm_embed_cleanup_all(void) {
    pthread_mutex_lock(&g_registry_mutex);

    for (size_t i = 0; i < g_sessions_cap; i++) {
        LlamaChatSession* s = g_sessions[i];
        if (s == NULL) continue;
        llama_free(s->ctx);
        pthread_mutex_destroy(&s->session_mutex);
        for (size_t j = 0; j < s->messages_len; j++) { free(s->messages[j].role); free(s->messages[j].content); }
        free(s->messages);
        free(s->prev_formatted);
        free(s->tokens);
        free(s);
        g_sessions[i] = NULL;
    }

    for (size_t i = 0; i < g_models_cap; i++) {
        LlamaModel* m = g_models[i];
        if (m == NULL) continue;
        pthread_mutex_lock(&m->emb_lock);
        if (m->emb_ctx != NULL) { llama_free(m->emb_ctx); m->emb_ctx = NULL; m->emb_ctx_need = 0; }
        pthread_mutex_unlock(&m->emb_lock);
        pthread_mutex_destroy(&m->emb_lock);
        llama_model_free(m->model);
        free(m->model_name);
        free(m);
        g_models[i] = NULL;
    }

    pthread_mutex_unlock(&g_registry_mutex);

    /* Teardown canonico del backend: llama_backend_free() chiude i device Metal e rimuove i
     * residency set rimasti. Deve girare PRIMA che libggml-metal venga scaricata: il destructor
     * della nostra dylib è chiamato prima delle sue dipendenze. */
    if (g_backend_initialized) { llama_backend_free(); g_backend_initialized = false; }
}

static void llm_embed_atexit(void) { llm_embed_cleanup_all(); }

/* Best-effort: registra il cleanup anche come destructor di dylib (dlclose) e come atexit
 * handler. NB: su macOS a exit NATURALE del processo queste vie NON bastano — libggml-metal
 * viene caricata DOPO la nostra dylib (llama_backend_init -> ggml_backend_load_all) e quindi
 * scaricata PRIMA, col suo distruttore che trova i residency set ancora vivi. Il teardown
 * affidabile resta la chiamata esplicita di llm_ai_shutdown() da parte dell'host prima
 * dell'exit (vedi il .NET: ProviderServer.Shutdown su ProcessExit). */
static void llm_embed_register_teardown(void) __attribute__((constructor));
static void llm_embed_register_teardown(void) { atexit(llm_embed_atexit); }
__attribute__((destructor)) static void llm_embed_dtor(void) { llm_embed_cleanup_all(); }

/* Chiamare SOLO con g_registry_mutex già acquisito. */
static LlamaModel* find_model_locked(int handle) {
    if (handle < 0 || (size_t)handle >= g_models_cap) return NULL;
    return g_models[handle];
}

static LlamaChatSession* find_session_locked(int handle) {
    if (handle < 0 || (size_t)handle >= g_sessions_cap) return NULL;
    return g_sessions[handle];
}

static void models_array_ensure(int handle) {
    if ((size_t)handle < g_models_cap) return;
    size_t new_cap = g_models_cap ? g_models_cap * 2 : 8;
    while (new_cap <= (size_t)handle) new_cap *= 2;
    g_models = (LlamaModel**)realloc(g_models, new_cap * sizeof(LlamaModel*));
    for (size_t i = g_models_cap; i < new_cap; i++) g_models[i] = NULL;
    g_models_cap = new_cap;
}

static void sessions_array_ensure(int handle) {
    if ((size_t)handle < g_sessions_cap) return;
    size_t new_cap = g_sessions_cap ? g_sessions_cap * 2 : 8;
    while (new_cap <= (size_t)handle) new_cap *= 2;
    g_sessions = (LlamaChatSession**)realloc(g_sessions, new_cap * sizeof(LlamaChatSession*));
    for (size_t i = g_sessions_cap; i < new_cap; i++) g_sessions[i] = NULL;
    g_sessions_cap = new_cap;
}

/* Appende in fondo a messages, duplicando role/content (owned dalla sessione). */
static void session_push_message(LlamaChatSession* s, const char* role, const char* content) {
    if (s->messages_len >= s->messages_cap) {
        s->messages_cap = s->messages_cap ? s->messages_cap * 2 : 8;
        s->messages = (ChatMessage*)realloc(s->messages, s->messages_cap * sizeof(ChatMessage));
    }
    s->messages[s->messages_len].role = strdup(role);
    s->messages[s->messages_len].content = strdup(content);
    s->messages_len++;
}

static void session_set_prev_formatted(LlamaChatSession* s, const char* text, size_t len) {
    free(s->prev_formatted);
    s->prev_formatted = (char*)malloc(len + 1);
    memcpy(s->prev_formatted, text, len);
    s->prev_formatted[len] = '\0';
    s->prev_formatted_len = len;
}

static void session_append_tokens(LlamaChatSession* s, const llama_token* toks, size_t n) {
    if (s->tokens_len + n > s->tokens_cap) {
        size_t new_cap = s->tokens_cap ? s->tokens_cap * 2 : 256;
        while (new_cap < s->tokens_len + n) new_cap *= 2;
        s->tokens = (llama_token*)realloc(s->tokens, new_cap * sizeof(llama_token));
        s->tokens_cap = new_cap;
    }
    memcpy(s->tokens + s->tokens_len, toks, n * sizeof(llama_token));
    s->tokens_len += n;
}

/* Costruisce il transcript completo formattato dal chat template a partire dai
 * messaggi posseduti dalla sessione. add_ass controlla se un suffisso
 * generation-prompt (es. l'apertura del turno assistant) viene appeso -- true
 * subito dopo un turno user, false dopo un turno system/assistant. Ritorna un
 * buffer malloc'd (mai NULL su successo, stringa vuota se formatted e' vuoto)
 * e scrive la lunghezza in *out_len; il chiamante libera con free(). Ritorna
 * NULL su errore reale (non su transcript vuoto). */
static char* apply_template(const LlamaModel* model, const LlamaChatSession* session, bool add_ass, size_t* out_len) {
    const char* tmpl = llama_model_chat_template(model->model, NULL);
    if (tmpl == NULL) {
        size_t cap = 256, len = 0;
        char* fallback = (char*)malloc(cap);
        fallback[0] = '\0';
        for (size_t i = 0; i < session->messages_len; i++) {
            const char* role = session->messages[i].role;
            const char* content = session->messages[i].content;
            size_t needed = len + strlen(role) + 2 + strlen(content) + 2;
            if (needed + 1 > cap) {
                while (needed + 1 > cap) cap *= 2;
                fallback = (char*)realloc(fallback, cap);
            }
            len += (size_t)snprintf(fallback + len, cap - len, "%s: %s\n", role, content);
        }
        *out_len = len;
        return fallback;
    }

    llama_chat_message* view = (llama_chat_message*)malloc(session->messages_len * sizeof(llama_chat_message));
    for (size_t i = 0; i < session->messages_len; i++) {
        view[i].role = session->messages[i].role;
        view[i].content = session->messages[i].content;
    }

    size_t buf_cap = 8192;
    char* buf = (char*)malloc(buf_cap);
    int written = llama_chat_apply_template(tmpl, view, session->messages_len, add_ass, buf, (int)buf_cap);
    if (written < 0) {
        free(view);
        free(buf);
        return NULL;
    }
    if ((size_t)written > buf_cap) {
        buf_cap = (size_t)written;
        buf = (char*)realloc(buf, buf_cap);
        written = llama_chat_apply_template(tmpl, view, session->messages_len, add_ass, buf, (int)buf_cap);
    }
    free(view);
    if (written < 0) {
        free(buf);
        return NULL;
    }
    buf[written] = '\0';
    *out_len = (size_t)written;
    return buf;
}

/* Tokenizza e decodifica solo il testo in `formatted` oltre quanto gia' in
 * `session->prev_formatted` (il delta), avanzando il KV-cache della sessione
 * esattamente di quella quantita'. Ritorna false su qualunque fallimento di
 * tokenize/decode/mismatch di forma del template -- il chiamante deve
 * trattare la sessione come inutilizzabile per QUESTA chiamata ma non
 * necessariamente corrotta (lo stato resta cosi' com'e', nessuna mutazione
 * parziale di prev_formatted in caso di fallimento). */
static bool decode_delta_locked(LlamaModel* model, LlamaChatSession* session, const char* formatted, size_t formatted_len) {
    size_t prev_len = session->prev_formatted ? session->prev_formatted_len : 0;
    const char* prev = session->prev_formatted ? session->prev_formatted : "";
    if (formatted_len < prev_len || memcmp(formatted, prev, prev_len) != 0) {
        return false;
    }

    const char* delta = formatted + prev_len;
    size_t delta_len = formatted_len - prev_len;
    if (delta_len == 0) {
        session_set_prev_formatted(session, formatted, formatted_len);
        return true;
    }

    bool is_first = !session->decoded_anything;
    int n_tok = -llama_tokenize(model->vocab, delta, (int32_t)delta_len, NULL, 0, is_first, true);
    if (n_tok <= 0) return false;

    llama_token* toks = (llama_token*)malloc((size_t)n_tok * sizeof(llama_token));
    if (llama_tokenize(model->vocab, delta, (int32_t)delta_len, toks, (int32_t)n_tok, is_first, true) < 0) {
        free(toks);
        return false;
    }

    llama_batch batch = llama_batch_get_one(toks, (int32_t)n_tok);
    if (llama_decode(session->ctx, batch) != 0) {
        free(toks);
        return false;
    }

    session_append_tokens(session, toks, (size_t)n_tok);
    free(toks);
    session->decoded_anything = true;
    session_set_prev_formatted(session, formatted, formatted_len);
    return true;
}

/* --------------------------------------------------------------------------
 * Formato meta per lo swap su disco -- LLMKV02: version tag lets a future
 * format change refuse to load an old meta file rather than misinterpret its
 * bytes. Bumped from LLMKV01 to LLMKV02 when kv_type_k/kv_type_v were added.
 * -------------------------------------------------------------------------- */
static const char kMetaMagic[8] = {'G', 'M', 'B', 'K', 'V', '0', '3', '\0'};
static const char kMetaMagicV02[8] = {'G', 'M', 'B', 'K', 'V', '0', '2', '\0'};

static bool write_u32(FILE* f, uint32_t v) { return fwrite(&v, sizeof(v), 1, f) == 1; }
static bool write_i32(FILE* f, int32_t v) { return fwrite(&v, sizeof(v), 1, f) == 1; }
static bool write_i64(FILE* f, int64_t v) { return fwrite(&v, sizeof(v), 1, f) == 1; }
static bool write_f32(FILE* f, float v) { return fwrite(&v, sizeof(v), 1, f) == 1; }
static bool write_bytes(FILE* f, const char* s, size_t len) {
    if (!write_u32(f, (uint32_t)len)) return false;
    if (len > 0 && fwrite(s, 1, len, f) != len) return false;
    return true;
}

static bool write_meta_file(const char* path, const LlamaChatSession* session) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) return false;

    bool ok = true;
    ok = ok && fwrite(kMetaMagic, 1, sizeof(kMetaMagic), f) == sizeof(kMetaMagic);
    ok = ok && write_i32(f, session->model_handle);
    ok = ok && write_i32(f, session->ctx_size);
    ok = ok && write_i32(f, session->kv_type_k);
    ok = ok && write_i32(f, session->kv_type_v);
    uint8_t decoded_anything = session->decoded_anything ? 1 : 0;
    ok = ok && fwrite(&decoded_anything, 1, 1, f) == 1;
    ok = ok && write_u32(f, (uint32_t)session->messages_len);
    for (size_t i = 0; ok && i < session->messages_len; i++) {
        ok = ok && write_bytes(f, session->messages[i].role, strlen(session->messages[i].role));
        ok = ok && write_bytes(f, session->messages[i].content, strlen(session->messages[i].content));
    }
    ok = ok && write_bytes(f, session->prev_formatted ? session->prev_formatted : "",
                           session->prev_formatted ? session->prev_formatted_len : 0);
    /* Sampling (magic 03): persisted so a swap-out/in round-trip restores the
     * exact same generation behavior. */
    uint8_t sampling_set = session->sampling_set ? 1 : 0;
    ok = ok && fwrite(&sampling_set, 1, 1, f) == 1;
    ok = ok && write_f32(f, session->sampling_temperature);
    ok = ok && write_f32(f, session->sampling_top_p);
    ok = ok && write_i32(f, session->sampling_top_k);
    ok = ok && write_i64(f, session->sampling_seed);

    fclose(f);
    return ok;
}

typedef struct {
    int32_t model_handle;
    int32_t ctx_size;
    int32_t kv_type_k;
    int32_t kv_type_v;
    bool decoded_anything;
    ChatMessage* messages;
    size_t messages_len;
    char* prev_formatted;
    size_t prev_formatted_len;
    bool sampling_set;
    float sampling_temperature;
    float sampling_top_p;
    int sampling_top_k;
    int64_t sampling_seed;
} LoadedMeta;

static bool read_u32(FILE* f, uint32_t* v) { return fread(v, sizeof(*v), 1, f) == 1; }
static bool read_i32(FILE* f, int32_t* v) { return fread(v, sizeof(*v), 1, f) == 1; }
static bool read_i64(FILE* f, int64_t* v) { return fread(v, sizeof(*v), 1, f) == 1; }
static bool read_f32(FILE* f, float* v) { return fread(v, sizeof(*v), 1, f) == 1; }
/* Legge una stringa con prefisso di lunghezza in un buffer malloc'd NUL-terminated. */
static bool read_bytes(FILE* f, char** out, size_t* out_len) {
    uint32_t len = 0;
    if (!read_u32(f, &len)) return false;
    char* s = (char*)malloc((size_t)len + 1);
    if (len > 0 && fread(s, 1, len, f) != len) { free(s); return false; }
    s[len] = '\0';
    *out = s;
    if (out_len) *out_len = len;
    return true;
}

static void loaded_meta_free(LoadedMeta* m) {
    for (size_t i = 0; i < m->messages_len; i++) {
        free(m->messages[i].role);
        free(m->messages[i].content);
    }
    free(m->messages);
    free(m->prev_formatted);
}

static bool read_meta_file(const char* path, LoadedMeta* out) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return false;

    memset(out, 0, sizeof(*out));
    out->kv_type_k = 1;
    out->kv_type_v = 1;

    bool ok = true;
    char magic[sizeof(kMetaMagic)] = {0};
    ok = ok && fread(magic, 1, sizeof(magic), f) == sizeof(magic);
    if (!ok || (memcmp(magic, kMetaMagic, sizeof(kMetaMagic)) != 0 &&
                memcmp(magic, kMetaMagicV02, sizeof(kMetaMagicV02)) != 0)) { fclose(f); return false; }
    bool is_v03 = memcmp(magic, kMetaMagic, sizeof(kMetaMagic)) == 0;

    int32_t model_handle = -1, ctx_size = 0, kv_type_k = 1, kv_type_v = 1;
    ok = ok && read_i32(f, &model_handle);
    ok = ok && read_i32(f, &ctx_size);
    ok = ok && read_i32(f, &kv_type_k);
    ok = ok && read_i32(f, &kv_type_v);
    uint8_t decoded_anything_byte = 0;
    ok = ok && fread(&decoded_anything_byte, 1, 1, f) == 1;
    uint32_t message_count = 0;
    ok = ok && read_u32(f, &message_count);

    ChatMessage* messages = ok && message_count > 0 ? (ChatMessage*)malloc(message_count * sizeof(ChatMessage)) : NULL;
    size_t messages_len = 0;
    for (uint32_t i = 0; ok && i < message_count; i++) {
        char* role = NULL; char* content = NULL;
        ok = ok && read_bytes(f, &role, NULL);
        ok = ok && read_bytes(f, &content, NULL);
        if (ok) { messages[messages_len].role = role; messages[messages_len].content = content; messages_len++; }
        else { free(role); free(content); }
    }
    char* prev_formatted = NULL;
    size_t prev_formatted_len = 0;
    ok = ok && read_bytes(f, &prev_formatted, &prev_formatted_len);

    /* Sampling (magic 03): opzionale. Su un file LLMKV02 (vecchio) i campi non
     * sono scritti -- lascia sampling_set=false (comportamento default). */
    bool sampling_set = false;
    float sampling_temperature = 0.0f, sampling_top_p = 0.0f;
    int sampling_top_k = 0;
    int64_t sampling_seed = 0;
    if (is_v03) {
        uint8_t sampling_set_byte = 0;
        ok = ok && fread(&sampling_set_byte, 1, 1, f) == 1;
        ok = ok && read_f32(f, &sampling_temperature);
        ok = ok && read_f32(f, &sampling_top_p);
        ok = ok && read_i32(f, &sampling_top_k);
        ok = ok && read_i64(f, &sampling_seed);
        if (ok) sampling_set = sampling_set_byte != 0;
    }

    fclose(f);
    if (!ok) {
        for (size_t i = 0; i < messages_len; i++) { free(messages[i].role); free(messages[i].content); }
        free(messages);
        free(prev_formatted);
        return false;
    }

    out->model_handle = model_handle;
    out->ctx_size = ctx_size;
    out->kv_type_k = kv_type_k;
    out->kv_type_v = kv_type_v;
    out->decoded_anything = decoded_anything_byte != 0;
    out->messages = messages;
    out->messages_len = messages_len;
    out->prev_formatted = prev_formatted;
    out->prev_formatted_len = prev_formatted_len;
    out->sampling_set = sampling_set;
    out->sampling_temperature = sampling_temperature;
    out->sampling_top_p = sampling_top_p;
    out->sampling_top_k = sampling_top_k;
    out->sampling_seed = sampling_seed;
    return true;
}

/* --------------------------------------------------------------------------
 * Superficie pubblica (extern "C" nell'header, invariata -- questo file e'
 * ora C puro, ma il confine ABI verso C# non cambia).
 * -------------------------------------------------------------------------- */

int llm_ai_load_model(const char* model_path, int n_gpu_layers) {
    if (model_path == NULL) return -1;

    ensure_backend_initialized();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) return -1;

    LlamaModel* entry = (LlamaModel*)calloc(1, sizeof(LlamaModel));
    entry->model = model;
    entry->vocab = llama_model_get_vocab(model);

    char name_buf[256] = {0};
    int name_len = llama_model_meta_val_str(model, "general.name", name_buf, sizeof(name_buf));
    entry->model_name = strdup(name_len > 0 ? name_buf : model_path);
    pthread_mutex_init(&entry->emb_lock, NULL);

    pthread_mutex_lock(&g_registry_mutex);
    int handle = g_next_model_handle++;
    models_array_ensure(handle);
    g_models[handle] = entry;
    pthread_mutex_unlock(&g_registry_mutex);
    return handle;
}

void llm_ai_unload_model(int model_handle) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* m = find_model_locked(model_handle);
    if (m == NULL) { pthread_mutex_unlock(&g_registry_mutex); return; }

    pthread_mutex_lock(&m->emb_lock);
    if (m->emb_ctx != NULL) { llama_free(m->emb_ctx); m->emb_ctx = NULL; m->emb_ctx_need = 0; }
    pthread_mutex_unlock(&m->emb_lock);
    pthread_mutex_destroy(&m->emb_lock);

    llama_model_free(m->model);
    free(m->model_name);
    free(m);
    g_models[model_handle] = NULL;
    pthread_mutex_unlock(&g_registry_mutex);
}

/* Cleanup TOTALE esplicito: libera TUTTE le sessioni e TUTTI i modelli residui, poi fa il
 * teardown canonico del backend. Equivalente al dlclose ordinato (che passa sempre), da
 * invocare dall'host (es. handler .NET di ProcessExit) PRIMA che il processo termini: su
 * macOS, senza questo, libggml-metal viene scaricata per prima (ordine inverso di caricamento:
 * llama_backend_init -> ggml_backend_load_all la carica DOPO la nostra dylib) e il suo
 * distruttore trova i residency set ancora vivi -> GGML_ASSERT([rsets->data count] == 0). */
void llm_ai_shutdown(void) {
    llm_embed_cleanup_all();
}

const char* llm_ai_model_name(int model_handle) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* model = find_model_locked(model_handle);
    const char* name = model == NULL ? NULL : model->model_name;
    pthread_mutex_unlock(&g_registry_mutex);
    return name;
}

int llm_ai_create_session(int model_handle, int ctx_size, int kv_type_k, int kv_type_v) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* model = find_model_locked(model_handle);
    if (model == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t)(ctx_size > 0 ? ctx_size : 4096);
    ctx_params.n_batch = ctx_params.n_ctx;
    /* Quantized KV-cache types (Q8_0/Q4_0) need flash attention -- AUTO (llama.cpp's own
     * default) enables it automatically when the backend/quant combination requires it, so
     * this only matters when kv_type_k/v differ from F16; harmless no-op otherwise. */
    ctx_params.type_k = (enum ggml_type)kv_type_k;
    ctx_params.type_v = (enum ggml_type)kv_type_v;

    llama_context* ctx = llama_init_from_model(model->model, ctx_params);
    if (ctx == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }

    LlamaChatSession* session = (LlamaChatSession*)calloc(1, sizeof(LlamaChatSession));
    session->model_handle = model_handle;
    session->ctx_size = (int)ctx_params.n_ctx;
    session->kv_type_k = kv_type_k;
    session->kv_type_v = kv_type_v;
    session->ctx = ctx;
    pthread_mutex_init(&session->session_mutex, NULL);

    int handle = g_next_session_handle++;
    sessions_array_ensure(handle);
    g_sessions[handle] = session;
    pthread_mutex_unlock(&g_registry_mutex);
    return handle;
}

void llm_ai_destroy_session(int session_handle) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaChatSession* s = find_session_locked(session_handle);
    if (s == NULL) { pthread_mutex_unlock(&g_registry_mutex); return; }

    llama_free(s->ctx);
    pthread_mutex_destroy(&s->session_mutex);
    for (size_t i = 0; i < s->messages_len; i++) { free(s->messages[i].role); free(s->messages[i].content); }
    free(s->messages);
    free(s->prev_formatted);
    free(s->tokens);
    free(s);
    g_sessions[session_handle] = NULL;
    pthread_mutex_unlock(&g_registry_mutex);
}

int llm_ai_session_append(int session_handle, const char* role, const char* content) {
    LlamaChatSession* session = NULL;
    LlamaModel* model = NULL;
    /* g_registry_mutex only protects the handle tables -- held just long enough to look
     * the pointers up, never across the actual decode below. See LlamaChatSession's
     * session_mutex doc comment for why: holding the registry lock across a decode would
     * serialize every session's decode against every other session's, process-wide. */
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    model = find_model_locked(session->model_handle);
    if (model == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);

    if (role == NULL || content == NULL) return -1;

    /* Per-session lock: safe for two DIFFERENT sessions to decode concurrently on separate
     * threads (each has its own llama_context/KV cache); never safe for the SAME session
     * handle to be touched by two threads at once. */
    pthread_mutex_lock(&session->session_mutex);

    session_push_message(session, role, content);

    bool add_ass = strcmp(role, "user") == 0;
    size_t formatted_len = 0;
    char* formatted = apply_template(model, session, add_ass, &formatted_len);
    if (formatted == NULL) { pthread_mutex_unlock(&session->session_mutex); return -1; }
    if (formatted_len == 0 && session->messages_len > 0) { free(formatted); pthread_mutex_unlock(&session->session_mutex); return -1; }

    bool ok = decode_delta_locked(model, session, formatted, formatted_len);
    free(formatted);
    pthread_mutex_unlock(&session->session_mutex);
    return ok ? 0 : -1;
}

char* llm_ai_session_generate(int session_handle, int max_tokens) {
    LlamaChatSession* session = NULL;
    LlamaModel* model = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return NULL; }
    model = find_model_locked(session->model_handle);
    if (model == NULL) { pthread_mutex_unlock(&g_registry_mutex); return NULL; }
    pthread_mutex_unlock(&g_registry_mutex);

    /* Held for the whole generation loop -- see llm_ai_session_append's comment. Two
     * DIFFERENT sessions' generate calls run fully concurrently (no lock shared between
     * them); the SAME session can never be generated on by two threads at once. */
    pthread_mutex_lock(&session->session_mutex);

    /* Build the sampler chain. Default (sampling_set==false) keeps the historical
     * behavior: min_p 0.05 + temperature 0.8 + entropy seed. When configured,
     * temperature<=0 means greedy (deterministic argmax); otherwise top_p/top_k
     * (when >0) refine the temperature sampler. Chain order mirrors llama.cpp:
     * top_k -> top_p -> min_p -> temp -> dist. */
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!session->sampling_set || session->sampling_temperature <= 0.0f) {
        if (session->sampling_set) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }
    } else {
        if (session->sampling_top_k > 0)
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(session->sampling_top_k));
        if (session->sampling_top_p > 0.0f && session->sampling_top_p < 1.0f)
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(session->sampling_top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(session->sampling_temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(session->sampling_seed == 0
                                                                    ? LLAMA_DEFAULT_SEED
                                                                    : (unsigned int)session->sampling_seed));
    }

    size_t result_cap = 4096, result_len = 0;
    char* result = (char*)malloc(result_cap);
    llama_token* generated_tokens = NULL;
    size_t generated_len = 0, generated_cap = 0;

    /* Clamping difensivo: non generare piu' token del contesto residuo. Senza questo,
     * una richiesta max_tokens > contesto disponibile fa fallire llama_decode a meta'
     * ("no KV slot for the batch" -- il KV-cache e' pieno), e la sessione resta in uno
     * stato inutilizzabile. session->tokens include sia il prefill (append) sia i token
     * gia' generati, quindi e' la misura esatta dei token nel KV-cache. */
    int requested = max_tokens > 0 ? max_tokens : 2048;
    uint32_t n_ctx = llama_n_ctx(session->ctx);
    int used = (int)session->tokens_len;
    int remaining = (int)n_ctx - used;
    int limit = requested;
    if (remaining <= 0) {
        llama_sampler_free(sampler);
        free(result);
        pthread_mutex_unlock(&session->session_mutex);
        return NULL; /* context already full -- nothing sensible to generate */
    }
    if (requested > remaining) limit = remaining;

    for (int generated = 0; generated < limit; generated++) {
        llama_token new_token = llama_sampler_sample(sampler, session->ctx, -1);
        if (llama_vocab_is_eog(model->vocab, new_token)) break;

        char piece_buf[256];
        int piece_len = llama_token_to_piece(model->vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len > 0) {
            if (result_len + (size_t)piece_len + 1 > result_cap) {
                while (result_len + (size_t)piece_len + 1 > result_cap) result_cap *= 2;
                result = (char*)realloc(result, result_cap);
            }
            memcpy(result + result_len, piece_buf, (size_t)piece_len);
            result_len += (size_t)piece_len;
        }

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(session->ctx, batch) != 0) break;

        if (generated_len >= generated_cap) {
            generated_cap = generated_cap ? generated_cap * 2 : 64;
            generated_tokens = (llama_token*)realloc(generated_tokens, generated_cap * sizeof(llama_token));
        }
        generated_tokens[generated_len++] = new_token;
    }
    result[result_len] = '\0';

    llama_sampler_free(sampler);

    /* Record the reply in the session's own transcript bookkeeping. Its template closing
     * markup (if any) is intentionally NOT decoded here -- it wasn't generated by the model,
     * so it isn't in the KV cache yet either; the next llm_ai_session_append call's diff will
     * pick it up together with the following turn. Still under session_mutex, not
     * g_registry_mutex -- this is session-local state, not the handle table. */
    session_push_message(session, "assistant", result);
    size_t new_prev_len = 0;
    char* new_prev = apply_template(model, session, /*add_ass=*/false, &new_prev_len);
    if (new_prev != NULL) {
        session_set_prev_formatted(session, new_prev, new_prev_len);
        free(new_prev);
    }
    session_append_tokens(session, generated_tokens, generated_len);
    free(generated_tokens);

    pthread_mutex_unlock(&session->session_mutex);
    return result;
}

/* ── API a livello di ID di token (usata dall'adattatore `llamacpp_ids`) ─────────────────────────────────
 * llama.cpp lavora a testo/chat, ma il suo contesto e' a token: queste due funzioni decodificano id gia'
 * calcolati dal chiamante (tokenizer e template NON sono quelli di llama.cpp) e generano id. Cosi' un consumatore a id
 * (FastLLM) usa qualunque GGUF che llama.cpp sa eseguire, con le stesse regole di streaming/stop degli altri motori.
 * Il KV della sessione e' quello della chat: i due stili non vanno mescolati sulla stessa sessione. */

/* Decodifica `n` id nel KV della sessione (a blocchi: llama_decode accetta al piu' n_batch token). 0 = ok, -1 = errore. */
/* Dati per scegliere la lunghezza del contesto di un modello: contesto di addestramento, byte di KV-cache per token (stima
 * F16 su tutti gli strati: la cache e' preallocata da llama.cpp per intero alla creazione del contesto) e dimensione dei
 * pesi. 0 = ok, -1 = handle non valido. I valori 0 vanno letti come "ignoto". */
int llm_ai_model_ctx_info(int model_handle, int* n_ctx_train, size_t* kv_bytes_per_token, size_t* model_bytes) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* m = find_model_locked(model_handle);
    if (m == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    const int64_t nl = llama_model_n_layer(m->model), nkv = llama_model_n_head_kv(m->model);
    const int64_t nh = llama_model_n_head(m->model), ne = llama_model_n_embd(m->model);
    if (n_ctx_train) *n_ctx_train = llama_model_n_ctx_train(m->model);
    if (kv_bytes_per_token) *kv_bytes_per_token = (nh > 0 && nl > 0 && nkv > 0) ? (size_t)(2 * 2 * nl * nkv * (ne / nh)) : 0;
    if (model_bytes) *model_bytes = (size_t)llama_model_size(m->model);
    pthread_mutex_unlock(&g_registry_mutex);
    return 0;
}

/* Byte di KV-cache (F16) che llama.cpp assegna per un contesto di `n_ctx` token, calcolati dai metadati del modello:
 * lunghezza di chiavi e valori per testa (`<arch>.attention.key_length/value_length`, altrimenti embedding/teste),
 * teste KV, strati, e per i modelli con finestra locale (SWA) il costo ridotto degli strati locali: min(n_ctx, finestra + 512).
 * Frazione di strati locali nota per Gemma 2 (1/2), Gemma 3 (5/6), Gemma 3n (4/5), Cohere2 (3/4); per le altre architetture con
 * finestra si assume nessuno strato locale (stima per eccesso: mai sotto il costo reale). 0 = ignoto / handle non valido. */
size_t llm_ai_model_kv_bytes(int model_handle, long long n_ctx) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* m = find_model_locked(model_handle);
    if (m == NULL) { pthread_mutex_unlock(&g_registry_mutex); return 0; }
    const llama_model* mdl = m->model;
    const long long nl = llama_model_n_layer(mdl), nkv = llama_model_n_head_kv(mdl), nh = llama_model_n_head(mdl), ne = llama_model_n_embd(mdl);
    const long long swa = llama_model_n_swa(mdl);
    char arch[64] = "", key[160], val[32];
    (void)llama_model_meta_val_str(mdl, "general.architecture", arch, sizeof arch);
    long long klen = 0, vlen = 0;
    snprintf(key, sizeof key, "%s.attention.key_length", arch);
    if (llama_model_meta_val_str(mdl, key, val, sizeof val) > 0) klen = atoll(val);
    snprintf(key, sizeof key, "%s.attention.value_length", arch);
    if (llama_model_meta_val_str(mdl, key, val, sizeof val) > 0) vlen = atoll(val);
    pthread_mutex_unlock(&g_registry_mutex);
    if (nl <= 0 || nkv <= 0 || nh <= 0) return 0;
    if (klen <= 0) klen = ne / nh;
    if (vlen <= 0) vlen = klen;
    double frac = 0.0;
    if (swa > 0) {
        if (strcmp(arch, "gemma2") == 0) frac = 0.5;
        else if (strcmp(arch, "gemma3") == 0) frac = 5.0 / 6.0;
        else if (strcmp(arch, "gemma3n") == 0) frac = 0.8;
        else if (strcmp(arch, "cohere2") == 0) frac = 0.75;
    }
    const double per_layer_token = (double)nkv * (double)(klen + vlen) * 2.0;   /* F16 */
    const double local_ctx = swa > 0 ? (double)(swa + 512 < n_ctx ? swa + 512 : n_ctx) : (double)n_ctx;
    return (size_t)((double)nl * per_layer_token * ((1.0 - frac) * (double)n_ctx + frac * local_ctx));
}

int llm_ai_session_append_ids(int session_handle, const int32_t* ids, int n) {
    LlamaChatSession* session = NULL;
    if (ids == NULL || n < 0) return -1;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    pthread_mutex_unlock(&g_registry_mutex);
    if (session == NULL) return -1;
    pthread_mutex_lock(&session->session_mutex);
    int rc = 0;
    if ((int)session->tokens_len + n > (int)llama_n_ctx(session->ctx)) rc = -1;   /* contesto pieno */
    const int chunk = 512;
    for (int off = 0; rc == 0 && off < n; off += chunk) {
        int len = n - off < chunk ? n - off : chunk;
        llama_token* tk = (llama_token*)malloc((size_t)len * sizeof(llama_token));
        if (tk == NULL) { rc = -1; break; }
        for (int i = 0; i < len; i++) tk[i] = (llama_token)ids[off + i];
        llama_batch batch = llama_batch_get_one(tk, len);
        if (llama_decode(session->ctx, batch) != 0) rc = -1;
        else session_append_tokens(session, tk, (size_t)len);
        free(tk);
    }
    if (rc == 0) session->decoded_anything = true;
    pthread_mutex_unlock(&session->session_mutex);
    return rc;
}

/* Genera fino a `max_new` id continuando dal KV corrente; si ferma sull'eos dato o su un token di fine generazione del
 * vocabolario (che viene scritto in `out` ma NON decodificato). Ritorna il numero di id scritti. */
int llm_ai_session_generate_ids(int session_handle, int max_new, int32_t eos, int32_t* out) {
    LlamaChatSession* session = NULL;
    LlamaModel* model = NULL;
    if (out == NULL || max_new <= 0) return -1;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session != NULL) model = find_model_locked(session->model_handle);
    pthread_mutex_unlock(&g_registry_mutex);
    if (session == NULL || model == NULL) return -1;
    pthread_mutex_lock(&session->session_mutex);

    /* Catena di campionamento come in generate_text, ma il seme cambia a ogni chiamata (qui si chiama un token per volta:
     * con un seme fisso ogni chiamata ripeterebbe la stessa estrazione). */
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    unsigned seed = session->sampling_seed != 0 ? (unsigned)(session->sampling_seed + (int64_t)session->tokens_len)
                                                : (unsigned)(time(NULL) ^ ((unsigned)session->tokens_len * 2654435761u));
    if (session->sampling_set && session->sampling_temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        float temp = session->sampling_set ? session->sampling_temperature : 0.8f;
        if (session->sampling_set && session->sampling_top_k > 0)
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(session->sampling_top_k));
        if (session->sampling_set && session->sampling_top_p > 0.0f && session->sampling_top_p < 1.0f)
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(session->sampling_top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    }

    int k = 0;
    const int n_ctx = (int)llama_n_ctx(session->ctx);
    for (; k < max_new; k++) {
        if ((int)session->tokens_len >= n_ctx) break;   /* contesto esaurito */
        llama_token t = llama_sampler_sample(sampler, session->ctx, -1);
        out[k] = (int32_t)t;
        if (eos >= 0 && (int32_t)t == eos) { k++; break; }
        if (llama_vocab_is_eog(model->vocab, t)) {   /* altro token di fine (p.es. <end_of_turn> di Gemma): lo si segnala con l eos richiesto, che il chiamante riconosce */
            if (eos >= 0) out[k] = eos;
            k++; break;
        }
        llama_batch batch = llama_batch_get_one(&t, 1);
        if (llama_decode(session->ctx, batch) != 0) break;
        session_append_tokens(session, &t, 1);
    }
    llama_sampler_free(sampler);
    pthread_mutex_unlock(&session->session_mutex);
    return k;
}

void llm_ai_free_string(char* s) {
    free(s);
}

int llm_ai_session_swap_out(int session_handle, const char* path_prefix) {
    if (path_prefix == NULL) return -1;

    LlamaChatSession* session = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);
    /* g_registry_mutex released before the (potentially large) file I/O below;
     * session_mutex held instead, for the same reason as llm_ai_session_append/
     * generate -- this is session-local work, not a handle-table change. */
    pthread_mutex_lock(&session->session_mutex);

    char kv_tmp[1024], kv_final[1024], meta_tmp[1024], meta_final[1024];
    snprintf(kv_tmp, sizeof(kv_tmp), "%s.kv.tmp", path_prefix);
    snprintf(kv_final, sizeof(kv_final), "%s.kv", path_prefix);
    snprintf(meta_tmp, sizeof(meta_tmp), "%s.meta.tmp", path_prefix);
    snprintf(meta_final, sizeof(meta_final), "%s.meta", path_prefix);

    if (!llama_state_save_file(session->ctx, kv_tmp, session->tokens, session->tokens_len)) {
        remove(kv_tmp);
        pthread_mutex_unlock(&session->session_mutex);
        return -1;
    }

    if (!write_meta_file(meta_tmp, session)) {
        remove(kv_tmp);
        remove(meta_tmp);
        pthread_mutex_unlock(&session->session_mutex);
        return -1;
    }

    /* Commit: rename() is atomic per-file on the same filesystem, so a crash
     * between these two renames never leaves a half-written file at the
     * final path -- worst case is a fresh .kv next to a stale/missing
     * .meta, which swap-in detects (model_handle/magic mismatch or missing
     * file) and refuses rather than silently loading mismatched state. */
    if (rename(kv_tmp, kv_final) != 0) {
        remove(kv_tmp);
        remove(meta_tmp);
        pthread_mutex_unlock(&session->session_mutex);
        return -1;
    }
    if (rename(meta_tmp, meta_final) != 0) {
        remove(meta_tmp);
        pthread_mutex_unlock(&session->session_mutex);
        return -1;
    }

    pthread_mutex_unlock(&session->session_mutex);
    return 0;
}

int llm_ai_session_swap_in(const char* path_prefix) {
    if (path_prefix == NULL) return -1;

    char kv_path[1024], meta_path[1024];
    snprintf(kv_path, sizeof(kv_path), "%s.kv", path_prefix);
    snprintf(meta_path, sizeof(meta_path), "%s.meta", path_prefix);

    LoadedMeta meta;
    if (!read_meta_file(meta_path, &meta)) return -1;

    /* llm_ai_create_session takes g_registry_mutex itself -- must be called
     * with no lock held here to avoid deadlock (pthread_mutex_t default type
     * is non-recursive). */
    int session_handle = llm_ai_create_session(meta.model_handle, meta.ctx_size, meta.kv_type_k, meta.kv_type_v);
    if (session_handle < 0) { loaded_meta_free(&meta); return -1; }

    llama_context* ctx = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    LlamaChatSession* session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); loaded_meta_free(&meta); return -1; } /* unreachable: just created it */
    ctx = session->ctx;
    pthread_mutex_unlock(&g_registry_mutex);

    size_t tokens_cap = (size_t)(meta.ctx_size > 0 ? meta.ctx_size : 4096);
    llama_token* tokens_out = (llama_token*)malloc(tokens_cap * sizeof(llama_token));
    size_t n_token_count = 0;
    bool loaded = llama_state_load_file(ctx, kv_path, tokens_out, tokens_cap, &n_token_count);
    if (!loaded) {
        free(tokens_out);
        loaded_meta_free(&meta);
        llm_ai_destroy_session(session_handle);
        return -1;
    }

    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); free(tokens_out); loaded_meta_free(&meta); return -1; } /* unreachable */
    /* Trasferisce la proprieta' di messages/prev_formatted da meta alla sessione
     * (evita una copia extra: meta non viene piu' usato dopo, solo liberato
     * senza il suo contenuto tramite l'azzeramento qui sotto). */
    for (size_t i = 0; i < session->messages_len; i++) { free(session->messages[i].role); free(session->messages[i].content); }
    free(session->messages);
    session->messages = meta.messages;
    session->messages_len = meta.messages_len;
    session->messages_cap = meta.messages_len;
    free(session->prev_formatted);
    session->prev_formatted = meta.prev_formatted;
    session->prev_formatted_len = meta.prev_formatted_len;
    session->decoded_anything = meta.decoded_anything;
    /* Sampling (magic 03): ripristina la stessa configurazione del momento
     * dello swap-out. Su un meta LLMKV02 sampling_set resta false (default). */
    session->sampling_set = meta.sampling_set;
    session->sampling_temperature = meta.sampling_temperature;
    session->sampling_top_p = meta.sampling_top_p;
    session->sampling_top_k = meta.sampling_top_k;
    session->sampling_seed = meta.sampling_seed;
    free(session->tokens);
    session->tokens = (llama_token*)realloc(tokens_out, n_token_count * sizeof(llama_token));
    session->tokens_len = n_token_count;
    session->tokens_cap = n_token_count;
    pthread_mutex_unlock(&g_registry_mutex);

    return session_handle;
}

int llm_ai_session_token_count(int session_handle) {
    LlamaChatSession* session = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);
    pthread_mutex_lock(&session->session_mutex);
    int count = (int)session->tokens_len;
    pthread_mutex_unlock(&session->session_mutex);
    return count;
}

int llm_ai_session_set_sampling(int session_handle, float temperature,
                                float top_p, int top_k, int64_t seed) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaChatSession* session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);

    pthread_mutex_lock(&session->session_mutex);
    session->sampling_set = true;
    session->sampling_temperature = temperature;
    session->sampling_top_p = top_p;
    session->sampling_top_k = top_k;
    session->sampling_seed = seed;
    pthread_mutex_unlock(&session->session_mutex);
    return 0;
}

int llm_ai_session_cache_clear(int session_handle) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaChatSession* session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);

    pthread_mutex_lock(&session->session_mutex);
    llama_memory_t mem = llama_get_memory(session->ctx);
    llama_memory_clear(mem, /*data=*/false);
    session->tokens_len = 0;
    pthread_mutex_unlock(&session->session_mutex);
    return 0;
}

int llm_ai_session_cache_trim(int session_handle, int keep) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaChatSession* session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);

    pthread_mutex_lock(&session->session_mutex);
    /* `keep` = quanti token VECCHI rimuovere (eviction dei piu' antichi). Rimuove
     * le posizioni [0, keep) della seq default. tokens_len viene ricalcolato da
     * cache_stats dalla cache reale (pos_max-pos_min+1), non aggiornato qui. */
    if (keep > 0) {
        llama_memory_t mem = llama_get_memory(session->ctx);
        llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/0, /*p1=*/keep);
    }
    pthread_mutex_unlock(&session->session_mutex);
    return 0;
}

int llm_ai_session_cache_stats(int session_handle, int* out_n_ctx,
                               int* out_n_tokens, size_t* out_bytes,
                               int* out_pos_min, int* out_pos_max) {
    pthread_mutex_lock(&g_registry_mutex);
    LlamaChatSession* session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);

    pthread_mutex_lock(&session->session_mutex);
    uint32_t n_ctx = llama_n_ctx(session->ctx);
    llama_memory_t mem = llama_get_memory(session->ctx);
    llama_pos pos_min = llama_memory_seq_pos_min(mem, /*seq_id=*/0);
    llama_pos pos_max = llama_memory_seq_pos_max(mem, /*seq_id=*/0);
    size_t state_bytes = llama_state_get_size(session->ctx);
    /* NTokens coerente con la cache REALE (pos_max-pos_min+1), non con il
     * conteggio libro: dopo un trim i token rimossi non contano piu'. */
    int tokens = (pos_min >= 0 && pos_max >= pos_min) ? (int)(pos_max - pos_min + 1) : 0;
    session->tokens_len = (size_t)tokens;
    pthread_mutex_unlock(&session->session_mutex);

    if (out_n_ctx) *out_n_ctx = (int)n_ctx;
    if (out_n_tokens) *out_n_tokens = tokens;
    if (out_bytes) *out_bytes = state_bytes;
    if (out_pos_min) *out_pos_min = (int)pos_min;
    if (out_pos_max) *out_pos_max = (int)pos_max;
    return 0;
}

/* ── Estensione 9.1 (LatentMAS) — KV-cache transfer IN-RAM ──────────────────
 * llama_state_get_data/set_data copiano lo stato del contesto (KV + tokens) in un
 * buffer di memoria — niente file (a differenza di swap_out/in). Valido tra sessioni
 * dello STESSO modello con la stessa config ctx. */
int llm_ai_session_kv_export(int session_handle, unsigned char* out, int cap) {
    LlamaChatSession* session = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);
    pthread_mutex_lock(&session->session_mutex);
    size_t needed = llama_state_get_size(session->ctx);
    if (needed == 0) { pthread_mutex_unlock(&session->session_mutex); return -1; }
    /* size query: out==NULL o cap<=0 → ritorna la dimensione necessaria */
    if (out == NULL || cap <= 0) { pthread_mutex_unlock(&session->session_mutex); return (int)needed; }
    if (needed > (size_t)cap) { pthread_mutex_unlock(&session->session_mutex); return -1; }
    size_t n = llama_state_get_data(session->ctx, out, needed);
    pthread_mutex_unlock(&session->session_mutex);
    return n > 0 ? (int)n : -1;
}

int llm_ai_session_kv_import(int session_handle, const unsigned char* buf, int n) {
    if (buf == NULL || n <= 0) return -1;
    LlamaChatSession* session = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);
    pthread_mutex_lock(&session->session_mutex);
    size_t applied = llama_state_set_data(session->ctx, buf, (size_t)n);
    pthread_mutex_unlock(&session->session_mutex);
    return applied > 0 ? 0 : -1;
}

/* ── FASE 5 (RecursiveMAS R5 — latent nativo) ──────────────────────────────
 * Contesto TEMPORANEO con embeddings=true per estrarre hidden senza toccare
 * le sessioni di generazione (il sampler richiede embeddings=false). */

static int llm_emb_run_once(LlamaModel* model, const llama_token* tokens, int n_tokens, float* out, int cap)
{
    if (model == NULL || model->model == NULL || (n_tokens > 0 && tokens == NULL) || out == NULL || cap <= 0) return -1;
    if (n_tokens <= 0) return -1;

    /* Contestuale al MODELLO: bert/bge hanno context breve (es. 512). n_ctx/n_batch/n_ubatch
     * vanno dimensionati sul context train del modello — non su 4096 fisso: un ubatch enorme
     * faceva esplodere il grafo bert ('gather i01 out of range'). Il testo è già troncato al
     * context PRIMA dal chiamante; qui il ctx copre n_tokens. */
    int32_t ctx_train = llama_model_n_ctx_train(model->model);
    uint32_t need = (uint32_t)(ctx_train > 0 ? ctx_train : 512);
    if ((uint32_t)n_tokens > need) return -1;   // guardia: mai decode oltre il context del modello

    /* CONTEXT EMBEDDING PERSISTENTE per modello (fix Metal, PR #17869): crearlo una volta e
     * riusarlo tra le chiamate. Su Metal llama_init_from_model+llama_free ripetuti per ogni
     * testo facevano scattare l'assert di cleanup '[rsets->data count] == 0' alla distruzione
     * del device (il RAG embedda migliaia di chunk → init/free a ogni chunk crashava). */
    pthread_mutex_lock(&model->emb_lock);
    if (model->emb_ctx == NULL || model->emb_ctx_need != (int)need)
    {
        if (model->emb_ctx != NULL) llama_free(model->emb_ctx);
        model->emb_ctx = NULL;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = need;
        cp.n_batch = need;
        cp.n_ubatch = need;
        cp.embeddings = true;
        model->emb_ctx = llama_init_from_model(model->model, cp);
        if (model->emb_ctx == NULL) { pthread_mutex_unlock(&model->emb_lock); return -1; }
        model->emb_ctx_need = (int)need;
    }
    llama_context* ctx = model->emb_ctx;

    llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { pthread_mutex_unlock(&model->emb_lock); return -1; }

    /* Pooling: per i modelli bert/bge il GGUF dichiara bert.pooling_type (CLS/MEAN) e llama lo
     * applica al context (llama-context.cpp: se params.pooling_type==UNSPECIFIED eredita
     * hparams.pooling_type). Con pooling != NONE l'embedding di SEQUENZA sta in embd_seq (via
     * llama_get_embeddings_seq), NON nelle righe per-token di llama_get_embeddings_ith — usare
     * _ith(-1) su un bert con pooling restituiva un vettore di ZERI (fix 2026-09-06: RAG build
     * con bge produceva embedding nulli → ogni query coseno degenerato ~0.016). */
    enum llama_pooling_type pooling = llama_pooling_type(ctx);
    float* emb = NULL;
    if (pooling == LLAMA_POOLING_TYPE_NONE)
        emb = llama_get_embeddings_ith(ctx, 0);
    else
        emb = llama_get_embeddings_seq(ctx, 0);   /* batch singolo: seq_id 0 */
    if (emb == NULL) { pthread_mutex_unlock(&model->emb_lock); return -1; }

    /* dimensione output effettiva (con pooling = dim embedding della sequenza) */
    int dim = llama_model_n_embd_out(model->model);
    int n = dim < cap ? dim : cap;
    memcpy(out, emb, (size_t)n * sizeof(float));

    /* normalizzazione L2 (il retrieval RAG usa coseno: embedding unitari = prodotto scalare) */
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)out[i] * (double)out[i];
    if (acc > 0.0)
    {
        double inv = 1.0 / sqrt(acc);
        for (int i = 0; i < n; i++) out[i] = (float)((double)out[i] * inv);
    }
    pthread_mutex_unlock(&model->emb_lock);
    return n;
}

int llm_ai_emb_dim(int model_handle)
{
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* model = find_model_locked(model_handle);
    int dim = (model != NULL) ? llama_model_n_embd(model->model) : -1;
    pthread_mutex_unlock(&g_registry_mutex);
    return dim;
}

int llm_ai_session_last_hidden(int session_handle, float* out, int cap)
{
    LlamaChatSession* session = NULL;
    LlamaModel* model = NULL;
    pthread_mutex_lock(&g_registry_mutex);
    session = find_session_locked(session_handle);
    if (session == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    model = find_model_locked(session->model_handle);
    if (model == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    pthread_mutex_unlock(&g_registry_mutex);

    pthread_mutex_lock(&session->session_mutex);
    int rc = llm_emb_run_once(model, session->tokens, (int)session->tokens_len, out, cap);
    pthread_mutex_unlock(&session->session_mutex);
    return rc;
}

int llm_ai_model_embed_text(int model_handle, const char* text, float* out, int cap)
{
    if (text == NULL || out == NULL || cap <= 0) return -1;
    pthread_mutex_lock(&g_registry_mutex);
    LlamaModel* model = find_model_locked(model_handle);
    if (model == NULL) { pthread_mutex_unlock(&g_registry_mutex); return -1; }
    const llama_vocab* vocab = model->vocab;
    int32_t ctx_train = llama_model_n_ctx_train(model->model);
    pthread_mutex_unlock(&g_registry_mutex);

    int n_tok = -llama_tokenize(vocab, text, (int32_t)strlen(text), NULL, 0, true, true);
    if (n_tok <= 0) return -1;
    /* Tokenizza TUTTO il testo (buffer pieno), poi passa solo i primi ctx_train token al
     * decode: bert/bge hanno context breve (512) e i chunk RAG (2048 char) lo superano —
     * l'encoder non accetta oltre il suo context (crash gather). Embeddare la parte iniziale
     * è il comportamento di retrieval corretto per questi modelli. */
    llama_token* toks = (llama_token*)malloc((size_t)n_tok * sizeof(llama_token));
    if (toks == NULL) return -1;
    int got = llama_tokenize(vocab, text, (int32_t)strlen(text), toks, (int32_t)n_tok, true, true);
    if (got < 0 || got > n_tok) { free(toks); return -1; }
    int use = got;
    if (ctx_train > 0 && use > ctx_train) use = ctx_train;
    int rc = llm_emb_run_once(model, toks, use, out, cap);
    free(toks);
    return rc;
}
