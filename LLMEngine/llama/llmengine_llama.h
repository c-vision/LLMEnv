/* ============================================================================
 * llmengine_llama.h
 * Native LLM driver -- wraps llama.cpp's C API (llama.h) behind a handle-based
 * resource manager, same pattern as sdl2_wrapper.h/.cpp and sqlite3_wrapper.h/.cpp
 * (see 03-Architecture/C-Target-Plan.md §12.1, and Native-LLM-Provider-Plan.md
 * for the full integration plan).
 *
 * FASE 2: model and session are separate handles. One model (GGUF weights) can
 * back many concurrent sessions (one llama_context + KV cache each) -- this is
 * what lets AgentServer give each AgentRecord its own persistent conversation
 * state without reloading weights per agent. Each session's KV cache survives
 * across calls: llm_ai_session_append tokenizes+decodes only the NEW text
 * since the last call (diffed against the chat-template-formatted transcript),
 * never the whole history from scratch.
 *
 * Backs Basic.AI.Rag.ISessionAwareLlmCompleter / ILlmCompleter implementations
 * in Basic.Providers.LlamaCppLlmCompleter (kind "llamacpp") -- zero HTTP/JSON,
 * in-process call from the C# host via P/Invoke. Implementation in
 * llmengine_llama.cpp is C++ (llama.cpp's own headers are C++), this
 * header's surface is pure C so it P/Invokes cleanly from .NET without a
 * Swift/ObjC-style bridging layer.
 * ============================================================================ */
#ifndef LLMENGINE_LLAMA_HOST_AI_LLAMACPP_H
#define LLMENGINE_LLAMA_HOST_AI_LLAMACPP_H

#include <stdint.h> /* int64_t per llm_ai_session_set_sampling */
#include <stddef.h> /* size_t per llm_ai_session_cache_stats */

#ifdef __cplusplus
extern "C" {
#endif

/* Loads a GGUF model's weights, returns an opaque handle >= 0, or -1 on
 * failure. n_gpu_layers: -1 offloads all layers to the platform's GPU backend
 * (Metal/CUDA/ROCm/Vulkan, whichever this build was compiled with); 0 forces
 * CPU-only. A model handle owns no KV cache/context by itself -- create one
 * or more sessions from it via llm_ai_create_session. */
int llm_ai_load_model(const char* model_path, int n_gpu_layers);

/* Releases a model's weights. Caller must destroy all sessions created from
 * this model first -- undefined behavior otherwise (matches llama.cpp's own
 * ownership: a context cannot outlive its model). Safe to call on an
 * already-invalid handle (no-op). */
void llm_ai_unload_model(int model_handle);

/* Human-readable model name (as embedded in the GGUF metadata), or NULL if
 * the handle is invalid. Caller does NOT own this pointer -- valid only
 * until llm_ai_unload_model(model_handle). */
const char* llm_ai_model_name(int model_handle);

/* Creates a new conversation session (llama_context + KV cache) bound to a
 * loaded model. ctx_size: context window in tokens (n_ctx). kv_type_k/
 * kv_type_v: ggml_type values (see ggml.h -- GGML_TYPE_F16=1 is llama.cpp's
 * own default/unquantized, GGML_TYPE_Q8_0=8 and GGML_TYPE_Q4_0=2 are the
 * quantized KV-cache types this wrapper's kvQuantize/kvTurboQuant config
 * keys map to) -- pass GGML_TYPE_F16 for both to get the original
 * unquantized behavior. Returns an opaque handle >= 0, or -1 on failure
 * (e.g. invalid model handle). */
int llm_ai_create_session(int model_handle, int ctx_size, int kv_type_k, int kv_type_v);

/* Releases a session's context + KV cache. Does NOT touch the underlying
 * model (other sessions on the same model are unaffected). Safe to call on
 * an already-invalid handle (no-op). */
void llm_ai_destroy_session(int session_handle);

/* Appends one chat turn (role = "system"/"user"/"assistant", content = the
 * message text) to the session's transcript and decodes ONLY the newly
 * added text into the KV cache -- prior turns are never re-tokenized or
 * re-decoded. Uses the model's own chat template (falls back to a plain
 * concatenation if the model has none). Returns 0 on success, -1 on error
 * (invalid handle, tokenize/decode failure, or a chat template that isn't
 * purely additive turn-to-turn). */
int llm_ai_session_append(int session_handle, const char* role, const char* content);

/* Generates a reply continuing from the session's current KV cache state
 * (i.e. after the most recent llm_ai_session_append call) -- no re-tokenize
 * of prior turns. The generated text is recorded as an "assistant" turn in
 * the session's own transcript bookkeeping (its closing template markup, if
 * any, is decoded lazily on the next llm_ai_session_append call rather than
 * here). Returns a malloc'd, NUL-terminated UTF-8 string the caller MUST
 * release via llm_ai_free_string -- never call free() directly from C# on
 * this pointer. Returns NULL on error (invalid handle, decode failure, out
 * of context). */
char* llm_ai_session_generate(int session_handle, int max_tokens);

/* API a livello di ID di token (tokenizer e template del chiamante): decodifica `n` id nel KV; 0 = ok, -1 = errore. */
int llm_ai_session_append_ids(int session_handle, const int32_t* ids, int n);
/* Genera fino a `max_new` id (si ferma su `eos` o su un token di fine del vocabolario, incluso in `out`). Ritorna il conteggio. */
int llm_ai_session_generate_ids(int session_handle, int max_new, int32_t eos, int32_t* out);

/* Configures per-session sampling for the NEXT llm_ai_session_generate call.
 * Until called, the session uses the historical default chain
 * (min_p=0.05 + temperature=0.8 + seeded-from-entropy) so existing callers
 * see no behavior change. Semantics (mirrors llama.cpp sampler chain):
 *   temperature > 0  -> temperature sampling; temperature <= 0 -> greedy
 *                      (deterministic: argmax of the logits, ignores top_p/
 *                      top_k/seed). top_p and top_k only apply when
 *                      temperature > 0.
 *   top_p     in (0,1]  (default 1.0 = disabled when 0/<=0 passed)
 *   top_k     > 0       (default 0 = disabled when <=0 passed)
 *   seed      any int64; LLAMA_DEFAULT_SEED is used when 0 is passed (seeded
 *                      from entropy). Persisted in the swap meta-file so a
 *                      swapped-out/in session restores the same sampling.
 * Returns 0 on success, -1 on invalid handle. */
int llm_ai_session_set_sampling(int session_handle, float temperature,
                                float top_p, int top_k, int64_t seed);

/* Releases a string returned by llm_ai_session_generate. */
void llm_ai_free_string(char* s);

/* KV-cache manager (P0, 2026-08-31) — primitive base sulla memoria del
 * contesto (llama_get_memory + llama_memory_*). La sessione usa la seq
 * default (0, single-sequence). Tutte ritornano 0 su successo, -1 su handle
 * invalido; cache_stats scrive i valori SOLO sui puntatori non-NULL. */

/* Svuota completamente la KV-cache della sessione (llama_memory_clear).
 * NB: lo stato libro (tokens/prev_formatted/decoded_anything) NON viene
 * toccato — dopo un clear la generazione riprende come se il contesto fosse
 * vuoto ma i turni restano nel transcript. */
int llm_ai_session_cache_clear(int session_handle);

/* Rimuove i primi `drop` token della seq (eviction dei piu' vecchi,
 * llama_memory_seq_rm su [0, drop)). drop <= 0 => no-op (ritorna 0). Il
 * conteggio dei token viene ricalcolato da cache_stats dalla cache reale. */
int llm_ai_session_cache_trim(int session_handle, int drop);

/* Statistiche della KV-cache:
 *   *out_n_ctx     = dimensione del contesto (n_ctx)
 *   *out_n_tokens  = token correnti nella seq (pos_max+1)
 *   *out_bytes     = byte approssimati dello stato (llama_state_get_size)
 *   *out_pos_min   = posizione minima presente
 *   *out_pos_max   = posizione massima presente
 * Ritorna 0 su successo, -1 su handle invalido. */
int llm_ai_session_cache_stats(int session_handle, int* out_n_ctx,
                               int* out_n_tokens, size_t* out_bytes,
                               int* out_pos_min, int* out_pos_max);

/* FASE 2b: disk swap-out/swap-in of a session, so a long-lived AgentServer
 * agent doesn't have to hold its KV cache in RAM forever. llama_state_save_file
 * only captures the raw KV cache + token history -- it knows nothing about
 * this wrapper's own chat-transcript bookkeeping (messages, prev_formatted,
 * decoded_anything) that llm_ai_session_append's incremental-decode diffing
 * depends on. Swap-out therefore writes TWO files under path_prefix:
 * "<path_prefix>.kv" (llama_state_save_file's own format) and
 * "<path_prefix>.meta" (a small versioned format for the wrapper-level
 * state) -- both written to ".tmp" siblings first and rename()'d into place
 * only once fully written, so a crash mid-write never leaves a corrupt pair
 * on disk (matches the commit-atomico requirement from the swap design).
 *
 * Swap-out does NOT destroy session_handle -- caller decides (typically:
 * verify success, then llm_ai_destroy_session to actually free the RAM).
 * Swap-in creates a brand-new session (its own new llama_context, sized
 * from the ctx_size recorded at swap-out time) and returns a NEW handle --
 * the original, swapped-out handle is gone and must not be reused. */
int llm_ai_session_swap_out(int session_handle, const char* path_prefix);
int llm_ai_session_swap_in(const char* path_prefix);

/* Diagnostic: number of tokens currently decoded into a session's KV cache
 * (i.e. session->tokens.size()), or -1 if the handle is invalid. Sampling is
 * not deterministic (LLAMA_DEFAULT_SEED means "seed from entropy", not a
 * fixed value) so generated text can't be used to verify a swap-out/swap-in
 * round-trip byte-for-byte -- token count can, since it depends only on what
 * was actually decoded, not on sampling. Also the natural basis for a future
 * PROVIDER.SWAPSTATUS-style diagnostic command. */
int llm_ai_session_token_count(int session_handle);

/* Estensione 9.1 (LatentMAS) — KV-cache transfer IN-RAM tra sessioni dello STESSO modello.
 * Esporta/importa lo stato KV+token di una sessione via llama_state_get_data/set_data
 * (niente I/O su disco, a differenza di swap_out/in). Valido solo tra sessioni con lo
 * stesso modello e stessa config ctx. Ritorna: export → byte scritti (o -1 su errore);
 * import → 0 ok / -1 errore. */
int llm_ai_session_kv_export(int session_handle, unsigned char* out, int cap);
int llm_ai_session_kv_import(int session_handle, const unsigned char* buf, int n);

/* FASE 5 (RecursiveMAS R5 — latent nativo): embedding/hidden states.
 * Queste API NON toccano le sessioni di generazione (che restano con
 * embeddings=false per il sampler): creano un contesto temporaneo con
 * embeddings=true e vi decodificano i token, leggendo l'hidden del token
 * finale via llama_get_embeddings_ith. Costo O(n_tokens) per chiamata —
 * adeguato per i test/gate e per l'estrazione sporadica, non per un loop
 * latente per-token (per quello servirebbe un contesto embeddings gemello). */

/* Dimensione embedding del modello (llama_model_n_embd). -1 se handle invalido. */
int llm_ai_emb_dim(int model_handle);

/* Hidden finale della sessione: decodifica session->tokens in un contesto
 * embeddings temporaneo e copia i primi n_embd float in out (cap >= n_embd).
 * Ritorna n_embd (>=0) o -1 su errore. */
int llm_ai_session_last_hidden(int session_handle, float* out, int cap);

/* Embedding di un testo: tokenizza `text` (is_first=1) e decodifica in un
 * contesto embeddings temporaneo; copia n_embd float in out. Ritorna n_embd o -1. */
int llm_ai_model_embed_text(int model_handle, const char* text, float* out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_LLAMA_HOST_AI_LLAMACPP_H */
