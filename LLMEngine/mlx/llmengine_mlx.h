/* ============================================================================
 * llmengine_mlx.h
 * Native LLM driver -- wraps MLX's C API (mlx-c) for Apple Silicon-specific
 * inference. FASE 1 of the native LLM provider plan (see
 * Native-LLM-Provider-Plan.md / Fase1-MLX-Qwen3MoE-Plan.md in the Obsidian
 * vault): unlike llmengine_llama (which wraps a complete, existing
 * inference engine), this wraps only tensor primitives -- the transformer
 * architecture graph (attention, RoPE, MoE routing, KV-cache) is written by
 * hand on top, one architecture at a time. First target: qwen3_moe.
 *
 * FASE 1.0 (this file, current step): scaffolding only -- a smoke test that
 * proves the mlx-c build works and the Metal backend is actually selected,
 * before any model-specific code is written. Nothing below is model-aware
 * yet.
 * ============================================================================ */
#ifndef LLM_HOST_AI_MLX_H
#define LLM_HOST_AI_MLX_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "llm_engine_opts.h"   /* registro opzioni: parametri azionabili da BASIC/CLI (niente env) */

#ifdef __cplusplus
extern "C" {
#endif

/* Runs a minimal, self-verifying computation on the MLX GPU (Metal) device:
 * builds two known 2x2 float32 matrices, multiplies them, and checks the
 * result against the hand-computed expected values. Prints the selected
 * device's description to stdout. Returns 0 if the computation ran AND
 * produced the correct numeric result, -1 otherwise (device unavailable,
 * op failure, or a wrong result -- treated the same as a hard failure,
 * never partially "OK"). This is intentionally the only entry point in
 * FASE 1.0 -- no model loading, no weights, no tokenizer yet. */
int llm_ai_mlx_smoketest(void);
/* Self-test UNITARIO della cache della mask sliding (P0-bis), senza modello: hash
 * dell'output di llm_mlx_gqa_sdpa(window>0) su q/k/v sintetici. Confrontare gli hash fra
 * processi con LLM_MASK_CACHE=0 e senza -> devono coincidere. 0 = ok. */
int llm_ai_mlx_attention_bw_selftest(int nkv, int n_heads, int hd, int nk, int reps, char* out, int cap_out);
int llm_ai_mlx_kv_view_selftest(int nkv, int cap, int need, int hd, char* out, int cap_out);
int llm_ai_mlx_mask_cache_selftest(int Lq, int Nk, int win, char* out, int cap);
int llm_ai_mlx_qwen4exp_embed_tile_selftest(int hidden, int hc_count, int vocab, int token_id, char* out, int cap);
/* Self-test UNITARIO del primitive oQ A8: risoluzione automatica delle metallib +
 * fail-open di quantized_linear_cfg su input eleggibile. 0 = ok, -2 = fail-open rotto. */
int llm_ai_mlx_oq_a8_selftest(int m, int n, int k, char* out, int cap);
/* P0-quinquies: blocco che ATTRAVERSA la finestra == calcolo incrementale (coerenza
 * maschera/buffer del percorso finestrato). mode 0=con mlx_slice, 1=senza. 0 = coerente. */
int llm_ai_mlx_sliding_selftest(int pre, int Lq, int win, int D, int mode, char* out, int cap);
/* P0-ter: microbench SYNTHETIC della GEMV di decode (ms/chiamata) per coppia dtype.
 * adt/sdt: 0=f32 1=f16 2=bf16. Nessun modello. */
int llm_ai_mlx_gemv_bench(int m, int n, int k, int bits, int group, int adt, int sdt, int reps, int pipe, char* out, int cap);
/* P0-ter (2026-09-16): costo BUILD vs ENCODING per nodo del grafo (us/nodo, min su reps).
 * Catena di `n_ops` Add elementwise su [1,1,1024]: nessun modello, millisecondi. */
int llm_ai_mlx_encode_cost_selftest(int n_ops, int reps, char* out, int cap);
/* P0-ter (2026-09-16): DRIFT del tempo di eval sullo STESSO grafo (isola lo stato interno di MLX). */
int llm_ai_mlx_eval_drift_selftest(int n_ops, int iters, char* out, int cap);
/* P0-ter: come sopra ma con FORMA CRESCENTE (isola costo-per-forma-nuova). */
int llm_ai_mlx_eval_shapedrift_selftest(int base_n, int grow, int iters, char* out, int cap);
/* P0-ter: drift della SDPA con Nk che cambia (grow>0) vs costante (grow=0). */
int llm_ai_mlx_eval_sdpadrift_selftest(int nkv_start, int nkv_grow, int iters, char* out, int cap);
/* P0-ter (2026-09-16): BANDA SOSTENIBILE (Fase 1). mode 0=triad, 1=read, 2=write; f32; MB. */
int llm_ai_mlx_bw_selftest(int size_mb, int mode, int reps, char* out, int cap);

/* FASE 1.1 checkpoint: loads `model.embed_tokens.{weight,scales,biases}`
 * from the given safetensors shard (quantized, group_size=64/bits=8 --
 * verified against the real model.safetensors.index.json, not assumed),
 * extracts the row for `token_id` via mlx_take_axis, dequantizes just
 * that row, and prints the first 8 dequantized float32 values to stdout.
 * Not a general-purpose loader yet -- proves the safetensors + quantized
 * gather + dequantize path works end-to-end on one real tensor before
 * building the full model loader. Returns 0 on success, -1 on any
 * failure (missing tensor, shape mismatch, NaN in output). */
int llm_ai_mlx_embedding_checkpoint(const char* safetensors_path, int token_id);

/* FASE 1.3 checkpoint: runs layer 0's full self-attention sub-block for a
 * single token -- input_layernorm (RMSNorm) -> Q/K/V projections (quantized
 * matmul) -> per-head reshape -> QK-norm (RMSNorm, BEFORE RoPE) -> RoPE
 * (traditional=false, base=rope_theta) -> GQA scaled-dot-product-attention
 * (mlx_fast_scaled_dot_product_attention, no manual repeat/tile -- verified
 * against mlx_lm/models/qwen3_moe.py that GQA broadcast is handled
 * natively) -> output projection -> residual add with the ORIGINAL
 * (pre-norm) embedding. Prints the first 8 values of both the raw
 * attention output and the post-residual result, for comparison against a
 * real mlx-lm run of the same layer on the same token (see
 * Fase1-MLX-Qwen3MoE-Plan.md FASE 1.3 for the reference values). Returns 0
 * on success, -1 on any failure. rope_theta is passed explicitly (not
 * hardcoded) since it must come from the model's own config.json. */
int llm_ai_mlx_attention_checkpoint(const char* safetensors_path, int token_id, float rope_theta);

/* FASE 1.4 checkpoint: runs layer 0's full block (attention, same as
 * llm_ai_mlx_attention_checkpoint, followed by the MoE FFN) for a single
 * token -- post_attention_layernorm (RMSNorm) -> router (quantized matmul
 * against `mlp.gate`, softmax w/ precise=true, argpartition top-8,
 * renormalized since norm_topk_prob=true for this model) -> for each of
 * the 8 selected experts: gate/up/down projections sliced out of the
 * stacked `mlp.switch_mlp.{gate,up,down}_proj` tensors via mlx_take_axis
 * (one quantized matmul per selected expert -- mathematically identical
 * to mlx_gather_qmm's unsorted dispatch for a single token, since the
 * sort-by-expert optimization mlx-lm uses only triggers at >=64 gathered
 * rows; verified against mlx_lm/models/switch_layers.py) -> SwiGLU
 * (silu(gate) * up) -> down projection -> weighted sum by the
 * renormalized router scores -> residual add with the post-attention
 * hidden state. Prints the selected expert ids, their renormalized
 * scores, the raw MoE output, and the final layer-0 output (first 8
 * values each), for comparison against a real mlx-lm run of the same
 * layer on the same token (see Fase1-MLX-Qwen3MoE-Plan.md FASE 1.4 for
 * the reference values). Returns 0 on success, -1 on any failure. */
int llm_ai_mlx_moe_checkpoint(const char* safetensors_path, int token_id, float rope_theta);

typedef void* llm_ai_mlx_model_handle;

/* FASE 1.5: carica il modello COMPLETO -- legge le dimensioni reali da
 * `model_dir/config.json` (nessun hardcoded: hidden_size, num_hidden_layers,
 * num_attention_heads, num_key_value_heads, head_dim, num_experts,
 * num_experts_per_tok, moe_intermediate_size, vocab_size, rms_norm_eps,
 * rope_theta, norm_topk_prob, tie_word_embeddings), poi carica TUTTI gli
 * shard safetensors elencati in `model_dir/model.safetensors.index.json`
 * (non solo il primo, a differenza dei checkpoint di Fase 1.1/1.3/1.4).
 * Ritorna un handle opaco riusabile per più chiamate a
 * llm_ai_mlx_generate, NULL su qualunque errore (file mancante, schema
 * inatteso, shard mancante). */
llm_ai_mlx_model_handle llm_ai_mlx_load_model(const char* model_dir);

/* Offload esperti MoE da disco (streaming): `dir` è la directory del modello da cui leggere solo i
 * k esperti attivi per step invece di materializzare tutti gli esperti. Va chiamata PRIMA di
 * `llm_ai_mlx_load_model` (la dir coincide con quella del load). `NULL`/"" disattiva. Nessun env. */
void llm_ai_mlx_set_moe_swap_dir(const char* dir);
/* TurboQuant: scelta MEMORIA (1, default: esperti packed, RSS basso) vs VELOCITA' (0: affine-2bit
 * residenti + gather_qmm MLX). API del motore, impostata dal consumer (BASIC/FastLLM), non env. */
void llm_ai_mlx_set_tq_memory_opt(int on);

void llm_ai_mlx_free_model(llm_ai_mlx_model_handle handle);

/* ── GGUF single-file supportato dal motore MLX (auto-route da BASIC/C#) ──────
 * llm_ai_mlx_gguf_supported: 1 se l'architettura GGUF (general.architecture) e' gestita
 *   direttamente dal motore MLX (oggi solo DS4 = "deepseek4"), altrimenti 0.
 * llm_ai_mlx_gguf_tokenizer_json: costruisce la JSON del tokenizer DS4 (BPE + pre-tokenizer
 *   JoyAI) leggendo vocab+merges dal GGUF. Ritorna una stringa malloc'ata (libera con
 *   llm_ai_mlx_free_str) e ne scrive la lunghezza in *out_len; NULL su errore. Pensata per
 *   essere passata a llmtoken_load_json(byte, len). */
int llm_ai_mlx_gguf_supported(const char* path);
char* llm_ai_mlx_gguf_tokenizer_json(const char* path, size_t* out_len);
void llm_ai_mlx_free_str(char* p);

/* Genera fino a max_new_tokens token id in modo autoregressivo (sampling
 * greedy/argmax, deterministico) a partire dagli id del prompt. Il prompt
 * viene processato un token alla volta (non in un'unica passata batched --
 * scelta deliberata, vedi commento nell'implementazione) per costruire il
 * KV-cache per tutti i layer, poi la decodifica prosegue incrementalmente
 * riusando il cache. Si ferma prima di max_new_tokens se genera
 * eos_token_id (ignorato se negativo). Scrive gli id generati in out_ids
 * (buffer di almeno max_new_tokens elementi). Ritorna il conteggio REALE
 * generato, o -1 su errore. */
int llm_ai_mlx_generate(llm_ai_mlx_model_handle handle, const int32_t* prompt_ids, int n_prompt_ids,
                         int32_t* out_ids, int max_new_tokens, int32_t eos_token_id);

typedef void* llm_ai_mlx_session_handle;

/* FASE 2b prerequisite: persistent KV-cache session, so a conversation's KV-cache survives
 * across calls instead of every llm_ai_mlx_generate call re-prefilling from scratch (same
 * "session" concept llmengine_llama.cpp already has via llm_ai_create_session, ported
 * here now that it's needed). Creates a session bound to a loaded model with an empty
 * KV-cache for every layer. Returns an opaque handle, or NULL on invalid model_handle. */
llm_ai_mlx_session_handle llm_ai_mlx_create_session(llm_ai_mlx_model_handle model_handle);

/* Releases a session's KV-cache. Does NOT touch the underlying model. Safe to call on an
 * already-invalid handle (no-op). */
void llm_ai_mlx_destroy_session(llm_ai_mlx_session_handle handle);

/* Decodes token_ids (already-tokenized -- chat templating/tokenization stays in C#, unlike
 * llama.cpp's native layer which owns it) into the session's persistent KV-cache, growing it
 * by n_ids positions. Also computes (but does not emit) the predicted continuation token, so
 * a following llm_ai_mlx_session_generate call starts from exactly where this left off --
 * mirrors llama.cpp's llm_ai_session_append + llm_ai_session_generate split. Returns 0 on
 * success, -1 on failure (invalid handle, decode failure). */
int llm_ai_mlx_session_append(llm_ai_mlx_session_handle handle, const int32_t* token_ids, int n_ids);

/* Generates up to max_new_tokens ids continuing from the session's current KV-cache state
 * (i.e. after the most recent llm_ai_mlx_session_append call) -- greedy/argmax, deterministic,
 * same sampling as llm_ai_mlx_generate. Stops early on eos_token_id (ignored if negative).
 * Writes to out_ids (buffer of at least max_new_tokens elements). Returns the REAL count
 * generated, or -1 on error (invalid handle, nothing appended yet, decode failure). */
int llm_ai_mlx_session_generate(llm_ai_mlx_session_handle handle, int max_new_tokens,
                                int32_t eos_token_id, int32_t* out_ids);

/* Self-speculative n-gram (GENERALE, tutti i modelli, greedy): costruisce la tabella n-gram dal
 * contesto (`build_ngram`) e genera con draft multipli verificati dal modello (`generate_ngram`,
 * output IDENTICO al greedy autoregressivo). Opzione di registro `spec_ngram` (0=off, N>=2=ordine). */
int llm_ai_mlx_session_build_ngram(llm_ai_mlx_session_handle handle, int order);
int llm_ai_mlx_session_generate_ngram(llm_ai_mlx_session_handle handle, int max_new_tokens,
                                      int32_t eos_token_id, int32_t* out_ids, int draft_len);

/* Speculative streaming (spec_ngram): helper di supporto al buffering del chunk. `spec_begin`
 * salva i checkpoint K/V (alias O(1)) del chunk-start; `spec_discard` scarta i token non
 * consumati (rollback cache al chunk-start + ri-forward dei consumati + tronca la history) e va
 * chiamato su ogni append esterno; `spec_free` libera i checkpoint alla distruzione. */
struct LlmMlxSession;
void llm_mlx_session_spec_begin(struct LlmMlxSession* s);
void llm_mlx_session_spec_discard(struct LlmMlxSession* s);
void llm_mlx_session_spec_free(struct LlmMlxSession* s);
/* True se la tabella n-gram predice >=2 token dalla history (draft[0]==pending): il dispatch usa
 * lo spec solo in quel caso, per evitare il fallback AR sincrono di generate_ngram. */
int llm_mlx_session_spec_can(struct LlmMlxSession* s);
/* Larghezza adattiva del draft spec (stile Gufo): K in [2,cap] da acceptance EWMA + full-accept. */
int llm_mlx_spec_adaptive_width(struct LlmMlxSession* s, int cap);

/* Fase 3 continuous batching (default OFF via LLM_BATCH_DECODE): genera fino a max_new_tokens
 * per N sessioni (stesso modello) in UN forward BATCHED per step (attention per-sessione +
 * FFN/proiezioni batched). out_ids: n*max_new_tokens, row-major (sessione i -> out_ids[i*max_new+t]).
 * Tutte le sessioni devono condividere il modello e avere has_pending. Ritorna 0 o -1. */
/* S6 — batching speculativo multi-sessione: draft MTP per sessione + UN verify batch per round.
 * Stesso contratto di session_generate_batch. Ritorna -3 se non eleggibile (no head MTP, o
 * qualche sessione non greedy): il chiamante deve ricadere sul batch normale. */
int llm_ai_mlx_session_generate_batch_spec(llm_ai_mlx_session_handle* handles, int n,
                                           int max_new_tokens, int32_t eos_token_id,
                                           int32_t* out_ids);
int llm_ai_mlx_session_generate_batch(llm_ai_mlx_session_handle* handles, int n,
                                      int max_new_tokens, int32_t eos_token_id, int32_t* out_ids);

/* Fase 4 (prefix cache automatico, usato dal server C): salva il prefisso della sessione `sh`
 * con chiave `name` (auto-save dopo una generazione) e appende i token riusando il prefisso
 * automatico salvato che meglio matcha (min_match = lunghezza minima per riusare). Entrambe
 * ritornano 0 su successo, -1 su errore. */
int llm_ai_mlx_model_save_prefix(llm_ai_mlx_model_handle mh, const char* name, llm_ai_mlx_session_handle sh);
int llm_ai_mlx_session_append_auto_prefix(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh,
                                          const int32_t* token_ids, int n_ids, int min_match);

/* §G-ter — agentic rewind: con LLM_KVDISK attivo il motore scrive checkpoint dei soli layer
 * GLOBAL ai confini di LLM_KVCHECKPOINT_BLOCK token (default 256) durante il append. Questa
 * funzione ripristina nella sessione il checkpoint piu' lungo <= keep_tokens, ripristinando la
 * KV da disco e impostando cache_offset/history a quella posizione; il chiamante appende poi il
 * suffisso (re-prefill minimo, es. dopo aver rimosso il reasoning di un agent loop).
 * `tokens` = prefisso (almeno n_tokens elementi). Ritorna la posizione ripristinata (>0) o -1
 * (nessun checkpoint disponibile / tier SSD OFF). */
int llm_ai_mlx_session_rewind(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh,
                              const int32_t* tokens, int n_tokens, int keep_tokens);

/* §G-ter EVICT: scrive su llm_kv_disk TUTTA la KV della sessione (tutti i layer) + lo stato
 * pending e LIBERA gli array dalla RAM (unified memory: RAM ∝ sessioni ATTIVE, non a tutte
 * quelle viste). Il restore è ON-DEMAND: il prossimo llm_ai_mlx_session_append/generate
 * richiama automaticamente unpark. Richiede LLM_KVDISK attivo. Ritorna 0 o -1. */
int llm_ai_mlx_session_park(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh);

/* Configures per-session sampling for the NEXT llm_ai_mlx_session_generate.
 * Until called, the session keeps historical greedy-argmax behavior.
 *   temperature > 0  -> temperature sampling; <= 0 -> greedy (deterministic)
 *   top_p     in (0,1] (default 1.0 = disabled when <=0 passed)
 *   top_k     > 0     (default 0 = disabled when <=0 passed)
 *   seed      any int64; when 0 a fixed deterministic default is used.
 * Sampling happens in host memory (same logit-read path as the constrained
 * variant), so generation becomes reproducible for a given seed. Returns 0 on
 * success, -1 on invalid handle. */
int llm_ai_mlx_session_set_sampling(llm_ai_mlx_session_handle handle,
                                     float temperature, float top_p, int top_k,
                                     int64_t seed);

/* Repetition penalty (>1.0 penalizza i token gia' visti; 1.0 = off). Indipendente da
 * set_sampling (non ne cambia la firma). */
int llm_ai_mlx_session_set_repetition_penalty(llm_ai_mlx_session_handle handle, float penalty);

/* §I: frequency_penalty / presence_penalty OpenAI-style, SEPARATE da repetition_penalty.
 * >0 => penalizza i token gia' visti (`logits[t] -= fp*count(t)`, `-= pp` se count>0); 0 = off.
 * Applicate nel path sampling (temperature>0). Valide per la generazione successiva. */
int llm_ai_mlx_session_set_frequency_penalty(llm_ai_mlx_session_handle handle, float penalty);
int llm_ai_mlx_session_set_presence_penalty(llm_ai_mlx_session_handle handle, float penalty);

/* Configures the chunked-prefill size for llm_ai_mlx_session_append: prompt
 * tokens beyond this size are processed as a sequence of chunks (KV cache
 * grows between them, causal mask preserved), so a huge prompt never builds
 * a single enormous Metal graph. chunk_size<=0 resets to the default (4096).
 * Returns 0 on success, -1 on invalid handle. */
int llm_ai_mlx_session_set_prefill_chunk(llm_ai_mlx_session_handle handle,
                                          int chunk_size);

/* T2 speculative — verify a 1 token (tappa 1, senza rollback):
 * verifica che il token proposto dal draft coincida col greedy del modello alla posizione
 * corrente (s->pending_next_id). Semantica:
 *   - proposed == greedy  -> il draft e' corretto: avanza la KV di 1 (forward_one_token +
 *     final_step), *out_accepted=1, *out_next_token = greedy (=proposed).
 *   - proposed != greedy  -> il draft e' sbagliato: NON avanza, *out_accepted=0,
 *     *out_next_token = greedy (il token corretto del modello).
 * Parita' col decode normale garantita: se il draft e' sempre corretto, emette gli stessi
 * token; se sbagliato, emette il greedy (stesso output di session_generate).
 * Returns 0 su successo, -1 su handle invalido o sessione senza pending. */
int llm_ai_mlx_session_verify_token(llm_ai_mlx_session_handle handle,
                                     int32_t proposed_token,
                                     int32_t* out_accepted,
                                     int32_t* out_next_token);

/* T2 speculative — verify BATCH dei K draft (tappa 2, lo speedup):
 * processa i K draft in UN forward batch (KV avanza di K in un solo grafo),
 * calcola i logits di ogni posizione in UN matmul, confronta l'argmax col draft
 * e accetta il piu' lungo prefisso confermato. Se non tutti i K sono accettati,
 * fa rollback della KV (mlx_slice su k_cache/v_cache per layer) ai token accettati.
 *
 * Parita' col decode normale SOLO con sampling greedy (temp<=0): l'accettazione
 * confronta l'argmax (greedy), e il next_token e' greedy. Con temp>0 l'output puo'
 * differire (il draft e' verificato col greedy) — usare solo per greedy.
 *
 * NB (2026-09-19, bug "(solved) - MLX-DFlash2-Diverge-da-Greedy-Qwen38"): "parita' col decode
 * normale" vale a livello ALGORITMICO (accettazione greedy + next_token greedy), NON bit-exact.
 * Il forward BATCH (K+1 token) differisce dal decode sequenziale (1 token) per rounding bf16,
 * amplificato dal residual stream con la profondita' (~1 ULP al layer 0 -> O(10) sull'hidden
 * finale): l'argmax puo' cambiare su logits quasi in parita' a contesto lungo. E' numerica (non
 * logica), model-agnostic (anche i densi), e NON degrada la qualita': batch e sequenziale sono
 * entrambe valutazioni valide dello stesso modello. Per l'uguaglianza stretta coi token dell'AR
 * servirebbe un verify fp32/sequenziale (opt-in, piu' lento).
 *
 * Returns 0 su successo, -1 su handle invalido o sessione senza pending.
 * Scrive *out_accepted (0..n_draft) e *out_next_token (il token corretto del modello
 * al primo mismatch, o il nuovo greedy dopo tutti i K se accettati). */
int llm_ai_mlx_session_verify_draft_batch(llm_ai_mlx_session_handle handle,
                                           const int32_t* draft_ids, int n_draft,
                                           int32_t* out_accepted,
                                           int32_t* out_next_token);

/* B — verify batch MULTI-SESSIONE (speculative batching, 2026-09-19): verifica i draft di N
 * sessioni (stesso modello) in un unico giro, batched sulle SESSIONI (forward N×1 per passo k)
 * e sequenziale sui DRAFT. draft_ids e' row-major (sessione i -> draft_ids[i*max_draft + k]),
 * draft_len[i] = numero di draft proposti per la sessione i (<= max_draft). Scrive
 * out_accepted[i] (token confermati, 0..draft_len[i]) e out_next[i] (token corretto del modello).
 * A differenza di verify_draft_batch NON fa checkpoint/rollback: feeda solo il prefisso
 * confermato, quindi la KV avanza esattamente di accepted[i]. Parita' col decode greedy SOLO
 * con sampling greedy (temp<=0). Returns 0 su successo, -1 su errore. */
int llm_ai_mlx_session_verify_draft_batch_multi(
    llm_ai_mlx_session_handle* handles, int n,
    const int32_t* draft_ids, const int* draft_len, int max_draft,
    int32_t* out_accepted, int32_t* out_next);

/* T2/DFlash2 — tronca la KV-cache ai primi n_tokens token (rollback). Usata per riallineare
 * una sessione DRAFT al prefisso confermato dal target dopo un accettazione parziale
 * (i guess draft non confermati vanno scartati). Dopo truncate has_pending=false: serve un
 * session_append prima del prossimo generate. Rifiuta se n_tokens > cache_offset o se la KV
 * quant cold (kvq) e' attiva. Returns 0 su successo, -1 su errore. */
int llm_ai_mlx_session_truncate(llm_ai_mlx_session_handle handle, int n_tokens);

/* DSpark — P(argmax | context) alla posizione "pending" (usa pending_h): la confidenza del
 * modello nel greedy corrente. Usata dalla verifica confidence-scheduled di DSpark per
 * troncare il blocco draft al prefisso con confidenza >= soglia. Ritorna 0 ok (prob in
 * [0,1]), -1 su errore/sessione senza pending. */
int llm_ai_mlx_session_greedy_prob(llm_ai_mlx_session_handle handle, float* out_prob);

/* LOGPROBS (ABI v2): logprob naturale del token `token_id` nella distribuzione del modello alla
 * posizione "pending" (pending_h). Calcola la stessa distribuzione di final_step (final_norm +
 * lm_head + softcap) e la normalizza in log-softmax: log p(id) = logit(id) - logsumexp(logits).
 * Esatta per qualunque id (greedy o campionato), non tocca lo stato della sessione. Per i modelli
 * DSV4.1 whole-model usa i logits gia' in host (`last_logits`). Ritorna 0 e scrive *out_logprob
 * (<0), oppure -1 (handle/stato invalido, id fuori vocabolario, nessun pending). */
int llm_ai_mlx_session_logprob_of(llm_ai_mlx_session_handle handle, int32_t token_id, float* out_logprob);

/* Logprob del token greedy corrente (pending_next_id) alla posizione pending. Scorciatoia di
 * logprob_of(pending_next_id): "confidenza del modello" sul prossimo token. 0 / -1. */
int llm_ai_mlx_session_last_logprob(llm_ai_mlx_session_handle handle, float* out_logprob);

/* TOPK (ABI v3): i k token piu' probabili alla posizione "pending" con le loro logprob naturali,
 * ordinati per logprob DECRESCENTE (out_ids[0]/out_logprobs[0] = argmax). Stessa distribuzione di
 * logprob_of (final_norm + lm_head + softcap + log-softmax), stato della sessione NON toccato:
 * e' la controparte "alternative" della logprob del scelto, per `top_logprobs` di OpenAI. Scrive
 * k voci (k ridotto al vocabolario se maggiore) e ritorna k, oppure -1 (argomenti invalidi/stato
 * senza pending). out_ids/out_logprobs hanno almeno k elementi. */
int llm_ai_mlx_session_topk(llm_ai_mlx_session_handle handle, int k, int32_t* out_ids,
                            float* out_logprobs);

/* Genera fino a max_new token come llm_ai_mlx_session_generate(max_new) ma, per OGNI token emesso,
 * scrive la sua logprob (out_logprobs[i]). La logprob e' catturata dalla distribuzione del modello
 * alla posizione di PARTENZA di quello step, PRIMA che il token avanzi: esatta anche con sampling
 * (si guarda up l'id effettivamente emesso nella distribuzione pre-step). Si ferma su EOS come
 * generate. out_ids/out_logprobs hanno almeno max_new elementi. Ritorna il conteggio reale (>=0)
 * o -1. Costo: un lm_head + read del vocabolario in host per token (oltre al decode). */
int llm_ai_mlx_session_generate_probs(llm_ai_mlx_session_handle handle, int max_new, int32_t eos,
                                      int32_t* out_ids, float* out_logprobs);

/* KV (ABI v2): azzera la KV-cache della sessione IN-PLACE (stesso handle), riportandola allo
 * stato di una sessione appena creata: cache k/v/aux liberate e ricreate vuote, cache_offset=0,
 * pending azzerato, history/n-gram azzerati, stato kvq/pagesel/park resettato. Il modello NON
 * viene toccato. 0 / -1. */
int llm_ai_mlx_session_reset(llm_ai_mlx_session_handle handle);

/* KV (ABI v2): token usati (cache_offset) e capacita' della KV. La capacita' e' il limite
 * configurato (cfg.kv_max_tokens > 0), altrimenti la lunghezza ALLOCATA della cache del layer 0
 * (per la KV dinamica corrente coincide con l'uso). Scrive i due valori non-NULL. 0 / -1. */
int llm_ai_mlx_session_kv_stats(llm_ai_mlx_session_handle handle, int* out_used, int* out_cap);

/* FASE 3: native tool dispatch -- decides exactly ONE token at the session's current
 * (already-pending, undecided) position via an argmax restricted to allowed_ids, instead of
 * the ordinary unrestricted argmax llm_ai_mlx_session_generate uses. No JSON, no grammar text:
 * the caller (C#) builds allowed_ids per step by walking the BPE-tokenized candidate strings
 * (tool names, enum literals) or a precomputed digit-token set (numeric arguments) -- this
 * function only ever sees a plain id list, never text. The CHOSEN id (not the discarded
 * unrestricted prediction) is what advances the session's KV-cache, so a following call
 * (constrained or not) continues from exactly where this left off. Deterministic scaffolding
 * between constrained tokens (parameter names, separators) costs nothing here -- inject it with
 * llm_ai_mlx_session_append instead, same as any other prefill. Returns 0 on success (writes
 * the chosen id to *out_token_id), -1 on failure (invalid handle, nothing pending yet -- same
 * precondition as llm_ai_mlx_session_generate -- or empty/invalid allowed_ids). */
int llm_ai_mlx_session_generate_constrained(llm_ai_mlx_session_handle handle,
                                             const int32_t* allowed_ids, int n_allowed,
                                             int32_t* out_token_id);

/* FASE 2b: swap-out su disco -- stesso contratto di llm_ai_session_swap_out in
 * llmengine_llama.h, con una differenza: il model handle di MLX e' un puntatore
 * in-processo (non un intero da registro come in llama.cpp), quindi swap_in lo riceve
 * esplicitamente dal chiamante invece di leggerlo da un meta-file. Precondizione: la
 * sessione deve avere gia' ricevuto almeno un append (fallisce esplicitamente su una
 * sessione vuota, niente di significativo da persistere). Scrive <path_prefix>.mlxkv
 * (safetensors: k/v cache per layer + pending_h) e <path_prefix>.mlxmeta (binario:
 * cache_offset/pending_next_id/has_pending/n_layers/hidden_size), entrambi via file
 * temporaneo + rename() per atomicita'. Ritorna 0 su successo, -1 su fallimento (nessun
 * file lasciato a meta' scritto). */
int llm_ai_mlx_session_swap_out(llm_ai_mlx_session_handle handle, const char* path_prefix);

/* Ricrea una sessione da file scritti da llm_ai_mlx_session_swap_out. n_layers/hidden_size
 * nel meta-file sono confrontati contro model_handle -- un modello diverso da quello di
 * swap-out fa fallire esplicitamente invece di produrre stato corrotto. Ritorna il nuovo
 * handle, o NULL su fallimento (file mancanti/corrotti, modello incompatibile). */
llm_ai_mlx_session_handle llm_ai_mlx_session_swap_in(llm_ai_mlx_model_handle model_handle,
                                                      const char* path_prefix);

/* FASE 5 (RecursiveMAS R5 — latent nativo): hidden states reali.
 * llm_ai_mlx_session_last_hidden legge pending_h — l'hidden [1,1,hidden_size] dell'ultimo
 * token, gia' mantenuto vivo per la FASE 3 constrained decode (NON una copia: il valore è
 * quello pre-lm_head, esattamente il `last_hidden` del paper). Cast esplicito a float32
 * PRIMA della lettura host (leggere direttamente i bit f16 li reinterpreta come f32 —
 * bug reale documentato in plain_linear). Ritorna hidden_size (>=0), -hidden_size se cap
 * troppo piccolo, -1 su errore/sessione vuota. */
int llm_ai_mlx_hidden_size(llm_ai_mlx_model_handle model_handle);

int llm_ai_mlx_session_last_hidden(llm_ai_mlx_session_handle session_handle, float* out, int cap);

/* Data-gen support DSpark: forward di UN chunk (n <= prefill_chunk) e copia l'hidden dell'ultimo
 * layer per OGNI posizione (out: n*hidden fp32). KV cresce (stream del corpus a chunk). Non tocca
 * pending/final_step. Ritorna n su successo, -1 su errore/cap insufficiente. */
/* MTP head (MTPLX/qwen3-next-mtp): draft di 1 token dalla head MTP del modello.
 * out_greedy = il greedy corrente della sessione (pending_next_id); out_draft = token draftato
 * dalla head. Per un MTP corretto, out_draft dovrebbe coincidere con out_greedy. */
int llm_ai_mlx_session_pending_id(llm_ai_mlx_session_handle session_handle, int32_t* out);
int llm_ai_mlx_mtp_draft1(llm_ai_mlx_session_handle session_handle, int32_t* out_draft, int32_t* out_greedy);

/* Drafta K token col head MTP (ricorrenza sul blocco con RoPE locale). Scrive K id in
 * out_drafts (>=0). Ritorna 0, -1/-2/-3 su errore. */
int llm_ai_mlx_mtp_draftk(llm_ai_mlx_session_handle session_handle, int K, int32_t* out_drafts, int32_t* out_greedy);

/* Draft depth-1 BATCHED su N sessioni (K=1): un solo forward MTP per N richieste (posizione
 * 0 -> RoPE identita', attenzione self). Scrive N id in out_drafts. 0 ok, -1/-2/-3 errore. */
int llm_ai_mlx_mtp_draft1_multi(llm_ai_mlx_session_handle* handles, int N, int32_t* out_drafts);

/* MTP speculative generate (k=1, schema MTPLX M2=[primary,draft]): emette il primary,
 * drafta col head MTP, verifica [primary,draft] in un forward batch e prosegue col bonus
 * dai logits del batch (nessun commit separato). Output identico all'AR greedy (parita'
 * algoritmica). Ritorna #token emessi, -1 errore, -2 modello senza head MTP. */
int llm_ai_mlx_session_generate_mtp(llm_ai_mlx_session_handle session_handle,
                                    int max_new_tokens, int32_t eos_token_id, int32_t* out_ids);

int llm_ai_mlx_session_capture_hidden(llm_ai_mlx_session_handle session_handle,
                                      const int32_t* token_ids, int n_ids,
                                      float* out, int cap);

/* EMBEDDING IDS-based (CAP_EMBED, ABI v4). `llm_ai_mlx_embed_dim` ritorna la dimensione
 * dell'embedding (hidden_size), o -1. `llm_ai_mlx_embed_ids` calcola il mean-pool dell'hidden
 * dell'ultimo layer su TUTTI i token di `token_ids` (n_ids), scrivendo `dim` float in `out`
 * (cap >= dim) e ritornando dim, oppure -1. Crea una sessione temporanea (la KV del chiamante
 * NON viene toccata); il chunking della cattura non cambia il risultato (la media e' lineare). */
int llm_ai_mlx_embed_dim(llm_ai_mlx_model_handle handle);
int llm_ai_mlx_embed_ids(llm_ai_mlx_model_handle handle, const int32_t* token_ids, int n_ids,
                         float* out, int cap);

/* Stima del footprint KV per-architettura (port da rapidMLX kv_estimation.py):
 * classifica i layer (full-growth / sliding sublineare / GDN stato fisso) e calcola
 * i byte della KV-cache per seq_len token (over-estimate-safe). 0 su successo. */
int llm_ai_mlx_model_kv_estimate(llm_ai_mlx_model_handle handle, int seq_len, double* out_bytes);

/* KV quant gate: dato un budget memoria per la KV, decide se quantizzare (e con
 * quanti bits) -- OFF se est<=budget, Q8 se <=2*budget, Q4 oltre. 0 su successo. */
int llm_ai_mlx_model_kv_recommend(llm_ai_mlx_model_handle handle, int seq_len,
                                  double budget_bytes, int* out_enabled, int* out_bits,
                                  double* out_estimate);

/* Persistenza della prefix cache (radix) su disco: salva/carica i KEYS (nome + token)
 * dei prefissi salvati. La KV resta in memoria (al load i prefissi hanno has_kv=false e
 * richiedono re-prefill al riuso). Write atomica (.tmp + rename). Ritorna 0 ok / n. caricati. */
int llm_ai_mlx_model_prefix_save(llm_ai_mlx_model_handle handle, const char* path);
int llm_ai_mlx_model_prefix_load(llm_ai_mlx_model_handle handle, const char* path);

/* F1 follow-on GDN — pin/unpin prefix in-memory e checkpoint stato ricorrente (FaseC) */
int llm_ai_mlx_prefix_pin(llm_ai_mlx_session_handle h, int n_tokens);
int llm_ai_mlx_prefix_unpin(llm_ai_mlx_session_handle h);
typedef void* llm_ai_mlx_state_handle;
int llm_ai_mlx_state_checkpoint(llm_ai_mlx_session_handle h, llm_ai_mlx_state_handle *out);
int llm_ai_mlx_state_restore(llm_ai_mlx_session_handle h, llm_ai_mlx_state_handle st);
void llm_ai_mlx_state_free(llm_ai_mlx_state_handle st);

/* Imatrix collector (DS4-style): accumulate per-(layer, routed expert, colonna) l'energia di
 * attivazione durante session_append su prompt di calibrazione. llm_ai_mlx_model_imatrix_begin
 * alloca (richiede modello MoE); la raccolta avviene automaticamente nei blocchi MoE con im.active;
 * llm_ai_mlx_model_imatrix_write salva il report (layer expert count <gate_cols valori> <down_cols
 * valori>, normalizzati per il conteggio) per un futuro quantizer. Ritorna 0 ok / -1. */
int llm_ai_mlx_model_imatrix_begin(llm_ai_mlx_model_handle handle);
int llm_ai_mlx_model_imatrix_write(llm_ai_mlx_model_handle handle, const char* path);

/* Guardia memoria MLX: statistiche (cache/active/peak in byte) e limite di memoria. La ritenzione
 * "active" (~picco grafo per eval su modelli grandi, non reclamata da clear_cache) puo' saturare
 * processi long-lived con molti eval grandi: il limite fa fallire in modo pulito invece del
 * Jetsam. Ritorna 0 ok / -1. */
int llm_ai_mlx_memory_stats(size_t* out_cache, size_t* out_active, size_t* out_peak);

/* §D R3.3 — training forward differenziabile (path denso): loss = masked cross-entropy (mean sul
 * batch) di `toks` con target `targets`, con un adapter LoRA su (layer, proj). Se out_dA/out_dB
 * non NULL calcola anche dLoss/dA, dLoss/dB via `mlx_value_and_grad` (autograd MLX, in-graph).
 * A [Ar,Ac] (Ac=rank), B [Br,Bc]. Ritorna 0 ok, -1 errore. */
/* LoRA model-level (registrazione/abilitazione; usate anche dal training R3.3). */
void llm_ai_mlx_model_lora_clear(llm_ai_mlx_model_handle h);
int  llm_ai_mlx_model_lora_add(llm_ai_mlx_model_handle h, int layer, const char* proj,
                               int rank, float scale, const float* a, const float* b,
                               int in_dim, int out_dim);
int  llm_ai_mlx_model_lora_set_enabled(llm_ai_mlx_model_handle h, bool enabled);

/* §D R4.6 — DEPLOY: salva/carica TUTTI i moduli LoRA registrati (adapter multi-modulo) su file.
 * Scrittura atomica. `load` azzera i moduli correnti e registra quelli del file. 0 ok / -1. */
int llm_ai_mlx_model_lora_save(llm_ai_mlx_model_handle h, const char* path);
int llm_ai_mlx_model_lora_load(llm_ai_mlx_model_handle h, const char* path);

/* §D R5: come sopra ma con MASK (1 = la posizione contribuisce alla loss), per l'SFT (loss solo
 * sulla completion). mask NULL = tutte le posizioni. */
int llm_ai_mlx_train_lora_loss_grad_mask(llm_ai_mlx_model_handle mh,
                                         const int32_t* toks, const int32_t* targets, const uint8_t* mask, int T,
                                         int layer, const char* proj,
                                         const float* A, int Ar, int Ac,
                                         const float* B, int Br, int Bc,
                                         float* out_loss, float* out_dA, float* out_dB);

/* §D R7 — DPO step: loss di preference + gradienti sul LoRA. La reference gira con l'adapter OFF.
 * Ritorna loss DPO + dLoss/dA,dB. 0 ok / -1. */
int llm_ai_mlx_train_dpo_step(llm_ai_mlx_model_handle mh,
                              const int32_t* tw, const int32_t* tgw, const uint8_t* mw, int Tw,
                              const int32_t* tl, const int32_t* tgl, const uint8_t* ml, int Tl,
                              int layer, const char* proj,
                              const float* A, int Ar, int Ac, const float* B, int Br, int Bc,
                              float beta, float* out_loss, float* out_dA, float* out_dB);

int llm_ai_mlx_train_lora_loss_grad(llm_ai_mlx_model_handle mh,
                                    const int32_t* toks, const int32_t* targets, int T,
                                    int layer, const char* proj,
                                    const float* A, int Ar, int Ac,
                                    const float* B, int Br, int Bc,
                                    float* out_loss, float* out_dA, float* out_dB);

/* §M/L25 — probe di attribuzione della ritenzione: stampa il bilancio `mlx_array_new*−free`
 * (net costante = nessun leak NOSTRO) + active/peak/cache MLX. Con build `-DLLM_COUNT_ARRAYS=1`
 * riporta i contatori in *out_new / *out_free / *out_net (altrimenti 0). */
int llm_ai_mlx_debug_array_balance(const char* tag, long* out_new, long* out_free, long* out_net);
int llm_ai_mlx_set_memory_limit_bytes(size_t limit_bytes);

/* §P12 (oMLX/SIMD: MEMORY TRAFFIC) — report di TRAFFICO MEMORIA per la generazione (decode).
 * Calcola i byte che il motore DEVE muovere per produrre un token:
 *  - `weights`: somma dei byte di TUTTI i tensori caricati (U32+scales+biases) MENO gli embedding
 *    (di cui in decode si legge solo la riga del token, non l'intera matrice);
 *  - `kv`: KV cache letta = 2 * n_layers * n_kv_heads * head_dim * 2B * `offset` (formula GQA;
 *    per architetture MLA — deepseek_v2/kimi — la KV e' compressa -> il valore e' una STIMA alta).
 * `offset` = cache_offset (contesto corrente). Scrive i byte e il numero di tensori; 0 / -1. */
int llm_ai_mlx_debug_mem_report(llm_ai_mlx_model_handle mh, int offset,
                                long* out_total, long* out_weights, long* out_kv, long* out_tensors);

/* Limite memoria MLX per-modello (MB; 0 = off): sotto pressione MLX evicta/GC e oltre il limite
 * fallisce pulito invece del Jetsam. Configurabile (default off) per evitare regressioni. */
int llm_ai_mlx_model_set_memory_limit_mb(llm_ai_mlx_model_handle handle, int limit_mb);
int llm_ai_mlx_model_memory_limit_mb(llm_ai_mlx_model_handle handle);

/* Matcha mlx_lm: embed+layer in bfloat16 (sincronizza g_bf16_all). 0 ok / -1. */
int llm_ai_mlx_model_set_bfloat16(llm_ai_mlx_model_handle handle, bool enabled);

/* Capienza della KV-cache (contesto) a runtime, per QUALUNQUE modello MLX. n = token (0 = default
 * del modello). È il setter dietro CAP_KV_MAX (kv_set_max) dell'ABI comune. 0 ok / -1. */
int llm_ai_mlx_model_set_kv_max(llm_ai_mlx_model_handle handle, int n);

/* KV quant (E3): quantizza la parte COLD della cache oltre `threshold` token, a `bits` (4|8).
 * Parametro `kvquant` del registro. 0 ok / -1. */
int llm_ai_mlx_model_set_kv_quantize(llm_ai_mlx_model_handle handle, bool enabled, int threshold, int bits);

/* Dtype di calcolo scelto dall'alto (host/BASIC), da chiamare DOPO
 * llm_ai_mlx_load_model: i bit bf16 sono letti solo a forward-time, quindi applicarli dopo
 * il load equivale -- e vince sul default architetturale e su LLM_BF16_DENSE perche' arriva
 * per ultimo. mode = "both" | "embed" | "matmul" | "bf16" | "f16" | "fp16" | "off" ("fp32" =
 * sinonimo di off, "bf16" = sinonimo di both). "f16"/"fp16" = both + float16 come dtype a 16 bit:
 * scelto perche' la GEMV 8-bit affine di mlx (B=1) prende il kernel veloce SOLO se x e le SCALE
 * del checkpoint hanno gli stessi bit -- scale F16 (DeepSeek-R1-distill) o scale BF16
 * (qwen3_5/qwen3.6/3.8, gemma4, GLM) chiedono il flavor accoppiato. MISURATO 2026-09-08 su
 * DeepSeek-R1-32B-8bit: f16 = 1,27x fp32, bf16 = 1,02x fp32 (il bf16 li' e' la coppia peggiore,
 * sotto il fp32). LA REGOLA NON E' SOLO DELLA GEMV B=1: MISURATA ANCHE A M GRANDE (prefill ~9000
 * token di testo reale, braccia alternate nello STESSO processo) vale ancora e di piu' - DeepSeek:
 * f16 27,11 s vs fp32 45,24 s = **1,67x**, mentre bf16 e' 47,76 s (sotto il fp32); Qwen3.6-27B-4bit
 * (scale BF16) bf16 18,85 s vs fp32 26,25 s = 1,39x. I MoE restano l'eccezione da non toccare per
 * analogia: mixtral/qwen3_moe hanno scale F16 e
 * stanno BENEFICIANDO in bf16 (+40-50% misurato, GEMM batched) -> NON estenderla per analogia, e
 * llama resta esclusa (correttezza, task #5). Il denso qwen2 affine 8-bit con scale F16 riceve
 * comunque f16 DI DEFAULT a load-time (llm_mlx_auto_pair_act_dtype): questa chiamata serve a
 * forzare, spegnere o cambiare flavor, non ad accendere. La mode e'
 * lo STATO COMPLETO dei due bit, non una richiesta di accensione: "embed" SPEGNE il bit matmul
 * (e viceversa), quindi chiamarla a caldo su un modello carico riporta il dtype esattamente a
 * quanto chiesto, qualunque fosse lo stato precedente.
 * Ritorna 0 applicato / -1 rifiuto del motore: quant_mode fuori dalla allow-list, oppure
 * model_type=llama (bf16 MISURATO 2026-09-08: output errato in modo silenzioso su
 * llama_attention_block) -- il motivo esatto e' sempre su stderr / -2 argomento non valido
 * (handle NULL o mode sconosciuta; la convalida precede la guardia architetturale).
 * La allow-list affine|none e' obbligatoria: i formati a blocchi (mxfp4/mxfp8/nvfp4) con bf16
 * sono documentati 10x piu' lenti o crashano. */
int llm_ai_mlx_model_set_bf16_mode(llm_ai_mlx_model_handle handle, const char* mode);

/* ============================================================================
 * DFlash2 (Qwen3.8-27B-DFlash2) -- drafter block-diffusion nativo.
 *
 * Runtime handle che esegue il forward del draft DFlash2 (5 layer qwen3 sliding
 * + two-tap grouped dynamic conv + fc di contesto + CandidateSelector lattice)
 * su un checkpoint draft BF16, con embed/lm_head bind-ati dal TARGET
 * (quantizzato affine 8-bit). Reference python esatto:
 * Rapid-MLX-NextGen vllm_mlx/speculative/dflash/drafters/qwen3_dflash.py.
 *
 * Il draft NON usa embed_tokens/lm_head propri: li legge dal modello target
 * caricato (embed sotto `language_model.model.embed_tokens`, lm_head sotto
 * `language_model.lm_head`). La KV-cache ctx del draft (una per layer) e'
 * mantenuta TRA le chiamate a propose (ctx_pos avanza di S a ogni round);
 * reset() la azzera.
 * ============================================================================ */
typedef void* llm_ai_mlx_dflash2_handle;

/* Crea il runtime DFlash2: carica il draft da `draft_dir` via llm_ai_mlx_load_model
 * e aggancia target_handle (gia' caricato) per embed/lm_head. Ritorna NULL su errore
 * (draft non dflash, config invalida, lookup embed/lm_head fallito). Il runtime NON
 * possiede target_handle: il chiamante lo libera dopo llm_ai_mlx_dflash2_free. */
llm_ai_mlx_dflash2_handle llm_ai_mlx_dflash2_create(llm_ai_mlx_model_handle target_handle,
                                                    const char* draft_dir);

/* Rilascia il runtime DFlash2 (draft model + cache ctx + handle pesi embed/lm_head
 * referenziati). Non tocca il target. No-op su handle NULL. */
void llm_ai_mlx_dflash2_free(llm_ai_mlx_dflash2_handle handle);

/* Esegue UN draft_block: block = [last_verified_token] + mask*(block_size-1), embed bf16
 * dal target, fc+hidden_norm sul ctx, i n_layers del draft (che aggiornano la cache ctx
 * per layer), norm finale, logits via lm_head del target sui hidden[:,1:], poi il greedy
 * walk del CandidateSelector. ctx_feature: fp32 host [S * (n_ctx_layers * hidden_size)]
 * (= hidden dei layer target dflash_ctx_layers, concat). Scrive fino a block_size-1 token
 * in out_tokens (cap max_out) e ritorna il numero di draft token prodotti, oppure -1 su
 * errore. ctx_pos interno avanza di S. */
int llm_ai_mlx_dflash2_propose(llm_ai_mlx_dflash2_handle handle,
                               int32_t last_verified_token,
                               const float* ctx_feature, int S,
                               int32_t* out_tokens, int max_out);

/* Azzera la cache ctx del draft e ctx_pos (nuovo round/sequenza). 0 ok / -1. */
int llm_ai_mlx_dflash2_reset(llm_ai_mlx_dflash2_handle handle);

/* Tappa 3 (multi-sessione): seleziona lo slot ctx (sessione) su cui operano propose/reset; gli
 * slot sono allocati on-demand. 0 ok / -1. Il single-session usa sempre lo slot 0. */
int llm_ai_mlx_dflash2_set_slot(llm_ai_mlx_dflash2_handle handle, int slot);

/* DEBUG (validazione golden): esegue il backbone del draft (senza toccare
 * cache/ctx_pos) e copia in out gli hidden per layer + norm finale (fp32 host),
 * ordine layer0..n_layers-1,final. Ogni snap [1,L,hidden]. Ritorna il numero di
 * float scritti, -1 su errore, -(n float necessari) se cap insufficiente. */
int llm_ai_mlx_dflash2_debug_layers(llm_ai_mlx_dflash2_handle handle,
                                    int32_t last_verified_token,
                                    const float* ctx_feature, int S,
                                    float* out, int cap);

/* Generate greedy DFlash2 su una sessione target con il prompt GIA' appended
 * (session_append). Parita' greedy (target authoritative). out_ids <= max_tokens. */
int llm_ai_mlx_dflash2_generate(llm_ai_mlx_dflash2_handle handle,
                                llm_ai_mlx_session_handle sess_handle,
                                int32_t eos_token, int max_tokens,
                                int32_t* out_ids);

/* Tappa 3 DFlash2 MULTI-SESSIONE: N sessioni (stesso target) in round-loop batched (propose
 * per-sessione sugli slot + UN verify N×K per round + emit per-sessione). out_ids row-major con
 * stride `stride` (= max(max_tokens)); out_counts[i] = token emessi per la sessione i. Ritorna 0
 * / -1. NON eseguito/testato: richiede la verifica su modello reale. */
int llm_ai_mlx_dflash2_generate_multi(llm_ai_mlx_dflash2_handle handle,
                                      llm_ai_mlx_session_handle* sessions, int n,
                                      int32_t eos_token, const int* max_tokens, int stride,
                                      int32_t* out_ids, int* out_counts);

/* Core micro-bench model-free (2026-09-13): misura i wrapper delle primitive core
 * contro la chiamata MLX diretta, con input sintetici deterministici. `op` in
 * {"rms_norm","qlinear","sdpa","all"}; iters<1 => 200. Scrive righe TSV in `out`
 * (OP <name> ... raw_ms= llm_ms= ratio= maxdiff=). Ritorna il numero di op misurate. */
int llm_ai_mlx_core_bench(const char* op, int iters, char* out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* LLM_HOST_AI_MLX_H */

/* DEBUG correttezza: hidden fp32 dopo ogni layer per un forward (n token da zero). */
int llm_ai_mlx_debug_all_layers_hidden(llm_ai_mlx_session_handle sess, const int32_t* ids, int n,
                                      float* out, int cap);
int llm_ai_mlx_debug_embed(llm_ai_mlx_session_handle sess, const int32_t* ids, int n, float* out, int cap);
