/* ============================================================================
 * llmtoken.h
 * Tokenizer BPE byte-level scritto a mano (FASE 1.2 del piano LLM nativo, vedi
 * Fase1-MLX-Qwen3MoE-Plan.md) -- compatibile col formato tokenizer.json di
 * HuggingFace "fast" tokenizers. Nessuna dipendenza da llama.cpp per la
 * tokenizzazione -- decisione esplicita dell'utente (2026-08-17): l'integrazione
 * LLM deve essere nativa e completa, non delegata a tool esterni.
 *
 * Due FAMIGLIE di encoder, decise da cosa contiene il vocabolario (non dal nome
 * del modello, che nel file non c'e'):
 *  - byte-level (schema GPT-2/GPT-4: Llama-3, Qwen, GLM): ogni byte del testo e'
 *    trasposto nel codepoint "mirrorato" corrispondente, lo spazio e' "Ġ", e lo
 *    split di pre-tokenizzazione viene da una regex Unicode (PCRE2).
 *  - metaspace/SentencePiece (Llama-1/2, Mistral/Mixtral): nessun byte-level e
 *    nessuna regex -- il confine di parola e' il marcatore dichiarato dal
 *    normalizer ("▁"), i simboli iniziali sono caratteri UTF-8, e un carattere
 *    fuori vocabolario diventa la sequenza dei suoi byte scritti <0xHH> quando il
 *    file dichiara model.byte_fallback.
 * Entrambe validate id per id contro `tokenizers` 0.22.2 di HF (la libreria che
 * mlx_lm/rapidMLX usano sugli stessi checkpoint): Mixtral 9/9 casi, GLM 9/9,
 * Qwen3-Coder 9/9, con testo italiano accentato, emoji e newline.
 * ============================================================================ */
#ifndef LLMTOKEN_H
#define LLMTOKEN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* llmtoken_handle;

/* Carica un tokenizer.json reale (vocab + merges + added_tokens + pattern di
 * pre-tokenizzazione, letti dal file, nessun default hardcodato). Ritorna
 * NULL su qualunque errore (file mancante, JSON malformato, schema non
 * riconosciuto -- es. modello diverso da BPE). `pre_tokenizer` assente o null
 * NON e' un errore se il vocabolario e' metaspace (un SentencePiece il pattern
 * di split non ce l'ha, il confine di parola e' il marcatore del normalizer);
 * per un vocabolario byte-level resta un errore, perche' lo split mancherebbe. */
llmtoken_handle llmtoken_load(const char* tokenizer_json_path);
/* Variante da buffer JSON in memoria (stesso schema del file tokenizer.json). */
llmtoken_handle llmtoken_load_json(const char* json, size_t len);

/* Carica un tokenizer TIKTOKEN (NON un tokenizer.json): `tiktoken_model_path` e' il file
 * `tiktoken.model` (una riga per token: "base64(token_bytes) SP rank") e
 * `tokenizer_config_json_path` e' il `tokenizer_config.json` del modello, letto per gli special
 * token in `added_tokens_decoder`. Usato dai modelli Kimi (es. kimi_linear, Kimi-Linear-48B) che
 * NON hanno un tokenizer.json BPE HF ma un vocabolario tiktoken. Algoritmo/pattern verificati
 * byte-exact contro il riferimento `tiktoken` (vedi tools/llmtoken_kimi_tiktoken_probe.py e
 * Todo/(open) - Kimi-Tiktoken-Tokenizer-Nativo.md). Ritorna NULL su errore. */
llmtoken_handle llmtoken_load_tiktoken(const char* tiktoken_model_path, const char* tokenizer_config_json_path);

/* Carica un tokenizer SentencePiece DIRETTAMENTE dal file `tokenizer.model`
 * (protobuf ModelProto di sentencepiece, modello BPE con byte_fallback). Nessun
 * Python e nessun `tokenizer.json`: il loader deriva vocab+merges dal protobuf e
 * costruisce internamente lo schema JSON del percorso BPE-metaspace gia' validato.
 * Usato dai modelli che hanno SOLO il tokenizer slow (es. ERNIE-4.5). Ritorna NULL
 * su errore (file assente/malformato, modello non-BPE). Vedi llmtoken_sp.c. */
llmtoken_handle llmtoken_load_sp(const char* model_path);

void llmtoken_free(llmtoken_handle handle);

/* Export per i METADATA GGUF (tokenizer.ggml.tokens/merges/model). Unica sorgente
 * di tokenizer.json: il converter NON ri-parsa il file. I puntatori restituiti
 * sono posseduti dall'handle (validi fino a llmtoken_free) e NON vanno
 * liberati dal chiamante. `tokens` e' denso su [0, n_tokens) coi buchi degli
 * added_tokens gia' riempiti. Ritornano NULL se l'handle e' invalido. */
const char* const* llmtoken_export_tokens(llmtoken_handle handle, size_t* out_n);
const char* const* llmtoken_export_merges(llmtoken_handle handle, size_t* out_n);
/* id -> tipo token GGUF (`tokenizer.ggml.token_type`): 1=NORMAL, 3=CONTROL, 4=USER_DEFINED.
 * Puntatore posseduto dall'handle (valido fino a llmtoken_free). */
const int32_t* llmtoken_export_token_types(llmtoken_handle handle, size_t* out_n);
/* "gpt2" (byte-level) | "llama" (metaspace/SentencePiece) per tokenizer.ggml.model. */
const char* llmtoken_gguf_model(llmtoken_handle handle);

/* Codifica `text` (UTF-8, NUL-terminated) in id di token: pre-scan dei token
 * speciali (match esatto, i più lunghi prima) sul testo grezzo -> per ogni
 * segmento non speciale, split via il pattern regex Unicode-aware del
 * tokenizer (PCRE2, non un'approssimazione ASCII-only) -> per ogni pezzo,
 * mappa byte->unicode e merge BPE iterativo per rango (stesso algoritmo di
 * GPT-2/GPT-4). Nel modo metaspace il segmento non viene spezzato (la regex non
 * esiste): viene normalizzato coi passi Prepend/Replace del file e spezzato in
 * CARATTERI UTF-8, poi mergato allo stesso modo. NOTA: non applica la
 * normalizzazione NFC prevista dal tokenizer -- no-op corretto per testo già in
 * forma NFC (ASCII e la stragrande maggioranza del testo "normale"), gap noto e
 * documentato per sequenze Unicode già decomposte (raro in pratica). Scrive fino
 * a max_ids id in out_ids; ritorna il conteggio REALE prodotto (se > max_ids,
 * richiamare con un buffer più grande -- stessa convenzione "query size" di
 * llama_state_seq_get_size). out_ids può essere NULL con max_ids=0 solo per
 * interrogare il conteggio. Ritorna -1 su errore (handle invalido, lookup
 * nel vocabolario fallito -- non dovrebbe mai accadere per un tokenizer
 * byte-level ben caricato, trattato come bug reale se succede). */
int llmtoken_encode(llmtoken_handle handle, const char* text, int32_t* out_ids, int max_ids);

/* Come llmtoken_encode ma con `flags`. LLMTOKEN_ENCODE_NO_SPECIAL: i token aggiunti/speciali NON sono riconosciuti nel testo (testo ordinario). */
#define LLMTOKEN_ENCODE_NO_SPECIAL 1
int llmtoken_encode_ex(llmtoken_handle handle, const char* text, int32_t* out_ids, int max_ids, int flags);

/* ANTI-INJECTION. Il testo dell'utente, dei tool e dei risultati NON deve poter produrre token di controllo (es. `<|im_end|>` scritto a mano).
 * `llmtoken_guard_text` racchiude fra guardie (U+E000 ... U+E001) ogni occorrenza di un token speciale/marcatore nella stringa non fidata e
 * rimuove le guardie gia' presenti nell'input; `llmtoken_encode_guarded` codifica il testo del prompt gia' renderizzato: gli span fra guardie
 * diventano testo ORDINARIO, il resto (i token di controllo del template) resta com'e'. Il risultato di guard_text va liberato con free(). */
char* llmtoken_guard_text(llmtoken_handle handle, const char* in);
int llmtoken_encode_guarded(llmtoken_handle handle, const char* text, int32_t* out_ids, int max_ids);

/* Decodifica `n_ids` id di token in testo UTF-8: per ogni id, se è un token
 * speciale ne concatena il contenuto letterale, altrimenti decodifica la
 * stringa byte-unicode del vocabolario codepoint per codepoint e la
 * rimappa ai byte originali (inverso esatto della mappa byte->unicode usata
 * in encode). Nel modo metaspace le stringhe sono già UTF-8 reale: il marcatore
 * di parola torna spazio e <0xHH> torna il byte che rappresenta; in coda viene
 * tolto UN spazio iniziale (lo Strip del file, definito sulla sequenza intera --
 * chiamare quindi con la sequenza completa, non col token singolo).
 * Scrive fino a max_bytes byte (incluso il NUL terminatore) in
 * out_text; ritorna la lunghezza REALE necessaria (esclusa NUL), stessa
 * convenzione "query size" di encode. Ritorna -1 su errore. */
int llmtoken_decode(llmtoken_handle handle, const int32_t* ids, int n_ids, char* out_text, int max_bytes);
int llmtoken_decode_ex(llmtoken_handle handle, const int32_t* ids, int n_ids, char* out_text, int max_bytes, int strip_leading_metaspace);

#ifdef __cplusplus
}
#endif

#endif /* LLMTOKEN_H */
