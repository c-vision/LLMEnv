/* ============================================================================
 * llmtoken_wordpiece.h
 * Tokenizer BERT WordPiece nel runtime C condiviso (Fase 4b del todo
 * "Tokenizer-WordPiece-Unicode-e-Retrieval-Sparse-Learned", parita' C#/C).
 *
 * Porting 1:1 di common/Databases/VectorDB/BertWordPieceTokenizer.cs
 * (BasicTokenizer HF + WordPiece greedy + Encode/EncodePair + troncamento che
 * preserva [CLS]/[SEP]), con i dati Unicode autosufficienti in
 * llmtoken_wp_unicode_data.inc. Comportamento verificato id-per-id contro
 * l'oracolo HF versionato samples/Fixtures/tokenizer/tokenizer_golden.json
 * (test: tests/llmtoken_wordpiece_selfcheck.c).
 *
 * Nestesso uso: vocab path (file testo una riga per token, indice = id, come
 * vocab.txt di BERT) + [CLS]/[SEP]/[UNK] letti dal vocabolario con fallback
 * canonici (101/102/100), identico al Load() C#.
 * ============================================================================ */
#ifndef LLMTOKEN_WORDPIECE_H
#define LLMTOKEN_WORDPIECE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llmtoken_wp_tok llmtoken_wp_tok;

/* Carica il vocabolario WordPiece. NULL su errore (file mancante/illeggibile,
 * vocabolario vuoto). Legge [CLS]/[SEP]/[UNK] dal vocab (fallback 101/102/100). */
llmtoken_wp_tok* llmtoken_wp_tok_load(const char* vocab_path);

void llmtoken_wp_tok_free(llmtoken_wp_tok* t);

int llmtoken_wp_tok_cls(const llmtoken_wp_tok* t);   /* id del [CLS] */
int llmtoken_wp_tok_sep(const llmtoken_wp_tok* t);   /* id del [SEP] */
int llmtoken_wp_tok_unk(const llmtoken_wp_tok* t);   /* id del [UNK] */
int llmtoken_wp_tok_pad(const llmtoken_wp_tok* t);   /* id 0 (padding) */

/* BasicTokenizer HF (uncased, strip_accents): la fase di pre-tokenizzazione.
 * Per ogni token prodotto, chiama cb(token_utf8, user); se cb ritorna != 0 la
 * scansione si interrompe e il valore ritornato viene propagato come conteggio
 * parziale. Ritorna il numero di token emessi (o -1 su errore di input). */
typedef int (*llmtoken_wp_basic_cb)(const char* token_utf8, void* user);
int llmtoken_wp_basic_tokenize(const char* text, llmtoken_wp_basic_cb cb, void* user);

/* WordPiece greedy su un singolo pre-token (gia' normalizzato/abbassato):
 * ritorna il numero di pezzi emessi via cb(piece_id, user) (oppure 1 con il solo
 * id di [UNK] se la parola fallisce o supera 100 codepoint). -1 su errore. */
typedef int (*llmtoken_wp_piece_cb)(int piece_id, void* user);
int llmtoken_wp_wordpiece(const llmtoken_wp_tok* t, const char* token, llmtoken_wp_piece_cb cb, void* user);

/* Encode singola sequenza: [CLS] ... [SEP] paddata a max_len.
 * Riempie input_ids[] e attention_mask[] (lunghezza max_len) e, se
 * token_type_ids non NULL, lo stesso array (tutti 0). Ritorna il numero di token
 * ATTIVI (indice dell'ultimo attivo+1, == max_len se non c'e' padding) o -1 se
 * max_len < 0. Il troncamento preserva [CLS] (testa) e [SEP] (coda), tagliando
 * SOLO il contenuto (difetto D2). */
int llmtoken_wp_encode(const llmtoken_wp_tok* t, const char* text, int max_len,
                  int32_t* input_ids, int32_t* attention_mask, int32_t* token_type_ids);

/* Encode coppia (cross-encoder): [CLS] query [SEP] document [SEP], con
 * token_type_ids 0 sul segmento query (incluso [CLS] e il suo [SEP]) e 1 sul
 * documento (incluso il [SEP] finale). Su overflow si taglia il CONTENUTO
 * (prima il documento, poi la query se non basta: longest-first), mai i marker;
 * i type id sono assegnati DOPO il taglio sul confine reale. Ritorna il numero
 * di token attivi o -1 su errore. */
int llmtoken_wp_encode_pair(const llmtoken_wp_tok* t, const char* query, const char* document, int max_len,
                       int32_t* input_ids, int32_t* attention_mask, int32_t* token_type_ids);

#ifdef __cplusplus
}
#endif

#endif /* LLMTOKEN_WORDPIECE_H */
