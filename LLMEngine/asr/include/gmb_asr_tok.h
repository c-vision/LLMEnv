/* gmb_asr_tok.h — tokenizer Qwen byte-level BPE (vocab.json + merges.txt) per Qwen3-ASR. */
#ifndef GMB_ASR_TOK_H
#define GMB_ASR_TOK_H
#include <stddef.h>
typedef struct GmbAsrTok GmbAsrTok;
GmbAsrTok* gmb_asr_tok_load(const char* model_dir);   /* vocab.json + merges.txt (dir HF) */
void gmb_asr_tok_free(GmbAsrTok* t);
/* decode: id -> stringa UTF-8 (buffer statico, valido fino alla call successiva). */
const char* gmb_asr_tok_decode(GmbAsrTok* t, int id);
/* encode: testo UTF-8 -> id (malloc, free col chiamante); *out_n = #id. NULL su errore. */
int* gmb_asr_tok_encode(GmbAsrTok* t, const char* utf8, int* out_n);
int  gmb_asr_tok_n_vocab(const GmbAsrTok* t);
#endif
