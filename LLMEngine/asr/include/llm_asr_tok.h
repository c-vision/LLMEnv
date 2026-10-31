/* llm_asr_tok.h — tokenizer Qwen byte-level BPE (vocab.json + merges.txt) per Qwen3-ASR. */
#ifndef LLM_ASR_TOK_H
#define LLM_ASR_TOK_H
#include <stddef.h>
typedef struct LlmAsrTok LlmAsrTok;
LlmAsrTok* llm_asr_tok_load(const char* model_dir);   /* vocab.json + merges.txt (dir HF) */
void llm_asr_tok_free(LlmAsrTok* t);
/* decode: id -> stringa UTF-8 (buffer statico, valido fino alla call successiva). */
const char* llm_asr_tok_decode(LlmAsrTok* t, int id);
/* encode: testo UTF-8 -> id (malloc, free col chiamante); *out_n = #id. NULL su errore. */
int* llm_asr_tok_encode(LlmAsrTok* t, const char* utf8, int* out_n);
int  llm_asr_tok_n_vocab(const LlmAsrTok* t);
#endif
