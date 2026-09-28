/* llm_dsv4_tokenizer.h — costruisce il tokenizer DeepSeek V4 leggendo vocab+merges DIRETTAMENTE
 * dal GGUF (niente tokenizer.json esterno). Riusa il BPE di LLMEnv (llmtoken) passando
 * una JSON in memoria con il pre-tokenizer JoyAI (Sequence di Split, identico a ds4).
 * Header-only. Test: tools/llm_dsv4_tokenizer_gguf_test.sh. */
#ifndef LLM_DSV4_TOKENIZER_H
#define LLM_DSV4_TOKENIZER_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "llm_gguf.h"
#include "llmtoken.h"
#include "llm_dsv4_tokenizer_json.h"

/* Costruisce l'handle BPE DeepSeek V4 dal GGUF. La JSON (vocab+merges+pre-tokenizer JoyAI) e'
 * generata da llm_dsv4_tokenizer_json.h, che NON dipende da llmtoken (cosi' il builder
 * puo' essere usato anche da llmengine_mlx.c via ABI); qui la si carica col BPE di LLMEnv. */
static inline llmtoken_handle llm_dsv4_tokenizer_from_gguf(const LlmEngineMoeGguf* g) {
    size_t len = 0;
    char* json = llm_dsv4_tokenizer_json_from_gguf(g, &len);
    if (json == NULL) return NULL;
    llmtoken_handle h = llmtoken_load_json(json, len);
    free(json);
    return h;
}

#endif /* LLM_DSV4_TOKENIZER_H */
