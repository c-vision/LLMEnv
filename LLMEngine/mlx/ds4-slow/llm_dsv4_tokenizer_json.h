/* llm_dsv4_tokenizer_json.h — costruisce la JSON del tokenizer DeepSeek V4 leggendo
 * vocab+merges DIRETTAMENTE dal GGUF (niente tokenizer.json esterno), con il pre-tokenizer
 * JoyAI (Sequence di Split, identico a ds4).
 *
 * Header-only e SENZA dipendenze da llmtoken: cosi' il costruttore della stringa
 * JSON puo' essere incluso da llmengine_mlx.c (che gia' include llm_gguf.h, header-only)
 * ed esportato via ABI. La creazione dell'handle BPE resta in llm_dsv4_tokenizer.h, che
 * include questo header e chiama llmtoken_load_json. */
#ifndef LLM_DSV4_TOKENIZER_JSON_H
#define LLM_DSV4_TOKENIZER_JSON_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "llm_gguf.h"

typedef struct { char* p; size_t len, cap; } LlmDsv4TokBuf;

static inline void llm_dsv4_tok_put(LlmDsv4TokBuf* b, const char* s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        b->cap = (b->cap == 0) ? 4096 : b->cap;
        while (b->len + n + 1 > b->cap) b->cap *= 2;
        b->p = (char*)realloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, s, n); b->len += n; b->p[b->len] = '\0';
}
static inline void llm_dsv4_tok_c(LlmDsv4TokBuf* b, char c) { llm_dsv4_tok_put(b, &c, 1); }
#define LLM_D4PUT(b, s) llm_dsv4_tok_put((b), (s), strlen(s))
static inline void llm_dsv4_tok_jsonstr(LlmDsv4TokBuf* b, const char* s, size_t n) {
    llm_dsv4_tok_c(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { llm_dsv4_tok_c(b, '\\'); llm_dsv4_tok_c(b, (char)c); }
        else if (c < 0x20) { char tmp[8]; int k = snprintf(tmp, sizeof tmp, "\\u%04x", c); llm_dsv4_tok_put(b, tmp, (size_t)k); }
        else llm_dsv4_tok_c(b, (char)c);
    }
    llm_dsv4_tok_c(b, '"');
}

/* Costruisce la JSON del tokenizer DeepSeek V4 (BPE + pre-tokenizer JoyAI) dal GGUF.
 * Ritorna una stringa malloc'ata (caller: free) e ne scrive la lunghezza in *out_len;
 * NULL in caso di errore (o vocab assente). Non richiede llmtoken. */
static inline char* llm_dsv4_tokenizer_json_from_gguf(const LlmEngineMoeGguf* g, size_t* out_len) {
    if (out_len) *out_len = 0;
    const char** toks = NULL; size_t* tlens = NULL;
    long nt = llm_gguf_kv_array_str(g, "tokenizer.ggml.tokens", &toks, &tlens);
    if (nt <= 0) return NULL;
    const char** mrg = NULL; size_t* mlens = NULL;
    long nm = llm_gguf_kv_array_str(g, "tokenizer.ggml.merges", &mrg, &mlens);
    if (nm < 0) nm = 0;

    LlmDsv4TokBuf b; memset(&b, 0, sizeof b);
    LLM_D4PUT(&b, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (long i = 0; i < nt; i++) {
        if (i) llm_dsv4_tok_c(&b, ',');
        llm_dsv4_tok_jsonstr(&b, toks[i], tlens[i]);
        char num[24]; int k = snprintf(num, sizeof num, ":%ld", i);
        llm_dsv4_tok_put(&b, num, (size_t)k);
    }
    LLM_D4PUT(&b, "},\"merges\":[");
    for (long i = 0; i < nm; i++) {
        if (i) llm_dsv4_tok_c(&b, ',');
        llm_dsv4_tok_jsonstr(&b, mrg[i], mlens[i]);
    }
    LLM_D4PUT(&b, "]},");
    /* added_tokens: i token SPECIALI del GGUF (tokenizer.ggml.token_type: 2=UNKNOWN, 3=CONTROL,
     * 4=USER_DEFINED; 1=NORMAL e 6=BYTE esclusi). Indispensabile: senza questi i marcatori di
     * turno (<｜User｜>/<｜Assistant｜>/<｜end▁of▁sentence｜>) verrebbero spezzati dal BPE e non
     * mappati al loro singolo id. llmtoken li tratta come match letterale (i più lunghi
     * prima) sul testo grezzo, prima del BPE. */
    {
        int32_t* tt = NULL;
        long ntt = llm_gguf_kv_array_i32(g, "tokenizer.ggml.token_type", &tt);
        LLM_D4PUT(&b, "\"added_tokens\":[");
        int first = 1;
        if (ntt > 0 && tt != NULL) {
            for (long i = 0; i < nt && i < ntt; i++) {
                int ty = (int)tt[i];
                if ((ty != 2 && ty != 3 && ty != 4) || tlens[i] == 0) continue;
                if (!first) llm_dsv4_tok_c(&b, ',');
                first = 0;
                LLM_D4PUT(&b, "{\"id\":");
                char num[24]; int k = snprintf(num, sizeof num, "%ld", i);
                llm_dsv4_tok_put(&b, num, (size_t)k);
                LLM_D4PUT(&b, ",\"content\":");
                llm_dsv4_tok_jsonstr(&b, toks[i], tlens[i]);
                LLM_D4PUT(&b, ",\"special\":true}");
            }
        }
        free(tt);
        LLM_D4PUT(&b, "],");
    }
    LLM_D4PUT(&b, "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[");
    static const char* splits[3] = {
        "\\p{N}{1,3}",
        "[\\x{4e00}-\\x{9fa5}\\x{3040}-\\x{309f}\\x{30a0}-\\x{30ff}]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    for (int i = 0; i < 3; i++) {
        const char* s = splits[i];
        LLM_D4PUT(&b, "{\"type\":\"Split\",\"behavior\":\"Isolated\",\"invert\":false,\"pattern\":{\"Regex\":");
        llm_dsv4_tok_jsonstr(&b, s, strlen(s));
        LLM_D4PUT(&b, "}},");
    }
    LLM_D4PUT(&b, "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}]}}");

    free(toks); free(tlens); free(mrg); free(mlens);
    if (out_len) *out_len = b.len;
    return b.p;
}

#endif /* LLM_DSV4_TOKENIZER_JSON_H */
