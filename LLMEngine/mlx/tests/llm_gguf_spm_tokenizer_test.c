/* llm_gguf_spm_tokenizer_test.c — il tokenizer SentencePiece costruito dai KV di un GGUF (`tokenizer.ggml.model = llama`) produce gli
 * stessi id di llama.cpp. Uso: llm_gguf_spm_tokenizer_test <spm.gguf> <cases.tsv>   (default: tools/Fixtures/tokenizer/gguf_spm/)
 * Compilazione: clang -I LLMToken tests/llm_gguf_spm_tokenizer_test.c -L LLMEngine/mlx/build-gmbasic -L build/dist/osx-arm64 \
 *   -lllm_ai_mlx -lllmtoken -Wl,-rpath,<stesse cartelle> -o /tmp/spm_tok_test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "llmtoken.h"
extern char* llm_ai_mlx_gguf_tokenizer_json(const char* path, size_t* out_len);

static void unesc(char* s) {
    char* o = s;
    for (char* p = s; *p; p++) {
        if (*p == '\\' && p[1]) { p++; *o++ = (*p == 'n') ? '\n' : (*p == 't') ? '\t' : *p; } else *o++ = *p;
    }
    *o = 0;
}

int main(int argc, char** argv) {
    const char* gguf = argc > 1 ? argv[1] : "tools/Fixtures/tokenizer/gguf_spm/spm.gguf";
    const char* cases = argc > 2 ? argv[2] : "tools/Fixtures/tokenizer/gguf_spm/cases.tsv";
    size_t jl = 0;
    char* js = llm_ai_mlx_gguf_tokenizer_json(gguf, &jl);
    if (!js) { fprintf(stderr, "FAIL: tokenizzatore non costruibile da %s\n", gguf); return 1; }
    llmtoken_handle h = llmtoken_load_json(js, jl);
    free(js);
    if (!h) { fprintf(stderr, "FAIL: llmtoken_load_json\n"); return 1; }
    FILE* f = fopen(cases, "r");
    if (!f) { fprintf(stderr, "FAIL: %s non apribile\n", cases); return 1; }
    char line[8192];
    int n = 0, bad = 0;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\n")] = 0;
        char* tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        char text[4096];
        snprintf(text, sizeof text, "%s", line);
        unesc(text);
        int32_t ids[1024];
        int k = llmtoken_encode(h, text, ids, 1024);
        char got[8192] = "";
        for (int i = 0; i < k; i++) { char t[16]; snprintf(t, sizeof t, i ? " %d" : "%d", ids[i]); strcat(got, t); }
        n++;
        if (strcmp(got, tab + 1) != 0) { bad++; printf("DIFF  %s\n  got: %s\n  ref: %s\n", line, got, tab + 1); }
    }
    fclose(f);
    printf("llm_gguf_spm_tokenizer_test: %d casi, %d diversi\n", n, bad);
    return bad != 0;
}
