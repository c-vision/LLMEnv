/* ============================================================================
 * llmtoken_sp_selfcheck.c
 * Conformance del loader SentencePiece nativo (llmtoken_sp.c): carica un
 * `tokenizer.model`, codifica ogni testo passato e stampa "text<TAB>id,id,...".
 * Si confronta con l'output di riferimento di sentencepiece (o dei golden id).
 *
 * Uso: llmtoken_sp_selfcheck <tokenizer.model> [testo ...]
 * (senza testi usa un set di default con accenti, emoji e casi misti).
 * ============================================================================ */
#include "llmtoken.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "uso: %s <tokenizer.model> [testo ...]\n", argv[0]);
        return 2;
    }
    llmtoken_handle h = llmtoken_load_sp(argv[1]);
    if (!h) {
        fprintf(stderr, "llmtoken_load_sp('%s') = NULL\n", argv[1]);
        return 1;
    }
    static const char* defs[] = {
        "The capital of Italy is",
        "Hello, world!",
        "citta' piu' bella",
        "emoji \xF0\x9F\x98\x80 test",
    };
    const char** texts = (const char**)defs;
    int n = (int)(sizeof defs / sizeof defs[0]);
    if (argc > 2) { texts = (const char**)(argv + 2); n = argc - 2; }

    int32_t ids[4096];
    for (int t = 0; t < n; t++) {
        int m = llmtoken_encode(h, texts[t], ids, 4096);
        printf("%s\t", texts[t]);
        if (m < 0) { printf("<ERRORE>\n"); continue; }
        for (int i = 0; i < m; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("\n");
    }
    llmtoken_free(h);
    return 0;
}
