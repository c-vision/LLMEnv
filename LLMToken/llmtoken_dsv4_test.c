/* llmtoken_dsv4_test.c — verifica il tokenizer DeepSeek V4 (pre-tokenizer Sequence:
 * 3 Split + ByteLevel) contro gli id del riferimento ds4 (--raw --dump-tokens). */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "llmtoken.h"
static const struct { const char* text; int n; int ids[8]; } cases[] = {
    { "The capital of France is", 5, { 671, 6102, 294, 8760, 344 } },
    { "Once upon a time",         4, { 16600, 4465, 260, 1014 } },
    { "2+2=",                     4, { 20, 13, 20, 31 } },
    { "The sky is",               3, { 671, 12709, 344 } },
    { "2+3=",                     4, { 20, 13, 21, 31 } },
    { "Hello, my name is",        5, { 19923, 14, 1026, 2329, 344 } },
};
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ai/Models/DeepSeek-V4.1-Flash/tokenizer.json";
    llmtoken_handle h = llmtoken_load(path);
    if (!h) { fprintf(stderr, "SKIP: tokenizer non trovato (%s)\n", path); return 0; }
    int fail = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int32_t ids[32];
        int n = llmtoken_encode(h, cases[i].text, ids, 32);
        int ok = (n == cases[i].n);
        for (int j = 0; ok && j < n; j++) if (ids[j] != cases[i].ids[j]) ok = 0;
        printf("%s %-24s ->", ok ? "PASS" : "FAIL", cases[i].text);
        for (int j = 0; j < n; j++) printf(" %d", ids[j]);
        printf("\n");
        if (!ok) fail++;
    }
    llmtoken_free(h);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail ? 1 : 0;
}
