/* llmtoken_tiktoken_test.c — verifica il loader TIKTOKEN nativo (kimi_linear)
 * contro gli id GOLDEN prodotti dal riferimento `tiktoken` (vedi
 * tools/llmtoken_kimi_tiktoken_probe.py). Copre ASCII, accenti, Han/CJK, cirillico, punteggiatura
 * e numeri. Fa anche round-trip encode->decode.
 * Uso: llmtoken_tiktoken_test [modello_dir] */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "llmtoken.h"

static const struct { const char* text; int n; int ids[12]; } cases[] = {
    { "Hello, my name is",                 5, { 19180, 11, 1106, 1530, 387 } },
    { "La capitale d'Italia \xC3\xA8 Roma.", 10, { 13724, 118404, 1861, 304, 6, 1744, 37187, 27002, 65203, 13 } },
    { "\xE6\xB1\x89\xE5\xAD\x97\xE6\xB5\x8B\xE8\xAF\x95 123", 4, { 38325, 9555, 220, 6694 } },
    { "2+2=4",                             5, { 17, 10, 17, 28, 19 } },
    { "Once upon a time",                  4, { 13426, 7625, 261, 1169 } },
    { "We're testing it's fine, they'll go.", 8, { 51104, 10298, 4643, 8201, 11, 55481, 970, 13 } },
    { "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 \xD0\xBC\xD0\xB8\xD1\x80", 4, { 159025, 63694, 5993, 17898 } },
    { "1234567890 42 1000000",            10, { 6694, 12972, 16242, 15, 220, 5512, 220, 1570, 1476, 15 } },
    { "a",                                 1, { 64 } },
};

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "/Users/gaetano/ai/Models/Kimi-Linear-48B-A3B-Instruct-4bit";
    char model[1024], cfg[1024];
    snprintf(model, sizeof model, "%s/tiktoken.model", dir);
    snprintf(cfg, sizeof cfg, "%s/tokenizer_config.json", dir);

    llmtoken_handle h = llmtoken_load_tiktoken(model, cfg);
    if (!h) { fprintf(stderr, "SKIP: tiktoken non caricabile (%s)\n", model); return 0; }

    int fail = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int32_t ids[64];
        int n = llmtoken_encode(h, cases[i].text, ids, 64);
        int ok = (n == cases[i].n);
        for (int j = 0; ok && j < n; j++) if (ids[j] != cases[i].ids[j]) ok = 0;
        /* round-trip: decode(encode(t)) == t */
        char back[512];
        int bn = ok ? llmtoken_decode(h, ids, n, back, sizeof back) : -1;
        if (ok && (bn < 0 || strcmp(back, cases[i].text) != 0)) ok = 0;
        printf("%s %-34s ->", ok ? "PASS" : "FAIL", cases[i].text);
        for (int j = 0; j < n; j++) printf(" %d", ids[j]);
        printf("\n");
        if (!ok) fail++;
    }
    llmtoken_free(h);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail ? 1 : 0;
}
