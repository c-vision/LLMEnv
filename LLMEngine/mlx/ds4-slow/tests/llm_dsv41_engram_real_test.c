/* llm_dsv41_engram_real_test.c — verifica Engram REALE end-to-end (senza pesi): legge il
 * token_map dal file (int32) del modello reale, calcola primes/offsets (scansione C) e multipliers
 * (PCG64 C) per il layer 1, esegue `llm_dsv41_engram_hash` e confronta con il riferimento NumPy.
 *   cc -O2 -o /tmp/llm_dsv41_engram_real_test native/llmengine_mlx/llm_dsv41_engram_real_test.c -lm
 * Uso: llm_dsv41_engram_real_test [path/engram_token_map.bin] */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "llm_dsv41_engram_params.h"
#include "llm_dsv41_engram.h"
#include "llm_dsv41_engram_real_ref.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/Users/gaetano/ai/Models/DeepSeek-V4.1-Flash/engram_token_map.bin";
    printf("llm_dsv41_engram_real — Engram reale (token_map file + primes + PCG64 + hash)\n");
    int fail = 0;

    /* token_map: int32 LE */
    FILE* f = fopen(path, "rb");
    if (!f) { printf("SKIP: %s assente\n", path); return 0; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    int n32 = (int)(sz / 4);
    int32_t* raw = (int32_t*)malloc((size_t)n32 * 4);
    size_t rd = fread(raw, 4, (size_t)n32, f); fclose(f);
    int64_t* map = (int64_t*)malloc((size_t)n32 * 8);
    for (int i = 0; i < n32; i++) map[i] = raw[i];
    free(raw);
    if ((int)rd != n32 || n32 != REF_ENR_MAP_N) {
        printf("FAIL token_map size got=%d exp=%d\n", n32, REF_ENR_MAP_N); return 1;
    }

    /* tokens = map[ids] */
    int64_t tokens[REF_ENR_L];
    for (int i = 0; i < REF_ENR_L; i++) tokens[i] = map[REF_ENR_IDS[i]];
    int mok = 1;
    for (int i = 0; i < REF_ENR_L; i++) if (tokens[i] != REF_ENR_TOKENS[i]) mok = 0;
    if (mok) printf("PASS %-46s\n", "token_map -> tokens");
    else { printf("FAIL %-46s\n", "token_map -> tokens"); fail = 1; }

    /* primes/offsets layer 1 (scansione C) + multipliers (PCG64 C) */
    int64_t seen[256]; int seen_n = 0;
    int64_t primes[64], offsets[64], mult[8];
    llm_dsv41_engram_scan(REF_ENR_VOCAB, REF_ENR_DEPTH, REF_ENR_NHEADS, seen, &seen_n, 256, primes, offsets);
    llm_dsv41_engram_multipliers(REF_ENR_COMP, 1, REF_ENR_DEPTH + 1, mult);
    int pok = 1, mok2 = 1;
    for (int i = 0; i < REF_ENR_NCOLS; i++) { if (primes[i] != REF_ENR_PRIMES[i]) pok = 0; if (offsets[i] != REF_ENR_OFFSETS[i]) pok = 0; }
    for (int i = 0; i <= REF_ENR_DEPTH; i++) if (mult[i] != REF_ENR_MULT[i]) mok2 = 0;
    if (pok) printf("PASS %-46s\n", "primes/offsets layer 1 (C)");
    else { printf("FAIL %-46s\n", "primes/offsets layer 1 (C)"); fail = 1; }
    if (mok2) printf("PASS %-46s\n", "multipliers layer 1 (PCG64 C)");
    else { printf("FAIL %-46s\n", "multipliers layer 1 (PCG64 C)"); fail = 1; }

    /* hash reale */
    int64_t hist_in[8]; for (int i = 0; i < REF_ENR_DEPTH; i++) hist_in[i] = -1;
    int64_t hashes[REF_ENR_L * REF_ENR_NCOLS], hist_out[8];
    llm_dsv41_engram_hash(tokens, REF_ENR_L, hist_in, REF_ENR_DEPTH, REF_ENR_PAD,
                          mult, primes, REF_ENR_NHEADS, offsets, hashes, hist_out);
    int hok = 1;
    for (int i = 0; i < REF_ENR_L * REF_ENR_NCOLS; i++) if (hashes[i] != REF_ENR_HASHES[i]) hok = 0;
    if (hok) printf("PASS %-46s\n", "hashes Engram reali (layer 1)");
    else { printf("FAIL %-46s\n", "hashes Engram reali (layer 1)"); fail = 1; }
    int htok = 1;
    for (int i = 0; i < REF_ENR_DEPTH; i++) if (hist_out[i] != REF_ENR_HIST[i]) htok = 0;
    if (htok) printf("PASS %-46s\n", "history Engram reali");
    else { printf("FAIL %-46s\n", "history Engram reali"); fail = 1; }

    free(map);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
