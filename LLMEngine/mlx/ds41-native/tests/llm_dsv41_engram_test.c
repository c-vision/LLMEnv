/* llm_dsv41_engram_test.c — self-test LOAD-FREE (nessun modello/device).
 *   cc -O2 -o /tmp/llm_dsv41_engram_test native/llmengine_mlx/llm_dsv41_engram_test.c -lm
 * Confronta llm_dsv41_engram.h con i riferimenti (tools/gen_dsv41_engram_reference.py). */
#include <stdio.h>
#include <stdint.h>
#include "llm_dsv41_engram.h"
#include "llm_dsv41_engram_ref.h"

static int g_fail = 0;
#define TOL 3e-4f
static int nclose(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > TOL * m) { printf("  [%d] got=%.7g exp=%.7g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}
static int i64eq(const int64_t* a, const int64_t* b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) { printf("  [%d] got=%lld exp=%lld\n", i, (long long)a[i], (long long)b[i]); return 0; }
    return 1;
}
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-46s\n", name); \
    else { printf("FAIL %-46s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

int main(void) {
    const int L = REF_EN_L, depth = REF_EN_DEPTH, nh = REF_EN_NHEADS;
    const int Hn = L * depth * nh;
    printf("llm_dsv41_engram — self-test load-free (vs riferimento)\n");

    int64_t tokens[16], hashes[64], hist[8];
    for (int i = 0; i < L; i++) tokens[i] = REF_EN_TOKEN_MAP[REF_EN_IDS[i]];

    /* T1 — hash senza maschera */
    {
        llm_dsv41_engram_hash(tokens, L, REF_EN_HIST, depth, REF_EN_PAD, REF_EN_MULT,
                              REF_EN_PRIMES, nh, REF_EN_OFFSETS, hashes, hist);
        CHECK(i64eq(hashes, REF_EN_HASH, Hn) && i64eq(hist, REF_EN_HISTOUT, depth),
              "T1 n-gram hash + history", "");
    }
    /* T2 — hash con posizioni immagine (-1): il lookback si blocca */
    {
        int64_t t2[16];
        for (int i = 0; i < L; i++) t2[i] = REF_EN_MASK[i] ? -1 : tokens[i];
        llm_dsv41_engram_hash(t2, L, REF_EN_HIST, depth, REF_EN_PAD, REF_EN_MULT,
                              REF_EN_PRIMES, nh, REF_EN_OFFSETS, hashes, hist);
        CHECK(i64eq(hashes, REF_EN_HASH2, Hn) && i64eq(hist, REF_EN_HISTOUT2, depth),
              "T2 hash con image_mask (-1 blocca lookback)", "");
    }
    /* T3 — Engram gate */
    {
        float out[16];
        llm_dsv41_engram_gate(REF_EN_H, REF_EN_HC, REF_EN_DIM, REF_EN_KEY, REF_EN_VALUE,
                              REF_EN_QW, REF_EN_KW, 1e-20f, out);
        CHECK(nclose(out, REF_EN_GATE_OUT, REF_EN_HC * REF_EN_DIM), "T3 Engram gate + iniezione", "");
    }

    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
