/* llm_dsv41_attn_test.c — self-test LOAD-FREE (nessun modello/device).
 *   cc -O2 -o /tmp/llm_dsv41_attn_test native/llmengine_mlx/llm_dsv41_attn_test.c -lm
 * Confronta llm_dsv41_attn.h con i riferimenti (tools/gen_dsv41_attn_reference.py). */
#include <stdio.h>
#include <string.h>
#include "llm_dsv41_attn.h"
#include "llm_dsv41_attn_ref.h"

static int g_fail = 0;
#define TOL 3e-4f

static int nclose(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        if (b[i] == -INFINITY || a[i] == -INFINITY) { if (a[i] != b[i]) return 0; continue; }
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > TOL * m) { printf("  [%d] got=%.7g exp=%.7g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}
static int ieq(const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) { printf("  [%d] got=%d exp=%d\n", i, a[i], b[i]); return 0; }
    return 1;
}
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-46s\n", name); \
    else { printf("FAIL %-46s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

int main(void) {
    printf("llm_dsv41_attn — self-test load-free (vs riferimento)\n");

    /* T1 — SWA indices (3 casi, esatti) */
    {
        int idx[64]; int ok = 1;
        int W0 = llm_dsv41_swa_indices(0, 5, 3, 0, idx);
        ok = ok && W0 == REF_SWA_W0 && ieq(idx, REF_SWA_IDX0, 5 * W0);
        int W1 = llm_dsv41_swa_indices(4, 2, 3, 2, idx);
        ok = ok && W1 == REF_SWA_W1 && ieq(idx, REF_SWA_IDX1, 2 * W1);
        int W2 = llm_dsv41_swa_indices(10, 3, 4, 4, idx);
        ok = ok && W2 == REF_SWA_W2 && ieq(idx, REF_SWA_IDX2, 3 * W2);
        CHECK(ok, "T1 SWA local window indices", "W=%d,%d,%d", W0, W1, W2);
    }

    /* T2 — Compressor step + cache remainder (2 casi) */
    {
        int H = 3, r = 2, L = 4; float pooled[16], nkv[16], ngate[16]; int cnew = 0;
        int ok = 1;
        int m0 = llm_dsv41_compressor_step(REF_CP_KV_IN0, REF_CP_GATE_IN0, L, H, r, 0,
                                           REF_CP_CACHE_KV0, REF_CP_CACHE_GATE0, REF_CP_CACHE_LEN0,
                                           pooled, nkv, ngate, &cnew);
        ok = ok && m0 == REF_CP_M0 && cnew == REF_CP_CNEW0 &&
             nclose(pooled, REF_CP_POOLED0, m0 * H) && nclose(nkv, REF_CP_NEW_KV0, cnew * H) &&
             nclose(ngate, REF_CP_NEW_GATE0, cnew * H);
        int m1 = llm_dsv41_compressor_step(REF_CP_KV_IN1, REF_CP_GATE_IN1, L, H, r, 1,
                                           REF_CP_CACHE_KV1, REF_CP_CACHE_GATE1, REF_CP_CACHE_LEN1,
                                           pooled, nkv, ngate, &cnew);
        ok = ok && m1 == REF_CP_M1 && cnew == REF_CP_CNEW1 &&
             nclose(pooled, REF_CP_POOLED1, m1 * H) && nclose(nkv, REF_CP_NEW_KV1, cnew * H) &&
             nclose(ngate, REF_CP_NEW_GATE1, cnew * H);
        CHECK(ok, "T2 Compressor step + cache remainder", "m=%d,%d cnew=%d,%d", m0, m1, REF_CP_CNEW0, REF_CP_CNEW1);
    }

    /* T3 — Index top-k */
    {
        int idx[8];
        int n = llm_dsv41_index_topk(REF_IT_SCORES, REF_IT_N, REF_IT_K, idx);
        CHECK(n == 3 && ieq(idx, REF_IT_IDX, REF_IT_K), "T3 index top-k", "n=%d", n);
    }

    /* T4b — Candidate block pre-filtering */
    {
        int out[8];
        int n = llm_dsv41_candidate_block_ids(REF_CB_SCORES, REF_CB_W, REF_CB_LEN, REF_CB_CNT, REF_CB_BLK, out);
        CHECK(n == REF_CB_CNT && ieq(out, REF_CB_OUT, REF_CB_CNT), "T4b candidate block ids", "n=%d", n);
    }

    /* T4 — Sparse attention con sink */
    {
        float out[8];
        int heads = 2, dim = 3, M = 4;
        llm_dsv41_sdpa_sink(REF_SD_Q, REF_SD_KEYS, REF_SD_IDX, M, heads, dim, REF_SD_SINK, REF_SD_SCALE[0], out);
        CHECK(nclose(out, REF_SD_OUT, heads * dim), "T4 sparse attention + sink", "");
    }

    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}