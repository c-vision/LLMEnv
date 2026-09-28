/* llm_dsv41_math_test.c — self-test LOAD-FREE (nessun modello/device).
 *   cc -O2 -o /tmp/llm_dsv41_math_test native/llmengine_mlx/llm_dsv41_math_test.c -lm
 * Confronta il port C (llm_dsv41_math.h) con i valori di riferimento MLX
 * (llm_dsv41_math_ref.h, generati da tools/gen_dsv41_math_reference.py). */
#include <stdio.h>
#include <string.h>
#include "llm_dsv41_math.h"
#include "llm_dsv41_math_ref.h"

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
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-46s\n", name); \
    else { printf("FAIL %-46s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

int main(void) {
    const float eps = 1e-6f, hc_eps = 1e-6f, norm_eps = 1e-20f;
    const int iters = 20, n = 4, D = 6, nD = n * D, size = (2 + n) * n;
    printf("llm_dsv41_math — self-test load-free (vs riferimento MLX)\n");

    /* T1 — sinkhorn */
    {
        float comb[16]; memcpy(comb, REF_SINK_COMB, sizeof comb);
        llm_dsv41_sinkhorn(comb, n, eps, iters);
        CHECK(nclose(comb, REF_SINK_OUT, 16), "T1 sinkhorn", "");
    }
    /* T2 — hc_mix_weights */
    {
        float pre[4], post[4], comb[16];
        llm_dsv41_hc_mix_weights(REF_MIX_MIXES, REF_MIX_SCALE, REF_MIX_BASE, n, hc_eps, iters, pre, post, comb);
        CHECK(nclose(pre, REF_MIX_PRE, 4) && nclose(post, REF_MIX_POST, 4) && nclose(comb, REF_MIX_COMB, 16),
              "T2 hc_mix_weights (pre/post/comb)", "");
    }
    /* T3 — hc_mixes */
    {
        float mixes[24], pre[4], post[4], comb[16];
        llm_dsv41_hc_mixes(REF_HCM_X, nD, REF_HCM_FN, size, REF_HCM_SCALE, REF_HCM_BASE, n,
                           norm_eps, hc_eps, iters, mixes, pre, post, comb);
        CHECK(nclose(pre, REF_HCM_PRE, 4) && nclose(post, REF_HCM_POST, 4) && nclose(comb, REF_HCM_COMB, 16),
              "T3 hc_mixes (pre/post/comb)", "");
    }
    /* T4 — hc_pre */
    {
        float out[6];
        llm_dsv41_hc_pre(REF_PRE_X, REF_PRE_P, n, D, out);
        CHECK(nclose(out, REF_PRE_OUT, 6), "T4 hc_pre", "");
    }
    /* T5 — hc_post */
    {
        float out[24];
        llm_dsv41_hc_post(REF_POST_X, D, REF_POST_RES, REF_POST_P, REF_POST_COMB, n, out);
        CHECK(nclose(out, REF_POST_OUT, 24), "T5 hc_post", "");
    }
    /* T6 — Compressor pooling (r=2, H=3, m=2) */
    {
        float out[6];
        llm_dsv41_compressor_pool(REF_CP_KV, REF_CP_GATE, 2, 3, 2, out);
        CHECK(nclose(out, REF_CP_OUT, 6), "T6 compressor pooling", "");
    }
    /* T7 — index score (heads=2, dim=3) */
    {
        float s = llm_dsv41_index_score(REF_IDX_Q, REF_IDX_K, REF_IDX_W, 2, 3);
        CHECK(nclose(&s, REF_IDX_OUT, 1), "T7 indexer score", "got=%.7g", s);
    }

    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
