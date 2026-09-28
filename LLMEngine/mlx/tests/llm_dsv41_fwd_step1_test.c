/* llm_dsv41_fwd_step1_test.c — step 1 del forward V4.1 C: embed + repeat(hc) + pre.
 *   cc -O2 -o /tmp/llm_dsv41_fwd_step1_test native/llmengine_mlx/llm_dsv41_fwd_step1_test.c -lm
 * Confronta con llm_dsv41_fwd_step1_ref.h (tools/gen_dsv41_fwd_step1_reference.py). */
#include <stdio.h>
#include "llm_dsv41_fwd_step1_ref.h"

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

/* Step 1: h[l,c,d] = embed[ids[l]*dim + d]; pre[l,c] = (c==0) */
static void fwd_step1(float* h, float* pre) {
    for (int l = 0; l < REF_S1_L; l++)
        for (int c = 0; c < REF_S1_HC; c++)
            for (int d = 0; d < REF_S1_DIM; d++)
                h[(l * REF_S1_HC + c) * REF_S1_DIM + d] = REF_S1_EMBED[REF_S1_IDS[l] * REF_S1_DIM + d];
    for (int l = 0; l < REF_S1_L; l++)
        for (int c = 0; c < REF_S1_HC; c++)
            pre[l * REF_S1_HC + c] = (c == 0) ? 1.0f : 0.0f;
}

int main(void) {
    printf("llm_dsv41_fwd_step1 — embed + repeat + pre\n");
    float h[128], pre[32];
    fwd_step1(h, pre);
    CHECK(nclose(h, REF_S1_H, REF_S1_L * REF_S1_HC * REF_S1_DIM), "T1 embed + repeat(hc_mult)", "");
    CHECK(nclose(pre, REF_S1_PRE, REF_S1_L * REF_S1_HC), "T2 pre (broadcast arange==0)", "");
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
