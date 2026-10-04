/* llm_dsv41_moe_test.c — self-test LOAD-FREE (nessun modello/device).
 *   cc -O2 -o /tmp/llm_dsv41_moe_test native/llmengine_mlx/llm_dsv41_moe_test.c -lm
 * Confronta llm_dsv41_moe.h con i riferimenti (tools/gen_dsv41_moe_reference.py). */
#include <stdio.h>
#include "llm_dsv41_moe.h"
#include "llm_dsv41_moe_ref.h"

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
static int ieq(const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) { printf("  [%d] got=%d exp=%d\n", i, a[i], b[i]); return 0; }
    return 1;
}
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-46s\n", name); \
    else { printf("FAIL %-46s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static int run_gate(int sf, int nrm, float rs, const int* ridx, const float* rw) {
    int idx[4]; float w[4];
    llm_dsv41_gate(REF_GATE_X, REF_GATE_D, REF_GATE_W, REF_GATE_E, REF_GATE_B, 1.0f, sf,
                   REF_GATE_ACT, nrm, rs, idx, w);
    return ieq(idx, ridx, REF_GATE_ACT) && nclose(w, rw, REF_GATE_ACT);
}

int main(void) {
    printf("llm_dsv41_moe — self-test load-free (vs riferimento)\n");

    /* T1 — Gate nei 3 score_func (+ norm/route_scale) */
    CHECK(run_gate(REF_GATE_SF0, REF_GATE_NRM0, REF_GATE_RS0, REF_GATE_IDX0, REF_GATE_WGT0) &&
          run_gate(REF_GATE_SF1, REF_GATE_NRM1, REF_GATE_RS1, REF_GATE_IDX1, REF_GATE_WGT1) &&
          run_gate(REF_GATE_SF2, REF_GATE_NRM2, REF_GATE_RS2, REF_GATE_IDX2, REF_GATE_WGT2),
          "T1 Gate (softmax/sigmoid/sqrtsoftplus + norm/scale)", "");

    /* T2 — Expert SwiGLU (con swiglu_limit) */
    {
        float y[8];
        llm_dsv41_expert_swiglu(REF_SW_GATE, REF_SW_UP, REF_SW_D, REF_SW_LIMIT, REF_SW_WEIGHT, y);
        CHECK(nclose(y, REF_SW_OUT, REF_SW_D), "T2 Expert SwiGLU + swiglu_limit", "");
    }

    /* T3 — MoE combine (somma routed + shared) */
    {
        float out[8];
        llm_dsv41_moe_combine(REF_MOE_ROUTED, REF_MOE_SHARED, REF_MOE_ACT, REF_MOE_D, out);
        CHECK(nclose(out, REF_MOE_OUT, REF_MOE_D), "T3 MoE combine", "");
    }

    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
