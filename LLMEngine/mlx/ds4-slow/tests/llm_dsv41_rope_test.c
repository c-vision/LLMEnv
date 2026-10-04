/* llm_dsv41_rope_test.c — self-test LOAD-FREE (nessun modello/device).
 *   cc -O2 -o /tmp/llm_dsv41_rope_test native/llmengine_mlx/llm_dsv41_rope_test.c -lm */
#include <stdio.h>
#include <string.h>
#include "llm_dsv41_rope.h"
#include "llm_dsv41_rope_ref.h"

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
    printf("llm_dsv41_rope — self-test load-free (vs riferimento)\n");
    int ok = 1;
    for (int ci = 0; ci < REF_RP_NCASE; ci++) {
        float x[16]; memcpy(x, REF_RP_X, REF_RP_HEADDIM * sizeof(float));
        float pos = 0, base = 0, bf = 0, bs = 0, fac = 0;
        int orig = 0, comp = 0, inv = 0;
        const float* rexp = NULL;
        switch (ci) {
            case 0: pos = REF_RP_POS0; base = REF_RP_BASE0; orig = REF_RP_ORIG0; bf = REF_RP_BF0;
                    bs = REF_RP_BS0; fac = REF_RP_FAC0; comp = REF_RP_COMP0; inv = REF_RP_INV0; rexp = REF_RP_OUT0; break;
            case 1: pos = REF_RP_POS1; base = REF_RP_BASE1; orig = REF_RP_ORIG1; bf = REF_RP_BF1;
                    bs = REF_RP_BS1; fac = REF_RP_FAC1; comp = REF_RP_COMP1; inv = REF_RP_INV1; rexp = REF_RP_OUT1; break;
            default: pos = REF_RP_POS2; base = REF_RP_BASE2; orig = REF_RP_ORIG2; bf = REF_RP_BF2;
                    bs = REF_RP_BS2; fac = REF_RP_FAC2; comp = REF_RP_COMP2; inv = REF_RP_INV2; rexp = REF_RP_OUT2; break;
        }
        llm_dsv41_rope(x, REF_RP_HEADDIM, pos, REF_RP_D, base, orig, bf, bs, fac, comp, inv);
        char name[48]; snprintf(name, sizeof name, "T%d rope case %d", ci + 1, ci);
        if (!nclose(x, rexp, REF_RP_HEADDIM)) { ok = 0; CHECK(0, name, "case %d", ci); }
        else CHECK(1, name, "");
    }
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    (void)ok;
    return g_fail == 0 ? 0 : 1;
}
