/* llm_dsv41_fwd_step2a_test.c — step 2a: mHC di layer 0 (hc_mixes + hc_pre_norm).
 *   cc -O2 -o /tmp/llm_dsv41_fwd_step2a_test native/llmengine_mlx/llm_dsv41_fwd_step2a_test.c -lm
 * Verifica contro llm_dsv41_fwd_step2a_ref.h (generato dal riferimento oMLX tiny). */
#include <stdio.h>
#include "llm_dsv41_math.h"
#include "llm_dsv41_fwd_step2a_ref.h"

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
    const int L = REF_S2_L, hc = REF_S2_HC, dim = REF_S2_DIM, size = REF_S2_SIZE, nD = hc * dim;
    const float norm_eps = 1e-20f, hc_eps = 1e-6f; const int iters = 20;
    printf("llm_dsv41_fwd_step2a — mHC layer0\n");
    float ap[32], ao[32], ac[128], x[64];
    int ok = 1;
    for (int l = 0; l < L; l++) {
        float mixes[64], pre[16], post[16], comb[64], hpre[16], xl[16];
        llm_dsv41_hc_mixes(REF_S2_H + l * nD, nD, REF_S2_HCFN, size, REF_S2_HCSCALE,
                           REF_S2_HCBASE, hc, norm_eps, hc_eps, iters, mixes, pre, post, comb);
        for (int i = 0; i < hc; i++) { ap[l * hc + i] = pre[i]; ao[l * hc + i] = post[i]; }
        for (int i = 0; i < hc * hc; i++) ac[l * hc * hc + i] = comb[i];
        llm_dsv41_hc_pre(REF_S2_H + l * nD, REF_S2_PRE + l * hc, hc, dim, hpre);
        llm_dsv41_rms_norm(hpre, REF_S2_ANORM, dim, norm_eps, xl);
        for (int i = 0; i < dim; i++) x[l * dim + i] = xl[i];
    }
    ok &= nclose(ap, REF_S2_AP, L * hc);
    ok &= nclose(ao, REF_S2_AO, L * hc);
    ok &= nclose(ac, REF_S2_AC, L * hc * hc);
    ok &= nclose(x, REF_S2_X, L * dim);
    CHECK(ok, "T1 mHC hc_mixes (ap/ao/ac) + hc_pre_norm(x)", "");
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
