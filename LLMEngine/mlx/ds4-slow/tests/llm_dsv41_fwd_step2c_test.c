/* llm_dsv41_fwd_step2c_test.c — step 2c: MoE layer 0 + hc_post -> output Block 0.
 *   cc -O2 -o /tmp/llm_dsv41_fwd_step2c_test native/llmengine_mlx/llm_dsv41_fwd_step2c_test.c -lm
 * Verifica contro llm_dsv41_fwd_step2c_ref.h. */
#include <stdio.h>
#include <math.h>
#include "llm_dsv41_math.h"
#include "llm_dsv41_moe.h"
#include "llm_dsv41_fwd_step2c_ref.h"

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
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-50s\n", name); \
    else { printf("FAIL %-50s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static void matvec(const float* w, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; r++) { float s = 0; for (int c = 0; c < cols; c++) s += w[r * cols + c] * x[c]; y[r] = s; }
}
static float silu(float g) { return g / (1.0f + expf(-g)); }

/* esperto routed (SwitchLinear): idx e' l'esperto; out = w2 @ (silu(w1@x)*w3@x * weight) */
static void expert(const float* w1, const float* w3, const float* w2, int e, int DI, int DIM,
                   const float* x, float weight, float* out) {
    float g[64], u[64], y[64];
    matvec(w1 + (size_t)e * DI * DIM, DI, DIM, x, g);
    matvec(w3 + (size_t)e * DI * DIM, DI, DIM, x, u);
    for (int i = 0; i < DI; i++) y[i] = silu(g[i]) * u[i] * weight;
    matvec(w2 + (size_t)e * DIM * DI, DIM, DI, y, out);
}

int main(void) {
    const int L = REF_S2C_L, DIM = REF_S2C_DIM, HC = REF_S2C_HC, SIZE = REF_S2C_SIZE;
    const int E = REF_S2C_E, ACT = REF_S2C_ACT, DI = REF_S2C_DI;
    const float norm_eps = 1e-20f, hc_eps = 1e-6f; const int iters = 20;
    printf("llm_dsv41_fwd_step2c — MoE + hc_post (Block 0)\n");
    float ffn_in[64], fp[32], fo[32], fc[128], ffn_out[64], h1[128];
    int ok = 1;
    for (int l = 0; l < L; l++) {
        const float* ha = REF_S2C_HATTN + l * HC * DIM;
        float mixes[64], pr[16], po[16], cb[64], hin[16], x[16];
        llm_dsv41_hc_mixes(ha, HC * DIM, REF_S2C_FFNFN, SIZE, REF_S2C_FFNSCALE, REF_S2C_FFNBASE,
                           HC, norm_eps, hc_eps, iters, mixes, pr, po, cb);
        for (int i = 0; i < HC; i++) { fp[l * HC + i] = pr[i]; fo[l * HC + i] = po[i]; }
        for (int i = 0; i < HC * HC; i++) fc[l * HC * HC + i] = cb[i];
        llm_dsv41_hc_pre(ha, REF_S2C_AP + l * HC, HC, DIM, hin);
        llm_dsv41_rms_norm(hin, REF_S2C_FFNNORM, DIM, norm_eps, x);
        for (int i = 0; i < DIM; i++) ffn_in[l * DIM + i] = x[i];
        /* MoE */
        int idx[8]; float wt[8];
        llm_dsv41_gate(x, DIM, REF_S2C_GW, E, REF_S2C_GB, 1.0f, LLM_DSV41_SCORE_SQRTSOFTPLUS,
                       ACT, 1, 1.0f, idx, wt);
        float acc[16] = {0};
        for (int k = 0; k < ACT; k++) {
            float de[16];
            expert(REF_S2C_E_W1, REF_S2C_E_W3, REF_S2C_E_W2, idx[k], DI, DIM, x, wt[k], de);
            for (int i = 0; i < DIM; i++) acc[i] += de[i];
        }
        /* shared */
        float sg[64], su[64], sy[64], so[16];
        matvec(REF_S2C_S_W1, DI, DIM, x, sg);
        matvec(REF_S2C_S_W3, DI, DIM, x, su);
        for (int i = 0; i < DI; i++) sy[i] = silu(sg[i]) * su[i];
        matvec(REF_S2C_S_W2, DIM, DI, sy, so);
        for (int i = 0; i < DIM; i++) { acc[i] += so[i]; ffn_out[l * DIM + i] = acc[i]; }
        /* h1 = hc_post(ffn_out, h_attn, fo, fc) */
        llm_dsv41_hc_post(acc, DIM, ha, REF_S2C_FO + l * HC, REF_S2C_FC + l * HC * HC, HC, h1 + l * HC * DIM);
    }
    ok &= nclose(fp, REF_S2C_FP, L * HC);
    ok &= nclose(fo, REF_S2C_FO, L * HC);
    ok &= nclose(fc, REF_S2C_FC, L * HC * HC);
    ok &= nclose(ffn_in, REF_S2C_FFNIN, L * DIM);
    ok &= nclose(ffn_out, REF_S2C_FFNOUT, L * DIM);
    ok &= nclose(h1, REF_S2C_H1, L * HC * DIM);
    ok &= nclose(fp, REF_S2C_PRE1, L * HC);
    CHECK(ok, "T1 MoE (gate+esperti+shared) + hc_post -> h1", "");
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
