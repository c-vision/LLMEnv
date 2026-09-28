/* llm_dsv41_fwd_step2b_test.c — step 2b: attention layer 0.
 *   cc -O2 -o /tmp/llm_dsv41_fwd_step2b_test native/llmengine_mlx/llm_dsv41_fwd_step2b_test.c -lm
 * Verifica rope+sdpa+wo contro llm_dsv41_fwd_step2b_ref.h. Il KV e' il WINDOW DEQUANTIZZATO dal
 * riferimento (il packing fp8 esatto e' nel todo KV-Packed-Layout). */
#include <stdio.h>
#include <stdlib.h>
#include "llm_dsv41_math.h"
#include "llm_dsv41_attn.h"
#include "llm_dsv41_rope.h"
#include "llm_dsv41_fwd_step2b_ref.h"

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

static void matvec(const float* w, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; r++) { float s = 0; for (int c = 0; c < cols; c++) s += w[r * cols + c] * x[c]; y[r] = s; }
}

int main(void) {
    const int L = REF_S2B_L, DIM = REF_S2B_DIM, H = REF_S2B_H, D = REF_S2B_D, DR = REF_S2B_DROPE, QD = REF_S2B_QD;
    const int OG = REF_S2B_OG, OL = REF_S2B_OLORA;
    const float scale = 1.0f / 1.0f; /* placeholder, ricalcolato sotto */
    const float attn_scale = 1.0f / sqrtf((float)D);
    printf("llm_dsv41_fwd_step2b — attention layer0\n");
    float out[64], qall[128], oall[128]; int ok = 1, okeq = 1;
    for (int l = 0; l < L; l++) {
        const float* x = REF_S2B_X + l * DIM;
        /* qr = q_norm(wq_a x) */
        float qr[8]; matvec(REF_S2B_WQA, QD, DIM, x, qr);
        float qrn[8]; llm_dsv41_rms_norm(qr, REF_S2B_QNORM, QD, 1e-20f, qrn);
        /* q = wq_b qr -> (H,D) ; rope */
        float q[64]; matvec(REF_S2B_WQB, H * D, QD, qrn, q);
        for (int h = 0; h < H; h++) llm_dsv41_rope(q + h * D, D, (float)l, DR, 10000.0f, 0, 32, 1, 4.0f, 0, 0);
        for (int i = 0; i < H * D; i++) qall[l * H * D + i] = q[i];
        /* keys: window dequant (single kv head) replicato per head */
        float keys[128]; /* M*H*D, M=L */
        for (int m = 0; m < L; m++)
            for (int h = 0; h < H; h++)
                for (int e = 0; e < D; e++)
                    keys[(m * H + h) * D + e] = REF_S2B_WIN_DEQ[m * D + e];
        int idx[8]; for (int m = 0; m < L; m++) idx[m] = (m <= l) ? m : -1;
        for (int m = 0; m < L; m++) if (idx[m] != REF_S2B_WI[l * L + m]) okeq = 0;
        float o[64];
        llm_dsv41_sdpa_sink(q, keys, idx, L, H, D, REF_S2B_SINK, attn_scale, o);
        for (int h = 0; h < H; h++) llm_dsv41_rope(o + h * D, D, (float)l, DR, 10000.0f, 0, 32, 1, 4.0f, 0, 1);
        for (int i = 0; i < H * D; i++) oall[l * H * D + i] = o[i];
        /* wo_a grouped einsum: grouped (OG, H*D/OG); weight (OG,OL,H*D/OG) */
        int grp = H * D / OG; float y[64];
        for (int g = 0; g < OG; g++)
            for (int r = 0; r < OL; r++) {
                float s = 0;
                for (int k = 0; k < grp; k++) s += o[g * grp + k] * REF_S2B_WOA[(g * OL + r) * grp + k];
                y[g * OL + r] = s;
            }
        matvec(REF_S2B_WOB, DIM, OG * OL, y, out + l * DIM);
    }
    (void)scale;
    CHECK(okeq, "T0 window indices", "");
    CHECK(nclose(qall, REF_S2B_Q, L * H * D), "T0b q (wq_a/q_norm/wq_b/rope)", "");
    CHECK(nclose(oall, REF_S2B_O, L * H * D), "T1 attention o (sdpa+sink+rope inv)", "");
    ok = nclose(out, REF_S2B_OUT, L * DIM);
    CHECK(ok, "T2 out (wo_a grouped + wo_b)", "");
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
