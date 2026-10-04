/* llm_dsv41_fwd_test.c — step 3: forward COMPLETO tiny (2 layer + final norm + head -> logits).
 *   cc -O2 -o /tmp/llm_dsv41_fwd_test native/llmengine_mlx/llm_dsv41_fwd_test.c -lm
 * Verifica vs llm_dsv41_fwd_ref.h (riferimento oMLX tiny, pesi random seedati). */
#include <stdio.h>
#include <math.h>
#include "llm_dsv41_math.h"
#include "llm_dsv41_attn.h"
#include "llm_dsv41_moe.h"
#include "llm_dsv41_rope.h"
#include "llm_dsv41_fwd_ref.h"

static int g_fail = 0;
#define TOL 4e-3f
static int nclose(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > TOL * m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-50s\n", name); \
    else { printf("FAIL %-50s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static void mv(const float* w, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; r++) { float s = 0; for (int c = 0; c < cols; c++) s += w[r * cols + c] * x[c]; y[r] = s; }
}
static float silu(float g) { return g / (1.0f + expf(-g)); }

typedef struct {
    const float *hcafn,*hcas,*hcab,*anorm,*wqa,*qn,*wqb,*woa,*wob,*sink,*win;
    const float *hcffn,*hcfs,*hcfb,*fnorm,*gw,*gb,*ew1,*ew3,*ew2,*sw1,*sw3,*sw2;
} LW;

static void moe(const LW* w, const float* x, float* out) {
    const int DIM = REF_FW_DIM, E = REF_FW_E, ACT = REF_FW_ACT, DI = REF_FW_DI;
    int idx[8]; float wt[8];
    llm_dsv41_gate(x, DIM, w->gw, E, w->gb, 1.0f, LLM_DSV41_SCORE_SQRTSOFTPLUS, ACT, 1, 1.0f, idx, wt);
    float acc[16] = {0};
    for (int k = 0; k < ACT; k++) {
        int e = idx[k]; float g[64], u[64], y[64], de[16];
        mv(w->ew1 + (size_t)e * DI * DIM, DI, DIM, x, g);
        mv(w->ew3 + (size_t)e * DI * DIM, DI, DIM, x, u);
        for (int i = 0; i < DI; i++) y[i] = silu(g[i]) * u[i] * wt[k];
        mv(w->ew2 + (size_t)e * DIM * DI, DIM, DI, y, de);
        for (int i = 0; i < DIM; i++) acc[i] += de[i];
    }
    float sg[64], su[64], sy[64], so[16];
    mv(w->sw1, DI, DIM, x, sg); mv(w->sw3, DI, DIM, x, su);
    for (int i = 0; i < DI; i++) sy[i] = silu(sg[i]) * su[i];
    mv(w->sw2, DIM, DI, sy, so);
    for (int i = 0; i < DIM; i++) out[i] = acc[i] + so[i];
}

static void attention(const LW* w, const float* x, int l, float* out) {
    const int H = REF_FW_H, D = REF_FW_D, DR = REF_FW_DROPE, QD = REF_FW_QD;
    const int OG = REF_FW_OG, OL = REF_FW_OLORA, Lq = REF_FW_L;
    const float astat = 1.0f / sqrtf((float)D);
    float qr[8]; mv(w->wqa, QD, REF_FW_DIM, x, qr);
    llm_dsv41_rms_norm(qr, w->qn, QD, 1e-20f, qr);
    float q[64]; mv(w->wqb, H * D, QD, qr, q);
    for (int h = 0; h < H; h++) llm_dsv41_rope(q + h * D, D, (float)l, DR, 10000.0f, 0, 32, 1, 4.0f, 0, 0);
    float keys[128];
    for (int m = 0; m < Lq; m++) for (int h = 0; h < H; h++) for (int e = 0; e < D; e++)
        keys[(m * H + h) * D + e] = w->win[m * D + e];
    int idx[8]; for (int m = 0; m < Lq; m++) idx[m] = (m <= l) ? m : -1;
    float o[64];
    llm_dsv41_sdpa_sink(q, keys, idx, Lq, H, D, w->sink, astat, o);
    for (int h = 0; h < H; h++) llm_dsv41_rope(o + h * D, D, (float)l, DR, 10000.0f, 0, 32, 1, 4.0f, 0, 1);
    int grp = H * D / OG; float y[64];
    for (int g = 0; g < OG; g++) for (int r = 0; r < OL; r++) {
        float s = 0; for (int k = 0; k < grp; k++) s += o[g * grp + k] * w->woa[(g * OL + r) * grp + k];
        y[g * OL + r] = s;
    }
    mv(w->wob, REF_FW_DIM, OG * OL, y, out);
}

static void block(const LW* w, const float* h_all, const float* pre_all, float* h_out, float* pre_out) {
    const int L = REF_FW_L, DIM = REF_FW_DIM, HC = REF_FW_HC, SIZE = REF_FW_SIZE;
    const float neps = 1e-20f, heps = 1e-6f; const int it = 20;
    for (int l = 0; l < L; l++) {
        const float* h = h_all + l * HC * DIM;
        float mixes[64], ap[16], ao[16], ac[64], hin[16], x[16];
        llm_dsv41_hc_mixes(h, HC * DIM, w->hcafn, SIZE, w->hcas, w->hcab, HC, neps, heps, it, mixes, ap, ao, ac);
        llm_dsv41_hc_pre(h, pre_all + l * HC, HC, DIM, hin);
        llm_dsv41_rms_norm(hin, w->anorm, DIM, neps, x);
        float attn_out[16]; attention(w, x, l, attn_out);
        float h2[32]; llm_dsv41_hc_post(attn_out, DIM, h, ao, ac, HC, h2);
        float m2[64], fp[16], fo[16], fc[64], fin[16];
        llm_dsv41_hc_mixes(h2, HC * DIM, w->hcffn, SIZE, w->hcfs, w->hcfb, HC, neps, heps, it, m2, fp, fo, fc);
        llm_dsv41_hc_pre(h2, ap, HC, DIM, fin);          /* nota: il riferimento usa `ap` */
        llm_dsv41_rms_norm(fin, w->fnorm, DIM, neps, fin);
        float fout[16]; moe(w, fin, fout);
        float h3[32]; llm_dsv41_hc_post(fout, DIM, h2, fo, fc, HC, h3);
        for (int i = 0; i < HC * DIM; i++) h_out[l * HC * DIM + i] = h3[i];
        for (int i = 0; i < HC; i++) pre_out[l * HC + i] = fp[i];
    }
}

#define LW0 {REF_FW_HCAFN0,REF_FW_HCAS0,REF_FW_HCAB0,REF_FW_ANORM0,REF_FW_WQA0,REF_FW_QN0,REF_FW_WQB0,REF_FW_WOA0,REF_FW_WOB0,REF_FW_SINK0,REF_FW_WIN0,REF_FW_HCFFN0,REF_FW_HCFFS0,REF_FW_HCFFB0,REF_FW_FNORM0,REF_FW_GW0,REF_FW_GB0,REF_FW_EW10,REF_FW_EW30,REF_FW_EW20,REF_FW_SW10,REF_FW_SW30,REF_FW_SW20}
#define LW1 {REF_FW_HCAFN1,REF_FW_HCAS1,REF_FW_HCAB1,REF_FW_ANORM1,REF_FW_WQA1,REF_FW_QN1,REF_FW_WQB1,REF_FW_WOA1,REF_FW_WOB1,REF_FW_SINK1,REF_FW_WIN1,REF_FW_HCFFN1,REF_FW_HCFFS1,REF_FW_HCFFB1,REF_FW_FNORM1,REF_FW_GW1,REF_FW_GB1,REF_FW_EW11,REF_FW_EW31,REF_FW_EW21,REF_FW_SW11,REF_FW_SW31,REF_FW_SW21}

int main(void) {
    const int L = REF_FW_L, DIM = REF_FW_DIM, HC = REF_FW_HC, V = REF_FW_VOCAB;
    printf("llm_dsv41_fwd — forward completo tiny (2 layer + head)\n");
    float h0[128], pre0[32];
    for (int l = 0; l < L; l++) {
        int id = l == 0 ? REF_FW_IDS[0] : REF_FW_IDS[1];
        for (int c = 0; c < HC; c++) for (int d = 0; d < DIM; d++)
            h0[(l * HC + c) * DIM + d] = REF_FW_EMBED[id * DIM + d];
        for (int c = 0; c < HC; c++) pre0[l * HC + c] = (c == 0) ? 1.0f : 0.0f;
    }
    const LW w0 = LW0, w1 = LW1;
    float h1[128], pre1[32], h2[128], pre2[32];
    block(&w0, h0, pre0, h1, pre1);
    block(&w1, h1, pre1, h2, pre2);
    float logits[128];
    for (int l = 0; l < L; l++) {
        float hp[16], hn[16];
        llm_dsv41_hc_pre(h2 + l * HC * DIM, pre2 + l * HC, HC, DIM, hp);
        llm_dsv41_rms_norm(hp, REF_FW_FNORM, DIM, 1e-20f, hn);
        mv(REF_FW_HEAD, V, DIM, hn, logits + l * V);
    }
    printf("logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           logits[0], logits[1], logits[2], logits[3],
           REF_FW_LOGITS[0], REF_FW_LOGITS[1], REF_FW_LOGITS[2], REF_FW_LOGITS[3]);
    CHECK(nclose(logits, REF_FW_LOGITS, L * V), "T1 forward completo -> logits", "");
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
