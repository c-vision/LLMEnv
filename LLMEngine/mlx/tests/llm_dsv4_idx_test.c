/* llm_dsv4_idx_test.c — test unitario dei kernel Metal dell'indexer (`dsv4_indexer_qat`,
 * `dsv4_indexer_scores`, `dsv4_indexer_topk`) contro le reference CPU. Deterministico. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_metal.h"
#include "llm_dsv4_quant.h"

static unsigned s_state = 24681357u;
static float fr(void) { s_state = s_state * 1103515245u + 12345u; return (float)((s_state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }

static float dotf(const float* a, const float* b, int n) { float s = 0.f; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }

static int test_qat(LlmDsv4Metal* b, int n_rows) {
    int hd = 128;
    float* ref = (float*)malloc(sizeof(float) * (size_t)n_rows * hd);
    float* gpu = (float*)malloc(sizeof(float) * (size_t)n_rows * hd);
    for (int i = 0; i < n_rows * hd; i++) { ref[i] = fr() * 4.0f; gpu[i] = ref[i]; }
    llm_dsv4_indexer_qat_rows(ref, n_rows, hd);
    int rc = llm_dsv4_metal_indexer_qat(b, gpu, n_rows, hd);
    double mx = 0.0;
    for (int i = 0; i < n_rows * hd; i++) { double d = fabs(ref[i] - gpu[i]); if (d > mx) mx = d; }
    int ok = (rc == 0 && mx == 0.0);
    printf("[QAT]  rows=%d rc=%d max|diff|=%.3e %s\n", n_rows, rc, mx, ok ? "ok" : "BAD");
    free(ref); free(gpu);
    return ok;
}

static int test_scores_topk(LlmDsv4Metal* b, int nh, int hd, int n_comp, int top_k) {
    float* q = (float*)malloc(sizeof(float) * (size_t)nh * hd);
    float* kv = (float*)malloc(sizeof(float) * (size_t)n_comp * hd);
    float* w = (float*)malloc(sizeof(float) * (size_t)nh);
    for (int i = 0; i < nh * hd; i++) q[i] = fr();
    for (int i = 0; i < n_comp * hd; i++) kv[i] = fr();
    for (int i = 0; i < nh; i++) w[i] = fr();
    float scale = 1.0f / sqrtf((float)(hd * nh));
    for (int i = 0; i < nh; i++) w[i] *= scale;

    float* ref = (float*)malloc(sizeof(float) * (size_t)n_comp);
    for (int c = 0; c < n_comp; c++) { float s = 0.f; for (int h = 0; h < nh; h++) { float d = dotf(q + (long)h * hd, kv + (long)c * hd, hd); if (d < 0.f) d = 0.f; s += d * w[h]; } ref[c] = s; }
    float* gpu = (float*)malloc(sizeof(float) * (size_t)n_comp);
    int rc1 = llm_dsv4_metal_indexer_scores(b, q, nh, hd, kv, n_comp, w, gpu);
    double mx = 0.0; for (int c = 0; c < n_comp; c++) { double d = fabs((double)ref[c] - (double)gpu[c]); if (d > mx) mx = d; }

    int* ra = (int*)calloc((size_t)n_comp, sizeof(int));
    for (int k = 0; k < top_k; k++) { int bi = 0; float bs = -1e30f; for (int c = 0; c < n_comp; c++) if (!ra[c] && ref[c] > bs) { bi = c; bs = ref[c]; } ra[bi] = 1; }
    int* ga = (int*)calloc((size_t)n_comp, sizeof(int));
    int rc2 = llm_dsv4_metal_indexer_topk(b, gpu, n_comp, top_k, ga);
    int same = 1; for (int c = 0; c < n_comp; c++) if (ra[c] != ga[c]) same = 0;

    int ok = (rc1 == 0 && rc2 == 0 && mx < 1e-4 && same);
    printf("[IDX]  nh=%d hd=%d n_comp=%d top_k=%d rc=%d/%d score=%.3e sel=%s %s\n",
           nh, hd, n_comp, top_k, rc1, rc2, mx, same ? "match" : "MISMATCH", ok ? "ok" : "BAD");
    free(q); free(kv); free(w); free(ref); free(gpu); free(ra); free(ga);
    return ok;
}

/* Test del top-k BATCHED (una threadgroup per token, radix-select) vs reference CPU per token.
 * Include score NEGATIVI (la reference somma dot*pesi, che possono essere <0), n_comp variabili
 * e duplicati (tie -> tie-break su indice minore). */
static int test_scores_topk_t(LlmDsv4Metal* b, int nt, int maxc, int top_k) {
    int* nc = (int*)malloc(sizeof(int) * (size_t)nt);
    float* sc = (float*)malloc(sizeof(float) * (size_t)nt * maxc);
    int* ga = (int*)malloc(sizeof(int) * (size_t)nt * maxc);
    int* ra = (int*)malloc(sizeof(int) * (size_t)nt * maxc);
    for (int t = 0; t < nt; t++) {
        s_state = s_state * 1103515245u + 12345u;
        int n = 1 + (int)((s_state >> 4) % (unsigned)maxc);
        nc[t] = n;
        for (int c = 0; c < n; c++) {
            float v = fr() * 4.0f - 1.5f;
            if ((c % 37) == 0 && c > 0) v = sc[(size_t)t * maxc + (c - 1)];
            sc[(size_t)t * maxc + c] = v;
        }
        for (int c = n; c < maxc; c++) sc[(size_t)t * maxc + c] = 0.0f;
        for (int c = 0; c < maxc; c++) ga[(size_t)t * maxc + c] = 0;
    }
    int rc = llm_dsv4_metal_indexer_topk_t(b, sc, nt, maxc, nc, top_k, ga);
    int same = 1;
    for (int t = 0; t < nt; t++) {
        int n = nc[t]; int k = top_k > n ? n : top_k;
        for (int c = 0; c < n; c++) ra[(size_t)t * maxc + c] = 0;
        for (int kk = 0; kk < k; kk++) {
            int bi = -1; float bs = -1e30f;
            for (int c = 0; c < n; c++) if (!ra[(size_t)t * maxc + c] && sc[(size_t)t * maxc + c] > bs) { bi = c; bs = sc[(size_t)t * maxc + c]; }
            ra[(size_t)t * maxc + bi] = 1;
        }
        for (int c = 0; c < n; c++) if (ra[(size_t)t * maxc + c] != ga[(size_t)t * maxc + c]) same = 0;
    }
    int ok = (rc == 0 && same);
    printf("[IDXt] nt=%d maxc=%d top_k=%d rc=%d sel=%s %s\n", nt, maxc, top_k, rc, same ? "match" : "MISMATCH", ok ? "ok" : "BAD");
    free(nc); free(sc); free(ga); free(ra);
    return ok;
}

int main(void) {
    if (!llm_dsv4_metal_available()) { printf("SKIP: nessun device Metal\n"); return 0; }
    LlmDsv4Metal* b = llm_dsv4_metal_create();
    if (!b) { printf("FAIL: llm_dsv4_metal_create\n"); return 1; }
    int ok = 1;
    ok &= test_qat(b, 64);
    ok &= test_qat(b, 1);
    ok &= test_scores_topk(b, 64, 128, 200, 12);  /* selezione parziale */
    ok &= test_scores_topk(b, 64, 128, 10, 10);   /* top_k == n_comp -> tutte ammesse */
    ok &= test_scores_topk(b, 32, 128, 130, 1);
    /* top-k batched (radix, una threadgroup per token) vs reference, con negativi e tie */
    ok &= test_scores_topk_t(b, 8, 300, 12);
    ok &= test_scores_topk_t(b, 4, 1024, 512);
    ok &= test_scores_topk_t(b, 3, 50, 50);    /* alcuni token n<=top_k -> tutte ammesse */
    ok &= test_scores_topk_t(b, 2, 200, 1);
    printf(ok ? "PASS indexer: QAT bit-exact, scores<1e-4, top-k identico alla reference CPU\n"
              : "FAIL indexer\n");
    llm_dsv4_metal_destroy(b);
    return ok ? 0 : 1;
}
