/* llm_dsv4_attn_test.c — test unitario del kernel Metal `dsv4_attn_mixed` contro la reference
 * CPU `llm_dsv4_attention_mixed` (cache.h). Deterministico, niente modello. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_metal.h"
#include "llm_dsv4_cache.h"

static unsigned s_state = 123456789u;
static float fr(void) { s_state = s_state * 1103515245u + 12345u; return (float)((s_state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }

static int run_case(LlmDsv4Metal* b, int n_head, int hd, int n_raw, int n_comp, int with_allowed) {
    float* q   = (float*)malloc(sizeof(float) * (size_t)n_head * hd);
    float* raw = n_raw  ? (float*)malloc(sizeof(float) * (size_t)n_raw  * hd) : NULL;
    float* cmp = n_comp ? (float*)malloc(sizeof(float) * (size_t)n_comp * hd) : NULL;
    int*   alw = (with_allowed && n_comp) ? (int*)malloc(sizeof(int) * (size_t)n_comp) : NULL;
    float* snk = (float*)malloc(sizeof(float) * (size_t)n_head);
    float* ref = (float*)malloc(sizeof(float) * (size_t)n_head * hd);
    float* gpu = (float*)malloc(sizeof(float) * (size_t)n_head * hd);
    for (int i = 0; i < n_head * hd; i++) q[i] = fr();
    for (int i = 0; i < n_raw * hd; i++) raw[i] = fr();
    for (int i = 0; i < n_comp * hd; i++) cmp[i] = fr();
    for (int c = 0; c < n_comp; c++) if (alw) alw[c] = (c % 3) != 0;
    for (int h = 0; h < n_head; h++) snk[h] = fr();

    llm_dsv4_attention_mixed(ref, NULL, q, raw, n_raw, cmp, n_comp, alw, snk, n_head, hd);
    int rc = llm_dsv4_metal_attention_mixed(b, q, n_head, hd, raw, n_raw, cmp, n_comp, alw, snk, gpu);

    double mx = 0.0, ss = 0.0, sr = 0.0;
    for (int i = 0; i < n_head * hd; i++) {
        double d = fabs((double)ref[i] - (double)gpu[i]);
        if (d > mx) mx = d;
        ss += d * d; sr += (double)ref[i] * (double)ref[i];
    }
    double rel = (sr > 0.0) ? sqrt(ss / sr) : sqrt(ss);
    int ok = (rc == 0 && mx < 1e-4);
    printf("[ATTN] h=%d hd=%d raw=%d comp=%d allowed=%d  rc=%d  max|diff|=%.3e rel=%.3e  %s\n",
           n_head, hd, n_raw, n_comp, with_allowed, rc, mx, rel, ok ? "ok" : "BAD");
    free(q); free(raw); free(cmp); free(alw); free(snk); free(ref); free(gpu);
    return ok;
}

int main(void) {
    if (!llm_dsv4_metal_available()) { printf("SKIP: nessun device Metal\n"); return 0; }
    LlmDsv4Metal* b = llm_dsv4_metal_create();
    if (!b) { printf("FAIL: llm_dsv4_metal_create\n"); return 1; }
    int ok = 1;
    ok &= run_case(b, 8, 128, 37, 53, 1);   /* percorso reale: allowed parziale */
    ok &= run_case(b, 8, 128, 11, 47, 1);   /* poche raw */
    ok &= run_case(b, 4, 128, 29, 0, 0);    /* solo raw (ratio=0 / no compresse) */
    ok &= run_case(b, 8, 128, 0, 40, 1);    /* solo compresse */
    ok &= run_case(b, 8, 512, 37, 53, 1);   /* hd reale DS4, split ns=2 */
    ok &= run_case(b, 64, 512, 128, 400, 1);  /* teste/finestra reali, ctx ~1.6k */
    ok &= run_case(b, 64, 512, 128, 3000, 1); /* ns=16 (ctx lungo): merge di 16 parziali */
    ok &= run_case(b, 64, 512, 5, 0, 0);    /* poche posizioni: split con simdgroup vuoti */
    printf(ok ? "PASS attn_mixed: entro 1e-4 vs reference CPU\n"
              : "FAIL attn_mixed\n");
    llm_dsv4_metal_destroy(b);
    return ok ? 0 : 1;
}
