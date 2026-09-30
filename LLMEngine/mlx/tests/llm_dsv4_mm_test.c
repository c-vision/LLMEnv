/* llm_dsv4_mm_test.c — MoE batched (mul_mm_id, kernel ds4) contro il path verificato pair/sum6 su pesi REALI.
 * Test differenziale: stessi (x, esperti, pesi) -> due implementazioni; errore relativo atteso ~1e-3 (staging f16).
 * Uso: mm_test [gguf] [layer] [nt] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
#include "llm_dsv4_engine.h"
#include "llm_dsv4_metal.h"

static unsigned st = 987654321u;
static float fr(void) { st = st * 1103515245u + 12345u; return (float)((st >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int il = argc > 2 ? atoi(argv[2]) : 5, nt = argc > 3 ? atoi(argv[3]) : 48;
    LlmDsv4Engine* e = llm_dsv4_engine_load(path, 256);
    if (!e) { printf("load fallita\n"); return 1; }
    const LlmDsv4Config* c = &e->w.cfg; const LlmDsv4PackedExperts* pe = &e->w.exps[il];
    const int embd = c->n_embd, ff = c->n_ff_exp, ne = c->n_expert;
    if (pe->gate_type != 16 || pe->down_type != 10) { printf("layer %d non IQ2/Q2K (%d/%d)\n", il, pe->gate_type, pe->down_type); return 1; }
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    size_t gnb = (size_t)llm_ggml_nbytes(pe->gate_type, embd), dnb = (size_t)llm_ggml_nbytes(pe->down_type, ff);
    float* x = malloc(sizeof(float) * (size_t)nt * embd); int* ids = malloc(sizeof(int) * (size_t)nt * 6); float* rw = malloc(sizeof(float) * (size_t)nt * 6);
    for (int i = 0; i < nt * embd; i++) x[i] = 0.05f * fr();
    for (int t = 0; t < nt; t++) for (int k = 0; k < 6; k++) {
        int v; int ok; do { v = (int)(((unsigned)(fr() * 32768.f + 32768.f) * 2654435761u) % (unsigned)ne); ok = 1; for (int j = 0; j < k; j++) if (ids[t*6+j] == v) ok = 0; } while (!ok);
        ids[t*6+k] = v; rw[t*6+k] = 0.1f + 0.4f * (fr() * 0.5f + 0.5f); }
    const void** gp = malloc(sizeof(void*) * 3 * (size_t)nt * 6); const void** up = gp + nt * 6; const void** dp = up + nt * 6;
    for (int k = 0; k < nt * 6; k++) { size_t ex = (size_t)ids[k];
        gp[k] = (const uint8_t*)pe->gate + ex * ff * gnb; up[k] = (const uint8_t*)pe->up + ex * ff * gnb; dp[k] = (const uint8_t*)pe->down + ex * embd * dnb; }
    float* ref = malloc(sizeof(float) * (size_t)nt * embd); float* got = malloc(sizeof(float) * (size_t)nt * embd);
    int r1 = llm_dsv4_metal_moe_iq2_q2k_batch(be, nt, gp, up, dp, x, rw, 6, c->swiglu_clamp_exp, embd, (uint32_t)gnb, ff, (uint32_t)dnb, NULL, NULL, NULL, 0, 0, ref);
    int r2 = llm_dsv4_metal_moe_mm_batch(be, nt, pe->gate, pe->up, pe->down, pe->gate_type, pe->down_type, ne, embd, ff, (uint32_t)gnb, (uint32_t)dnb, ids, rw, x, c->swiglu_clamp_exp, NULL, NULL, NULL, 0, 0, got);
    { int nr = 0, ng = 0; for (int i = 0; i < nt * embd; i++) { nr += !isfinite(ref[i]); ng += !isfinite(got[i]); }
      printf("non-finiti ref=%d got=%d  ref[0..3]=%g %g %g %g got[0..3]=%g %g %g %g clamp=%g\n", nr, ng, ref[0], ref[1], ref[2], ref[3], got[0], got[1], got[2], got[3], c->swiglu_clamp_exp); }
    { float* mids[6]; float* mb = malloc(sizeof(float) * 6 * ff); for (int k = 0; k < 6; k++) mids[k] = mb + k * ff;
      int r = llm_dsv4_metal_pair_swiglu_batch(be, gp, up, x, mids, rw, 6, c->swiglu_clamp_exp, embd, (uint32_t)gnb, ff);
      int nn = 0; for (int i = 0; i < 6 * ff; i++) nn += !isfinite(mb[i]);
      printf("pair_swiglu_batch rc=%d nonfinite=%d mid[0..2]=%g %g %g\n", r, nn, mb[0], mb[1], mb[2]); }
    double mx = 0, ss = 0, sr = 0;
    for (int i = 0; i < nt * embd; i++) { double d = fabs((double)ref[i] - got[i]); if (d > mx) mx = d; ss += d * d; sr += (double)ref[i] * ref[i]; }
    double rel = sqrt(ss / (sr > 0 ? sr : 1));
    printf("[MM] layer=%d nt=%d rc=%d/%d  max|diff|=%.3e  rel=%.3e  |ref|rms=%.3e\n", il, nt, r1, r2, mx, rel, sqrt(sr / ((double)nt * embd)));
    { double t0 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_moe_iq2_q2k_batch(be, nt, gp, up, dp, x, rw, 6, c->swiglu_clamp_exp, embd, (uint32_t)gnb, ff, (uint32_t)dnb, NULL, NULL, NULL, 0, 0, ref);
      double t1 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_moe_mm_batch(be, nt, pe->gate, pe->up, pe->down, pe->gate_type, pe->down_type, ne, embd, ff, (uint32_t)gnb, (uint32_t)dnb, ids, rw, x, c->swiglu_clamp_exp, NULL, NULL, NULL, 0, 0, got);
      double t2 = now(); printf("tempo (caldo, 5 iter): pair/sum6 %.1f ms  |  mul_mm_id %.1f ms per chunk di %d token\n", (t1 - t0) * 200, (t2 - t1) * 200, nt); }
    {   /* variante con TABELLA DI INDIRIZZI per esperto (buffer separati, come lo store) */
        const void** ga = calloc((size_t)ne, sizeof(void*)); const void** ua = calloc((size_t)ne, sizeof(void*)); const void** da = calloc((size_t)ne, sizeof(void*));
        for (int k = 0; k < nt * 6; k++) { size_t ex = (size_t)ids[k]; ga[ex] = (const uint8_t*)pe->gate + ex * ff * gnb; ua[ex] = (const uint8_t*)pe->up + ex * ff * gnb; da[ex] = (const uint8_t*)pe->down + ex * embd * dnb; }
        float* got2 = malloc(sizeof(float) * (size_t)nt * embd);
        int r3 = llm_dsv4_metal_moe_mm_addr_batch(be, nt, ga, ua, da, pe->gate_type, pe->down_type, ne, embd, ff, (uint32_t)gnb, (uint32_t)dnb, ids, rw, x, c->swiglu_clamp_exp, NULL, NULL, NULL, 0, 0, got2);
        double m2 = 0, s2 = 0, sr2 = 0; int nf = 0; for (int i = 0; i < nt * embd; i++) { nf += !isfinite(got2[i]); double d = fabs((double)ref[i] - got2[i]); if (d > m2) m2 = d; s2 += d*d; sr2 += (double)ref[i]*ref[i]; }
        printf("[MM-ADDR] rc=%d nonfiniti=%d max|diff|=%.3e rel=%.3e\n", r3, nf, m2, sqrt(s2 / (sr2 > 0 ? sr2 : 1)));
        double t0 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_moe_mm_addr_batch(be, nt, ga, ua, da, pe->gate_type, pe->down_type, ne, embd, ff, (uint32_t)gnb, (uint32_t)dnb, ids, rw, x, c->swiglu_clamp_exp, NULL, NULL, NULL, 0, 0, got2);
        printf("tempo addr (caldo): %.1f ms per chunk di %d token\n", (now() - t0) * 200, nt);
        if (r3 != 0 || nf) { printf("FAIL mm_addr\n"); return 1; } }
    int ok = r1 == 0 && r2 == 0 && rel < 5e-3;
    printf(ok ? "PASS mm_id: routed MoE batched == pair/sum6 (rel < 5e-3)\n" : "FAIL mm_id\n");
    return ok ? 0 : 1;
}
