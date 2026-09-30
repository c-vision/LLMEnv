/* llm_dsv4_q4_test.c — MoE Q4_K per chunk (kernel ds4 slots6 pair_swiglu + sum6) contro il path per-token gemv_batch (Q8K) su pesi
 * REALI di un layer Q4_K. Atteso: errore relativo ~1e-2 (il riferimento quantizza le attivazioni a Q8_K, il nuovo usa f32).
 * Uso: q4_test [gguf] [nt]   (un solo layer Q4_K, un solo processo col modello) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_engine.h"
#include "llm_dsv4_metal.h"
#include "llmcodec_ggml_quants.h"
static void dot12(const uint8_t* row, int n, const float* y, float* out) { static float* w = NULL; static int cap = 0; if (cap < n) { free(w); w = malloc(4 * (size_t)n); cap = n; }
    llmcodec_ggml_dequant_row(12, row, w, n); double s = 0; for (int k = 0; k < n; k++) s += (double)w[k] * y[k]; *out = (float)s; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static unsigned st = 8642u;
static float fr(void) { st = st * 1103515245u + 12345u; return (float)((st >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int nt = argc > 2 ? atoi(argv[2]) : 8;
    LlmDsv4Engine* e = llm_dsv4_engine_load(path, 256);
    if (!e) { printf("load fallita\n"); return 1; }
    const LlmDsv4Config* c = &e->w.cfg; int il = -1;
    for (int i = 0; i < c->n_layer; i++) if (e->w.exps[i].gate_type == 12 && e->w.exps[i].up_type == 12 && e->w.exps[i].down_type == 12) { il = i; break; }
    if (il < 0) { printf("nessun layer Q4_K\n"); return 1; }
    const LlmDsv4PackedExperts* pe = &e->w.exps[il];
    const int embd = c->n_embd, ff = c->n_ff_exp, ne = c->n_expert;
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    size_t gnb = (size_t)llm_ggml_nbytes(pe->gate_type, embd), dnb = (size_t)llm_ggml_nbytes(pe->down_type, ff);
    float* x = malloc(sizeof(float) * (size_t)nt * embd); int* ids = malloc(sizeof(int) * (size_t)nt * 6); float* rw = malloc(sizeof(float) * (size_t)nt * 6);
    for (int i = 0; i < nt * embd; i++) x[i] = 0.7f * fr();
    for (int t = 0; t < nt; t++) for (int k = 0; k < 6; k++) { int v, ok; do { v = (int)(((unsigned)(fr() * 32768.f + 32768.f) * 2654435761u) % (unsigned)ne); ok = 1; for (int j = 0; j < k; j++) if (ids[t*6+j] == v) ok = 0; } while (!ok); ids[t*6+k] = v; rw[t*6+k] = 0.1f + 0.4f * (fr() * 0.5f + 0.5f); }
    const void** gp = malloc(sizeof(void*) * 3 * (size_t)nt * 6); const void** up = gp + nt * 6; const void** dp = up + nt * 6;
    for (int k = 0; k < nt * 6; k++) { size_t ex = (size_t)ids[k]; gp[k] = pe->gate + ex * ff * gnb; up[k] = pe->up + ex * ff * gnb; dp[k] = pe->down + ex * embd * dnb; }
    float* ref = calloc((size_t)nt * embd, sizeof(float)); float* got = malloc(sizeof(float) * (size_t)nt * embd);
    /* riferimento: per token gemv_batch Q8K (gate/up), swiglu*peso su CPU, gemv_batch Q8K (down), somma */
    for (int t = 0; t < nt; t++) {
        float* g = malloc(4 * (size_t)ff * 6); float* u = malloc(4 * (size_t)ff * 6); float* mid = malloc(4 * (size_t)ff * 6); float* dn = malloc(4 * (size_t)embd * 6);
        LlmDsv4GpuJob jb[12], jd[6];
        for (int i = 0; i < 6; i++) { int k = t * 6 + i;
            jb[2*i]   = (LlmDsv4GpuJob){ gp[k], 0, 12, gnb, (uint32_t)embd, (uint32_t)ff, x + (size_t)t * embd, g + (size_t)i * ff };
            jb[2*i+1] = (LlmDsv4GpuJob){ up[k], 0, 12, gnb, (uint32_t)embd, (uint32_t)ff, x + (size_t)t * embd, u + (size_t)i * ff }; }
        if (llm_dsv4_metal_gemv_batch(be, jb, 12)) { printf("gemv_batch ref fallita\n"); return 1; }
        for (int i = 0; i < 6; i++) for (int j = 0; j < ff; j++) { float gg = g[i*ff+j], uu = u[i*ff+j]; float cl = c->swiglu_clamp_exp;
            if (cl > 1e-6f) { gg = fminf(gg, cl); uu = fmaxf(-cl, fminf(cl, uu)); } mid[i*ff+j] = gg / (1.f + expf(-gg)) * uu * rw[t*6+i]; }
        for (int i = 0; i < 6; i++) jd[i] = (LlmDsv4GpuJob){ dp[t*6+i], 0, 12, dnb, (uint32_t)ff, (uint32_t)embd, mid + (size_t)i * ff, dn + (size_t)i * embd };
        if (llm_dsv4_metal_gemv_batch(be, jd, 6)) { printf("gemv_batch ref down fallita\n"); return 1; }
        for (int i = 0; i < 6; i++) for (int o = 0; o < embd; o++) ref[(size_t)t * embd + o] += dn[(size_t)i * embd + o];
        free(g); free(u); free(mid); free(dn);
    }
    int r2 = llm_dsv4_metal_moe_q4k_batch(be, nt, gp, up, dp, x, rw, 6, c->swiglu_clamp_exp, embd, (uint32_t)gnb, ff, (uint32_t)dnb, NULL, NULL, NULL, 0, 0, got);
    double mx = 0, ss = 0, sr = 0; int nn = 0;
    for (int i = 0; i < nt * embd; i++) { nn += !isfinite(got[i]); double d = fabs((double)ref[i] - got[i]); if (d > mx) mx = d; ss += d * d; sr += (double)ref[i] * ref[i]; }
    double rel = sqrt(ss / (sr > 0 ? sr : 1));
    printf("[Q4] layer=%d nt=%d rc=%d nonfiniti=%d max|diff|=%.3e rel=%.3e |ref|rms=%.3e\n", il, nt, r2, nn, mx, rel, sqrt(sr / ((double)nt * embd)));
    {   /* riferimento ESATTO su CPU (dequant f32 x attivazioni f32) per il token 0: chi e' piu' vicino, il nuovo o il vecchio Q8K? */
        float* mid = malloc(4 * (size_t)ff * 6); float* ex = calloc((size_t)embd, 4);
        for (int i = 0; i < 6; i++) for (int j = 0; j < ff; j++) { float g = 0, u = 0;
            dot12((const uint8_t*)gp[i] + (size_t)j * gnb, embd, x, &g); dot12((const uint8_t*)up[i] + (size_t)j * gnb, embd, x, &u);
            float cl = c->swiglu_clamp_exp; if (cl > 1e-6f) { g = fminf(g, cl); u = fmaxf(-cl, fminf(cl, u)); } mid[i*ff+j] = g / (1.f + expf(-g)) * u * rw[i]; }
        for (int i = 0; i < 6; i++) for (int o = 0; o < embd; o++) { float d = 0; dot12((const uint8_t*)dp[i] + (size_t)o * dnb, ff, mid + (size_t)i * ff, &d); ex[o] += d; }
        double sn = 0, so = 0, sr2 = 0; for (int o = 0; o < embd; o++) { double dn = got[o] - ex[o], dO = ref[o] - ex[o]; sn += dn*dn; so += dO*dO; sr2 += (double)ex[o]*ex[o]; }
        printf("vs ESATTO (token 0): nuovo rel=%.3e   vecchio(Q8K) rel=%.3e\n", sqrt(sn / sr2), sqrt(so / sr2)); free(mid); free(ex); }
    double t0 = now(); for (int k = 0; k < 3; k++) llm_dsv4_metal_moe_q4k_batch(be, nt, gp, up, dp, x, rw, 6, c->swiglu_clamp_exp, embd, (uint32_t)gnb, ff, (uint32_t)dnb, NULL, NULL, NULL, 0, 0, got);
    printf("tempo (caldo): q4k_batch %.2f ms per chunk di %d token\n", (now() - t0) * 1000 / 3, nt);
    int ok = r2 == 0 && nn == 0 && rel < 2e-2;
    printf(ok ? "PASS moe q4k batch\n" : "FAIL moe q4k batch\n");
    return ok ? 0 : 1;
}
