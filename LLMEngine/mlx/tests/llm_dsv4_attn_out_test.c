/* llm_dsv4_attn_out_test.c — output grouped-LoRA batched (mul_mm_id Q8_0) vs path per-token verificato (attn_out_fused_batch),
 * su pesi reali. Atteso: errore relativo ~1e-3 (activation a half). Uso: attn_out_test [gguf] [layer] [nt] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_engine.h"
#include "llm_dsv4_metal.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static unsigned st = 24680u;
static float fr(void) { st = st * 1103515245u + 12345u; return (float)((st >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int il = argc > 2 ? atoi(argv[2]) : 5, nt = argc > 3 ? atoi(argv[3]) : 64;
    LlmDsv4Engine* e = llm_dsv4_engine_load(path, 256);
    if (!e) { printf("load fallita\n"); return 1; }
    const LlmDsv4Config* c = &e->w.cfg; const LlmDsv4PackedExperts* pe = &e->w.exps[il];
    const int ng = c->n_out_group, rank = c->n_lora_o, embd = c->n_embd, gd = c->n_head_dim * (c->n_head / ng), ol = ng * rank, ht = c->n_head * c->n_head_dim;
    if (pe->out_a_type != 8 || pe->out_b_type != 8) { printf("out non Q8_0\n"); return 1; }
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    size_t ar = (size_t)llm_ggml_nbytes(pe->out_a_type, gd), br = (size_t)llm_ggml_nbytes(pe->out_b_type, ol);
    float* h = malloc(sizeof(float) * (size_t)nt * ht); float* ref = malloc(sizeof(float) * (size_t)nt * embd); float* got = malloc(sizeof(float) * (size_t)nt * embd);
    for (int i = 0; i < nt * ht; i++) h[i] = fr();
    int r1 = llm_dsv4_metal_attn_out_fused_batch(be, nt, pe->out_a, ar, 0, ng, gd, rank, pe->out_b, br, ol, embd, ht, h, ref);
    int r2 = llm_dsv4_metal_attn_out_mm_batch(be, nt, pe->out_a, ar, ng, gd, rank, pe->out_b, br, ol, embd, ht, h, got);
    double mx = 0, ss = 0, sr = 0;
    for (int i = 0; i < nt * embd; i++) { double d = fabs((double)ref[i] - got[i]); if (d > mx) mx = d; ss += d * d; sr += (double)ref[i] * ref[i]; }
    double rel = sqrt(ss / (sr > 0 ? sr : 1));
    printf("[OUT] layer=%d nt=%d ng=%d rank=%d rc=%d/%d max|diff|=%.3e rel=%.3e |ref|rms=%.3e\n", il, nt, ng, rank, r1, r2, mx, rel, sqrt(sr / ((double)nt * embd)));
    double t0 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_attn_out_fused_batch(be, nt, pe->out_a, ar, 0, ng, gd, rank, pe->out_b, br, ol, embd, ht, h, ref);
    double t1 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_attn_out_mm_batch(be, nt, pe->out_a, ar, ng, gd, rank, pe->out_b, br, ol, embd, ht, h, got);
    double t2 = now();
    printf("tempo (caldo): per-token-chain %.2f ms | mul_mm_id %.2f ms per chunk di %d token\n", (t1 - t0) * 200, (t2 - t1) * 200, nt);
    int ok = r1 == 0 && r2 == 0 && rel < 5e-3;
    printf(ok ? "PASS attn_out mm_id\n" : "FAIL attn_out mm_id\n");
    return ok ? 0 : 1;
}
