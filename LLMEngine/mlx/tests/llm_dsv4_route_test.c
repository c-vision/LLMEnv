/* llm_dsv4_route_test.c — routing BATCHED su GPU contro llm_dsv4_moe_route (CPU). SINTETICO (nessun modello): pesi/bias/x casuali con le dimensioni reali
 * (256 esperti, embd 4096, top-6). Atteso: selezioni identiche e pesi rel ~1e-6. Uso: route_test [nt] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_moe_fwd.h"
#include "llm_dsv4_metal.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static unsigned st = 5551212u;
static float fr(void) { st = st * 1103515245u + 12345u; return (float)((st >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }
int main(int argc, char** argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 512;
    LlmDsv4Config c; memset(&c, 0, sizeof c); c.n_expert = 256; c.n_expert_used = 6; c.n_embd = 4096; c.n_hash_layer = 3; c.expert_weight_scale = 1.5f;
    LlmDsv4LayerWeights L; memset(&L, 0, sizeof L);
    L.ffn_gate_inp = malloc(4 * (size_t)256 * 4096); for (long i = 0; i < 256L * 4096; i++) L.ffn_gate_inp[i] = 0.03f * fr();
    float* bias = malloc(4 * 256); for (int i = 0; i < 256; i++) bias[i] = 0.2f * fr(); L.exp_probs_b = bias;
    int nvocab = 1000; float* tid2eid = malloc(4 * (size_t)nvocab * 6); for (int i = 0; i < nvocab * 6; i++) tid2eid[i] = (float)((unsigned)(fr() * 32768.f + 32768.f) % 256u);
    L.ffn_gate_tid2eid = tid2eid;
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    float* x = malloc(4 * (size_t)nt * 4096); for (long i = 0; i < (long)nt * 4096; i++) x[i] = 1.2f * fr();
    int* tokens = malloc(4 * (size_t)nt); for (int t = 0; t < nt; t++) tokens[t] = (t * 37) % nvocab;
    int ok = 1;
    for (int il = 5; il >= 1; il -= 4) {   /* il=5: top-k con bias; il=1: layer hash (esperti da tabella) */
        int* rs = malloc(4 * (size_t)nt * 6); float* rw = malloc(4 * (size_t)nt * 6); int* gs = malloc(4 * (size_t)nt * 6); float* gw = malloc(4 * (size_t)nt * 6);
        double t0 = now();
        for (int t = 0; t < nt; t++) llm_dsv4_moe_route(&c, &L, x + (size_t)t * 4096, il, tokens[t], rs + (size_t)t * 6, rw + (size_t)t * 6);
        double t1 = now();
        int* hs = NULL; if (il < c.n_hash_layer) { hs = malloc(4 * (size_t)nt * 6); for (int t = 0; t < nt; t++) for (int j = 0; j < 6; j++) hs[t*6+j] = (int)tid2eid[(size_t)tokens[t]*6+j]; }
        int r = llm_dsv4_metal_route_batch(be, nt, 256, 4096, 6, L.ffn_gate_inp, il < c.n_hash_layer ? NULL : bias, hs, c.expert_weight_scale, x, gs, gw);
        double t2 = now(); for (int k = 0; k < 3; k++) llm_dsv4_metal_route_batch(be, nt, 256, 4096, 6, L.ffn_gate_inp, il < c.n_hash_layer ? NULL : bias, hs, c.expert_weight_scale, x, gs, gw);
        double t3 = now();
        int diff = 0; double mw = 0, sw = 0, sr = 0;
        for (int i = 0; i < nt * 6; i++) { diff += (rs[i] != gs[i]); double d = fabs((double)rw[i] - gw[i]); if (d > mw) mw = d; sw += d*d; sr += (double)rw[i]*rw[i]; }
        printf("[ROUTE] il=%d nt=%d rc=%d selezioni diverse=%d/%d  pesi rel=%.2e max=%.2e | CPU %.1f ms  GPU %.2f ms\n", il, nt, r, diff, nt * 6, sqrt(sw / sr), mw, (t1 - t0) * 1000, (t3 - t2) * 1000 / 3);
        if (r != 0 || diff > nt * 6 / 200 || sqrt(sw / sr) > 1e-4) ok = 0;
        (void)t2; free(rs); free(rw); free(gs); free(gw); free(hs);
    }
    printf(ok ? "PASS route batched\n" : "FAIL route batched\n");
    return ok ? 0 : 1;
}
