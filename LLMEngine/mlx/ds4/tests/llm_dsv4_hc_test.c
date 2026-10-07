/* llm_dsv4_hc_test.c — mHC BATCHED su GPU (pre/post) contro la reference CPU (llm_dsv4_block_pre/post). SINTETICO: nessun modello
 * (pesi casuali con le dimensioni reali HC=4, D=4096), quindi leggero e sicuro. Atteso: rel ~1e-5 (accumuli float vs double). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_block.h"
#include "llm_dsv4_metal.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static unsigned st = 97531u;
static float fr(void) { st = st * 1103515245u + 12345u; return (float)((st >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }
int main(int argc, char** argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 64;
    LlmDsv4Config c; memset(&c, 0, sizeof c); c.n_hc = 4; c.n_embd = 4096; c.rms_eps = 1e-6f; c.hc_eps = 1e-6f; c.n_hc_sinkhorn_iter = 20;
    const int HC = c.n_hc, D = c.n_embd, HCD = HC * D, SIZE = (2 + HC) * HC;
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    float* fn = malloc(4 * (size_t)SIZE * HCD); for (long i = 0; i < (long)SIZE * HCD; i++) fn[i] = 0.02f * fr();
    float scale[3] = { 0.9f, 0.8f, 1.1f }, base[24]; for (int i = 0; i < SIZE; i++) base[i] = 0.3f * fr();
    float* nw = malloc(4 * (size_t)D); for (int i = 0; i < D; i++) nw[i] = 1.0f + 0.2f * fr();
    float* res = malloc(4 * (size_t)nt * HCD); for (long i = 0; i < (long)nt * HCD; i++) res[i] = 2.0f * fr();
    float* bo = malloc(4 * (size_t)nt * D); for (long i = 0; i < (long)nt * D; i++) bo[i] = fr();
    float *ref_n = malloc(4 * (size_t)nt * D), *gpu_n = malloc(4 * (size_t)nt * D), *sp = malloc(4 * (size_t)nt * SIZE);
    float *rp = malloc(4 * (size_t)nt * HC), *rc = malloc(4 * (size_t)nt * HC * HC);
    for (int t = 0; t < nt; t++) llm_dsv4_block_pre(&c, fn, scale, base, nw, res + (size_t)t * HCD, ref_n + (size_t)t * D, rp + (size_t)t * HC, rc + (size_t)t * HC * HC);
    int r1 = llm_dsv4_metal_hc_pre_batch(be, nt, HC, D, c.n_hc_sinkhorn_iter, c.hc_eps, c.rms_eps, res, fn, scale, base, nw, gpu_n, sp);
    double sn = 0, sr = 0, sp_d = 0, spr = 0, mxn = 0;
    for (long i = 0; i < (long)nt * D; i++) { double d = fabs((double)ref_n[i] - gpu_n[i]); if (d > mxn) mxn = d; sn += d * d; sr += (double)ref_n[i] * ref_n[i]; }
    for (int t = 0; t < nt; t++) { for (int i = 0; i < HC; i++) { double d = sp[(size_t)t*SIZE + HC + i] - rp[(size_t)t*HC + i]; sp_d += d*d; spr += (double)rp[(size_t)t*HC+i]*rp[(size_t)t*HC+i]; }
        for (int i = 0; i < HC*HC; i++) { double d = sp[(size_t)t*SIZE + 2*HC + i] - rc[(size_t)t*HC*HC + i]; sp_d += d*d; spr += (double)rc[(size_t)t*HC*HC+i]*rc[(size_t)t*HC*HC+i]; } }
    printf("[HC-PRE] nt=%d rc=%d  normed rel=%.2e max=%.2e | post/comb rel=%.2e\n", nt, r1, sqrt(sn / sr), mxn, sqrt(sp_d / spr));
    /* post: usa gli stessi post/comb della CPU (sp con pre|post|comb della GPU: qui il confronto e' della sola formula) */
    float *ref_o = malloc(4 * (size_t)nt * HCD), *gpu_o = malloc(4 * (size_t)nt * HCD);
    float* sp2 = malloc(4 * (size_t)nt * SIZE); for (int t = 0; t < nt; t++) { for (int i = 0; i < HC; i++) { sp2[(size_t)t*SIZE + i] = 0; sp2[(size_t)t*SIZE + HC + i] = rp[(size_t)t*HC + i]; }
        for (int i = 0; i < HC*HC; i++) sp2[(size_t)t*SIZE + 2*HC + i] = rc[(size_t)t*HC*HC + i]; }
    for (int t = 0; t < nt; t++) llm_dsv4_block_post(&c, bo + (size_t)t * D, res + (size_t)t * HCD, rp + (size_t)t * HC, rc + (size_t)t * HC * HC, ref_o + (size_t)t * HCD);
    int r2 = llm_dsv4_metal_hc_post_batch(be, nt, HC, D, bo, res, sp2, gpu_o);
    double s2 = 0, sr2 = 0, mx2 = 0; for (long i = 0; i < (long)nt * HCD; i++) { double d = fabs((double)ref_o[i] - gpu_o[i]); if (d > mx2) mx2 = d; s2 += d*d; sr2 += (double)ref_o[i]*ref_o[i]; }
    printf("[HC-POST] rc=%d rel=%.2e max=%.2e\n", r2, sqrt(s2 / sr2), mx2);
    double t0 = now(); for (int t = 0; t < nt; t++) llm_dsv4_block_pre(&c, fn, scale, base, nw, res + (size_t)t * HCD, ref_n + (size_t)t * D, rp + (size_t)t * HC, rc + (size_t)t * HC * HC);
    double t1 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_hc_pre_batch(be, nt, HC, D, c.n_hc_sinkhorn_iter, c.hc_eps, c.rms_eps, res, fn, scale, base, nw, gpu_n, sp);
    double t2 = now();
    printf("tempo pre: CPU %.2f ms | GPU %.2f ms per chunk di %d token\n", (t1 - t0) * 1000, (t2 - t1) * 200, nt);
    int ok = r1 == 0 && r2 == 0 && sqrt(sn / sr) < 1e-4 && sqrt(s2 / sr2) < 1e-5 && sqrt(sp_d / spr) < 1e-4;
    printf(ok ? "PASS mHC batched\n" : "FAIL mHC batched\n");
    return ok ? 0 : 1;
}
