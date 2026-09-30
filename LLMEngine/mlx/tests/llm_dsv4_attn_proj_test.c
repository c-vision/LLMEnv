/* llm_dsv4_attn_proj_test.c — proiezioni Q-LoRA+KV del chunk via mul_mm_id Q8_0 vs catena per-token verificata
 * (attn_proj_fused_batch), pesi REALI. Atteso: errore relativo ~1e-3 (attivazioni half). Uso: attn_proj_test [gguf] [layer] [nt] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_engine.h"
#include "llm_dsv4_metal.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static unsigned st = 13579u;
static float fr(void) { st = st * 1103515245u + 12345u; return (float)((st >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }
static double rel(const float* a, const float* b, size_t n, double* mx) { double ss = 0, sr = 0; *mx = 0; for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > *mx) *mx = d; ss += d*d; sr += (double)a[i]*a[i]; } return sqrt(ss / (sr > 0 ? sr : 1)); }
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int il = argc > 2 ? atoi(argv[2]) : 5, nt = argc > 3 ? atoi(argv[3]) : 64;
    LlmDsv4Engine* e = llm_dsv4_engine_load(path, 256);
    if (!e) { printf("load fallita\n"); return 1; }
    const LlmDsv4Config* c = &e->w.cfg; const LlmDsv4PackedExperts* pe = &e->w.exps[il]; const LlmDsv4LayerWeights* L = &e->w.layers[il];
    const int lq = c->n_lora_q, embd = c->n_embd, qd = c->n_head * c->n_head_dim, hd = c->n_head_dim;
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    int compressed = llm_dsv4_layer_compress_ratio(c, il) != 0;
    float fb = (compressed && c->compress_rope_freq_base > 0.0f) ? c->compress_rope_freq_base : c->rope_freq_base;
    float fs = (!compressed || c->rope_scale_factor <= 0.0f) ? 1.0f : 1.0f / c->rope_scale_factor;
    float ef = (compressed && c->rope_scale_factor > 1.0f) ? 1.0f : 0.0f, af = 1.0f;
    if (ef != 0.0f && fs > 0.0f) af /= 1.0f + 0.1f * logf(1.0f / fs);
    float rc_[7] = { fb, fs, ef, af, c->rope_yarn_beta_fast, c->rope_yarn_beta_slow, (float)(compressed ? c->rope_orig_ctx : 0) };
    size_t qar = llm_ggml_nbytes(pe->q_a_type, embd), kvr = llm_ggml_nbytes(pe->kv_type, embd), qbr = llm_ggml_nbytes(pe->q_b_type, lq);
    float* x = malloc(sizeof(float) * (size_t)nt * embd);
    for (int i = 0; i < nt * embd; i++) x[i] = fr();
    float *q1 = malloc(4*(size_t)nt*qd), *k1 = malloc(4*(size_t)nt*hd), *n1 = malloc(4*(size_t)nt*lq), *q2 = malloc(4*(size_t)nt*qd), *k2 = malloc(4*(size_t)nt*hd), *n2 = malloc(4*(size_t)nt*lq);
    int pos0 = 100;
    int r1 = llm_dsv4_metal_attn_proj_fused_batch(be, nt, pe->q_a, qar, pe->kv, kvr, pe->q_b, qbr, x, embd, L->attn_q_a_norm, L->attn_kv_a_norm, lq, hd, qd, c->n_head, c->n_rot, rc_, pos0, c->rms_eps, q1, k1, n1);
    int r2 = llm_dsv4_metal_attn_proj_mm_batch(be, nt, pe->q_a, qar, pe->kv, kvr, pe->q_b, qbr, x, embd, L->attn_q_a_norm, L->attn_kv_a_norm, lq, hd, qd, c->n_head, c->n_rot, rc_, pos0, c->rms_eps, q2, k2, n2);
    double mq, mk, mn; double rq = rel(q1, q2, (size_t)nt*qd, &mq), rk = rel(k1, k2, (size_t)nt*hd, &mk), rn = rel(n1, n2, (size_t)nt*lq, &mn);
    printf("[PROJ] layer=%d nt=%d rc=%d/%d  q rel=%.2e max=%.2e | kv rel=%.2e max=%.2e | qrn rel=%.2e max=%.2e\n", il, nt, r1, r2, rq, mq, rk, mk, rn, mn);
    double t0 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_attn_proj_fused_batch(be, nt, pe->q_a, qar, pe->kv, kvr, pe->q_b, qbr, x, embd, L->attn_q_a_norm, L->attn_kv_a_norm, lq, hd, qd, c->n_head, c->n_rot, rc_, pos0, c->rms_eps, q1, k1, n1);
    double t1 = now(); for (int k = 0; k < 5; k++) llm_dsv4_metal_attn_proj_mm_batch(be, nt, pe->q_a, qar, pe->kv, kvr, pe->q_b, qbr, x, embd, L->attn_q_a_norm, L->attn_kv_a_norm, lq, hd, qd, c->n_head, c->n_rot, rc_, pos0, c->rms_eps, q2, k2, n2);
    double t2 = now();
    printf("tempo (caldo): per-token-chain %.2f ms | mul_mm_id %.2f ms per chunk di %d token\n", (t1 - t0) * 200, (t2 - t1) * 200, nt);
    int ok = r1 == 0 && r2 == 0 && rq < 5e-3 && rk < 5e-3 && rn < 5e-3;
    printf(ok ? "PASS attn_proj mm_id\n" : "FAIL attn_proj mm_id\n");
    return ok ? 0 : 1;
}
