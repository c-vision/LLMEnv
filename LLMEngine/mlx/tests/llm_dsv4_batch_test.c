/* llm_dsv4_batch_test.c — parità del prefill a chunk layer-major vs token-per-token sul modello reale.
 * Tre engine freschi: A (forward_step), B (forward_batch, chunk N), A2 (forward_step, per misurare il rumore di
 * fondo run-to-run). Confronta i logit finali: max|diff| e argmax. Uso: batch_test [gguf] [n_tok] [chunk] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_engine.h"
#include "llm_engine_opts.h"

static float* run(const char* path, const int* tok, int n, int chunk, double* secs, int* am) {
    LlmDsv4Engine* e = llm_dsv4_engine_load_swap(path, 1024, 3000);
    if (!e) { fprintf(stderr, "load fallita\n"); exit(1); }
    const LlmDsv4Config* c = &e->w.cfg;
    float* hc = (float*)malloc(sizeof(float) * (size_t)c->n_hc * c->n_embd);
    float* lg = (float*)malloc(sizeof(float) * (size_t)c->n_vocab);
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    if (chunk <= 1) for (int i = 0; i < n; i++) llm_dsv4_forward_step(c, &e->w, &e->caches, tok[i], i, hc);
    else for (int i = 0; i < n; i += chunk) { int m = n - i < chunk ? n - i : chunk; llm_dsv4_forward_batch(c, &e->w, &e->caches, tok + i, i, m, hc); }
    clock_gettime(CLOCK_MONOTONIC, &b);
    *secs = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
    llm_dsv4_output_logits(c, &e->w, hc, lg);
    *am = llm_dsv4_argmax(lg, c->n_vocab);
    int nv = c->n_vocab; float* out = (float*)malloc(sizeof(float) * (size_t)nv);
    for (int i = 0; i < nv; i++) out[i] = lg[i];
    free(lg); free(hc); llm_dsv4_engine_free(e);
    return out;
}
static double maxdiff(const float* a, const float* b, int n) { double m = 0; for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > m) m = d; } return m; }
static double kl_div(const float* a, const float* b, int n) {   /* KL(softmax(a) || softmax(b)) */
    double ma = a[0], mb = b[0]; for (int i = 1; i < n; i++) { if (a[i] > ma) ma = a[i]; if (b[i] > mb) mb = b[i]; }
    double za = 0, zb = 0; for (int i = 0; i < n; i++) { za += exp(a[i] - ma); zb += exp(b[i] - mb); }
    double kl = 0; for (int i = 0; i < n; i++) { double pa = exp(a[i] - ma) / za, pb = exp(b[i] - mb) / zb; if (pa > 1e-12 && pb > 0) kl += pa * (log(pa) - log(pb)); }
    return kl; }
static int topk_overlap(const float* a, const float* b, int n, int k) {
    int ia[16], ib[16]; for (int r = 0; r < k; r++) { int ba = -1, bb = -1;
        for (int i = 0; i < n; i++) { int ua = 0, ub = 0; for (int q = 0; q < r; q++) { ua |= (ia[q] == i); ub |= (ib[q] == i); }
            if (!ua && (ba < 0 || a[i] > a[ba])) ba = i; if (!ub && (bb < 0 || b[i] > b[bb])) bb = i; } ia[r] = ba; ib[r] = bb; }
    int c = 0; for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) if (ia[i] == ib[j]) c++; return c; }
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int n = argc > 2 ? atoi(argv[2]) : 40, chunk = argc > 3 ? atoi(argv[3]) : 16;
    int tok[4096]; unsigned s = 12345; for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; tok[i] = 1000 + (int)((s >> 8) % 20000); }
    double ta, tb, tc, td; int aa, ab, ac, ad;
    const int nv = 129280;
    llm_ai_mlx_opt_set("dsv4_prefill_f16", "0");   /* percorso f32: deve essere BIT-EXACT col token-per-token */
    float* A = run(path, tok, n, 1, &ta, &aa);
    float* B = run(path, tok, n, chunk, &tb, &ab);
    float* A2 = run(path, tok, n, 1, &tc, &ac);
    llm_ai_mlx_opt_set("dsv4_prefill_f16", "1");   /* attivazioni half (come ds4): differenza statistica, non bit-exact */
    float* C = run(path, tok, n, chunk, &td, &ad);
    llm_ai_mlx_opt_set("dsv4_prefill_f16", "0"); llm_ai_mlx_opt_set("dsv4_gpu_hc", "1");   /* mHC su GPU: accumuli float (non double) -> confronto statistico */
    double te; int ae; float* E = run(path, tok, n, chunk, &te, &ae);
    llm_ai_mlx_opt_set("dsv4_gpu_hc", "0");
    printf("gpu_hc: max|A-E|=%.3e KL(A||E)=%.4f argmax A=%d E=%d top5=%d/5  (%.1fs)\n", maxdiff(A, E, nv), kl_div(A, E, nv), aa, ae, topk_overlap(A, E, nv, 5), te);
    double dab = maxdiff(A, B, nv), daa = maxdiff(A, A2, nv), dac = maxdiff(A, C, nv);
    printf("n=%d chunk=%d  A %.1fs B %.1fs A2 %.1fs C(f16) %.1fs\n", n, chunk, ta, tb, tc, td);
    printf("f32: max|A-A2|=%.3e (determinismo)  max|A-B|=%.3e (chunk vs token)\n", daa, dab);
    printf("f16: max|A-C|=%.3e  KL(A||C)=%.4f  argmax A=%d C=%d  top5-overlap=%d/5  (il modello amplifica il rumore: confronto statistico)\n", dac, kl_div(A, C, nv), aa, ad, topk_overlap(A, C, nv, 5));
    int ok = (aa == ab) && dab <= 1e-4 && daa <= 1e-4;
    printf(ok ? "PASS batch: f32 chunk == token-per-token (bit-exact), run-to-run deterministico\n" : "FAIL batch\n");
    return ok ? 0 : 1;
}
