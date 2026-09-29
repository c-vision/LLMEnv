/* llm_dsv4_batch_test.c — parità del prefill a chunk layer-major vs token-per-token sul modello reale.
 * Tre engine freschi: A (forward_step), B (forward_batch, chunk N), A2 (forward_step, per misurare il rumore di
 * fondo run-to-run). Confronta i logit finali: max|diff| e argmax. Uso: batch_test [gguf] [n_tok] [chunk] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_engine.h"

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
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int n = argc > 2 ? atoi(argv[2]) : 40, chunk = argc > 3 ? atoi(argv[3]) : 16;
    int tok[4096]; unsigned s = 12345; for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; tok[i] = 1000 + (int)((s >> 8) % 20000); }
    double ta, tb, tc; int aa, ab, ac;
    float* A = run(path, tok, n, 1, &ta, &aa);
    float* B = run(path, tok, n, chunk, &tb, &ab);
    float* A2 = run(path, tok, n, 1, &tc, &ac);
    const int nv = 129280;
    double dab = maxdiff(A, B, nv), daa = maxdiff(A, A2, nv);
    printf("n=%d chunk=%d  A %.1fs  B %.1fs  A2 %.1fs | argmax A=%d B=%d A2=%d | max|A-B|=%.3e  max|A-A2|=%.3e\n",
           n, chunk, ta, tb, tc, aa, ab, ac, dab, daa);
    int ok = (aa == ab) && dab <= 1e-2 + 4 * daa;
    printf(ok ? "PASS batch: logit e argmax equivalenti al token-per-token\n" : "FAIL batch\n");
    return ok ? 0 : 1;
}
