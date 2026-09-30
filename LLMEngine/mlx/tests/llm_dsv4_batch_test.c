/* llm_dsv4_batch_test.c — parità del prefill a chunk layer-major vs token-per-token sul modello reale.
 * Tre engine freschi: A (forward_step), B (forward_batch, chunk N), A2 (forward_step, per misurare il rumore di
 * fondo run-to-run). Confronta i logit finali: max|diff| e argmax. Uso: batch_test [gguf] [n_tok] [chunk] */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "llm_dsv4_engine.h"
#include "llm_engine_opts.h"

static int g_cap = 3000;   /* bundle nella cache esperti (argv[5]) */
static float* run(const char* path, const int* tok, int n, int chunk, double* secs, int* am) {
    LlmDsv4Engine* e = llm_dsv4_engine_load_swap(path, 1024, g_cap);
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
    /* UN CASO ALLA VOLTA (ogni caso = piu' caricamenti del modello, quindi mai tutti insieme):
     *   f32  : token-per-token vs chunk, opzioni f32 -> deve essere BIT-EXACT (2 caricamenti)           [default]
     *   half : f32 vs attivazioni half (dsv4_prefill_f16=1)  -> statistico (2 caricamenti)
     *   hc   : mHC CPU vs GPU (dsv4_gpu_hc=1)               -> statistico (2 caricamenti)
     *   raw  : attenzione layer 0-1 GPU vs CPU              -> statistico (2 caricamenti) */
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int n = argc > 2 ? atoi(argv[2]) : 40, chunk = argc > 3 ? atoi(argv[3]) : 16;
    const char* cs = argc > 4 ? argv[4] : "f32"; if (argc > 5) g_cap = atoi(argv[5]);
    int tok[4096]; unsigned s = 12345; for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; tok[i] = 1000 + (int)((s >> 8) % 20000); }
    const int nv = 129280; double ta, tb; int aa, ab;
    llm_ai_mlx_opt_set("dsv4_prefill_f16", "0"); llm_ai_mlx_opt_set("dsv4_gpu_hc", "0"); llm_ai_mlx_opt_set("dsv4_gpu_raw_attn", "1"); llm_ai_mlx_opt_set("dsv4_prefill_mm", "0");
    float* A = run(path, tok, n, 1, &ta, &aa);                     /* riferimento: token-per-token, percorso f32 */
    if (!strcmp(cs, "half")) llm_ai_mlx_opt_set("dsv4_prefill_f16", "1");
    if (!strcmp(cs, "hc"))   llm_ai_mlx_opt_set("dsv4_gpu_hc", "1");
    if (!strcmp(cs, "raw"))  llm_ai_mlx_opt_set("dsv4_gpu_raw_attn", "0");
    float* B = run(path, tok, n, chunk, &tb, &ab);                 /* chunk con la variante del caso */
    double dab = maxdiff(A, B, nv);
    printf("caso=%s n=%d chunk=%d  A %.1fs  B %.1fs | argmax A=%d B=%d | max|A-B|=%.3e KL(A||B)=%.4f top5=%d/5\n", cs, n, chunk, ta, tb, aa, ab, dab, kl_div(A, B, nv), topk_overlap(A, B, nv, 5));
    int ok = !strcmp(cs, "f32") ? (aa == ab && dab <= 1e-4) : (aa == ab);
    printf(ok ? "PASS caso %s\n" : "FAIL caso %s\n", cs);
    return ok ? 0 : 1;
}
