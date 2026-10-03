/* llm_dsv4_prefill_bench.c — prefill puro a chunk su N token sintetici (un solo caricamento), t/s.
 * Uso: prefill_bench [gguf] [n_tok] [chunk] [cache_mb] [tokfile]   (aggira il limite di lunghezza del prompt del CLI) */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "llm_dsv4_engine.h"
#include "llm_engine_opts.h"
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int n = argc > 2 ? atoi(argv[2]) : 4096, chunk = argc > 3 ? atoi(argv[3]) : 512;
    if (argc > 4) llm_ai_mlx_opt_set("dsv4_cache_mb", argv[4]);
    int* tok = (int*)malloc(sizeof(int) * (size_t)n); unsigned s = 12345;
    for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; tok[i] = 1000 + (int)((s >> 8) % 20000); }
    if (argc > 5 && argv[5][0]) { FILE* f = fopen(argv[5], "r"); int v, k = 0;   /* token id separati da spazio (es. da `ds4 --dump-tokens`) */
        if (!f) fprintf(stderr, "tokfile %s non apribile: uso token sintetici\n", argv[5]);
        else { while (k < n && fscanf(f, "%d", &v) == 1) tok[k++] = v;
               fclose(f); if (k < n) { fprintf(stderr, "solo %d token\n", k); n = k; } } }
    if (argc > 6) llm_ai_mlx_opt_set("dsv4_prefetch_hot", argv[6]);   /* item B: prefetch top-K hot del layer L+1 (0=off) */
    LlmDsv4Engine* e = llm_dsv4_engine_load_swap(path, n + 64, 3000);
    if (!e) { fprintf(stderr, "load fallita\n"); return 1; }
    const LlmDsv4Config* c = &e->w.cfg;
    float* hc = (float*)malloc(sizeof(float) * (size_t)c->n_hc * c->n_embd);
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < n; i += chunk) { int m = n - i < chunk ? n - i : chunk;
        llm_dsv4_forward_batch(c, &e->w, &e->caches, tok + i, i, m, hc);
        clock_gettime(CLOCK_MONOTONIC, &b);
        printf("  %d/%d  %.1fs\n", i + m, n, (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9); fflush(stdout); }
    double dt = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
    printf("prefill n=%d chunk=%d: %.1fs = %.1f t/s\n", n, chunk, dt, n / dt);
    llm_dsv4_engine_free(e); return 0;
}
