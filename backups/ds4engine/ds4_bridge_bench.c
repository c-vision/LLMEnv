/* ds4_bridge_bench.c — misura rigorosa del motore ds4 via il bridge token-level.
 * Uso: ds4_bridge_bench <gguf> <n_prefill> <ctx> <stream> [tokfile] [n_decode] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ds4_bridge.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int n = argc > 2 ? atoi(argv[2]) : 2048;
    int ctx = argc > 3 ? atoi(argv[3]) : 4096;
    int stream = argc > 4 ? atoi(argv[4]) : 1;
    const char* tokf = argc > 5 ? argv[5] : NULL;
    int ndec = argc > 6 ? atoi(argv[6]) : 0;
    if (ctx < n + ndec + 8) ctx = n + ndec + 8;

    int* toks = (int*)malloc(sizeof(int) * (size_t)(n + ndec + 32));
    if (tokf && tokf[0]) { FILE* f = fopen(tokf, "r"); int k = 0, v; while (k < n && fscanf(f, "%d", &v) == 1) toks[k++] = v; fclose(f); if (k < n) n = k; }
    else { unsigned s = 12345u; for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; toks[i] = 1 + (int)((s >> 8) % 100000); } }
    if (n <= 0) { fprintf(stderr, "nessun token\n"); return 1; }

    fprintf(stderr, "[bench] load %s (ctx=%d stream=%d n=%d dec=%d)\n", path, ctx, stream, n, ndec);
    ds4_bridge* b = ds4_bridge_open(path, ctx, stream, 0);
    if (!b) { fprintf(stderr, "open FALLITA\n"); return 1; }
    int vocab = ds4_bridge_vocab(b);
    fprintf(stderr, "[bench] vocab=%d\n", vocab);

    /* warm-up (build grafo / primo submit) su 1 token */
    ds4_bridge_sync(b, toks, 1);

    double t0 = now();
    ds4_bridge_sync(b, toks, n);
    double t1 = now();
    double pf = (double)n / (t1 - t0);

    double dec_tps = 0.0, d0 = 0.0, d1 = 0.0;
    if (ndec > 0) {
        int cur = n;
        int last = ds4_bridge_argmax(b);
        d0 = now();
        for (int i = 0; i < ndec; i++) {
            toks[cur++] = last;
            ds4_bridge_sync(b, toks, cur);
            last = ds4_bridge_argmax(b);
        }
        d1 = now();
        dec_tps = (double)ndec / (d1 - d0);
    }

    printf("BRIDGE n=%d ctx=%d stream=%d | prefill %.3fs = %.1f t/s | decode %.1f t/s | last_tok=%d\n",
           n, ctx, stream, t1 - t0, pf, dec_tps, ndec > 0 ? ds4_bridge_argmax(b) : -1);
    ds4_bridge_close(b);
    free(toks);
    return 0;
}
