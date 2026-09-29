/* tools/dflash_stream.c — A1: DFlash2 a contesto PERSISTENTE. Confronta
 *  (1) AR puro, (2) DFlash2 ONE-SHOT (N), (3) DFlash2 STREAMING (loop di chiamate con chunk C,
 *  che esercita la continuazione: no re-capture/reset).
 * Uso: /tmp/dflashstream <dylib> <target_dir> <draft_dir> <id,id,...> <N> [chunk] */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef void* model_h; typedef void* sess_h; typedef void* df_h;

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "uso: %s <dylib> <target> <draft> <ids> <N> [chunk]\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    model_h (*load)(const char*) = dlsym(h, "llm_ai_mlx_load_model");
    sess_h  (*create)(model_h) = dlsym(h, "llm_ai_mlx_create_session");
    int (*append)(sess_h, const int32_t*, int) = dlsym(h, "llm_ai_mlx_session_append");
    int (*gen)(sess_h, int, int32_t, int32_t*) = dlsym(h, "llm_ai_mlx_session_generate");
    df_h (*dfcreate)(model_h, const char*) = dlsym(h, "llm_ai_mlx_dflash2_create");
    int (*dfgen)(df_h, sess_h, int32_t, int, int32_t*) = dlsym(h, "llm_ai_mlx_dflash2_generate");
    int (*optset)(const char*, const char*) = dlsym(h, "llm_ai_mlx_opt_set");
    if (optset) optset("spec_ngram", "0");
    if (!load || !create || !append || !gen || !dfcreate || !dfgen) { fprintf(stderr, "dlsym fallito\n"); return 3; }

    int32_t ids[4096]; int n = 0; { const char* p = argv[4]; char* e = NULL;
        while (*p && n < 4096) { long v = strtol(p, &e, 10); if (e == p) break; ids[n++] = (int32_t)v; p = e; if (*p == ',') p++; } }
    int N = atoi(argv[5]);
    int chunk = argc > 6 ? atoi(argv[6]) : 16;
    if (chunk < 2) chunk = 2;

    model_h m = load(argv[2]);
    if (!m) { fprintf(stderr, "load_model NULL\n"); return 4; }

    /* WARMUP: la prima generate paga il load dei kernel/warm (~s). Bruciamola qui. */
    { sess_h sw = create(m); append(sw, ids, n); int32_t tw[8]; gen(sw, 4, -1, tw); }

    /* AR baseline */
    sess_h sa = create(m); append(sa, ids, n);
    int32_t ar[4096]; struct timespec a0, a1; clock_gettime(CLOCK_MONOTONIC, &a0);
    int na = gen(sa, N, -1, ar);
    clock_gettime(CLOCK_MONOTONIC, &a1);
    double ar_ms = (a1.tv_sec-a0.tv_sec)*1e3+(a1.tv_nsec-a0.tv_nsec)/1e6;
    printf("AR n=%d %.1f ms (%.1f tok/s)\n", na, ar_ms, na>0?na*1e3/ar_ms:0);

    /* DFlash2 ONE-SHOT */
    sess_h sd = create(m); append(sd, ids, n);
    df_h d = dfcreate(m, argv[3]);
    if (!d) { fprintf(stderr, "dflash2_create NULL\n"); return 5; }
    int32_t df[4096]; struct timespec d0, d1; clock_gettime(CLOCK_MONOTONIC, &d0);
    int nd = dfgen(d, sd, -1, N, df);
    clock_gettime(CLOCK_MONOTONIC, &d1);
    double df_ms = (d1.tv_sec-d0.tv_sec)*1e3+(d1.tv_nsec-d0.tv_nsec)/1e6;
    printf("DF1 n=%d %.1f ms (%.1f tok/s)\n", nd, df_ms, nd>0?nd*1e3/df_ms:0);

    /* DFlash2 STREAMING (continuazione) */
    sess_h ss = create(m); append(ss, ids, n);
    int32_t ds[4096]; int ns = 0; int calls = 0;
    struct timespec s0, s1; clock_gettime(CLOCK_MONOTONIC, &s0);
    while (ns < N && calls < 4096) {
        int want = N - ns; if (want > chunk) want = chunk;
        int32_t buf[256];
        int got = dfgen(d, ss, -1, want, buf);
        calls++;
        if (got <= 0) break;
        for (int i = 0; i < got && ns < 4096; i++) ds[ns++] = buf[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &s1);
    double ds_ms = (s1.tv_sec-s0.tv_sec)*1e3+(s1.tv_nsec-s0.tv_nsec)/1e6;
    printf("DFS n=%d %.1f ms (%.1f tok/s) chiamate=%d chunk=%d\n", ns, ds_ms, ns>0?ns*1e3/ds_ms:0, calls, chunk);

    int min = nd < ns ? nd : ns, first = -1;
    for (int i = 0; i < min; i++) if (df[i] != ds[i]) { first = i; break; }
    printf("PARITA one-shot vs streaming: %s (primo diff a %d, nd=%d ns=%d)\n",
           (first < 0 && nd == ns) ? "OK" : "DIVERGE", first, nd, ns);
    if (first >= 0) {
        printf("  ctx one-shot:"); for (int i = (first>6?first-6:0); i < first+2 && i < nd; i++) printf(" %d", df[i]); printf("\n");
        printf("  ctx stream  :"); for (int i = (first>6?first-6:0); i < first+2 && i < ns; i++) printf(" %d", ds[i]); printf("\n");
    }
    int mina = na < ns ? na : ns, fa = -1;
    for (int i = 0; i < mina; i++) if (ar[i] != ds[i]) { fa = i; break; }
    printf("PARITA AR vs streaming: %s (primo diff a %d)\n", (fa < 0 && na == ns) ? "OK" : "DIVERGE", fa);
    return (first < 0 && nd == ns) ? 0 : 1;
}
