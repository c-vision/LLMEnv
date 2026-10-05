/* tools/llmengine/dflash_stream.c — A1: DFlash2 a contesto PERSISTENTE. Confronta, TUTTO A WARM:
 *  (1) AR puro, (2) DFlash2 ONE-SHOT (N), (3) DFlash2 STREAMING (chunk), (4) wiring session_generate.
 * Warmup COMPLETO (compila i kernel del target E del draft) prima di ogni misura.
 * Uso: /tmp/dflashstream <dylib> <target_dir> <draft_dir> <id,id,...> <N> [chunk] */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef void* model_h; typedef void* sess_h; typedef void* df_h;

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

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
    df_h d = dfcreate(m, argv[3]);
    if (!d) { fprintf(stderr, "dflash2_create NULL\n"); return 5; }

    /* WARMUP COMPLETO: compila i kernel del target (AR) e del draft (DFlash2). */
    { sess_h sw = create(m); append(sw, ids, n); int32_t tw[256];
      gen(sw, 64, -1, tw);
      dfgen(d, sw, -1, 64, tw); }

    /* AR baseline */
    sess_h sa = create(m); append(sa, ids, n);
    int32_t ar[4096]; double a0 = now_ms();
    int na = gen(sa, N, -1, ar);
    double ar_ms = now_ms() - a0;
    printf("AR n=%d %.1f ms (%.1f tok/s)\n", na, ar_ms, na>0?na*1e3/ar_ms:0);

    /* DFlash2 ONE-SHOT */
    sess_h sd = create(m); append(sd, ids, n);
    int32_t df[4096]; double d0 = now_ms();
    int nd = dfgen(d, sd, -1, N, df);
    double df_ms = now_ms() - d0;
    printf("DF1 n=%d %.1f ms (%.1f tok/s)\n", nd, df_ms, nd>0?nd*1e3/df_ms:0);
    {   /* F20: acceptance (draft proposti vs accettati) come metrica di primo livello. */
        int (*specstats)(sess_h, long long*, long long*, double*) =
            dlsym(h, "llm_ai_mlx_session_spec_stats");
        long long sdd = 0, sda = 0; double sdr = 0;
        if (specstats && specstats(sd, &sdd, &sda, &sdr) == 0)
            printf("DF1 acceptance: %lld/%lld = %.3f\n", sda, sdd, sdr);
    }
    { int minb = na < nd ? na : nd, fb = -1;
      for (int i = 0; i < minb; i++) if (ar[i] != df[i]) { fb = i; break; }
      printf("PARITA AR vs DF1: %s (primo diff %d, na=%d nd=%d)\n", (fb<0 && na==nd)?"OK":"DIVERGE", fb, na, nd);
      if (fb >= 0) printf("   ar[%d]=%d df[%d]=%d\n", fb, ar[fb], fb, df[fb]);
    }

    /* DFlash2 STREAMING (continuazione) */
    sess_h ss = create(m); append(ss, ids, n);
    int32_t ds[4096]; int ns = 0; int calls = 0;
    double s0 = now_ms();
    while (ns < N && calls < 4096) {
        int want = N - ns; if (want > chunk) want = chunk;
        int32_t buf[256];
        int got = dfgen(d, ss, -1, want, buf);
        calls++;
        if (got <= 0) break;
        for (int i = 0; i < got && ns < 4096; i++) ds[ns++] = buf[i];
    }
    double ds_ms = now_ms() - s0;
    printf("DFS n=%d %.1f ms (%.1f tok/s) chiamate=%d chunk=%d\n", ns, ds_ms, ns>0?ns*1e3/ds_ms:0, calls, chunk);

    int min = nd < ns ? nd : ns, first = -1;
    for (int i = 0; i < min; i++) if (df[i] != ds[i]) { first = i; break; }
    printf("PARITA one-shot vs streaming: %s (primo diff a %d, nd=%d ns=%d)\n",
           (first < 0 && nd == ns) ? "OK" : "DIVERGE", first, nd, ns);

    /* WIRING: session_generate con l'opzione dflash_model (dispatch spec streaming interno). */
    if (optset) optset("dflash_model", argv[3]);
    sess_h sg = create(m); append(sg, ids, n);
    int32_t dg[4096]; double g0 = now_ms();
    int ng = gen(sg, N, -1, dg);
    double dgw_ms = now_ms() - g0;
    printf("DFW n=%d %.1f ms (%.1f tok/s) [wiring session_generate]\n", ng, dgw_ms, ng>0?ng*1e3/dgw_ms:0);
    int minw = na < ng ? na : ng, fw = -1;
    for (int i = 0; i < minw; i++) if (ar[i] != dg[i]) { fw = i; break; }
    printf("PARITA AR vs wiring: %s\n", (fw < 0 && na == ng) ? "OK" : "DIVERGE");
    return (first < 0 && nd == ns && fw < 0 && na == ng) ? 0 : 1;
}
