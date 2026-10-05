/* tools/llmengine/dflash_parity.c — parita' DFlash2 vs greedy AR (unita'-first, senza dispatch).
 * Carica il target, due sessioni (stesso prompt): una genera AR (session_generate, n-gram
 * bit-exact), l'altra usa DFlash2 (dflash2_generate). Confronta gli id.
 * Uso: /tmp/dflashparity <dylib> <target_dir> <draft_dir> <id,id,...> <N> */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef void* model_h; typedef void* sess_h; typedef void* df_h;

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "uso: %s <dylib> <target> <draft> <ids> <N>\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    model_h (*load)(const char*) = dlsym(h, "llm_ai_mlx_load_model");
    sess_h  (*create)(model_h) = dlsym(h, "llm_ai_mlx_create_session");
    int (*append)(sess_h, const int32_t*, int) = dlsym(h, "llm_ai_mlx_session_append");
    int (*gen)(sess_h, int, int32_t, int32_t*) = dlsym(h, "llm_ai_mlx_session_generate");
    df_h (*dfcreate)(model_h, const char*) = dlsym(h, "llm_ai_mlx_dflash2_create");
    int (*dfgen)(df_h, sess_h, int32_t, int, int32_t*) = dlsym(h, "llm_ai_mlx_dflash2_generate");
    int (*optset)(const char*, const char*) = dlsym(h, "llm_ai_mlx_opt_set");
    if (optset) optset("spec_ngram", "0");   /* AR PURO per il baseline */
    if (!load || !create || !append || !gen || !dfcreate || !dfgen) { fprintf(stderr, "dlsym fallito\n"); return 3; }

    int32_t ids[4096]; int n = 0; { const char* p = argv[4]; char* e = NULL;
        while (*p && n < 4096) { long v = strtol(p, &e, 10); if (e == p) break; ids[n++] = (int32_t)v; p = e; if (*p == ',') p++; } }
    int N = atoi(argv[5]);

    model_h m = load(argv[2]);
    if (!m) { fprintf(stderr, "load_model NULL\n"); return 4; }

    /* AR (session_generate: n-gram bit-exact su questo target senza head MTP) */
    sess_h sa = create(m); append(sa, ids, n);
    int32_t ar[1024]; struct timespec a0, a1; clock_gettime(CLOCK_MONOTONIC, &a0);
    int na = gen(sa, N, -1, ar);
    clock_gettime(CLOCK_MONOTONIC, &a1);
    double ar_ms = (a1.tv_sec-a0.tv_sec)*1e3+(a1.tv_nsec-a0.tv_nsec)/1e6;
    printf("AR n=%d: ", na); for (int i = 0; i < na; i++) printf("%d ", ar[i]); printf("\n");
    printf("AR %.1f ms (%.1f tok/s)\n", ar_ms, na>0?na*1e3/ar_ms:0);

    /* DFlash2 */
    sess_h sd = create(m); append(sd, ids, n);
    df_h d = dfcreate(m, argv[3]);
    if (!d) { fprintf(stderr, "dflash2_create NULL\n"); return 5; }
    int32_t df[1024]; struct timespec d0, d1; clock_gettime(CLOCK_MONOTONIC, &d0);
    int nd = dfgen(d, sd, -1, N, df);
    clock_gettime(CLOCK_MONOTONIC, &d1);
    double df_ms = (d1.tv_sec-d0.tv_sec)*1e3+(d1.tv_nsec-d0.tv_nsec)/1e6;
    printf("DF n=%d: ", nd); for (int i = 0; i < nd; i++) printf("%d ", df[i]); printf("\n");
    printf("DF %.1f ms (%.1f tok/s)\n", df_ms, nd>0?nd*1e3/df_ms:0);

    int min = na < nd ? na : nd, first = -1;
    for (int i = 0; i < min; i++) if (ar[i] != df[i]) { first = i; break; }
    printf("PARITA: %s (primo diff a %d, na=%d nd=%d)\n", (first < 0 && na == nd) ? "OK" : "DIVERGE", first, na, nd);
    if (first >= 0) { printf("  contesto AR: "); for (int i = (first>4?first-4:0); i < first+2 && i < na; i++) printf("%d ", ar[i]); printf("\n  contesto DF: "); for (int i = (first>4?first-4:0); i < first+2 && i < nd; i++) printf("%d ", df[i]); printf("\n"); }
    return first < 0 && na == nd ? 0 : 1;
}
