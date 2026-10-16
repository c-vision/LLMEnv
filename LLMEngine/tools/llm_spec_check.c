/* llm_spec_check.c — misura la speculativa su un PROMPT NATURALE in un solo comando:
 * tokenizza il testo (LLMToken, nativo) -> AR (greedy) vs DFlash2 (tok/s) + parity.
 * Nato per evitare il prompt SINTETICO di decode_profile (id 100,101,...) che falsa l'acceptance.
 *
 * Uso:
 *   llm_spec_check <dylib> <target_dir> <draft_dir> <N> "testo del prompt" [reps]
 *   echo "testo" | llm_spec_check <dylib> <target_dir> <draft_dir> <N> - [reps]
 * reps: ripete la misura su sessioni FRESCHE e riporta il MIN (meno rumoroso su GPU condivisa);
 *       con reps>1 stampa anche la media AR/DF.
 *
 * Build:
 *   cc -O2 -I LLMToken -o llm_spec_check LLMEngine/tools/llm_spec_check.c \
 *      -L build/LLMToken -lllmtoken -Wl,-rpath,$(pwd)/build/LLMToken
 */
#include "llmtoken.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach_time.h>

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

typedef void* M; typedef void* S; typedef void* DF;

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "uso: %s <dylib> <target_dir> <draft_dir> <N> \"prompt\"|-\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    M  (*load)(const char*)            = dlsym(h, "llm_ai_mlx_load_model");
    S  (*create)(M)                    = dlsym(h, "llm_ai_mlx_create_session");
    int(*append)(S, const int32_t*, int) = dlsym(h, "llm_ai_mlx_session_append");
    int(*gen)(S, int, int32_t, int32_t*) = dlsym(h, "llm_ai_mlx_session_generate");
    DF (*dfcreate)(M, const char*)     = dlsym(h, "llm_ai_mlx_dflash2_create");
    int(*dfgen)(DF, S, int32_t, int, int32_t*) = dlsym(h, "llm_ai_mlx_dflash2_generate");
    int(*optset)(const char*, const char*) = dlsym(h, "llm_ai_mlx_opt_set");
    if (!load || !create || !append || !gen || !dfcreate || !dfgen) { fprintf(stderr, "dlsym fallito\n"); return 3; }
    if (optset) optset("spec_ngram", "0");   /* AR = greedy puro */

    int N = atoi(argv[4]);
    /* tokenizer da <target_dir>/tokenizer.json */
    char tp[4096]; snprintf(tp, sizeof tp, "%s/tokenizer.json", argv[2]);
    llmtoken_handle tk = llmtoken_load(tp);
    if (!tk) { fprintf(stderr, "tokenizer: %s non caricato\n", tp); return 3; }

    char* text = NULL;
    if (strcmp(argv[5], "-") == 0) {
        size_t cap = 4096, n = 0; text = (char*)malloc(cap); size_t r;
        while ((r = fread(text + n, 1, cap - n - 1, stdin)) > 0) { n += r; if (n + 1 >= cap) { cap *= 2; text = (char*)realloc(text, cap); } }
        text[n] = '\0';
    } else text = strdup(argv[5]);

    int need = llmtoken_encode(tk, text, NULL, 0);
    if (need < 0) { fprintf(stderr, "encode errore\n"); return 4; }
    int32_t* ids = (int32_t*)malloc((size_t)need * sizeof(int32_t));
    int n = llmtoken_encode(tk, text, ids, need);
    printf("# prompt: %d token\n", n);

    M m = load(argv[2]);
    if (!m) { fprintf(stderr, "load_model NULL\n"); return 5; }

    int32_t* ar = (int32_t*)malloc((size_t)(N + 8) * sizeof(int32_t));
    int32_t* df = (int32_t*)malloc((size_t)(N + 8) * sizeof(int32_t));
    if (!ar || !df) { fprintf(stderr, "malloc\n"); return 5; }
    int reps = (argc > 6) ? atoi(argv[6]) : 1;
    if (reps < 1) reps = 1; if (reps > 9) reps = 9;
    DF d = dfcreate(m, argv[3]);           /* il draft si carica UNA volta */
    if (!d) { fprintf(stderr, "dflash2_create NULL\n"); return 5; }

    double arm[9], dfm[9]; int na = 0, nd = 0, first = -1;
    for (int r = 0; r < reps; r++) {
        S sa = create(m); append(sa, ids, n);
        double a0 = now_ms();
        int nna = gen(sa, N, -1, ar);
        arm[r] = now_ms() - a0;
        S sd = create(m); append(sd, ids, n);
        double d0 = now_ms();
        int nnd = dfgen(d, sd, -1, N, df);
        dfm[r] = now_ms() - d0;
        if (r == 0) { na = nna; nd = nnd; int mn = nna < nnd ? nna : nnd;
            for (int i = 0; i < mn; i++) if (ar[i] != df[i]) { first = i; break; } }
    }
    double sa_ = 0, sd_ = 0;               /* min = meno rumoroso (best-case) su macchina condivisa */
    for (int r = 0; r < reps; r++) { sa_ = r ? (arm[r] < sa_ ? arm[r] : sa_) : arm[0]; sd_ = r ? (dfm[r] < sd_ ? dfm[r] : sd_) : dfm[0]; }
    double a_sum = 0, d_sum = 0; for (int r = 0; r < reps; r++) { a_sum += arm[r]; d_sum += dfm[r]; }
    double a_avg = a_sum / reps, d_avg = d_sum / reps;
    printf("reps=%d\n", reps);
    printf("AR  %7.1f ms  %6.1f tok/s  (%d)\n", sa_, na > 0 ? na * 1e3 / sa_ : 0, na);
    printf("DF  %7.1f ms  %6.1f tok/s  (%d)  guadagno(min) %+.0f%%\n", sd_, nd > 0 ? nd * 1e3 / sd_ : 0, nd,
           sa_ > 0 && sd_ > 0 ? (sa_ / sd_ - 1.0) * 100.0 : 0);
    if (reps > 1) printf("    media AR %.1f / DF %.1f ms  (DF min %.1f max %.1f)\n", a_avg, d_avg, sd_, dfm[0] > dfm[1] ? dfm[0] : dfm[1]);
    printf("PARITA: %s (primo diff a %d)\n", (first < 0 && na == nd) ? "OK" : "DIVERGE", first);
    free(ids); free(text); llmtoken_free(tk);
    return 0;
}
