/* decode_profile.c — F18: profilo di decode riutilizzabile (tok/s + memoria MLX).
 *
 * Misura, per il motore LLMEnv (`libllm_ai_mlx.dylib`), su un modello reale:
 *   - PRE-FILL (append) di N prompt token: tempo;
 *   - DECODE di G token (greedy, temp=0): tok/s;
 *   - memoria MLX: active / peak / cache (via `llm_ai_mlx_memory_stats`);
 * a CONTESTI crescenti (sweep) e con qualunque opzione del registro (`--opt nome=valore`).
 *
 * Pensato per la metodologia del todo Splash: MISURARE prima di proporre. Confronta due
 * configurazioni (es. kvquant off vs 8:8192, o dflash off vs on) a parita' di contesto.
 *
 * Uso:
 *   decode_profile <dylib> <modello> <nprompt|a,b,c> <ngen> [--opt nome=valore ...] [--quiet]
 *
 * Esempi:
 *   decode_profile $DYLIB ~/ai/Models/Llama-3.2-1B-Instruct-4bit 2048,8192,16384 64 \
 *       --opt spec_ngram=0 --opt kvquant=8:512 --quiet
 *   decode_profile $DYLIB ~/ai/Models/Qwen3.8-27B-8bit 4096 64 \
 *       --opt dflash_model=~/ai/Models/Qwen3.8-27B-DFlash2
 *
 * Nota: temp=0 e spec_ngram=0 sono forzati se non specificati (misura greedy riproducibile).
 * Compila: cc -O2 -o decode_profile decode_profile.c
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void* M; typedef void* S;

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "uso: %s <dylib> <modello> <nprompt|a,b,c> <ngen> [--opt k=v ...] [--quiet]\n", argv[0]);
        return 2;
    }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    M   (*load)(const char*)                 = dlsym(h, "llm_ai_mlx_load_model");
    S   (*create)(M)                         = dlsym(h, "llm_ai_mlx_create_session");
    int (*append)(S, const int32_t*, int)    = dlsym(h, "llm_ai_mlx_session_append");
    int (*gen)(S, int, int32_t, int32_t*)    = dlsym(h, "llm_ai_mlx_session_generate");
    int (*optset)(const char*, const char*)  = dlsym(h, "llm_ai_mlx_opt_set");
    int (*mem)(size_t*, size_t*, size_t*)    = dlsym(h, "llm_ai_mlx_memory_stats");
    int (*destroy)(S)                        = dlsym(h, "llm_ai_mlx_destroy_session");
    if (!load || !create || !append || !gen || !mem) { fprintf(stderr, "dlsym fallito\n"); return 3; }

    const char* model = argv[2];
    int NG = atoi(argv[4]);
    if (NG <= 0) NG = 64;
    int quiet = 0;
    /* Opzioni del registro: default riproducibile (greedy, niente n-gram) salvo override. */
    int have_temp = 0, have_spec = 0;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--quiet")) { quiet = 1; continue; }
        if (!strcmp(argv[i], "--opt") && i + 1 < argc) {
            const char* kv = argv[++i]; char* eq = strchr(kv, '=');
            if (!eq || !optset) continue;
            char name[128]; size_t n = (size_t)(eq - kv); if (n >= sizeof name) n = sizeof name - 1;
            memcpy(name, kv, n); name[n] = '\0';
            if (!strcmp(name, "temp")) have_temp = 1;
            if (!strcmp(name, "spec_ngram")) have_spec = 1;
            optset(name, eq + 1);
        }
    }
    if (optset && !have_temp) optset("temp", "0");
    if (optset && !have_spec) optset("spec_ngram", "0");

    M m = load(model);
    if (!m) { fprintf(stderr, "load_model NULL\n"); return 4; }

    printf("# decode_profile  modello=%s  gen=%d  greedy=%d\n", model, NG, 1);
    printf("# %-10s %-12s %-12s %-10s %-8s %-8s %-8s\n",
           "nprompt", "prefill_ms", "decode_ms", "tok/s", "act_MB", "peak_MB", "cache_MB");

    char* sweep = strdup(argv[3]);
    if (!sweep) return 4;
    for (char* tok = strtok(sweep, ","); tok; tok = strtok(NULL, ",")) {
        int NP = atoi(tok);
        if (NP <= 0) continue;
        S s = create(m);
        int32_t* ids = (int32_t*)malloc((size_t)NP * sizeof(int32_t));
        for (int i = 0; i < NP; i++) ids[i] = 100 + (i % 2000);
        double t0 = now_ms();
        int ra = append(s, ids, NP);
        double tp = now_ms() - t0;
        free(ids);
        if (ra < 0) { fprintf(stderr, "append(%d) rc=%d\n", NP, ra); if (destroy) destroy(s); continue; }
        int32_t w[8]; gen(s, 2, -1, w);              /* warmup kernel */
        int32_t* out = (int32_t*)malloc((size_t)NG * sizeof(int32_t));
        double t1 = now_ms();
        int g = gen(s, NG, -1, out);
        double tg = now_ms() - t1;
        size_t c = 0, a = 0, p = 0; mem(&c, &a, &p);
        printf("  %-10d %-12.1f %-12.1f %-10.1f %-8.1f %-8.1f %-8.1f\n",
               NP, tp, tg, tg > 0 ? g * 1e3 / tg : 0.0,
               a / 1e6, p / 1e6, c / 1e6);
        if (!quiet) printf("      (gen=%d token)\n", g);
        {   /* F20: acceptance (draft proposti vs accettati) se la spec e' attiva. */
            int (*specstats)(S, long long*, long long*, double*) =
                dlsym(h, "llm_ai_mlx_session_spec_stats");
            long long dd = 0, aa = 0; double rr = 0;
            if (specstats && specstats(s, &dd, &aa, &rr) == 0 && dd > 0)
                printf("      acceptance %lld/%lld = %.3f\n", aa, dd, rr);
        }
        free(out);
        if (destroy) destroy(s);
    }
    free(sweep);
    return 0;
}
