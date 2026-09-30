/* kv_restore_bench.c — D15 micro-bench (leggero): TTFT cold (prefill) vs warm (restore da cache).
 * Uso: kv_restore_bench <dylib> <model> <nprompt>
 * Misura: (1) prefill di P token (cold); (2) park della sessione (spill KV su kvdisk) e re-append
 * di 1 token (unpark automatico = restore dell'intera KV da disco + prefill del delta). Confronta
 * il COSTO PREFILL(P) col COSTO RESTORE. Se restore < prefill, la prefix-cache riduce il TTFT. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void* M; typedef void* S;

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "uso: %s <dylib> <model> <nprompt>\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    M   (*load)(const char*)                 = dlsym(h, "llm_ai_mlx_load_model");
    S   (*create)(M)                         = dlsym(h, "llm_ai_mlx_create_session");
    int (*append)(S, const int32_t*, int)    = dlsym(h, "llm_ai_mlx_session_append");
    int (*gen)(S, int, int32_t, int32_t*)    = dlsym(h, "llm_ai_mlx_session_generate");
    int (*optset)(const char*, const char*)  = dlsym(h, "llm_ai_mlx_opt_set");
    int (*park)(S, M)                        = dlsym(h, "llm_ai_mlx_session_park");
    int (*mem)(size_t*, size_t*, size_t*)    = dlsym(h, "llm_ai_mlx_memory_stats");
    if (!load || !create || !append || !gen || !park) { fprintf(stderr, "dlsym fallito\n"); return 3; }

    int P = atoi(argv[3]);
    if (optset) { optset("temp", "0"); optset("spec_ngram", "0"); optset("kvquant", "off");
                  optset("kvdisk", "/tmp/llm_kvrestore.scratch"); optset("kvdisk_cap_mb", "0"); }
    M m = load(argv[2]);
    if (!m) { fprintf(stderr, "load NULL\n"); return 4; }
    int32_t* ids = (int32_t*)malloc((size_t)P * sizeof(int32_t));
    for (int i = 0; i < P; i++) ids[i] = 100 + (i % 2000);
    int32_t one[2] = {0}, one2[2] = {0};

    /* COLD: sessione fresca, append(P) + gen(1) dal contesto P. */
    S s1 = create(m);
    double t0 = now_ms();
    int rc = append(s1, ids, P);
    double t_prefill = now_ms() - t0;
    int g = gen(s1, 1, -1, one);
    double t_ttft_cold = now_ms() - t0;
    size_t c, a, pk; if (mem) mem(&c, &a, &pk);
    printf("COLD : append(P=%d) rc=%d = %.1f ms | TTFT(cold) = %.1f ms | gen=%d | tok=%d | active=%.1f MB\n",
           P, rc, t_prefill, t_ttft_cold, g, one[0], mem ? a/1e6 : 0);

    /* WARM: sessione fresca, append(P) + park + gen(1) dal MEDESIMO contesto P (unpark auto). */
    S s2 = create(m);
    append(s2, ids, P);
    int prc = park(s2, m);
    double t1 = now_ms();
    int g2 = gen(s2, 1, -1, one2);
    double t_restore = now_ms() - t1;
    printf("WARM : park rc=%d | TTFT(warm: restore KV da disco + gen) = %.1f ms | gen=%d | tok=%d\n",
           prc, t_restore, g2, one2[0]);

    int match = (g == 1 && g2 == 1 && one[0] == one2[0]);
    printf("ESITO: prefill(P)=%.1f ms vs restore=%.1f ms -> %s | stesso primo token: %s\n",
           t_prefill, t_restore, (t_restore < t_prefill) ? "RESTORE PIU' VELOCE (cache utile)" : "restore NON conviene",
           match ? "SI" : "NO");
    free(ids);
    return 0;
}
