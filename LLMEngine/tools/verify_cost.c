/* verify_cost.c — quanto costa UN verify di 8 draft rispetto a UN passo di decode (M=1)?
 * Modello: qwen3-8B-4bit. Nessun draft vero: si passano 8 token qualsiasi (tutti sbagliati) ->
 * accepted=1, cioe' il caso peggiore realistico (rollback + re-append del prefisso accettato).
 *
 * Uso: verify_cost <dylib> <model_dir> [reps]
 * Stampa: ms/verify(M=8), ms/token(AR), rapporto, e il punto di break-even dell'acceptance.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <dylib> <model_dir> [reps]\n", argv[0]); return 2; }
    const int reps = argc > 3 ? atoi(argv[3]) : 10;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    void* (*load)(const char*) = dlsym(h, "llm_ai_mlx_load_model");
    void* (*create)(void*) = dlsym(h, "llm_ai_mlx_create_session");
    int (*append)(void*, const int32_t*, int) = dlsym(h, "llm_ai_mlx_session_append");
    int (*gen)(void*, int, int32_t, int32_t*) = dlsym(h, "llm_ai_mlx_session_generate");
    int (*verify)(void*, const int32_t*, int, int32_t*, int32_t*) = dlsym(h, "llm_ai_mlx_session_verify_draft_batch");
    if (!load || !create || !append || !gen || !verify) { fprintf(stderr, "dlsym fallito\n"); return 3; }

    void* m = load(argv[2]); if (!m) { fprintf(stderr, "load modello fallito\n"); return 4; }
    void* s = create(m); if (!s) { fprintf(stderr, "create sessione fallito\n"); return 4; }

    int32_t ids[512]; for (int i = 0; i < 512; i++) ids[i] = 100 + (i * 37) % 30000;
    if (append(s, ids, 256) != 0) { fprintf(stderr, "append fallito\n"); return 5; }

    /* draft deliberatamente sbagliati: accepted=1 -> caso peggiore (rollback completo + re-append) */
    int32_t draft[8]; for (int i = 0; i < 8; i++) draft[i] = 4242 + i;
    int32_t acc = 0, next = 0;

    for (int w = 0; w < 3; w++) (void)verify(s, draft, 8, &acc, &next);   /* warmup */
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) { if (verify(s, draft, 8, &acc, &next) != 0) { fprintf(stderr, "verify rc!=0\n"); return 6; } }
    double t_verify = (now_ms() - t0) / reps;

    int32_t out[1];
    for (int w = 0; w < 3; w++) (void)gen(s, 1, -1, out);                 /* warmup */
    t0 = now_ms();
    for (int r = 0; r < reps; r++) { if (gen(s, 1, -1, out) != 1) { fprintf(stderr, "gen rc!=1\n"); return 7; } }
    double t_ar = (now_ms() - t0) / reps;

    const int D = 8;
    printf("verify M=%d : %8.2f ms  (accepted=%d)\n", D, t_verify, acc);
    printf("decode AR   : %8.2f ms/token\n", t_ar);
    printf("rapporto    : %8.2f x un passo di decode (su %d draft)\n", t_verify / t_ar, D);
    printf("break-even  : servono %.2f token accettati perche' la spec paghi\n", t_verify / t_ar);
    printf("max guadagno teorico a piena accettazione: %.2fx\n", (double)D / (t_verify / t_ar));
    return 0;
}
