/* llm_decode_time.c — tempo di prefill/decode reale via API engine (dlopen, nessuna modifica
 * all'engine). Uso: llm_decode_time <dylib> <model_dir> <ngen> [reps] */
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / (double)tb.denom / 1e6;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "uso: %s <dylib> <model_dir> <ngen> [reps]\n", argv[0]); return 2; }
    int ngen = atoi(argv[3]);
    int reps = argc > 4 ? atoi(argv[4]) : 2;
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    void* (*load_model)(const char*) = dlsym(h, "llm_ai_mlx_load_model");
    void* (*create_session)(void*) = dlsym(h, "llm_ai_mlx_create_session");
    int (*append)(void*, const int32_t*, int) = dlsym(h, "llm_ai_mlx_session_append");
    int (*topk)(void*, int, int32_t*, float*) = dlsym(h, "llm_ai_mlx_session_topk");
    void (*destroy_session)(void*) = dlsym(h, "llm_ai_mlx_destroy_session");
    void (*free_model)(void*) = dlsym(h, "llm_ai_mlx_free_model");
    if (!load_model || !create_session || !append || !topk) { fprintf(stderr, "dlsym: mancano simboli\n"); return 4; }

    void* model = load_model(argv[2]);
    if (!model) { fprintf(stderr, "load_model fallito\n"); return 5; }

    int32_t prompt[] = {1, 22172, 1996, 3457, 315, 9583, 374};   /* ids fissi, solo timing */
    int np = (int)(sizeof(prompt)/sizeof(prompt[0]));
    for (int r = 0; r < reps; r++) {
        void* s = create_session(model);
        double t0 = now_ms();
        if (append(s, prompt, np) != 0) { fprintf(stderr, "append fallito\n"); return 6; }
        double t_pre = now_ms() - t0;
        int32_t id = 0; float lp = 0;
        double t1 = now_ms();
        for (int i = 0; i < ngen; i++) {
            if (topk(s, 1, &id, &lp) != 1) break;
            if (append(s, &id, 1) != 0) break;
        }
        double t_dec = now_ms() - t1;
        printf("rep %d: prefill(%d tok)=%.2f ms | decode %d tok=%.2f ms | %.2f ms/token | %.1f tok/s\n",
               r, np, t_pre, ngen, t_dec, t_dec / ngen, 1000.0 * ngen / t_dec);
        if (destroy_session) destroy_session(s);
    }
    if (free_model) free_model(model);
    return 0;
}
