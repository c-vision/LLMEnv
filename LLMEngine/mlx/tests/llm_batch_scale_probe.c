/* batch_scale_probe.c — curva di scaling del BATCH (per P0-ter: il verify speculativo usa
 * M = K+1 token). Per ogni N: SESSIONE NUOVA + append di N token, cronometrato; prima un
 * warm-up (append+generate) per materializzare i pesi (altrimenti il primo tocco domina).
 * Uso: batch_scale_probe <model_dir> [N1 N2 ...] */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <model_dir> <N...>\n", argv[0]); return 2; }
    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (mh == NULL) { fprintf(stderr, "load fallita\n"); return 3; }

    /* warm-up: append+generate su una sessione usa-e-getta (primo tocco dei pesi) */
    {
        /* warm-up: 8 tok (decode) + 64 tok (PREFILL: inizializza anche il path oQ A8 lazy,
         * che si crea al primo uso con M>=16: senza questo la prima misura paga metallib+
         * pipeline e sembra catastrofica). */
        int32_t w[64]; for (int i = 0; i < 64; i++) w[i] = 100 + i;
        llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(mh);
        if (s) { (void)llm_ai_mlx_session_append(s, w, 8);
                 int32_t o[4]; (void)llm_ai_mlx_session_generate(s, 4, -1, o);
                 llm_ai_mlx_destroy_session(s); }
        s = llm_ai_mlx_create_session(mh);
        if (s) { (void)llm_ai_mlx_session_append(s, w, 64); llm_ai_mlx_destroy_session(s); }
        fprintf(stderr, "[scale] warm-up fatto\n");
    }
    printf("%6s %12s %12s\n", "N", "tot ms", "ms/tok");
    for (int a = 2; a < argc; a++) {
        int N = atoi(argv[a]);
        int32_t* ids = (int32_t*)malloc((size_t)N * sizeof(int32_t));
        for (int i = 0; i < N; i++) ids[i] = 100 + (i % 900);
        for (int it = 0; it < 2; it++) {   /* it=0 warm-up del path per QUESTA N, it=1 misurata */
            llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(mh);
            if (s == NULL) { free(ids); return 4; }
            double t0 = now_s();
            int rc = llm_ai_mlx_session_append(s, ids, N);
            double t = (now_s() - t0) * 1000.0;
            if (it == 1)
                printf("%6d %12.1f %12.2f%s\n", N, t, t / N, rc == 0 ? "" : "  ERRORE");
            fflush(stdout);
            llm_ai_mlx_destroy_session(s);
        }
        free(ids);
    }
    return 0;
}
