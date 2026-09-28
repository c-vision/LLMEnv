/* specbench.c — BENCH dello speculative batching multi-sessione (B) e delle alternative.
 *
 * Confronta, per lo STESSO totale di token (N sessioni x T token, greedy):
 *   G0) decode greedy BATCHED  (generate_batch: 1 forward N x 1 per token)  -> T forward
 *   S1) speculazione PER-SESSIONE (verify_draft_batch 1 x K per round)      -> N*ceil(T/K) forward
 *   S2) speculative batching (A): verify_draft_batch_multi col kernel N x K -> ceil(T/K) forward
 *
 * Draft = la sequenza greedy stessa (drafter perfetto, acceptance = K): misura il COSTO del
 * meccanismo, non la qualita' del drafter. MISURATO (Qwen3.8-27B, N=4 T=32 K=8):
 *   G0 30.1 tok/s | S1 54.1 | S2/A 108.5  -> (A) = 3.6x sul greedy batched.
 * (La vecchia versione (B) del multi-verify, K passi N x 1, dava ~= G0: nessun guadagno.)
 *
 * Build (dalla dir native/llmengine_mlx):
 *   clang -O2 -I. goldens/specbench.c -Lbuild -lllm_ai_mlx \
 *         -Wl,-rpath,"$PWD/build" -o /tmp/specbench
 * Uso: /tmp/specbench [model_dir] [N] [T] [K]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "llmengine_mlx.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    const char* MODEL = argc > 1 ? argv[1] : "/Users/gaetano/ai/Models/Qwen3.8-27B-8bit";
    int N = argc > 2 ? atoi(argv[2]) : 4;
    int T = argc > 3 ? atoi(argv[3]) : 32;
    int K = argc > 4 ? atoi(argv[4]) : 8;
    const int32_t prompt[] = {760, 6511, 314, 9338, 369};
    const int NP = 5;

    llm_ai_mlx_model_handle model = llm_ai_mlx_load_model(MODEL);
    if (!model) { fprintf(stderr, "load modello fallito: %s\n", MODEL); return 1; }

    int32_t* G = (int32_t*)malloc(sizeof(int32_t) * T);
    { llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(model);
      llm_ai_mlx_session_append(s, prompt, NP);
      int ng = llm_ai_mlx_session_generate(s, T, -1, G);
      if (ng <= 0) { fprintf(stderr, "reference greedy fallito\n"); return 1; }
      T = ng; llm_ai_mlx_destroy_session(s); }
    printf("N=%d T=%d K=%d (drafter perfetto: draft = greedy)\n", N, T, K);

    llm_ai_mlx_session_handle h[256];
    int32_t out[256];

    /* G0 — greedy batched */
    {
        for (int i = 0; i < N; i++) { h[i] = llm_ai_mlx_create_session(model); llm_ai_mlx_session_append(h[i], prompt, NP); }
        double t0 = now_s();
        for (int t = 0; t < T; t++) if (llm_ai_mlx_session_generate_batch(h, N, 1, -1, out) != 0) break;
        double dt = now_s() - t0;
        printf("G0 greedy-batch   : %.3fs  %7.1f tok/s  (%d tok, %d forward)\n", dt, (double)(N * T) / dt, N * T, T);
        for (int i = 0; i < N; i++) llm_ai_mlx_destroy_session(h[i]);
    }

    /* S1 — speculazione per-sessione (verify_draft_batch) */
    {
        for (int i = 0; i < N; i++) { h[i] = llm_ai_mlx_create_session(model); llm_ai_mlx_session_append(h[i], prompt, NP); }
        double t0 = now_s();
        for (int i = 0; i < N; i++) {
            int off = 0;
            while (off < T) {
                int k = T - off; if (k > K) k = K;
                int32_t a = -1, nx = -1;
                if (llm_ai_mlx_session_verify_draft_batch(h[i], G + off, k, &a, &nx) != 0 || a <= 0) break;
                off += a;
            }
        }
        double dt = now_s() - t0;
        printf("S1 per-sess spec  : %.3fs  %7.1f tok/s  (%d tok, ~%d forward 1xK)\n", dt, (double)(N * T) / dt, N * T, N * ((T + K - 1) / K));
        for (int i = 0; i < N; i++) llm_ai_mlx_destroy_session(h[i]);
    }

    /* S2 — speculative batching (B): verify_draft_batch_multi */
    {
        for (int i = 0; i < N; i++) { h[i] = llm_ai_mlx_create_session(model); llm_ai_mlx_session_append(h[i], prompt, NP); }
        int* off = (int*)calloc((size_t)N, sizeof(int));
        int done = 0;
        int32_t* d = (int32_t*)malloc(sizeof(int32_t) * (size_t)N * K);
        int* dl = (int*)malloc(sizeof(int) * (size_t)N);
        int32_t* acc = (int32_t*)malloc(sizeof(int32_t) * (size_t)N);
        int32_t* nxt = (int32_t*)malloc(sizeof(int32_t) * (size_t)N);
        double t0 = now_s();
        while (done < N) {
            int na = 0;
            llm_ai_mlx_session_handle ah[256];
            for (int i = 0; i < N; i++) {
                if (off[i] >= T) continue;
                int k = T - off[i]; if (k > K) k = K;
                memcpy(d + (size_t)na * K, G + off[i], sizeof(int32_t) * k);
                dl[na] = k; ah[na] = h[i]; na++;
            }
            if (na == 0) break;
            if (llm_ai_mlx_session_verify_draft_batch_multi(ah, na, d, dl, K, acc, nxt) != 0) break;
            for (int a = 0; a < na; a++) {
                int i = -1;
                for (int q = 0; q < N; q++) if (h[q] == ah[a]) { i = q; break; }
                if (i < 0) continue;
                off[i] += acc[a];
                if (off[i] >= T) done++;
            }
        }
        double dt = now_s() - t0;
        printf("S2 (A) NxK verify : %.3fs  %7.1f tok/s  (%d tok, ~%d forward NxK)\n", dt, (double)(N * T) / dt, N * T, (T + K - 1) / K);
        free(off); free(d); free(dl); free(acc); free(nxt);
        for (int i = 0; i < N; i++) llm_ai_mlx_destroy_session(h[i]);
    }

    free(G); llm_ai_mlx_free_model(model);
    return 0;
}
