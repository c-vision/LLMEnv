/* llm_kvdisk_park_test.c — A/B §G-ter EVICT: park (KV su disco, RAM liberata) + unpark on-demand.
 * Uso: LLM_KVDISK=<path> llm_kvdisk_park_test <model_dir>
 *
 * A: prefill 600 -> gen NG (ref1) -> PARK (RAM liberata) -> gen NG (auto-unpark -> got2).
 * B: prefill 600 -> gen NG (ref1b) -> gen NG (ref2). Verifica: ref1==ref1b e got2==ref2. */
#include "llmengine_mlx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int llm_ai_mlx_model_prefix_debug(llm_ai_mlx_model_handle mh);

#define NP 600
#define NG 8
#define BLK 256

static int append_chunks(llm_ai_mlx_session_handle sh, const int32_t* p, int n) {
    int off = 0;
    while (off < n) { int c = (n - off) < BLK ? (n - off) : BLK;
        if (llm_ai_mlx_session_append(sh, p + off, c) != 0) return -1; off += c; }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: LLM_KVDISK=<path> %s <model_dir>\n", argv[0]); return 2; }
    int32_t prompt[NP], ref1[NG], got2[NG], ref1b[NG], ref2[NG];
    for (int i = 0; i < NP; i++) prompt[i] = 1 + (i % 1000);

    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (!mh) { fprintf(stderr, "load_model fallita\n"); return 3; }

    /* A: prefill + gen + PARK + gen (auto-unpark) */
    llm_ai_mlx_session_handle A = llm_ai_mlx_create_session(mh);
    if (append_chunks(A, prompt, NP) != 0) { fprintf(stderr, "append A fallito\n"); return 4; }
    if (llm_ai_mlx_session_generate(A, NG, -1, ref1) != NG) { fprintf(stderr, "gen A1 fallita\n"); return 5; }
    if (llm_ai_mlx_session_park(A, mh) != 0) { fprintf(stderr, "park fallito (LLM_KVDISK attivo?)\n"); return 6; }
    llm_ai_mlx_model_prefix_debug(mh);   /* atteso parks=1 */
    if (llm_ai_mlx_session_generate(A, NG, -1, got2) != NG) { fprintf(stderr, "gen A2 (dopo unpark) fallita\n"); return 7; }
    llm_ai_mlx_model_prefix_debug(mh);   /* atteso unparks=1 */

    /* B: controllo senza park */
    llm_ai_mlx_session_handle B = llm_ai_mlx_create_session(mh);
    if (append_chunks(B, prompt, NP) != 0) { fprintf(stderr, "append B fallito\n"); return 8; }
    if (llm_ai_mlx_session_generate(B, NG, -1, ref1b) != NG) { fprintf(stderr, "gen B1 fallita\n"); return 9; }
    if (llm_ai_mlx_session_generate(B, NG, -1, ref2) != NG) { fprintf(stderr, "gen B2 fallita\n"); return 10; }

    int ok1 = (memcmp(ref1, ref1b, sizeof(ref1)) == 0);
    int ok2 = (memcmp(got2, ref2, sizeof(ref2)) == 0);
    printf("REF1:");  for (int i = 0; i < NG; i++) printf(" %d", ref1[i]);  printf("\n");
    printf("REF1b:"); for (int i = 0; i < NG; i++) printf(" %d", ref1b[i]); printf("\n");
    printf("GOT2:");  for (int i = 0; i < NG; i++) printf(" %d", got2[i]);  printf("\n");
    printf("REF2:");  for (int i = 0; i < NG; i++) printf(" %d", ref2[i]);  printf("\n");
    printf("unpark: %s | IDS after-unpark %s\n", ok1 ? "OK" : "FAIL", ok2 ? "IDENTICI" : "DIVERSI");
    llm_ai_mlx_destroy_session(A);
    llm_ai_mlx_destroy_session(B);
    return (ok1 && ok2) ? 0 : 1;
}
