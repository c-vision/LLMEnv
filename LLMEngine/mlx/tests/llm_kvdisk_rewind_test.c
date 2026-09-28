/* llm_kvdisk_rewind_test.c — A/B §G-ter: checkpoint a blocchi + agentic rewind.
 * Uso: LLM_KVDISK=<path> llm_kvdisk_rewind_test <model_dir>
 *
 * A: prefill 600 token in chunk da 256/256/88 (checkpoint a 256 e 512) -> gen NG = RIFERIMENTO.
 * B (nuova sessione): rewind(tokens, 600, keep=600) -> ripristina il checkpoint piu' lungo <= 600
 * (=512) da disco, poi append del solo suffisso (88 token) -> gen deve riprodurre il riferimento. */
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
    int32_t prompt[NP], ref[NG], got[NG];
    for (int i = 0; i < NP; i++) prompt[i] = 1 + (i % 1000);

    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (!mh) { fprintf(stderr, "load_model fallita\n"); return 3; }

    /* A: prefill a chunk -> checkpoint a 256/512, poi RIFERIMENTO */
    llm_ai_mlx_session_handle A = llm_ai_mlx_create_session(mh);
    if (append_chunks(A, prompt, NP) != 0) { fprintf(stderr, "append A fallito\n"); return 4; }
    if (llm_ai_mlx_session_generate(A, NG, -1, ref) != NG) { fprintf(stderr, "gen REF fallita\n"); return 5; }
    printf("REF:");
    for (int i = 0; i < NG; i++) printf(" %d", ref[i]);
    printf("\n");
    llm_ai_mlx_model_prefix_debug(mh);   /* atteso ckpt_writes>=2 */

    /* B: rewind da disco + prefill del solo suffisso. K6: per architetture ibride (layer sliding)
     * la posizione arretra di una finestra (>=256 -> 256); per il denso resta 512. */
    llm_ai_mlx_session_handle B = llm_ai_mlx_create_session(mh);
    int p = llm_ai_mlx_session_rewind(B, mh, prompt, NP, NP);
    printf("rewind -> p=%d (atteso 512; 256 se ibrida con K6)\n", p);
    if (p <= 0 || p > 512 || (p % BLK) != 0) { fprintf(stderr, "rewind: posizione inattesa\n"); return 6; }
    if (llm_ai_mlx_session_append(B, prompt + p, NP - p) != 0) { fprintf(stderr, "append suffisso fallito\n"); return 7; }
    if (llm_ai_mlx_session_generate(B, NG, -1, got) != NG) { fprintf(stderr, "gen RIUSO fallita\n"); return 8; }
    printf("GOT:");
    for (int i = 0; i < NG; i++) printf(" %d", got[i]);
    printf("\n");
    llm_ai_mlx_model_prefix_debug(mh);   /* atteso rewinds>=1 */

    int ok = (memcmp(ref, got, sizeof(ref)) == 0);
    printf("IDS %s\n", ok ? "IDENTICI" : "DIVERSI");
    llm_ai_mlx_destroy_session(A);
    llm_ai_mlx_destroy_session(B);
    return ok ? 0 : 1;
}
