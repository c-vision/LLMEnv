/* llm_kvdisk_e2e_test.c — A/B §G: spill del prefix cache su SSD + restore (IDS identici).
 * Uso: LLM_KVDISK=<path> llm_kvdisk_e2e_test <model_dir>
 *
 * Flusso: prefill 64 token -> gen ids1 (8) -> salva prefisso "p" (72 token) -> gen ids2 (8 =
 * RIFERIMENTO) -> salva "q"/"r" per forzare l'evict LRU di "p" (SPILL su disco) -> nuova
 * sessione con append_auto_prefix(query = prompt+ids1, 72 token) -> RESTORE da disco ->
 * gen deve riprodurre ids2. Verifica: IDS identici + reused==72 (+ restores>=1 con LLM_KVDISK). */
#include "llmengine_mlx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* definite nel motore ma non dichiarate in llmengine_mlx.h */
int llm_ai_mlx_model_set_prefix_max(llm_ai_mlx_model_handle mh, int n);
int llm_ai_mlx_model_prefix_debug(llm_ai_mlx_model_handle mh);

#define NP 64
#define NG 8

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: LLM_KVDISK=<path> %s <model_dir>\n", argv[0]); return 2; }
    int32_t prompt[NP], query[NP + NG], ref[NG], got[NG];
    for (int i = 0; i < NP; i++) prompt[i] = 1 + (i % 1000);

    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (!mh) { fprintf(stderr, "load_model fallita: %s\n", argv[1]); return 3; }
    llm_ai_mlx_model_set_prefix_max(mh, 2);   /* il 3o save forza l'evict LRU (spill) */

    llm_ai_mlx_session_handle A = llm_ai_mlx_create_session(mh);
    if (llm_ai_mlx_session_append(A, prompt, NP) != 0) { fprintf(stderr, "append fallito\n"); return 4; }
    int32_t ids1[NG];
    if (llm_ai_mlx_session_generate(A, NG, -1, ids1) != NG) { fprintf(stderr, "gen1 fallita\n"); return 5; }
    if (llm_ai_mlx_model_save_prefix(mh, "p", A) != 0) { fprintf(stderr, "save p fallito\n"); return 6; }
    if (llm_ai_mlx_session_generate(A, NG, -1, ref) != NG) { fprintf(stderr, "gen2 (REF) fallita\n"); return 7; }
    printf("REF:");
    for (int i = 0; i < NG; i++) printf(" %d", ref[i]);
    printf("\n");

    int32_t x = 7; llm_ai_mlx_session_append(A, &x, 1);
    if (llm_ai_mlx_model_save_prefix(mh, "q", A) != 0) { fprintf(stderr, "save q fallito\n"); return 8; }
    int32_t y = 9; llm_ai_mlx_session_append(A, &y, 1);
    if (llm_ai_mlx_model_save_prefix(mh, "r", A) != 0) { fprintf(stderr, "save r fallito\n"); return 9; }
    llm_ai_mlx_model_prefix_debug(mh);   /* atteso: p con has_kv=0 on_disk=1 (se LLM_KVDISK) */

    /* query = prompt + ids1 (72 token) = esattamente i token del prefisso "p" */
    memcpy(query, prompt, sizeof(prompt));
    memcpy(query + NP, ids1, sizeof(ids1));
    llm_ai_mlx_session_handle C = llm_ai_mlx_create_session(mh);
    int reused = llm_ai_mlx_session_append_auto_prefix(C, mh, query, NP + NG, 8);
    if (reused < 0) { fprintf(stderr, "auto_prefix fallito\n"); return 10; }
    if (llm_ai_mlx_session_generate(C, NG, -1, got) != NG) { fprintf(stderr, "gen RIUSO fallita\n"); return 11; }
    llm_ai_mlx_model_prefix_debug(mh);   /* atteso: restores>=1 se il riuso ha ripristinato da disco */
    printf("GOT:");
    for (int i = 0; i < NG; i++) printf(" %d", got[i]);
    printf("\nreused=%d\n", reused);

    int ok = (memcmp(ref, got, sizeof(ref)) == 0);
    printf("IDS %s\n", ok ? "IDENTICI" : "DIVERSI");
    llm_ai_mlx_destroy_session(A);
    llm_ai_mlx_destroy_session(C);
    return ok ? 0 : 1;
}
