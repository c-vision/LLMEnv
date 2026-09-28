/* specaccept.c — probe di RIPRODUZIONE per l'artefatto di acceptance del path
 * verify_draft_batch generico (B13). Non assume nulla: misura.
 *
 * Idea: il verificatore batch deve accettare per intero il PROPIO continuation
 * greedy. Sequenza:
 *   S1: append(prompt); generate G  -> G = greedy autoregressivo del target
 *       (path final_step => llm_mlx_final_norm => rms_norm_cfg => cast bf16).
 *   S2: append(stesso prompt) => pending_next_id == G[0].
 *       verify_draft_batch(S2, draft=G, N): il draft E' il greedy, quindi un
 *       verificatore NUMERICAMENTE COERENTE accetta tutti N (accepted==N).
 *       Se accepted < N, l'argmax del path batch differisce da quello greedy:
 *       -> acceptance regression dimostrata (draft corretti scartati).
 *
 * Causa attesa: verify_draft_batch usa mlx_fast_rms_norm RAW (:13579) mentre
 * greedy0/df2 usano llm_mlx_final_norm (che sotto g_bf16_all casta a bf16).
 *
 * Build (dalla dir native/llmengine_mlx):
 *   clang -O2 -I. goldens/specaccept.c -Lbuild -lllm_ai_mlx \
 *         -Wl,-rpath,"$PWD/build" -o /tmp/specaccept
 * Uso:
 *   /tmp/specaccept [model_dir] [n_gen]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "llmengine_mlx.h"

int main(int argc, char** argv) {
    const char* MODEL = argc > 1 ? argv[1] : "/Users/gaetano/ai/Models/Qwen3.8-27B-8bit";
    int N = argc > 2 ? atoi(argv[2]) : 24;
    /* prompt reale Qwen-tokenizzato: "The capital of France is" (dal golden dflash2) */
    const int32_t prompt[] = {760, 6511, 314, 9338, 369};
    const int NP = 5;

    llm_ai_mlx_model_handle model = llm_ai_mlx_load_model(MODEL);
    if (!model) { fprintf(stderr, "load modello fallito: %s\n", MODEL); return 1; }

    int32_t* G = (int32_t*)malloc(sizeof(int32_t) * N);

    /* Pass 1: greedy autoregressivo di riferimento. */
    llm_ai_mlx_session_handle s1 = llm_ai_mlx_create_session(model);
    if (llm_ai_mlx_session_append(s1, prompt, NP) != 0) { fprintf(stderr, "append s1 fallito\n"); return 1; }
    int ng = llm_ai_mlx_session_generate(s1, N, -1, G);
    printf("greedy di riferimento (%d token):", ng);
    for (int i = 0; i < ng; i++) printf(" %d", G[i]);
    printf("\n");

    /* Pass 2: verify_draft_batch con draft == vero greedy. */
    llm_ai_mlx_session_handle s2 = llm_ai_mlx_create_session(model);
    if (llm_ai_mlx_session_append(s2, prompt, NP) != 0) { fprintf(stderr, "append s2 fallito\n"); return 1; }
    int32_t accepted = -1, next = -1;
    int rc = llm_ai_mlx_session_verify_draft_batch(s2, G, ng, &accepted, &next);
    printf("verify_draft_batch(draft=greedy): rc=%d accepted=%d/%d  next=%d\n", rc, accepted, ng, next);
    if (rc == 0 && accepted < ng)
        printf(">>> ACCEPTANCE REGRESSION: il verificatore batch rifiuta i PROPRI token greedy a %d/%d\n", accepted, ng);
    else if (rc == 0 && accepted == ng)
        printf(">>> coerente: batch accetta tutto il greedy (%d/%d)\n", accepted, ng);

    /* Pass 3 (B, speculative batching): verify_draft_batch_multi con 2 sessioni, draft == vero
     * greedy. Il verificatore multi deve accettare per INTERO il proprio greedy su OGNI sessione
     * (N/N), esattamente come il single-session. */
    {
        const int NM = 2;
        llm_ai_mlx_session_handle sm[2];
        int32_t* draft_ids = (int32_t*)malloc(sizeof(int32_t) * NM * ng);
        int* draft_len = (int*)malloc(sizeof(int) * NM);
        int32_t* macc = (int32_t*)malloc(sizeof(int32_t) * NM);
        int32_t* mnext = (int32_t*)malloc(sizeof(int32_t) * NM);
        for (int i = 0; i < NM; i++) {
            sm[i] = llm_ai_mlx_create_session(model);
            llm_ai_mlx_session_append(sm[i], prompt, NP);
            memcpy(draft_ids + (size_t)i * ng, G, sizeof(int32_t) * ng);
            draft_len[i] = ng;
        }
        int rcm = llm_ai_mlx_session_verify_draft_batch_multi(sm, NM, draft_ids, draft_len, ng, macc, mnext);
        printf("verify_draft_batch_multi(draft=greedy, NM=%d): rc=%d accepted=[%d,%d]/%d next=[%d,%d]\n",
               NM, rcm, macc[0], macc[1], ng, mnext[0], mnext[1]);
        if (rcm == 0 && macc[0] == ng && macc[1] == ng)
            printf(">>> multi coerente: ogni sessione accetta tutto il greedy (%d/%d)\n", ng, ng);
        else
            printf(">>> MULTI REGRESSION: accepted=%d/%d, %d/%d\n", macc[0], ng, macc[1], ng);
        for (int i = 0; i < NM; i++) llm_ai_mlx_destroy_session(sm[i]);
        free(draft_ids); free(draft_len); free(macc); free(mnext);
    }

    /* Pass 4 (B): draft con un token SBAGLIATO alla posizione `bad` -> accepted==bad e
     * next==G[bad] (token corretto del modello). Verifica sia single che multi (mismatch parziale
     * + ramo "committa il token corretto"). */
    if (ng > 3) {
        int bad = 5 < ng ? 5 : ng - 1;
        int32_t* wrong = (int32_t*)malloc(sizeof(int32_t) * ng);
        memcpy(wrong, G, sizeof(int32_t) * ng);
        wrong[bad] = G[bad] + 7;
        int32_t a1 = -1, x1 = -1;
        llm_ai_mlx_session_handle s4 = llm_ai_mlx_create_session(model);
        llm_ai_mlx_session_append(s4, prompt, NP);
        int rc4 = llm_ai_mlx_session_verify_draft_batch(s4, wrong, ng, &a1, &x1);
        printf("single mismatch@%d: rc=%d accepted=%d next=%d (atteso accepted=%d next=%d)\n",
               bad, rc4, a1, x1, bad, G[bad]);
        const int NM2 = 2;
        llm_ai_mlx_session_handle sm2[2];
        int32_t* d2 = (int32_t*)malloc(sizeof(int32_t) * NM2 * ng);
        int dl2[2] = { ng, ng };
        int32_t ma2[2], mn2[2];
        for (int i = 0; i < NM2; i++) {
            sm2[i] = llm_ai_mlx_create_session(model);
            llm_ai_mlx_session_append(sm2[i], prompt, NP);
            memcpy(d2 + (size_t)i * ng, wrong, sizeof(int32_t) * ng);
        }
        int rc4m = llm_ai_mlx_session_verify_draft_batch_multi(sm2, NM2, d2, dl2, ng, ma2, mn2);
        printf("multi  mismatch@%d: rc=%d accepted=[%d,%d] next=[%d,%d] (atteso accepted=%d next=%d)\n",
               bad, rc4m, ma2[0], ma2[1], mn2[0], mn2[1], bad, G[bad]);
        if (rc4m == 0 && ma2[0] == bad && ma2[1] == bad && mn2[0] == G[bad] && mn2[1] == G[bad])
            printf(">>> multi mismatch coerente (accepted=%d, next=G[%d])\n", bad, bad);
        else
            printf(">>> MULTI MISMATCH REGRESSION\n");
        for (int i = 0; i < NM2; i++) llm_ai_mlx_destroy_session(sm2[i]);
        free(d2); free(wrong);
        llm_ai_mlx_destroy_session(s4);
    }

    /* NOTA: nessun contrasto con verify_token singolo: verify_token AVANZA gia' la
     * cache internamente sull'accettazione, quindi ri-appendere il token nel loop lo
     * double-avanza e disallinea -> falso accepted=1. Pass 2 (batch accetta il proprio
     * greedy 24/24) e' il discriminante pulito e sufficiente. */

    free(G);
    llm_ai_mlx_destroy_session(s1); llm_ai_mlx_destroy_session(s2);
    llm_ai_mlx_free_model(model);
    return 0;
}
