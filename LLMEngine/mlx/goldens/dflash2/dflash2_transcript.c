/* Transcript harness: prompt fisso -> generate DFlash2 greedy end-to-end,
 * confronto col transcript del reference (parita' target-authoritative). */
#include <stdio.h>
#include <stdlib.h>
#include "llmengine_mlx.h"

int main(void) {
    const char* TARGET="/Users/gaetano/ai/Models/Qwen3.8-27B-8bit";
    const char* DRAFT ="/Users/gaetano/ai/Models/Qwen3.8-27B-DFlash2";
    const int32_t prompt[] = {760, 6511, 314, 9338, 369};   /* "The capital of France is" */
    const int32_t golden[] = {11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13,
                              198, 760, 6511, 314, 14898, 369, 21047, 13, 198, 760};
    const int N = 20;
    int32_t eos = 248071;  /* Qwen3.8 tokenizer <|eot|>? uso 151645-like fallback: non ferma */

    llm_ai_mlx_model_handle target = llm_ai_mlx_load_model(TARGET);
    if (!target) { fprintf(stderr, "target load fail\n"); return 1; }
    llm_ai_mlx_dflash2_handle dfl = llm_ai_mlx_dflash2_create(target, DRAFT);
    if (!dfl) { fprintf(stderr, "create fail\n"); return 1; }
    llm_ai_mlx_session_handle sess = llm_ai_mlx_create_session(target);
    if (llm_ai_mlx_session_append(sess, prompt, 5) != 0) { fprintf(stderr, "append fail\n"); return 1; }

    int32_t out[64];
    int n = llm_ai_mlx_dflash2_generate(dfl, sess, eos, N, out);
    printf("generate returned %d\n", n);
    int match = (n == N);
    for (int i = 0; i < n && i < N; i++) {
        if (out[i] != golden[i]) match = 0;
        printf("  out[%d]=%d golden=%d%s\n", i, (int)out[i], golden[i], out[i]!=golden[i]?"  <-- MISMATCH":"");
    }
    if (n > N) for (int i = N; i < n; i++) printf("  extra[%d]=%d\n", i, (int)out[i]);
    printf("TRANSCRIPT: %s\n", match ? "MATCH" : "MISMATCH");
    llm_ai_mlx_dflash2_free(dfl);
    llm_ai_mlx_destroy_session(sess);
    llm_ai_mlx_free_model(target);
    return match ? 0 : 2;
}
