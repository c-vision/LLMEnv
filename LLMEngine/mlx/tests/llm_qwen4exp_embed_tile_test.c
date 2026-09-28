/* llm_qwen4exp_embed_tile_test.c — Self-test unitario Qwen4-Exp tile hc_count (bug MLX-Qwen4Exp-Attn-HyperConnection-Mix-Failed).
 * Uso: llm_qwen4exp_embed_tile_test [hidden] [hc_count] [vocab] [token_id]  (nessun modello: deterministico, millisecondi).
 * Verifica che l'embedding del token nel path LLM_ASYNC_DECODE (llm_mlx_embed_row_from_idx)
 * sia tileato hc_count volte con l'ultimo asse pari a hc_count*hidden e valori replicati. */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int hidden = argc > 1 ? atoi(argv[1]) : 64;
    int hc_count = argc > 2 ? atoi(argv[2]) : 4;
    int vocab = argc > 3 ? atoi(argv[3]) : 32;
    int token_id = argc > 4 ? atoi(argv[4]) : 5;
    char buf[512];
    int rc = llm_ai_mlx_qwen4exp_embed_tile_selftest(hidden, hc_count, vocab, token_id, buf, (int)sizeof(buf));
    printf("rc=%d %s\n", rc, buf);
    return rc == 0 ? 0 : 1;
}
