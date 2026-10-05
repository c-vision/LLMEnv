/* llm_mlx_execplan_test.c — §9.A M10 EXECUTION PLANNER: test MODEL-FREE.
 * Verifica parse del piano per-layer (ops + skip), selezione per-layer, layer non pianificato
 * e reset. Nessun modello. Ritorna 0 = PASS. */
#include "llm_mlx_execplan.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    if (llm_mlx_execplan_active()) { printf("FAIL: piano attivo prima del load\n"); return 1; }

    const char* plan =
        "{\"version\":1,\"layers\":["
        "{\"ops\":[\"attention\",\"ffn\"]},"
        "{\"skip\":true},"
        "{\"ops\":[\"attention\"]}]}";
    if (llm_mlx_execplan_load_str(plan) != 0) { printf("FAIL: load\n"); return 1; }
    if (!llm_mlx_execplan_active()) { printf("FAIL: non attivo dopo load\n"); return 1; }
    if (llm_mlx_execplan_nlayers() != 3) { printf("FAIL: nlayers=%d\n", llm_mlx_execplan_nlayers()); return 1; }

    char ops[LLM_MLX_EXECPLAN_MAX_OPS][LLM_MLX_EXECPLAN_OP_LEN];
    int skip = 0;

    int n = llm_mlx_execplan_layer(0, ops, LLM_MLX_EXECPLAN_MAX_OPS, &skip);
    if (n != 2 || skip != 0 || strcmp(ops[0], "attention") != 0 || strcmp(ops[1], "ffn") != 0) {
        printf("FAIL: L0 n=%d skip=%d ops=%s,%s\n", n, skip, n > 0 ? ops[0] : "-", n > 1 ? ops[1] : "-"); return 1;
    }

    n = llm_mlx_execplan_layer(1, ops, LLM_MLX_EXECPLAN_MAX_OPS, &skip);
    if (n != 0 || skip != 1) { printf("FAIL: L1 n=%d skip=%d\n", n, skip); return 1; }

    n = llm_mlx_execplan_layer(2, ops, LLM_MLX_EXECPLAN_MAX_OPS, &skip);
    if (n != 1 || skip != 0 || strcmp(ops[0], "attention") != 0) { printf("FAIL: L2 n=%d\n", n); return 1; }

    /* Layer non coperto -> -1 (il chiamante usa il default). */
    if (llm_mlx_execplan_layer(3, ops, LLM_MLX_EXECPLAN_MAX_OPS, &skip) != -1) { printf("FAIL: L3 doveva essere non pianificato\n"); return 1; }

    llm_mlx_execplan_reset();
    if (llm_mlx_execplan_active()) { printf("FAIL: reset non ha azzerato\n"); return 1; }

    printf("OK llm_mlx_execplan_test\n");
    return 0;
}
