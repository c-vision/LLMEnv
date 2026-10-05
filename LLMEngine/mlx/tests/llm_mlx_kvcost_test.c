/* llm_mlx_kvcost_test.c — §9.A M10a: test MODEL-FREE del modello di costo KV. */
#include "llm_mlx_kvcost.h"

#include <stdio.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    LlmKvCostResult r;

    /* Banda di ricarica bassa -> persistere costa (1e6) piu' del ri-prefill (1000). */
    if (llm_mlx_kvcost_decide(1000, 1.0, 1e9, 1e3, 1.0, &r) != LLM_KVRECOMPUTE) return fail("atteso RECOMPUTE");
    if (r.cost_recompute != 1000.0) return fail("cost_recompute");

    /* Banda alta -> ricaricare (1) costa meno del ri-prefill (1000). */
    if (llm_mlx_kvcost_decide(1000, 1.0, 1e9, 1e9, 1.0, &r) != LLM_KVPERSIST) return fail("atteso PERSIST");

    /* Tenere residente quasi gratis -> KEEP. */
    if (llm_mlx_kvcost_decide(1000, 1.0, 1e9, 1e9, 1e-12, &r) != LLM_KVKEEP) return fail("atteso KEEP");

    /* persist non praticabile (bw<=0) -> mai PERSIST. */
    if (llm_mlx_kvcost_decide(1000, 1.0, 1e9, 0.0, 1e9, &r) != LLM_KVRECOMPUTE) return fail("bw<=0 -> RECOMPUTE");

    printf("OK llm_mlx_kvcost_test\n");
    return 0;
}
