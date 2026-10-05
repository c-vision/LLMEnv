/* llm_mlx_kvdag_test.c — §9.A M10b: test MODEL-FREE del DAG KV + espansione PREFETCH. */
#include "llm_mlx_kvdag.h"

#include <stdio.h>
#include <string.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    /* Grafo minimale: base <- derived <- index; il prefetch di index deve espandersi in ordine. */
    llm_mlx_kvdag_reset();
    if (llm_mlx_kvdag_add("base", LLM_KVNODE_BASE, 0) != 0) return fail("add base");
    if (llm_mlx_kvdag_add("der", LLM_KVNODE_DERIVED, 0) != 0) return fail("add der");
    if (llm_mlx_kvdag_add("idx", LLM_KVNODE_INDEX, 1) != 0) return fail("add idx");
    if (llm_mlx_kvdag_dep("der", "base") != 0) return fail("dep der");
    if (llm_mlx_kvdag_dep("idx", "der") != 0) return fail("dep idx");
    char out[256];
    int n = llm_mlx_kvdag_prefetch("idx", out, sizeof out);
    if (n != 3) return fail("prefetch n!=3");
    if (strcmp(out, "READ base,REUSE der,RECONSTRUCT idx") != 0) { printf("FAIL: [%s]\n", out); return 1; }

    /* Grafo CSA2 rappresentativo (7 nodi): l'espansione e' deterministica e topologica. */
    char o2[512];
    int n2 = llm_mlx_kvdag_demo_csa2(o2, sizeof o2);
    if (n2 != 7) return fail("demo n!=7");
    const char* exp =
        "READ v41.base,REUSE v41.reuse.l0,RECONSTRUCT v41.index.l0,"
        "REUSE v41.reuse.l1,RECONSTRUCT v41.index.l1,REUSE v41.shared,RECONSTRUCT v41.out";
    if (strcmp(o2, exp) != 0) { printf("FAIL demo: [%s]\n", o2); return 1; }

    printf("OK llm_mlx_kvdag_test\n");
    return 0;
}
