/* llm_mlx_mobj_test.c — §9.A M10: test MODEL-FREE del MEMORY OBJECT runtime manager unificato. */
#include "llm_mlx_mobj.h"

#include <stdio.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    llm_mlx_mobj_reset();
    if (llm_mlx_mobj_register("kv.sess1", LLM_MOBJ_KV_RUNTIME, LLM_MOBJ_RESIDENT, 1000) != 0) return fail("reg kv");
    if (llm_mlx_mobj_register("prefix.disk", LLM_MOBJ_KV_PERSISTENT, LLM_MOBJ_DISK, 5000) != 0) return fail("reg prefix");
    if (llm_mlx_mobj_register("w", LLM_MOBJ_WEIGHTS, LLM_MOBJ_RESIDENT, 200000) != 0) return fail("reg weights");

    unsigned long long t = 0; int n = 0;
    llm_mlx_mobj_stats(&t, &n);
    if (t != 206000 || n != 3) return fail("stats totali");

    llm_mlx_mobj_stats_cat(LLM_MOBJ_KV_RUNTIME, &t, &n);
    if (t != 1000 || n != 1) return fail("stats kv_runtime");

    if (llm_mlx_mobj_evict_priority(LLM_MOBJ_TEMP) >= llm_mlx_mobj_evict_priority(LLM_MOBJ_WEIGHTS)) return fail("priorita eviction");
    if (llm_mlx_mobj_evict_priority(LLM_MOBJ_PREFIX) >= llm_mlx_mobj_evict_priority(LLM_MOBJ_KV_RUNTIME)) return fail("priorita prefix");

    llm_mlx_mobj_set_budget(100000);
    if (!llm_mlx_mobj_over_budget()) return fail("doveva superare il budget");
    llm_mlx_mobj_set_budget(1000000);
    if (llm_mlx_mobj_over_budget()) return fail("non doveva superare");

    if (llm_mlx_mobj_unregister("w") != 0) return fail("unregister");
    llm_mlx_mobj_stats(&t, &n);
    if (t != 6000 || n != 2) return fail("stats dopo unregister");

    llm_mlx_mobj_reset();
    if (llm_mlx_mobj_count() != 0) return fail("reset");

    printf("OK llm_mlx_mobj_test\n");
    return 0;
}
