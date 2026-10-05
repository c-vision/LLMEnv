/* llm_mlx_expolicy_test.c — §9.A EXPERT_RESIDENCY_POLICY: test MODEL-FREE.
 * Verifica parse policy, model-aware derive, apply alle opzioni registro. Nessun modello. */
#include "llm_mlx_expolicy.h"
#include "llm_engine_opts.h"

#include <stdio.h>
#include <string.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    if (llm_mlx_expolicy_active()) return fail("attiva prima del load");

    const char* p = "{\"mode\":\"OFFLOAD\",\"cache_mb\":4096,\"hotness\":true,\"prefetch_topn\":8,\"prefetch_cap\":16,\"async_workers\":2}";
    if (llm_mlx_expolicy_load_str(p) != 0) return fail("load_str");
    if (!llm_mlx_expolicy_active()) return fail("non attiva");
    if (strcmp(llm_mlx_expolicy_mode_name(), "OFFLOAD") != 0) return fail("mode");
    if (llm_mlx_expolicy_cache_mb() != 4096) return fail("cache_mb");
    if (!llm_mlx_expolicy_hotness()) return fail("hotness");
    if (llm_mlx_expolicy_prefetch_topn() != 8) return fail("prefetch_topn");

    llm_mlx_expolicy_apply();
    int v = 0;
    if (llm_ai_mlx_opt_get_int("LLMENGINE_MOE_CACHE_MB", &v) != 0 || v != 4096) return fail("apply cache_mb");
    if (llm_ai_mlx_opt_get_int("LLMENGINE_MOE_PREFETCH_TOPN", &v) != 0 || v != 8) return fail("apply prefetch_topn");
    if (llm_ai_mlx_opt_get_int("LLMENGINE_MOE_HOTNESS", &v) != 0 || v != 1) return fail("apply hotness");
    if (llm_ai_mlx_opt_get_int("LLMENGINE_MOE_ASYNC_WORKERS", &v) != 0 || v != 2) return fail("apply async_workers");

    /* Model-aware: budget >= totale -> RESIDENT; budget < totale -> OFFLOAD + prefetch=top_k. */
    if (llm_mlx_expolicy_derive(64, 8, 4LL * 1024 * 1024, 512LL * 1024 * 1024) != 0) return fail("derive1");
    if (strcmp(llm_mlx_expolicy_mode_name(), "RESIDENT") != 0) return fail("derive1 non RESIDENT");
    if (llm_mlx_expolicy_derive(64, 8, 4LL * 1024 * 1024, 16LL * 1024 * 1024) != 0) return fail("derive2");
    if (strcmp(llm_mlx_expolicy_mode_name(), "OFFLOAD") != 0) return fail("derive2 non OFFLOAD");
    if (llm_mlx_expolicy_prefetch_topn() != 8) return fail("derive2 prefetch!=top_k");
    if (!llm_mlx_expolicy_hotness()) return fail("derive2 hotness off");

    llm_mlx_expolicy_reset();
    if (llm_mlx_expolicy_active()) return fail("reset");

    printf("OK llm_mlx_expolicy_test\n");
    return 0;
}
