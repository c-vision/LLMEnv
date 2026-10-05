/* llm_mlx_kvclasses_test.c — §9.A M10d: test MODEL-FREE delle classi KV (RUNTIME/PERSISTENT). */
#include "llm_mlx_kvclasses.h"

#include <stdio.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    llm_mlx_kvclasses_reset();
    if (llm_mlx_kvclasses_register("prefix.ram", LLM_KVCLASS_RUNTIME, 1000) != 0) return fail("reg rt");
    if (llm_mlx_kvclasses_register("prefix.kvdisk", LLM_KVCLASS_PERSISTENT, 5000) != 0) return fail("reg ps");
    if (llm_mlx_kvclasses_register("kvq.pool", LLM_KVCLASS_RUNTIME, 2000) != 0) return fail("reg rt2");

    uint64_t rt = 0, ps = 0; int nr = 0, np = 0;
    llm_mlx_kvclasses_stats(&rt, &ps, &nr, &np);
    if (rt != 3000 || ps != 5000 || nr != 2 || np != 1) return fail("stats iniziali");

    llm_mlx_kvclasses_set_limits(2500, -1);
    if (!llm_mlx_kvclasses_over_limit(LLM_KVCLASS_RUNTIME)) return fail("rt doveva superare 2500");
    if (llm_mlx_kvclasses_over_limit(LLM_KVCLASS_PERSISTENT)) return fail("ps illimitato");

    if (llm_mlx_kvclasses_update("kvq.pool", 1000) != 0) return fail("update");
    llm_mlx_kvclasses_stats(&rt, &ps, &nr, &np);
    if (rt != 2000) return fail("stats dopo update");

    if (llm_mlx_kvclasses_unregister("prefix.ram") != 0) return fail("unregister");
    llm_mlx_kvclasses_stats(&rt, &ps, &nr, &np);
    if (rt != 1000 || nr != 1) return fail("stats dopo unregister");

    llm_mlx_kvclasses_reset();
    llm_mlx_kvclasses_stats(&rt, &ps, &nr, &np);
    if (rt != 0 || ps != 0 || nr != 0 || np != 0) return fail("reset");

    printf("OK llm_mlx_kvclasses_test\n");
    return 0;
}
