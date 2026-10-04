/* llm_dsv41_numpy_rng_test.c — verifica del port C di np.random.default_rng vs riferimento NumPy.
 *   cc -O2 -o /tmp/llm_dsv41_numpy_rng_test native/llmengine_mlx/llm_dsv41_numpy_rng_test.c -lm */
#include <stdio.h>
#include <stdint.h>
#include "llm_dsv41_numpy_rng.h"
#include "llm_dsv41_numpy_rng_ref.h"

int main(void) {
    printf("llm_dsv41_numpy_rng — np.random.default_rng port\n");
    int fail = 0;
    for (int c = 0; c < REF_RNG_NCASES; c++) {
        int64_t out[REF_RNG_MAXN];
        llm_dsv41_numpy_integers_i64(REF_RNG_SEED[c], REF_RNG_HIGH[c], out, REF_RNG_N[c]);
        int ok = 1;
        const int64_t* exp = REF_RNG_EXPECT + (size_t)c * REF_RNG_MAXN;
        for (int i = 0; i < REF_RNG_N[c]; i++)
            if (out[i] != exp[i]) { printf("  case %d [%d] got=%lld exp=%lld\n", c, i,
                                           (long long)out[i], (long long)exp[i]); ok = 0; break; }
        char nm[64]; snprintf(nm, sizeof nm, "case %d (seed %lld, high %lld)", c,
                              (long long)REF_RNG_SEED[c], (long long)REF_RNG_HIGH[c]);
        if (ok) printf("PASS %-46s\n", nm);
        else { printf("FAIL %-46s\n", nm); fail = 1; }
    }
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
