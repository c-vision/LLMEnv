/* llm_dsv41_engram_params_test.c — verifica della scansione primes/offsets Engram contro i
 * `engram_num_embeddings` REALI del config (deepseek-ai/DeepSeek-V4.1-Flash). Puro C.
 *   cc -O2 -o /tmp/llm_dsv41_engram_params_test native/llmengine_mlx/llm_dsv41_engram_params_test.c -lm */
#include <stdio.h>
#include <stdint.h>
#include "llm_dsv41_tensors.h"
#include "llm_dsv41_engram_params.h"

int main(void) {
    printf("llm_dsv41_engram_params — primes/offsets vs num_embeddings reali\n");
    LlmDsv41Config c = llm_dsv41_config_real();
    const int depth = c.engram_max_ngram_size - 1;
    const int heads = c.engram_n_heads;
    int64_t seen[256]; int seen_n = 0;
    int64_t primes[64], offsets[64];
    int fail = 0;
    for (int idx = 0; idx < c.n_engram_layers; idx++) {
        int64_t sum = llm_dsv41_engram_scan(c.engram_vocab_size, depth, heads,
                                            seen, &seen_n, 256, primes, offsets);
        int64_t want = (int64_t)c.engram_num_embeddings[idx];
        char nm[64]; snprintf(nm, sizeof nm, "layer %d (%d): sum", idx, c.engram_layer_ids[idx]);
        if (sum == want) printf("PASS %-46s (%lld)\n", nm, (long long)sum);
        else { printf("FAIL %-46s got=%lld exp=%lld\n", nm, (long long)sum, (long long)want); fail = 1; }
    }
    printf("  n primes visti = %d (atteso %d)\n", seen_n, c.n_engram_layers * depth * heads);

    /* multipliers reali (NumPy PCG64): attesi dai seed 10007/140098 */
    static const int64_t exp1[4] = { 76632096046245LL, 4839876093313LL, 35959672319349LL, 73987337458391LL };
    static const int64_t exp14[4] = { 67716810739261LL, 51510806800915LL, 30921347202721LL, 82619226485591LL };
    int64_t m1[4], m14[4];
    llm_dsv41_engram_multipliers(c.engram_compressed_vocab_size, 1, c.engram_max_ngram_size, m1);
    llm_dsv41_engram_multipliers(c.engram_compressed_vocab_size, 14, c.engram_max_ngram_size, m14);
    int ok1 = 1, ok14 = 1;
    for (int i = 0; i < 4; i++) { if (m1[i] != exp1[i]) ok1 = 0; if (m14[i] != exp14[i]) ok14 = 0; }
    if (ok1) printf("PASS %-46s\n", "multipliers layer 1 (PCG64 C)");
    else { printf("FAIL %-46s got=[%lld %lld %lld %lld]\n", "multipliers layer 1 (PCG64 C)",
                  (long long)m1[0], (long long)m1[1], (long long)m1[2], (long long)m1[3]); fail = 1; }
    if (ok14) printf("PASS %-46s\n", "multipliers layer 14 (PCG64 C)");
    else { printf("FAIL %-46s\n", "multipliers layer 14 (PCG64 C)"); fail = 1; }

    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
