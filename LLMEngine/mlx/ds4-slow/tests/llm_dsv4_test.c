/* llm_dsv4_test.c — verifica config/policy DeepSeek V4 Flash (port DS4). */
#include <stdio.h>
#include "llm_dsv4.h"
#include <string.h>
static int g_cnt, g_comp, g_idx, g_hash;
static int sink(void* u, const char* n) {
    (void)u; g_cnt++;
    if (strstr(n, "attn_compressor_")) g_comp++;
    if (strstr(n, "indexer_compressor_")) g_idx++;
    if (strstr(n, "ffn_gate_tid2eid")) g_hash++;
    return 0;
}
int main(void) {
    LlmDsv4Config c = llm_dsv4_config_flash();
    int fail = 0;
    /* ratio policy Flash: 0,0,4,128,4,128,...,4 */
    int exp[8] = { 0, 0, 4, 128, 4, 128, 4, 128 };
    int ok = 1;
    for (int i = 0; i < 8; i++) if (llm_dsv4_layer_compress_ratio(&c, i) != exp[i]) { printf("  ratio[%d]=%d exp=%d\n", i, llm_dsv4_layer_compress_ratio(&c, i), exp[i]); ok = 0; }
    if (llm_dsv4_layer_compress_ratio(&c, 42) != 4) { printf("  ratio[42]=%d exp=4\n", llm_dsv4_layer_compress_ratio(&c, 42)); ok = 0; }
    if (llm_dsv4_layer_compress_ratio(&c, 43) != -1) ok = 0;
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "ratio CSA/HCA Flash"); if (!ok) fail++;
    int v = llm_dsv4_validate(&c);
    printf("%s %-40s (v=%d)\n", v == 0 ? "PASS" : "FAIL", "validate Flash (reale)", v); if (v) fail++;
    /* invarianti attese */
    ok = (c.n_layer == 43 && c.n_embd == 4096 && c.n_head == 64 && c.n_head_kv == 1 &&
          c.n_head_dim == 512 && c.n_rot == 64 && c.n_expert == 256 && c.n_expert_used == 6 &&
          c.n_swa == 128 && c.n_indexer_top_k == 512 && c.n_hc == 4);
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "costanti V4 Flash"); if (!ok) fail++;
    /* negative: head_dim non multiplo di 32 */
    LlmDsv4Config b = c; b.n_head_dim = 500;
    printf("%s %-40s\n", llm_dsv4_validate(&b) == LLM_DSV4_E_DIM ? "PASS" : "FAIL", "reject head_dim%32");
    if (llm_dsv4_validate(&b) != LLM_DSV4_E_DIM) fail++;

    /* mappa nomi: conta e verifica le condizionali (compressor/indexer/hash) */
    {
        g_cnt = g_comp = g_idx = g_hash = 0;
        int rc = llm_dsv4_expected_names(&c, sink, NULL);
        /* attesi: glob 6 + 43*23 base + exp_probs_b (solo non-hash: 43-3) + compressor(41*4)
         * + indexer(21*6) + hash(3) = 1328 (come il GGUF reale ds4flash.gguf) */
        int exp = 6 + 43*23 + (43-3) + 41*4 + 21*6 + 3;
        int ok = (rc == 0 && g_cnt == exp && g_hash == 3);
        printf("%s %-40s (cnt=%d exp=%d hash=%d)\n", ok ? "PASS" : "FAIL", "expected_names V4 Flash", g_cnt, exp, g_hash);
        if (!ok) fail++;
    }
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
