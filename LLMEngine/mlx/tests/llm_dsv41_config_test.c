/* llm_dsv41_config_test.c — verifica del parsing config.json DeepSeek V4.1.
 * Confronta `llm_dsv41_config_from_json` con i due config reali del release (inferenza e root)
 * e con un JSON inline con chiavi-alias. Nessun MLX: solo cJSON.
 *   cc -O2 -Ibuild/_deps/cjson-src -o /tmp/llm_dsv41_config_test \
 *      native/llmengine_mlx/llm_dsv41_config_test.c build/_deps/cjson-build/libcjson.a -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm_dsv41_config.h"

static int g_fail = 0;
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-46s\n", name); \
    else { printf("FAIL %-46s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)
#define EQ(a, b) ((a) == (b))

static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc((size_t)n + 1);
    size_t rd = fread(b, 1, (size_t)n, f); fclose(f); b[rd] = '\0';
    return b;
}

#define CRC(cond, what) do { if (cond) printf("PASS %-46s\n", what); \
    else { printf("FAIL [%s] %s\n", label, what); g_fail++; } } while (0)

static void check_real(const char* path, const char* label) {
    char* json = slurp(path);
    if (!json) { printf("SKIP %s (file assente: %s)\n", label, path); return; }
    LlmDsv41Config c;
    int ok = llm_dsv41_config_from_json(json, &c);
    free(json);
    CRC(ok, "parse");
    CRC(EQ(c.dim, 5120) && EQ(c.n_layers, 40) && EQ(c.n_heads, 64) && EQ(c.head_dim, 512) &&
        EQ(c.rope_head_dim, 64) && EQ(c.q_lora_rank, 1280) && EQ(c.o_groups, 8) &&
        EQ(c.o_lora_rank, 1024) && EQ(c.n_routed_experts, 384) && EQ(c.n_activated_experts, 6) &&
        EQ(c.moe_inter_dim, 2304) && EQ(c.window_size, 128) && EQ(c.vocab_size, 129280) &&
        EQ(c.hc_mult, 4), "shape");
    CRC(EQ(c.index_n_heads, 32) && EQ(c.index_head_dim, 128) && EQ(c.index_topk, 512) &&
        EQ(c.candidate_source_layer, 20), "index/candidate");
    CRC(EQ(c.in_kv_source[2], 1) && EQ(c.in_kv_source[8], 1) && EQ(c.in_kv_source[14], 1) &&
        EQ(c.in_kv_source[20], 1) && !EQ(c.in_kv_source[3], 1), "kv_source");
    CRC(EQ(c.in_index_source[2], 1) && EQ(c.in_index_source[36], 1) && !EQ(c.in_index_source[3], 1),
        "index_source");
    CRC(EQ(c.compress_ratios[0], 0) && EQ(c.compress_ratios[2], 2) && EQ(c.compress_ratios[20], 1) &&
        EQ(c.compress_ratios[39], 1), "compress_ratios");
    CRC(EQ(c.n_engram_layers, 2) && EQ(c.engram_layer_ids[0], 1) && EQ(c.engram_layer_ids[1], 14),
        "engram");
    CRC(EQ(c.original_seq_len, 65536) && EQ((int)c.rope_factor, 16) &&
        EQ((int)c.beta_fast, 32) && EQ((int)c.beta_slow, 1), "rope/YaRN");
    CRC(c.route_scale > 1.4f && c.route_scale < 1.6f &&
        c.swiglu_limit > 9.9f && c.swiglu_limit < 10.1f, "route/swiglu");
    CRC(c.score_func == LLM_DSV41_SCORE_SQRTSOFTPLUS, "score_func");
    CRC(EQ(c.vision_n_layers, 32) && EQ(c.image_token_id, 129264), "vision");
    CRC(EQ(c.n_mtp_layers, 3), "n_mtp_layers");
    CRC(llm_dsv41_config_validate(&c) == LLM_DSV41_OK, "validate");
}

int main(void) {
    printf("llm_dsv41_config — parsing config.json\n");
    check_real("/tmp/dsv41_config.json", "T1 inference config");
    check_real("/tmp/dsv41_root_config.json", "T2 root config (text_config+alias)");

    /* T3: JSON inline con soli alias + rope_scaling, per il percorso alias. */
    const char* j =
        "{\"model_type\":\"deepseek_v41\",\"text_config\":{"
        "\"hidden_size\":64,\"moe_intermediate_size\":32,\"num_hidden_layers\":2,"
        "\"num_attention_heads\":4,\"head_dim\":32,\"qk_rope_head_dim\":4,"
        "\"q_lora_rank\":8,\"o_lora_rank\":8,\"o_groups\":2,\"vocab_size\":128,"
        "\"n_routed_experts\":4,\"n_shared_experts\":1,\"num_experts_per_tok\":2,"
        "\"scoring_func\":\"softmax\",\"routed_scaling_factor\":1.0,"
        "\"rms_norm_eps\":1e-6,\"sliding_window\":16,\"hc_mult\":4,"
        "\"compress_ratios\":[0,0],\"kv_source_layer_ids\":[],\"index_source_layer_ids\":[],"
        "\"rope_scaling\":{\"factor\":4,\"beta_fast\":32,\"beta_slow\":1,"
        "\"original_max_position_embeddings\":8192},\"engram_layer_ids\":[],"
        "\"engram_num_embeddings\":[],\"index_head_dim\":32,\"index_n_heads\":2,\"index_topk\":8}}";
    LlmDsv41Config a;
    CHECK(llm_dsv41_config_from_json(j, &a), "T3 alias parse", "");
    CHECK(EQ(a.dim, 64) && EQ(a.n_layers, 2) && EQ(a.n_heads, 4) && EQ(a.rope_head_dim, 4) &&
          EQ(a.n_activated_experts, 2) && EQ(a.window_size, 16), "T3 alias shape", "");
    CHECK(a.score_func == LLM_DSV41_SCORE_SOFTMAX, "T3 alias score_func", "got %d", a.score_func);
    CHECK(EQ(a.original_seq_len, 8192) && EQ((int)a.rope_factor, 4), "T3 rope_scaling", "");
    CHECK(llm_dsv41_config_validate(&a) == LLM_DSV41_OK, "T3 validate", "");

    /* T4: config_real() della libreria deve validare OK. */
    LlmDsv41Config r = llm_dsv41_config_real();
    CHECK(llm_dsv41_config_validate(&r) == LLM_DSV41_OK, "T4 config_real validate", "");

    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail != 0;
}
