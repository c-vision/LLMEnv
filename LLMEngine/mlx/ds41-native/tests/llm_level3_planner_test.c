/* llm_level3_planner_test.c — self-test LOAD-FREE (nessun modello/device).
 *   cc -O2 -o /tmp/llm_level3_planner_test native/llmengine_mlx/llm_level3_planner_test.c -lm
 * Copre: validate()/ced_layout_supported() (port dsv41) + le decisioni del planner Level-3. */
#include "llm_level3_planner.h"

static int g_fail = 0;
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-52s\n", name); \
    else { printf("FAIL %-52s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static LlmDsv41Layout valid_ced_layout(void) {
    LlmDsv41Layout L; memset(&L, 0, sizeof L);
    L.model_type = "deepseek_v41";
    L.n_layers = 6; L.n_heads = 16; L.o_groups = 8; L.head_dim = 128;
    L.rope_head_dim = 32; L.index_head_dim = 64;
    L.n_shared_experts = 1; L.n_routed_experts = 8; L.n_activated_experts = 2;
    L.window_size = 128;
    int r[] = {0, 2, 2, 1, 1, 1};
    memcpy(L.compress_ratios, r, sizeof r); L.compress_ratio_n = 6;
    int ks[] = {1, 3}; memcpy(L.kv_source_layers, ks, sizeof ks); L.kv_source_n = 2;
    int is[] = {1, 3}; memcpy(L.index_source_layers, is, sizeof is); L.index_source_n = 2;
    L.engram_layer_n = 0; L.engram_num_embeddings_n = 0;
    L.ced_prefill = 1;
    return L;
}

int main(void) {
    printf("llm_level3_planner — self-test load-free\n");

    /* T1 — layout CED valido. */
    {
        LlmDsv41Layout L = valid_ced_layout();
        int rc = llm_dsv41_validate(&L);
        CHECK(rc == LLM_DSV41_OK && llm_dsv41_ced_layout_supported(&L) == 1,
              "T1 dsv41 layout valido + CED supportato", "rc=%d", rc);
    }

    /* T2 — errori di validate(). */
    {
        LlmDsv41Layout L;
        L = valid_ced_layout(); L.compress_ratio_n = 5;                       CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_RATIOS,  "T2a ratios corti", "");
        L = valid_ced_layout(); L.o_groups = 7;                               CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_HEADS,   "T2b heads/o_groups", "");
        L = valid_ced_layout(); L.rope_head_dim = 200;                        CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_ROPE,    "T2c rope_head_dim", "");
        L = valid_ced_layout(); L.n_shared_experts = 2;                       CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_INDEX,   "T2d shared experts != 1", "");
        L = valid_ced_layout(); L.n_activated_experts = 9;                    CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_EXPERTS, "T2e activated > routed", "");
        /* ratio 2 su un layer senza kv_source compatibile */
        L = valid_ced_layout(); L.kv_source_n = 0;                            CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_KV_SOURCE,"T2f nessun kv_source", "");
        /* engram: 1 layer ma 0 tabelle */
        L = valid_ced_layout(); L.engram_layer_ids[0] = 0; L.engram_layer_n = 1; CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_ENGRAM, "T2g engram mismatch", "");
        /* ced richiesto ma layout non supportato (engram nel decoder half) */
        L = valid_ced_layout(); L.engram_layer_ids[0] = 5; L.engram_layer_n = 1;
        L.engram_num_embeddings_n = 1;                                       CHECK(llm_dsv41_validate(&L) == LLM_DSV41_E_CED_LAYOUT, "T2h ced layout non supportato", "");
    }

    /* T3 — ced_layout_supported: rifiuti mirati. */
    {
        LlmDsv41Layout L;
        L = valid_ced_layout(); L.ced_prefill = 0; L.n_layers = 7;            CHECK(llm_dsv41_ced_layout_supported(&L) == 0, "T3a n_layers dispari", "");
        L = valid_ced_layout(); L.compress_ratios[5] = 2;                     CHECK(llm_dsv41_ced_layout_supported(&L) == 0, "T3b decoder half ratio != 1", "");
        L = valid_ced_layout(); L.engram_layer_ids[0] = 5; L.engram_layer_n = 1; L.engram_num_embeddings_n = 1; CHECK(llm_dsv41_ced_layout_supported(&L) == 0, "T3c engram nel decoder half", "");
        L = valid_ced_layout(); L.window_size = 0;                            CHECK(llm_dsv41_ced_layout_supported(&L) == 0, "T3d window 0", "");
    }

    /* T4 — planner prefill: PFlash. */
    {
        LlmL3PlannerInput in; LlmL3Plan p;
        memset(&in, 0, sizeof in); in.model_type = "qwen3_5_moe"; in.is_prefill = 1;
        in.n_tokens = 65536; in.is_cold_session = 1; in.pflash_mode = 1; in.pflash_threshold = 32768;
        llm_l3_plan(&in, &p);
        int skip = 0, persist = 0;
        for (int i = 0; i < p.n; i++) { if (p.items[i].action == LLM_L3_SKIP && strcmp(p.items[i].reason, "pflash") == 0) skip = 1;
                                        if (p.items[i].action == LLM_L3_PERSIST) persist = 1; }
        CHECK(skip && persist, "T4a prefill 64k cold AUTO -> SKIP pflash + PERSIST", "n=%d", p.n);

        in.has_tools = 1; llm_l3_plan(&in, &p);
        int exec = 0; for (int i = 0; i < p.n; i++) if (p.items[i].action == LLM_L3_EXECUTE && strcmp(p.items[i].reason, "pflash_tools_failclosed") == 0) exec = 1;
        CHECK(exec, "T4b tools -> EXECUTE (fail-closed)", "n=%d", p.n);

        in.has_tools = 0; in.n_tokens = 8192; llm_l3_plan(&in, &p);
        exec = 0; for (int i = 0; i < p.n; i++) if (p.items[i].action == LLM_L3_EXECUTE) exec = 1;
        CHECK(exec, "T4c prefill 8k < soglia -> EXECUTE", "n=%d", p.n);
    }

    /* T5 — planner prefill: CED (solo dsv41 con layout CED). */
    {
        LlmDsv41Layout L = valid_ced_layout();
        LlmL3PlannerInput in; LlmL3Plan p; memset(&in, 0, sizeof in);
        in.model_type = "deepseek_v41"; in.is_prefill = 1; in.n_tokens = 4096; in.is_cold_session = 1;
        in.pflash_mode = 0; in.ced_enabled = 1; in.layout = &L;
        llm_l3_plan(&in, &p);
        int ced = 0; for (int i = 0; i < p.n; i++) if (p.items[i].action == LLM_L3_SKIP && strcmp(p.items[i].reason, "ced") == 0) ced = 1;
        CHECK(ced, "T5a dsv41 + ced -> SKIP decoder_half", "n=%d", p.n);
        in.ced_enabled = 0; llm_l3_plan(&in, &p);
        ced = 0; for (int i = 0; i < p.n; i++) if (strcmp(p.items[i].reason, "ced") == 0) ced = 1;
        CHECK(!ced, "T5b ced off -> nessuno skip decoder", "n=%d", p.n);
        in.ced_enabled = 1; in.n_tokens = 100; /* <= window_size */
        llm_l3_plan(&in, &p);
        ced = 0; for (int i = 0; i < p.n; i++) if (strcmp(p.items[i].reason, "ced") == 0) ced = 1;
        CHECK(!ced, "T5c prompt <= window -> nessun CED", "n=%d", p.n);
    }

    /* T6 — planner decode: riuso. */
    {
        LlmL3PlannerInput in; LlmL3Plan p; memset(&in, 0, sizeof in);
        in.model_type = "qwen3_5_moe"; in.is_prefill = 0; in.prefix_cache_hit = 1; in.moe_routing_unchanged = 1;
        llm_l3_plan(&in, &p);
        int kvc = 0, moe = 0;
        for (int i = 0; i < p.n; i++) { if (strcmp(p.items[i].component, "kv_prefix") == 0) kvc = 1;
                                        if (strcmp(p.items[i].component, "moe_routing") == 0) moe = 1; }
        CHECK(kvc && moe, "T6 decode -> REUSE kv_prefix + moe_routing", "n=%d", p.n);
    }

    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
