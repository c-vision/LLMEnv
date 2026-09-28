/* llm_dsv41_engine_test.c — verifica dell'integrazione engine (ARCH_TABLE) DeepSeek V4.1:
 * carica il checkpoint sintetico sharded via `llm_ai_mlx_load_model` (whole-model), appende il
 * prompt e genera. Copre greedy (primo token = argmax ref), sampling temperature=0 (greedy) e
 * determinismo del sampling con seed fisso. Build: tools/dsv41_engine_test.sh */
#include <stdio.h>
#include <stdint.h>
#include "llmengine_mlx.h"

static const int32_t PROMPT[4] = {3, 7, 5, 2};
static const int EXPECT_FIRST = 23;   /* argmax(REF_CSA_LOGITS[3*64..4*64]) */

static int run(llm_ai_mlx_model_handle m, float temp, int seed, int32_t* out, int* n) {
    llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(m);
    if (!s) return -1;
    if (temp >= 0.0f) llm_ai_mlx_session_set_sampling(s, temp, 0.0f, 0, (int64_t)seed);
    int rc = llm_ai_mlx_session_append(s, PROMPT, 4);
    if (rc == 0) *n = llm_ai_mlx_session_generate(s, 3, -1, out);
    else *n = -1;
    llm_ai_mlx_destroy_session(s);
    return rc;
}

int main(void) {
    printf("llm_dsv41_engine — integrazione ARCH_TABLE (load + append + generate)\n");
    llm_ai_mlx_model_handle m = llm_ai_mlx_load_model("/tmp/dsv41_st/sharded");
    if (!m) { printf("FAIL load_model\n"); return 1; }
    int fail = 0;
    int32_t out[4]; int n;

    /* T1: greedy di default -> primo token = argmax ref */
    run(m, -1.0f, 0, out, &n);
    if (n == 3 && out[0] == EXPECT_FIRST) printf("PASS %-46s\n", "greedy default");
    else { printf("FAIL %-46s (got %d exp %d)\n", "greedy default", out[0], EXPECT_FIRST); fail = 1; }

    /* T2: sampling temperature=0 -> greedy */
    run(m, 0.0f, 0, out, &n);
    if (n == 3 && out[0] == EXPECT_FIRST) printf("PASS %-46s\n", "sampling temp=0 (greedy)");
    else { printf("FAIL %-46s (got %d exp %d)\n", "sampling temp=0 (greedy)", out[0], EXPECT_FIRST); fail = 1; }

    /* T3: sampling temp=1 seed fisso -> deterministico tra due run */
    int32_t a[4], b[4]; int na, nb;
    run(m, 1.0f, 1234, a, &na);
    run(m, 1.0f, 1234, b, &nb);
    printf("  sample seed=1234: run1=[%d %d %d] run2=[%d %d %d]\n", a[0], a[1], a[2], b[0], b[1], b[2]);
    if (na == 3 && nb == 3 && a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
        printf("PASS %-46s\n", "sampling seed fisso (deterministico)");
    else { printf("FAIL %-46s\n", "sampling seed fisso (deterministico)"); fail = 1; }

    /* T4: generate_constrained -> il token scelto appartiene ad allowed_ids */
    {
        llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(m);
        int32_t allowed[3] = {5, 7, 9};
        int32_t id = -1;
        int rc = llm_ai_mlx_session_append(s, PROMPT, 4);
        if (rc == 0) rc = llm_ai_mlx_session_generate_constrained(s, allowed, 3, &id);
        int in = (id == 5 || id == 7 || id == 9);
        printf("  constrained chose id=%d (allowed {5,7,9})\n", id);
        if (rc == 0 && in) printf("PASS %-46s\n", "constrained (allowed_ids)");
        else { printf("FAIL %-46s\n", "constrained (allowed_ids)"); fail = 1; }
        llm_ai_mlx_destroy_session(s);
    }

    /* T5: generate_batch (2 sessioni, stesso prompt) -> stesse 3 uscite greedy */
    {
        llm_ai_mlx_session_handle hs[2];
        hs[0] = llm_ai_mlx_create_session(m);
        hs[1] = llm_ai_mlx_create_session(m);
        llm_ai_mlx_session_append(hs[0], PROMPT, 4);
        llm_ai_mlx_session_append(hs[1], PROMPT, 4);
        int32_t b[6] = {0};
        int rc = llm_ai_mlx_session_generate_batch(hs, 2, 3, -1, b);
        printf("  batch: [%d %d %d] [%d %d %d]\n", b[0], b[1], b[2], b[3], b[4], b[5]);
        if (rc == 0 && b[0] == EXPECT_FIRST && b[3] == EXPECT_FIRST &&
            b[0] == b[3] && b[1] == b[4] && b[2] == b[5])
            printf("PASS %-46s\n", "generate_batch (2 sessioni)");
        else { printf("FAIL %-46s\n", "generate_batch (2 sessioni)"); fail = 1; }
        llm_ai_mlx_destroy_session(hs[0]);
        llm_ai_mlx_destroy_session(hs[1]);
    }

    /* T6: prefix cache -- save del prefisso + append_auto_prefix -> stesse uscite */
    {
        llm_ai_mlx_session_handle a = llm_ai_mlx_create_session(m);
        llm_ai_mlx_session_append(a, PROMPT, 4);
        int32_t ga[4]; int na = llm_ai_mlx_session_generate(a, 3, -1, ga);
        int rc = llm_ai_mlx_model_save_prefix(m, "p", a);

        llm_ai_mlx_session_handle b = llm_ai_mlx_create_session(m);
        int rc2 = llm_ai_mlx_session_append_auto_prefix(b, m, PROMPT, 4, 1);
        int32_t gb[4]; int nb = llm_ai_mlx_session_generate(b, 3, -1, gb);
        printf("  prefix: fresh=[%d %d %d] reuse=[%d %d %d]\n", ga[0], ga[1], ga[2], gb[0], gb[1], gb[2]);
        if (rc == 0 && rc2 == 0 && na == 3 && nb == 3 &&
            ga[0] == gb[0] && ga[1] == gb[1] && ga[2] == gb[2])
            printf("PASS %-46s\n", "prefix cache (save + auto reuse)");
        else { printf("FAIL %-46s\n", "prefix cache (save + auto reuse)"); fail = 1; }
        llm_ai_mlx_destroy_session(a);
        llm_ai_mlx_destroy_session(b);
    }

    /* T7: park -- swap_out + swap_in (rebuild da history) e unpark in-place */
    {
        llm_ai_mlx_session_handle a = llm_ai_mlx_create_session(m);
        llm_ai_mlx_session_append(a, PROMPT, 4);
        int rc0 = llm_ai_mlx_session_swap_out(a, "/tmp/dsv41_park");
        llm_ai_mlx_session_handle b = llm_ai_mlx_session_swap_in(m, "/tmp/dsv41_park");
        int32_t gb[4] = {0}; int nb = b ? llm_ai_mlx_session_generate(b, 3, -1, gb) : -1;

        llm_ai_mlx_session_handle c = llm_ai_mlx_create_session(m);
        llm_ai_mlx_session_append(c, PROMPT, 4);
        llm_ai_mlx_session_swap_out(c, "/tmp/dsv41_park2");
        int32_t gc[4] = {0}; int nc = llm_ai_mlx_session_generate(c, 3, -1, gc); /* unpark in-place */
        printf("  park: swap_in=[%d %d %d] unpark=[%d %d %d]\n", gb[0], gb[1], gb[2], gc[0], gc[1], gc[2]);
        if (rc0 == 0 && nb == 3 && nc == 3 && gb[0] == EXPECT_FIRST && gc[0] == EXPECT_FIRST &&
            gb[0] == gc[0] && gb[1] == gc[1] && gb[2] == gc[2])
            printf("PASS %-46s\n", "park (swap_in + unpark)");
        else { printf("FAIL %-46s\n", "park (swap_in + unpark)"); fail = 1; }
        llm_ai_mlx_destroy_session(a);
        if (b) llm_ai_mlx_destroy_session(b);
        llm_ai_mlx_destroy_session(c);
    }

    /* T8: greedy_prob (P del token greedy in (0,1]) */
    {
        llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(m);
        llm_ai_mlx_session_append(s, PROMPT, 4);
        float p = -1.0f;
        int rc = llm_ai_mlx_session_greedy_prob(s, &p);
        if (rc == 0 && p > 0.0f && p <= 1.0f) printf("PASS %-46s (p=%.4f)\n", "greedy_prob", p);
        else { printf("FAIL %-46s (p=%.4f)\n", "greedy_prob", p); fail = 1; }
        llm_ai_mlx_destroy_session(s);
    }

    /* T9: truncate (riavvolge al prompt e riproduce le stesse uscite) */
    {
        llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(m);
        llm_ai_mlx_session_append(s, PROMPT, 4);
        int32_t a[4]; int na = llm_ai_mlx_session_generate(s, 3, -1, a);
        int rc = llm_ai_mlx_session_truncate(s, 4);
        int32_t b[4]; int nb = llm_ai_mlx_session_generate(s, 3, -1, b);
        printf("  truncate: pre=[%d %d %d] post=[%d %d %d]\n", a[0], a[1], a[2], b[0], b[1], b[2]);
        if (rc == 0 && na == 3 && nb == 3 && a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
            printf("PASS %-46s\n", "truncate (rewind)");
        else { printf("FAIL %-46s\n", "truncate (rewind)"); fail = 1; }
        llm_ai_mlx_destroy_session(s);
    }

    llm_ai_mlx_free_model(m);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
