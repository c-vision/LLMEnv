/* llm_pflash_test.c — self-test LOAD-FREE di llm_pflash.h (nessun modello,
 * nessun device MLX, millisecondi). Compilazione:
 *   cc -O2 -o /tmp/llm_pflash_test native/llmengine_mlx/llm_pflash_test.c -lm
 *   /tmp/llm_pflash_test
 * Esito: stampa una riga per caso e termina con rc=0 se TUTTI passano.
 *
 * Copre (specchio dei test token-level del riferimento
 * `tests/test_pflash_needle.py` + invarianti nostre):
 *  T1  haystack 4096/8192 coi default -> RIFIUTO "insufficient_middle_budget"
 *      e prompt byte-identico (safety: mai cancellare il corpo);
 *  T2  haystack 16384 -> compressione REALE e needle (token rari ripetuti
 *      nella query) sopravvissuto su tutte e 3 le posizioni 5%/50%/95%;
 *  T3  budget invariant a keep_ratio 0.02 (compressione materialmente < 25%);
 *  T4  determinismo (due run -> output identico) e ordine preservato
 *      (l'output e' una sotto-sequenza dell'input);
 *  T5  sink+tail preservati ESATTAMENTE nel caso compresso;
 *  T6  gate strutturali: off / threshold / tools / protected_prompt;
 *  T7  parsing env (default AUTO; LLM_PFLASH=auto|always|off, soglia, include-tools).
 *
 * NOTA (2026-09-18): il default e' stato PROMOSSO da OFF ad AUTO dopo il gate di
 * qualita' (needle recall 4/4 posizioni 5/25/50/75% a ~20k + control senza
 * fabbricazione). T6a/T7a sono state aggiornate di conseguenza.
 */

#include "llm_pflash.h"

static int g_fail = 0;

#define CHECK(cond, name, fmt, ...)                                            \
    do {                                                                       \
        if (cond) {                                                            \
            printf("PASS %-46s " fmt "\n", name, ##__VA_ARGS__);               \
        } else {                                                               \
            printf("FAIL %-46s " fmt "\n", name, ##__VA_ARGS__);               \
            g_fail++;                                                          \
        }                                                                      \
    } while (0)

/* Haystack sintetico identico al riferimento: sink + filler + needle +
 * filler + query tail. Il needle (id rari >= 500000) riappare nella query
 * tail: e' il caso che lo scoring per overlap e' progettato per preservare. */
static int build_haystack(int ctx_tokens, double needle_position_frac,
                          int32_t* prompt) {
    static const int32_t sink_n = 64;
    static const int32_t filler = 7;
    static const int32_t needle[4] = {500001, 500002, 500003, 500004};
    static const int32_t query_tail[6] = {600000, 500001, 500002, 500003, 500004, 600001};
    int pos = 0;
    for (int i = 0; i < sink_n; i++) prompt[pos++] = i;
    int middle_budget = ctx_tokens - sink_n - 4 - 6;
    int insert_at = (int)((double)middle_budget * needle_position_frac);
    for (int i = 0; i < middle_budget; i++) {
        if (i == insert_at)
            for (int k = 0; k < 4; k++) prompt[pos++] = needle[k];
        prompt[pos++] = filler;
    }
    for (int k = 0; k < 6; k++) prompt[pos++] = query_tail[k];
    return pos;
}

/* L'output e' una sotto-sequenza (ordine preservato) dell'input? */
static int is_subsequence(const int32_t* in, int n, const int32_t* out, int m) {
    int j = 0;
    for (int i = 0; i < n && j < m; i++) if (in[i] == out[j]) j++;
    return j == m;
}

static int contains(const int32_t* v, int n, int32_t x) {
    for (int i = 0; i < n; i++) if (v[i] == x) return 1;
    return 0;
}

int main(void) {
    printf("llm_pflash_test — self-test load-free del compressore PFlash\n");
    enum { MAXN = 32768 };
    static int32_t prompt[MAXN];
    static int32_t out[MAXN];
    static int32_t out2[MAXN];

    /* T1 — safety sui contesti corti coi default validati. */
    {
        const double fracs[3] = {0.05, 0.5, 0.95};
        for (int f = 0; f < 3; f++) {
            for (int ci = 0; ci < 2; ci++) {
                int ctx = ci == 0 ? 4096 : 8192;
                int m = build_haystack(ctx, fracs[f], prompt);
                LlmPflashConfig cfg;
                llm_pflash_config_defaults(&cfg);
                cfg.mode = LLM_PFLASH_MODE_ALWAYS;
                LlmPflashResult r;
                int ok = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r);
                int same = (r.kept_tokens == m) && memcmp(out, prompt, (size_t)m * 4) == 0;
                char name[64];
                snprintf(name, sizeof(name), "T1 ctx=%d pos=%.2f unchanged", ctx, fracs[f]);
                CHECK(ok && !r.compressed && strcmp(r.reason, LLM_PFLASH_RSN_MIDDLE) == 0 && same,
                      name, "reason=%s kept=%d", r.reason, r.kept_tokens);
            }
        }
    }

    /* T2 — compressione reale a 16k + needle preservato (3 posizioni). */
    {
        const double fracs[3] = {0.05, 0.5, 0.95};
        for (int f = 0; f < 3; f++) {
            int m = build_haystack(16384, fracs[f], prompt);
            LlmPflashConfig cfg;
            llm_pflash_config_defaults(&cfg);
            cfg.mode = LLM_PFLASH_MODE_ALWAYS;
            LlmPflashResult r;
            int ok = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r);
            int needle_ok = contains(out, r.kept_tokens, 500001) &&
                            contains(out, r.kept_tokens, 500002) &&
                            contains(out, r.kept_tokens, 500003) &&
                            contains(out, r.kept_tokens, 500004);
            int subseq = is_subsequence(prompt, m, out, r.kept_tokens);
            char name[64];
            snprintf(name, sizeof(name), "T2 ctx=16384 pos=%.2f needle+order", fracs[f]);
            CHECK(ok && r.compressed && needle_ok && subseq, name,
                  "kept=%d/%d ratio=%.3f", r.kept_tokens, m,
                  (double)r.kept_tokens / (double)m);
        }
    }

    /* T3 — budget invariant aggressivo (keep_ratio 0.02). */
    {
        int m = build_haystack(16384, 0.5, prompt);
        LlmPflashConfig cfg;
        llm_pflash_config_defaults(&cfg);
        cfg.mode = LLM_PFLASH_MODE_ALWAYS;
        cfg.keep_ratio = 0.02;
        cfg.min_keep_tokens = 64;
        cfg.sink_tokens = 16;
        cfg.tail_tokens = 16;
        cfg.query_window = 64;
        cfg.stride_blocks = 0;
        LlmPflashResult r;
        int ok = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r);
        int expected_max = (int)(m * 0.02) + cfg.sink_tokens + cfg.tail_tokens + 32;
        CHECK(ok && r.compressed && r.kept_tokens <= expected_max && r.kept_tokens < m / 4,
              "T3 budget invariant ratio=0.02", "kept=%d max=%d (<%d)", r.kept_tokens,
              expected_max, m / 4);
    }

    /* T4 — determinismo + sotto-sequenza. */
    {
        int m = build_haystack(16384, 0.5, prompt);
        LlmPflashConfig cfg;
        llm_pflash_config_defaults(&cfg);
        cfg.mode = LLM_PFLASH_MODE_ALWAYS;
        LlmPflashResult r1, r2;
        int ok1 = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r1);
        int ok2 = llm_pflash_compress(prompt, m, &cfg, 0, 0, out2, MAXN, &r2);
        int same = ok1 && ok2 && r1.kept_tokens == r2.kept_tokens &&
                   memcmp(out, out2, (size_t)r1.kept_tokens * 4) == 0;
        CHECK(same && is_subsequence(prompt, m, out, r1.kept_tokens),
              "T4 determinism + subsequence", "kept=%d", r1.kept_tokens);
    }

    /* T5 — sink+tail preservati esattamente. */
    {
        int m = build_haystack(16384, 0.5, prompt);
        LlmPflashConfig cfg;
        llm_pflash_config_defaults(&cfg);
        cfg.mode = LLM_PFLASH_MODE_ALWAYS;
        LlmPflashResult r;
        int ok = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r);
        int sink_ok = 1, tail_ok = 1;
        for (int i = 0; i < cfg.sink_tokens && i < r.kept_tokens; i++)
            if (out[i] != prompt[i]) sink_ok = 0;
        for (int i = 0; i < cfg.tail_tokens && i < r.kept_tokens; i++) {
            int a = out[r.kept_tokens - 1 - i];
            int b = prompt[m - 1 - i];
            if (a != b) tail_ok = 0;
        }
        CHECK(ok && r.compressed && sink_ok && tail_ok, "T5 sink+tail preserved",
              "sink=%d tail=%d", sink_ok, tail_ok);
    }

    /* T6 — gate strutturali. */
    {
        int m = build_haystack(16384, 0.5, prompt);
        LlmPflashConfig cfg;
        LlmPflashResult r;

        llm_pflash_config_defaults(&cfg);
        cfg.mode = LLM_PFLASH_MODE_OFF; /* esplicito: dal 2026-09-18 il default e' AUTO */
        int ok = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r);
        CHECK(ok && !r.compressed && strcmp(r.reason, LLM_PFLASH_RSN_OFF) == 0,
              "T6a mode=off", "reason=%s", r.reason);

        llm_pflash_config_defaults(&cfg);
        cfg.mode = LLM_PFLASH_MODE_AUTO; /* soglia 32768 > 16384 */
        ok = llm_pflash_compress(prompt, m, &cfg, 0, 0, out, MAXN, &r);
        CHECK(ok && !r.compressed && strcmp(r.reason, LLM_PFLASH_RSN_THRESHOLD) == 0,
              "T6b auto sotto soglia", "reason=%s", r.reason);

        llm_pflash_config_defaults(&cfg);
        cfg.mode = LLM_PFLASH_MODE_ALWAYS;
        ok = llm_pflash_compress(prompt, m, &cfg, /*has_tools=*/1, 0, out, MAXN, &r);
        CHECK(ok && !r.compressed && strcmp(r.reason, LLM_PFLASH_RSN_TOOLS) == 0,
              "T6c tools fail-closed", "reason=%s", r.reason);

        ok = llm_pflash_compress(prompt, m, &cfg, 0, /*integrity=*/1, out, MAXN, &r);
        CHECK(ok && !r.compressed && strcmp(r.reason, LLM_PFLASH_RSN_PROTECTED) == 0,
              "T6d protected prompt", "reason=%s", r.reason);
    }

    /* T7 — parsing env. */
    {
        LlmPflashConfig cfg;
        unsetenv("LLM_PFLASH");
        /* PROMOSSO default ON 2026-09-18: env assente => AUTO (attivo solo sopra soglia). */
        int off = llm_pflash_config_from_env(&cfg);
        CHECK(off == 1 && cfg.mode == LLM_PFLASH_MODE_AUTO &&
                  !llm_pflash_should_attempt(&cfg, 32767) && llm_pflash_should_attempt(&cfg, 32768),
              "T7a env unset -> auto (default ON)", "mode=%d", cfg.mode);
        setenv("LLM_PFLASH", "auto", 1);
        setenv("LLM_PFLASH_THRESHOLD", "1000", 1);
        int on = llm_pflash_config_from_env(&cfg);
        CHECK(on == 1 && cfg.mode == LLM_PFLASH_MODE_AUTO && cfg.threshold == 1000 &&
                  !llm_pflash_should_attempt(&cfg, 999) && llm_pflash_should_attempt(&cfg, 1000),
              "T7b env auto+threshold", "mode=%d thr=%d", cfg.mode, cfg.threshold);
        setenv("LLM_PFLASH", "always", 1);
        setenv("LLM_PFLASH_KEEP_RATIO", "0.35", 1);
        on = llm_pflash_config_from_env(&cfg);
        CHECK(on == 1 && cfg.mode == LLM_PFLASH_MODE_ALWAYS &&
                  cfg.keep_ratio > 0.349 && cfg.keep_ratio < 0.351 &&
                  llm_pflash_should_attempt(&cfg, 1),
              "T7c env always+ratio", "mode=%d ratio=%.3f", cfg.mode, cfg.keep_ratio);
        setenv("LLM_PFLASH", "off", 1);
        CHECK(llm_pflash_config_from_env(&cfg) == 0, "T7d env off -> 0", "ok");
        unsetenv("LLM_PFLASH");
        unsetenv("LLM_PFLASH_THRESHOLD");
        unsetenv("LLM_PFLASH_KEEP_RATIO");
    }

    if (g_fail == 0) printf("ALL PASS\n");
    else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
