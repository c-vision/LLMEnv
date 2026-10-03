/* moe_sim.c — K1: replay offline della cache esperti da un trace `.moetrace`.
 *
 * Uso:
 *   moe_sim --trace <file.moetrace> [--expert-size BYTES] [--budget B1,B2,...] [--slots S1,S2,...]
 *           [--verify] [--json]
 *
 * Sweep di capacita' → hit-rate VERA, resident-rate (K2), byte letti, esperti unici.
 * `--verify` ri-gioca alla capacita' registrata nel trace (header) e confronta con il
 * summary del trace stesso (stesso trace, stessa policy → gli stessi hit/miss/eviction).
 *
 * La policy e' quella del runtime (usa llmengine_moe_cache): nessuna copia divergente. */
#include "llmengine_moe_sim.h"
#include "llmengine_moe_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* argv0) {
    fprintf(stderr,
        "uso: %s --trace <file.moetrace> [--expert-size BYTES] [--budget B1,B2,...] "
        "[--slots S1,S2,...] [--verify] [--json]\n", argv0);
}

static uint64_t parse_u64(const char* s) {
    if (s == NULL) return 0;
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 0);
    return (end && end != s) ? (uint64_t)v : 0;
}

/* Lista di budget in byte: ogni voce "B" o "Nk"/"Nm"/"Ng" (KiB/MiB/GiB). */
static int parse_budget_list(const char* s, uint64_t* out, int max, int* n) {
    if (s == NULL) return 0;
    char* copy = strdup(s);
    if (copy == NULL) return 0;
    char* save = NULL;
    for (char* tok = strtok_r(copy, ",", &save); tok != NULL; tok = strtok_r(NULL, ",", &save)) {
        uint64_t mult = 1;
        size_t len = strlen(tok);
        if (len > 1) {
            char c = tok[len - 1];
            if (c == 'k' || c == 'K') { mult = 1024ull; tok[len - 1] = 0; }
            else if (c == 'm' || c == 'M') { mult = 1024ull * 1024ull; tok[len - 1] = 0; }
            else if (c == 'g' || c == 'G') { mult = 1024ull * 1024ull * 1024ull; tok[len - 1] = 0; }
        }
        uint64_t v = parse_u64(tok);
        if (v == 0 || *n >= max) { free(copy); return 0; }
        out[(*n)++] = v * mult;
    }
    free(copy);
    return 1;
}

static void print_row(const LlmEngineMoeSimResult* r, int json) {
    if (json) {
        printf("{\"budget_bytes\":%llu,\"slots\":%llu,\"requests\":%llu,\"steps\":%llu,"
               "\"hits\":%llu,\"resident_hits\":%llu,\"hit_rate\":%.6f,\"resident_rate\":%.6f,"
               "\"misses\":%llu,\"evictions\":%llu,\"bytes_loaded\":%llu,\"unique_experts\":%llu}\n",
               (unsigned long long)r->budget_bytes, (unsigned long long)r->slots,
               (unsigned long long)r->requests, (unsigned long long)r->steps,
               (unsigned long long)r->hits, (unsigned long long)r->resident_hits,
               r->hit_rate, r->resident_rate,
               (unsigned long long)r->misses, (unsigned long long)r->evictions,
               (unsigned long long)r->bytes_loaded, (unsigned long long)r->unique_experts);
        return;
    }
    printf("budget=%-12llu slots=%-8llu hit_rate=%6.2f%% resident_rate=%6.2f%% "
           "hits=%-10llu resident=%-10llu misses=%-10llu evict=%-8llu bytes=%-14llu unique=%llu\n",
           (unsigned long long)r->budget_bytes, (unsigned long long)r->slots,
           r->hit_rate * 100.0, r->resident_rate * 100.0,
           (unsigned long long)r->hits, (unsigned long long)r->resident_hits,
           (unsigned long long)r->misses, (unsigned long long)r->evictions,
           (unsigned long long)r->bytes_loaded, (unsigned long long)r->unique_experts);
}

int main(int argc, char** argv) {
    const char* trace = NULL;
    uint64_t expert_size = 0;
    uint64_t budgets[64]; int n_budgets = 0;
    uint64_t slots[64]; int n_slots = 0;
    int verify = 0, json = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) trace = argv[++i];
        else if (strcmp(argv[i], "--expert-size") == 0 && i + 1 < argc) expert_size = parse_u64(argv[++i]);
        else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) { if (!parse_budget_list(argv[++i], budgets, 64, &n_budgets)) { usage(argv[0]); return 2; } }
        else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            char* copy = strdup(argv[++i]); char* save = NULL;
            for (char* tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
                if (n_slots < 64) slots[n_slots++] = parse_u64(tok);
            }
            free(copy);
        }
        else if (strcmp(argv[i], "--verify") == 0) verify = 1;
        else if (strcmp(argv[i], "--json") == 0) json = 1;
        else { usage(argv[0]); return 2; }
    }
    if (trace == NULL) { usage(argv[0]); return 2; }

    uint64_t hdr_budget = 0, hdr_expert = 0;
    if (llmengine_moe_trace_meta(trace, &hdr_budget, &hdr_expert) != LLMENGINE_MOE_OK) {
        fprintf(stderr, "errore: trace non leggibile: %s\n", trace);
        return 1;
    }
    if (expert_size == 0) expert_size = hdr_expert;
    if (expert_size == 0) { fprintf(stderr, "errore: --expert-size mancante (header senza expert size)\n"); return 1; }

    /* budget espliciti + slots convertiti */
    for (int i = 0; i < n_slots; i++) if (n_budgets < 64) budgets[n_budgets++] = slots[i] * expert_size;
    if (n_budgets == 0 && hdr_budget > 0) budgets[n_budgets++] = hdr_budget;
    if (n_budgets == 0) { fprintf(stderr, "errore: nessun budget (usa --budget/--slots o registra il budget nel trace)\n"); return 1; }

    int fails = 0;
    for (int i = 0; i < n_budgets; i++) {
        LlmEngineMoeSimResult r;
        LlmEngineMoeStatus s = llmengine_moe_sim_replay(trace, budgets[i], expert_size, &r);
        if (s != LLMENGINE_MOE_OK) {
            fprintf(stderr, "errore: replay a budget=%llu fallito (status %d)\n",
                    (unsigned long long)budgets[i], (int)s);
            fails++;
            continue;
        }
        print_row(&r, json);
    }

    if (verify) {
        LlmEngineMoeTraceSummary sum;
        if (llmengine_moe_trace_summary(trace, &sum) != LLMENGINE_MOE_OK) {
            fprintf(stderr, "errore: summary del trace fallito\n");
            return 1;
        }
        LlmEngineMoeSimResult r;
        uint64_t vb = hdr_budget ? hdr_budget : budgets[0];
        if (llmengine_moe_sim_replay(trace, vb, expert_size, &r) != LLMENGINE_MOE_OK) {
            fprintf(stderr, "errore: replay di verifica fallito\n");
            return 1;
        }
        int ok = (r.hits == sum.cache_hits) && (r.misses == sum.cache_misses) && (r.evictions == sum.evictions);
        printf("VERIFY budget=%llu sim(hits=%llu misses=%llu evict=%llu) vs trace(hits=%llu misses=%llu evict=%llu) -> %s\n",
               (unsigned long long)vb, (unsigned long long)r.hits, (unsigned long long)r.misses,
               (unsigned long long)r.evictions, (unsigned long long)sum.cache_hits,
               (unsigned long long)sum.cache_misses, (unsigned long long)sum.evictions, ok ? "OK" : "FAIL");
        if (!ok) fails++;
    }
    return fails ? 1 : 0;
}
