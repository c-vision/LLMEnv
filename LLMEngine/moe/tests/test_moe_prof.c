/* test_moe_prof.c — M12 gate: timing per fase (enable/disable, min/max, somma=totale),
 * trace binario .moetrace (round-trip, corruzione) e confronto fra due run. */
#include "llmengine_moe_prof.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACE_A "/tmp/llmengine_moe_prof_a.moetrace"
#define TRACE_B "/tmp/llmengine_moe_prof_b.moetrace"

static void write_run(const char* path, const char* phase_ns[LLMENGINE_MOE_PH_COUNT],
                      uint64_t hits, uint64_t misses, uint64_t evicts) {
    LlmEngineMoeProfiler* p = llmengine_moe_prof_create();
    assert(p != NULL);
    assert(llmengine_moe_prof_trace_open(p, path) == LLMENGINE_MOE_OK);
    for (int ph = 0; ph < LLMENGINE_MOE_PH_COUNT; ++ph) {
        uint64_t ns = strtoull(phase_ns[ph], NULL, 10);
        llmengine_moe_prof_add(p, (LlmEngineMoeProfPhase)ph, ns); /* deterministico, no wall noise */
        llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_PHASE_TIME, 0, 0, (uint32_t)ph, ns);
    }
    for (uint64_t i = 0; i < hits; ++i) llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_CACHE_HIT, 0, (uint32_t)i, 0, 0);
    for (uint64_t i = 0; i < misses; ++i) llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_CACHE_MISS, 0, (uint32_t)i, 0, 0);
    for (uint64_t i = 0; i < evicts; ++i) llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_EVICT, 0, (uint32_t)i, 0, 0);
    llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_ROUTER, 1, 0, 0, 0);
    llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_LOAD_BEGIN, 1, 0, 0, 4096);
    llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_LOAD_END, 1, 0, 0, 4096);
    llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_PREFETCH_BEGIN, 1, 1, 0, 0);
    llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_PREFETCH_END, 1, 1, 0, 0);
    llmengine_moe_prof_trace_close(p);
    llmengine_moe_prof_destroy(p);
}

int main(void) {
    /* 1. profiler in-process: add deterministico + somma delle fasi = totale */
    LlmEngineMoeProfiler* p = llmengine_moe_prof_create();
    assert(p != NULL);
    assert(llmengine_moe_prof_enabled(p, LLMENGINE_MOE_PH_COMPUTE) == 1);
    llmengine_moe_prof_add(p, LLMENGINE_MOE_PH_ROUTER, 100);
    llmengine_moe_prof_add(p, LLMENGINE_MOE_PH_ROUTER, 300);
    llmengine_moe_prof_add(p, LLMENGINE_MOE_PH_COMPUTE, 1000);
    LlmEngineMoeProfStats s;
    llmengine_moe_prof_stats(p, &s);
    assert(s.phase[LLMENGINE_MOE_PH_ROUTER].count == 2);
    assert(s.phase[LLMENGINE_MOE_PH_ROUTER].total_ns == 400);
    assert(s.phase[LLMENGINE_MOE_PH_ROUTER].min_ns == 100);
    assert(s.phase[LLMENGINE_MOE_PH_ROUTER].max_ns == 300);
    assert(s.phase[LLMENGINE_MOE_PH_COMPUTE].total_ns == 1000);
    assert(s.total_ns == 1400); /* §42: somma delle fasi = totale */
    uint64_t sum = 0;
    for (int i = 0; i < LLMENGINE_MOE_PH_COUNT; ++i) sum += s.phase[i].total_ns;
    assert(sum == s.total_ns);

    /* 2. fase disabilitata: add no-op, end ritorna 0 */
    llmengine_moe_prof_enable(p, LLMENGINE_MOE_PH_DISK, 0);
    assert(llmengine_moe_prof_enabled(p, LLMENGINE_MOE_PH_DISK) == 0);
    llmengine_moe_prof_add(p, LLMENGINE_MOE_PH_DISK, 9999);
    llmengine_moe_prof_begin(p, LLMENGINE_MOE_PH_DISK);
    assert(llmengine_moe_prof_end(p, LLMENGINE_MOE_PH_DISK) == 0);
    llmengine_moe_prof_stats(p, &s);
    assert(s.phase[LLMENGINE_MOE_PH_DISK].total_ns == 0);

    /* 3. reset azzera, non tocca enabled */
    llmengine_moe_prof_reset(p);
    llmengine_moe_prof_stats(p, &s);
    assert(s.total_ns == 0);
    assert(llmengine_moe_prof_enabled(p, LLMENGINE_MOE_PH_DISK) == 0);

    /* 4. trace: due run confrontabili */
    const char* a_ns[LLMENGINE_MOE_PH_COUNT] = {"1000","2000","3000","4000","5000","6000","7000"};
    const char* b_ns[LLMENGINE_MOE_PH_COUNT] = {"1000","2000","3000","4000","5000","6000","9000"};
    write_run(TRACE_A, a_ns, 8, 2, 1);
    write_run(TRACE_B, b_ns, 5, 5, 3);

    LlmEngineMoeTraceSummary sa, sb;
    assert(llmengine_moe_trace_summary(TRACE_A, &sa) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_trace_summary(TRACE_B, &sb) == LLMENGINE_MOE_OK);
    /* una PHASE_TIME deterministica per fase (add + event) */
    assert(sa.timing.phase[LLMENGINE_MOE_PH_COMPUTE].count == 1);
    assert(sa.timing.phase[LLMENGINE_MOE_PH_COMPUTE].total_ns == 6000);
    assert(sa.timing.phase[LLMENGINE_MOE_PH_ROUTER].total_ns == 1000);
    assert(sa.timing.total_ns == 28000);
    assert(sb.timing.total_ns == 30000);
    assert(sa.cache_hits == 8 && sa.cache_misses == 2 && sa.evictions == 1);
    assert(sb.cache_hits == 5 && sb.cache_misses == 5 && sb.evictions == 3);
    assert(sa.router == 1 && sa.loads == 1 && sa.prefetches == 1);

    LlmEngineMoeTraceDiff d;
    llmengine_moe_trace_diff(&sa, &sb, &d);
    assert(d.cache_hits_delta == 3);
    assert(d.cache_misses_delta == -3);
    assert(d.evictions_delta == -2);
    assert(d.total_delta_ns == -2000);
    assert(d.phase_delta_ns[LLMENGINE_MOE_PH_COMBINE] == 7000 - 9000);

    /* 5. reader manuale: conta i record e rilegge i tipi */
    LlmEngineMoeTraceReader* r = NULL;
    assert(llmengine_moe_trace_open(TRACE_A, &r) == LLMENGINE_MOE_OK);
    uint64_t n = 0, phase_times = 0;
    LlmEngineMoeTraceRecord rec;
    int rc;
    while ((rc = llmengine_moe_trace_next(r, &rec)) == 1) {
        n++;
        if (rec.type == LLMENGINE_MOE_TRACE_PHASE_TIME) phase_times++;
    }
    assert(rc == 0);
    assert(n == sa.records);
    assert(phase_times >= LLMENGINE_MOE_PH_COUNT);
    llmengine_moe_trace_close(r);

    /* 6. file inesistente -> IO; file corrotto -> FORMAT */
    LlmEngineMoeTraceReader* bad = NULL;
    assert(llmengine_moe_trace_open("/tmp/llmengine_moe_prof_missing.moetrace", &bad) == LLMENGINE_MOE_ERR_IO);
    FILE* f = fopen("/tmp/llmengine_moe_prof_corrupt.moetrace", "wb");
    assert(f != NULL);
    fputs("not a moetrace file at all", f);
    fclose(f);
    LlmEngineMoeTraceSummary junk;
    assert(llmengine_moe_trace_summary("/tmp/llmengine_moe_prof_corrupt.moetrace", &junk) == LLMENGINE_MOE_ERR_FORMAT);

    llmengine_moe_prof_destroy(p);
    remove(TRACE_A);
    remove(TRACE_B);
    remove("/tmp/llmengine_moe_prof_corrupt.moetrace");

    /* 7. end() emette PHASE_TIME quando il trace e' attivo */
    const char* t7 = "/tmp/llmengine_moe_prof_end.moetrace";
    LlmEngineMoeProfiler* p7 = llmengine_moe_prof_create();
    assert(p7 != NULL && llmengine_moe_prof_trace_open(p7, t7) == LLMENGINE_MOE_OK);
    llmengine_moe_prof_begin(p7, LLMENGINE_MOE_PH_UPLOAD);
    (void)llmengine_moe_prof_end(p7, LLMENGINE_MOE_PH_UPLOAD); /* emette 1 PHASE_TIME (wall, anche 0) */
    llmengine_moe_prof_trace_close(p7);
    LlmEngineMoeTraceSummary s7;
    assert(llmengine_moe_trace_summary(t7, &s7) == LLMENGINE_MOE_OK);
    assert(s7.timing.phase[LLMENGINE_MOE_PH_UPLOAD].count == 1);
    llmengine_moe_prof_destroy(p7);
    remove(t7);

    printf("test_moe_prof: PASS\n");
    return 0;
}
