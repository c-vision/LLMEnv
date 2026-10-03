/* llmengine_moe_prof.c — Profiler MoE (M12): timing per fase + trace binario .moetrace. */
#include "llmengine_moe_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LLMENGINE_MOE_TRACE_MAGIC "MOET"
#define LLMENGINE_MOE_TRACE_VERSION 1u
#define LLMENGINE_MOE_TRACE_ENDIAN  0x01020304u

typedef struct {
    uint8_t magic[4];
    uint32_t version;
    uint32_t endian;
    uint32_t phase_count;
    uint32_t reserved;
    uint64_t reserved2;
} LlmEngineMoeTraceHeader;

struct LlmEngineMoeProfiler {
    LlmEngineMoeProfStats stats;
    int enabled[LLMENGINE_MOE_PH_COUNT];
    int active[LLMENGINE_MOE_PH_COUNT];
    uint64_t start[LLMENGINE_MOE_PH_COUNT];
    FILE* trace;
};

static const char* const k_phase_names[LLMENGINE_MOE_PH_COUNT] = {
    "ROUTER", "CACHE", "DISK", "DECOMPRESS", "UPLOAD", "COMPUTE", "COMBINE"
};

static const char* const k_event_names[] = {
    "NONE", "ROUTER", "CACHE_HIT", "CACHE_MISS", "LOAD_BEGIN", "LOAD_END",
    "PREFETCH_BEGIN", "PREFETCH_END", "COMPUTE_BEGIN", "COMPUTE_END", "EVICT",
    "PHASE_TIME"
};

const char* llmengine_moe_prof_phase_name(LlmEngineMoeProfPhase ph) {
    return (ph < LLMENGINE_MOE_PH_COUNT) ? k_phase_names[ph] : "?";
}

const char* llmengine_moe_trace_event_name(LlmEngineMoeTraceEvent ev) {
    return (ev <= LLMENGINE_MOE_TRACE_PHASE_TIME) ? k_event_names[ev] : "?";
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

LlmEngineMoeProfiler* llmengine_moe_prof_create(void) {
    LlmEngineMoeProfiler* p = (LlmEngineMoeProfiler*)calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    for (int i = 0; i < LLMENGINE_MOE_PH_COUNT; ++i) p->enabled[i] = 1;
    return p;
}

void llmengine_moe_prof_destroy(LlmEngineMoeProfiler* p) {
    if (p == NULL) return;
    llmengine_moe_prof_trace_close(p);
    free(p);
}

void llmengine_moe_prof_enable(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph, int on) {
    if (p == NULL || ph >= LLMENGINE_MOE_PH_COUNT) return;
    p->enabled[ph] = on ? 1 : 0;
}

void llmengine_moe_prof_enable_all(LlmEngineMoeProfiler* p, int on) {
    if (p == NULL) return;
    for (int i = 0; i < LLMENGINE_MOE_PH_COUNT; ++i) p->enabled[i] = on ? 1 : 0;
}

int llmengine_moe_prof_enabled(const LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph) {
    return (p == NULL || ph >= LLMENGINE_MOE_PH_COUNT) ? 0 : p->enabled[ph];
}

void llmengine_moe_prof_add(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph, uint64_t ns) {
    if (p == NULL || ph >= LLMENGINE_MOE_PH_COUNT || !p->enabled[ph]) return;
    LlmEngineMoeProfPhaseStat* st = &p->stats.phase[ph];
    st->total_ns += ns;
    st->count++;
    if (st->count == 1 || ns < st->min_ns) st->min_ns = ns;
    if (ns > st->max_ns) st->max_ns = ns;
    p->stats.total_ns += ns;
}

void llmengine_moe_prof_begin(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph) {
    if (p == NULL || ph >= LLMENGINE_MOE_PH_COUNT || !p->enabled[ph]) return;
    p->start[ph] = now_ns();
    p->active[ph] = 1;
}

uint64_t llmengine_moe_prof_end(LlmEngineMoeProfiler* p, LlmEngineMoeProfPhase ph) {
    if (p == NULL || ph >= LLMENGINE_MOE_PH_COUNT || !p->enabled[ph] || !p->active[ph]) return 0;
    uint64_t dt = now_ns() - p->start[ph];
    p->active[ph] = 0;
    llmengine_moe_prof_add(p, ph, dt);
    if (p->trace != NULL)
        llmengine_moe_prof_event(p, LLMENGINE_MOE_TRACE_PHASE_TIME, 0, 0, (uint32_t)ph, dt);
    return dt;
}

void llmengine_moe_prof_stats(const LlmEngineMoeProfiler* p, LlmEngineMoeProfStats* out) {
    if (out == NULL) return;
    if (p == NULL) { memset(out, 0, sizeof(*out)); return; }
    *out = p->stats;
}

void llmengine_moe_prof_reset(LlmEngineMoeProfiler* p) {
    if (p == NULL) return;
    memset(&p->stats, 0, sizeof(p->stats));
    memset(p->active, 0, sizeof(p->active));
}

/* ── trace ─────────────────────────────────────────────────────────────────── */
LlmEngineMoeStatus llmengine_moe_prof_trace_open_ex(LlmEngineMoeProfiler* p, const char* path,
                                          uint64_t cache_budget, uint64_t expert_size) {
    if (p == NULL || path == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (p->trace != NULL) { fclose(p->trace); p->trace = NULL; }
    FILE* f = fopen(path, "wb");
    if (f == NULL) return LLMENGINE_MOE_ERR_IO;
    LlmEngineMoeTraceHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, LLMENGINE_MOE_TRACE_MAGIC, 4);
    h.version = LLMENGINE_MOE_TRACE_VERSION;
    h.endian = LLMENGINE_MOE_TRACE_ENDIAN;
    h.phase_count = LLMENGINE_MOE_PH_COUNT;
    h.reserved = (uint32_t)expert_size;   /* K1: byte per esperto (dimensionamento a slot) */
    h.reserved2 = cache_budget;           /* K1: budget cache del run (verifica del replay) */
    if (fwrite(&h, sizeof(h), 1, f) != 1) { fclose(f); return LLMENGINE_MOE_ERR_IO; }
    p->trace = f;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_prof_trace_open(LlmEngineMoeProfiler* p, const char* path) {
    return llmengine_moe_prof_trace_open_ex(p, path, 0, 0);
}

LlmEngineMoeStatus llmengine_moe_trace_meta(const char* path, uint64_t* cache_budget, uint64_t* expert_size) {
    if (cache_budget) *cache_budget = 0;
    if (expert_size) *expert_size = 0;
    if (path == NULL) return LLMENGINE_MOE_ERR_INVALID;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return LLMENGINE_MOE_ERR_IO;
    LlmEngineMoeTraceHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return LLMENGINE_MOE_ERR_FORMAT; }
    fclose(f);
    if (memcmp(h.magic, LLMENGINE_MOE_TRACE_MAGIC, 4) != 0 || h.endian != LLMENGINE_MOE_TRACE_ENDIAN ||
        h.version != LLMENGINE_MOE_TRACE_VERSION) return LLMENGINE_MOE_ERR_FORMAT;
    if (cache_budget) *cache_budget = h.reserved2;
    if (expert_size) *expert_size = h.reserved;
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_prof_trace_close(LlmEngineMoeProfiler* p) {
    if (p == NULL || p->trace == NULL) return;
    fflush(p->trace);
    fclose(p->trace);
    p->trace = NULL;
}

int llmengine_moe_prof_trace_active(const LlmEngineMoeProfiler* p) {
    return (p != NULL && p->trace != NULL) ? 1 : 0;
}

void llmengine_moe_prof_event(LlmEngineMoeProfiler* p, LlmEngineMoeTraceEvent ev,
                        uint32_t layer, uint32_t expert, uint32_t phase, uint64_t value) {
    if (p == NULL || p->trace == NULL) return;
    LlmEngineMoeTraceRecord r;
    r.type = (uint32_t)ev;
    r.layer = layer;
    r.expert = expert;
    r.phase = phase;
    r.ts_ns = now_ns();
    r.value = value;
    (void)fwrite(&r, sizeof(r), 1, p->trace);
}

/* ── reader ────────────────────────────────────────────────────────────────── */
struct LlmEngineMoeTraceReader {
    FILE* f;
    uint32_t phase_count;
};

LlmEngineMoeStatus llmengine_moe_trace_open(const char* path, LlmEngineMoeTraceReader** out) {
    if (path == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return LLMENGINE_MOE_ERR_IO;
    LlmEngineMoeTraceHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return LLMENGINE_MOE_ERR_FORMAT; }
    if (memcmp(h.magic, LLMENGINE_MOE_TRACE_MAGIC, 4) != 0 || h.endian != LLMENGINE_MOE_TRACE_ENDIAN ||
        h.version != LLMENGINE_MOE_TRACE_VERSION || h.phase_count != LLMENGINE_MOE_PH_COUNT) {
        fclose(f);
        return LLMENGINE_MOE_ERR_FORMAT;
    }
    LlmEngineMoeTraceReader* r = (LlmEngineMoeTraceReader*)calloc(1, sizeof(*r));
    if (r == NULL) { fclose(f); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    r->f = f;
    r->phase_count = h.phase_count;
    *out = r;
    return LLMENGINE_MOE_OK;
}

int llmengine_moe_trace_next(LlmEngineMoeTraceReader* r, LlmEngineMoeTraceRecord* out) {
    if (r == NULL || out == NULL) return -1;
    size_t n = fread(out, sizeof(*out), 1, r->f);
    if (n == 1) return 1;
    if (feof(r->f)) return 0;
    return -1; /* troncato/malformato */
}

void llmengine_moe_trace_close(LlmEngineMoeTraceReader* r) {
    if (r == NULL) return;
    fclose(r->f);
    free(r);
}

/* ── summary / diff ────────────────────────────────────────────────────────── */
LlmEngineMoeStatus llmengine_moe_trace_summary(const char* path, LlmEngineMoeTraceSummary* out) {
    if (path == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    LlmEngineMoeTraceReader* r = NULL;
    LlmEngineMoeStatus st = llmengine_moe_trace_open(path, &r);
    if (st != LLMENGINE_MOE_OK) return st;
    LlmEngineMoeTraceRecord rec;
    int rc;
    while ((rc = llmengine_moe_trace_next(r, &rec)) == 1) {
        out->records++;
        switch ((LlmEngineMoeTraceEvent)rec.type) {
            case LLMENGINE_MOE_TRACE_ROUTER:        out->router++; break;
            case LLMENGINE_MOE_TRACE_CACHE_HIT:     out->cache_hits++; break;
            case LLMENGINE_MOE_TRACE_CACHE_MISS:    out->cache_misses++; break;
            case LLMENGINE_MOE_TRACE_LOAD_BEGIN:    out->loads++; break;
            case LLMENGINE_MOE_TRACE_PREFETCH_BEGIN: out->prefetches++; break;
            case LLMENGINE_MOE_TRACE_EVICT:         out->evictions += rec.value ? rec.value : 1; break;
            case LLMENGINE_MOE_TRACE_PHASE_TIME: {
                if (rec.phase >= LLMENGINE_MOE_PH_COUNT) break;
                LlmEngineMoeProfPhaseStat* s = &out->timing.phase[rec.phase];
                s->total_ns += rec.value;
                s->count++;
                if (s->count == 1 || rec.value < s->min_ns) s->min_ns = rec.value;
                if (rec.value > s->max_ns) s->max_ns = rec.value;
                out->timing.total_ns += rec.value;
                break;
            }
            default: break;
        }
    }
    llmengine_moe_trace_close(r);
    return (rc < 0) ? LLMENGINE_MOE_ERR_FORMAT : LLMENGINE_MOE_OK;
}

void llmengine_moe_trace_diff(const LlmEngineMoeTraceSummary* a, const LlmEngineMoeTraceSummary* b,
                        LlmEngineMoeTraceDiff* out) {
    if (a == NULL || b == NULL || out == NULL) return;
    memset(out, 0, sizeof(*out));
    for (int i = 0; i < LLMENGINE_MOE_PH_COUNT; ++i)
        out->phase_delta_ns[i] = (int64_t)a->timing.phase[i].total_ns -
                                 (int64_t)b->timing.phase[i].total_ns;
    out->total_delta_ns = (int64_t)a->timing.total_ns - (int64_t)b->timing.total_ns;
    out->router_delta = (int64_t)a->router - (int64_t)b->router;
    out->cache_hits_delta = (int64_t)a->cache_hits - (int64_t)b->cache_hits;
    out->cache_misses_delta = (int64_t)a->cache_misses - (int64_t)b->cache_misses;
    out->loads_delta = (int64_t)a->loads - (int64_t)b->loads;
    out->prefetches_delta = (int64_t)a->prefetches - (int64_t)b->prefetches;
    out->evictions_delta = (int64_t)a->evictions - (int64_t)b->evictions;
}
