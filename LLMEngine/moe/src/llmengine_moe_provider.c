/* llmengine_moe_provider.c — registro multi-modello (M14). */
#include "llmengine_moe_provider.h"
#include "llmengine_moe_backend.h"
#include "llmengine_moe_adapter_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LLMENGINE_MOE_PROVIDER_MAX_NAME 64u
#define LLMENGINE_MOE_PROVIDER_INIT_CAP 4u

/* M18: la persistenza `.stats` scrive accanto al container → OPT-IN (env LLMENGINE_MOE_STATS=1), per NON
 * sporcare le fixture (`samples/Fixtures/...`, che per AGENTS §11 non vanno mai modificate). */
static int moe_stats_enabled(void) {
    const char* e = getenv("LLMENGINE_MOE_STATS");
    return e != NULL && *e && strcmp(e, "0") != 0;
}

typedef struct {
    char name[LLMENGINE_MOE_PROVIDER_MAX_NAME];
    LlmEngineMoeRuntime* rt;
    LlmEngineMoeRuntimeConfig cfg;
    uint64_t last_used;      /* M21: LRU */
    int priority;            /* M21: orchestrazione — più alto = tenuto di più (evict dal più basso) */
} MoeEntry;

struct LlmEngineMoeProvider {
    MoeEntry* entries;
    uint32_t count;
    uint32_t cap;
    uint64_t mem_available;   /* M21: memoria disponibile iniettata dal chiamante (0 = ignota) */
    uint64_t lru_clock;
};

static void provider_ensure_capacity(LlmEngineMoeProvider* p, uint64_t required, int exempt_idx);  /* M21 */

LlmEngineMoeProvider* llmengine_moe_provider_create(void) {
    LlmEngineMoeProvider* p = (LlmEngineMoeProvider*)calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    p->cap = LLMENGINE_MOE_PROVIDER_INIT_CAP;
    p->entries = (MoeEntry*)calloc(p->cap, sizeof(*p->entries));
    if (p->entries == NULL) { free(p); return NULL; }
    return p;
}

void llmengine_moe_provider_destroy(LlmEngineMoeProvider* p) {
    if (p == NULL) return;
    for (uint32_t i = 0; i < p->count; ++i) {
        if (moe_stats_enabled()) (void)llmengine_moe_stats_save(p->entries[i].rt, NULL);   /* M18 (opt-in) */
        llmengine_moe_destroy(p->entries[i].rt);
    }
    free(p->entries);
    free(p);
}

static int find_entry(const LlmEngineMoeProvider* p, const char* name) {
    for (uint32_t i = 0; i < p->count; ++i)
        if (strncmp(p->entries[i].name, name, LLMENGINE_MOE_PROVIDER_MAX_NAME) == 0) return (int)i;
    return -1;
}

LlmEngineMoeStatus llmengine_moe_provider_unload(LlmEngineMoeProvider* p, const char* name) {
    if (p == NULL || name == NULL) return LLMENGINE_MOE_ERR_INVALID;
    int idx = find_entry(p, name);
    if (idx < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    if (moe_stats_enabled()) (void)llmengine_moe_stats_save(p->entries[idx].rt, NULL);   /* M18 (opt-in) */
    llmengine_moe_destroy(p->entries[idx].rt);
    for (uint32_t i = (uint32_t)idx; i + 1 < p->count; ++i) p->entries[i] = p->entries[i + 1];
    p->count--;
    return LLMENGINE_MOE_OK;
}

/* Unico formato supportato: GGUF (ingest diretto, riconosciuto dal magic 'GGUF', senza conversione
 * offline). Il container `GGUF` è DEPRECATO e NON è più selezionato qui: se il file non è GGUF
 * ritorna un adapter vuoto (self=NULL) e il chiamante segnala l'errore. */
static LlmEngineMoeModelAdapter make_adapter_for(const char* path) {
    LlmEngineMoeModelAdapter empty; memset(&empty, 0, sizeof empty);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return empty;
    unsigned char m[4] = {0, 0, 0, 0};
    size_t got = fread(m, 1, 4, f);
    fclose(f);
    if (got == 4 && m[0] == 'G' && m[1] == 'G' && m[2] == 'U' && m[3] == 'F') {
        LlmEngineMoeModelAdapter a = llmengine_moe_gguf_adapter();
        if (a.self != NULL) return a;
    }
    return empty;
}

LlmEngineMoeStatus llmengine_moe_provider_load(LlmEngineMoeProvider* p, const char* name, const char* path,
                                   const LlmEngineMoeRuntimeConfig* cfg) {
    if (p == NULL || name == NULL || path == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (strlen(name) >= LLMENGINE_MOE_PROVIDER_MAX_NAME) return LLMENGINE_MOE_ERR_INVALID;

    /* sostituzione: se esiste, scarica prima */
    int idx = find_entry(p, name);
    if (idx >= 0) {
        LlmEngineMoeStatus us = llmengine_moe_provider_unload(p, name);
        if (us != LLMENGINE_MOE_OK) return us;
    }

    LlmEngineMoeRuntimeConfig c;
    if (cfg != NULL) c = *cfg; else llmengine_moe_config_defaults(&c);

    /* M21: per un load RESIDENT, libera memoria PRIMA di allocare (stima = dimensione file). */
    if (c.mode == LLMENGINE_MOE_MODE_RESIDENT) {
        FILE* f = fopen(path, "rb");
        if (f != NULL) { if (fseek(f, 0, SEEK_END) == 0) { long sz = ftell(f); if (sz > 0) provider_ensure_capacity(p, (uint64_t)sz, -1); } fclose(f); }
    }

    LlmEngineMoeRuntime* rt = llmengine_moe_create();
    if (rt == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    LlmEngineMoeModelAdapter ad = make_adapter_for(path);
    if (llmengine_moe_register_adapter(rt, &ad) != LLMENGINE_MOE_OK) { llmengine_moe_destroy(rt); return LLMENGINE_MOE_ERR_ADAPTER; }
    LlmEngineMoeStatus ls = llmengine_moe_load_model_ex(rt, path, &c);
    if (ls != LLMENGINE_MOE_OK) { llmengine_moe_destroy(rt); return ls; }
    if (moe_stats_enabled()) (void)llmengine_moe_stats_load(rt, NULL);   /* M18 (opt-in) */

    if (p->count == p->cap) {
        uint32_t nc = p->cap * 2;
        MoeEntry* ne = (MoeEntry*)realloc(p->entries, nc * sizeof(*ne));
        if (ne == NULL) { llmengine_moe_destroy(rt); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        p->entries = ne;
        p->cap = nc;
    }
    MoeEntry* e = &p->entries[p->count++];
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->rt = rt;
    e->cfg = c;
    e->last_used = ++p->lru_clock;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeRuntime* llmengine_moe_provider_get(LlmEngineMoeProvider* p, const char* name) {
    if (p == NULL || name == NULL) return NULL;
    int idx = find_entry(p, name);
    return idx < 0 ? NULL : p->entries[idx].rt;
}

LlmEngineMoeStatus llmengine_moe_provider_stats(LlmEngineMoeProvider* p, const char* name, LlmEngineMoeStreamStats* out) {
    if (p == NULL || name == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    int idx = find_entry(p, name);
    if (idx < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    return llmengine_moe_stream_stats(p->entries[idx].rt, out);
}

LlmEngineMoeStatus llmengine_moe_provider_reconfigure(LlmEngineMoeProvider* p, const char* name,
                                          const LlmEngineMoeRuntimeConfig* cfg) {
    if (p == NULL || name == NULL || cfg == NULL) return LLMENGINE_MOE_ERR_INVALID;
    int idx = find_entry(p, name);
    if (idx < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    LlmEngineMoeRuntime* rt = p->entries[idx].rt;

    LlmEngineMoeStatus s = llmengine_moe_set_device(rt, cfg->device);
    if (s != LLMENGINE_MOE_OK) return s;
    if (cfg->cache_mb > 0 || llmengine_moe_mode(rt) == LLMENGINE_MOE_MODE_SWAP) {
        s = llmengine_moe_set_cache_mb(rt, cfg->cache_mb);
        if (s != LLMENGINE_MOE_OK && s != LLMENGINE_MOE_ERR_INVALID) return s;
    }
    if (cfg->async_workers > 0) {
        s = llmengine_moe_set_async(rt, cfg->async_workers, cfg->prefetch_cap);
        if (s != LLMENGINE_MOE_OK) return s;
    } else {
        (void)llmengine_moe_set_async(rt, 0, 0);
    }
    s = llmengine_moe_set_prefetch(rt, cfg->prefetch_topn, 0);
    if (s != LLMENGINE_MOE_OK && s != LLMENGINE_MOE_ERR_INVALID) return s;

    p->entries[idx].cfg = *cfg;
    return LLMENGINE_MOE_OK;
}

uint32_t llmengine_moe_provider_count(const LlmEngineMoeProvider* p) { return p == NULL ? 0 : p->count; }

const char* llmengine_moe_provider_name_at(const LlmEngineMoeProvider* p, uint32_t i) {
    return (p == NULL || i >= p->count) ? NULL : p->entries[i].name;
}

/* ── M21: scheduler di residenza (LRU demote/evict sulla memoria disponibile iniettata) ──
 * UNA implementazione, condivisa da C# (IL/PCode) e backend C (C99/C99Native/VM-C). */
static uint64_t entry_expert_bytes(LlmEngineMoeRuntime* rt) {
    LlmEngineMoeStreamStats s;
    if (llmengine_moe_stream_stats(rt, &s) != LLMENGINE_MOE_OK) return 0;
    return s.expert_bytes;
}

void llmengine_moe_provider_set_mem_available(LlmEngineMoeProvider* p, uint64_t bytes) {
    if (p != NULL) p->mem_available = bytes;
}

void llmengine_moe_provider_mark_used(LlmEngineMoeProvider* p, const char* name) {
    if (p == NULL || name == NULL) return;
    int idx = find_entry(p, name);
    if (idx >= 0) p->entries[idx].last_used = ++p->lru_clock;
}

/* M21 orchestrazione: priorità del modello (più alto = tenuto di più; evict dal più basso). */
void llmengine_moe_provider_set_priority(LlmEngineMoeProvider* p, const char* name, int priority) {
    if (p == NULL || name == NULL) return;
    int idx = find_entry(p, name);
    if (idx >= 0) p->entries[idx].priority = priority;
}

/* Libera memoria (prima DEMOTE degli ACTIVE LRU, poi EVICT dei LRU) finché
 * mem_available + freed >= required. `exempt_idx` = l'indice del modello da NON toccare.
 * M21 orchestrazione: si evitta/ demota prima il PRIORITY più BASSO; a pari priorità il LRU. */
static int provider_more_evictable(const MoeEntry* a, const MoeEntry* b) {
    if (a->priority != b->priority) return a->priority < b->priority;
    return a->last_used < b->last_used;
}

static void provider_ensure_capacity(LlmEngineMoeProvider* p, uint64_t required, int exempt_idx) {
    if (p->mem_available == 0 || required == 0) return;   /* memoria ignota: nessuna policy */
    uint64_t freed = 0;
    while (p->mem_available + freed < required) {
        int victim = -1, demote = 0;
        for (uint32_t i = 0; i < p->count; ++i) {          /* 1) ACTIVE non-esente più evictable */
            if ((int)i == exempt_idx) continue;
            if (llmengine_moe_residency(p->entries[i].rt) != LLMENGINE_MOE_RESIDENCY_ACTIVE) continue;
            if (victim < 0 || provider_more_evictable(&p->entries[i], &p->entries[victim])) victim = (int)i;
        }
        if (victim >= 0) demote = 1;
        else for (uint32_t i = 0; i < p->count; ++i) {      /* 2) qualsiasi non-esente più evictable */
            if ((int)i == exempt_idx) continue;
            if (victim < 0 || provider_more_evictable(&p->entries[i], &p->entries[victim])) victim = (int)i;
        }
        if (victim < 0) return;                            /* niente da liberare */
        if (demote) {
            freed += entry_expert_bytes(p->entries[victim].rt);
            (void)llmengine_moe_demote(p->entries[victim].rt);
        } else {
            freed += entry_expert_bytes(p->entries[victim].rt);   /* anche l'EVICT libera RAM */
            llmengine_moe_destroy(p->entries[victim].rt);
            for (uint32_t i = (uint32_t)victim; i + 1 < p->count; ++i) p->entries[i] = p->entries[i + 1];
            p->count--;
            if (exempt_idx > victim) exempt_idx--;
        }
    }
}

LlmEngineMoeStatus llmengine_moe_provider_promote(LlmEngineMoeProvider* p, const char* name) {
    if (p == NULL || name == NULL) return LLMENGINE_MOE_ERR_INVALID;
    int idx = find_entry(p, name);
    if (idx < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    provider_ensure_capacity(p, entry_expert_bytes(p->entries[idx].rt), idx);
    idx = find_entry(p, name);   /* la compattazione può aver spostato l'indice */
    if (idx < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    LlmEngineMoeStatus s = llmengine_moe_promote(p->entries[idx].rt);
    if (s == LLMENGINE_MOE_OK) p->entries[idx].last_used = ++p->lru_clock;
    return s;
}

LlmEngineMoeStatus llmengine_moe_provider_demote(LlmEngineMoeProvider* p, const char* name) {
    if (p == NULL || name == NULL) return LLMENGINE_MOE_ERR_INVALID;
    int idx = find_entry(p, name);
    if (idx < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    return llmengine_moe_demote(p->entries[idx].rt);
}
