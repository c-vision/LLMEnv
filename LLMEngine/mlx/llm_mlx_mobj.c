/* llm_mlx_mobj.c — §9.A M10: MEMORY OBJECT runtime manager unificato. Vedi llm_mlx_mobj.h. */
#include "llm_mlx_mobj.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char tag[64]; int cat; int res; unsigned long long bytes; } Mob;
static Mob* g_objs = NULL;
static int  g_n = 0, g_cap = 0;
static unsigned long long g_budget = 0;   /* 0 = illimitato */

void llm_mlx_mobj_reset(void) { free(g_objs); g_objs = NULL; g_n = 0; g_cap = 0; g_budget = 0; }
int  llm_mlx_mobj_count(void) { return g_n; }

static int mobj_find(const char* tag) {
    for (int i = 0; i < g_n; i++) if (strcmp(g_objs[i].tag, tag) == 0) return i;
    return -1;
}

int llm_mlx_mobj_register(const char* tag, int cat, int res, unsigned long long bytes) {
    if (!tag || !*tag || cat < 0 || cat >= LLM_MOBJ_NCAT) return -1;
    int i = mobj_find(tag);
    if (i < 0) {
        if (g_n == g_cap) { g_cap = g_cap ? g_cap * 2 : 16; g_objs = (Mob*)realloc(g_objs, sizeof(Mob) * (size_t)g_cap); if (!g_objs) return -1; }
        i = g_n++;
        snprintf(g_objs[i].tag, sizeof g_objs[i].tag, "%s", tag);
    }
    g_objs[i].cat = cat; g_objs[i].res = res; g_objs[i].bytes = bytes;
    return 0;
}

int llm_mlx_mobj_update(const char* tag, unsigned long long bytes) {
    int i = tag ? mobj_find(tag) : -1;
    if (i < 0) return -1;
    g_objs[i].bytes = bytes;
    return 0;
}

int llm_mlx_mobj_unregister(const char* tag) {
    int i = tag ? mobj_find(tag) : -1;
    if (i < 0) return -1;
    g_objs[i] = g_objs[g_n - 1]; g_n--;
    return 0;
}

void llm_mlx_mobj_stats(unsigned long long* total_bytes, int* n_objects) {
    unsigned long long t = 0;
    for (int i = 0; i < g_n; i++) t += g_objs[i].bytes;
    if (total_bytes) *total_bytes = t;
    if (n_objects) *n_objects = g_n;
}

void llm_mlx_mobj_stats_cat(int cat, unsigned long long* bytes, int* n) {
    unsigned long long t = 0; int c = 0;
    for (int i = 0; i < g_n; i++) if (g_objs[i].cat == cat) { t += g_objs[i].bytes; c++; }
    if (bytes) *bytes = t;
    if (n) *n = c;
}

void llm_mlx_mobj_set_budget(unsigned long long bytes) { g_budget = bytes; }
int  llm_mlx_mobj_over_budget(void) {
    if (g_budget == 0) return 0;
    unsigned long long t = 0; llm_mlx_mobj_stats(&t, NULL);
    return t > g_budget ? 1 : 0;
}

int llm_mlx_mobj_evict_priority(int cat) {
    /* K3: evince prima cio' che e' ricostruibile/effimero, mai il trunk. */
    switch (cat) {
        case LLM_MOBJ_TEMP:          return 0;
        case LLM_MOBJ_PREFIX:        return 1;
        case LLM_MOBJ_KV_PERSISTENT: return 2;
        case LLM_MOBJ_EXPERT_RAM:    return 3;
        case LLM_MOBJ_EXPERT_DEVICE: return 4;
        case LLM_MOBJ_KV_RUNTIME:    return 5;
        case LLM_MOBJ_WEIGHTS:       return 9;
        default:                     return 9;
    }
}

const char* llm_mlx_mobj_cat_name(int cat) {
    switch (cat) {
        case LLM_MOBJ_WEIGHTS:       return "weights";
        case LLM_MOBJ_EXPERT_RAM:    return "expert_ram";
        case LLM_MOBJ_EXPERT_DEVICE: return "expert_device";
        case LLM_MOBJ_KV_RUNTIME:    return "kv_runtime";
        case LLM_MOBJ_KV_PERSISTENT: return "kv_persistent";
        case LLM_MOBJ_PREFIX:        return "prefix";
        case LLM_MOBJ_TEMP:          return "temp";
        default:                     return "unknown";
    }
}
