/* llm_mlx_kvclasses.c — §9.A M10d: manager delle classi KV (RUNTIME vs PERSISTENT). Vedi .h. */
#include "llm_mlx_kvclasses.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char tag[64]; int cls; uint64_t bytes; } KvObj;

static KvObj*   g_objs = NULL;
static int      g_n = 0, g_cap = 0;
static int64_t  g_lim_rt = -1, g_lim_ps = -1;

void llm_mlx_kvclasses_reset(void) {
    free(g_objs); g_objs = NULL; g_n = 0; g_cap = 0; g_lim_rt = -1; g_lim_ps = -1;
}

static int kvclass_find(const char* tag) {
    for (int i = 0; i < g_n; i++) if (strcmp(g_objs[i].tag, tag) == 0) return i;
    return -1;
}

int llm_mlx_kvclasses_register(const char* tag, int cls, uint64_t bytes) {
    if (tag == NULL || !*tag) return -1;
    if (cls != LLM_KVCLASS_RUNTIME && cls != LLM_KVCLASS_PERSISTENT) return -1;
    int i = kvclass_find(tag);
    if (i < 0) {
        if (g_n == g_cap) { g_cap = g_cap ? g_cap * 2 : 16; g_objs = (KvObj*)realloc(g_objs, sizeof(KvObj) * (size_t)g_cap); if (!g_objs) return -1; }
        i = g_n++;
        snprintf(g_objs[i].tag, sizeof g_objs[i].tag, "%s", tag);
    }
    g_objs[i].cls = cls; g_objs[i].bytes = bytes;
    return 0;
}

int llm_mlx_kvclasses_update(const char* tag, uint64_t bytes) {
    int i = tag ? kvclass_find(tag) : -1;
    if (i < 0) return -1;
    g_objs[i].bytes = bytes;
    return 0;
}

int llm_mlx_kvclasses_unregister(const char* tag) {
    int i = tag ? kvclass_find(tag) : -1;
    if (i < 0) return -1;
    g_objs[i] = g_objs[g_n - 1]; g_n--;
    return 0;
}

void llm_mlx_kvclasses_stats(uint64_t* runtime_bytes, uint64_t* persistent_bytes,
                             int* n_runtime, int* n_persistent) {
    uint64_t rt = 0, ps = 0; int nr = 0, npm = 0;
    for (int i = 0; i < g_n; i++) {
        if (g_objs[i].cls == LLM_KVCLASS_RUNTIME) { rt += g_objs[i].bytes; nr++; }
        else { ps += g_objs[i].bytes; npm++; }
    }
    if (runtime_bytes) *runtime_bytes = rt;
    if (persistent_bytes) *persistent_bytes = ps;
    if (n_runtime) *n_runtime = nr;
    if (n_persistent) *n_persistent = npm;
}

void llm_mlx_kvclasses_set_limits(int64_t runtime_bytes, int64_t persistent_bytes) {
    g_lim_rt = runtime_bytes; g_lim_ps = persistent_bytes;
}

int llm_mlx_kvclasses_over_limit(int cls) {
    uint64_t rt = 0, ps = 0;
    llm_mlx_kvclasses_stats(&rt, &ps, NULL, NULL);
    if (cls == LLM_KVCLASS_RUNTIME)     return (g_lim_rt >= 0 && (int64_t)rt > g_lim_rt) ? 1 : 0;
    if (cls == LLM_KVCLASS_PERSISTENT)  return (g_lim_ps >= 0 && (int64_t)ps > g_lim_ps) ? 1 : 0;
    return 0;
}
