/* llm_mlx_kvdag.c — §9.A M10b: DAG del KV + espansione PREFETCH. Vedi llm_mlx_kvdag.h. */
#include "llm_mlx_kvdag.h"

#include <stdio.h>
#include <string.h>

typedef struct { char id[64]; int kind; int reconstructable; int ndeps; int deps[16]; } KvNode;
static KvNode g_nodes[LLM_KVDAG_MAX];
static int    g_nn = 0;

static int kvnode_find(const char* id) {
    if (!id) return -1;
    for (int i = 0; i < g_nn; i++) if (strcmp(g_nodes[i].id, id) == 0) return i;
    return -1;
}
void llm_mlx_kvdag_reset(void) { g_nn = 0; }
int  llm_mlx_kvdag_count(void) { return g_nn; }

int llm_mlx_kvdag_add(const char* id, int kind, int reconstructable) {
    if (!id || !*id || g_nn >= LLM_KVDAG_MAX) return -1;
    if (kvnode_find(id) >= 0) return -1;
    KvNode* n = &g_nodes[g_nn++];
    memset(n, 0, sizeof *n);
    snprintf(n->id, sizeof n->id, "%s", id);
    n->kind = kind; n->reconstructable = reconstructable;
    return 0;
}

int llm_mlx_kvdag_dep(const char* id, const char* dep) {
    int i = kvnode_find(id), j = kvnode_find(dep);
    if (i < 0 || j < 0 || g_nodes[i].ndeps >= 16) return -1;
    g_nodes[i].deps[g_nodes[i].ndeps++] = j;
    return 0;
}

static const char* kvnode_op(int kind) {
    switch (kind) {
        case LLM_KVNODE_DERIVED: return "REUSE";
        case LLM_KVNODE_INDEX:   return "RECONSTRUCT";
        default:                 return "READ";
    }
}

/* DFS post-order (dipendenze prima), dedup. */
static void kvdag_visit(int i, int* visited, int* order, int* no, int max) {
    if (visited[i]) return;
    visited[i] = 1;
    for (int d = 0; d < g_nodes[i].ndeps; d++) kvdag_visit(g_nodes[i].deps[d], visited, order, no, max);
    if (*no < max) order[(*no)++] = i;
}

int llm_mlx_kvdag_prefetch(const char* id, char* out, size_t cap) {
    if (out && cap) out[0] = 0;
    int i = kvnode_find(id);
    if (i < 0 || out == NULL || cap == 0) return -1;
    int visited[LLM_KVDAG_MAX]; int order[LLM_KVDAG_MAX]; int no = 0;
    memset(visited, 0, sizeof visited);
    kvdag_visit(i, visited, order, &no, LLM_KVDAG_MAX);
    size_t o = 0;
    for (int k = 0; k < no; k++) {
        const KvNode* n = &g_nodes[order[k]];
        o += (size_t)snprintf(out + o, cap - o, "%s%s %s", k ? "," : "", kvnode_op(n->kind), n->id);
    }
    return no;
}

int llm_mlx_kvdag_demo_csa2(char* out, size_t cap) {
    llm_mlx_kvdag_reset();
    /* Grafo CSA2 rappresentativo (7 nodi): un BASE condiviso, due KV DERIVED (riuso), i loro
     * INDICI (reindex) e un aggregato. */
    if (llm_mlx_kvdag_add("v41.base",       LLM_KVNODE_BASE,    0) != 0) return -1;
    if (llm_mlx_kvdag_add("v41.reuse.l0",   LLM_KVNODE_DERIVED, 0) != 0) return -1;
    if (llm_mlx_kvdag_add("v41.reuse.l1",   LLM_KVNODE_DERIVED, 0) != 0) return -1;
    if (llm_mlx_kvdag_add("v41.shared",     LLM_KVNODE_DERIVED, 0) != 0) return -1;
    if (llm_mlx_kvdag_add("v41.index.l0",   LLM_KVNODE_INDEX,   1) != 0) return -1;
    if (llm_mlx_kvdag_add("v41.index.l1",   LLM_KVNODE_INDEX,   1) != 0) return -1;
    if (llm_mlx_kvdag_add("v41.out",        LLM_KVNODE_INDEX,   1) != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.reuse.l0", "v41.base") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.reuse.l1", "v41.base") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.shared",   "v41.base") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.index.l0", "v41.reuse.l0") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.index.l1", "v41.reuse.l1") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.out",      "v41.index.l0") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.out",      "v41.index.l1") != 0) return -1;
    if (llm_mlx_kvdag_dep("v41.out",      "v41.shared")   != 0) return -1;
    return llm_mlx_kvdag_prefetch("v41.out", out, cap);
}
