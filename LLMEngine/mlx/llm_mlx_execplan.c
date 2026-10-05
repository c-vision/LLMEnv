/* llm_mlx_execplan.c — §9.A M10 EXECUTION PLANNER (runtime manager).
 * Vedi llm_mlx_execplan.h. Piano per-layer (ops + skip) ingerito a runtime da JSON (dato PUL)
 * e consumato da `layer_forward`. Nessuna dipendenza PUL. */
#include "llm_mlx_execplan.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char ops[LLM_MLX_EXECPLAN_MAX_OPS][LLM_MLX_EXECPLAN_OP_LEN];
    int  nops;
    int  skip;
    int  present;
} ExecLayerPlan;

static ExecLayerPlan* g_layers = NULL;
static int g_nlayers = 0;
static int g_active = 0;

void llm_mlx_execplan_reset(void) {
    free(g_layers); g_layers = NULL; g_nlayers = 0; g_active = 0;
}

static int execplan_parse(const char* json) {
    if (json == NULL) return -1;
    cJSON* root = cJSON_Parse(json);
    if (root == NULL) return -1;
    cJSON* layers = cJSON_GetObjectItemCaseSensitive(root, "layers");
    if (!cJSON_IsArray(layers)) { cJSON_Delete(root); return -1; }
    int n = cJSON_GetArraySize(layers);
    if (n <= 0) { cJSON_Delete(root); return -1; }
    ExecLayerPlan* lp = (ExecLayerPlan*)calloc((size_t)n, sizeof(ExecLayerPlan));
    if (lp == NULL) { cJSON_Delete(root); return -1; }
    int i = 0; cJSON* it = NULL;
    cJSON_ArrayForEach(it, layers) {
        if (i >= n) break;
        ExecLayerPlan* L = &lp[i];
        cJSON* ops = cJSON_GetObjectItemCaseSensitive(it, "ops");
        if (cJSON_IsArray(ops)) {
            cJSON* o = NULL;
            cJSON_ArrayForEach(o, ops)
                if (cJSON_IsString(o) && L->nops < LLM_MLX_EXECPLAN_MAX_OPS)
                    snprintf(L->ops[L->nops++], LLM_MLX_EXECPLAN_OP_LEN, "%s", o->valuestring);
        }
        cJSON* sk = cJSON_GetObjectItemCaseSensitive(it, "skip");
        L->skip = (cJSON_IsBool(sk) ? (cJSON_IsTrue(sk) ? 1 : 0)
                                    : (cJSON_IsNumber(sk) ? (sk->valueint != 0) : 0));
        L->present = 1;
        i++;
    }
    cJSON_Delete(root);
    llm_mlx_execplan_reset();
    g_layers = lp; g_nlayers = n; g_active = 1;
    return 0;
}

int llm_mlx_execplan_load_str(const char* json) { return execplan_parse(json); }

int llm_mlx_execplan_load(const char* path) {
    if (path == NULL || !*path) return -1;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return -1; }
    char* buf = (char*)malloc((size_t)n + 1);
    if (buf == NULL) { fclose(f); return -1; }
    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return -1; }
    buf[n] = 0; fclose(f);
    int rc = execplan_parse(buf);
    free(buf);
    return rc;
}

int llm_mlx_execplan_active(void) { return g_active; }
int llm_mlx_execplan_nlayers(void) { return g_nlayers; }

int llm_mlx_execplan_layer(int layer, char ops[][LLM_MLX_EXECPLAN_OP_LEN], int max, int* skip) {
    if (skip) *skip = 0;
    if (!g_active || layer < 0 || layer >= g_nlayers) return -1;
    ExecLayerPlan* L = &g_layers[layer];
    if (!L->present) return -1;
    if (skip) *skip = L->skip;
    int n = L->nops; if (n > max) n = max;
    for (int i = 0; i < n; i++) snprintf(ops[i], LLM_MLX_EXECPLAN_OP_LEN, "%s", L->ops[i]);
    return n;
}
