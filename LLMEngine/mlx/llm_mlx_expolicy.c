/* llm_mlx_expolicy.c — §9.A EXPERT_RESIDENCY_POLICY (applicazione LIVE). Vedi llm_mlx_expolicy.h. */
#include "llm_mlx_expolicy.h"
#include "llm_engine_opts.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int  g_active = 0;
static int  g_mode = LLM_MLX_RESIDENCY_OFFLOAD;   /* default: streaming esperti */
static int  g_mode_set = 0;
static long g_cache_mb = 0;
static int  g_hotness = 0;
static int  g_prefetch_topn = 0;
static int  g_prefetch_cap = 0;
static int  g_async_workers = 0;

void llm_mlx_expolicy_reset(void) {
    g_active = 0; g_mode = LLM_MLX_RESIDENCY_OFFLOAD; g_mode_set = 0;
    g_cache_mb = 0; g_hotness = 0; g_prefetch_topn = 0; g_prefetch_cap = 0; g_async_workers = 0;
}

int llm_mlx_expolicy_active(void) { return g_active; }
const char* llm_mlx_expolicy_mode_name(void) {
    if (!g_mode_set) return "";
    return g_mode == LLM_MLX_RESIDENCY_RESIDENT ? "RESIDENT" : "OFFLOAD";
}
long llm_mlx_expolicy_cache_mb(void) { return g_cache_mb; }
int  llm_mlx_expolicy_hotness(void) { return g_hotness; }
int  llm_mlx_expolicy_prefetch_topn(void) { return g_prefetch_topn; }

static int expolicy_parse(const char* json) {
    if (json == NULL) return -1;
    cJSON* root = cJSON_Parse(json);
    if (root == NULL) return -1;
    llm_mlx_expolicy_reset();
    cJSON* mode = cJSON_GetObjectItemCaseSensitive(root, "mode");
    if (cJSON_IsString(mode) && mode->valuestring) {
        g_mode_set = 1;
        g_mode = (strcasecmp(mode->valuestring, "RESIDENT") == 0) ? LLM_MLX_RESIDENCY_RESIDENT : LLM_MLX_RESIDENCY_OFFLOAD;
    }
    cJSON* cm = cJSON_GetObjectItemCaseSensitive(root, "cache_mb");
    if (cJSON_IsNumber(cm) && cm->valuedouble > 0) g_cache_mb = (long)cm->valuedouble;
    cJSON* ht = cJSON_GetObjectItemCaseSensitive(root, "hotness");
    if (cJSON_IsBool(ht)) g_hotness = cJSON_IsTrue(ht) ? 1 : 0;
    else if (cJSON_IsNumber(ht)) g_hotness = ht->valueint != 0 ? 1 : 0;
    cJSON* pt = cJSON_GetObjectItemCaseSensitive(root, "prefetch_topn");
    if (cJSON_IsNumber(pt) && pt->valuedouble > 0) g_prefetch_topn = (int)pt->valuedouble;
    cJSON* pc = cJSON_GetObjectItemCaseSensitive(root, "prefetch_cap");
    if (cJSON_IsNumber(pc) && pc->valuedouble > 0) g_prefetch_cap = (int)pc->valuedouble;
    cJSON* aw = cJSON_GetObjectItemCaseSensitive(root, "async_workers");
    if (cJSON_IsNumber(aw) && aw->valuedouble > 0) g_async_workers = (int)aw->valuedouble;
    cJSON_Delete(root);
    g_active = 1;
    return 0;
}

int llm_mlx_expolicy_load_str(const char* json) { return expolicy_parse(json); }

int llm_mlx_expolicy_load(const char* path) {
    if (path == NULL || !*path) return -1;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return -1; }
    char* buf = (char*)malloc((size_t)n + 1);
    if (buf == NULL) { fclose(f); return -1; }
    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return -1; }
    buf[n] = 0; fclose(f);
    int rc = expolicy_parse(buf);
    free(buf);
    return rc;
}

int llm_mlx_expolicy_derive(long long n_experts, long long top_k, long long expert_bytes, long long budget_bytes) {
    if (n_experts <= 0 || expert_bytes <= 0) return -1;
    llm_mlx_expolicy_reset();
    g_mode_set = 1;
    long long total = n_experts * expert_bytes;
    if (budget_bytes <= 0 || budget_bytes >= total) {
        /* Tutto entra in RAM: RESIDENT, nessun prefetch necessario. */
        g_mode = LLM_MLX_RESIDENCY_RESIDENT;
        g_cache_mb = (long)(total / (1024LL * 1024LL));
        g_hotness = 0;
        g_prefetch_topn = 0;
    } else {
        /* Streaming: cache = budget, hotness ON (evince per route), prefetch = top_k. */
        g_mode = LLM_MLX_RESIDENCY_OFFLOAD;
        g_cache_mb = (long)(budget_bytes / (1024LL * 1024LL));
        g_hotness = 1;
        g_prefetch_topn = (int)(top_k > 0 ? top_k : 1);
    }
    g_active = 1;
    return 0;
}

int llm_mlx_expolicy_apply(void) {
    if (!g_active) return 0;
    int n = 0;
    if (g_mode_set) {
        llm_ai_mlx_opt_set("offload", g_mode == LLM_MLX_RESIDENCY_RESIDENT ? "off" : "on"); n++;
    }
    if (g_cache_mb > 0) { char b[32]; snprintf(b, sizeof b, "%ld", g_cache_mb); llm_ai_mlx_opt_set("LLMENGINE_MOE_CACHE_MB", b); n++; }
    { char b[8]; snprintf(b, sizeof b, "%d", g_hotness); llm_ai_mlx_opt_set("LLMENGINE_MOE_HOTNESS", b); n++; }
    if (g_prefetch_topn > 0) { char b[32]; snprintf(b, sizeof b, "%d", g_prefetch_topn); llm_ai_mlx_opt_set("LLMENGINE_MOE_PREFETCH_TOPN", b); n++; }
    if (g_prefetch_cap > 0) { char b[32]; snprintf(b, sizeof b, "%d", g_prefetch_cap); llm_ai_mlx_opt_set("LLMENGINE_MOE_PREFETCH_CAP", b); n++; }
    if (g_async_workers > 0) { char b[32]; snprintf(b, sizeof b, "%d", g_async_workers); llm_ai_mlx_opt_set("LLMENGINE_MOE_ASYNC_WORKERS", b); n++; }
    return n;
}
