/* llm_dsv41_ckpt.h — lettura di un checkpoint DeepSeek V4.1 (safetensors, singolo file o
 * sharded con `model.safetensors.index.json`). Header-only.
 *
 * Usa `mlx_load_safetensors` (MLX mappa F8_E4M3/F8_E8M0 a uint8: i pesi fp8 e le scale ue8m0
 * restano byte grezzi, li decodifica `llm_dsv41_loader.h`). L'index e' parsato con cJSON.
 * Nessun P/Invoke, C puro. Server al loader dei pesi, non al runtime.
 */
#ifndef LLM_DSV41_CKPT_H
#define LLM_DSV41_CKPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include <mlx/c/mlx.h>
#include <cJSON.h>

#define LLM_DSV41_CKPT_MAX_SHARDS 64

typedef struct {
    char path[1200];
    mlx_map_string_to_array map;
    bool loaded;
} LlmDsv41Shard;

typedef struct {
    bool sharded;
    char root[1024];          /* directory (sharded) o path del file singolo */
    cJSON* index_root;        /* root del model.safetensors.index.json (o NULL) */
    cJSON* weight_map;        /* index_root["weight_map"] (o NULL) */
    LlmDsv41Shard shards[LLM_DSV41_CKPT_MAX_SHARDS];
    int n_shards;
} LlmDsv41Ckpt;

/* risolve il nome -> indice shard (aggiungendolo se serve). -1 se assente. */
static inline int llm_dsv41_ckpt_shard_for(LlmDsv41Ckpt* ck, const char* name) {
    if (!ck->sharded) return ck->n_shards ? 0 : -1;
    if (!ck->weight_map) return -1;
    cJSON* it = cJSON_GetObjectItemCaseSensitive(ck->weight_map, name);
    if (!cJSON_IsString(it) || !it->valuestring) return -1;
    char full[1200];
    snprintf(full, sizeof full, "%s/%s", ck->root, it->valuestring);
    for (int i = 0; i < ck->n_shards; i++)
        if (strcmp(ck->shards[i].path, full) == 0) return i;
    if (ck->n_shards >= LLM_DSV41_CKPT_MAX_SHARDS) return -1;
    LlmDsv41Shard* sh = &ck->shards[ck->n_shards];
    snprintf(sh->path, sizeof sh->path, "%s", full);
    sh->map = mlx_map_string_to_array_new();
    sh->loaded = false;
    return ck->n_shards++;
}

/* apre il checkpoint (directory con index, o singolo .safetensors). */
static inline bool llm_dsv41_ckpt_open(LlmDsv41Ckpt* ck, const char* path) {
    memset(ck, 0, sizeof *ck);
    struct stat st;
    if (stat(path, &st) != 0) return false;
    if (S_ISDIR(st.st_mode)) {
        ck->sharded = true;
        snprintf(ck->root, sizeof ck->root, "%s", path);
        char ipath[1200];
        snprintf(ipath, sizeof ipath, "%s/model.safetensors.index.json", path);
        FILE* f = fopen(ipath, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        if (n <= 0) { fclose(f); return false; }
        char* buf = (char*)malloc((size_t)n + 1);
        if (!buf) { fclose(f); return false; }
        size_t rd = fread(buf, 1, (size_t)n, f);
        fclose(f);
        buf[rd] = '\0';
        ck->index_root = cJSON_Parse(buf);
        free(buf);
        if (!ck->index_root) return false;
        ck->weight_map = cJSON_GetObjectItemCaseSensitive(ck->index_root, "weight_map");
        if (!cJSON_IsObject(ck->weight_map)) return false;
        return true;
    }
    ck->sharded = false;
    snprintf(ck->root, sizeof ck->root, "%s", path);
    LlmDsv41Shard* sh = &ck->shards[0];
    snprintf(sh->path, sizeof sh->path, "%s", path);
    sh->map = mlx_map_string_to_array_new();
    sh->loaded = false;
    ck->n_shards = 1;
    return true;
}

/* La primitiva `Load` di MLX e' CPU-only: caricare con stream GPU fa fallire l'eval
 * (`Load::eval_gpu Not implemented`). Si carica su stream CPU; gli op successivi con stream
 * GPU fanno il trasferimento. */
static inline mlx_stream llm_dsv41_cpu_stream(void) {
    static mlx_stream cpu = { 0 };
    static int inited = 0;
    if (!inited) { cpu = mlx_default_cpu_stream_new(); inited = 1; }
    return cpu;
}

static inline bool llm_dsv41_ckpt_load_shard(LlmDsv41Ckpt* ck, int idx, mlx_stream s) {
    LlmDsv41Shard* sh = &ck->shards[idx];
    (void)s;
    if (sh->loaded) return true;
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    int rc = mlx_load_safetensors(&sh->map, &meta, sh->path, llm_dsv41_cpu_stream());
    mlx_map_string_to_string_free(meta);
    if (rc != 0) { fprintf(stderr, "llm_dsv41_ckpt: load %s fallita\n", sh->path); return false; }
    sh->loaded = true;
    return true;
}

/* recupera un tensore per nome (handle valido finche' il checkpoint e' aperto). */
static inline bool llm_dsv41_ckpt_get(LlmDsv41Ckpt* ck, const char* name, mlx_array* out,
                                      mlx_stream s) {
    int idx = llm_dsv41_ckpt_shard_for(ck, name);
    if (idx < 0) return false;
    if (!llm_dsv41_ckpt_load_shard(ck, idx, s)) return false;
    if (mlx_map_string_to_array_get(out, ck->shards[idx].map, name) != 0) return false;
    return true;
}

static inline bool llm_dsv41_ckpt_has(LlmDsv41Ckpt* ck, const char* name) {
    int idx = llm_dsv41_ckpt_shard_for(ck, name);
    if (idx < 0) return false;
    if (!llm_dsv41_ckpt_load_shard(ck, idx, llm_dsv41_cpu_stream())) return false;
    mlx_array v = mlx_array_new();
    return mlx_map_string_to_array_get(&v, ck->shards[idx].map, name) == 0;
}

static inline void llm_dsv41_ckpt_close(LlmDsv41Ckpt* ck) {
    for (int i = 0; i < ck->n_shards; i++)
        if (ck->shards[i].loaded) mlx_map_string_to_array_free(ck->shards[i].map);
    if (ck->index_root) cJSON_Delete(ck->index_root);
    memset(ck, 0, sizeof *ck);
}

#endif /* LLM_DSV41_CKPT_H */
