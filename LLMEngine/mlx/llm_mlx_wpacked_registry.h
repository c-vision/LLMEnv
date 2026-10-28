/* llm_mlx_wpacked_registry.h — M3 (passo 1): risoluzione <base> -> shard + descrittore packed,
 * con cache. Non tocca il loader esistente: fornisce l'API che il loader usera' quando
 * weights_backend="packed". La risoluzione shard usa model.safetensors.index.json (cJSON). */
#ifndef LLM_MLX_WPACKED_REGISTRY_H
#define LLM_MLX_WPACKED_REGISTRY_H

#include "llm_mlx_wpacked.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cache dinamica (puntatori stabili): un modello a 48 layer ha ~350 tensori packed; il limite fisso a 8 faceva
 * ricadere sul peso MLX (gia' rilasciato) dal terzo layer in poi. */

typedef struct {
    char base[256];
    char shard[256];
    LlmMlxPackedW w;
} LlmMlxPackedEntry;

typedef struct { LlmMlxPackedEntry** e; int n, cap; } LlmMlxPackedReg;

static inline void llm_mlx_packed_reg_init(LlmMlxPackedReg* r) { if (r) { memset(r, 0, sizeof *r); } }

static inline void llm_mlx_packed_reg_free(LlmMlxPackedReg* r) {
    if (!r) return;
    for (int i = 0; i < r->n; i++) { llm_mlx_packed_close(&r->e[i]->w); free(r->e[i]); }
    free(r->e);
    memset(r, 0, sizeof *r);
}

/* Ritorna il descrittore per <base> (risolto sullo shard dell'index), o NULL. Cache per base. */
static inline const LlmMlxPackedW* llm_mlx_packed_reg_get(LlmMlxPackedReg* r, const char* model_dir,
                                                          const char* base, int bits, int group) {
    if (!r || !model_dir || !base) return NULL;
    for (int i = 0; i < r->n; i++) if (strcmp(r->e[i]->base, base) == 0) return &r->e[i]->w;
    char ip[1200]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", model_dir);
    FILE* f = fopen(ip, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1); if (!j) { fclose(f); return NULL; }
    if (fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return NULL; }
    fclose(f); j[n] = 0;
    cJSON* doc = cJSON_Parse(j); free(j);
    if (!doc) return NULL;
    cJSON* wm = cJSON_GetObjectItemCaseSensitive(doc, "weight_map");
    char key[512]; snprintf(key, sizeof key, "%s.weight", base);
    cJSON* sh = wm ? cJSON_GetObjectItemCaseSensitive(wm, key) : NULL;
    if (!cJSON_IsString(sh)) { cJSON_Delete(doc); return NULL; }
    if (r->n >= r->cap) {
        int nc = r->cap ? r->cap * 2 : 64;
        LlmMlxPackedEntry** ne = (LlmMlxPackedEntry**)realloc(r->e, (size_t)nc * sizeof *ne);
        if (!ne) { cJSON_Delete(doc); return NULL; }
        r->e = ne; r->cap = nc;
    }
    char path[1600]; snprintf(path, sizeof path, "%s/%s", model_dir, sh->valuestring);
    LlmMlxPackedEntry* e = (LlmMlxPackedEntry*)calloc(1, sizeof *e);
    if (!e) { cJSON_Delete(doc); return NULL; }
    if (llm_mlx_packed_open(path, base, bits, 0, &e->w) != 0)   /* gruppo: dal file */ { free(e); cJSON_Delete(doc); return NULL; }
    snprintf(e->base, sizeof e->base, "%s", base);
    snprintf(e->shard, sizeof e->shard, "%s", sh->valuestring);
    r->e[r->n++] = e;
    cJSON_Delete(doc);
    return &e->w;
}

#endif
