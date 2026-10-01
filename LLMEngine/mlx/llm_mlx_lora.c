/* llm_mlx_lora.c — dominio LoRA (fase 4): helper + API esportate. Estratto VERBATIM da
 * llmengine_mlx.c (split fase 2). */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- LoRA applicazione nel forward (fase 4): helper + API esportate --- */
int llm_mlx_model_lora_get(LlmMlxModel* m, int layer, const char* proj) {
    if (!m->lora_enabled) return -1;
    for (int i = 0; i < m->lora_n; i++)
        if (m->lora_layer[i] == layer && strcmp(m->lora_proj[i], proj) == 0) return i;
    return -1;
}

/* Registra un modulo LoRA su (layer, proj). A [in,rank] row-major, B [rank,out] row-major.
 * I dati vengono copiati in array MLX. Ritorna 0 ok, -1 errore. */
int llm_ai_mlx_model_lora_add(llm_ai_mlx_model_handle h, int layer, const char* proj,
                              int rank, float scale, const float* a, const float* b,
                              int in_dim, int out_dim) {
    if (h == NULL || proj == NULL || rank <= 0 || !a || !b || in_dim <= 0 || out_dim <= 0) return -1;
    LlmMlxModel* m = (LlmMlxModel*)h;
    if (m->lora_n >= m->lora_cap) {           /* R4.1: crescita dinamica (nessun cap a 16) */
        int ncap = m->lora_cap ? m->lora_cap * 2 : 64;
        mlx_array* nA = (mlx_array*)realloc(m->lora_A, (size_t)ncap * sizeof(mlx_array));
        mlx_array* nB = (mlx_array*)realloc(m->lora_B, (size_t)ncap * sizeof(mlx_array));
        int* nL = (int*)realloc(m->lora_layer, (size_t)ncap * sizeof(int));
        char (*nP)[32] = (char (*)[32])realloc(m->lora_proj, (size_t)ncap * 32);
        float* nS = (float*)realloc(m->lora_scale, (size_t)ncap * sizeof(float));
        int* nE = (int*)realloc(m->lora_expert, (size_t)ncap * sizeof(int));
        if (!nA || !nB || !nL || !nP || !nS || !nE) return -1;
        m->lora_A = nA; m->lora_B = nB; m->lora_layer = nL; m->lora_proj = nP;
        m->lora_scale = nS; m->lora_expert = nE; m->lora_cap = ncap;
    }
    int idx = m->lora_n;
    m->lora_layer[idx] = layer;
    llm_mlx_mode_set(m->lora_proj[idx], sizeof(m->lora_proj[idx]), proj);
    m->lora_scale[idx] = scale;
    m->lora_expert[idx] = -1; /* proiezione normale */
    int as[2] = {in_dim, rank}, bs[2] = {rank, out_dim};
    m->lora_A[idx] = mlx_array_new_data(a, as, 2, MLX_FLOAT32);
    m->lora_B[idx] = mlx_array_new_data(b, bs, 2, MLX_FLOAT32);
    m->lora_n++;
    return 0;
}

/* Abilita/disabilita l'applicazione LoRA nel forward. */
/* Registra un modulo LoRA per-esperto MoE su (layer, proj, expert). */
int llm_ai_mlx_model_lora_add_expert(llm_ai_mlx_model_handle h, int layer, int expert,
                                     const char* proj, int rank, float scale,
                                     const float* a, const float* b, int in_dim, int out_dim) {
    if (h == NULL || proj == NULL || rank <= 0 || !a || !b || in_dim <= 0 || out_dim <= 0 || expert < 0) return -1;
    if (llm_ai_mlx_model_lora_add(h, layer, proj, rank, scale, a, b, in_dim, out_dim) != 0) return -1;
    LlmMlxModel* m = (LlmMlxModel*)h;
    m->lora_expert[m->lora_n - 1] = expert;
    return 0;
}

int llm_ai_mlx_model_lora_set_enabled(llm_ai_mlx_model_handle h, bool enabled) {
    if (h == NULL) return -1;
    ((LlmMlxModel*)h)->lora_enabled = enabled;
    return 0;
}

/* Pulisce i moduli LoRA. */
void llm_ai_mlx_model_lora_clear(llm_ai_mlx_model_handle h) {
    if (h == NULL) return;
    LlmMlxModel* m = (LlmMlxModel*)h;
    for (int i = 0; i < m->lora_n; i++) { mlx_array_free(m->lora_A[i]); mlx_array_free(m->lora_B[i]); }
    m->lora_n = 0;
    m->lora_enabled = false;
}

/* §D R4.6 — DEPLOY: salva/carica TUTTI i moduli LoRA registrati (adapter multi-modulo).
 * Formato: u32 magic 'GLRO', u32 ver, i32 n; per modulo i32 layer, i32 expert, i32 in, i32 rank,
 * i32 out, f32 scale, char proj[32], f32 A[in*rank], f32 B[rank*out]. Scrittura atomica. */
int llm_ai_mlx_model_lora_save(llm_ai_mlx_model_handle h, const char* path) {
    if (!h || !path) return -1;
    LlmMlxModel* m = (LlmMlxModel*)h;
    char tmp[1200]; snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb"); if (!f) return -1;
    uint32_t magic = 0x4F524C47u, ver = 1u; int32_t n = m->lora_n;
    int ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&ver, 4, 1, f) == 1 && fwrite(&n, 4, 1, f) == 1;
    for (int i = 0; ok && i < m->lora_n; i++) {
        int in = (int)mlx_array_shape(m->lora_A[i])[0];
        int rank = (int)mlx_array_shape(m->lora_A[i])[1];
        int out = (int)mlx_array_shape(m->lora_B[i])[1];
        int32_t layer = m->lora_layer[i], expert = m->lora_expert[i];
        int32_t in_, rank_, out_; in_ = in; rank_ = rank; out_ = out;
        float scale = m->lora_scale[i];
        ok = fwrite(&layer, 4, 1, f) == 1 && fwrite(&expert, 4, 1, f) == 1 && fwrite(&in_, 4, 1, f) == 1 &&
             fwrite(&rank_, 4, 1, f) == 1 && fwrite(&out_, 4, 1, f) == 1 && fwrite(&scale, 4, 1, f) == 1 &&
             fwrite(m->lora_proj[i], 1, 32, f) == 32;
        if (ok) ok = mlx_array_eval(m->lora_A[i]) == 0 && mlx_array_eval(m->lora_B[i]) == 0;
        if (ok) ok = fwrite(mlx_array_data_float32(m->lora_A[i]), 4, (size_t)in * rank, f) == (size_t)in * rank &&
                      fwrite(mlx_array_data_float32(m->lora_B[i]), 4, (size_t)rank * out, f) == (size_t)rank * out;
    }
    fclose(f);
    if (!ok) { remove(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}

/* Carica un adapter multi-modulo: azzera i moduli correnti e registra quelli del file. */
int llm_ai_mlx_model_lora_load(llm_ai_mlx_model_handle h, const char* path) {
    if (!h || !path) return -1;
    FILE* f = fopen(path, "rb"); if (!f) return -1;
    uint32_t magic = 0, ver = 0; int32_t n = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1 ||
        magic != 0x4F524C47u || ver != 1u || n < 0) { fclose(f); return -1; }
    llm_ai_mlx_model_lora_clear(h);
    int rc = 0;
    for (int i = 0; i < n && rc == 0; i++) {
        int32_t layer, expert, in, rank, out; float scale; char proj[32];
        if (fread(&layer, 4, 1, f) != 1 || fread(&expert, 4, 1, f) != 1 || fread(&in, 4, 1, f) != 1 ||
            fread(&rank, 4, 1, f) != 1 || fread(&out, 4, 1, f) != 1 || fread(&scale, 4, 1, f) != 1 ||
            fread(proj, 1, 32, f) != 32) { rc = -1; break; }
        proj[31] = 0;
        float* A = (float*)malloc((size_t)in * rank * 4);
        float* B = (float*)malloc((size_t)rank * out * 4);
        if (!A || !B || fread(A, 4, (size_t)in * rank, f) != (size_t)in * rank || fread(B, 4, (size_t)rank * out, f) != (size_t)rank * out) {
            free(A); free(B); rc = -1; break;
        }
        if (expert >= 0) rc = llm_ai_mlx_model_lora_add_expert(h, layer, expert, proj, rank, scale, A, B, in, out);
        else rc = llm_ai_mlx_model_lora_add(h, layer, proj, rank, scale, A, B, in, out);
        free(A); free(B);
    }
    fclose(f);
    if (rc != 0) { llm_ai_mlx_model_lora_clear(h); return -1; }
    return 0;
}

/* Model Rewrite smoketest (fase 5): spec replace/remove + validazione TRAINING_REQUIRED. */
