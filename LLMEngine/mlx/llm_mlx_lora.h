/* llm_mlx_lora.h — ADAPTER / LoRA (roadmap 8.11 fase 4).
 *
 * Modulo low-level per adapter LoRA/QLoRA: struttura di un adapter (lista di moduli
 * LoRA, ognuno per una proiezione del modello), registry per NOME con HOT-SWAP (cambia
 * adapter a runtime senza ricaricare il modello), e la matematica LoRA
 *   contrib = (x @ A @ B) * scale   (A in [in,rank], B in [rank,out])
 * da sommare all'output della proiezione base.
 *
 * Questo e' il fondamento puro (struttura dati + matematica, testabile senza MLX);
 * l'applicazione nel forward MLX (aggiungere il contributo nel matmul) e' il passo
 * successivo nel motore. Hot-swap: un modello con piu' adapter (coding, sql, vision)
 * puo' attivare/disattivare senza ricostruire (PEFT-style).
 *
 * Convenzione: puro C, nessuno stato globale.
 */
#ifndef LLM_MLX_LORA_H
#define LLM_MLX_LORA_H

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define LLM_MAX_LORA_MODULES 64
#define LLM_MAX_ADAPTERS 16
#define LLM_LORA_NAME 64

/* Un modulo LoRA per una proiezione: y = base(x) + (x @ A @ B) * scale.
 * A/B sono puntatori opachi (in MLX: mlx_array; qui float* per il test puro). */
typedef struct {
    int layer;
    char proj[32];       /* es. "q_proj", "v_proj", "gate_proj" */
    int in_dim, out_dim, rank;
    float scale;
    float* A;            /* [in_dim, rank] (row-major) */
    float* B;            /* [rank, out_dim] */
} LlmLoraModule;

typedef struct {
    char name[LLM_LORA_NAME];
    LlmLoraModule modules[LLM_MAX_LORA_MODULES];
    int n_modules;
    bool enabled;
} LlmLoraAdapter;

typedef struct {
    LlmLoraAdapter adapters[LLM_MAX_ADAPTERS];
    int n_adapters;
    int active;          /* indice dell'adapter attivo, -1 = nessuno */
} LlmLoraRegistry;

static inline void llm_lora_registry_init(LlmLoraRegistry* r) {
    if (!r) return;
    memset(r, 0, sizeof(*r));
    r->active = -1;
}

/* Registra (o sostituisce) un adapter per nome. Ritorna true ok. */
static inline bool llm_lora_register(LlmLoraRegistry* r, const char* name, LlmLoraAdapter* a) {
    if (!r || !name) return false;
    for (int i = 0; i < r->n_adapters; i++)
        if (strcmp(r->adapters[i].name, name) == 0) {
            r->adapters[i] = *a;
            r->adapters[i].enabled = false;
            return true;
        }
    if (r->n_adapters >= LLM_MAX_ADAPTERS) return false;
    r->adapters[r->n_adapters] = *a;
    snprintf(r->adapters[r->n_adapters].name, LLM_LORA_NAME, "%s", name);
    r->adapters[r->n_adapters].enabled = false;
    r->n_adapters++;
    return true;
}

/* Attiva un adapter (hot-swap: disattiva il precedente). Ritorna true se trovato. */
static inline bool llm_lora_enable(LlmLoraRegistry* r, const char* name) {
    if (!r || !name) return false;
    if (r->active >= 0 && r->active < r->n_adapters)
        r->adapters[r->active].enabled = false;
    for (int i = 0; i < r->n_adapters; i++)
        if (strcmp(r->adapters[i].name, name) == 0) {
            r->adapters[i].enabled = true;
            r->active = i;
            return true;
        }
    return false;
}

/* Disattiva l'adapter attivo. */
static inline void llm_lora_disable(LlmLoraRegistry* r) {
    if (!r || r->active < 0 || r->active >= r->n_adapters) return;
    r->adapters[r->active].enabled = false;
    r->active = -1;
}

/* --- Matematica LoRA pura (CPU float): usata dal path di MERGE (llm_lora_merge_contrib) e dai
 * test. Il FORWARD del motore usa llm_mlx_lora_apply_mlx (matmul MLX nativi, llm_mlx_ops.h):
 * una sola implementazione per dominio (T1 verificato: non sono duplicazioni).
 * contrib = (x @ A @ B) * scale per una riga di x: out[i] += sum_j sum_k x[i][j]*A[j][k]*B[k][l]*scale. */
static inline void llm_lora_apply_row(const LlmLoraModule* m, const float* x, float* out) {
    int in = m->in_dim, outd = m->out_dim, rank = m->rank;
    float* tmp = (float*)calloc((size_t)outd, sizeof(float));
    /* x @ A -> [rank] */
    float* xa = (float*)calloc((size_t)rank, sizeof(float));
    for (int k = 0; k < rank; k++)
        for (int j = 0; j < in; j++)
            xa[k] += x[j] * m->A[j * rank + k];
    /* xa @ B -> [out] */
    for (int l = 0; l < outd; l++)
        for (int k = 0; k < rank; k++)
            tmp[l] += xa[k] * m->B[k * outd + l];
    for (int l = 0; l < outd; l++) out[l] += tmp[l] * m->scale;
    free(tmp); free(xa);
}

/* --- MODEL MERGE ADAPTERS (fase 4): somma ponderata dei contributi di piu' adapter ---
 * Per K adapter sullo STESSO modulo (layer,proj), l'output e'
 *   y = base(x) + sum_k w_k * (x @ A_k @ B_k) * scale_k
 * (merge a runtime, pattern DARE/TIES: pesi w_k sommano idealmente a 1). Ritorna il
 * numero di contributi applicati. */
static inline int llm_lora_merge_contrib(const LlmLoraModule* mods, const float* w, int n_mods,
                                         const float* x, float* out) {
    if (n_mods <= 0) return 0;
    int applied = 0;
    for (int k = 0; k < n_mods; k++) {
        float* tmp = (float*)calloc((size_t)mods[k].out_dim, sizeof(float));
        LlmLoraModule m2 = mods[k];
        m2.scale = mods[k].scale * (w ? w[k] : 1.0f); /* scala ponderata */
        llm_lora_apply_row(&m2, x, tmp);
        for (int l = 0; l < mods[k].out_dim; l++) out[l] += tmp[l];
        free(tmp);
        applied++;
    }
    return applied;
}

#endif /* LLM_MLX_LORA_H */
