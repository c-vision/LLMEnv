/* llm_mlx_rewrite.h — MODEL REWRITE (roadmap 8.11 fase 5).
 *
 * Specifica dichiarativa di trasformazioni del modello per-layer (replace/remove/insert
 * di attention/MLP/norm), con il flag TRAINING_REQUIRED: le sostituzioni architetturali
 * NON conservano le capacita' apprese (lezione paragrafo 2.1 — ARCHITECTURAL COMPATIBILITY
 * != FUNCTIONAL COMPATIBILITY), quindi di default richiedono retraining/fine-tuning.
 *
 * Questo e' il fondamento puro (spec + registry + validazione), testabile senza MLX.
 * L'applicazione al grafo reale del motore (ARCH_TABLE / layer dispatch) e' il passo
 * successivo e richiede la ricostruzione del modello (Model Builder, fase 7).
 *
 * Convenzione: puro C, nessuno stato globale.
 */
#ifndef LLM_MLX_REWRITE_H
#define LLM_MLX_REWRITE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LLM_MAX_REWRITES 64

typedef enum {
    LLM_RW_NONE = 0,
    LLM_RW_REPLACE_ATTENTION,   /* sostituisce il tipo di attenzione (es. full -> linear) */
    LLM_RW_REPLACE_MLP,         /* sostituisce l'MLP (es. denso -> MoE) */
    LLM_RW_REPLACE_NORM,        /* sostituisce la normalizzazione */
    LLM_RW_REMOVE_LAYER,        /* rimuove un layer */
    LLM_RW_INSERT_LAYER,        /* inserisce un layer */
} LlmkxRewriteKind;

typedef struct {
    int layer;                  /* layer di destinazione (-1 = tutti / inserimento) */
    LlmkxRewriteKind kind;
    bool training_required;     /* default true per le sostituzioni architetturali */
    bool enabled;               /* la rewrite e' attiva nel plan */
} LlmkxRewrite;

typedef struct {
    LlmkxRewrite items[LLM_MAX_REWRITES];
    int count;
} LlmkxRewritePlan;

static inline void llm_rewrite_plan_init(LlmkxRewritePlan* p) {
    if (!p) return;
    memset(p, 0, sizeof(*p));
}

/* Aggiunge una rewrite. Le sostituzioni architetturali (ATTENTION/MLP/NORM/REMOVE/INSERT)
 * sono marcate training_required di default (paragrafo 2.1). Ritorna true se aggiunta. */
static inline bool llm_rewrite_add(LlmkxRewritePlan* p, int layer, LlmkxRewriteKind kind,
                                   bool training_required, bool enabled) {
    if (!p || p->count >= LLM_MAX_REWRITES) return false;
    p->items[p->count].layer = layer;
    p->items[p->count].kind = kind;
    p->items[p->count].training_required =
        (kind == LLM_RW_REPLACE_ATTENTION || kind == LLM_RW_REPLACE_MLP ||
         kind == LLM_RW_REPLACE_NORM || kind == LLM_RW_REMOVE_LAYER ||
         kind == LLM_RW_INSERT_LAYER) ? training_required : false;
    p->items[p->count].enabled = enabled;
    p->count++;
    return true;
}

/* Valida il plan: nessuna rimozione consecutiva di tutti i layer, nessun sostituzione su
 * layer fuori range (n_layers), e segnala se una sostituzione richiede training (quindi
 * l'esecuzione va bloccata senza fine-tuning). Ritorna true se il plan e' valido. */
static inline bool llm_rewrite_validate(const LlmkxRewritePlan* p, int n_layers, bool allow_training_rewrites) {
    if (!p || n_layers <= 0) return false;
    int removable = 0;
    for (int i = 0; i < p->count; i++) {
        const LlmkxRewrite* r = &p->items[i];
        if (!r->enabled) continue;
        if (r->layer < -1 || r->layer >= n_layers) return false; /* fuori range */
        if (r->kind == LLM_RW_REMOVE_LAYER) removable++;
        if (r->training_required && !allow_training_rewrites) return false; /* richiede training, bloccato */
    }
    if (removable >= n_layers) return false; /* rimuoverebbe tutti i layer */
    return true;
}

/* Numero di rewrites attive che richiedono training. */
static inline int llm_rewrite_training_count(const LlmkxRewritePlan* p) {
    int n = 0;
    if (!p) return 0;
    for (int i = 0; i < p->count; i++) if (p->items[i].enabled && p->items[i].training_required) n++;
    return n;
}

#endif /* LLM_MLX_REWRITE_H */
