/* llm_level3_planner.h — Level-3 "model-aware computation elimination" + DeepSeek V4.1 layout.
 *
 * Livello 3 (v. Todo/(open) - Model-Aware-Computation-Elimination-Execution-Planner.md):
 * NON "come eseguo bene un'operazione", ma "QUESTA operazione va eseguita?".
 * Questo modulo e' il PRIMO output del planner: **REPORT-ONLY** (nessun cambio di comportamento:
 * decide e stampa; l'engine resta eager e invariato). Header-only (include-only, nessuna modifica
 * al CMake: il modulo elenca i sorgenti esplicitamente — aggiungere un .c e' infrastruttura).
 *
 * Contiene:
 *  1) `LlmDsv41Layout` + `llm_dsv41_validate()` + `llm_dsv41_ced_layout_supported()`:
 *     port 1:1 della logica VERIFICABILE di `omlx/patches/deepseek_v41/config.py`
 *     (`validate()` e `ced_layout_supported()`). NIENTE forward: quello richiede i pesi e non e'
 *     validabile senza (v. TODO). Serve a riconoscere/validare il layout e a decidere il CED.
 *  2) `llm_l3_plan()`: decisioni di eliminazione per (fase, architettura, stato). Stampa le
 *     azioni SKIP/REUSE/PERSIST con la ragione. Gated: ogni SKIP e' lossy/approx e va abilitato.
 *
 * Compilazione self-test:
 *   cc -O2 -o /tmp/llm_level3_planner_test native/llmengine_mlx/llm_level3_planner_test.c -lm
 */
#ifndef LLM_LEVEL3_PLANNER_H
#define LLM_LEVEL3_PLANNER_H

#include <stdio.h>
#include <string.h>

/* ============================ DeepSeek V4.1 layout ============================ */
/* Port dei campi rilevanti di config.py. Array a capacita' fissa (config reali ben sotto). */
typedef struct {
    const char* model_type;      /* "deepseek_v41" */
    int n_layers;
    int n_heads;
    int head_dim;
    int o_groups;
    int rope_head_dim;
    int index_head_dim;
    int n_shared_experts;
    int n_routed_experts;
    int n_activated_experts;
    int window_size;
    int compress_ratios[64];
    int compress_ratio_n;
    int kv_source_layers[16];    int kv_source_n;
    int index_source_layers[16]; int index_source_n;
    int engram_layer_ids[16];       int engram_layer_n;
    int engram_num_embeddings[16];  int engram_num_embeddings_n;
    int ced_prefill;
} LlmDsv41Layout;

/* Esiti validate() (negativi = errore, con ragione). */
#define LLM_DSV41_OK                    0
#define LLM_DSV41_E_RATIOS             -1  /* compress_ratios non coprono i layer */
#define LLM_DSV41_E_HEADS              -2  /* n_heads % o_groups o head_dim % 32 */
#define LLM_DSV41_E_ROPE               -3  /* rope_head_dim invalido */
#define LLM_DSV41_E_INDEX              -4  /* index_head_dim % 32 o shared != 1 */
#define LLM_DSV41_E_EXPERTS            -5  /* activated fuori [1, routed] */
#define LLM_DSV41_E_KV_SOURCE          -6  /* layer con ratio senza source compatibile */
#define LLM_DSV41_E_ENGRAM             -7  /* engram layer/table count diversi */
#define LLM_DSV41_E_CED_LAYOUT         -8  /* ced_prefill richiesto ma layout non supportato */

static inline int llm_dsv41_in(const int* set, int n, int x) {
    for (int i = 0; i < n; i++) if (set[i] == x) return 1;
    return 0;
}

/* Forward decl (usata da validate(), definita sotto). */
static inline int llm_dsv41_ced_layout_supported(const LlmDsv41Layout* L);

/* Ritorna LLM_DSV41_OK o un codice negativo. */
static inline int llm_dsv41_validate(const LlmDsv41Layout* L) {
    if (L == NULL) return LLM_DSV41_E_RATIOS;
    if (L->compress_ratio_n < L->n_layers) return LLM_DSV41_E_RATIOS;
    if (L->o_groups <= 0 || (L->n_heads % L->o_groups) || (L->head_dim % 32)) return LLM_DSV41_E_HEADS;
    if ((L->rope_head_dim % 2) ||
        L->rope_head_dim > (L->head_dim < L->index_head_dim ? L->head_dim : L->index_head_dim))
        return LLM_DSV41_E_ROPE;
    if ((L->index_head_dim % 32) || L->n_shared_experts != 1) return LLM_DSV41_E_INDEX;
    if (!(L->n_activated_experts >= 1 && L->n_activated_experts <= L->n_routed_experts))
        return LLM_DSV41_E_EXPERTS;
    /* Ogni layer con ratio>0 deve avere un kv/index source compatibile (stesso ratio). */
    int source = -1, index = -1;
    for (int i = 0; i < L->n_layers; i++) {
        int ratio = L->compress_ratios[i];
        if (llm_dsv41_in(L->kv_source_layers, L->kv_source_n, i)) { source = i; index = -1; }
        if (llm_dsv41_in(L->index_source_layers, L->index_source_n, i)) index = i;
        if (ratio && (source < 0 || index < 0 || L->compress_ratios[source] != ratio))
            return LLM_DSV41_E_KV_SOURCE;
    }
    /* engram: numero di layer == numero di tabelle (config.py: len(layer_ids)==len(num_embeddings)) */
    if (L->engram_layer_n != L->engram_num_embeddings_n) return LLM_DSV41_E_ENGRAM;
    if (L->ced_prefill && !llm_dsv41_ced_layout_supported(L)) return LLM_DSV41_E_CED_LAYOUT;
    return LLM_DSV41_OK;
}

/* Port 1:1 di config.py::ced_layout_supported(). */
static inline int llm_dsv41_ced_layout_supported(const LlmDsv41Layout* L) {
    if (L == NULL || L->n_layers % 2 != 0 || L->window_size <= 0) return 0;
    int mid = L->n_layers / 2;
    if (!llm_dsv41_in(L->kv_source_layers, L->kv_source_n, mid)) return 0;
    if (!llm_dsv41_in(L->index_source_layers, L->index_source_n, mid)) return 0;
    if (L->compress_ratios[mid] != 1) return 0;
    for (int i = mid + 1; i < L->n_layers; i++) if (L->compress_ratios[i] != 1) return 0;
    for (int i = mid + 1; i < L->n_layers; i++) if (llm_dsv41_in(L->kv_source_layers, L->kv_source_n, i)) return 0;
    for (int i = 0; i < L->engram_layer_n; i++) if (L->engram_layer_ids[i] >= mid) return 0;
    return 1;
}

/* ============================ Level-3 planner ============================ */
typedef enum { LLM_L3_EXECUTE = 0, LLM_L3_SKIP, LLM_L3_REUSE, LLM_L3_PERSIST } LlmL3Action;

typedef struct {
    const char* model_type;
    int is_prefill;              /* 1 prefill, 0 decode */
    long n_tokens;               /* token del prompt (prefill) o 1 (decode) */
    int is_cold_session;         /* cache_offset == 0 && history_len == 0 */
    int has_tools;               /* prompt con tool definitions */
    int pflash_mode;             /* 0 off, 1 auto, 2 always */
    int pflash_threshold;        /* default 32768 */
    int ced_enabled;             /* opt-in (approx) */
    const LlmDsv41Layout* layout;/* nullable: se il modello e' dsv41 */
    int prefix_cache_hit;        /* un prefisso KV compatibile e' disponibile */
    int moe_routing_unchanged;   /* caller: top-k identico al token precedente (candidato) */
} LlmL3PlannerInput;

typedef struct {
    LlmL3Action action;
    const char* component;       /* "prefill_tokens" | "decoder_half" | "kv_prefix" | ... */
    const char* reason;          /* "pflash" | "ced" | "prefix_cache" | "resident" | ... */
    long detail;                 /* es. token coinvolti (0 se n/a) */
    int approx;                  /* 1 = lossy/approx (richiede gate), 0 = esatto */
} LlmL3Decision;

typedef struct { LlmL3Decision items[12]; int n; } LlmL3Plan;

static inline void llm_l3_add(LlmL3Plan* p, LlmL3Action a, const char* comp,
                              const char* reason, long detail, int approx) {
    if (p->n >= 12) return;
    LlmL3Decision* d = &p->items[p->n++];
    d->action = a; d->component = comp; d->reason = reason; d->detail = detail; d->approx = approx;
}

/* Decide il piano (REPORT-ONLY). Ritorna il numero di decisioni. */
static inline int llm_l3_plan(const LlmL3PlannerInput* in, LlmL3Plan* out) {
    out->n = 0;
    /* Pesì invarianti residenti: sono in memoria per tutta la sessione. */
    llm_l3_add(out, LLM_L3_PERSIST, "model_weights", "resident", 0, 0);

    if (in->is_prefill) {
        /* PFlash: eliminazione sul lato SEQUENZA (reale, promossa default AUTO). */
        int pflash_on = (in->pflash_mode != 0) && !in->has_tools && in->is_cold_session;
        int above = (in->pflash_mode == 2) || (in->n_tokens >= (long)in->pflash_threshold);
        if (pflash_on && above) {
            /* stima keep_ratio 0.20: i token scartati ~= n - ceil(max(min_keep,n*0.2)) */
            long keep = (long)((double)in->n_tokens * 0.20);
            if (keep < 2048) keep = 2048;
            long skipped = in->n_tokens - keep;
            if (skipped > 0) llm_l3_add(out, LLM_L3_SKIP, "prefill_tokens", "pflash", skipped, 1);
        } else {
            llm_l3_add(out, LLM_L3_EXECUTE, "prefill_tokens",
                       in->has_tools ? "pflash_tools_failclosed" : "pflash_off_or_below", in->n_tokens, 0);
        }
        /* CED: eliminazione sul lato LAYER (solo modelli encoder/decoder tipo deepseek_v41). */
        int ced_ok = in->is_prefill && in->ced_enabled && in->is_cold_session && in->layout != NULL &&
                     llm_dsv41_ced_layout_supported(in->layout) &&
                     in->n_tokens > (long)in->layout->window_size;
        if (ced_ok)
            llm_l3_add(out, LLM_L3_SKIP, "decoder_half", "ced", in->n_tokens, 1);
    } else {
        /* Decode: riuso invece di ricalcolo. */
        if (in->prefix_cache_hit) llm_l3_add(out, LLM_L3_REUSE, "kv_prefix", "prefix_cache", 0, 0);
        if (in->moe_routing_unchanged) llm_l3_add(out, LLM_L3_REUSE, "moe_routing", "routing_invariant", 0, 1);
        llm_l3_add(out, LLM_L3_EXECUTE, "decode_step", "default", 1, 0);
    }
    return out->n;
}

static inline const char* llm_l3_action_name(LlmL3Action a) {
    switch (a) {
        case LLM_L3_SKIP:    return "SKIP";
        case LLM_L3_REUSE:   return "REUSE";
        case LLM_L3_PERSIST: return "PERSIST";
        default:             return "EXECUTE";
    }
}

static inline void llm_l3_print(const LlmL3Plan* p, FILE* f) {
    fprintf(f, "[l3plan] %d decisioni (report-only, nessun cambio di comportamento)\n", p->n);
    for (int i = 0; i < p->n; i++) {
        const LlmL3Decision* d = &p->items[i];
        fprintf(f, "[l3plan]   %-8s %-16s %-24s detail=%ld%s\n",
                llm_l3_action_name(d->action), d->component, d->reason, d->detail,
                d->approx ? " (approx,gated)" : "");
    }
}

#endif /* LLM_LEVEL3_PLANNER_H */
