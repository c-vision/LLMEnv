/* ============================================================================
 * llm_mlx_stats.c — dominio introspezione modello: stats/parametri attivi, KV estimate e
 * recommend, imatrix collector pubblico, memory stats/limit e guardia saturazione, FLOPS/memory.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llm_mlx_array_count.h"   /* L25: contatori GLOBALI (prima degli header di progetto) */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#if LLM_COUNT_ARRAYS
long g_arr_new_ct = 0, g_arr_free_ct = 0, g_arr_ne_free = 0;
void llm_mlx_array_balance(const char* tag) {
    fprintf(stderr, "[ARRAYS] %s new=%ld free=%ld net=%ld (free_nonempty=%ld)\n", tag,
            g_arr_new_ct, g_arr_free_ct, g_arr_new_ct - g_arr_free_ct, g_arr_ne_free);
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

/* ─────────────────────────────────────────────────────────────────────────────

 * MODEL STATS / MEMORY / FLOPS (roadmap fase 2 — MODEL MEMORY / MODEL FLOPS /

 * MODEL WEIGHTS). Stime basate sul config (ordine di grandezza, per il benchmark

 * farm e la capability matrix), NON misure: parametri, KV-cache a seq_len,

 * FLOPS prefill approssimati.

 *

 * `llm_ai_mlx_model_stats(handle, seq_len, out_params, out_kv_bytes)`:

 *   out_params  = n. parametri del modello (pesi)

 *   out_kv_bytes= byte della KV-cache a seq_len (FP16)

 * `llm_ai_mlx_model_flops(handle, seq_len)`: FLOPS prefill approssimati (attention

 * O(seq^2) + MLP O(seq)). Ritorna -1 su handle nullo.

 */

int llm_ai_mlx_model_stats(llm_ai_mlx_model_handle handle, int seq_len,

                           double* out_params, double* out_kv_bytes) {

    if (handle == NULL) return -1;

    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;

    long long h = c->hidden_size, nh = c->n_heads, nkv = c->n_kv_heads, hd = c->head_dim;

    if (h <= 0 || nh <= 0) return -1;

    /* attenzione per layer (GQA + output gate qwen3_next: q_proj = 2*nh*hd) */

    double attn_per = (double)(2 * nh * hd) * h + (double)(2 * nkv * hd) * h + (double)(nh * hd) * h; /* q(+gate), k, v, o */

    double mlp_per;

    if (c->n_experts > 0) {

        mlp_per = (double)h * c->n_experts                       /* router */

                + (double)c->n_experts * 3 * h * c->moe_intermediate_size /* switch_mlp gate/up/down */

                + 3 * h * c->moe_intermediate_size;              /* shared_expert (approx) */

    } else {

        /* FIX 2026-09-17 (misurato): usare l'intermediate REALE. L'approx `3h^2` scambiava
         * l'intermediate con h -> su Qwen3-8B (h=4096, intermediate=12288) stimava 50M
         * invece di 151M parametri di MLP per layer: -3,6B parametri totali = -39% sui byte
         * dichiarati (2439,8 MB contro ~4000 MB reali), che rendeva falso ogni GB/s. */
        mlp_per = 3.0 * h * (c->intermediate_size > 0 ? (double)c->intermediate_size : 3.0 * h);

    }

    /* FIX 2026-09-17: l'embedding d'INGRESSO e' vocab*h (il lm_head e' contato a parte dal
     * chiamante come `head`); prima `2.0*h` (~8k parametri) lo ignorava quasi del tutto. */
    double params = (double)c->n_layers * (attn_per + mlp_per)
                  + (double)c->vocab_size * h + 2.0 * h;

    if (out_params) *out_params = params;

    if (out_kv_bytes) {

        double kv = 2.0 * c->n_layers * nkv * hd * (seq_len > 0 ? seq_len : 1) * 2.0; /* k+v, FP16=2B */

        *out_kv_bytes = kv;

    }

    return 0;
}

/* Byte/parametri ATTIVI per token (MoE: solo i top_k esperti vengono letti). Serve al roofline
 * di banda: sui MoE il totale (tutti gli esperti) NON e' il traffico per token: gpt-oss-20b
 * risultava 12,6 GB/token e la 35B-A3B 18,2 GB/token, cioe' numeri che rendono fittizi i GB/s. */
int llm_ai_mlx_model_active_params(llm_ai_mlx_model_handle handle, double* out_active) {
    if (handle == NULL || out_active == NULL) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    long long h = c->hidden_size, nh = c->n_heads, nkv = c->n_kv_heads, hd = c->head_dim;
    if (h <= 0 || nh <= 0) return -1;
    double attn_per = (double)(2 * nh * hd) * h + (double)(2 * nkv * hd) * h + (double)(nh * hd) * h;
    double mlp_per;
    if (c->n_experts > 0) {
        int act = (c->top_k > 0 && c->top_k < c->n_experts) ? c->top_k : c->n_experts;
        mlp_per = (double)h * c->n_experts                        /* router (sempre letto) */
                + (double)act * 3 * h * c->moe_intermediate_size  /* SOLO i top_k esperti */
                + 3 * h * c->moe_intermediate_size;               /* shared_expert (sempre) */
    } else {
        /* FIX 2026-09-17: intermediate reale (vedi llm_ai_mlx_model_stats). */
        mlp_per = 3.0 * h * (c->intermediate_size > 0 ? (double)c->intermediate_size : 3.0 * h);
    }
    *out_active = (double)c->n_layers * (attn_per + mlp_per)
                + (double)c->vocab_size * h + 2.0 * h;   /* FIX: + embedding d'ingresso */
    return 0;
}

/* KV footprint estimate PER-ARCHITETTURA (port da rapidMLX `kv_estimation.py`):
 * classifica ogni layer (full-attention a crescita lineare / sliding-window a
 * finestra rotante sublineare / GatedDeltaNet ricorrente a stato fisso / borrower
 * KV-sharing a zero) e stima i byte della KV-cache per T = seq_len token:
 *
 *   bytes(T) = fixed_baseline + per_token_growth * T + sliding_slot * slots(T)
 *
 * dove slots(T) = rotating_cache_slots(window, T) e' un upper bound del buffer
 * RotatingKVCache di mlx_lm (step 256, keep 4). Over-estimate-safe: un layer
 * non classificabile e' addebitato come full-growth; mai sotto-stima.
 * Sostituisce la formula uniforme di llm_ai_mlx_model_stats (che contava TUTTI
 * i layer come full-growth, sovrastimando fino a ~2x per gli ibridi). */
int llm_ai_mlx_model_kv_estimate(llm_ai_mlx_model_handle handle, int seq_len, double* out_bytes) {
    if (handle == NULL || out_bytes == NULL) return -1;
    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;
    if (c->n_layers <= 0 || c->n_kv_heads <= 0 || c->head_dim <= 0) return -1;
    int T = seq_len > 0 ? seq_len : 1;
    const int dtype_bytes = 2; /* FP16 */
    const int kv_heads = c->n_kv_heads, hd = c->head_dim;
    const bool is_gdn = (strcmp(c->model_type, "qwen3_5") == 0 || strcmp(c->model_type, "qwen3_5_moe") == 0);
    const bool has_sliding = (c->is_sliding_layer != NULL && c->sliding_window > 0);

    /* rotating_cache_slots(window, T): upper bound del buffer rotante
     * min(ceil((T+keep)/step)*step, ceil((window+keep)/step)*step), step=256 keep=4 */
    int slots = 0;
    if (has_sliding) {
        const int STEP = 256, KEEP = 4, window = c->sliding_window;
        int effective = (T < window ? T : window) + KEEP;
        slots = ((effective + STEP - 1) / STEP) * STEP;
    }

    /* n_full / n_sliding / n_gdn per tipo di layer */
    int n_full = 0, n_sliding = 0;
    if (c->is_sliding_layer != NULL) {
        for (int i = 0; i < c->n_layers; i++) {
            if (c->is_sliding_layer[i]) n_sliding++; else n_full++;
        }
    } else if (is_gdn && c->full_attention_interval > 0) {
        n_full = c->n_layers / c->full_attention_interval;
    } else {
        n_full = c->n_layers;
    }
    int n_gdn = is_gdn ? (c->n_layers - n_full) : 0;

    /* fixed baseline: stato ricorrente GDN per-sequence (4 bytes/elem, come mlx_lm
     * mx.zeros f32) -- num_v*head_k*head_v + conv (kernel-1)*(2*key_dim+value_dim) */
    double fixed_baseline = 0;
    if (n_gdn > 0 && c->linear_num_value_heads > 0 && c->linear_num_key_heads > 0 &&
        c->linear_key_head_dim > 0 && c->linear_value_head_dim > 0 && c->linear_conv_kernel_dim > 0) {
        int key_dim = c->linear_num_key_heads * c->linear_key_head_dim;
        int value_dim = c->linear_num_value_heads * c->linear_value_head_dim;
        double per_layer = 4.0 * ((double)c->linear_num_value_heads * c->linear_key_head_dim * c->linear_value_head_dim
                                  + (double)(c->linear_conv_kernel_dim - 1) * (2.0 * key_dim + value_dim));
        fixed_baseline = per_layer * n_gdn;
    } else if (n_gdn > 0) {
        /* stato non dimensionabile -> addebito full-growth (mai sotto-stima) */
        n_full += n_gdn; n_gdn = 0;
    }

    /* dims BASE per tutti i layer (over-estimate-safe, identico alla reference
     * rapidMLX: per gemma4/gpt-oss i layer full usano le dims locali nel calcolo;
     * mai sotto-stima) */
    double per_token_growth = (double)n_full * 2.0 * dtype_bytes * kv_heads * hd;
    double sliding_bytes = (double)n_sliding * 2.0 * dtype_bytes * kv_heads * hd * slots;

    *out_bytes = fixed_baseline + per_token_growth * T + sliding_bytes;
    return 0;
}

/* KV quant GATE (item 6): dato un budget di memoria per la KV-cache, decide se la
 * quantizzazione e' necessaria e con quanti bits. Policy (over-estimate-safe):
 *   est(seq_len) <= budget          -> OFF (nessuna quantizzazione)
 *   est <= 2*budget                 -> Q8  (la parte cold a 8 bit)
 *   est >  2*budget                 -> Q4  (piu' aggressivo)
 * Ritorna 0 su successo e scrive enabled/bits + la stima effettiva. Base:
 * llm_ai_mlx_model_kv_estimate. */
int llm_ai_mlx_model_kv_recommend(llm_ai_mlx_model_handle handle, int seq_len,
                                  double budget_bytes, int* out_enabled, int* out_bits,
                                  double* out_estimate) {
    if (handle == NULL || out_enabled == NULL || out_bits == NULL) return -1;
    double est = 0;
    if (llm_ai_mlx_model_kv_estimate(handle, seq_len, &est) != 0) return -1;
    if (out_estimate) *out_estimate = est;
    if (budget_bytes <= 0 || est <= budget_bytes) {
        *out_enabled = 0; *out_bits = 0;
        return 0;
    }
    *out_enabled = 1;
    *out_bits = (est > 2.0 * budget_bytes) ? 4 : 8;
    return 0;
}

/* Imatrix collector pubblici (DS4-style): begin alloca gli accumulatori per (layer, expert,
 * colonna); la raccolta avviene in automatico nei blocchi MoE durante session_append con
 * im.active; write salva il report. End non e' necessario (libera il model destroy). */
int llm_ai_mlx_model_imatrix_begin(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return -1;
    return llm_mlx_imatrix_begin((LlmMlxModel*)handle);
}
int llm_ai_mlx_model_imatrix_write(llm_ai_mlx_model_handle handle, const char* path) {
    if (handle == NULL || path == NULL) return -1;
    return llm_mlx_imatrix_write((LlmMlxModel*)handle, path);
}



/* Diagnostica/guardia memoria MLX. La ritenzione "active" (vedi finding 2026-09-02: ~picco grafo
 * per eval su modelli grandi, NON reclamata da clear_cache — cache=0 ma active cresce) puo'
 * saturare il sistema in processi long-lived che fanno molti eval grandi. Esporre le statistiche
 * permette watchdog/supervisione; impostare un limite fa fallire gli eval in modo PULITO (errore
 * MLX) invece di far uccidere il processo da Jetsam. Ritorna 0 ok / -1. */

int llm_ai_mlx_memory_stats(size_t* out_cache, size_t* out_active, size_t* out_peak) {
    size_t c = 0, a = 0, p = 0;
    if (out_cache && mlx_get_cache_memory(&c) != 0) return -1;
    if (out_active && mlx_get_active_memory(&a) != 0) return -1;
    if (out_peak && mlx_get_peak_memory(&p) != 0) return -1;
    if (out_cache) *out_cache = c;
    if (out_active) *out_active = a;
    if (out_peak) *out_peak = p;
    return 0;
}

/* §M/L25 — probe di ATTRIBUZIONE della ritenzione: bilancio deterministico
 * `mlx_array_new* − mlx_array_free` (net deve restare COSTANTE a regime) + active/peak MLX.
 * Richiede build con `-DLLM_COUNT_ARRAYS=1` (default 0 = no-op, zero overhead). */
int llm_ai_mlx_debug_array_balance(const char* tag, long* out_new, long* out_free, long* out_net) {
    size_t a = 0, p = 0, c = 0;
    (void)mlx_get_active_memory(&a); (void)mlx_get_peak_memory(&p); (void)mlx_get_cache_memory(&c);
#if LLM_COUNT_ARRAYS
    llm_mlx_array_balance(tag ? tag : "probe");
    if (out_new) *out_new = g_arr_new_ct;
    if (out_free) *out_free = g_arr_free_ct;
    if (out_net) *out_net = g_arr_new_ct - g_arr_free_ct;
#else
    fprintf(stderr, "[ARRAYS] %s (counting OFF: ricompilare con -DLLM_COUNT_ARRAYS=1)\n", tag ? tag : "probe");
    if (out_new) *out_new = 0;
    if (out_free) *out_free = 0;
    if (out_net) *out_net = 0;
#endif
    fprintf(stderr, "[ARRAYS] %s active=%.1f MB peak=%.1f MB cache=%.1f MB\n",
            tag ? tag : "probe", a / 1048576.0, p / 1048576.0, c / 1048576.0);
    return 0;
}

/* §P12 — vedi llmengine_mlx.h: byte/token (pesi + KV) per la generazione. */
int llm_ai_mlx_debug_mem_report(llm_ai_mlx_model_handle mh, int offset,
                                long* out_total, long* out_weights, long* out_kv, long* out_tensors) {
    if (mh == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    long total = 0, embed = 0, nt = 0;
    for (int si = 0; si < m->weights.count; ++si) {
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(m->weights.shards[si]);
        const char* key = NULL;
        mlx_array val = mlx_array_new();   /* placeholder: next() lo sostituisce (handle borrowato) */
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
            long nb = (long)mlx_array_nbytes(val);
            total += nb; nt++;
            if (key != NULL && (strstr(key, "embed_tokens") != NULL || strstr(key, "embedding") != NULL ||
                                strstr(key, "wte") != NULL))
                embed += nb;   /* in decode si legge 1 riga, non l'intera matrice */
        }
        mlx_map_string_to_array_iterator_free(it);
    }
    long kv = 0;
    if (m->cfg.n_layers > 0 && m->cfg.n_kv_heads > 0 && m->cfg.head_dim > 0 && offset > 0)
        kv = (long)m->cfg.n_layers * (long)m->cfg.n_kv_heads * (long)m->cfg.head_dim * 4L * (long)offset; /* K+V, 2B */
    if (out_total) *out_total = total;
    if (out_weights) *out_weights = total - embed;
    if (out_kv) *out_kv = kv;
    if (out_tensors) *out_tensors = nt;
    return 0;
}

int llm_ai_mlx_set_memory_limit_bytes(size_t limit_bytes) {
    size_t prev = 0;
    return mlx_set_memory_limit(&prev, limit_bytes) == 0 ? 0 : -1;
}

/* Guardia saturazione memoria: tetto su mlx ACTIVE (non la cache, che MLX libera gia'; la
 * ritenzione "active" dell'allocatore su eval grandi NON e' soggetta a mlx_set_memory_limit,
 * misurato 2026-09-02). Ritorna 0 = ok / -2 = budget active superato (il chiamante deve riciclare
 * il processo / ridurre lo scope). Costo: una get_active_memory per chiamata — trascurabile; con
 * limite off (default) e' un solo test su int. */
int llm_mlx_active_guard(LlmMlxModel* m) {
    if (m == NULL || m->mem_active_limit_mb <= 0) return 0;
    size_t active = 0;
    if (mlx_get_active_memory(&active) != 0) return 0;
    if (active > (size_t)m->mem_active_limit_mb * 1048576ULL) return -2;
    return 0;
}

/* Limite memoria per-modello (MB; 0 = off). "mlxMemoryLimitMB" = tetto su mlx ACTIVE enforced da
 * LLMEnv (guardia anti-saturazione), NON il limite cache di MLX (che non limita la ritenzione
 * active, verificato). Default off -> nessuna regressione. */
int llm_ai_mlx_model_set_memory_limit_mb(llm_ai_mlx_model_handle handle, int limit_mb) {
    if (handle == NULL) return -1;
    ((LlmMlxModel*)handle)->mem_active_limit_mb = limit_mb;
    return 0;
}

int llm_ai_mlx_model_memory_limit_mb(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return -1;
    return ((LlmMlxModel*)handle)->mem_active_limit_mb;
}

int llm_ai_mlx_model_flops(llm_ai_mlx_model_handle handle, int seq_len) {

    if (handle == NULL) return -1;    const ModelConfig* c = &((LlmMlxModel*)handle)->cfg;

    double params = 0; llm_ai_mlx_model_stats(handle, seq_len, &params, NULL);

    long long L = seq_len > 0 ? seq_len : 1;

    /* MLP + attention proj: ~2 * params * L (matmul). Attention core: 2*L^2*nh*hd. */

    double flops = 2.0 * params * L + 2.0 * L * L * c->n_heads * c->head_dim;

    return (int)(flops / 1e9); /* GFLOPs */

}



/* Stampa le stime (MODEL MEMORY / FLOPS) e ritorna 0 ok. */

int llm_ai_mlx_model_memory(llm_ai_mlx_model_handle handle, int seq_len) {

    if (handle == NULL) return -1;

    double params = 0, kv = 0;

    if (llm_ai_mlx_model_stats(handle, seq_len, &params, &kv) != 0) return -1;

    int gflops = llm_ai_mlx_model_flops(handle, seq_len);

    fprintf(stderr, "MODEL stats (seq=%d):\n", seq_len);

    fprintf(stderr, "  params ~= %.1fM (%.2fB)\n", params / 1e6, params / 1e9);

    fprintf(stderr, "  KV-cache ~= %.1f MB (FP16, k+v, %d layers)\n", kv / 1e6, ((LlmMlxModel*)handle)->cfg.n_layers);

    fprintf(stderr, "  prefill FLOPS ~= %d GFLOPs\n", gflops);

    return 0;

}
