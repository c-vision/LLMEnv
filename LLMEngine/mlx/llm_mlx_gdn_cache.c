/* ============================================================================
 * llm_mlx_gdn_cache.c — dominio cache per-layer di pesi/stato GDN e Qwen4-Exp: scalare nel
 * dtype di attivazione, conv1d/dt_bias/A_log GDN, stato QSA indexer e conv PLE. Estratto
 * VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* P2 invarianti GDN (2026-09-03): peso conv1d (BF16) castato FP32 UNA volta per layer e
 * cachato (prima: cast a ogni forward, in tutti e 3 i path GDN). Ritorna puntatore al handle
 * cachato o NULL se non disponibile (fallback legacy del chiamante).
 * NB (2026-09-13): il kernel GDN fuso (llm_mlx_gdn_prefill) ha InT = dtype di q/k/v e state
 * float; forzare qui il dtype di attivazione (bf16) rompe l'uniformita' con q_s/k_s f32 e fa
 * fallire la build del kernel Metal. Il residuo f32 del conv GDN resta quindi un target
 * separato (richiede di portare TUTTO il blocco GDN-a-16bit, non solo il conv). */
/* Scalare nel dtype di attivazione (g_act16): evita che un moltiplicatore f32 promuova a
 * f32 un tensore bf16 (era il caso di inv1/inv2 nel GDN: promozione + AsType per layer).
 * bf16/f16 sono scritti come bit pattern (mlx_array_new_data copia), zero nodi nel grafo. */
mlx_array llm_mlx_act_scalar(float v) {
    if (g_act16 == MLX_BFLOAT16) {
        uint32_t u; memcpy(&u, &v, sizeof(u));
        uint16_t b = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); /* round-nearest-even */
        return mlx_array_new_data(&b, NULL, 0, MLX_BFLOAT16);
    }
    if (g_act16 == MLX_FLOAT16) {
        _Float16 h = (_Float16)v;
        return mlx_array_new_data(&h, NULL, 0, MLX_FLOAT16);
    }
    return mlx_array_new_float32(v);
}
mlx_array* llm_mlx_gdn_conv_cached(LlmMlxModel* m, int layer_idx) {
    if (m == NULL) return NULL;
    if (m->gdn_cw == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->gdn_cw = (mlx_array*)calloc((size_t)n, sizeof(mlx_array));
        if (m->gdn_cw == NULL) return NULL;
        m->gdn_cw_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->gdn_cw_n) return NULL;
    if (m->gdn_cw[layer_idx].ctx == NULL) {
        char nb[256];
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.conv1d.weight", nb, sizeof(nb));
        mlx_array w = mlx_array_new(), cw = mlx_array_new();
        /* dtype di ATTIVAZIONE (reference mlx_lm: conv1d in inputs.dtype, bf16): con f32
         * il conv_input (concat conv_state+qkv) promuoveva tutto a f32 e il blocco GDN
         * pagava ~8 AsType/layer. Il cast f32 era nato per far combaciare il weight col
         * conv_input f32; allineato il conv_state al dtype di attivazione, il weight segue. */
        bool okk = multishard_get(&m->weights, nb, &w) &&
                   mlx_astype(&cw, w, llm_mlx_act_dt(m->cfg.bfloat16_input), m->stream) == 0;
        mlx_array_free(w);
        if (!okk) { mlx_array_free(cw); return NULL; }
        mlx_array_set(&m->gdn_cw[layer_idx], cw);
        mlx_array_free(cw);
    }
    return &m->gdn_cw[layer_idx];
}

/* P2 invarianti GDN: linear_attn.dt_bias e' un peso costante letto a ogni forward -> cache per-layer. */
mlx_array* llm_mlx_gdn_dt_bias(LlmMlxModel* m, int layer_idx) {
    if (m == NULL) return NULL;
    if (m->gdn_dt == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->gdn_dt = (mlx_array*)calloc((size_t)n, sizeof(mlx_array));
        if (m->gdn_dt == NULL) return NULL;
        m->gdn_dt_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->gdn_dt_n) return NULL;
    if (m->gdn_dt[layer_idx].ctx == NULL) {
        char nb[256];
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.dt_bias", nb, sizeof(nb));
        mlx_array w = mlx_array_new();
        bool okk = multishard_get(&m->weights, nb, &w);
        if (!okk) { mlx_array_free(w); return NULL; }
        mlx_array_set(&m->gdn_dt[layer_idx], w);
        mlx_array_free(w);
    }
    return &m->gdn_dt[layer_idx];
}

/* Qwen4-Exp: stato persistente dell'indexer QSA per-layer (ring raw + chiavi compresse
 * growable). Lazy come gdn_dt/gdn_cw sopra: alloca l'array [n_layers] al primo layer QSA
 * incontrato, inizializza la singola cache (ring/compressed vuoti, offset=0) alla prima
 * chiamata su QUEL layer. Mai liberata/reinizializzata fra un token e l'altro -- e'
 * esattamente lo stato che deve persistere (stesso ruolo di k_cache/v_cache per
 * l'attenzione full, ma con una shape che cresce in modo diverso, da cui la struct
 * dedicata invece di un mlx_array). */
LlmQwen4ExpQSACache* llm_mlx_qwen4exp_model_qsa_cache(LlmMlxModel* m, int layer_idx) {
    if (m == NULL) return NULL;
    if (m->qsa_indexer_cache == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->qsa_indexer_cache = (LlmQwen4ExpQSACache*)calloc((size_t)n, sizeof(LlmQwen4ExpQSACache));
        if (m->qsa_indexer_cache == NULL) return NULL;
        m->qsa_indexer_cache_n = n;
        for (int i = 0; i < n; i++) m->qsa_indexer_cache[i].ring.ctx = NULL; /* calloc gia' azzera, esplicito per chiarezza */
    }
    if (layer_idx < 0 || layer_idx >= m->qsa_indexer_cache_n) return NULL;
    LlmQwen4ExpQSACache* c = &m->qsa_indexer_cache[layer_idx];
    if (c->dim == 0 && c->ring.ctx == NULL && c->compressed.ctx == NULL)
        llm_mlx_qwen4exp_qsa_cache_init(c, m->cfg.indexer_compress_ratio, m->cfg.indexer_head_dim);
    return c;
}

/* PLE: stato del conv1d persistente, per-layer (analogo a llm_mlx_gdn_alog_exp sotto ma
 * per un mlx_array intero invece di un peso costante). *out_has_state = true se la cache
 * conteneva gia' un valore da una chiamata precedente (continuazione decode), false se
 * questo layer PLE non e' mai stato forwardato prima (fresh, il chiamante deve passare
 * conv_state_in.ctx==NULL a llm_mlx_qwen4exp_ple_layer). */
mlx_array* llm_mlx_qwen4exp_ple_conv_cache(LlmMlxModel* m, int layer_idx, bool* out_has_state) {
    if (m == NULL) return NULL;
    if (m->ple_conv_cache == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->ple_conv_cache = (mlx_array*)calloc((size_t)n, sizeof(mlx_array));
        if (m->ple_conv_cache == NULL) return NULL;
        m->ple_conv_cache_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->ple_conv_cache_n) return NULL;
    if (out_has_state) *out_has_state = (m->ple_conv_cache[layer_idx].ctx != NULL);
    return &m->ple_conv_cache[layer_idx];
}

/* Imatrix: inizia la raccolta sul modello (alloca gli accumulatori dalle shape del config).
 * Richiede un modello MoE (n_experts>0). Ritorna 0 ok / -1. */

/* P2 invarianti GDN: exp(A_log) e' costante per layer (A_log e' un peso) e veniva ricalcolato a
 * ogni forward nei 3 path GDN. Cache per-layer; fallback legacy se non disponibile. */
mlx_array* llm_mlx_gdn_alog_exp(LlmMlxModel* m, int layer_idx) {
    if (m == NULL) return NULL;
    if (m->gdn_ae == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->gdn_ae = (mlx_array*)calloc((size_t)n, sizeof(mlx_array));
        if (m->gdn_ae == NULL) return NULL;
        m->gdn_ae_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->gdn_ae_n) return NULL;
    if (m->gdn_ae[layer_idx].ctx == NULL) {
        char nb[256];
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.A_log", nb, sizeof(nb));
        mlx_array w = mlx_array_new(), w32 = mlx_array_new(), e = mlx_array_new();
        /* Reference mlx_lm: `mx.exp(-mx.exp(A_log.astype(mx.float32)) * softplus(a + dt_bias))`
         * -> exp calcolato in f32 (prima era in bf16 e poi promosso per-token). */
        bool okk = multishard_get(&m->weights, nb, &w) &&
                   mlx_astype(&w32, w, MLX_FLOAT32, m->stream) == 0 &&
                   mlx_exp(&e, w32, m->stream) == 0;
        mlx_array_free(w); mlx_array_free(w32);
        if (!okk) { mlx_array_free(e); return NULL; }
        mlx_array_set(&m->gdn_ae[layer_idx], e);
        mlx_array_free(e);
    }
    return &m->gdn_ae[layer_idx];
}
