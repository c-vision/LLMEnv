/* llmengine_moe_qwen3_forward.h — M15.3: forward Qwen3-MoE (decode a singolo token con KV cache).
 *
 * Architettura (Qwen3 MoE): RMSNorm → GQA attention (q_norm/k_norm per-head, RoPE, causale) →
 * residual → RMSNorm → MoE (router → top-k → esperti SwiGLU → combine pesato) → residual →
 * final norm → lm_head. Gli esperti sono fetchati via callback (dequant-on-read dal runtime).
 * Tutti i tensori dense sono float32 (l'adattatore dequantizza f16→f32 al load). */
#ifndef LLMENGINE_MOE_QWEN3_FORWARD_H
#define LLMENGINE_MOE_QWEN3_FORWARD_H

#include "llmengine_moe_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t hidden, n_heads, n_kv_heads, head_dim, n_layers, n_experts, topk, inter, vocab, ctx;
    float eps, rope_theta;
    /* varianti (descriptor generico d'architettura) */
    uint32_t rope_variant;   /* 0 = half-split (Qwen/Llama), 1 = interleaved (GPT-NeoX) */
    uint32_t router_kind;    /* 0 = softmax sui top-k, 1 = sigmoid + renormalizzazione */
    uint32_t norm_kind;      /* 0 = RMSNorm, 1 = LayerNorm */
    uint32_t flags;          /* bit0 tied, bit1 qk_norm, bit2 bias, bit3 moe, bit4 mla, bit5 shared */
    uint32_t arch_family;    /* 0 unknown, 1 qwen3_moe, ... */
} Q3Cfg;

/* Pesi di UN layer. AZZERARE (memset) prima di riempire i campi: i campi opzionali
 * (q_norm/k_norm/shared_*) a NULL indicano l'assenza della variante. */
typedef struct {
    const float* input_ln;      /* [hidden] */
    const float* post_attn_ln;  /* [hidden] */
    const float* q_w;           /* [hidden, hidden]  (out,in) row-major */
    const float* k_w;           /* [kv_heads*head_dim, hidden] */
    const float* v_w;           /* [kv_heads*head_dim, hidden] */
    const float* o_w;           /* [hidden, hidden] */
    const float* q_norm;        /* [head_dim] (per-head) o [H*D] (QK_NORM_FULL); NULL se assente */
    const float* k_norm;        /* [head_dim] o [KVH*D]; NULL se assente */
    const float* router;        /* [n_experts, hidden] */
    /* SHARED-EXPERT (M7.1, GLM/DeepSeek/Qwen2-MoE): un MLP SwiGLU sempre applicato, sommato al
     * routed. `shared_inter` = dim intermedia (n_expert_shared * n_ff_exp). NULL se assente. */
    const float* shared_gate;   /* [shared_inter, hidden] */
    const float* shared_up;     /* [shared_inter, hidden] */
    const float* shared_down;   /* [hidden, shared_inter] */
    uint32_t     shared_inter;
} Q3Layer;

typedef struct {
    float* k;  /* [n_layers][n_kv_heads][ctx][head_dim] */
    float* v;
} Q3KvCache;

/* Fetch di UN esperto: out[hidden] = expert(x[hidden]). */
typedef LlmEngineMoeStatus (*Q3ExpertFn)(void* ctx, uint32_t layer, uint32_t expert,
                                   const float* x, float* out);

/* CARICA il blob di (layer, expert) senza applicarlo: usato dal prefill BATCHED per caricare
 * un esperto UNA volta e applicarlo a tutte le posizioni che lo hanno selezionato (M17). */
typedef LlmEngineMoeStatus (*Q3ExpertBlobFn)(void* ctx, uint32_t layer, uint32_t expert,
                                       const void** blob, size_t* size);

/* M6 v2 — feed ROUTER-BASED del prefetch: l'adapter notifica gli esperti scelti dal router per il
 * layer (top-k del decode, unione delle posizioni nel prefill), così il predittore cross-layer può
 * prefetchare il layer successivo (prima l'adapter reale NON notificava → prefetch inattivo). */
typedef void (*Q3RouteFn)(void* ctx, uint32_t layer, const uint32_t* experts, uint32_t n);

/* ── primitive (testabili in isolamento) ── */
void llmengine_moe_rmsnorm(const float* x, const float* w, float eps, uint32_t n, float* out);
/* LayerNorm (M7.2): normalizza a media 0 e varianza 1. `w` opzionale (NULL = senza scale).
 * Bias NON gestito (i modelli MoE con LayerNorm bias sono rari; vedi M7). */
void llmengine_moe_layernorm(const float* x, const float* w, float eps, uint32_t n, float* out);
void llmengine_moe_softmax(float* x, uint32_t n);
/* RoPE su un vettore [n_heads*head_dim], posizione assoluta `pos`, stile half-split (Qwen). */
void llmengine_moe_rope(float* vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta);
/* Come sopra, con variante: 0 = half-split (rotate_half), 1 = interleaved (GPT-NeoX). */
void llmengine_moe_rope_ex(float* vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta, uint32_t variant);
/* out[rows] = W·x, W row-major [rows, cols]. */
void llmengine_moe_matvec(const float* w, uint32_t rows, uint32_t cols, const float* x, float* out);

/* ── forward ── */
/* Un token: x[hidden] → out[hidden] (post-layernorm finale esclusa). `pos` = indice posizione.
 * `route_fn` (opzionale, M6 v2) riceve gli esperti scelti dal router per il layer. */
LlmEngineMoeStatus llmengine_moe_qwen3_decode_layer(const Q3Cfg* cfg, const Q3Layer* ly,
                                        Q3ExpertFn expert_fn, void* ectx,
                                        Q3RouteFn route_fn, void* route_ctx,
                                        uint32_t layer, uint32_t pos,
                                        const float* x, float* out, Q3KvCache* kv);

/* PREFILL BATCHED (M17): tutti i token del prompt in un colpo, con gli esperti CONDIVISI tra le
 * posizioni (per layer si prende l'UNIONE dei top-k e ogni esperto e' caricato UNA volta). Riduce
 * le letture da O(seq x topk x layer) a O(union x layer). `x`/`out` = [seq x hidden]; le posizioni
 * KV vanno da `pos0` a `pos0+seq-1`. `out` = hidden dopo `n_layers` per ogni posizione.
 * `route_fn` (opzionale, M6 v2) riceve l'UNIONE degli esperti del layer tra le posizioni. */
LlmEngineMoeStatus llmengine_moe_qwen3_prefill(const Q3Cfg* cfg, const Q3Layer* ly, uint32_t n_layers,
                                   Q3ExpertBlobFn blob_fn, void* bctx,
                                   Q3RouteFn route_fn, void* route_ctx,
                                   uint32_t seq, uint32_t pos0,
                                   const float* x, float* out, Q3KvCache* kv);

/* Applica UN esperto a `n` posizioni X[n][hidden] → out[n][hidden], con il peso caricato una
 * sola volta (M6.4: variante sorgente-agnostica — GGUF dequant-on-read, container MLX). */
typedef LlmEngineMoeStatus (*Q3ExpertBatchFn)(void* ctx, uint32_t layer, uint32_t expert,
                                        const float* X, uint32_t n, float* out);

/* Come `llmengine_moe_qwen3_prefill`, ma con un callback di applicazione batch sorgente-agnostico
 * invece del blob MLX. Stessa matematica (M17). */
LlmEngineMoeStatus llmengine_moe_qwen3_prefill_batchfn(const Q3Cfg* cfg, const Q3Layer* ly, uint32_t n_layers,
                                           Q3ExpertBatchFn batch_fn, void* bctx,
                                           Q3RouteFn route_fn, void* route_ctx,
                                           uint32_t seq, uint32_t pos0,
                                           const float* x, float* out, Q3KvCache* kv);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_QWEN3_FORWARD_H */
