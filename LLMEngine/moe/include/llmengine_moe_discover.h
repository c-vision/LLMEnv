/* llmengine_moe_discover.h — Auto-Discovery dell'architettura da un modello su disco.
 *
 * Deriva un `LlmEngineMoeArchDesc` e un name-map strutturale da `config.json` + index
 * safetensors (o header `model.safetensors`), SENZA convertire il modello. La parte strutturale
 * (~80-90%) è deterministica; le varianti semantiche (RoPE/attention/router/shared-expert)
 * possono essere corrette con override espliciti e restano soggette a verifica vs oracolo.
 *
 * Vedi `Todo/(open) - Model-Architecture-Auto-Discovery.md`. Implementazione in C, mai Python. */
#ifndef LLMENGINE_MOE_DISCOVER_H
#define LLMENGINE_MOE_DISCOVER_H

#include "llmengine_moe_arch.h"
#include "llmengine_moe_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    LlmEngineMoeArchDesc desc;                       /* descriptor derivato (geometria numerica) */
    uint32_t       n_tensors;                  /* tensori visti nell'index/header */
    uint32_t       role_counts[LLMENGINE_MOE_ROLE_COUNT];
    uint32_t       experts_per_layer;          /* rilevato dai nomi degli esperti (0 = non visto) */
    uint32_t       max_layer_seen;             /* n. layer implicito dai nomi (max indice + 1) */
    int            confidence;                 /* 0..100: stima di affidabilità del discovery */
    int            source_gguf;                /* 1 se derivato da metadati GGUF, 0 = HF safetensors */
    char           model_type[64];
    /* parametri MLA (DeepSeek-V2/V3) + struttura MoE per layer (0 se non applicabile) */
    uint32_t       mla_kv_lora, mla_qk_nope, mla_qk_rope, mla_v_head, mla_q_lora;
    uint32_t       first_k_dense;              /* n. layer iniziali con FFN denso (first_k_dense_replace) */
    uint32_t       dense_inter;                /* intermediate_size del FFN denso (≠ moe_intermediate_size) */
    uint32_t       n_shared_experts;
    /* YaRN (DeepSeek): rope_scaling */
    float          yarn_factor; uint32_t yarn_orig_max;
    float          yarn_beta_fast, yarn_beta_slow, yarn_mscale, yarn_mscale_all_dim;
} LlmEngineMoeDiscover;

/* Discovery da una directory HF/MLX (config.json + index safetensors). */
LlmEngineMoeStatus llmengine_moe_discover_hf(const char* dir, LlmEngineMoeDiscover* out);

/* Discovery da un file GGUF (metadati + tensor-info, nessun dato tensore). */
LlmEngineMoeStatus llmengine_moe_discover_gguf(const char* path, LlmEngineMoeDiscover* out);

/* Auto-detect: directory → HF, file `.gguf` → GGUF. */
LlmEngineMoeStatus llmengine_moe_discover_auto(const char* path, LlmEngineMoeDiscover* out);

/* Applica un override esplicito al descriptor (per le varianti incerte).
 * Chiavi: arch_family, hidden, n_layers, n_heads, n_kv, head_dim, n_experts, topk, inter,
 * vocab, ctx, eps, rope_theta, rope_variant, router_kind, norm_kind, flags.
 * Ritorna LLMENGINE_MOE_OK se la chiave è riconosciuta. */
LlmEngineMoeStatus llmengine_moe_discover_override(LlmEngineMoeArchDesc* d, const char* key, const char* value);

/* Report testuale (multiriga) del discovery. Ritorna n. caratteri scritti (>=0). */
int llmengine_moe_discover_report(const LlmEngineMoeDiscover* d, char* out, size_t cap);

/* M8 — LLM locale (processo esterno) per gli override INCERTI.
 * `llmengine_moe_discover_prompt`: costruisce il prompt (contesto del discovery + campi incerti +
 * schema JSON atteso) in `out`. Ritorna n. caratteri (>=0) o negativo se `cap` insufficiente.
 * `llmengine_moe_discover_apply_reply`: applica la risposta dell'LLM (JSON tollerante:
 * `{"chiave": valore, ...}`, valori numerici o stringhe-enum) al descriptor. Valida le chiavi
 * note e SCARTA quelle sconosciute/fuori range. Ritorna LLMENGINE_MOE_OK se ha applicato >=1 campo. */
int          llmengine_moe_discover_prompt(const LlmEngineMoeDiscover* d, char* out, size_t cap);
LlmEngineMoeStatus llmengine_moe_discover_apply_reply(LlmEngineMoeArchDesc* desc, const char* reply);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_DISCOVER_H */
