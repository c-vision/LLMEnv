/* llmengine_moe_arch.h — descriptor GENERICO di architettura + ruoli dei tensori.
 *
 * Il descriptor (geometria numerica) non ha più un blob `GMAD` proprietario: la geometria è
 * fornita dal `.gmbarch` (llmdesc) oppure derivata dal discovery (KV GGUF / config.json). Qui
 * restano solo la struct `LlmEngineMoeArchDesc`, gli enum e la name-map dei ruoli.
 * Vedi `Todo/(open) - Model-Architecture-Auto-Discovery.md`. */
#ifndef LLMENGINE_MOE_ARCH_H
#define LLMENGINE_MOE_ARCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* arch_family. 0 = unknown -> adapter di fallback "container". */
enum {
    LLMENGINE_MOE_ARCH_UNKNOWN    = 0,
    LLMENGINE_MOE_ARCH_QWEN3_MOE  = 1,   /* Qwen3-MoE (riferimento bit-exact) */
    LLMENGINE_MOE_ARCH_QWEN2_MOE  = 2,
    LLMENGINE_MOE_ARCH_GLM_MOE    = 3,
    LLMENGINE_MOE_ARCH_DEEPSEEK   = 4,   /* MLA */
    LLMENGINE_MOE_ARCH_LLAMA      = 5,   /* dense */
    LLMENGINE_MOE_ARCH_GEMMA      = 6,
    LLMENGINE_MOE_ARCH_QWEN3_NEXT = 7    /* ibrida (linear+full attention) */
};
const char* llmengine_moe_arch_family_name(uint32_t family);

/* rope_variant (cb+52) */
enum { LLMENGINE_MOE_ROPE_HALF = 0, LLMENGINE_MOE_ROPE_INTERLEAVED = 1 };
/* router_kind (cb+56) */
enum { LLMENGINE_MOE_ROUTER_SOFTMAX = 0, LLMENGINE_MOE_ROUTER_SIGMOID = 1 };
/* norm_kind (cb+60) */
enum { LLMENGINE_MOE_NORM_RMS = 0, LLMENGINE_MOE_NORM_LAYER = 1 };
/* flags (cb+64) */
enum {
    LLMENGINE_MOE_FLAG_TIED    = 1u << 0,
    LLMENGINE_MOE_FLAG_QK_NORM = 1u << 1,
    LLMENGINE_MOE_FLAG_BIAS    = 1u << 2,
    LLMENGINE_MOE_FLAG_MOE     = 1u << 3,
    LLMENGINE_MOE_FLAG_MLA     = 1u << 4,
    LLMENGINE_MOE_FLAG_SHARED  = 1u << 5,
    LLMENGINE_MOE_FLAG_NO_RENORM = 1u << 6,  /* top-k NON rinormalizzati (es. OLMoE norm_topk_prob=false) */
    LLMENGINE_MOE_FLAG_QK_NORM_FULL = 1u << 7 /* q/k-norm su n_embd intero (es. OLMoE), non per-head (Qwen3) */
};

typedef struct {
    uint32_t arch_family;   /* enum sopra */
    uint32_t hidden, n_layers, n_heads, n_kv_heads, head_dim;
    uint32_t n_experts, topk, inter, vocab, ctx;
    uint32_t rope_variant, router_kind, norm_kind, flags;
    float    eps, rope_theta;
} LlmEngineMoeArchDesc;

/* Ruolo canonico di un tensore (name-map, derivato dalla nomenclatura). */
enum {
    LLMENGINE_MOE_ROLE_UNKNOWN = 0,
    LLMENGINE_MOE_ROLE_INPUT_LN,
    LLMENGINE_MOE_ROLE_POST_ATTN_LN,
    LLMENGINE_MOE_ROLE_Q, LLMENGINE_MOE_ROLE_K, LLMENGINE_MOE_ROLE_V, LLMENGINE_MOE_ROLE_O,
    LLMENGINE_MOE_ROLE_Q_NORM, LLMENGINE_MOE_ROLE_K_NORM,
    LLMENGINE_MOE_ROLE_ROUTER,               /* mlp.gate / ffn_gate_inp */
    LLMENGINE_MOE_ROLE_EXPERT_GATE, LLMENGINE_MOE_ROLE_EXPERT_UP, LLMENGINE_MOE_ROLE_EXPERT_DOWN,
    LLMENGINE_MOE_ROLE_EMBED, LLMENGINE_MOE_ROLE_FINAL_NORM, LLMENGINE_MOE_ROLE_LM_HEAD,
    LLMENGINE_MOE_ROLE_COUNT
};
const char* llmengine_moe_role_name(int role);

/* Classifica il nome di UN tensore in un ruolo (struttura, dai suffissi noti).
 * Ritorna LLMENGINE_MOE_ROLE_UNKNOWN se non riconosciuto. */
int llmengine_moe_role_of_tensor(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_ARCH_H */
