/* llmengine_moe_arch.c — nome famiglia d'architettura e name-map ruoli dei tensori.
 * Vedi include/llmengine_moe_arch.h per il contratto. Niente più blob `GMAD`. */
#include "llmengine_moe_arch.h"
#include <string.h>

const char* llmengine_moe_arch_family_name(uint32_t family) {
    switch (family) {
        case LLMENGINE_MOE_ARCH_QWEN3_MOE:  return "qwen3_moe";
        case LLMENGINE_MOE_ARCH_QWEN2_MOE:  return "qwen2_moe";
        case LLMENGINE_MOE_ARCH_GLM_MOE:    return "glm_moe";
        case LLMENGINE_MOE_ARCH_DEEPSEEK:   return "deepseek_mla";
        case LLMENGINE_MOE_ARCH_LLAMA:      return "llama";
        case LLMENGINE_MOE_ARCH_GEMMA:      return "gemma";
        case LLMENGINE_MOE_ARCH_QWEN3_NEXT: return "qwen3_next";
        default:                      return "unknown";
    }
}

const char* llmengine_moe_role_name(int role) {
    switch (role) {
        case LLMENGINE_MOE_ROLE_INPUT_LN:      return "input_layernorm";
        case LLMENGINE_MOE_ROLE_POST_ATTN_LN:  return "post_attention_layernorm";
        case LLMENGINE_MOE_ROLE_Q:             return "q_proj";
        case LLMENGINE_MOE_ROLE_K:             return "k_proj";
        case LLMENGINE_MOE_ROLE_V:             return "v_proj";
        case LLMENGINE_MOE_ROLE_O:             return "o_proj";
        case LLMENGINE_MOE_ROLE_Q_NORM:        return "q_norm";
        case LLMENGINE_MOE_ROLE_K_NORM:        return "k_norm";
        case LLMENGINE_MOE_ROLE_ROUTER:        return "router";
        case LLMENGINE_MOE_ROLE_EXPERT_GATE:   return "experts.gate_proj";
        case LLMENGINE_MOE_ROLE_EXPERT_UP:     return "experts.up_proj";
        case LLMENGINE_MOE_ROLE_EXPERT_DOWN:   return "experts.down_proj";
        case LLMENGINE_MOE_ROLE_EMBED:         return "embed_tokens";
        case LLMENGINE_MOE_ROLE_FINAL_NORM:    return "final_norm";
        case LLMENGINE_MOE_ROLE_LM_HEAD:       return "lm_head";
        default:                         return "unknown";
    }
}

/* Classificazione strutturale dal nome. Ordine deliberato: i suffissi più specifici
 * (q_norm vs q_proj; *_exps vs *_inp; gate_proj vs mlp.gate) vanno testati per primi. */
int llmengine_moe_role_of_tensor(const char* name) {
    if (name == NULL) return LLMENGINE_MOE_ROLE_UNKNOWN;

    /* norm per-head (prima di q/k generici) */
    if (strstr(name, "q_norm")) return LLMENGINE_MOE_ROLE_Q_NORM;
    if (strstr(name, "k_norm")) return LLMENGINE_MOE_ROLE_K_NORM;

    /* MoE: esperti stacked/routed PRIMA del router (gate) */
    if (strstr(name, "gate_proj") || strstr(name, "ffn_gate_exps") || strstr(name, ".w1."))
        return LLMENGINE_MOE_ROLE_EXPERT_GATE;
    if (strstr(name, "up_proj")   || strstr(name, "ffn_up_exps")   || strstr(name, ".w3."))
        return LLMENGINE_MOE_ROLE_EXPERT_UP;
    if (strstr(name, "down_proj") || strstr(name, "ffn_down_exps") || strstr(name, ".w2."))
        return LLMENGINE_MOE_ROLE_EXPERT_DOWN;

    /* router (gate d'ingresso del MoE) */
    if (strstr(name, "mlp.gate") || strstr(name, "ffn_gate_inp") || strstr(name, "gate.weight") ||
        strstr(name, "router"))
        return LLMENGINE_MOE_ROLE_ROUTER;

    /* attention projection */
    if (strstr(name, "q_proj") || strstr(name, "attn_q") || strstr(name, "q_a_proj")) return LLMENGINE_MOE_ROLE_Q;
    if (strstr(name, "k_proj") || strstr(name, "attn_k")) return LLMENGINE_MOE_ROLE_K;
    if (strstr(name, "v_proj") || strstr(name, "attn_v")) return LLMENGINE_MOE_ROLE_V;
    if (strstr(name, "o_proj") || strstr(name, "attn_output") || strstr(name, "o_a_proj")) return LLMENGINE_MOE_ROLE_O;

    /* norm di blocco */
    if (strstr(name, "post_attention_layernorm") || strstr(name, "ffn_norm")) return LLMENGINE_MOE_ROLE_POST_ATTN_LN;
    if (strstr(name, "input_layernorm") || strstr(name, "attn_norm") || strstr(name, "attention_norm"))
        return LLMENGINE_MOE_ROLE_INPUT_LN;

    /* testa / corpo */
    if (strstr(name, "embed_tokens") || strstr(name, "token_embd")) return LLMENGINE_MOE_ROLE_EMBED;
    if (strstr(name, "lm_head") || strstr(name, "output.weight")) return LLMENGINE_MOE_ROLE_LM_HEAD;
    if (strstr(name, "model.norm") || strstr(name, "output_norm") || strstr(name, "final_norm"))
        return LLMENGINE_MOE_ROLE_FINAL_NORM;

    return LLMENGINE_MOE_ROLE_UNKNOWN;
}
