/* llmengine_moe_arch.c — implementazione del descriptor generico `GMAD` v2 e del name-map ruoli.
 * Vedi include/llmengine_moe_arch.h per il contratto. */
#include "llmengine_moe_arch.h"
#include <string.h>

/* ── lettura/scrittura little-endian esplicita (indipendente dall'host) ── */
static void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff); p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff); p[3] = (uint8_t)((v >> 24) & 0xff);
}
static uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

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

void llmengine_moe_arch_encode(const LlmEngineMoeArchDesc* d, uint8_t out[LLMENGINE_MOE_GMAD_SIZE]) {
    memset(out, 0, LLMENGINE_MOE_GMAD_SIZE);
    put_u32(out + 0,  LLMENGINE_MOE_GMAD_MAGIC);
    put_u32(out + 4,  LLMENGINE_MOE_GMAD_VERSION);
    put_u32(out + 8,  d->arch_family);
    put_u32(out + 12, d->hidden);
    put_u32(out + 16, d->n_layers);
    put_u32(out + 20, d->n_heads);
    put_u32(out + 24, d->n_kv_heads);
    put_u32(out + 28, d->head_dim);
    put_u32(out + 32, d->n_experts);
    put_u32(out + 36, d->topk);
    put_u32(out + 40, d->inter);
    put_u32(out + 44, d->vocab);
    put_u32(out + 48, d->ctx);
    put_u32(out + 52, d->rope_variant);
    put_u32(out + 56, d->router_kind);
    put_u32(out + 60, d->norm_kind);
    put_u32(out + 64, d->flags);
    memcpy(out + 68, &d->eps, 4);
    memcpy(out + 72, &d->rope_theta, 4);
}

int llmengine_moe_arch_decode(const uint8_t* buf, size_t n, LlmEngineMoeArchDesc* out) {
    if (buf == NULL || out == NULL || n < 48) return -1;
    memset(out, 0, sizeof(*out));
    uint32_t magic = rd_u32(buf);
    if (magic == LLMENGINE_MOE_GMAD_MAGIC && n >= LLMENGINE_MOE_GMAD_USED) {   /* GMAD v2 */
        out->arch_family  = rd_u32(buf + 8);
        out->hidden       = rd_u32(buf + 12);
        out->n_layers     = rd_u32(buf + 16);
        out->n_heads      = rd_u32(buf + 20);
        out->n_kv_heads   = rd_u32(buf + 24);
        out->head_dim     = rd_u32(buf + 28);
        out->n_experts    = rd_u32(buf + 32);
        out->topk         = rd_u32(buf + 36);
        out->inter        = rd_u32(buf + 40);
        out->vocab        = rd_u32(buf + 44);
        out->ctx          = rd_u32(buf + 48);
        out->rope_variant = rd_u32(buf + 52);
        out->router_kind  = rd_u32(buf + 56);
        out->norm_kind    = rd_u32(buf + 60);
        out->flags        = rd_u32(buf + 64);
        memcpy(&out->eps, buf + 68, 4);
        memcpy(&out->rope_theta, buf + 72, 4);
        return 0;
    }
    if (magic == 0x46433351u) {                                    /* 'Q3CF' v1 (retro-compat) */
        out->arch_family  = LLMENGINE_MOE_ARCH_QWEN3_MOE;
        out->hidden       = rd_u32(buf + 8);
        out->n_heads      = rd_u32(buf + 12);
        out->n_kv_heads   = rd_u32(buf + 16);
        out->head_dim     = rd_u32(buf + 20);
        out->n_layers     = rd_u32(buf + 24);
        out->n_experts    = rd_u32(buf + 28);
        out->topk         = rd_u32(buf + 32);
        out->inter        = rd_u32(buf + 36);
        out->vocab        = rd_u32(buf + 40);
        out->ctx          = rd_u32(buf + 44);
        memcpy(&out->eps, buf + 48, 4);
        memcpy(&out->rope_theta, buf + 52, 4);
        out->flags = LLMENGINE_MOE_FLAG_QK_NORM | LLMENGINE_MOE_FLAG_MOE;
        return 0;
    }
    return -1;
}
