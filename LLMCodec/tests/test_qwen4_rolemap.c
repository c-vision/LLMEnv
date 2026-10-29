/* test_qwen4_rolemap.c — verifica la mappa HF -> GGUF di ABI/qwen4_exp.gmbarch
 * (fonte di verita' unica dei nomi). Uso: llmcodec_qwen4_rolemap_selfcheck [path] */
#include "llmcodec_rolemap.h"

#include <stdio.h>
#include <string.h>
#ifndef LLMCODEC_Q4_ABI
#define LLMCODEC_Q4_ABI "../../ABI/qwen4_exp.gmbarch"
#endif

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : LLMCODEC_Q4_ABI;
    LlmCodecRoleMap rm;
    if (llmcodec_rolemap_load(path, &rm) != 0) { printf("FAIL load %s\n", path); return 1; }
    static const struct { const char* hf; const char* want; } t[] = {
        {"language_model.model.layers.0.attn_hyper_connection.hc_norm.weight", "blk.%d.hc_attn_norm.weight"},
        {"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "blk.%d.hc_attn_down.weight"},
        {"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "blk.%d.hc_attn_up.weight"},
        {"language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight", "blk.%d.hc_attn_inject.weight"},
        {"language_model.model.layers.0.mlp_hyper_connection.hc_norm.weight", "blk.%d.hc_ffn_norm.weight"},
        {"language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", "blk.%d.hc_ffn_down.weight"},
        {"language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "blk.%d.hc_ffn_up.weight"},
        {"language_model.model.layers.0.mlp_hyper_connection.block_inject_weight.weight", "blk.%d.hc_ffn_inject.weight"},
        {"language_model.model.layers.0.linear_attn.in_proj_qkv.weight", "blk.%d.attn_qkv.weight"},
        {"language_model.model.layers.0.linear_attn.in_proj_z.weight", "blk.%d.attn_gate.weight"},
        {"language_model.model.layers.0.linear_attn.conv1d.weight", "blk.%d.ssm_conv1d.weight"},
        {"language_model.model.layers.0.linear_attn.dt_bias", "blk.%d.ssm_dt.bias"},
        {"language_model.model.layers.0.linear_attn.A_log", "blk.%d.ssm_a"},
        {"language_model.model.layers.0.linear_attn.in_proj_b.weight", "blk.%d.ssm_beta.weight"},
        {"language_model.model.layers.0.linear_attn.in_proj_a.weight", "blk.%d.ssm_alpha.weight"},
        {"language_model.model.layers.0.linear_attn.norm.weight", "blk.%d.ssm_norm.weight"},
        {"language_model.model.layers.0.linear_attn.out_proj.weight", "blk.%d.ssm_out.weight"},
        {"language_model.model.layers.1.ple.key_proj.weight", "blk.%d.ple_key.weight"},
        {"language_model.model.layers.1.ple.value_proj.weight", "blk.%d.ple_value.weight"},
        {"language_model.model.layers.1.ple.norm_key.weight", "blk.%d.ple_norm_key.weight"},
        {"language_model.model.layers.1.ple.norm_query.weight", "blk.%d.ple_norm_query.weight"},
        {"language_model.model.layers.1.ple.norm_conv.weight", "blk.%d.ple_norm_conv.weight"},
        {"language_model.model.layers.1.ple.conv1d.weight", "blk.%d.ple_conv1d.weight"},
        {"language_model.model.layers.1.ple.ple_embedding.layer_multipliers", "blk.%d.ple_layer_multipliers"},
        {"language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets", "ple.head_offsets"},
        {"language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes", "ple.head_vocab_sizes"},
        {"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.7.weight", "ple.shard.%i.weight"},
        {"language_model.model.layers.0.self_attn.indexer.q_proj.weight", "blk.%d.indexer.q_proj.weight"},
        {"language_model.model.layers.0.self_attn.indexer.k_proj.weight", "blk.%d.indexer.k_proj.weight"},
        {"language_model.model.layers.0.self_attn.indexer.q_norm.weight", "blk.%d.indexer.q_norm.weight"},
        {"language_model.model.layers.0.self_attn.indexer.k_norm.weight", "blk.%d.indexer.k_norm.weight"},
        {"language_model.model.layers.0.mlp.shared_expert.gate_proj.weight", "blk.%d.ffn_gate_shexp.weight"},
        {"language_model.model.layers.0.mlp.shared_expert.up_proj.weight", "blk.%d.ffn_up_shexp.weight"},
        {"language_model.model.layers.0.mlp.shared_expert.down_proj.weight", "blk.%d.ffn_down_shexp.weight"},
        {"language_model.model.layers.0.mlp.shared_expert_gate.weight", "blk.%d.ffn_gate_inp_shexp.weight"},
        {"language_model.model.layers.0.self_attn.q_proj.weight", "blk.%d.attn_q.weight"},
        {"language_model.model.layers.0.self_attn.k_proj.weight", "blk.%d.attn_k.weight"},
        {"language_model.model.layers.0.self_attn.v_proj.weight", "blk.%d.attn_v.weight"},
        {"language_model.model.layers.0.self_attn.o_proj.weight", "blk.%d.attn_output.weight"},
        {"language_model.model.layers.0.self_attn.q_norm.weight", "blk.%d.attn_q_norm.weight"},
        {"language_model.model.layers.0.self_attn.k_norm.weight", "blk.%d.attn_k_norm.weight"},
        {"language_model.model.layers.0.mlp.switch_mlp.up_proj.weight", "blk.%d.ffn_%s_exps.weight"},
        {"language_model.model.layers.0.mlp.gate.weight", "blk.%d.ffn_gate_inp.weight"},
        {"language_model.model.embed_tokens.weight", "token_embd.weight"},
        {"language_model.lm_head.weight", "output.weight"},
        {"language_model.model.norm.weight", "output_norm.weight"},
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        const char* g = llmcodec_rolemap_gguf(&rm, t[i].hf);
        if (!g || strcmp(g, t[i].want) != 0) {
            printf("  FAIL %s -> %s (atteso %s)\n", t[i].hf, g ? g : "(null)", t[i].want);
            bad++;
        }
    }
    printf("[qwen4-rolemap] casi=%zu falliti=%d\n", sizeof t / sizeof t[0], bad);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
