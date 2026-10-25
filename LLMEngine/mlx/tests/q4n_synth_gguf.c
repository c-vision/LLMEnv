/* q4n_synth_gguf.c — emette un GGUF qwen4exp sintetico per il test P7e.
 * Separato dal test perche' llm_gguf.h e llmcodec_gguf_writer.h definiscono
 * entrambi LlmCodecGgufTensor con layout diversi (stessa TU = conflitto). */
#include "llmcodec_gguf_writer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAXTS 64
static char g_names[MAXTS][128];
static float g_data[MAXTS][4];
static uint64_t g_dims[MAXTS][1];
static LlmCodecGgufTensor g_ts[MAXTS];
static int g_nts = 0;

static void add_ts(const char* name, float sig) {
    snprintf(g_names[g_nts], sizeof g_names[g_nts], "%s", name);
    for (int i = 0; i < 4; i++) g_data[g_nts][i] = sig + 0.01f * i;
    g_dims[g_nts][0] = 4;
    g_ts[g_nts].name = g_names[g_nts];
    g_ts[g_nts].ggml_type = LLMCODEC_GGML_F32;
    g_ts[g_nts].dims = g_dims[g_nts];
    g_ts[g_nts].ndim = 1;
    g_ts[g_nts].data = g_data[g_nts];
    g_ts[g_nts]._offset = 0;
    g_nts++;
}

int q4n_write_synth_gguf(const char* path) {
    LlmCodecGgufMetaKV kv[48]; int nkv = 0;
    uint32_t u[48]; float fv[48]; int nu = 0, nf = 0;
#define KV_STR(k, v) do { kv[nkv].key = (k); kv[nkv].type = LLMCODEC_GGUF_MT_STRING; kv[nkv].str = (v); nkv++; } while (0)
#define KV_U(k, v) do { u[nu] = (uint32_t)(v); kv[nkv].key = (k); kv[nkv].type = LLMCODEC_GGUF_MT_U32; kv[nkv].u32 = u[nu]; nkv++; nu++; } while (0)
#define KV_F(k, v) do { fv[nf] = (v); kv[nkv].key = (k); kv[nkv].type = LLMCODEC_GGUF_MT_F32; kv[nkv].f32 = fv[nf]; nkv++; nf++; } while (0)
    KV_STR("general.architecture", "qwen4exp");
    KV_U("qwen4exp.embedding_length", 8);
    KV_U("qwen4exp.block_count", 2);
    KV_U("qwen4exp.attention.head_count", 2);
    KV_U("qwen4exp.attention.head_count_kv", 1);
    KV_U("qwen4exp.attention.key_length", 4);
    KV_U("qwen4exp.attention.value_length", 4);
    KV_U("qwen4exp.attention.indexer.head_count", 1);
    KV_U("qwen4exp.attention.indexer.key_length", 4);
    KV_U("qwen4exp.attention.indexer.top_k", 4);
    KV_U("qwen4exp.expert_count", 4);
    KV_U("qwen4exp.expert_used_count", 2);
    KV_U("qwen4exp.expert_feed_forward_length", 4);
    KV_U("qwen4exp.expert_shared_feed_forward_length", 4);
    KV_U("qwen4exp.hyper_connection.count", 2);
    KV_U("qwen4exp.hyper_connection.low_rank", 3);
    KV_U("qwen4exp.ssm.conv_kernel", 4);
    KV_U("qwen4exp.ssm.group_count", 3);
    KV_U("qwen4exp.ssm.state_size", 4);
    KV_U("qwen4exp.ssm.inner_size", 2);
    KV_U("qwen4exp.ple.ngram_size", 3);
    KV_U("qwen4exp.ple.heads_per_ngram", 1);
    KV_U("qwen4exp.ple.conv_kernel", 4);
    KV_U("qwen4exp.ple.eos_token_id", 7);
    KV_U("qwen4exp.ple.layers", 1);
    KV_U("qwen4exp.rope.dimension_count", 4);
    KV_U("qwen4exp.full_attention_interval", 2);
    KV_U("qwen4exp.nextn_predict_layers", 0);
    KV_F("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f);
    KV_F("qwen4exp.rope.freq_base", 1e7f);

    const char* common[] = {"hc_attn_norm.weight","hc_attn_down.weight","hc_attn_up.weight","hc_attn_inject.weight",
                            "hc_ffn_norm.weight","hc_ffn_down.weight","hc_ffn_up.weight","hc_ffn_inject.weight",
                            "ffn_gate_inp.weight","ffn_gate_inp_shexp.weight",
                            "ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight",
                            "ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
    const char* linear[] = {"attn_qkv.weight","attn_gate.weight","ssm_conv1d.weight","ssm_dt.bias","ssm_a",
                            "ssm_beta.weight","ssm_alpha.weight","ssm_norm.weight","ssm_out.weight"};
    const char* full[] = {"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight",
                          "attn_q_norm.weight","attn_k_norm.weight",
                          "indexer.q_proj.weight","indexer.k_proj.weight",
                          "indexer.q_norm.weight","indexer.k_norm.weight"};
    const char* ple[] = {"ple_key.weight","ple_value.weight","ple_norm_key.weight",
                         "ple_norm_query.weight","ple_norm_conv.weight","ple_conv1d.weight"};
    char nm[128];
    float sig = 1.0f;
    for (int il = 0; il < 2; il++) {
        for (size_t i = 0; i < sizeof common / sizeof common[0]; i++) {
            snprintf(nm, sizeof nm, "blk.%d.%s", il, common[i]); add_ts(nm, sig += 1.0f);
        }
        const char** list = il == 0 ? linear : full;
        size_t n = il == 0 ? sizeof linear / sizeof linear[0] : sizeof full / sizeof full[0];
        for (size_t i = 0; i < n; i++) { snprintf(nm, sizeof nm, "blk.%d.%s", il, list[i]); add_ts(nm, sig += 1.0f); }
        if (il == 1) for (size_t i = 0; i < sizeof ple / sizeof ple[0]; i++) { snprintf(nm, sizeof nm, "blk.%d.%s", il, ple[i]); add_ts(nm, sig += 1.0f); }
    }
    return llmcodec_gguf_write(path, kv, nkv, g_ts, g_nts, 0);
}
