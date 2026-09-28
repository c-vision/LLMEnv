/* test_moe_discover.c — test del descriptor generico (GMAD v2) e dell'auto-discovery.
 * Crea una directory temporanea con config.json + index safetensors minimi e verifica che il
 * discovery derivi i campi attesi; verifica role-map, round-trip encode/decode e override. */
#include "llmengine_moe_discover.h"
#include "llmengine_moe_arch.h"
#include "llmengine_moe_gguf.h"
#include "llmengine_moe_ggml.h"
#include "llmcodec_fp16.h"   /* f32->f16 = atomo del CODEC */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); fails++; } } while (0)

static void write_file(const char* path, const char* text) {
    FILE* f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f) { fwrite(text, 1, strlen(text), f); fclose(f); }
}

/* ── writer GGUF minimale (per il test di discovery da metadati) ── */
static void w_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w_str(FILE* f, const char* s) { uint64_t n = strlen(s); w_u64(f, n); fwrite(s, 1, n, f); }
static void w_kv_u32(FILE* f, const char* k, uint32_t v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_UINT32); w_u32(f, v); }
static void w_kv_f32(FILE* f, const char* k, float v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_FLOAT32); fwrite(&v, 4, 1, f); }
static void w_kv_str(FILE* f, const char* k, const char* v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_STRING); w_str(f, v); }
static void w_tensor(FILE* f, const char* name, uint32_t nd, const uint64_t* dims, uint32_t type, uint64_t off) {
    w_str(f, name); w_u32(f, nd);
    for (uint32_t i = 0; i < nd; ++i) w_u64(f, dims[i]);
    w_u32(f, type); w_u64(f, off);
}

static void write_test_gguf(const char* path) {
    FILE* f = fopen(path, "wb");
    CHECK(f != NULL);
    if (!f) return;
    const uint32_t kv_count = 11, tensor_count = 6;
    w_u32(f, 0x46554747u);      /* 'GGUF' */
    w_u32(f, 3);                /* version */
    w_u64(f, tensor_count);
    w_u64(f, kv_count);
    w_kv_str(f, "general.architecture", "qwen3moe");
    w_kv_u32(f, "qwen3moe.block_count", 2);
    w_kv_u32(f, "qwen3moe.embedding_length", 16);
    w_kv_u32(f, "qwen3moe.attention.head_count", 2);
    w_kv_u32(f, "qwen3moe.attention.head_count_kv", 1);
    w_kv_u32(f, "qwen3moe.attention.key_length", 8);
    w_kv_u32(f, "qwen3moe.expert_count", 4);
    w_kv_u32(f, "qwen3moe.expert_used_count", 2);
    w_kv_u32(f, "qwen3moe.expert_feed_forward_length", 8);
    w_kv_f32(f, "qwen3moe.rope.freq_base", 1e7f);
    w_kv_u32(f, "general.alignment", 32);
    uint64_t e2[2] = { 16, 32 }, e1[1] = { 16 }, qn[1] = { 8 }, e3[3] = { 16, 8, 4 };
    w_tensor(f, "token_embd.weight", 2, e2, 1, 0);
    w_tensor(f, "output.weight", 2, e2, 1, 0);
    w_tensor(f, "output_norm.weight", 1, e1, 0, 0);
    w_tensor(f, "blk.0.attn_q_norm.weight", 1, qn, 0, 0);
    w_tensor(f, "blk.0.ffn_gate_inp.weight", 2, e2, 1, 0);
    w_tensor(f, "blk.0.ffn_gate_exps.weight", 3, e3, 1, 0);
    fclose(f);
}

/* GGUF con DATI: token_embd F16 [4,2] + output_norm F32 [4] + esperti Q8_0 [32,1,2]. */
static void write_test_gguf_data(const char* path) {
    FILE* f = fopen(path, "wb");
    CHECK(f != NULL);
    if (!f) return;
    const uint32_t kv_count = 10, tensor_count = 4;
    w_u32(f, 0x46554747u); w_u32(f, 3);
    w_u64(f, tensor_count); w_u64(f, kv_count);
    w_kv_str(f, "general.architecture", "qwen3moe");
    w_kv_u32(f, "qwen3moe.block_count", 1);
    w_kv_u32(f, "qwen3moe.embedding_length", 4);
    w_kv_u32(f, "qwen3moe.attention.head_count", 1);
    w_kv_u32(f, "qwen3moe.attention.head_count_kv", 1);
    w_kv_u32(f, "qwen3moe.attention.key_length", 4);
    w_kv_u32(f, "qwen3moe.expert_count", 2);
    w_kv_u32(f, "qwen3moe.expert_used_count", 1);
    w_kv_u32(f, "qwen3moe.expert_feed_forward_length", 4);
    w_kv_u32(f, "general.alignment", 32);
    uint64_t d_emb[2] = { 4, 2 }, d_norm[1] = { 4 }, d_exp[3] = { 32, 1, 2 };
    w_tensor(f, "token_embd.weight", 2, d_emb, LLMENGINE_MOE_GGML_F16, 0);
    w_tensor(f, "output.weight", 2, d_emb, LLMENGINE_MOE_GGML_F16, 16);
    w_tensor(f, "output_norm.weight", 1, d_norm, LLMENGINE_MOE_GGML_F32, 32);
    w_tensor(f, "blk.0.ffn_gate_exps.weight", 3, d_exp, LLMENGINE_MOE_GGML_Q8_0, 48);
    long p = ftell(f); long pad = (32 - (p % 32)) % 32;
    for (long i = 0; i < pad; ++i) fputc(0, f);
    for (int i = 0; i < 8; ++i) { uint16_t h = llmcodec_f16_from_f32((float)(i + 1)); fwrite(&h, 2, 1, f); }   /* 16 B */
    for (int i = 0; i < 8; ++i) { uint16_t h = llmcodec_f16_from_f32(0.0f); fwrite(&h, 2, 1, f); }             /* 16 B */
    float one = 1.0f; fwrite(&one, 4, 1, f); float z = 0.0f; for (int i = 1; i < 4; ++i) fwrite(&z, 4, 1, f);
    for (uint32_t blk = 0; blk < 2; ++blk) {                                                        /* 2 blocchi Q8_0 */
        uint16_t dh = llmcodec_f16_from_f32(blk == 0 ? 1.0f : 2.0f); fwrite(&dh, 2, 1, f);
        for (int j = 0; j < 32; ++j) { int8_t q = (int8_t)(j + 1); fwrite(&q, 1, 1, f); }
    }
    fclose(f);
}

int main(void) {
    /* ── role-map ── */
    CHECK(llmengine_moe_role_of_tensor("model.layers.0.self_attn.q_proj.weight") == LLMENGINE_MOE_ROLE_Q);
    CHECK(llmengine_moe_role_of_tensor("model.layers.0.self_attn.q_norm.weight") == LLMENGINE_MOE_ROLE_Q_NORM);
    CHECK(llmengine_moe_role_of_tensor("model.layers.0.mlp.gate.weight") == LLMENGINE_MOE_ROLE_ROUTER);
    CHECK(llmengine_moe_role_of_tensor("model.layers.0.mlp.switch_mlp.gate_proj.weight") == LLMENGINE_MOE_ROLE_EXPERT_GATE);
    CHECK(llmengine_moe_role_of_tensor("model.embed_tokens.weight") == LLMENGINE_MOE_ROLE_EMBED);
    CHECK(llmengine_moe_role_of_tensor("lm_head.weight") == LLMENGINE_MOE_ROLE_LM_HEAD);
    CHECK(llmengine_moe_role_of_tensor("model.norm.weight") == LLMENGINE_MOE_ROLE_FINAL_NORM);
    CHECK(llmengine_moe_role_of_tensor("blk.0.ffn_gate_exps.weight") == LLMENGINE_MOE_ROLE_EXPERT_GATE);
    CHECK(llmengine_moe_role_of_tensor("model.layers.0.mlp.experts.3.down_proj.weight") == LLMENGINE_MOE_ROLE_EXPERT_DOWN);

    /* ── encode/decode round-trip ── */
    LlmEngineMoeArchDesc d; memset(&d, 0, sizeof(d));
    d.arch_family = LLMENGINE_MOE_ARCH_QWEN3_MOE; d.hidden = 16; d.n_layers = 1;
    d.n_heads = 2; d.n_kv_heads = 1; d.head_dim = 8; d.n_experts = 2; d.topk = 1;
    d.inter = 8; d.vocab = 32; d.ctx = 512; d.eps = 1e-6f; d.rope_theta = 1e7f;
    d.flags = LLMENGINE_MOE_FLAG_QK_NORM | LLMENGINE_MOE_FLAG_MOE;
    uint8_t blob[LLMENGINE_MOE_GMAD_SIZE];
    llmengine_moe_arch_encode(&d, blob);
    LlmEngineMoeArchDesc r;
    CHECK(llmengine_moe_arch_decode(blob, sizeof(blob), &r) == 0);
    CHECK(r.arch_family == d.arch_family && r.hidden == 16 && r.n_layers == 1 && r.n_experts == 2);
    CHECK(r.head_dim == 8 && r.ctx == 512 && r.flags == d.flags);
    /* blob troncato/estraneo → errore */
    CHECK(llmengine_moe_arch_decode(blob, 10, &r) != 0);

    /* ── discovery su dir temporanea ── */
    char tmpl[] = "/tmp/llm-discover-XXXXXX";
    char* dir = mkdtemp(tmpl);
    CHECK(dir != NULL);
    if (dir == NULL) { fprintf(stderr, "test_moe_discover: %d FAIL\n", fails); return fails ? 1 : 0; }

    char p[1200];
    snprintf(p, sizeof(p), "%s/config.json", dir);
    write_file(p,
        "{\n"
        "  \"model_type\": \"qwen3_moe\",\n"
        "  \"hidden_size\": 16,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"head_dim\": 8,\n"
        "  \"num_experts\": 4,\n"
        "  \"num_experts_per_tok\": 2,\n"
        "  \"moe_intermediate_size\": 8,\n"
        "  \"vocab_size\": 32,\n"
        "  \"max_position_embeddings\": 4096,\n"
        "  \"rms_norm_eps\": 1e-06,\n"
        "  \"rope_theta\": 10000000.0\n"
        "}\n");
    snprintf(p, sizeof(p), "%s/model.safetensors.index.json", dir);
    write_file(p,
        "{ \"weight_map\": {\n"
        "  \"model.embed_tokens.weight\": \"a\",\n"
        "  \"model.norm.weight\": \"a\",\n"
        "  \"lm_head.weight\": \"a\",\n"
        "  \"model.layers.1.self_attn.q_proj.weight\": \"a\",\n"
        "  \"model.layers.1.self_attn.q_norm.weight\": \"a\",\n"
        "  \"model.layers.1.self_attn.k_norm.weight\": \"a\",\n"
        "  \"model.layers.1.mlp.gate.weight\": \"a\",\n"
        "  \"model.layers.1.mlp.switch_mlp.gate_proj.weight\": \"a\"\n"
        "} }\n");

    LlmEngineMoeDiscover res;
    CHECK(llmengine_moe_discover_hf(dir, &res) == LLMENGINE_MOE_OK);
    CHECK(res.desc.arch_family == LLMENGINE_MOE_ARCH_QWEN3_MOE);
    CHECK(res.desc.hidden == 16 && res.desc.n_layers == 2 && res.desc.head_dim == 8);
    CHECK(res.desc.n_experts == 4 && res.desc.topk == 2 && res.desc.inter == 8);
    CHECK(res.desc.ctx == 4096);
    CHECK((res.desc.flags & LLMENGINE_MOE_FLAG_QK_NORM) != 0);
    CHECK((res.desc.flags & LLMENGINE_MOE_FLAG_MOE) != 0);
    CHECK(res.experts_per_layer == 4);           /* stacked → da config */
    CHECK(res.n_tensors == 8);
    CHECK(res.role_counts[LLMENGINE_MOE_ROLE_Q] == 1);
    CHECK(res.role_counts[LLMENGINE_MOE_ROLE_EXPERT_GATE] == 1);
    CHECK(res.confidence > 50);

    char rep[4096];
    CHECK(llmengine_moe_discover_report(&res, rep, sizeof(rep)) > 0);
    CHECK(strstr(rep, "qwen3_moe") != NULL);

    /* ── override ── */
    CHECK(llmengine_moe_discover_override(&res.desc, "router_kind", "1") == LLMENGINE_MOE_OK);
    CHECK(res.desc.router_kind == LLMENGINE_MOE_ROUTER_SIGMOID);
    CHECK(llmengine_moe_discover_override(&res.desc, "boh", "1") != LLMENGINE_MOE_OK);

    /* ── M8: prompt + apply_reply (LLM locale, processo esterno) ── */
    {
        char pr[4096];
        CHECK(llmengine_moe_discover_prompt(&res, pr, sizeof(pr)) > 0);
        CHECK(strstr(pr, "rope_variant") != NULL && strstr(pr, "router_kind") != NULL);
        LlmEngineMoeArchDesc ar; memset(&ar, 0, sizeof(ar)); ar.arch_family = LLMENGINE_MOE_ARCH_UNKNOWN;
        CHECK(llmengine_moe_discover_apply_reply(&ar,
              "bla {\"arch_family\":\"glm_moe\",\"rope_variant\":\"interleaved\",\"router_kind\":1,"
              "\"norm_kind\":\"layernorm\",\"bogus\":9} tail") == LLMENGINE_MOE_OK);
        CHECK(ar.arch_family == LLMENGINE_MOE_ARCH_GLM_MOE);
        CHECK(ar.rope_variant == LLMENGINE_MOE_ROPE_INTERLEAVED);
        CHECK(ar.router_kind == LLMENGINE_MOE_ROUTER_SIGMOID);
        CHECK(ar.norm_kind == LLMENGINE_MOE_NORM_LAYER);
        CHECK(llmengine_moe_discover_apply_reply(&ar, "nessun campo valido") != LLMENGINE_MOE_OK);
    }

    /* ── discovery da GGUF ── */
    char gp[1200];
    snprintf(gp, sizeof(gp), "%s/t.gguf", dir);
    write_test_gguf(gp);
    LlmEngineMoeDiscover gr;
    CHECK(llmengine_moe_discover_gguf(gp, &gr) == LLMENGINE_MOE_OK);
    CHECK(gr.source_gguf == 1);
    CHECK(gr.desc.arch_family == LLMENGINE_MOE_ARCH_QWEN3_MOE);
    CHECK(gr.desc.hidden == 16 && gr.desc.n_layers == 2 && gr.desc.head_dim == 8);
    CHECK(gr.desc.n_experts == 4 && gr.desc.topk == 2 && gr.desc.inter == 8);
    CHECK(gr.desc.vocab == 32);                      /* da token_embd dims[1] */
    CHECK(gr.experts_per_layer == 4);                /* da ffn_gate_exps dims[2] */
    CHECK((gr.desc.flags & LLMENGINE_MOE_FLAG_QK_NORM) != 0);
    CHECK((gr.desc.flags & LLMENGINE_MOE_FLAG_MOE) != 0);
    remove(gp);

    /* ── dequant ggml (unit) ── */
    {
        uint8_t q8[34]; uint16_t dh = llmcodec_f16_from_f32(1.0f); memcpy(q8, &dh, 2);
        for (int j = 0; j < 32; ++j) q8[2 + j] = (uint8_t)(int8_t)(j - 2);
        float o8[32];
        CHECK(llmengine_moe_ggml_dequant_row(LLMENGINE_MOE_GGML_Q8_0, q8, o8, 32) == 0);
        CHECK(o8[0] == -2.0f && o8[31] == 29.0f);
        CHECK(llmengine_moe_ggml_blck_size(LLMENGINE_MOE_GGML_Q8_0) == 32 && llmengine_moe_ggml_type_size(LLMENGINE_MOE_GGML_Q8_0) == 34);

        uint8_t q4[18]; uint16_t d4 = llmcodec_f16_from_f32(0.5f); memcpy(q4, &d4, 2);
        for (int j = 0; j < 16; ++j) q4[2 + j] = 0x88;
        float o4[32];
        CHECK(llmengine_moe_ggml_dequant_row(LLMENGINE_MOE_GGML_Q4_0, q4, o4, 32) == 0);
        CHECK(o4[0] == 0.0f && o4[16] == 0.0f);
        q4[2] = 0x99;
        CHECK(llmengine_moe_ggml_dequant_row(LLMENGINE_MOE_GGML_Q4_0, q4, o4, 32) == 0);
        CHECK(o4[0] == 0.5f && o4[16] == 0.5f);
        CHECK(llmengine_moe_ggml_dequant_row(LLMENGINE_MOE_GGML_Q2_K, q4, o4, 2) < 0);   /* non implementato */
        CHECK(llmengine_moe_ggml_type_size(LLMENGINE_MOE_GGML_Q4_K) == 144);

        /* Q4_K: d=1, dmin=0, scales tutti a sc=1/m=0, qs nibble lo=1 hi=2 →
         * y[k] = 1 se (k%64)<32 altrimenti 2. */
        uint8_t q4k[144]; memset(q4k, 0, sizeof(q4k));
        uint16_t dk1 = llmcodec_f16_from_f32(1.0f), dk0 = llmcodec_f16_from_f32(0.0f);
        memcpy(q4k, &dk1, 2); memcpy(q4k + 2, &dk0, 2);
        uint8_t sc4[12] = { 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
        memcpy(q4k + 4, sc4, 12);
        for (int l = 0; l < 128; ++l) q4k[16 + l] = 0x21;
        float yk[256];
        CHECK(llmengine_moe_ggml_dequant_row(LLMENGINE_MOE_GGML_Q4_K, q4k, yk, 256) == 0);
        int ok4k = 1;
        for (int k = 0; k < 256; ++k) if (fabsf(yk[k] - (((k % 64) < 32) ? 1.0f : 2.0f)) > 1e-6f) ok4k = 0;
        CHECK(ok4k);
        CHECK(llmengine_moe_ggml_blck_size(LLMENGINE_MOE_GGML_Q4_K) == 256 && llmengine_moe_ggml_type_size(LLMENGINE_MOE_GGML_Q4_K) == 144);

        /* Q6_K: ql=qh=0 → q=-32; scales=1; d=1 → y = -32 ovunque. */
        uint8_t q6k[210]; memset(q6k, 0, sizeof(q6k));
        memset(q6k + 192, 1, 16);
        uint16_t d6 = llmcodec_f16_from_f32(1.0f); memcpy(q6k + 208, &d6, 2);
        float y6[256];
        CHECK(llmengine_moe_ggml_dequant_row(LLMENGINE_MOE_GGML_Q6_K, q6k, y6, 256) == 0);
        int ok6k = 1;
        for (int k = 0; k < 256; ++k) if (fabsf(y6[k] + 32.0f) > 1e-6f) ok6k = 0;
        CHECK(ok6k);
        CHECK(llmengine_moe_ggml_type_size(LLMENGINE_MOE_GGML_Q6_K) == 210);
    }

    /* ── accesso dati GGUF (F16 + esperto Q8_0) ── */
    {
        char dp[1200]; snprintf(dp, sizeof(dp), "%s/d.gguf", dir);
        write_test_gguf_data(dp);
        LlmEngineMoeGguf g;
        CHECK(llmengine_moe_gguf_read(dp, &g) == LLMENGINE_MOE_OK);
        FILE* f = fopen(dp, "rb");
        CHECK(f != NULL);
        const LlmCodecGgufTensor* te = llmengine_moe_gguf_tensor(&g, "token_embd.weight");
        const LlmCodecGgufTensor* tx = llmengine_moe_gguf_tensor(&g, "blk.0.ffn_gate_exps.weight");
        CHECK(te != NULL && tx != NULL);
        float emb[8];
        CHECK(llmengine_moe_gguf_tensor_f32(f, &g, te, emb, 8) == LLMENGINE_MOE_OK);
        CHECK(emb[0] == 1.0f && emb[7] == 8.0f);
        float e0[32], e1[32];
        CHECK(llmengine_moe_gguf_expert_f32(f, &g, tx, 0, e0, 32) == LLMENGINE_MOE_OK);
        CHECK(llmengine_moe_gguf_expert_f32(f, &g, tx, 1, e1, 32) == LLMENGINE_MOE_OK);
        CHECK(e0[0] == 1.0f && e0[31] == 32.0f);     /* d=1 * q */
        CHECK(e1[0] == 2.0f && e1[31] == 64.0f);     /* d=2 * q */
        fclose(f);
        llmengine_moe_gguf_free(&g);
        remove(dp);
    }

    char rm[1200];
    snprintf(rm, sizeof(rm), "%s/config.json", dir); remove(rm);
    snprintf(rm, sizeof(rm), "%s/model.safetensors.index.json", dir); remove(rm);
    rmdir(dir);

    if (fails == 0) printf("test_moe_discover: OK\n");
    else fprintf(stderr, "test_moe_discover: %d FAIL\n", fails);
    return fails ? 1 : 0;
}
