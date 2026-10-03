/* test_moe_gguf_store.c — store esperti GGUF in stile DS4 (llmengine_moe_gguf_store.h):
 *   - RESIDENT: puntatori dentro il mmap del file (zero-copy);
 *   - SWAP:     bundle gate+up+down via pread + cache LRU (hit/miss/eviction/metriche).
 * Verifica parità bit-exact tra RESIDENT, SWAP e il dequant-on-read via FILE*, su un GGUF MoE
 * F16 minimo (stessa fixture di test_moe_gguf_adapter). */
#include "llmengine_moe_gguf.h"
#include "llmengine_moe_gguf_store.h"
#include "llmengine_moe_gguf_expert.h"
#include "llmengine_moe_ggml.h"
#include "llmengine_moe.h"
#include "llmengine_moe_adapter_gguf.h"
#include "llmcodec_fp16.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); fails++; } } while (0)

static void w_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w_str(FILE* f, const char* s) { uint64_t n = strlen(s); w_u64(f, n); fwrite(s, 1, n, f); }
static void w_kv_u32(FILE* f, const char* k, uint32_t v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_UINT32); w_u32(f, v); }
static void w_kv_str(FILE* f, const char* k, const char* v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_STRING); w_str(f, v); }

typedef struct { const char* name; uint32_t nd; uint64_t dims[4]; uint32_t type; } TDef;

/* hidden=4, `inter`/`experts` parametrici (2 layer) — rampe diverse per layer per le chiavi.
 * `inter` grande serve al test D19: bundle grande → cap dello store piccolo → eviction. */
static void write_tiny_moe_gguf_ex(const char* path, uint32_t experts, uint32_t inter) {
    TDef T[32]; uint32_t NT = 0;
    const TDef base[] = {
        { "token_embd.weight",            2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "output.weight",                2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "output_norm.weight",           1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.0.attn_norm.weight",       1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.0.ffn_norm.weight",        1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.0.attn_q.weight",          2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.attn_k.weight",          2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.attn_v.weight",          2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.attn_output.weight",     2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.attn_q_norm.weight",     1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.0.attn_k_norm.weight",     1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.0.ffn_gate_inp.weight",    2, {4,2,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.ffn_gate_exps.weight",   3, {4,2,2,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.ffn_up_exps.weight",     3, {4,2,2,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.0.ffn_down_exps.weight",   3, {2,4,2,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.attn_norm.weight",       1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.1.ffn_norm.weight",        1, {4,0,0,0}, LLMENGINE_MOE_GGML_F32 },
        { "blk.1.attn_q.weight",          2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.attn_k.weight",          2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.attn_v.weight",          2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.attn_output.weight",     2, {4,4,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.ffn_gate_inp.weight",    2, {4,2,0,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.ffn_gate_exps.weight",   3, {4,2,2,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.ffn_up_exps.weight",     3, {4,2,2,0}, LLMENGINE_MOE_GGML_F16 },
        { "blk.1.ffn_down_exps.weight",   3, {2,4,2,0}, LLMENGINE_MOE_GGML_F16 },
    };
    for (uint32_t i = 0; i < (uint32_t)(sizeof(base) / sizeof(base[0])); ++i) T[NT++] = base[i];
    /* D19: esperti/inter parametrici (router [hidden, experts]; gate/up [hidden, inter, experts];
     * down [inter, hidden, experts]). */
    for (uint32_t i = 0; i < NT; ++i) {
        if (strstr(T[i].name, "ffn_gate_inp")) T[i].dims[1] = experts;
        else if (strstr(T[i].name, "ffn_gate_exps") || strstr(T[i].name, "ffn_up_exps")) {
            T[i].dims[1] = inter; T[i].dims[2] = experts;
        } else if (strstr(T[i].name, "ffn_down_exps")) {
            T[i].dims[0] = inter; T[i].dims[2] = experts;
        }
    }
    FILE* f = fopen(path, "wb");
    CHECK(f != NULL); if (!f) return;
    w_u32(f, 0x46554747u); w_u32(f, 3);
    w_u64(f, NT); w_u64(f, 10);
    w_kv_str(f, "general.architecture", "qwen3moe");
    w_kv_u32(f, "qwen3moe.block_count", 2);
    w_kv_u32(f, "qwen3moe.embedding_length", 4);
    w_kv_u32(f, "qwen3moe.attention.head_count", 1);
    w_kv_u32(f, "qwen3moe.attention.head_count_kv", 1);
    w_kv_u32(f, "qwen3moe.attention.key_length", 4);
    w_kv_u32(f, "qwen3moe.expert_count", experts);
    w_kv_u32(f, "qwen3moe.expert_used_count", 1);
    w_kv_u32(f, "qwen3moe.expert_feed_forward_length", inter);
    w_kv_u32(f, "general.alignment", 32);
    uint64_t off = 0;
    for (uint32_t i = 0; i < NT; ++i) {
        uint64_t ne = 1; for (uint32_t d = 0; d < T[i].nd; ++d) ne *= T[i].dims[d];
        w_str(f, T[i].name); w_u32(f, T[i].nd);
        for (uint32_t d = 0; d < T[i].nd; ++d) w_u64(f, T[i].dims[d]);
        w_u32(f, T[i].type); w_u64(f, off);
        off += ne * (T[i].type == LLMENGINE_MOE_GGML_F32 ? 4 : 2);
    }
    long p = ftell(f); long pad = (32 - (p % 32)) % 32;
    for (long i = 0; i < pad; ++i) fputc(0, f);
    for (uint32_t i = 0; i < NT; ++i) {
        uint64_t ne = 1; for (uint32_t d = 0; d < T[i].nd; ++d) ne *= T[i].dims[d];
        for (uint64_t k = 0; k < ne; ++k) {
            float val = 1.0f;
            if (strstr(T[i].name, "blk.0.ffn_gate_exps")) val = 0.5f * (float)(k + 1);
            else if (strstr(T[i].name, "blk.0.ffn_up_exps")) val = 0.25f * (float)(k + 1);
            else if (strstr(T[i].name, "blk.1.ffn_gate_exps")) val = 0.125f * (float)(k + 1);
            else if (strstr(T[i].name, "blk.1.ffn_up_exps")) val = 0.75f * (float)(k + 1);
            else if (strstr(T[i].name, "ffn_down_exps")) {
                static const float pat[8] = {1,0,0,1,1,0,0,1}; val = pat[k % 8];
            }
            if (T[i].type == LLMENGINE_MOE_GGML_F32) { fwrite(&val, 4, 1, f); }
            else { uint16_t h = llmcodec_f16_from_f32(val); fwrite(&h, 2, 1, f); }
        }
    }
    fclose(f);
}

static void write_tiny_moe_gguf(const char* path) { write_tiny_moe_gguf_ex(path, 2, 2); }

int main(void) {
    char tmpl[] = "/tmp/gguf-store-XXXXXX";
    char* dir = mkdtemp(tmpl);
    CHECK(dir != NULL);
    if (!dir) return 1;
    char gp[1200]; snprintf(gp, sizeof(gp), "%s/tiny.gguf", dir);
    write_tiny_moe_gguf(gp);

    LlmEngineMoeGguf g;
    CHECK(llmengine_moe_gguf_read(gp, &g) == LLMENGINE_MOE_OK);
    CHECK(g.n_tensors > 0);

    GgufStore* rs = llmengine_moe_gguf_store_open(gp, &g, 2, 0 /*RESIDENT*/, 0);
    GgufStore* ss = llmengine_moe_gguf_store_open(gp, &g, 2, 1 /*SWAP*/,    0);
    CHECK(rs != NULL && ss != NULL);
    if (!rs || !ss) return 1;

    FILE* f = fopen(gp, "rb"); CHECK(f != NULL);

    float x[4] = { 1, 0.5f, -1, 2 };
    for (uint32_t layer = 0; layer < 2; ++layer) {
        for (uint32_t e = 0; e < 2; ++e) {
            const uint8_t *ur, *dr, *us, *ds;
            const uint8_t* gr = llmengine_moe_gguf_store_expert(rs, layer, e, &ur, &dr);
            const uint8_t* gs = llmengine_moe_gguf_store_expert(ss, layer, e, &us, &ds);
            CHECK(gr && gs);
            if (!gr || !gs) continue;

            /* RESIDENT: puntatori DENTRO il mmap (zero-copy) */
            GgufStoreLayer* L = &rs->layers[layer];
            CHECK(gr == rs->base + L->g.off + (uint64_t)e * L->g.nb);
            CHECK(ur == rs->base + L->u.off + (uint64_t)e * L->u.nb);
            CHECK(dr == rs->base + L->d.off + (uint64_t)e * L->d.nb);

            /* SWAP vs RESIDENT: byte identici (bundle = pread, RESIDENT = mmap) */
            CHECK(L->g.nb == ss->layers[layer].g.nb);
            CHECK(memcmp(gr, gs, L->g.nb) == 0);
            CHECK(memcmp(ur, us, L->u.nb) == 0);
            CHECK(memcmp(dr, ds, L->d.nb) == 0);

            /* swiglu Mem == swiglu FILE (fallback dequant-on-read) */
            char nm[128];
            snprintf(nm, sizeof(nm), "blk.%u.ffn_gate_exps.weight", layer);
            const LlmCodecGgufTensor* gate = llmengine_moe_gguf_tensor(&g, nm);
            snprintf(nm, sizeof(nm), "blk.%u.ffn_up_exps.weight", layer);
            const LlmCodecGgufTensor* up = llmengine_moe_gguf_tensor(&g, nm);
            snprintf(nm, sizeof(nm), "blk.%u.ffn_down_exps.weight", layer);
            const LlmCodecGgufTensor* down = llmengine_moe_gguf_tensor(&g, nm);
            float of[4], om[4];
            CHECK(llmengine_moe_gguf_expert_swiglu(f, &g, gate, up, down, e, 4, 2, x, of) == LLMENGINE_MOE_OK);
            CHECK(llmengine_moe_gguf_expert_swiglu_mem(gate->type, up->type, down->type,
                                                 gr, ur, dr, 4, 2, x, om) == LLMENGINE_MOE_OK);
            for (int h = 0; h < 4; ++h) CHECK(fabs((double)of[h] - (double)om[h]) < 1e-6);
        }
    }
    /* metriche SWAP: 4 esperimenti distinti → 4 miss, 0 hit; poi ripetizione → hit */
    CHECK(ss->misses == 4);
    CHECK(ss->bytes_read > 0);
    { const uint8_t *u, *d; (void)llmengine_moe_gguf_store_expert(ss, 0, 0, &u, &d); }
    CHECK(ss->hits == 1);
    CHECK(llmengine_moe_gguf_store_resident_bytes(ss) > 0);

    fclose(f);
    llmengine_moe_gguf_store_close(rs);
    llmengine_moe_gguf_store_close(ss);
    llmengine_moe_gguf_free(&g);

    /* ── integrazione runtime: SWAP → lo store dell'adapter alimenta AI.STREAMSTATS ── */
    LlmEngineMoeRuntime* rt = llmengine_moe_create();
    CHECK(rt != NULL);
    if (rt != NULL) {
        LlmEngineMoeModelAdapter ad = llmengine_moe_gguf_adapter();
        CHECK(llmengine_moe_register_adapter(rt, &ad) == LLMENGINE_MOE_OK);
        LlmEngineMoeRuntimeConfig cfg; memset(&cfg, 0, sizeof(cfg));
        cfg.mode = LLMENGINE_MOE_MODE_SWAP; cfg.device = LLMENGINE_MOE_DEVICE_CPU; cfg.cache_mb = 1;
        CHECK(llmengine_moe_load_model_ex(rt, gp, &cfg) == LLMENGINE_MOE_OK);
        float in1[1] = { 0 }, out1[4] = { 0 };
        CHECK(llmengine_moe_forward(rt, in1, 1, out1) == LLMENGINE_MOE_OK);
        LlmEngineMoeStreamStats st;
        CHECK(llmengine_moe_stream_stats(rt, &st) == LLMENGINE_MOE_OK);
        CHECK(st.mode == LLMENGINE_MOE_MODE_SWAP);
        CHECK(st.cache_misses >= 2);       /* 2 layer × 1 esperto top-1 */
        CHECK(st.bytes_read > 0);          /* lo store SWAP ha letto via pread */
        CHECK(llmengine_moe_cache_mb(rt) == 1);
        llmengine_moe_destroy(rt);
    }

    /* ── D19: policy hotness sul cache REALE (oracolo deterministico LRU vs freq) ──
     * Modello 10 esperti con bundle grande → cap dello store = 8 (clamp) → eviction. */
    {
        char gp10[1200]; snprintf(gp10, sizeof(gp10), "%s/tiny10.gguf", dir);
        write_tiny_moe_gguf_ex(gp10, 10, 8192);
        LlmEngineMoeGguf g10;
        CHECK(llmengine_moe_gguf_read(gp10, &g10) == LLMENGINE_MOE_OK);

        /* OFF (LRU): riempie e0..e7, l'accesso a e8 evince il PIÙ VECCHIO (e0) → e0 = MISS */
        GgufStore* so = llmengine_moe_gguf_store_open(gp10, &g10, 1, 1, 1);
        CHECK(so != NULL);
        if (so != NULL) {
            llmengine_moe_gguf_store_set_dims(so, 1, 10);
            const uint8_t *u, *d;
            for (uint32_t e = 0; e < 8; ++e)
                CHECK(llmengine_moe_gguf_store_expert(so, 0, e, &u, &d) != NULL);
            CHECK(llmengine_moe_gguf_store_expert(so, 0, 8, &u, &d) != NULL);
            uint64_t m0 = so->misses;
            CHECK(llmengine_moe_gguf_store_expert(so, 0, 0, &u, &d) != NULL);
            CHECK(so->misses == m0 + 1);   /* e0 era stato evitto dall'LRU */
            llmengine_moe_gguf_store_close(so);
        }

        /* ON (hotness): freq(e0)=1000 → l'eviction sceglie un esperto FREDDO (e1), e0 resta → HIT */
        GgufStore* sh = llmengine_moe_gguf_store_open(gp10, &g10, 1, 1, 1);
        CHECK(sh != NULL);
        if (sh != NULL) {
            llmengine_moe_gguf_store_set_dims(sh, 1, 10);
            llmengine_moe_gguf_store_set_hotness(sh, 1);
            uint32_t hot[1] = { 0 };
            for (int i = 0; i < 1000; ++i) llmengine_moe_gguf_store_note_route(sh, 0, hot, 1);
            CHECK(llmengine_moe_gguf_store_freq(sh, 0, 0) == 1000);
            const uint8_t *u, *d;
            for (uint32_t e = 0; e < 8; ++e)
                CHECK(llmengine_moe_gguf_store_expert(sh, 0, e, &u, &d) != NULL);
            CHECK(llmengine_moe_gguf_store_expert(sh, 0, 8, &u, &d) != NULL);   /* evince un freddo */
            uint64_t h0 = sh->hits;
            CHECK(llmengine_moe_gguf_store_expert(sh, 0, 0, &u, &d) != NULL);
            CHECK(sh->hits == h0 + 1);   /* e0 (caldo) è sopravvissuto */
            /* profilo GMSTATS 1: save → load su uno store nuovo (round-trip freq) */
            char pp[1200]; snprintf(pp, sizeof(pp), "%s.stats", gp10);
            CHECK(llmengine_moe_gguf_store_profile_save(sh, pp) == 0);
            GgufStore* s2 = llmengine_moe_gguf_store_open(gp10, &g10, 1, 1, 1);
            CHECK(s2 != NULL);
            if (s2 != NULL) {
                llmengine_moe_gguf_store_set_dims(s2, 1, 10);
                CHECK(llmengine_moe_gguf_store_profile_load(s2, pp) == 0);
                CHECK(llmengine_moe_gguf_store_freq(s2, 0, 0) == 1000);
                llmengine_moe_gguf_store_close(s2);
            }
            remove(pp);
            llmengine_moe_gguf_store_close(sh);
        }
        llmengine_moe_gguf_free(&g10);
        remove(gp10);
    }

    if (fails == 0) { printf("test_moe_gguf_store: OK\n"); return 0; }
    fprintf(stderr, "test_moe_gguf_store: %d FAIL\n", fails);
    return 1;
}
