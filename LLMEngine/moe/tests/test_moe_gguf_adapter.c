/* test_moe_gguf_adapter.c — ingest diretto GGUF: esperto SwiGLU (oracle a mano) + forward adapter.
 * Costruisce un GGUF MoE F16 minimo con pesi tutti a 1.0, così l'output dell'esperto è calcolabile
 * analiticamente, e verifica che l'adapter GGUF carichi e produca logits deterministici. */
#include "llmengine_moe_adapter_gguf.h"
#include "llmengine_moe_gguf.h"
#include "llmengine_moe_gguf_expert.h"
#include "llmengine_moe_ggml.h"
#include "llmengine_moe_provider.h"
#include "llmengine_moe_prof.h"
#include "llmengine_moe_sim.h"
#include "llmcodec_fp16.h"   /* f32->f16 = atomo del CODEC */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); fails++; } } while (0)



static void w_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w_str(FILE* f, const char* s) { uint64_t n = strlen(s); w_u64(f, n); fwrite(s, 1, n, f); }
static void w_kv_u32(FILE* f, const char* k, uint32_t v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_UINT32); w_u32(f, v); }
static void w_kv_str(FILE* f, const char* k, const char* v) { w_str(f, k); w_u32(f, LLMCODEC_GGUF_STRING); w_str(f, v); }

typedef struct { const char* name; uint32_t nd; uint64_t dims[4]; uint32_t type; } TDef;

/* K5: variante con metadata/tensor volutamente INCOERENTI (`meta_experts` > esperti stacked) e
 * router forzato su `high_expert` (>=0): serve al test negativo "esperto non risolvibile". */
static void write_tiny_moe_gguf_ex(const char* path, int with_shared, uint32_t meta_experts, int high_expert) {
    TDef T[18];
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
    };
    uint32_t NT = 0;
    for (uint32_t i = 0; i < (uint32_t)(sizeof(base) / sizeof(base[0])); ++i) T[NT++] = base[i];
    /* K5: il router ha `meta_experts` righe (coi metadata), i tensor stacked restano a 2. */
    for (uint32_t i = 0; i < NT; ++i)
        if (strcmp(T[i].name, "blk.0.ffn_gate_inp.weight") == 0) T[i].dims[1] = meta_experts;
    if (with_shared) {   /* M7.1: shared-expert 2D (n_ff_shexp=3) */
        T[NT++] = (TDef){ "blk.0.ffn_gate_shexp.weight", 2, {4,3,0,0}, LLMENGINE_MOE_GGML_F16 };
        T[NT++] = (TDef){ "blk.0.ffn_up_shexp.weight",   2, {4,3,0,0}, LLMENGINE_MOE_GGML_F16 };
        T[NT++] = (TDef){ "blk.0.ffn_down_shexp.weight", 2, {3,4,0,0}, LLMENGINE_MOE_GGML_F16 };
    }
    FILE* f = fopen(path, "wb");
    CHECK(f != NULL);
    if (!f) return;
    w_u32(f, 0x46554747u); w_u32(f, 3);
    w_u64(f, NT); w_u64(f, 10);
    w_kv_str(f, "general.architecture", "qwen3moe");
    w_kv_u32(f, "qwen3moe.block_count", 1);
    w_kv_u32(f, "qwen3moe.embedding_length", 4);
    w_kv_u32(f, "qwen3moe.attention.head_count", 1);
    w_kv_u32(f, "qwen3moe.attention.head_count_kv", 1);
    w_kv_u32(f, "qwen3moe.attention.key_length", 4);
    w_kv_u32(f, "qwen3moe.expert_count", meta_experts);
    w_kv_u32(f, "qwen3moe.expert_used_count", 1);
    w_kv_u32(f, "qwen3moe.expert_feed_forward_length", 2);
    w_kv_u32(f, "general.alignment", 32);
    /* offsets: bytes di ogni tensore (F16 = 2 B/elem, F32 = 4 B/elem) */
    uint64_t off = 0;
    for (uint32_t i = 0; i < NT; ++i) {
        uint64_t ne = 1; for (uint32_t d = 0; d < T[i].nd; ++d) ne *= T[i].dims[d];
        uint64_t bytes = ne * (T[i].type == LLMENGINE_MOE_GGML_F32 ? 4 : 2);
        w_str(f, T[i].name); w_u32(f, T[i].nd);
        for (uint32_t d = 0; d < T[i].nd; ++d) w_u64(f, T[i].dims[d]);
        w_u32(f, T[i].type); w_u64(f, off);
        off += bytes;
    }
    long p = ftell(f); long pad = (32 - (p % 32)) % 32;
    for (long i = 0; i < pad; ++i) fputc(0, f);
    for (uint32_t i = 0; i < NT; ++i) {
        uint64_t ne = 1; for (uint32_t d = 0; d < T[i].nd; ++d) ne *= T[i].dims[d];
        for (uint64_t k = 0; k < ne; ++k) {
            float val = 1.0f;
            if (strstr(T[i].name, "ffn_gate_exps")) val = 0.5f * (float)(k + 1);   /* ramp: layout-sensitive */
            else if (strstr(T[i].name, "ffn_gate_inp")) {   /* K5: riga `high_expert` domina */
                uint64_t ne0 = T[i].dims[0] ? T[i].dims[0] : 1;
                uint32_t e = (uint32_t)(k / ne0);
                val = (high_expert >= 0 && e == (uint32_t)high_expert) ? 100.0f : 1.0f;
            }
            else if (strstr(T[i].name, "ffn_down_exps")) { static const float pat[8] = {1,0,0,1,1,0,0,1}; val = pat[k % 8]; }
            else if (strstr(T[i].name, "ffn_gate_shexp")) val = 0.25f * (float)(k + 1);   /* ramp: shared applicato */
            if (T[i].type == LLMENGINE_MOE_GGML_F32) fwrite(&val, 4, 1, f);
            else { uint16_t h = llmcodec_f16_from_f32(val); fwrite(&h, 2, 1, f); }
        }
    }
    fclose(f);
}

static void write_tiny_moe_gguf(const char* path, int with_shared) {
    write_tiny_moe_gguf_ex(path, with_shared, 2, -1);
}

int main(void) {
    char tmpl[] = "/tmp/gguf-adapter-XXXXXX";
    char* dir = mkdtemp(tmpl);
    CHECK(dir != NULL);
    if (!dir) { fprintf(stderr, "test_moe_gguf_adapter: %d FAIL\n", fails); return 1; }
    char gp[1200]; snprintf(gp, sizeof(gp), "%s/tiny.gguf", dir);
    write_tiny_moe_gguf(gp, 0);

    LlmEngineMoeGguf g;
    CHECK(llmengine_moe_gguf_read(gp, &g) == LLMENGINE_MOE_OK);
    FILE* f = fopen(gp, "rb");
    CHECK(f != NULL);
    const LlmCodecGgufTensor* tg = llmengine_moe_gguf_tensor(&g, "blk.0.ffn_gate_exps.weight");
    const LlmCodecGgufTensor* tu = llmengine_moe_gguf_tensor(&g, "blk.0.ffn_up_exps.weight");
    const LlmCodecGgufTensor* td = llmengine_moe_gguf_tensor(&g, "blk.0.ffn_down_exps.weight");
    CHECK(tg && tu && td);

    /* ── esperto SwiGLU layout-sensitive (hidden=4, inter=2 non quadrato) ──
     * gate_exps flat = 0.5*(k+1): expert0 righe [0.5,1,1.5,2],[2.5,3,3.5,4]; x=ones → gate·x=[5,13].
     * up_exps = 1 → up·x=[4,4]. down_exps pattern [1,0,0,1,1,0,0,1] → out=[t0,t1,t0,t1]. */
    float x[4] = {1, 1, 1, 1}, eo[4];
    CHECK(llmengine_moe_gguf_expert_swiglu(f, &g, tg, tu, td, 0, 4, 2, x, eo) == LLMENGINE_MOE_OK);
    double t0 = (5.0 / (1.0 + exp(-5.0))) * 4.0;
    double t1 = (13.0 / (1.0 + exp(-13.0))) * 4.0;
    double exp_eo[4] = { t0, t1, t0, t1 };
    for (int h = 0; h < 4; ++h) CHECK(fabs((double)eo[h] - exp_eo[h]) < 1e-3);
    /* ── esperto BATCH == esperto singolo (dequant una volta, n posizioni) ── */
    float xb[3][4] = { {1,1,1,1}, {0.5f,0.25f,1,2}, {2,0,1,1} };
    float os[4], ob[3][4];
    CHECK(llmengine_moe_gguf_expert_swiglu(f, &g, tg, tu, td, 0, 4, 2, xb[0], os) == LLMENGINE_MOE_OK);
    CHECK(llmengine_moe_gguf_expert_swiglu_batch(f, &g, tg, tu, td, 0, 4, 2, &xb[0][0], 3, &ob[0][0]) == LLMENGINE_MOE_OK);
    for (int i = 0; i < 3; ++i) {
        CHECK(llmengine_moe_gguf_expert_swiglu(f, &g, tg, tu, td, 0, 4, 2, xb[i], os) == LLMENGINE_MOE_OK);
        for (int h = 0; h < 4; ++h) CHECK(fabs((double)ob[i][h] - (double)os[h]) < 1e-6);
    }
    fclose(f);
    llmengine_moe_gguf_free(&g);

    /* ── adapter GGUF: load + forward deterministico ── */
    LlmEngineMoeModelAdapter a1 = llmengine_moe_gguf_adapter();
    LlmEngineMoeModelAdapter a2 = llmengine_moe_gguf_adapter();
    CHECK(a1.self != NULL && a2.self != NULL);
    CHECK(a1.load(a1.self, gp, NULL) == LLMENGINE_MOE_OK);
    CHECK(a2.load(a2.self, gp, NULL) == LLMENGINE_MOE_OK);
    float in_ids[1] = {0.0f};
    float o1[4] = {0}, o2[4] = {0};
    CHECK(a1.forward(a1.self, NULL, in_ids, 1, 4, o1) == LLMENGINE_MOE_OK);
    CHECK(a2.forward(a2.self, NULL, in_ids, 1, 4, o2) == LLMENGINE_MOE_OK);
    int finite = 1, same = 1;
    for (int i = 0; i < 4; ++i) {
        if (!isfinite(o1[i])) finite = 0;
        if (fabs((double)o1[i] - (double)o2[i]) > 1e-5) same = 0;
    }
    CHECK(finite);
    CHECK(same);

    /* ── M7.1 shared-expert: con l'esperto condiviso il forward CAMBIA (viene applicato) ── */
    char gs[1200]; snprintf(gs, sizeof(gs), "%s/tiny_shared.gguf", dir);
    write_tiny_moe_gguf(gs, 1);
    LlmEngineMoeModelAdapter bs = llmengine_moe_gguf_adapter();
    CHECK(bs.self != NULL && bs.load(bs.self, gs, NULL) == LLMENGINE_MOE_OK);
    float osh[4] = {0};
    CHECK(bs.forward(bs.self, NULL, in_ids, 1, 4, osh) == LLMENGINE_MOE_OK);
    int shared_differs = 0;
    for (int i = 0; i < 4; ++i) if (fabs((double)osh[i] - (double)o1[i]) > 1e-4) shared_differs = 1;
    CHECK(shared_differs);   /* o1 = modello SENZA shared, stesso input/pos0 */
    bs.destroy(bs.self);

    /* ── prefill BATCHED (seq=3) == decode incrementale (3× seq=1) ── */
    float in3[3] = {0, 1, 2}, obatch[4] = {0}, oinc[4] = {0}, o1t[4] = {0};
    LlmEngineMoeModelAdapter a3 = llmengine_moe_gguf_adapter();
    LlmEngineMoeModelAdapter a4 = llmengine_moe_gguf_adapter();
    CHECK(a3.self != NULL && a4.self != NULL);
    CHECK(a3.load(a3.self, gp, NULL) == LLMENGINE_MOE_OK);
    CHECK(a4.load(a4.self, gp, NULL) == LLMENGINE_MOE_OK);
    CHECK(a3.forward(a3.self, NULL, in3, 3, 4, obatch) == LLMENGINE_MOE_OK);
    for (int i = 0; i < 3; ++i) {
        float one[1] = { in3[i] };
        CHECK(a4.forward(a4.self, NULL, one, 1, 4, o1t) == LLMENGINE_MOE_OK);
    }
    memcpy(oinc, o1t, sizeof(oinc));
    int prefill_same = 1;
    for (int i = 0; i < 4; ++i) if (fabs((double)obatch[i] - (double)oinc[i]) > 1e-4) prefill_same = 0;
    CHECK(prefill_same);
    a3.destroy(a3.self);
    a4.destroy(a4.self);
    a1.destroy(a1.self);
    a2.destroy(a2.self);

    /* ── K1/K2: trace di PRODUZIONE dallo store GGUF + replay offline (gate `moe_sim --verify`) ──
     * Il trace deve venire dal cache REALE del path SWAP (GgufStore), coi confini di step dati
     * dal route-callback; il replay offline deve riprodurre hits/misses/evictions. */
    {
        LlmEngineMoeProvider* p = llmengine_moe_provider_create();
        CHECK(p != NULL);
        LlmEngineMoeRuntimeConfig rc;
        llmengine_moe_config_defaults(&rc);
        rc.mode = LLMENGINE_MOE_MODE_SWAP;
        rc.cache_mb = 1;            /* store SWAP (cap min 8 bundle) */
        rc.async_workers = 0;       /* solo eventi dello store (nessun async) */
        rc.prefetch_topn = 0;
        rc.hotness = 1;             /* D19: hotness alimentata dal route-callback */
        CHECK(llmengine_moe_provider_load(p, "tiny", gp, &rc) == LLMENGINE_MOE_OK);
        LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, "tiny");
        CHECK(rt != NULL);
        const LlmEngineMoeModelMeta* meta = llmengine_moe_meta(rt);
        CHECK(meta != NULL && meta->layer_count > 0 && meta->expert_count > 0);
        uint64_t expert = meta ? meta->routed_bytes / ((uint64_t)meta->layer_count * meta->expert_count) : 0;
        CHECK(expert > 0);

        char tp[1200]; snprintf(tp, sizeof(tp), "%s/tiny.moetrace", dir);
        LlmEngineMoeProfiler* prof = llmengine_moe_prof_create();
        CHECK(prof != NULL);
        CHECK(llmengine_moe_prof_trace_open_ex(prof, tp, rc.cache_mb * 1024ull * 1024ull, expert) == LLMENGINE_MOE_OK);
        llmengine_moe_set_profiler(rt, prof);

        float in1[1] = { 0.0f }, out4[4];
        for (int i = 0; i < 4; ++i)
            CHECK(llmengine_moe_forward(rt, in1, 1, out4) == LLMENGINE_MOE_OK);
        llmengine_moe_prof_trace_close(prof);
        llmengine_moe_set_profiler(rt, NULL);
        llmengine_moe_prof_destroy(prof);

        /* K2: gli stats (path store) riportano la resident hit rate vera */
        LlmEngineMoeStreamStats st;
        CHECK(llmengine_moe_stream_stats(rt, &st) == LLMENGINE_MOE_OK);
        CHECK(st.cache_misses > 0);
        CHECK(st.cache_hits > 0);
        CHECK(st.cache_resident_hits > 0);
        CHECK(st.cache_resident_hits <= st.cache_hits);

        /* D19: la hotness è alimentata dal route-callback anche nel path GGUF (freq osservata) */
        const LlmEngineMoeExpertCache* hc = llmengine_moe_expert_cache(rt);
        CHECK(hc != NULL);
        CHECK(llmengine_moe_cache_fetches(hc) > 0);
        CHECK(llmengine_moe_cache_freq(hc, (LlmEngineMoeExpertKey){ 0, 0 }) > 0);

        /* K1: replay offline == summary del trace (stesso trace, stessa policy) */
        LlmEngineMoeTraceSummary sum;
        CHECK(llmengine_moe_trace_summary(tp, &sum) == LLMENGINE_MOE_OK);
        LlmEngineMoeSimResult sim;
        CHECK(llmengine_moe_sim_replay(tp, 0, 0, &sim) == LLMENGINE_MOE_OK);
        CHECK(sim.hits == sum.cache_hits);
        CHECK(sim.misses == sum.cache_misses);
        CHECK(sim.evictions == sum.evictions);
        CHECK(sim.resident_hits == st.cache_resident_hits);   /* K2: store == replay */

        llmengine_moe_provider_destroy(p);
        remove(tp);
    }

    /* ── K5: esperto non risolvibile (router > tensor stacked) → stato DISTINTO, mai silenzioso ── */
    {
        char gb[1200]; snprintf(gb, sizeof(gb), "%s/tiny_broken.gguf", dir);
        write_tiny_moe_gguf_ex(gb, 0, 4, 3);   /* metadata 4 esperti, tensor 2, router forza #3 */
        LlmEngineMoeProvider* p = llmengine_moe_provider_create();
        CHECK(p != NULL);
        LlmEngineMoeRuntimeConfig rc;
        llmengine_moe_config_defaults(&rc);
        rc.mode = LLMENGINE_MOE_MODE_SWAP;
        rc.cache_mb = 1;
        rc.async_workers = 0;
        rc.prefetch_topn = 0;
        CHECK(llmengine_moe_provider_load(p, "bad", gb, &rc) == LLMENGINE_MOE_OK);
        LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, "bad");
        CHECK(rt != NULL);
        float in1[1] = { 0.0f }, ob[4];
        CHECK(llmengine_moe_forward(rt, in1, 1, ob) == LLMENGINE_MOE_ERR_EXPERT_MISSING);
        LlmEngineMoeStreamStats stb;
        CHECK(llmengine_moe_stream_stats(rt, &stb) == LLMENGINE_MOE_OK);
        CHECK(stb.unresolvable_experts >= 1);
        llmengine_moe_provider_destroy(p);
        remove(gb);
    }

    remove(gp);
    rmdir(dir);
    if (fails == 0) printf("test_moe_gguf_adapter: OK\n");
    else fprintf(stderr, "test_moe_gguf_adapter: %d FAIL\n", fails);
    return fails ? 1 : 0;
}
