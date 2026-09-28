/* llm_dsv41_loader_test.c — verifica del loader safetensors DeepSeek V4.1.
 *
 * T1) copertura nomi: confronta `llm_dsv41_expected_names(llm_dsv41_config_real())` con il
 *     `model.safetensors.index.json` reale (passato come argv[1]); report missing/extra.
 * T2) dequant: legge i probe fp8/ue8m0, fp4/ue8m0, bf16 dal checkpoint sintetico e confronta
 *     con `llm_dsv41_st_ref.h`.
 * T3) load + forward: carica il checkpoint tiny sintetico e riproduce REF_CSA_LOGITS.
 *
 * Build: tools/dsv41_loader_test.sh */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <mlx/c/mlx.h>
#include <cJSON.h>

#include "llm_dsv41_ckpt.h"
#include "llm_dsv41_loader.h"
#include "llm_dsv41_mlx.h"
#include "llm_dsv41_st_ref.h"
#include "llm_dsv41_fwd_csa_ref.h"

static int g_fail = 0;
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-46s\n", name); \
    else { printf("FAIL %-46s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static int nclose(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > tol * m) { printf("  [%d] got=%.7g exp=%.7g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}

/* ── hash set di stringhe (per la copertura nomi) ───────────────────────────── */
typedef struct { char** k; size_t cap; size_t n; } StrSet;
static uint64_t fnv(const char* s) { uint64_t h = 1469598103934665603ULL; for (; *s; s++) { h ^= (uint8_t)*s; h *= 1099511628211ULL; } return h; }
static void sset_init(StrSet* h, size_t cap) { h->cap = cap; h->k = (char**)calloc(cap, sizeof(char*)); h->n = 0; }
static void sset_add(StrSet* h, const char* s) { size_t i = fnv(s) & (h->cap - 1); while (h->k[i] && strcmp(h->k[i], s)) i = (i + 1) & (h->cap - 1); if (!h->k[i]) { h->k[i] = strdup(s); h->n++; } }
static int sset_has(StrSet* h, const char* s) { size_t i = fnv(s) & (h->cap - 1); while (h->k[i]) { if (!strcmp(h->k[i], s)) return 1; i = (i + 1) & (h->cap - 1); } return 0; }
static void sset_free(StrSet* h) { for (size_t i = 0; i < h->cap; i++) free(h->k[i]); free(h->k); }

static StrSet g_expected;
static int expected_sink(void* user, const char* name) { (void)user; sset_add(&g_expected, name); return 0; }
static StrSet g_real;
static int check_missing_sink(void* user, const char* name) {
    (void)user;
    if (!sset_has(&g_real, name)) { if (*(int*)user < 5) printf("  missing: %s\n", name); (*(int*)user)++; }
    return 0;
}

static int coverage(const char* index_path) {
    FILE* f = fopen(index_path, "rb");
    if (!f) { printf("coverage: impossibile aprire %s\n", index_path); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    size_t rd = fread(buf, 1, (size_t)n, f); fclose(f); buf[rd] = '\0';
    cJSON* root = cJSON_Parse(buf); free(buf);
    if (!root) { printf("coverage: JSON non valido\n"); return 0; }
    cJSON* wm = cJSON_GetObjectItemCaseSensitive(root, "weight_map");
    if (!cJSON_IsObject(wm)) { cJSON_Delete(root); printf("coverage: no weight_map\n"); return 0; }

    sset_init(&g_real, 1u << 21);
    for (cJSON* it = wm->child; it; it = it->next) sset_add(&g_real, it->string);
    sset_init(&g_expected, 1u << 20);
    LlmDsv41Config cfg = llm_dsv41_config_real();
    llm_dsv41_expected_names(&cfg, expected_sink, NULL);
    /* DSpark/MTP (`mtp.*`) e Vision (`vision.*`/`aligner.*`/`image_*`) nella checklist. */
    llm_dsv41_expected_dspark_names(&cfg, expected_sink, NULL);
    llm_dsv41_expected_vision_names(&cfg, expected_sink, NULL);

    size_t expected_n = g_expected.n;
    int missing = 0;
    llm_dsv41_expected_names(&cfg, check_missing_sink, &missing);
    llm_dsv41_expected_dspark_names(&cfg, check_missing_sink, &missing);
    llm_dsv41_expected_vision_names(&cfg, check_missing_sink, &missing);

    size_t real_n = g_real.n;
    int extra = 0;
    for (cJSON* it = wm->child; it; it = it->next) {
        const char* nm = it->string;
        if (strstr(nm, ".scale")) continue;
        if (!sset_has(&g_expected, nm)) { if (extra < 5) printf("  extra: %s\n", nm); extra++; }
    }
    printf("  coverage: expected=%zu real=%zu | missing=%d extra=%d\n",
           expected_n, real_n, missing, extra);
    sset_free(&g_expected); sset_free(&g_real);
    cJSON_Delete(root);
    return (missing == 0 && extra == 0);
}

int main(int argc, char** argv) {
    const char* index = argc > 1 ? argv[1] : NULL;
    const char* ckpt = argc > 2 ? argv[2] : "/tmp/dsv41_st/tiny.safetensors";
    printf("llm_dsv41_loader — safetensors V4.1\n");
    mlx_stream s = mlx_default_gpu_stream_new();

    if (index) CHECK(coverage(index), "T1 copertura nomi vs index reale", "");
    else printf("SKIP T1 (nessun index passato)\n");

    /* T2: probe dequant */
    LlmDsv41Ckpt ck;
    if (!llm_dsv41_ckpt_open(&ck, ckpt)) { printf("FAIL ckpt_open %s\n", ckpt); return 1; }
    {
        mlx_array w = mlx_array_new(), sc = mlx_array_new();
        int ok = llm_dsv41_ckpt_get(&ck, "probe.fp8.weight", &w, s) &&
                 llm_dsv41_ckpt_get(&ck, "probe.fp8.scale", &sc, s);
        if (!ok) { printf("FAIL probe fp8 presenti\n"); g_fail++; }
        else {
            mlx_array_eval(w); mlx_array_eval(sc);
            float dst[REF_ST_FP8_OUT * REF_ST_FP8_IN];
            llmcodec_dsv41_dequant_fp8_block(mlx_array_data_uint8(w), mlx_array_data_uint8(sc),
                                        REF_ST_FP8_OUT, REF_ST_FP8_IN, dst);
            CHECK(nclose(dst, REF_ST_FP8, REF_ST_FP8_OUT * REF_ST_FP8_IN, 1e-5f),
                  "T2 dequant fp8+ue8m0", "");
        }
    }
    {
        mlx_array w = mlx_array_new(), sc = mlx_array_new();
        int ok = llm_dsv41_ckpt_get(&ck, "probe.fp4.weight", &w, s) &&
                 llm_dsv41_ckpt_get(&ck, "probe.fp4.scale", &sc, s);
        if (!ok) { printf("FAIL probe fp4 presenti\n"); g_fail++; }
        else {
            mlx_array_eval(w); mlx_array_eval(sc);
            float dst[REF_ST_FP4_OUT * REF_ST_FP4_IN];
            llmcodec_dsv41_dequant_fp4_expert(mlx_array_data_uint8(w), mlx_array_data_uint8(sc),
                                         REF_ST_FP4_OUT, REF_ST_FP4_IN, dst);
            CHECK(nclose(dst, REF_ST_FP4, REF_ST_FP4_OUT * REF_ST_FP4_IN, 1e-5f),
                  "T2 dequant fp4+ue8m0", "");
        }
    }
    {
        bool ok = true;
        mlx_array b = llm_dsv41_load_plain(&ck, "probe.bf16.weight", s, &ok);
        float dst[REF_ST_BF16_N];
        if (!ok || !llm_dsv41_copy_f32(b, dst, REF_ST_BF16_N)) { printf("FAIL probe bf16\n"); g_fail++; }
        else CHECK(nclose(dst, REF_ST_BF16, REF_ST_BF16_N, 1e-5f), "T2 load bf16->f32", "");
    }
    /* T2b: `load_weight` risolve la scala come `X.scale` (convenzione reale, NON `X.weight.scale`). */
    {
        bool ok = true;
        mlx_array a = llm_dsv41_load_weight(&ck, "probe.fp8.weight", REF_ST_FP8_OUT, REF_ST_FP8_IN, 0, s, &ok);
        float dst[REF_ST_FP8_OUT * REF_ST_FP8_IN];
        if (!ok || !llm_dsv41_copy_f32(a, dst, REF_ST_FP8_OUT * REF_ST_FP8_IN)) { printf("FAIL T2b fp8 load_weight\n"); g_fail++; }
        else CHECK(nclose(dst, REF_ST_FP8, REF_ST_FP8_OUT * REF_ST_FP8_IN, 1e-5f), "T2b load_weight fp8 X.scale", "");
    }
    {
        bool ok = true;
        mlx_array a = llm_dsv41_load_weight(&ck, "probe.fp4.weight", REF_ST_FP4_OUT, REF_ST_FP4_IN, 1, s, &ok);
        float dst[REF_ST_FP4_OUT * REF_ST_FP4_IN];
        if (!ok || !llm_dsv41_copy_f32(a, dst, REF_ST_FP4_OUT * REF_ST_FP4_IN)) { printf("FAIL T2b fp4 load_weight\n"); g_fail++; }
        else CHECK(nclose(dst, REF_ST_FP4, REF_ST_FP4_OUT * REF_ST_FP4_IN, 1e-5f), "T2b load_weight fp4 X.scale", "");
    }
    llm_dsv41_ckpt_close(&ck);

    /* T3: load completo + forward */
    LlmDsv41Config cfg = llm_dsv41_config_csa_ref();
    LlmDsv41Weights w;
    int unloaded = -1;
    if (!llm_dsv41_load_checkpoint(&w, &cfg, ckpt, s, &unloaded)) {
        printf("FAIL load_checkpoint\n"); return 1;
    }
    printf("  load_checkpoint: %d layer, nomi non materializzati (indexer/engram)=%d\n",
           w.n_layers, unloaded);
    mlx_array logits = mlx_array_new();
    if (!llm_dsv41_mlx_forward(&w, REF_CSA_IDS, REF_CSA_S, &logits, s)) {
        printf("FAIL forward da checkpoint\n");
        llm_dsv41_weights_free(&w); return 1;
    }
    mlx_array_eval(logits);
    const float* got = mlx_array_data_float32(logits);
    CHECK(nclose(got, REF_CSA_LOGITS, REF_CSA_S * REF_CSA_VOCAB, 5e-4f),
          "T3 load checkpoint -> forward -> logits", "");
    CHECK(nclose(got, REF_CSA_LOGITS, REF_CSA_VOCAB, 1e-5f),
          "T3 token0 (causalita' compresse)", "");
    printf("  logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           got[0], got[1], got[2], got[3],
           REF_CSA_LOGITS[0], REF_CSA_LOGITS[1], REF_CSA_LOGITS[2], REF_CSA_LOGITS[3]);

    mlx_array_free(logits);
    llm_dsv41_weights_free(&w);

    /* T4: variante sharded (index.json + 2 shard) + config.json, via load_model_dir. */
    struct stat stt;
    if (stat("/tmp/dsv41_st/sharded", &stt) == 0) {
        LlmDsv41Weights ws;
        mlx_array lg = mlx_array_new();
        if (!llm_dsv41_load_model_dir(&ws, "/tmp/dsv41_st/sharded", s, NULL)) {
            printf("FAIL load_model_dir sharded\n"); g_fail++;
        } else {
            llm_dsv41_mlx_forward(&ws, REF_CSA_IDS, REF_CSA_S, &lg, s);
            mlx_array_eval(lg);
            CHECK(nclose(mlx_array_data_float32(lg), REF_CSA_LOGITS, REF_CSA_S * REF_CSA_VOCAB, 5e-4f),
                  "T4 checkpoint sharded -> forward -> logits", "");
            CHECK(nclose(mlx_array_data_float32(lg), REF_CSA_LOGITS, REF_CSA_VOCAB, 1e-5f),
                  "T4 token0 (causalita' compresse)", "");
            mlx_array_free(lg);
            llm_dsv41_weights_free(&ws);
        }
    }

    /* T5: S=8 > window=4 e index_topk=2 < m=4 (top-k reale + SWA oltre finestra). */
    {
        LlmDsv41Config c8 = llm_dsv41_config_csa_ref();
        c8.index_topk = 2;
        LlmDsv41Weights w8;
        if (!llm_dsv41_load_checkpoint(&w8, &c8, ckpt, s, NULL)) {
            printf("FAIL load_checkpoint topk\n"); g_fail++;
        } else {
            mlx_array lg = mlx_array_new();
            if (!llm_dsv41_mlx_forward(&w8, REF_CSA8_IDS, REF_CSA8_S, &lg, s)) {
                printf("FAIL forward topk\n"); g_fail++;
            } else {
                mlx_array_eval(lg);
                CHECK(nclose(mlx_array_data_float32(lg), REF_CSA8_LOGITS,
                             REF_CSA8_S * REF_CSA8_VOCAB, 5e-4f),
                      "T5 top-k reale (S=8, topk=2) -> logits", "");
                CHECK(nclose(mlx_array_data_float32(lg), REF_CSA8_LOGITS,
                             REF_CSA8_VOCAB, 1e-5f),
                      "T5 token0 (causalita' compresse)", "");
                mlx_array_free(lg);
            }
            llm_dsv41_weights_free(&w8);
        }
    }

    printf(g_fail ? "FAILURES: %d\n" : "ALL PASS\n", g_fail);
    return g_fail != 0;
}
