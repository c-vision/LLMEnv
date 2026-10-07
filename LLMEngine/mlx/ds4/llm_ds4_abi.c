/* llm_ds4_abi.c — il motore DS4 nativo esposto come ABI `LlmEngine` (llmengine.h):
 * `llmengine_llm_engine_ds4`.
 *
 * Variante coperta: DeepSeek-V4 (GGUF firma "deepseek4"). La variante DeepSeek-V4.1
 * (`llm_dsv41_*`, che richiede un `mlx_stream` e una `LlmDsv41Cache` per sessione) si aggancia
 * qui allo stesso modo in un secondo momento.
 *
 * TUTTO il codice concreto DS4 (tipi `LlmDsv4Engine`, caches, metal) vive in `ds4/`: qui
 * c'e' solo l'adattamento all'interfaccia. */
#include "llmengine.h"

#define LLM_DS4_IMPL 1
#include "llm_dsv4_engine.h"
#include "llm_dsv4_autoswap.h"   /* auto-sizing della cache esperti (vive con DS4) */
#include "llmengine_mlx.h"       /* llm_ai_mlx_opt_is_set/get_int (registro opzioni) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Impostato da set_expert_swap PRIMA del load (lo swap esperti si abilita dopo il load). */
static int g_ds4_swap = 0;

static void ds4_set_expert_swap(const char* dir) { g_ds4_swap = (dir && *dir) ? 1 : 0; }

typedef struct { LlmDsv4Engine* e; } Ds4Model;

static LlmEngineLlmModel ds4_load(const char* path) {
    LlmDsv4Engine* e;
    Ds4Model* m;
    if (!path || !*path) return NULL;
    e = llm_dsv4_engine_load(path, 4096);   /* ctx di default; kv_set_max lo corregge */
    if (!e) return NULL;
    if (g_ds4_swap) {
        /* Cap della cache esperti: `offload_cache_mb`/`dsv4_cache_mb` (MB) se dati, altrimenti
         * AUTO-SIZING (come il ds4 originale). Il modello reale e' ~97 GB: lo streaming e' default. */
        int mb = 0, bundles = 0;
        if (llm_ai_mlx_opt_is_set("offload_cache_mb") == 1 &&
            llm_ai_mlx_opt_get_int("offload_cache_mb", &mb) == 0 && mb > 0) bundles = mb / 7;
        else if (llm_ai_mlx_opt_get_int("dsv4_cache_mb", &mb) == 0 && mb > 0) bundles = mb / 7;
        else bundles = llm_dsv4_autoswap_bundles(e);
        if (bundles < 256) bundles = 256;
        (void)llm_dsv4_engine_enable_swap(e, bundles);
        llm_dsv4_gpu_ctx_reinit(&e->w.cfg, e->w.exps);
    }
    m = (Ds4Model*)calloc(1, sizeof *m);
    if (!m) { llm_dsv4_engine_free(e); return NULL; }
    m->e = e;
    return (LlmEngineLlmModel)m;
}

static void ds4_free(LlmEngineLlmModel mm) {
    Ds4Model* m = (Ds4Model*)mm;
    if (!m) return;
    if (m->e) llm_dsv4_engine_free(m->e);
    free(m);
}

/* Il DS4 nativo ha UNA sessione (lo stato vive nell'engine): la sessione e' l'engine stesso. */
static LlmEngineLlmSession ds4_session(LlmEngineLlmModel mm) { return (LlmEngineLlmSession)mm; }
static void ds4_destroy(LlmEngineLlmSession ss) { (void)ss; }

static int ds4_kv_set_max(LlmEngineLlmModel mm, int n) {
    Ds4Model* m = (Ds4Model*)mm;
    if (!m || !m->e || n <= 0) return -1;
    if ((int)m->e->n_ctx != n) { m->e->n_ctx = n; llm_dsv4_engine_reset(m->e); }
    return 0;
}

static int ds4_append(LlmEngineLlmSession ss, const int32_t* ids, int n) {
    Ds4Model* m = (Ds4Model*)ss;
    if (!m || !m->e || n < 0) return -1;
    llm_dsv4_engine_append(m->e, (const int*)ids, n);
    return 0;
}

static int ds4_generate(LlmEngineLlmSession ss, int max_new, int32_t eos, int32_t* out) {
    Ds4Model* m = (Ds4Model*)ss;
    if (!m || !m->e) return -1;
    return llm_dsv4_engine_generate_step(m->e, (int*)out, max_new, eos);
}

static int ds4_set_sampling(LlmEngineLlmSession ss, const LlmEngineLlmSampling* sp) {
    Ds4Model* m = (Ds4Model*)ss;
    if (!m || !m->e || !sp) return -1;
    llm_dsv4_engine_set_sampling(m->e, sp->temperature, sp->top_k, sp->top_p, sp->seed);
    return 0;
}

static const LlmEngineLlmEngine DS4_ENGINE = {
    .name = "ds4",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_KV_MAX,
    .load_model = ds4_load,
    .free_model = ds4_free,
    .create_session = ds4_session,
    .destroy_session = ds4_destroy,
    .append_ids = ds4_append,
    .set_sampling = ds4_set_sampling,
    .generate_ids = ds4_generate,
    .kv_set_max = ds4_kv_set_max,
    .set_expert_swap = ds4_set_expert_swap,
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds4(void) { return &DS4_ENGINE; }
