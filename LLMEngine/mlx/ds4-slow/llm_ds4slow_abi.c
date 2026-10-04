/* llm_ds4slow_abi.c — la NOSTRA implementazione DS4 nativa ("ds4-slow") esposta come ABI
 * `LlmEngine` (llmengine.h), allo stesso titolo del motore ds4 vendorizzato
 * (`llmengine_llm_engine_ds4`, in LLMEngine/ds4engine/). Cosi' un consumatore puo' scegliere
 * l'una o l'altra implementazione con la STESSA interfaccia.
 *
 * Variante coperta: DeepSeek-V4 (GGUF firma "deepseek4"). La variante DeepSeek-V4.1
 * (`llm_dsv41_*`, che richiede un `mlx_stream` e una `LlmDsv41Cache` per sessione) si aggancia
 * qui allo stesso modo in un secondo momento.
 *
 * TUTTO il codice concreto DS4 (tipi `LlmDsv4Engine`, caches, metal) vive in `ds4-slow/`: qui
 * c'e' solo l'adattamento all'interfaccia. */
#include "llmengine.h"

#define LLM_DS4SLOW_IMPL 1
#include "llm_dsv4_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Impostato da set_expert_swap PRIMA del load (lo swap esperti si abilita dopo il load). */
static int g_ds4slow_swap = 0;

static void ds4slow_set_expert_swap(const char* dir) { g_ds4slow_swap = (dir && *dir) ? 1 : 0; }

typedef struct { LlmDsv4Engine* e; } Ds4SlowModel;

static LlmEngineLlmModel ds4slow_load(const char* path) {
    LlmDsv4Engine* e;
    Ds4SlowModel* m;
    if (!path || !*path) return NULL;
    e = llm_dsv4_engine_load(path, 4096);   /* ctx di default; kv_set_max lo corregge */
    if (!e) return NULL;
    if (g_ds4slow_swap) {
        /* Cap della cache esperti: dall'opzione `dsv4_cache_mb` (MB) se data, altrimenti un default
         * prudente. Il modello reale e' ~97 GB -> lo streaming e' il default. */
        int mb = 0, bundles = 8000;
        if (llm_ai_mlx_opt_get_int("dsv4_cache_mb", &mb) == 0 && mb > 0)
            bundles = (int)(((long long)mb * 1024 * 1024) / (7ll * 1024 * 1024));
        if (bundles < 256) bundles = 256;
        (void)llm_dsv4_engine_enable_swap(e, bundles);
        llm_dsv4_gpu_ctx_reinit(&e->w.cfg, e->w.exps);
    }
    m = (Ds4SlowModel*)calloc(1, sizeof *m);
    if (!m) { llm_dsv4_engine_free(e); return NULL; }
    m->e = e;
    return (LlmEngineLlmModel)m;
}

static void ds4slow_free(LlmEngineLlmModel mm) {
    Ds4SlowModel* m = (Ds4SlowModel*)mm;
    if (!m) return;
    if (m->e) llm_dsv4_engine_free(m->e);
    free(m);
}

/* Il DS4 nativo ha UNA sessione (lo stato vive nell'engine): la sessione e' l'engine stesso. */
static LlmEngineLlmSession ds4slow_session(LlmEngineLlmModel mm) { return (LlmEngineLlmSession)mm; }
static void ds4slow_destroy(LlmEngineLlmSession ss) { (void)ss; }

static int ds4slow_kv_set_max(LlmEngineLlmModel mm, int n) {
    Ds4SlowModel* m = (Ds4SlowModel*)mm;
    if (!m || !m->e || n <= 0) return -1;
    if ((int)m->e->n_ctx != n) { m->e->n_ctx = n; llm_dsv4_engine_reset(m->e); }
    return 0;
}

static int ds4slow_append(LlmEngineLlmSession ss, const int32_t* ids, int n) {
    Ds4SlowModel* m = (Ds4SlowModel*)ss;
    if (!m || !m->e || n < 0) return -1;
    llm_dsv4_engine_append(m->e, (const int*)ids, n);
    return 0;
}

static int ds4slow_generate(LlmEngineLlmSession ss, int max_new, int32_t eos, int32_t* out) {
    Ds4SlowModel* m = (Ds4SlowModel*)ss;
    if (!m || !m->e) return -1;
    return llm_dsv4_engine_generate_step(m->e, (int*)out, max_new, eos);
}

static const LlmEngineLlmEngine DS4SLOW_ENGINE = {
    .name = "ds4-slow",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_KV_MAX,
    .load_model = ds4slow_load,
    .free_model = ds4slow_free,
    .create_session = ds4slow_session,
    .destroy_session = ds4slow_destroy,
    .append_ids = ds4slow_append,
    .generate_ids = ds4slow_generate,
    .kv_set_max = ds4slow_kv_set_max,
    .set_expert_swap = ds4slow_set_expert_swap,
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds4slow(void) { return &DS4SLOW_ENGINE; }
