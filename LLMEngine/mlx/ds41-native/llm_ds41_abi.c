/* llm_ds41_abi.c — adapter DS4.1 -> ABI comune LlmEngine.
 *
 * Espone `llmengine_llm_engine_ds41()` con GLI STESSI 9 metodi del vendored `ds4-engine`
 * (`llmengine_llm_engine_ds4`) e di `ds4` (`llmengine_llm_engine_ds4`):
 *   load_model, free_model, create_session, destroy_session, append_ids, set_sampling,
 *   generate_ids, kv_set_max, set_expert_swap.
 * Cosi' un consumatore (FastLLM e altri) sceglie l'engine per `model_type` e li usa tutti allo
 * stesso modo, senza rami specifici nel codice condiviso. Vive con DS4.1 (ds41-native/). */
#define LLM_DS41_ABI_IMPL 1
#include "llm_dsv41_engine.h"
#include "llmengine.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    LlmDsv41Engine* e;
    mlx_device device;
    mlx_stream stream;
} Ds41Model;

typedef struct {
    Ds41Model* m;
    LlmDsv41Cache* cache;
    int cache_offset;
    int32_t pending;
    bool has_pending;
    LlmDsv41Sampling sm;
    int32_t* hist;   /* token gia' generati (per repetition/frequency/presence penalty) */
    int hist_n, hist_cap;
} Ds41Session;

static void ds41_hist_push(Ds41Session* s, const int32_t* ids, int n) {
    if (n <= 0) return;
    if (s->hist_n + n > s->hist_cap) {
        int nc = s->hist_cap ? s->hist_cap : 256;
        int32_t* nv;
        while (nc < s->hist_n + n) nc *= 2;
        nv = (int32_t*)realloc(s->hist, (size_t)nc * sizeof(int32_t));
        if (!nv) return;
        s->hist = nv; s->hist_cap = nc;
    }
    memcpy(s->hist + s->hist_n, ids, (size_t)n * sizeof(int32_t));
    s->hist_n += n;
}

static LlmEngineLlmModel ds41_load(const char* dir) {
    Ds41Model* m;
    mlx_device gpu, chosen;
    mlx_stream st;
    bool av = false;
    if (!dir || !*dir) return NULL;
    gpu = mlx_device_new_type(MLX_GPU, 0);
    mlx_device_is_available(&av, gpu);
    chosen = av ? gpu : mlx_device_new_type(MLX_CPU, 0);
    if (!av) mlx_device_free(gpu);
    mlx_set_default_device(chosen);
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return NULL; }
    (void)mlx_set_default_stream(st);
    m = (Ds41Model*)calloc(1, sizeof *m);
    if (!m) { mlx_device_free(chosen); return NULL; }
    m->device = chosen;
    m->stream = st;
    m->e = llm_dsv41_engine_load(dir, st);
    if (!m->e) { free(m); mlx_device_free(chosen); return NULL; }
    return (LlmEngineLlmModel)m;
}

static void ds41_free(LlmEngineLlmModel mm) {
    Ds41Model* m = (Ds41Model*)mm;
    if (!m) return;
    if (m->e) llm_dsv41_engine_free(m->e);
    mlx_stream_free(m->stream);
    mlx_device_free(m->device);
    free(m);
}

static LlmEngineLlmSession ds41_session(LlmEngineLlmModel mm) {
    Ds41Model* m = (Ds41Model*)mm;
    Ds41Session* s;
    if (!m) return NULL;
    s = (Ds41Session*)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->m = m;
    s->sm.active = false;
    return (LlmEngineLlmSession)s;
}

static void ds41_destroy(LlmEngineLlmSession ss) {
    Ds41Session* s = (Ds41Session*)ss;
    if (!s) return;
    if (s->cache) { llm_dsv41_cache_free(s->cache); free(s->cache); s->cache = NULL; }
    free(s->hist);
    free(s);
}

static int ds41_append(LlmEngineLlmSession ss, const int32_t* ids, int n) {
    Ds41Session* s = (Ds41Session*)ss;
    int rc;
    if (!s || n < 0) return -1;
    s->sm.hist = s->hist; s->sm.hist_n = s->hist_n;
    rc = llm_dsv41_engine_append(s->m->e, (void**)&s->cache, ids, n, &s->cache_offset,
                                 &s->pending, &s->has_pending, &s->sm, s->m->stream);
    if (rc == 0) ds41_hist_push(s, ids, n);
    return rc;
}

static int ds41_set_sampling(LlmEngineLlmSession ss, const LlmEngineLlmSampling* sp) {
    Ds41Session* s = (Ds41Session*)ss;
    if (!s || !sp) return -1;
    s->sm.active = (sp->temperature > 0.0f);
    s->sm.temperature = sp->temperature;
    s->sm.top_p = sp->top_p;
    s->sm.top_k = sp->top_k;
    s->sm.seed = sp->seed;
    s->sm.repetition_penalty = sp->repetition_penalty;
    s->sm.frequency_penalty = sp->frequency_penalty;
    s->sm.presence_penalty = sp->presence_penalty;
    return 0;
}

static int ds41_generate(LlmEngineLlmSession ss, int max_new, int32_t eos, int32_t* out) {
    Ds41Session* s = (Ds41Session*)ss;
    int n;
    if (!s || !out) return -1;
    s->sm.hist = s->hist; s->sm.hist_n = s->hist_n;
    n = llm_dsv41_engine_generate(s->m->e, (void**)&s->cache, &s->cache_offset,
                                  s->pending, s->has_pending, max_new, eos, out, &s->sm, s->m->stream);
    if (n > 0) ds41_hist_push(s, out, n);
    return n;
}

static int ds41_kv_set_max(LlmEngineLlmModel mm, int n) { (void)mm; (void)n; return -1; }
static void ds41_set_expert_swap(const char* dir) { (void)dir; }

static const LlmEngineLlmEngine DS41_ENGINE = {
    .name = "ds41-native",
    .caps = LLMMODELIO_CAP_IDS,
    .load_model = ds41_load,
    .free_model = ds41_free,
    .create_session = ds41_session,
    .destroy_session = ds41_destroy,
    .append_ids = ds41_append,
    .set_sampling = ds41_set_sampling,
    .generate_ids = ds41_generate,
    .kv_set_max = ds41_kv_set_max,
    .set_expert_swap = ds41_set_expert_swap,
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds41(void) { return &DS41_ENGINE; }
