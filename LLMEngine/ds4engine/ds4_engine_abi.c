/* ds4_engine_abi.c — il motore ds4 vendorizzato esposto come ABI `LlmEngine` (llmengine.h),
 * così FastLLM/LLMEngine lo usano al posto del motore DS4 nativo (`m->n`).
 *
 * Mappa: load_model -> ds4_engine_open; create_session -> ds4_session_create;
 * append_ids -> ds4_session_sync (prefisso completo); generate_ids -> argmax + sync incrementale.
 * La tokenizzazione/testualizzazione resta al consumer (FMLX usa il tokenizer del GGUF), quindi
 * l'ABI espone solo CAP_IDS (+ CAP_KV_MAX per il contesto). */
#include "llmengine.h"
#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- stato --- */
typedef struct {
    ds4_engine* e;
    int ctx;
    int stream;
} Ds4Model;

typedef struct {
    Ds4Model* m;
    ds4_session* s;
    int* toks;
    int ntok, cap;
    LlmEngineLlmSampling sp;   /* sampling del server (temperature/top_p/top_k/seed) */
    int has_sp;
    uint64_t rng;
} Ds4Sess;

/* set_expert_swap: PRIMA del load. Memorizza su un globale letto da load_model. */
static int g_stream = 0;
static void ds4_set_expert_swap(const char* dir) { g_stream = (dir && *dir) ? 1 : 0; }

static LlmEngineLlmModel ds4_load_model(const char* path) {
    ds4_engine_options opt;
    ds4_engine* e = NULL;
    Ds4Model* m;
    if (!path || !*path) return NULL;
    memset(&opt, 0, sizeof opt);
    opt.model_path = path;
    opt.context_size = 0;          /* il contesto vero lo fissa ds4_session_create (kv_set_max) */
    opt.ssd_streaming = g_stream ? true : false;
    if (ds4_engine_open(&e, &opt) != 0) return NULL;
    m = (Ds4Model*)calloc(1, sizeof *m);
    if (!m) { ds4_engine_close(e); return NULL; }
    m->e = e;
    m->ctx = (int)ds4_engine_prefill_chunk(e);   /* default: la chunk size non e' il ctx; 0 = default */
    m->ctx = 0;
    m->stream = g_stream;
    return (LlmEngineLlmModel)m;
}

static void ds4_free_model(LlmEngineLlmModel mm) {
    Ds4Model* m = (Ds4Model*)mm;
    if (!m) return;
    if (m->e) ds4_engine_close(m->e);
    free(m);
}

static int ds4_kv_set_max(LlmEngineLlmModel mm, int n) {
    Ds4Model* m = (Ds4Model*)mm;
    if (!m) return -1;
    m->ctx = n > 0 ? n : 0;
    return 0;
}

static LlmEngineLlmSession ds4_create_session(LlmEngineLlmModel mm) {
    Ds4Model* m = (Ds4Model*)mm;
    Ds4Sess* s;
    int ctx;
    if (!m || !m->e) return NULL;
    s = (Ds4Sess*)calloc(1, sizeof *s);
    if (!s) return NULL;
    ctx = m->ctx > 0 ? m->ctx : 32768;
    if (ds4_session_create(&s->s, m->e, ctx) != 0) { free(s); return NULL; }
    s->m = m;
    return (LlmEngineLlmSession)s;
}

static void ds4_destroy_session(LlmEngineLlmSession ss) {
    Ds4Sess* s = (Ds4Sess*)ss;
    if (!s) return;
    if (s->s) ds4_session_free(s->s);
    free(s->toks);
    free(s);
}

static int ds4_sync(Ds4Sess* s) {
    ds4_tokens t;
    char err[256];
    err[0] = 0;
    memset(&t, 0, sizeof t);
    t.v = s->toks; t.len = s->ntok; t.cap = s->ntok;
    if (ds4_session_sync(s->s, &t, err, sizeof err) != 0) return -1;
    return 0;
}

static int ds4_push_tok(Ds4Sess* s, int32_t id) {
    if (s->ntok >= s->cap) {
        int nc = s->cap ? s->cap * 2 : 256;
        int* nv = (int*)realloc(s->toks, (size_t)nc * sizeof(int));
        if (!nv) return -1;
        s->toks = nv; s->cap = nc;
    }
    s->toks[s->ntok++] = (int)id;
    return 0;
}

static int ds4_append_ids(LlmEngineLlmSession ss, const int32_t* ids, int n) {
    Ds4Sess* s = (Ds4Sess*)ss;
    int i;
    if (!s || n < 0) return -1;
    for (i = 0; i < n; i++) if (ds4_push_tok(s, ids[i]) != 0) return -1;
    /* Prefill: cronometra la sync (che processa i token appesi) e stampa i tok/s per N>=32. */
    { struct timespec t0, t1; int rc;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      rc = ds4_sync(s);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      if (n >= 32) {
          double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
          fprintf(stderr, "[ds4-prefill] N=%d %.3fs %.0f tok/s\n", n, ms / 1000.0, (double)n * 1000.0 / ms);
      }
      return rc; }
}

static int ds4_set_sampling(LlmEngineLlmSession ss, const LlmEngineLlmSampling* sp) {
    Ds4Sess* s = (Ds4Sess*)ss;
    if (!s || !sp) return -1;
    s->sp = *sp;
    s->has_sp = 1;
    s->rng = (uint64_t)(sp->seed ? sp->seed : 0x9e3779b97f4a7c15ULL);
    return 0;
}

static int ds4_generate_ids(LlmEngineLlmSession ss, int max_new, int32_t eos, int32_t* out) {
    Ds4Sess* s = (Ds4Sess*)ss;
    int i;
    if (!s) return -1;
    for (i = 0; i < max_new; i++) {
        int id;
        if (s->has_sp && s->sp.temperature > 0.f)
            id = ds4_session_sample(s->s, s->sp.temperature, s->sp.top_k, s->sp.top_p, 0.f, &s->rng);
        else
            id = ds4_session_argmax(s->s);
        if (id < 0) return i;
        if (out) out[i] = (int32_t)id;
        if (ds4_push_tok(s, (int32_t)id) != 0) return i;
        if (ds4_sync(s) != 0) return i;
        if (eos >= 0 && id == (int)eos) return i + 1;
    }
    return max_new;
}

static const LlmEngineLlmEngine DS4_ENGINE = {
    .name = "ds4engine",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_KV_MAX,
    .load_model = ds4_load_model,
    .free_model = ds4_free_model,
    .create_session = ds4_create_session,
    .destroy_session = ds4_destroy_session,
    .append_ids = ds4_append_ids,
    .set_sampling = ds4_set_sampling,
    .generate_ids = ds4_generate_ids,
    .kv_set_max = ds4_kv_set_max,
    .set_expert_swap = ds4_set_expert_swap,
    /* tokenize/detokenize/prefix/logprobs/sampling: non esposti (restano al consumer / non serve). */
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds4(void) { return &DS4_ENGINE; }
