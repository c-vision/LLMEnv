/* llm_ds4_engine_abi.c — adattatore vtable LlmEngine -> motore ds4 vendorizzato.
 * Espone la vtable `llmengine_llm_engine_ds4n()` (n = native). CAP_IDS: il motore lavora a
 * token-id; tokenize/detokenize restano al chiamante (bridge non espone il tokenizer).
 * Una sessione per handle: `create_session` restituisce lo stesso handle del modello. */
#include "llmengine.h"
#include "ds4_bridge.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    ds4_bridge* b;
    int32_t* toks; int n, cap;
    int ctx;
} ds4n_sess;

static const char* ds4n_name(void) { return "ds4"; }

static LlmEngineLlmModel ds4n_load(const char* path) {
    if (!path) return NULL;
    ds4_bridge* b = ds4_bridge_open(path, 0, 0, 0);
    if (!b) return NULL;
    ds4n_sess* s = calloc(1, sizeof *s);
    if (!s) { ds4_bridge_close(b); return NULL; }
    s->b = b; s->ctx = 4096;
    return (LlmEngineLlmModel)s;
}
static void ds4n_free(LlmEngineLlmModel m) {
    ds4n_sess* s = (ds4n_sess*)m;
    if (!s) return;
    ds4_bridge_close(s->b); free(s->toks); free(s);
}
static LlmEngineLlmSession ds4n_create(LlmEngineLlmModel m) { return (LlmEngineLlmSession)m; }
static void ds4n_destroy(LlmEngineLlmSession sess) { (void)sess; }

static int ds4n_append(LlmEngineLlmSession sess, const int32_t* ids, int n) {
    ds4n_sess* s = (ds4n_sess*)sess;
    if (!s || !ids || n < 0) return -1;
    if (s->n + n > s->cap) {
        int nc = s->cap ? s->cap : 256; while (nc < s->n + n) nc *= 2;
        int32_t* t = realloc(s->toks, (size_t)nc * sizeof(int32_t)); if (!t) return -1;
        s->toks = t; s->cap = nc;
    }
    memcpy(s->toks + s->n, ids, (size_t)n * sizeof(int32_t)); s->n += n;
    return ds4_bridge_sync(s->b, s->toks, s->n) == 0 ? 0 : -1;   /* ABI: 0 = ok, come gli altri motori */
}
static int ds4n_generate(LlmEngineLlmSession sess, int max_new, int32_t eos, int32_t* out) {
    ds4n_sess* s = (ds4n_sess*)sess;
    if (!s || max_new <= 0 || !out) return -1;
    int k = 0;
    for (; k < max_new; k++) {
        int t = ds4_bridge_argmax(s->b);
        if (t < 0) break;
        out[k] = t;
        if (eos >= 0 && t == eos) { k++; break; }
        if (ds4n_append(sess, &out[k], 1) < 0) break;
    }
    return k;
}
static int ds4n_kv_stats(LlmEngineLlmSession sess, int* used, int* cap) {
    ds4n_sess* s = (ds4n_sess*)sess; if (!s) return -1;
    if (used) *used = s->n; if (cap) *cap = s->ctx; return 0;
}
static void ds4n_free_str(char* p) { free(p); }

static const LlmEngineLlmEngine DS4N_ENGINE = {
    .name = "ds4",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_KV,
    .load_model = ds4n_load,
    .free_model = ds4n_free,
    .create_session = ds4n_create,
    .destroy_session = ds4n_destroy,
    .append_ids = ds4n_append,
    .generate_ids = ds4n_generate,
    .kv_stats = ds4n_kv_stats,
    .free_str = ds4n_free_str,
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds4n(void) { return &DS4N_ENGINE; }
