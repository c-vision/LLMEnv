/* llm_ds4_engine_abi.c — adattatore vtable LlmEngine -> motore ds4 vendorizzato.
 * Espone la vtable `llmengine_llm_engine_ds4n()` (n = native). CAP_IDS: il motore lavora a
 * token-id; tokenize/detokenize restano al chiamante (bridge non espone il tokenizer).
 * Una sessione per handle: `create_session` restituisce lo stesso handle del modello.
 *
 * CONTESTO. ds4 dimensiona KV e buffer all'apertura, quindi la lunghezza del contesto deve essere nota PRIMA di aprire:
 * `load_model` controlla solo il file (apertura "inspect_only"), `kv_set_max(n)` registra il contesto richiesto e l'apertura vera
 * avviene alla prima `create_session` (o all'uso). n = 0 -> automatico: 131072 token, dimezzati fino a 32768 se ds4 non riesce
 * a pianificare la memoria. Una richiesta esplicita (n > 0) non viene ridotta: se non entra, l'errore e' esplicito. */
#include "llmengine.h"
#include "ds4_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4N_AUTO_CTX 131072
#define DS4N_MIN_CTX  32768

typedef struct {
    ds4_bridge* b;
    char* path;
    int32_t* toks; int n, cap;
    int ctx;    /* contesto effettivo (0 = non ancora aperto) */
    int want;   /* contesto richiesto con kv_set_max (0 = automatico) */
} ds4n_sess;

static const char* ds4n_name(void) { return "ds4"; }

static LlmEngineLlmModel ds4n_load(const char* path) {
    if (!path) return NULL;
    ds4_bridge* chk = ds4_bridge_open_ex(path, 0, 0, 0, 1);   /* validazione: nessun peso caricato */
    if (!chk) return NULL;
    ds4_bridge_close(chk);
    ds4n_sess* s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->path = strdup(path);
    if (!s->path) { free(s); return NULL; }
    return (LlmEngineLlmModel)s;
}
static void ds4n_free(LlmEngineLlmModel m) {
    ds4n_sess* s = (ds4n_sess*)m;
    if (!s) return;
    if (s->b) ds4_bridge_close(s->b);
    free(s->path); free(s->toks); free(s);
}

/* Apre il motore se non lo e' gia' (con il contesto richiesto o automatico). 0 = ok. */
static int ds4n_ensure(ds4n_sess* s) {
    if (s->b) return 0;
    int ctx = s->want > 0 ? s->want : DS4N_AUTO_CTX;
    for (;;) {
        s->b = ds4_bridge_open(s->path, ctx, 0, 0);
        if (s->b) break;
        if (s->want > 0 || ctx <= DS4N_MIN_CTX) return -1;   /* richiesta esplicita o minimo raggiunto */
        ctx /= 2; if (ctx < DS4N_MIN_CTX) ctx = DS4N_MIN_CTX;
    }
    s->ctx = ctx; s->n = 0;
    fprintf(stderr, "[ds4] contesto %d token (%s)\n", ctx, s->want > 0 ? "richiesto" : "automatico");
    return 0;
}
static int ds4n_kv_set_max(LlmEngineLlmModel m, int n) {
    ds4n_sess* s = (ds4n_sess*)m;
    if (!s) return -1;
    if (n < 0) n = 0;
    if (s->b && n > 0 && n != s->ctx) {   /* gia' aperto con un altro contesto: si richiude, riapre al prossimo uso */
        ds4_bridge_close(s->b); s->b = NULL; s->ctx = 0; s->n = 0;
    } else if (s->b && n == 0 && s->want != 0) {
        ds4_bridge_close(s->b); s->b = NULL; s->ctx = 0; s->n = 0;
    }
    s->want = n;
    return 0;
}
static LlmEngineLlmSession ds4n_create(LlmEngineLlmModel m) {
    ds4n_sess* s = (ds4n_sess*)m;
    if (!s || ds4n_ensure(s) != 0) return NULL;
    return (LlmEngineLlmSession)m;
}
static void ds4n_destroy(LlmEngineLlmSession sess) { (void)sess; }

static int ds4n_append(LlmEngineLlmSession sess, const int32_t* ids, int n) {
    ds4n_sess* s = (ds4n_sess*)sess;
    if (!s || !ids || n < 0) return -1;
    if (ds4n_ensure(s) != 0) return -1;
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
    if (ds4n_ensure(s) != 0) return -1;
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
    if (used) *used = s->n; if (cap) *cap = s->ctx ? s->ctx : s->want; return 0;
}
static void ds4n_free_str(char* p) { free(p); }

static const LlmEngineLlmEngine DS4N_ENGINE = {
    .name = "ds4",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_KV | LLMMODELIO_CAP_KV_MAX,
    .load_model = ds4n_load,
    .free_model = ds4n_free,
    .create_session = ds4n_create,
    .destroy_session = ds4n_destroy,
    .append_ids = ds4n_append,
    .generate_ids = ds4n_generate,
    .kv_stats = ds4n_kv_stats,
    .kv_set_max = ds4n_kv_set_max,
    .free_str = ds4n_free_str,
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds4n(void) { return &DS4N_ENGINE; }
