/* ds4_bridge.c — bridge token-level verso ds4.
 *
 * Due oggetti distinti, come nell'upstream:
 *   - `ds4_bridge` = il MOTORE (pesi, KV dimensionata all'apertura);
 *   - `ds4_bridge_session` = una conversazione (KV propria). Piu' sessioni sullo stesso motore
 *     sono legali e servono al server (`serve --batch`).
 *
 * Le funzioni "senza session" restano per compatibilita': usano una sessione di DEFAULT creata
 * lazy al primo uso. Il codice nuovo usa le `ds4_bridge_session_*`.
 */
#include "ds4.h"
#include "ds4_bridge.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

struct ds4_bridge { ds4_engine* e; ds4_session* s; int ctx; };
struct ds4_bridge_session { ds4_session* s; ds4_engine* e; int ctx; };

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/stat.h>
#endif

/* Cartella con `metal/*.metal` (i kernel di ds4 si leggono come file, relativi alla CWD). Ordine: accanto all'eseguibile
 * installato (`<bin>/../share/fastllm/ds4`, come lo mette il pacchetto di FastLLM), poi la cartella dei sorgenti usata alla
 * compilazione (build di sviluppo). Scrive il percorso in `out`; 0 = nessuna trovata (si resta nella CWD). */
static int ds4_resource_dir(char* out, size_t cap) {
#ifdef __APPLE__
    char raw[PATH_MAX], real[PATH_MAX], probe[PATH_MAX + 64];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) == 0 && realpath(raw, real)) {
        char* slash = strrchr(real, '/');
        struct stat st;
        if (slash) {
            *slash = 0;
            snprintf(out, cap, "%s/../share/fastllm/ds4", real);
            snprintf(probe, sizeof probe, "%s/metal/dense.metal", out);
            if (stat(probe, &st) == 0) return 1;
        }
    }
#endif
#ifdef DS4_RESOURCE_DIR
    snprintf(out, cap, "%s", DS4_RESOURCE_DIR);
    return 1;
#else
    (void)out; (void)cap;
    return 0;
#endif
}

ds4_bridge* ds4_bridge_open_ex(const char* model, int ctx, int stream, int prefill_chunk, int inspect_only);

ds4_bridge* ds4_bridge_open(const char* model, int ctx, int stream, int prefill_chunk) {
    return ds4_bridge_open_ex(model, ctx, stream, prefill_chunk, 0);
}

ds4_bridge* ds4_bridge_open_ex(const char* model, int ctx, int stream, int prefill_chunk, int inspect_only) {
    return ds4_bridge_open_opts(model, ctx, stream, prefill_chunk, 0, inspect_only);
}

ds4_bridge* ds4_bridge_open_opts(const char* model, int ctx, int stream, int prefill_chunk,
                                 uint64_t cache_bytes, int inspect_only) {
    ds4_bridge_options o;
    memset(&o, 0, sizeof o);
    o.ctx = ctx; o.stream = stream; o.prefill_chunk = prefill_chunk;
    o.cache_bytes = cache_bytes; o.inspect_only = inspect_only;
    return ds4_bridge_open2(model, &o);
}

ds4_bridge* ds4_bridge_open2(const char* model, const ds4_bridge_options* o) {
    ds4_bridge_options d;
    int ctx;
    ds4_engine_options opt;
    ds4_engine* e = NULL;
    ds4_bridge* b;
    char oldcwd[PATH_MAX];
    int have_cwd, orc;
    memset(&d, 0, sizeof d);
    if (o) d = *o;
    ctx = d.ctx > 0 ? d.ctx : 32768;      /* context_size=0 non crea la sessione */
    memset(&opt, 0, sizeof opt);
    opt.inspect_only = d.inspect_only ? true : false;
    opt.model_path = model;
    opt.context_size = ctx;
    opt.ssd_streaming = d.stream ? true : false;
    /* 0 = auto-sizing (ds4_engine_configure_streaming_auto_cache): non si inventa una formula. */
    if (d.cache_bytes > 0) opt.ssd_streaming_cache_bytes = d.cache_bytes;
    if (d.prefill_chunk > 0) opt.prefill_chunk = (uint32_t)d.prefill_chunk;
    /* MTP/DSpark: opzioni dell'upstream, passate cosi' come sono. */
    if (d.mtp_path && *d.mtp_path) opt.mtp_path = d.mtp_path;
    opt.dspark = d.dspark ? true : false;
    opt.dspark_strict = d.dspark_strict ? true : false;
    /* ds4 carica i kernel .metal dalla CWD: ci si sposta nella resource dir per l'apertura
     * (nessuna env), poi si ripristina la CWD. */
    have_cwd = (getcwd(oldcwd, sizeof oldcwd) != NULL);
    { char rdir[PATH_MAX]; if (ds4_resource_dir(rdir, sizeof rdir) && chdir(rdir) != 0) { /* resta la CWD corrente */ } }
    orc = ds4_engine_open(&e, &opt);
    if (have_cwd) { if (chdir(oldcwd) != 0) { /* ignore */ } }
    if (orc != 0) { fprintf(stderr, "bridge: open fallita\n"); return NULL; }
    /* NESSUNA sessione qui: la crea chi ne ha bisogno. L'apertura e' gia' costosa (pesi + pianificazione). */
    b = (ds4_bridge*)calloc(1, sizeof *b);
    if (!b) { ds4_engine_close(e); return NULL; }
    b->e = e; b->s = NULL; b->ctx = ctx;
    return b;
}

int ds4_bridge_has_mtp(ds4_bridge* b) {
    return (b && b->e) ? (ds4_engine_has_mtp(b->e) ? 1 : 0) : 0;
}

void ds4_bridge_close(ds4_bridge* b) {
    if (!b) return;
    if (b->s) ds4_session_free(b->s);
    if (b->e) ds4_engine_close(b->e);
    free(b);
}

/* Sessione di default: creata al primo uso dalle funzioni di compatibilita'. */
static ds4_session* bridge_default(ds4_bridge* b) {
    if (!b || !b->e) return NULL;
    if (!b->s) {
        if (ds4_session_create(&b->s, b->e, b->ctx) != 0) {
            b->s = NULL;
            fprintf(stderr, "bridge: session create fallita (contesto %d)\n", b->ctx);
            return NULL;
        }
    }
    return b->s;
}

ds4_bridge_session* ds4_bridge_session_new(ds4_bridge* b) {
    ds4_session* s = NULL;
    ds4_bridge_session* bs;
    if (!b || !b->e) return NULL;
    if (ds4_session_create(&s, b->e, b->ctx) != 0) {
        fprintf(stderr, "bridge: session create fallita (contesto %d)\n", b->ctx);
        return NULL;
    }
    bs = (ds4_bridge_session*)calloc(1, sizeof *bs);
    if (!bs) { ds4_session_free(s); return NULL; }
    bs->s = s;
    bs->e = b->e;
    bs->ctx = b->ctx;
    return bs;
}

/* Riporta la sessione allo stato iniziale (KV svuotata): si butta la `ds4_session` e se ne crea
 * una nuova sullo stesso motore. E' l'unico modo corretto, perche' ds4 non espone un reset. */
int ds4_bridge_session_reset(ds4_bridge_session* s) {
    ds4_session* fresh = NULL;
    if (!s || !s->e) return -1;
    if (ds4_session_create(&fresh, s->e, s->ctx) != 0) return -1;
    if (s->s) ds4_session_free(s->s);
    s->s = fresh;
    return 0;
}

void ds4_bridge_session_free(ds4_bridge_session* s) {
    if (!s) return;
    if (s->s) ds4_session_free(s->s);
    free(s);
}

static int bridge_sync_on(ds4_session* s, const int* toks, int n) {
    int* v;
    ds4_tokens t;
    int rc;
    if (!s || n < 0) return -1;
    v = (int*)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    if (!v) return -1;
    if (n > 0) memcpy(v, toks, sizeof(int) * (size_t)n);
    memset(&t, 0, sizeof t); t.v = v; t.len = n; t.cap = n;
    { char err[256]; err[0] = 0;
      rc = ds4_session_sync(s, &t, err, sizeof err);
      if (rc != 0) fprintf(stderr, "bridge: sync rc=%d err=%s\n", rc, err);
    }
    free(v);
    return rc;
}

int ds4_bridge_session_sync(ds4_bridge_session* s, const int* toks, int n) {
    return s ? bridge_sync_on(s->s, toks, n) : -1;
}

int ds4_bridge_session_argmax(ds4_bridge_session* s) {
    return (s && s->s) ? ds4_session_argmax(s->s) : -1;
}

int ds4_bridge_session_sample(ds4_bridge_session* s, float temperature, int top_k, float top_p,
                              float min_p, uint64_t* rng) {
    if (!s || !s->s || !rng) return -1;
    /* temp<=0 e' GREEDY, non "sampling a temperatura zero": convenzione dell'ABI. */
    if (!(temperature > 0.0f)) return ds4_session_argmax(s->s);
    return ds4_session_sample(s->s, temperature, top_k, top_p, min_p, rng);
}

int ds4_bridge_session_logits(ds4_bridge_session* s, float* out, int cap) {
    return (s && s->s) ? ds4_session_copy_logits(s->s, out, cap) : -1;
}

/* ---- compatibilita': delega alla sessione di default ---------------------------------------- */

int ds4_bridge_sync(ds4_bridge* b, const int* toks, int n) {
    return bridge_sync_on(bridge_default(b), toks, n);
}

int ds4_bridge_argmax(ds4_bridge* b) {
    ds4_session* s = bridge_default(b);
    return s ? ds4_session_argmax(s) : -1;
}

int ds4_bridge_sample(ds4_bridge* b, float temperature, int top_k, float top_p, float min_p,
                      uint64_t* rng) {
    ds4_session* s = bridge_default(b);
    if (!s || !rng) return -1;
    if (!(temperature > 0.0f)) return ds4_session_argmax(s);
    return ds4_session_sample(s, temperature, top_k, top_p, min_p, rng);
}

int ds4_bridge_logits(ds4_bridge* b, float* out, int cap) {
    ds4_session* s = bridge_default(b);
    return s ? ds4_session_copy_logits(s, out, cap) : -1;
}

int ds4_bridge_vocab(ds4_bridge* b) { return b ? ds4_engine_vocab_size(b->e) : -1; }
