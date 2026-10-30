/* llm_ds4_engine_abi.c — adattatore vtable LlmEngine -> motore ds4 vendorizzato.
 * Espone la vtable `llmengine_llm_engine_ds4n()` (n = native). CAP_IDS: il motore lavora a
 * token-id; tokenize/detokenize restano al chiamante (bridge non espone il tokenizer).
 *
 * MODELLO e SESSIONE sono due cose distinte (come nell'upstream): `load_model` apre solo il file
 * (inspect), il motore si apre alla prima `create_session`, e ogni `create_session` crea una
 * SESSIONE vera (`ds4_session`) con la sua KV. Il server puo' quindi avere piu' conversazioni sullo
 * stesso motore (`serve --batch`). `destroy_session` libera la sessione, `free_model` il motore.
 *
 * SAMPLING. `set_sampling` e' per-sessione: copia i parametri e semina il generatore; `generate_ids`
 * usa `ds4_session_sample` (upstream) quando temperatura > 0, altrimenti `ds4_session_argmax`.
 * Senza `set_sampling` il comportamento resta GREEDY (compatibilita'). `min_p` non e' nella struct
 * ABI e si legge dal registro opzioni; i tre penalty non esistono nel sampler di ds4: si dichiara
 * una volta su stderr invece di tacerlo.
 *
 * OFFLOAD. `set_expert_swap` e' globale e arriva prima del load: registra la volonta', e
 * `ds4n_ensure` apre con `ssd_streaming` (cache in byte da `offload_cache_mb` solo se l'utente
 * l'ha impostata; altrimenti auto-sizing del motore). `prefill` passa al chunk di prefill.
 *
 * CONTESTO. ds4 dimensiona KV e buffer all'apertura, quindi la lunghezza del contesto deve essere
 * nota PRIMA: `kv_set_max(n)` la registra e, se il motore e' gia' aperto, lo richiude (solo quando
 * non ci sono sessioni vive: con sessioni attive il cambio e' un errore esplicito, non un danno
 * silenzioso). n = 0 -> automatico: 131072 token, dimezzati fino a 32768 se ds4 non riesce a
 * pianificare la memoria. Una richiesta esplicita (n > 0) non viene ridotta: se non entra, errore. */
#include "llmengine.h"
#include "ds4_bridge.h"
#include "llm_engine_opts.h"   /* registro opzioni (minp/offload/prefill): nessuna env */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>            /* access(): il path di MTP si verifica PRIMA di aprire */

#define DS4N_AUTO_CTX 131072
#define DS4N_MIN_CTX  32768

typedef struct {
    ds4_bridge* b;          /* NULL finche' qualcuno non chiede una sessione */
    char* path;
    int ctx;                /* contesto effettivo del motore aperto (0 = non aperto) */
    int want;               /* contesto richiesto con kv_set_max (0 = automatico) */
    int sessions;           /* sessioni vive: protegge il cambio di contesto */
} ds4n_model;

typedef struct {
    ds4n_model* m;
    ds4_bridge_session* bs;
    int32_t* toks; int n, cap;
    /* Sampling per-sessione. `has_sp` distingue "mai impostato" (greedy, storico) da
     * "impostato a temperatura 0" (greedy esplicito). */
    LlmEngineLlmSampling sp;
    int has_sp;
    uint64_t rng;
    int warned_penalties;
} ds4n_session;

static const char* ds4n_name(void) { return "ds4"; }

/* `set_expert_swap` e' GLOBALE nell'ABI e arriva PRIMA di `load_model` (come per il motore
 * nativo): qui si registra la volonta', l'apertura vera avviene in `ds4n_ensure`. */
static int g_ds4n_offload;
static void ds4n_set_expert_swap(const char* dir) { g_ds4n_offload = (dir && *dir) ? 1 : 0; }

/* Overload: `--offload-experts` (set_expert_swap) O il registro `offload=on`. */
static int ds4n_want_offload(void) {
    int on = 0;
    if (g_ds4n_offload) return 1;
    if (llm_ai_mlx_opt_is_set("offload") == 1 && llm_ai_mlx_opt_get_bool("offload", &on) == 0)
        return on ? 1 : 0;
    return 0;
}
/* `prefill` (0 = default del motore): nessuna env, nessun default inventato. */
static int ds4n_prefill_chunk(void) {
    int v = 0;
    if (llm_ai_mlx_opt_get_int("prefill", &v) == 0 && v > 0) return v;
    return 0;
}
/* `offload_cache_mb` SOLO se l'utente l'ha impostata: il default del registro (14336) non e' una
 * scelta e usarlo cambierebbe il comportamento senza dirlo. 0 = auto del motore. */
static uint64_t ds4n_cache_bytes(void) {
    int mb = 0;
    if (llm_ai_mlx_opt_is_set("offload_cache_mb") == 1 &&
        llm_ai_mlx_opt_get_int("offload_cache_mb", &mb) == 0 && mb > 0)
        return (uint64_t)mb << 20;
    return 0;
}
/* `min_p` vive solo nel registro: la struct ABI non lo porta. */
static float ds4n_min_p(void) {
    double v = 0.0;
    if (llm_ai_mlx_opt_get_float("minp", &v) == 0 && v > 0.0 && v < 1.0) return (float)v;
    return 0.0f;
}

static LlmEngineLlmModel ds4n_load(const char* path) {
    ds4n_model* m;
    ds4_bridge* chk;
    if (!path) return NULL;
    chk = ds4_bridge_open_ex(path, 0, 0, 0, 1);   /* validazione: nessun peso caricato */
    if (!chk) return NULL;
    ds4_bridge_close(chk);
    m = (ds4n_model*)calloc(1, sizeof *m);
    if (!m) return NULL;
    m->path = strdup(path);
    if (!m->path) { free(m); return NULL; }
    return (LlmEngineLlmModel)m;
}

static void ds4n_free(LlmEngineLlmModel mm) {
    ds4n_model* m = (ds4n_model*)mm;
    if (!m) return;
    if (m->sessions > 0)
        fprintf(stderr, "[ds4] free_model con %d sessioni ancora vive: chiudile con destroy_session\n", m->sessions);
    if (m->b) ds4_bridge_close(m->b);
    free(m->path); free(m);
}

/* Apre il MOTORE se non lo e' gia' (contesto richiesto o automatico). 0 = ok. */
static int ds4n_ensure(ds4n_model* m) {
    ds4_bridge_options o;
    char mtp[1200];
    int ctx, offload, dspark = 0, strict = 0;
    if (m->b) return 0;
    mtp[0] = '\0';
    if (llm_ai_mlx_opt_get_str("ds4_mtp", mtp, sizeof mtp) != 0) mtp[0] = '\0';
    if (llm_ai_mlx_opt_get_bool("ds4_dspark", &dspark) != 0) dspark = 0;
    if (llm_ai_mlx_opt_get_bool("ds4_dspark_strict", &strict) != 0) strict = 0;
    /* Il file si verifica PRIMA di aprire: un path sbagliato deve dire quale, non far fallire
     * l'apertura del modello per un motivo che sembra del modello. */
    if (mtp[0] && access(mtp, R_OK) != 0) {
        fprintf(stderr, "[ds4] ds4_mtp non leggibile: %s\n", mtp);
        return -1;
    }
    if (dspark && !mtp[0]) {
        fprintf(stderr, "[ds4] ds4_dspark richiede ds4_mtp (il draft GGUF)\n");
        return -1;
    }
    ctx = m->want > 0 ? m->want : DS4N_AUTO_CTX;
    offload = ds4n_want_offload();
    for (;;) {
        memset(&o, 0, sizeof o);
        o.ctx = ctx;
        o.stream = offload;
        o.prefill_chunk = ds4n_prefill_chunk();
        o.cache_bytes = ds4n_cache_bytes();
        o.mtp_path = mtp[0] ? mtp : NULL;
        o.dspark = dspark;
        o.dspark_strict = strict;
        m->b = ds4_bridge_open2(m->path, &o);
        if (m->b) break;
        if (m->want > 0 || ctx <= DS4N_MIN_CTX) {   /* richiesta esplicita o minimo raggiunto */
            fprintf(stderr, "[ds4] impossibile aprire il motore con %d token di contesto (memoria insufficiente?): %s\n", ctx,
                    m->want > 0 ? "riduci --context, oppure omettilo per la scelta automatica" : "nemmeno il minimo entra");
            return -1;
        }
        ctx /= 2; if (ctx < DS4N_MIN_CTX) ctx = DS4N_MIN_CTX;
    }
    m->ctx = ctx;
    if (dspark && !ds4_bridge_has_mtp(m->b))
        fprintf(stderr, "[ds4] ds4_dspark richiesto ma il modello non espone MTP: speculazione inattiva\n");
    fprintf(stderr, "[ds4] contesto %d token (%s)%s%s%s\n", ctx, m->want > 0 ? "richiesto" : "automatico",
            offload ? ", esperti in streaming" : "",
            (offload && ds4n_cache_bytes()) ? ", cache esperti da registro" : "",
            mtp[0] ? ", MTP" : "");
    return 0;
}

static int ds4n_kv_set_max(LlmEngineLlmModel mm, int n) {
    ds4n_model* m = (ds4n_model*)mm;
    if (!m) return -1;
    if (n < 0) n = 0;
    if (m->sessions > 0 && n != m->want) {
        /* Riaprire il motore invaliderebbe le KV delle sessioni vive: si rifiuta, non si rompe. */
        fprintf(stderr, "[ds4] contesto non modificabile con %d sessioni attive: distruggile prima\n", m->sessions);
        return -1;
    }
    if (m->b && n > 0 && n != m->ctx) {
        ds4_bridge_close(m->b); m->b = NULL; m->ctx = 0;
    } else if (m->b && n == 0 && m->want != 0) {
        ds4_bridge_close(m->b); m->b = NULL; m->ctx = 0;
    }
    m->want = n;
    return 0;
}

static LlmEngineLlmSession ds4n_create(LlmEngineLlmModel mm) {
    ds4n_model* m = (ds4n_model*)mm;
    ds4n_session* s;
    if (!m || ds4n_ensure(m) != 0) return NULL;
    s = (ds4n_session*)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->bs = ds4_bridge_session_new(m->b);
    if (!s->bs) { free(s); return NULL; }
    s->m = m;
    s->rng = 0x9e3779b97f4a7c15ULL;
    m->sessions++;
    return (LlmEngineLlmSession)s;
}

static void ds4n_destroy(LlmEngineLlmSession ss) {
    ds4n_session* s = (ds4n_session*)ss;
    if (!s) return;
    if (s->m && s->m->sessions > 0) s->m->sessions--;
    ds4_bridge_session_free(s->bs);
    free(s->toks);
    free(s);
}

static int ds4n_append(LlmEngineLlmSession ss, const int32_t* ids, int n) {
    ds4n_session* s = (ds4n_session*)ss;
    if (!s || !s->bs || !ids || n < 0) return -1;
    if (s->n + n > s->cap) {
        int nc = s->cap ? s->cap : 256; while (nc < s->n + n) nc *= 2;
        int32_t* t = realloc(s->toks, (size_t)nc * sizeof(int32_t)); if (!t) return -1;
        s->toks = t; s->cap = nc;
    }
    memcpy(s->toks + s->n, ids, (size_t)n * sizeof(int32_t)); s->n += n;
    return ds4_bridge_session_sync(s->bs, s->toks, s->n) == 0 ? 0 : -1;   /* ABI: 0 = ok */
}

static int ds4n_generate(LlmEngineLlmSession ss, int max_new, int32_t eos, int32_t* out) {
    ds4n_session* s = (ds4n_session*)ss;
    int k = 0;
    if (!s || !s->bs || max_new <= 0 || !out) return -1;
    for (; k < max_new; k++) {
        int t;
        if (s->has_sp && s->sp.temperature > 0.0f)
            t = ds4_bridge_session_sample(s->bs, s->sp.temperature, s->sp.top_k, s->sp.top_p,
                                          ds4n_min_p(), &s->rng);
        else
            t = ds4_bridge_session_argmax(s->bs);
        if (t < 0) break;
        out[k] = t;
        if (eos >= 0 && t == eos) { k++; break; }
        if (ds4n_append(ss, &out[k], 1) < 0) break;
    }
    return k;
}

static int ds4n_set_sampling(LlmEngineLlmSession ss, const LlmEngineLlmSampling* sp) {
    ds4n_session* s = (ds4n_session*)ss;
    if (!s) return -1;
    if (!sp) { s->has_sp = 0; return 0; }
    s->sp = *sp;
    s->has_sp = 1;
    /* seed 0 = deterministico: lo stesso default del motore MLX (0x9e3779b97f4a7c15). */
    s->rng = (uint64_t)(sp->seed ? sp->seed : 0x9e3779b97f4a7c15ULL);
    /* I tre penalty NON esistono nel sampler di ds4 (`ds4_session_sample` non li prende): si
     * dichiara una volta, non si tace. Gli altri motori li applicano sui logit prima del sampler. */
    if ((sp->repetition_penalty > 1.0f || sp->frequency_penalty != 0.0f ||
         sp->presence_penalty != 0.0f) && !s->warned_penalties) {
        fprintf(stderr, "[ds4] repetition/frequency/presence penalty non supportati da questo motore: ignorati\n");
        s->warned_penalties = 1;
    }
    return 0;
}

static int ds4n_kv_stats(LlmEngineLlmSession ss, int* used, int* cap) {
    ds4n_session* s = (ds4n_session*)ss;
    if (!s || !s->m) return -1;
    if (used) *used = s->n;
    if (cap) *cap = s->m->ctx ? s->m->ctx : s->m->want;
    return 0;
}

/* CAP_KV promette `kv_clear` E `kv_stats`: prima qui c'era solo kv_stats (caps bugiarde).
 * Il clear butta la sessione ds4 e ne crea una nuova: e' l'unico reset che l'upstream permette. */
static int ds4n_kv_clear(LlmEngineLlmSession ss) {
    ds4n_session* s = (ds4n_session*)ss;
    if (!s || !s->bs) return -1;
    if (ds4_bridge_session_reset(s->bs) != 0) return -1;
    s->n = 0;
    return 0;
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
    .set_sampling = ds4n_set_sampling,
    .set_expert_swap = ds4n_set_expert_swap,
    .kv_clear = ds4n_kv_clear,
    .kv_stats = ds4n_kv_stats,
    .kv_set_max = ds4n_kv_set_max,
    .free_str = ds4n_free_str,
};

const LlmEngineLlmEngine* llmengine_llm_engine_ds4n(void) { return &DS4N_ENGINE; }
