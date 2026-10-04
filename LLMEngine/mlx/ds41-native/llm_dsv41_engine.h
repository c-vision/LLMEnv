/* llm_dsv41_engine.h — adattatore "whole-model" DeepSeek V4.1 per il motore MLX.
 *
 * Il forward V4.1 e' whole-model (mHC a 4 stream + cache a 7 slot): non entra nella coppia
 * attn/ffn per-layer di `LlmMlxArchOps`. Questo header espone un carrier opaco
 * (`LlmDsv41Engine`) + step prefill/decode, presi in carico dal motore nei punti di dispatch
 * (load, session_append, session_generate) quando `model_type == "deepseek_v41"`.
 *
 * Usa solo API pubbliche della GPM? No: qui e' codice del motore (stesso processo/stream MLX),
 * NON un bridge C#<->C. Header-only, C puro.
 */
#ifndef LLM_DSV41_ENGINE_H
#define LLM_DSV41_ENGINE_H

#include "llm_dsv41_loader.h"
#include "llm_dsv41_container.h"   /* load dal container GGUF (ggml/RAW) */
#include "llm_dsv41_mlx_cache.h"

typedef struct {
    char name[64];
    int32_t* tokens;
    int n;
    LlmDsv41Cache cache;
} LlmDsv41Prefix;

typedef struct {
    LlmDsv41Weights w;
    int vocab;
    LlmDsv41Prefix* prefixes;
    int n_prefixes, cap;
    LlmDsv41DSpark dspark;   /* pesi MTP/DSpark (se il checkpoint li ha) */
    bool has_dspark;
    bool mtp_enabled;        /* abilita la generazione speculativa MTP */
    long mtp_proposed;       /* blocchi speculati (P[1]==target) */
    long mtp_accepted;       /* token accettati dalle proposte */
    long mtp_rollbacks;      /* rollback cache eseguiti (accept < block) */
    long mtp_prop_ok;        /* forward_spec ha prodotto una proposta */
} LlmDsv41Engine;

static inline LlmDsv41Engine* llm_dsv41_engine_load(const char* dir, mlx_stream s) {
    LlmDsv41Engine* e = (LlmDsv41Engine*)calloc(1, sizeof *e);
    if (!e) return NULL;
    int unloaded = 0;
    if (!llm_dsv41_load_model_dir(&e->w, dir, s, &unloaded)) { free(e); return NULL; }
    e->vocab = e->w.cfg.vocab_size;
    if (e->w.cfg.n_mtp_layers > 0 && e->w.cfg.n_dspark_targets > 0) {
        e->has_dspark = llm_dsv41_load_dspark(&e->dspark, &e->w.cfg, dir, s);
    }
    return e;
}

/* Carica l'engine dal CONTAINER `GGUF` (ggml/RAW). `cfg_dir` = dir HSF da cui leggere config.json
 * (il container non lo contiene). MTP/DSpark (container separato) NON caricato qui. */
static inline LlmDsv41Engine* llm_dsv41_engine_load_container(const char* cfg_dir,
                                                              const char* container_path, mlx_stream s) {
    LlmDsv41Engine* e = (LlmDsv41Engine*)calloc(1, sizeof *e);
    if (!e) return NULL;
    int unloaded = 0;
    if (!llm_dsv41_load_container_dir(&e->w, cfg_dir, container_path, s, &unloaded)) { free(e); return NULL; }
    e->vocab = e->w.cfg.vocab_size;
    return e;
}


/* Carica l'engine in modalita' STREAMING dal container: embed/norm/head caricati una volta,
 * i layer letti on-demand (RAM bounded). Il forward usa llm_dsv41_mlx_L(w,l) (vedi llm_dsv41_mlx.h). */
static inline LlmDsv41Engine* llm_dsv41_engine_load_container_stream(const char* cfg_dir,
                                                                     const char* container_path, mlx_stream s) {
    LlmDsv41Engine* e = (LlmDsv41Engine*)calloc(1, sizeof *e);
    if (!e) return NULL;
    LlmDsv41Stream* st = (LlmDsv41Stream*)calloc(1, sizeof(LlmDsv41Stream));
    if (!st || !llm_dsv41_stream_open(st, cfg_dir, container_path)) { free(st); free(e); return NULL; }
    bool ok = true;
    e->w.cfg    = st->cfg;
    e->w.stream = st;
    e->w.embed  = llm_dsv41_cont_dense(&st->g, "embed.weight", &ok);
    e->w.norm   = llm_dsv41_cont_dense(&st->g, "norm.weight", &ok);
    e->w.head   = llm_dsv41_cont_dense(&st->g, "head.weight", &ok);
    if (!ok) { mlx_array_free(e->w.embed); mlx_array_free(e->w.norm); mlx_array_free(e->w.head);
               llm_dsv41_stream_close(st); free(st); free(e); return NULL; }
    e->vocab = e->w.cfg.vocab_size;
    return e;
}

static inline void llm_dsv41_engine_free(LlmDsv41Engine* e) {
    if (!e) return;
    for (int i = 0; i < e->n_prefixes; i++) {
        free(e->prefixes[i].tokens);
        llm_dsv41_cache_free(&e->prefixes[i].cache);
    }
    free(e->prefixes);
    if (e->has_dspark) llm_dsv41_dspark_free(&e->dspark);
    if (e->w.stream) {   /* STREAMING: embed/norm/head + lo stream (i layer non sono materializzati) */
        mlx_array_free(e->w.embed); mlx_array_free(e->w.norm); mlx_array_free(e->w.head);
        llm_dsv41_stream_close((LlmDsv41Stream*)e->w.stream);
        free(e->w.stream); e->w.stream = NULL;
    }
    llm_dsv41_weights_free(&e->w);
    free(e);
}

/* ── prefix cache (snapshot del cache 7-slot) ─────────────────────────────── */
static inline void llm_dsv41_cache_clone(LlmDsv41Cache* dst, const LlmDsv41Cache* src) {
    llm_dsv41_cache_init(dst);
    for (int l = 0; l < LLM_DSV41_MAX_LAYERS; l++)
        for (int i = 0; i < 7; i++)
            if (src->layers[l].present[i]) {
                mlx_array_set(&dst->layers[l].slot[i], src->layers[l].slot[i]);
                dst->layers[l].present[i] = true;
            }
    dst->offset = src->offset;
    if (src->history && src->history_n > 0) {
        dst->history = (int64_t*)malloc((size_t)src->history_n * sizeof(int64_t));
        memcpy(dst->history, src->history, (size_t)src->history_n * sizeof(int64_t));
        dst->history_n = src->history_n;
    }
    if (src->last_logits && src->last_logits_n > 0) {
        dst->last_logits = (float*)malloc((size_t)src->last_logits_n * sizeof(float));
        memcpy(dst->last_logits, src->last_logits, (size_t)src->last_logits_n * sizeof(float));
        dst->last_logits_n = src->last_logits_n;
    }
    if (src->dspark_ctx && src->n_dspark_ctx > 0) {
        dst->dspark_ctx = (LlmDsv41DSparkCache*)calloc((size_t)src->n_dspark_ctx, sizeof(LlmDsv41DSparkCache));
        if (dst->dspark_ctx) {
            dst->n_dspark_ctx = src->n_dspark_ctx;
            for (int i = 0; i < src->n_dspark_ctx; i++) {
                dst->dspark_ctx[i].max_size = src->dspark_ctx[i].max_size;
                dst->dspark_ctx[i].offset = src->dspark_ctx[i].offset;
                dst->dspark_ctx[i].has = src->dspark_ctx[i].has;
                if (src->dspark_ctx[i].has)
                    mlx_array_set(&dst->dspark_ctx[i].keys, src->dspark_ctx[i].keys);
            }
        }
    }
}

static inline bool llm_dsv41_engine_save_prefix(LlmDsv41Engine* e, const char* name,
                                                const LlmDsv41Cache* cache, const int32_t* tokens, int n) {
    if (!cache || !tokens || n <= 0) return false;
    int idx = -1;
    for (int i = 0; i < e->n_prefixes; i++) if (strcmp(e->prefixes[i].name, name) == 0) { idx = i; break; }
    if (idx < 0) {
        if (e->n_prefixes >= e->cap) {
            int nc = e->cap ? e->cap * 2 : 4;
            LlmDsv41Prefix* np = (LlmDsv41Prefix*)realloc(e->prefixes, (size_t)nc * sizeof(LlmDsv41Prefix));
            if (!np) return false;
            e->prefixes = np; e->cap = nc;
        }
        idx = e->n_prefixes++;
        memset(&e->prefixes[idx], 0, sizeof e->prefixes[idx]);
    } else {
        free(e->prefixes[idx].tokens);
        llm_dsv41_cache_free(&e->prefixes[idx].cache);
        e->prefixes[idx].tokens = NULL;
    }
    LlmDsv41Prefix* pe = &e->prefixes[idx];
    snprintf(pe->name, sizeof pe->name, "%s", name);
    pe->tokens = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    memcpy(pe->tokens, tokens, (size_t)n * sizeof(int32_t));
    pe->n = n;
    llm_dsv41_cache_clone(&pe->cache, cache);
    return true;
}

/* miglior match = prefisso salvato piu' lungo di `tokens`. Ritorna la lunghezza d (0 se nessuno). */
static inline int llm_dsv41_engine_match_prefix(const LlmDsv41Engine* e, const int32_t* tokens, int n) {
    int best = 0;
    for (int i = 0; i < e->n_prefixes; i++) {
        const LlmDsv41Prefix* pe = &e->prefixes[i];
        int d = pe->n < n ? pe->n : n;
        int k = 0; while (k < d && pe->tokens[k] == tokens[k]) k++;
        if (k > best) best = k;
    }
    return best;
}

static inline int32_t llm_dsv41_argmax(const float* lg, int n) {
    int bi = 0;
    for (int i = 1; i < n; i++) if (lg[i] > lg[bi]) bi = i;
    return (int32_t)bi;
}

/* Parametri di sampling della sessione (mirror di llm_mlx_sampling_t, senza dipendere dal .c).
 * NOTE: sampler standard host-side (temperature/top-k/top-p + penalita'); NON garantito
 * bit-identico al sampler di final_step (RNG diverso). temperature<=0 o active=false -> greedy. */
typedef struct {
    bool active;
    float temperature, top_p;
    int top_k;
    int64_t seed;
    float repetition_penalty, frequency_penalty, presence_penalty;
    const int32_t* hist;
    int hist_n;
} LlmDsv41Sampling;

static inline uint64_t llm_dsv41_rng(uint64_t* st) {
    uint64_t x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    return x;
}

static double* llm_dsv41_sort_p;
static int llm_dsv41_sort_cmp(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    double pa = llm_dsv41_sort_p[ia], pb = llm_dsv41_sort_p[ib];
    if (pa != pb) return pa > pb ? -1 : 1;
    return ia < ib ? -1 : (ia > ib ? 1 : 0);
}

static inline int32_t llm_dsv41_sample(const float* lg, int n, const LlmDsv41Sampling* sm) {
    if (!sm || !sm->active || sm->temperature <= 0.0f) return llm_dsv41_argmax(lg, n);
    float temp = sm->temperature > 1e-5f ? sm->temperature : 1e-5f;
    double* p = (double*)malloc((size_t)n * sizeof(double));
    int* cnt = (int*)calloc((size_t)n, sizeof(int));
    int* ord = (int*)malloc((size_t)n * sizeof(int));
    if (!p || !cnt || !ord) { free(p); free(cnt); free(ord); return llm_dsv41_argmax(lg, n); }
    for (int i = 0; i < n; i++) { p[i] = (double)lg[i]; ord[i] = i; }
    if (sm->hist && sm->hist_n > 0) {
        for (int j = 0; j < sm->hist_n; j++) { int t = sm->hist[j]; if (t >= 0 && t < n) cnt[t]++; }
        for (int i = 0; i < n; i++) if (cnt[i]) {
            if (sm->repetition_penalty > 0.0f && sm->repetition_penalty != 1.0f)
                p[i] = (p[i] > 0.0) ? p[i] / sm->repetition_penalty : p[i] * sm->repetition_penalty;
            p[i] -= (double)sm->frequency_penalty * (double)cnt[i];
            p[i] -= (double)sm->presence_penalty;
        }
    }
    double mx = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mx) mx = p[i];
    for (int i = 0; i < n; i++) p[i] = exp((p[i] - mx) / (double)temp);
    llm_dsv41_sort_p = p;
    qsort(ord, (size_t)n, sizeof(int), llm_dsv41_sort_cmp);
    int keep = n;
    if (sm->top_k > 0 && sm->top_k < keep) keep = sm->top_k;
    if (sm->top_p > 0.0f && sm->top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < keep; i++) sum += p[ord[i]];
        double cum = 0.0; int k = 0;
        for (; k < keep; k++) { cum += p[ord[k]]; if (cum >= (double)sm->top_p * sum) { k++; break; } }
        keep = k > 0 ? k : 1;
    }
    double sum = 0.0;
    for (int i = 0; i < keep; i++) sum += p[ord[i]];
    uint64_t st = (uint64_t)sm->seed ^ ((uint64_t)(sm->hist_n + 1) * 0x9E3779B97F4A7C15ULL);
    if (st == 0) st = 0x9E3779B97F4A7C15ULL;
    double r = ((double)(llm_dsv41_rng(&st) >> 11)) * (1.0 / 9007199254740992.0) * sum;
    double acc = 0.0; int32_t pick = (int32_t)ord[0];
    for (int i = 0; i < keep; i++) { acc += p[ord[i]]; if (r <= acc) { pick = (int32_t)ord[i]; break; } }
    free(p); free(cnt); free(ord);
    return pick;
}


/* Processa `ids[n]` con la cache persistente (prefill se nuova, altrimenti decode/chunk).
 * Aggiorna `*cache_offset` e ritorna l'argmax dell'ultima riga. */
static inline bool llm_dsv41_engine_step(LlmDsv41Engine* e, LlmDsv41Cache** cache,
                                         const int32_t* ids, int n, int32_t* out_next,
                                         int* cache_offset, const LlmDsv41Sampling* sm,
                                         const int* cap_layers, int n_cap, mlx_array* out_capture,
                                         mlx_stream s) {
    if (!*cache) {
        *cache = (LlmDsv41Cache*)malloc(sizeof **cache);
        if (!*cache) return false;
        llm_dsv41_cache_init(*cache);
    }
    mlx_array lg = mlx_array_new();
    if (!llm_dsv41_mlx_forward_cached(&e->w, *cache, ids, n, &lg, cap_layers, n_cap, out_capture, s)) {
        mlx_array_free(lg);
        return false;
    }
    mlx_array_eval(lg);
    const float* p = mlx_array_data_float32(lg);
    if (!p) { mlx_array_free(lg); return false; }
    LlmDsv41Cache* c0 = *cache;
    /* memorizza i logits dell'ultima riga (posizione pending) per generate_constrained */
    if (!c0->last_logits || c0->last_logits_n < e->vocab) {
        free(c0->last_logits);
        c0->last_logits = (float*)malloc((size_t)e->vocab * sizeof(float));
        c0->last_logits_n = e->vocab;
    }
    memcpy(c0->last_logits, p + (size_t)(n - 1) * (size_t)e->vocab,
           (size_t)e->vocab * sizeof(float));
    *out_next = llm_dsv41_sample(c0->last_logits, e->vocab, sm);
    mlx_array_free(lg);
    *cache_offset += n;
    return true;
}

/* Variante per MTP: ritorna TUTTI i logits (mlx_array [1,n,vocab]) e la cattura hidden opzionale. */
static inline bool llm_dsv41_engine_step_logits(LlmDsv41Engine* e, LlmDsv41Cache** cache,
                                                const int32_t* ids, int n, int* cache_offset,
                                                mlx_array* out_logits, mlx_array* out_cap,
                                                mlx_stream s) {
    if (!*cache) {
        *cache = (LlmDsv41Cache*)malloc(sizeof **cache);
        if (!*cache) return false;
        llm_dsv41_cache_init(*cache);
    }
    if (!llm_dsv41_mlx_forward_cached(&e->w, *cache, ids, n, out_logits,
                                      out_cap ? e->w.cfg.dspark_target_layer_ids : NULL,
                                      out_cap ? e->w.cfg.n_dspark_targets : 0, out_cap, s))
        return false;
    *cache_offset += n;
    return true;
}

/* aggiorna `c->last_logits` (host) con la riga `row` dei logits forniti. */
static inline void llm_dsv41_engine_set_last_logits(LlmDsv41Engine* e, LlmDsv41Cache* c,
                                                    mlx_array lg, int row) {
    mlx_array_eval(lg);
    const float* p = mlx_array_data_float32(lg);
    if (!p) return;
    if (!c->last_logits || c->last_logits_n < e->vocab) {
        free(c->last_logits);
        c->last_logits = (float*)malloc((size_t)e->vocab * sizeof(float));
        c->last_logits_n = e->vocab;
    }
    memcpy(c->last_logits, p + (size_t)row * (size_t)e->vocab, (size_t)e->vocab * sizeof(float));
}

/* session_append: prefill/decode del chunk appena appeso; setta il prossimo token greedy. */
static inline int llm_dsv41_engine_append(void* est, void** cache, const int32_t* ids, int n,
                                          int* cache_offset, int32_t* pending, bool* has_pending,
                                          const LlmDsv41Sampling* sm, mlx_stream s) {
    int32_t next = 0;
    if (!llm_dsv41_engine_step((LlmDsv41Engine*)est, (LlmDsv41Cache**)cache, ids, n, &next,
                               cache_offset, sm, NULL, 0, NULL, s))
        return -1;
    *pending = next;
    *has_pending = true;
    return 0;
}

/* Speculazione batch testabile: dati `next` (token target dopo il pending) e la proposta
 * `P[0..k]` (P[0]=anchor, P[1..k] proposta), verifica con UN forward target, accetta il prefisso
 * e (se accept<k) fa rollback della cache ricommittando i soli accettati. Emette i token accettati
 * in `out_ids`, ritorna quanti e setta `*out_pending`. `room` = spazio rimasto. */
static inline int llm_dsv41_engine_mtp_speculate(LlmDsv41Engine* e, LlmDsv41Cache** cache,
        int* cache_offset, int32_t next, const int32_t* P, int k, int room, int32_t eos,
        int32_t* out_ids, int32_t* out_pending, mlx_stream s) {
    const int V = e->vocab;
    LlmDsv41Cache* c = (LlmDsv41Cache*)*cache;
    if (k < 1 || room <= 0) { *out_pending = next; return 0; }
    if (k > 64) k = 64;
    int32_t B[64];
    for (int i = 0; i < k; i++) B[i] = P[i + 1];
    LlmDsv41Cache* snap = (LlmDsv41Cache*)malloc(sizeof *snap);
    if (!snap) { *out_pending = next; return 0; }
    llm_dsv41_cache_clone(snap, c);
    int snap_off = *cache_offset;
    mlx_array Lb = mlx_array_new(), mhb = mlx_array_new();
    if (!llm_dsv41_engine_step_logits(e, cache, B, k, cache_offset, &Lb, &mhb, s)) {
        mlx_array_free(Lb); mlx_array_free(mhb); llm_dsv41_cache_free(snap); free(snap);
        *out_pending = next; return 0;
    }
    mlx_array_eval(Lb);
    const float* lp = mlx_array_data_float32(Lb);
    int accept = 1;
    for (int i = 1; i < k; i++) {
        int32_t t = llm_dsv41_argmax(lp + (size_t)(i - 1) * V, V);
        if (P[i + 1] == t) accept++; else break;
    }
    int nacc = 0; bool hit_eos = false;
    for (int i = 1; i <= accept; i++) {
        int32_t t = P[i];
        if (eos >= 0 && t == eos) { hit_eos = true; break; }
        if (nacc >= room) break;
        out_ids[nacc++] = t;
    }
    if (accept == k && nacc == k && !hit_eos) {
        llm_dsv41_engine_set_last_logits(e, c, Lb, k - 1);
        *out_pending = llm_dsv41_argmax(lp + (size_t)(k - 1) * V, V);
        e->mtp_accepted += nacc;
        llm_dsv41_cache_free(snap); free(snap);
        mlx_array_free(Lb); mlx_array_free(mhb);
        return nacc;
    }
    /* rollback */
    llm_dsv41_cache_free(c);
    llm_dsv41_cache_clone(c, snap);
    *cache_offset = snap_off;
    llm_dsv41_cache_free(snap); free(snap);
    if (nacc > 0) {
        mlx_array L2 = mlx_array_new(), mh2 = mlx_array_new();
        if (llm_dsv41_engine_step_logits(e, cache, B, nacc, cache_offset, &L2, &mh2, s))
            llm_dsv41_engine_set_last_logits(e, c, L2, nacc - 1);
        mlx_array_free(L2); mlx_array_free(mh2);
    }
    int idx = (nacc < accept ? nacc : accept) - 1;
    *out_pending = hit_eos ? eos : (idx >= 0 ? llm_dsv41_argmax(lp + (size_t)idx * V, V) : next);
    e->mtp_accepted += nacc; e->mtp_rollbacks++;
    mlx_array_free(Lb); mlx_array_free(mhb);
    return nacc;
}

/* Generazione speculativa MTP: propone un blocco DSpark e chiama `_speculate` (batch + rollback).
 * Output IDENTICO al greedy (parità); in non-greedy fa fallback token-per-token. */
static inline int llm_dsv41_engine_mtp_generate(void* est, void** cache, int* cache_offset,
                                                int32_t current_id, bool has_pending, int max_new,
                                                int32_t eos, int32_t* out_ids,
                                                const LlmDsv41Sampling* sm, mlx_stream s) {
    LlmDsv41Engine* e = (LlmDsv41Engine*)est;
    LlmDsv41Cache* c = (LlmDsv41Cache*)*cache;
    const int V = e->vocab;
    if (!e->has_dspark || !has_pending || max_new <= 0 || !out_ids || !c) return -1;
    if (!c->dspark_ctx) {
        c->n_dspark_ctx = e->dspark.n_stages;
        c->dspark_ctx = (LlmDsv41DSparkCache*)calloc((size_t)c->n_dspark_ctx, sizeof(LlmDsv41DSparkCache));
        if (!c->dspark_ctx) return -1;
        for (int i = 0; i < c->n_dspark_ctx; i++) c->dspark_ctx[i].max_size = e->w.cfg.window_size;
    }
    const int* tgt = e->w.cfg.dspark_target_layer_ids;
    const int nt = e->w.cfg.n_dspark_targets;
    const int block = e->w.cfg.dspark_block_size;
    const bool greedy = (!sm || !sm->active || sm->temperature <= 0.0f);
    int count = 0;
    while (count < max_new) {
        if (eos >= 0 && current_id == eos) break;
        out_ids[count++] = current_id;
        mlx_array lg = mlx_array_new(), mh = mlx_array_new();
        if (!llm_dsv41_engine_step_logits(e, (LlmDsv41Cache**)cache, &current_id, 1, cache_offset,
                                          &lg, &mh, s)) { mlx_array_free(lg); mlx_array_free(mh); return -1; }
        llm_dsv41_engine_set_last_logits(e, *cache, lg, 0);
        int32_t next = llm_dsv41_sample(c->last_logits, V, sm);
        if (count >= max_new || !greedy || block <= 1) {
            mlx_array_free(lg); mlx_array_free(mh); current_id = next; continue;
        }
        mlx_array ids = mlx_array_new_data(&current_id, (const int[]){ 1, 1 }, 2, MLX_INT32);
        mlx_array op = mlx_array_new(), ob = mlx_array_new(), oc = mlx_array_new();
        bool proposed = llm_dsv41_dspark_forward_spec(&e->dspark, ids, mh, c->dspark_ctx, &op, &ob, &oc, s);
        mlx_array_free(ids); mlx_array_free(mh); mlx_array_free(lg);
        if (!proposed) { current_id = next; continue; }
        mlx_array_eval(op);
        const int32_t* P = mlx_array_data_int32(op);
        e->mtp_prop_ok++;
        if (P[1] != next) { mlx_array_free(op); mlx_array_free(ob); mlx_array_free(oc); current_id = next; continue; }
        e->mtp_proposed++;
        int room = max_new - count;
        int32_t pend = next;
        int em = llm_dsv41_engine_mtp_speculate(e, (LlmDsv41Cache**)cache, cache_offset, next, P,
                                                block, room, eos, out_ids + count, &pend, s);
        count += em; current_id = pend;
        mlx_array_free(op); mlx_array_free(ob); mlx_array_free(oc);
    }
    return count;
}

/* session_generate: emette `current_id` e lo decodifica per il prossimo, in greedy. */
static inline int llm_dsv41_engine_generate(void* est, void** cache, int* cache_offset,
                                            int32_t current_id, bool has_pending, int max_new,
                                            int32_t eos, int32_t* out_ids,
                                            const LlmDsv41Sampling* sm, mlx_stream s) {
    LlmDsv41Engine* e0 = (LlmDsv41Engine*)est;
    if (e0->mtp_enabled && e0->has_dspark)
        return llm_dsv41_engine_mtp_generate(est, cache, cache_offset, current_id, has_pending,
                                             max_new, eos, out_ids, sm, s);
    if (!has_pending || max_new <= 0 || out_ids == NULL) return -1;
    LlmDsv41Engine* e = (LlmDsv41Engine*)est;
    int count = 0;
    for (int step = 0; step < max_new; step++) {
        if (eos >= 0 && current_id == eos) break;
        out_ids[count++] = current_id;
        int32_t next = 0;
        if (!llm_dsv41_engine_step(e, (LlmDsv41Cache**)cache, &current_id, 1, &next, cache_offset, sm,
                                   NULL, 0, NULL, s))
            return -1;
        current_id = next;
    }
    return count;
}

/* generate_constrained: decide UN token alla posizione pending con argmax ristretto ad
 * `allowed`, poi lo decodifica (avanza la cache) e calcola il nuovo pending. */
static inline bool llm_dsv41_engine_constrained(void* est, void** cache, int* cache_offset,
                                                const int32_t* allowed, int n_allowed,
                                                int32_t* out_id, int32_t* out_next,
                                                const LlmDsv41Sampling* sm, mlx_stream s) {
    LlmDsv41Engine* e = (LlmDsv41Engine*)est;
    LlmDsv41Cache* c = (LlmDsv41Cache*)(*cache);
    if (!c || !c->last_logits || n_allowed <= 0) return false;
    int V = e->vocab;
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    unsigned char* okmask = (unsigned char*)calloc((size_t)V, 1);
    if (!lg || !okmask) { free(lg); free(okmask); return false; }
    memcpy(lg, c->last_logits, (size_t)V * sizeof(float));
    for (int i = 0; i < n_allowed; i++) { int t = allowed[i]; if (t >= 0 && t < V) okmask[t] = 1; }
    for (int i = 0; i < V; i++) if (!okmask[i]) lg[i] = -INFINITY;
    int32_t chosen = llm_dsv41_argmax(lg, V);
    free(lg); free(okmask);
    if (!llm_dsv41_engine_step(e, (LlmDsv41Cache**)cache, &chosen, 1, out_next, cache_offset, sm, NULL, 0, NULL, s))
        return false;
    *out_id = chosen;
    return true;
}

/* append con riuso del prefisso salvato piu' lungo (>= min_match). */
static inline int llm_dsv41_engine_append_auto_prefix(LlmDsv41Engine* e, void** cache,
        const int32_t* tokens, int n, int min_match, int* cache_offset, int32_t* pending,
        bool* has_pending, const LlmDsv41Sampling* sm, mlx_stream s) {
    if (!*cache) {
        *cache = (LlmDsv41Cache*)malloc(sizeof(LlmDsv41Cache));
        if (!*cache) return -1;
        llm_dsv41_cache_init((LlmDsv41Cache*)*cache);
    }
    LlmDsv41Cache* c = (LlmDsv41Cache*)*cache;
    if (c->offset == 0 && e->n_prefixes > 0) {
        int d = llm_dsv41_engine_match_prefix(e, tokens, n);
        if (d >= min_match) {
            for (int i = 0; i < e->n_prefixes; i++) {
                const LlmDsv41Prefix* pe = &e->prefixes[i];
                int dd = pe->n < n ? pe->n : n, k = 0;
                while (k < dd && pe->tokens[k] == tokens[k]) k++;
                if (k == d && pe->cache.offset == d) {
                    llm_dsv41_cache_free(c);
                    llm_dsv41_cache_clone(c, &pe->cache);
                    break;
                }
            }
        }
    }
    int start = c->offset;
    *cache_offset = start;
    if (start >= n) {
        if (c->last_logits) { *pending = llm_dsv41_sample(c->last_logits, e->vocab, sm); *has_pending = true; return 0; }
        return -1;
    }
    int32_t next = 0;
    if (!llm_dsv41_engine_step(e, (LlmDsv41Cache**)cache, tokens + start, n - start, &next,
                               cache_offset, sm, NULL, 0, NULL, s))
        return -1;
    *pending = next;
    *has_pending = true;
    return 0;
}

/* rebuild del cache da una sequenza (per park/unpark/swap_in). */
static inline bool llm_dsv41_engine_prefill(LlmDsv41Engine* e, void** cache, const int32_t* tokens,
                                            int n, int* cache_offset, int32_t* pending,
                                            bool* has_pending, const LlmDsv41Sampling* sm,
                                            mlx_stream s) {
    if (n <= 0) return false;
    LlmDsv41Cache* c = (LlmDsv41Cache*)*cache;
    if (c) { llm_dsv41_cache_free(c); llm_dsv41_cache_init(c); }
    else { *cache = (LlmDsv41Cache*)malloc(sizeof(LlmDsv41Cache)); if (!*cache) return false; llm_dsv41_cache_init((LlmDsv41Cache*)*cache); }
    *cache_offset = 0;
    int32_t next = 0;
    if (!llm_dsv41_engine_step(e, (LlmDsv41Cache**)cache, tokens, n, &next, cache_offset, sm, NULL, 0, NULL, s))
        return false;
    *pending = next;
    *has_pending = true;
    return true;
}

#endif /* LLM_DSV41_ENGINE_H */
