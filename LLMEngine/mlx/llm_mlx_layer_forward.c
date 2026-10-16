/* ============================================================================
 * Dominio KV-quant/pagesel + layer_forward: g_kvq_cold, llm_mlx_kvq_flush_batch, llm_kvq_sdpa_hook_dispatch, llm_kvq_append_cold, llm_pagesel_ensure/update/sdpa_dispatch, llm_mlx_set_layer_ops, layer_forward.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_kv_quant_sdpa_kernel.h"
#include "llm_mlx_kvq_splash.h"   /* KV-quant port da Splash (store/dequant) */
#include "llm_mlx_kvq_pool.h"     /* page pool int8 (storage primario cold) */
#include "llm_mlx_flash_decode_kernel.h"
#include "llm_mlx_fused_rope_append.h"
#include "llm_mlx_execplan.h"   /* §9.A M10 EXECUTION PLANNER (runtime manager) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* A/B bit-level GDN (LLM_GDN_AB): tag di run per il dump per-layer di h. NULL = spento. */
const char* g_layerh_run = NULL;

/* Definizioni UNICHE degli hook SDPA (dichiarati extern in llm_mlx_ops.h).
 * Prima erano `static` per-TU -> l'hook installato qui non era visibile agli archi
 * attention (altre TU) e non veniva mai invocato. */
llm_sdpa_hook_slot llm_kvq_hook = { NULL, NULL };
llm_sdpa_hook_slot llm_pagesel_hook = { NULL, NULL };
llm_sdpa_hook_slot llm_train_sdpa_hook = { NULL, NULL };

/* (B7, 2026-09-30) Il registry `g_kvq_pools` e' `static` NELL'HEADER llm_mlx_kvq_pool.h -> una
 * copia PER TU. Il pool di generazione vive in QUESTA TU (layer_forward.c); session_destroy/_reset
 * (altra TU, llm_mlx_session_lifecycle.c) non poteva raggiungerlo -> il pool non veniva MAI
 * liberato (byte/token cold trattenuti per sessione) e il cap di 8 si esauriva (oltre 8 sessioni:
 * `llm_kvq_pool_for` ritornava NULL e la KV-quant si spegneva IN SILENZIO). Queste due funzioni
 * non-static sono l'unico punto d'accesso dal lifecycle. */
void llm_mlx_kvq_pool_release_session(void* model, void* sess) {
    for (int i = 0; i < g_kvq_pools_n; i++) {
        if (g_kvq_pools[i].model == model && g_kvq_pools[i].sess == sess) {
            llm_kvq_pool_free(&g_kvq_pools[i].p);
            g_kvq_pools[i] = g_kvq_pools[--g_kvq_pools_n];
            memset(&g_kvq_pools[g_kvq_pools_n], 0, sizeof(g_kvq_pools[0]));
            return;
        }
    }
}
void llm_mlx_kvq_pool_reset_session(void* model, void* sess) {
    for (int i = 0; i < g_kvq_pools_n; i++)
        if (g_kvq_pools[i].model == model && g_kvq_pools[i].sess == sess) {
            llm_kvq_pool* p = &g_kvq_pools[i].p;
            if (p->committed_l) for (int l = 0; l < p->layers; l++) p->committed_l[l] = 0;
            return;
        }
}

/* ── LLM_KV_DEQUANT_READ (2026-09-13): attenzione decode che legge la KV cold QUANTIZZATA
 * direttamente (kernel Metal dequant-on-read, llm_mlx_kv_quant_sdpa_kernel.h), senza
 * dequant+concat. Il cold e' "a blocchi": a ogni step si quantizza SOLO il token che scade
 * dalla finestra e lo si concatena ai quantizzati persistenti (nessun dequant del cold).
 * Solo decode (n_tokens==1): il prefill usa il path normale. Default OFF. */
static LlmKvqCold g_kvq_cold = { {0}, {0}, {0}, {0}, {0}, {0}, 0 };

/* CAUSA MISURATA 2026-09-17 (AiWiki "MLX-KVQuant-Contesto-Lungo-5x-Peggiore"): ogni
 * llm_kvq_append_cold fa un mlx_concatenate_axis del buffer cold ESISTENTE (old_nc token)
 * con quello nuovo -- gli array MLX sono funzionali, quindi ogni concat ricopia l'INTERO
 * buffer (O(old_nc) per 6 array: weight/scale/bias di K e V). Prima di questo fix
 * scattava a OGNI step di decode (n_new=1 appena la finestra recente supera la soglia) --
 * stesso pattern gia' diagnosticato e risolto per la cache fp16 da LLM_KVSEG (vedi
 * commento ~riga 167): "anche col buffer fisso, la scrittura ricopia l'intero buffer,
 * O(pos) per token". Qui la cura e' piu' semplice: NON serve una cache segmentata separata
 * (main+segmento) come LLM_KVSEG -- basta lasciare che la finestra "recente" FP16 (gia'
 * letta dal kernel dequant-on-read senza quantizzazione, llm_mlx_kv_quant_sdpa_kernel.h)
 * cresca oltre la soglia di LLM_KVQ_FLUSH_BATCH token prima di quantizzare+concatenare
 * TUTTI insieme in un solo colpo, invece che uno alla volta -- riduce il numero di concat
 * O(nc) di un fattore LLM_KVQ_FLUSH_BATCH, a costo di LLM_KVQ_FLUSH_BATCH-1 token in piu'
 * (bounded, non cresce col contesto) nel loop "recente" del kernel, che e' gia' O(soglia)
 * costante oggi. Override: LLM_KVQ_FLUSH_BATCH. */
static int llm_mlx_kvq_flush_batch(void) {
    static int v = 0;
    if (v == 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_KVQ_FLUSH_BATCH");
        v = (e != NULL && atoi(e) > 0) ? atoi(e) : 64;
    }
    return v;
}

static bool llm_kvq_sdpa_hook_dispatch(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                       float scale, void* ctx, mlx_stream s) {
    LlmKvqCold* c = (LlmKvqCold*)ctx;
    if (c->nc <= 0 || c->kw.ctx == NULL) return false;
    if (mlx_array_ndim(q) != 4 || mlx_array_ndim(k) != 4 || mlx_array_ndim(v) != 4) return false;
    int nh = mlx_array_shape(q)[1], lq = mlx_array_shape(q)[2], hd = mlx_array_shape(q)[3];
    /* (B) Splash P0 — DECISIONE 2026-10-01: il port `splash_attn` (kernel JIT, V dim-major, solo-pool,
     * tronca a 256 token) NON e' production: la parity gate e' fallita (broadcast (1,8,512,64) vs
     * (1,8,1335,64); non fonde la finestra recente fp16). Il kernel AOT `llm_mlx_kvq_sdpa_decode`
     * (dequant-on-read, online-softmax, scala folded, modalita' Sym Splash, split-K) e' la soluzione
     * COMPLETA -> resta la produzione. Il port non e' piu' cablato qui (resta solo come riferimento/
     * selftest in `llm_mlx_cb.c`). Vedi `Todo/(open) - Splash-Port-...` e `Kernel-Metal-Da-Portare`
     * se si decidera' di completarlo (fondere recente + tiling >256). */
    int nkv = mlx_array_shape(k)[1], nr = mlx_array_shape(k)[2];
    if (getenv("LLM_KVQ_DBG"))
        fprintf(stderr, "[KVQHOOK] nc(cold)=%d nr(recent)=%d kcap=%d nh=%d hd=%d\n",
                (int)c->nc, nr, (int)mlx_array_shape(c->kw)[1], nh, hd);
    if (hd % 64 != 0) return false;
    if (mlx_array_dtype(q) != mlx_array_dtype(k) || mlx_array_dtype(q) != mlx_array_dtype(v)) return false;
    mlx_array q2 = mlx_array_new(), k2 = mlx_array_new(), v2 = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    int qsh[3] = {nh, lq, hd}, ksh[3] = {nkv, nr, hd};
    int kcap = (int)mlx_array_shape(c->kw)[1];                 /* capacita' del pool (K) */
    int cw[3] = {nkv, kcap, hd}, cs[3] = {nkv, kcap, 1};       /* Splash: int8 byte + 1 scala/token */
    int cwv[3] = {nkv, kcap, hd};
    bool ok = mlx_reshape(&q2, q, qsh, 3, s) == 0 &&
              mlx_reshape(&k2, k, ksh, 3, s) == 0 &&
              mlx_reshape(&v2, v, ksh, 3, s) == 0 &&
              mlx_reshape(&kw, c->kw, cw, 3, s) == 0 &&
              mlx_reshape(&ks, c->ks, cs, 3, s) == 0 &&
              mlx_reshape(&kb, c->kb, cs, 3, s) == 0 &&
              mlx_reshape(&vw, c->vw, cwv, 3, s) == 0 &&
              mlx_reshape(&vs, c->vs, cs, 3, s) == 0 &&
              mlx_reshape(&vb, c->vb, cs, 3, s) == 0;
    mlx_array o2 = mlx_array_new();
    if (ok) ok = llm_mlx_kvq_sdpa_decode(&o2, q2, kw, ks, kb, vw, vs, vb, k2, v2, c->nc, c->kbits, c->vbits, scale, hd, c->ptab, s);
    if (ok) {
        mlx_array o2c = mlx_array_new();
        ok = (mlx_array_dtype(o2) == mlx_array_dtype(q)) ? (mlx_array_set(&o2c, o2) == 0)
                                                        : (mlx_astype(&o2c, o2, mlx_array_dtype(q), s) == 0);
        if (ok) { int osh[4] = {1, nh, lq, hd}; ok = mlx_reshape(out, o2c, osh, 4, s) == 0; }
        mlx_array_free(o2c);
    }
    mlx_array_free(q2); mlx_array_free(k2); mlx_array_free(v2);
    mlx_array_free(kw); mlx_array_free(ks); mlx_array_free(kb);
    mlx_array_free(vw); mlx_array_free(vs); mlx_array_free(vb); mlx_array_free(o2);
    return ok;
}

/* Appende i primi n_new token di `cache` ([1,nkv,N,hd]) ai quantizzati persistenti w/sc/bi
 * (concat lungo l'asse token, NESSUN dequant). Ritorna il nuovo n_cold. */
static int llm_kvq_append_cold(mlx_array* w, mlx_array* sc, mlx_array* bi, int old_nc,
                               mlx_array cache, int n_new, int bits, int gs, mlx_stream s) {
    if (n_new <= 0) return old_nc;
    int nkv = mlx_array_shape(cache)[1], hd = mlx_array_shape(cache)[3];
    int s0[4] = {0,0,0,0}, s1[4] = {1,nkv,n_new,hd}, st[4] = {1,1,1,1};
    mlx_array part = mlx_array_new();
    if (mlx_slice(&part, cache, s0, 4, s1, 4, st, 4, s) != 0) { mlx_array_free(part); return old_nc; }
    mlx_vector_array q = mlx_vector_array_new();
    mlx_optional_int og = {gs, true}, ob = {bits, true};
    int res = old_nc;
    if (mlx_quantize(&q, part, og, ob, "affine", g_mlx_empty_arr, s) == 0 && mlx_vector_array_size(q) >= 3) {
        mlx_array w2 = mlx_array_new(), s2 = mlx_array_new(), b2 = mlx_array_new();
        mlx_vector_array_get(&w2, q, 0); mlx_vector_array_get(&s2, q, 1); mlx_vector_array_get(&b2, q, 2);
        if (old_nc > 0 && w->ctx != NULL) {
            mlx_array cw = mlx_array_new(), cs2 = mlx_array_new(), cb = mlx_array_new();
            mlx_vector_array vw = mlx_vector_array_new(), vs = mlx_vector_array_new(), vb = mlx_vector_array_new();
            mlx_vector_array_append_value(vw, *w); mlx_vector_array_append_value(vw, w2);
            mlx_vector_array_append_value(vs, *sc); mlx_vector_array_append_value(vs, s2);
            mlx_vector_array_append_value(vb, *bi); mlx_vector_array_append_value(vb, b2);
            if (mlx_concatenate_axis(&cw, vw, 2, s) == 0 &&
                mlx_concatenate_axis(&cs2, vs, 2, s) == 0 &&
                mlx_concatenate_axis(&cb, vb, 2, s) == 0) {
                mlx_array_set(w, cw); mlx_array_set(sc, cs2); mlx_array_set(bi, cb);
                res = old_nc + n_new;
            }
            mlx_array_free(cw); mlx_array_free(cs2); mlx_array_free(cb);
            mlx_vector_array_free(vw); mlx_vector_array_free(vs); mlx_vector_array_free(vb);
        } else {
            mlx_array_set(w, w2); mlx_array_set(sc, s2); mlx_array_set(bi, b2);
            res = n_new;
        }
        mlx_array_free(w2); mlx_array_free(s2); mlx_array_free(b2);
    }
    mlx_vector_array_free(q); mlx_array_free(part);
    return res;
}

/* §G-bis leva 1 — sparse page-level attention: stato e dispatch.
 * I summary di pagina (ps_mean[layer] = media delle key per pagina/kv-head) sono mantenuti
 * INCREMENTALMENTE ai confini di pagina (amortizzato O(N) totale, MAI per-step: altrimenti i
 * byte KV letti non calano e lo sparse torna net≈0, §M). Al decode il dispatch calcola gli
 * score q-vs-summary (O(pagine)), seleziona le top-k pagine (+ sink + recent) e chiama
 * llm_mlx_pagesel_decode_sdpa. layer_forward installa/azzera l'hook attorno a m->arch->attn.
 * ATTENZIONE: l'op pagesel chiama a sua volta llm_mlx_gqa_sdpa -> va azzerato l'hook durante la
 * chiamata, altrimenti ricorsione infinita. */

static bool llm_pagesel_ensure(LlmMlxModel* m, LlmMlxSession* s) {
    if (!m || !s) return false;
    if (s->ps_mean && s->ps_done) return true;
    int nl = s->n_layers; if (nl <= 0) return false;
    s->ps_mean = (mlx_array*)calloc((size_t)nl, sizeof(mlx_array));
    s->ps_done = (int*)calloc((size_t)nl, sizeof(int));
    if (!s->ps_mean || !s->ps_done) { free(s->ps_mean); free(s->ps_done); s->ps_mean = NULL; s->ps_done = NULL; return false; }
    int cap = m->ps_cap > 0 ? m->ps_cap : 512;
    s->ps_cap = cap;
    for (int l = 0; l < nl; l++) {
        int shape[4] = {1, m->cfg.n_kv_heads, cap, m->cfg.head_dim};
        if (mlx_zeros(&s->ps_mean[l], shape, 4, MLX_FLOAT32, m->stream) != 0) return false;
    }
    return true;
}

/* Riassume (media su asse 2) le pagine COMPLETATE di k_cache [1,nkv,L,hd] in ps_mean[layer]. */
static void llm_pagesel_update(LlmMlxModel* m, LlmMlxSession* s, int layer, mlx_array k_cache) {
    if (!m || !s || !s->ps_mean || !s->ps_done) return;
    if (layer < 0 || layer >= s->n_layers || mlx_array_ndim(k_cache) != 4) return;
    int L = (int)mlx_array_shape(k_cache)[2];
    int page = m->ps_page, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    while (s->ps_done[layer] + page <= L) {
        int base = s->ps_done[layer], p = base / page;
        if (p >= s->ps_cap) break;
        int a[4] = {0, 0, base, 0}, b[4] = {1, nkv, base + page, hd}, st[4] = {1, 1, 1, 1};
        mlx_array pg = mlx_array_new(), mean = mlx_array_new();
        bool pok = (mlx_slice(&pg, k_cache, a, 4, b, 4, st, 4, m->stream) == 0 &&
                    mlx_mean_axis(&mean, pg, 2, /*keepdims=*/true, m->stream) == 0);
        if (pok) {
            int u0[4] = {0, 0, p, 0}, u1[4] = {1, nkv, p + 1, hd}, us[4] = {1, 1, 1, 1};
            mlx_array upd = mlx_array_new();
            /* mlx_slice_update(res, BASE, UPDATE, ...): base = ps_mean, update = mean della pagina */
            if (mlx_slice_update(&upd, s->ps_mean[layer], mean, u0, 4, u1, 4, us, 4, m->stream) == 0)
                mlx_array_set(&s->ps_mean[layer], upd);
            mlx_array_free(upd);
            s->ps_done[layer] = base + page;
        }
        mlx_array_free(pg); mlx_array_free(mean);
        if (!pok) break;
    }
}

static bool llm_pagesel_sdpa_dispatch(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                      float scale, void* ctx, mlx_stream stream) {
    LlmPageselCtx* c = (LlmPageselCtx*)ctx;
    if (!c || !c->m || !c->s || !c->s->ps_mean || !c->s->ps_done) return false;
    LlmMlxModel* m = c->m; LlmMlxSession* s = c->s;
    if (mlx_array_ndim(k) != 4) return false;
    int nkv = m->cfg.n_kv_heads, nh = m->cfg.n_heads, hd = m->cfg.head_dim;
    if (nkv <= 0 || hd <= 0 || nh % nkv != 0) return false;
    int N = (int)mlx_array_shape(k)[2];
    int page = m->ps_page;
    if (N < m->ps_min || page <= 0) return false;
    int n_filled = s->ps_done[c->layer] / page;
    if (n_filled < m->ps_window + 1) return false;
    int g = nh / nkv;
    bool ok = true;
    mlx_array mean_sl = mlx_array_new(), mean_nh = mlx_array_new(), meanT = mlx_array_new();
    mlx_array sc = mlx_array_new(), scm = mlx_array_new();
    int a4[4] = {0, 0, 0, 0}, b4[4] = {1, nkv, n_filled, hd}, st4[4] = {1, 1, 1, 1};
    if (mlx_slice(&mean_sl, s->ps_mean[c->layer], a4, 4, b4, 4, st4, 4, stream) != 0) ok = false;
    if (ok) {
        if (g > 1) { if (mlx_repeat_axis(&mean_nh, mean_sl, g, /*axis=*/1, stream) != 0) ok = false; }
        else if (mlx_array_set(&mean_nh, mean_sl) != 0) ok = false;
    }
    int perm[4] = {0, 1, 3, 2};
    if (ok && mlx_transpose_axes(&meanT, mean_nh, perm, 4, stream) != 0) ok = false;
    if (ok && mlx_matmul(&sc, q, meanT, stream) != 0) ok = false;         /* [1,nh,1,n_filled] */
    if (ok && mlx_max_axis(&scm, sc, 1, /*keepdims=*/true, stream) != 0) ok = false; /* [1,1,1,n_filled] */

    int sel[512]; int n_sel = 0;
    if (ok) {
        int tk = m->ps_topk < n_filled ? m->ps_topk : n_filled;
        mlx_array part = mlx_array_new();
        if (mlx_argpartition_axis(&part, scm, -tk, -1, stream) != 0) { ok = false; mlx_array_free(part); }
        else {
            int pnd = (int)mlx_array_ndim(part);
            int last = (int)mlx_array_shape(part)[pnd - 1];
            int s0[4] = {0, 0, 0, last - tk}, s1[4] = {1, 1, 1, last}, sv[4] = {1, 1, 1, 1};
            mlx_array view = mlx_array_new(), idxc = mlx_array_new();
            if (pnd != 4 || mlx_slice(&view, part, s0, 4, s1, 4, sv, 4, stream) != 0 ||
                mlx_contiguous(&idxc, view, false, stream) != 0 || mlx_array_eval(idxc) != 0) ok = false;
            if (ok) {
                const int32_t* d = mlx_array_data_int32(idxc);
                size_t n = mlx_array_size(idxc);
                for (size_t i = 0; i < n && n_sel < 500; i++)
                    if (d[i] >= 0 && d[i] < n_filled) sel[n_sel++] = (int)d[i];
            }
            mlx_array_free(view); mlx_array_free(idxc); mlx_array_free(part);
        }
    }
    mlx_array_free(mean_sl); mlx_array_free(mean_nh); mlx_array_free(meanT);
    mlx_array_free(sc); mlx_array_free(scm);
    if (!ok) return false;
    sel[n_sel++] = 0;                                            /* sink */
    for (int i = 0; i < m->ps_window && i < n_filled; i++) sel[n_sel++] = n_filled - 1 - i; /* recent */
    for (int i = 0; i < n_sel; i++) for (int j = i + 1; j < n_sel; j++) if (sel[j] < sel[i]) { int t = sel[i]; sel[i] = sel[j]; sel[j] = t; }
    int uniq[512], nu = 0;
    for (int i = 0; i < n_sel; i++) if (nu == 0 || uniq[nu - 1] != sel[i]) uniq[nu++] = sel[i];

    /* Azzera l'hook durante le chiamate a gqa_sdpa (l'op pagesel la usa internamente). */
    if (m->ps_verify) {
        llm_pagesel_hook.fn = NULL; llm_pagesel_hook.ctx = NULL;
        mlx_array full = mlx_array_new(), sp = mlx_array_new();
        bool fok = llm_mlx_gqa_sdpa(&full, q, k, v, scale, 0, stream);
        if (fok && llm_mlx_pagesel_decode_sdpa(&sp, q, k, v, scale, page, uniq, nu, stream) &&
            mlx_array_eval(full) == 0 && mlx_array_eval(sp) == 0) {
            const float* fd = mlx_array_data_float32(full);
            const float* sd = mlx_array_data_float32(sp);
            size_t n = mlx_array_size(full); float md = 0.0f;
            for (size_t i = 0; i < n; i++) { float dd = fd[i] - sd[i]; if (dd < 0) dd = -dd; if (dd > md) md = dd; }
            fprintf(stderr, "[pagesel] layer=%d N=%d filled=%d sel=%d maxdiff=%.4g\n", c->layer, N, n_filled, nu, md);
        }
        mlx_array_free(sp);
        llm_pagesel_hook.fn = llm_pagesel_sdpa_dispatch; llm_pagesel_hook.ctx = c;
        bool rok = fok && mlx_array_set(out, full) == 0;   /* in verifica usa l'output pieno */
        mlx_array_free(full);
        return rok;
    }
    llm_pagesel_hook.fn = NULL; llm_pagesel_hook.ctx = NULL;
    bool r = llm_mlx_pagesel_decode_sdpa(out, q, k, v, scale, page, uniq, nu, stream);
    llm_pagesel_hook.fn = llm_pagesel_sdpa_dispatch; llm_pagesel_hook.ctx = c;
    return r;
}

/* COMPOSIZIONE del layer (`.llmarch` structure `layer`): la sequenza del forward e' DATA, non
 * hardcoded. Default (nessuna pipeline impostata) = canonica [attention, ffn]. Il motore la
 * imposta dal `.llmarch` risolto (llm_mlx_resolve_arch). */
#define LLM_MLX_MAX_LOPS 16
static char g_mlx_layer_ops_buf[LLM_MLX_MAX_LOPS][24];
static const char* g_mlx_layer_ops[LLM_MLX_MAX_LOPS];
static int  g_mlx_layer_nops = 0;
void llm_mlx_set_layer_ops(const char* const* ops, int n) {
    if (n < 0) n = 0; if (n > LLM_MLX_MAX_LOPS) n = LLM_MLX_MAX_LOPS;
    for (int i = 0; i < n; i++) { snprintf(g_mlx_layer_ops_buf[i], sizeof g_mlx_layer_ops_buf[i], "%s", ops[i] ? ops[i] : ""); g_mlx_layer_ops[i] = g_mlx_layer_ops_buf[i]; }
    g_mlx_layer_nops = n;
}

bool layer_forward(LlmMlxModel* m, LlmMlxSession* kvq_sess, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out) {    /* KV quant: DEQUANTIZZA la parte cold salvata da un turno precedente (kvq_*[layer]) e concatena
     * con i recenti FP16 -> cache completa per l'attn. Resetta kvq_len=0 (la cache torna FP16). */
    /* CAUSA MISURATA 2026-09-17 (AiWiki "MLX-KVQuant-Contesto-Lungo-5x-Peggiore"): con
     * LLM_KVQUANT=1 da solo (senza LLM_KV_DEQUANT_READ), il decode NON usava affatto questo
     * kernel dequant-on-read -- finiva nel percorso piu' sotto (righe ~14060+) che ad OGNI
     * step dequantizza l'INTERO cold, lo concatena ai recenti, fa attenzione normale, poi
     * lo riquantizza (3 operazioni O(nc) per step, altro che O(nc/split)): e' quello, non il
     * kernel qui sotto, il vero motivo della pendenza 5-8x peggiore della fp16 misurata nel
     * bug doc. LLM_KV_DEQUANT_READ era un flag SEPARATO (2026-09-13, mai attivato dal
     * benchmark del bug doc) che evita quel ciclo -- e con lo split-K corretto qui sotto
     * (vedi llm_mlx_kv_quant_sdpa_kernel.h) e il batching del flush (llm_kvq_append_cold)
     * arriva a battere la fp16 (misurato: 32k 61,9 tok/s, 64k 59,8 tok/s, contro 8,71/4,27
     * del percorso vecchio -- fp16 stessa e' 49-50/29-31). Reso automatico quando kv_quantize
     * e' attivo, invece di richiedere un secondo flag separato che nessun utente/benchmark
     * conosceva: LLM_KV_DEQUANT_READ resta utilizzabile per forzarlo anche in scenari dove
     * kv_quantize non e' ancora stato attivato dal chiamante. */
    /* AUTO (2026-09-29): attiva la KV int8 Splash quando il contesto supera la soglia.
     * Il crossover misurato e' ~8-10k token (sotto conviene l'fp, sopra l'int8 e' ~4x piu'
     * veloce e ~1.7x meno memoria). Solo per modelli COMPATIBILI col kernel: head_dim
     * multiplo di 32, GQA valido, nessuna sliding-window/NSA. Default del registro = auto. */
    if (m->cfg.kv_quantize_auto && !m->cfg.kv_quantize) {
        bool compat = !m->cfg.nsa_sparse && m->cfg.head_dim > 0 &&
                      (m->cfg.head_dim % 32 == 0) && m->cfg.n_kv_heads > 0 &&
                      (m->cfg.n_heads % m->cfg.n_kv_heads == 0) &&
                      (m->cfg.is_sliding_layer == NULL);
        /* PREFILL-DEFER (2026-10-01, OPZIONALE): nel prefill (n_tokens>1) la KV-quant usa il
         * dequant-read che RICOSTRUISCE la KV fp16 full (+ pool int8) -> picco ~80GB@257k. Con
         * LLM_KVQ_DEFER_PREFILL=1 si differisce l'int8 al decode (n_tokens==1): prefill fp16 ->
         * picco ~meta' (65.6GB@257k) MA prefill ~5x piu' lento (2327s vs 486s). Utile solo a
         * contesto estremo (1M/2M) per far entrare il DFlash2. Default = comportamento veloce. */
        /* W5 FIX (2026-10-03): DIFFERISCI l'attivazione int8 al DECODE (n_tokens==1). Attivarla
         * durante il prefill CHUNKED la ROMPE: il 2o+ chunk non legge la cold (il read-fused è
         * n_tokens==1), quindi l'attenzione del prefill vede solo la finestra recente -> output
         * errato oltre la dimensione del chunk (~8192; misurato: 7800 ok, 8200 rotto). Con il defer
         * il prefill resta FP16 (corretto) e l'int8 parte al primo decode. Override debug per
         * l'attivazione immediata: LLM_KVQ_DEFER_PREFILL=0. (Il fix "vero" sarebbe il kernel int8
         * di PREFILL — vedi W3.) */
        const char* _dpe = llm_ai_mlx_opt_env("LLM_KVQ_DEFER_PREFILL");
        bool defer_ok = (n_tokens == 1) || (_dpe != NULL && strcmp(_dpe, "0") == 0);
        if (compat && defer_ok && cache_offset + n_tokens > m->cfg.kv_quantize_auto_trigger) {
            m->cfg.kv_quantize = true;
            m->cfg.kv_quantize_bits = 8;
            m->cfg.kv_quantize_threshold = 2048;
            fprintf(stderr, "[kvquant] auto-ON a %d token (int8 Splash, finestra 2048)\n",
                    cache_offset + n_tokens);
        }
    }
    bool kvq_read = m->cfg.kv_quantize;
    /* KV-quant (port Splash): il cold vive nel PAGE POOL int8 = storage PRIMARIO.
     * La cache fp e' SOLO la finestra recente; il read fused legge il pool. */
    llm_kvq_pool* kvqp = NULL;
    KvDisk* kvq_disk = llm_kvdisk_enabled(m) ? &m->kvdisk : NULL;   /* B9: tier SSD per lo spill */
    if (kvq_sess && m->cfg.kv_quantize && m->cfg.kv_quantize_threshold > 0)
        kvqp = llm_kvq_pool_for((void*)m, (void*)kvq_sess, (int)m->cfg.n_layers,
                                (int)m->cfg.n_kv_heads, (int)m->cfg.head_dim, LLM_KVQ_FMT_INT8,
                                m->cfg.kv_quantize_threshold, kvq_disk, m->stream);
    /* B9: se il pool e' stato spilliato (sessione ripresa), ripristina le pagine PRIMA dell'attenzione. */
    if (kvqp && kvqp->disk && llm_kvq_pool_offloaded(kvqp))
        llm_kvq_pool_faultin_all(kvqp, kvqp->disk, kvqp->key_base, m->stream);
    bool kvq_read_used = false;
    if (kvq_read && kvqp && kvqp->committed_l[layer_idx] > 0) {
        g_kvq_cold.kw = kvqp->keyData[layer_idx];
        g_kvq_cold.ks = kvqp->keyScales[layer_idx];
        g_kvq_cold.kb = kvqp->bias[layer_idx];
        g_kvq_cold.vw = kvqp->valueData[layer_idx];
        g_kvq_cold.vs = kvqp->valueScales[layer_idx];
        g_kvq_cold.vb = kvqp->bias[layer_idx];
        g_kvq_cold.nc = kvqp->committed_l[layer_idx];
        g_kvq_cold.kbits = 8; g_kvq_cold.vbits = 8;
        g_kvq_cold.ptab = kvqp->ptab;
        llm_kvq_hook.fn = llm_kvq_sdpa_hook_dispatch;
        llm_kvq_hook.ctx = &g_kvq_cold;
        kvq_read_used = true;
    }
    if (getenv("LLM_KVQ_DBG") && layer_idx == 0)
        fprintf(stderr, "[KVQREAD] L0 kvq_read=%d used=%d committed=%d nc=%d off=%d nt=%d\n",
                (int)kvq_read, (int)kvq_read_used, kvqp ? kvqp->committed_l[0] : -1,
                (int)g_kvq_cold.nc, cache_offset, n_tokens);

    if (!kvq_read_used && kvq_sess && kvq_sess->kvq_len && kvq_sess->kvq_len[layer_idx] > 0) {
        int cold = kvq_sess->kvq_len[layer_idx];
        int bits = kvq_sess->kvq_bits;
        mlx_array deq = mlx_array_new();
        if (llm_mlx_kvq_splash_dequant(&deq, kvq_sess->kvq_w[layer_idx], kvq_sess->kvq_sc[layer_idx], m->stream) == 0) {
            mlx_vector_array rc = mlx_vector_array_new();
            mlx_vector_array_append_value(rc, deq);
            mlx_vector_array_append_value(rc, *k_cache);
            mlx_array full = mlx_array_new();
            if (mlx_concatenate_axis(&full, rc, 2, m->stream) == 0)
                mlx_array_set(k_cache, full);
            mlx_array_free(full);
            mlx_vector_array_free(rc);
        }
        mlx_array_free(deq);
        kvq_sess->kvq_len[layer_idx] = 0;
    }
    /* HETERO-KV: dequantizza la parte cold di V (K piu' preciso di V) e concatena con i recenti. */
    if (!kvq_read_used && kvq_sess && kvq_sess->kvq_v_len && kvq_sess->kvq_v_len[layer_idx] > 0) {
        int vbits = kvq_sess->kvq_v_bits;
        mlx_array deqv = mlx_array_new();
        if (llm_mlx_kvq_splash_dequant(&deqv, kvq_sess->kvq_v_w[layer_idx], kvq_sess->kvq_v_sc[layer_idx], m->stream) == 0) {
            mlx_vector_array rc = mlx_vector_array_new();
            mlx_vector_array_append_value(rc, deqv);
            mlx_vector_array_append_value(rc, *v_cache);
            mlx_array full = mlx_array_new();
            if (mlx_concatenate_axis(&full, rc, 2, m->stream) == 0)
                mlx_array_set(v_cache, full);
            mlx_array_free(full);
            mlx_vector_array_free(rc);
        }
        mlx_array_free(deqv);
        kvq_sess->kvq_v_len[layer_idx] = 0;
    }
    mlx_array h = mlx_array_new();
    /* Sliding-window KV trim: per i layer sliding (is_sliding_layer + sliding_window>0) i token
     * piu' vecchi di `sliding_window` sono INACCESSIBILI a qualunque query futura (attenzione
     * causale a finestra) -> la cache oltre la finestra va scartata. Senza trim la cache cresce
     * O(pos) e ogni step di decode ri-concatena la cache intera per layer -> decode O(pos)
     * (gemma4 31B: 82ms/step a 2K ctx, crollo a ~1-2 tok/s a 8K). Con trim il concat + la SDPA
     * a finestra diventano O(window) = costanti col contesto. Parita' verificata vs mlx_lm a
     * 1500 token (oltre la finestra 1024): output identico. */
    if (m->cfg.is_sliding_layer && m->cfg.sliding_window > 0 &&
        m->cfg.is_sliding_layer[layer_idx] && has_cache) {
        const int* csh = mlx_array_shape(*k_cache);
        if (csh != NULL && mlx_array_ndim(*k_cache) == 4) {
            int seq = csh[2];
            int win = m->cfg.sliding_window;
            if (seq > win) {
                const int start[4] = {0, 0, seq - win, 0};
                const int stop[4]  = {csh[0], csh[1], seq, csh[3]};
                const int stride[4]= {1, 1, 1, 1};
                mlx_array kt = mlx_array_new(), vt = mlx_array_new();
                if (mlx_slice(&kt, *k_cache, start, 4, stop, 4, stride, 4, m->stream) == 0 &&
                    mlx_slice(&vt, *v_cache, start, 4, stop, 4, stride, 4, m->stream) == 0) {
                    mlx_array_set(k_cache, kt);
                    mlx_array_set(v_cache, vt);
                }
                mlx_array_free(kt);
                mlx_array_free(vt);
            }
        }
    }
    static double prof_attn = 0.0, prof_ffn = 0.0; static int prof_run = 0;
    bool gprof = llm_env_prof();
    bool gphase = llm_env_phase();
    bool gmem = llm_env_memprof();
    static double ph_attn = 0.0, ph_ffn = 0.0; static int ph_layers = 0;
    struct timespec t0, t1;
    if (gprof) clock_gettime(CLOCK_MONOTONIC, &t0);
    {   /* Dump dell'INPUT del layer 0 (= embedding) per isolare la prima divergenza. */
        const char* dl = llm_env_get("LLM_DUMP_LAYERS");
        if (dl && layer_idx == 0) {
            mlx_array hf = mlx_array_new();
            if (mlx_astype(&hf, x, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
                const float* d = mlx_array_data_float32(hf);
                size_t nh = (size_t)n_tokens * (size_t)m->cfg.hidden_size;
                char p[512]; snprintf(p, sizeof(p), "%s/in_00_n%d.bin", dl, n_tokens);
                FILE* f = fopen(p, "wb");
                if (f) { fwrite(d, sizeof(float), nh, f); fclose(f); }
            }
            mlx_array_free(hf);
        }
    }
    /* F3-bis (qwen4_exp): dump dell'INPUT del layer 0 = embedding post-tile (hc_width reale,
     * non cfg.hidden_size). LLM_QWEN4EXP_LAYER_DUMP attiva il dump per-layer; questo stampa
     * [Q4_EMB] con gli stessi campi (absmax/mean/sum/first4) per il confronto col reference. */
    if (layer_idx == 0 && llm_env_isset("LLM_QWEN4EXP_LAYER_DUMP")) {
        mlx_array hf = mlx_array_new();
        if (mlx_astype(&hf, x, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
            const float* d = mlx_array_data_float32(hf);
            int nd = mlx_array_ndim(hf);
            if (d && nd >= 2) {
                int W = (int)mlx_array_shape(hf)[nd - 1];
                int S = (int)mlx_array_shape(hf)[nd - 2];
                if (W > 0 && S > 0) {
                    const float* last = d + (size_t)(S - 1) * (size_t)W;
                    float am = 0.0f; double sum = 0.0;
                    for (int i = 0; i < W; i++) { float a = fabsf(last[i]); if (a > am) am = a; sum += (double)last[i]; }
                    fprintf(stderr, "[Q4_EMB] absmax=%.6f mean=%.6f sum=%.6f first4=%.6f,%.6f,%.6f,%.6f W=%d\n",
                            am, sum / (double)W, sum, last[0], last[1], last[2], last[3], W);
                }
            }
        }
        mlx_array_free(hf);
    }
    /* §G-bis leva 1: aggiorna i summary di pagina (prefill+decode) e, al decode, installa
     * l'hook sparse page-level attorno alla chiamata del blocco attn. Skip per layer sliding
     * (il trim della cache renderrebbe le posizioni incoerenti) e per la KV-quant cold. */
    LlmPageselCtx psctx;
    bool ps_upd = m->pagesel_on && has_cache && kvq_sess != NULL &&
                  !(m->cfg.is_sliding_layer && layer_idx < m->cfg.n_layers && m->cfg.is_sliding_layer[layer_idx]) &&
                  !(kvq_sess->kvq_len && kvq_sess->kvq_len[layer_idx] > 0) &&
                  (m->cfg.n_heads % m->cfg.n_kv_heads == 0);
    bool ps_hook = ps_upd && n_tokens == 1;
    if (ps_upd && !llm_pagesel_ensure(m, kvq_sess)) ps_upd = false;
    bool ok = true;
    /* COMPOSIZIONE del layer: esegue la pipeline dichiarata (`layer` del `.llmarch`) o la canonica
     * [attention, ffn]. Ogni op -> la primitiva del layer (il blocco per-arch). */
    static const char* def_layer_ops[] = { "attention", "ffn" };
    const char* const* LOP = g_mlx_layer_nops > 0 ? (const char* const*)g_mlx_layer_ops : def_layer_ops;
    int NLOP = g_mlx_layer_nops > 0 ? g_mlx_layer_nops : 2;
    /* §9.A M10 EXECUTION PLANNER (runtime manager): se un piano e' attivo e copre questo layer,
     * usa la sua pipeline per-layer; SKIP -> identita' (out = x). Piano ingerito da JSON
     * (compile-time EXECPLAN del producer PUL) via llm_mlx_execplan_load. */
    if (llm_mlx_execplan_active()) {
        char ep_buf[LLM_MLX_EXECPLAN_MAX_OPS][LLM_MLX_EXECPLAN_OP_LEN];
        const char* ep_ptrs[LLM_MLX_EXECPLAN_MAX_OPS];
        int ep_skip = 0;
        int ep_n = llm_mlx_execplan_layer(layer_idx, ep_buf, LLM_MLX_EXECPLAN_MAX_OPS, &ep_skip);
        if (ep_n >= 0) {
            if (ep_skip) { ok = mlx_array_set(out, x) == 0; NLOP = 0; }
            else if (ep_n > 0) { for (int ei = 0; ei < ep_n; ei++) ep_ptrs[ei] = ep_buf[ei]; LOP = ep_ptrs; NLOP = ep_n; }
        }
    }
    for (int lop = 0; lop < NLOP && ok; lop++) {
        const char* lopn = LOP[lop];
        if (strcmp(lopn, "attention") == 0) {
            if (ps_hook) { psctx.m = m; psctx.s = kvq_sess; psctx.layer = layer_idx;
                           llm_pagesel_hook.fn = llm_pagesel_sdpa_dispatch; llm_pagesel_hook.ctx = &psctx; }
            ok = m->arch->attn(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, &h);
            /* SONDA deterministica (A): identita' dei BUFFER (mlx_array={void*ctx}) di input/output
             * del layer. Serve a vedere l'aliasing tra stato (KV/aux) e intermedi quando si aggiunge
             * un nodo di lettura (cattura snap): se un leaf condivide il buffer con lo stato, la
             * donation in-place diventa copia e il forward diverge. Zero costo se l'env e' assente. */
            if (getenv("LLM_LAYER_CTX")) {
                fprintf(stderr, "[layerctx] L=%d x=%p h=%p kc=%p vc=%p has=%d n=%d\n",
                        layer_idx, x.ctx, h.ctx,
                        (k_cache && k_cache->ctx) ? k_cache->ctx : NULL,
                        (v_cache && v_cache->ctx) ? v_cache->ctx : NULL,
                        (int)has_cache, n_tokens);
            }
            if (ps_hook) { llm_pagesel_hook.fn = NULL; llm_pagesel_hook.ctx = NULL; }
            if (ps_upd && ok) llm_pagesel_update(m, kvq_sess, layer_idx, *k_cache);
            if (kvq_read_used) { llm_kvq_hook.fn = NULL; llm_kvq_hook.ctx = NULL; }
            if (gprof) { clock_gettime(CLOCK_MONOTONIC, &t1); prof_attn += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9; }
            if (gphase && ok) { clock_gettime(CLOCK_MONOTONIC, &t0); ok = mlx_array_eval(h) == 0; clock_gettime(CLOCK_MONOTONIC, &t1); ph_attn += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9; }
            {   /* Dump per-layer del hidden POST-ATTENTION (pre-FFN, fp32). */
                const char* dl = llm_env_get("LLM_DUMP_LAYERS");
                if (ok && dl) {
                    mlx_array hf = mlx_array_new();
                    if (mlx_astype(&hf, h, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
                        const float* d = mlx_array_data_float32(hf);
                        size_t nh = (size_t)n_tokens * (size_t)m->cfg.hidden_size;
                        char p[512]; snprintf(p, sizeof(p), "%s/attn_%02d_n%d.bin", dl, layer_idx, n_tokens);
                        FILE* f = fopen(p, "wb");
                        if (f) { fwrite(d, sizeof(float), nh, f); fclose(f); }
                    }
                    mlx_array_free(hf);
                }
            }
        } else if (strcmp(lopn, "ffn") == 0) {
            if (gprof) clock_gettime(CLOCK_MONOTONIC, &t0);
            ok = m->arch->ffn(m, layer_idx, n_tokens, h, out);
            if (ok) llm_mlx_dbg_dump("out", layer_idx, *out);
            if (gphase && ok) { clock_gettime(CLOCK_MONOTONIC, &t0); ok = mlx_array_eval(*out) == 0; clock_gettime(CLOCK_MONOTONIC, &t1); ph_ffn += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9; }
        }
    }
    /* DIAGNOSTICA (P10 oQ A8, §6ter "prossimo esperimento"): LLMMETAL_OQ_A8_SYNC_LAYER forza il
     * DRAIN del grafo lazy a fine di OGNI layer (invece che a fine di OGNI singola primitiva
     * OqA8, come LLMMETAL_OQ_A8_SYNC) — testa se il blocco lazy/ordering e' un accumulo fra PIU'
     * invocazioni della primitiva nello stesso grafo (16 layer x ~5 matmul oQ A8 = decine di
     * `allocator::malloc` per l'output PRIMA che qualunque eval reale giri, dato che il motore
     * NON valuta per-layer nel path normale — verificato: tutte le mlx_array_eval per-layer nel
     * motore sono dietro gphase/gmem, mai attive di default) invece che di un singolo primitive
     * isolato dal resto. Se drenare qui basta (a differenza di LLMMETAL_OQ_A8_SYNC_X, che NON
     * bastava), la causa e' l'accumulo cross-invocazione, non la staleness di un input di una
     * singola chiamata. Default OFF, nessun impatto quando non impostata. */
    if (ok && llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_SYNC_LAYER")) ok = mlx_array_eval(*out) == 0;
    if (gmem && ok) {
        size_t peak = 0, active = 0, cache = 0;
        mlx_reset_peak_memory();
        ok = mlx_array_eval(*out) == 0;
        mlx_get_peak_memory(&peak);
        mlx_get_active_memory(&active);
        mlx_get_cache_memory(&cache);
        fprintf(stderr, "[MEMPROF] layer %3d peak=%.1f MB active=%.1f MB cache=%.1f MB\n",
                layer_idx, peak / 1048576.0, active / 1048576.0, cache / 1048576.0);
        fflush(stderr);
    }
    if (getenv("LLM_KVQ_MEMDBG") && kvq_sess && (layer_idx == 0 || layer_idx == m->cfg.n_layers - 1)) {
        size_t kb = (*k_cache).ctx ? mlx_array_size(*k_cache) * mlx_array_itemsize(*k_cache) : 0;
        size_t vb = (*v_cache).ctx ? mlx_array_size(*v_cache) * mlx_array_itemsize(*v_cache) : 0;
        size_t qk = 0, qv = 0;
        if (kvq_sess->kvq_w && kvq_sess->kvq_w[layer_idx].ctx) {
            qk += mlx_array_size(kvq_sess->kvq_w[layer_idx]) * mlx_array_itemsize(kvq_sess->kvq_w[layer_idx]);
            qk += mlx_array_size(kvq_sess->kvq_sc[layer_idx]) * mlx_array_itemsize(kvq_sess->kvq_sc[layer_idx]);
            qk += mlx_array_size(kvq_sess->kvq_bi[layer_idx]) * mlx_array_itemsize(kvq_sess->kvq_bi[layer_idx]);
        }
        if (kvq_sess->kvq_v_w && kvq_sess->kvq_v_w[layer_idx].ctx) {
            qv += mlx_array_size(kvq_sess->kvq_v_w[layer_idx]) * mlx_array_itemsize(kvq_sess->kvq_v_w[layer_idx]);
            qv += mlx_array_size(kvq_sess->kvq_v_sc[layer_idx]) * mlx_array_itemsize(kvq_sess->kvq_v_sc[layer_idx]);
            qv += mlx_array_size(kvq_sess->kvq_v_bi[layer_idx]) * mlx_array_itemsize(kvq_sess->kvq_v_bi[layer_idx]);
        }
        fprintf(stderr, "[KVQMEM] L%02d n=%d kcache=%.2fMB vcache=%.2fMB kvqK=%.2fMB kvqV=%.2fMB kvq_len=%d\n",
                layer_idx, n_tokens, kb / 1048576.0, vb / 1048576.0, qk / 1048576.0, qv / 1048576.0,
                kvq_sess->kvq_len ? kvq_sess->kvq_len[layer_idx] : -1);
        fflush(stderr);
    }
    ph_layers++;
    if (gphase && layer_idx + 1 >= m->cfg.n_layers)
        fprintf(stderr, "[LLM_PHASE] layers=%d attn=%.3fs ffn=%.3fs total=%.3fs\n", ph_layers, ph_attn, ph_ffn, ph_attn + ph_ffn);
    {   /* Dump per-layer del hidden post-FFN (fp32) per il confronto con mlx_lm. */
        const char* dl = llm_env_get("LLM_DUMP_LAYERS");
        if (ok && dl) {
            mlx_array hf = mlx_array_new();
            if (mlx_astype(&hf, *out, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
                const float* d = mlx_array_data_float32(hf);
                size_t nh = (size_t)n_tokens * (size_t)m->cfg.hidden_size;
                char p[512]; snprintf(p, sizeof(p), "%s/layer_%02d_n%d.bin", dl, layer_idx, n_tokens);
                FILE* f = fopen(p, "wb");
                if (f) { fwrite(d, sizeof(float), nh, f); fclose(f); }
            }
            mlx_array_free(hf);
        }
    }
    static int dump_attn_done = 0;
    if (ok && !dump_attn_done && llm_env_get("LLM_DEBUG_POSTATTN") && layer_idx == 0) {
        mlx_array hf = mlx_array_new();
        if (mlx_astype(&hf, *out, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
            const float* d = mlx_array_data_float32(hf);
            size_t nh = (size_t)n_tokens * (size_t)m->cfg.hidden_size;
            char p[512]; snprintf(p, sizeof(p), "%s.postffn", llm_env_get("LLM_DEBUG_POSTATTN"));
            FILE* f = fopen(p, "wb");
            if (f) { fwrite(d, sizeof(float), nh, f); fclose(f); }
        }
        mlx_array_free(hf);
        mlx_array hb = mlx_array_new();
        if (mlx_astype(&hb, h, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hb) == 0) {
            const float* d = mlx_array_data_float32(hb);
            size_t nh = (size_t)n_tokens * (size_t)m->cfg.hidden_size;
            FILE* f = fopen(llm_env_get("LLM_DEBUG_POSTATTN"), "wb");
            if (f) { fwrite(d, sizeof(float), nh, f); fclose(f); }
        }
        mlx_array_free(hb);
        dump_attn_done = 1;
    }
    if (gprof) { clock_gettime(CLOCK_MONOTONIC, &t1); prof_ffn += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9; }
    static int dump_layer_max = -1;
    const char* dh = llm_env_get("LLM_DEBUG_HIDDIR");
    /* LLM_DEBUG_HIDDIR_ALL=1: dumpa OGNI passata (sovrascrive) -> per la diagnosi batch-vs-seq
     * serve l'ultimo token della sequenza (la guardia static coprirebbe solo la prima passata). */
    if (ok && dh && (llm_env_isset("LLM_DEBUG_HIDDIR_ALL") || layer_idx > dump_layer_max) && layer_idx < 32) {
        mlx_array hf = mlx_array_new();
        if (mlx_astype(&hf, *out, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
            const float* d = mlx_array_data_float32(hf);
            size_t nh = (size_t)n_tokens * (size_t)m->cfg.hidden_size;
            char p[512]; snprintf(p, sizeof(p), "%s/l%d.bin", dh, layer_idx);
            FILE* f = fopen(p, "wb");
            if (f) { fwrite(d, sizeof(float), nh, f); fclose(f); }
        }
        mlx_array_free(hf);
        dump_layer_max = layer_idx;
    }
    if (gprof && layer_idx + 1 >= m->cfg.n_layers) { fprintf(stderr, "[LLM_PROF] attn=%.2fs ffn=%.2fs total=%.2fs\n", prof_attn, prof_ffn, prof_attn + prof_ffn); prof_attn = prof_ffn = 0.0; (void)prof_run; }
    /* KV quant: DOPO l'attn (cache aggiornata con i nuovi token), se supera la soglia quantizza la
     * parte COLD e la salva in kvq_*[layer] (persistente quantizzata); k_cache resta SOLO i recenti. */
    /* KV-quant (port Splash ARCHITETTURALE): il cold [0, N-W) va nel PAGE POOL
     * int8 (storage primario, slice_update); la cache fp conserva SOLO la
     * finestra recente [N-W, N). Nessun Concat del cold, nessuna cache fp del
     * cold: il +1266 MB del path "in aggiunta" non puo' esistere. */
    if (ok && kvq_sess && kvqp && m->cfg.kv_quantize_threshold > 0 &&
        (*k_cache).ctx != NULL && (*v_cache).ctx != NULL &&
        mlx_array_ndim(*k_cache) == 4 && mlx_array_ndim(*v_cache) == 4 &&
        mlx_array_shape(*k_cache)[1] == (int)m->cfg.n_kv_heads &&
        mlx_array_shape(*k_cache)[3] == (int)m->cfg.head_dim) {
        int thr = m->cfg.kv_quantize_threshold;
        int nkv = (int)mlx_array_shape(*k_cache)[1], hd = (int)mlx_array_shape(*k_cache)[3];
        int N = (int)mlx_array_shape(*k_cache)[2];
        int total = cache_offset + n_tokens;          /* posizione ASSOLUTA totale */
        int upto = total - thr;                       /* cold assoluto = [0, upto) */
        if (thr > 0 && upto > 0) {
            /* La cache fp puo' essere una FINESTRA [total-N, total) (concat, dopo un trim) oppure
             * un buffer PADDED a capacita' con i validi in [0, total) (in-place, alla prima
             * attivazione). valid_start = posizione assoluta del primo token valido nella cache. */
            int valid_start = (N <= total) ? (total - N) : 0;
            int valid_count = (N <= total) ? N : total;
            int committed = kvqp->committed_l[layer_idx];
            int n_new = upto - committed;
            if (getenv("LLM_KVQ_DBG") && layer_idx == 0) {
                static int dbgn = 0;
                if (dbgn < 8) { dbgn++;
                    fprintf(stderr, "[KVQDBG] L0 total=%d thr=%d upto=%d committed=%d valid_start=%d valid_count=%d n_new=%d i0=%d i1=%d N=%d\n",
                        total, thr, upto, committed, valid_start, valid_count, n_new, committed-valid_start, upto-valid_start, N);
                }
            }
            int one[4] = {1,1,1,1};
            if (n_new > 0) {
                int i0 = committed - valid_start, i1 = upto - valid_start;
                if (i0 >= 0 && i1 <= valid_count && i1 > i0) {
                    int s0[4] = {0,0,i0,0}, s1[4] = {1,nkv,i1,hd};
                    mlx_array kn = mlx_array_new(), vn = mlx_array_new();
                    bool st_ok = false;
                    if (mlx_slice(&kn, *k_cache, s0,4,s1,4,one,4,m->stream) == 0 &&
                        mlx_slice(&vn, *v_cache, s0,4,s1,4,one,4,m->stream) == 0)
                        st_ok = llm_kvq_pool_store(kvqp, layer_idx, kn, vn, committed, n_new, m->stream);
                    if (getenv("LLM_KVQ_DBG") && layer_idx == 0)
                        fprintf(stderr, "[KVQSTORE] pos=%d n=%d -> committed_now=%d ok=%d\n",
                                committed, n_new, kvqp->committed_l[0], (int)st_ok);
                    mlx_array_free(kn); mlx_array_free(vn);
                }
            }
            /* cache fp = ultimi thr token VALIDI, in ordine assoluto [total-thr, total) */
            if (valid_count > thr) {
                int r0[4] = {0,0,valid_count-thr,0};
                int r1[4] = {1,nkv,valid_count,hd};
                mlx_array kr = mlx_array_new(), krc = mlx_array_new();
                mlx_array vr = mlx_array_new(), vrc = mlx_array_new();
                if (mlx_slice(&kr, *k_cache, r0,4,r1,4,one,4,m->stream) == 0 &&
                    mlx_contiguous(&krc, kr, false, m->stream) == 0) {
                    mlx_array_set(k_cache, krc);
                    /* EVAL (2026-10-01): senza, il `contiguous` resta LAZY e la vista slice
                     * trattiene il BACKING full della KV fp16 finche' il grafo non e' consumato
                     * -> +1x KV fp16 (~60KB/token, misurato +4.5GB@71k). Valutare qui libera il full. */
                    mlx_array_eval(*k_cache);
                }
                if (mlx_slice(&vr, *v_cache, r0,4,r1,4,one,4,m->stream) == 0 &&
                    mlx_contiguous(&vrc, vr, false, m->stream) == 0) {
                    mlx_array_set(v_cache, vrc);
                    mlx_array_eval(*v_cache);
                }
                mlx_array_free(kr); mlx_array_free(krc); mlx_array_free(vr); mlx_array_free(vrc);
            }
        }
    }
    /* (i path affine/kvq "in aggiunta" sono rimossi: sostituiti dal page pool Splash) */
    mlx_array_free(h);
    return ok;
}
