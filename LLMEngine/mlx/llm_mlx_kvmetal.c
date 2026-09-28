/* ============================================================================
 * Dominio KV Metal + KV segmenti: llm_mlx_kvmetal_populate_batch, llm_mlx_kvmetal_step, llm_mlx_max_abs_diff (diagnostica verify), llm_mlx_kvseg_merge_two.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_kvmetal_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* LLMMETAL_KVMETAL (sperimentale, 2026-09-06, default OFF): un passo di decode via KV Metal
 * reale (native/llmmetal_kv/llmmetal_kv.h tramite llm_kvmetal_bridge.h) invece di
 * mlx_fast_scaled_dot_product_attention su k_full/v_full concatenati -- vedi commento su
 * llm_mlx_kvmetal_enabled per il contratto completo. Precondizioni verificate dal
 * CHIAMANTE (eligibilita' runtime); qui si assume gia' tutto valido tranne l'inizializzazione
 * lazy del contesto Metal, che PUO' fallire (Metal non disponibile, vincoli del kernel non
 * rispettati) -- in quel caso ritorna false e NON tocca attn_out, il chiamante fa fallback.
 *
 * page_tokens fisso a 16 (stesso block_size di default di llm_mlx_kvpage.h). n_pages
 * dimensionato su m->cfg.kv_max_tokens (richiesto > 0 dall'eligibilita' del chiamante) --
 * se cache_offset supera quella capacita' (es. rolling-window che avanza la posizione oltre
 * la finestra) llm_kvmetal_write fallisce per bound-check interno e questa funzione ritorna
 * false: fallback sicuro, MAI un overflow o un crash, semplicemente il path smette di
 * aiutare oltre quella soglia. Layout my kernel: pagina = posizione assoluta / page_tokens,
 * SENZA riuso di slot (nessuna eviction) -- corretto solo finche' cache_offset resta entro
 * kv_max_tokens dall'inizio della sessione; un vero rolling-window con eviction reale delle
 * pagine e' lavoro futuro (vedi piano, D14 limite 3). */
/* BUG REALE trovato il 2026-09-07 (prima verifica end-to-end mai eseguita di LLMMETAL_KVMETAL,
 * sbloccata dal fix del parsing kvMaxTokens, commit 060df81c): llm_mlx_kvmetal_step (sotto)
 * viene chiamata SOLO per n_tokens==1 (decode) -- durante un forward BATCH (prefill/append,
 * n_tokens>1) il pool K/V Metal non veniva MAI popolato. Risultato: dopo un prefill di N
 * token, le pagine che coprono quei token restavano vuote (memoria del buffer Metal non
 * inizializzata) nel pool -- il decode successivo faceva attention su "token" fantasma per
 * tutta la storia del prefill, non sui K/V reali. Root cause dei max_abs_diff 0.13-1.4
 * osservati (non spiegabili da precisione fp16, un ordine di grandezza troppo grandi).
 * Questa funzione chiude il buco: scrive OGNI posizione del batch nel pool, una per una
 * (page_id/offset ricalcolati per ciascuna), SENZA calcolare attenzione (quella resta
 * decode-only, llm_mlx_kvmetal_step sotto). k_rope/v_t arrivano come [1,nkv,n_tokens,D]
 * (H-major, dal transpose in llama_attention_block) -- serve [1,n_tokens,nkv,D] (T-major)
 * per estrarre uno slab [nkv,D] contiguo per token, stesso layout che llmmetal_kv_write
 * si aspetta (vedi llmmetal_kv.h). */
bool llm_mlx_kvmetal_populate_batch(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m,
                                            int cache_offset, int n_tokens,
                                            mlx_array k_rope, mlx_array v_t) {
    if (!kvmetal_ctx_arr || n_tokens <= 0) return false;
    const int page_tokens = 16;
    const int nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;

    LlmKvMetalCtx* ctx = (LlmKvMetalCtx*)kvmetal_ctx_arr[layer_idx];
    if (!ctx) {
        int n_pages = (m->cfg.kv_max_tokens + page_tokens - 1) / page_tokens;
        if (n_pages < 1) n_pages = 1;
        ctx = llm_kvmetal_create(n_pages, page_tokens, m->cfg.n_heads, nkv, hd);
        if (!ctx) return false;
        kvmetal_ctx_arr[layer_idx] = ctx;
    }

    mlx_array kf = mlx_array_new(), vf = mlx_array_new();
    bool ok = (mlx_array_dtype(k_rope) == MLX_FLOAT16) ? mlx_array_set(&kf, k_rope) == 0
                                                        : mlx_astype(&kf, k_rope, MLX_FLOAT16, m->stream) == 0;
    ok = ok && ((mlx_array_dtype(v_t) == MLX_FLOAT16) ? mlx_array_set(&vf, v_t) == 0
                                                       : mlx_astype(&vf, v_t, MLX_FLOAT16, m->stream) == 0);
    mlx_array kt = mlx_array_new(), vtt = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3}; /* [1,nkv,T,D] -> [1,T,nkv,D] */
        ok = mlx_transpose_axes(&kt, kf, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&vtt, vf, perm, 4, m->stream) == 0;
    }
    mlx_array kc = mlx_array_new(), vc = mlx_array_new();
    if (ok) ok = mlx_contiguous(&kc, kt, false, m->stream) == 0 &&
                 mlx_contiguous(&vc, vtt, false, m->stream) == 0;
    /* Un solo mlx_eval su [kc,vc] invece di 2 mlx_array_eval separate -- stesso motivo
     * del gemello in llm_mlx_kvmetal_step sotto (un round-trip CPU<->GPU invece di due). */
    if (ok) {
        mlx_array kv2[2]; kv2[0] = kc; kv2[1] = vc;
        mlx_vector_array kv_vec = mlx_vector_array_new_data(kv2, 2);
        ok = mlx_eval(kv_vec) == 0;
        mlx_vector_array_free(kv_vec);
    }
    const uint16_t* kd = ok ? (const uint16_t*)mlx_array_data_float16(kc) : NULL;
    const uint16_t* vd = ok ? (const uint16_t*)mlx_array_data_float16(vc) : NULL;
    if (ok && kd && vd) {
        const size_t tok_elems = (size_t)nkv * (size_t)hd;
        for (int t = 0; t < n_tokens && ok; t++) {
            int pos = cache_offset + t;
            int page_id = pos / page_tokens, off = pos % page_tokens;
            ok = llm_kvmetal_write(ctx, page_id, off, kd + (size_t)t * tok_elems, vd + (size_t)t * tok_elems) == 0;
        }
    } else {
        ok = false;
    }
    mlx_array_free(kf); mlx_array_free(vf); mlx_array_free(kt); mlx_array_free(vtt);
    mlx_array_free(kc); mlx_array_free(vc);
    return ok;
}

bool llm_mlx_kvmetal_step(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m, int cache_offset,
                                  mlx_array q_rope, mlx_array k_rope, mlx_array v_t,
                                  mlx_array* attn_out) {
    if (!kvmetal_ctx_arr) return false;
    const int page_tokens = 16;
    const int nq = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;

    LlmKvMetalCtx* ctx = (LlmKvMetalCtx*)kvmetal_ctx_arr[layer_idx];
    if (!ctx) {
        int n_pages = (m->cfg.kv_max_tokens + page_tokens - 1) / page_tokens;
        if (n_pages < 1) n_pages = 1;
        ctx = llm_kvmetal_create(n_pages, page_tokens, nq, nkv, hd);
        if (!ctx) return false;
        kvmetal_ctx_arr[layer_idx] = ctx;
    }

    /* LLMMETAL_KVMETAL_TIME=1 (diagnostica, default OFF, mai nel percorso caldo se non impostata):
     * split del costo per-layer in eval_qkv (mlx_array_eval forza la materializzazione
     * dell'INTERO grafo lazy accumulato fino a qui, non solo di q/k/v -- sospetto principale
     * del rallentamento, non ancora isolato da waitUntilCompleted) vs metal_dispatch (write +
     * commit/waitUntilCompleted del kernel) vs wrap (mlx_array_new_data + astype indietro). */
    const bool kvtime = llm_env_isset("LLMMETAL_KVMETAL_TIME");
    struct timespec kt0, kt1, kt2, kt3;
    if (kvtime) clock_gettime(CLOCK_MONOTONIC, &kt0);

    mlx_dtype qdt = mlx_array_dtype(q_rope);
    mlx_array qf = mlx_array_new(), kf = mlx_array_new(), vf = mlx_array_new();
    bool ok = (mlx_array_dtype(q_rope) == MLX_FLOAT16) ? mlx_array_set(&qf, q_rope) == 0
                                                        : mlx_astype(&qf, q_rope, MLX_FLOAT16, m->stream) == 0;
    ok = ok && ((mlx_array_dtype(k_rope) == MLX_FLOAT16) ? mlx_array_set(&kf, k_rope) == 0
                                                          : mlx_astype(&kf, k_rope, MLX_FLOAT16, m->stream) == 0);
    ok = ok && ((mlx_array_dtype(v_t) == MLX_FLOAT16) ? mlx_array_set(&vf, v_t) == 0
                                                       : mlx_astype(&vf, v_t, MLX_FLOAT16, m->stream) == 0);
    mlx_array qc = mlx_array_new(), kc = mlx_array_new(), vc = mlx_array_new();
    if (ok) ok = mlx_contiguous(&qc, qf, false, m->stream) == 0 &&
                 mlx_contiguous(&kc, kf, false, m->stream) == 0 &&
                 mlx_contiguous(&vc, vf, false, m->stream) == 0;
    /* Un solo mlx_eval su [qc,kc,vc] invece di 3 mlx_array_eval separate (2026-09-07,
     * verifica LLMMETAL_KVMETAL_TIME): ciascuna mlx_array_eval e' un round-trip CPU<->GPU
     * indipendente (submit+wait), anche se le tre array condividono lo stesso grafo a monte
     * (RMSNorm del layer) -- 3 sync separati pagano 3 volte il costo fisso del round-trip.
     * mlx_eval su un vettore costruisce e sottomette UN SOLO grafo combinato, un solo wait.
     * Non cambia NESSUN valore numerico (stesso identico calcolo), solo il numero di sync. */
    if (ok) {
        mlx_array qkv3[3]; qkv3[0] = qc; qkv3[1] = kc; qkv3[2] = vc;
        mlx_vector_array qkv_vec = mlx_vector_array_new_data(qkv3, 3);
        ok = mlx_eval(qkv_vec) == 0;
        mlx_vector_array_free(qkv_vec);
    }
    if (kvtime) clock_gettime(CLOCK_MONOTONIC, &kt1);
    const uint16_t* qd = ok ? (const uint16_t*)mlx_array_data_float16(qc) : NULL;
    const uint16_t* kd = ok ? (const uint16_t*)mlx_array_data_float16(kc) : NULL;
    const uint16_t* vd = ok ? (const uint16_t*)mlx_array_data_float16(vc) : NULL;
    bool result = false;
    if (ok && qd && kd && vd) {
        int page_id = cache_offset / page_tokens, off = cache_offset % page_tokens;
        bool wrote = llm_kvmetal_write(ctx, page_id, off, kd, vd) == 0;
        int n_pages_used = cache_offset / page_tokens + 1;
        int last_page_len = (cache_offset % page_tokens) + 1;
        int* page_table = (int*)malloc((size_t)n_pages_used * sizeof(int));
        uint16_t* out_raw = (uint16_t*)malloc((size_t)nq * hd * sizeof(uint16_t));
        if (page_table && out_raw) {
            for (int i = 0; i < n_pages_used; i++) page_table[i] = i;
            bool attn_ok = wrote && llm_kvmetal_decode_attention(
                ctx, qd, page_table, n_pages_used, last_page_len,
                1.0f / sqrtf((float)hd), out_raw) == 0;
            if (kvtime) clock_gettime(CLOCK_MONOTONIC, &kt2);
            if (attn_ok) {
                int oshape[4] = {1, nq, 1, hd};
                mlx_array raw = mlx_array_new_data(out_raw, oshape, 4, MLX_FLOAT16);
                result = (qdt == MLX_FLOAT16) ? mlx_array_set(attn_out, raw) == 0
                                               : mlx_astype(attn_out, raw, qdt, m->stream) == 0;
                mlx_array_free(raw);
            }
        }
        free(page_table); free(out_raw);
    }
    mlx_array_free(qf); mlx_array_free(kf); mlx_array_free(vf);
    mlx_array_free(qc); mlx_array_free(kc); mlx_array_free(vc);
    if (kvtime) {
        clock_gettime(CLOCK_MONOTONIC, &kt3);
        double eval_ms = (kt1.tv_sec - kt0.tv_sec) * 1000.0 + (kt1.tv_nsec - kt0.tv_nsec) / 1e6;
        double dispatch_ms = (kt2.tv_sec - kt1.tv_sec) * 1000.0 + (kt2.tv_nsec - kt1.tv_nsec) / 1e6;
        double wrap_ms = (kt3.tv_sec - kt2.tv_sec) * 1000.0 + (kt3.tv_nsec - kt2.tv_nsec) / 1e6;
        fprintf(stderr, "[kvmetal-time] layer=%d eval_qkv=%.3fms metal_dispatch=%.3fms wrap=%.3fms\n",
                layer_idx, eval_ms, dispatch_ms, wrap_ms);
    }
    return result;
}

/* Max abs diff fra due array (convertiti in float32 e valutati). Ritorna -1 se
 * la conversione/lettura fallisce. SOLO diagnostica: forza un eval per chiamata,
 * quindi non va mai nel percorso di produzione (usato da LLM_*_VERIFY). */
/* LLM_KVSEG — merge softmax di DUE attention parziali (main + segmento) con max-trick.
 *
 * Ogni SDPA parziale produce o_i = (1/l_i) * sum(exp(s_i) * v_i) con l_i = sum(exp(s_i))
 * il denominatore NON normalizzato (softmax "scaldato"). Il merge esatto e':
 *   m = max globale dei punteggi (per stabilita');  w_i = exp(s_i - m)  (per-elemento);
 *   l = sum(w_1) + sum(w_2);  o = (sum(w_1 o_1') + sum(w_2 o_2')) / l
 * dove o_i' = o_i * l_i (denormalizzazione: sum(exp(s_i) v_i)).
 * Le SDPA standard (llm_mlx_gqa_sdpa) ritornano o_i GIA' normalizzate, quindi:
 *   o = (l1 * o1 * e1 + l2 * o2 * e2) / (l1*e1 + l2*e2)
 * con e_i = exp(m_i - m) e m_i = massimo locale dei punteggi della parte i.
 *
 * q_rope: [1,H,1,D]. k_i: [1,Hkv,L_i,D] (row-contiguous non richiesto da matmul).
 * out: [1,H,1,D] come le SDPA. Tutto nel grafo lazy (nessun eval qui). */
bool llm_mlx_kvseg_merge_two(LlmMlxModel* m, mlx_array q_rope,
                                    mlx_array k1, mlx_array v1,
                                    mlx_array k2, mlx_array v2,
                                    mlx_array* out) {
    const int nq = m->cfg.n_heads, nkv = m->cfg.n_kv_heads;
    if (nq <= 0 || nkv <= 0 || nq % nkv != 0) return false;
    const int g = nq / nkv;   /* GQA: ogni kv head serve a g query head */
    const float scale = 1.0f / sqrtf((float)m->cfg.head_dim);
    /* Armonizzazione dtype: i punteggi e il merge si calcolano in fp32 (robusto),
     * le SDPA parziali le fa gqa_sdpa (gestisce la GQA internamente). Alla fine il
     * risultato torna al dtype di q_rope (come il ramo eager). */
    mlx_dtype out_dt = mlx_array_dtype(q_rope);
    mlx_array q32 = mlx_array_new();
    bool ok = mlx_astype(&q32, q_rope, MLX_FLOAT32, m->stream) == 0;
    /* Espandi le head KV per il matmul dei punteggi: [1,nkv,D,L] -> repeat contiguo
     * lungo l'asse head -> [1,nq,D,L] con kv_head r ripetuta g volte (pattern GQA di
     * fast_sdpa: q head h usa kv head h/g). */
    const int perm[4] = {0, 1, 3, 2};
    mlx_array k1t = mlx_array_new(), k2t = mlx_array_new();
    mlx_array k1e = mlx_array_new(), k2e = mlx_array_new();
    mlx_array mm1 = mlx_array_new(), mm2 = mlx_array_new();
    mlx_array sc1 = mlx_array_new(), sc2 = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&k1t, k1, perm, 4, m->stream) == 0 &&
                 mlx_transpose_axes(&k2t, k2, perm, 4, m->stream) == 0;
    if (ok) ok = mlx_repeat_axis(&k1e, k1t, g, 1, m->stream) == 0 &&
                 mlx_repeat_axis(&k2e, k2t, g, 1, m->stream) == 0;
    if (ok) ok = mlx_matmul(&mm1, q32, k1e, m->stream) == 0 &&
                 mlx_matmul(&mm2, q32, k2e, m->stream) == 0;
    mlx_array_free(k1t); mlx_array_free(k2t);
    mlx_array_free(k1e); mlx_array_free(k2e);
    mlx_array scale_arr = mlx_array_new();
    if (ok) {
        float sv[1] = {scale};
        mlx_array sa = mlx_array_new_data(sv, NULL, 0, MLX_FLOAT32);
        ok = sa.ctx != NULL;
        if (ok) scale_arr = sa; else mlx_array_free(sa);
    }
    if (ok) ok = mlx_multiply(&sc1, mm1, scale_arr, m->stream) == 0 &&
                 mlx_multiply(&sc2, mm2, scale_arr, m->stream) == 0;
    mlx_array_free(mm1); mlx_array_free(mm2);
    mlx_array m1 = mlx_array_new(), m2 = mlx_array_new();
    if (ok) ok = mlx_max_axis(&m1, sc1, 3, true, m->stream) == 0 &&
                 mlx_max_axis(&m2, sc2, 3, true, m->stream) == 0;
    mlx_array gm = mlx_array_new();
    if (ok) ok = mlx_maximum(&gm, m1, m2, m->stream) == 0;
    mlx_array dm1 = mlx_array_new(), dm2 = mlx_array_new();
    mlx_array e1 = mlx_array_new(), e2 = mlx_array_new();
    mlx_array ls1 = mlx_array_new(), ls2 = mlx_array_new();
    if (ok) ok = mlx_subtract(&dm1, m1, gm, m->stream) == 0 &&
                 mlx_subtract(&dm2, m2, gm, m->stream) == 0;
    if (ok) ok = mlx_exp(&e1, dm1, m->stream) == 0 && mlx_exp(&e2, dm2, m->stream) == 0;
    mlx_array_free(dm1); mlx_array_free(dm2);
    mlx_array gsc1 = mlx_array_new(), gsc2 = mlx_array_new();
    if (ok) ok = mlx_subtract(&gsc1, sc1, gm, m->stream) == 0 &&
                 mlx_subtract(&gsc2, sc2, gm, m->stream) == 0;
    mlx_array_free(sc1); mlx_array_free(sc2);
    mlx_array ex1 = mlx_array_new(), ex2 = mlx_array_new();
    if (ok) ok = mlx_exp(&ex1, gsc1, m->stream) == 0 && mlx_exp(&ex2, gsc2, m->stream) == 0;
    mlx_array_free(gsc1); mlx_array_free(gsc2);
    if (ok) ok = mlx_sum_axis(&ls1, ex1, 3, true, m->stream) == 0 &&
                 mlx_sum_axis(&ls2, ex2, 3, true, m->stream) == 0;
    mlx_array_free(ex1); mlx_array_free(ex2);
    mlx_array L1 = mlx_array_new(), L2 = mlx_array_new();
    if (ok) ok = mlx_multiply(&L1, ls1, e1, m->stream) == 0 &&
                 mlx_multiply(&L2, ls2, e2, m->stream) == 0;
    mlx_array_free(ls1); mlx_array_free(ls2); mlx_array_free(e1); mlx_array_free(e2);
    mlx_array o1 = mlx_array_new(), o2 = mlx_array_new();
    if (ok) ok = llm_mlx_gqa_sdpa(&o1, q_rope, k1, v1, scale, 0, m->stream);
    if (ok) ok = llm_mlx_gqa_sdpa(&o2, q_rope, k2, v2, scale, 0, m->stream);
    /* Armonizza o1/o2 a fp32 per il merge (gqa_sdpa può ritornare nel dtype di q). */
    mlx_array o1f = mlx_array_new(), o2f = mlx_array_new();
    if (ok) ok = mlx_astype(&o1f, o1, MLX_FLOAT32, m->stream) == 0 &&
                 mlx_astype(&o2f, o2, MLX_FLOAT32, m->stream) == 0;
    mlx_array_free(o1); mlx_array_free(o2);
    mlx_array n1 = mlx_array_new(), n2 = mlx_array_new();
    if (ok) ok = mlx_multiply(&n1, L1, o1f, m->stream) == 0 &&
                 mlx_multiply(&n2, L2, o2f, m->stream) == 0;
    mlx_array_free(o1f); mlx_array_free(o2f);
    mlx_array num = mlx_array_new(), den = mlx_array_new();
    if (ok) ok = mlx_add(&num, n1, n2, m->stream) == 0 &&
                 mlx_add(&den, L1, L2, m->stream) == 0;
    mlx_array_free(n1); mlx_array_free(n2); mlx_array_free(L1); mlx_array_free(L2);
    mlx_array res = mlx_array_new();
    if (ok) ok = mlx_divide(&res, num, den, m->stream) == 0;
    mlx_array_free(num); mlx_array_free(den);
    if (ok && out_dt != MLX_FLOAT32)
        ok = mlx_astype(out, res, out_dt, m->stream) == 0;
    else if (ok) ok = mlx_array_set(out, res) == 0;
    mlx_array_free(res);
    mlx_array_free(q32);
    mlx_array_free(gm);  mlx_array_free(m1);  mlx_array_free(m2);
    mlx_array_free(scale_arr);
    return ok;
}

double llm_mlx_max_abs_diff(mlx_array a, mlx_array b, size_t n, mlx_stream s) {
    mlx_array af = mlx_array_new(), bf = mlx_array_new();
    double md = -1.0;
    if (mlx_astype(&af, a, MLX_FLOAT32, s) == 0 && mlx_astype(&bf, b, MLX_FLOAT32, s) == 0 &&
        mlx_array_eval(af) == 0 && mlx_array_eval(bf) == 0) {
        const float* ad = mlx_array_data_float32(af);
        const float* bd = mlx_array_data_float32(bf);
        if (ad != NULL && bd != NULL) {
            md = 0.0;
            for (size_t i = 0; i < n; i++) {
                double d = fabs((double)ad[i] - (double)bd[i]);
                if (d > md) md = d;
            }
        }
    }
    mlx_array_free(af); mlx_array_free(bf);
    return md;
}
