/* ============================================================================
 * llm_mlx_session_spec.c — dominio speculative/verify (T2), truncate, greedy_prob, constrained e n-gram self-speculative.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_ngram.h"
#include "llm_mlx_qwen4exp.h"
#include "llm_dsv41_engine.h"
#include "llm_dsv4_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* T2 speculative — verify a 1 token (tappa 1). Vedi il commento nell'header. */
int llm_ai_mlx_session_verify_token(llm_ai_mlx_session_handle handle,
                                     int32_t proposed_token,
                                     int32_t* out_accepted,
                                     int32_t* out_next_token) {
    if (handle == NULL || out_accepted == NULL || out_next_token == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    if (out_next_token) *out_next_token = s->pending_next_id;

    /* Il greedy corrente e' gia' calcolato (pending_next_id): confronto diretto. */
    if (proposed_token != s->pending_next_id) {
        if (out_accepted) *out_accepted = 0;
        return 0;
    }

    /* Draft corretto: avanza la KV di 1 (stesso pattern del corpo di session_generate). */
    LlmMlxModel* m = s->model;
    OutputTensors t;
    bool ok = load_output_tensors(m, &t);
    if (!ok) return -1;

    mlx_array h = mlx_array_new();
    ok = forward_one_token(m, s, t.embed_w, t.embed_s, t.embed_b, proposed_token, s->cache_offset,
                           s->k_cache, s->v_cache, s->has_cache, &h);
        int32_t next_id = 0;
        if (ok) ok = final_step(m, h, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                llm_mlx_session_sampling(s), &next_id);
        if (llm_da_enabled()) { g_da.final_steps++; g_da.readbacks++; }  /* final_step legge l'argmax */
    if (ok) {
        mlx_array_free(s->pending_h);
        s->pending_h = mlx_array_new();
        mlx_array_set(&s->pending_h, h);
        s->pending_next_id = next_id;
    }
    mlx_array_free(h);
    s->cache_offset++;
    free_output_tensors(&t);

    if (!ok) return -1;
    if (out_accepted) *out_accepted = 1;
    if (out_next_token) *out_next_token = proposed_token;
    return 0;
}

/* T2 speculative — verify BATCH (tappa 2). Vedi il commento nell'header.
 * Sampling: usa l'argmax (greedy) per il confronto E per il next_token; parita'
 * col decode normale garantita SOLO con sampling greedy (temp<=0). */
int llm_ai_mlx_session_verify_draft_batch(llm_ai_mlx_session_handle handle,
                                           const int32_t* draft_ids, int n_draft,
                                           int32_t* out_accepted,
                                           int32_t* out_next_token) {
    if (handle == NULL || draft_ids == NULL || n_draft <= 0 ||
        out_accepted == NULL || out_next_token == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    LlmMlxModel* m = s->model;
    /* P1 Passo 2a (2026-09-19): i modelli SENZA layer ricorrenti (GDN) possono troncare la KV
     * per SLICE (rollback O(1), NIENTE re-forward del prefisso accettato). Il checkpoint +
     * re-append serve SOLO allo stato ricorrente 4D / conv_state 3D, non troncabile per slice
     * (era il bug 2026-09-19 sul verify GDN). LLM_SPEC_REAPPEND=1 forza il vecchio path (A/B). */
    bool has_linear = false;
    if (m->cfg.is_linear_layer != NULL)
        for (int l = 0; l < m->cfg.n_layers; l++)
            if (m->cfg.is_linear_layer[l]) { has_linear = true; break; }
    const bool slice_rollback = !has_linear && llm_ai_mlx_opt_env("LLM_SPEC_REAPPEND") == NULL;
    /* S5 (2026-09-21): sui modelli ibridi (GDN) cattura lo stato ricorrente PER-TOKEN
     * durante il batch -> su rifiuto si ripristina con un alias O(1), SENZA re-forward. */
    const bool cap_ok = has_linear && llm_ai_mlx_opt_env("LLM_SPEC_NOCAP") == NULL;
    /* Il greedy dopo il prefisso (posizione 0 del draft) e' gia' calcolato. */
    int32_t greedy0 = s->pending_next_id;

    OutputTensors t;
    bool ok = load_output_tensors(m, &t);
    if (!ok) return -1;

    int n_prev = s->cache_offset;

    /* 1. Forward batch dei K draft (KV avanza di K in un grafo).
     * CHECKPOINT O(1) delle cache prima del batch (alias immutabili): il rollback
     * per-slice qui sotto vale solo per KV 4D, mentre sui modelli GDN (qwen3_5)
     * k_cache/v_cache portano stato ricorrente 4D e conv_state 3D -> slice 4D =
     * errore MLX "[slice] ... dimension 3" (bug 2026-09-19). Pattern dal runtime
     * DFlash2: ripristino del checkpoint + ri-append del SOLO prefisso confermato. */
    const int n_ck = m->cfg.n_layers;
    mlx_array* ck = (mlx_array*)calloc((size_t)n_ck, sizeof(mlx_array));
    mlx_array* cv = (mlx_array*)calloc((size_t)n_ck, sizeof(mlx_array));
    if (ck != NULL && cv != NULL) {
        for (int l = 0; l < n_ck; l++) {
            ck[l] = mlx_array_new(); cv[l] = mlx_array_new();
            (void)llm_mlx_cache_checkpoint(&ck[l], s->k_cache[l], m->stream);
            (void)llm_mlx_cache_checkpoint(&cv[l], s->v_cache[l], m->stream);
        }
    }
    mlx_array h_batch = mlx_array_new();
    struct timespec _st0,_st1; clock_gettime(CLOCK_MONOTONIC,&_st0);
    if (cap_ok) g_spec_capture = 1;
    ok = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, draft_ids, n_draft,
                              n_prev, s->k_cache, s->v_cache, s->has_cache, &h_batch);
    if (cap_ok) g_spec_capture = 0;
    clock_gettime(CLOCK_MONOTONIC,&_st1);
    double _t_batch = (_st1.tv_sec-_st0.tv_sec)*1e3+(_st1.tv_nsec-_st0.tv_nsec)/1e6;
    if (!ok) {
        mlx_array_free(h_batch); free_output_tensors(&t);
        if (ck && cv) for (int l = 0; l < n_ck; l++) { llm_mlx_cache_commit(&ck[l], m->stream); llm_mlx_cache_commit(&cv[l], m->stream); }
        free(ck); free(cv);
        return -1;
    }

    /* 2. rms_norm + lm_head sull'intero batch [1, K, hidden] -> logits [1, K, vocab]. */
    mlx_array normed = mlx_array_new();
    ok = mlx_fast_rms_norm(&normed, h_batch, t.final_norm_w, m->cfg.rms_norm_eps, m->stream) == 0;
    mlx_array logits = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b, m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    mlx_array capped = mlx_array_new();
    if (ok && m->cfg.final_logit_softcap > 0.0f)
        ok = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
    else if (ok) ok = mlx_array_set(&capped, logits) == 0;

    /* 3. argmax su axis -1 -> K token greedy in un passo. */
    mlx_array am = mlx_array_new();
    if (ok) ok = mlx_argmax_axis(&am, capped, -1, /*keepdims=*/false, m->stream) == 0;
    if (ok) ok = mlx_array_eval(am) == 0;
    struct timespec _st2; clock_gettime(CLOCK_MONOTONIC,&_st2);
    double _t_head = (_st2.tv_sec-_st1.tv_sec)*1e3+(_st2.tv_nsec-_st1.tv_nsec)/1e6;

    int accepted = 0;
    int32_t next_token = greedy0;
    if (ok) {
        const uint32_t* d = mlx_array_data_uint32(am);
        size_t n_out = mlx_array_size(am);
        ok = (d != NULL && n_out == (size_t)n_draft);
        if (ok) {
            /* Confronto: greedy0 (dopo il prefisso) vs draft[0], poi greedy[i] (dopo
             * draft[i]) vs draft[i+1]. Nota: greedy[0] predice il token DOPO draft[0]. */
            if (greedy0 == draft_ids[0]) {
                accepted = 1;
                for (int i = 0; i < n_draft - 1; i++) {
                    if (d[i] == draft_ids[i + 1]) accepted++;
                    else break;
                }
            }
            if (accepted >= 1 && accepted <= n_draft)
                next_token = (accepted == n_draft) ? (int32_t)d[n_draft - 1] : (int32_t)d[accepted - 1];
        }
    }

    mlx_array_free(normed);
    mlx_array_free(logits);
    mlx_array_free(capped);
    mlx_array_free(am);

    if (!ok) {
        mlx_array_free(h_batch); free_output_tensors(&t);
        if (ck && cv) for (int l = 0; l < n_ck; l++) { llm_mlx_cache_commit(&ck[l], m->stream); llm_mlx_cache_commit(&cv[l], m->stream); }
        free(ck); free(cv);
        return -1;
    }

    /* 4. Rollback: se non tutti i K accettati, RIPRISTINA IL CHECKPOINT e ri-avanza il
     * SOLO prefisso confermato (stesso pattern del runtime DFlash2). Vale per ogni
     * forma di cache (4D attenzione, 4D stato ricorrente, 3D conv_state). */
    int keep = n_prev + accepted;
    bool pending_h_from_reappend = false;
    if (accepted < n_draft) {
      if (cap_ok) {
        /* S5: GDN -> stato dal SNAPSHOT per-token (alias O(1)); attenzione -> slice.
         * Nessun re-forward. pending_h arriva da h_batch[accepted-1] allo step 5. */
        for (int layer = 0; layer < n_ck; layer++) {
            const bool lin = m->cfg.is_linear_layer && m->cfg.is_linear_layer[layer];
            if (lin) {
                /* stato dopo `accepted` token = state_seq[:, accepted-1] (take_axis: array
                 * NUOVO, non slice-view) + conv_input[:, accepted : accepted+cstate_len]. */
                if (accepted >= 1 && g_spec_snap_k[layer].ctx != NULL && g_spec_snap_v[layer].ctx != NULL) {
                    int ix[1] = {accepted - 1}, ish[1] = {1};
                    mlx_array idx = mlx_array_new_data(ix, ish, 1, MLX_INT32);
                    mlx_array st = mlx_array_new();
                    if (mlx_take_axis(&st, g_spec_snap_k[layer], idx, 1, m->stream) == 0) {
                        mlx_array_free(s->k_cache[layer]); s->k_cache[layer] = st;
                    } else { mlx_array_free(st); ok = false; }
                    mlx_array_free(idx);
                    int nd = mlx_array_ndim(g_spec_snap_v[layer]);
                    if (nd == 3) {
                        const int* cvs = mlx_array_shape(g_spec_snap_v[layer]);
                        int csl = cvs[1] - n_draft;                       /* cstate_len */
                        int cv0[3] = {0, accepted, 0}, cv1[3] = {1, accepted + csl, cvs[2]}, cvst[3] = {1,1,1};
                        mlx_array vt = mlx_array_new();
                        if (mlx_slice(&vt, g_spec_snap_v[layer], cv0, 3, cv1, 3, cvst, 3, m->stream) == 0) {
                            mlx_array_free(s->v_cache[layer]); s->v_cache[layer] = vt;
                        } else { mlx_array_free(vt); ok = false; }
                    }
                } else {
                    /* fail-loud: mai stato silenziosamente sbagliato se lo snapshot manca */
                    fprintf(stderr, "spec: snapshot GDN mancante (layer %d, accepted %d)\n", layer, accepted);
                    ok = false;
                }
            } else {
                const int* shape = mlx_array_shape(s->k_cache[layer]);
                int n_kv = shape[1], hd = shape[3];
                int sr[4] = {0, 0, 0, 0};
                int sp[4] = {1, n_kv, keep, hd};
                int ss[4] = {1, 1, 1, 1};
                mlx_array kt = mlx_array_new(), vt = mlx_array_new();
                bool kr = mlx_slice(&kt, s->k_cache[layer], sr, 4, sp, 4, ss, 4, m->stream) == 0;
                bool vr = mlx_slice(&vt, s->v_cache[layer], sr, 4, sp, 4, ss, 4, m->stream) == 0;
                if (kr && vr) {
                    mlx_array_free(s->k_cache[layer]); s->k_cache[layer] = kt;
                    mlx_array_free(s->v_cache[layer]); s->v_cache[layer] = vt;
                } else { mlx_array_free(kt); mlx_array_free(vt); ok = false; }
            }
        }
      } else if (slice_rollback) {
        /* Solo-attenzione: la KV 4D [B,nkv,cap,hd] si TRONCA ai soli token accettati
         * (nessun re-forward: pending_h arriva da h_batch allo step 5). */
        for (int layer = 0; layer < n_ck; layer++) {
            const int* shape = mlx_array_shape(s->k_cache[layer]);
            int n_kv = shape[1], hd = shape[3];
            int sr[4] = {0, 0, 0, 0};
            int sp[4] = {1, n_kv, keep, hd};
            int ss[4] = {1, 1, 1, 1};
            mlx_array kt = mlx_array_new(), vt = mlx_array_new();
            bool kr = mlx_slice(&kt, s->k_cache[layer], sr, 4, sp, 4, ss, 4, m->stream) == 0;
            bool vr = mlx_slice(&vt, s->v_cache[layer], sr, 4, sp, 4, ss, 4, m->stream) == 0;
            if (kr && vr) {
                mlx_array_free(s->k_cache[layer]); s->k_cache[layer] = kt;
                mlx_array_free(s->v_cache[layer]); s->v_cache[layer] = vt;
            } else { mlx_array_free(kt); mlx_array_free(vt); ok = false; }
        }
      } else {
        for (int layer = 0; layer < n_ck; layer++) {
            (void)llm_mlx_cache_rollback(&s->k_cache[layer], ck[layer], m->stream);
            (void)llm_mlx_cache_rollback(&s->v_cache[layer], cv[layer], m->stream);
        }
        if (accepted > 0) {
            /* Ri-avanza [draft_ids[0..accepted)) dal checkpoint: KV e stato ricorrente
             * tornano coerenti con cache_offset=keep; pending_h dal hidden ri-calcolato. */
            mlx_array h_re = mlx_array_new();
            if (forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, draft_ids, accepted,
                                     n_prev, s->k_cache, s->v_cache, s->has_cache, &h_re)) {
                int st2[3] = {0, accepted - 1, 0};
                int sp2[3] = {1, accepted, mlx_array_dim(h_re, 2)};
                int ss2[3] = {1, 1, 1};
                mlx_array ph2 = mlx_array_new();
                if (mlx_slice(&ph2, h_re, st2, 3, sp2, 3, ss2, 3, m->stream) == 0) {
                    mlx_array_free(s->pending_h);
                    s->pending_h = ph2;
                    pending_h_from_reappend = true;
                } else mlx_array_free(ph2);
            } else ok = false;
            mlx_array_free(h_re);
        }
      }
    }
    for (int l = 0; l < n_ck; l++) { llm_mlx_cache_commit(&ck[l], m->stream); llm_mlx_cache_commit(&cv[l], m->stream); }
    free(ck); free(cv);

    /* 5. Riallinea cache_offset, pending_h/pending_next_id. */
    s->cache_offset = keep;
    if (ok && accepted > 0) {
        /* BUGFIX 2026-09-19: `pending_next_id` va aggiornato SEMPRE che accepted>0, anche quando
         * pending_h viene dal re-append (`h_re`). Prima era dentro `!pending_h_from_reappend` ->
         * nel ramo re-append (modelli GDN) restava STALE (= greedy0) e `commitToken` del token
         * corretto falliva -> la generazione speculativa si fermava (trovato dal test
         * `specparity` col path LLM_SPEC_REAPPEND=1). */
        if (!pending_h_from_reappend) {
            int start[3] = {0, accepted - 1, 0};
            int stop[3] = {1, accepted, mlx_array_dim(h_batch, 2)}; /* vedi nota hc_width sopra */
            int stride[3] = {1, 1, 1};
            mlx_array ph = mlx_array_new();
            if (ok) ok = mlx_slice(&ph, h_batch, start, 3, stop, 3, stride, 3, m->stream) == 0;
            if (ok) {
                mlx_array_free(s->pending_h);
                s->pending_h = ph;
            } else mlx_array_free(ph);
        }
        s->pending_next_id = next_token;
    }

    /* Materializza SUBITO pending_h (slice di h_batch). Senza questo, l'eval del
     * forward MTP successivo (che legge pending_h) forza la materializzazione dello
     * slice DOPO il free di h_batch -> costo nascosto nel MTP (~127ms osservati).
     * Eager-eval qui = il costo resta attribuito al verify (dove appartiene). */
    if (ok && s->pending_h.ctx != NULL) (void)mlx_array_eval(s->pending_h);

    mlx_array_free(h_batch);
    free_output_tensors(&t);

    if (!ok) return -1;
    { struct timespec _st3; clock_gettime(CLOCK_MONOTONIC,&_st3);
      double _t_roll = (_st3.tv_sec-_st2.tv_sec)*1e3+(_st3.tv_nsec-_st2.tv_nsec)/1e6;
      if (getenv("LLM_SPEC_TIME"))
          fprintf(stderr, "[spec-time] K=%d acc=%d | batch=%.1fms head(eval)=%.1fms rollback=%.1fms\n",
                  n_draft, accepted, _t_batch, _t_head, _t_roll); }
    *out_accepted = accepted;
    *out_next_token = next_token;
    return 0;
}

/* forward_tokens_batch_multi — (A) kernel N×K: N sessioni, ks[s] token ciascuna, in UN grafo.
 * Compone l'attenzione PER-SESSIONE (m->arch->attn con n_tokens=ks[s], al cache_offset della
 * sessione) con la FFN BATCHED su sum(ks) (m->arch->ffn): i pesi del layer si leggono una sola
 * volta per N*K token — e' il guadagno rispetto a K forward N×1 (misurato: (B) N×1 = 35 tok/s
 * con N=4,K=8, cioe' nessun guadagno sul greedy batched; il forward N×K e' la fonte reale).
 * toks e' COMPATTO in ordine sessione; offs[s] = offset della sessione s in toks.
 * out_h -> [1, sum(ks), hidden]. NON gestisce sliding-window/KV-quant (lo fa layer_forward): il
 * chiamante deve garantire l'eligibilita' (vedi il guard in verify_draft_batch_multi). */
static bool forward_tokens_batch_multi(LlmMlxModel* m, LlmMlxSession** sess, const int* ks, int n_sess,
                                       const int32_t* toks, const int* offs,
                                       mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                       mlx_array* out_h) {
    return forward_tokens_batch_multi_cap(m, sess, ks, n_sess, toks, offs,
                                          embed_w, embed_s, embed_b, /*caplayers=*/NULL, /*ncap=*/0,
                                          /*capout=*/NULL, /*capcap=*/0, out_h);
}

/* Tappa 2 DFlash2 multi-sessione: come forward_tokens_batch_multi ma con CATTURA opzionale degli
 * hidden post-layer dei `caplayers` (dflash_ctx_layers), fp32, per-sessione. `capout` e' COMPATTO
 * in ordine sessione: layout [total_rows, ncap, H] con total_rows = sum(ks) (il chiamante indicizza
 * per sessione con `offs`). Un solo grafo: gli snap si accumulano lazy e si materializzano TUTTI
 * insieme a fine forward (come df2_capture_batch). */
bool forward_tokens_batch_multi_cap(LlmMlxModel* m, LlmMlxSession** sess, const int* ks, int n_sess,
                                           const int32_t* toks, const int* offs,
                                           mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                           const int* caplayers, int ncap,
                                           float* capout, int capcap, mlx_array* out_h) {
    int total = 0;
    for (int s = 0; s < n_sess; s++) total += ks[s];
    if (total <= 0) return false;
    mlx_array x = mlx_array_new();
    if (!embed_tokens_batch(m, embed_w, embed_s, embed_b, toks, total, &x)) { mlx_array_free(x); return false; }
    mlx_array h = x;
    mlx_array snaps[8];
    int snaps_n = 0;
    bool ok_cap = true;
    for (int layer = 0; layer < m->cfg.n_layers; layer++) {
        mlx_array hb = mlx_array_new();
        mlx_vector_array hs = mlx_vector_array_new();
        bool ok = true;
        for (int s = 0; s < n_sess && ok; s++) {
            int k = ks[s];
            if (k <= 0) continue;
            mlx_array x_s = mlx_array_new();
            const int st[3] = {0, offs[s], 0};
            const int sp[3] = {1, offs[s] + k, m->cfg.hidden_size};
            const int ss[3] = {1, 1, 1};
            ok = mlx_slice(&x_s, h, st, 3, sp, 3, ss, 3, m->stream) == 0;
            mlx_array h_s = mlx_array_new();
            /* SIDE-CHANNEL PER-SESSIONE (come df2_capture_batch/session_generate): lo stato
             * ricorrente dei modelli ibridi (GDN qwen3_5, MLA kimi) vive in `sess->aux_cache`,
             * e `arch->attn` lo legge da `m->aux_cache` (globale). Senza questo set il kernel N×K
             * leggerebbe un aux_cache stantio/NULL -> risultato errato o SIGSEGV (stessa classe
             * del fix 2026-09-16). q4e (PLE/ngram) idem via m->q4e_ctx_ids. */
            mlx_array* saved_aux = m->aux_cache;
            const int32_t* saved_q4e = m->q4e_ctx_ids;
            int saved_q4e_len = m->q4e_ctx_len;
            m->aux_cache = sess[s]->aux_cache;
            m->q4e_ctx_ids = sess[s]->history;
            m->q4e_ctx_len = sess[s]->history_len;
            if (ok) ok = m->arch->attn(m, layer, sess[s]->cache_offset, k, x_s,
                                       &sess[s]->k_cache[layer], &sess[s]->v_cache[layer],
                                       sess[s]->has_cache[layer], &h_s);
            m->aux_cache = saved_aux;
            m->q4e_ctx_ids = saved_q4e;
            m->q4e_ctx_len = saved_q4e_len;
            sess[s]->has_cache[layer] = true;
            if (ok) mlx_vector_array_append_value(hs, h_s);
            mlx_array_free(x_s);
            mlx_array_free(h_s);
        }
        if (ok) ok = mlx_concatenate_axis(&hb, hs, 1, m->stream) == 0;
        mlx_vector_array_free(hs);
        mlx_array layer_out = mlx_array_new();
        if (ok) ok = m->arch->ffn(m, layer, total, hb, &layer_out);
        mlx_array_free(hb);
        mlx_array_free(h);
        if (!ok) { mlx_array_free(layer_out); return false; }
        h = layer_out;
        /* Capture post-layer (tappa 2 DFlash2): snap fp32 LAZY del layer se e' un ctx layer. */
        if (ok && caplayers != NULL && snaps_n < 8) {
            for (int c = 0; c < ncap; c++)
                if (caplayers[c] == layer) {
                    mlx_array s32 = mlx_array_new();
                    ok = mlx_astype(&s32, h, MLX_FLOAT32, m->stream) == 0;
                    if (ok) snaps[snaps_n++] = s32; else mlx_array_free(s32);
                    break;
                }
        }
    }
    /* Materializza gli snap TUTTI insieme a fine forward e copia in capout [total, ncap, H]. */
    if (caplayers != NULL) {
        if (snaps_n != ncap) ok_cap = false;
        else {
            for (int i = 0; ok_cap && i < snaps_n; i++) ok_cap = mlx_array_eval(snaps[i]) == 0;
            int64_t need = (int64_t)total * ncap * m->cfg.hidden_size;
            if (ok_cap && (capout == NULL || (int64_t)capcap < need)) ok_cap = false;
            if (ok_cap) {
                for (int c = 0; c < ncap && ok_cap; c++) {
                    const float* d = mlx_array_data_float32(snaps[c]);
                    if (d == NULL) { ok_cap = false; break; }
                    for (int p = 0; p < total; p++)
                        memcpy(capout + ((size_t)p * ncap + c) * m->cfg.hidden_size,
                               d + (size_t)p * m->cfg.hidden_size,
                               (size_t)m->cfg.hidden_size * sizeof(float));
                }
            }
            for (int i = 0; i < snaps_n; i++) mlx_array_free(snaps[i]);
        }
    }
    if (!ok_cap) { mlx_array_free(h); return false; }
    mlx_array_set(out_h, h);
    mlx_array_free(h);
    return true;
}

/* (A) speculative batching — verify batch MULTI-SESSIONE con forward N×K in UN grafo.
 * Ogni sessione i propone K_i token (draft_ids row-major, draft_len[i]); il target verifica con
 * argmax (greedy) su TUTTI i draft in un colpo (un forward N×K: attenzione per-sessione + FFN
 * batched), poi accetta il piu' lungo prefisso confermato per sessione e, sui mismatch, ripristina
 * il checkpoint e ri-avanza il solo prefisso confermato (stesso pattern di verify_draft_batch).
 * Fallback per-sessione (verify_draft_batch) per i modelli con sliding-window/KV-quant, che
 * arch->attn da solo NON gestisce (lo fa layer_forward).
 * out_accepted[i]/out_next[i] per sessione. Parita' col greedy SOLO con temp<=0. Ritorna 0/-1. */
int llm_ai_mlx_session_verify_draft_batch_multi(
    llm_ai_mlx_session_handle* handles, int n,
    const int32_t* draft_ids, const int* draft_len, int max_draft,
    int32_t* out_accepted, int32_t* out_next) {
    if (handles == NULL || n <= 0 || draft_ids == NULL || draft_len == NULL || max_draft <= 0 ||
        out_accepted == NULL || out_next == NULL) return -1;
    LlmMlxSession* s0 = (LlmMlxSession*)handles[0];
    if (s0 == NULL) return -1;
    LlmMlxModel* m = s0->model;
    { int _g = llm_mlx_active_guard(m); if (_g != 0) return _g; }
    int total = 0;
    for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        out_accepted[i] = 0;
        if (s == NULL || s->model != m || !s->has_pending) return -1;
        out_next[i] = s->pending_next_id;
        if (draft_len[i] < 0 || draft_len[i] > max_draft) return -1;
        total += draft_len[i];
    }
    if (total <= 0) return 0;   /* niente draft: pending invariati */

    /* Fallback per-sessione: sliding-window o KV-quant richiedono layer_forward (che arch->attn da
     * solo non replica). Riusa la primitiva single-session verificata (S1). */
    if ((m->cfg.is_sliding_layer && m->cfg.sliding_window > 0) || m->cfg.kv_quantize) {
        for (int i = 0; i < n; i++) {
            LlmMlxSession* s = (LlmMlxSession*)handles[i];
            int K = draft_len[i];
            if (K <= 0) { out_accepted[i] = 0; out_next[i] = s->pending_next_id; continue; }
            if (llm_ai_mlx_session_verify_draft_batch(handles[i], draft_ids + (size_t)i * max_draft, K,
                                                      &out_accepted[i], &out_next[i]) != 0) return -1;
        }
        return 0;
    }

    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;

    const int n_ck = m->cfg.n_layers;
    int32_t* toks = (int32_t*)malloc(sizeof(int32_t) * (size_t)total);
    int* offs = (int*)malloc(sizeof(int) * (size_t)n);
    int* ks = (int*)malloc(sizeof(int) * (size_t)n);
    int* nprev = (int*)malloc(sizeof(int) * (size_t)n);
    mlx_array* ck = (mlx_array*)calloc((size_t)n * n_ck, sizeof(mlx_array));
    mlx_array* cv = (mlx_array*)calloc((size_t)n * n_ck, sizeof(mlx_array));
    bool ok = (toks != NULL && offs != NULL && ks != NULL && nprev != NULL && ck != NULL && cv != NULL);
    int o = 0;
    if (ok) for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        offs[i] = o; ks[i] = draft_len[i]; nprev[i] = s->cache_offset;
        if (draft_len[i] > 0)
            memcpy(toks + o, draft_ids + (size_t)i * max_draft, sizeof(int32_t) * draft_len[i]);
        o += draft_len[i];
    }
    /* CHECKPOINT O(1) di tutte le cache PRIMA del forward (rollback per i mismatch). */
    if (ok) for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        for (int l = 0; l < n_ck; l++) {
            ck[(size_t)i * n_ck + l] = mlx_array_new();
            cv[(size_t)i * n_ck + l] = mlx_array_new();
            (void)llm_mlx_cache_checkpoint(&ck[(size_t)i * n_ck + l], s->k_cache[l], m->stream);
            (void)llm_mlx_cache_checkpoint(&cv[(size_t)i * n_ck + l], s->v_cache[l], m->stream);
        }
    }
    mlx_array out_h = mlx_array_new();
    struct timespec _q0,_q1,_q2; clock_gettime(CLOCK_MONOTONIC,&_q0);
    if (ok) ok = forward_tokens_batch_multi(m, (LlmMlxSession**)handles, ks, n, toks, offs,
                                            t.embed_w, t.embed_s, t.embed_b, &out_h);
    clock_gettime(CLOCK_MONOTONIC,&_q1);
    /* norm + lm_head sull'intero [1,total,hidden] -> argmax [total] (un solo grafo). */
    mlx_array normed = mlx_array_new(), logits = mlx_array_new(), capped = mlx_array_new(), am = mlx_array_new();
    if (ok) ok = mlx_fast_rms_norm(&normed, out_h, t.final_norm_w, m->cfg.rms_norm_eps, m->stream) == 0;
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                      m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    if (ok) {
        if (m->cfg.final_logit_softcap > 0.0f)
            ok = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
        else
            ok = mlx_array_set(&capped, logits) == 0;
    }
    if (ok) ok = mlx_argmax_axis(&am, capped, -1, /*keepdims=*/false, m->stream) == 0;
    if (ok) ok = mlx_array_eval(am) == 0;
    const uint32_t* amd = ok ? mlx_array_data_uint32(am) : NULL;
    if (ok && (amd == NULL || mlx_array_size(am) != (size_t)total)) ok = false;

    if (ok) for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        int K = ks[i];
        if (K <= 0) { out_accepted[i] = 0; out_next[i] = s->pending_next_id; continue; }
        const uint32_t* d = amd + offs[i];
        const int32_t* dr = draft_ids + (size_t)i * max_draft;
        int accepted = 0;
        if (s->pending_next_id == dr[0]) {
            accepted = 1;
            for (int k = 0; k < K - 1; k++) { if ((int32_t)d[k] == dr[k + 1]) accepted++; else break; }
        }
        out_accepted[i] = accepted;
        out_next[i] = (accepted >= 1)
            ? (int32_t)(accepted == K ? d[K - 1] : d[accepted - 1])
            : s->pending_next_id;
    }

    /* Rollback per sessione: ripristina il checkpoint e ri-avanza il SOLO prefisso confermato. */
    if (ok) for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        int K = ks[i];
        int accepted = out_accepted[i];
        bool pending_h_from_reappend = false;
        if (accepted < K && K > 0) {
            for (int l = 0; l < n_ck; l++) {
                (void)llm_mlx_cache_rollback(&s->k_cache[l], ck[(size_t)i * n_ck + l], m->stream);
                (void)llm_mlx_cache_rollback(&s->v_cache[l], cv[(size_t)i * n_ck + l], m->stream);
            }
            if (accepted > 0) {
                mlx_array h_re = mlx_array_new();
                if (forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                         draft_ids + (size_t)i * max_draft, accepted,
                                         nprev[i], s->k_cache, s->v_cache, s->has_cache, &h_re)) {
                    int st2[3] = {0, accepted - 1, 0};
                    int sp2[3] = {1, accepted, mlx_array_dim(h_re, 2)};
                    int ss2[3] = {1, 1, 1};
                    mlx_array ph2 = mlx_array_new();
                    if (mlx_slice(&ph2, h_re, st2, 3, sp2, 3, ss2, 3, m->stream) == 0) {
                        mlx_array_free(s->pending_h);
                        s->pending_h = ph2;
                        pending_h_from_reappend = true;
                    } else mlx_array_free(ph2);
                } else ok = false;
                mlx_array_free(h_re);
            }
        }
        if (ok && accepted > 0 && !pending_h_from_reappend) {
            int start[3] = {0, offs[i] + accepted - 1, 0};
            int stop[3] = {1, offs[i] + accepted, mlx_array_dim(out_h, 2)};
            int stride[3] = {1, 1, 1};
            mlx_array ph = mlx_array_new();
            if (mlx_slice(&ph, out_h, start, 3, stop, 3, stride, 3, m->stream) == 0) {
                mlx_array_free(s->pending_h);
                s->pending_h = ph;
            } else mlx_array_free(ph);
        }
        s->cache_offset = nprev[i] + accepted;
        s->pending_next_id = out_next[i];
    }
    clock_gettime(CLOCK_MONOTONIC,&_q2);
    if (getenv("LLM_SPEC_TIME"))
        fprintf(stderr, "[multi-time] N=%d total=%d fwd=%.1fms rollback=%.1fms\n", n, total,
            (_q1.tv_sec-_q0.tv_sec)*1e3+(_q1.tv_nsec-_q0.tv_nsec)/1e6,
            (_q2.tv_sec-_q1.tv_sec)*1e3+(_q2.tv_nsec-_q1.tv_nsec)/1e6);

    for (int i = 0; i < n; i++) for (int l = 0; l < n_ck; l++) {
        llm_mlx_cache_commit(&ck[(size_t)i * n_ck + l], m->stream);
        llm_mlx_cache_commit(&cv[(size_t)i * n_ck + l], m->stream);
    }
    free(ck); free(cv); free(toks); free(offs); free(ks); free(nprev);
    mlx_array_free(out_h); mlx_array_free(normed); mlx_array_free(logits);
    mlx_array_free(capped); mlx_array_free(am);
    free_output_tensors(&t);
    return ok ? 0 : -1;
}

/* T2/DFlash2 — tronca la KV-cache della sessione ai primi n_tokens token (rollback).
 * Usata per riallineare una sessione DRAFT al prefisso confermato dal target quando
 * il blocco proposto e' stato solo parzialmente accettato: la sessione draft ha
 * appeso token NON confermati (i suoi guess) che vanno scartati prima di sincronizzarsi
 * di nuovo coi token veri. Dopo truncate la sessione NON ha pending (has_pending=false):
 * serve un session_append prima del prossimo generate (l'append ricalcola pending via
 * final_step). Rifiuta se n_tokens > cache_offset (non puo' crescere) o se la KV quant
 * cold (kvq) e' attiva (lo stato quantizzato non e' tagliabile in modo sicuro).
 * Returns 0 su successo, -1 su errore. */
int llm_ai_mlx_session_truncate(llm_ai_mlx_session_handle handle, int n_tokens) {
    if (handle == NULL || n_tokens < 0) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (n_tokens > s->cache_offset) return -1;
    if (n_tokens == s->cache_offset) return 0;
    /* DeepSeek V4.1 whole-model: reset + rebuild dalla history (fino a n_tokens). */
    if (s->model && s->model->dsv41) {
    LlmMlxModel* m = s->model;
        if (n_tokens > s->history_len) return -1;
        if (n_tokens == 0) {
            if (s->dsv41) { llm_dsv41_cache_free((LlmDsv41Cache*)s->dsv41); free(s->dsv41); s->dsv41 = NULL; }
            s->cache_offset = 0; s->has_pending = false; s->pending_next_id = 0; s->history_len = 0;
            return 0;
        }
        llm_mlx_sampling_t sm0 = llm_mlx_session_sampling(s);
        LlmDsv41Sampling ds = { sm0.active, sm0.temperature, sm0.top_p, sm0.top_k, sm0.seed,
                                sm0.repetition_penalty, sm0.frequency_penalty, sm0.presence_penalty,
                                sm0.hist, sm0.hist_n };
        int32_t pending = 0;
        if (!llm_dsv41_engine_prefill((LlmDsv41Engine*)m->dsv41, &s->dsv41, s->history, n_tokens,
                                      &s->cache_offset, &pending, &s->has_pending, &ds, m->stream))
            return -1;
        s->pending_next_id = pending;
        s->history_len = n_tokens;
        return 0;
    }
    if (s->kvq_len != NULL) {
        for (int l = 0; l < s->n_layers; l++)
            if (s->kvq_len[l] > 0) return -1; /* kvq cold non tagliabile: errore esplicito */
    }
    bool ok = true;
    for (int layer = 0; ok && layer < s->n_layers; layer++) {
        const int* shape = mlx_array_shape(s->k_cache[layer]);
        if (shape == NULL || mlx_array_ndim(s->k_cache[layer]) != 4) { ok = false; break; }
        int start[4] = {0, 0, 0, 0};
        int stop[4] = {shape[0], shape[1], n_tokens, shape[3]};
        int stride[4] = {1, 1, 1, 1};
        mlx_array kt = mlx_array_new(), vt = mlx_array_new();
        bool kr = mlx_slice(&kt, s->k_cache[layer], start, 4, stop, 4, stride, 4, s->model->stream) == 0;
        bool vr = mlx_slice(&vt, s->v_cache[layer], start, 4, stop, 4, stride, 4, s->model->stream) == 0;
        if (kr && vr) {
            mlx_array_free(s->k_cache[layer]);
            s->k_cache[layer] = kt;
            mlx_array_free(s->v_cache[layer]);
            s->v_cache[layer] = vt;
        } else {
            mlx_array_free(kt); mlx_array_free(vt);
            ok = false;
        }
    }
    if (!ok) return -1;
    s->cache_offset = n_tokens;
    if (s->has_pending) {
        mlx_array_free(s->pending_h);
        s->pending_h = mlx_array_new();
    }
    s->has_pending = false;
    s->pending_next_id = 0;
    return 0;
}

/* DSpark — confidence del greedy corrente della sessione: P(argmax | context) sulla posizione
 * "pending" (pending_h, gia' tenuto vivo). Riusa il path di final_step (rms_norm + lm_head,
 * softcap se config) e calcola softmax->max sul device. E' la confidenza che il DRAFT model ha
 * nel token che ha appena proposto: usata dalla verifica confidence-scheduled di DSpark per
 * troncare il blocco al prefisso piu' lungo con confidenza >= soglia (niente verifica sprecata
 * su token che il draft stesso ritiene incerti). Costo: una norm + una lm_head + riduzioni
 * sul vocabolario — accettabile per step di draft; NON chiamata se il gate non e' attivo.
 * Returns 0 su successo (scrive *out_prob in [0,1]), -1 su errore/sessione senza pending. */
int llm_ai_mlx_session_greedy_prob(llm_ai_mlx_session_handle handle, float* out_prob) {
    if (handle == NULL || out_prob == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    LlmMlxModel* m = s->model;
    if (m->dsv41) {
        LlmDsv41Cache* c = (LlmDsv41Cache*)s->dsv41;
        if (!c || !c->last_logits || c->last_logits_n <= 0) return -1;
        int V = c->last_logits_n;
        float mxv = c->last_logits[0];
        for (int i = 1; i < V; i++) if (c->last_logits[i] > mxv) mxv = c->last_logits[i];
        double sum = 0.0;
        for (int i = 0; i < V; i++) sum += exp((double)(c->last_logits[i] - mxv));
        *out_prob = (float)(1.0 / sum);   /* P(argmax) = exp(max-mx)/sum = 1/sum */
        return 0;
    }
    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;
    mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_final_norm(m, s->pending_h, t.final_norm_w, &normed);
    mlx_array logits = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                      m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    mlx_array capped = mlx_array_new();
    if (ok && m->cfg.final_logit_softcap > 0.0f) {
        ok = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
    } else if (ok) {
        ok = mlx_array_set(&capped, logits) == 0;
    }
    mlx_array capped32 = mlx_array_new();
    if (ok) ok = llm_mlx_logits_f32(&capped32, capped, m->stream);
    mlx_array sm = mlx_array_new();
    if (ok) ok = mlx_softmax_axis(&sm, capped32, -1, /*precise=*/true, m->stream) == 0;
    mlx_array pm = mlx_array_new();
    if (ok) ok = mlx_max_axis(&pm, sm, -1, /*keepdims=*/false, m->stream) == 0;
    if (ok) ok = mlx_array_eval(pm) == 0;
    if (ok) {
        const float* d = mlx_array_data_float32(pm);
        ok = (d != NULL && mlx_array_size(pm) == 1);
        if (ok) *out_prob = d[0];
    }
    mlx_array_free(normed);
    mlx_array_free(logits);
    mlx_array_free(capped);
    mlx_array_free(capped32);
    mlx_array_free(sm);
    mlx_array_free(pm);
    free_output_tensors(&t);
    return ok ? 0 : -1;
}

/* FASE 3: dispatch nativo a puntatore di funzione -- vedi commento su
 * LlmMlxSession::pending_h e final_step_constrained. Un solo passo: ri-decide
 * la posizione corrente (gia' "pending" da un append o generate precedente)
 * con un argmax ristretto ad allowed_ids invece di accettare la scelta
 * unconstrained gia' calcolata, poi propaga il token SCELTO (non quello
 * scartato) in avanti per preparare la posizione successiva -- stessa
 * identica forma del corpo del loop di llm_ai_mlx_session_generate, solo con
 * la selezione ristretta al posto dell'argmax completo. */
int llm_ai_mlx_session_generate_constrained(llm_ai_mlx_session_handle handle,
                                             const int32_t* allowed_ids, int n_allowed,
                                             int32_t* out_token_id) {
    if (handle == NULL || allowed_ids == NULL || n_allowed <= 0 || out_token_id == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    LlmMlxModel* m = s->model;

    /* DeepSeek V4.1 whole-model: argmax ristretto ad allowed_ids sulla posizione pending, poi
     * decode del token scelto e nuovo pending. */
    if (m->dsv41) {
        llm_mlx_sampling_t sm0 = llm_mlx_session_sampling(s);
        LlmDsv41Sampling ds = { sm0.active, sm0.temperature, sm0.top_p, sm0.top_k, sm0.seed,
                                sm0.repetition_penalty, sm0.frequency_penalty, sm0.presence_penalty,
                                sm0.hist, sm0.hist_n };
        int32_t chosen = 0, next = 0;
        if (!llm_dsv41_engine_constrained(m->dsv41, &s->dsv41, &s->cache_offset,
                                          allowed_ids, n_allowed, &chosen, &next, &ds, m->stream))
            return -1;
        s->pending_next_id = next;
        s->has_pending = true;
        *out_token_id = chosen;
        return 0;
    }

    OutputTensors t;
    bool ok = load_output_tensors(m, &t);

    int32_t chosen_id = 0;
    if (ok) {
        ok = final_step_constrained(m, s->pending_h, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                     allowed_ids, n_allowed, &chosen_id);
    }

    if (ok) {
        mlx_array h = mlx_array_new();
        ok = forward_one_token(m, s, t.embed_w, t.embed_s, t.embed_b, chosen_id, s->cache_offset,
                                s->k_cache, s->v_cache, s->has_cache, &h);
        int32_t next_id = 0;
        if (ok) ok = final_step(m, h, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                llm_mlx_session_sampling(s), &next_id);
        if (ok) {
            mlx_array_free(s->pending_h);
            s->pending_h = mlx_array_new();
            mlx_array_set(&s->pending_h, h);
            s->pending_next_id = next_id;
        }
        mlx_array_free(h);
        s->cache_offset++;
    }

    free_output_tensors(&t);
    if (ok) *out_token_id = chosen_id;
    return ok ? 0 : -1;
}



/* ─────────────────────────────────────────────────────────────────────────────
 * MODEL IR EXPORT (§8.3 + roadmap fase 1/7 del piano): scrive la descrizione
 * strutturale del modello (Model IR) in JSON su un file (model.gmir), leggibile da
 * un Model Builder per ricostruire/validare/exportare l'esecuzione. Completa la
 * fase 1 (Model Format Abstraction): il modello non e' un file opaco ma un IR
 * dichiarativo (tokenizer/architettura/layer quant/moa). Ritorna 0 ok, -1 errore.
 */






/* DS4-Q1.b item 4 (roadmap): smoketest del quantizzatore pesato dall'imatrix. Scenario
 * realistico (non un caso di comodo): 8 colonne POCO importanti ma con outlier ampi
 * (tipico di colonne rare che il naive min/max spende male su tutto il gruppo) + 56 colonne
 * MOLTO importanti con segnale piccolo -- il naive min/max quantizzatore sceglie una scala
 * dominata dagli outlier, rendendo il segnale importante indistinguibile; il quantizzatore
 * pesato deve scartare (con errore alto, ma peso basso) gli outlier per una scala molto piu'
 * fine sul segnale importante. Poi verifica via mlx_dequantize (la VERA API di dequant del
 * motore, non solo la mia formula host) che il packing prodotto sia effettivamente caricabile.
 * Ritorna 0 ok / -1. */

/* N-GRAM self-speculative DRAFT BATCH (paragrafo 8.4/8.10 del piano): propone K draft,
 * li verifica con UN forward batch (greedy intermedi dall''hidden batch) e accetta i primi
 * M che coincidono col greedy. M==K -> K token con 1 forward (guadagno); M<K -> rollback
 * (Cache ABI) + riforward di M. Correttezza: il primo draft deve == greedy noto. */
/* Longest-suffix prompt-lookup: cerca il piu' LUNGO suffisso della history (L in 2..MAXL) che
 * ricorre prima (finestra W), e ritorna il seguito (fino a max_draft). Draft piu' affidabili su
 * span brevi ripetuti con prefisso variabile (che l'ordine fisso manca). Bounded (W*MAXL). */
static int spec_lookup_suffix(LlmMlxSession* s, int32_t* draft, int max_draft) {
    const int n = s->history_len; const int32_t* H = s->history;
    if (H == NULL || n < 3 || max_draft <= 0) return 0;
    const int MAXL = 24, W = 2048;
    int lo = n - W; if (lo < 0) lo = 0;
    int Lmax = n - 1; if (Lmax > MAXL) Lmax = MAXL;
    for (int L = Lmax; L >= 2; L--) {
        int p = -1;
        for (int q = n - L - 1; q >= lo; q--) {
            if (memcmp(&H[q], &H[n - L], (size_t)L * sizeof(int32_t)) == 0) { p = q; break; }
        }
        if (p < 0) continue;
        int nd = 0, e = p + L;
        while (nd < max_draft && e < n) draft[nd++] = H[e++];
        if (nd > 0) return nd;
    }
    return 0;
}

/* Draft self-speculativo: PRIMA il n-gram a ordine fisso, POI il longest-suffix come fallback. */
static int spec_build_draft(LlmMlxSession* s, int32_t* draft, int max_draft) {
    if (s == NULL || draft == NULL || max_draft <= 0 || s->history_len < 2) return 0;
    int nd = 0;
    if (s->ngram_built && s->ngram.buckets != NULL) {
        int32_t* ctx = (int32_t*)malloc((s->history_len + (size_t)max_draft + 1) * sizeof(int32_t));
        if (ctx != NULL) {
            const int ctx_len = s->history_len;
            for (int i = 0; i < ctx_len; i++) ctx[i] = s->history[i];
            for (int i = 0; i < max_draft; i++) {
                int32_t d = llm_mlx_ngram_predict(&s->ngram, ctx, ctx_len + i);
                if (d < 0) break;
                draft[i] = d; ctx[ctx_len + i] = d; nd++;
            }
            free(ctx);
        }
    }
    if (nd < 2) nd = spec_lookup_suffix(s, draft, max_draft);
    return nd;
}

int llm_ai_mlx_session_generate_ngram(llm_ai_mlx_session_handle handle, int max_new_tokens,
                                      int32_t eos_token_id, int32_t* out_ids, int draft_len) {
    if (handle == NULL || max_new_tokens <= 0 || out_ids == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    if (draft_len < 2) draft_len = 2;
    if (draft_len > 16) draft_len = 16;
    if (!s->ngram_built) draft_len = 2;
    LlmMlxModel* m = s->model;

    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;

    bool ok = true;
    int generated = 0;
    int32_t current = s->pending_next_id;
    while (ok && generated < max_new_tokens) {
        if (eos_token_id >= 0 && current == eos_token_id) break;
        const int remaining = max_new_tokens - generated;
        int dl = draft_len; if (dl > remaining) dl = remaining;

        int32_t draft[16];
        int nd = spec_build_draft(s, draft, dl);

        /* Nessuno spec possibile (draft < 2 o primo != pending): forward AR di `current`.
           SEMPRE il forward: senza di esso il pending non avanza -> token costante. */
        if (dl < 2 || nd < 2 || draft[0] != current) {
            out_ids[generated++] = current;
            session_history_append(s, &current, 1);
            mlx_array h = mlx_array_new();
            bool ok2 = forward_one_token(m, s, t.embed_w, t.embed_s, t.embed_b, current, s->cache_offset,
                                         s->k_cache, s->v_cache, s->has_cache, &h);
            int32_t next = 0;
            if (ok2) ok2 = final_step(m, h, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                      llm_mlx_session_sampling(s), &next);
            if (ok2) { mlx_array_free(s->pending_h); s->pending_h = mlx_array_new(); mlx_array_set(&s->pending_h, h); }
            mlx_array_free(h);
            s->cache_offset++;
            if (!ok2) break;
            current = next;
            continue;
        }

        int co_orig = s->cache_offset;
        mlx_array* ckpt_k = (mlx_array*)malloc((size_t)s->n_layers * sizeof(mlx_array));
        mlx_array* ckpt_v = (mlx_array*)malloc((size_t)s->n_layers * sizeof(mlx_array));
        if (!ckpt_k || !ckpt_v) { free(ckpt_k); free(ckpt_v); free_output_tensors(&t); return -1; }
        for (int i = 0; i < s->n_layers; i++) {
            ckpt_k[i] = mlx_array_new(); ckpt_v[i] = mlx_array_new();
            llm_mlx_cache_checkpoint(&ckpt_k[i], s->k_cache[i], m->stream);
            llm_mlx_cache_checkpoint(&ckpt_v[i], s->v_cache[i], m->stream);
        }

        mlx_array hb = mlx_array_new();
        bool okb = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, draft, nd, co_orig,
                                        s->k_cache, s->v_cache, s->has_cache, &hb);
        int32_t greedy[16];
        llm_mlx_sampling_t smpg = llm_mlx_session_sampling(s);
        const bool greedy_mode = (!smpg.active || smpg.temperature <= 0.0f);
        if (okb && greedy_mode) {
            /* Batch: una final_norm + UNA matmul lm_head per tutti i draft. */
            okb = final_step_greedy_batch(m, hb, t.final_norm_w, t.out_w, t.out_s, t.out_b, greedy, nd);
        } else if (okb) {
            for (int i = 0; i < nd && okb; i++) {
                mlx_array h_i = mlx_array_new();
                const int start[3] = {0, i, 0};
                const int stop[3] = {1, i + 1, mlx_array_dim(hb, 2)}; /* vedi nota hc_width sopra */
                const int strides[3] = {1, 1, 1};
                okb = mlx_slice(&h_i, hb, start, 3, stop, 3, strides, 3, m->stream) == 0;
                if (okb) okb = final_step(m, h_i, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                          smpg, &greedy[i]);
                mlx_array_free(h_i);
            }
        }

        if (!okb) {   /* forward batch fallito: NIENTE emit/avanzamento (cache resterebbe incoerente) */
            for (int i = 0; i < s->n_layers; i++) {
                llm_mlx_cache_rollback(&s->k_cache[i], ckpt_k[i], m->stream);
                llm_mlx_cache_rollback(&s->v_cache[i], ckpt_v[i], m->stream);
            }
            s->cache_offset = co_orig;
            mlx_array_free(hb);
            for (int i = 0; i < s->n_layers; i++) { mlx_array_free(ckpt_k[i]); mlx_array_free(ckpt_v[i]); }
            free(ckpt_k); free(ckpt_v);
            break;
        }
        int M = 1;
        if (okb) { for (int i = 1; i < nd; i++) { if (greedy[i - 1] == draft[i]) M++; else break; } }
        int32_t next = okb ? greedy[M - 1] : 0;

        if (M == nd && okb) {
            for (int i = 0; i < s->n_layers; i++) llm_mlx_cache_commit(&ckpt_k[i], m->stream);
            for (int i = 0; i < s->n_layers; i++) llm_mlx_cache_commit(&ckpt_v[i], m->stream);
            s->cache_offset = co_orig + nd;
            mlx_array h_last = mlx_array_new();
            const int start[3] = {0, nd - 1, 0};
            const int stop[3] = {1, nd, mlx_array_dim(hb, 2)}; /* vedi nota hc_width sopra */
            const int strides[3] = {1, 1, 1};
            if (mlx_slice(&h_last, hb, start, 3, stop, 3, strides, 3, m->stream) == 0) {
                mlx_array_free(s->pending_h); s->pending_h = mlx_array_new(); mlx_array_set(&s->pending_h, h_last);
            }
            mlx_array_free(h_last);
            for (int i = 0; i < nd && generated < max_new_tokens; i++) {
                out_ids[generated++] = draft[i];
                session_history_append(s, &draft[i], 1);
            }
        } else {
            for (int i = 0; i < s->n_layers; i++) llm_mlx_cache_rollback(&s->k_cache[i], ckpt_k[i], m->stream);
            for (int i = 0; i < s->n_layers; i++) llm_mlx_cache_rollback(&s->v_cache[i], ckpt_v[i], m->stream);
            s->cache_offset = co_orig;
            if (okb && M > 0) {
                mlx_array hb2 = mlx_array_new();
                okb = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, draft, M, co_orig,
                                           s->k_cache, s->v_cache, s->has_cache, &hb2);
                if (okb) {
                    mlx_array h_last = mlx_array_new();
                    const int start[3] = {0, M - 1, 0};
                    const int stop[3] = {1, M, mlx_array_dim(hb2, 2)}; /* vedi nota hc_width sopra */
                    const int strides[3] = {1, 1, 1};
                    okb = mlx_slice(&h_last, hb2, start, 3, stop, 3, strides, 3, m->stream) == 0;
                    if (okb) { mlx_array_free(s->pending_h); s->pending_h = mlx_array_new(); mlx_array_set(&s->pending_h, h_last); }
                    mlx_array_free(h_last);
                }
                mlx_array_free(hb2);
            }
            s->cache_offset = co_orig + M;
            for (int i = 0; i < M && generated < max_new_tokens; i++) {
                out_ids[generated++] = draft[i];
                session_history_append(s, &draft[i], 1);
            }
        }
        mlx_array_free(hb);
        for (int i = 0; i < s->n_layers; i++) { mlx_array_free(ckpt_k[i]); mlx_array_free(ckpt_v[i]); }
        free(ckpt_k); free(ckpt_v);
        if (!okb) break;
        current = next;
    }
    if (current >= 0) s->pending_next_id = current;

    free_output_tensors(&t);
    return generated;
}

/* ─── Speculative streaming (spec_ngram): supporto al buffering del chunk ───────────────────
 * Vedi i campi `spec_*` in llm_mlx_internal.h. Il consumer streaming chiama generate(1) per
 * token; qui il motore drafta in anticipo un chunk (generate_ngram) e lo bufferizza. */

/* Salva i checkpoint K/V (alias O(1), nessuna copia) all'inizio del chunk corrente. Rilascia
 * eventuali alias precedenti sovrascrivendoli con mlx_array_set. */
void llm_mlx_session_spec_begin(LlmMlxSession* s) {
    if (s == NULL) return;
    if (s->spec_ckpt_k == NULL) {
        s->spec_ckpt_k = (mlx_array*)malloc((size_t)s->n_layers * sizeof(mlx_array));
        s->spec_ckpt_v = (mlx_array*)malloc((size_t)s->n_layers * sizeof(mlx_array));
        if (s->spec_ckpt_k == NULL || s->spec_ckpt_v == NULL) return;
        for (int i = 0; i < s->n_layers; i++) { s->spec_ckpt_k[i] = mlx_array_new(); s->spec_ckpt_v[i] = mlx_array_new(); }
    }
    for (int i = 0; i < s->n_layers; i++) {
        llm_mlx_cache_checkpoint(&s->spec_ckpt_k[i], s->k_cache[i], s->model->stream);
        llm_mlx_cache_checkpoint(&s->spec_ckpt_v[i], s->v_cache[i], s->model->stream);
    }
    s->spec_ctx_len = s->history_len;
    s->spec_ctx_off = s->cache_offset;
    s->spec_ckpt_valid = true;
}

void llm_mlx_session_spec_free(LlmMlxSession* s) {
    if (s == NULL) return;
    if (s->spec_ckpt_k) {
        for (int i = 0; i < s->n_layers; i++) mlx_array_free(s->spec_ckpt_k[i]);
        free(s->spec_ckpt_k); s->spec_ckpt_k = NULL;
    }
    if (s->spec_ckpt_v) {
        for (int i = 0; i < s->n_layers; i++) mlx_array_free(s->spec_ckpt_v[i]);
        free(s->spec_ckpt_v); s->spec_ckpt_v = NULL;
    }
    s->spec_ckpt_valid = false; s->spec_buf_len = 0; s->spec_buf_pos = 0;
}

/* Scarta i token del chunk non consumati (append esterno, nuovo turno). Riporta la cache al
 * chunk-start e ri-avanza i soli token consumati (riproduce esattamente lo stato greedy), poi
 * tronca la history. No-op se il buffer e' esaurito. */
void llm_mlx_session_spec_discard(LlmMlxSession* s) {
    if (s == NULL) return;
    if (s->spec_ckpt_valid && s->spec_ckpt_k != NULL) {
        if (!s->parked && s->spec_buf_pos < s->spec_buf_len) {
            const int consumed = s->spec_buf_pos;
            LlmMlxModel* m = s->model;
            for (int i = 0; i < s->n_layers; i++) {
                llm_mlx_cache_rollback(&s->k_cache[i], s->spec_ckpt_k[i], m->stream);
                llm_mlx_cache_rollback(&s->v_cache[i], s->spec_ckpt_v[i], m->stream);
            }
            s->cache_offset = s->spec_ctx_off;
            OutputTensors t;
            if (load_output_tensors(m, &t)) {
                for (int i = 0; i < consumed; i++) {
                    mlx_array h = mlx_array_new();
                    bool okr = forward_one_token(m, s, t.embed_w, t.embed_s, t.embed_b, s->spec_buf[i],
                                                 s->cache_offset, s->k_cache, s->v_cache, s->has_cache, &h);
                    mlx_array_free(h);
                    if (!okr) break;
                    s->cache_offset++;
                }
                free_output_tensors(&t);
            }
            if (s->history_len > s->spec_ctx_len + consumed) s->history_len = s->spec_ctx_len + consumed;
        }
        /* rilascia gli alias dei checkpoint (senza free dell'array handle) */
        for (int i = 0; i < s->n_layers; i++) {
            llm_mlx_cache_commit(&s->spec_ckpt_k[i], s->model->stream);
            llm_mlx_cache_commit(&s->spec_ckpt_v[i], s->model->stream);
        }
        s->spec_ckpt_valid = false;
    }
    s->spec_buf_len = 0; s->spec_buf_pos = 0;
}

/* True se la tabella n-gram PREDICE almeno 2 token dalla history corrente (draft[0] ==
 * pending_next_id). Serve al dispatch per NON entrare nel fallback AR di generate_ngram quando la
 * tabella non predice: quel fallback e' un forward SINCRONO, piu' lento del path principale (async
 * con overlap I/O) sui modelli I/O-bound -- misurato 3.7x su OLMoE (562 vs 150 ms/tok). */
int llm_mlx_session_spec_can(LlmMlxSession* s) {
    if (s == NULL || !s->ngram_built || s->pending_next_id < 0 || s->history_len <= 0) return 0;
    int32_t d[16];
    int nd = spec_build_draft(s, d, 8);
    return (nd >= 2 && d[0] == s->pending_next_id) ? 1 : 0;
}
