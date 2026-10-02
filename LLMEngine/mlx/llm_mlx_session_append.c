/* ============================================================================
 * Dominio session append: llm_ai_mlx_session_append_impl, llm_ai_mlx_session_append (PFlash), llm_ai_mlx_session_capture_hidden.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_pflash.h"
#include "llm_dsv4_engine.h"
#include "llm_dsv41_engine.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_dispatch_profile.h"
#include "llm_mlx_array_count.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int llm_ai_mlx_session_append_impl(llm_ai_mlx_session_handle handle, const int32_t* token_ids, int n_ids) {
    if (handle == NULL || token_ids == NULL || n_ids <= 0) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    LlmMlxModel* m = s->model;
    /* PRECARICA il drafter DFlash2 al PRIMO append (= prima del prefill): il picco del prefill
     * (array vivi fino al primo forward del decode) include allora il drafter, e il primo forward
     * dello streaming drena il picco -> evita l'OOM OS a 1M/2M. */
    if (m->dsv4 == NULL && m->dsv41 == NULL && s->cache_offset == 0 && s->history_len == 0)
        llm_mlx_df2_preload(m);
    /* Append esterno (nuovo turno/prompt): i token draftati in anticipo e non ancora consumati
     * dallo spec streaming vengono scartati (cache riportata al chunk-start + history troncata). */
    llm_mlx_session_spec_discard(s);
    /* DeepSeek V4.1 whole-model: prefill/decode del chunk + set del prossimo token (greedy o
     * sampling secondo la sessione). */
    if (m->dsv4) {
        llm_dsv4_engine_append((LlmDsv4Engine*)m->dsv4, token_ids, n_ids);
        session_history_append(s, token_ids, n_ids);
        s->has_pending = true;
        s->pending_next_id = ((LlmDsv4Engine*)m->dsv4)->pending;
        return 0;
    }
    if (m->dsv41) {
        llm_mlx_sampling_t sm0 = llm_mlx_session_sampling(s);
        LlmDsv41Sampling ds = { sm0.active, sm0.temperature, sm0.top_p, sm0.top_k, sm0.seed,
                                sm0.repetition_penalty, sm0.frequency_penalty, sm0.presence_penalty,
                                sm0.hist, sm0.hist_n };
        int rc = llm_dsv41_engine_append(m->dsv41, &s->dsv41, token_ids, n_ids,
                                         &s->cache_offset, &s->pending_next_id, &s->has_pending,
                                         &ds, m->stream);
        if (rc == 0) session_history_append(s, token_ids, n_ids);
        return rc;
    }
    if (s->parked) (void)llm_session_unpark(m, s); /* §G-ter: restore on-demand dell'evict */
    { int _g = llm_mlx_active_guard(m); if (_g != 0) return _g; }

    OutputTensors t;
    bool ok = load_output_tensors(m, &t);

    /* CHUNKED prefill (fisso 2026-08-31): invece di un unico grafo per TUTTI
     * gli n_ids token (che per prompt enormi esplode la memoria del grafo
     * Metal o fallisce la build del batch), i token vengono processati in
     * sequenza di chunk da `prefill_chunk_size` (default 2048). Ogni chunk
     * passa attraverso ogni layer in UNA chiamata (mask_mode="causal" +
     * KV-cache che cresce: il chunk i dipende solo dai token 0..i-1 gia'
     * decodificati), quindi il risultato e' matematicamente identico al batch
     * singolo, solo con grafi piu' piccoli e piu' sync CPU<->GPU ma memoria
     * costante per chunk. L'ultimo chunk produce h_batch (per l'estrazione
     * di h_last/pending_h in coda). */
    int chunk_size = s->prefill_chunk_size > 0 ? s->prefill_chunk_size : 2048;
    /* gemma4 (PREFILL, misurato 2026-09-16): la scelta ottimale del chunk dipende dal CONTESTO
     * TOTALE, non dal modello in astratto.
     *  - contesto <= 16K: chunk GRANDE (2048) + flash-prefill: il kernel evita la
     *    materializzazione Lq*Lk dei 10 layer full hd=512 (fuori dal limite flash di MLX).
     *    MISURATO a 4K (finestra pulita, load 2,2-3,2, A/B interleaved): nostro 7535/9200 ms
     *    vs mlx_lm 9176/8661 ms = **parita' ±15%**, e **2,6x meglio del default precedente**
     *    (kernel OFF + chunk 128: 19449/19735 ms).
     *  - contesto > 16K: chunk PICCOLO (128) e il kernel e' disabilitato dalla soglia Nk
     *    (e' per-query senza key-tiling: a 32K faceva 408 s contro 139 s del percorso senza
     *    kernel). Con 128 si limita la materializzazione del percorso generale.
     *    Crossover misurato ~16K (li' le config sono entro il 3%).
     * ATTENZIONE ai numeri: un A/B a carico alto NON e' confrontabile (mlx_lm soffre il
     * carico piu' di noi) -> le prime misure "+26% vs mlx_lm" erano un artefatto e sono
     * state ritirate (v. todo, RETTIFICA 2026-09-16).
     * Override espliciti: LLM_PREFILL_CHUNK ha precedenza (chunk_env > 0 al create). */
    /* gemma4: chunk 128 e' il MISURATO migliore col kernel flash-prefill SPENTO (default):
     * vedi llm_mlx_flash_prefill_enabled(). Con il kernel abilitato conviene 2048, ma resta
     * comunque peggio del default -> se lo abiliti, passa LLM_PREFILL_CHUNK=2048 a mano. */
    if (strncmp(m->cfg.model_type, "gemma4", 6) == 0 &&
        llm_ai_mlx_opt_env("LLM_PREFILL_CHUNK") == NULL && llm_mlx_flash_prefill_enabled())
        chunk_size = 2048;
    /* h_batch possiede l'output dell'ultimo chunk (ownership trasferita da
     * forward_tokens_batch via out_h). NOTA sul pattern: mlx_array_set(&dst, src)
     * fa condividere lo stesso ArrayDesc (refcount C++), e mlx_array_free(src)
     * dopo decrementa -- se dst resta in vita e src viene liberato, il refcount
     * NON arriva a zero finche' dst lo tiene, MA ASan ha mostrato il caso limite
     * (h_chunk riusato nel loop dopo free). Per evitare ogni rischio: si trasferisce
     * l'ownership della struct (h_batch = h_chunk) e NON si libera h_chunk. */
    /* Qwen4-Exp N-gram/PLE: side-channel per la finestra di contesto token, impostato UNA
     * VOLTA per l'intera append (copre tutti i chunk: ogni chunk ha bisogno solo di un
     * prefisso di questo buffer, mai di tutto). s->history NON include ancora token_ids
     * qui (l'append avviene DOPO il loop, vedi sotto) -- combined = history ++ token_ids. */
    int32_t* q4e_combined = NULL;
    if (m->cfg.ple_layer_ids_n > 0) {
        q4e_combined = (int32_t*)malloc((size_t)(s->history_len + n_ids) * sizeof(int32_t));
        if (q4e_combined != NULL) {
            if (s->history_len > 0) memcpy(q4e_combined, s->history, (size_t)s->history_len * sizeof(int32_t));
            memcpy(q4e_combined + s->history_len, token_ids, (size_t)n_ids * sizeof(int32_t));
            m->q4e_ctx_ids = q4e_combined; m->q4e_ctx_len = s->history_len + n_ids;
            m->aux_cache = s->aux_cache;
        }
    }

    mlx_array h_batch = mlx_array_new();
    /* G0 sul PREFILL: il conteggio di cio' che il grafo del prefill possiede in byte, per
     * famiglia e per forma. Misurato (2026-09-09): su Mixtral-8x22B l'allocatore Metal passa da
     * 79,39 GB (pesi) a 130,41 GB durante UN prefill di 32 token, e non scende piu'; la crescita
     * e' lineare (~870 MB/layer su 56 layer, LLM_MEMPROF). Qui sotto lo stesso strumento che ha
     * nominato i cast da 48 MB, applicato al grafo pigro del prefill PRIMA che venga evalutato
     * (dopo l'eval non e' camminabile: REGOLA OPERATIVA in llm_mlx_dispatch_profile.h). */
    const bool dp = llm_mlx_dp_enabled();
    const int dp_lay = dp ? m->cfg.n_layers : 0;
    int dp_chunks = 0;
    struct timespec dp_t0, dp_t1;
    if (dp) { llm_mlx_dp_reset(); clock_gettime(CLOCK_MONOTONIC, &dp_t0); }
    int offset = 0;
    int last_chunk_n = 0;
    /* MLA (Kimi): la cache ausiliaria per-layer (k_pe) deve essere visibile al modello durante
     * il prefill, non solo nel ramo PLE (Qwen4-Exp). Senza, il prefill NON popola
     * s->aux_cache e il decode MLA perde il k_pe del contesto -> degenerazione. */
    m->aux_cache = s->aux_cache;
    while (ok && offset < n_ids) {
        int chunk_n = n_ids - offset;
        if (chunk_n > chunk_size) chunk_n = chunk_size;
        mlx_array h_chunk = mlx_array_new();
        /* §G-ter P3 (2026-09-19): checkpoint del rewind ai confini di blocco DENTRO il
         * prefill (prima una sola volta a fine append -> il rewind non trovava prefissi
         * intermedi). La chiave e' il prefisso history ++ token_ids[:pos-history_len]. */
        int ckpt_after_chunk = 0;
        {
            int abs_end = s->cache_offset + offset + chunk_n;
            if (m->kvckpt_block > 0 && llm_kvdisk_enabled(m) &&
                (abs_end % m->kvckpt_block) == 0)
                ckpt_after_chunk = abs_end;
        }
        ok = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                   token_ids + offset, chunk_n,
                                   s->cache_offset + offset,
                                   s->k_cache, s->v_cache, s->has_cache, &h_chunk);
        if (dp && ok) {
            const size_t nrt = 1 + 2 * (size_t)dp_lay;
            mlx_array* rts = (mlx_array*)malloc(nrt * sizeof(mlx_array));
            if (rts != NULL) {
                rts[0] = h_chunk;
                for (int l = 0; l < dp_lay; l++) {
                    rts[1 + l] = s->k_cache[l];
                    rts[1 + (size_t)dp_lay + l] = s->v_cache[l];
                }
                llm_mlx_dp_count(rts, (int)nrt);
                free(rts);
            }
            dp_chunks++;
        }
        if (ok && ckpt_after_chunk > 0) {
            /* §G-ter P3: checkpoint intermedio (chiave = history ++ token_ids[:resto]). */
            int h_hist = s->history_len; if (h_hist > ckpt_after_chunk) h_hist = ckpt_after_chunk;
            int n_from_tok = ckpt_after_chunk - h_hist;
            if (n_from_tok >= 0 && n_from_tok <= offset + chunk_n) {
                int32_t* keybuf = (int32_t*)malloc((size_t)ckpt_after_chunk * sizeof(int32_t));
                if (keybuf != NULL) {
                    if (h_hist > 0 && s->history) memcpy(keybuf, s->history, (size_t)h_hist * sizeof(int32_t));
                    if (n_from_tok > 0) memcpy(keybuf + h_hist, token_ids, (size_t)n_from_tok * sizeof(int32_t));
                    (void)llm_session_ckpt_write_k(m, s, ckpt_after_chunk, keybuf, ckpt_after_chunk);
                    free(keybuf);
                }
            }
        }
        if (ok) {
            /* Trasferimento di ownership: h_batch adotta l'output di questo chunk.
             * Il vecchio h_batch (chunk precedente) va liberato prima. h_chunk NON
             * va liberato (la struct ctx e' ora posseduta da h_batch). */
            mlx_array_free(h_batch);
            h_batch = h_chunk;
            last_chunk_n = chunk_n;
        } else {
            mlx_array_free(h_chunk);
        }
        offset += chunk_n;
        if (ok && offset < n_ids) {
            /* Materializza KV-cache + hidden di QUESTO chunk PRIMA del successivo e
             * svuota la cache allocator MLX. E' la strategia di mlx_lm generate_step
             * (prefill_step_size 2048, per chunk: forward + mx.eval dei cache-state +
             * mx.clear_cache). Senza questo eval i chunk si impilano in un unico grafo
             * lazy gigante (n_chunk x 60 layer x ~30 op) che l'eval finale esegue con
             * overhead superlineare e picco memoria alto: documentato gemma4 8K chunk
             * 2048 impilato = 41.9 tok/s vs single-chunk 8192 = 275. L'eval per-chunk
             * rende il chunking efficiente (memoria ~costante per chunk). */
            mlx_array_eval(h_batch);
            if (!llm_no_clear_cache()) mlx_clear_cache();
        }
    }
    m->q4e_ctx_ids = NULL; m->q4e_ctx_len = 0;
    m->aux_cache = NULL;
    free(q4e_combined);
    if (dp) {
        clock_gettime(CLOCK_MONOTONIC, &dp_t1);
        char dp_tag[64];
        snprintf(dp_tag, sizeof(dp_tag), "prefill/N=%d", n_ids);
        /* seconds=0 VOLUTO: il report stamperebbe ns/nodo e GB/s usando il tempo di COSTRUZIONE
         * del grafo come se fosse il tempo di dispatch (dopo il report arriva l'eval). */
        llm_mlx_dp_report(dp_tag, dp_chunks > 0 ? dp_chunks : 1, dp_lay, 0.0, 24);
        fprintf(stderr, "LLM_DISPATCH[%s]:   build grafo = %.1f s su %d chunk\n", dp_tag,
                (double)(dp_t1.tv_sec - dp_t0.tv_sec) +
                    (double)(dp_t1.tv_nsec - dp_t0.tv_nsec) * 1e-9, dp_chunks);
    }
    /* Pad della KV-cache al multiplo di 256 (step di mlx_lm `KVCache.update_and_fetch`:
     * n_steps = ceil((prev+q)/step) calcolato DURANTE il prefill). Senza, se la lunghezza non
     * e' multipla di 256 il PRIMO decode migra il buffer (Full + SliceUpdate/Broadcast extra
     * nel grafo del decode). Qui il lavoro si sposta nel prefill (una volta per sessione). */
    if (ok && s->has_cache) {
        const int used = s->cache_offset + n_ids;
        const int padded = ((used + 255) / 256) * 256;
        if (padded != used)
            for (int l = 0; l < m->cfg.n_layers; l++) {
                if (!s->has_cache[l] || s->k_cache[l].ctx == NULL || s->v_cache[l].ctx == NULL) continue;
                if (mlx_array_ndim(s->k_cache[l]) != 4 || mlx_array_ndim(s->v_cache[l]) != 4) continue;
                const int* ksh = mlx_array_shape(s->k_cache[l]);
                if (ksh == NULL) continue;
                const int nkv = ksh[1], cap = ksh[2], kd = ksh[3];
                if (cap != used) continue;
                const int vd = (int)mlx_array_shape(s->v_cache[l])[3];
                int zk[4] = {1, nkv, padded, kd}, zv[4] = {1, nkv, padded, vd};
                mlx_array zK = mlx_array_new(), zV = mlx_array_new();
                if (mlx_zeros(&zK, zk, 4, mlx_array_dtype(s->k_cache[l]), m->stream) != 0 ||
                    mlx_zeros(&zV, zv, 4, mlx_array_dtype(s->v_cache[l]), m->stream) != 0) {
                    mlx_array_free(zK); mlx_array_free(zV); continue;
                }
                int s0[4] = {0, 0, 0, 0}, sk[4] = {1, nkv, used, kd}, sv[4] = {1, nkv, used, vd}, st[4] = {1, 1, 1, 1};
                mlx_array K1 = mlx_array_new(), V1 = mlx_array_new();
                bool pok = mlx_slice_update(&K1, zK, s->k_cache[l], s0, 4, sk, 4, st, 4, m->stream) == 0 &&
                           mlx_slice_update(&V1, zV, s->v_cache[l], s0, 4, sv, 4, st, 4, m->stream) == 0;
                if (pok) {
                    mlx_array_set(&s->k_cache[l], K1);
                    mlx_array_set(&s->v_cache[l], V1);
                    mlx_array_eval(s->k_cache[l]);
                    mlx_array_eval(s->v_cache[l]);
                }
                mlx_array_free(zK); mlx_array_free(zV); mlx_array_free(K1); mlx_array_free(V1);
            }
    }

    if (ok) { s->cache_offset += n_ids; session_history_append(s, token_ids, n_ids); }
    /* NOTA (2026-09-29): la vecchia policy auto via env `LLM_KVQUANT_AUTO` e' stata RIMOSSA.
     * Ora l'attivazione automatica a contesto lungo e' una policy di REGISTRO (`kvquant=auto[:soglia]`,
     * default `auto:8192`) gestita in `llm_mlx_layer_forward.c` (compatibile e sicura mid-session:
     * il pool int8 Splash sostituisce l'fp, nessun latch fragile). Vedi `() - KV-int8-Splash-...`. */

    /* §G-ter: checkpoint del rewind ai confini di blocco (auto-prefill a chunk => i confini
     * cadono esattamente). Solo se il tier SSD e' attivo; nessun costo quando OFF. */
    if (ok && m->kvckpt_block > 0 && llm_kvdisk_enabled(m) &&
        (s->cache_offset % m->kvckpt_block) == 0)
        (void)llm_session_ckpt_write_k(m, s, s->cache_offset, NULL, 0);

    /* pending_h/pending_next_id (contratto FASE 3, native tool dispatch) devono
     * restare una singola riga [1,1,hidden] -- si estrae l'ULTIMA riga del
     * batch (l'ultimo token appena appeso) prima di chiamare final_step,
     * esattamente il ruolo che aveva l'unico `h` disponibile nel vecchio loop.
     * NB: h_batch è l'output dell'ULTIMO chunk (dimensione [1, last_chunk_n,
     * hidden]) -- lo slice usa last_chunk_n, non n_ids. */
    mlx_array h_last = mlx_array_new();
    if (ok) {
        /* Larghezza dell'ultimo asse presa dal tensore REALE (mlx_array_dim), non da
         * m->cfg.hidden_size: per qwen4_exp ogni layer restituisce hc_width
         * (hc_count*hidden_size), non hidden_size -- usare la costante troncava
         * silenziosamente il tensore al solo primo "stream" (bug reale trovato
         * 2026-09-06 testando il checkpoint reale, mai un crash: shape valida ma
         * sbagliata). Per ogni altra architettura hc_width==hidden_size, nessun
         * cambiamento di comportamento. */
        const int start[3] = {0, last_chunk_n - 1, 0};
        const int stop[3] = {1, last_chunk_n, mlx_array_dim(h_batch, 2)};
        const int strides[3] = {1, 1, 1};
        ok = mlx_slice(&h_last, h_batch, start, 3, stop, 3, strides, 3, m->stream) == 0;
    }
    if (ok) {
        int32_t next_id = 0;
        ok = final_step(m, h_last, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                        llm_mlx_session_sampling(s), &next_id);
        if (ok) {
            s->pending_next_id = next_id;
            s->has_pending = true;
            mlx_array_free(s->pending_h);
            s->pending_h = mlx_array_new();
            mlx_array_set(&s->pending_h, h_last);
        }
    }
    mlx_array_free(h_last);
    mlx_array_free(h_batch);
    free_output_tensors(&t);
    /* Timeline del working set: i ~56 GB sopra il modello su Mixtral esistono gia' dopo il
     * prefill, o li creano i passi di decode? (llm_mlx_dp_memory e' no-op senza LLM_DISPATCH) */
    llm_mlx_dp_memory("post-prefill");
    llm_mlx_array_balance("post-prefill");
    llm_ai_mlx_debug_array_balance("post-prefill", NULL, NULL, NULL);
    return ok ? 0 : -1;
}

/* PFlash (#287) — hook di PREFILL: comprime il prompt a contesto lungo PRIMA di costruire la KV.
 * Default AUTO (`LLM_PFLASH=off|auto|always`, profilo validato del riferimento: keep_ratio 0.20,
 * sink 256, tail 2048, min_keep 2048, block 128, query_window 512, stride 8).
 * PROMOSSO default ON il 2026-09-18 dopo il gate di qualita' (needle recall 4/4 a 5/25/50/75%,
 * ~20k, profilo validato + control senza fabbricazione); sotto threshold (32768) non comprime e
 * il kill-switch e' `LLM_PFLASH=off`.
 *
 * Perche' proprio qui e con queste regole:
 *  - SOLO sessioni FREDDE (cache_offset==0 && history_len==0): un append mid-conversazione non e'
 *    un prompt completo; comprimerlo scarterebbe token gia' "detti" e sposterebbe le posizioni
 *    dei turni successivi.
 *  - La sequenza compressa e' una "positional fiction" (posizione i != posizione i del prompt
 *    originale) => il percorso di prefix-cache va SALTATO per la richiesta compressa: v. il
 *    guard in llm_ai_mlx_session_append_reuse_prefix. Nota sul salvataggio: i prefissi sono
 *    chiavati sulla sequenza ESATTA dei token, quindi un prefisso compresso puo' essere riusato
 *    solo da una sequenza identica (posizioni coerenti per costruzione).
 *  - Fail-open: su qualunque errore di allocazione/capacita' si usa il prompt ORIGINALE (mai un
 *    prompt troncato a meta').
 *  - has_tools/requires_prompt_integrity non sono noti a questo livello (l'API riceve solo token
 *    id, il templating vive in C#): v1 li passa a 0; il gate "tools" del compressore resta
 *    disponibile per un chiamante che abbia il segnale. Il kill-switch e' l'env (default OFF). */
/* ── P4 — MEMORY PLAN + ADMISSION (2026-09-19) ─────────────────────────────────
 * Budget deterministico e riproducibile:
 *   limit  = min(iogpu.wired_limit_mb, RAM_fisica*3/4)   [il cap Metal/wired del sistema]
 *   budget = limit - pesi_MLX(misurati al primo append, prima che la KV cresca)
 *            - riserva (LLM_P4_RESERVE_MB, default 2048)
 * KV bytes/token dai config: per OGNI layer di attenzione 2(K,V)*n_kv_heads*head_dim*2B;
 * i layer GDN/linear NON consumano per-token (stato ricorrente COSTANTE, contato a parte);
 * i layer sliding sono capped alla finestra (oltre la finestra non crescono).
 * L'APPEND verifica che il KV richiesto (offset+n) + lo stato entri nel budget: se no
 * RIFIUTA esplicitamente (niente OOM silenzioso, criterio P4). */

int llm_ai_mlx_session_append(llm_ai_mlx_session_handle handle, const int32_t* token_ids, int n_ids) {
    LlmMlxSession* s = (LlmMlxSession*)handle;
    /* P4 ADMISSION: il KV richiesto deve entrare nel budget dichiarato (niente OOM). */
    if (s == NULL || s->model == NULL) return -1;
    llm_p4_plan_init(s->model);
    {
        const uint64_t need = llm_p4_kv_needed(s->model, (int)s->cache_offset, n_ids);
        const uint64_t other = llm_p4_used_sessions(s->model, s);
        if (s->model->p4_enforce && (need + other) > s->model->p4_budget_bytes) {
            fprintf(stderr, "[p4-admission] RIFIUTATO: KV richiesto %.2fGB + altre sessioni %.2fGB "
                            "> budget %.2fGB (sessioni attive=%d, offset=%d + n=%d, kv/token=%.1fKB). "
                            "Alza LLM_P4_RESERVE_MB o accorcia il contesto.\n",
                    need / 1073741824.0, other / 1073741824.0,
                    s->model->p4_budget_bytes / 1073741824.0, s->model->p4_nsessions,
                    (int)s->cache_offset, n_ids, s->model->p4_kv_per_token / 1024.0);
            return -3;
        }
    }
    LlmPflashConfig cfg;
    if (s == NULL || token_ids == NULL || n_ids <= 0) return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    if (s->model && s->model->dsv41) return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids); /* niente pflash */
    if (!llm_pflash_config_from_env(&cfg)) return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    if (s->cache_offset != 0 || s->history_len != 0) return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    if (!llm_pflash_should_attempt(&cfg, n_ids)) return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    int32_t* buf = (int32_t*)malloc((size_t)n_ids * sizeof(int32_t));
    if (buf == NULL) return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    LlmPflashResult r;
    int rc;
    if (!llm_pflash_compress(token_ids, n_ids, &cfg, 0, 0, buf, n_ids, &r)) {
        free(buf);
        return llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    }
    if (r.compressed) {
        fprintf(stderr, "[pflash] compressed %d -> %d tokens (ratio %.3f, middle=%d)\n",
                r.original_tokens, r.kept_tokens,
                (double)r.kept_tokens / (double)r.original_tokens, r.middle_tokens_kept);
        rc = llm_ai_mlx_session_append_impl(handle, buf, r.kept_tokens);
    } else {
        fprintf(stderr, "[pflash] skipped (%s) tokens=%d\n", r.reason, r.original_tokens);
        rc = llm_ai_mlx_session_append_impl(handle, token_ids, n_ids);
    }
    free(buf);
    return rc;
}

/* Data-gen per il SUPPORT DSpark: forward dei token (UN chunk, n <= prefill_chunk_size) e copia
 * in host l'hidden dell'ULTIMO layer per OGNI posizione del chunk (out: n*hidden fp32). La KV
 * della sessione cresce (cache_offset += n) cosi' il chiamante può streamare il corpus a chunk e
 * ottenere h_p per ogni posizione p con attenzione causale corretta sul passato. A differenza di
 * session_append NON esegue final_step/lm_head e NON aggiorna pending (puro data-gen, nessun
 * generate dopo). Ritorna n (righe catturate) su successo, -1 su errore/cap insufficiente. */
int llm_ai_mlx_session_capture_hidden(llm_ai_mlx_session_handle handle,
                                      const int32_t* token_ids, int n_ids,
                                      float* out, int cap) {
    if (handle == NULL || token_ids == NULL || n_ids <= 0 || out == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    LlmMlxModel* m = s->model;
    { int _g = llm_mlx_active_guard(m); if (_g != 0) return _g; }
    const int chunk_size = s->prefill_chunk_size > 0 ? s->prefill_chunk_size : 2048;
    if (n_ids > chunk_size) return -1; /* un chunk alla volta (data-gen: chiamare per chunk) */
    const int64_t need = (int64_t)n_ids * m->cfg.hidden_size;
    if ((int64_t)cap < need) return -1;

    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;
    mlx_array h = mlx_array_new();
    bool ok = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                   token_ids, n_ids, s->cache_offset,
                                   s->k_cache, s->v_cache, s->has_cache, &h);
    if (ok) {
        mlx_array hf = mlx_array_new();
        ok = mlx_astype(&hf, h, MLX_FLOAT32, m->stream) == 0;
        if (ok) ok = mlx_array_eval(hf) == 0;
        if (ok) {
            const float* d = mlx_array_data_float32(hf);
            size_t total = (size_t)n_ids * m->cfg.hidden_size;
            ok = (d != NULL && mlx_array_size(hf) == total);
            if (ok) memcpy(out, d, total * sizeof(float));
        }
        /* NOTA: mlx_array_detach NON e' API mlx-c upstream (era una patch debug locale del bug
         * retention, inefficace) — rimosso per la build su mlx-c pulito/0.32. */

        mlx_array_free(hf);
    }
    mlx_array_free(h);
    if (ok) {
        s->cache_offset += n_ids;
        session_history_append(s, token_ids, n_ids);
    }
    /* Svuota la cache allocator MLX: in data-gen si chiama capture a chunk consecutivi e senza
     * clear_cache il pool GPU cresce a ogni chunk (stesso motivo del chunked prefill 8b451f40)
     * fino a far uccidere il processo (Jetsam/OOM su macOS). */
    mlx_clear_cache();
    free_output_tensors(&t);
    return ok ? n_ids : -1;
}
