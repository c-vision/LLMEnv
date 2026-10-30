/* ============================================================================
 * llm_mlx_session_generate.c — dominio generazione sessione: pipeline async greedy (D16bis), generate/generate_batch_spec/generate_batch e setter di sampling/prefill.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_array_count.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_dispatch_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>   /* auto-detect draft DFlash2 (stat) */

static int g_async_decode_state = -1;
static bool llm_mlx_async_decode_enabled(void) {
    if (g_async_decode_state < 0) g_async_decode_state = (llm_ai_mlx_opt_env("LLM_NO_ASYNC_DECODE") != NULL) ? 0 : 1;
    return g_async_decode_state == 1;
}

/* D16bis: pipeline di decode asincrona (schema mlx_lm/generate.py -- async_eval + doppio
 * buffering, MAI kernel Metal ne' mlx_compile, vedi il piano). Path SOLO greedy (il chiamante
 * verifica sampling.temperature<=0 prima di entrare qui). Ad ogni iterazione:
 *   1. si costruisce il grafo del token in sospeso `id_pend` (mlx_array lazy, prodotto
 *      dall'argmax dell'iterazione PRECEDENTE e MAI letto finora) dentro COPIE DI STAGING di
 *      k_cache/v_cache (mlx_array_set = riassegnazione di un puntatore refcounted, non una
 *      copia dei dati -- stesso costo di un mlx_array_set qualunque altrove in questo file);
 *   2. si calcola il nuovo id (lazy) e si mette in coda TUTTO (nuovo id + cache di staging
 *      aggiornate) con mlx_async_eval, che NON blocca;
 *   3. SOLO ORA si legge (bloccante) il valore host di `id_pend` -- per costruzione la GPU ha
 *      avuto l'intera costruzione+dispatch del grafo di QUESTA iterazione (misurato: 704 nodi
 *      con Primitive su llama 1B denso, 4264-4437 sui grandi MoE -- la stima «~500+ op» che
 *      stava qui era vera SOLO per il denso; LLM_DISPATCH=1, Todo L26) per
 *      finire il lavoro accodato per `id_pend` alla iterazione PRECEDENTE, invece di bloccare
 *      subito dopo averlo prodotto (quello che fa il loop eager oggi, zero overlap).
 * Se `id_pend` risulta EOS lo staging viene scartato SENZA commit -- `s->k_cache`/`v_cache`/
 * `cache_offset` restano esattamente come prima di questa iterazione (stessa semantica "eos
 * NON emesso" del loop eager); l'unico costo e' una passata di forward calcolata e poi
 * scartata per il token che si e' rivelato EOS, mai visibile all'esterno (out_ids/cache_offset
 * identici al path eager). Se non e' EOS, lo staging viene committato normalmente. */
static double llm_mlx_dp_ms(const struct timespec* a, const struct timespec* b) {
    return (double)(b->tv_sec - a->tv_sec) * 1000.0 + (double)(b->tv_nsec - a->tv_nsec) * 1e-6;
}

static bool llm_ai_mlx_session_generate_async_greedy(LlmMlxSession* s, LlmMlxModel* m, OutputTensors* t,
                                                      int max_new_tokens, int32_t eos_token_id,
                                                      int32_t* out_ids, int* out_generated_count) {
    const int L = s->n_layers;

    int generated_count = 0;
    bool ok = true;
    bool eos_hit = false;
    int32_t last_id = s->pending_next_id;
    mlx_array id_pend = mlx_array_new();
    {
        const int shp[1] = {1};
        mlx_array seed = mlx_array_new_data(&s->pending_next_id, shp, 1, MLX_INT32);
        mlx_array_set(&id_pend, seed);
        mlx_array_free(seed);
    }

    /* G0 (Todo L3/L26): conta i nodi con primitiva che OGNI passo di decode costruisce. Il
     * prefill non va sottratto: a generazione iniziata il suo grafo NON e' piu' camminabile
     * (baseline misurata = 0 nodi, vedi la REGOLA OPERATIVA nel header). Costo zero se il flag
     * LLM_DISPATCH non e' attivo: ogni entry point controlla la flag e ritorna subito. */
    const bool dp = llm_mlx_dp_enabled();
    int dp_steps = 0;
    struct timespec dp_t0, dp_t1, dp_b0 = {0, 0}, dp_b1 = {0, 0}, dp_b2 = {0, 0}, dp_b3 = {0, 0};
    if (dp) {
        llm_mlx_dp_reset();
        llm_mlx_dp_set_moe(m->cfg.n_experts, m->cfg.top_k);
        clock_gettime(CLOCK_MONOTONIC, &dp_t0);
    }
    /* LLM_ASYNC_TIME: timer PROPRI (non il profiler che cammina il grafo) per separare le fasi
     * del loop async: build (forward+final_step) / submit (async_eval) / wait (eval del token
     * precedente) / rest (read+history+pending_h+clear_cache). */
    const bool atime = getenv("LLM_ASYNC_TIME") != NULL;
    double at_build = 0, at_submit = 0, at_wait = 0, at_rest = 0; int at_n = 0;
    struct timespec at_a = {0, 0}, at_b = {0, 0}, at_c = {0, 0}, at_d = {0, 0}, at_e = {0, 0};
#define AT_MS(x, y) ((double)((y).tv_sec - (x).tv_sec) * 1e3 + (double)((y).tv_nsec - (x).tv_nsec) * 1e-6)

    /* P0-ter DECODE (2026-09-16): pipeline a LAG>1. Con LAG=1 (default) la readback
     * (`mlx_array_eval`) avviene subito dopo l'async_eval del token N -> DREna lo stream e
     * include il grafo di N appena accodato: il pipeline serializza (wall ~ CPU+GPU invece di
     * max) e il 'wait' misurato e' G-CPU (~0,81 ms a ctx4096). Con LAG=2 si legge l'id DUE
     * iterazioni dopo: la GPU ha ~2x CPU (~4,3 ms) per completare ~3 ms -> wait -> 0.
     * Pura schedulazione: gli IDS devono restare identici. LLM_ASYNC_LAG=2 per attivarlo. */
    static int s_lag_state = -1;
    if (s_lag_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_ASYNC_LAG");
        s_lag_state = (e != NULL) ? atoi(e) : 1;
        if (s_lag_state < 1) s_lag_state = 1;
        if (s_lag_state > 2) s_lag_state = 2;
    }
    const int s_lag = s_lag_state;
    mlx_array s_fifo[5]; int s_fifo_n = 0;
    if (s_lag > 1) { mlx_array seed = mlx_array_new(); mlx_array_set(&seed, id_pend); s_fifo[s_fifo_n++] = seed; }

    for (int step = 0; ok && step < max_new_tokens; step++) {
        if (dp) clock_gettime(CLOCK_MONOTONIC, &dp_b0);
        if (atime) clock_gettime(CLOCK_MONOTONIC, &at_a);
        const int stage_cache_offset = s->cache_offset;

        mlx_array h = mlx_array_new();
        /* MLA (Kimi): la cache ausiliaria per-layer (k_pe) vive nella sessione; va passata al
         * modello per la durata del forward (come fa il path eager). Senza, il decode MLA
         * perde il k_pe del prefill e degenera. */
        m->aux_cache = s->aux_cache;
        /* COMPILED+ASYNC (esperimento 2026-09-16, NEGATIVO -> default OFF, LLM_COMPILE_ASYNC=1):
         * la closure compilata riduce la costruzione del grafo ma il costo CPU/token NON e' li':
         * con 375 primitive il `build` costa ~0,26 ms (0,7 us/op) mentre `mlx_async_eval` costa
         * ~1,15 ms (~3 us/nodo di encoding Metal). Con la closure: build 0,30 (+0,04) e submit
         * 1,32 (+0,16) = +14% CPU/token, IDS identici. Il collo e' l'ENCODING, non il grafo:
         * la leva e' ridurre il NUMERO di dispatch Metal (fusione), non compilarlo. */
        const bool use_compiled_async =
            (llm_ai_mlx_opt_env("LLM_COMPILE_ASYNC") != NULL) && (step >= 1) && llm_mlx_compile_enabled() &&
            llm_mlx_fixkv_enabled() && llm_mlx_static_cache_ready(s) && m->cfg.kv_max_tokens <= 0 &&
            !m->cfg.kv_quantize && llm_mlx_compile_arch_supported(m);
        mlx_array s_latest = mlx_array_new();
        if (s_lag > 1 && s_fifo_n > 0) mlx_array_set(&s_latest, s_fifo[s_fifo_n - 1]);
        else mlx_array_set(&s_latest, id_pend);
        if (use_compiled_async) {
            ok = llm_mlx_session_run_compiled_step_arr(s, s_latest, stage_cache_offset, &h);
            if (!ok) {
                fprintf(stderr, "mlx_compile+async: compiled step fallito, passo %d (fallback)\n", step);
                ok = forward_one_token_dev(m, s, t->embed_w, t->embed_s, t->embed_b, s_latest,
                                           stage_cache_offset, s->k_cache, s->v_cache, s->has_cache, &h);
            }
        } else {
            ok = forward_one_token_dev(m, s, t->embed_w, t->embed_s, t->embed_b, s_latest, stage_cache_offset,
                                        s->k_cache, s->v_cache, s->has_cache, &h);
        }
        mlx_array_free(s_latest);
        if (llm_da_enabled()) g_da.fwd++;
        m->aux_cache = NULL;
        /* DIAGNOSTICO (LLM_KV_MATERIALIZE=1 = ogni token, LLM_KV_MATERIALIZE_EVERY=<K> = ogni
         * K token): materializza le cache KV. Serve a verificare l'ipotesi che la LINEAGE LAZY
         * della cache (slice_update della cache precedente, che la referenzia — E il successivo
         * mlx_slice(0:need) di llm_mlx_kv_append_inplace che vi si appoggia, vedi todo P0-ter
         * continuazione 2026-09-16) si approfondisca di un livello per iterazione -> costo di
         * scheduling crescente (il degrado misurato: 7,7 -> 22 ms/token, indipendente dal
         * contesto). Materializzando OGNI token (K=1, LLM_KV_MATERIALIZE) la lineage riparte
         * da un buffer puro ma si paga il sync TOTALE ogni volta (gia' misurato: sposta il
         * costo submit->build, NON lo riduce). LLM_KV_MATERIALIZE_EVERY=K tappa la profondita'
         * della catena a K invece di azzerarla sempre o lasciarla crescere al plateau naturale:
         * ipotesi da verificare (non ancora misurata, serve macchina silenziosa + coppie
         * alternate). Default OFF per entrambe le varianti, zero overhead se non impostate. */
        static int kv_mat_every = -1;
        if (kv_mat_every < 0) {
            const char* mev = llm_env_get("LLM_KV_MATERIALIZE_EVERY");
            kv_mat_every = (mev != NULL) ? atoi(mev) : (llm_env_get("LLM_KV_MATERIALIZE") ? 1 : 0);
            if (kv_mat_every < 0) kv_mat_every = 0;
        }
        if (kv_mat_every > 0) {
            static int kv_mat_step = 0;
            if ((++kv_mat_step % kv_mat_every) == 0) {
                mlx_vector_array kvv = mlx_vector_array_new();
                for (int l = 0; l < s->n_layers; l++) {
                    if (s->k_cache[l].ctx != NULL) mlx_vector_array_append_value(kvv, s->k_cache[l]);
                    if (s->v_cache[l].ctx != NULL) mlx_vector_array_append_value(kvv, s->v_cache[l]);
                }
                if (mlx_vector_array_size(kvv) > 0) mlx_eval(kvv);
                mlx_vector_array_free(kvv);
            }
        }
        if (step < 4) llm_mlx_array_balance("fwd");
        mlx_array next_id = mlx_array_new();
        if (ok) ok = final_step_greedy_dev(m, h, t->final_norm_w, t->out_w, t->out_s, t->out_b, &next_id);
        if (llm_da_enabled()) g_da.final_steps++;
        if (step < 4) llm_mlx_array_balance("fin");
        if ((step & 63) == 0) llm_mlx_array_balance("decode");

        if (ok) {
            /* REPLICA mlx_lm `mx.async_eval(y, logprobs)`: si mette in coda il SOLO output
             * del passo (l'id greedy); le cache KV aggiornate nel grafo si materializzano
             * transitivamente come dipendenze. Nessuna frontier bespoke delle cache (era
             * quella a tenerle referenziate -> use_count>1 -> niente donazione). */
            if (dp) { mlx_array _a[1] = {next_id}; clock_gettime(CLOCK_MONOTONIC, &dp_b1);
                      llm_mlx_dp_count(_a, 1); clock_gettime(CLOCK_MONOTONIC, &dp_b2); dp_steps++; }
            if (atime) clock_gettime(CLOCK_MONOTONIC, &at_b);
            if (next_id.ctx != NULL) {
                mlx_vector_array vec = mlx_vector_array_new();
                mlx_vector_array_append_value(vec, next_id);
                mlx_async_eval(vec);
                if (llm_da_enabled()) g_da.async_submits++;
                mlx_vector_array_free(vec);
            }
        }
        if (atime) clock_gettime(CLOCK_MONOTONIC, &at_c);

        if (s_lag > 1) {
            /* ---- percorso LAG>=2 (sperimentale, LLM_ASYNC_LAG=2) ---- */
            if (ok && s_fifo_n < 5) s_fifo[s_fifo_n++] = next_id;   /* ownership a FIFO */
            else if (ok) mlx_array_free(next_id);
            if (atime) clock_gettime(CLOCK_MONOTONIC, &at_d);
            bool fifo_ok = ok;
            while (fifo_ok && s_fifo_n > s_lag) {
                mlx_array old = s_fifo[0];
                for (int i = 1; i < s_fifo_n; i++) s_fifo[i - 1] = s_fifo[i];
                s_fifo_n--;
                int32_t idh = 0;
                bool rok = (mlx_array_eval(old) == 0);
                if (rok) { const int32_t* dd = mlx_array_data_int32(old); rok = (dd != NULL); if (rok) idh = dd[0]; }
                mlx_array_free(old);
                if (!rok) { fifo_ok = false; break; }
                if (eos_token_id >= 0 && idh == eos_token_id) { eos_hit = true; last_id = idh; break; }
                if (generated_count < max_new_tokens) {
                    out_ids[generated_count++] = idh;
                    if (llm_da_enabled()) g_da.tokens++;
                    session_history_append(s, &idh, 1);
                    last_id = idh;
                }
            }
            if (!fifo_ok) { mlx_array_free(h); ok = false; fprintf(stderr, "LLM_ASYNC_DECODE(lag): passo %d fallito\n", step); break; }
            s->cache_offset++;   /* avanzato per BUILD (1 per iterazione), non per emit */
            mlx_array_free(s->pending_h);
            s->pending_h = mlx_array_new();
            mlx_array_set(&s->pending_h, h);
            mlx_array_free(h);
            if (eos_hit) break;
            goto lag_iter_end;
        }

        int32_t id_pend_host = 0;
        bool read_ok = ok && mlx_array_eval(id_pend) == 0;
        if (llm_da_enabled()) g_da.readbacks++;
        if (read_ok) {
            const int32_t* d = mlx_array_data_int32(id_pend);
            read_ok = (d != NULL);
            if (read_ok) id_pend_host = d[0];
        }
        if (atime) clock_gettime(CLOCK_MONOTONIC, &at_d);
        if (dp) {
            clock_gettime(CLOCK_MONOTONIC, &dp_b3);
            llm_mlx_dp_stage(llm_mlx_dp_ms(&dp_b0, &dp_b1) * 1e-3,
                             llm_mlx_dp_ms(&dp_b2, &dp_b3) * 1e-3);
            char memtag[32];
            snprintf(memtag, sizeof(memtag), "step=%d", step);
            llm_mlx_dp_memory(memtag); /* traiettoria: i buffer vivi crescono passo-per-passo? */
        }
        if (!ok || !read_ok) {
            mlx_array_free(h); mlx_array_free(next_id);
            ok = false;
            fprintf(stderr, "LLM_ASYNC_DECODE: passo %d fallito\n", step);
            break;
        }

        if (eos_token_id >= 0 && id_pend_host == eos_token_id) {
            mlx_array_free(h); mlx_array_free(next_id);
            last_id = id_pend_host;
            eos_hit = true;
            break;
        }

        out_ids[generated_count++] = id_pend_host;
        if (llm_da_enabled()) g_da.tokens++;
        session_history_append(s, &id_pend_host, 1);
        s->cache_offset++;
        mlx_array_free(s->pending_h);
        s->pending_h = mlx_array_new();
        mlx_array_set(&s->pending_h, h);
        mlx_array_free(h);
        last_id = id_pend_host;

        mlx_array_free(id_pend);
        id_pend = next_id;

        /* mlx_lm `generate_step`: `if n % 256 == 0: mx.clear_cache()` (dopo la lettura del
         * token). Replica esatta per non far crescere il pool GPU a contesto lungo. */
        if (!llm_no_clear_cache() && step % 256 == 0) mlx_clear_cache();
        if (llm_da_enabled()) g_da.clear_cache++;

        if (atime) { clock_gettime(CLOCK_MONOTONIC, &at_e);
            at_build += AT_MS(at_a, at_b); at_submit += AT_MS(at_b, at_c);
            at_wait += AT_MS(at_c, at_d); at_rest += AT_MS(at_d, at_e); at_n++; }
    lag_iter_end:;

        /* Traccia per-finestra (LLM_ASYNC_TRACE=1): mostra la PROGRESSIONE del costo, non la
         * media cumulativa. Serve a distinguere un costo che dipende dal contesto da uno che
         * dipende dal numero di iterazioni. */
        if (atime && llm_env_get("LLM_ASYNC_TRACE") && at_n > 0 && ((step & 63) == 63)) {
            fprintf(stderr, "[ASYNC_TRACE] step=%d build=%.3f submit=%.3f wait=%.3f rest=%.3f tot=%.3f\n",
                    step + 1, at_build / at_n, at_submit / at_n, at_wait / at_n, at_rest / at_n,
                    (at_build + at_submit + at_wait + at_rest) / at_n);
            at_build = at_submit = at_wait = at_rest = 0; at_n = 0;
        }
        if (generated_count >= max_new_tokens) break;
    }
    if (atime && at_n > 0)
        fprintf(stderr, "[ASYNC_TIME] step=%d  build=%.3f submit=%.3f wait=%.3f rest=%.3f ms/token  (tot=%.3f)\n",
                at_n, at_build / at_n, at_submit / at_n, at_wait / at_n, at_rest / at_n,
                (at_build + at_submit + at_wait + at_rest) / at_n);

    if (s_lag > 1) {
        /* Draining: emetti i token rimasti nella FIFO fino al budget; l'ULTIMO rimasto e' il
         * pending_next_id della prossima chiamata (contratto invariato). */
        while (ok && !eos_hit && s_fifo_n > 0 && generated_count < max_new_tokens) {
            mlx_array old = s_fifo[0];
            for (int i = 1; i < s_fifo_n; i++) s_fifo[i - 1] = s_fifo[i];
            s_fifo_n--;
            int32_t idh = 0; bool rok = (mlx_array_eval(old) == 0);
            if (rok) { const int32_t* dd = mlx_array_data_int32(old); rok = (dd != NULL); if (rok) idh = dd[0]; }
            mlx_array_free(old);
            if (!rok) { ok = false; break; }
            if (eos_token_id >= 0 && idh == eos_token_id) { eos_hit = true; last_id = idh; break; }
            out_ids[generated_count++] = idh;
            if (llm_da_enabled()) g_da.tokens++;
            session_history_append(s, &idh, 1);
            last_id = idh;
        }
        /* l'ultimo rimasto -> pending della prossima generate */
        while (s_fifo_n > 0) {
            mlx_array old = s_fifo[0];
            for (int i = 1; i < s_fifo_n; i++) s_fifo[i - 1] = s_fifo[i];
            s_fifo_n--;
            if (mlx_array_eval(old) == 0) { const int32_t* dd = mlx_array_data_int32(old); if (dd != NULL) last_id = dd[0]; }
            mlx_array_free(old);
        }
    }
    if (ok && !eos_hit) {
        /* Flush: l'id in sospeso dell'ultima iterazione (mai letto se il loop e' finito per
         * max_new_tokens prima del prossimo giro) va materializzato per mantenere
         * pending_next_id un valore host noto, come richiede il resto dell'API di sessione. */
        if (mlx_array_eval(id_pend) == 0) {
            const int32_t* d = mlx_array_data_int32(id_pend);
            if (d != NULL) last_id = d[0];
        }
    }
    s->pending_next_id = last_id;
    mlx_array_free(id_pend);

    if (dp) {
        clock_gettime(CLOCK_MONOTONIC, &dp_t1);
        const double dp_sec = (double)(dp_t1.tv_sec - dp_t0.tv_sec)
                            + (double)(dp_t1.tv_nsec - dp_t0.tv_nsec) * 1e-9;
        char dp_tag[64];
        snprintf(dp_tag, sizeof(dp_tag), "%s/N=%d", m->cfg.model_type ? m->cfg.model_type : "?", s->cache_offset);
        /* 64 = stampa TUTTE le famiglie: col taglio a 14 il 12% dei nodi restava senza nome, e
         * su qwen3_moe finiva sotto taglio proprio GatherQMM (2/layer) -- il dato decisivo. */
        llm_mlx_dp_report(dp_tag, dp_steps, L, dp_sec, 64);
    }

    *out_generated_count = generated_count;
    return ok;
}

/* Forward decl (definizione piu' avanti): usata dall'audit M1 per i bytes/token. */
int llm_ai_mlx_model_stats(llm_ai_mlx_model_handle handle, int seq_len, double* out_params, double* out_kv_bytes);
/* P0-ter (2026-09-16): parametri ATTIVI per token (MoE: solo top_k esperti) - per il roofline di banda. */
int llm_ai_mlx_model_active_params(llm_ai_mlx_model_handle handle, double* out_active);
int llm_ai_mlx_model_flops(llm_ai_mlx_model_handle handle, int seq_len);

/* DFlash2 (A1): handle per-processo (l'opzione `dflash_model` e' globale). Creato lazy alla prima
 * generate col target coerente; riusato -> il ctx del draft persiste tra le chiamate (streaming). */
static llm_ai_mlx_dflash2_handle g_df2_handle = NULL;
static LlmMlxModel* g_df2_target = NULL;
/* AUTO-DETECT del draft DFlash2: sibling "<dir>/<stem-senza-quant>-DFlash2" del modello target.
 * Es. /…/Qwen3.8-27B-8bit -> /…/Qwen3.8-27B-DFlash2. Cosi' la spec DFlash2 si attiva senza passare
 * `dflash_model` a mano (il default del registro e' vuoto perche' il draft e' model-specific).
 * Ritorna 1 e scrive `out` se trovato. */
static int llm_mlx_df2_autodetect(LlmMlxModel* m, char* out, int cap) {
    if (m == NULL || m->model_dir[0] == '\0') return 0;
    const char* slash = strrchr(m->model_dir, '/');
    if (slash == NULL || slash == m->model_dir) return 0;
    static const char* quants[] = { "-8bit", "-6bit", "-4bit", "-2bit", "-Q8", "-Q6", "-Q4",
                                    "-mxfp4", "-MLX-8Bit", "-MLX-4Bit", "-8Bit", "-4Bit" };
    char stem[512];
    snprintf(stem, sizeof stem, "%s", slash + 1);
    for (size_t i = 0; i < sizeof(quants) / sizeof(quants[0]); i++) {
        const size_t ql = strlen(quants[i]), sl = strlen(stem);
        if (sl > ql && strcmp(stem + sl - ql, quants[i]) == 0) { stem[sl - ql] = '\0'; break; }
    }
    char cand[1024];
    snprintf(cand, sizeof cand, "%.*s/%s-DFlash2", (int)(slash - m->model_dir), m->model_dir, stem);
    struct stat st;
    if (stat(cand, &st) == 0 && S_ISDIR(st.st_mode)) { snprintf(out, (size_t)cap, "%s", cand); return 1; }
    return 0;
}

/* 1 se la generazione usera' un drafter DFlash2 (opt esplicita o auto-detect). SIDE-EFFECT FREE:
 * non crea il drafter ne' tocca i global g_df2_*. Serve a decidere se EVITARE la radix
 * prefix-cache (llm_ai_mlx_session_append_auto_prefix), perche' il path di riuso prefisso non
 * esegue il prefill e quindi non arma la cattura-ctx del drafter -> draft degenerato (loop).
 * Diagnosi 2026-10-03 (test DFlash2Native: greedy con auto-detect andava in loop `<think>` x40). */
int llm_mlx_df2_will_use(LlmMlxModel* m) {
    if (m == NULL || m->deleg != NULL) return 0;
    char path[1024];
    if (llm_ai_mlx_opt_get_str("dflash_model", path, sizeof(path)) == 0 && path[0] != 0) return 1;
    return llm_mlx_df2_autodetect(m, path, sizeof(path));
}

static llm_ai_mlx_dflash2_handle llm_mlx_df2_for(LlmMlxModel* m) {
    if (g_df2_target == m) return g_df2_handle;   /* gia' tentato per questo target */
    char path[1024];
    if (llm_ai_mlx_opt_get_str("dflash_model", path, sizeof(path)) != 0 || path[0] == 0) {
        /* opt vuota: prova l'auto-detect del sibling <stem>-DFlash2 accanto al modello.
         * (2026-10-01) La divergenza sui GDN e' RISOLTA: il greedy usa forward_tokens_batch
         * (la cattura-snap perturbava il forward ricorrente) e il ctx e' ricatturato su cache
         * scratch. Dump token-id: ON == OFF bit-identici. Auto-detect ora per TUTTI i target. */
        if (!llm_mlx_df2_autodetect(m, path, sizeof(path))) return NULL;
    }
    g_df2_target = m;
    /* LARGE-CONTEXT (2026-10-01): il prefill di contesti lunghi lascia un picco di buffer IN-FLIGHT
     * (a 257k ~80GB attivi, che scendono a ~38GB solo al primo EVAL del decode: la coda GPU si
     * drena). Il drafter veniva caricato PRIMA di quel drain -> picco+drafter = OOM OS. Forziamo il
     * drain (synchronize = attende la coda) e il ritorno dei buffer liberati all'OS (clear_cache),
     * ripetuto, PRIMA di caricare il drafter. */
    for (int _i = 0; _i < 4; _i++) { mlx_synchronize(m->stream); mlx_clear_cache(); }
    g_df2_handle = llm_ai_mlx_dflash2_create((llm_ai_mlx_model_handle)m, path);
    if (getenv("LLM_SPEC_DEBUG"))
        fprintf(stderr, "df2_for: path=%s handle=%p\n", path, (void*)g_df2_handle);
    return g_df2_handle;
}

/* PRECARICA il drafter DFlash2 PRIMA del prefill (2026-10-01): il prefill di contesti lunghi
 * lascia un picco di array VIVI (~80GB a 257k) che viene DRAINATO solo dal PRIMO FORWARD del
 * decode (sync/clear non bastano). Se il drafter viene caricato DOPO il prefill, il picco + il
 * drafter -> OOM OS. Caricandolo PRIMA, il picco del prefill lo include (80+4 < limite) e il
 * primo forward dello streaming lo drena -> 1M/2M entrano. */
void llm_mlx_df2_preload(LlmMlxModel* m) {
    if (!m) return;
    llm_ai_mlx_dflash2_handle h = llm_mlx_df2_for(m);
    /* Arma la CATTURA ctx durante il prefill (evita il re-forward nello streaming). */
    if (h != NULL) (void)llm_mlx_df2_pf_arm((DFlash2Runtime*)h, m);
}

int llm_ai_mlx_session_generate(llm_ai_mlx_session_handle handle, int max_new_tokens,
                                 int32_t eos_token_id, int32_t* out_ids) {
    if (handle == NULL || max_new_tokens <= 0 || out_ids == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (s->parked) (void)llm_session_unpark(s->model, s); /* §G-ter: restore on-demand PRIMA del check */
    if (!s->has_pending) return -1; /* nothing appended yet -- no predicted continuation to start from */
    LlmMlxModel* m = s->model;
    /* MTP AUTO (2026-09-21): se il modello ha una head MTP (sidecar MTPLX) e il sampling
     * e' GREEDY, il decode usa il path speculativo (output identico al greedy AR, ~1.4x).
     * `LLM_MTP_AUTO=0` disattiva; `LLM_MTP_DEPTH` sceglie la profondita' (default 1). */
    if (!m->deleg && llm_mlx_mtp_get(m) != NULL) {
        const char* au = llm_ai_mlx_opt_env("LLM_MTP_AUTO");
        llm_mlx_sampling_t sm0 = llm_mlx_session_sampling(s);
        if ((au == NULL || au[0] != '0') && (!sm0.active || sm0.temperature <= 0.0f)) {
            /* STREAMING BUFFER: il consumer chiama generate(1) per token. Senza buffer il path MTP
             * emette il pending e fa break PRIMA del forward -> il pending non avanza -> token
             * ripetuto (stesso bug del vecchio n-gram). Genera un CHUNK MTP e lo bufferizza. */
            int cnt = 0;
            while (cnt < max_new_tokens) {
                if (s->spec_buf_pos >= s->spec_buf_len) {
                    llm_mlx_session_spec_begin(s);   /* checkpoint per lo scarto su append esterno */
                    int got = llm_ai_mlx_session_generate_mtp(handle, 16, eos_token_id, s->spec_buf);
                    if (getenv("LLM_SPEC_DEBUG")) fprintf(stderr, "mtp_branch: got=%d\n", got);
                    if (got <= 0) break;
                    if (got > 0) llm_mlx_dp_spec_add(got);
                    s->spec_buf_len = got; s->spec_buf_pos = 0;
                }
                int32_t tok = s->spec_buf[s->spec_buf_pos++];
                out_ids[cnt++] = tok;
                if (eos_token_id >= 0 && tok == eos_token_id) break;
            }
            if (cnt > 0) return cnt;   /* fallback a n-gram/AR se MTP non produce */
        }
    }
    /* DFlash2 (A1) greedy: drafter esterno a contesto PERSISTENTE, streaming buffer come MTP. */
    if (!m->deleg) {
        llm_ai_mlx_dflash2_handle dfh = llm_mlx_df2_for(m);
        if (dfh != NULL && !s->df2_off) {
            llm_mlx_sampling_t smd = llm_mlx_session_sampling(s);
            if (!smd.active || smd.temperature <= 0.0f) {
                int cnt = 0;
                while (cnt < max_new_tokens) {
                    if (s->spec_buf_pos >= s->spec_buf_len) {
                        if (s->df2_off) break;   /* speculazione disattivata: svuotato il buffer, si esce (il chiamante richiama e passa ad AR/n-gram) */
                        llm_mlx_session_spec_begin(s);
                        int got = llm_ai_mlx_dflash2_generate(dfh, handle, eos_token_id, 16, s->spec_buf);
                        if (getenv("LLM_SPEC_DEBUG")) fprintf(stderr, "df2_branch: got=%d\n", got);
                        if (got <= 0) break;
                        if (got > 0) llm_mlx_dp_spec_add(got);
                        s->spec_buf_len = got; s->spec_buf_pos = 0;
                        if (llm_mlx_df2_unprofitable(dfh, m)) { s->df2_off = true; fprintf(stderr, "[df2] speculazione DISATTIVATA: costo per token > AR (draft non abbinato al target?); si prosegue in AR/n-gram\n"); }
                    }
                    int32_t tok = s->spec_buf[s->spec_buf_pos++];
                    out_ids[cnt++] = tok;
                    if (eos_token_id >= 0 && tok == eos_token_id) break;
                }
                if (cnt > 0) return cnt;
            }
        }
    }
    /* N-GRAM self-speculative GENERALE (tutti i modelli, greedy): draft dalla tabella n-gram del
     * contesto, verificati dal modello con UN forward batch -> output IDENTICO al greedy AR.
     * Config-driven (opzione di registro `spec_ngram`, nessuna env). Default 8 (built-in). */
    if (!m->deleg) {
        int ord = 2;   /* default built-in (bigram: piu' copertura; l'8 dava nd<2 su code/ripetiz.) */
        if (llm_ai_mlx_opt_get_int("spec_ngram", &ord) != 0) ord = 8;
        /* Modelli IBRIDI con stato ricorrente (GDN qwen3_5, MLA+KDA kimi): lo stato vive in
         * sess->aux_cache, che il checkpoint/rollback del draft NON copre -> il rollback
         * lascerebbe un aux stantio/NULL (array vuoto CMLX / risultato errato, cfr. il
         * side-channel di forward_tokens_batch_multi_cap). Finche' il rollback non include
         * l'aux, il percorso self-speculative si disattiva per questi modelli. */
        /* Stesso vincolo per ogni architettura con stato ricorrente/convoluzionale fuori da K/V: LFM2 (conv corta),
         * Mamba, Falcon-H1, Jamba, Granite-hybrid, OLMo-hybrid, RWKV. Senza questa esclusione il rollback riporta
         * K/V ma non lo stato: il testo degenera (LFM2-8B generava solo virgolette). */
        const bool hybrid_recurrent = llm_mlx_model_has_recurrent_state(m) != 0;   /* elenco unico in llm_mlx_prefix.c */
        if (ord >= 2 && !hybrid_recurrent) {
            llm_mlx_sampling_t smn = llm_mlx_session_sampling(s);
            if (!smn.active || smn.temperature <= 0.0f) {
                if (!s->ngram_built) llm_ai_mlx_session_build_ngram(handle, ord);
                /* Speculative streaming: il consumer chiama generate(1) per token; il motore
                 * genera in anticipo un chunk e lo bufferizza. Usato SOLO se la tabella n-gram
                 * predice (>=2 token): altrimenti si usa il path principale, perche' il fallback
                 * AR sincrono di generate_ngram e' piu' lento dell'async sui modelli I/O-bound. */
                /* (B) NON ricalcolare spec_can all'ingresso: la prima iterazione del loop riusa il
                 * risultato (prima era chiamata 2 volte -> 2 build del draft per ingresso). */
                const bool buf_has = (s->spec_buf_pos < s->spec_buf_len);
                const bool can0 = buf_has ? false : llm_mlx_session_spec_can(s);
                /* L1 — AUTO-DISABLE (stile Lightning-mlx/Gufo, deterministico): se l'acceptance
                 * OSSERVATA e' bassa la spec costa piu' di quanto rende -> si torna al path AR
                 * principale. Soglia: EWMA dei token accettati per passata < 1.5 con >= 8 campioni
                 * (evita di spegnere per un singolo passo sfortunato). Latch di sessione. */
                const bool spec_profitable = (s->spec_drafts < 8) || (s->spec_acc_ewma >= 1.5f);
                if ((buf_has || can0) && spec_profitable) {
                    int cnt = 0;
                    bool first_iter = true;
                    while (cnt < max_new_tokens) {
                        if (s->spec_buf_pos >= s->spec_buf_len) {
                        const bool can_now = first_iter ? can0 : llm_mlx_session_spec_can(s);
                        first_iter = false;
                        if (!can_now) break;  /* n-gram non predice */
                        int w = llm_mlx_spec_adaptive_width(s, 16);   /* larghezza adattiva (Gufo) */
                        int chunk = w + 2; if (chunk < 2) chunk = 2; if (chunk > 60) chunk = 60;
                        llm_mlx_session_spec_begin(s);
                        int got = llm_ai_mlx_session_generate_ngram(handle, chunk, eos_token_id, s->spec_buf, w);
                            if (getenv("LLM_SPEC_DEBUG")) fprintf(stderr, "ngram_branch: got=%d chunk=%d w=%d\n", got, chunk, w);
                            if (got < 0) break;
                            if (got > 0) llm_mlx_dp_spec_add(got);
                            s->spec_buf_len = got; s->spec_buf_pos = 0;
                            if (got == 0) break;
                        }
                        int32_t tok = s->spec_buf[s->spec_buf_pos++];
                        out_ids[cnt++] = tok;
                        if (eos_token_id >= 0 && tok == eos_token_id) break;
                    }
                    if (cnt > 0) return cnt;
                }
            }
        }
    }
    /* Modello DELEGATO (DS4/DS4.1): generate via engine autonomo (stessa ABI a 9 metodi). */
    if (m->deleg) {
        int n = m->deleg->generate_ids
                ? m->deleg->generate_ids(s->deleg_sess, max_new_tokens, eos_token_id, out_ids) : -1;
        if (n > 0) session_history_append(s, out_ids, n);
        return n;
    }
    { int _g = llm_mlx_active_guard(m); if (_g != 0) return _g; }

    /* P0-ter: audit STRUTTURALE del decode (LLM_DECODE_AUDIT, default OFF). I contatori sono
     * deterministici -> validi anche a macchina carica (i tempi no, vedi §J). */
    struct timespec da_t0 = {0, 0}, da_t1 = {0, 0};
    llm_da_reset();
    if (llm_da_enabled()) {
        clock_gettime(CLOCK_MONOTONIC, &da_t0);
        /* M1: byte di PESI per token = params*(bits/8 + 4/group) (packed + scale/bias f16 per gruppo). */
        double prm = 0.0;
        /* NB: `m` (modello), NON `handle` (sessione): la stats casta a LlmMlxModel*. */
        double pact = 0.0;
        if (llm_ai_mlx_model_stats((llm_ai_mlx_model_handle)m, 1, &prm, NULL) == 0 && prm > 0.0) {
            const double bits = (double)(m->cfg.quant_bits > 0 ? m->cfg.quant_bits : 16);
            const double g = (double)(m->cfg.quant_group_size > 0 ? m->cfg.quant_group_size : 64);
            const double bpw = bits / 8.0 + (m->cfg.quant_bits > 0 ? 4.0 / g : 0.0);
            double head = 0.0;
            if (m->cfg.vocab_size > 0 && m->cfg.hidden_size > 0)
                head = (double)m->cfg.vocab_size * (double)m->cfg.hidden_size;   /* lm_head, letto per intero */
            /* MoE: il traffico per token sono i top_k esperti, non tutti. */
            (void)llm_ai_mlx_model_active_params((llm_ai_mlx_model_handle)m, &pact);
            if (!(pact > 0.0)) pact = prm;
            llm_da_set_weight_bytes((long)((pact + head) * bpw));
            llm_da_set_weight_bytes_total((long)((prm + head) * bpw));
        }
    }
    char da_tag[96] = {0};
    snprintf(da_tag, sizeof(da_tag), "%s/N=%d", m->cfg.model_type ? m->cfg.model_type : "?", s->cache_offset);
#define LLM_DA_REPORT() do { \
        if (llm_da_enabled() && da_t0.tv_sec != 0) { \
            clock_gettime(CLOCK_MONOTONIC, &da_t1); \
            llm_da_report(da_tag, (double)(da_t1.tv_sec - da_t0.tv_sec) \
                                + (double)(da_t1.tv_nsec - da_t0.tv_nsec) * 1e-9); \
        } } while (0)

    OutputTensors t;
    bool ok = load_output_tensors(m, &t);

    /* D16bis: pipeline asincrona, path SOLO greedy (temperature<=0) -- il path con
     * sampling attivo resta quello eager sotto, invariato (richiede comunque l'intero
     * vettore logits in host, nessun beneficio ovvio di pipelining senza riscrivere anche
     * il sampler come grafo MLX). Mutuamente esclusivo con RMLX10 Fase B (LLM_COMPILE): se
     * entrambi i flag sono attivi vince LLM_ASYNC_DECODE. */
    if (ok) {
        llm_mlx_sampling_t smp = llm_mlx_session_sampling(s);
        /* F1 (todo Qwen3.8-Flash-Next, 2026-09-18): il path ASYNC non imposta il side-channel
         * della PLE (`q4e_ctx_ids`/`q4e_ctx_len`) -> per i modelli con layer PLE
         * (`ple_layer_ids_n>0`, es. qwen4_exp) `qwen4_exp_ple_inject` fallisce e il decode
         * muore ("PLE richiesto ma non disponibile"). Finche' il side-channel non e' integrato
         * nel pipeline async, quei modelli usano il path EAGER (che lo imposta, riga ~20207 e
         * funziona: misurato 1.8 tok/s). Vedi Todo/(open) - Qwen38-FlashNext-60tok.md (F1). */
                if (llm_mlx_async_decode_enabled() && m->cfg.ple_layer_ids_n == 0 &&
                    (!smp.active || smp.temperature <= 0.0f)) {
            int generated_count = 0;
            bool async_ok = llm_ai_mlx_session_generate_async_greedy(s, m, &t, max_new_tokens, eos_token_id,
                                                                      out_ids, &generated_count);
            free_output_tensors(&t);
            LLM_DA_REPORT();
            return async_ok ? generated_count : -1;
        }
    }

    int generated_count = 0;
    int32_t current_id = s->pending_next_id;
    /* LLM_DISPATCH: profilo per-token del graph di DECODE (nodi per tipo, MB/token, build vs eval).
     * Il profiler esisteva solo per il prefill; cablato qui nel loop eager (il path dei modelli
     * PLE/qwen4). Radici = h (ultimo layer) + le KV cache di tutti i layer. */
    const bool dp_dec = llm_mlx_dp_enabled();
    double dp_build_s = 0.0, dp_eval_s = 0.0;
    if (dp_dec) { llm_mlx_dp_reset(); llm_mlx_dp_set_moe(m->cfg.n_experts, m->cfg.top_k); }
    for (int step = 0; ok && step < max_new_tokens; step++) {
        if (eos_token_id >= 0 && current_id == eos_token_id) break;
        struct timespec dpt0, dpt1, dpt2;
        if (dp_dec) clock_gettime(CLOCK_MONOTONIC, &dpt0);
        out_ids[generated_count++] = current_id;
        if (llm_da_enabled()) g_da.tokens++;
        session_history_append(s, &current_id, 1);
        /* NB: NESSUN break qui. Serve SEMPRE il forward che predice il prossimo pending, anche
         * all'ultimo token: il server chiama generate(s, 1) in un ciclo, e senza il forward il
         * pending non avanzerebbe -> token costante (bug "Parole Parole ...") e il path sampling
         * (che entra nel loop eager) non verrebbe MAI applicato. */

        mlx_array h = mlx_array_new();
        /* RMLX10 Fase B: dal SECONDO token di decode in poi, se il gate compile
         * e' attivo (LLM_COMPILE && LLM_FIXKV && Mixtral && cache statiche
         * uniformi) esegue la closure compilata; il PRIMO token resta eager
         * (fixkv) per convertire la cache eager->static (cap uniforme). */
        bool use_compiled = (step >= 1) && llm_mlx_compile_enabled() &&
                            llm_mlx_fixkv_enabled() && llm_mlx_static_cache_ready(s) &&
                            m->cfg.kv_max_tokens <= 0 && !m->cfg.kv_quantize &&
                            llm_mlx_compile_arch_supported(m);
        if (use_compiled) {
            ok = llm_mlx_session_run_compiled_step(s, current_id, s->cache_offset, &h);
            if (!ok) fprintf(stderr, "mlx_compile: compiled step fallito, passo %d\n", step);
        } else {
            /* Qwen4-Exp N-gram/PLE: session_history_append (sopra, prima di questo blocco)
             * ha GIA' appeso current_id -- s->history copre esattamente [0, cache_offset]
             * qui, la finestra di contesto che un layer PLE potrebbe richiedere. Side-channel
             * impostato solo per la durata di QUESTA chiamata (vedi campo in LlmMlxModel). */
            m->q4e_ctx_ids = s->history; m->q4e_ctx_len = s->history_len;
            m->aux_cache = s->aux_cache;
            { const bool _ft = getenv("LLM_FWD_TIME") != NULL; struct timespec _fa, _fb;
              if (_ft) clock_gettime(CLOCK_MONOTONIC, &_fa);
              ok = forward_one_token(m, s, t.embed_w, t.embed_s, t.embed_b, current_id, s->cache_offset,
                                    s->k_cache, s->v_cache, s->has_cache, &h);
              if (_ft) { clock_gettime(CLOCK_MONOTONIC, &_fb);
                fprintf(stderr, "[FWD_TIME] step=%d fwd=%.2f ms\n", step,
                        (double)(_fb.tv_sec-_fa.tv_sec)*1e3 + (double)(_fb.tv_nsec-_fa.tv_nsec)*1e-6); } }
            m->q4e_ctx_ids = NULL; m->q4e_ctx_len = 0;
            m->aux_cache = NULL;
        }
        if (dp_dec && ok) {
            const int nl = m->cfg.n_layers;
            size_t nrt = 1 + 2 * (size_t)nl;
            mlx_array* rts = (mlx_array*)malloc(nrt * sizeof(mlx_array));
            if (rts != NULL) {
                rts[0] = h;
                for (int l = 0; l < nl; l++) { rts[1 + l] = s->k_cache[l]; rts[1 + (size_t)nl + l] = s->v_cache[l]; }
                llm_mlx_dp_count(rts, (int)nrt);
                free(rts);
            }
            clock_gettime(CLOCK_MONOTONIC, &dpt1);
            dp_build_s += (double)(dpt1.tv_sec - dpt0.tv_sec) + (double)(dpt1.tv_nsec - dpt0.tv_nsec) * 1e-9;
        }
        /* Debug parita' Fase B (default OFF): dump h fp32 di OGNI step. */
        {
            const char* dh = getenv("LLM_DUMP_H");
            if (ok && dh) {
                mlx_array hf = mlx_array_new();
                if (mlx_astype(&hf, h, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(hf) == 0) {
                    const float* d = mlx_array_data_float32(hf);
                    char p[512];
                    snprintf(p, sizeof(p), "%s/step%05d.bin", dh, step);
                    FILE* f = fopen(p, "wb");
                    if (f && d) {
                        size_t nh = (size_t)m->cfg.hidden_size;
                        fwrite(d, sizeof(float), nh, f);
                    }
                    if (f) fclose(f);
                }
                mlx_array_free(hf);
            }
        }
        int32_t next_id = 0;
        { const bool _ft = getenv("LLM_FWD_TIME") != NULL; struct timespec _fa, _fb;
          if (_ft) clock_gettime(CLOCK_MONOTONIC, &_fa);
          if (ok) ok = final_step(m, h, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                llm_mlx_session_sampling(s), &next_id);
          if (_ft) { clock_gettime(CLOCK_MONOTONIC, &_fb);
            fprintf(stderr, "[FWD_TIME] step=%d final=%.2f ms\n", step,
                    (double)(_fb.tv_sec-_fa.tv_sec)*1e3 + (double)(_fb.tv_nsec-_fa.tv_nsec)*1e-6); } }
        if (dp_dec) {
            clock_gettime(CLOCK_MONOTONIC, &dpt2);
            dp_eval_s += (double)(dpt2.tv_sec - dpt1.tv_sec) + (double)(dpt2.tv_nsec - dpt1.tv_nsec) * 1e-9;
        }
        if (ok) {
            mlx_array_free(s->pending_h);
            s->pending_h = mlx_array_new();
            mlx_array_set(&s->pending_h, h);
        }
        mlx_array_free(h);
        s->cache_offset++;
        if (!ok) break;
        current_id = next_id;
    }
    s->pending_next_id = current_id;
    if (dp_dec) {
        llm_mlx_dp_stage(dp_build_s, dp_eval_s);
        llm_mlx_dp_report("decode/token", generated_count > 0 ? generated_count : 1,
                          m->cfg.n_layers, dp_build_s + dp_eval_s, 12);
    }

    free_output_tensors(&t);
    LLM_DA_REPORT();
    return ok ? generated_count : -1;
}

/* Fase 3 continuous batching (2026-09-11): genera fino a max_new_tokens per N sessioni (stesso
 * modello) in UN forward BATCHED per step (attention per-sessione + FFN/proiezioni batched =
 * N token -> 1 GEMM). out_ids: n*max_new, row-major per sessione (sessione i -> out_ids[i*max_new + t]).
 * Le sessioni devono condividere il modello e avere has_pending. Ritorna 0 su successo, -1 su errore.
 * LLM_BATCH_DECODE=1 (default OFF): il chiamante decide. */
/* S6 — BATCHING SPECULATIVO multi-sessione (2,30x a N=4 nel probe, ora nel motore).
 *
 * Per ogni round: il "primary" di ogni sessione e' il suo pending_next_id (greedy gia' calcolato);
 * si drafteggia MTP per N sessioni (K=1 -> `llm_ai_mlx_mtp_draft1_multi`, un solo forward
 * batchato; K>1 -> `llm_ai_mlx_mtp_draftk` per sessione); si verifica TUTTO con UN forward batch
 * (`llm_ai_mlx_session_verify_draft_batch_multi`); si emettono i draft accettati. Il vantaggio e'
 * che i pesi del trunk vengono letti una volta per N*(K+1) token invece di uno per sessione.
 *
 * Contratto: come `llm_ai_mlx_session_generate_batch` (out_ids[i*max_new_tokens + k], ritorno 0).
 * Eleggibilita': head MTP presente, tutte le sessioni con pending, tutte GREEDY (l'MTP non ha
 * senso col sampling). Se non eleggibile ritorna -3 (il chiamante ricade sul batch normale).
 * Rispetta `LLM_MTP_DEPTH` (default 1) e `LLM_BATCH_SPEC=0` (kill-switch). */
int llm_ai_mlx_session_generate_batch_spec(llm_ai_mlx_session_handle* handles, int n,
                                           int max_new_tokens, int32_t eos_token_id,
                                           int32_t* out_ids) {
    if (handles == NULL || n <= 0 || max_new_tokens <= 0 || out_ids == NULL || n > 32) return -1;
    LlmMlxSession* s0 = (LlmMlxSession*)handles[0];
    if (s0 == NULL || s0->model == NULL) return -1;
    LlmMlxModel* m = s0->model;
    { const char* ks = llm_ai_mlx_opt_env("LLM_BATCH_SPEC"); if (ks != NULL && strcmp(ks, "0") == 0) return -3; }
    if (llm_mlx_mtp_get(m) == NULL) return -3;
    for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        if (s == NULL || s->model != m || !s->has_pending) return -1;
        if (llm_mlx_session_sampling(s).active) return -3;   /* MTP: solo greedy */
    }
    int K = 1; { const char* d = llm_ai_mlx_opt_env("LLM_MTP_DEPTH"); if (d) { int v = atoi(d); if (v >= 1 && v <= 8) K = v; } }
    const int maxd = K + 1;
    int32_t* blocks = (int32_t*)malloc(sizeof(int32_t) * (size_t)n * maxd);
    int* dlen = (int*)calloc((size_t)n, sizeof(int));
    int* gen = (int*)calloc((size_t)n, sizeof(int));
    int* done = (int*)calloc((size_t)n, sizeof(int));
    int32_t* acc = (int32_t*)malloc(sizeof(int32_t) * (size_t)n);
    int32_t* nxt = (int32_t*)malloc(sizeof(int32_t) * (size_t)n);
    bool ok = (blocks && dlen && gen && done && acc && nxt);
    while (ok) {
        int any = 0;
        for (int i = 0; i < n; i++) {
            LlmMlxSession* s = (LlmMlxSession*)handles[i];
            dlen[i] = 0;
            if (done[i] || gen[i] >= max_new_tokens) continue;
            int32_t p = -1;
            if (llm_ai_mlx_session_pending_id(handles[i], &p) != 0) { ok = false; break; }
            blocks[(size_t)i * maxd] = p;
            out_ids[(size_t)i * max_new_tokens + gen[i]] = p; gen[i]++;
            if (eos_token_id >= 0 && p == eos_token_id) { done[i] = 1; continue; }
            if (gen[i] >= max_new_tokens) continue;
            dlen[i] = maxd; any = 1;
        }
        if (!ok || !any) break;
        if (K == 1) {
            int32_t bd[32];
            if (llm_ai_mlx_mtp_draft1_multi(handles, n, bd) != 0)
                for (int i = 0; i < n; i++) bd[i] = blocks[(size_t)i * maxd];
            for (int i = 0; i < n; i++) if (dlen[i] > 0) blocks[(size_t)i * maxd + 1] = bd[i];
        } else {
            for (int i = 0; i < n; i++) {
                if (dlen[i] == 0) continue;
                int32_t dr[8];
                if (llm_ai_mlx_mtp_draftk(handles[i], K, dr, NULL) != 0)
                    for (int k = 0; k < K; k++) dr[k] = blocks[(size_t)i * maxd];
                for (int k = 0; k < K; k++) blocks[(size_t)i * maxd + 1 + k] = dr[k];
            }
        }
        if (llm_ai_mlx_session_verify_draft_batch_multi(handles, n, blocks, dlen, maxd, acc, nxt) != 0) {
            ok = false; break;
        }
        for (int i = 0; i < n; i++) {
            if (dlen[i] == 0) continue;
            for (int k = 1; k < acc[i] && gen[i] < max_new_tokens; k++) {
                int32_t tk = blocks[(size_t)i * maxd + k];
                out_ids[(size_t)i * max_new_tokens + gen[i]] = tk; gen[i]++;
                if (eos_token_id >= 0 && tk == eos_token_id) { done[i] = 1; break; }
            }
        }
    }
    free(blocks); free(dlen); free(gen); free(done); free(acc); free(nxt);
    return ok ? 0 : -1;
}

int llm_ai_mlx_session_generate_batch(llm_ai_mlx_session_handle* handles, int n,
                                      int max_new_tokens, int32_t eos_token_id, int32_t* out_ids) {
    if (handles == NULL || n <= 0 || max_new_tokens <= 0 || out_ids == NULL) return -1;
    LlmMlxSession* s0 = (LlmMlxSession*)handles[0];
    if (s0 == NULL) return -1;
    LlmMlxModel* m = s0->model;
    { int _g = llm_mlx_active_guard(m); if (_g != 0) return _g; }
    for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        if (s == NULL || s->model != m || !s->has_pending) return -1;
    }
    /* S6 — percorso SPECULATIVO multi-sessione quando eleggibile (greedy + head MTP):
     * un draft batchato + un verify batch per round, N*(K+1) token per lettura dei pesi.
     * Con max_new_tokens==1 non conviene (round singolo) -> batch normale. */
    const char* bspec = llm_ai_mlx_opt_env("LLM_BATCH_SPEC");
    if (bspec != NULL && strcmp(bspec, "0") != 0 && max_new_tokens > 1 && m->deleg == NULL) {
        int rc = llm_ai_mlx_session_generate_batch_spec(handles, n, max_new_tokens,
                                                       eos_token_id, out_ids);
        if (rc == 0) return 0;
        if (rc == -1) return -1;   /* errore reale; -3 = non eleggibile -> batch normale */
    }
    /* Modello DELEGATO (DS4/DS4.1): batch sequenziale (una sessione per volta, stesso engine;
     * layout out_ids[i*max_new + k]). */
    if (m->deleg) {
        for (int i = 0; i < n; i++) {
            LlmMlxSession* s = (LlmMlxSession*)handles[i];
            int nc = m->deleg->generate_ids
                     ? m->deleg->generate_ids(s->deleg_sess, max_new_tokens, eos_token_id,
                                              out_ids + (size_t)i * max_new_tokens)
                     : -1;
            if (nc < 0) return -1;
            if (nc > 0) session_history_append(s, out_ids + (size_t)i * max_new_tokens, nc);
        }
        return 0;
    }
    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;

    int32_t* cur = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    int* gen = (int*)calloc((size_t)n, sizeof(int));
    int32_t* toks = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    LlmMlxSession** act = (LlmMlxSession**)malloc((size_t)n * sizeof(LlmMlxSession*));
    bool ok = (cur != NULL && gen != NULL && toks != NULL && act != NULL);
    if (ok) for (int i = 0; i < n; i++) cur[i] = ((LlmMlxSession*)handles[i])->pending_next_id;

    for (int step = 0; ok && step < max_new_tokens; step++) {
        int na = 0;
        for (int i = 0; i < n; i++) {
            if (gen[i] >= max_new_tokens) continue;
            if (eos_token_id >= 0 && cur[i] == eos_token_id) continue;
            toks[na] = cur[i];
            act[na] = (LlmMlxSession*)handles[i];
            na++;
        }
        if (na == 0) break;
        mlx_array out_h = mlx_array_new();
        ok = forward_one_token_batch(m, act, na, toks, t.embed_w, t.embed_s, t.embed_b, &out_h);
        for (int a = 0; a < na && ok; a++) {
            LlmMlxSession* s = act[a];
            mlx_array h_a = mlx_array_new();
            const int st[3] = {0, a, 0};
            const int sp[3] = {1, a + 1, m->cfg.hidden_size};
            const int ss[3] = {1, 1, 1};
            ok = mlx_slice(&h_a, out_h, st, 3, sp, 3, ss, 3, m->stream) == 0;
            int32_t next = -1;
            if (ok) ok = final_step(m, h_a, t.final_norm_w, t.out_w, t.out_s, t.out_b,
                                    llm_mlx_session_sampling(s), &next);
            if (ok) {
                int idx = -1;
                for (int q = 0; q < n; q++)
                    if (handles[q] == (llm_ai_mlx_session_handle)s) { idx = q; break; }
                if (idx >= 0) {
                    out_ids[(size_t)idx * max_new_tokens + gen[idx]] = cur[idx];
                    gen[idx]++;
                    cur[idx] = next;
                }
                s->pending_next_id = next;
                s->cache_offset++; /* come session_generate: il token appena forwardato e' in cache */
            }
            mlx_array_free(h_a);
        }
        mlx_array_free(out_h);
    }

    free(cur); free(gen); free(toks); free(act);
    free_output_tensors(&t);
    return ok ? 0 : -1;
}

int llm_ai_mlx_session_set_sampling(llm_ai_mlx_session_handle handle,
                                     float temperature, float top_p, int top_k,
                                     int64_t seed) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    s->sampling_set = true;
    s->sampling_temperature = temperature;
    s->sampling_top_p = top_p;
    s->sampling_top_k = top_k;
    /* min-p dal registro (`minp`, default 0.05). Rende VIVA l'opzione (prima mai letta);
     * override per-sessione via llm_ai_mlx_session_set_min_p (additiva, dopo set_sampling). */
    { double mp = 0.0; if (llm_ai_mlx_opt_get_float("minp", &mp) != 0) mp = 0.0; s->sampling_min_p = (float)mp; }
    s->sampling_seed = seed;
    if (s->sampling_repetition_penalty <= 0.0f) s->sampling_repetition_penalty = 1.0f;  /* default off */
        return 0;
}

/* Repetition penalty (>1.0 penalizza i token gia' visti; 1.0 = off). Additiva rispetto a
 * set_sampling (non ne cambia la firma). Usata dal server per `repetition_penalty`. */
int llm_ai_mlx_session_set_min_p(llm_ai_mlx_session_handle handle, float min_p) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    s->sampling_min_p = min_p;
    return 0;
}

int llm_ai_mlx_session_set_repetition_penalty(llm_ai_mlx_session_handle handle, float penalty) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    s->sampling_repetition_penalty = penalty;
    return 0;
}

/* §I: frequency_penalty / presence_penalty (OpenAI-style) SEPARATE da repetition_penalty.
 * >0 penalizza i token gia' visti: `logits[t] -= fp*count(t)` e `-= pp` (se count>0). 0 = off.
 * Si applicano nel path SAMPLING (temperature>0), come repetition_penalty. */
int llm_ai_mlx_session_set_frequency_penalty(llm_ai_mlx_session_handle handle, float penalty) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    s->sampling_frequency_penalty = penalty > 0.0f ? penalty : 0.0f;
    return 0;
}
int llm_ai_mlx_session_set_presence_penalty(llm_ai_mlx_session_handle handle, float penalty) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    s->sampling_presence_penalty = penalty > 0.0f ? penalty : 0.0f;
    return 0;
}

int llm_ai_mlx_session_set_prefill_chunk(llm_ai_mlx_session_handle handle,
                                          int chunk_size) {
    if (handle == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    /* 0 = default adattivo per architettura (denso 2048 / MoE 4096, misurato). */
    s->prefill_chunk_size = chunk_size > 0
        ? chunk_size
        : (s->model->cfg.n_experts > 0 ? 4096 : 2048);
    return 0;
}
