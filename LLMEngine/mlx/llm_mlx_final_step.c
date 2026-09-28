/* ============================================================================
 * Dominio sampling/final_step: llm_mlx_logit_softcap, e9 top-k (e9_worse/swap/cmp/select), llm_mlx_final_norm, final_step_sampling_dev, final_step, final_step_greedy_dev, final_step_constrained.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* FASE 1b.4.4 (Gemma4): soft-cap sui logit finali -- tanh(x/softcap)*softcap,
 * verificato identico a mlx_lm.models.gemma4_text.logit_softcap. Monotona (non
 * cambia quale indice vince l'argmax rispetto ai logit grezzi, in aritmetica
 * esatta), applicata comunque per fedeltà completa all'implementazione reale --
 * su modelli senza soft-cap (final_logit_softcap == 0, default per ogni
 * architettura non-Gemma4) è un no-op: la funzione non viene nemmeno chiamata. */
bool llm_mlx_logit_softcap(mlx_array* res, mlx_array logits, float softcap, mlx_stream stream) {
    mlx_array cap = mlx_array_new_float32(softcap);
    mlx_array scaled = mlx_array_new(), t = mlx_array_new();
    bool ok = mlx_divide(&scaled, logits, cap, stream) == 0;
    if (ok) ok = mlx_tanh(&t, scaled, stream) == 0;
    if (ok) ok = mlx_multiply(res, t, cap, stream) == 0;
    mlx_array_free(cap); mlx_array_free(scaled); mlx_array_free(t);
    return ok;
}

/* RMSNorm finale + proiezione lm_head (o embed_tokens se
 * tie_word_embeddings). Selezione del token:
 *   - sampling==false (o temperature<=0): argmax greedy (comportamento storico)
 *   - sampling attivo: temperature + top_k + top_p in host memory, poi
 *     campionamento dalla distribuzione (seed deterministico riproducibile).
 * h: [1,1,hidden]. */

/* E9: top-k stabile con parita' ESATTA rispetto al vecchio selection-sort O(k·vocab).
 * Il nuovo path O(vocab·log k) (min-heap) produce lo STESSO ordine: value desc, e su tie
 * index asc (earlier-index-wins, come il selection-sort con `>`). "better" e' un ordine
 * totale stretto (indici unici), quindi qsort sul min-heap dei k migliori da' la stessa
 * permutazione unica del selection-sort -> parita' di order[] e topk_min. */
static bool e9_worse(const e9_topk_t a, const e9_topk_t b) {
    if (a.value != b.value) return a.value < b.value;
    return a.index > b.index;
}
static void e9_swap(e9_topk_t* a, e9_topk_t* b) { e9_topk_t t = *a; *a = *b; *b = t; }
static int e9_topk_cmp(const void* av, const void* bv) {
    const e9_topk_t* a = (const e9_topk_t*)av;
    const e9_topk_t* b = (const e9_topk_t*)bv;
    if (a->value != b->value) return a->value > b->value ? -1 : 1;
    return a->index < b->index ? -1 : (a->index > b->index ? 1 : 0);
}
/* Seleziona i `kn` valori migliori di scaled[0..vocab) in order[] (value desc, tie index asc):
 * min-heap O(vocab·log kn) + qsort dei soli candidati. `e9_worse` e' un ordine totale stretto
 * (indici unici), quindi la permutazione e' unica e il top-K ordinato e' il PREFISSO del top-V
 * ordinato -- proprieta' su cui si appoggia la crescita di K del ramo top_p. Restituisce il
 * numero di candidati scritti (min(kn, vocab)). */
static int e9_topk_select(int* order, const float* scaled, size_t vocab, int kn) {
    e9_topk_t* heap = (e9_topk_t*)malloc((size_t)(kn + 1) * sizeof(e9_topk_t)); /* 1-indexed */
    int heapsz = 0;
    /* fase 1: riempi il min-heap di size kn con i primi kn (root = peggiore). */
    for (size_t i = 0; i < vocab && heapsz < kn; i++) {
        heapsz++;
        heap[heapsz].value = scaled[i];
        heap[heapsz].index = (int)i;
        int c = heapsz;
        while (c > 1 && e9_worse(heap[c], heap[c/2])) { e9_swap(&heap[c], &heap[c/2]); c /= 2; }
    }
    /* fase 2: per ogni token restante, sostituisci la radice (peggiore) se migliore. */
    for (size_t i = (size_t)kn; i < vocab; i++) {
        e9_topk_t it = { scaled[i], (int)i };
        if (e9_worse(heap[1], it)) {
            heap[1] = it;
            int c = 1;
            for (;;) {
                int l = 2*c, r = 2*c+1, m = c;
                if (l <= heapsz && e9_worse(heap[l], heap[m])) m = l;
                if (r <= heapsz && e9_worse(heap[r], heap[m])) m = r;
                if (m == c) break;
                e9_swap(&heap[c], &heap[m]); c = m;
            }
        }
    }
    qsort(heap + 1, (size_t)heapsz, sizeof(e9_topk_t), e9_topk_cmp);
    for (int i = 0; i < heapsz; i++) order[i] = heap[i + 1].index;
    free(heap);
    return heapsz;
}

/* Qwen4-Exp NON ha una RMSNorm finale semplice: Qwen4ExpTextModel.__call__ termina con
 * `self.hyper_connection_mixer(hidden_states)` (GatedResidual, use_combine=False) che
 * riduce l'hidden state da hc_width a hidden_size -- il risultato va DIRETTAMENTE a
 * lm_head, senza un'altra norma dopo (verificato: TextModel.__call__ chiama
 * `self.lm_head(hidden)` subito dopo `self.model(...)`, nessun `norm(hidden)` in mezzo).
 * `norm_w` qui e' gia' risolto da load_output_tensors a
 * `hyper_connection_mixer.hc_norm.weight` (fallback qwen4_exp, vedi sopra) -- e'
 * esattamente il tensore che serve come PRIMO argomento di llm_mlx_gated_residual
 * (la sua stessa ZeroCenteredRMSNorm interna). Ogni altro chiamante di final_step*
 * passa da qui invece di rms_norm_cfg diretto, cosi' il branch qwen4_exp e' UNO
 * solo invece di duplicato in ogni variante. */
bool llm_mlx_final_norm(LlmMlxModel* m, mlx_array h, mlx_array norm_w, mlx_array* out_normed) {
    if (strcmp(m->cfg.model_type, "qwen4_exp") != 0)
        return rms_norm_cfg(out_normed, h, norm_w, m->cfg.rms_norm_eps, m->stream);

    mlx_array down_w = mlx_array_new(), down_sc = mlx_array_new(), down_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_sc = mlx_array_new(), up_b = mlx_array_new();
    bool ok = multishard_get_triple(&m->weights, "language_model.model.hyper_connection_mixer.input_mix_weight_down", &down_w, &down_sc, &down_b);
    ok = ok && multishard_get_triple(&m->weights, "language_model.model.hyper_connection_mixer.input_mix_weight_up", &up_w, &up_sc, &up_b);
    if (!ok) fprintf(stderr, "llm_mlx_final_norm: hyper_connection_mixer.input_mix_weight_{down,up} lookup fallito\n");
    mlx_array no_inject = mlx_array_new(); /* ctx==NULL -> use_combine=false (solo mixed) */
    if (ok) ok = qwen4_exp_gated_residual_q(m, h, norm_w, down_w, down_sc, down_b, up_w, up_sc, up_b,
                                             no_inject, no_inject, no_inject, out_normed, NULL, NULL);
    mlx_array_free(down_w); mlx_array_free(down_sc); mlx_array_free(down_b);
    mlx_array_free(up_w); mlx_array_free(up_sc); mlx_array_free(up_b);
    mlx_array_free(no_inject);
    return ok;
}

/* S1 (Todo/(open) - MLX-Decode-Composizione §11): sampling interamente ON-DEVICE.
 * Supporta temperature + top_k + top_p; le penalty (repetition/frequency/presence) NON sono
 * supportate in v1 -> ritorna false e il chiamante ricade sul path host (mai un risultato
 * silenzioso diverso). RNG: `mlx_random_key(seed ^ (hist_n+1)*cost)` -> sequenze diverse dal
 * path host con il flag ON (opt-in documentato). `capped`: logits post-softcap, [vocab] o
 * [1,vocab]; il risultato è un solo intero letto in host (niente vettore del vocabolario). */
static bool final_step_sampling_dev(LlmMlxModel* m, mlx_array capped,
                                    const llm_mlx_sampling_t* sp, int32_t* out_id) {
    static int s_dbg = -1;
    if (s_dbg < 0) s_dbg = (getenv("LLM_SAMPLER_ONDEVICE_DBG") != NULL) ? 1 : 0;
    if (sp == NULL || out_id == NULL) return false;
    if (sp->repetition_penalty > 1.0f || sp->frequency_penalty != 0.0f || sp->presence_penalty != 0.0f) {
        if (s_dbg) fprintf(stderr, "[sampler-ondevice] penalty attive -> path host\n");
        return false;   /* penalty: path host */
    }
    if (sp->temperature <= 0.0f) return false;
    mlx_stream s = m->stream;
    const size_t vocab = mlx_array_size(capped);
    if (vocab == 0) return false;
    bool ok = true;
    const char* stp = "logits_f32";
    mlx_array x = mlx_array_new();
    mlx_array t = mlx_array_new_float(sp->temperature);
    mlx_array scaled = mlx_array_new();
    ok = llm_mlx_logits_f32(&x, capped, s);   /* bool: true = successo */
    if (ok) { stp = "divide"; ok = mlx_divide(&scaled, x, t, s) == 0; }
    mlx_array_free(x); mlx_array_free(t);
    /* Appiattisce a 1-D [vocab]: i logit arrivano [1,1,vocab] e le slice/argsort/cumsum
     * sotto usano indici 1-D (stessa convenzione del path host, che legge flat). */
    {
        mlx_array flat = mlx_array_new();
        const int fsh[1] = {(int)vocab};
        if (ok) { stp = "reshape_flat"; ok = mlx_reshape(&flat, scaled, fsh, 1, s) == 0; }
        if (ok) { mlx_array_set(&scaled, flat); }
        mlx_array_free(flat);
    }

    const int k = (sp->top_k > 0 && sp->top_k < (int)vocab) ? sp->top_k : (int)vocab;
    mlx_array vals = mlx_array_new(), idx = mlx_array_new();
    bool has_idx = false;
    if (ok && k < (int)vocab) {
        mlx_array order = mlx_array_new();
        if (ok) { stp = "argsort"; ok = mlx_argsort(&order, scaled, s) == 0; }
        if (ok) {
            const int st[1] = {(int)vocab - k}, en[1] = {(int)vocab}, stp1[1] = {1};
            stp = "slice";
            ok = mlx_slice(&idx, order, st, 1, en, 1, stp1, 1, s) == 0;
        }
        if (ok) { stp = "take_along"; ok = mlx_take_along_axis(&vals, scaled, idx, 0, s) == 0; }
        has_idx = ok;
        mlx_array_free(order);
    } else if (ok) {
        stp = "set_vals";
        ok = mlx_array_set(&vals, scaled) == 0;
    }
    mlx_array_free(scaled);

    mlx_array probs = mlx_array_new();
    if (ok) { stp = "softmax"; ok = mlx_softmax_axis(&probs, vals, 0, /*precise=*/true, s) == 0; }
    if (ok && sp->top_p > 0.0f && sp->top_p < 1.0f) {
        /* Nucleo (top_p) sul vettore ASCENDENTE: si tiene il token i se la somma delle
         * probabilità dei token SOPRA di lui (cum_rev - p_i) è < top_p. La maschera si applica
         * ai LOGIT (-inf), non alle probs: `random_categorical` vuole logits. */
        mlx_array cum = mlx_array_new(), above = mlx_array_new(), thr = mlx_array_new_float(sp->top_p);
        mlx_array mask = mlx_array_new(), neginf = mlx_array_new_float(-INFINITY), masked = mlx_array_new();
        stp = "top_p";
        ok = mlx_cumsum(&cum, probs, 0, /*reverse=*/true, /*inclusive=*/true, s) == 0 &&
             mlx_subtract(&above, cum, probs, s) == 0 &&
             mlx_less(&mask, above, thr, s) == 0 &&
             mlx_where(&masked, mask, vals, neginf, s) == 0;
        if (ok) mlx_array_set(&vals, masked);
        mlx_array_free(cum); mlx_array_free(above); mlx_array_free(thr);
        mlx_array_free(mask); mlx_array_free(neginf); mlx_array_free(masked);
    }
    mlx_array_free(probs);

    mlx_array key = mlx_array_new(), pick = mlx_array_new(), id = mlx_array_new();
    if (ok) {
        uint64_t seed = (uint64_t)(sp->seed == 0 ? (int64_t)0x9E3779B97F4A7C15ULL : sp->seed);
        seed ^= (uint64_t)(sp->hist_n + 1) * 0x9E3779B97F4A7C15ULL;
        stp = "random_key";
        ok = mlx_random_key(&key, seed) == 0;
    }
    if (ok) { stp = "categorical"; ok = mlx_random_categorical(&pick, vals, 0, key, s) == 0; }
    mlx_array_free(vals); mlx_array_free(key);
    if (ok && has_idx) {
        mlx_array pick1 = mlx_array_new();
        const int p1[1] = {1};
        stp = "reshape_pick";
        ok = mlx_reshape(&pick1, pick, p1, 1, s) == 0;
        if (ok) { stp = "take"; ok = mlx_take(&id, idx, pick1, s) == 0; }
        mlx_array_free(pick1);
    } else if (ok) {
        ok = mlx_array_set(&id, pick) == 0;
    }
    mlx_array_free(pick); mlx_array_free(idx);
    mlx_array id32 = mlx_array_new();
    if (ok) { stp = "astype"; ok = mlx_astype(&id32, id, MLX_INT32, s) == 0; }
    mlx_array_free(id);
    if (ok) { stp = "eval"; ok = mlx_array_eval(id32) == 0; }
    if (ok) {
        const int32_t* d = mlx_array_data_int32(id32);
        ok = d != NULL && mlx_array_size(id32) == 1;
        if (ok) *out_id = d[0];
        /* Branch-first: una riga UNA volta per processo che prova che il ramo on-device e'
         * davvero in uso (senza, un output identico non distingue on-device da fallback host).
         * LLM_SAMPLER_ONDEVICE_DBG=1, zero overhead quando non impostata. */
        static int s_pr = 0;
        if (s_dbg && !s_pr) { s_pr = 1; fprintf(stderr, "[sampler-ondevice] ATTIVO (campionamento sul device)\n"); }
    } else if (s_dbg) {
        fprintf(stderr, "[sampler-ondevice] FALLITA a: %s -> path host\n", stp);
    }
    mlx_array_free(id32);
    return ok;
}

bool final_step(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                       mlx_array out_w, mlx_array out_s, mlx_array out_b,
                       llm_mlx_sampling_t sampling, int32_t* out_id) {
         mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_final_norm(m, h, norm_w, &normed);
    mlx_array logits = mlx_array_new();
    /* Ternary-Bonsai: lm_head e' elencato -> si ruota il suo ingresso. */
    if (ok) ok = llm_bonsai_rotate(&normed, m, -1, "lm_head.weight", false);
    if (ok) ok = quantized_linear_cfg(&logits, normed, out_w, out_s, out_b, m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    mlx_array capped = mlx_array_new();
    if (ok && m->cfg.final_logit_softcap > 0.0f) {
        ok = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
    } else if (ok) {
        ok = mlx_array_set(&capped, logits) == 0;
    }

    /* F3 qwen4_exp: dump dei LOGITS (f32, raw) in un file per il confronto con l'oracolo.
     * LLM_QWEN4EXP_LOGITS=<path>. LLM_QWEN4EXP_LOGITS_DIR=<dir> scrive una chiamata per
     * file numerato (`logits_%04d.bin`, nell'ordine di final_step: prefill + ogni decode),
     * cosi' si confronta il logit di OGNI passo di decode con la reference. */
    if (ok && (llm_env_isset("LLM_QWEN4EXP_LOGITS") || llm_env_isset("LLM_QWEN4EXP_LOGITS_DIR"))) {
        mlx_array lf = mlx_array_new();
        if (mlx_astype(&lf, capped, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(lf) == 0) {
            const float* ld = mlx_array_data_float32(lf);
            size_t n = mlx_array_size(lf);
            const char* path = getenv("LLM_QWEN4EXP_LOGITS");
            const char* dir = getenv("LLM_QWEN4EXP_LOGITS_DIR");
            char pbuf[1200];
            if (dir != NULL) {
                static int q4_logit_step = 0;
                snprintf(pbuf, sizeof(pbuf), "%s/logits_%04d.bin", dir, q4_logit_step++);
                path = pbuf;
            }
            FILE* fp = path ? fopen(path, "wb") : NULL;
            if (fp) { fwrite(ld, sizeof(float), n, fp); fclose(fp); }
        }
        mlx_array_free(lf);
    }

    /* S1 (todo MLX-Decode-Composizione §11, opt-in LLM_SAMPLER_ONDEVICE=1): campionamento
     * ON-DEVICE (niente lettura del vocabolario in host). Se le penalty sono attive o il
     * device fallisce, si prosegue sul path host sotto (fallback esatto). */
    if (ok && sampling.active && sampling.temperature > 0.0f && llm_mlx_sampler_ondevice_enabled()) {
        int32_t dev_id = -1;
        if (final_step_sampling_dev(m, capped, &sampling, &dev_id)) {
            *out_id = dev_id;
            mlx_array_free(normed);
            mlx_array_free(logits);
            mlx_array_free(capped);
            return true;
        }
    }

    /* Path greedy: argmax sul device (come storicamente). */
    if (ok && (!sampling.active || sampling.temperature <= 0.0f)) {
        mlx_array am = mlx_array_new();
        if (ok) ok = mlx_argmax_axis(&am, capped, -1, /*keepdims=*/false, m->stream) == 0;
    if (ok) ok = mlx_array_eval(am) == 0;
        if (ok) {
            const uint32_t* d = mlx_array_data_uint32(am);
            ok = (d != NULL && mlx_array_size(am) == 1);
            if (ok) *out_id = (int32_t)d[0];
        }
        if (!ok) fprintf(stderr, "final_step: fallito\n");
        mlx_array_free(normed);
        mlx_array_free(logits);
        mlx_array_free(capped);
        mlx_array_free(am);
        return ok;
    }

    /* Path sampling: leggi i logit in host (stesso pattern di
     * final_step_constrained) e campiona. */
    mlx_array capped32 = mlx_array_new();
    if (ok) ok = llm_mlx_logits_f32(&capped32, capped, m->stream);
    if (ok) ok = mlx_array_eval(capped32) == 0;
    const float* data = ok ? mlx_array_data_float32(capped32) : NULL;
    size_t vocab = ok ? mlx_array_size(capped32) : 0;
    ok = ok && (data != NULL && vocab > 0);

    if (ok) {
        float temperature = sampling.temperature > 0.0f ? sampling.temperature : 1.0f;
        /* 0. repetition penalty (CTRL): penalizza i token gia' visti (prompt+generati), sui
         * logit GREZZI prima della temperatura. >0 -> /penalty; <0 -> *penalty. */
        float* scaled = (float*)malloc(vocab * sizeof(float));
        for (size_t i = 0; i < vocab; i++) scaled[i] = data[i];
        if (sampling.repetition_penalty > 1.0f && sampling.hist != NULL) {
            for (int j = 0; j < sampling.hist_n; j++) {
                int t = sampling.hist[j];
                if (t >= 0 && (size_t)t < vocab)
                    scaled[t] = scaled[t] > 0.0f ? scaled[t] / sampling.repetition_penalty
                                                 : scaled[t] * sampling.repetition_penalty;
            }
        }
        /* 0b. frequency/presence penalty (OpenAI, separati da repetition_penalty): sui logit
         * GREZZI, prima della temperatura, contando le occorrenze nella history (prompt+generati):
         *   logits[t] -= frequency_penalty * count(t)
         *   logits[t] -= presence_penalty   (se count(t) > 0, UNA volta) */
        if ((sampling.frequency_penalty != 0.0f || sampling.presence_penalty != 0.0f) && sampling.hist != NULL) {
            uint32_t* cnt = (uint32_t*)calloc(vocab, sizeof(uint32_t));
            if (cnt) {
                for (int j = 0; j < sampling.hist_n; j++) {
                    int t = sampling.hist[j];
                    if (t >= 0 && (size_t)t < vocab) cnt[t]++;
                }
                for (size_t i = 0; i < vocab; i++) {
                    if (cnt[i]) {
                        scaled[i] -= sampling.frequency_penalty * (float)cnt[i];
                        scaled[i] -= sampling.presence_penalty;
                    }
                }
                free(cnt);
            }
        }

        /* 1. temperatura: dividi i logit. */
        for (size_t i = 0; i < vocab; i++) scaled[i] /= temperature;

        /* 2. top_k: tieni solo i k logit piu' alti. O(vocab·log k) via min-heap, con parita'
         * ESATTA col vecchio selection-sort O(k·vocab): stesso order[] (value desc, tie index
         * asc) e stesso topk_min. k>=vocab => nessun filtro reale (heap riempito da tutto). */
        int k = sampling.top_k > 0 ? sampling.top_k : (int)vocab;
        if (k > (int)vocab) k = (int)vocab;
        if (k < 1) k = 1;
        const bool top_k_active = sampling.top_k > 0 && sampling.top_k < (int)vocab;
        const bool top_p_active = sampling.top_p > 0.0f && sampling.top_p < 1.0f;
        /* guardia no-op: con top_k disabilitato (o >= vocab) e top_p disabilitato, la selezione
         * dei k migliori NON cambia nulla -- topk_min sarebbe il min globale, quindi la guardia
         * `scaled[i] >= topk_min` resta vera per ogni i e order[] non viene letto (e' usato solo
         * dal mask top_p). Saltare heap+qsort su 128k logit e' la differenza tra 64 e ~300 tok/s
         * (misura 2026-09-08: greedy 424 vs temperatura 64, stesso modello). Parita' bit-exact:
         * sum/scan non cambiano. LLM_SAMPLER_LEGACY=1 riesegue il percorso vecchio (misura). */
        const bool legacy = llm_mlx_sampler_legacy_enabled();
        const bool need_sel = top_k_active || top_p_active || legacy;
        /* Con SOLA top_p il nucleo si chiude quasi sempre nelle prime poche centinaia di logit:
         * selezionare crescendo (1024, poi x4, fino a vocab) costa O(vocab·log K) per tentativo
         * invece di heapify+qsort dell'intero vocabolario (~20 ms/token: 16,0 vs 23,0 tok/s su
         * Qwen3.6-27B, 9,4 vs 11,9 su gemma-4-31b, 65,7 vs 315,8 su llama, misurati 2026-09-08).
         * ESATTEZZA: la permutazione e' totale e unica, quindi il top-K ordinato e' il PREFISSO
         * del top-vocab ordinato -> se il cumulativo supera top_p entro K il cutoff coincide alla
         * cifra; se non lo supera, K cresce fino a vocab = percorso di riferimento. Crescita
         * ammessa SOLO qui perche' e' l'unico caso in cui topk_min = -INFINITY, cioe' probs/sum
         * non dipendono dalla selezione (si ricalcolano una volta sola). Negli altri rami il
         * filtro top-k agisce SUL softmax, quindi order[] serve gia' completo. */
        const bool growable = top_p_active && !top_k_active && !legacy;
        int kn = need_sel ? (growable && k > 1024 ? 1024 : k) : 0;
        int* order = need_sel ? (int*)malloc((size_t)kn * sizeof(int)) : NULL;
        uint8_t* outside = (need_sel && growable && kn < (int)vocab) ? (uint8_t*)malloc(vocab) : NULL;
        float* probs = (float*)malloc(vocab * sizeof(float));
        double sum = 0.0, max_logit = -INFINITY;
        for (size_t i = 0; i < vocab; i++) {
            if (scaled[i] > max_logit) max_logit = scaled[i];
        }
        float topk_min = -INFINITY;
        bool have_order = false;
        if (need_sel && !growable) {
            e9_topk_select(order, scaled, vocab, kn);
            topk_min = scaled[order[kn - 1]];
            have_order = true;
        }
        for (size_t i = 0; i < vocab; i++) {
            float x = (scaled[i] >= topk_min) ? scaled[i] : -INFINITY;
            probs[i] = (float)exp((double)x - max_logit);
            sum += probs[i];
        }

        /* 3. top_p (nucleo): cumulativo in ordine decrescente sui candidati finche' p >= top_p. */
        if (top_p_active) {
            double cum = 0.0;
            int scanned = 0, cutoff = kn;
            for (;;) {
                if (!have_order) { e9_topk_select(order, scaled, vocab, kn); have_order = true; }
                bool closed = false;
                for (int i = scanned; i < kn; i++) {   /* il prefisso e' gia' stato sommato */
                    cum += probs[order[i]] / sum;
                    if (cum >= sampling.top_p) { cutoff = i + 1; closed = true; break; }
                }
                scanned = kn;
                if (closed || kn >= (int)vocab) break;
                kn = kn * 4 > k ? k : kn * 4;          /* nucleo non chiuso: cresce K e riprova */
                order = (int*)realloc(order, (size_t)kn * sizeof(int));
                have_order = false;
                cutoff = kn;
            }
            for (int j = cutoff; j < kn; j++) probs[order[j]] = 0.0f;
            if (outside) {
                /* i logit FUORI dai kn candidati non sono in order[]: il percorso di riferimento
                 * (k == vocab) li azzerava con `j = cutoff..k-1`, quindi vanno azzerati qui. */
                memset(outside, 0, vocab);
                for (int i = 0; i < cutoff; i++) outside[order[i]] = 1;
                for (size_t i = 0; i < vocab; i++) if (!outside[i]) probs[i] = 0.0f;
            }
        }
        /* ricalcola la somma dopo il mask top_p */
        sum = 0.0;
        for (size_t i = 0; i < vocab; i++) sum += probs[i];

        /* 4. campiona dalla distribuzione con seed deterministico (xorshift64*).
         * BUG FIX: L'RNG NON va ri-seeded con `sampling.seed` a ogni token (darebbe lo STESSO
         * `r` per tutti i token -> stesso token -> output degenerato "Parole Parole ...").
         * Il seed e' mescolato con la posizione corrente (hist_n = token gia' visti) -> un `r`
         * diverso per token, ma riproducibile (stesso seed + stesso prompt -> stesso output). */
        uint64_t rng = (uint64_t)(sampling.seed == 0 ? (int64_t)0x9E3779B97F4A7C15ULL : sampling.seed);
        rng ^= (uint64_t)(sampling.hist_n + 1) * 0x9E3779B97F4A7C15ULL;
        rng = (rng * 0x2545F4914F6CDD1DULL);
        double r = (double)((rng = (rng * 0x2545F4914F6CDD1DULL)) >> 11) / 9007199254740992.0; /* ~ [0,1) */
        double target = r * sum;
        double acc = 0.0;
        int32_t chosen = -1;
        for (size_t i = 0; i < vocab; i++) {
            acc += probs[i];
            if (acc >= target) { chosen = (int32_t)i; break; }
        }
        if (chosen < 0) { /* fallback (arrotondamento) — ultimo non-zero */
            for (size_t i = 0; i < vocab; i++) if (probs[i] > 0.0f) chosen = (int32_t)i;
        }
        ok = (chosen >= 0);
        if (ok) *out_id = chosen;
        
        free(scaled);
        free(order);
        free(outside);
        free(probs);
    }
    if (!ok) fprintf(stderr, "final_step(sampling): fallito\n");
    mlx_array_free(normed);
    mlx_array_free(logits);
    mlx_array_free(capped);
    mlx_array_free(capped32);
    return ok;
}

/* D16bis (2026-09-06, LLM_ASYNC_DECODE): variante greedy di final_step che NON valuta
 * ne' legge l'id in host -- ritorna l'argmax come mlx_array lazy (int32). Serve a
 * costruire il grafo del passo N+1 usando l'id del passo N come mlx_array simbolico,
 * senza il round-trip host che oggi forza il loop di generate a bloccarsi ad ogni
 * token (vedi §9 "RapidMLX -- analisi codice reale" nel piano: e' lo stesso schema di
 * mlx_lm/generate.py, che produce mx.array non un int, cosi' l'embedding del token
 * successivo si costruisce senza aspettare il valore). Chiamare mlx_async_eval sul
 * risultato (insieme alle k/v cache aggiornate) per mettere in coda il command buffer
 * senza bloccare; leggere il valore host solo un passo dopo (vedi
 * llm_ai_mlx_session_generate_async). Path SOLO greedy: se serve sampling con
 * temperature/top_p/top_k il chiamante deve restare su final_step (richiede comunque
 * l'intero vettore logits in host). */
bool final_step_greedy_dev(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                                   mlx_array out_w, mlx_array out_s, mlx_array out_b,
                                   mlx_array* out_id_arr) {
    mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_final_norm(m, h, norm_w, &normed);
    mlx_array logits = mlx_array_new();
    /* Ternary-Bonsai: lm_head e' elencato -> si ruota il suo ingresso. */
    if (ok) ok = llm_bonsai_rotate(&normed, m, -1, "lm_head.weight", false);
    if (ok) ok = quantized_linear_cfg(&logits, normed, out_w, out_s, out_b, m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    mlx_array capped = mlx_array_new();
    if (ok && m->cfg.final_logit_softcap > 0.0f) {
        ok = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
    } else if (ok) {
        ok = mlx_array_set(&capped, logits) == 0;
    }
    mlx_array am = mlx_array_new();
    if (ok) ok = mlx_argmax_axis(&am, capped, -1, /*keepdims=*/false, m->stream) == 0;
    /* L'argmax e' gia' un array intero: `mlx_astype(..., INT32)` era un nodo AsType per token
     * inutile (gli id reali stanno in [0,vocab), stessi bit int32/uint32). Passato diretto:
     * 0 AsType nel passo greedy. mlx_lm fa `.item()` senza cast. Opt-out LLM_ARGMAX_CAST=1. */
    static int amcast = -1; if (amcast < 0) amcast = getenv("LLM_ARGMAX_CAST") != NULL ? 1 : 0;
    if (ok) ok = amcast ? (mlx_astype(out_id_arr, am, MLX_INT32, m->stream) == 0)
                        : (mlx_array_set(out_id_arr, am) == 0);
    if (!ok) fprintf(stderr, "final_step_greedy_dev: fallito\n");
    mlx_array_free(normed);
    mlx_array_free(logits);
    mlx_array_free(capped);
    mlx_array_free(am);
    return ok;
}

/* FASE 3: variante di final_step che restringe l'argmax a un insieme di id
 * ammessi (dispatch nativo a puntatore di funzione -- vedi
 * llm_ai_mlx_session_generate_constrained/AgentToolLoop). Nessuna operazione
 * di masking lato device: i logit vengono valutati e letti in host memory
 * (stesso identico tensore che final_step calcola comunque), poi il massimo
 * e' scelto qui solo tra allowed_ids -- evita di dover esprimere il masking
 * come grafo MLX, e il costo di eval+read e' lo stesso che final_step gia'
 * paga per il tensore completo.
 * FASE 1b.4.4 (Gemma4): il soft-cap NON è applicato qui deliberatamente --
 * `tanh(x/cap)*cap` è monotona, quindi non cambia quale `allowed_ids[i]` ha il
 * valore più alto rispetto ai logit grezzi (in aritmetica esatta): applicarlo
 * qui aggiungerebbe complessità/costo (una tanh per candidato) per zero
 * differenza nella scelta finale. */
bool final_step_constrained(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                                    mlx_array out_w, mlx_array out_s, mlx_array out_b,
                                    const int32_t* allowed_ids, int n_allowed, int32_t* out_id) {
    mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_final_norm(m, h, norm_w, &normed);
    mlx_array logits = mlx_array_new();
    mlx_array logits32 = mlx_array_new();
    /* Ternary-Bonsai: lm_head e' elencato -> si ruota il suo ingresso. */
    if (ok) ok = llm_bonsai_rotate(&normed, m, -1, "lm_head.weight", false);
    if (ok) ok = quantized_linear_cfg(&logits, normed, out_w, out_s, out_b, m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    if (ok) ok = llm_mlx_logits_f32(&logits32, logits, m->stream);
    if (ok) ok = mlx_array_eval(logits32) == 0;
    if (ok) {
        const float* data = mlx_array_data_float32(logits32);
        size_t vocab = mlx_array_size(logits32);
        ok = (data != NULL);
        if (ok) {
            int32_t best_id = -1;
            float best_val = -INFINITY;
            for (int i = 0; i < n_allowed; i++) {
                int32_t id = allowed_ids[i];
                if (id < 0 || (size_t)id >= vocab) continue;
                float v = data[id];
                if (best_id < 0 || v > best_val) { best_val = v; best_id = id; }
            }
            ok = best_id >= 0;
            if (ok) *out_id = best_id;
        }
    }
    if (!ok) fprintf(stderr, "final_step_constrained: fallito\n");
    mlx_array_free(normed);
    mlx_array_free(logits);
    mlx_array_free(logits32);
    return ok;
}
