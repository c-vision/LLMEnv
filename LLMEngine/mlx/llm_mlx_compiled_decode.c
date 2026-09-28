/* ============================================================================
 * Dominio compiled decode (RMLX10 FASE B): llm_mlx_compile_enabled, llm_mlx_compile_dense_enabled, llm_mlx_compile_qwen35_enabled, llm_mlx_compile_arch_supported, llm_mlx_static_cache_ready, llm_mlx_session_kv_ensure_cap, llm_mlx_embed_row_from_idx, llm_ai_mlx_qwen4exp_embed_tile_selftest, forward_one_token_dev, mixtral_attention_block_dyn, llm_mlx_decode_step_fn, llm_mlx_session_run_compiled_step(_arr).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_lora.h"
#include <cJSON.h>
#include "llm_mlx_bonsai_hadamard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* RMLX10 FASE B (2026-09-03): decode-step COMPILED (mlx_compile).
 *
 * Idea: a ogni token il motore eager RICOSTRUISCE il grafo lazy di 56 layer
 * (~2000+ op) e lo rilancia. mlx_compile traccia UNA volta la funzione del
 * passo e alle chiamate successive riusa il tape compilato (fuse di kernel +
 * niente re-trace/re-simplify). Perche' la cache di compile sia keyed su
 * shape (non sul valore), serve:
 *   - KV-cache STATICA (shape fissa): prerequisito della Fase A (LLM_FIXKV);
 *   - `pos` come INPUT ARRAY (mai host-int, che verrebbe baked alla trace):
 *     RoPE via mlx_fast_rope_dynamic, scrittura e maschera SDPA derivate su
 *     device da `pos`. Qui: mixtral_attention_block_dyn (variante pos-array
 *     di mixtral_attention_block in modalita' static) + closure funzionale.
 * Gate: LLM_COMPILE=1 (con LLM_FIXKV=1 attivo e model_type mixtral; la cache
 * statica e' condizione necessaria). LLM_NO_COMPILE=1 = fallback eager.
 * La closure viene creata lazy al primo step compilato e vive nella sessione
 * (cache di compile keyed per fun_id: stessa closure = cache-hit).
 * ═══════════════════════════════════════════════════════════════════════════ */
static int g_compile_state = -1;
bool llm_mlx_compile_enabled(void) {
    if (g_compile_state < 0)
        g_compile_state = (getenv("LLM_COMPILE") != NULL && getenv("LLM_NO_COMPILE") == NULL) ? 1 : 0;
    return g_compile_state == 1;
}

/* LLM_COMPILE_DENSE (default OFF): estende la closure compiled_decode ai densi GQA plain
 * oggi solo "llama". NON e' un semplice flip di gate: il blocco `mixtral_attention_block_dyn`
 * deve modellare FEDELMENTE l'attention eager di quell'arch. Per llama la RoPE a 3 fasce
 * llama3 e' ora replicata nel ramo `use_llama3_rope` del `_dyn` (vedi); senza di essa il
 * compiled diverge da eager (misurato: flip argmax da step5). Mixtral (RoPE plain) e'
 * invariato. OBBLIGO: parita' greedy token-id vs async-eager (PRODUZIONE) verificata PRIMA
 * di qualunque claim di velocita'. */
static int g_compile_dense_state = -1;
static bool llm_mlx_compile_dense_enabled(void) {
    if (g_compile_dense_state < 0)
        g_compile_dense_state = (getenv("LLM_COMPILE_DENSE") != NULL && getenv("LLM_NO_COMPILE") == NULL) ? 1 : 0;
    return g_compile_dense_state == 1;
}
static int g_compile_qwen35_state = -1;
bool llm_mlx_compile_qwen35_enabled(void) {
    if (g_compile_qwen35_state < 0)
        g_compile_qwen35_state = (getenv("LLM_COMPILE_QWEN35") != NULL) ? 1 : 0;
    return g_compile_qwen35_state == 1;
}
bool llm_mlx_compile_arch_supported(const LlmMlxModel* m) {
    if (m == NULL) return false;
    if (strcmp(m->cfg.model_type, "mixtral") == 0) return true;
    if (llm_mlx_compile_dense_enabled() && strcmp(m->cfg.model_type, "llama") == 0) return true;
    /* M1 (decode compilato a shape stabile): qwen3_5 (GDN + full-attn). Default OFF; l'aggancio
     * della closure PER-LAYER sui blocchi GDN (stato fisso) e' il passo successivo: qui si
     * verifica se il path compilato dello step intero puo' agganciare (richiede cache statiche
     * per TUTTI i layer, quindi serve anche LLM_FIXKV). */
    if (llm_mlx_compile_qwen35_enabled() && strcmp(m->cfg.model_type, "qwen3_5") == 0) return true;
    return false;
}

/* Cache statiche uniformi? (tutti i layer shape [1,nkv,cap,hd], cap>0) --
 * prerequisito per chiamare la closure compilata. */
bool llm_mlx_static_cache_ready(const LlmMlxSession* s) {
    const LlmMlxModel* m = s->model;
    if (m == NULL) return false;
    if (s->n_layers <= 0) return false;
    if (mlx_array_ndim(s->k_cache[0]) != 4 || mlx_array_ndim(s->v_cache[0]) != 4) return false;
    int cap = (int)mlx_array_shape(s->k_cache[0])[2];
    if (cap <= 0) return false;
    for (int l = 1; l < s->n_layers; l++) {
        if (mlx_array_ndim(s->k_cache[l]) != 4 || mlx_array_ndim(s->v_cache[l]) != 4) return false;
        const int* ks = mlx_array_shape(s->k_cache[l]);
        const int* vs = mlx_array_shape(s->v_cache[l]);
        if (ks == NULL || vs == NULL) return false;
        if (ks[2] != cap || vs[2] != cap) return false;
    }
    return true;
}

/* Crescita HOST della cache statica a `needed` token (pow2): padding di zeri in
 * coda su tutti i layer. Serve quando pos+1 supera la capacita' corrente: il
 * change di shape provoca UN recompile della closure (raro, crescita logaritmica),
 * ma il write/mask restano corretti (pos <= cap-1 sempre). */
static bool llm_mlx_session_kv_ensure_cap(LlmMlxSession* s, int needed) {
    if (needed <= 0) return false;
    if (!llm_mlx_static_cache_ready(s)) return false;
    int cap = (int)mlx_array_shape(s->k_cache[0])[2];
    if (cap >= needed) return true;
    int newcap = llm_mlx_kv_cap_for(needed);
    if (newcap <= cap) return false;
    for (int l = 0; l < s->n_layers; l++) {
        const int* ks = mlx_array_shape(s->k_cache[l]);
        const int* vs = mlx_array_shape(s->v_cache[l]);
        int zsh[4] = {ks[0], ks[1], newcap - cap, ks[3]};
        mlx_dtype dt = mlx_array_dtype(s->k_cache[l]);
        mlx_array kz = mlx_array_new(), vz = mlx_array_new();
        bool zok = mlx_zeros(&kz, zsh, 4, dt, s->model->stream) == 0 &&
                   mlx_zeros(&vz, zsh, 4, dt, s->model->stream) == 0;
        if (zok) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, s->k_cache[l]);
            mlx_vector_array_append_value(kv, kz);
            mlx_array kfull = mlx_array_new();
            zok = mlx_concatenate_axis(&kfull, kv, 2, s->model->stream) == 0;
            mlx_vector_array_free(kv);
            if (zok) mlx_array_set(&s->k_cache[l], kfull);
            mlx_array_free(kfull);
        }
        if (zok) {
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, s->v_cache[l]);
            mlx_vector_array_append_value(vv, vz);
            mlx_array vfull = mlx_array_new();
            zok = mlx_concatenate_axis(&vfull, vv, 2, s->model->stream) == 0;
            mlx_vector_array_free(vv);
            if (zok) mlx_array_set(&s->v_cache[l], vfull);
            mlx_array_free(vfull);
        }
        mlx_array_free(kz); mlx_array_free(vz);
        if (!zok) return false;
    }
    return true;
}

/* Embedding del token da INDICE ARRAY (per la trace: tid e' un input/tracer,
 * mai un host-int). Equivalente di embed_token ma su mlx_array [1] int32. */
bool llm_mlx_embed_row_from_idx(LlmMlxModel* m, mlx_array embed_w, mlx_array embed_s,
                                       mlx_array embed_b, mlx_array tid, mlx_array* out_x) {
    const bool quantized = (embed_s.ctx != NULL);
    int e_group = m->cfg.quant_group_size, e_bits = m->cfg.quant_bits;
    char e_mode[16];
    llm_mlx_mode_set(e_mode, sizeof(e_mode), m->cfg.quant_mode);
    if (quantized)
        llm_mlx_quant_derive(embed_w, embed_s, e_group, e_bits, e_mode, &e_group, &e_bits, e_mode, sizeof(e_mode));
    mlx_array w_row = mlx_array_new(), s_row = mlx_array_new(), b_row = mlx_array_new();
    bool g = mlx_take_axis(&w_row, embed_w, tid, 0, m->stream) == 0;
    if (g && quantized) {
        g = mlx_take_axis(&s_row, embed_s, tid, 0, m->stream) == 0;
        if (g && embed_b.ctx != NULL && mlx_array_size(embed_b) > 0)
            g = mlx_take_axis(&b_row, embed_b, tid, 0, m->stream) == 0;
    }
    mlx_array x_flat = mlx_array_new();
    bool ok = g;
    if (ok) {
        if (quantized) {
            mlx_optional_int gsz = {e_group, true};
            mlx_optional_int bts = {e_bits, true};
            mlx_optional_dtype dt = {llm_mlx_act_dt(m->cfg.bfloat16_input), true};
            ok = mlx_dequantize(&x_flat, w_row, s_row, b_row, gsz, bts, e_mode, g_mlx_empty_arr, dt, m->stream) == 0;
        } else {
            ok = mlx_astype(&x_flat, w_row, llm_mlx_act_dt(m->cfg.bfloat16_input), m->stream) == 0;
        }
    }
    if (ok) {
        const int x_shape[3] = {1, 1, m->cfg.hidden_size};
        ok = mlx_reshape(out_x, x_flat, x_shape, 3, m->stream) == 0;
    }
    if (ok && m->cfg.embed_scale > 0.0f) {
        mlx_array scl = mlx_array_new_float32(m->cfg.embed_scale);
        mlx_array scaled = mlx_array_new();
        ok = mlx_multiply(&scaled, *out_x, scl, m->stream) == 0;
        mlx_array_free(scl);
        if (ok) mlx_array_set(out_x, scaled);
        mlx_array_free(scaled);
    }
    /* Qwen4-Exp (bug 2026-09-17, AiWiki "MLX-Qwen4Exp-Attn-HyperConnection-Mix-Failed"): manca
     * qui il tile hc_count che embed_token/embed_tokens_batch gia' fanno (vedi commento sopra
     * "Qwen4-Exp: l'intero modello lavora su stati tileati hc_count volte") -- questa e' la
     * TERZA via di embedding, usata SOLO dal path LLM_ASYNC_DECODE (default ON dal 2026-09-07,
     * forward_one_token_dev), quindi il gap non si vedeva a prefill (embed_tokens_batch) ne'
     * col path sync esplicito (embed_token), solo nel decode di default: l'hidden state restava
     * a hidden_size invece di hc_count*hidden_size, e qwen4_exp_gated_residual_q lo rifiutava
     * silenziosamente al layer 0 (check `hshape[hndim-1] != hc_count*hidden`). */
    if (ok && strcmp(m->cfg.model_type, "qwen4_exp") == 0 && m->cfg.hc_count > 1) {
        mlx_array tiled = mlx_array_new();
        const int reps[3] = {1, 1, m->cfg.hc_count};
        ok = mlx_tile(&tiled, *out_x, reps, 3, m->stream) == 0;
        if (ok) mlx_array_set(out_x, tiled);
        mlx_array_free(tiled);
    }
    mlx_array_free(w_row); mlx_array_free(s_row); mlx_array_free(b_row); mlx_array_free(x_flat);
    return ok;
}

/* Self-test UNITARIO (2026-09-17) — Qwen4-Exp: llm_mlx_embed_row_from_idx deve tileare
 * hc_count volte come embed_token/embed_tokens_batch (bug "MLX-Qwen4Exp-Attn-HyperConnection-
 * Mix-Failed": mancava SOLO qui, path LLM_ASYNC_DECODE, default ON -- vedi bug doc). Nessun
 * modello/checkpoint reale (97 GB, non caricabile in sicurezza in questa sessione): pesi di
 * embedding sintetici, LlmMlxModel calloc'ato con SOLO i campi cfg toccati da questa funzione
 * (hidden_size/embed_scale/bfloat16_input/model_type/hc_count) -- zero-init e' gia' un pattern
 * usato altrove nel motore per questa stessa struct (vedi commento su llama3_freqs_valid).
 * Verifica sia la SHAPE (ultimo asse = hc_count*hidden) sia il CONTENUTO (i valori del token
 * devono ripetersi identici hc_count volte, non essere un'altra riga/rumore). Driver:
 * llm_qwen4exp_embed_tile_test [hidden] [hc_count] [vocab] [token_id]. */
int llm_ai_mlx_qwen4exp_embed_tile_selftest(int hidden, int hc_count, int vocab, int token_id,
                                            char* out, int cap) {
    if (out == NULL || cap < 32 || hidden < 1 || hc_count < 2 || vocab < 1 ||
        token_id < 0 || token_id >= vocab) return -1;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL) { snprintf(out, (size_t)cap, "FAIL: device GPU"); return -1; }
    if (mlx_set_default_device(dev) != 0) { mlx_device_free(dev); snprintf(out, (size_t)cap, "FAIL: set_default_device"); return -1; }
    mlx_stream str = mlx_stream_new();
    if (mlx_get_default_stream(&str, dev) != 0 || str.ctx == NULL) { mlx_device_free(dev); snprintf(out, (size_t)cap, "FAIL: default_stream"); return -1; }

    LlmMlxModel* m = (LlmMlxModel*)calloc(1, sizeof(LlmMlxModel));
    if (m == NULL) { mlx_device_free(dev); snprintf(out, (size_t)cap, "FAIL: calloc model"); return -1; }
    m->stream = str;
    m->cfg.hidden_size = hidden;
    m->cfg.embed_scale = 0.0f;
    m->cfg.bfloat16_input = false;
    m->cfg.hc_count = hc_count;
    snprintf(m->cfg.model_type, sizeof(m->cfg.model_type), "qwen4_exp");

    float* wd = (float*)malloc(sizeof(float) * (size_t)vocab * (size_t)hidden);
    int rc = -1;
    if (wd) {
        for (int r = 0; r < vocab; r++)
            for (int c = 0; c < hidden; c++)
                wd[r * hidden + c] = (float)(r * 100 + c);
        const int wsh[2] = {vocab, hidden};
        mlx_array embed_w = mlx_array_new_data(wd, wsh, 2, MLX_FLOAT32);
        mlx_array embed_s = mlx_array_new(), embed_b = mlx_array_new(); /* ctx==NULL -> non quantizzato */
        int32_t tid_host = token_id;
        const int tsh[1] = {1};
        mlx_array tid = mlx_array_new_data(&tid_host, tsh, 1, MLX_INT32);
        mlx_array x = mlx_array_new();
        bool ok = llm_mlx_embed_row_from_idx(m, embed_w, embed_s, embed_b, tid, &x);
        if (ok) {
            mlx_array xf = mlx_array_new();
            ok = mlx_astype(&xf, x, MLX_FLOAT32, str) == 0 && mlx_array_eval(xf) == 0;
            int nd = ok ? mlx_array_ndim(xf) : 0;
            const int* sh = ok ? mlx_array_shape(xf) : NULL;
            int last = (sh != NULL && nd > 0) ? sh[nd - 1] : -1;
            bool shape_ok = ok && last == hc_count * hidden;
            bool values_ok = false;
            if (shape_ok) {
                const float* d = mlx_array_data_float32(xf);
                values_ok = true;
                for (int rep = 0; rep < hc_count && values_ok; rep++)
                    for (int c = 0; c < hidden && values_ok; c++)
                        if (d[rep * hidden + c] != (float)(token_id * 100 + c)) values_ok = false;
            }
            snprintf(out, (size_t)cap, "%s ndim=%d ultimo_asse=%d atteso=%d valori_ripetuti=%s",
                     (shape_ok && values_ok) ? "PASS" : "FAIL", nd, last, hc_count * hidden,
                     values_ok ? "si" : "NO");
            rc = (shape_ok && values_ok) ? 0 : -2;
            mlx_array_free(xf);
        } else {
            snprintf(out, (size_t)cap, "FAIL: llm_mlx_embed_row_from_idx ha rifiutato (ok=false)");
            rc = -2;
        }
        mlx_array_free(x); mlx_array_free(tid); mlx_array_free(embed_s); mlx_array_free(embed_b); mlx_array_free(embed_w);
    } else {
        snprintf(out, (size_t)cap, "FAIL: malloc embed_w");
    }
    free(wd);
    free(m);
    mlx_device_free(dev);
    return rc;
}

/* D16bis (LLM_ASYNC_DECODE): come forward_one_token, ma l'id del token e' un mlx_array
 * (possibilmente non ancora materializzato/letto in host, prodotto dall'argmax lazy del
 * passo precedente via final_step_greedy_dev) invece di un int32_t host. Usa
 * llm_mlx_embed_row_from_idx (gia' generico per architettura) al posto di embed_token --
 * il resto (layer_forward per-layer, m->arch->attn/ffn) e' IDENTICO a forward_one_token,
 * nessuna funzione per-famiglia toccata. */
bool forward_one_token_dev(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                   mlx_array token_id_arr, int cache_offset,
                                   mlx_array* k_cache, mlx_array* v_cache, bool* has_cache, mlx_array* out_h) {
    mlx_array x = mlx_array_new();
    if (!llm_mlx_embed_row_from_idx(m, embed_w, embed_s, embed_b, token_id_arr, &x)) {
        mlx_array_free(x);
        return false;
    }

    /* Ternary-Bonsai: embed_tokens e' in inverse_weight_names -> si ruota la sua USCITA.
     * Direzione (R vs R^T) incerta dal naming: LLM_BONSAI_EMBED_INV=1 abilita (default off). */
    {
        const int emb_inv = (getenv("LLM_BONSAI_EMBED_INV") != NULL) ? (atoi(getenv("LLM_BONSAI_EMBED_INV")) != 0 ? 1 : 0) : 1;
        (void)llm_bonsai_rotate(&x, m, -1, "embed_tokens.weight", emb_inv == 1);
    }    mlx_array h = x;
    bool step_ok = true;
    for (int layer = 0; step_ok && layer < m->cfg.n_layers; layer++) {
        mlx_array layer_out = mlx_array_new();
        step_ok = layer_forward(m, kvq_sess, layer, cache_offset, /*n_tokens=*/1, h, &k_cache[layer], &v_cache[layer], has_cache[layer], &layer_out);
        has_cache[layer] = true;
        mlx_array_free(h);
        h = layer_out;
    }
    if (step_ok) mlx_array_set(out_h, h);
    mlx_array_free(h);
    return step_ok;
}

/* Attenzione Mixtral in variante pos-ARRAY (per la closure compilata). Unica
 * differenza vs il path fixkv di mixtral_attention_block: RoPE con offset
 * dinamico (mlx_fast_rope_dynamic), scrittura nel buffer statico con pos
 * array (llm_mlx_kv_write_pos_dyn), SDPA su prefisso con pos array
 * (llm_mlx_gqa_sdpa_prefix_dyn). Cache attesa: buffer statici cap uniformi
 * (pos+1 <= cap, garantito dal chiamante via llm_mlx_session_kv_ensure_cap).
 * Pura: nessun eval/host-read (compatibile con i tracer di compile_trace). */
static bool mixtral_attention_block_dyn(LlmMlxModel* m, int layer_idx, mlx_array pos,
                                        mlx_array x, mlx_array* k_cache, mlx_array* v_cache,
                                        mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "mlx_compile[mixtral_attn_dyn l=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)
    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
    if (!ok) CHK(false, "input_layernorm lookup");
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, ln_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "input_layernorm"); }
    /* Q2-b: energia dell'input condiviso di q/k/v (riga attention, expert=-1). Solo in
     * calibrazione (m->im.active): in inferenza costo zero (nessuna lettura host). */
    if (m->im.active && m->im.attn_qkv != NULL && layer_idx >= 0 && layer_idx < m->im.n_layers) {
        int ndq = mlx_array_ndim(normed), lcq = ndq >= 1 ? (int)mlx_array_dim(normed, ndq - 1) : 0;
        int nrq = lcq > 0 ? (int)(mlx_array_size(normed) / (size_t)lcq) : 0;  /* righe: firme diverse fra blocchi */
        if (lcq == m->im.gate_cols && nrq > 0) {
            llm_mlx_imatrix_energy_add(m, m->im.attn_qkv + (size_t)layer_idx * m->im.gate_cols,
                                       m->im.gate_cols, normed, nrq);
            m->im.attn_cnt[layer_idx]++;
        }
    }

    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &qw, &qs, &qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &kw, &ks, &kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &vw, &vs, &vb);
    CHK(ok, "q/k/v_proj lookup");

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = quantized_linear_cfg(&k, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = quantized_linear_cfg(&v, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, 1, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, 1, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_heads, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&k_t, k_heads, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&v_t, v_heads, perm, 4, m->stream) == 0;
        CHK(ok, "qkv transpose");
    }
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) {
        mlx_array no_freqs = mlx_array_new();
        if (m->cfg.use_llama3_rope) {
            /* RoPE "llama3" (3 fasce): stesse freqs custom del path eager
             * llama_attention_block, base disabilitato. Senza questo ramo il blocco
             * `_dyn` (nato per mixtral, RoPE plain) DIVERGE da eager sui modelli
             * llama3-scaled (misurato 2026-09-10: flip argmax da step5). */
            mlx_array custom_freqs = llm_mlx_llama3_freqs_cached(m);
            mlx_optional_float no_base = {0.0f, false};
            ok = mlx_fast_rope_dynamic(&q_rope, q_t, m->cfg.head_dim, /*traditional=*/false, no_base, /*scale=*/1.0f,
                                       pos, custom_freqs, m->stream) == 0 &&
                 mlx_fast_rope_dynamic(&k_rope, k_t, m->cfg.head_dim, false, no_base, 1.0f,
                                       pos, custom_freqs, m->stream) == 0;
            CHK(ok, "rope_dynamic (llama3)");
            mlx_array_free(custom_freqs);
        } else {
            mlx_optional_float base = {m->cfg.rope_theta, true};
            ok = mlx_fast_rope_dynamic(&q_rope, q_t, m->cfg.head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                                       pos, no_freqs, m->stream) == 0 &&
                 mlx_fast_rope_dynamic(&k_rope, k_t, m->cfg.head_dim, false, base, 1.0f,
                                       pos, no_freqs, m->stream) == 0;
            CHK(ok, "rope_dynamic");
        }
        mlx_array_free(no_freqs);
    }
    int cap = 0;
    {
        const int* csh = mlx_array_shape(*k_cache);
        if (csh != NULL) cap = csh[2];
    }
    if (ok && cap <= 0) { ok = false; CHK(ok, "cache cap"); }
    mlx_array k_buf = mlx_array_new(), v_buf = mlx_array_new();
    if (ok) ok = llm_mlx_kv_write_pos_dyn(&k_buf, *k_cache, k_rope, pos, cap, m->stream);
    if (ok) ok = llm_mlx_kv_write_pos_dyn(&v_buf, *v_cache, v_t, pos, cap, m->stream);
    if (ok) { ok = mlx_array_set(k_cache, k_buf) == 0; CHK(ok, "k static store"); }
    if (ok) { ok = mlx_array_set(v_cache, v_buf) == 0; CHK(ok, "v static store"); }
    mlx_array attn = mlx_array_new();
    if (ok) {
        ok = llm_mlx_gqa_sdpa_prefix_dyn(&attn, q_rope, *k_cache, *v_cache, pos,
                                         1.0f / sqrtf((float)m->cfg.head_dim), m->stream);
        CHK(ok, "sdpa prefix dyn");
    }
    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream) == 0;
        CHK(ok, "attn transpose back");
        if (ok) {
            const int flat_shape[3] = {1, 1, m->cfg.n_heads * m->cfg.head_dim};
            ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, m->stream) == 0;
            CHK(ok, "attn flatten");
        }
    }
    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &os, &ob);
    /* Q2-b: energia dell'input di o_proj (uscita attention). Solo in calibrazione e solo se lo
     * stesso layer ha gia' accumulato la sezione q/k/v (contatore condiviso dal writer). */
    if (ok && m->im.active && m->im.attn_o != NULL && m->im.attn_cnt != NULL &&
        layer_idx >= 0 && layer_idx < m->im.n_layers && m->im.attn_cnt[layer_idx] > 0) {
        int ndo = mlx_array_ndim(attn_flat), lco = ndo >= 1 ? (int)mlx_array_dim(attn_flat, ndo - 1) : 0;
        int nro = lco > 0 ? (int)(mlx_array_size(attn_flat) / (size_t)lco) : 0;
        if (lco == m->im.attn_o_cols && nro > 0)
            llm_mlx_imatrix_energy_add(m, m->im.attn_o + (size_t)layer_idx * m->im.attn_o_cols,
                                       m->im.attn_o_cols, attn_flat, nro);
    }
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, os, ob, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "o_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "o_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&attn_out, attn_flat, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, m->stream) == 0; CHK(ok, "residual add (attn)"); }
    mlx_array all[] = {
        ln_w, normed, qw, qs, qb, kw, ks, kb, vw, vs, vb,
        q, k, v, q_heads, k_heads, v_heads,
        q_t, k_t, v_t, q_rope, k_rope, k_buf, v_buf, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

/* Closure del decode-step (funzionale, per mlx_compile).
 * Inputs: [0] token_id int32[1]; [1] pos int32[1]; [2..nL+1) k_cache[l];
 *         [nL+2..2nL+1) v_cache[l].
 * Outputs: [0] h_out [1,1,hidden]; poi k_cache[l] aggiornati; poi v_cache[l].
 * Pura: costruisce SOLO grafo lazy (i tracer di compile_trace non hanno dati:
 * mai eval / read host dentro questa funzione). I pesi sono letti dal modello
 * (payload) e diventano costanti alla trace. */
static int llm_mlx_decode_step_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    if (payload == NULL) return 1;
    LlmMlxSession* s = (LlmMlxSession*)payload;
    LlmMlxModel* m = s->model;
    if (m == NULL) return 1;
    int L = s->n_layers;
    size_t nin = mlx_vector_array_size(in);
    if (nin != (size_t)(2 + 2 * L)) { fprintf(stderr, "mlx_compile: inputs=%zu attesi %d\n", nin, 2 + 2 * L); return 1; }
    mlx_array tid = mlx_array_new(), pos = mlx_array_new();
    mlx_vector_array_get(&tid, in, 0);
    mlx_vector_array_get(&pos, in, 1);

    OutputTensors t;
    if (!load_output_tensors(m, &t)) { fprintf(stderr, "mlx_compile: load_output_tensors failed\n"); return 1; }
    mlx_array h = mlx_array_new();
    bool ok = llm_mlx_embed_row_from_idx(m, t.embed_w, t.embed_s, t.embed_b, tid, &h);
    if (!ok) { fprintf(stderr, "mlx_compile: embed failed\n"); return 1; }

    mlx_array* ks = NULL; mlx_array* vs = NULL;
    if (ok) {
        ks = (mlx_array*)malloc((size_t)L * sizeof(mlx_array));
        vs = (mlx_array*)malloc((size_t)L * sizeof(mlx_array));
        if (ks == NULL || vs == NULL) { ok = false; }
    }
    for (int l = 0; ok && l < L; l++) {
        mlx_vector_array_get(&ks[l], in, 2 + l);
        mlx_vector_array_get(&vs[l], in, 2 + L + l);
        mlx_array h_attn = mlx_array_new();
        ok = mixtral_attention_block_dyn(m, l, pos, h, &ks[l], &vs[l], &h_attn);
        if (ok) {
            mlx_array layer_out = mlx_array_new();
            ok = m->arch->ffn(m, l, /*n_tokens=*/1, h_attn, &layer_out);
            mlx_array_free(h_attn);
        mlx_array_free(h);
        h = layer_out;
        } else {
            mlx_array_free(h_attn);
        }
    }
    free_output_tensors(&t);
    mlx_array_free(tid); mlx_array_free(pos);
    if (!ok) {
        if (ks) { for (int l = 0; l < L; l++) mlx_array_free(ks[l]); free(ks); }
        if (vs) { for (int l = 0; l < L; l++) mlx_array_free(vs[l]); free(vs); }
        mlx_array_free(h);
        fprintf(stderr, "mlx_compile: step graph failed at some layer\n");
        return 1;
    }
    /* Outputs: [0] h_finale, poi k_cache[l] aggiornati, poi v_cache[l].
     * NOTA: *out arriva da closure.cpp come vettore ctx==NULL: NON si usa
     * append_value (lancerebbe "expected a non-empty mlx_vector_array"); si
     * costruisce il vettore e lo si imposta con mlx_vector_array_set. */
    {
        size_t n_out = (size_t)(1 + 2 * L);
        mlx_array* outs = (mlx_array*)malloc(n_out * sizeof(mlx_array));
        if (outs == NULL) {
            for (int l = 0; l < L; l++) { mlx_array_free(ks[l]); mlx_array_free(vs[l]); }
            free(ks); free(vs); mlx_array_free(h);
            return 1;
        }
        outs[0] = h;
        for (int l = 0; l < L; l++) outs[1 + l] = ks[l];
        for (int l = 0; l < L; l++) outs[1 + L + l] = vs[l];
        mlx_vector_array built = mlx_vector_array_new_data(outs, n_out);
        free(outs);
        int rc = built.ctx != NULL ? mlx_vector_array_set(out, built) : 1;
        mlx_vector_array_free(built);
        for (int l = 0; l < L; l++) { mlx_array_free(ks[l]); mlx_array_free(vs[l]); }
        free(ks); free(vs);
        mlx_array_free(h);
        return rc;
    }
}

/* Esegue UN passo di decode tramite la closure compilata. `token_id` = token da
 * far avanzare; `pos` = cache_offset (posizione assoluta del token). Ritorna in
 * *out_h l'hidden state finale [1,1,hidden] e aggiorna s->k_cache/v_cache.
 * La prima chiamata traccia+compila (lenta); le successive = cache-hit. */
bool llm_mlx_session_run_compiled_step_arr(LlmMlxSession* s, mlx_array tid_in, int pos,
                                                  mlx_array* out_h);
/* Variante HOST-INT (path eager): costruisce l'array del token e delega. */
bool llm_mlx_session_run_compiled_step(LlmMlxSession* s, int32_t token_id, int pos,
                                              mlx_array* out_h) {
    int shape[1] = {1};
    mlx_array tid = mlx_array_new_data(&token_id, shape, 1, MLX_INT32);
    bool ok = llm_mlx_session_run_compiled_step_arr(s, tid, pos, out_h);
    mlx_array_free(tid);
    return ok;
}
/* Variante ARRAY: il token arriva come mlx_array invece che come intero host. Serve al
 * path ASYNC (compiled+async): l'id del token precedente e' l'array LAZY `id_pend`
 * (argmax non ancora letto sull'host) -> il grafo compilato lo consuma senza sync,
 * quindi la CPU non ricostruisce piu' le ~375 primitive del decode: patcha 2 input. */
bool llm_mlx_session_run_compiled_step_arr(LlmMlxSession* s, mlx_array tid_in, int pos,
                                                  mlx_array* out_h) {
    LlmMlxModel* m = s->model;
    if (m == NULL) return false;
    if (!llm_mlx_static_cache_ready(s)) return false;
    if (!llm_mlx_session_kv_ensure_cap(s, pos + 1)) return false;

    if (!s->compiled_decode_valid) {
        /* Numeri: il kernel-FUSION di mlx_compile altera i risultati di questo
         * grafo (~1e-3 rel, verificato 2026-09-03: con MLX_COMPILE_MODE_ENABLED
         * il decode compilato DIVERGE da eager; con NO_FUSE e' bit-exact).
         * Default = NO_FUSE (parita'); LLM_COMPILE_FUSE=1 riattiva il fuse. */
        if (getenv("LLM_COMPILE_FUSE")) mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
        else mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
        mlx_closure raw = mlx_closure_new_func_payload(llm_mlx_decode_step_fn, s, NULL);
        mlx_closure comp = mlx_closure_new();
        if (raw.ctx == NULL) return false;
        int rc = mlx_compile(&comp, raw, /*shapeless=*/false);
        mlx_closure_free(raw);
        if (rc != 0) return false;
        rc = mlx_closure_set(&s->compiled_decode, comp);
        mlx_closure_free(comp);
        if (rc != 0) return false;
        s->compiled_decode_valid = true;
        if (getenv("LLM_COMPILE_VERBOSE"))
            fprintf(stderr, "[compiled-decode] closure creata arch=%s fuse=%s n_layers=%d\n",
                    m->cfg.model_type, getenv("LLM_COMPILE_FUSE") ? "ON" : "off", s->n_layers);
    }

    mlx_vector_array in = mlx_vector_array_new();
    int tshape[1] = {1};
    mlx_array posa = mlx_array_new_data(&pos, tshape, 1, MLX_INT32);
    mlx_vector_array_append_value(in, tid_in);
    mlx_vector_array_append_value(in, posa);
    for (int l = 0; l < s->n_layers; l++) mlx_vector_array_append_value(in, s->k_cache[l]);
    for (int l = 0; l < s->n_layers; l++) mlx_vector_array_append_value(in, s->v_cache[l]);
    mlx_array_free(posa);

    mlx_vector_array res = mlx_vector_array_new();
    int rc = mlx_closure_apply(&res, s->compiled_decode, in);
    mlx_vector_array_free(in);
    bool ok = (rc == 0);
    if (ok) {
        size_t nr = mlx_vector_array_size(res);
        if (nr != (size_t)(1 + 2 * s->n_layers)) { ok = false; }
        else {
            mlx_array h = mlx_array_new();
            mlx_vector_array_get(&h, res, 0);
            mlx_array_set(out_h, h);
            mlx_array_free(h);
            for (int l = 0; l < s->n_layers; l++) {
                mlx_array kk = mlx_array_new(), vv = mlx_array_new();
                mlx_vector_array_get(&kk, res, 1 + l);
                mlx_vector_array_get(&vv, res, 1 + s->n_layers + l);
                mlx_array_set(&s->k_cache[l], kk);
                mlx_array_set(&s->v_cache[l], vv);
                mlx_array_free(kk); mlx_array_free(vv);
            }
        }
    }
    mlx_vector_array_free(res);
    if (!ok && rc != 0) fprintf(stderr, "mlx_compile: closure apply failed (rc=%d)\n", rc);
    return ok;
}
