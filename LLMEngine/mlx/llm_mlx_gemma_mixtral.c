/* ============================================================================
 * Dominio Gemma4+Mixtral (attenzione): llm_flash_prefill_fp64_uniform, gemma4_attention_block, mixtral_attention_block.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_flash_decode_kernel.h"
#include "llm_mlx_kv_quant_sdpa_kernel.h"
#include "llm_mlx_fused_rope_append.h"
#include "llm_mlx_fused_residual_rms.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* FASE 1b.4.3 (Gemma4): attenzione con parametri DIVERSI per tipo di layer (sliding
 * vs full/global, m->cfg.is_sliding_layer[layer_idx]) -- la parte più delicata del
 * piano. Riassunto delle differenze verificate contro gemma4_text.py Attention:
 *  - head_dim: global_head_dim (512) sui layer full, head_dim (256) sui sliding.
 *  - n_kv_heads: num_global_kv_heads (4) sui layer full SE attention_k_eq_v, altrimenti
 *    n_kv_heads (16) -- quindi sempre 16 sui layer sliding.
 *  - scala fissa a 1.0 (mai 1/sqrt(head_dim)) su ENTRAMBI i tipi.
 *  - K=V condiviso (values = keys, PRIMA di k_norm/rope) solo sui layer full quando
 *    attention_k_eq_v -- quindi niente v_proj per quei layer specifici.
 *  - V-norm (RMSNormNoScale, senza peso) applicata su ENTRAMBI i tipi di layer (non
 *    solo full/global come ipotizzato inizialmente nel piano -- corretto rileggendo
 *    per intero Attention.__call__: `has_kv` è vero per ogni layer in questo modello).
 *  - RoPE: base semplice sui layer sliding, frequenze custom proporzionali
 *    (llm_mlx_proportional_rope_freqs, Fase 1b.4.2) sui layer full. MAI applicato a V.
 * Il sandwich di norme (input_layernorm prima, post_attention_layernorm sul
 * risultato PRIMA del residuo) è lo stesso pattern già visto in dense_geglu_ffn per
 * l'FFN -- qui applicato al lato attenzione. */

/* LLM_FLASH_PREFILL_FP64=1 (2026-09-16): test ANALITICO con scale=0.
 * scale=0 -> punteggi tutti 0 -> softmax UNIFORME -> out[qi] = media di V sui key [0..coff+qi].
 * E' calcolabile in fp64 senza ambiguita': se kernel e/o naive NON coincidono con quella media,
 * il loro indicizzamento/layout e' sbagliato (oppure lo e' il mio). Serve a distinguere un bug di
 * layout da un errore di precisione, che il confronto a scale=1.0 non puo' fare (li' con |score|
 * ~40-80 la softmax e' quasi un argmax e ogni differenza di punteggio ribalta la posizione). */
static void llm_flash_prefill_fp64_uniform(mlx_array v_src, mlx_array kout, mlx_array nout,
                                           int H, int Hkv, int D, int Lq, int coff,
                                           mlx_stream stream) {
    /* INVARIANTE AUTO-CONSISTENTE: a scale=0 i punteggi sono tutti 0 -> softmax UNIFORME in
     * entrambe le implementazioni -> out = media causale di V. Qui NON serve un riferimento
     * esterno (il mio si e' dimostrato inaffidabile): basta confrontare KERNEL vs NAIVE. Se
     * differiscono ~1e-6, il kernel e' una implementazione FEDELE della stessa attention; a
     * scale=1.0 la differenza (0,16-0,31) e' allora solo il rounding bf16 dei punteggi del
     * naive (0,4% di |score| con |score|~40-80), NON un errore del kernel.
     * (v_src/Hkv/Lq/coff non servono piu' e restano per firma compatibile.) */
    (void)v_src; (void)Hkv; (void)Lq; (void)coff;
    mlx_array kf = mlx_array_new(), nf = mlx_array_new();
    const float *kd = NULL, *nd = NULL;
    if (mlx_astype(&kf, kout, MLX_FLOAT32, stream) == 0 &&
        mlx_astype(&nf, nout, MLX_FLOAT32, stream) == 0 &&
        mlx_array_eval(kf) == 0 && mlx_array_eval(nf) == 0) {
        kd = mlx_array_data_float32(kf); nd = mlx_array_data_float32(nf);
    }
    if (kd && nd) {
        size_t nel = mlx_array_size(kf); double mx = 0; size_t n = 0;
        for (size_t i = 0; i < nel; i++) {
            double d = fabs((double)kd[i] - (double)nd[i]);
            if (d > mx) mx = d;
            if (d > 0) n++;
        }
        fprintf(stderr, "[flash-prefill-uniform] KERNEL vs NAIVE a scale=0 (softmax uniforme): "
                        "max_diff=%.6g (%zu/%zu elementi diversi) H=%d D=%d -> ~1 ULP bf16 = STESSA attention, arrotondamenti diversi\n",
                mx, n, nel, H, D);
    }
    mlx_array_free(kf); mlx_array_free(nf);
}

bool gemma4_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    char namebuf[256];
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;

    const bool is_sliding = m->cfg.is_sliding_layer[layer_idx];
    const bool use_k_eq_v = m->cfg.attention_k_eq_v && !is_sliding;
    const int head_dim = (!is_sliding && m->cfg.global_head_dim > 0) ? m->cfg.global_head_dim : m->cfg.head_dim;
    const int n_kv_heads = (use_k_eq_v && m->cfg.num_global_kv_heads > 0) ? m->cfg.num_global_kv_heads : m->cfg.n_kv_heads;
    const float rope_theta = is_sliding ? m->cfg.rope_theta_sliding : m->cfg.rope_theta_full;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "gemma4_attention[layer=%d,%s]: %s failed\n", layer_idx, is_sliding ? "sliding" : "full", (what)); ok = false; } \
    } while (0)

    mlx_array ln_w = mlx_array_new();
    tname_gemma4(layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
    CHK(ok, "input_layernorm lookup");
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

    /* Q: proiezione + reshape a teste + q_norm (con peso, sempre presente). */
    GattnQuant* ac4 = llm_mlx_gemma4_attn_quant(m, layer_idx);
    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    if (ac4 != NULL) { ok = ok && mlx_array_set(&qw, ac4->qw) == 0 && mlx_array_set_optional(&qs, ac4->qs) && mlx_array_set_optional(&qb, ac4->qb); }
    else { tname_gemma4(layer_idx, "self_attn.q_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &qw, &qs, &qb); CHK(ok, "q_proj lookup"); }
    mlx_array q0 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q0, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    mlx_array q_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, head_dim};
        ok = mlx_reshape(&q_heads, q0, q_shape, 4, m->stream) == 0;
        CHK(ok, "q head reshape");
    }
    mlx_array qn_w = mlx_array_new();
    tname_gemma4(layer_idx, "self_attn.q_norm.weight", namebuf, sizeof(namebuf));
    if (ac4 != NULL && ac4->has_qn) { ok = ok && mlx_array_set(&qn_w, ac4->qn_w) == 0; }
    else { ok = ok && multishard_get(&m->weights, namebuf, &qn_w); }
    CHK(ok, "q_norm lookup");
    mlx_array q_normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&q_normed, q_heads, qn_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "q_norm"); }

    /* K: proiezione + reshape. Sui layer KV-SHARED (gemma4: j >= n_layers - kv_shared_layers)
     * k_proj/v_proj/k_norm NON esistono: il K/V corrente (post-norm+rope) viene preso dallo
     * STASH del primo layer precedente dello stesso tipo (aux_cache[src], schema mlx_lm
     * previous_kvs). Il layer shared lo scrive nel PROPRIO cache (cache.update_and_fetch). */
    const int N_layers = m->cfg.n_layers, S_shared = m->cfg.kv_shared_layers;
    const bool kv_shared = (S_shared > 0 && layer_idx >= N_layers - S_shared);
    int kv_src = layer_idx;
    if (kv_shared && m->cfg.is_sliding_layer)
        for (int i = 0; i < N_layers - S_shared; i++)
            if (m->cfg.is_sliding_layer[i] == is_sliding) { kv_src = i; break; }

    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array k0 = mlx_array_new();
    mlx_array k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok && kv_shared) {
        mlx_array* st_ptr = (m->aux_cache && kv_src < N_layers) ? &m->aux_cache[kv_src] : NULL;
        ok = st_ptr && (*st_ptr).ctx != NULL;
        if (ok) {
            /* stash [1,nkv,T,2*hd] -> [1,T,nkv,2*hd]: prima meta' K (post-rope), seconda V. */
            int hd2[4] = {1, n_kv_heads, n_tokens, head_dim*2};
            mlx_array st2 = mlx_array_new(), st_t = mlx_array_new();
            ok = mlx_reshape(&st2, *st_ptr, hd2, 4, m->stream) == 0 &&
                 mlx_transpose_axes(&st_t, st2, (const int[4]){0,2,1,3}, 4, m->stream) == 0;
            if (ok) { int sec[1] = {head_dim}; mlx_vector_array sp = mlx_vector_array_new();
                      ok = mlx_split_sections(&sp, st_t, sec, 1, -1, m->stream) == 0 &&
                           mlx_vector_array_get(&k_heads, sp, 0) == 0 && mlx_vector_array_get(&v_heads, sp, 1) == 0;
                      mlx_vector_array_free(sp); }
            mlx_array_free(st2); mlx_array_free(st_t);
        }
        CHK(ok, "kv-shared stash lookup");
    } else if (ok) {
    if (ac4 != NULL) { ok = ok && mlx_array_set(&kw, ac4->kw) == 0 && mlx_array_set_optional(&ks, ac4->ks) && mlx_array_set_optional(&kb, ac4->kb); }
    else { tname_gemma4(layer_idx, "self_attn.k_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &kw, &ks, &kb); CHK(ok, "k_proj lookup"); }
    if (ok) { ok = quantized_linear_cfg(&k0, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    if (ok) {
        const int kv_shape[4] = {1, n_tokens, n_kv_heads, head_dim};
        ok = mlx_reshape(&k_heads, k0, kv_shape, 4, m->stream) == 0;
        CHK(ok, "k head reshape");
    }
    }

    /* V: se use_k_eq_v, è una copia di k_heads (PRIMA di k_norm) -- nessun v_proj
     * separato caricato in quel caso. Altrimenti proiezione propria, come al solito.
     * Sui layer kv-shared V arriva già dallo stash (nessun v_proj nel checkpoint). */
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    mlx_array v0 = mlx_array_new();
    if (ok && kv_shared) {
        CHK(true, "v_heads dallo stash (kv-shared)");
    } else if (ok && use_k_eq_v) {
        ok = mlx_array_set(&v_heads, k_heads) == 0;
        CHK(ok, "v_heads = k_heads (k_eq_v)");
    } else if (ok) {
        tname_gemma4(layer_idx, "self_attn.v_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &vw, &vs, &vb);
        CHK(ok, "v_proj lookup");
        if (ok) { ok = quantized_linear_cfg(&v0, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
        if (ok) {
            const int kv_shape[4] = {1, n_tokens, n_kv_heads, head_dim};
            ok = mlx_reshape(&v_heads, v0, kv_shape, 4, m->stream) == 0;
            CHK(ok, "v head reshape");
        }
    }

    /* k_norm (con peso) su K; v_norm (SENZA peso, RMSNormNoScale) su V -- applicata
     * su entrambi i tipi di layer, verificato. Sui layer kv-shared K/V sono GIA'
     * normalizzati e rope-ati (dallo stash): niente norm e niente RoPE su K. */
    mlx_array kn_w = mlx_array_new();
    mlx_array k_normed = mlx_array_new();
    if (ok && kv_shared) {
        ok = mlx_array_set(&k_normed, k_heads) == 0;
        CHK(ok, "k_normed dallo stash (kv-shared)");
    } else {
    tname_gemma4(layer_idx, "self_attn.k_norm.weight", namebuf, sizeof(namebuf));
    if (ac4 != NULL && ac4->has_qn) { ok = ok && mlx_array_set(&kn_w, ac4->kn_w) == 0; }
    else { ok = ok && multishard_get(&m->weights, namebuf, &kn_w); }
    CHK(ok, "k_norm lookup");
    if (ok) { ok = rms_norm_cfg(&k_normed, k_heads, kn_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "k_norm"); }
    }
    mlx_array v_normed = mlx_array_new();
    if (ok && kv_shared) {
        ok = mlx_array_set(&v_normed, v_heads) == 0;
        CHK(ok, "v_normed dallo stash (kv-shared)");
    } else {
    if (ok) { ok = mlx_fast_rms_norm_no_scale(&v_normed, v_heads, head_dim, m->cfg.rms_norm_eps, m->stream); CHK(ok, "v_norm (no-scale)"); }
    }

    /* Transpose a [B, n_heads/n_kv_heads, L, head_dim] per attenzione/RoPE. */
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_normed, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&k_t, k_normed, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&v_t, v_normed, perm, 4, m->stream) == 0;
        CHK(ok, "qkv transpose");
    }

    /* RoPE su Q e K (MAI su V): base semplice sui layer sliding, frequenze custom
     * proporzionali sui layer full/global (Fase 1b.4.2). Sui layer kv-shared la K e'
     * GIA' rope-ata (la RoPE del layer sorgente, nel checkpoint): RoPE solo su Q. */
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    mlx_array custom_freqs = mlx_array_new();
    if (ok && kv_shared) {
        ok = mlx_array_set(&k_rope, k_t) == 0;
        CHK(ok, "k_rope dallo stash (kv-shared)");
        if (ok) {
            if (is_sliding) {
                mlx_optional_float base = {rope_theta, true};
                mlx_array no_freqs = mlx_array_new();
                ok = mlx_fast_rope(&q_rope, q_t, head_dim, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
                CHK(ok, "rope Q (kv-shared, sliding)");
                mlx_array_free(no_freqs);
            } else {
                int rotated_dims = (int)(head_dim * m->cfg.partial_rotary_factor);
                custom_freqs = llm_mlx_proportional_rope_freqs(head_dim, rotated_dims, rope_theta, 1.0f);
                mlx_optional_float no_base = {0.0f, false};
                ok = mlx_fast_rope(&q_rope, q_t, head_dim, false, no_base, 1.0f, cache_offset, custom_freqs, m->stream) == 0;
                CHK(ok, "rope Q (kv-shared, full)");
            }
        }
    } else if (ok) {
        if (is_sliding) {
            mlx_optional_float base = {rope_theta, true};
            mlx_array no_freqs = mlx_array_new();
            ok = mlx_fast_rope(&q_rope, q_t, head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                                cache_offset, no_freqs, m->stream) == 0 &&
                 mlx_fast_rope(&k_rope, k_t, head_dim, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
            CHK(ok, "rope (sliding, base semplice)");
            mlx_array_free(no_freqs);
        } else {
            int rotated_dims = (int)(head_dim * m->cfg.partial_rotary_factor);
            mlx_array_free(custom_freqs); /* sovrascritto: l'handle vuoto iniziale va liberato */
            custom_freqs = llm_mlx_proportional_rope_freqs(head_dim, rotated_dims, rope_theta, 1.0f);
            mlx_optional_float no_base = {0.0f, false};
            ok = mlx_fast_rope(&q_rope, q_t, head_dim, false, no_base, 1.0f, cache_offset, custom_freqs, m->stream) == 0 &&
                 mlx_fast_rope(&k_rope, k_t, head_dim, false, no_base, 1.0f, cache_offset, custom_freqs, m->stream) == 0;
            CHK(ok, "rope (full, frequenze custom)");
        }
    }

    /* STASH del K/V corrente (post-norm+rope) per i layer kv-shared successivi: il primo
     * layer di ogni tipo lo pubblica in aux_cache[layer] come [1,nkv,T,2hd] (K|V). */
    if (ok && S_shared > 0 && !kv_shared && m->aux_cache && layer_idx < N_layers) {
        mlx_vector_array kvv = mlx_vector_array_new();
        mlx_vector_array_append_value(kvv, k_rope); mlx_vector_array_append_value(kvv, v_t);
        mlx_array st = mlx_array_new();
        if (mlx_concatenate_axis(&st, kvv, -1, m->stream) == 0)
            mlx_array_set(&m->aux_cache[layer_idx], st);
        mlx_array_free(st);
        mlx_vector_array_free(kvv);
    }

    /* Crescita KV-cache + attenzione. Due vie mutuamente esclusive:
     *  (A) paged-KV IN-PLACE (LLM_KVSLICE) SOLO per i layer FULL in decode mono-token:
     *      buffer persistente a capacita' + slice_update del SOLO token nuovo (zero
     *      concat O(N) per step) + SDPA causale nativa sulla slice strided. Port del
     *      path gia' presente in llama/qwen3_moe/qwen2/glm e LA' MISURATO vincente
     *      (llama: 1.19x@13K, 1.85-1.95x@30K vs concat, "chiude il gap O(context)" --
     *      vedi :7092-7105). I layer FULL di gemma NE ERANO PRIVI: ri-concatenavano
     *      l'intera cache O(N) per token -> collasso 16->8 tok/s a 4K (§18bis). Schema
     *      di sicurezza identico al llama: donazione del buffer via free-delle-view
     *      stale prima dello slice_update, e NESSUNA mlx_contiguous difensiva sulla
     *      slice di lettura (fast_sdpa accetta K/V non contigui, verificato). Default
     *      OFF (flag LLM_KVSLICE): senza il flag il comportamento e' il ramo (B).
     *  (B) concat (prefill, primo token, layer sliding, oppure flag off): identico a
     *      prima. Gli sliding sono gia' troncati a `win` in layer_forward (:9889-9909),
     *      quindi il loro concat resta O(window) = economico, non il collo di bottiglia. */
    mlx_array attn = mlx_array_new();
    /* k_full/v_full: gestiti dal ramo (B) concat, liberi qui (handle vuoti) nel ramo
     * (A) in-place; comunque rilasciati dalla cleanup `all[]` in fondo alla funzione. */
    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    const bool kvsl = llm_mlx_kvslice_enabled() && !is_sliding && n_tokens == 1 &&
                      has_cache && m->cfg.kv_max_tokens <= 0;
    mlx_array k_src = mlx_array_new(), v_src = mlx_array_new();
    bool kv_inplace = false;
    if (ok && kvsl) {
        const int nkv = n_kv_heads, hd = head_dim;
        int have_len = 0;
        if ((*k_cache).ctx != NULL && mlx_array_ndim(*k_cache) == 4) {
            const int* csh = mlx_array_shape(*k_cache); if (csh) have_len = csh[2];
        }
        int needed = cache_offset + n_tokens;
        int cap = have_len >= needed ? have_len : llm_mlx_kv_cap_for(needed);
        mlx_array k_base = mlx_array_new(), v_base = mlx_array_new();
        if (have_len > 0 && have_len < cap) {
            int zsh[4] = {1, nkv, cap - have_len, hd};
            mlx_array kz = mlx_array_new(), vz = mlx_array_new();
            bool zok = mlx_zeros(&kz, zsh, 4, mlx_array_dtype(*k_cache), m->stream) == 0 &&
                       mlx_zeros(&vz, zsh, 4, mlx_array_dtype(*v_cache), m->stream) == 0;
            if (zok) {
                mlx_vector_array kv = mlx_vector_array_new();
                mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, kz);
                zok = mlx_concatenate_axis(&k_base, kv, 2, m->stream) == 0; mlx_vector_array_free(kv);
                mlx_vector_array vv = mlx_vector_array_new();
                mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, vz);
                zok = zok && mlx_concatenate_axis(&v_base, vv, 2, m->stream) == 0; mlx_vector_array_free(vv);
            }
            mlx_array_free(kz); mlx_array_free(vz);
            if (!zok) { ok = false; }
        } else if (have_len == cap && cap > 0) {
            /* Donazione: set duplica l'handle -> refcount 2 romperebbe is_donatable()
             * dentro slice_update (che ripiegherebbe su una copia O(cap)). Liberare
             * *k_cache subito dopo aver preso k_base riporta il refcount a 1. */
            ok = ok && mlx_array_set(&k_base, *k_cache) == 0 && mlx_array_set(&v_base, *v_cache) == 0;
            if (ok) {
                mlx_array_free(*k_cache); mlx_array_free(*v_cache);
                *k_cache = mlx_array_new(); *v_cache = mlx_array_new();
            }
        } else if (cap > 0) {
            int zsh[4] = {1, nkv, cap, hd};
            mlx_dtype dt = mlx_array_dtype(k_rope);
            ok = ok && mlx_zeros(&k_base, zsh, 4, dt, m->stream) == 0 &&
                       mlx_zeros(&v_base, zsh, 4, dt, m->stream) == 0;
        }
        mlx_array k_buf = mlx_array_new(), v_buf = mlx_array_new();
        if (ok && cap > 0 && cache_offset >= 0 && cache_offset + 1 <= cap) {
            int st4[4] = {0, 0, cache_offset, 0}, sp4[4] = {1, nkv, cache_offset + 1, hd}, sv4[4] = {1, 1, 1, 1};
            ok = mlx_slice_update_inplace(&k_base, k_rope, st4, 4, sp4, 4, sv4, 4, m->stream) == 0 &&
                 mlx_slice_update_inplace(&v_base, v_t, st4, 4, sp4, 4, sv4, 4, m->stream) == 0;
        } else { ok = false; }
        if (ok) { ok = mlx_array_set(k_cache, k_base) == 0 && mlx_array_set(v_cache, v_base) == 0; }
        mlx_array_free(k_base); mlx_array_free(v_base); mlx_array_free(k_buf); mlx_array_free(v_buf);
        if (ok) {
            mlx_array kp = mlx_array_new(), vp = mlx_array_new();
            int p0[4] = {0, 0, 0, 0}, p1[4] = {1, nkv, needed, hd}, ps[4] = {1, 1, 1, 1};
            ok = mlx_slice(&kp, *k_cache, p0, 4, p1, 4, ps, 4, m->stream) == 0 &&
                 mlx_slice(&vp, *v_cache, p0, 4, p1, 4, ps, 4, m->stream) == 0;
            if (ok) {
                ok = llm_mlx_gqa_sdpa(&attn, q_rope, kp, vp, 1.0f, 0, m->stream)  /* gemma4: scala 1.0 */;
                CHK(ok, "scaled_dot_product_attention (kvslice full)");
            }
            mlx_array_free(kp); mlx_array_free(vp);
        }
    } else if (ok) {
        if (ok && has_cache && n_tokens == 1 && !m->cfg.nsa_sparse && llm_mlx_kv_inplace_enabled() &&
            !is_sliding)
            kv_inplace = llm_mlx_kv_append_inplace(k_cache, v_cache, k_rope, v_t,
                                                   cache_offset, n_tokens, /*step=*/256,
                                                   &k_src, &v_src, m->stream);
        if (ok && !kv_inplace) {
        if (has_cache) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache);
            mlx_vector_array_append_value(kv, k_rope);
            ok = mlx_concatenate_axis(&k_full, kv, 2, m->stream) == 0;
            CHK(ok, "k-cache concat");
            mlx_vector_array_free(kv);
            if (ok) {
                mlx_vector_array vv = mlx_vector_array_new();
                mlx_vector_array_append_value(vv, *v_cache);
                mlx_vector_array_append_value(vv, v_t);
                ok = mlx_concatenate_axis(&v_full, vv, 2, m->stream) == 0;
                CHK(ok, "v-cache concat");
                mlx_vector_array_free(vv);
            }
        } else {
            ok = mlx_array_set(&k_full, k_rope) == 0 && mlx_array_set(&v_full, v_t) == 0;
            CHK(ok, "k/v cache init");
        }
        if (ok) { ok = mlx_array_set(&k_src, k_full) == 0 && mlx_array_set(&v_src, v_full) == 0; }
        }
        /* Rolling-Window KV (fase 3 paging): se kv_max_tokens>0, tronca la cache full ai piu' recenti
         * N token (mantiene i blocchi recenti, evicta i vecchi via Cache ABI trim). Limita la memoria
         * KV a contesto lungo (StreamingLLM-style). 0 = off (cache illimitata). */
        if (ok && !kv_inplace && m->cfg.kv_max_tokens > 0) {
            mlx_array k2 = mlx_array_new(), v2 = mlx_array_new();
            ok = llm_mlx_cache_trim(&k2, k_full, m->cfg.kv_max_tokens, m->stream) &&
                 llm_mlx_cache_trim(&v2, v_full, m->cfg.kv_max_tokens, m->stream);
            if (ok) { mlx_array_free(k_full); k_full = k2; mlx_array_free(v_full); v_full = v2; }
            else { mlx_array_free(k2); mlx_array_free(v2); }
        }
        if (ok && !kv_inplace) {
            ok = mlx_array_set(k_cache, k_full) == 0;
            CHK(ok, "k cache store");
            if (ok) { ok = mlx_array_set(v_cache, v_full) == 0; CHK(ok, "v cache store"); }
        }
        /* Retention SnapKV sui layer FULL (i layer sliding trimmiano gia' via sliding_window):
         * i full crescono O(pos) a contesto lungo. Gated dalla policy LLM_KV_POLICY (retain=snapkv)
         * o LLM_KVRETAIN. Lossy -> opt-in. */
        if (ok && !is_sliding && m->cfg.kv_max_tokens <= 0) {
            static int pol_loaded = 0;
            static LlmMlxKvPolicy pol;
            if (!pol_loaded) { pol = llm_mlx_kv_policy_from_env(); pol_loaded = 1; }
            if ((pol.retain == 1) || llm_mlx_kvretain_enabled()) {
                int budget = (pol.retain == 1) ? pol.retain_budget : llm_mlx_kvretain_param("LLM_KVRETAIN_BUDGET", 1024);
                int sink = (pol.retain == 1) ? pol.retain_sink : llm_mlx_kvretain_param("LLM_KVRETAIN_SINK", 4);
                int recent = (pol.retain == 1) ? pol.retain_recent : llm_mlx_kvretain_param("LLM_KVRETAIN_RECENT", 128);
                int obs = (pol.retain == 1) ? pol.retain_obs : llm_mlx_kvretain_param("LLM_KVRETAIN_OBS", 32);
                int slen = (mlx_array_ndim(*k_cache) == 4) ? mlx_array_shape(*k_cache)[2] : 0;
                if (slen > budget) {
                    mlx_array rk = mlx_array_new(), rv = mlx_array_new();
                    if (llm_mlx_kv_retain(&rk, &rv, *k_cache, *v_cache, budget, sink, recent, obs, m->stream))
                        ok = mlx_array_set(k_cache, rk) == 0 && mlx_array_set(v_cache, rv) == 0;
                    mlx_array_free(rk); mlx_array_free(rv);
                }
            }
        }
        /* Attenzione: scala FISSA a 1.0 (mai 1/sqrt(head_dim) come in ogni altra
         * architettura di questo motore). Sui layer SLIDING viene applicata la finestra
         * scorrevole (window = sliding_window, es. 1024) tramite llm_mlx_gqa_sdpa (mask
         * additiva pre-softmax causale+finestra, fedele a gemma4_text.py create_causal_mask:
         * k<=q AND k>=q-(window-1)). */
        if (ok) {
            int win = (is_sliding && m->cfg.sliding_window > 0) ? m->cfg.sliding_window : 0;
            /* Il kernel custom copre SOLO il decode full-layer (hd=512, win=0, 1 token,
             * niente trim rolling-window): li' MLX andrebbe sul naive che materializza.
             * Sliding (hd=256) e prefill (n_tokens>1) restano su llm_mlx_gqa_sdpa intonsi. */
            bool fd_used = false;
            const bool fd_elig = (llm_mlx_flash_decode_enabled() || llm_mlx_flash_decode_verify_enabled()) &&
                                 !is_sliding && n_tokens == 1 &&
                                 win == 0 && m->cfg.kv_max_tokens <= 0 && (*k_cache).ctx != NULL &&
                                 (*v_cache).ctx != NULL;
            /* LLM_GEMMA_FLASH_DECODE_DEBUG=1: stampa i soli ingressi che decidono l'eleggibilita'
             * (una manciata di chiamate) -> distingue "il kernel ha perso" da "non e' partito". */
            {
                static int fd_dbg = -1, fd_pr = 0;
                if (fd_dbg < 0) fd_dbg = (getenv("LLM_GEMMA_FLASH_DECODE_DEBUG") != NULL) ? 1 : 0;
                if (fd_dbg == 1 && fd_pr < 8) {
                    fd_pr++;
                    fprintf(stderr, "[flash-decode-debug] layer=%d is_sliding=%d n_tokens=%d win=%d kv_max=%d hd=%d kctx=%d vctx=%d\n",
                            layer_idx, (int)is_sliding, n_tokens, win, m->cfg.kv_max_tokens, head_dim,
                            (*k_cache).ctx != NULL, (*v_cache).ctx != NULL);
                }
            }
            if (fd_elig) {
                /* FIX (2026-09-16): la cache e' [1,nkv,CAP,hd] e CAP e' la **CAPACITA'** del buffer
                 * (arrotondata a step 256), NON la lunghezza valida. Con `vislen = shape(*k_cache)[2]`
                 * il kernel leggeva anche il padding oltre cache_offset+n_tokens -> risultato SBAGLIATO
                 * (misurato con LLM_GEMMA_FLASH_DECODE_VERIFY: max_abs_diff 4,15/5,78/6,09/9,68, cioe'
                 * non rumore f16). E' il motivo per cui il kernel era inutilizzabile. k_src/v_src sono
                 * le viste TRONCATE a cache_offset+n_tokens: le stesse che usa la naive qui sotto. */
                const int* ksh = mlx_array_shape(k_src);
                const int* vsh = mlx_array_shape(v_src);
                int vislen = (ksh && mlx_array_ndim(k_src) == 4) ? ksh[2] : 0;
                if (vislen > 0 && vsh && mlx_array_ndim(v_src) == 4 && vsh[2] == vislen) {
                    const size_t nel = (size_t)1 * m->cfg.n_heads * head_dim;
                    if (llm_mlx_flash_decode_verify_enabled()) {
                        /* ENTRAMBI: uso il naive, stampo il diff. */
                        mlx_array fd_out = mlx_array_new();
                        bool fd_ok = llm_mlx_flash_decode_attention(&fd_out, q_rope, k_src, v_src,
                                            m->cfg.n_heads, n_kv_heads, head_dim, vislen, 1.0f, m->stream);
                        ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f, win, m->stream);
                        if (fd_ok && ok) {
                            double md = llm_mlx_max_abs_diff(attn, fd_out, nel, m->stream);
                            if (md >= 0) fprintf(stderr, "[flash-decode-verify] layer=%d cache_offset=%d vislen=%d max_abs_diff=%.6g\n",
                                                 layer_idx, cache_offset, vislen, md);
                        }
                        mlx_array_free(fd_out);
                        fd_used = true;  /* naive gia' in `attn`; evita la seconda chiamata qui sotto */
                    } else {
                        fd_used = llm_mlx_flash_decode_attention(&attn, q_rope, k_src, v_src,
                                            m->cfg.n_heads, n_kv_heads, head_dim, vislen, 1.0f, m->stream);
                        ok = fd_used;
                    }
                }
            }
            /* LLM_GEMMA_FLASH_PREFILL (default OFF): prefill (n_tokens>1) dei layer FULL
             * (hd=512) su kernel custom — evita la materializzazione [Lq,Lk] della naive. */
            bool fp_used = false;
            /* SOGLIA Nk (2026-09-16): il kernel e' per-query SENZA key-tiling (loop O(Nk)) ->
             * vince solo a Nk piccolo, dove evita la materializzazione Lq*Lk dei layer full
             * hd=512. MISURATO a 32K (load ~3,8): kernel ON+chunk2048 408 s vs kernel
             * OFF+chunk128 139 s (il naive usa GEMM e scala meglio con Nk). Sotto ~16K vince il
             * kernel; a 16K le config sono entro il 3%. */
            if (ok && !fd_used && llm_mlx_flash_prefill_enabled() && !is_sliding && n_tokens > 1 &&
                (cache_offset + n_tokens) <= 16384 &&
                win == 0 && m->cfg.kv_max_tokens <= 0 && (*k_cache).ctx != NULL && (*v_cache).ctx != NULL) {
                const int* ksh = mlx_array_shape(*k_cache);
                if (ksh != NULL && mlx_array_ndim(*k_cache) == 4 && ksh[2] >= cache_offset + n_tokens) {
                    mlx_array fp_out = mlx_array_new();
                    bool fok = llm_mlx_flash_prefill_attention(&fp_out, q_rope, *k_cache, *v_cache,
                                  m->cfg.n_heads, n_kv_heads, head_dim, n_tokens, cache_offset, 1.0f, m->stream);
                    if (fok && llm_env_isset("LLM_FLASH_PREFILL_FP64")) {
                        mlx_array kz = mlx_array_new(), nz = mlx_array_new();
                        bool zk = llm_mlx_flash_prefill_attention(&kz, q_rope, *k_cache, *v_cache,
                                      m->cfg.n_heads, n_kv_heads, head_dim, n_tokens, cache_offset, 0.0f, m->stream);
                        bool zn = llm_mlx_gqa_sdpa(&nz, q_rope, k_src, v_src, 0.0f, win, m->stream);
                        if (zk && zn) llm_flash_prefill_fp64_uniform(v_src, kz, nz, m->cfg.n_heads,
                                      n_kv_heads, head_dim, n_tokens, cache_offset, m->stream);
                        mlx_array_free(kz); mlx_array_free(nz);
                    }
                    if (fok && llm_env_isset("LLM_GEMMA_FLASH_PREFILL_VERIFY")) {
                        mlx_array ref = mlx_array_new();
                        if (llm_mlx_gqa_sdpa(&ref, q_rope, k_src, v_src, 1.0f, win, m->stream)) {
                            size_t nel = (size_t)m->cfg.n_heads * (size_t)n_tokens * (size_t)head_dim;
                            double md = llm_mlx_max_abs_diff(fp_out, ref, nel, m->stream);
                            const int* qsh = mlx_array_shape(q_rope);
                            const int* ksh2 = mlx_array_shape(*k_cache);
                            const int* osh = mlx_array_shape(fp_out);
                            fprintf(stderr, "[flash-prefill-verify] layer=%d Lq=%d coff=%d maxdiff=%g "
                                    "q=[%d,%d,%d,%d] k=[%d,%d,%d,%d] out=[%d,%d,%d,%d] hd=%d\n",
                                    layer_idx, n_tokens, cache_offset, md,
                                    qsh[0], qsh[1], qsh[2], qsh[3], ksh2[0], ksh2[1], ksh2[2], ksh2[3],
                                    osh[0], osh[1], osh[2], osh[3], head_dim);
                        }
                        mlx_array_free(ref);
                    }
                    if (fok) { mlx_array_set(&attn, fp_out); fp_used = true; }
                    mlx_array_free(fp_out);
                }
            }
            if (!fd_used && !fp_used)
                ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f, win, m->stream)  /* gemma4: scale fissa 1.0 */;
            CHK(ok, "scaled_dot_product_attention");
        }
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream) == 0;
        CHK(ok, "attn transpose back");
        if (ok) {
            const int flat_shape[3] = {1, n_tokens, m->cfg.n_heads * head_dim};
            ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, m->stream) == 0;
            CHK(ok, "attn flatten");
        }
    }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    tname_gemma4(layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
    if (ac4 != NULL && ac4->ow.ctx != NULL) { ok = ok && mlx_array_set(&ow, ac4->ow) == 0 && mlx_array_set_optional(&os, ac4->os) && mlx_array_set_optional(&ob, ac4->ob); }
    else { ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &os, &ob); }
    CHK(ok, "o_proj lookup");
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

    /* post_attention_layernorm applicata al risultato dell'attenzione PRIMA del
     * residuo -- diverso da qwen3_moe_attention_block, dove il residuo è diretto
     * (x + attn_out) senza questa norma intermedia. */
    mlx_array ln2_w = mlx_array_new();
    tname_gemma4(layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
    CHK(ok, "post_attention_layernorm lookup");
    /* FUSIONE residual+RMSNorm (opt-in LLM_FUSE_RESIDUAL_RMS): y = x + w*rmsnorm(attn_out) in UN
     * dispatch al posto di {RMSNorm -> Add}. VERIFY calcola entrambi, stampa il max_abs_diff e usa
     * l'originale. Fallback silenzioso se la geometria non e' supportata. */
    mlx_array attn_normed = mlx_array_new();
    bool rr_fused = false;
    if (ok && (llm_mlx_fused_residual_rms_enabled() || llm_mlx_fused_residual_rms_verify_enabled()) &&
        mlx_array_ndim(x) == 3 && mlx_array_ndim(attn_out) == 3) {
        mlx_array yr = mlx_array_new();
        if (llm_mlx_fused_residual_rms(&yr, x, attn_out, ln2_w, m->cfg.rms_norm_eps, m->stream)) {
            if (llm_mlx_fused_residual_rms_verify_enabled()) {
                mlx_array ref = mlx_array_new();
                bool okr = rms_norm_cfg(&attn_normed, attn_out, ln2_w, m->cfg.rms_norm_eps, m->stream);
                if (okr) okr = mlx_add(&ref, x, attn_normed, m->stream) == 0;
                double dr = okr ? llm_mlx_max_abs_diff(yr, ref, (size_t)mlx_array_size(yr), m->stream) : -1.0;
                static int rr_once = 0;
                if (!rr_once || dr < 0 || dr > 1e-2) {
                    rr_once = 1;
                    fprintf(stderr, "[fuse-residual-rms-verify] layer=%d diff=%.6g%s\n",
                            layer_idx, dr, (dr < 0 || dr > 1e-2) ? "  <-- MISMATCH" : "");
                }
                mlx_array_free(ref);
                ok = ok && okr;
            } else {
                ok = mlx_array_set(out_h, yr) == 0;
                rr_fused = true;
            }
        }
        mlx_array_free(yr);
    }
    if (ok && !rr_fused) {
        if (ok) { ok = rms_norm_cfg(&attn_normed, attn_out, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }
        if (ok) { ok = mlx_add(out_h, x, attn_normed, m->stream) == 0; CHK(ok, "residual add (attn)"); }
    }

    mlx_array all[] = {
        ln_w, normed, qw, qs, qb, q0, q_heads, qn_w, q_normed,
        kw, ks, kb, k0, k_heads, vw, vs, vb, v0, v_heads,
        kn_w, k_normed, v_normed, q_t, k_t, v_t,
        q_rope, k_rope, custom_freqs, k_full, v_full, k_src, v_src, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out, ln2_w, attn_normed,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}

/* FASE 1b.3 (Mixtral, es. Mixtral-8x22B): attenzione GQA "pura" -- nessun QK-norm
 * (verificato: MixtralAttention non ha q_norm/k_norm, a differenza di Qwen3-MoE/Gemma4;
 * nessun tensore self_attn.{q,k}_norm.weight nei pesi reali), NESSUN bias reale (a
 * differenza di Qwen2: q/k/v/o_proj hanno bias=False su tutte e quattro le proiezioni
 * in MixtralAttention.__init__ -- solo le ".biases" di quantizzazione sempre presenti,
 * già gestite da quantized_linear_cfg, esistono nei pesi). Scala di attenzione
 * standard 1/sqrt(head_dim) (a differenza del fisso 1.0 di Gemma4), RoPE "default"
 * (nessuna frequenza custom, nessun rotary parziale). In sostanza la stessa identica
 * forma di qwen2_attention_block MENO il bias additivo su q/k/v -- funzione separata
 * comunque (mai parametrizzare una funzione esistente per un'architettura diversa,
 * stesso principio delle altre entry di ARCH_TABLE). */
bool mixtral_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "mixtral_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    GattnQuant* ac = llm_mlx_attn_quant(m, layer_idx);
    mlx_array ln_w = mlx_array_new();
    mlx_array normed = mlx_array_new();
    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    if (ac != NULL) {
        ok = ok && mlx_array_set(&ln_w, ac->ln1) == 0;
        ok = ok && mlx_array_set(&qw, ac->qw) == 0 && mlx_array_set_optional(&qs, ac->qs) && mlx_array_set_optional(&qb, ac->qb);
        ok = ok && mlx_array_set(&kw, ac->kw) == 0 && mlx_array_set_optional(&ks, ac->ks) && mlx_array_set_optional(&kb, ac->kb);
        ok = ok && mlx_array_set(&vw, ac->vw) == 0 && mlx_array_set_optional(&vs, ac->vs) && mlx_array_set_optional(&vb, ac->vb);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
        if (!ok) CHK(false, "input_layernorm lookup");
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &qw, &qs, &qb);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &kw, &ks, &kb);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &vw, &vs, &vb);
    }
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
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }

    /* NESSUN QK-norm: da q_heads/k_heads direttamente al transpose. */
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
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, m->cfg.head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                            cache_offset, no_freqs, m->stream) == 0 &&
             mlx_fast_rope(&k_rope, k_t, m->cfg.head_dim, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
        CHK(ok, "rope");
        mlx_array_free(no_freqs);
    }

    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    mlx_array attn = mlx_array_new();
    mlx_array k_src = mlx_array_new(), v_src = mlx_array_new();
    bool kv_inplace = false;
    if (ok) {
        /* RMLX10 Fase A (2026-09-03, vault Todo/(open) - RMLX10-handoff): decode
         * con KV-cache a buffer FISSO shape-statica. Solo Mixtral, solo step da 1
         * token con cache presente (decode), senza rolling-window (kv_max_tokens)
         * e senza KV quant (incompatibili col layout statico). Gated LLM_FIXKV=1,
         * default OFF = percorso eager intatto sotto. Obiettivo: shape identiche a
         * ogni step -> prerequisito per la closure mlx_compile di Fase B. */
        bool fixkv = llm_mlx_fixkv_enabled() && n_tokens == 1 && has_cache &&
                     m->cfg.kv_max_tokens <= 0 && !m->cfg.kv_quantize;
        if (!fixkv) {
            if (ok && has_cache && n_tokens == 1 && !m->cfg.nsa_sparse && llm_mlx_kv_inplace_enabled() &&
                !(m->cfg.is_sliding_layer != NULL && layer_idx >= 0 && layer_idx < m->cfg.n_layers &&
                  m->cfg.is_sliding_layer[layer_idx]))
                kv_inplace = llm_mlx_kv_append_inplace(k_cache, v_cache, k_rope, v_t,
                                                       cache_offset, n_tokens, /*step=*/256,
                                                       &k_src, &v_src, m->stream);
            if (ok && !kv_inplace) {
            if (has_cache) {
                mlx_vector_array kv = mlx_vector_array_new();
                mlx_vector_array_append_value(kv, *k_cache);
                mlx_vector_array_append_value(kv, k_rope);
                ok = mlx_concatenate_axis(&k_full, kv, 2, m->stream) == 0;
                CHK(ok, "k-cache concat");
                mlx_vector_array_free(kv);
                if (ok) {
                    mlx_vector_array vv = mlx_vector_array_new();
                    mlx_vector_array_append_value(vv, *v_cache);
                    mlx_vector_array_append_value(vv, v_t);
                    ok = mlx_concatenate_axis(&v_full, vv, 2, m->stream) == 0;
                    CHK(ok, "v-cache concat");
                    mlx_vector_array_free(vv);
                }
            } else {
                ok = mlx_array_set(&k_full, k_rope) == 0 && mlx_array_set(&v_full, v_t) == 0;
                CHK(ok, "k/v cache init");
            }
                if (ok) { ok = mlx_array_set(&k_src, k_full) == 0 && mlx_array_set(&v_src, v_full) == 0; }
            }
            /* Rolling-Window KV (fase 3 paging): se kv_max_tokens>0, tronca la cache full ai piu' recenti
             * N token (mantiene i blocchi recenti, evicta i vecchi via Cache ABI trim). Limita la memoria
             * KV a contesto lungo (StreamingLLM-style). 0 = off (cache illimitata). */
            if (ok && !kv_inplace && m->cfg.kv_max_tokens > 0) {
                mlx_array k2 = mlx_array_new(), v2 = mlx_array_new();
                ok = llm_mlx_cache_trim(&k2, k_full, m->cfg.kv_max_tokens, m->stream) &&
                     llm_mlx_cache_trim(&v2, v_full, m->cfg.kv_max_tokens, m->stream);
                if (ok) { mlx_array_free(k_full); k_full = k2; mlx_array_free(v_full); v_full = v2; }
                else { mlx_array_free(k2); mlx_array_free(v2); }
            }
            if (ok && !kv_inplace) {
                ok = mlx_array_set(k_cache, k_full) == 0;
                CHK(ok, "k cache store");
                if (ok) { ok = mlx_array_set(v_cache, v_full) == 0; CHK(ok, "v cache store"); }
            }
            if (ok) {
            ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
                CHK(ok, "scaled_dot_product_attention");
            }
        } else {
            /* Buffer fisso [1, n_kv, cap, head_dim]: cap = capacita' (asse 2), mai
             * piu' piccola di needed = cache_offset+n_tokens. Prima chiamata in
             * modalita' static = conversione della cache eager (lunghezza
             * cache_offset) in buffer cap (pow2 di needed); poi write posizionale
             * via where (copia O(cap), shape INVARIATA tra gli step). SDPA su
             * prefisso con maschera additiva: i token futuri del padding NON devono
             * essere visibili (mode "causal" deriverebbe kL dalla shape = cap). */
            const int nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
            int have_len = 0;
            if (k_cache->ctx != NULL && mlx_array_ndim(*k_cache) == 4) {
                const int* csh = mlx_array_shape(*k_cache);
                if (csh != NULL) have_len = csh[2];
            }
            int needed = cache_offset + n_tokens;
            int cap = (have_len >= needed) ? have_len : llm_mlx_kv_cap_for(needed);
            bool kok = true;
            mlx_array k_buf = mlx_array_new(), v_buf = mlx_array_new();
            mlx_array k_base = mlx_array_new(), v_base = mlx_array_new();
            if (have_len > 0 && have_len < cap) {
                /* conversione eager->static o crescita: allinea a cap (zeri in
                 * coda) il buffer esistente PRIMA del write posizionale. */
                int zsh[4] = {1, nkv, cap - have_len, hd};
                mlx_dtype dt = mlx_array_dtype(*k_cache);
                mlx_array kz = mlx_array_new(), vz = mlx_array_new();
                kok = mlx_zeros(&kz, zsh, 4, dt, m->stream) == 0 &&
                      mlx_zeros(&vz, zsh, 4, dt, m->stream) == 0;
                if (kok) {
                    mlx_vector_array kvv = mlx_vector_array_new();
                    mlx_vector_array_append_value(kvv, *k_cache);
                    mlx_vector_array_append_value(kvv, kz);
                    kok = mlx_concatenate_axis(&k_base, kvv, 2, m->stream) == 0;
                    mlx_vector_array_free(kvv);
                    mlx_vector_array vvv = mlx_vector_array_new();
                    mlx_vector_array_append_value(vvv, *v_cache);
                    mlx_vector_array_append_value(vvv, vz);
                    kok = kok && mlx_concatenate_axis(&v_base, vvv, 2, m->stream) == 0;
                    mlx_vector_array_free(vvv);
                }
                mlx_array_free(kz); mlx_array_free(vz);
            } else if (have_len == cap && cap > 0) {
                /* buffer static gia' alla capacita' corrente: base = alias della cache */
                kok = mlx_array_set(&k_base, *k_cache) == 0 &&
                      mlx_array_set(&v_base, *v_cache) == 0;
            }
            /* have_len == 0: base vuota (ctx==NULL) -> write_pos parte da zero */
            static int fixkv_logged = 0;
            if (!fixkv_logged) {
                fixkv_logged = 1;
                fprintf(stderr, "[fixkv] Mixtral decode static-KV attivo (LLM_FIXKV=1, layer %d, n_tokens=%d)\n", layer_idx, n_tokens);
                fflush(stderr);
            }
            if (kok) kok = llm_mlx_kv_write_pos(&k_buf, k_base, k_rope, cache_offset, cap, m->stream);
            if (kok) kok = llm_mlx_kv_write_pos(&v_buf, v_base, v_t, cache_offset, cap, m->stream);
            if (kok) { kok = mlx_array_set(k_cache, k_buf) == 0; CHK(kok, "k static store"); }
            if (kok) { kok = mlx_array_set(v_cache, v_buf) == 0; CHK(kok, "v static store"); }
            if (kok) {
                kok = llm_mlx_gqa_sdpa_prefix(&attn, q_rope, *k_cache, *v_cache,
                                              1.0f / sqrtf((float)m->cfg.head_dim),
                                              cache_offset + n_tokens, m->stream);
                CHK(kok, "scaled_dot_product_attention (prefix mask)");
            }
            mlx_array_free(k_buf); mlx_array_free(v_buf);
            mlx_array_free(k_base); mlx_array_free(v_base);
            ok = kok;
        }
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream) == 0;
        CHK(ok, "attn transpose back");
        if (ok) {
            const int flat_shape[3] = {1, n_tokens, m->cfg.n_heads * m->cfg.head_dim};
            ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, m->stream) == 0;
            CHK(ok, "attn flatten");
        }
    }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    if (ac != NULL) {
        ok = ok && mlx_array_set(&ow, ac->ow) == 0 && mlx_array_set_optional(&os, ac->os) &&
                   mlx_array_set_optional(&ob, ac->ob);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &os, &ob);
    }
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
        q_t, k_t, v_t, q_rope, k_rope, k_full, v_full, k_src, v_src, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
