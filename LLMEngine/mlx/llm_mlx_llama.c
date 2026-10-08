/* ============================================================================
 * llm_mlx_llama.c — dominio Llama: llm_mlx_llama3_freqs_cached, llama_try_fused_qkv_rope e llama_attention_block (GQA).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_fused_rope_append.h"
#include "llm_mlx_flash_decode_kernel.h"
#include "llm_mlx_kv_quant_sdpa_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

mlx_array llm_mlx_llama3_freqs_cached(LlmMlxModel* m) {
    mlx_array out = mlx_array_new();
    if (m == NULL || m->cfg.head_dim <= 0 || !m->cfg.use_llama3_rope) return out;
    static int nocache = -1; /* LLM_ROPE_FREQS_NOCACHE=1: comportamento PRE-fix (A/B) */
    if (nocache < 0) nocache = (llm_ai_mlx_opt_env("LLM_ROPE_FREQS_NOCACHE") != NULL) ? 1 : 0;
    if (nocache) {
        mlx_array_free(out);
        return llm_mlx_llama3_rope_freqs(
            m->cfg.head_dim, m->cfg.rope_theta, m->cfg.llama3_rope_factor,
            m->cfg.llama3_low_freq_factor, m->cfg.llama3_high_freq_factor,
            m->cfg.llama3_old_context_len);
    }
    if (!m->llama3_freqs_valid) {
        mlx_array f = llm_mlx_llama3_rope_freqs(
            m->cfg.head_dim, m->cfg.rope_theta, m->cfg.llama3_rope_factor,
            m->cfg.llama3_low_freq_factor, m->cfg.llama3_high_freq_factor,
            m->cfg.llama3_old_context_len);
        if (f.ctx == NULL) return out;
        if (m->llama3_freqs.ctx != NULL) mlx_array_free(m->llama3_freqs);
        m->llama3_freqs = f;
        m->llama3_freqs_valid = true;
    }
    mlx_array_set(&out, m->llama3_freqs);
    return out;
}

static bool llama_try_fused_qkv_rope(LlmMlxModel* m, int cache_offset,
                                     mlx_array q, mlx_array k, mlx_array v,
                                     mlx_array* q_rope, mlx_array* k_rope, mlx_array* v_t) {
    /* head_dim MOLTO piccolo (es. 4 — modelli sintetici/test): il kernel Metal
     * `fused_rope_qkv_layout` non compila per quella specializzazione del template MLX (errore di
     * address space in utils.h) -> si usa il path NON fuso. Nessun modello reale ha head_dim<8. */
    if (m->cfg.head_dim <= 0 || (m->cfg.head_dim & 1) != 0 || m->cfg.head_dim < 8) return false;
    mlx_array freqs = mlx_array_new();
    float rope_base = m->cfg.rope_theta;
    if (m->cfg.use_llama3_rope) {
        mlx_array_free(freqs); /* sovrascritto: l'handle vuoto iniziale va liberato */
        freqs = llm_mlx_llama3_freqs_cached(m);
        rope_base = 0.0f; /* base disabilitato, come no_base={0,false} nel path originale */
    }
    bool fok = llm_mlx_fused_rope_append(
        q_rope, k_rope, v_t, q, k, v,
        m->cfg.n_heads, m->cfg.n_kv_heads, m->cfg.head_dim, m->cfg.head_dim,
        rope_base, 1.0f, /*traditional=*/false, freqs, cache_offset, m->stream);
    mlx_array_free(freqs);
    return fok;
}

/* FASE 1b.2 (Llama 3.x, es. Llama-3.2-1B-Instruct): stessa forma di
 * mixtral_attention_block (GQA pura -- nessun QK-norm, nessun bias reale, scala
 * standard 1/sqrt(head_dim)) -- verificato su LlamaAttention in mlx_lm/models/
 * llama.py, struttura identica a MixtralAttention -- ma con il RoPE
 * "llama3" (scaling a 3 fasce di frequenza, llm_mlx_llama3_rope_freqs) quando
 * cfg.use_llama3_rope, altrimenti RoPE default -- stesso branch pattern già usato in
 * gemma4_attention_block per passare frequenze custom a mlx_fast_rope (base
 * disabilitato, freqs esplicite). Funzione dedicata comunque, mai riusare
 * mixtral_attention_block per un'altra architettura anche quando la struttura
 * combacia: ARCH_TABLE resta un mapping 1:1 fra model_type e funzioni concrete. */
bool llama_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                   mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "llama_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
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
    bool packed = ok && ac && llm_mlx_qkv_apply(&ac->packed_qkv, normed, &q, &k, &v,
        gsz, bts, m->cfg.quant_mode, m->stream, quantized_linear_cfg);
    if (ok && !packed) { ok = quantized_linear_cfg(&q, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok && !packed) { ok = quantized_linear_cfg(&k, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok && !packed) { ok = quantized_linear_cfg(&v, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    /* G1 (piano antigravity 2026-09-08): reshape+transpose+RoPE di Q/K + transpose
     * di V fusi in UN solo kernel Metal, SOLO nel decode a 1 token (8 dispatch -> 1,
     * il collo "dispatch-count" di §3.2 Gap-Closure). Il risultato fuso vive in
     * fq/fk/fv e NON tocca `ok`: se la geometria non e' supportata o il launch
     * fallisce, need_orig resta true e il path originale sotto gira invariato.
     * LLM_FUSE_ROPE_VERIFY calcola entrambi, stampa il diff e USA l'originale. */
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new(), v_t = mlx_array_new();
    bool fuse_try = n_tokens == 1 && (llm_mlx_fuse_rope_enabled() || llm_mlx_fuse_rope_verify_enabled());
    mlx_array fq = mlx_array_new(), fk = mlx_array_new(), fv = mlx_array_new();
    bool fused_ok = false;
    if (ok && fuse_try) fused_ok = llama_try_fused_qkv_rope(m, cache_offset, q, k, v, &fq, &fk, &fv);
    bool need_orig = !fused_ok || llm_mlx_fuse_rope_verify_enabled();

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new();
    if (ok && need_orig) {
        if (n_tokens == 1) {
            /* Decode mono-token: [1,1,nh,hd] e [1,nh,1,hd] hanno lo STESSO ordine lineare
             * (i singleton non spostano dati) -> reshape diretto, NIENTE transpose.
             * Bit-identico, -1 Transpose per q/k/v per layer (il reference mlx_lm lo paga). */
            const int qh[4] = {1, m->cfg.n_heads, 1, m->cfg.head_dim};
            const int kvh[4] = {1, m->cfg.n_kv_heads, 1, m->cfg.head_dim};
            ok = mlx_reshape(&q_t, q, qh, 4, m->stream) == 0 &&
                 mlx_reshape(&k_t, k, kvh, 4, m->stream) == 0 &&
                 mlx_reshape(&v_t, v, kvh, 4, m->stream) == 0;
            CHK(ok, "qkv layout (decode, no transpose)");
        } else {
            const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
            const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
            ok = mlx_reshape(&q_heads, q, q_shape, 4, m->stream) == 0 &&
                 mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
                 mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
            if (ok) {
                const int perm[4] = {0, 2, 1, 3};
                ok = mlx_transpose_axes(&q_t, q_heads, perm, 4, m->stream) == 0 &&
                     mlx_transpose_axes(&k_t, k_heads, perm, 4, m->stream) == 0 &&
                     mlx_transpose_axes(&v_t, v_heads, perm, 4, m->stream) == 0;
            }
            CHK(ok, "qkv layout (prefill)");
        }
    }

    if (ok && need_orig) {
        mlx_array no_freqs = mlx_array_new();
        if (m->cfg.use_llama3_rope) {
            /* RoPE "llama3": frequenze custom, base disabilitato -- stesso schema
             * di gemma4_attention_block per il ramo proportional. */
            mlx_array custom_freqs = llm_mlx_llama3_freqs_cached(m);
            mlx_optional_float no_base = {0.0f, false};
            ok = mlx_fast_rope(&q_rope, q_t, m->cfg.head_dim, /*traditional=*/false, no_base, /*scale=*/1.0f,
                                cache_offset, custom_freqs, m->stream) == 0 &&
                 mlx_fast_rope(&k_rope, k_t, m->cfg.head_dim, false, no_base, 1.0f, cache_offset, custom_freqs, m->stream) == 0;
            CHK(ok, "rope (llama3)");
            mlx_array_free(custom_freqs);
        } else {
            mlx_optional_float base = {m->cfg.rope_theta, true};
            ok = mlx_fast_rope(&q_rope, q_t, m->cfg.head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                                cache_offset, no_freqs, m->stream) == 0 &&
                 mlx_fast_rope(&k_rope, k_t, m->cfg.head_dim, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
            CHK(ok, "rope (default)");
        }
        mlx_array_free(no_freqs);
    }

    if (fused_ok) {
        if (llm_mlx_fuse_rope_verify_enabled()) {
            /* CONFRONTO diagnostico: il risultato usato resta quello originale. */
            const size_t nq = (size_t)m->cfg.n_heads * m->cfg.head_dim;
            const size_t nkv = (size_t)m->cfg.n_kv_heads * m->cfg.head_dim;
            double dq = llm_mlx_max_abs_diff(fq, q_rope, nq, m->stream);
            double dk = llm_mlx_max_abs_diff(fk, k_rope, nkv, m->stream);
            double dv = llm_mlx_max_abs_diff(fv, v_t, nkv, m->stream);
            bool anom = dq < 0 || dk < 0 || dv < 0 || dq > 1e-2 || dk > 1e-2 || dv > 1e-2;
            static int vrf_printed_once = 0;
            if (!vrf_printed_once || anom) {
                vrf_printed_once = 1;
                fprintf(stderr, "[fuse-rope-verify] layer=%d cache_offset=%d q=%.6g k=%.6g v=%.6g%s\n",
                        layer_idx, cache_offset, dq, dk, dv, anom ? "  <-- MISMATCH" : "");
            }
        } else {
            ok = mlx_array_set(&q_rope, fq) == 0 && mlx_array_set(&k_rope, fk) == 0 &&
                 mlx_array_set(&v_t, fv) == 0;
            CHK(ok, "fused qkv rope adopt");
        }
    }
    mlx_array_free(fq); mlx_array_free(fk); mlx_array_free(fv);

    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    mlx_array attn = mlx_array_new();
    bool kvsl = llm_mlx_kvslice_enabled() && n_tokens == 1 && has_cache && m->cfg.kv_max_tokens <= 0;
    mlx_array k_src = mlx_array_new(), v_src = mlx_array_new();
    bool kv_inplace = false;
    if (ok && kvsl) {
        /* paged-KV step 3 (sperimentale, LLM_KVSLICE): buffer persistente a capacita' + slice_update
         * del solo blocco nuovo; SDPA sul prefisso reale (niente concat esplicito per step). */
        /* LLM_KVSEG (default OFF): se il modello viene RIUSATO per una nuova sessione
         * (prefill nuovo, passato come has_cache=false), lo stato segmentato della
         * sessione precedente va azzerato qui — gli array main/seg di g_kvseg sono
         * per-sessione, mai per-modello. Altrimenti il primo step di decode della
         * sessione nuova ripartirebbe dai segmenti della sessione morta. */
        if (llm_mlx_kvseg_enabled() && !has_cache) llm_mlx_kvseg_reset_layer(layer_idx);
        const int nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
        /* LLM_KVSEG (default OFF): cache SEGMENTATA (main statico + segmento O(SEG)) sopra
         * il layout KVSLICE. Solo decode mono-token. Fallimento/non-applicabilita' ->
         * fallback silenzioso al KVSLICE standard, mai errore bloccante. */
        mlx_array seg_k1 = mlx_array_new(), seg_v1 = mlx_array_new();
        mlx_array seg_k2 = mlx_array_new(), seg_v2 = mlx_array_new();
        int seg_pieces = 0;
        bool seg_used = false;
        if (ok && llm_mlx_kvseg_enabled()) {
            if (llm_mlx_kvseg_step(m->stream, nkv, hd, layer_idx, cache_offset, n_tokens,
                                   k_rope, v_t, k_cache, v_cache,
                                   &seg_k1, &seg_v1, &seg_k2, &seg_v2, &seg_pieces)) {
                seg_used = true;
                if (llm_da_enabled()) g_da.kv_seg++;
            }
        }
        if (seg_used && llm_mlx_kvseg_verify_enabled()) {
            /* LLM_KVSEG_VERIFY (default OFF): confronta il merge a due pezzi contro la
             * SDPA sul CONCAT esplicito dei pezzi reali (main+seg) -- e' il baseline
             * onesto: la cache dichiarativa *k_cache NON viene aggiornata nel path
             * segmentato (i token vivono in main/seg), quindi un confronto contro
             * *k_cache sarebbe contro dati stale. Stampa il diff e PROSEGUE col path
             * segmentato (non cambia il flusso: serve solo a validare il merge). */
            if (seg_pieces == 2) {
                mlx_array bk = mlx_array_new(), bv = mlx_array_new();
                mlx_vector_array kb = mlx_vector_array_new(), vb = mlx_vector_array_new();
                mlx_vector_array_append_value(kb, seg_k2);   /* main */
                mlx_vector_array_append_value(kb, seg_k1);   /* segmento corrente */
                mlx_vector_array_append_value(vb, seg_v2);
                mlx_vector_array_append_value(vb, seg_v1);
                bool cok = mlx_concatenate_axis(&bk, kb, 2, m->stream) == 0 &&
                           mlx_concatenate_axis(&bv, vb, 2, m->stream) == 0;
                mlx_vector_array_free(kb); mlx_vector_array_free(vb);
                if (cok) {
                    mlx_array seg_attn = mlx_array_new();
                    mlx_array base_attn = mlx_array_new();
                    bool sok2 = llm_mlx_kvseg_merge_two(m, q_rope, seg_k1, seg_v1, seg_k2, seg_v2, &seg_attn);
                    bool bok = llm_mlx_gqa_sdpa(&base_attn, q_rope, bk, bv,
                                                1.0f / sqrtf((float)m->cfg.head_dim), 0, m->stream);
                    if (sok2 && bok) {
                        size_t ndiff = (size_t)m->cfg.n_heads * (size_t)m->cfg.head_dim;
                        double md = llm_mlx_max_abs_diff(seg_attn, base_attn, ndiff, m->stream);
                        fprintf(stderr, "[kvseg-verify] layer=%d off=%d merge_vs_concat max_abs_diff=%s%.6g%s\n",
                                layer_idx, cache_offset, (md < 0 ? "<eval-fail " : ""), md,
                                (md >= 0 && md > 1e-2 ? "  <-- MISMATCH" : ""));
                    }
                    mlx_array_free(seg_attn); mlx_array_free(base_attn);
                }
                mlx_array_free(bk); mlx_array_free(bv);
            } else {
                /* 1 pezzo: segmento vuoto, il merge è identico alla SDPA singola. */
                fprintf(stderr, "[kvseg-verify] layer=%d off=%d pieces=1 (segmento vuoto, niente da confrontare)\n",
                        layer_idx, cache_offset);
            }
        }
        if (seg_used) {
            if (seg_pieces == 2) {
                ok = llm_mlx_kvseg_merge_two(m, q_rope, seg_k1, seg_v1, seg_k2, seg_v2, &attn);
                CHK(ok, "kvseg merge two-piece attention");
            } else {
                ok = llm_mlx_gqa_sdpa(&attn, q_rope, seg_k1, seg_v1,
                                      1.0f / sqrtf((float)m->cfg.head_dim), 0, m->stream);
                CHK(ok, "scaled_dot_product_attention (kvseg 1-piece)");
            }
        }
        mlx_array_free(seg_k1); mlx_array_free(seg_v1);
        mlx_array_free(seg_k2); mlx_array_free(seg_v2);
        if (seg_used && ok) {
            /* Path segmentato completato: salta il KVSLICE standard (il goto salta
             * alla fine del ramo, dove il concat baseline NON deve girare). */
            goto kvseg_done;
        }
        if (seg_used && !ok) {
            /* Fail-closed: il path segmentato NON può essere sostituito dal KVSLICE
             * standard qui sotto, perche' la cache dichiarativa di *k_cache NON e' stata
             * aggiornata dall'append segmentato (i token vivono in main/seg): ripiegare
             * sul concat baseline produrrebbe attention su una cache STALE (token
             * mancanti) = output silenziosamente sbagliato. Errore esplicito. */
            CHK(ok, "kvseg: merge/1-piece fallito con path segmentato gia' attivo");
        }
        int have_len = 0;
        if ((*k_cache).ctx != NULL && mlx_array_ndim(*k_cache) == 4) { const int* csh = mlx_array_shape(*k_cache); if (csh) have_len = csh[2]; }
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
            /* FIX 2026-09-07 notte: mlx_array_set duplica l'handle (condivide lo shared_ptr
             * array_desc_ con l'originale) mantenendo *k_cache vivo insieme a k_base --
             * refcount 2 rompe silenziosamente la donazione del buffer dentro
             * mlx_slice_update sotto (is_donatable() richiede array_desc_.use_count()==1 E
             * data.use_count()==1, array.h:294-296) e mlx_slice_update ripiega sulla copia
             * completa O(cap) invece di alias O(1)+scrittura O(nuovi token). Fix: libero
             * *k_cache SUBITO DOPO aver preso k_base, cosi' al momento dello slice_update
             * sotto il refcount e' tornato a 1 e la donazione puo' scattare -- stesso schema
             * "free stale views before the write" documentato in mlx-serve
             * (github.com/ddalcu/mlx-serve, transformer.zig, dove lo stesso bug costava il
             * 45% del warm TTFT a contesto lungo). Non cambia alcun valore, solo se la scrittura
             * dona il buffer o lo ricopia. */
            ok = ok && mlx_array_set(&k_base, *k_cache) == 0 && mlx_array_set(&v_base, *v_cache) == 0;
            if (ok) {
                mlx_array_free(*k_cache); mlx_array_free(*v_cache);
                *k_cache = mlx_array_new(); *v_cache = mlx_array_new();
            }
        } else if (cap > 0) {
            int zsh[4] = {1, nkv, cap, hd};
            mlx_dtype dt = (llm_mlx_act_dt(m->cfg.bfloat16_input));
            ok = ok && mlx_zeros(&k_base, zsh, 4, dt, m->stream) == 0 &&
                       mlx_zeros(&v_base, zsh, 4, dt, m->stream) == 0;
        }
        mlx_array k_buf = mlx_array_new(), v_buf = mlx_array_new();
        /* BUG PRE-ESISTENTE trovato e fissato 2026-09-07 notte (indipendente dal fix di
         * refcount sopra): la guardia confrontava `cache_offset >= have_len`, ma `have_len`
         * e' letto dalla shape FISICA del buffer (riga 6160) -- dopo un grow con padding
         * (ramo "have_len < cap" sopra) diventa la capacita' piena con padding, non il
         * conteggio di token reali, e resta quella capacita' padded per ogni chiamata
         * successiva finche' non si rientra nel ramo grow. In stato stazionario
         * (have_len==cap, il ramo appena sopra corretto per il refcount) questo rende
         * `cache_offset >= have_len` insieme a `cache_offset + 1 <= cap` una coppia
         * matematicamente impossibile (equivalgono a cache_offset>=cap E cache_offset<cap),
         * quindi lo slice_update non veniva MAI eseguito oltre il primo token dopo ogni
         * raddoppio -- LLM_KVSLICE non ha mai realmente decodificato oltre 1 token per bucket
         * di capacita'. Il conteggio reale di token e' gia' `cache_offset` (contratto di
         * questa funzione, sempre corretto indipendentemente dal padding) -- la guardia
         * corretta e' solo "la posizione di scrittura sta dentro la capacita'". */
        /* LLM_KVSLICE_TIME (2026-09-08 notte, diagnostico): timing fine-grained dei 3
         * sotto-passaggi del path (slice_update scrittura, slice estrazione lettura, SDPA)
         * con mlx_array_eval forzato ad ogni passaggio -- per isolare QUALE cresce con la
         * posizione, dato che LLM_PHASE aveva isolato "attn cresce, ffn no" ma non il
         * sotto-passaggio esatto dentro attn. Cambia il pattern di sync (aggiunge eval
         * intermedi non presenti nel decode normale), quindi i numeri assoluti non sono il
         * tok/s reale -- solo la FORMA relativa (quale fase cresce di piu' con N) e' valida.
         * ATTENZIONE (trovato lo stesso giorno, vedi TODO KvPagePool §9): anche la modalita'
         * "=2" (un solo eval qui) forza comunque un eval PER LAYER, mentre il decode reale
         * (llm_ai_mlx_session_generate -> forward_one_token -> final_step) sincronizza UNA
         * SOLA VOLTA PER TOKEN su tutti i layer insieme, lasciando MLX libero di sovrapporre
         * l'invio dei comandi Metal di piu' layer. Questo strumento perde quel pipelining
         * incrociato e puo' quindi SOVRASTIMARE il costo reale (contraddizione trovata: il
         * sotto-blocco qui misurato, senza le proiezioni Q/K/V/O, risultava piu' costoso del
         * blocco intero misurato da LLM_PHASE, che le include -- impossibile se non per
         * questo artefatto). Usare solo per un segnale di massima, non per numeri assoluti. */
        const char* ktime_env = llm_env_get("LLM_KVSLICE_TIME");
        bool ktime = ktime_env != NULL;
        bool ktime_multi = ktime && strcmp(ktime_env, "2") != 0; /* =2: un solo eval finale, per
            escludere che gli eval intermedi stessi introducano il costo osservato */
        struct timespec kt0, kt1;
        double t_write = 0, t_read = 0, t_sdpa = 0;
        if (ktime) clock_gettime(CLOCK_MONOTONIC, &kt0);
        if (ok && cap > 0 && cache_offset >= 0 && cache_offset + 1 <= cap) {
            int st4[4] = {0, 0, cache_offset, 0}, sp4[4] = {1, nkv, cache_offset + 1, hd}, sv4[4] = {1, 1, 1, 1};
            ok = mlx_slice_update_inplace(&k_base, k_rope, st4, 4, sp4, 4, sv4, 4, m->stream) == 0 &&
                 mlx_slice_update_inplace(&v_base, v_t, st4, 4, sp4, 4, sv4, 4, m->stream) == 0;
            if (ktime_multi && ok) { ok = mlx_array_eval(k_base) == 0 && mlx_array_eval(v_base) == 0; }
        } else { ok = false; }
        if (ktime) { clock_gettime(CLOCK_MONOTONIC, &kt1); t_write = (kt1.tv_sec-kt0.tv_sec)+(kt1.tv_nsec-kt0.tv_nsec)/1e9; kt0=kt1; }
        if (ok) { ok = mlx_array_set(k_cache, k_base) == 0 && mlx_array_set(v_cache, v_base) == 0; }
        mlx_array_free(k_base); mlx_array_free(v_base); mlx_array_free(k_buf); mlx_array_free(v_buf);
        if (ok) {
            /* Estrae il prefisso [0..needed) dal buffer padded a `cap` come slice strided
             * e la passa COSI' COM'E' a fast_scaled_dot_product_attention -- niente
             * mlx_contiguous. Trovato il 2026-09-07 notte che mlx_contiguous qui era una
             * copia difensiva mai verificata necessaria: fast_sdpa accetta K/V non contigui
             * senza errori. Verificato: output byte-identico al baseline concat (Llama-3.2-1B,
             * N=2048 e N=30000, confronto letterale dei token id greedy) sia con che senza
             * questa modifica. Misurato A/B pulito (decode128, niente crescite ripetute nella
             * finestra): con mlx_contiguous ancora piu' lento del baseline in tutto il range
             * (0.77x-0.88x, N=5000-13000); SENZA (questo path) supera il baseline e il
             * vantaggio CRESCE col contesto -- N=5000 0.95x (quasi pari), N=13000 1.19-1.22x,
             * N=30000 1.85-1.95x (confermato su 2 run interleaved). La donazione risolve la
             * scrittura, questo risolve la lettura -- insieme chiudono il gap O(context) del
             * baseline concat. Dettagli in TODO KvPagePool §1. */
            mlx_array kp = mlx_array_new(), vp = mlx_array_new();
            int p0[4] = {0, 0, 0, 0}, p1[4] = {1, nkv, needed, hd}, ps[4] = {1, 1, 1, 1};
            ok = mlx_slice(&kp, *k_cache, p0, 4, p1, 4, ps, 4, m->stream) == 0 &&
                 mlx_slice(&vp, *v_cache, p0, 4, p1, 4, ps, 4, m->stream) == 0;
            if (ktime_multi && ok) { ok = mlx_array_eval(kp) == 0 && mlx_array_eval(vp) == 0; }
            if (ktime) { clock_gettime(CLOCK_MONOTONIC, &kt1); t_read = (kt1.tv_sec-kt0.tv_sec)+(kt1.tv_nsec-kt0.tv_nsec)/1e9; kt0=kt1; }
            if (ok) {
                ok = llm_mlx_gqa_sdpa(&attn, q_rope, kp, vp, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
                CHK(ok, "scaled_dot_product_attention (kvslice)");
                if (ktime && ok) ok = mlx_array_eval(attn) == 0;
            }
            if (ktime) { clock_gettime(CLOCK_MONOTONIC, &kt1); t_sdpa = (kt1.tv_sec-kt0.tv_sec)+(kt1.tv_nsec-kt0.tv_nsec)/1e9; }
            if (ktime) fprintf(stderr, "[LLM_KVSLICE_TIME] layer=%d cache_offset=%d cap=%d write=%.6fs read=%.6fs sdpa=%.6fs\n",
                    layer_idx, cache_offset, cap, t_write, t_read, t_sdpa);
            mlx_array_free(kp); mlx_array_free(vp);
        }
        kvseg_done:
            ;   /* label per il goto del path segmentato: salta il concat baseline */
    } else {
    /* #68 LLMMETAL_KVMETAL_AUTH (default OFF): attention dal pool Metal fp16 O(1) AUTORITATIVO.
     * Quando il pool di QUESTO layer e' stato popolato dal prefill e la posizione sta dentro
     * la capacita' fissa (=> il trim eager sarebbe un no-op: pool ed eager attendono sulle
     * STESSE posizioni), l'output viene dalle pagine del pool e si SALTA l'intero ramo eager
     * (concat/trim/store/SDPA), cioe' il Concatenate O(pos) che e' il 98% dei byte/decode denso
     * (causa #61). CONTRATT: chi attiva AUTH deve dare LLM_KV_MAX_TOKENS >= lunghezza sessione;
     * se il passo del pool fallisce (capacita' insufficiente, contratto violato) ok=false e
     * errore esplicito -- mai un risultato silenzioso costruito su pagine vuote.
     * >>> FALSIFICATO 2026-09-10 (Llama-3.2-1B, finestra condivisa ma ordine stabile, 2 rep):
     * ctx=4096 AUTH ~91 tok/s vs LLM_KVF16 ~332 vs baseline eager ~255; ctx=256 (dove NON c'e'
     * Concatenate O(pos) da risparmiare) AUTH ~112 vs ~458. IDS greedy BYTE-IDENTICI al baseline
     * (path CORRETTO), ma 2,7-4x PIU' LENTO ANCHE CON IL CONCAT DEL TUTTO SOPPRESSO. Causa: non
     * era il concat -- e' il bridge stesso: a ogni passo, per ogni layer, marshal host->device di
     * q/k/v + UN llm_kvmetal_write + decode_attention SINCRONO (waitUntilCompleted) spezzano il
     * pipelining MLX "un solo sync per token su tutti i layer"; il costo fisso per-layer-per-token
     * DOMINA ogni risparmio O(pos). Stesso esito gia' documentato su qwen2/DeepSeek (:3531, R2,06
     * a ctx=16384). La leva reale e' una scrittura di pagina IN-DEVICE dal grafo (niente
     * round-trip host per token), NON questo bridge. RESTA default-OFF come evidenza misurata. */
    bool kv_used_pool = false;
    if (ok && llm_mlx_kvmetal_auth_enabled() && llm_mlx_kvmetal_auth_layer_ready(layer_idx)
        && n_tokens == 1 && has_cache
        && !llm_mlx_is_act16(mlx_array_dtype(q_rope))
        && m->cfg.kv_max_tokens > 0 && m->cfg.n_kv_heads > 0
        && m->cfg.head_dim % 32 == 0 && m->cfg.head_dim <= 256
        && m->cfg.n_heads % m->cfg.n_kv_heads == 0
        && cache_offset < m->cfg.kv_max_tokens
        && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers) {
        mlx_array attn_pool = mlx_array_new();
        bool step_ok = llm_mlx_kvmetal_step(g_kvmetal_ctx, layer_idx, m, cache_offset, q_rope, k_rope, v_t, &attn_pool);
        if (step_ok && llm_da_enabled()) g_da.kv_metal++;
        if (step_ok) { ok = mlx_array_set(&attn, attn_pool) == 0; CHK(ok, "kvmetal-auth decode attention"); kv_used_pool = ok; }
        else { ok = false; CHK(ok, "kvmetal-auth: passo pool fallito (LLM_KV_MAX_TOKENS < lunghezza sessione)"); }
        mlx_array_free(attn_pool);
    }
    if (ok && !kv_used_pool) {
    /* KVF16 solo con attivazione fp32: li' dimezza DAVVERO i byte (4->2). Con attivazioni
     * bf16/f16 la cache e' gia' a 2 byte/elemento: il cast non risparmia nulla e aggiunge
     * 4 AsType per layer (144/token su Qwen3-8B, 12% dei nodi; misurato LLM_DISPATCH
     * 2026-09-19) allontanando le numeriche dal riferimento mlx_lm (cache bf16). */
    if (ok && llm_mlx_kv_f16_enabled() && mlx_array_dtype(k_rope) == MLX_FLOAT32) {
        /* LLM_KVF16 (#61): cache KV in fp16 per dimezzare il Concatenate O(pos). k_rope/v_t
         * riassegnati a fp16: il concat/init sotto li usa cosi' com'e'. q->fp16 e ritorno
         * attn->act sono armonizzati in llm_mlx_gqa_sdpa_sinks. Default-OFF: mai in prod. */
        mlx_array kr16 = mlx_array_new(), vt16 = mlx_array_new();
        if (mlx_astype(&kr16, k_rope, MLX_FLOAT16, m->stream) == 0 &&
            mlx_astype(&vt16, v_t,   MLX_FLOAT16, m->stream) == 0) {
            /* FIX ownership (2026-09-13): `k_rope = kr16` sovrascriveva l'handle raw perdendo
             * l'originale FP32/bf16 (buffer mai liberato). mlx_array_set condivide e sostituisce
             * in-place, poi si libera il nuovo handle. Vedi Bugs/(open) - MLX-Core-Prefill-Buffer-Retention.md. */
            mlx_array_set(&k_rope, kr16); mlx_array_set(&v_t, vt16);
            mlx_array_free(kr16); mlx_array_free(vt16);
        }
        else { mlx_array_free(kr16); mlx_array_free(vt16); }
    }
    if (ok && has_cache && n_tokens == 1 && !m->cfg.nsa_sparse && llm_mlx_kv_inplace_enabled() &&
        !(m->cfg.is_sliding_layer != NULL && layer_idx >= 0 && layer_idx < m->cfg.n_layers &&
          m->cfg.is_sliding_layer[layer_idx]))
        kv_inplace = llm_mlx_kv_append_inplace(k_cache, v_cache, k_rope, v_t,
                                               cache_offset, n_tokens, /*step=*/256,
                                               &k_src, &v_src, m->stream);
    if (ok && !kv_inplace) {
    if (ok) {
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

    /* Retention (SnapKV), dichiarabile in modo DICHIARATIVO via LLM_KV_POLICY="retain=snapkv,
     * budget=...,sink=...,recent=...,obs=..." (policy unificata, vedi llm_mlx_ops.h) oppure via i
     * singoli env LLM_KVRETAIN*. Tiene budget = sink (iniziali) + top-by-score (proxy query sulle
     * ultime `obs` righe di K) + recent (finali); riduce la RSS a contesto lungo. Lossy -> opt-in.
     * `cache_offset` resta la posizione ASSOLUTA (solo RoPE); cache store/SDPA usano il tenuto (token
     * tutti passati -> nessuna mask). Solo se il rolling trim (kv_max_tokens) e' spento. */
    if (ok && m->cfg.kv_max_tokens <= 0) {
        static int pol_loaded = 0;
        static LlmMlxKvPolicy pol;
        if (!pol_loaded) { pol = llm_mlx_kv_policy_from_env(); pol_loaded = 1; }
        bool do_retain = (pol.retain == 1) || llm_mlx_kvretain_enabled();
        if (do_retain) {
            int budget = (pol.retain == 1) ? pol.retain_budget : llm_mlx_kvretain_param("LLM_KVRETAIN_BUDGET", 1024);
            int sink = (pol.retain == 1) ? pol.retain_sink : llm_mlx_kvretain_param("LLM_KVRETAIN_SINK", 4);
            int recent = (pol.retain == 1) ? pol.retain_recent : llm_mlx_kvretain_param("LLM_KVRETAIN_RECENT", 128);
            int obs = (pol.retain == 1) ? pol.retain_obs : llm_mlx_kvretain_param("LLM_KVRETAIN_OBS", 32);
            int slen = (mlx_array_ndim(*k_cache) == 4) ? mlx_array_shape(*k_cache)[2] : 0;
            if (slen > budget) {
                mlx_array rk = mlx_array_new(), rv = mlx_array_new();
                if (llm_mlx_kv_retain(&rk, &rv, *k_cache, *v_cache, budget, sink, recent, obs, m->stream)) {
                    ok = mlx_array_set(k_cache, rk) == 0 && mlx_array_set(v_cache, rv) == 0;
                    CHK(ok, "kvretain store");
                }
                mlx_array_free(rk); mlx_array_free(rv);
            }
        }
    }

    if (ok && n_tokens > 1) {
        /* LLMMETAL_KVMETAL: popola il pool per un forward BATCH (prefill/append) -- vedi commento
         * su llm_mlx_kvmetal_populate_batch sopra per il bug che questo chiude (pool mai
         * scritto durante il prefill, decode successivo attendeva su pagine vuote). Stessa
         * eligibilita' del ramo decode sotto, MENO n_tokens==1 (qui e' l'opposto, >1) --
         * fallback silenzioso se una condizione non e' soddisfatta, mai un errore bloccante
         * (ok non viene toccato da un fallimento qui: il forward prosegue con la SDPA
         * normale, semplicemente il pool Metal resta disallineato per quel layer/sessione). */
        bool kvmetal_populate = (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled()
                                 || llm_mlx_kvmetal_auth_enabled())
            && g_kvmetal_ctx != NULL && !g_kvmetal_disabled
            && !llm_mlx_is_act16(mlx_array_dtype(k_rope))
            && m->cfg.kv_max_tokens > 0 && m->cfg.n_kv_heads > 0
            && m->cfg.head_dim % 32 == 0 && m->cfg.head_dim <= 256
            && m->cfg.n_heads % m->cfg.n_kv_heads == 0
            && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers;
        if (kvmetal_populate &&
            llm_mlx_kvmetal_populate_batch(g_kvmetal_ctx, layer_idx, m, cache_offset, n_tokens, k_rope, v_t)
            && g_kvmetal_pop_ok != NULL)
            g_kvmetal_pop_ok[layer_idx] = 1;   /* #68: pool di questo layer completo -> auth pronto */
    }

    if (ok) {
        /* LLMMETAL_KVMETAL (sperimentale, default OFF): eligibilita' runtime -- ogni condizione
         * che fallisce fa fallback silenzioso alla SDPA normale sotto, mai un errore.
         * kvsl e' gia' garantito false qui (siamo nel ramo else di "if (ok && kvsl)"). */
        bool kvmetal_active = (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled())
            && g_kvmetal_ctx != NULL && !g_kvmetal_disabled
            && n_tokens == 1 && has_cache
            && !llm_mlx_is_act16(mlx_array_dtype(q_rope))
            && m->cfg.kv_max_tokens > 0 && m->cfg.n_kv_heads > 0
            && m->cfg.head_dim % 32 == 0 && m->cfg.head_dim <= 256
            && m->cfg.n_heads % m->cfg.n_kv_heads == 0
            && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers;
        mlx_array attn_metal = mlx_array_new();
        bool have_attn_metal = false;
        if (kvmetal_active)
            have_attn_metal = llm_mlx_kvmetal_step(g_kvmetal_ctx, layer_idx, m, cache_offset, q_rope, k_rope, v_t, &attn_metal);
        if (have_attn_metal && llm_da_enabled()) g_da.kv_metal++;

        if (have_attn_metal && !llm_mlx_kvmetal_verify_enabled()) {
            /* LLMMETAL_KVMETAL puro: usa il risultato del kernel Metal al posto della SDPA. */
            ok = mlx_array_set(&attn, attn_metal) == 0;
            CHK(ok, "kvmetal decode attention");
        } else {
            ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
            CHK(ok, "scaled_dot_product_attention");
            if (ok && have_attn_metal && llm_mlx_kvmetal_verify_enabled()) {
                /* LLMMETAL_KVMETAL_VERIFY: confronta col risultato Metal, stampa il diff su
                 * stderr, NON cambia il risultato usato (attn resta quello normale) -- zero
                 * rischio, serve a validare la correttezza su un modello reale. */
                mlx_array af = mlx_array_new(), amf = mlx_array_new();
                if (mlx_astype(&af, attn, MLX_FLOAT32, m->stream) == 0 &&
                    mlx_astype(&amf, attn_metal, MLX_FLOAT32, m->stream) == 0 &&
                    mlx_array_eval(af) == 0 && mlx_array_eval(amf) == 0) {
                    const float* ad = mlx_array_data_float32(af);
                    const float* bd = mlx_array_data_float32(amf);
                    size_t nel = (size_t)m->cfg.n_heads * m->cfg.head_dim;
                    if (ad && bd) {
                        double maxdiff = 0;
                        for (size_t vi = 0; vi < nel; vi++) {
                            double d = fabs((double)ad[vi] - (double)bd[vi]);
                            if (d > maxdiff) maxdiff = d;
                        }
                        fprintf(stderr, "[kvmetal-verify] layer=%d cache_offset=%d max_abs_diff=%.6g\n",
                                layer_idx, cache_offset, maxdiff);
                    }
                }
                mlx_array_free(af); mlx_array_free(amf);
            }
        }
        mlx_array_free(attn_metal);
    }
    }
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int flat_shape[3] = {1, n_tokens, m->cfg.n_heads * m->cfg.head_dim};
        if (n_tokens == 1) {
            /* Decode: attn [1,nh,1,hd] -> [1,1,nh*hd] e' lo stesso ordine lineare -> reshape
             * diretto, niente transpose (bit-identico). */
            ok = mlx_reshape(&attn_flat, attn, flat_shape, 3, m->stream) == 0;
            CHK(ok, "attn flatten (decode, no transpose)");
        } else {
            const int perm[4] = {0, 2, 1, 3};
            ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream) == 0;
            CHK(ok, "attn transpose back");
            if (ok) ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, m->stream) == 0;
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

/* FASE 1b.4.1 (Gemma4): FFN GeGLU (gelu_approx invece di SwiGLU) + il sandwich a due
 * norme di Gemma4 (pre_feedforward_layernorm PRIMA della FFN, post_feedforward_layernorm
 * SUL RISULTATO della FFN prima del residuo -- diverso da dense_swiglu_ffn, che ha una
 * sola norma prima e nessuna dopo) + la moltiplicazione finale per layer_scalar (uno
 * scalare appreso per layer, applicato incondizionatamente in coda a OGNI layer Gemma4 --
 * verificato in gemma4_text.py DecoderLayer.__call__, riga finale prima del return).
 *
 * gelu_approx(x) = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) -- verificato
 * identico a mlx.nn.layers.activations.gelu_approx. */
