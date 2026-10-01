/* ============================================================================
 * Dominio Qwen2 attenzione + cache pesi attenzione: qwen2_attention_block (GQA, nessuna QK-norm), llm_mlx_attn_quant, llm_mlx_gemma4_attn_quant.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
bool qwen2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen2_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
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

    /* Bias reale (NON di quantizzazione): self_attn.{q,k,v}_proj.bias, un
     * tensore float separato da .biases (quantizzazione, già in qb/kb/vb). */
    mlx_array q_bias = mlx_array_new(), k_bias = mlx_array_new(), v_bias = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj.bias", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &q_bias);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj.bias", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &k_bias);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj.bias", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &v_bias);
    CHK(ok, "q/k/v_proj.bias lookup");

    mlx_array q0 = mlx_array_new(), k0 = mlx_array_new(), v0 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q0, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    if (ok) { ok = quantized_linear_cfg(&k0, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    if (ok) { ok = quantized_linear_cfg(&v0, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
    /* §D R6 (multi-arch): LoRA su q/k/v_proj (come llama/qwen3) — prima solo o_proj. */

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) {
        ok = mlx_add(&q, q0, q_bias, m->stream) == 0 &&
             mlx_add(&k, k0, k_bias, m->stream) == 0 &&
             mlx_add(&v, v0, v_bias, m->stream) == 0;
        CHK(ok, "q/k/v bias add");
    }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }

    /* NESSUN QK-norm qui (a differenza di qwen3_moe_attention_block): si va
     * direttamente da q_heads/k_heads al transpose. */
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
    /* LLM_KVSLICE esteso a qwen2 (2026-09-07 notte fonda, per DeepSeek-R1-Distill-Qwen-32B):
     * stesso pattern verificato su Llama (vince, GQA 4x) e Qwen3-MoE (vince poco, GQA 8x).
     * Vedi TODO KvPagePool §1 per il ragionamento completo sul GQA ratio come predittore.
     * >>> FALSIFICATO 2026-09-08 (A/B 12 run, auto-pair f16 su entrambe le braccia, quindi
     * solo KV): su questo modello lo slice PEGGIORA il decode a contesto lungo — a N=4096 e' in
     * pari, a N=16384 vale 0,44x / 0,68x / 0,85x contro la concat, 3 rep appaiate su 3 nella
     * stessa direzione; il fattore di degrado 4096->16384 passa da 0,5625 (concat) a 0,355
     * (slice), cioe' lo slice AMPLIFICA la dipendenza da pos invece di toglierla. Motivo
     * plausibile (lettura del codice QUI SOTTO, non misurata: LLM_KVSLICE_TIME non e' cablato su
     * questo ramo): mlx_slice_update e' un update FUNZIONALE (materializza l'intero buffer da
     * `cap`), la crescita di cap avviene ancora via concat di zero, e in piu' servono due
     * mlx_slice di lettura per K e V -> ~3 materializzazioni O(pos) per token per layer contro
     * le 2 del ramo concat. CONSEGUENZA SECONDARIA: cade anche il predittore "GQA ratio" su cui
     * si base' l'estensione — qwen2 e' 40/8 = 5x, confinante con Llama (4x, dove VINCE), e
     * invece PERDE 2,3x. Il ramo resta quindi spento di default e NON va promosso ne' esteso
     * per analogia ad altre famiglie: ogni famiglia richiede la sua A/B appaiata. */
    bool kvsl = llm_mlx_kvslice_enabled() && n_tokens == 1 && has_cache && m->cfg.kv_max_tokens <= 0;
    mlx_array k_src = mlx_array_new(), v_src = mlx_array_new();
    bool kv_inplace = false;
    if (ok && kvsl) {
        const int nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
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
                ok = llm_mlx_gqa_sdpa(&attn, q_rope, kp, vp, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
                CHK(ok, "scaled_dot_product_attention (kvslice)");
            }
            mlx_array_free(kp); mlx_array_free(vp);
        }
    } else {
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

    if (ok && n_tokens > 1) {
        /* LLMMETAL_KVMETAL esteso a qwen2 (2026-09-09, DeepSeek-R1-Distill-Qwen-32B): popola il
         * pool Metal per un forward BATCH (prefill/append) -- stesso bug che chiude il path
         * llama/glm (pool scritto solo nel decode -> il primo passo attendeva pagine vuote).
         * Eligibilita' = quella del ramo decode MENO n_tokens==1. AmMESSI fp32 e fp16: il
         * kernel interna fa astype->FLOAT16, quindi su sorgente f16 e' identita' (DeepSeek e'
         * promosso a f16 -> KV nativa fp16, zero perdita); ESCLUSO bfloat16 (bf16->fp16 perde
         * mantissa ed esce dal range fp16 -> non validato, fallback a SDPA). */
        mlx_dtype pdt = mlx_array_dtype(k_rope);
        bool kvmetal_populate = (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled())
            && g_kvmetal_ctx != NULL && !g_kvmetal_disabled
            && (pdt == MLX_FLOAT32 || pdt == MLX_FLOAT16)
            && m->cfg.kv_max_tokens > 0 && m->cfg.n_kv_heads > 0
            && m->cfg.head_dim % 32 == 0 && m->cfg.head_dim <= 256
            && m->cfg.n_heads % m->cfg.n_kv_heads == 0
            && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers;
        if (kvmetal_populate)
            llm_mlx_kvmetal_populate_batch(g_kvmetal_ctx, layer_idx, m, cache_offset, n_tokens, k_rope, v_t);
    }

    if (ok) {
        /* LLMMETAL_KVMETAL decode step (sperimentale, default OFF): sostituisce la SDPA con il
         * kernel Metal a pagine (crescita O(1) ammortizzata, non concat O(pos) del ramo
         * sotto). Falsificato il KVSLICE su qwen2 (amplifica il degrado O(pos)); questo e'
         * un meccanismo DIVERSO (pool fisso + split-K decode).
         * >>> FALSIFICATO ANCHE QUESTO 2026-09-09 (A/B appaiato 3 rep, ctx=16384, DeepSeek
         * f16, cap 20000 non-troncante, finestra priva di procs mlx concorrenti): il paging
         * E' CORRETTO (LLMMETAL_KVMETAL_VERIFY max_abs_diff ~1e-4 su tutti i layer = rumore fp16,
         * l'astype interna ->FLOAT16 su sorgente f16 e' identita') ma E' PIU' LENTO: decode
         * mediana 3,6 tok/s (KVMETAL) vs 7,4 (concat) = ~R2,06 DI PENALIZZO; prefill 348 vs
         * 490 tok/s. Causa: a ogni passo di decode, per ognuno dei 60 layer, il path marshal
         * q/k/v in fp16 + UN llm_kvmetal_write (host->device) + split-K read -> il costo di
         * dispatch per-token-per-layer DOMINA il risparmio sulla materializzazione O(pos) che
         * il paging voleva evitare. Su un denso 32B a 60 layer il paging non paga. RESTA OFF
         * e NON va promosso ne' esteso (qwen3_moe/Coder-30B e' bf16 + MoE: peggio ancora, il
         * guard qui sotto ammette solo fp32/f16 e blocca bf16 proprio per la perdita di
         * mantissa bf16->fp16). Il gap O(pos) del contesto lungo su DeepSeek resta quindi
         * SENZA leva nell'engine attuale: serve un kernel che scriva la pagina IN-DEVICE dal
         * grafo (niente round-trip host per token), non questo bridge. */
        mlx_dtype qdt = mlx_array_dtype(q_rope);
        bool kvmetal_active = (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled())
            && g_kvmetal_ctx != NULL && !g_kvmetal_disabled
            && n_tokens == 1 && has_cache
            && (qdt == MLX_FLOAT32 || qdt == MLX_FLOAT16)
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
            ok = mlx_array_set(&attn, attn_metal) == 0;
            CHK(ok, "kvmetal decode attention");
        } else {
            ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
            CHK(ok, "scaled_dot_product_attention");
            if (ok && have_attn_metal && llm_mlx_kvmetal_verify_enabled()) {
                double md = llm_mlx_max_abs_diff(attn, attn_metal,
                    (size_t)m->cfg.n_heads * m->cfg.head_dim, m->stream);
                if (md >= 0) fprintf(stderr, "[kvmetal-verify] qwen2 layer=%d cache_offset=%d max_abs_diff=%.6g\n",
                                      layer_idx, cache_offset, md);
            }
        }
        mlx_array_free(attn_metal);
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

    if (ok) { ok = mlx_add(out_h, x, attn_out, m->stream) == 0; CHK(ok, "residual add (attn)"); }

    mlx_array all[] = {
        ln_w, normed, qw, qs, qb, kw, ks, kb, vw, vs, vb,
        q_bias, k_bias, v_bias, q0, k0, v0, q, k, v,
        q_heads, k_heads, v_heads,
        q_t, k_t, v_t, q_rope, k_rope, k_full, v_full, k_src, v_src, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}

/* FASE 1b.0: FFN Qwen3-MoE (router softmax/top-k + dispatch raggruppato per
 * esperto via mlx_gather_qmm + SwiGLU + somma pesata), estratta identica dal
 * secondo metà del vecchio layer_forward monolitico. h: hidden state
 * POST-attenzione prodotto da qwen3_moe_attention_block (o un'altra funzione
 * di attenzione compatibile) -- usato sia per la post_attention_layernorm sia
 * per il residuo finale. *out riceve l'hidden state finale del layer. Il
 * ramo FFN denso (Fase 1b.1, nessun router/MoE) sarà una funzione sorella con
 * la stessa firma, mai una modifica a questa. */


/* FASE 1b.6 (Qwen3.6 / qwen3_5_moe): FFN MoE con SHARED EXPERT — il MoE standard
 * (gate router + switch_mlp + gather esperti top-k) PIU' un MLP denso condiviso
 * (shared_expert + shared_expert_gate) applicato a TUTTI i token.
 *   moe_out      = somma top-k esperti pesati (stesso pattern di qwen3_moe_ffn_block)
 *   shared_out   = shared_expert_gate(x) * shared_expert(x)   # SwiGLU denso
 *   out          = h + moe_out + shared_out
 * Riusa il pattern di qwen3_moe_ffn_block per il MoE core; aggiunge shared_expert.
 * Naming tensori verificato sul modello Qwen3.6-35B-A3B (mlp.shared_expert.*). */

/* Cache pesi ATTENZIONE per-layer (P1/P3): naming standard self_attn.* + input_layernorm.
 * Compilata lazy al primo forward del layer; il wiring nei blocchi e' incrementale. */
GattnQuant* llm_mlx_attn_quant(LlmMlxModel* m, int layer_idx) {
    if (m == NULL || !llm_mlx_wcache_enabled()) return NULL;
    if (m->attn_quant == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->attn_quant = (GattnQuant*)calloc((size_t)n, sizeof(GattnQuant));
        if (m->attn_quant == NULL) return NULL;
        m->attn_quant_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->attn_quant_n) return NULL;
    GattnQuant* c = &m->attn_quant[layer_idx];
    if (c->valid) return c;
    char nb[256];
    bool ok = true;
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &c->ln1);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &c->qw, &c->qs, &c->qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &c->kw, &c->ks, &c->kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &c->vw, &c->vs, &c->vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &c->ow, &c->os, &c->ob);
    if (!ok) return NULL;
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", nb, sizeof(nb));
    bool qo = multishard_get(&m->weights, nb, &c->qn_w);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", nb, sizeof(nb));
    bool ko = multishard_get(&m->weights, nb, &c->kn_w);
    c->has_qn = qo && ko;
    c->valid = true;
    return c;
}
GattnQuant* llm_mlx_gemma4_attn_quant(LlmMlxModel* m, int layer_idx) {
    if (m == NULL || !llm_mlx_wcache_enabled()) return NULL;
    if (m->attn_quant == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->attn_quant = (GattnQuant*)calloc((size_t)n, sizeof(GattnQuant));
        if (m->attn_quant == NULL) return NULL;
        m->attn_quant_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->attn_quant_n) return NULL;
    GattnQuant* c = &m->attn_quant[layer_idx];
    if (c->valid) return c;
    char nb[256];
    tname_gemma4(layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    bool ok = multishard_get(&m->weights, nb, &c->ln1);
    tname_gemma4(layer_idx, "self_attn.q_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &c->qw, &c->qs, &c->qb);
    tname_gemma4(layer_idx, "self_attn.k_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &c->kw, &c->ks, &c->kb);
    if (!ok) return NULL;
    tname_gemma4(layer_idx, "self_attn.v_proj", nb, sizeof(nb));
    bool vok = multishard_get_triple(&m->weights, nb, &c->vw, &c->vs, &c->vb);
    c->has_v = vok;
    if (!vok) { c->vw = mlx_array_new(); c->vs = mlx_array_new(); c->vb = mlx_array_new(); }
    tname_gemma4(layer_idx, "self_attn.o_proj", nb, sizeof(nb));
    bool ook = multishard_get_triple(&m->weights, nb, &c->ow, &c->os, &c->ob);
    if (!ook) { c->ow = mlx_array_new(); c->os = mlx_array_new(); c->ob = mlx_array_new(); }
    tname_gemma4(layer_idx, "self_attn.q_norm.weight", nb, sizeof(nb));
    bool qq = multishard_get(&m->weights, nb, &c->qn_w);
    tname_gemma4(layer_idx, "self_attn.k_norm.weight", nb, sizeof(nb));
    bool kk = multishard_get(&m->weights, nb, &c->kn_w);
    c->has_qn = qq && kk;
    c->valid = true;
    return c;
}
