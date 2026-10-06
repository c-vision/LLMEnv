/* ============================================================================
 * Dominio Qwen3-MoE + Qwen3.5/Next attenzione FULL: llm_bonsai_signs_right, qwen3_moe_attention_block (GQA+QK-norm), qwen3_next_attention_block (output gate).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* FASE 1b.0: attenzione Qwen3-MoE (attenzione GQA + QK-norm + RoPE + KV-cache),
 * estratta identica (nessun cambio di comportamento) da quello che prima era
 * il primo metà di un unico layer_forward monolitico. Generalizza
 * llm_ai_mlx_attention_checkpoint: stesso identico algoritmo verificato in
 * Fase 1.3, parametrizzato su layer_idx/dimensioni di config invece che
 * hardcodato al layer 0, con l'aggiunta della crescita del KV-cache
 * (concatenazione lungo l'asse sequenza) e dell'offset di RoPE reale (non
 * sempre 0). x: [1,n_tokens,hidden]. k_cache/v_cache: posseduti dal
 * chiamante, persistono tra le chiamate per lo stesso layer (append-only per
 * la durata di una llm_ai_mlx_generate). *out_h riceve l'hidden state
 * POST-attenzione (residuo x+attn_out) -- il chiamante lo passa poi a
 * qwen3_moe_ffn_block (o un'altra *_ffn_block, per un'altra architettura che
 * riusi questa stessa attenzione, vedi Fase1b-MLX-Multi-Architecture-Plan.md
 * §1.2 punto A: Llama/Mistral/Gemma dense possono riusare questa funzione,
 * cambiando solo l'FFN). QK-norm (righe qn_w/kn_w) è specifico di Qwen3 --
 * altre architetture avranno una propria *_attention_block che la ometta,
 * non un flag qui (vedi piano §1.2 punto B sul perché renderla condizionale
 * "in place" sarebbe più rischioso che una funzione dedicata). */
GattnQuant* llm_mlx_attn_quant(LlmMlxModel* m, int layer_idx);
bool qwen3_moe_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen3_moe_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
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

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    bool packed = ok && ac && llm_mlx_qkv_apply(&ac->packed_qkv, normed, &q, &k, &v,
        gsz, bts, m->cfg.quant_mode, m->stream, quantized_linear_cfg);
    if (ok && !packed) { ok = quantized_linear_cfg(&q, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok && !packed) { ok = quantized_linear_cfg(&k, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok && !packed) { ok = quantized_linear_cfg(&v, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    /* QK-norm: caricata PRIMA della reshape per distinguere la variante.
     * - qwen3/qwen3_moe: peso = head_dim -> norm per-head DOPO la reshape (canonica).
     * - OLMoE: peso = hidden (=n_embd) -> norm PIATTA su q/k PRIMA della reshape
     *   (llama.cpp olmoe.cpp:92-102: build_norm(Qcur) piatto, poi reshape_3d). */
    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    if (ac != NULL && ac->has_qn) {
        ok = ok && mlx_array_set_optional(&qn_w, ac->qn_w) && mlx_array_set_optional(&kn_w, ac->kn_w);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &qn_w);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &kn_w);
    }
    const int qn_dim = ok ? (int)mlx_array_size(qn_w) : 0;
    /* QK-norm FLAT (OLMoE: una norm sull'intero q/k, dim = n_heads*head_dim) vs PER-HEAD (dim =
     * head_dim). Usare n_heads*head_dim, NON hidden_size: dopo un prune delle teste `hidden_size`
     * NON cambia ma il q/k piatti SÌ (n_keep*head_dim) -> con hidden_size la flat NON veniva
     * riconosciuta e la norm per-head leggeva un peso di lunghezza sbagliata (crash). */
    const bool qk_norm_flat = ok && qn_dim == m->cfg.n_heads * m->cfg.head_dim && qn_dim != m->cfg.head_dim;

    mlx_array q_pre = q, k_pre = k;
    mlx_array q_flat_n = mlx_array_new(), k_flat_n = mlx_array_new();
    if (qk_norm_flat) {
        ok = rms_norm_cfg(&q_flat_n, q, qn_w, m->cfg.rms_norm_eps, m->stream) &&
             rms_norm_cfg(&k_flat_n, k, kn_w, m->cfg.rms_norm_eps, m->stream);
        CHK(ok, "qk-norm (flat/OLMoE)");
        q_pre = q_flat_n; k_pre = k_flat_n;
    }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q_pre, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k_pre, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }

    mlx_array q_normed = mlx_array_new(), k_normed = mlx_array_new();
    if (ok) {
        if (qk_norm_flat) {
            ok = mlx_array_set(&q_normed, q_heads) == 0 && mlx_array_set(&k_normed, k_heads) == 0;
        } else {
            ok = rms_norm_cfg(&q_normed, q_heads, qn_w, m->cfg.rms_norm_eps, m->stream) &&
                 rms_norm_cfg(&k_normed, k_heads, kn_w, m->cfg.rms_norm_eps, m->stream);
            CHK(ok, "qk-norm");
        }
    }

    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_normed, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&k_t, k_normed, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&v_t, v_heads, perm, 4, m->stream) == 0;
        CHK(ok, "qkv transpose");
    }

    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) {
        if (m->cfg.use_yarn_rope) {
            ok = llm_mlx_yarn_rope_apply(&q_rope, &k_rope, m->cfg.head_dim, m->cfg.rope_theta,
                                         m->cfg.yarn_factor, m->cfg.yarn_orig_max,
                                         m->cfg.yarn_beta_fast, m->cfg.yarn_beta_slow, m->cfg.yarn_mscale,
                                         q_t, k_t, cache_offset, m->stream);
            CHK(ok, "rope (yarn)");
        } else {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, m->cfg.head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                            cache_offset, no_freqs, m->stream) == 0 &&
             mlx_fast_rope(&k_rope, k_t, m->cfg.head_dim, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
        CHK(ok, "rope");
        mlx_array_free(no_freqs);
        }
    }

    /* Crescita del KV-cache: concatena lungo l'asse sequenza (asse 2 nel
     * layout [B, n_kv_heads, L, head_dim]) con quanto già accumulato per
     * questo layer. Alla prima chiamata (has_cache=false) il cache è
     * semplicemente k_rope/v_t stessi. */
    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    mlx_array attn = mlx_array_new();
    /* LLM_KVSLICE esteso a Qwen3-MoE (2026-09-07 notte fonda): stesso pattern verificato su
     * Llama (vince, GQA 4x) e GLM (perde, GQA 12x) -- Qwen3-Coder-30B-A3B ha GQA 8x
     * (32 query heads / 4 kv heads), un caso intermedio scelto apposta per testare l'ipotesi
     * "GQA alto rompe KVSLICE". Vedi TODO KvPagePool §1 e LLMEnv-vs-RapidMLX-Comparison §19
     * per il ragionamento completo e i numeri. Esclude nsa_sparse (percorso sperimentale
     * separato, default OFF, non combinato con questo). */
    bool kvsl = llm_mlx_kvslice_enabled() && n_tokens == 1 && has_cache && m->cfg.kv_max_tokens <= 0
        && !m->cfg.nsa_sparse;
    /* Append KV in-place (in-place-first, fallback concat): k_src/v_src = sorgente SDPA. */
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
        if (ok) {
            ok = mlx_array_set(k_cache, k_full) == 0;
            CHK(ok, "k cache store");
            if (ok) { ok = mlx_array_set(v_cache, v_full) == 0; CHK(ok, "v cache store"); }
        }
        if (ok) { ok = mlx_array_set(&k_src, k_full) == 0 && mlx_array_set(&v_src, v_full) == 0; }
    }

    /* E3 KV quant (opzionale, default OFF): se kv_quantize attivo e la cache supera la
     * soglia, quantizza la parte COLD (piu' vecchia) a Q8/Q4 e la ricompatta (dequant +
     * recenti FP16). Con kv_quantize=false (default) questo blocco non gira: parita' col
     * comportamento precedente garantita. Il risparmio memoria reale (mantenere la cache
     * quantizzata senza dequantizzare subito) e' il passo successivo; qui verifica il flusso. */
    if (0 && ok && m->cfg.kv_quantize && m->cfg.kv_quantize_threshold > 0) { /* KV quant ora in layer_forward */
        int bits = m->cfg.kv_quantize_bits > 0 ? m->cfg.kv_quantize_bits : 8;
        int n_cold = 0;
        mlx_vector_array qk = mlx_vector_array_new();
        if (llm_mlx_kv_apply_policy(&qk, *k_cache, m->cfg.kv_quantize_threshold, bits, 64, &n_cold, m->stream)
            && n_cold > 0) {
            /* dequantizza la parte cold e ricompatta con i recenti FP16 */
            mlx_array w=mlx_array_new(), sc=mlx_array_new(), bi=mlx_array_new();
            mlx_array deq=mlx_array_new();
            mlx_vector_array_get(&w, qk, 0); mlx_vector_array_get(&sc, qk, 1); mlx_vector_array_get(&bi, qk, 2);
            if (llm_mlx_kv_dequantize(&deq, w, sc, bi, bits, 64, m->stream)) {
                mlx_array recent = mlx_array_new();
                /* slice recenti: [B,Hkv, n_cold..N, D] */
                int r0[4]={0,0,n_cold,0}, r1[4]={1,(int)mlx_array_shape(*k_cache)[1],(int)mlx_array_shape(*k_cache)[2],(int)mlx_array_shape(*k_cache)[3]};
                int ones[4]={1,1,1,1};
                if (mlx_slice(&recent, *k_cache, r0, 4, r1, 4, ones, 4, m->stream)==0) {
                    mlx_vector_array rc=mlx_vector_array_new();
                    mlx_vector_array_append_value(rc, deq);
                    mlx_vector_array_append_value(rc, recent);
                    mlx_array k_full2=mlx_array_new();
                    if (mlx_concatenate_axis(&k_full2, rc, 2, m->stream)==0)
                        mlx_array_set(k_cache, k_full2);
                    mlx_vector_array_free(rc); mlx_array_free(k_full2);
                }
                mlx_array_free(recent);
            }
            mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi); mlx_array_free(deq);
        }
        mlx_vector_array_free(qk);
    }

    if (ok) {
        float scale = 1.0f / sqrtf((float)m->cfg.head_dim);
        mlx_array no_mask = mlx_array_new();
        mlx_array no_sinks = mlx_array_new();
        /* mask_mode="causal": MLX calcola offset=kL-qL internamente
         * (verificato in mlx/fast.cpp) e maschera q_idx=[offset..qL+offset)
         * contro k_idx=[0..kL) -- corretto sia per L=1 (degenera esattamente
         * al vecchio comportamento mask_mode="", "il token vede tutta la
         * cronologia in cache") sia per L>1 (query nuove che continuano da
         * una cache di cache_offset token precedenti, vedi Aggiornamento
         * batched prefill nel piano). */
        /* NSA sparse (opzionale, default OFF): su decode (n_tokens=1) se nsa_sparse attivo,
         * seleziona top-k token causali di K e fa attention sparsa (O(topk) non O(N)).
         * Con default OFF -> attention piena (parita'). kvsl e' gia' garantito false qui. */
        if (m->cfg.nsa_sparse && n_tokens == 1 && m->cfg.nsa_topk > 0) {
            ok = llm_mlx_nsa_decode_sdpa(&attn, q_rope, *k_cache, *v_cache,
                1.0f / sqrtf((float)m->cfg.head_dim), m->cfg.nsa_topk, m->cfg.nsa_window, m->stream);
        } else {
            ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
        }
        CHK(ok, "scaled_dot_product_attention");
        mlx_array_free(no_mask); mlx_array_free(no_sinks); /* sentinelle vuote (non nel loop all[]) */
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
        ln_w, normed, qw, qs, qb, kw, ks, kb, vw, vs, vb, q, k, v,
        q_heads, k_heads, v_heads, qn_w, kn_w, q_normed, k_normed,
        q_t, k_t, v_t, q_rope, k_rope, k_full, v_full, k_src, v_src, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}

/* FASE 1b.5 (Qwen3.8 / qwen3_5): attenzione FULL dei layer full con OUTPUT GATE
 * (Qwen3NextAttention, attn_output_gate=true). Differenza da qwen3_moe_attention_block:
 * q_proj produce n_heads*head_dim*2 = queries + gate; dopo l'SDPA e prima di o_proj
 * si applica `o_proj(sdpa(...) * sigmoid(gate))`. Riferimento: mlx_lm Qwen3NextAttention.__call__.
 * Il resto (QK-norm, RoPE, KV-cache concat, SDPA causale, o_proj, residual) è identico. */
/* Orientamento dei sign: default "right" (H·diag(s)); LLM_BONSAI_SIGNS=left per A/B empirico. */
bool llm_bonsai_signs_right(void) {
    static int st = -1;
    if (st < 0) { const char* e = llm_ai_mlx_opt_env("LLM_BONSAI_SIGNS"); st = (e != NULL && strcmp(e, "left") == 0) ? 0 : 1; }
    return st == 1;
}
/* Ternary-Bonsai: definita piu' sotto (vicino a dense_swiglu_ffn), ma serve prima qui. */

bool qwen3_next_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen3_next_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
    if (!ok) CHK(false, "input_layernorm lookup");
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, ln_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "input_layernorm"); }
    /* Ternary-Bonsai: self_attn.{q,k,v}_proj sono elencati e condividono l'ingresso (`normed`);
     * il gate dell'output-gate e' DENTRO q_proj, quindi coperto dalla stessa rotazione. */
    if (ok) ok = llm_bonsai_rotate(&normed, m, layer_idx, "self_attn.q_proj.weight", false);
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

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = quantized_linear_cfg(&k, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = quantized_linear_cfg(&v, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    /* Output gate: q_proj = [B,L, n_heads*head_dim*2]. Il reference Qwen3NextAttention fa
     *   q_proj_output.reshape(B,L,num_heads,-1)  # [B,L,nh, head_dim*2]
     *   queries, gate = mx.split(..., 2, axis=-1)  # interleaved DENTRO ogni testa
     *   gate = gate.reshape(B,L,-1)
     * quindi queries e gate NON sono due meta' piatte: sono interleaved per testa
     * (dims 0..head_dim-1 = query, head_dim..2*head_dim-1 = gate, per OGNI head).
     * Bisogna reshape a [B,L,nh,head_dim*2] e split lungo l'ultimo asse, non flat split. */
    mlx_array q_full=mlx_array_new(), gate_full=mlx_array_new();
    if (ok) {
        int qshape4[4] = {1, n_tokens, m->cfg.n_heads, 2 * m->cfg.head_dim};
        mlx_array q_resh = mlx_array_new();
        ok = mlx_reshape(&q_resh, q, qshape4, 4, m->stream) == 0;
        int qfg[2] = { m->cfg.head_dim, 2 * m->cfg.head_dim };
        mlx_vector_array qsplit = mlx_vector_array_new();
        if (ok) ok = mlx_split_sections(&qsplit, q_resh, qfg, 2, /*axis=*/-1, m->stream) == 0;
        if (ok) ok = mlx_vector_array_get(&q_full, qsplit, 0) == 0 &&
                        mlx_vector_array_get(&gate_full, qsplit, 1) == 0;
        /* gate_full: [B,L,nh,head_dim] -> [B,L,nh*head_dim] */
        if (ok) {
            const int gflat[3] = {1, n_tokens, m->cfg.n_heads * m->cfg.head_dim};
            mlx_array g2 = mlx_array_new();
            ok = mlx_reshape(&g2, gate_full, gflat, 3, m->stream) == 0;
            if (ok) { mlx_array_free(gate_full); gate_full = g2; } else mlx_array_free(g2);
        }
        mlx_vector_array_free(qsplit);
        mlx_array_free(q_resh);
        CHK(ok, "q_proj output-gate split (interleaved)");
    }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q_full, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }

    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &qn_w);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &kn_w);
    mlx_array q_normed = mlx_array_new(), k_normed = mlx_array_new();
    if (ok) {
        ok = rms_norm_cfg(&q_normed, q_heads, qn_w, m->cfg.rms_norm_eps, m->stream) &&
             rms_norm_cfg(&k_normed, k_heads, kn_w, m->cfg.rms_norm_eps, m->stream);
        CHK(ok, "qk-norm");
    }

    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_normed, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&k_t, k_normed, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&v_t, v_heads, perm, 4, m->stream) == 0;
        CHK(ok, "qkv transpose");
    }

    /* RoPE con partial_rotary_factor (Qwen3.8: 0.25): il reference Qwen3NextAttention usa
     * nn.RoPE(dims=int(head_dim*partial_rotary_factor)) = ruota SOLO i primi `rotated_dims`
     * di ogni testa con base standard. mlx_fast_rope(dims=rotated_dims) su un head [..,head_dim]
     * ruota i primi rotated_dims ("dims must not exceed the input's last dimension") -- NON
     * serve la variante proportional-freqs di gemma4. */
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) {
        float prf = m->cfg.partial_rotary_factor;
        int head_dim = m->cfg.head_dim;
        int rope_dims = (prf > 0.0f && prf < 1.0f) ? (int)(head_dim * prf) : head_dim;
        if (rope_dims <= 0) rope_dims = head_dim;
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, rope_dims, /*traditional=*/false, base, /*scale=*/1.0f,
                            cache_offset, no_freqs, m->stream) == 0 &&
             mlx_fast_rope(&k_rope, k_t, rope_dims, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
        CHK(ok, "rope (partial, rotate first rotated_dims)");
        mlx_array_free(no_freqs);
    }

    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    mlx_array k_src = mlx_array_new(), v_src = mlx_array_new();
    bool kv_inplace = false;
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
    /* Rolling-Window KV (fase 3 paging): se kv_max_tokens>0, tronca la cache full ai piu' recenti
     * N token (mantiene i blocchi recenti, evicta i vecchi via Cache ABI trim). Limita la memoria
     * KV a contesto lungo (StreamingLLM-style). 0 = off (cache illimitata). */
    if (ok && m->cfg.kv_max_tokens > 0) {
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

    /* §refactor modulare: SDPA causale via llm_mlx_gqa_sdpa (window=0 = full attention). */
    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);         CHK(ok, "scaled_dot_product_attention");
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

    /* o_proj(sdpa(...) * sigmoid(gate)): gate flatten [B,L, n_heads*head_dim],
     * sigmoid, moltiplica l'output dell'attenzione prima della proiezione. */
    mlx_array gate_sig = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&gate_sig, gate_full, m->stream) == 0; CHK(ok, "gate sigmoid"); }
    mlx_array attn_gated = mlx_array_new();
    if (ok) { ok = mlx_multiply(&attn_gated, attn_flat, gate_sig, m->stream) == 0; CHK(ok, "attn*gate"); }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &os, &ob);
    /* Ternary-Bonsai: self_attn.o_proj e' elencato. */
    if (ok) ok = llm_bonsai_rotate(&attn_gated, m, layer_idx, "self_attn.o_proj.weight", false);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_gated, ow, os, ob, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "o_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "o_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&attn_out, attn_gated, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, m->stream) == 0; CHK(ok, "residual add (attn)"); }

    mlx_array all[] = {
        ln_w, normed, qw, qs, qb, kw, ks, kb, vw, vs, vb, q, k, v,
        q_full, gate_full, q_heads, k_heads, v_heads, qn_w, kn_w, q_normed, k_normed,
        q_t, k_t, v_t, q_rope, k_rope, k_full, v_full, k_src, v_src, attn,
        attn_bt, attn_flat, gate_sig, attn_gated, ow, os, ob, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
