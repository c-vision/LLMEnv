/* ============================================================================
 * llm_mlx_gptoss_ffn.c — dominio GPT-OSS: blocco attenzione (q/k/v/o bias + sinks +
 * sliding 128) e FFN MoE (32 esperti top4, SwiGLU speciale + bias, mxfp4 fuso). Estratto
 * VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_moe_fused_kernel.h"
#include "llm_mlx_swiglu_gptoss_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool gpt_oss_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "gpt_oss_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln_w = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "input_layernorm.weight", &ln_w);
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

    /* Sinks appresi (per-head) -- carica il tensore self_attn.sinks [n_heads] */
    mlx_array sinks = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "self_attn.sinks", &sinks);
    CHK(ok, "sinks lookup");
    /* il multishard puo' restituire una VIEW a stride (bug reale: la C leggeva i
     * sinks con stride 2 -> valori sbagliati). Forza copia contigua. */
    if (ok) {
        mlx_array sc = mlx_array_new();
        ok = mlx_contiguous(&sc, sinks, /*allow_col_major=*/false, m->stream) == 0;
        if (ok) { mlx_array_free(sinks); sinks = sc; } else mlx_array_free(sc);
    }

    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_triple_sfx(m, layer_idx, "self_attn.q_proj", &qw, &qs, &qb);
    ok = ok && llm_mlx_gptoss_get_triple_sfx(m, layer_idx, "self_attn.k_proj", &kw, &ks, &kb);
    ok = ok && llm_mlx_gptoss_get_triple_sfx(m, layer_idx, "self_attn.v_proj", &vw, &vs, &vb);

    /* Bias lineari reali (q_proj.bias, separati da .biases di quantizzazione) */
    mlx_array q_lb = mlx_array_new(), k_lb = mlx_array_new(), v_lb = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "self_attn.q_proj.bias", &q_lb);
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "self_attn.k_proj.bias", &k_lb);
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "self_attn.v_proj.bias", &v_lb);
    CHK(ok, "q/k/v_proj.bias lookup");

    mlx_array q0 = mlx_array_new(), k0 = mlx_array_new(), v0 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q0, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
    if (ok) { ok = quantized_linear_cfg(&k0, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
    if (ok) { ok = quantized_linear_cfg(&v0, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
    /* §D R6 (multi-arch): LoRA su q/k/v_proj. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q0, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k0, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v0, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) {
        mlx_array qb = mlx_array_new(), kb = mlx_array_new(), vb = mlx_array_new();
        if (n_tokens == 1) {
            ok = llm_mlx_gptoss_bias_prep(m, layer_idx, 0, q_lb, m->cfg.n_heads * m->cfg.head_dim, &qb) &&
                 llm_mlx_gptoss_bias_prep(m, layer_idx, 1, k_lb, m->cfg.n_kv_heads * m->cfg.head_dim, &kb) &&
                 llm_mlx_gptoss_bias_prep(m, layer_idx, 2, v_lb, m->cfg.n_kv_heads * m->cfg.head_dim, &vb);
        } else {
            ok = mlx_array_set(&qb, q_lb) == 0 && mlx_array_set(&kb, k_lb) == 0 && mlx_array_set(&vb, v_lb) == 0;
        }
        ok = ok && mlx_add(&q, q0, qb, m->stream) == 0 &&
                   mlx_add(&k, k0, kb, m->stream) == 0 &&
                   mlx_add(&v, v0, vb, m->stream) == 0;
        mlx_array_free(qb); mlx_array_free(kb); mlx_array_free(vb);
        CHK(ok, "q/k/v linear bias add");
    }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    static int notr = -1;
    if (notr < 0) notr = (llm_env_get("LLM_GPTOSS_NOTRANSPOSE") != NULL) ? 1 : 0;
    if (ok && n_tokens == 1 && !notr) {
        /* Decode mono-token: [1,1,H,D] e [1,H,1,D] hanno lo STESSO ordine lineare (i singleton
         * non spostano dati) -> reshape diretto, NIENTE transpose. Bit-identico (stesso pattern
         * verificato su llama): -3 Transpose/layer, che il reference paga. */
        const int qh[4] = {1, m->cfg.n_heads, 1, m->cfg.head_dim};
        const int kvh[4] = {1, m->cfg.n_kv_heads, 1, m->cfg.head_dim};
        ok = mlx_reshape(&q_t, q, qh, 4, m->stream) == 0 &&
             mlx_reshape(&k_t, k, kvh, 4, m->stream) == 0 &&
             mlx_reshape(&v_t, v, kvh, 4, m->stream) == 0;
        CHK(ok, "qkv layout (decode, no transpose)");
    } else if (ok) {
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }
    if (ok && (n_tokens != 1 || notr)) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_heads, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&k_t, k_heads, perm, 4, m->stream) == 0 &&
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

    /* LLM_KVF16 (#61) esteso a gpt_oss: cache KV in fp16 SOLO per i layer FULL (quelli
     * con O(pos) nel concat). I layer sliding (window>0) restano fp32: il loro ramo SDPA
     * usa la mask additiva e NON ha l'armonizzazione q<->k/v dtype. Ogni layer ha la sua
     * cache, quindi dtype misti tra layer sono leciti. */
    if (ok && llm_mlx_kv_f16_gptoss_enabled()
        && !(m->cfg.is_sliding_layer[layer_idx] && m->cfg.sliding_window > 0)
        && mlx_array_dtype(k_rope) != MLX_FLOAT16) {
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
            mlx_vector_array_free(kv);
            CHK(ok, "k-cache concat");
            if (ok) {
                mlx_vector_array vv = mlx_vector_array_new();
                mlx_vector_array_append_value(vv, *v_cache);
                mlx_vector_array_append_value(vv, v_t);
                ok = mlx_concatenate_axis(&v_full, vv, 2, m->stream) == 0;
                mlx_vector_array_free(vv);
                CHK(ok, "v-cache concat");
            }
        } else {
            ok = mlx_array_set(&k_full, k_rope) == 0 && mlx_array_set(&v_full, v_t) == 0;
            CHK(ok, "k/v cache init");
        }
    }
        if (ok) { ok = mlx_array_set(&k_src, k_full) == 0 && mlx_array_set(&v_src, v_full) == 0; }
    }
    if (ok && !kv_inplace) {
        ok = mlx_array_set(k_cache, k_full) == 0;
        CHK(ok, "k cache store");
        if (ok) { ok = mlx_array_set(v_cache, v_full) == 0; CHK(ok, "v cache store"); }
    }

    mlx_array attn = mlx_array_new();
    if (ok) {
        int win = (m->cfg.is_sliding_layer[layer_idx] && m->cfg.sliding_window > 0) ? m->cfg.sliding_window : 0;
        ok = llm_mlx_gqa_sdpa_sinks(&attn, q_rope, k_src, v_src,
                                    1.0f / sqrtf((float)m->cfg.head_dim), win, sinks, m->stream);
        CHK(ok, "scaled_dot_product_attention (sinks)");
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int flat_shape[3] = {1, n_tokens, m->cfg.n_heads * m->cfg.head_dim};
        if (n_tokens == 1 && !notr) {
            /* decode: [1,H,1,D] -> [1,1,H*D] e' lo stesso ordine lineare, il transpose back e'
             * un no-op (bit-identico, -1 Transpose/layer). */
            ok = mlx_reshape(&attn_flat, attn, flat_shape, 3, m->stream) == 0;
            CHK(ok, "attn flatten (decode, no transpose)");
        } else {
            const int perm[4] = {0, 2, 1, 3};
            ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream) == 0;
            CHK(ok, "attn transpose back");
            if (ok) {
                ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, m->stream) == 0;
                CHK(ok, "attn flatten");
            }
        }
    }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_triple_sfx(m, layer_idx, "self_attn.o_proj", &ow, &os, &ob);
    CHK(ok, "o_proj lookup");
    mlx_array o0 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o0, attn_flat, ow, os, ob, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "o_proj matmul"); }
    mlx_array o_lb = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "self_attn.o_proj.bias", &o_lb);
    mlx_array attn_out = mlx_array_new();
    if (ok) {
        mlx_array ob = mlx_array_new();
        if (n_tokens == 1) ok = llm_mlx_gptoss_bias_prep(m, layer_idx, 3, o_lb, m->cfg.hidden_size, &ob);
        else ok = mlx_array_set(&ob, o_lb) == 0;
        ok = ok && mlx_add(&attn_out, o0, ob, m->stream) == 0;
        mlx_array_free(ob);
        CHK(ok, "o_proj linear bias add");
    }

    if (ok) { ok = mlx_add(out_h, x, attn_out, m->stream) == 0; CHK(ok, "residual add (attn)"); }

    mlx_array all[] = {
        ln_w, normed, sinks, qw, qs, qb, kw, ks, kb, vw, vs, vb,
        q_lb, k_lb, v_lb, q0, k0, v0, q, k, v,
        q_heads, k_heads, v_heads, q_t, k_t, v_t, q_rope, k_rope,
        k_full, v_full, k_src, v_src, attn, attn_bt, attn_flat, ow, os, ob, o0, o_lb, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}

/* GPT-OSS MoE FFN: router `mlp.router` (bias) -> top-4 -> softmax SOLO sui top-k
 * logit (diverso da qwen3_moe, che fa softmax su tutti + rinormalizza) -> gather
 * esperti `mlp.experts.{gate,up,down}_proj` (mxfp4 32/4 SENZA .biases quant + bias
 * lineare) con SwiGLU speciale gpt-oss: clip(gate,7) * sigmoid(1.702*gate) *
 * (clip(up,±7)+1) -> down_proj -> somma pesata -> residuo. Reference:
 * mlx_lm/models/gpt_oss.py (SwiGLU alpha=1.702 limit=7.0; SwitchGLU bias=True). */

/* gpt_oss_expert_unsort: output gather sorted (righe [L,...D]) -> ordine originale -> [T,K,1,D]. */
static bool gpt_oss_expert_unsort(mlx_array* res, mlx_array raw, mlx_array inv_idx, int T, int K, mlx_stream s) {
    const int* sh = mlx_array_shape(raw);
    int nd = mlx_array_ndim(raw);
    int D = sh[nd - 1];
    int L = (int)(mlx_array_size(raw) / (size_t)D);
    int s2[2] = {L, D};
    mlx_array flat = mlx_array_new();
    bool ok = mlx_reshape(&flat, raw, s2, 2, s) == 0;
    mlx_array un = mlx_array_new();
    if (ok) ok = mlx_take_axis(&un, flat, inv_idx, /*axis=*/0, s) == 0;
    if (ok) { int s4[4] = {T, K, 1, D}; ok = mlx_reshape(res, un, s4, 4, s) == 0; }
    mlx_array_free(flat); mlx_array_free(un);
    return ok;
}

/* LLM_MXFP4_PAD512 (2026-09-11): gli esperti di gpt-oss hanno K=2880 (Din=Dff), NON
 * multiplo di 512 -> il gather_qmv di MLX sceglie il kernel GENERICO. Paddando K a 3072
 * (zeri, blocchi di quantizzazione interi) si innesca `gather_qmv_fast` (~5% model-free,
 * tools/moe_mxfp4_fastbranch_probe.py: K=3072 -5% tempo / +12% GB/s vs K=2880). Pad UNA
 * volta per layer (cache; i pesi non si ricostruiscono per token) + padding dell'attivazione
 * (economico). La matematica e' invariata: i 192 valori di pad sono zeri e le loro 6 scale
 * sono 0 -> x_pad . w_pad^T = x . w^T. Default OFF (misura prima di promuovere). */
#define LLM_PAD512_MAX_LAYERS 256
static LlmPad512 g_pad512[LLM_PAD512_MAX_LAYERS];
static int g_pad512_state = -1;
static bool llm_mlx_mxfp4_pad512_enabled(void) {
    if (g_pad512_state < 0) g_pad512_state = (getenv("LLM_MXFP4_PAD512") != NULL) ? 1 : 0;
    return g_pad512_state == 1;
}
void llm_mlx_pad512_reset(void) {
    for (int i = 0; i < LLM_PAD512_MAX_LAYERS; i++) {
        LlmPad512* p = &g_pad512[i];
        if (!p->ready) continue;
        mlx_array_free(p->guw); mlx_array_free(p->gus);
        mlx_array_free(p->uw);  mlx_array_free(p->us);
        mlx_array_free(p->dw);  mlx_array_free(p->ds);
        memset(p, 0, sizeof(*p));
    }
}
/* LLM_FUSE_GATEUP (2026-09-12, adattato da rapid-MLX moe_fusion.py): fonde le proiezioni
 * gate+up degli esperti in UN solo gather_qmm. La quantizzazione (affine/mxfp4) quantizza
 * ogni riga di output indipendentemente -> concatenare gate+up lungo l'ASSE DI OUTPUT e'
 * BIT-IDENTICO a due gather separate, con un dispatch Metal in meno per layer MoE. Cache
 * per-layer dei pesi concatenati [E, 2*inter, in] (concatenare per token sarebbe proibitivo).
 * Default OFF; non compone con PAD512 (asserti separati) ne' col kernel fuso mxfp4. */
static LlmGateUp g_gateup[LLM_PAD512_MAX_LAYERS];
static int g_gateup_state = -1;
static bool llm_mlx_fuse_gateup_enabled(void) {
    if (g_gateup_state < 0) g_gateup_state = (getenv("LLM_FUSE_GATEUP") != NULL) ? 1 : 0;
    return g_gateup_state == 1;
}
void llm_mlx_gateup_reset(void) {
    for (int i = 0; i < LLM_PAD512_MAX_LAYERS; i++) {
        LlmGateUp* p = &g_gateup[i];
        if (!p->ready) continue;
        mlx_array_free(p->w); mlx_array_free(p->s); mlx_array_free(p->b);
        memset(p, 0, sizeof(*p));
    }
}
/* Concatena due array lungo `axis` (dtype e dim preservati). */
static bool llm_mlx_concat2_axis(mlx_array* out, mlx_array a, mlx_array b, int axis, mlx_stream s) {
    mlx_vector_array v = mlx_vector_array_new();
    mlx_vector_array_append_value(v, a);
    mlx_vector_array_append_value(v, b);
    bool ok = mlx_concatenate_axis(out, v, axis, s) == 0;
    mlx_vector_array_free(v);
    return ok;
}
/* Concatena `pad` zeri lungo l'ULTIMO asse, dtype preservato. */
static bool llm_mlx_pad_last(mlx_array* out, mlx_array a, int pad, mlx_stream s) {
    int nd = mlx_array_ndim(a);
    const int* sh = mlx_array_shape(a);
    if (nd < 1 || sh == NULL || pad <= 0) return false;
    int zsh[8];
    for (int i = 0; i < nd && i < 8; i++) zsh[i] = sh[i];
    zsh[nd - 1] = pad;
    mlx_array z = mlx_array_new();
    bool ok = mlx_zeros(&z, zsh, nd, mlx_array_dtype(a), s) == 0;
    if (ok) {
        mlx_vector_array v = mlx_vector_array_new();
        mlx_vector_array_append_value(v, a);
        mlx_vector_array_append_value(v, z);
        ok = mlx_concatenate_axis(out, v, nd - 1, s) == 0;
        mlx_vector_array_free(v);
    }
    mlx_array_free(z);
    return ok;
}

/* LLMMETAL_GPTOSS_SWIGLU_METAL (2026-09-13): attivazione SwiGLU gpt-oss in UN dispatch Metal
 * (llm_mlx_swiglu_gptoss_kernel.h) invece di ~13 op eager + Broadcast di scalari. mlx_lm la
 * compila in un nodo; questo e' l'equivalente senza mlx_compile. Default ON (kill-switch =0):
 * verificato IDS greedy identico e prefill in miglioramento. */
static int g_gptoss_swiglu_metal_state = -1;
static bool llm_mlx_fuse_swiglu_gptoss_enabled(void) {
    if (g_gptoss_swiglu_metal_state < 0) {
        const char* e = getenv("LLMMETAL_GPTOSS_SWIGLU_METAL");
        g_gptoss_swiglu_metal_state = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    return g_gptoss_swiglu_metal_state == 1;
}

bool gpt_oss_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256], nb1[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "gpt_oss_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln2_w = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "post_attention_layernorm.weight", &ln2_w);
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }

    /* Router con bias lineare */
    mlx_array gw = mlx_array_new(), gs = mlx_array_new(), gb = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_triple_sfx(m, layer_idx, "mlp.router", &gw, &gs, &gb);
    CHK(ok, "router lookup");
    mlx_array g_lb = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.router.bias", &g_lb);
    mlx_array gates0 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gates0, normed2, gw, gs, gb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "router matmul"); }
    mlx_array gates = mlx_array_new();
    if (ok) {
        mlx_array gbb = mlx_array_new();
        if (n_tokens == 1) ok = llm_mlx_gptoss_bias_prep(m, layer_idx, 4, g_lb, m->cfg.n_experts, &gbb);
        else ok = mlx_array_set(&gbb, g_lb) == 0;
        ok = ok && mlx_add(&gates, gates0, gbb, m->stream) == 0;
        mlx_array_free(gbb);
        CHK(ok, "router bias add");
    }
    static int dump_gates_done = 0;
    if (ok && !dump_gates_done && llm_env_get("LLM_DEBUG_GATES")) {
        mlx_array gf = mlx_array_new();
        if (mlx_astype(&gf, gates, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(gf) == 0) {
            const float* d = mlx_array_data_float32(gf);
            size_t ne = (size_t)n_tokens * (size_t)m->cfg.n_experts;
            FILE* f = fopen(llm_env_get("LLM_DEBUG_GATES"), "wb");
            if (f) { fwrite(d, sizeof(float), ne, f); fclose(f); }
        }
        mlx_array_free(gf);
        dump_gates_done = 1;
    }

    /* top-4: argpartition (indici) -> top_inds; poi top-k VALUES via take_along_axis
     * sui logit -> softmax SOLO sui top-k (reference: mlx_topk + softmax(experts)). */
    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, gates, /*kth=*/-m->cfg.top_k, /*axis=*/-1, m->stream) == 0;
        CHK(ok, "router argpartition");
        if (ok) {
            const int ne = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, ne - m->cfg.top_k};
            const int stop[3] = {1, n_tokens, ne};
            const int strides[3] = {1, 1, 1};
            mlx_array view = mlx_array_new();
            ok = mlx_slice(&view, part, start, 3, stop, 3, strides, 3, m->stream) == 0;
            CHK(ok, "top-k slice");
            if (ok) ok = mlx_contiguous(&top_inds, view, /*allow_col_major=*/false, m->stream) == 0;
            mlx_array_free(view);
        }
    }
    mlx_array topk_vals = mlx_array_new();
    if (ok) {
        const int vi_shape[3] = {1, n_tokens, m->cfg.top_k};
        mlx_array idx = mlx_array_new();
        ok = mlx_reshape(&idx, top_inds, vi_shape, 3, m->stream) == 0;
        if (ok) ok = mlx_take_along_axis(&topk_vals, gates, idx, /*axis=*/-1, m->stream) == 0;
        mlx_array_free(idx);
        CHK(ok, "top-k values gather");
    }
    mlx_array weights = mlx_array_new();
    if (ok) { ok = mlx_softmax_axis(&weights, topk_vals, -1, /*precise=*/true, m->stream) == 0; CHK(ok, "top-k softmax"); }

    const int T = n_tokens, K = m->cfg.top_k, L = T * K;
    /* Q2-b: imatrix (calibrazione) — esperto per riga sorted + riga sorted -> token-major. */
    uint32_t* im_expert = NULL; uint32_t* im_src = NULL; int im_rows = 0;
    /* Reference (mlx_lm switch_layers): sorted_indices solo se indices.size >= 64;
     * sotto usa gather_qmm UNSORTED. Il kernel sorted su gruppi singleton (L<64) produce
     * output errati -> replica la soglia della reference. Per L<64 (decode T=1) il path
     * resta TUTTO su device (niente eval/host top_inds): gli indici esperti si derivano con
     * reshape/astype dell'array top_inds lazy -> un solo sync per token (final_step). */
    const bool unsort_moe = L < 64;

    /* Esperti: weight + scales (mxfp4, SENZA .biases quant -- verificato sul
     * safetensors reale) + .bias lineare. Bias quant vuoto (gather_qmm: null ok). */
    mlx_array gp_w = mlx_array_new(), gp_s = mlx_array_new(), gp_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_s = mlx_array_new(), up_b = mlx_array_new();
    mlx_array dn_w = mlx_array_new(), dn_s = mlx_array_new(), dn_b = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.gate_proj.weight", &gp_w) && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.gate_proj.scales", &gp_s);
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.up_proj.weight", &up_w) && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.up_proj.scales", &up_s);
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.down_proj.weight", &dn_w) && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.down_proj.scales", &dn_s);
    mlx_array gp_lb = mlx_array_new(), up_lb = mlx_array_new(), dn_lb = mlx_array_new();
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.gate_proj.bias", &gp_lb);
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.up_proj.bias", &up_lb);
    ok = ok && llm_mlx_gptoss_get_sfx(m, layer_idx, "mlp.experts.down_proj.bias", &dn_lb);
    CHK(ok, "experts lookup");

    /* LLM_MXFP4_PAD512: pad K 2880->3072 (pesi cachati + attivazione) per gather_qmv_fast. */
    mlx_array gp_w_u = mlx_array_new(), gp_s_u = mlx_array_new();
    mlx_array up_w_u = mlx_array_new(), up_s_u = mlx_array_new();
    mlx_array dn_w_u = mlx_array_new(), dn_s_u = mlx_array_new();
    int pad_in = 0, pad_ff = 0;
    bool pad512 = ok && llm_mlx_mxfp4_pad512_enabled() && layer_idx >= 0 &&
                  layer_idx < LLM_PAD512_MAX_LAYERS &&
                  mlx_array_ndim(gp_w) == 3 && mlx_array_ndim(dn_w) == 3 &&
                  mlx_array_ndim(gp_s) == 3 && mlx_array_ndim(dn_s) == 3;
    if (pad512) {
        const int din = m->cfg.hidden_size;
        const int dff = (int)mlx_array_dim(gp_w, 1);
        pad_in = (512 - (din % 512)) % 512;
        pad_ff = (512 - (dff % 512)) % 512;
        if (pad_in == 0 && pad_ff == 0) pad512 = false;
    }
    if (pad512) {
        LlmPad512* p = &g_pad512[layer_idx];
        if (!p->ready) {
            const int pi8 = pad_in / 8, pi32 = pad_in / 32;
            const int pf8 = pad_ff / 8, pf32 = pad_ff / 32;
            bool cok = llm_mlx_pad_last(&p->guw, gp_w, pi8, m->stream) &&
                       llm_mlx_pad_last(&p->gus, gp_s, pi32, m->stream) &&
                       llm_mlx_pad_last(&p->uw,  up_w, pi8, m->stream) &&
                       llm_mlx_pad_last(&p->us,  up_s, pi32, m->stream) &&
                       llm_mlx_pad_last(&p->dw,  dn_w, pf8, m->stream) &&
                       llm_mlx_pad_last(&p->ds,  dn_s, pf32, m->stream);
            if (cok) p->ready = true;
            else { llm_mlx_pad512_reset(); pad512 = false;
                   fprintf(stderr, "gpt_oss_ffn[layer=%d]: PAD512 cache build failed -> no pad\n", layer_idx); }
        }
        if (pad512 && p->ready) {
            ok = ok && mlx_array_set(&gp_w_u, p->guw) == 0 && mlx_array_set_optional(&gp_s_u, p->gus) &&
                       mlx_array_set(&up_w_u, p->uw) == 0 && mlx_array_set_optional(&up_s_u, p->us) &&
                       mlx_array_set(&dn_w_u, p->dw) == 0 && mlx_array_set_optional(&dn_s_u, p->ds);
        } else pad512 = false;
    }
    mlx_array gp_w_s = pad512 ? gp_w_u : gp_w, gp_s_s = pad512 ? gp_s_u : gp_s;
    mlx_array up_w_s = pad512 ? up_w_u : up_w, up_s_s = pad512 ? up_s_u : up_s;
    mlx_array dn_w_s = pad512 ? dn_w_u : dn_w, dn_s_s = pad512 ? dn_s_u : dn_s;

    /* gather bias lineare per top_inds: bias [n_experts, out] -> [n_tokens, top_k, out]
     * via mlx_take_axis (mx.take(bias, idx, axis=0): idx [T,K] -> [T,K,out]). Sempre su
     * device: idx2 = reshape di top_inds astype uint32 (ordine partition). */
    mlx_array gp_lb_g = mlx_array_new(), up_lb_g = mlx_array_new(), dn_lb_g = mlx_array_new();
    if (ok) {
        const int ti2_shape[2] = {n_tokens, m->cfg.top_k};
        mlx_array idx2 = mlx_array_new();
        {
            int ts[2] = {n_tokens, m->cfg.top_k};
            mlx_array tt = mlx_array_new();
            ok = mlx_reshape(&tt, top_inds, ts, 2, m->stream) == 0;
            if (ok) ok = mlx_astype(&idx2, tt, MLX_UINT32, m->stream) == 0;
            mlx_array_free(tt);
        }
        ok = ok && mlx_take_axis(&gp_lb_g, gp_lb, idx2, 0, m->stream) == 0 &&
             mlx_take_axis(&up_lb_g, up_lb, idx2, 0, m->stream) == 0 &&
             mlx_take_axis(&dn_lb_g, dn_lb, idx2, 0, m->stream) == 0;
        mlx_array_free(idx2);
        CHK(ok, "expert linear bias gather");
        if (ok) {
            const int bg_shape[4] = {n_tokens, m->cfg.top_k, 1, m->cfg.hidden_size};
            mlx_array g2 = mlx_array_new(), u2 = mlx_array_new(), d2 = mlx_array_new();
            ok = mlx_reshape(&g2, gp_lb_g, bg_shape, 4, m->stream) == 0 &&
                 mlx_reshape(&u2, up_lb_g, bg_shape, 4, m->stream) == 0 &&
                 mlx_reshape(&d2, dn_lb_g, bg_shape, 4, m->stream) == 0;
            if (ok) {
                mlx_array_free(gp_lb_g); gp_lb_g = g2;
                mlx_array_free(up_lb_g); up_lb_g = u2;
                mlx_array_free(dn_lb_g); dn_lb_g = d2;
            } else { mlx_array_free(g2); mlx_array_free(u2); mlx_array_free(d2); }
            CHK(ok, "expert bias reshape");
        }
    }

    mlx_array idx_tok = mlx_array_new();
    mlx_array inv_idx = mlx_array_new();
    mlx_array srt_idx = mlx_array_new();
    mlx_array perm_idx = mlx_array_new();
    if (ok && unsort_moe) {
        /* indici su device: rhs per gather_qmm = top_inds flatten [L] uint32 (ordine partition) */
        const int lsh2[1] = {L};
        mlx_array ti = mlx_array_new();
        ok = mlx_reshape(&ti, top_inds, lsh2, 1, m->stream) == 0;
        if (ok) ok = mlx_astype(&srt_idx, ti, MLX_UINT32, m->stream) == 0;
        mlx_array_free(ti);
        CHK(ok, "srt_idx device");
    } else if (ok) {
        /* Leva A sul prefill (L>=64): ordine sorted su DEVICE (argsort), niente readback host
         * per layer. flat = top_inds flatten [L] (int32). order = argsort(flat) (esperti asc);
         * inv = argsort(order) (inversa); tok = order/K (token per slot sorted); srt = esperti
         * in ordine sorted (rhs uint32); perm_idx = order (act partition -> sorted per down). */
        const int lsh2[1] = {L};
        mlx_array flat = mlx_array_new(), ord = mlx_array_new(), invtmp = mlx_array_new();
        mlx_array srt_i = mlx_array_new(), tkn = mlx_array_new();
        mlx_array ord_f = mlx_array_new(), ord_d = mlx_array_new(), ord_floor = mlx_array_new();
        mlx_array flat_u = mlx_array_new();
        ok = mlx_reshape(&flat, top_inds, lsh2, 1, m->stream) == 0;
        if (ok && mlx_array_dtype(flat) != MLX_INT32) ok = mlx_astype(&flat, flat, MLX_INT32, m->stream) == 0;
        if (ok) ok = mlx_argsort(&ord, flat, m->stream) == 0;
        if (ok) ok = mlx_argsort(&invtmp, ord, m->stream) == 0;
        if (ok) ok = mlx_take_along_axis(&srt_i, flat, ord, /*axis=*/0, m->stream) == 0;
        if (ok) { /* tok = floor(order/K) via float */
            ok = mlx_astype(&ord_f, ord, MLX_FLOAT32, m->stream) == 0;
            if (ok) { mlx_array kk = mlx_array_new_float32((float)K); ok = mlx_divide(&ord_d, ord_f, kk, m->stream) == 0; mlx_array_free(kk); }
            if (ok) ok = mlx_floor(&ord_floor, ord_d, m->stream) == 0;
            if (ok) ok = mlx_astype(&tkn, ord_floor, MLX_INT32, m->stream) == 0;
        }
        if (ok) ok = mlx_astype(&flat_u, srt_i, MLX_UINT32, m->stream) == 0;
        if (ok) {
            mlx_array_set(&idx_tok, tkn);
            mlx_array_set(&inv_idx, invtmp);
            mlx_array_set(&srt_idx, flat_u);
            mlx_array_set(&perm_idx, ord);
        }
        mlx_array_free(flat);
        mlx_array_free(srt_i);
        mlx_array_free(ord_f); mlx_array_free(ord_d); mlx_array_free(ord_floor);
        CHK(ok, "device sorted indices");
    }
    mlx_array no_lhs = mlx_array_new();
    mlx_array x_perm = mlx_array_new();
    if (ok && unsort_moe) {
        /* x rows in ordine partition: token t ripetuto K volte -> normed2 [1,T,H]
         * -> [1,T,1,H] -> repeat axis 2 K -> [1,T,K,1,H] -> reshape [L,1,H]. */
        const int r4[4] = {1, T, 1, m->cfg.hidden_size};
        const int r5[5] = {1, T, K, 1, m->cfg.hidden_size};
        const int r3[3] = {L, 1, m->cfg.hidden_size};
        mlx_array xc = mlx_array_new();
        ok = mlx_reshape(&xc, normed2, r4, 4, m->stream) == 0;
        if (ok) {
            mlx_array xr = mlx_array_new();
            ok = mlx_repeat_axis(&xr, xc, K, 2, m->stream) == 0;
            if (ok) ok = mlx_reshape(&xr, xr, r5, 5, m->stream) == 0;
            if (ok) ok = mlx_reshape(&x_perm, xr, r3, 3, m->stream) == 0;
            mlx_array_free(xr);
        }
        mlx_array_free(xc);
        CHK(ok, "x_perm (unsort device)");
    } else if (ok) {
        mlx_array xp = mlx_array_new();
        ok = mlx_take_axis(&xp, normed2, idx_tok, /*axis=*/1, m->stream) == 0;
        if (ok) { const int xps[3] = {L, 1, m->cfg.hidden_size}; ok = mlx_reshape(&x_perm, xp, xps, 3, m->stream) == 0; }
        mlx_array_free(xp);
        CHK(ok, "x_perm");
    }
    int xp_group, xp_bits; char xp_mode[16];
    llm_mlx_quant_derive(gp_w, gp_s, gsz, bts, "affine", &xp_group, &xp_bits, xp_mode, sizeof(xp_mode));
    mlx_optional_int xp_gsz = {xp_group, true};
    mlx_optional_int xp_bts = {xp_bits, true};

    /* Leva f16-attivazione mxfp4 pura (LLM_MXFP4_ACT_F16, default OFF — vedi helper): casta
     * la lhs delle gather ESPERTI a f16 SOLO quando i pesi sono mxfp4 4/g32 E l'attivazione
     * arriva f32 (percorso mxfp4 puro: g_bf16_all falso -> rms_norm non riduce). Non tocca
     * il resto del modello. Fallback silenzioso a f32 se l'astype fallisce (mai bloccante). */
    const bool mxfp4_act16 = llm_mlx_mxfp4_act_f16_enabled()
        && strcmp(xp_mode, "mxfp4") == 0 && xp_bits == 4 && xp_group == 32
        && mlx_array_dtype(normed2) == MLX_FLOAT32;
    if (ok && mxfp4_act16) {
        mlx_array xp16 = mlx_array_new();
        if (mlx_astype(&xp16, x_perm, MLX_FLOAT16, m->stream) == 0) {
            mlx_array_free(x_perm); x_perm = xp16;
        } else { mlx_array_free(xp16); }
    }

    /* ── Kernel MoE fuso MXFP4 (gpt-oss, LLMENGINE_MOE_FUSE_MOE_MXFP4, default OFF) ──────────────
     * Decode puro (n_tokens==1, unsort_moe, esperti mxfp4 4-bit/g32): 2 dispatch Metal
     * (gate+up+attivazione clamped, poi down+score+bias) + 1 sum, invece di 3 gather_qmm +
     * activation + scale/sum. Quando ATTIVO (non VERIFY) moe_out e' prodotto QUI e i passi
     * gather_qmm sotto sono saltati (moe_fused_done). In VERIFY (default con _VERIFY=1) NON
     * tocca moe_out -- il path gather_qmm resta autoritativo -- calcola solo il fused e
     * stampa il diff. Fallback silenzioso se una guardia non e' soddisfatta o il kernel
     * fallisce. NB: la collapse dei gather di bias/x_perm (sopra) non e' ancora fusA qui --
     * il guadagno pieno di dispatch si avra' ad attivazione, spostando anche quelli. */
    bool moe_fused_done = false;
    bool moe_fused_valid = false;
    const bool mxfp4_fuse_verify = llm_mlx_fuse_moe_mxfp4_verify_enabled();
    mlx_array fused_summed = mlx_array_new();
    mlx_array moe_out = mlx_array_new();
    if (ok && n_tokens == 1 && unsort_moe
        && strcmp(xp_mode, "mxfp4") == 0 && xp_bits == 4 && xp_group == 32
        && (llm_mlx_fuse_moe_mxfp4_enabled() || mxfp4_fuse_verify)) {
        const int din = m->cfg.hidden_size;
        const int dff = mlx_array_dim(gp_w, 1);   /* intermedia reale dal tensore impilato */
        const int k = m->cfg.top_k;
        /* T = dtype dell'ATTIVAZIONE (normed2), non del residuo h: h e' f32 dai layer>=1
         * ma rms_norm riduce a bf16 (compute dtype), e il kernel deriva il proprio T da x
         * (=normed2) e pretende bias dello STESSO dtype. Derivare T da h castava i bias a
         * f32 -> guardia gate_lb==T(x=bf16) falliva su ogni layer tranne il 0. */
        const mlx_dtype T = mlx_array_dtype(normed2);
        const int xsh[1] = {din}, ksh[1] = {k};
        mlx_array x_flat = mlx_array_new(), sc_flat = mlx_array_new();
        mlx_array glb = mlx_array_new(), ulb = mlx_array_new(), dlb = mlx_array_new();
        bool fok = mlx_reshape(&x_flat, normed2, xsh, 1, m->stream) == 0;
        /* scores: "weights" e' softmax su gate in dtype T. Il kernel (fedelta' bit) usa
         * score in T, come l'eager (multiply(down_out[,], w_arr) con entrambi T). */
        if (fok) fok = mlx_reshape(&sc_flat, weights, ksh, 1, m->stream) == 0;
        /* bias lineare per-esperto: dtype deve coincidere con x (kernel li dichiara T*) */
        if (fok) fok = (mlx_array_dtype(gp_lb) == T) ? mlx_array_set(&glb, gp_lb) == 0
                                                     : mlx_astype(&glb, gp_lb, T, m->stream) == 0;
        if (fok) fok = (mlx_array_dtype(up_lb) == T) ? mlx_array_set(&ulb, up_lb) == 0
                                                     : mlx_astype(&ulb, up_lb, T, m->stream) == 0;
        if (fok) fok = (mlx_array_dtype(dn_lb) == T) ? mlx_array_set(&dlb, dn_lb) == 0
                                                     : mlx_astype(&dlb, dn_lb, T, m->stream) == 0;
        /* srt_idx (decode/unsort) e' gia' [L]=[k] uint32 in ordine partition */
        if (fok) fok = llm_mlx_moe_fused_decode_mxfp4(&fused_summed, x_flat, srt_idx,
                     gp_w, gp_s, glb, up_w, up_s, ulb, dn_w, dn_s, dlb, sc_flat,
                     din, dff, k, m->stream);
        moe_fused_valid = fok;
        if (fok && !mxfp4_fuse_verify) {
            const int o3[3] = {1, n_tokens, din};
            fok = mlx_reshape(&moe_out, fused_summed, o3, 3, m->stream) == 0;
            moe_fused_done = fok;
            if (!fok) fprintf(stderr, "gpt_oss_ffn[layer=%d]: fused mxfp4 output build failed -> gather_qmm\n", layer_idx);
        }
        mlx_array_free(x_flat); mlx_array_free(sc_flat);
        mlx_array_free(glb); mlx_array_free(ulb); mlx_array_free(dlb);
    }

    if (ok && !moe_fused_done && pad512) {
        mlx_array xp_pad = mlx_array_new();
        if (llm_mlx_pad_last(&xp_pad, x_perm, pad_in, m->stream)) { mlx_array_free(x_perm); x_perm = xp_pad; }
        else mlx_array_free(xp_pad);
    }
    mlx_array gate_raw = mlx_array_new(), up_raw = mlx_array_new();
    /* LLM_FUSE_GATEUP: un solo gather_qmm su [E,2*inter,H] invece di due (bit-identico). */
    LlmGateUp* gu = (layer_idx >= 0 && layer_idx < LLM_PAD512_MAX_LAYERS) ? &g_gateup[layer_idx] : NULL;
    bool gu_use = ok && !moe_fused_done && !pad512 && gu != NULL && llm_mlx_fuse_gateup_enabled() && getenv("LLMENGINE_MOE_MLX_MOE_SWAP_DIR") == NULL;
    if (gu_use && !gu->ready) {
        bool cok = llm_mlx_concat2_axis(&gu->w, gp_w_s, up_w_s, 1, m->stream)
                && llm_mlx_concat2_axis(&gu->s, gp_s_s, up_s_s, 1, m->stream);
        if (cok && gp_b.ctx != NULL && up_b.ctx != NULL
            && mlx_array_size(gp_b) > 0 && mlx_array_size(up_b) > 0)
            cok = llm_mlx_concat2_axis(&gu->b, gp_b, up_b, 1, m->stream);
        if (cok) gu->ready = true; else gu_use = false;
    }
    if (gu_use && !gu->ready) gu_use = false;
    if (ok && !moe_fused_done && !gu_use) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_raw, x_perm, gp_w_s, gp_s_s, gp_b, no_lhs, srt_idx,
                                true, xp_gsz, xp_bts, xp_mode, !unsort_moe, m->stream) == 0;
    CHK(ok, "gq gate");
    if (ok && !moe_fused_done && !gu_use) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_raw, x_perm, up_w_s, up_s_s, up_b, no_lhs, srt_idx,
                                true, xp_gsz, xp_bts, xp_mode, !unsort_moe, m->stream) == 0;
    CHK(ok, "gq up");
    if (ok && !moe_fused_done && gu_use) {
        mlx_array gu_raw = mlx_array_new();
        ok = llm_mlx_moe_gather_mm(&m->cfg, &gu_raw, x_perm, gu->w, gu->s, gu->b, no_lhs, srt_idx,
                                true, xp_gsz, xp_bts, xp_mode, !unsort_moe, m->stream) == 0;
        if (ok) {
            const int nd = mlx_array_ndim(gu_raw);
            const int dim = mlx_array_dim(gu_raw, nd - 1);
            const int inter = dim / 2;
            const int* sh = mlx_array_shape(gu_raw);
            int st[8], sp[8], ss[8];
            for (int i = 0; i < nd; i++) { st[i] = 0; sp[i] = sh[i]; ss[i] = 1; }
            sp[nd - 1] = inter;
            ok = mlx_slice(&gate_raw, gu_raw, st, nd, sp, nd, ss, nd, m->stream) == 0;
            if (ok) { st[nd - 1] = inter; sp[nd - 1] = dim;
                      ok = mlx_slice(&up_raw, gu_raw, st, nd, sp, nd, ss, nd, m->stream) == 0; }
        }
        mlx_array_free(gu_raw);
        CHK(ok, "gq gate_up fused slice");
    }
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    if (ok && !moe_fused_done && unsort_moe) {
        /* gate_raw/up_raw gia' in ordine partition [L,1,H] -> [T,K,1,H] (nessun unsort) */
        const int gk[4] = {T, K, 1, m->cfg.hidden_size};
        ok = mlx_reshape(&gate_out, gate_raw, gk, 4, m->stream) == 0 &&
             mlx_reshape(&up_out, up_raw, gk, 4, m->stream) == 0;
        if (ok) ok = mlx_add(&gate_out, gate_out, gp_lb_g, m->stream) == 0 &&
                     mlx_add(&up_out, up_out, up_lb_g, m->stream) == 0;
    } else if (ok && !moe_fused_done) {
        /* stile mlx (single unsort finale): gate/up restano in ordine SORTED [L,1,H], bias per-slot
         * gatherato in ordine sorted -> niente unsort/perm intermedi (3 gather full-matrix in meno
         * per layer vs la vecchia arrange-partition->perm-sorted). */
        const int lb[3] = {L, 1, m->cfg.hidden_size};
        mlx_array gb = mlx_array_new(), ub = mlx_array_new();
        mlx_array gt = mlx_array_new(), ut = mlx_array_new();
        ok = mlx_take_axis(&gt, gp_lb, srt_idx, /*axis=*/0, m->stream) == 0 &&
             mlx_take_axis(&ut, up_lb, srt_idx, /*axis=*/0, m->stream) == 0;
        if (ok) ok = mlx_reshape(&gb, gt, lb, 3, m->stream) == 0 &&
                     mlx_reshape(&ub, ut, lb, 3, m->stream) == 0;
        if (ok) ok = mlx_add(&gate_out, gate_raw, gb, m->stream) == 0 &&
                     mlx_add(&up_out, up_raw, ub, m->stream) == 0;
        mlx_array_free(gb); mlx_array_free(ub); mlx_array_free(gt); mlx_array_free(ut);
    }
    CHK(ok, "arrange gate/up");
    if (ok && unsort_moe && !moe_fused_done)
        CHK(ok, "expert gate/up linear bias add");
    static int dump_gu_done = 0;
    if (ok && !moe_fused_done && !dump_gu_done && llm_env_get("LLM_DEBUG_GATEUP")) {
        mlx_array gv = mlx_array_new();
        if (mlx_astype(&gv, gate_out, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(gv) == 0) {
            const float* d = mlx_array_data_float32(gv);
            size_t ng = (size_t)n_tokens * (size_t)m->cfg.top_k * (size_t)m->cfg.hidden_size;
            char p[512]; snprintf(p, sizeof(p), "%s.gate", llm_env_get("LLM_DEBUG_GATEUP"));
            FILE* f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(float), ng, f); fclose(f); }
        }
        mlx_array_free(gv);
        mlx_array uv = mlx_array_new();
        if (mlx_astype(&uv, up_out, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(uv) == 0) {
            const float* d = mlx_array_data_float32(uv);
            size_t ng = (size_t)n_tokens * (size_t)m->cfg.top_k * (size_t)m->cfg.hidden_size;
            char p[512]; snprintf(p, sizeof(p), "%s.up", llm_env_get("LLM_DEBUG_GATEUP"));
            FILE* f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(float), ng, f); fclose(f); }
        }
        mlx_array_free(uv);
        dump_gu_done = 1;
    }

    mlx_array clip_g = mlx_array_new(), clip_u = mlx_array_new();
    mlx_array up1 = mlx_array_new(), alpha_g = mlx_array_new();
    mlx_array sig = mlx_array_new(), act = mlx_array_new();
    bool gptoss_swiglu_done = false;
    /* Kernel Metal fuso (default ON): collassa le ~13 op eager in 1 dispatch, come il
     * `Compiled` di mlx_lm. Fallback eager se il kernel non e' disponibile. */
    if (ok && !moe_fused_done && llm_mlx_fuse_swiglu_gptoss_enabled()) {
        gptoss_swiglu_done = llm_mlx_swiglu_gptoss_metal(gate_out, up_out, &act, m->stream);
    }
    if (ok && !moe_fused_done && !gptoss_swiglu_done) {
        /* Scalari nel DTYPE DI ATTIVAZIONE (bf16): un f32 forte promuoveva l'intero SwiGLU a f32
         * (382 AsType, ~2,2 GB/token sul prefill gpt-oss). Vedi llm_mlx_scalar_act. */
        mlx_array seven = llm_mlx_scalar_act(7.0f, m->stream);
        mlx_array mseven = llm_mlx_scalar_act(-7.0f, m->stream);
        mlx_array one = llm_mlx_scalar_act(1.0f, m->stream);
        mlx_array alpha = llm_mlx_scalar_act(1.702f, m->stream);
        ok = mlx_minimum(&clip_g, gate_out, seven, m->stream) == 0 &&
             mlx_clip(&clip_u, up_out, mseven, seven, m->stream) == 0;
        if (ok) { ok = mlx_multiply(&alpha_g, clip_g, alpha, m->stream) == 0 && mlx_sigmoid(&sig, alpha_g, m->stream) == 0; }
        if (ok) { ok = mlx_multiply(&sig, sig, clip_g, m->stream) == 0; }
        if (ok) { ok = mlx_add(&up1, clip_u, one, m->stream) == 0; }
        if (ok) { ok = mlx_multiply(&act, sig, up1, m->stream) == 0; }
        mlx_array_free(seven); mlx_array_free(mseven); mlx_array_free(one); mlx_array_free(alpha);
        CHK(ok, "swiglu");
    }
    /* Leva f16-attivazione: anche la lhs della gather down (act) va a f16 per il ~2x su
     * tutte e tre le proiezioni. act e' f32 qui perche' l'add dei bias (f32) promuove
     * l'uscita gate/up f16 -> f32; il cast la riporti a f16 solo per la matmul low-bit. */
    if (ok && !moe_fused_done && mxfp4_act16) {
        mlx_array act16 = mlx_array_new();
        if (mlx_astype(&act16, act, MLX_FLOAT16, m->stream) == 0) {
            mlx_array_free(act); act = act16;
        } else { mlx_array_free(act16); }
    }
    if (ok && !moe_fused_done && pad512) {
        mlx_array act_pad = mlx_array_new();
        if (llm_mlx_pad_last(&act_pad, act, pad_ff, m->stream)) { mlx_array_free(act); act = act_pad; }
        else mlx_array_free(act_pad);
    }
    mlx_array down_out = mlx_array_new();
    if (ok && !moe_fused_done) {
        mlx_array down_raw = mlx_array_new();
        if (ok && unsort_moe) {
            /* act partition [T,K,1,H] -> [L,1,H]; down unsorted (rhs top_inds); reshape -> partition */
            mlx_array actx = mlx_array_new();
            const int ax[3] = {L, 1, (int)mlx_array_dim(act, mlx_array_ndim(act) - 1)};
            ok = mlx_reshape(&actx, act, ax, 3, m->stream) == 0;
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_raw, actx, dn_w_s, dn_s_s, dn_b, no_lhs, srt_idx,
                                        true, xp_gsz, xp_bts, xp_mode, false, m->stream) == 0;
            if (ok) { const int dk[4] = {T, K, 1, m->cfg.hidden_size}; ok = mlx_reshape(&down_out, down_raw, dk, 4, m->stream) == 0; }
            mlx_array_free(actx);
        } else if (ok) {
            /* stile mlx: act gia' sorted [L,1,H]; down sorted; UN SOLO unsort finale -> partition */
            ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_raw, act, dn_w_s, dn_s_s, dn_b, no_lhs, srt_idx,
                                true, xp_gsz, xp_bts, xp_mode, true, m->stream) == 0;
            if (ok) ok = gpt_oss_expert_unsort(&down_out, down_raw, inv_idx, T, K, m->stream);
        }
        CHK(ok, "gq down");
        mlx_array_free(down_raw);
    }
    if (ok && !moe_fused_done) ok = mlx_add(&down_out, down_out, dn_lb_g, m->stream) == 0;
    CHK(ok, "expert down linear bias add");
    static int dump_dn_done = 0;
    if (ok && !moe_fused_done && !dump_dn_done && llm_env_get("LLM_DEBUG_GATEUP")) {
        mlx_array dv = mlx_array_new();
        if (mlx_astype(&dv, down_out, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(dv) == 0) {
            const float* d = mlx_array_data_float32(dv);
            size_t ng = (size_t)n_tokens * (size_t)m->cfg.top_k * (size_t)m->cfg.hidden_size;
            char p[512]; snprintf(p, sizeof(p), "%s.down", llm_env_get("LLM_DEBUG_GATEUP"));
            FILE* f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(float), ng, f); fclose(f); }
        }
        mlx_array_free(dv);
        dump_dn_done = 1;
    }

    /* Q2-b: readback di srt_idx/inv_idx (solo calibrazione) PRIMA che vengano liberati sotto. */
    if (m->im.active && ok && srt_idx.ctx != NULL && inv_idx.ctx != NULL) {
        const int Mi = L;
        if (mlx_array_eval(srt_idx) == 0 && mlx_array_eval(inv_idx) == 0 &&
            mlx_array_data_uint32(srt_idx) != NULL && mlx_array_data_uint32(inv_idx) != NULL) {
            im_expert = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
            im_src = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
            if (im_expert && im_src) {
                memcpy(im_expert, mlx_array_data_uint32(srt_idx), (size_t)Mi * sizeof(uint32_t));
                memcpy(im_src, mlx_array_data_uint32(inv_idx), (size_t)Mi * sizeof(uint32_t));
                im_rows = Mi;
            }
        }
    }
    mlx_array_free(idx_tok); mlx_array_free(inv_idx); mlx_array_free(srt_idx); mlx_array_free(perm_idx);
    mlx_array_free(no_lhs); mlx_array_free(x_perm); mlx_array_free(gate_raw); mlx_array_free(up_raw);
    /* pesi softmax top-k: [n_tokens, top_k] -> [n_tokens, top_k, 1, 1] */
    mlx_array scaled = mlx_array_new();
    if (ok && !moe_fused_done) {
        const int ws_shape[4] = {n_tokens, m->cfg.top_k, 1, 1};
        mlx_array w_arr = mlx_array_new();
        ok = mlx_reshape(&w_arr, weights, ws_shape, 4, m->stream) == 0;
        if (ok) ok = mlx_multiply(&scaled, down_out, w_arr, m->stream) == 0;
        mlx_array_free(w_arr);
        CHK(ok, "moe score scale");
    }
    mlx_array summed = mlx_array_new();
    if (ok && !moe_fused_done) { ok = mlx_sum_axis(&summed, scaled, /*axis=*/1, /*keepdims=*/false, m->stream) == 0; CHK(ok, "moe sum over top_k"); }
    if (ok && !moe_fused_done) {
        const int out_shape[3] = {1, n_tokens, m->cfg.hidden_size};
        ok = mlx_reshape(&moe_out, summed, out_shape, 3, m->stream) == 0;
        CHK(ok, "moe_out reshape");
    }

    /* VERIFY: confronta il fused (fused_summed [din] f32) col risultato gather_qmm autoritativo
     * (summed [T,1,H], T=1 -> [din]), USA SEMPRE il gather_qmm -- zero rischio, stampa solo. */
    if (moe_fused_valid && !moe_fused_done && mxfp4_fuse_verify) {
        const int din = m->cfg.hidden_size;
        mlx_array af = mlx_array_new(), bf = mlx_array_new(), ref_flat = mlx_array_new();
        const int rsh[1] = {din};
        if (mlx_reshape(&ref_flat, summed, rsh, 1, m->stream) == 0 &&
            mlx_astype(&af, fused_summed, MLX_FLOAT32, m->stream) == 0 &&
            mlx_astype(&bf, ref_flat, MLX_FLOAT32, m->stream) == 0 &&
            mlx_array_eval(af) == 0 && mlx_array_eval(bf) == 0) {
            const float* ad = mlx_array_data_float32(af);
            const float* bd = mlx_array_data_float32(bf);
            if (ad && bd) {
                double maxdiff = 0, sumsq = 0, refsq = 0;
                for (int vi = 0; vi < din; vi++) {
                    double d = fabs((double)ad[vi] - (double)bd[vi]);
                    if (d > maxdiff) maxdiff = d;
                    sumsq += d * d; refsq += (double)bd[vi] * (double)bd[vi];
                }
                double rms = refsq > 0 ? sqrt(sumsq / refsq) : sqrt(sumsq);
                fprintf(stderr, "[fuse-moe-mxfp4-verify] gpt_oss layer=%d max_abs_diff=%.6g rel_rms=%.6g\n",
                        layer_idx, maxdiff, rms);
            }
        }
        mlx_array_free(af); mlx_array_free(bf); mlx_array_free(ref_flat);
    }
    mlx_array_free(fused_summed);

    if (ok) { ok = mlx_add(out, h, moe_out, m->stream) == 0; CHK(ok, "final residual add"); }

    /* Q2-b: accumulo imatrix — energia di gate/up (input normed2) e down (riga `act`). */
    if (m->im.active && ok && im_expert && im_src && im_rows > 0 && act.ctx != NULL)
        llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, im_expert, im_src, im_rows);
    free(im_expert); free(im_src);
    mlx_array all[] = {
        ln2_w, normed2, gw, gs, gb, g_lb, gates0, gates, part, top_inds, topk_vals, weights,
        gp_w, gp_s, gp_b, up_w, up_s, up_b, dn_w, dn_s, dn_b,
        gp_lb, up_lb, dn_lb, gp_lb_g, up_lb_g, dn_lb_g,
        gate_out, up_out,
        clip_g, clip_u, up1, alpha_g, sig, act, down_out, scaled, summed, moe_out,
        gp_w_u, gp_s_u, up_w_u, up_s_u, dn_w_u, dn_s_u, /* PAD512 (default OFF: vuoti; il contatore ownership li vedeva come leak) */
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
