/* ============================================================================
 * llm_mlx_attn_batch.c — dominio attenzione/forward BATCHATO varlen multi-sessione (FASE 3b) + forward_tokens_batch prefill.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_dispatch_profile.h"
#include "llm_mlx_fused_residual_rms.h"
#include "llm_mlx_ops.h"
#include "oss/llm_oss_attn.h"   /* helper attenzione GPT-OSS (sinks/kv-f16/sdpa/oproj) */
#include "llm_mlx_kv_quant_sdpa_kernel.h"
#include "llm_mlx_gdn_gate_kernel.h"

/* HOOK DFlash2 per il prefill (2026-10-01): definizioni uniche. Attivo solo se armato. */
int g_df2_pf_active = 0;
void (*g_df2_pf_hook)(int layer, mlx_array h, int n_tokens, int cache_offset) = NULL;
#include "llm_mlx_gdn_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* ============================================================================
 * FASE 3b (2026-09-11): attenzione BATCHATA (varlen) per il decode multi-sessione.
 * Proiezioni QKV/O e norm BATCHATE su tutte le sessioni (N GEMV -> 1 GEMM); rope,
 * cache append e SDPA restano PER-SESSIONE (la SDPA dipende dalla cache di ciascuna
 * sessione, lunghezze diverse). Default OFF (LLM_BATCH_VARLEN), SOLO llama; qualunque
 * condizione fuori dal subset => return false: il chiamante usa il loop per-sessione.
 * Parita' verificata con `tools/mlx_batch_test.sh parity` (deve dare IDS identici).
 * ========================================================================== */
static int g_batch_varlen_state = -1;
static bool llm_mlx_batch_varlen_enabled(void) {
    if (g_batch_varlen_state < 0) {
        g_batch_varlen_state = 1; /* default ON (llama-only, fallback sicuro su ogni caso non coperto) */
        const char* e = llm_ai_mlx_opt_env("LLM_BATCH_VARLEN");
        if (e != NULL && (e[0] == '0' || e[0] == 'n' || e[0] == 'N' ||
                          e[0] == 'f' || e[0] == 'F' || e[0] == 'o' || e[0] == 'O'))
            g_batch_varlen_state = 0;
    }
    return g_batch_varlen_state == 1;
}

static bool llama_attention_block_batch(LlmMlxModel* m, int layer_idx, LlmMlxSession** sess,
                                        int n_sess, mlx_array x, mlx_array* out_h) {
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const mlx_stream st = m->stream;
    GattnQuant* ac = llm_mlx_attn_quant(m, layer_idx);
    const bool is_qwen2 = strcmp(m->cfg.model_type, "qwen2") == 0;
    const bool is_qwen3 = strcmp(m->cfg.model_type, "qwen3") == 0 ||
                          strcmp(m->cfg.model_type, "qwen3_moe") == 0;
    const bool is_gptoss = strcmp(m->cfg.model_type, "gpt_oss") == 0;
    const bool is_mixtral = strcmp(m->cfg.model_type, "mixtral") == 0;
    const bool is_glm4 = strcmp(m->cfg.model_type, "glm4_moe") == 0;
    /* RoPE parziale (glm4_moe: ruota solo head_dim*partial_rotary_factor dims). */
    int rope_dims = hd;
    if (is_glm4 && m->cfg.partial_rotary_factor > 0.0f && m->cfg.partial_rotary_factor < 1.0f)
        rope_dims = (int)(hd * m->cfg.partial_rotary_factor);
    if (llm_env_isset("LLM_BATCH_VARLEN_DEBUG")) {
        static int g_varlen_dbg_once = 0;
        if (!g_varlen_dbg_once) { g_varlen_dbg_once = 1;
            fprintf(stderr, "[varlen-dbg] n_sess=%d model=%s kv_max=%d kvq=%d kvslice=%d kvseg=%d kvmetal=%d auth=%d ver=%d ac=%p valid=%d has_v=%d hc0=%d lora=%d\n",
                n_sess, m->cfg.model_type, m->cfg.kv_max_tokens, (int)m->cfg.kv_quantize,
                (int)llm_mlx_kvslice_enabled(), (int)llm_mlx_kvseg_enabled(),
                (int)llm_mlx_kvmetal_enabled(), (int)llm_mlx_kvmetal_auth_enabled(),
                (int)llm_mlx_kvmetal_verify_enabled(), (void*)ac, (int)(ac?ac->valid:0), (int)(ac?ac->has_v:0),
                (int)(sess && sess[0] ? sess[0]->has_cache[layer_idx] : -1),
                (int)(llm_mlx_model_lora_get(m, layer_idx, "q_proj") >= 0)); }
    }
    if (n_sess < 2 || nh <= 0 || nkv <= 0 || hd <= 0) return false;
    if (m->cfg.nsa_sparse) return false;
    if (is_mixtral && llm_mlx_fixkv_enabled()) return false; /* cache statica mixtral non coperta */
    if (m->cfg.kv_max_tokens > 0 || m->cfg.kv_quantize) return false;
    if (llm_mlx_kvslice_enabled() || llm_mlx_kvseg_enabled()) return false;
    if (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_auth_enabled() || llm_mlx_kvmetal_verify_enabled()) return false;
    if (!ac || !ac->valid) return false;
    if (llm_mlx_model_lora_get(m, layer_idx, "q_proj") >= 0 ||
        llm_mlx_model_lora_get(m, layer_idx, "k_proj") >= 0 ||
        llm_mlx_model_lora_get(m, layer_idx, "v_proj") >= 0 ||
        llm_mlx_model_lora_get(m, layer_idx, "o_proj") >= 0) return false;
    for (int s = 0; s < n_sess; s++)
        if (sess[s] == NULL || !sess[s]->has_cache[layer_idx]) return false;

    bool ok = true;
    mlx_array ln_w = mlx_array_new(), qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    ok = ok && mlx_array_set(&ln_w, ac->ln1) == 0;
    ok = ok && mlx_array_set(&qw, ac->qw) == 0 && mlx_array_set_optional(&qs, ac->qs) && mlx_array_set_optional(&qb, ac->qb);
    ok = ok && mlx_array_set(&kw, ac->kw) == 0 && mlx_array_set_optional(&ks, ac->ks) && mlx_array_set_optional(&kb, ac->kb);
    ok = ok && mlx_array_set(&vw, ac->vw) == 0 && mlx_array_set_optional(&vs, ac->vs) && mlx_array_set_optional(&vb, ac->vb);
    ok = ok && mlx_array_set(&ow, ac->ow) == 0 && mlx_array_set_optional(&os, ac->os) && mlx_array_set_optional(&ob, ac->ob);

    mlx_array normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&normed, x, ln_w, m->cfg.rms_norm_eps, st);
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&q, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, st);
    if (ok) ok = quantized_linear_cfg(&k, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, st);
    if (ok) ok = quantized_linear_cfg(&v, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, st);

    /* qwen2 / gpt-oss / glm4_moe: bias REALI su q/k/v_proj
     * (tensori float separati dai .biases di quantizzazione). Llama non li ha. */
    if (ok && (is_qwen2 || is_gptoss || is_glm4)) {
        char bnb[256];
        mlx_array q_bias = mlx_array_new(), k_bias = mlx_array_new(), v_bias = mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj.bias", bnb, sizeof(bnb));
        ok = ok && multishard_get(&m->weights, bnb, &q_bias);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj.bias", bnb, sizeof(bnb));
        ok = ok && multishard_get(&m->weights, bnb, &k_bias);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj.bias", bnb, sizeof(bnb));
        ok = ok && multishard_get(&m->weights, bnb, &v_bias);
        mlx_array q2 = mlx_array_new(), k2 = mlx_array_new(), v2 = mlx_array_new();
        if (ok) ok = mlx_add(&q2, q, q_bias, st) == 0 &&
                     mlx_add(&k2, k, k_bias, st) == 0 &&
                     mlx_add(&v2, v, v_bias, st) == 0;
        if (ok) { mlx_array_free(q); q = q2; mlx_array_free(k); k = k2; mlx_array_free(v); v = v2; }
        else { mlx_array_free(q2); mlx_array_free(k2); mlx_array_free(v2); }
        mlx_array_free(q_bias); mlx_array_free(k_bias); mlx_array_free(v_bias);
    }

    /* gpt-oss: sinks in oss/ (llm_oss_attn.h). */
    mlx_array sinks = mlx_array_new();
    if (ok && is_gptoss) ok = llm_oss_load_sinks(m, layer_idx, st, &sinks);

    const int qsh[4] = {1, n_sess, nh, hd};
    const int kvsh[4] = {1, n_sess, nkv, hd};
    const int perm[4] = {0, 2, 1, 3};
    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) ok = mlx_reshape(&q_heads, q, qsh, 4, st) == 0 &&
                 mlx_reshape(&k_heads, k, kvsh, 4, st) == 0 &&
                 mlx_reshape(&v_heads, v, kvsh, 4, st) == 0;

    /* qwen3 / qwen3_moe: QK-norm sulle teste (rms_norm per-head) prima del
     * transpose. qwen2/llama non l'hanno (ac->has_qn falso). */
    const bool has_qn = is_qwen3 && ac != NULL && ac->has_qn;
    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    mlx_array q_heads_n = mlx_array_new(), k_heads_n = mlx_array_new();
    if (ok && has_qn) {
        ok = mlx_array_set_optional(&qn_w, ac->qn_w) && mlx_array_set_optional(&kn_w, ac->kn_w);
        if (ok) ok = rms_norm_cfg(&q_heads_n, q_heads, qn_w, m->cfg.rms_norm_eps, st) &&
                     rms_norm_cfg(&k_heads_n, k_heads, kn_w, m->cfg.rms_norm_eps, st);
    }
    mlx_array q_src = has_qn ? q_heads_n : q_heads;
    mlx_array k_src = has_qn ? k_heads_n : k_heads;
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&q_t, q_src, perm, 4, st) == 0 &&
                 mlx_transpose_axes(&k_t, k_src, perm, 4, st) == 0 &&
                 mlx_transpose_axes(&v_t, v_heads, perm, 4, st) == 0;

    mlx_array freqs = mlx_array_new();
    if (ok && m->cfg.use_llama3_rope) {
        mlx_array_free(freqs); /* sovrascritto: l'handle vuoto iniziale va liberato */
        freqs = llm_mlx_llama3_freqs_cached(m);
    }

    mlx_vector_array av = mlx_vector_array_new();
    for (int s = 0; s < n_sess && ok; s++) {
        const int co = sess[s]->cache_offset;
        const int q0[4] = {0, 0, s, 0}, q1[4] = {1, nh, s + 1, hd}, stride[4] = {1, 1, 1, 1};
        const int k0[4] = {0, 0, s, 0}, k1[4] = {1, nkv, s + 1, hd};
        mlx_array qs_e = mlx_array_new(), ks_e = mlx_array_new(), vs_e = mlx_array_new();
        ok = ok && mlx_slice(&qs_e, q_t, q0, 4, q1, 4, stride, 4, st) == 0 &&
                   mlx_slice(&ks_e, k_t, k0, 4, k1, 4, stride, 4, st) == 0 &&
                   mlx_slice(&vs_e, v_t, k0, 4, k1, 4, stride, 4, st) == 0;
        mlx_array qr = mlx_array_new(), kr = mlx_array_new();
        if (ok && m->cfg.use_yarn_rope) {
            ok = llm_mlx_yarn_rope_apply(&qr, &kr, hd, m->cfg.rope_theta,
                                         m->cfg.yarn_factor, m->cfg.yarn_orig_max,
                                         m->cfg.yarn_beta_fast, m->cfg.yarn_beta_slow, m->cfg.yarn_mscale,
                                         qs_e, ks_e, co, st);
        } else if (ok && m->cfg.use_llama3_rope) {
            mlx_optional_float nb = {0.0f, false};
            ok = mlx_fast_rope(&qr, qs_e, rope_dims, false, nb, 1.0f, co, freqs, st) == 0 &&
                 mlx_fast_rope(&kr, ks_e, rope_dims, false, nb, 1.0f, co, freqs, st) == 0;
        } else if (ok) {
            mlx_optional_float base = {m->cfg.rope_theta, true};
            mlx_array nof = mlx_array_new();
            ok = mlx_fast_rope(&qr, qs_e, rope_dims, false, base, 1.0f, co, nof, st) == 0 &&
                 mlx_fast_rope(&kr, ks_e, rope_dims, false, base, 1.0f, co, nof, st) == 0;
            mlx_array_free(nof);
        }
        const bool kvf16 = is_gptoss ? llm_oss_kv_f16_layer(m, layer_idx) : llm_mlx_kv_f16_enabled();
        if (ok && kvf16 && mlx_array_dtype(kr) != MLX_FLOAT16) {
            mlx_array kr16 = mlx_array_new(), vt16 = mlx_array_new();
            if (mlx_astype(&kr16, kr, MLX_FLOAT16, st) == 0 && mlx_astype(&vt16, vs_e, MLX_FLOAT16, st) == 0) {
                mlx_array_free(kr); kr = kr16; mlx_array_free(vs_e); vs_e = vt16;
            } else { mlx_array_free(kr16); mlx_array_free(vt16); }
        }
        mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
        if (ok) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, sess[s]->k_cache[layer_idx]);
            mlx_vector_array_append_value(kv, kr);
            ok = mlx_concatenate_axis(&k_full, kv, 2, st) == 0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, sess[s]->v_cache[layer_idx]);
            mlx_vector_array_append_value(vv, vs_e);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, st) == 0; mlx_vector_array_free(vv);
        }
        if (ok) ok = mlx_array_set(&sess[s]->k_cache[layer_idx], k_full) == 0 &&
                     mlx_array_set(&sess[s]->v_cache[layer_idx], v_full) == 0;
        mlx_array attn_s = mlx_array_new();
        if (ok) {
            if (is_gptoss) {
                ok = llm_oss_attn_sdpa(&attn_s, m, layer_idx, qr, sess[s]->k_cache[layer_idx],
                                       sess[s]->v_cache[layer_idx], hd, sinks, st);
            } else {
                ok = llm_mlx_gqa_sdpa(&attn_s, qr, sess[s]->k_cache[layer_idx],
                                      sess[s]->v_cache[layer_idx],
                                      1.0f / sqrtf((float)hd), /*window=*/0, st);
            }
        }
        if (ok) mlx_vector_array_append_value(av, attn_s);
        mlx_array_free(qs_e); mlx_array_free(ks_e); mlx_array_free(vs_e);
        mlx_array_free(qr); mlx_array_free(kr);
        mlx_array_free(k_full); mlx_array_free(v_full);
        mlx_array_free(attn_s);
    }

    mlx_array attn = mlx_array_new();
    if (ok) ok = mlx_concatenate_axis(&attn, av, 2, st) == 0;
    mlx_vector_array_free(av);
    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, st) == 0;
    if (ok) { const int fsh[3] = {1, n_sess, nh * hd}; ok = mlx_reshape(&attn_flat, attn_bt, fsh, 3, st) == 0; }
    mlx_array attn_prj = mlx_array_new(), attn_out = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&attn_prj, attn_flat, ow, os, ob, gsz, bts, m->cfg.quant_mode, st);
    if (ok && is_gptoss) {
        ok = llm_oss_oproj_bias(m, layer_idx, attn_prj, &attn_out, st);
    } else if (ok) {
        ok = mlx_array_set(&attn_out, attn_prj) == 0;
    }
    if (ok) ok = mlx_add(out_h, x, attn_out, st) == 0;

    mlx_array all[] = { ln_w, qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob,
                        normed, q, k, v, q_heads, k_heads, v_heads, qn_w, kn_w,
                        q_heads_n, k_heads_n, q_t, k_t, v_t, freqs, sinks,
                        attn, attn_bt, attn_flat, attn_prj, attn_out };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);
    return ok;
}

/* gemma4 (Fase 1b.4): attenzione batchata. head_dim/n_kv_heads/rope_theta variano per tipo di
 * layer (sliding vs full/global); K=V sui layer full (v_heads = k_heads PRIMA di k_norm);
 * q_norm/k_norm (con peso) + v_norm (no-scale); RoPE sliding (base semplice) vs full (freq
 * proporzionali, partial rotary); SDPA a scala FISSA 1.0 con window sui layer sliding;
 * `post_attention_layernorm` sull'output PRIMA del residuo (specifico di gemma4).
 * Scaling su n_sess per proiezioni/norm; rope/cache/SDPA per-sessione (lunghezze diverse). */
static bool gemma4_attention_block_batch(LlmMlxModel* m, int layer_idx, LlmMlxSession** sess,
                                         int n_sess, mlx_array x, mlx_array* out_h) {
    const mlx_stream st = m->stream;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const bool is_sliding = m->cfg.is_sliding_layer[layer_idx];
    const bool use_k_eq_v = m->cfg.attention_k_eq_v && !is_sliding;
    const int nh = m->cfg.n_heads;
    const int head_dim = (!is_sliding && m->cfg.global_head_dim > 0) ? m->cfg.global_head_dim : m->cfg.head_dim;
    const int nkv = (use_k_eq_v && m->cfg.num_global_kv_heads > 0) ? m->cfg.num_global_kv_heads : m->cfg.n_kv_heads;
    const float rope_theta = is_sliding ? m->cfg.rope_theta_sliding : m->cfg.rope_theta_full;
    if (n_sess < 2 || nh <= 0 || nkv <= 0 || head_dim <= 0) return false;
    if (m->cfg.nsa_sparse || m->cfg.kv_max_tokens > 0 || m->cfg.kv_quantize) return false;
    if (llm_mlx_kvslice_enabled() || llm_mlx_kvseg_enabled()) return false;
    if (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_auth_enabled() || llm_mlx_kvmetal_verify_enabled()) return false;
    if (llm_mlx_flash_decode_enabled() || llm_mlx_flash_prefill_enabled()) return false;
    if (llm_mlx_model_lora_get(m, layer_idx, "q_proj") >= 0 ||
        llm_mlx_model_lora_get(m, layer_idx, "k_proj") >= 0 ||
        llm_mlx_model_lora_get(m, layer_idx, "v_proj") >= 0 ||
        llm_mlx_model_lora_get(m, layer_idx, "o_proj") >= 0) return false;
    for (int s = 0; s < n_sess; s++)
        if (sess[s] == NULL || !sess[s]->has_cache[layer_idx]) return false;
    GattnQuant* ac = llm_mlx_gemma4_attn_quant(m, layer_idx);
    if (!ac || !ac->valid || !ac->has_qn) return false;
    if (!use_k_eq_v && !ac->has_v) return false;

    bool ok = true;
    mlx_array ln_w = mlx_array_new();
    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    ok = ok && mlx_array_set(&ln_w, ac->ln1) == 0;
    ok = ok && mlx_array_set(&qw, ac->qw) == 0 && mlx_array_set_optional(&qs, ac->qs) && mlx_array_set_optional(&qb, ac->qb);
    ok = ok && mlx_array_set(&kw, ac->kw) == 0 && mlx_array_set_optional(&ks, ac->ks) && mlx_array_set_optional(&kb, ac->kb);
    ok = ok && mlx_array_set(&ow, ac->ow) == 0 && mlx_array_set_optional(&os, ac->os) && mlx_array_set_optional(&ob, ac->ob);
    ok = ok && mlx_array_set_optional(&qn_w, ac->qn_w) && mlx_array_set_optional(&kn_w, ac->kn_w);
    if (!use_k_eq_v)
        ok = ok && mlx_array_set(&vw, ac->vw) == 0 && mlx_array_set_optional(&vs, ac->vs) && mlx_array_set_optional(&vb, ac->vb);

    mlx_array normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&normed, x, ln_w, m->cfg.rms_norm_eps, st);
    mlx_array q0 = mlx_array_new(), k0 = mlx_array_new(), v0 = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&q0, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, st);
    if (ok) ok = quantized_linear_cfg(&k0, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, st);
    if (ok && !use_k_eq_v) ok = quantized_linear_cfg(&v0, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, st);

    const int qsh[4] = {1, n_sess, nh, head_dim};
    const int kvsh[4] = {1, n_sess, nkv, head_dim};
    const int perm[4] = {0, 2, 1, 3};
    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) ok = mlx_reshape(&q_heads, q0, qsh, 4, st) == 0 &&
                 mlx_reshape(&k_heads, k0, kvsh, 4, st) == 0;
    if (ok && use_k_eq_v) ok = mlx_array_set(&v_heads, k_heads) == 0;
    else if (ok) ok = mlx_reshape(&v_heads, v0, kvsh, 4, st) == 0;

    mlx_array q_normed = mlx_array_new(), k_normed = mlx_array_new(), v_normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&q_normed, q_heads, qn_w, m->cfg.rms_norm_eps, st);
    if (ok) ok = rms_norm_cfg(&k_normed, k_heads, kn_w, m->cfg.rms_norm_eps, st);
    if (ok) ok = mlx_fast_rms_norm_no_scale(&v_normed, v_heads, head_dim, m->cfg.rms_norm_eps, st);
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&q_t, q_normed, perm, 4, st) == 0 &&
                 mlx_transpose_axes(&k_t, k_normed, perm, 4, st) == 0 &&
                 mlx_transpose_axes(&v_t, v_normed, perm, 4, st) == 0;

    mlx_array custom_freqs = mlx_array_new();
    if (ok && !is_sliding) {
        int rotated_dims = (int)(head_dim * m->cfg.partial_rotary_factor);
        mlx_array_free(custom_freqs); /* sovrascritto: l'handle vuoto iniziale va liberato */
        custom_freqs = llm_mlx_proportional_rope_freqs(head_dim, rotated_dims, rope_theta, 1.0f);
    }

    mlx_vector_array av = mlx_vector_array_new();
    for (int s = 0; s < n_sess && ok; s++) {
        const int co = sess[s]->cache_offset;
        const int q0i[4] = {0, 0, s, 0}, q1i[4] = {1, nh, s + 1, head_dim}, stride[4] = {1, 1, 1, 1};
        const int k0i[4] = {0, 0, s, 0}, k1i[4] = {1, nkv, s + 1, head_dim};
        mlx_array qs_e = mlx_array_new(), ks_e = mlx_array_new(), vs_e = mlx_array_new();
        ok = ok && mlx_slice(&qs_e, q_t, q0i, 4, q1i, 4, stride, 4, st) == 0 &&
                   mlx_slice(&ks_e, k_t, k0i, 4, k1i, 4, stride, 4, st) == 0 &&
                   mlx_slice(&vs_e, v_t, k0i, 4, k1i, 4, stride, 4, st) == 0;
        mlx_array qr = mlx_array_new(), kr = mlx_array_new();
        if (ok && is_sliding) {
            mlx_optional_float base = {rope_theta, true};
            mlx_array nof = mlx_array_new();
            ok = mlx_fast_rope(&qr, qs_e, head_dim, false, base, 1.0f, co, nof, st) == 0 &&
                 mlx_fast_rope(&kr, ks_e, head_dim, false, base, 1.0f, co, nof, st) == 0;
            mlx_array_free(nof);
        } else if (ok) {
            mlx_optional_float no_base = {0.0f, false};
            ok = mlx_fast_rope(&qr, qs_e, head_dim, false, no_base, 1.0f, co, custom_freqs, st) == 0 &&
                 mlx_fast_rope(&kr, ks_e, head_dim, false, no_base, 1.0f, co, custom_freqs, st) == 0;
        }
        mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
        if (ok) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, sess[s]->k_cache[layer_idx]);
            mlx_vector_array_append_value(kv, kr);
            ok = mlx_concatenate_axis(&k_full, kv, 2, st) == 0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, sess[s]->v_cache[layer_idx]);
            mlx_vector_array_append_value(vv, vs_e);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, st) == 0; mlx_vector_array_free(vv);
        }
        if (ok) ok = mlx_array_set(&sess[s]->k_cache[layer_idx], k_full) == 0 &&
                     mlx_array_set(&sess[s]->v_cache[layer_idx], v_full) == 0;
        mlx_array attn_s = mlx_array_new();
        if (ok) {
            int win = (is_sliding && m->cfg.sliding_window > 0) ? m->cfg.sliding_window : 0;
            ok = llm_mlx_gqa_sdpa(&attn_s, qr, sess[s]->k_cache[layer_idx], sess[s]->v_cache[layer_idx],
                                  1.0f, win, st);
        }
        if (ok) mlx_vector_array_append_value(av, attn_s);
        mlx_array_free(qs_e); mlx_array_free(ks_e); mlx_array_free(vs_e);
        mlx_array_free(qr); mlx_array_free(kr);
        mlx_array_free(k_full); mlx_array_free(v_full);
        mlx_array_free(attn_s);
    }

    mlx_array attn = mlx_array_new();
    if (ok) ok = mlx_concatenate_axis(&attn, av, 2, st) == 0;
    mlx_vector_array_free(av);
    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, st) == 0;
    if (ok) { const int fsh[3] = {1, n_sess, nh * head_dim}; ok = mlx_reshape(&attn_flat, attn_bt, fsh, 3, st) == 0; }
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
    if (ok) ok = quantized_linear_cfg(&attn_out, attn_flat, ow, os, ob, gsz, bts, m->cfg.quant_mode, st);
    /* gemma4: post_attention_layernorm sull'output dell'attenzione PRIMA del residuo
     * (diverso dalle altre famiglie, che fanno x + attn_out diretto). */
    mlx_array ln2_w = mlx_array_new();
    char ln2b[256];
    tname_gemma4(layer_idx, "post_attention_layernorm.weight", ln2b, sizeof(ln2b));
    if (ok) ok = multishard_get(&m->weights, ln2b, &ln2_w);
    mlx_array attn_normed = mlx_array_new();
    /* FUSIONE residual+RMSNorm (opt-in LLM_FUSE_RESIDUAL_RMS): y = x + w*rmsnorm(attn_out) in UN
     * dispatch al posto di {RMSNorm -> Add}. Fallback silenzioso all'originale. */
    bool rr_fused = false;
    if (ok && (llm_mlx_fused_residual_rms_enabled() || llm_mlx_fused_residual_rms_verify_enabled()) &&
        mlx_array_ndim(x) == 3 && mlx_array_ndim(attn_out) == 3) {
        mlx_array yr = mlx_array_new();
        if (llm_mlx_fused_residual_rms(&yr, x, attn_out, ln2_w, m->cfg.rms_norm_eps, st)) {
            if (llm_mlx_fused_residual_rms_verify_enabled()) {
                mlx_array ref = mlx_array_new();
                bool okr = rms_norm_cfg(&attn_normed, attn_out, ln2_w, m->cfg.rms_norm_eps, st);
                if (okr) okr = mlx_add(&ref, x, attn_normed, st) == 0;
                double dr = okr ? llm_mlx_max_abs_diff(yr, ref, (size_t)mlx_array_size(yr), st) : -1.0;
                if (dr < 0 || dr > 1e-2)
                    fprintf(stderr, "[fuse-residual-rms-verify] layer=%d diff=%.6g  <-- MISMATCH\n", layer_idx, dr);
                mlx_array_free(ref);
            } else {
                ok = mlx_array_set(out_h, yr) == 0;
                rr_fused = true;
            }
        }
        mlx_array_free(yr);
    }
    if (ok && !rr_fused) {
        if (ok) ok = rms_norm_cfg(&attn_normed, attn_out, ln2_w, m->cfg.rms_norm_eps, st);
        if (ok) ok = mlx_add(out_h, x, attn_normed, st) == 0;
    }

    mlx_array all[] = { ln_w, qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob, qn_w, kn_w,
                        normed, q0, k0, v0, q_heads, k_heads, v_heads, q_normed, k_normed, v_normed,
                        q_t, k_t, v_t, custom_freqs, attn, attn_bt, attn_flat, attn_out, ln2_w, attn_normed };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);
    return ok;
}

/* Kimi Linear — MLA assorbita BATCHED (n_sess sessioni, 1 token ciascuna).
 * Proiezioni/norm/o_proj BATCHATE su [1,n_sess,hidden] (N GEMV -> 1 GEMM); append cache,
 * pe_scores e SDPA PER-SESSIONE (k_cache[layer] e aux_cache[layer] di ogni sessione, con
 * lunghezze diverse). Parita' col path per-sessione (fallback di forward_one_token_batch):
 * stessi op/ordini del blocco singolo, solo applicati su un asse sessione. */
static bool kimi_mla_attention_block_batch(LlmMlxModel* m, int layer_idx, LlmMlxSession** sess,
                                           int n_sess, mlx_array x, mlx_array* out_h) {
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const char* mode = m->cfg.quant_mode;
    const int H = m->cfg.n_heads;
    const int qn = m->cfg.qk_nope_head_dim, qr = m->cfg.qk_rope_head_dim;
    const int vhd = m->cfg.v_head_dim, kvl = m->cfg.kv_lora_rank;
    const int qh = qn + qr, N = n_sess;
    mlx_stream st = m->stream;
    if (n_sess < 2 || H <= 0 || qn <= 0 || vhd <= 0 || kvl <= 0) return false;
    if (m->cfg.nsa_sparse || m->cfg.kv_max_tokens > 0 || m->cfg.kv_quantize) return false;
    if (llm_mlx_kvslice_enabled() || llm_mlx_kvseg_enabled()) return false;
    if (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_auth_enabled() || llm_mlx_kvmetal_verify_enabled()) return false;
    if (llm_mlx_model_lora_get(m, layer_idx, "q_proj") >= 0) return false;
    for (int s = 0; s < n_sess; s++)
        if (sess[s] == NULL || !sess[s]->has_cache[layer_idx] || sess[s]->aux_cache == NULL) return false;

    bool ok = true;
    char nb[256];
    mlx_array ln=mlx_array_new(), qw=mlx_array_new(),qs=mlx_array_new(),qb=mlx_array_new();
    mlx_array aw=mlx_array_new(),as=mlx_array_new(),ab=mlx_array_new(),aln=mlx_array_new();
    mlx_array ew=mlx_array_new(),es=mlx_array_new(),eb=mlx_array_new();
    mlx_array uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new();
    mlx_array ow=mlx_array_new(),os=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &ln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_proj_with_mqa", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &aw,&as,&ab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &aln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.embed_q", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ew,&es,&eb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.unembed_out", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &uw,&us,&ub);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os,&ob);

    mlx_array normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&normed, x, ln, m->cfg.rms_norm_eps, st);
    mlx_array q=mlx_array_new(), q4=mlx_array_new(), qt=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&q, normed, qw,qs,qb, gsz,bts, mode, st);
    if (ok) { int sh[4]={1,N,H,qh}; ok = mlx_reshape(&q4, q, sh, 4, st)==0; }
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&qt, q4, perm, 4, st)==0; }

    mlx_array ckv=mlx_array_new(), ckv3=mlx_array_new(), compressed=mlx_array_new();
    mlx_array kpe3=mlx_array_new(), kpe=mlx_array_new(), kvl4=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&ckv, normed, aw,as,ab, gsz,bts, mode, st);
    if (ok) { int sh[3]={1,N,kvl+qr}; ok = mlx_reshape(&ckv3, ckv, sh, 3, st)==0; }
    if (ok) { int _sec[1]={kvl}; mlx_vector_array _sp=mlx_vector_array_new();
              ok = mlx_split_sections(&_sp, ckv3, _sec, 1, -1, st)==0 &&
                   mlx_vector_array_get(&compressed, _sp, 0)==0 && mlx_vector_array_get(&kpe3, _sp, 1)==0;
              mlx_vector_array_free(_sp); }
    if (ok) { int sh[4]={1,1,N,qr}; ok = mlx_reshape(&kpe, kpe3, sh, 4, st)==0; }
    if (ok) { int sh[4]={1,1,N,kvl}; mlx_array kvln=mlx_array_new();
              ok = rms_norm_cfg(&kvln, compressed, aln, m->cfg.rms_norm_eps, st) && mlx_reshape(&kvl4, kvln, sh, 4, st)==0; mlx_array_free(kvln); }

    const float scale = 1.0f / sqrtf((float)qh);
    mlx_vector_array av = mlx_vector_array_new();
    for (int si = 0; si < N && ok; si++) {
        mlx_array* kc = &sess[si]->k_cache[layer_idx];
        mlx_array* ac = &sess[si]->aux_cache[layer_idx];
        const int co = sess[si]->cache_offset;
        /* q_nope/q_pe della sessione si: qt [1,H,N,qh] -> [1,H,1,qh] */
        mlx_array q_s=mlx_array_new(), q_nope=mlx_array_new(), q_pe=mlx_array_new();
        { int s0[4]={0,0,si,0}, sp[4]={1,H,si+1,qh}, sv[4]={1,1,1,1}; ok = mlx_slice(&q_s, qt, s0,4,sp,4,sv,4,st)==0; }
        if (ok) { int _sec[1]={qn}; mlx_vector_array _sp=mlx_vector_array_new();
                  ok = mlx_split_sections(&_sp, q_s, _sec, 1, -1, st)==0 &&
                       mlx_vector_array_get(&q_nope, _sp, 0)==0 && mlx_vector_array_get(&q_pe, _sp, 1)==0;
                  mlx_vector_array_free(_sp); }
        /* kv_latent / k_pe della sessione si */
        mlx_array kvl_s=mlx_array_new(), kpe_s=mlx_array_new();
        if (ok) { int s0[4]={0,0,si,0}, sp[4]={1,1,si+1,kvl}, sv[4]={1,1,1,1}; ok = mlx_slice(&kvl_s, kvl4, s0,4,sp,4,sv,4,st)==0; }
        if (ok) { int s0[4]={0,0,si,0}, sp[4]={1,1,si+1,qr}, sv[4]={1,1,1,1}; ok = mlx_slice(&kpe_s, kpe, s0,4,sp,4,sv,4,st)==0; }
        /* append alle cache per-sessione (k_full/aux_full) */
        mlx_array k_full=mlx_array_new(), aux_full=mlx_array_new();
        if (ok) { mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*kc); mlx_vector_array_append_value(v,kvl_s);
                  ok = mlx_concatenate_axis(&k_full, v, 2, st)==0; mlx_vector_array_free(v); }
        if (ok) {
            if ((*ac).ctx != NULL && mlx_array_ndim(*ac) > 0) {
                mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*ac); mlx_vector_array_append_value(v,kpe_s);
                ok = mlx_concatenate_axis(&aux_full, v, 2, st)==0; mlx_vector_array_free(v);
            } else ok = mlx_array_set(&aux_full, kpe_s)==0;
        }
        if (ok) ok = mlx_array_set(kc, k_full)==0;
        if (ok) ok = mlx_array_set(ac, aux_full)==0;
        /* pe_scores = (q_pe*scale) @ aux^T + mascherina causale additiva */
        mlx_array kpe_ct=mlx_array_new(), qpes=mlx_array_new(), spe=mlx_array_new();
        if (ok) { int nd=(int)mlx_array_ndim(*ac); int perm[8]; for(int i=0;i<nd;i++)perm[i]=i; int t=perm[nd-1];perm[nd-1]=perm[nd-2];perm[nd-2]=t;
                  ok = mlx_transpose_axes(&kpe_ct, *ac, perm, nd, st)==0; }
        mlx_array sf=mlx_array_new_float32(scale), scl=mlx_array_new();
        if (ok) ok = mlx_astype(&scl, sf, mlx_array_dtype(q_pe), st)==0;
        if (ok) ok = mlx_multiply(&qpes, q_pe, scl, st)==0;
        mlx_array_free(sf); mlx_array_free(scl);
        if (ok) ok = mlx_matmul(&spe, qpes, kpe_ct, st)==0;
        mlx_array addmask=mlx_array_new(), spe_m=mlx_array_new();
        if (ok) {
            int S=(int)mlx_array_shape(*kc)[2];
            if (llm_mla_maskcache_enabled()) {
                /* pad-mask cachata: 1 costruzione per token invece di n_layers (IDS identici, zero
                 * op MLX: bit pattern). ~10 op x n_layers rimosse dal grafo per token. */
                mlx_array m = llm_mlx_pad_mask_cached(S, co + 1, mlx_array_dtype(spe), st);
                if (m.ctx == NULL) ok = false;
                else { mlx_array_set(&addmask, m); mlx_array_free(m); }
            } else {
            mlx_array iq=mlx_array_new(), ik=mlx_array_new(), iqe=mlx_array_new(), ike=mlx_array_new(), cmp=mlx_array_new();
            mlx_arange(&iq, 0, 1, 1, MLX_INT32, st);
            mlx_arange(&ik, 0, S, 1, MLX_INT32, st);
            mlx_array coff=mlx_array_new_int(co);
            mlx_array ip=mlx_array_new(); ok = mlx_add(&ip, iq, coff, st)==0; mlx_array_free(coff);
            if (ok) { int sh[2]={1,1}; ok = mlx_reshape(&iqe, ip, sh, 2, st)==0; }
            if (ok) { int sh[2]={1,S}; ok = mlx_reshape(&ike, ik, sh, 2, st)==0; }
            if (ok) ok = mlx_greater_equal(&cmp, iqe, ike, st)==0;
            mlx_array zerof=mlx_array_new_float32(0.0f), ninff=mlx_array_new_float32(-INFINITY);
            mlx_array zero=mlx_array_new(), ninf=mlx_array_new();
            if (ok) ok = mlx_astype(&zero, zerof, mlx_array_dtype(spe), st)==0 && mlx_astype(&ninf, ninff, mlx_array_dtype(spe), st)==0;
            if (ok) ok = mlx_where(&addmask, cmp, zero, ninf, st)==0;
            mlx_array_free(zerof); mlx_array_free(ninff); mlx_array_free(zero); mlx_array_free(ninf);
            if (ok) { int sh[4]={1,1,1,S}; ok = mlx_reshape(&addmask, addmask, sh, 4, st)==0; }
            mlx_array_free(iq);mlx_array_free(ik);mlx_array_free(iqe);mlx_array_free(ike);mlx_array_free(cmp);mlx_array_free(ip);
            }
        }
        if (ok) ok = mlx_add(&spe_m, spe, addmask, st)==0;
        /* SDPA assorbita per-sessione (decode, 1 token) */
        mlx_array k_attn=mlx_array_new(), attn=mlx_array_new(), out_s=mlx_array_new();
        mlx_optional_int gsz_o={gsz,true}, bts_o={bts,true};
        mlx_array no_sinks=mlx_array_new();
        if (ok) ok = mlx_quantized_matmul(&k_attn, q_nope, ew, es, eb, /*transpose=*/true, gsz_o, bts_o, mode, st)==0;
        if (ok) ok = mlx_fast_scaled_dot_product_attention(&attn, k_attn, *kc, *kc, scale, "", spe_m, no_sinks, st)==0;
        if (ok) ok = mlx_quantized_matmul(&out_s, attn, uw, us, ub, /*transpose=*/true, gsz_o, bts_o, mode, st)==0;
        if (ok) ok = mlx_vector_array_append_value(av, out_s)==0;
        mlx_array_free(q_s); mlx_array_free(q_nope); mlx_array_free(q_pe);
        mlx_array_free(kvl_s); mlx_array_free(kpe_s); mlx_array_free(k_full); mlx_array_free(aux_full);
        mlx_array_free(kpe_ct); mlx_array_free(qpes); mlx_array_free(spe); mlx_array_free(addmask); mlx_array_free(spe_m);
        mlx_array_free(k_attn); mlx_array_free(attn); mlx_array_free(out_s); mlx_array_free(no_sinks);
    }

    mlx_array attn_all=mlx_array_new(), attn_bt=mlx_array_new(), attn_flat=mlx_array_new(), proj=mlx_array_new();
    if (ok) ok = mlx_concatenate_axis(&attn_all, av, 2, st)==0;
    mlx_vector_array_free(av);
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn_all, perm, 4, st)==0; }
    if (ok) { int sh[3]={1,N,H*vhd}; ok = mlx_reshape(&attn_flat, attn_bt, sh, 3, st)==0; }
    if (ok) ok = quantized_linear_cfg(&proj, attn_flat, ow,os,ob, gsz,bts, mode, st);
    if (ok) ok = mlx_add(out_h, x, proj, st)==0;

    mlx_array_free(ln);mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);
    mlx_array_free(aw);mlx_array_free(as);mlx_array_free(ab);mlx_array_free(aln);
    mlx_array_free(ew);mlx_array_free(es);mlx_array_free(eb);mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);
    mlx_array_free(ow);mlx_array_free(os);mlx_array_free(ob);mlx_array_free(normed);
    mlx_array_free(q);mlx_array_free(q4);mlx_array_free(qt);
    mlx_array_free(ckv);mlx_array_free(ckv3);mlx_array_free(compressed);mlx_array_free(kpe3);mlx_array_free(kpe);mlx_array_free(kvl4);
    mlx_array_free(attn_all);mlx_array_free(attn_bt);mlx_array_free(attn_flat);mlx_array_free(proj);
    return ok;
}

/* Kimi Linear — KDA (Gated DeltaNet) BATCHED (n_sess sessioni, 1 token ciascuna).
 * Proiezioni (q/k/v/a/b/gate) e norm/o_proj BATCHATE su [1,n_sess,hidden] (N GEMV -> 1 GEMM);
 * conv+ricorrenza PER-SESSIONE con lo STESSO kernel GDN (`llm_mlx_gdn_prefill`) e lo STESSO
 * loop del path singolo -> parita' bit-identica. Stati per-sessione: ssm in k_cache[layer],
 * conv in v_cache[layer]. */
static bool kimi_delta_attention_block_batch(LlmMlxModel* m, int layer_idx, LlmMlxSession** sess,
                                             int n_sess, mlx_array x, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const char* mode = m->cfg.quant_mode;
    char nb[256];
    const int N = n_sess;
    const int H = m->cfg.linear_attn_num_heads > 0 ? m->cfg.linear_attn_num_heads : m->cfg.n_heads;
    const int D = m->cfg.linear_attn_head_dim > 0 ? m->cfg.linear_attn_head_dim : m->cfg.head_dim;
    const int kk = m->cfg.linear_attn_conv_kernel > 0 ? m->cfg.linear_attn_conv_kernel : 4;
    const int P = H * D, CD = 3 * P, csl = kk - 1;
    mlx_stream s = m->stream;
    if (n_sess < 2 || H <= 0 || D <= 0) return false;
    if (m->cfg.nsa_sparse || m->cfg.kv_max_tokens > 0 || m->cfg.kv_quantize) return false;
    if (llm_mlx_kvslice_enabled() || llm_mlx_kvseg_enabled()) return false;
    if (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_auth_enabled() || llm_mlx_kvmetal_verify_enabled()) return false;
    for (int i = 0; i < n_sess; i++)
        if (sess[i] == NULL || !sess[i]->has_cache[layer_idx]) return false;

    mlx_array ln=mlx_array_new(), h=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &ln);
    if (ok) ok = rms_norm_cfg(&h, x, ln, m->cfg.rms_norm_eps, s);

    mlx_array qw=mlx_array_new(),qs=mlx_array_new(),qb=mlx_array_new();
    mlx_array kw=mlx_array_new(),ks=mlx_array_new(),kb=mlx_array_new();
    mlx_array vw=mlx_array_new(),vs=mlx_array_new(),vb=mlx_array_new();
    mlx_array faw=mlx_array_new(),fas=mlx_array_new(),fab=mlx_array_new();
    mlx_array fbw=mlx_array_new(),fbs=mlx_array_new(),fbb=mlx_array_new();
    mlx_array bw=mlx_array_new(),bs_=mlx_array_new(),bb=mlx_array_new();
    mlx_array gaw=mlx_array_new(),gas=mlx_array_new(),gab=mlx_array_new();
    mlx_array gbw=mlx_array_new(),gbs=mlx_array_new(),gbb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &kw,&ks,&kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &vw,&vs,&vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.f_a_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &faw,&fas,&fab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.f_b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &fbw,&fbs,&fbb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &bw,&bs_,&bb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.g_a_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &gaw,&gas,&gab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.g_b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &gbw,&gbs,&gbb);

    /* Proiezioni batched (1 GEMM su N sessioni invece di N GEMV). */
    mlx_array qp=mlx_array_new(),kp=mlx_array_new(),vp=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&qp,h,qw,qs,qb, gsz,bts, mode, s);
    if (ok) ok = quantized_linear_cfg(&kp,h,kw,ks,kb, gsz,bts, mode, s);
    if (ok) ok = quantized_linear_cfg(&vp,h,vw,vs,vb, gsz,bts, mode, s);
    mlx_array qkv=mlx_array_new();
    if (ok) { mlx_array qa[3]={qp,kp,vp}; mlx_vector_array qv=mlx_vector_array_new_data(qa,3); ok = mlx_concatenate_axis(&qkv,qv,-1,s)==0; mlx_vector_array_free(qv); }

    mlx_array cqw=mlx_array_new(),ckw=mlx_array_new(),cvw=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_conv.conv.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &cqw);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_conv.conv.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &ckw);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_conv.conv.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &cvw);
    mlx_array cw32 = mlx_array_new();
    if (ok) ok = kimi_kda_conv_weight(m, layer_idx, s, cqw, ckw, cvw, mlx_array_dtype(qp), &cw32);

    mlx_array fah=mlx_array_new(),araw=mlx_array_new(),braw=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&fah,h,faw,fas,fab, gsz,bts, mode, s);
    if (ok) ok = quantized_linear_cfg(&araw,fah,fbw,fbs,fbb, gsz,bts, mode, s);
    if (ok) ok = quantized_linear_cfg(&braw,h,bw,bs_,bb, gsz,bts, mode, s);
    mlx_array a4=mlx_array_new();
    if (ok) { int sh[4]={1,N,H,D}; ok = mlx_reshape(&a4, araw, sh, 4, s)==0; }
    mlx_array beta=mlx_array_new();
    if (ok) ok = mlx_sigmoid(&beta, braw, s)==0;

    mlx_array Alog=mlx_array_new(), dtb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.A_log", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &Alog);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.dt_bias", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &dtb);
    int alsh[2]={H,1}, dtsh[2]={H,D};
    mlx_array Alog2=mlx_array_new(), dtb2=mlx_array_new();
    if (ok) ok = mlx_reshape(&Alog2, Alog, alsh, 2, s)==0 && mlx_reshape(&dtb2, dtb, dtsh, 2, s)==0;
    mlx_array apd=mlx_array_new(), spexp=mlx_array_new(), sp1=mlx_array_new(), splog=mlx_array_new();
    mlx_array onef=mlx_array_new_float32(1.0f), one=mlx_array_new();
    if (ok) ok = mlx_astype(&one, onef, mlx_array_dtype(a4), s)==0;
    mlx_array_free(onef);
    if (ok) ok = mlx_add(&apd, a4, dtb2, s)==0;
    if (ok) ok = mlx_exp(&spexp, apd, s)==0 && mlx_add(&sp1, spexp, one, s)==0 && mlx_log(&splog, sp1, s)==0;
    mlx_array eAlog=mlx_array_new();
    if (ok) ok = mlx_exp(&eAlog, Alog2, s)==0;
    mlx_array neg=mlx_array_new(), g=mlx_array_new();
    if (ok) ok = mlx_multiply(&neg, eAlog, splog, s)==0;
    if (ok) ok = mlx_negative(&neg, neg, s)==0 && mlx_exp(&g, neg, s)==0;

    mlx_array gah=mlx_array_new(), graw=mlx_array_new(), gate4=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&gah,h,gaw,gas,gab, gsz,bts, mode, s);
    if (ok) ok = quantized_linear_cfg(&graw,gah,gbw,gbs,gbb, gsz,bts, mode, s);
    if (ok) { int sh[4]={1,N,H,D}; ok = mlx_reshape(&gate4, graw, sh, 4, s)==0; }

    const double inv = pow((double)D, -0.5);
    mlx_array inv2f=mlx_array_new_float32((float)(inv*inv)), inv1f=mlx_array_new_float32((float)inv);
    mlx_array inv2=mlx_array_new(), inv1=mlx_array_new();
    if (ok) ok = mlx_astype(&inv2, inv2f, mlx_array_dtype(qp), s)==0 && mlx_astype(&inv1, inv1f, mlx_array_dtype(qp), s)==0;
    mlx_array_free(inv2f); mlx_array_free(inv1f);

    /* conv + ricorrenza PER-SESSIONE (stesso kernel/loop del path singolo). */
    const int ssh[4]={1,H,D,D}, cshape[3]={1,csl,CD};
    mlx_vector_array yv = mlx_vector_array_new();
    for (int si = 0; si < N && ok; si++) {
        mlx_array* kc = &sess[si]->k_cache[layer_idx];
        mlx_array* vc = &sess[si]->v_cache[layer_idx];
        mlx_array qkv_s=mlx_array_new();
        if (ok) { int s0[3]={0,si,0}, sp[3]={1,si+1,CD}, sv[3]={1,1,1}; ok = mlx_slice(&qkv_s, qkv, s0,3,sp,3,sv,3,s)==0; }
        mlx_array cstate=mlx_array_new();
        if (ok) {
            if ((*vc).ctx != NULL && mlx_array_ndim(*vc) > 0) ok = mlx_array_set(&cstate, *vc)==0;
            else ok = mlx_zeros(&cstate, cshape, 3, mlx_array_dtype(qkv), s)==0;
        }
        mlx_array cin=mlx_array_new(), cout=mlx_array_new(), csig=mlx_array_new(), csilu=mlx_array_new();
        if (ok) { mlx_array ca[2]={cstate,qkv_s}; mlx_vector_array cv=mlx_vector_array_new_data(ca,2); ok = mlx_concatenate_axis(&cin,cv,1,s)==0; mlx_vector_array_free(cv); }
        if (ok) {
            int str[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, idil[1]={1};
            ok = mlx_conv_general(&cout, cin, cw32, str,1, pl,1, ph,1, kd,1, idil,1, CD, false, s)==0;
        }
        if (ok) ok = mlx_sigmoid(&csig,cout,s)==0 && mlx_multiply(&csilu,cout,csig,s)==0;
        mlx_array qp2=mlx_array_new(),kp2=mlx_array_new(),vp2=mlx_array_new();
        if (ok) { int secs[2]={P,2*P}; mlx_vector_array spl=mlx_vector_array_new();
                  ok = mlx_split_sections(&spl, csilu, secs, 2, -1, s)==0;
                  if (ok) ok = mlx_vector_array_get(&qp2,spl,0)==0 && mlx_vector_array_get(&kp2,spl,1)==0 && mlx_vector_array_get(&vp2,spl,2)==0;
                  mlx_vector_array_free(spl); }
        mlx_array q4=mlx_array_new(),k4=mlx_array_new(),v4=mlx_array_new();
        if (ok) { int sh[4]={1,1,H,D}; ok = mlx_reshape(&q4,qp2,sh,4,s)==0 && mlx_reshape(&k4,kp2,sh,4,s)==0 && mlx_reshape(&v4,vp2,sh,4,s)==0; }
        mlx_array qn=mlx_array_new(),kn=mlx_array_new(),qs2=mlx_array_new(),ks2=mlx_array_new();
        if (ok) ok = mlx_fast_rms_norm_no_scale(&qn,q4,D,1e-6f,s) && mlx_fast_rms_norm_no_scale(&kn,k4,D,1e-6f,s);
        if (ok) ok = mlx_multiply(&qs2,qn,inv2,s)==0 && mlx_multiply(&ks2,kn,inv1,s)==0;
        mlx_array a4_s=mlx_array_new(), beta_s=mlx_array_new(), g_s=mlx_array_new(), g32=mlx_array_new();
        if (ok) { int s0[4]={0,si,0,0}, sp[4]={1,si+1,H,D}, sv[4]={1,1,1,1}; ok = mlx_slice(&a4_s, a4, s0,4,sp,4,sv,4,s)==0; }
        if (ok) { int s0[3]={0,si,0}, sp[3]={1,si+1,H}, sv[3]={1,1,1}; ok = mlx_slice(&beta_s, beta, s0,3,sp,3,sv,3,s)==0; }
        if (ok) { int s0[4]={0,si,0,0}, sp[4]={1,si+1,H,D}, sv[4]={1,1,1,1}; ok = mlx_slice(&g_s, g, s0,4,sp,4,sv,4,s)==0; }
        if (ok) ok = mlx_astype(&g32, g_s, MLX_FLOAT32, s)==0;
        mlx_array state_s=mlx_array_new();
        if (ok) {
            if ((*kc).ctx != NULL && mlx_array_ndim(*kc) > 0) ok = mlx_array_set(&state_s, *kc)==0;
            else ok = mlx_zeros(&state_s, ssh, 4, MLX_FLOAT32, s)==0;
        }
        mlx_array y_s=mlx_array_new();
        bool kda_kernel = false;
        mlx_array state_nk=mlx_array_new();
        if (ok && !llm_env_isset("LLM_GDN_KERNEL_OFF")) {
            if (llm_mlx_gdn_prefill(&y_s, &state_nk, qs2, ks2, v4, g32, beta_s, state_s, 1, /*g_vector=*/1, s) == 0) {
                mlx_array_set(&state_s, state_nk); kda_kernel = true;
            }
        }
        if (ok && !kda_kernel) {
            int t0[4]={0,0,0,0}, t1[4]={1,1,H,D}, on4[4]={1,1,1,1};
            mlx_array qt=mlx_array_new(),kt=mlx_array_new(),vt=mlx_array_new(),gt=mlx_array_new();
            ok = mlx_slice(&qt,qs2,t0,4,t1,4,on4,4,s)==0 && mlx_slice(&kt,ks2,t0,4,t1,4,on4,4,s)==0 &&
                 mlx_slice(&vt,v4,t0,4,t1,4,on4,4,s)==0 && mlx_slice(&gt,g_s,t0,4,t1,4,on4,4,s)==0;
            int b0[3]={0,0,0}, b1[3]={1,1,H}, on3[3]={1,1,1};
            mlx_array bt=mlx_array_new();
            if (ok) ok = mlx_slice(&bt,beta_s,b0,3,b1,3,on3,3,s)==0;
            mlx_array qsq=mlx_array_new(),ksq=mlx_array_new(),vsq=mlx_array_new(),gsq=mlx_array_new(),bsq=mlx_array_new(),ysq=mlx_array_new();
            if (ok) ok = mlx_squeeze_axis(&qsq,qt,1,s)==0 && mlx_squeeze_axis(&ksq,kt,1,s)==0 && mlx_squeeze_axis(&vsq,vt,1,s)==0 &&
                         mlx_squeeze_axis(&gsq,gt,1,s)==0 && mlx_squeeze_axis(&bsq,bt,1,s)==0;
            if (ok) ok = llm_mlx_gated_delta_step_ext(&ysq,&state_s,qsq,ksq,vsq,gsq,bsq,/*g_vector=*/true,s);
            if (ok) { int sh[4]={1,1,H,D}; ok = mlx_reshape(&y_s, ysq, sh, 4, s)==0; }
            mlx_array_free(qt);mlx_array_free(kt);mlx_array_free(vt);mlx_array_free(gt);mlx_array_free(bt);
            mlx_array_free(qsq);mlx_array_free(ksq);mlx_array_free(vsq);mlx_array_free(gsq);mlx_array_free(bsq);mlx_array_free(ysq);
        }
        mlx_array_free(state_nk);
        mlx_array cnew=mlx_array_new();
        if (ok) { int c0[3]={0,1,0}, c1[3]={1,1+csl,CD}, cst[3]={1,1,1}; ok = mlx_slice(&cnew, cin, c0,3,c1,3,cst,3,s)==0; }
        if (ok) ok = mlx_array_set(kc, state_s) == 0 && mlx_array_set(vc, cnew) == 0;
        if (ok) ok = mlx_vector_array_append_value(yv, y_s)==0;
        mlx_array_free(qkv_s);mlx_array_free(cstate);mlx_array_free(cin);mlx_array_free(cout);mlx_array_free(csig);mlx_array_free(csilu);
        mlx_array_free(qp2);mlx_array_free(kp2);mlx_array_free(vp2);mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v4);
        mlx_array_free(qn);mlx_array_free(kn);mlx_array_free(qs2);mlx_array_free(ks2);
        mlx_array_free(a4_s);mlx_array_free(beta_s);mlx_array_free(g_s);mlx_array_free(g32);
        mlx_array_free(state_s);mlx_array_free(y_s);mlx_array_free(cnew);
    }
    mlx_array y_stack=mlx_array_new();
    if (ok) ok = mlx_concatenate_axis(&y_stack, yv, 1, s)==0;
    mlx_vector_array_free(yv);

    mlx_array onorm=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_norm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &onorm);
    mlx_array yn=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&yn, y_stack, onorm, m->cfg.rms_norm_eps, s);
    mlx_array gsig=mlx_array_new(), gated=mlx_array_new();
    if (ok) ok = mlx_sigmoid(&gsig, gate4, s)==0 && mlx_multiply(&gated, yn, gsig, s)==0;
    int flat[3]={1,N,P};
    mlx_array gf=mlx_array_new();
    if (ok) ok = mlx_reshape(&gf, gated, flat, 3, s)==0;
    mlx_array ow=mlx_array_new(),os_=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os_,&ob);
    mlx_array r=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&r, gf, ow,os_,ob, gsz,bts, mode, s);
    if (ok) ok = mlx_add(out_h, x, r, s)==0;

    mlx_array_free(ln);mlx_array_free(h);
    mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);
    mlx_array_free(kw);mlx_array_free(ks);mlx_array_free(kb);
    mlx_array_free(vw);mlx_array_free(vs);mlx_array_free(vb);
    mlx_array_free(faw);mlx_array_free(fas);mlx_array_free(fab);
    mlx_array_free(fbw);mlx_array_free(fbs);mlx_array_free(fbb);
    mlx_array_free(bw);mlx_array_free(bs_);mlx_array_free(bb);
    mlx_array_free(gaw);mlx_array_free(gas);mlx_array_free(gab);
    mlx_array_free(gbw);mlx_array_free(gbs);mlx_array_free(gbb);
    mlx_array_free(qp);mlx_array_free(kp);mlx_array_free(vp);mlx_array_free(qkv);
    mlx_array_free(cqw);mlx_array_free(ckw);mlx_array_free(cvw);mlx_array_free(cw32);
    mlx_array_free(fah);mlx_array_free(araw);mlx_array_free(braw);mlx_array_free(a4);mlx_array_free(beta);
    mlx_array_free(Alog);mlx_array_free(dtb);mlx_array_free(Alog2);mlx_array_free(dtb2);
    mlx_array_free(apd);mlx_array_free(spexp);mlx_array_free(sp1);mlx_array_free(splog);mlx_array_free(one);
    mlx_array_free(eAlog);mlx_array_free(neg);mlx_array_free(g);
    mlx_array_free(gah);mlx_array_free(graw);mlx_array_free(gate4);
    mlx_array_free(inv2);mlx_array_free(inv1);
    mlx_array_free(onorm);mlx_array_free(yn);mlx_array_free(gsig);mlx_array_free(gated);mlx_array_free(gf);
    mlx_array_free(ow);mlx_array_free(os_);mlx_array_free(ob);mlx_array_free(r);
    mlx_array_free(y_stack);
    return ok;
}

bool forward_one_token_batch(LlmMlxModel* m, LlmMlxSession** sess, int n_sess,
                                    const int32_t* token_ids,
                                    mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                    mlx_array* out_h) {
    if (n_sess <= 0 || sess == NULL || token_ids == NULL) return false;
    mlx_array x = mlx_array_new();
    if (!embed_tokens_batch(m, embed_w, embed_s, embed_b, token_ids, n_sess, &x)) {
        mlx_array_free(x); return false;
    }
    mlx_array h = x;
    for (int layer = 0; layer < m->cfg.n_layers; layer++) {
        bool ok = true;
        mlx_array hb = mlx_array_new();
        const char* mt = m->cfg.model_type;
        /* Famiglie dense GQA coperte dal varlen. gpt-oss incluso: la sua matmul quantizzata bf16
         * differiva di ~1 ulp tra M=1 e M>=2 per via del kernel MLX qmv_wide -- risolto alla
         * radice con patches/mlx_disable_qmv_wide.patch (qmv_wide disabilitato di default). */
        /* NB: "llama" ESCLUSO dal varlen (bug 2026-09-19): llama_attention_block_batch diverge
         * dal path single-session gia' al layer 0 (attenzione, maxdiff misurato 0.09 su
         * Llama-3.2-1B/N=2: 'The finality...' invece di 'The capital of France is Paris.').
         * Con LLM_BATCH_VARLEN=0 (fallback per-sessione) l'output e' corretto. Da investigare
         * a fondo: le altre famiglie restano sul varlen. Vedi bench tests/AI.Tests. */
        bool gqa_ok = strcmp(mt, "qwen2") == 0 ||
                      strcmp(mt, "qwen3") == 0 || strcmp(mt, "qwen3_moe") == 0 ||
                      strcmp(mt, "gpt_oss") == 0 || strcmp(mt, "mixtral") == 0 ||
                      strcmp(mt, "glm4_moe") == 0 || strcmp(mt, "gemma4") == 0;
        const bool batch_attn_on = llm_mlx_batch_varlen_enabled() && !m->cfg.nsa_sparse;
        bool varlen = batch_attn_on && gqa_ok;
        bool attn_batched = false;
        if (varlen) {
            if (strcmp(mt, "gemma4") == 0)
                ok = gemma4_attention_block_batch(m, layer, sess, n_sess, h, &hb);
            else
                ok = llama_attention_block_batch(m, layer, sess, n_sess, h, &hb);
            attn_batched = ok;
        } else if (batch_attn_on && strcmp(mt, "kimi_linear") == 0) {
            /* Kimi Linear: MLA e KDA entrambi batched (proiezioni su N sessioni). */
            const bool is_kda = (m->cfg.is_linear_layer != NULL && layer >= 0 &&
                                 layer < m->cfg.n_layers && m->cfg.is_linear_layer[layer]);
            ok = is_kda ? kimi_delta_attention_block_batch(m, layer, sess, n_sess, h, &hb)
                        : kimi_mla_attention_block_batch(m, layer, sess, n_sess, h, &hb);
            attn_batched = ok;
        }
        if (!attn_batched) {
            /* fallback: attenzione per-sessione (path storico, gia' parity-safe). */
            mlx_array_free(hb); hb = mlx_array_new();
            mlx_vector_array hs = mlx_vector_array_new();
            ok = true;
            for (int s = 0; s < n_sess && ok; s++) {
                mlx_array x_s = mlx_array_new();
                const int st[3] = {0, s, 0};
                const int sp[3] = {1, s + 1, m->cfg.hidden_size};
                const int ss[3] = {1, 1, 1};
                ok = mlx_slice(&x_s, h, st, 3, sp, 3, ss, 3, m->stream) == 0;
                mlx_array h_s = mlx_array_new();
                /* SIDE-CHANNEL PER-SESSIONE (come session_generate/df2_capture_batch): senza,
                 * sui modelli ibridi GDN (qwen3_5) arch->attn legge un aux_cache stantio. */
                mlx_array* saved_aux = m->aux_cache;
                const int32_t* saved_q4e = m->q4e_ctx_ids;
                int saved_q4e_len = m->q4e_ctx_len;
                m->aux_cache = sess[s]->aux_cache;
                m->q4e_ctx_ids = sess[s]->history;
                m->q4e_ctx_len = sess[s]->history_len;
                if (ok) ok = m->arch->attn(m, layer, sess[s]->cache_offset, 1, x_s,
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
        }
        mlx_array layer_out = mlx_array_new();
        if (ok) ok = m->arch->ffn(m, layer, n_sess, hb, &layer_out);
        mlx_array_free(hb);
        mlx_array_free(h);
        if (!ok) { mlx_array_free(layer_out); return false; }
        h = layer_out;
    }
    mlx_array_set(out_h, h);
    mlx_array_free(h);
    return true;
}

/* Batched prefill: analogo di embed_token per n_tokens id insieme -- un solo
 * gather (mlx_take_axis con un array di n_tokens indici, stesso primitivo,
 * stessa semantica gia' usata altrove in questo file per indici multipli)
 * invece di n_tokens gather separati. */
bool embed_tokens_batch(LlmMlxModel* m, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                const int32_t* token_ids, int n_tokens, mlx_array* out_x) {
    const bool quantized = (embed_s.ctx != NULL); /* vedi embed_token sopra (FASE 1b.3, Mixtral) */
    /* §quant per-tensore (GPT-OSS): deriva group/bits/mode dalle shape COMPLETE di
     * embed_w/embed_s (il row gatherato e' 1D e la derive fallirebbe al fallback).
     * GPT-OSS embed = affine 64/8 (ratio 16). */
    int e_group = m->cfg.quant_group_size, e_bits = m->cfg.quant_bits;
    char e_mode[16];
    llm_mlx_mode_set(e_mode, sizeof(e_mode), m->cfg.quant_mode);
    if (quantized)
        llm_mlx_quant_derive(embed_w, embed_s, e_group, e_bits, e_mode, &e_group, &e_bits, e_mode, sizeof(e_mode));
    const int idx_shape[1] = {n_tokens};
    mlx_array indices = mlx_array_new_data(token_ids, idx_shape, 1, MLX_INT32);
    mlx_array w_row = mlx_array_new(), s_row = mlx_array_new(), b_row = mlx_array_new();
    bool g = mlx_take_axis(&w_row, embed_w, indices, 0, m->stream) == 0;
    if (g && quantized) {
        g = mlx_take_axis(&s_row, embed_s, indices, 0, m->stream) == 0;
        if (g && embed_b.ctx != NULL && mlx_array_size(embed_b) > 0)
            g = mlx_take_axis(&b_row, embed_b, indices, 0, m->stream) == 0;
    }
    mlx_array x_flat = mlx_array_new();
    bool ok2 = g;
    if (ok2) {
        if (quantized) {
            mlx_optional_int gsz = {e_group, true};
            mlx_optional_int bts = {e_bits, true};
            mlx_optional_dtype dt = {llm_mlx_act_dt(m->cfg.bfloat16_input), true};
            ok2 = mlx_dequantize(&x_flat, w_row, s_row, b_row, gsz, bts, e_mode, g_mlx_empty_arr, dt, m->stream) == 0;
        } else {
            /* Bug reale trovato dal vivo (Mixtral): embed_tokens.weight non quantizzato
             * è salvato in F16 nel safetensors reale -- senza questo cast esplicito a
             * float32, ogni lettura host successiva (mlx_array_data_float32) reinterpreta
             * i bit float16 come float32 (stride sbagliato), producendo valori quasi-zero
             * plausibili ma completamente sbagliati (osservato dal vivo: tutti gli 8
             * valori letti ~0.000000 invece dei veri pesi di embedding). */
            ok2 = mlx_astype(&x_flat, w_row, llm_mlx_act_dt(m->cfg.bfloat16_input), m->stream) == 0;
        }
    }
    if (ok2) {
        const int x_shape[3] = {1, n_tokens, m->cfg.hidden_size};
        ok2 = mlx_reshape(out_x, x_flat, x_shape, 3, m->stream) == 0;
    }
    /* Ternary-Bonsai: stesso hook di embed_token (era ASSENTE nel path batch -> prefill
     * non de-ruotato = garbage). L'embed e' in inverse_weight_names -> invert=true. */
    if (ok2) {
        const int emb_inv_b = (llm_ai_mlx_opt_env("LLM_BONSAI_EMBED_INV") != NULL) ? (atoi(llm_ai_mlx_opt_env("LLM_BONSAI_EMBED_INV")) != 0 ? 1 : 0) : 1;
        ok2 = llm_bonsai_rotate(out_x, m, -1, "embed_tokens.weight", emb_inv_b == 1);
    }
    /* FASE 1b.4.5 (Gemma4): embed_scale = sqrt(hidden_size), sempre (gemma4_text.py:402).
     * NOTA lifetime: mlx_array_set() ASSEGNA sull'array esistente (*out_x = scaled, refcount
     * C++ condiviso) -- NON chiamare mlx_array_free(*out_x) prima: mlx_array_free_ cancella
     * il ctx ma NON azzera l'handle, quindi un set successivo scriverebbe in memoria liberata
     * (use-after-free osservato). scaled va liberato DOPO il set (il set copia il valore). */
    if (ok2 && m->cfg.embed_scale > 0.0f) {
        mlx_array scaled = mlx_array_new();
        mlx_array scl = mlx_array_new_float32(m->cfg.embed_scale);
        ok2 = mlx_multiply(&scaled, *out_x, scl, m->stream) == 0;
        mlx_array_free(scl);
        if (ok2) mlx_array_set(out_x, scaled);
        mlx_array_free(scaled);
    }
    /* Qwen4-Exp: vedi embed_token sopra -- stesso tiling, versione batch. */
    if (ok2 && strcmp(m->cfg.model_type, "qwen4_exp") == 0 && m->cfg.hc_count > 1) {
        mlx_array tiled = mlx_array_new();
        const int reps[3] = {1, 1, m->cfg.hc_count};
        ok2 = mlx_tile(&tiled, *out_x, reps, 3, m->stream) == 0;
        if (ok2) mlx_array_set(out_x, tiled);
        mlx_array_free(tiled);
    }
    mlx_array_free(indices);
    mlx_array_free(w_row);
    mlx_array_free(s_row);
    mlx_array_free(b_row);
    mlx_array_free(x_flat);
    return ok2;
}

/* GPT-2/GPT-BigCode: somma le position embeddings APPRESE (m->pos_emb_w [n_pos, hidden]) agli
 * embed. offset = posizione assoluta del primo token; n = n_tokens (righe offset..offset+n-1).
 * No-op (copia) se m->pos_emb_w assente (tutte le arch RoPE). */
bool llm_mlx_pos_emb_add(LlmMlxModel* m, mlx_array x, int offset, int n, mlx_array* out) {
    if (m->pos_emb_w.ctx == NULL) return mlx_array_set(out, x) == 0;
    int st[2] = {offset, 0}, sp[2] = {offset + n, m->cfg.hidden_size}, sv[2] = {1, 1};
    mlx_array rows = mlx_array_new();
    bool ok = mlx_slice(&rows, m->pos_emb_w, st, 2, sp, 2, sv, 2, m->stream) == 0;
    mlx_array re = mlx_array_new();
    if (ok) { int sh[3] = {1, n, m->cfg.hidden_size}; ok = mlx_reshape(&re, rows, sh, 3, m->stream) == 0; }
    mlx_array sum = mlx_array_new();
    if (ok) ok = mlx_add(&sum, x, re, m->stream) == 0;
    if (ok) mlx_array_set(out, sum);
    mlx_array_free(rows); mlx_array_free(re); mlx_array_free(sum);
    return ok;
}

/* Batched prefill: analogo di forward_one_token per n_tokens id insieme --
 * imbeve tutti gli id in un colpo solo, poi li fa passare attraverso tutti i
 * layer con UNA chiamata per layer (layer_forward con n_tokens=L) invece di
 * n_tokens chiamate separate. *out_h e' l'uscita COMPLETA [1,n_tokens,hidden]
 * dell'ultimo layer -- il chiamante estrae la riga che gli serve
 * (tipicamente l'ultima, per pending_h/final_step in
 * llm_ai_mlx_session_append). */
/* oMLX 736c90f (guasto 3): eval per-layer durante il PREFILL. Opt-in. */
static bool llm_mlx_prefill_eval_layer_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_PREFILL_EVAL_LAYER") != NULL) ? 1 : 0;
    return st == 1;
}
bool forward_tokens_batch(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                  const int32_t* token_ids, int n_tokens, int cache_offset,
                                  mlx_array* k_cache, mlx_array* v_cache, bool* has_cache, mlx_array* out_h) {
    /* RMLX10 Fase A: se un decode precedente ha lasciato buffer statici PADDED
     * (LLM_FIXKV=1), un forward BATCH (prefill/append/chunked) deve prima
     * ricompattare la cache a cache_offset token validi: il concat eager
     * appende in coda e il padding romperebbe causalita'/parita'. No-op se la
     * cache non e' mai passata dal path fixkv (shape2 == cache_offset). */
    llm_mlx_kv_compact_prefix(m->cfg.n_layers, cache_offset, m->stream, k_cache, v_cache);
    mlx_array x = mlx_array_new();
    if (!embed_tokens_batch(m, embed_w, embed_s, embed_b, token_ids, n_tokens, &x)) {
        mlx_array_free(x);
        return false;
    }
    /* GPT-2/GPT-BigCode: somma le position embeddings apprese (no-op per le altre arch). */
    mlx_array h = mlx_array_new();
    if (!llm_mlx_pos_emb_add(m, x, cache_offset, n_tokens, &h)) { mlx_array_free(x); mlx_array_free(h); return false; }
    mlx_array_free(x);
    bool step_ok = true;
    for (int layer = 0; step_ok && layer < m->cfg.n_layers; layer++) {
        mlx_array layer_out = mlx_array_new();
        step_ok = layer_forward(m, kvq_sess, layer, cache_offset, n_tokens, h, &k_cache[layer], &v_cache[layer], has_cache[layer], &layer_out);
        has_cache[layer] = true;
        mlx_array_free(h);
        h = layer_out;
        /* HOOK DFlash2 (2026-10-01): cattura l'hidden dei layer-selezionati DURANTE il prefill
         * (evita il re-forward del ctx nello streaming). Attivo solo se armato (g_df2_pf_active). */
        if (step_ok && g_df2_pf_active && g_df2_pf_hook) g_df2_pf_hook(layer, h, n_tokens, cache_offset);
        if (step_ok && g_layerh_run != NULL && getenv("LLM_GDN_AB"))
            fprintf(stderr, "[layerh] run=%s L=%d sum=%.6g\n", g_layerh_run, layer, llm_mlx_checksum_f32(h, m->stream));
        if (step_ok && layer <= 1 && getenv("LLM_BONSAI_DBG")) {
            mlx_array vf = mlx_array_new();
            if (mlx_astype(&vf, h, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(vf) == 0) {
                const float* d = mlx_array_data_float32(vf);
                if (d) { fprintf(stderr, "[layer%d] out[:6] =", layer); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
            }
            mlx_array_free(vf);
        }

        /* Senza eval, l'INTERO chunk tiene vivi gli intermedi di TUTTI i layer (la CPU
         * accoda in ~1s, la GPU gira ~10s) -> il picco cresce col chunk. Materializzando
         * il layer-carried dopo OGNI layer il live set scende a ~un layer. Solo prefill
         * (L>=256): il decode a 1 token resta lazy. Output invariato (cambia solo la
         * schedulazione, non i valori). Opt-in: misurare prima di proporre il default. */
        if (step_ok && n_tokens >= 256 && llm_mlx_prefill_eval_layer_enabled()) {
            if (mlx_array_eval(h) != 0) step_ok = false;
        }
    }
    if (step_ok) mlx_array_set(out_h, h);
    mlx_array_free(h);
    return step_ok;
}
