/* ============================================================================
 * llm_mlx_qwen35.c — dominio Qwen3.5/Next: qwen3_next_linear_block (GDN), qwen3_5_attention_block (ibrido), qwen3_next_attention_dispatch, qwen4_full_qsa_block (QSA Flash-Next).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_qwen4exp.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_gdn_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

bool qwen3_next_linear_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    /* §13.5: k_cache[layer] = stato ricorrente GDN S [B,Hv,Dv,Dk]; v_cache[layer] =
     * conv_state [B,kk-1,conv_dim]; has_cache[layer] = true se lo stato esiste gia'
     * (decode che continua il prefill). Riuso i due slot perche' i layer lineari GDN non
     * usano la KV-cache dell'attention (solo i layer full la usano). */
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
    const int B = 1;
    const int Hk = m->cfg.linear_num_key_heads;
    const int Hv = m->cfg.linear_num_value_heads;
    const int Dk = m->cfg.linear_key_head_dim;
    const int Dv = m->cfg.linear_value_head_dim;
    const int key_dim = Hk * Dk;
    const int value_dim = Hv * Dv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int S = n_tokens;
    mlx_stream s = m->stream;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen3_next_linear[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, ln_w, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }

    /* Qwen3-Next: proiezioni MERGED. in_proj_qkvz -> [B,S, 2*key_dim+2*value_dim]; in_proj_ba -> [B,S,Hv*2].
     * Ordine per-k-head (reference mlx_lm qwen3_next fix_query_key_value_ordering): reshape a Hk teste,
     * split ultimo asse [Dk, 2*Dk, 2*Dk+(Hv/Hk)*Dv] => q,k,v,z per head; ba reshape a Hk, split [Hv/Hk]
     * => b,a. Riassembla: qkv (conv-input) = concat[q_flat,k_flat,v_flat] (flat head-major), z_flat
     * = value_dim, bv=b, av=a [B,S,Hv]. Il resto (conv depthwise, silu, split post-conv, rms q/k,
     * recurrence GDN, norm-gate silu(z)*rms(out), out_proj) e' IDENTICO al blocco qwen3_5. */
    const int R = Hv / Hk;
    const int qz_per = 2 * Dk + 2 * R * Dv;
    mlx_array qz_w=mlx_array_new(), qz_sc=mlx_array_new(), qz_b=mlx_array_new();
    mlx_array ba_w=mlx_array_new(), ba_sc=mlx_array_new(), ba_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_qkvz", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &qz_w, &qz_sc, &qz_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_ba", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ba_w, &ba_sc, &ba_b);
    CHK(ok, "merged in_proj lookups");
    mlx_array qkvz=mlx_array_new(), ba=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkvz, h, qz_w, qz_sc, qz_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_qkvz"); }
    if (ok) { ok = quantized_linear_cfg(&ba, h, ba_w, ba_sc, ba_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_ba"); }
    /* split qkvz in q,k,v,z per-k-head (ultimo asse, qz_per per gruppo) */
    mlx_array q4h=mlx_array_new(), k4h=mlx_array_new(), v4h=mlx_array_new(), z4h=mlx_array_new();
    if (ok) {
        int qz_sh[4] = { B, S, Hk, qz_per };
        mlx_array qz4 = mlx_array_new();
        ok = mlx_reshape(&qz4, qkvz, qz_sh, 4, s) == 0;
        if (ok) {
            int secs3[3] = { Dk, 2 * Dk, 2 * Dk + R * Dv };
            mlx_vector_array vv = mlx_vector_array_new();
            ok = mlx_split_sections(&vv, qz4, secs3, 3, /*axis=*/-1, s) == 0;
            if (ok) ok = mlx_vector_array_get(&q4h, vv, 0) == 0 && mlx_vector_array_get(&k4h, vv, 1) == 0 &&
                            mlx_vector_array_get(&v4h, vv, 2) == 0 && mlx_vector_array_get(&z4h, vv, 3) == 0;
            mlx_vector_array_free(vv);
        }
        mlx_array_free(qz4);
        CHK(ok, "split qkvz");
    }
    /* split ba in b,a per-k-head (ultimo asse, 2R per gruppo) */
    mlx_array b4h=mlx_array_new(), a4h=mlx_array_new();
    if (ok) {
        int ba_sh[4] = { B, S, Hk, 2 * R };
        mlx_array ba4 = mlx_array_new();
        ok = mlx_reshape(&ba4, ba, ba_sh, 4, s) == 0;
        if (ok) {
            int secs1[1] = { R };
            mlx_vector_array vv = mlx_vector_array_new();
            ok = mlx_split_sections(&vv, ba4, secs1, 1, /*axis=*/-1, s) == 0;
            if (ok) ok = mlx_vector_array_get(&b4h, vv, 0) == 0 && mlx_vector_array_get(&a4h, vv, 1) == 0;
            mlx_vector_array_free(vv);
        }
        mlx_array_free(ba4);
        CHK(ok, "split ba");
    }
    /* riassemblaggio flat head-major: qkv conv-input, z flat value_dim, bv/av [B,S,Hv] */
    mlx_array q_flat=mlx_array_new(), k_flat=mlx_array_new(), v_flat=mlx_array_new();
    mlx_array z=mlx_array_new(), av=mlx_array_new(), bv=mlx_array_new();
    mlx_array qkv=mlx_array_new();
    if (ok) {
        int kf[3] = { B, S, key_dim }, vf[3] = { B, S, value_dim }, hf[3] = { B, S, Hv };
        ok = mlx_reshape(&q_flat, q4h, kf, 3, s) == 0 && mlx_reshape(&k_flat, k4h, kf, 3, s) == 0 &&
             mlx_reshape(&v_flat, v4h, vf, 3, s) == 0 && mlx_reshape(&z, z4h, vf, 3, s) == 0 &&
             mlx_reshape(&bv, b4h, hf, 3, s) == 0 && mlx_reshape(&av, a4h, hf, 3, s) == 0;
        if (ok) {
            mlx_array parts[3] = { q_flat, k_flat, v_flat };
            mlx_vector_array cv = mlx_vector_array_new_data(parts, 3);
            ok = mlx_concatenate_axis(&qkv, cv, /*axis=*/-1, s) == 0;
            mlx_vector_array_free(cv);
        }
        CHK(ok, "qkv reassembly");
    }
    mlx_array_free(q4h); mlx_array_free(k4h); mlx_array_free(v4h); mlx_array_free(z4h);
    mlx_array_free(b4h); mlx_array_free(a4h);
    mlx_array_free(q_flat); mlx_array_free(k_flat); mlx_array_free(v_flat);

    /* conv1d depthwise su concat([conv_state, qkv]) */
    mlx_array conv_w = mlx_array_new();
    mlx_array* cw32p = llm_mlx_gdn_conv_cached(m, layer_idx);
    if (cw32p != NULL) {
        ok = ok && mlx_array_set(&conv_w, *cw32p) == 0;
        CHK(ok, "conv1d cast cache");
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.conv1d.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &conv_w);
        CHK(ok, "conv1d lookup");
        /* conv1d.weight e' BF16 (verificato sull'header safetensors); mlx_conv_general richiede
         * input e weight con lo stesso dtype. qkv e' float32 -> cast esplicito del weight. */
        if (ok) {
            mlx_array cw32 = mlx_array_new();
            ok = mlx_astype(&cw32, conv_w, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
            if (ok) { mlx_array_free(conv_w); conv_w = cw32; }
            else mlx_array_free(cw32);
            CHK(ok, "conv1d weight cast");
        }
    }
    int kk = m->cfg.linear_conv_kernel_dim;
    int cstate_len = kk - 1;
    int cshape[3] = { B, cstate_len, conv_dim };
    mlx_array conv_state = mlx_array_new();
    /* §13.5: conv_state persistente (v_cache[layer]). Se has_cache e v_cache ha un valore
     * (decode che continua), riuso; altrimenti zero (primo prefill del layer). */
    if (ok) {
        if (has_cache && v_cache && v_cache->ctx != NULL && mlx_array_ndim(*v_cache) > 0) {
            ok = mlx_array_set(&conv_state, *v_cache) == 0;
        } else {
            ok = mlx_zeros(&conv_state, cshape, 3, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
        }
        CHK(ok, "conv_state init");
    }
    mlx_array conv_input = mlx_array_new();
    /* concat [conv_state, qkv] lungo l'asse 1 (sequenza), come il riferimento
     * mx.concatenate([...], axis=1). mlx_concatenate senza axis unirebbe sull'ultimo
     * asse (conv_dim) -> forma sbagliata. */
    if (ok) { mlx_array cv_arr[2] = { conv_state, qkv }; mlx_vector_array cv = mlx_vector_array_new_data(cv_arr, 2); ok = mlx_concatenate_axis(&conv_input, cv, /*axis=*/1, s) == 0; CHK(ok, "concat conv_state+qkv"); mlx_vector_array_free(cv); }
    /* conv1d depthwise (groups=conv_dim) via mlx_conv_general: mlx_conv1d non supporta
     * groups != 1 (errore reale osservato: "Can only handle groups != 1 in 1D or 2D").
     * Il conv1d GDN e' depthwise (groups=conv_dim=10240, kernel kk), stride 1, padding 0.
     * Weight [conv_dim, kk, 1]; input [B, S+kk-1, conv_dim] -> output [B, S, conv_dim]. */
    mlx_array conv_out = mlx_array_new();
    if (ok) {
        int c_stride[1] = {1};
        int c_pad_lo[1] = {0};
        int c_pad_hi[1] = {0};
        int c_kdil[1] = {1};
        int c_idil[1] = {1};
        ok = mlx_conv_general(&conv_out, conv_input, conv_w, c_stride, 1, c_pad_lo, 1, c_pad_hi, 1,
                              c_kdil, 1, c_idil, 1, /*groups=*/conv_dim, /*flip=*/false, s) == 0;
        CHK(ok, "conv1d (depthwise)");
    }
    /* silu(conv_out) */
    mlx_array sig = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&sig, conv_out, s) == 0; }
    mlx_array silu = mlx_array_new();
    if (ok) { ok = mlx_multiply(&silu, conv_out, sig, s) == 0; CHK(ok, "silu"); }

    /* split in q,k,v per dim (mlx_split_sections: [0,key), [key,2key), [2key,conv)) lungo
     * l'ULTIMO asse (conv_dim), come il riferimento verificato (mx.split(..., -1)). */
    int secs[2] = { key_dim, 2 * key_dim };
    mlx_vector_array kv_split = mlx_vector_array_new();
    mlx_array q_part=mlx_array_new(), k_part=mlx_array_new(), v_part=mlx_array_new();
    if (ok) { ok = mlx_split_sections(&kv_split, silu, secs, 2, /*axis=*/-1, s) == 0; CHK(ok, "split q/k/v"); }
    if (ok) { ok = mlx_vector_array_get(&q_part, kv_split, 0) == 0 &&
                    mlx_vector_array_get(&k_part, kv_split, 1) == 0 &&
                    mlx_vector_array_get(&v_part, kv_split, 2) == 0; CHK(ok, "get q/k/v"); }

    /* reshape a teste: q,k [B,S,Hk,Dk], v [B,S,Hv,Dv] */
    int q_shape[4] = { B, S, Hk, Dk };
    int v_shape[4] = { B, S, Hv, Dv };
    mlx_array q4=mlx_array_new(), k4=mlx_array_new(), v4=mlx_array_new();
    if (ok) { ok = mlx_reshape(&q4, q_part, q_shape, 4, s) == 0 &&
                    mlx_reshape(&k4, k_part, q_shape, 4, s) == 0 &&
                    mlx_reshape(&v4, v_part, v_shape, 4, s) == 0; CHK(ok, "reshape q/k/v"); }
    

    /* inv_scale = Dk**-0.5; q = inv^2 * rms_norm(q); k = inv * rms_norm(k). eps=1e-6
     * come il riferimento (mx.fast.rms_norm(..., None, 1e-6)). */
    const double inv_scale = pow((double)Dk, -0.5);
    mlx_array inv2 = llm_mlx_act_scalar((float)(inv_scale * inv_scale));
    mlx_array inv1 = llm_mlx_act_scalar((float)inv_scale);
    mlx_array q_n=mlx_array_new(), k_n=mlx_array_new();
    if (ok) { ok = mlx_fast_rms_norm_no_scale(&q_n, q4, Dk, 1e-6f, s) &&
                    mlx_fast_rms_norm_no_scale(&k_n, k4, Dk, 1e-6f, s); CHK(ok, "rms_norm q/k"); }
    mlx_array q_s=mlx_array_new(), k_s=mlx_array_new();
    if (ok) { ok = mlx_multiply(&q_s, q_n, inv2, s) == 0 &&
                    mlx_multiply(&k_s, k_n, inv1, s) == 0; CHK(ok, "inv_scale q/k"); }
    

    /* beta = sigmoid(b); g = compute_g(A_log, a, dt_bias) = exp(-exp(A_log)*softplus(a+dt_bias)) */
    mlx_array A_log=mlx_array_new(), dt_bias=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.A_log", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &A_log);
    mlx_array* _dt = llm_mlx_gdn_dt_bias(m, layer_idx);
    if (_dt != NULL) { ok = ok && mlx_array_set(&dt_bias, *_dt) == 0; }
    else {
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.dt_bias", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &dt_bias);
    }
    CHK(ok, "A_log/dt_bias lookup");
    mlx_array beta=mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&beta, bv, s) == 0; CHK(ok, "beta sigmoid"); }
    /* softplus(a+dt_bias) = log(1+exp(a+dt_bias)); a: [B,S,Hv], dt_bias: [Hv] -> broadcast */
    mlx_array a_plus_dt=mlx_array_new();
    if (ok) { ok = mlx_add(&a_plus_dt, av, dt_bias, s) == 0; CHK(ok, "a+dt_bias"); }
    mlx_array sp_exp=mlx_array_new();
    if (ok) { ok = mlx_exp(&sp_exp, a_plus_dt, s) == 0; CHK(ok, "softplus exp"); }
    mlx_array one_arr=llm_mlx_act_scalar(1.0f);
    mlx_array sp_1p=mlx_array_new();
    if (ok) { ok = mlx_add(&sp_1p, sp_exp, one_arr, s) == 0; CHK(ok, "softplus 1+"); }
    mlx_array sp_log=mlx_array_new();
    if (ok) { ok = mlx_log(&sp_log, sp_1p, s) == 0; CHK(ok, "softplus log"); }
    mlx_array eAlog=mlx_array_new();
        mlx_array* _ae = llm_mlx_gdn_alog_exp(m, layer_idx);
        if (_ae != NULL) { ok = ok && mlx_array_set(&eAlog, *_ae) == 0; CHK(ok, "exp A_log cache"); }
        else if (ok) { ok = mlx_exp(&eAlog, A_log, s) == 0; CHK(ok, "exp A_log"); }
    mlx_array neg_exp=mlx_array_new();
    if (ok) { ok = mlx_multiply(&neg_exp, eAlog, sp_log, s) == 0; CHK(ok, "exp(A_log)*softplus"); }
    mlx_array neg_g=mlx_array_new();
    if (ok) { ok = mlx_negative(&neg_g, neg_exp, s) == 0; CHK(ok, "negate exp(A_log)*softplus"); }
    mlx_array g=mlx_array_new();
    if (ok) { ok = mlx_exp(&g, neg_g, s) == 0; CHK(ok, "g = exp(-...)"); }

    /* repeat q,k da Hk a Hv teste (Hv//Hk = 3) lungo l'asse teste (-2), come
     * gated_delta_ops (mx.repeat(x, repeat_factor, -2)). */
    const int repeat_factor = Hv / Hk;
    mlx_array q_rep=mlx_array_new(), k_rep=mlx_array_new();
    if (ok) {
        if (repeat_factor > 1) {
            ok = mlx_repeat_axis(&q_rep, q_s, repeat_factor, /*axis=*/2, s) == 0 &&
                 mlx_repeat_axis(&k_rep, k_s, repeat_factor, /*axis=*/2, s) == 0;
        } else {
            ok = mlx_array_set(&q_rep, q_s) == 0 && mlx_array_set(&k_rep, k_s) == 0;
        }
        CHK(ok, "repeat q/k to Hv");
    }

    /* Stato ricorrente S [B,Hv,Dv,Dk] (k_cache[layer]). §13.5: se has_cache e k_cache ha
     * un valore (decode che continua il prefill), riuso lo stato finale del passo
     * precedente; altrimenti zero (primo prefill del layer). Il loop lo evolve poi di S
     * token (prefill: l'intero batch; decode: 1 step). */
    int state_shape[4] = { B, Hv, Dv, Dk };
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache) > 0) {
            ok = mlx_array_set(&state, *k_cache) == 0;
        } else {
            ok = mlx_zeros(&state, state_shape, 4, MLX_FLOAT32, s) == 0;
        }
        CHK(ok, "state init");
    }

    /* §2026-09-01 KERNEL METAL GDN FUSO: per il PREFILL (S>1) la ricorrenza viene calcolata in
     * un solo kernel Metal (llm_mlx_gdn_prefill, modulo llm_mlx_gdn_kernel.h) invece del loop
     * per-token (~32 chiamate MLX per step, il collo del prefill). Stessa ricorrenza esatta.
     * Se il kernel non e' disponibile (dimensioni/Dk) o fallisce, fallback al loop qui sotto. */
    mlx_array y_stack = mlx_array_new();
    bool kernel_done = false;
    if (ok && S >= 1 && !llm_env_isset("LLM_GDN_KERNEL_OFF")) {
      if (g_spec_capture && S > 1) {
        /* S5(b): UN SOLO call del kernel FUSO con l'output `state_seq` (stato dopo OGNI
         * token) -> rollback O(1) senza l'overhead di launch per-token. */
        mlx_array st_n = mlx_array_new(), sseq_arr = mlx_array_new();
        if (llm_mlx_gdn_prefill_sseq(&y_stack, &st_n, &sseq_arr, q_s, k_s, v4, g, beta, state, S, 0, s) == 0) {
            (void)mlx_array_set(&state, st_n);
            if (layer_idx < LLM_SPEC_MAXL) {
                (void)mlx_array_set(&g_spec_snap_k[layer_idx], sseq_arr);   /* [B,S,Hv,Dv,Dk] */
                (void)mlx_array_set(&g_spec_snap_v[layer_idx], conv_input); /* [B,cstate_len+S,conv_dim] */
            }
            kernel_done = true;
        }
        mlx_array_free(st_n); mlx_array_free(sseq_arr);
        CHK(ok, "gdn kernel sseq (capture)");
      } else {
        mlx_array state_new_k = mlx_array_new();
        if (llm_mlx_gdn_prefill(&y_stack, &state_new_k, q_s, k_s, v4, g, beta, state, S, /*g_vector=*/0, s) == 0) {
            mlx_array_set(&state, state_new_k);
            kernel_done = true;
        }
        mlx_array_free(state_new_k);
      }
    }
    
    if (llm_mlx_gdn_kernel_dbg_enabled())
        fprintf(stderr, "[gdn-kernel] layer=%d S=%d -> %s\n", layer_idx, S,
                kernel_done ? "KERNEL FUSO" : "LOOP per-token");

    /* Loop ricorrente su S token (gated_delta_update, reference mlx-lm). Per ogni t:
     *   decay = g[...,None,None]; state *= decay
     *   kv_mem = (state * k[...,None,:]).sum(-1)
     *   delta = (v - kv_mem) * beta[...,None]
     *   state += k[...,None,:] * delta[...,None]
     *   y_t   = (state * q[...,None,:]).sum(-1)
     * Raccoglie y_t in un vettore e li stack lungo l'asse 1 -> [B,S,Hv,Dv]. */
    mlx_vector_array ys = mlx_vector_array_new();
    if (!kernel_done) for (int t = 0; ok && t < S; t++) {
        int st0[4] = {0, t, 0, 0};
        int st1[4] = {1, t + 1, Hv, Dk};
        int stv[4] = {1, t + 1, Hv, Dv};
        int ones4[4] = {1, 1, 1, 1};
        int ones3[3] = {1, 1, 1};
        mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
        mlx_array g_t=mlx_array_new(), beta_t=mlx_array_new();
        if (ok) {
            ok = mlx_slice(&q_t, q_rep, st0, 4, st1, 4, ones4, 4, s) == 0 &&
                 mlx_slice(&k_t, k_rep, st0, 4, st1, 4, ones4, 4, s) == 0 &&
                 mlx_slice(&v_t, v4, st0, 4, stv, 4, ones4, 4, s) == 0;
            int gt0[3] = {0, t, 0}, gt1[3] = {1, t + 1, Hv};
            ok = ok && mlx_slice(&g_t, g, gt0, 3, gt1, 3, ones3, 3, s) == 0 &&
                       mlx_slice(&beta_t, beta, gt0, 3, gt1, 3, ones3, 3, s) == 0;
        }
        /* squeeze assi 1 (singleton) */
        mlx_array q_sq=mlx_array_new(), k_sq=mlx_array_new(), v_sq=mlx_array_new();
        mlx_array g_sq=mlx_array_new(), beta_sq=mlx_array_new();
        if (ok) {
            ok = mlx_squeeze_axis(&q_sq, q_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&k_sq, k_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&v_sq, v_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&g_sq, g_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&beta_sq, beta_t, 1, s) == 0;
        }
        /* §refactor: il passo ricorrente GDN ora delega al modulo componibile
         * llm_mlx_gated_delta_step (stessa matematica, unica sorgente). */
        mlx_array y_t = mlx_array_new();
        if (ok) ok = llm_mlx_gated_delta_step(&y_t, &state, q_sq, k_sq, v_sq, g_sq, beta_sq, s);
        CHK(ok, "gated_delta_step");
        /* y_t: [B,Hv,Dv] viene stackato DIRETTAMENTE lungo l'asse 1 (stack di array
         * 3D [B,Hv,Dv] su axis=1 -> [B,S,Hv,Dv]). NON espandere prima: espandere a
         * 4D [B,1,Hv,Dv] e poi stack su axis=1 produce 5D [B,S,1,Hv,Dv] (bug reale
         * osservato). */
        if (ok) { ok = mlx_vector_array_append_value(ys, y_t) == 0; CHK(ok, "collect y"); }
        mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
        mlx_array_free(g_t); mlx_array_free(beta_t);
        mlx_array_free(q_sq); mlx_array_free(k_sq); mlx_array_free(v_sq);
        mlx_array_free(g_sq); mlx_array_free(beta_sq);
        mlx_array_free(y_t);
    }
    if (!kernel_done && ok) { ok = mlx_stack_axis(&y_stack, ys, /*axis=*/1, s) == 0; CHK(ok, "stack y"); }

    /* §13.5: salva lo stato ricorrente finale in k_cache[layer] e il conv_state aggiornato
     * in v_cache[layer], perche' il decode (forward_one_token) continui da dove ha lasciato
     * il prefill. conv_state aggiornato = ultimi cstate_len token di conv_input (come
     * mlx-lm GatedDeltaNet.__call__: cache[0] = conv_input[:, -n_keep:, :]). Eval prima di
     * salvare (state/conv_input sono lazy). */
    if (ok) {
        mlx_array cstate_new = mlx_array_new();
        /* conv_input: [B, cstate_len+S, conv_dim]; prendi gli ultimi cstate_len lungo l'asse 1 */
        int cst0[3] = {0, S, 0};
        int cst1[3] = {1, S + cstate_len, conv_dim};
        int cstrides[3] = {1, 1, 1};
        ok = mlx_slice(&cstate_new, conv_input, cst0, 3, cst1, 3, cstrides, 3, s) == 0;
        if (ok) { mlx_array_free(conv_state); conv_state = cstate_new; }
        else mlx_array_free(cstate_new);
        CHK(ok, "conv_state update");
        /* RMLX/bug P0 (2026-09-03): NIENTE eval per-layer GDN. state/conv_state restano LAZY
         * (refcount) e vengono materializzati dall' eval a fine token/chunk (dipendenza
         * dell'hidden finale); LLM_GDN_EVAL=1 ripristina il vecchio comportamento (2 sync/layer). */
        if (ok && llm_mlx_gdn_eval_forced())
            ok = mlx_array_eval(state) == 0 && mlx_array_eval(conv_state) == 0;
        if (ok) ok = mlx_array_set(k_cache, state) == 0 && mlx_array_set(v_cache, conv_state) == 0;
        CHK(ok, "gdn state store");
    }

    /* RMSNormGated(y, z) = silu(z) * rms_norm(y, norm_w, eps). norm_w = linear_attn.norm.weight [Dv]. */
    mlx_array norm_w=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &norm_w);
    CHK(ok, "norm.weight lookup");
    /* z [B,S,value_dim] -> [B,S,Hv,Dv] */
    int z_shape[4] = { B, S, Hv, Dv };
    mlx_array z4=mlx_array_new();
    if (ok) { ok = mlx_reshape(&z4, z, z_shape, 4, s) == 0; CHK(ok, "reshape z"); }
    mlx_array y_norm=mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&y_norm, y_stack, norm_w, m->cfg.rms_norm_eps, s); CHK(ok, "rms_norm y"); }
    /* silu(z) = z * sigmoid(z) */
    mlx_array z_sig=mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&z_sig, z4, s) == 0; }
    mlx_array silu_z=mlx_array_new();
    if (ok) { ok = mlx_multiply(&silu_z, z4, z_sig, s) == 0; CHK(ok, "silu(z)"); }
    mlx_array gated=mlx_array_new();
    if (ok) { ok = mlx_multiply(&gated, y_norm, silu_z, s) == 0; CHK(ok, "RMSNormGated"); }
    /* out_proj: [B,S,value_dim] -> [B,S,hidden_size] */
    int flat_shape[3] = { B, S, value_dim };
    mlx_array gated_flat=mlx_array_new();
    if (ok) { ok = mlx_reshape(&gated_flat, gated, flat_shape, 3, s) == 0; CHK(ok, "flatten gated"); }
    mlx_array out_w=mlx_array_new(), out_sc=mlx_array_new(), out_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.out_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &out_w, &out_sc, &out_b);
    CHK(ok, "out_proj lookup");
    mlx_array r=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&r, gated_flat, out_w, out_sc, out_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "out_proj");
         }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "out_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&r, gated_flat, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    /* residual: h = x + r (il layer lineare ritorna la somma, come DecoderLayer.__call__). */
    mlx_array h_out=mlx_array_new();
    if (ok) { ok = mlx_add(&h_out, x, r, s) == 0; CHK(ok, "residual add (gdn)"); }
    mlx_array_set(out_h, h_out);
    mlx_array_free(h_out);

    mlx_array_free(q_part); mlx_array_free(k_part); mlx_array_free(v_part);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_n); mlx_array_free(k_n); mlx_array_free(q_s); mlx_array_free(k_s);
    mlx_array_free(A_log); mlx_array_free(dt_bias); mlx_array_free(beta);
    mlx_array_free(a_plus_dt); mlx_array_free(sp_exp); mlx_array_free(sp_1p); mlx_array_free(sp_log);
    mlx_array_free(eAlog); mlx_array_free(neg_exp); mlx_array_free(g);
    mlx_array_free(q_rep); mlx_array_free(k_rep);
    mlx_array_free(conv_state); mlx_array_free(state); mlx_vector_array_free(ys); mlx_array_free(y_stack);
    mlx_array_free(norm_w); mlx_array_free(z4); mlx_array_free(y_norm);
    mlx_array_free(z_sig); mlx_array_free(silu_z); mlx_array_free(gated); mlx_array_free(gated_flat);
    mlx_array_free(out_w); mlx_array_free(out_sc); mlx_array_free(out_b); mlx_array_free(r);
    mlx_array_free(inv2); mlx_array_free(inv1); mlx_array_free(one_arr);
    mlx_vector_array_free(kv_split);
    /* FIX LEAK (2026-09-02): questi handle non erano mai liberati -> ogni forward GDN
     * (prefill o decode) tratteneva le attivazioni (h, qkv, z, av, bv, conv_input,
     * conv_out, sig, silu, neg_g) e gli handle dei pesi caricati -> memoria MLX "active"
     * cresceva di ~GB per eval su qwen3_5 (Jetsam a lungo termine / saturazione
     * imprevedibile). Tutti sono INPUT LAZY a questo punto (i consumer li referenziano
     * gia' nel grafo): liberare l'handle e' sicuro, nessun doppio free (non compaiono
     * nella cleanup esistente). */
    mlx_array_free(ln_w); mlx_array_free(h);
    mlx_array_free(qz_w); mlx_array_free(qz_sc); mlx_array_free(qz_b);
    mlx_array_free(ba_w); mlx_array_free(ba_sc); mlx_array_free(ba_b);
    mlx_array_free(conv_w);
    mlx_array_free(qkvz); mlx_array_free(ba);
    mlx_array_free(qkv); mlx_array_free(z); mlx_array_free(av); mlx_array_free(bv);
    mlx_array_free(conv_input); mlx_array_free(conv_out);
    mlx_array_free(sig); mlx_array_free(silu); mlx_array_free(neg_g);
    return ok;
}

/* FASE 1b.5: dispatch ibrido qwen3_5 — layer lineari -> gdn_attention_block,
 * layer full -> qwen3_moe_attention_block (GQA+QK-norm, attenzione standard) oppure,
 * se attn_output_gate (Qwen3NextAttention), qwen3_next_attention_block (output gate
 * su q_proj: `o_proj(sdpa(...) * sigmoid(gate))`). La distinzione lineare/full viene
 * da cfg.is_linear_layer (derivato da full_attention_interval). */
bool qwen3_5_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    /* LLM_PHASE_ATTN: contatori CUMULATIVI (stile LLM_PHASE) che separano il tempo speso nei layer
     * GDN (linear attention) da quello nei layer full-attention. Stampa all'ultimo layer: i due
     * print consecutivi delimitano un forward intero, quindi il delta e' il costo per forward.
     * Il timer NON e' utilizzabile per-layer con LLM_GDN_TIME (che sincronizza dentro il layer e
     * in un path a eval unico flusha lavoro di piu' layer): qui il sync e' al confine del blocco. */
    static double ph_g = 0.0, ph_f = 0.0; static long n_g = 0, n_f = 0;
    const bool gphase_a = llm_ai_mlx_opt_env("LLM_PHASE_ATTN") != NULL;
    struct timespec _a0, _a1;
    bool gdn = (m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx]);
    bool oka;
    if (gphase_a) clock_gettime(CLOCK_MONOTONIC, &_a0);
    if (gdn) {
        oka = gdn_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    } else if (m->cfg.attn_output_gate) {
        oka = qwen3_next_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    } else {
        oka = qwen3_moe_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    }
    if (gphase_a) {
        if (oka) oka = mlx_array_eval(*out_h) == 0;
        clock_gettime(CLOCK_MONOTONIC, &_a1);
        double dt = (_a1.tv_sec - _a0.tv_sec) + (_a1.tv_nsec - _a0.tv_nsec) / 1e9;
        if (gdn) { ph_g += dt; n_g++; } else { ph_f += dt; n_f++; }
        if (layer_idx + 1 >= m->cfg.n_layers)
            fprintf(stderr, "[LLM_PHASE_ATTN] gdn=%.3fs (%ld layer) full=%.3fs (%ld layer) fwdseq=%ld\n",
                    ph_g, n_g, ph_f, n_f, n_g + n_f);
    }
    return oka;
}

bool qwen3_next_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    if (m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx]) {
        return qwen3_next_linear_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    }
    /* Full (ogni full_attention_interval): Qwen3NextAttention — q_proj = 2x n_heads*head_dim
     * (queries + gate), o_proj(output * sigmoid(gate)) => qwen3_next_attention_block. */
    return qwen3_next_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
}

/* QSAAttention reale (Qwen4-Exp, qwen4_exp.py righe 767-890) -- layer
 * "qwen_sparse_attention" della Flash-Next. Carica i pesi VERI quantizzati dal checkpoint
 * con lo stesso schema di qwen3_next_attention_block/qwen4_linear_gdn_block
 * (multishard_get_triple + quantized_linear_cfg + hook LoRA), poi delega tutta la
 * matematica (norm, RoPE parziale, cache, selezione sparsa, SDPA) alle funzioni
 * verificate in llm_mlx_qwen4exp.h (contro riferimenti numpy indipendenti, incluso lo
 * scenario a 14 step dove la sparsificazione si attiva davvero -- vedi Todo C17).
 *
 * x: [1,L,hidden] -- GIA' normalizzato/mixato dall'hyper-connection del chiamante
 * (self.attn_hyper_connection nella reference): come qwen4_linear_gdn_block, questa
 * funzione NON fa il proprio input_layernorm.
 * k_cache/v_cache: KV-cache della sola attenzione principale (bhtd), stesso schema di
 * ogni altro attention_block. Lo stato dell'INDEXER (ring raw + chiavi compresse) vive
 * altrove (m->qsa_indexer_cache[layer_idx]) perche' ha una forma diversa da un mlx_array
 * singolo (vedi LlmQwen4ExpQSACache in llm_mlx_qwen4exp.h).
 *
 * Agganciata in ARCH_TABLE tramite qwen4_exp_attention_dispatch (che applica
 * attn_hyper_connection PRIMA di chiamare questa funzione, e hc_combine dopo) -- vedi
 * Todo C17 per lo stato di verifica end-to-end (nessun checkpoint reale ancora caricato
 * in questa sessione). */
bool qwen4_full_qsa_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
    mlx_stream s = m->stream;
    const int num_heads = m->cfg.n_heads, num_kv_heads = m->cfg.n_kv_heads, head_dim = m->cfg.head_dim;
    const int idx_heads = m->cfg.indexer_n_heads, idx_head_dim = m->cfg.indexer_head_dim;
    const float prf = m->cfg.partial_rotary_factor;
    const int rotary_dim = (prf > 0.0f && prf < 1.0f) ? (int)(head_dim * prf) : head_dim;
    const int idx_rotary_dim = (prf > 0.0f && prf < 1.0f) ? (int)(idx_head_dim * prf) : idx_head_dim;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen4_full_qsa[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    /* --- 1. Indexer QSA: proiezione quantizzata + norm pesi + cache + eventuale maschera --- */
    mlx_array idx_w=mlx_array_new(), idx_sc=mlx_array_new(), idx_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.indexer.index_qk_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &idx_w, &idx_sc, &idx_b);
    CHK(ok, "indexer.index_qk_proj lookup");
    mlx_array idx_qln=mlx_array_new(), idx_kln=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.indexer.q_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &idx_qln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.indexer.k_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &idx_kln);
    CHK(ok, "indexer layernorm lookup");

    mlx_array idx_projected = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&idx_projected, x, idx_w, idx_sc, idx_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "indexer.index_qk_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "indexer.index_qk_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&idx_projected, x, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }

    mlx_array idx_query_bhtd = mlx_array_new(), idx_raw_key = mlx_array_new();
    if (ok) { ok = llm_mlx_qwen4exp_qsa_indexer_postproject(&idx_query_bhtd, &idx_raw_key, idx_projected, idx_qln,
                                                             idx_heads, idx_head_dim, idx_rotary_dim, m->cfg.rope_theta,
                                                             cache_offset, s); CHK(ok, "indexer postproject"); }

    LlmQwen4ExpQSACache* idxc = llm_mlx_qwen4exp_model_qsa_cache(m, layer_idx);
    CHK(idxc != NULL, "indexer cache alloc");
    mlx_array idx_compressed = mlx_array_new();
    if (ok && idxc) { ok = llm_mlx_qwen4exp_qsa_cache_update(idxc, idx_raw_key, idx_kln, idx_rotary_dim, m->cfg.rope_theta,
                                                              &idx_compressed, s); CHK(ok, "indexer cache update"); }

    const int physical_kv_length = cache_offset + n_tokens;
    mlx_array sparse_mask = mlx_array_new();
    if (ok) { ok = llm_mlx_qwen4exp_qsa_select_mask(&sparse_mask, idx_query_bhtd, idx_compressed,
                                                     m->cfg.indexer_compress_ratio, m->cfg.indexer_budget,
                                                     cache_offset, physical_kv_length, s); CHK(ok, "indexer select_mask"); }

    /* --- 2. QSAAttention principale: pesi veri + delega alla postproject verificata --- */
    mlx_array qw=mlx_array_new(), qsc=mlx_array_new(), qb=mlx_array_new();
    mlx_array kw=mlx_array_new(), ksc=mlx_array_new(), kb=mlx_array_new();
    mlx_array vw=mlx_array_new(), vsc=mlx_array_new(), vb=mlx_array_new();
    mlx_array ow=mlx_array_new(), osc=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &qw, &qsc, &qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &kw, &ksc, &kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &vw, &vsc, &vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &osc, &ob);
    CHK(ok, "q/k/v/o_proj lookup");
    mlx_array qn_w=mlx_array_new(), kn_w=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &qn_w);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &kn_w);
    CHK(ok, "q/k_norm lookup");

    mlx_array q_lin=mlx_array_new(), k_lin=mlx_array_new(), v_lin=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q_lin, x, qw, qsc, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "q_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q_lin, x, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok) { ok = quantized_linear_cfg(&k_lin, x, kw, ksc, kb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "k_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k_lin, x, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok) { ok = quantized_linear_cfg(&v_lin, x, vw, vsc, vb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "v_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v_lin, x, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }

    mlx_array gated = mlx_array_new();
    if (ok) { ok = llm_mlx_qwen4exp_qsa_attention_postproject(&gated, k_cache, v_cache, q_lin, k_lin, v_lin,
                                                               qn_w, kn_w, has_cache, cache_offset,
                                                               num_heads, num_kv_heads, head_dim, rotary_dim,
                                                               m->cfg.rope_theta, m->cfg.rms_norm_eps, sparse_mask, s);
              CHK(ok, "attention postproject"); }
    if (ok) { ok = quantized_linear_cfg(out_h, gated, ow, osc, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "o_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(out_h, gated, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }

    mlx_array_free(idx_w); mlx_array_free(idx_sc); mlx_array_free(idx_b);
    mlx_array_free(idx_qln); mlx_array_free(idx_kln); mlx_array_free(idx_projected);
    mlx_array_free(idx_query_bhtd); mlx_array_free(idx_raw_key); mlx_array_free(idx_compressed);
    mlx_array_free(sparse_mask);
    mlx_array_free(qw); mlx_array_free(qsc); mlx_array_free(qb);
    mlx_array_free(kw); mlx_array_free(ksc); mlx_array_free(kb);
    mlx_array_free(vw); mlx_array_free(vsc); mlx_array_free(vb);
    mlx_array_free(ow); mlx_array_free(osc); mlx_array_free(ob);
    mlx_array_free(qn_w); mlx_array_free(kn_w);
    mlx_array_free(q_lin); mlx_array_free(k_lin); mlx_array_free(v_lin); mlx_array_free(gated);
    return ok;
#undef CHK
}
