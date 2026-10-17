/* ============================================================================
 * llm_mlx_olmo_hybrid.c — dominio OLMo-Hybrid (AllenAI), hybrid a due tipi di layer.
 *
 * Modellato 1:1 su mlx_lm/models/olmo_hybrid.py (+ gated_delta.py condiviso).
 *   layer linear_attention (GatedDeltaNet, PRE-norm):
 *       h   = x + gdn(input_layernorm(x))
 *       out = h + mlp(post_attention_layernorm(h))
 *   layer full_attention (GQA senza RoPE, POST-norm):
 *       h   = x + post_attention_layernorm(self_attn(x))
 *       out = h + post_feedforward_layernorm(mlp(h))
 *
 * GatedDeltaNet di olmo: proiezioni q/k/v/a/b/g SEPARATE, conv1d depthwise SEPARATE
 * (q/k/v_conv1d, poi concatenate in una sola conv equivalente), normalize_qk, beta con
 * `linear_allow_neg_eigval` (beta in [0,2]), gated RMSNorm `o_norm` con `layer_norm_epsilon`.
 * Il config ha rope_parameters.rope_theta=null -> i layer full NON usano RoPE.
 *
 * Stato di sessione: k_cache[layer] = stato ricorrente [B,Hv,Dv,Dk]; v_cache[layer] = conv_state
 * [B, kk-1, 2*key_dim+value_dim] (q/k/v conv concat, come il conv_dim di qwen3_5). Layer full: KV cache.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_gdn_kernel.h"
#include "llm_mlx_swiglu_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool olmo_hybrid_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool olmo_hybrid_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* Carica un conv1d depthwise e lo normalizza a layout MLX (dim, kk, 1), cast all'act dtype. */
static bool oh_conv_w(LlmMlxModel* m, int layer_idx, const char* suffix, mlx_array* out, mlx_stream s) {
    char nb[256];
    tname(m->cfg.layer_prefix, layer_idx, suffix, nb, sizeof(nb));
    mlx_array cw = mlx_array_new();
    if (!multishard_get(&m->weights, nb, &cw)) return false;
    /* HF: [dim, 1, kk] -> MLX: [dim, kk, 1]. Se gia' [dim, kk, 1] nessuna trasposizione. */
    if (mlx_array_ndim(cw) == 3 && mlx_array_shape(cw)[1] == 1) {
        mlx_array t = mlx_array_new(); int ax[3] = {0, 2, 1};
        if (mlx_transpose_axes(&t, cw, ax, 3, s) == 0) { mlx_array_free(cw); cw = t; } else mlx_array_free(t);
    }
    mlx_array c32 = mlx_array_new();
    bool ok = mlx_astype(&c32, cw, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
    mlx_array_free(cw);
    if (ok) mlx_array_set(out, c32); else mlx_array_free(c32);
    return ok;
}

/* ---------- GatedDeltaNet (layer linear_attention), PRE-norm + residuo ---------- */
static bool olmo_hybrid_gdn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int B = 1, S = n_tokens;
    const int Hk = m->cfg.linear_num_key_heads, Hv = m->cfg.linear_num_value_heads;
    const int Dk = m->cfg.linear_key_head_dim, Dv = m->cfg.linear_value_head_dim;
    const int key_dim = Hk * Dk, value_dim = Hv * Dv, conv_dim = 2 * key_dim + value_dim;
    const int kk = m->cfg.linear_conv_kernel_dim, cstate_len = kk - 1;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "olmo_hybrid.gdn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    /* input_layernorm + proiezioni q/k/v/a/b/g */
    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &ln_w);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, ln_w, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    mlx_array av = mlx_array_new(), bv = mlx_array_new(), gen = mlx_array_new();
    if (ok) {
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new(), lin = mlx_array_new();
#define OH_LIN(suf, src, dst) do { tname(m->cfg.layer_prefix, layer_idx, suf, nb, sizeof(nb)); \
        ok = ok && multishard_get_triple(&m->weights, nb, &w, &sc, &bb) && \
             quantized_linear_cfg(&lin, src, w, sc, bb, gsz, bts, m->cfg.quant_mode, s); \
        if (ok) mlx_array_set(&dst, lin); } while (0)
        OH_LIN("linear_attn.q_proj", h, q);
        OH_LIN("linear_attn.k_proj", h, k);
        OH_LIN("linear_attn.v_proj", h, v);
        OH_LIN("linear_attn.a_proj", h, av);
        OH_LIN("linear_attn.b_proj", h, bv);
        OH_LIN("linear_attn.g_proj", h, gen);
#undef OH_LIN
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb); mlx_array_free(lin);
        CHK(ok, "projections");
    }

    /* conv depthwise: concat [q,k,v] -> un solo conv (pesi q/k/v concat) */
    mlx_array qkv = mlx_array_new();
    if (ok) { mlx_array a3[3] = { q, k, v }; mlx_vector_array vv = mlx_vector_array_new_data(a3, 3);
              ok = mlx_concatenate_axis(&qkv, vv, -1, s) == 0; mlx_vector_array_free(vv); CHK(ok, "concat qkv"); }
    mlx_array qcw = mlx_array_new(), kcw = mlx_array_new(), vcw = mlx_array_new();
    if (ok) { ok = oh_conv_w(m, layer_idx, "linear_attn.q_conv1d.weight", &qcw, s)
                 && oh_conv_w(m, layer_idx, "linear_attn.k_conv1d.weight", &kcw, s)
                 && oh_conv_w(m, layer_idx, "linear_attn.v_conv1d.weight", &vcw, s); CHK(ok, "conv weights"); }
    mlx_array conv_w = mlx_array_new();
    if (ok) { mlx_array a3[3] = { qcw, kcw, vcw }; mlx_vector_array vv = mlx_vector_array_new_data(a3, 3);
              ok = mlx_concatenate_axis(&conv_w, vv, 0, s) == 0; mlx_vector_array_free(vv); CHK(ok, "concat convw"); }
    mlx_array conv_state = mlx_array_new();
    if (ok) {
        if (has_cache && v_cache && v_cache->ctx != NULL && mlx_array_ndim(*v_cache) > 0) ok = mlx_array_set(&conv_state, *v_cache) == 0;
        else { int cs[3] = { B, cstate_len, conv_dim }; ok = mlx_zeros(&conv_state, cs, 3, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0; }
        CHK(ok, "conv_state init");
    }
    mlx_array conv_input = mlx_array_new();
    if (ok) { mlx_array a2[2] = { conv_state, qkv }; mlx_vector_array vv = mlx_vector_array_new_data(a2, 2);
              ok = mlx_concatenate_axis(&conv_input, vv, 1, s) == 0; mlx_vector_array_free(vv); CHK(ok, "concat conv_state"); }
    mlx_array conv_out = mlx_array_new();
    if (ok) {
        int st[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, id[1]={1};
        ok = mlx_conv_general(&conv_out, conv_input, conv_w, st,1, pl,1, ph,1, kd,1, id,1, conv_dim, false, s) == 0;
        CHK(ok, "conv1d");
    }
    mlx_array sig = mlx_array_new(), silu = mlx_array_new();
    if (ok) { ok = llm_mlx_silu_parity(&silu, &sig, conv_out, s) == 0; CHK(ok, "silu"); }
    mlx_array qp = mlx_array_new(), kp = mlx_array_new(), vp = mlx_array_new();
    if (ok) { int secs[2] = { key_dim, 2 * key_dim }; mlx_vector_array sp = mlx_vector_array_new();
              ok = mlx_split_sections(&sp, silu, secs, 2, -1, s) == 0;
              if (ok) ok = mlx_vector_array_get(&qp, sp, 0) == 0 && mlx_vector_array_get(&kp, sp, 1) == 0 && mlx_vector_array_get(&vp, sp, 2) == 0;
              mlx_vector_array_free(sp); CHK(ok, "split qkv"); }
    mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new();
    if (ok) { int qs[4] = { B, S, Hk, Dk }, vs[4] = { B, S, Hv, Dv };
              ok = mlx_reshape(&q4, qp, qs, 4, s) == 0 && mlx_reshape(&k4, kp, qs, 4, s) == 0 && mlx_reshape(&v4, vp, vs, 4, s) == 0;
              CHK(ok, "reshape qkv"); }

    /* normalize_qk: inv_scale=Dk^-0.5; rms_eps = 1e-6 * inv_scale^2; q*=inv^2, k*=inv */
    const double inv_scale = pow((double)Dk, -0.5);
    const float rms_eps = (float)(1e-6 * inv_scale * inv_scale);
    mlx_array q_n = mlx_array_new(), k_n = mlx_array_new();
    if (ok) { ok = mlx_fast_rms_norm_no_scale(&q_n, q4, Dk, rms_eps, s) && mlx_fast_rms_norm_no_scale(&k_n, k4, Dk, rms_eps, s); CHK(ok, "rms q/k"); }
    mlx_array inv2 = llm_mlx_act_scalar((float)(inv_scale * inv_scale));
    mlx_array inv1 = llm_mlx_act_scalar((float)inv_scale);
    mlx_array q_s = mlx_array_new(), k_s = mlx_array_new();
    if (ok) { ok = mlx_multiply(&q_s, q_n, inv2, s) == 0 && mlx_multiply(&k_s, k_n, inv1, s) == 0; CHK(ok, "inv_scale"); }

    /* beta = sigmoid(b) (*2 se allow_neg_eigval); g = exp(-exp(A_log)*softplus(a+dt_bias)) */
    mlx_array A_log = mlx_array_new(), dt_bias = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.A_log", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &A_log);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.dt_bias", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &dt_bias);
    CHK(ok, "A_log/dt_bias");
    mlx_array beta = mlx_array_new(), g = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&beta, bv, s) == 0; CHK(ok, "beta sigmoid"); }
    if (ok && m->cfg.hybrid.allow_neg_eigval) { mlx_array two = llm_mlx_act_scalar(2.0f), b2 = mlx_array_new();
        ok = mlx_multiply(&b2, beta, two, s) == 0; if (ok) mlx_array_set(&beta, b2); mlx_array_free(b2); mlx_array_free(two); CHK(ok, "beta*2"); }
    if (ok) {
        mlx_array eAlog = mlx_array_new(), apd = mlx_array_new(), spl = mlx_array_new();
        mlx_array ne = mlx_array_new(), ng = mlx_array_new(), zero = llm_mlx_act_scalar(0.0f);
        ok = mlx_exp(&eAlog, A_log, s) == 0 && mlx_add(&apd, av, dt_bias, s) == 0 && mlx_logaddexp(&spl, apd, zero, s) == 0
          && mlx_multiply(&ne, eAlog, spl, s) == 0 && mlx_negative(&ng, ne, s) == 0 && mlx_exp(&g, ng, s) == 0;
        mlx_array_free(eAlog); mlx_array_free(apd); mlx_array_free(spl); mlx_array_free(ne); mlx_array_free(ng); mlx_array_free(zero);
        CHK(ok, "g compute");
    }

    /* repeat q/k da Hk a Hv */
    const int rf = Hv / Hk;
    mlx_array q_r = mlx_array_new(), k_r = mlx_array_new();
    if (ok) { if (rf > 1) ok = mlx_repeat_axis(&q_r, q_s, rf, 2, s) == 0 && mlx_repeat_axis(&k_r, k_s, rf, 2, s) == 0;
              else ok = mlx_array_set(&q_r, q_s) == 0 && mlx_array_set(&k_r, k_s) == 0; CHK(ok, "repeat"); }

    /* stato ricorrente */
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache) > 0) ok = mlx_array_set(&state, *k_cache) == 0;
        else { int ss[4] = { B, Hv, Dv, Dk }; ok = mlx_zeros(&state, ss, 4, MLX_FLOAT32, s) == 0; }
        CHK(ok, "state init");
    }
    /* scan: kernel (prefill) o loop per-token */
    mlx_array y_stack = mlx_array_new();
    bool kernel_done = false;
    if (ok && S >= 1) {
        mlx_array sn = mlx_array_new();
        if (llm_mlx_gdn_prefill(&y_stack, &sn, q_r, k_r, v4, g, beta, state, S, 0, s) == 0) { mlx_array_set(&state, sn); kernel_done = true; }
        mlx_array_free(sn);
    }
    mlx_vector_array ys = mlx_vector_array_new();
    if (!kernel_done) for (int t = 0; ok && t < S; t++) {
        int s0[4]={0,t,0,0}, s1[4]={1,t+1,Hv,Dk}, sv[4]={1,t+1,Hv,Dv}, o4[4]={1,1,1,1}, o3[3]={1,1,1};
        mlx_array qt=mlx_array_new(),kt=mlx_array_new(),vt=mlx_array_new(),gt=mlx_array_new(),bt=mlx_array_new();
        if (ok) { ok = mlx_slice(&qt,q_r,s0,4,s1,4,o4,4,s)==0 && mlx_slice(&kt,k_r,s0,4,s1,4,o4,4,s)==0 && mlx_slice(&vt,v4,s0,4,sv,4,o4,4,s)==0;
                  int g0[3]={0,t,0}, g1[3]={1,t+1,Hv};
                  ok = ok && mlx_slice(&gt,g,g0,3,g1,3,o3,3,s)==0 && mlx_slice(&bt,beta,g0,3,g1,3,o3,3,s)==0; }
        mlx_array qsq=mlx_array_new(),ksq=mlx_array_new(),vsq=mlx_array_new(),gsq=mlx_array_new(),bsq=mlx_array_new();
        if (ok) ok = mlx_squeeze_axis(&qsq,qt,1,s)==0 && mlx_squeeze_axis(&ksq,kt,1,s)==0 && mlx_squeeze_axis(&vsq,vt,1,s)==0
                  && mlx_squeeze_axis(&gsq,gt,1,s)==0 && mlx_squeeze_axis(&bsq,bt,1,s)==0;
        mlx_array yt = mlx_array_new();
        if (ok) ok = llm_mlx_gated_delta_step(&yt, &state, qsq, ksq, vsq, gsq, bsq, s);
        if (ok) ok = mlx_vector_array_append_value(ys, yt) == 0;
        mlx_array_free(qt);mlx_array_free(kt);mlx_array_free(vt);mlx_array_free(gt);mlx_array_free(bt);
        mlx_array_free(qsq);mlx_array_free(ksq);mlx_array_free(vsq);mlx_array_free(gsq);mlx_array_free(bsq);mlx_array_free(yt);
    }
    if (!kernel_done && ok) { ok = mlx_stack_axis(&y_stack, ys, 1, s) == 0; CHK(ok, "stack y"); }
    mlx_vector_array_free(ys);
    /* store stati */
    if (ok) {
        mlx_array cnew = mlx_array_new();
        int c0[3]={0,S,0}, c1[3]={1,S+cstate_len,conv_dim}, cst[3]={1,1,1};
        ok = mlx_slice(&cnew, conv_input, c0,3, c1,3, cst,3, s) == 0;
        if (ok) ok = mlx_array_set(k_cache, state) == 0 && mlx_array_set(v_cache, cnew) == 0;
        mlx_array_free(cnew);
        CHK(ok, "store state");
    }

    /* gated RMSNorm: normed = rms_norm(y, o_norm.weight, layer_norm_epsilon); out = silu(gate)*normed (f32) */
    mlx_array on_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.o_norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &on_w);
    int zsh[4] = { B, S, Hv, Dv };
    mlx_array gen4 = mlx_array_new();
    if (ok) { ok = mlx_reshape(&gen4, gen, zsh, 4, s) == 0; CHK(ok, "reshape gate"); }
    mlx_array y_norm = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&y_norm, y_stack, on_w, m->cfg.mamba.ln_eps, s); CHK(ok, "o_norm"); }
    mlx_array gated = mlx_array_new();
    if (ok) {
        mlx_array z32=mlx_array_new(), yn32=mlx_array_new(), zs=mlx_array_new(), sl=mlx_array_new(), g32=mlx_array_new();
        ok = mlx_astype(&z32, gen4, MLX_FLOAT32, s) == 0 && mlx_astype(&yn32, y_norm, MLX_FLOAT32, s) == 0
          && mlx_sigmoid(&zs, z32, s) == 0 && mlx_multiply(&sl, z32, zs, s) == 0 && mlx_multiply(&g32, yn32, sl, s) == 0
          && mlx_astype(&gated, g32, mlx_array_dtype(y_stack), s) == 0;
        mlx_array_free(z32); mlx_array_free(yn32); mlx_array_free(zs); mlx_array_free(sl); mlx_array_free(g32);
        CHK(ok, "precise_swiglu");
    }
    mlx_array flat = mlx_array_new(), o = mlx_array_new();
    if (ok) { int f3[3] = { B, S, value_dim }; ok = mlx_reshape(&flat, gated, f3, 3, s) == 0; CHK(ok, "reshape flat"); }
    if (ok) {
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.o_proj", nb, sizeof(nb));
        ok = multishard_get_triple(&m->weights, nb, &w, &sc, &bb) && quantized_linear_cfg(&o, flat, w, sc, bb, gsz, bts, m->cfg.quant_mode, s);
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb);
        CHK(ok, "o_proj");
    }
    if (ok) { ok = mlx_add(&*out_h, x, o, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(ln_w); mlx_array_free(h);
    mlx_array_free(q); mlx_array_free(k); mlx_array_free(v); mlx_array_free(av); mlx_array_free(bv); mlx_array_free(gen);
    mlx_array_free(qkv); mlx_array_free(qcw); mlx_array_free(kcw); mlx_array_free(vcw); mlx_array_free(conv_w);
    mlx_array_free(conv_state); mlx_array_free(conv_input); mlx_array_free(conv_out); mlx_array_free(sig); mlx_array_free(silu);
    mlx_array_free(qp); mlx_array_free(kp); mlx_array_free(vp); mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_n); mlx_array_free(k_n); mlx_array_free(inv2); mlx_array_free(inv1); mlx_array_free(q_s); mlx_array_free(k_s);
    mlx_array_free(A_log); mlx_array_free(dt_bias); mlx_array_free(beta); mlx_array_free(g);
    mlx_array_free(q_r); mlx_array_free(k_r); mlx_array_free(state); mlx_array_free(y_stack);
    mlx_array_free(on_w); mlx_array_free(gen4); mlx_array_free(y_norm); mlx_array_free(gated); mlx_array_free(flat); mlx_array_free(o);
    return ok;
#undef CHK
}

/* ---------- full_attention (GQA + q/k RMSNorm flat, SENZA RoPE) + POST-norm + residuo ---------- */
static bool olmo_hybrid_full_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "olmo_hybrid.full[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array qw=mlx_array_new(),qsc=mlx_array_new(),qb=mlx_array_new();
    mlx_array kw=mlx_array_new(),ksc=mlx_array_new(),kb=mlx_array_new();
    mlx_array vw=mlx_array_new(),vsc=mlx_array_new(),vb=mlx_array_new();
    mlx_array ow=mlx_array_new(),osc=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qsc, &qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &kw, &ksc, &kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &vw, &vsc, &vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow, &osc, &ob);
    mlx_array qn_w=mlx_array_new(), kn_w=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &qn_w);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &kn_w);
    CHK(ok, "proj/norm lookup");

    mlx_array q_lin=mlx_array_new(), k_lin=mlx_array_new(), v_lin=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&q_lin, x, qw, qsc, qb, gsz, bts, m->cfg.quant_mode, s);
    if (ok) ok = quantized_linear_cfg(&k_lin, x, kw, ksc, kb, gsz, bts, m->cfg.quant_mode, s);
    if (ok) ok = quantized_linear_cfg(&v_lin, x, vw, vsc, vb, gsz, bts, m->cfg.quant_mode, s);
    CHK(ok, "qkv");
    mlx_array q_n=mlx_array_new(), k_n=mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&q_n, q_lin, qn_w, m->cfg.rms_norm_eps, s) && rms_norm_cfg(&k_n, k_lin, kn_w, m->cfg.rms_norm_eps, s); CHK(ok, "qk norm"); }
    mlx_array q4=mlx_array_new(),k4=mlx_array_new(),v4=mlx_array_new();
    if (ok) { int qs[4]={1,n_tokens,nh,hd}, ks[4]={1,n_tokens,nkv,hd};
              ok = mlx_reshape(&q4,q_n,qs,4,s)==0 && mlx_reshape(&k4,k_n,ks,4,s)==0 && mlx_reshape(&v4,v_lin,ks,4,s)==0; CHK(ok, "reshape"); }
    mlx_array q_t=mlx_array_new(),k_t=mlx_array_new(),v_t=mlx_array_new();
    if (ok) { const int p[4]={0,2,1,3}; ok = mlx_transpose_axes(&q_t,q4,p,4,s)==0 && mlx_transpose_axes(&k_t,k4,p,4,s)==0 && mlx_transpose_axes(&v_t,v4,p,4,s)==0; CHK(ok, "transpose"); }
    /* NIENTE RoPE (rope_theta=null nel config) */
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache) == 4 && mlx_array_size(*k_cache) > 0) {
            mlx_array kn=mlx_array_new(), vn=mlx_array_new();
            mlx_vector_array kv=mlx_vector_array_new(); mlx_vector_array_append_value(kv,*k_cache); mlx_vector_array_append_value(kv,k_t);
            ok = mlx_concatenate_axis(&kn,kv,2,s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,v_t);
            ok = ok && mlx_concatenate_axis(&vn,vv,2,s)==0; mlx_vector_array_free(vv);
            if (ok) { mlx_array_set(k_cache, kn); mlx_array_set(v_cache, vn); }
            mlx_array_free(kn); mlx_array_free(vn);
        } else { mlx_array_set(k_cache, k_t); mlx_array_set(v_cache, v_t); }
        CHK(ok, "kv cache");
    }
    mlx_array attn=mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_t, *k_cache, *v_cache, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }
    mlx_array abld=mlx_array_new(), aflat=mlx_array_new();
    if (ok) { const int p[4]={0,2,1,3}; ok = mlx_transpose_axes(&abld, attn, p, 4, s) == 0; }
    if (ok) { int r3[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&aflat, abld, r3, 3, s) == 0; CHK(ok, "reshape attn"); }
    mlx_array o_out=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o_out, aflat, ow, osc, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj"); }
    mlx_array pan_w=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pan_w);
    mlx_array pan=mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&pan, o_out, pan_w, m->cfg.rms_norm_eps, s); CHK(ok, "post_attn norm"); }
    if (ok) { ok = mlx_add(&*out_h, x, pan, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(qw);mlx_array_free(qsc);mlx_array_free(qb);mlx_array_free(kw);mlx_array_free(ksc);mlx_array_free(kb);
    mlx_array_free(vw);mlx_array_free(vsc);mlx_array_free(vb);mlx_array_free(ow);mlx_array_free(osc);mlx_array_free(ob);
    mlx_array_free(qn_w);mlx_array_free(kn_w);mlx_array_free(q_lin);mlx_array_free(k_lin);mlx_array_free(v_lin);
    mlx_array_free(q_n);mlx_array_free(k_n);mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v4);
    mlx_array_free(q_t);mlx_array_free(k_t);mlx_array_free(v_t);mlx_array_free(attn);mlx_array_free(abld);mlx_array_free(aflat);
    mlx_array_free(o_out);mlx_array_free(pan_w);mlx_array_free(pan);
    return ok;
#undef CHK
}

bool olmo_hybrid_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool lin = m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx];
    if (lin) return olmo_hybrid_gdn_block(m, layer_idx, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    return olmo_hybrid_full_attn_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
}

bool olmo_hybrid_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool lin = m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx];
    if (lin) return dense_swiglu_ffn(m, layer_idx, n_tokens, h, out);
    return olmo3_ffn_block(m, layer_idx, n_tokens, h, out);
}
