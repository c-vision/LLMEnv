/* ============================================================================
 * llm_mlx_internlm2.c — dominio InternLM2 (model_type "internlm2").
 *
 * Struttura (mlx_lm `internlm2`):
 *  - attn: `attention.wqkv` FUSO (Q|K|V in un Linear, con bias) + `attention.wo` (bias).
 *    Layout del fuso: reshape [1,L,n_kv_heads,2+n_kv_groups,head_dim] dove
 *    n_kv_groups = n_heads/n_kv_heads; per ogni gruppo: [n_kv_groups teste Q, K, V].
 *    q = [:n_kv_groups], k = [-2], v = [-1]. RoPE tradizionale=false.
 *  - mlp: `feed_forward.w1` (gate) / `w3` (up) / `w2` (down), swiglu(w1,w3) -> w2.
 *  - norm: `attention_norm`, `ffn_norm` (RMSNorm).
 * Confronta `qwen_attention_block` (c_attn fuso ma split [Q_all,K_all,V_all]).
 * TRAP: quantized_linear_cfg / llm_mlx_gqa_sdpa ritornano bool TRUE=ok.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool internlm2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens,
                               mlx_array x, mlx_array* k_cache, mlx_array* v_cache,
                               bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const int groups = (nkv > 0) ? nh / nkv : 1;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "internlm2_attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention_norm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "attention_norm"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.wqkv", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "wqkv"); }

    /* reshape [1,L,nkv,2+groups,hd] -> q/gruppi, k, v */
    mlx_array q5=mlx_array_new(), k5=mlx_array_new(), v5=mlx_array_new();
    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        int sh5[5] = {1, n_tokens, nkv, 2 + groups, hd};
        mlx_array qkv5 = mlx_array_new();
        ok = mlx_reshape(&qkv5, qkv, sh5, 5, s) == 0;
        if (ok) {
            const int st[5] = {0,0,0,0,0}, stp[5] = {1,1,1,1,1};
            int qst[5] = {0,0,0,0,0}, qsp[5] = {1,n_tokens,nkv,groups,hd};
            int kst[5] = {0,0,0,2,0}, ksp[5] = {1,n_tokens,nkv,3,hd};
            int vst[5] = {0,0,0,3,0}, vsp[5] = {1,n_tokens,nkv,4,hd};
            ok = mlx_slice(&q5, qkv5, qst,5, qsp,5, stp,5, s)==0 &&
                 mlx_slice(&k5, qkv5, kst,5, ksp,5, stp,5, s)==0 &&
                 mlx_slice(&v5, qkv5, vst,5, vsp,5, stp,5, s)==0;
            (void)st; (void)stp;
        }
        mlx_array qr=mlx_array_new(), kr=mlx_array_new(), vr=mlx_array_new();
        if (ok) { int shq[4]={1,n_tokens,nh,hd};  ok = mlx_reshape(&qr, q5, shq,4, s)==0; }
        if (ok) { int shk[4]={1,n_tokens,nkv,hd}; ok = mlx_reshape(&kr, k5, shk,4, s)==0 && mlx_reshape(&vr, v5, shk,4, s)==0; }
        if (ok) { const int perm[4]={0,2,1,3};
                  ok = mlx_transpose_axes(&q_t, qr, perm,4, s)==0 &&
                       mlx_transpose_axes(&k_t, kr, perm,4, s)==0 &&
                       mlx_transpose_axes(&v_t, vr, perm,4, s)==0; }
        mlx_array_free(qkv5); mlx_array_free(qr); mlx_array_free(kr); mlx_array_free(vr);
        CHK(ok, "qkv split");
    }

    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, hd, false, base, 1.0f, cache_offset, no_freqs, s)==0 &&
             mlx_fast_rope(&k_rope, k_t, hd, false, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs); CHK(ok, "rope");
    }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_rope);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
            CHK(ok, "kv concat");
        } else { ok = mlx_array_set(&k_full, k_rope)==0 && mlx_array_set(&v_full, v_t)==0; CHK(ok, "kv init"); }
    }
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.wo", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &osp, &osb);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "wo"); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, normed, qw,qs,qb, qkv, q5,k5,v5, q_t,k_t,v_t, q_rope,k_rope,
                        k_full,v_full, attn, attn_bt, attn_flat, ow,osp,osb, attn_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool internlm2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "internlm2_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ffn_norm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "ffn_norm"); }

    /* gate = w1, up = w3 : swiglu(w1, w3) = silu(w1) * w3 ; down = w2. */
    mlx_array g1w=mlx_array_new(), g1s=mlx_array_new(), g1b=mlx_array_new();
    mlx_array u3w=mlx_array_new(), u3s=mlx_array_new(), u3b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.w1", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &g1w,&g1s,&g1b);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.w3", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &u3w,&u3s,&u3b);
    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gate, normed, g1w, g1s, g1b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w1 gate"); }
    if (ok) { ok = quantized_linear_cfg(&up,   normed, u3w, u3s, u3b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w3 up"); }
    mlx_array act = mlx_array_new();
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new();
              ok = mlx_sigmoid(&sg, gate, s)==0 && mlx_multiply(&sl, gate, sg, s)==0 && mlx_multiply(&act, sl, up, s)==0;
              mlx_array_free(sg); mlx_array_free(sl); CHK(ok, "swiglu"); }

    mlx_array dw=mlx_array_new(), ds=mlx_array_new(), db=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.w2", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &dw,&ds,&db);
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, act, dw, ds, db, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w2 down"); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok, "residual (ffn)"); }

    mlx_array all[] = { lnw, normed, g1w,g1s,g1b, u3w,u3s,u3b, gate, up, act, dw,ds,db, dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
