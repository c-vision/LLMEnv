/* ============================================================================
 * llm_mlx_gemma3.c — dominio Gemma3 ("gemma3_text").
 *
 * Arch: blocchi a 4 RMSNorm con scala (1+w) (gemma3 "RMSNorm" memorizza il DELTA: scale=1+weight),
 * attenzione GQA con q/k/v/o (no bias) + qk-norm (1+w, su head_dim) applicata PRIMA del rope;
 * RoPE (traditional=false, dims=head_dim) con base PER-LAYER (sliding: rope_local_base_freq;
 * global: rope_theta); SLIDING WINDOW sui layer con (layer_idx+1)%pattern!=0; scale SDPA =
 * query_pre_attn_scalar^-0.5. FFN GeGLU (gelu_approx(gate)*up). Embed scalati per sqrt(hidden).
 * Blocco: h = x + post_attention_layernorm(attn(input_layernorm(x)));
 *         out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h))).
 * Naming canonico `model.layers.` / `model.embed_tokens` / `model.norm`.
 *
 * TRAP: quantized_linear_cfg / llm_mlx_gqa_sdpa ritornano bool TRUE=ok -> MAI `== 0`.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* RMSNorm con scala (1+w) (gemma3: il weight memorizzato e' il delta). */
static bool g3_rms(LlmMlxModel* m, mlx_array* out, mlx_array x, mlx_array w, mlx_stream s) {
    mlx_array w1 = mlx_array_new(), one = mlx_array_new_float32(1.0f);
    bool ok = mlx_add(&w1, w, one, s) == 0;
    if (ok) ok = rms_norm_cfg(out, x, w1, m->cfg.rms_norm_eps, s);
    mlx_array_free(w1); mlx_array_free(one);
    return ok;
}

/* gelu_approx(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). */
static bool g3_gelu(mlx_array* out, mlx_array x, mlx_stream s) {
    mlx_array three=mlx_array_new_float32(3.0f), c=mlx_array_new_float32(0.044715f);
    mlx_array s2p=mlx_array_new_float32(0.7978845608028654f), one=mlx_array_new_float32(1.0f), half=mlx_array_new_float32(0.5f);
    mlx_array x3=mlx_array_new(), x3s=mlx_array_new(), inner=mlx_array_new();
    mlx_array inscale=mlx_array_new(), th=mlx_array_new(), op1=mlx_array_new(), g=mlx_array_new();
    bool ok = mlx_power(&x3, x, three, s) == 0;
    if (ok) ok = mlx_multiply(&x3s, x3, c, s) == 0;
    if (ok) ok = mlx_add(&inner, x, x3s, s) == 0;
    if (ok) ok = mlx_multiply(&inscale, inner, s2p, s) == 0;
    if (ok) ok = mlx_tanh(&th, inscale, s) == 0;
    if (ok) ok = mlx_add(&op1, th, one, s) == 0;
    if (ok) ok = mlx_multiply(&g, x, op1, s) == 0;
    if (ok) ok = mlx_multiply(out, g, half, s) == 0;
    mlx_array_free(three); mlx_array_free(c); mlx_array_free(s2p); mlx_array_free(one); mlx_array_free(half);
    mlx_array_free(x3); mlx_array_free(x3s); mlx_array_free(inner); mlx_array_free(inscale);
    mlx_array_free(th); mlx_array_free(op1); mlx_array_free(g);
    return ok;
}

bool gemma3_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const bool sliding = m->cfg.is_sliding_layer && layer_idx < m->cfg.n_layers && m->cfg.is_sliding_layer[layer_idx];
    const float base_theta = sliding ? m->cfg.rope_theta_sliding : m->cfg.rope_theta_full;
    const float scale = (m->cfg.query_pre_attn_scalar > 0.0f) ? 1.0f/sqrtf(m->cfg.query_pre_attn_scalar)
                                                              : 1.0f/sqrtf((float)hd);
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "gemma3[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array iln = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &iln);
    mlx_array normed = mlx_array_new();
    if (ok) ok = g3_rms(m, &normed, x, iln, s); CHK(ok, "input_layernorm");

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qbb=mlx_array_new();
    mlx_array kw=mlx_array_new(), ks=mlx_array_new(), kbb=mlx_array_new();
    mlx_array vw=mlx_array_new(), vs=mlx_array_new(), vbb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qbb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &kw,&ks,&kbb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &vw,&vs,&vbb);
    mlx_array q0=mlx_array_new(), k0=mlx_array_new(), v0=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q0, normed, qw,qs,qbb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "q_proj"); }
    if (ok) { ok = quantized_linear_cfg(&k0, normed, kw,ks,kbb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "k_proj"); }
    if (ok) { ok = quantized_linear_cfg(&v0, normed, vw,vs,vbb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "v_proj"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        const int qs4[4]={1,n_tokens,nh,hd}, kvs4[4]={1,n_tokens,nkv,hd};
        mlx_array qh=mlx_array_new(), kh=mlx_array_new(), vh=mlx_array_new();
        ok = mlx_reshape(&qh, q0, qs4,4, s)==0 && mlx_reshape(&kh, k0, kvs4,4, s)==0 && mlx_reshape(&vh, v0, kvs4,4, s)==0;
        if (ok) { const int perm[4]={0,2,1,3};
                  ok = mlx_transpose_axes(&q_t, qh, perm,4, s)==0 && mlx_transpose_axes(&k_t, kh, perm,4, s)==0 && mlx_transpose_axes(&v_t, vh, perm,4, s)==0; }
        mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh);
        CHK(ok, "head reshape/transpose");
    }

    /* qk-norm (1+w) su head_dim, PRIMA del rope. */
    mlx_array q_n=mlx_array_new(), k_n=mlx_array_new();
    if (ok) ok = mlx_array_set(&q_n, q_t)==0 && mlx_array_set(&k_n, k_t)==0;
    if (ok) {
        mlx_array qln=mlx_array_new(), kln=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &qln);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &kln);
        mlx_array qq=mlx_array_new(), kk=mlx_array_new();
        if (ok) ok = g3_rms(m, &qq, q_t, qln, s) && g3_rms(m, &kk, k_t, kln, s);
        if (ok) { mlx_array_set(&q_n, qq); mlx_array_set(&k_n, kk); }
        mlx_array_free(qq); mlx_array_free(kk); mlx_array_free(qln); mlx_array_free(kln);
        CHK(ok, "qk_norm");
    }

    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        mlx_optional_float base = {base_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_n, hd, /*traditional=*/false, base, 1.0f, cache_offset, no_freqs, s)==0
          && mlx_fast_rope(&k_rope, k_n, hd, false, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs); CHK(ok, "rope");
    }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache) llm_mlx_kv_trim(k_cache, v_cache, cache_offset, s);
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
    if (ok) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, scale, sliding ? m->cfg.sliding_window : 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&osp,&osb);
    mlx_array o = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj"); }
    /* h = x + post_attention_layernorm(o) */
    mlx_array paln=mlx_array_new(), po=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &paln);
    if (ok) { ok = g3_rms(m, &po, o, paln, s); CHK(ok, "post_attention_layernorm"); }
    if (ok) { ok = mlx_add(out_h, x, po, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { iln, normed, qw,qs,qbb, kw,ks,kbb, vw,vs,vbb, q0,k0,v0, q_t,k_t,v_t, q_n,k_n,
                        q_rope,k_rope, k_full,v_full, attn, attn_bt,attn_flat, ow,osp,osb, o, paln,po };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool gemma3_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "gemma3_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array pfln=mlx_array_new(), h_norm=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "pre_feedforward_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pfln);
    if (ok) { ok = g3_rms(m, &h_norm, h, pfln, s); CHK(ok, "pre_feedforward_layernorm"); }

    mlx_array gw=mlx_array_new(),gs=mlx_array_new(),gb=mlx_array_new();
    mlx_array uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new();
    mlx_array dw=mlx_array_new(),ds=mlx_array_new(),db=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &gw,&gs,&gb);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.up_proj", nb, sizeof(nb));   ok = ok && multishard_get_triple(&m->weights, nb, &uw,&us,&ub);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &dw,&ds,&db);
    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gate, h_norm, gw,gs,gb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "gate"); }
    if (ok) { ok = quantized_linear_cfg(&up,   h_norm, uw,us,ub, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "up"); }
    mlx_array gg=mlx_array_new(), act=mlx_array_new();
    if (ok) { ok = g3_gelu(&gg, gate, s); CHK(ok, "gelu"); }
    if (ok) { ok = mlx_multiply(&act, gg, up, s)==0; CHK(ok, "geglu"); }
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, act, dw,ds,db, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "down"); }

    /* out = h + post_feedforward_layernorm(dn) */
    mlx_array pfln2=mlx_array_new(), po=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_feedforward_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pfln2);
    if (ok) { ok = g3_rms(m, &po, dn, pfln2, s); CHK(ok, "post_feedforward_layernorm"); }
    if (ok) { ok = mlx_add(out, h, po, s)==0; CHK(ok, "residual (ffn)"); }

    mlx_array all[] = { pfln, h_norm, gw,gs,gb, uw,us,ub, dw,ds,db, gate, up, gg, act, dn, pfln2, po };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
