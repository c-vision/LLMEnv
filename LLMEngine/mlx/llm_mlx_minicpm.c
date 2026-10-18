/* ============================================================================
 * llm_mlx_minicpm.c — dominio MiniCPM v1/v2 (model_type "minicpm").
 *
 * Struttura (mlx_lm `minicpm`): llama-like (RMSNorm, q/k/v/o SENZA bias, RoPE pieno
 * traditional=cfg.rope_traditional, gate/up/down swiglu) + **muP**:
 *  - embed × scale_emb                      -> cfg.embed_scale (= scale_emb)
 *  - ogni ramo residuo × (scale_depth/sqrt(n_layers))  -> cfg.granite.residual_multiplier
 *  - logits ÷ (hidden/dim_model_base)       -> cfg.falcon_h1_out_scale (= dim_model_base/hidden)
 * Quindi h = x + attn(ln(x))*rm ; out = h + mlp(post_ln(h))*rm.
 * TRAP: quantized_linear_cfg/llm_mlx_gqa_sdpa ritornano TRUE=ok.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float minicpm_rm(LlmMlxModel* m) {
    return m->cfg.granite.residual_multiplier != 0.0f ? m->cfg.granite.residual_multiplier : 1.0f;
}

bool minicpm_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const float rm = minicpm_rm(m);
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "minicpm_attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    mlx_array kw=mlx_array_new(), ks=mlx_array_new(), kb=mlx_array_new();
    mlx_array vw=mlx_array_new(), vs=mlx_array_new(), vb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &kw,&ks,&kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &vw,&vs,&vb);
    mlx_array q=mlx_array_new(), k=mlx_array_new(), v=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q, normed, qw,qs,qb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"q_proj"); }
    if (ok) { ok = quantized_linear_cfg(&k, normed, kw,ks,kb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"k_proj"); }
    if (ok) { ok = quantized_linear_cfg(&v, normed, vw,vs,vb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"v_proj"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) { int qs4[4]={1,n_tokens,nh,hd}, kvs4[4]={1,n_tokens,nkv,hd};
              mlx_array qh=mlx_array_new(), kh=mlx_array_new(), vh=mlx_array_new();
              ok = mlx_reshape(&qh, q, qs4,4, s)==0 && mlx_reshape(&kh, k, kvs4,4, s)==0 && mlx_reshape(&vh, v, kvs4,4, s)==0;
              if (ok) { const int perm[4]={0,2,1,3};
                        ok = mlx_transpose_axes(&q_t, qh, perm,4, s)==0 && mlx_transpose_axes(&k_t, kh, perm,4, s)==0 && mlx_transpose_axes(&v_t, vh, perm,4, s)==0; }
              mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh); CHK(ok,"reshape"); }

    mlx_array q_r=mlx_array_new(), k_r=mlx_array_new();
    if (ok) { mlx_optional_float base = { m->cfg.rope_theta, true }; mlx_array nf = mlx_array_new();
              ok = mlx_fast_rope(&q_r, q_t, hd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, s)==0 &&
                   mlx_fast_rope(&k_r, k_t, hd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, s)==0;
              mlx_array_free(nf); CHK(ok,"rope"); }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv=mlx_vector_array_new(); mlx_vector_array_append_value(kv,*k_cache); mlx_vector_array_append_value(kv,k_r);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
        } else { ok = mlx_array_set(&k_full, k_r)==0 && mlx_array_set(&v_full, v_t)==0; }
        CHK(ok,"kv");
    }
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok,"kv store"); }

    mlx_array attn=mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_r, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok,"sdpa"); }

    mlx_array abt=mlx_array_new(), afl=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&abt, attn, perm,4, s)==0; }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&afl, abt, fl,3, s)==0; CHK(ok,"flat"); }

    mlx_array ow=mlx_array_new(), os=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os,&ob);
    mlx_array ao=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&ao, afl, ow,os,ob, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"o_proj"); }
    /* residuo × rm */
    if (ok && rm != 1.0f) { mlx_array sc=mlx_array_new_float32(rm), ao2=mlx_array_new();
                            ok = mlx_multiply(&ao2, ao, sc, s)==0; if (ok) mlx_array_set(&ao, ao2);
                            mlx_array_free(sc); mlx_array_free(ao2); }
    if (ok) { ok = mlx_add(out_h, x, ao, s)==0; CHK(ok,"residual (attn)"); }

    mlx_array all[] = { lnw,normed, qw,qs,qb, kw,ks,kb, vw,vs,vb, q,k,v, q_t,k_t,v_t, q_r,k_r,
                        k_full,v_full, attn, abt,afl, ow,os,ob, ao };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool minicpm_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const float rm = minicpm_rm(m);
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "minicpm_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "post_attention_layernorm"); }

    mlx_array gw=mlx_array_new(), gs=mlx_array_new(), gb=mlx_array_new();
    mlx_array uw=mlx_array_new(), us=mlx_array_new(), ub=mlx_array_new();
    mlx_array dyw=mlx_array_new(), dys=mlx_array_new(), dyb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &gw,&gs,&gb);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.up_proj", nb, sizeof nb);   ok = ok && multishard_get_triple(&m->weights, nb, &uw,&us,&ub);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &dyw,&dys,&dyb);
    mlx_array gate=mlx_array_new(), up=mlx_array_new(), act=mlx_array_new(), dn=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gate, normed, gw,gs,gb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"gate"); }
    if (ok) { ok = quantized_linear_cfg(&up,   normed, uw,us,ub, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"up"); }
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new();
              ok = mlx_sigmoid(&sg,gate,s)==0 && mlx_multiply(&sl,gate,sg,s)==0 && mlx_multiply(&act,sl,up,s)==0;
              mlx_array_free(sg); mlx_array_free(sl); CHK(ok,"swiglu"); }
    if (ok) { ok = quantized_linear_cfg(&dn, act, dyw,dys,dyb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"down"); }
    if (ok && rm != 1.0f) { mlx_array sc=mlx_array_new_float32(rm), dn2=mlx_array_new();
                            ok = mlx_multiply(&dn2, dn, sc, s)==0; if (ok) mlx_array_set(&dn, dn2);
                            mlx_array_free(sc); mlx_array_free(dn2); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok,"residual (ffn)"); }

    mlx_array all[] = { lnw,normed, gw,gs,gb, uw,us,ub, dyw,dys,dyb, gate, up, act, dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
