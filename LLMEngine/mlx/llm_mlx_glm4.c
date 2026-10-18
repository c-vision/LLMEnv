/* ============================================================================
 * llm_mlx_glm4.c — dominio GLM-4 (chatglm/Zhipu) DENSE (model_type "glm4").
 * Modellato 1:1 su mlx_lm/models/glm4.py.
 *   Layer: x = x + post_self_attn_layernorm( attn(input_layernorm(x)) )
 *          out = h + post_mlp_layernorm( mlp(post_attention_layernorm(h)) )
 *   4 RMSNorm per layer. Attention: GQA + bias(q/k/v) + RoPE PARZIALE (traditional=true,
 *   dims=int(head_dim*partial_rotary_factor)), o_proj senza bias. MLP FUSO: gate_up_proj
 *   [2*inter] -> split gate/up -> swiglu -> down_proj.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool glm4_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool glm4_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* lineare quantizzata + bias lineare opzionale (q/k/v_proj.bias). */
static bool glm4_lin_bias(mlx_array* out, LlmMlxModel* m, int layer_idx, const char* suf, mlx_array x,
                          int gsz, int bts, mlx_stream s) {
    char nb[256];
    tname(m->cfg.layer_prefix, layer_idx, suf, nb, sizeof(nb));
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new();
    if (!multishard_get_triple(&m->weights, nb, &w, &sc, &bb)) { mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb); return false; }
    bool ok = quantized_linear_cfg(out, x, w, sc, bb, gsz, bts, m->cfg.quant_mode, s);
    /* Il bias lineare (`suf + ".bias"`) e' gia' nello slot `biases` del triple
     * (multishard_get_triple fallback `.biases` -> `.bias`) e query il path plain di
     * quantized_linear_cfg lo APPLICA (res + biases). NON ri-aggiungerlo qui (double-bias). */
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb);
    return ok;
}

bool glm4_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "glm4.attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &ln_w);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, ln_w, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) ok = glm4_lin_bias(&q, m, layer_idx, "self_attn.q_proj", h, gsz, bts, s);
    if (ok) ok = glm4_lin_bias(&k, m, layer_idx, "self_attn.k_proj", h, gsz, bts, s);
    if (ok) ok = glm4_lin_bias(&v, m, layer_idx, "self_attn.v_proj", h, gsz, bts, s);
    CHK(ok, "qkv");

    mlx_array q4=mlx_array_new(),k4=mlx_array_new(),v4=mlx_array_new();
    if (ok) { int qs[4]={1,n_tokens,nh,hd}, ks[4]={1,n_tokens,nkv,hd};
              ok = mlx_reshape(&q4,q,qs,4,s)==0 && mlx_reshape(&k4,k,ks,4,s)==0 && mlx_reshape(&v4,v,ks,4,s)==0; CHK(ok,"reshape"); }
    mlx_array q_t=mlx_array_new(),k_t=mlx_array_new(),v_t=mlx_array_new();
    if (ok) { const int p[4]={0,2,1,3}; ok = mlx_transpose_axes(&q_t,q4,p,4,s)==0 && mlx_transpose_axes(&k_t,k4,p,4,s)==0 && mlx_transpose_axes(&v_t,v4,p,4,s)==0; CHK(ok,"transpose"); }
    /* RoPE PARZIALE (traditional=true): ruota i primi int(hd*partial_rotary_factor) dim. */
    mlx_array q_r=mlx_array_new(), k_r=mlx_array_new();
    if (ok) {
        float prf = m->cfg.partial_rotary_factor;
        int rd = (prf > 0.0f && prf < 1.0f) ? (int)(hd * prf) : hd;
        mlx_optional_float base = { m->cfg.rope_theta, true };
        mlx_array nf = mlx_array_new();
        ok = mlx_fast_rope(&q_r, q_t, rd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, s)==0 &&
             mlx_fast_rope(&k_r, k_t, rd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, s)==0;
        mlx_array_free(nf);
        CHK(ok, "rope");
    }
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache)==4 && mlx_array_size(*k_cache)>0) {
            mlx_array kn=mlx_array_new(), vn=mlx_array_new();
            mlx_vector_array kv=mlx_vector_array_new(); mlx_vector_array_append_value(kv,*k_cache); mlx_vector_array_append_value(kv,k_r);
            ok = mlx_concatenate_axis(&kn,kv,2,s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,v_t);
            ok = ok && mlx_concatenate_axis(&vn,vv,2,s)==0; mlx_vector_array_free(vv);
            if (ok) { mlx_array_set(k_cache,kn); mlx_array_set(v_cache,vn); }
            mlx_array_free(kn); mlx_array_free(vn);
        } else { mlx_array_set(k_cache,k_r); mlx_array_set(v_cache,v_t); }
        CHK(ok, "kv");
    }
    mlx_array attn=mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_r, *k_cache, *v_cache, 1.0f/sqrtf((float)hd), 0, s); CHK(ok,"sdpa"); }
    mlx_array abld=mlx_array_new(), aflat=mlx_array_new();
    if (ok) { const int p[4]={0,2,1,3}; ok = mlx_transpose_axes(&abld, attn, p, 4, s) == 0; }
    if (ok) { int r3[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&aflat, abld, r3, 3, s) == 0; CHK(ok,"reshape attn"); }
    mlx_array ow=mlx_array_new(), os=mlx_array_new(), ob=mlx_array_new(), o_out=mlx_array_new();
    if (ok) { tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof(nb));
              ok = multishard_get_triple(&m->weights, nb, &ow, &os, &ob)
                   && quantized_linear_cfg(&o_out, aflat, ow, os, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"o_proj"); }
    /* post_self_attn_layernorm + residuo */
    mlx_array psa_w=mlx_array_new(), psa=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_self_attn_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &psa_w);
    if (ok) { ok = rms_norm_cfg(&psa, o_out, psa_w, m->cfg.rms_norm_eps, s); CHK(ok,"post_self_attn_norm"); }
    if (ok) { ok = mlx_add(&*out_h, x, psa, s) == 0; CHK(ok,"residual"); }

    mlx_array_free(ln_w); mlx_array_free(h); mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4); mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
    mlx_array_free(q_r); mlx_array_free(k_r); mlx_array_free(attn); mlx_array_free(abld); mlx_array_free(aflat);
    mlx_array_free(ow); mlx_array_free(os); mlx_array_free(ob); mlx_array_free(o_out); mlx_array_free(psa_w); mlx_array_free(psa);
    return ok;
#undef CHK
}

bool glm4_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "glm4.ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array pa_w=mlx_array_new(), normed=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pa_w);
    if (ok) { ok = rms_norm_cfg(&normed, h, pa_w, m->cfg.rms_norm_eps, s); CHK(ok,"post_attn_norm"); }

    /* MLP fuso: gate_up_proj [2*inter] -> split gate/up -> swiglu -> down */
    mlx_array gw=mlx_array_new(), gs=mlx_array_new(), gb=mlx_array_new(), gu=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_up_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &gw, &gs, &gb);
    if (ok) { ok = quantized_linear_cfg(&gu, normed, gw, gs, gb, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"gate_up"); }
    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (ok) { int half = (int)mlx_array_shape(gu)[mlx_array_ndim(gu)-1] / 2;
              int sec[1] = { half }; mlx_vector_array sp=mlx_vector_array_new();
              ok = mlx_split_sections(&sp, gu, sec, 1, -1, s)==0;
              if (ok) ok = mlx_vector_array_get(&gate, sp, 0)==0 && mlx_vector_array_get(&up, sp, 1)==0;
              mlx_vector_array_free(sp); CHK(ok,"split gate_up"); }
    mlx_array sw=mlx_array_new(), dn=mlx_array_new();
    if (ok) { mlx_array sig=mlx_array_new(), silu=mlx_array_new();
              ok = mlx_sigmoid(&sig,gate,s)==0 && mlx_multiply(&silu,gate,sig,s)==0 && mlx_multiply(&sw,silu,up,s)==0;
              mlx_array_free(sig); mlx_array_free(silu); CHK(ok,"swiglu"); }
    mlx_array dw=mlx_array_new(), ds=mlx_array_new(), db=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &dw, &ds, &db);
    if (ok) { ok = quantized_linear_cfg(&dn, sw, dw, ds, db, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"down"); }
    /* post_mlp_layernorm + residuo */
    mlx_array pm_w=mlx_array_new(), pm=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_mlp_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pm_w);
    if (ok) { ok = rms_norm_cfg(&pm, dn, pm_w, m->cfg.rms_norm_eps, s); CHK(ok,"post_mlp_norm"); }
    if (ok) { ok = mlx_add(out, h, pm, s) == 0; CHK(ok,"residual"); }

    mlx_array_free(pa_w); mlx_array_free(normed); mlx_array_free(gw); mlx_array_free(gs); mlx_array_free(gb); mlx_array_free(gu);
    mlx_array_free(gate); mlx_array_free(up); mlx_array_free(sw); mlx_array_free(dn);
    mlx_array_free(dw); mlx_array_free(ds); mlx_array_free(db); mlx_array_free(pm_w); mlx_array_free(pm);
    return ok;
#undef CHK
}
