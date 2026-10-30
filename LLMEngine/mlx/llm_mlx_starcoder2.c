/* ============================================================================
 * llm_mlx_starcoder2.c — dominio StarCoder2 (model_type "starcoder2").
 *
 * Struttura (mlx_lm `starcoder2`):
 *  - attn: q/k/v/o_proj SEPARATI con bias=True; LayerNorm (input_layernorm,
 *    post_attention_layernorm — weight+bias); RoPE standard (traditional=false).
 *  - mlp: c_fc -> gelu (ESATTO, erf) -> c_proj (entrambi bias=True).
 *  - residuo seriale: h = x + attn(ln(x)); out = h + mlp(post_ln(h)).
 * I bias lineari vivono nello slot `biases` della tripletta (fallback .biases->.bias) e il
 * path plain di quantized_linear_cfg li APPLICA -> nessun add esplicito.
 * TRAP: quantized_linear_cfg/llm_mlx_gqa_sdpa ritornano TRUE=ok.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static bool sc2_gelu(mlx_array* out, mlx_array x, mlx_stream s) {
    /* gelu esatto: 0.5*x*(1+erf(x/sqrt(2))). */
    mlx_array e = mlx_array_new(), e1 = mlx_array_new(), h = mlx_array_new(), hp = mlx_array_new();
    mlx_array c = mlx_array_new_float32((float)(1.0/sqrt(2.0))), xc = mlx_array_new();
    bool ok = mlx_astype(&c, c, mlx_array_dtype(x), s)==0;
    if (ok) ok = mlx_multiply(&xc, x, c, s)==0;
    if (ok) ok = mlx_erf(&e, xc, s)==0;
    if (ok) { mlx_array one = mlx_array_new_float32(1.0f); mlx_array onea = mlx_array_new();
              ok = mlx_astype(&onea, one, mlx_array_dtype(x), s)==0 && mlx_add(&e1, e, onea, s)==0;
              mlx_array_free(one); mlx_array_free(onea); }
    if (ok) { mlx_array hf = mlx_array_new_float32(0.5f); mlx_array hfa = mlx_array_new();
              ok = mlx_astype(&hfa, hf, mlx_array_dtype(x), s)==0 && mlx_multiply(&h, hfa, x, s)==0;
              mlx_array_free(hf); mlx_array_free(hfa); }
    if (ok) ok = mlx_multiply(&hp, h, e1, s)==0;
    mlx_array_free(e); mlx_array_free(e1); mlx_array_free(h); mlx_array_free(c); mlx_array_free(xc);
    if (ok) { mlx_array_set(out, hp); }
    mlx_array_free(hp);
    return ok;
}

bool starcoder2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "starcoder2_attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw=mlx_array_new(), lnb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.bias", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnb);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = mlx_fast_layer_norm(&normed, x, lnw, lnb, m->cfg.rms_norm_eps, s)==0; CHK(ok, "input_layernorm"); }

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
    if (ok && qs.ctx != NULL) { ok = llm_mlx_add_linear_bias(m, layer_idx, "self_attn.q_proj", &q, s) && llm_mlx_add_linear_bias(m, layer_idx, "self_attn.k_proj", &k, s) && llm_mlx_add_linear_bias(m, layer_idx, "self_attn.v_proj", &v, s); CHK(ok, "qkv bias"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) { int qs4[4]={1,n_tokens,nh,hd}, kvs4[4]={1,n_tokens,nkv,hd};
              mlx_array qh=mlx_array_new(), kh=mlx_array_new(), vh=mlx_array_new();
              ok = mlx_reshape(&qh, q, qs4,4, s)==0 && mlx_reshape(&kh, k, kvs4,4, s)==0 && mlx_reshape(&vh, v, kvs4,4, s)==0;
              if (ok) { const int perm[4]={0,2,1,3};
                        ok = mlx_transpose_axes(&q_t, qh, perm,4, s)==0 && mlx_transpose_axes(&k_t, kh, perm,4, s)==0 && mlx_transpose_axes(&v_t, vh, perm,4, s)==0; }
              mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh); CHK(ok,"reshape"); }

    mlx_array q_r=mlx_array_new(), k_r=mlx_array_new();
    if (ok) { mlx_optional_float base = { m->cfg.rope_theta, true }; mlx_array nf = mlx_array_new();
              ok = mlx_fast_rope(&q_r, q_t, hd, false, base, 1.0f, cache_offset, nf, s)==0 &&
                   mlx_fast_rope(&k_r, k_t, hd, false, base, 1.0f, cache_offset, nf, s)==0;
              mlx_array_free(nf); CHK(ok,"rope"); }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache) llm_mlx_kv_trim(k_cache, v_cache, cache_offset, s);
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv=mlx_vector_array_new(); mlx_vector_array_append_value(kv,*k_cache); mlx_vector_array_append_value(kv,k_r);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
        } else { ok = mlx_array_set(&k_full, k_r)==0 && mlx_array_set(&v_full, v_t)==0; }
        CHK(ok,"kv");
    }
    if (ok && k_cache && v_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok,"kv store"); }

    mlx_array attn=mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_r, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok,"sdpa"); }

    mlx_array abt=mlx_array_new(), afl=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&abt, attn, perm,4, s)==0; }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&afl, abt, fl,3, s)==0; CHK(ok,"flat"); }

    mlx_array ow=mlx_array_new(), os=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os,&ob);
    mlx_array ao=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&ao, afl, ow,os,ob, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"o_proj"); }
    if (ok && os.ctx != NULL) { ok = llm_mlx_add_linear_bias(m, layer_idx, "self_attn.o_proj", &ao, s); CHK(ok, "o bias"); }
    if (ok) { ok = mlx_add(out_h, x, ao, s)==0; CHK(ok,"residual (attn)"); }

    mlx_array all[] = { lnw,lnb,normed, qw,qs,qb, kw,ks,kb, vw,vs,vb, q,k,v, q_t,k_t,v_t, q_r,k_r,
                        k_full,v_full, attn, abt,afl, ow,os,ob, ao };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool starcoder2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "starcoder2_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw=mlx_array_new(), lnb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &lnw);
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.bias", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &lnb);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = mlx_fast_layer_norm(&normed, h, lnw, lnb, m->cfg.rms_norm_eps, s)==0; CHK(ok,"post_attention_layernorm"); }

    mlx_array fw=mlx_array_new(), fs=mlx_array_new(), fb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.c_fc", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &fw,&fs,&fb);
    mlx_array fc = mlx_array_new(), g = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&fc, normed, fw,fs,fb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"c_fc"); }
    if (ok && fs.ctx != NULL) { ok = llm_mlx_add_linear_bias(m, layer_idx, "mlp.c_fc", &fc, s); CHK(ok, "c_fc bias"); }
    if (ok) { ok = sc2_gelu(&g, fc, s); CHK(ok,"gelu"); }
    mlx_array pw=mlx_array_new(), ps=mlx_array_new(), pb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.c_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &pw,&ps,&pb);
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, g, pw,ps,pb, gsz,bts, m->cfg.quant_mode, s); CHK(ok,"c_proj"); }
    if (ok && ps.ctx != NULL) { ok = llm_mlx_add_linear_bias(m, layer_idx, "mlp.c_proj", &dn, s); CHK(ok, "c_proj bias"); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok,"residual (ffn)"); }

    mlx_array all[] = { lnw,lnb,normed, fw,fs,fb, fc, g, pw,ps,pb, dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
