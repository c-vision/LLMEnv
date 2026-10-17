/* ============================================================================
 * llm_mlx_granite_hybrid.c — dominio Granite-MoE-Hybrid (IBM), variante DENSE
 * (granite-4.0-h-micro, num_local_experts == 0). Hybrid a due tipi di layer.
 *
 * Modellato 1:1 su mlx_lm/models/granitemoehybrid.py (+ ssm.py).
 *   layer:  residual = x
 *           hidden   = input_layernorm(x)
 *           hidden   = mamba(hidden) | self_attn(hidden)          [type da layer_types]
 *           hidden   = residual + hidden * residual_multiplier
 *           residual = hidden
 *           normed   = post_attention_layernorm(hidden)
 *           hidden   = residual + mlp(normed) * residual_multiplier   [mlp = dense SwiGLU]
 * Embedding: embed * embedding_multiplier (via cfg.embed_scale). Logits: / logits_scaling
 * (via cfg.falcon_h1_out_scale = 1/logits_scaling). Attention: GQA SENZA RoPE
 * (position_embedding_type "nope"), scale = attention_multiplier.
 *
 * Mixer mamba: Mamba2 SSD (in_proj -> [gate|conv_in|dt]; conv1d depthwise; SSD; gated
 * RMSNorm; out_proj). Naming "mamba.*". Stato: k_cache = ssm [1,H,Dh,Ds]; v_cache = conv state.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool granite_hybrid_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                       mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool granite_hybrid_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* attn GQA senza RoPE, ritorna o_proj (NESSUN residuo/norm). */
static bool granite_attn(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                         mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const float scale = m->cfg.granite.attn_multiplier > 0.0f ? m->cfg.granite.attn_multiplier : 1.0f / sqrtf((float)hd);
    char nb[256];
    mlx_stream st = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "granite.attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)
    mlx_array qw=mlx_array_new(),qs=mlx_array_new(),qb=mlx_array_new();
    mlx_array kw=mlx_array_new(),ks=mlx_array_new(),kb=mlx_array_new();
    mlx_array vw=mlx_array_new(),vs=mlx_array_new(),vb=mlx_array_new();
    mlx_array ow=mlx_array_new(),os=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &kw, &ks, &kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &vw, &vs, &vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow, &os, &ob);
    CHK(ok, "proj lookup");
    mlx_array ql=mlx_array_new(), kl=mlx_array_new(), vl=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&ql, x, qw, qs, qb, gsz, bts, m->cfg.quant_mode, st);
    if (ok) ok = quantized_linear_cfg(&kl, x, kw, ks, kb, gsz, bts, m->cfg.quant_mode, st);
    if (ok) ok = quantized_linear_cfg(&vl, x, vw, vs, vb, gsz, bts, m->cfg.quant_mode, st);
    CHK(ok, "qkv");
    mlx_array q4=mlx_array_new(),k4=mlx_array_new(),v4=mlx_array_new();
    if (ok) { int qs4[4]={1,n_tokens,nh,hd}, ks4[4]={1,n_tokens,nkv,hd};
              ok = mlx_reshape(&q4,ql,qs4,4,st)==0 && mlx_reshape(&k4,kl,ks4,4,st)==0 && mlx_reshape(&v4,vl,ks4,4,st)==0; CHK(ok,"reshape"); }
    mlx_array q_t=mlx_array_new(),k_t=mlx_array_new(),v_t=mlx_array_new();
    if (ok) { const int p[4]={0,2,1,3}; ok = mlx_transpose_axes(&q_t,q4,p,4,st)==0 && mlx_transpose_axes(&k_t,k4,p,4,st)==0 && mlx_transpose_axes(&v_t,v4,p,4,st)==0; CHK(ok,"transpose"); }
    if (ok && !m->cfg.granite.attn_no_rope) {   /* granite (dense) usa RoPE; granitemoehybrid "nope" no */
        mlx_optional_float base = { m->cfg.rope_theta, true };
        mlx_array nf = mlx_array_new();
        mlx_array qr = mlx_array_new(), kr = mlx_array_new();
        ok = mlx_fast_rope(&qr, q_t, hd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, st)==0 &&
             mlx_fast_rope(&kr, k_t, hd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, st)==0;
        mlx_array_free(nf);
        if (ok) { mlx_array_set(&q_t, qr); mlx_array_set(&k_t, kr); }
        mlx_array_free(qr); mlx_array_free(kr);
        CHK(ok, "rope");
    }
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache)==4 && mlx_array_size(*k_cache)>0) {
            mlx_array kn=mlx_array_new(), vn=mlx_array_new();
            mlx_vector_array kv=mlx_vector_array_new(); mlx_vector_array_append_value(kv,*k_cache); mlx_vector_array_append_value(kv,k_t);
            ok = mlx_concatenate_axis(&kn,kv,2,st)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,v_t);
            ok = ok && mlx_concatenate_axis(&vn,vv,2,st)==0; mlx_vector_array_free(vv);
            if (ok) { mlx_array_set(k_cache,kn); mlx_array_set(v_cache,vn); }
            mlx_array_free(kn); mlx_array_free(vn);
        } else { mlx_array_set(k_cache,k_t); mlx_array_set(v_cache,v_t); }
        CHK(ok, "kv");
    }
    mlx_array attn=mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_t, *k_cache, *v_cache, scale, 0, st); CHK(ok,"sdpa"); }
    mlx_array abld=mlx_array_new(), aflat=mlx_array_new();
    if (ok) { const int p[4]={0,2,1,3}; ok = mlx_transpose_axes(&abld, attn, p, 4, st) == 0; }
    if (ok) { int r3[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&aflat, abld, r3, 3, st) == 0; CHK(ok,"reshape attn"); }
    if (ok) { ok = quantized_linear_cfg(out, aflat, ow, os, ob, gsz, bts, m->cfg.quant_mode, st); CHK(ok,"o_proj"); }
    mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);mlx_array_free(kw);mlx_array_free(ks);mlx_array_free(kb);
    mlx_array_free(vw);mlx_array_free(vs);mlx_array_free(vb);mlx_array_free(ow);mlx_array_free(os);mlx_array_free(ob);
    mlx_array_free(ql);mlx_array_free(kl);mlx_array_free(vl);mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v4);
    mlx_array_free(q_t);mlx_array_free(k_t);mlx_array_free(v_t);mlx_array_free(attn);mlx_array_free(abld);mlx_array_free(aflat);
    return ok;
#undef CHK
}

/* Mamba2 SSD mixer (naming "mamba.*"). x gia' normato. Ritorna out_proj(...) in `mix_out`. */
static bool granite_mamba2(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array x,
                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* mix_out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int H = m->cfg.granite.ssm_n_heads > 0 ? m->cfg.granite.ssm_n_heads : 1;
    const int Dh = m->cfg.granite.ssm_head_dim > 0 ? m->cfg.granite.ssm_head_dim : 1;
    const int Inner = H * Dh;
    const int G = m->cfg.n_group > 0 ? m->cfg.n_group : 1;
    const int Ds = m->cfg.mamba_state_size > 0 ? m->cfg.mamba_state_size : 16;
    const int Kc = m->cfg.linear_conv_kernel_dim > 0 ? m->cfg.linear_conv_kernel_dim : 4;
    const int Hpg = H / G;
    const int ConvDim = Inner + 2 * G * Ds;
    const float ts_lo = m->cfg.mamba_ts_lo, ts_hi = m->cfg.mamba_ts_hi;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "granite.mamba[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)
    mlx_array h = mlx_array_new(); mlx_array_set(&h, x);   /* input gia' normato */

    mlx_array ipw=mlx_array_new(),ips=mlx_array_new(),ipb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mamba.in_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ipw, &ips, &ipb);
    mlx_array proj=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&proj, h, ipw, ips, ipb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj"); }
    mlx_array gate=mlx_array_new(), conv_in=mlx_array_new(), dt=mlx_array_new();
    if (ok) { int sec[2]={Inner, Inner+ConvDim}; mlx_vector_array v=mlx_vector_array_new();
        ok = mlx_split_sections(&v, proj, sec, 2, -1, s)==0;
        if (ok) ok = mlx_vector_array_get(&gate,v,0)==0 && mlx_vector_array_get(&conv_in,v,1)==0 && mlx_vector_array_get(&dt,v,2)==0;
        mlx_vector_array_free(v); CHK(ok, "split proj"); }
    mlx_array cw=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mamba.conv1d.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cw);
    { mlx_array c32=mlx_array_new(); if (mlx_astype(&c32, cw, MLX_FLOAT32, s)==0) { mlx_array_free(cw); cw=c32; } else mlx_array_free(c32); }
    if (mlx_array_ndim(cw)==3 && mlx_array_shape(cw)[1]==1) { int ax[3]={0,2,1};
        mlx_array cwt=mlx_array_new(); if (mlx_transpose_axes(&cwt,cw,ax,3,s)==0) { mlx_array_free(cw); cw=cwt; } else mlx_array_free(cwt); }
    mlx_array cbias=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mamba.conv1d.bias", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cbias);
    mlx_array cstate=mlx_array_new();
    if (ok) { if (has_cache && mlx_array_size(*v_cache)>0) mlx_array_set(&cstate, *v_cache);
        else { int cs[3]={1,Kc-1,ConvDim}; mlx_zeros(&cstate, cs, 3, MLX_FLOAT32, s); } }
    mlx_array cf=mlx_array_new();
    if (ok) { mlx_array arr[2]={cstate,conv_in}; mlx_vector_array v=mlx_vector_array_new_data(arr,2);
        ok = mlx_concatenate_axis(&cf, v, 1, s)==0; mlx_vector_array_free(v); CHK(ok,"conv concat"); }
    if (ok) { int ss[3]={0,n_tokens,0}, se[3]={1,n_tokens+(Kc-1),ConvDim}, stp[3]={1,1,1};
        mlx_array sl=mlx_array_new(); ok = mlx_slice(&sl, cf, ss,3,se,3,stp,3,s)==0; if (ok) mlx_array_set(v_cache, sl); mlx_array_free(sl); }
    mlx_array cout=mlx_array_new();
    if (ok) { int str[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, id[1]={1};
        ok = mlx_conv_general(&cout, cf, cw, str,1,pl,1,ph,1,kd,1,id,1, ConvDim, false, s)==0; CHK(ok,"conv1d"); }
    if (ok && cbias.ctx!=NULL && mlx_array_size(cbias)>0) { mlx_array c2=mlx_array_new(); ok = mlx_add(&c2, cout, cbias, s)==0; if (ok) mlx_array_set(&cout,c2); mlx_array_free(c2); }
    mlx_array xc=mlx_array_new();
    if (ok) { mlx_array sig=mlx_array_new(); ok = mlx_sigmoid(&sig, cout, s)==0 && mlx_multiply(&xc, cout, sig, s)==0; mlx_array_free(sig); CHK(ok,"silu"); }
    mlx_array hidden=mlx_array_new(), Bv=mlx_array_new(), Cv=mlx_array_new();
    if (ok) { int sec[2]={Inner, Inner+G*Ds}; mlx_vector_array v=mlx_vector_array_new();
        ok = mlx_split_sections(&v, xc, sec, 2, -1, s)==0;
        if (ok) ok = mlx_vector_array_get(&hidden,v,0)==0 && mlx_vector_array_get(&Bv,v,1)==0 && mlx_vector_array_get(&Cv,v,2)==0;
        mlx_vector_array_free(v); CHK(ok,"split conv"); }
    mlx_array Alog=mlx_array_new(), Dv=mlx_array_new(), dtb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mamba.A_log", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &Alog);
    tname(m->cfg.layer_prefix, layer_idx, "mamba.D", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &Dv);
    tname(m->cfg.layer_prefix, layer_idx, "mamba.dt_bias", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &dtb);
    mlx_array Aneg=mlx_array_new();
    if (ok) { mlx_array ea=mlx_array_new(); ok = mlx_exp(&ea,Alog,s)==0 && mlx_negative(&Aneg,ea,s)==0; mlx_array_free(ea); }
    mlx_array hd=mlx_array_new(), bg=mlx_array_new(), cg=mlx_array_new();
    if (ok) { int sh[4]={1,n_tokens,H,Dh}; ok = mlx_reshape(&hd,hidden,sh,4,s)==0; }
    if (ok) { int sh[4]={1,n_tokens,G,Ds}; ok = mlx_reshape(&bg,Bv,sh,4,s)==0; int sc[4]={1,n_tokens,G,Ds}; ok = ok && mlx_reshape(&cg,Cv,sc,4,s)==0; }
    mlx_array Bt=mlx_array_new(), Ct=mlx_array_new();
    if (ok) { int reps[4]={1,1,Hpg,1}; ok = mlx_tile(&Bt,bg,reps,4,s)==0 && mlx_tile(&Ct,cg,reps,4,s)==0; }
    mlx_array state=mlx_array_new();
    if (ok) { if (has_cache && mlx_array_size(*k_cache)>0) mlx_array_set(&state, *k_cache);
        else { int st4[4]={1,H,Dh,Ds}; mlx_zeros(&state, st4, 4, MLX_FLOAT32, s); } }
    mlx_array ystack=mlx_array_new(); bool yset=false;
    for (int t=0; t<n_tokens && ok; t++) {
        mlx_array xt=mlx_array_new(), Btt=mlx_array_new(), Ctt=mlx_array_new(), dtt=mlx_array_new();
        { int ss[4]={0,t,0,0}, se[4]={1,t+1,H,Dh}, stp[4]={1,1,1,1}; ok=mlx_slice(&xt,hd,ss,4,se,4,stp,4,s)==0; if(ok) ok=mlx_squeeze_axis(&xt,xt,1,s)==0; }
        { int ss[4]={0,t,0,0}, se[4]={1,t+1,H,Ds}, stp[4]={1,1,1,1}; ok=ok&&mlx_slice(&Btt,Bt,ss,4,se,4,stp,4,s)==0&&mlx_slice(&Ctt,Ct,ss,4,se,4,stp,4,s)==0; if(ok) ok=mlx_squeeze_axis(&Btt,Btt,1,s)==0&&mlx_squeeze_axis(&Ctt,Ctt,1,s)==0; }
        { int ss[3]={0,t,0}, se[3]={1,t+1,H}, stp[3]={1,1,1}; ok=ok&&mlx_slice(&dtt,dt,ss,3,se,3,stp,3,s)==0; if(ok) ok=mlx_squeeze_axis(&dtt,dtt,0,s)==0; }
        mlx_array dsum=mlx_array_new(), dsp=mlx_array_new(), dcl=mlx_array_new(), db3=mlx_array_new();
        { int sh[2]={1,H}; ok=ok&&mlx_reshape(&db3,dtb,sh,2,s)==0; }
        ok=ok&&mlx_add(&dsum,dtt,db3,s)==0;
        { mlx_array zero=mlx_array_new(); int zs[1]={1}; mlx_zeros(&zero,zs,1,MLX_FLOAT32,s); ok=ok&&mlx_logaddexp(&dsp,dsum,zero,s)==0; mlx_array_free(zero); }
        { mlx_array lo=mlx_array_new_float32(ts_lo), hi=mlx_array_new_float32(ts_hi); ok=ok&&mlx_clip(&dcl,dsp,lo,hi,s)==0; mlx_array_free(lo); mlx_array_free(hi); }
        mlx_array_free(db3);
        mlx_array a3=mlx_array_new(); { int sh[2]={1,H}; ok=ok&&mlx_reshape(&a3,Aneg,sh,2,s)==0; }
        mlx_array dAm=mlx_array_new(), dAe=mlx_array_new();
        ok=ok&&mlx_multiply(&dAm,a3,dcl,s)==0&&mlx_exp(&dAe,dAm,s)==0;
        mlx_array xde=mlx_array_new(), Bde=mlx_array_new(), outer=mlx_array_new(), outerd=mlx_array_new();
        ok=ok&&mlx_expand_dims(&xde,xt,-1,s)==0&&mlx_expand_dims(&Bde,Btt,-1,s)==0;
        { mlx_array Bde2=mlx_array_new(); ok=ok&&mlx_swapaxes(&Bde2,Bde,2,3,s)==0; ok=ok&&mlx_multiply(&outer,xde,Bde2,s)==0; mlx_array_free(Bde2); }
        mlx_array dA4=mlx_array_new(), dcl4=mlx_array_new();
        { mlx_array t=mlx_array_new(); ok=ok&&mlx_expand_dims(&t,dAe,-1,s)==0&&mlx_expand_dims(&dA4,t,-1,s)==0; mlx_array_free(t); }
        { mlx_array t=mlx_array_new(); ok=ok&&mlx_expand_dims(&t,dcl,-1,s)==0&&mlx_expand_dims(&dcl4,t,-1,s)==0; mlx_array_free(t); }
        { mlx_array stA=mlx_array_new(); ok=ok&&mlx_multiply(&stA,state,dA4,s)==0; ok=ok&&mlx_multiply(&outerd,outer,dcl4,s)==0;
          mlx_array nst=mlx_array_new(); ok=ok&&mlx_add(&nst,stA,outerd,s)==0; if(ok) mlx_array_set(&state,nst); mlx_array_free(stA); mlx_array_free(nst); }
        mlx_array Cde=mlx_array_new(); ok=ok&&mlx_expand_dims(&Cde,Ctt,-1,s)==0;
        mlx_array yp=mlx_array_new(), ys=mlx_array_new();
        ok=ok&&mlx_matmul(&yp,state,Cde,s)==0&&mlx_squeeze_axis(&ys,yp,3,s)==0;
        mlx_array D3=mlx_array_new(); { int sh[2]={1,H}; ok=ok&&mlx_reshape(&D3,Dv,sh,2,s)==0; mlx_array D3b=mlx_array_new(); ok=ok&&mlx_expand_dims(&D3b,D3,-1,s)==0;
          mlx_array dx=mlx_array_new(); ok=ok&&mlx_multiply(&dx,xt,D3b,s)==0; mlx_array yt=mlx_array_new(); ok=ok&&mlx_add(&yt,ys,dx,s)==0;
          if (!yset) { mlx_array ye=mlx_array_new(); ok=ok&&mlx_expand_dims(&ye,yt,1,s)==0; if(ok){mlx_array_set(&ystack,ye);yset=true;} mlx_array_free(ye); }
          else { mlx_array cc=mlx_array_new(); mlx_array yte=mlx_array_new(); ok=ok&&mlx_expand_dims(&yte,yt,1,s)==0; mlx_array arr[2]={ystack,yte}; mlx_vector_array vv=mlx_vector_array_new_data(arr,2);
                 ok=ok&&mlx_concatenate_axis(&cc,vv,1,s)==0; mlx_vector_array_free(vv); if(ok) mlx_array_set(&ystack,cc); mlx_array_free(cc); mlx_array_free(yte); }
          mlx_array_free(dx); mlx_array_free(yt); mlx_array_free(D3b); }
        mlx_array_free(D3); mlx_array_free(Cde); mlx_array_free(yp); mlx_array_free(ys);
        mlx_array_free(dA4); mlx_array_free(dcl4); mlx_array_free(outer); mlx_array_free(outerd);
        mlx_array_free(xde); mlx_array_free(Bde); mlx_array_free(dAm); mlx_array_free(dAe); mlx_array_free(a3);
        mlx_array_free(dsum); mlx_array_free(dsp); mlx_array_free(dcl);
        mlx_array_free(xt); mlx_array_free(Btt); mlx_array_free(Ctt); mlx_array_free(dtt);
    }
    if (ok) mlx_array_set(k_cache, state);
    mlx_array gn=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mamba.norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &gn);
    { mlx_array g32=mlx_array_new(); if (mlx_astype(&g32,gn,MLX_FLOAT32,s)==0) { mlx_array_free(gn); gn=g32; } else mlx_array_free(g32); }
    mlx_array ow=mlx_array_new(), ows=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mamba.out_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &ows, &ob);
    mlx_array yflat=mlx_array_new();
    if (ok) { int sh[3]={n_tokens,H,Dh}; mlx_array ysh=mlx_array_new(); ok=mlx_reshape(&ysh,ystack,sh,3,s)==0; CHK(ok,"reshape y"); int sh2[2]={n_tokens,Inner}; ok=ok&&mlx_reshape(&yflat,ysh,sh2,2,s)==0; mlx_array_free(ysh); }
    mlx_array sw=mlx_array_new();
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok=mlx_sigmoid(&sg,gate,s)==0&&mlx_multiply(&sl,gate,sg,s)==0&&mlx_multiply(&sw,yflat,sl,s)==0; mlx_array_free(sg); mlx_array_free(sl); }
    if (ok) { mlx_array nn=mlx_array_new(); ok=mlx_fast_rms_norm(&nn, sw, gn, m->cfg.rms_norm_eps, s)==0; if(ok) mlx_array_set(&sw,nn); mlx_array_free(nn); CHK(ok,"rms gated"); }
    if (ok) { ok = quantized_linear_cfg(mix_out, sw, ow, ows, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"out_proj"); }
    mlx_array_free(h); mlx_array_free(ipw); mlx_array_free(ips); mlx_array_free(ipb); mlx_array_free(proj);
    mlx_array_free(gate); mlx_array_free(conv_in); mlx_array_free(dt); mlx_array_free(cw); mlx_array_free(cbias);
    mlx_array_free(cstate); mlx_array_free(cf); mlx_array_free(cout); mlx_array_free(xc); mlx_array_free(hidden);
    mlx_array_free(Bv); mlx_array_free(Cv); mlx_array_free(Alog); mlx_array_free(Aneg); mlx_array_free(Dv); mlx_array_free(dtb);
    mlx_array_free(hd); mlx_array_free(bg); mlx_array_free(cg); mlx_array_free(Bt); mlx_array_free(Ct); mlx_array_free(state);
    mlx_array_free(ystack); mlx_array_free(gn); mlx_array_free(yflat); mlx_array_free(sw); mlx_array_free(ow); mlx_array_free(ows); mlx_array_free(ob);
    return ok;
#undef CHK
}

bool granite_hybrid_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                       mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true, lin = m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx];
    const float rm = m->cfg.granite.residual_multiplier != 0.0f ? m->cfg.granite.residual_multiplier : 1.0f;
    char nb[256]; mlx_stream s = m->stream;
    mlx_array lnw=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array xl=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&xl, x, lnw, m->cfg.rms_norm_eps, s);
    mlx_array mixout=mlx_array_new();
    if (ok) {
        if (lin) ok = granite_mamba2(m, layer_idx, n_tokens, xl, k_cache, v_cache, has_cache, &mixout);
        else ok = granite_attn(m, layer_idx, cache_offset, n_tokens, xl, k_cache, v_cache, has_cache, &mixout);
    }
    if (ok) { mlx_array sc=mlx_array_new_float32(rm), mo=mlx_array_new(); ok = mlx_multiply(&mo, mixout, sc, s)==0;
              if (ok) ok = mlx_add(out_h, x, mo, s)==0; mlx_array_free(sc); mlx_array_free(mo); }
    mlx_array_free(lnw); mlx_array_free(xl); mlx_array_free(mixout);
    return ok;
}

bool granite_hybrid_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const float rm = m->cfg.granite.residual_multiplier != 0.0f ? m->cfg.granite.residual_multiplier : 1.0f;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "granite.ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)
    mlx_array lnw=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed=mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "ffn_norm"); }
    mlx_array gw=mlx_array_new(),gs=mlx_array_new(),gb=mlx_array_new();
    mlx_array uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new();
    mlx_array dw=mlx_array_new(),ds=mlx_array_new(),db=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &gw, &gs, &gb);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.up_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &uw, &us, &ub);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &dw, &ds, &db);
    CHK(ok, "mlp lookup");
    mlx_array g=mlx_array_new(), u=mlx_array_new(), sw=mlx_array_new(), dn=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&g, normed, gw, gs, gb, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"gate"); }
    if (ok) { ok = quantized_linear_cfg(&u, normed, uw, us, ub, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"up"); }
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok = mlx_sigmoid(&sg,g,s)==0 && mlx_multiply(&sl,g,sg,s)==0 && mlx_multiply(&sw,sl,u,s)==0; mlx_array_free(sg); mlx_array_free(sl); }
    if (ok) { ok = quantized_linear_cfg(&dn, sw, dw, ds, db, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"down"); }
    if (ok) { mlx_array sc=mlx_array_new_float32(rm), dns=mlx_array_new(); ok = mlx_multiply(&dns, dn, sc, s)==0;
              if (ok) ok = mlx_add(out, h, dns, s)==0; mlx_array_free(sc); mlx_array_free(dns); }
    CHK(ok, "mlp");
    mlx_array_free(lnw); mlx_array_free(normed); mlx_array_free(gw);mlx_array_free(gs);mlx_array_free(gb);
    mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);mlx_array_free(dw);mlx_array_free(ds);mlx_array_free(db);
    mlx_array_free(g);mlx_array_free(u);mlx_array_free(sw);mlx_array_free(dn);
    return ok;
#undef CHK
}
