/* ============================================================================
 * llm_mlx_jamba.c — dominio Jamba (ibrido attenzione + Mamba S6, denso).
 *   blocco:  h = input_layernorm(x); h = attn(h) OPPURE mamba(h); out = x + h
 *   ffn:     hn = pre_ff_layernorm(h); swiglu(gate,up) -> down; out = h + d
 *   attn = GQA senza RoPE; mamba = S6 (conv1d + scan per-token, RMSNorm su dt/B/C).
 * Modellato 1:1 su mlx_lm/models/jamba.py.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool jamba_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                      mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool jamba_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* ---- mixer Mamba S6 (conv + scan, coda out_proj(swiglu)) ---- */
static bool jamba_mixer(LlmMlxModel* m, int li, int n_tokens, mlx_array x,
                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int Inner = 2 * m->cfg.hidden_size;
    const int St = m->cfg.mamba.state_size > 0 ? m->cfg.mamba.state_size : 16;
    const int Kc = m->cfg.linear_conv_kernel_dim > 0 ? m->cfg.linear_conv_kernel_dim : 4;
    const int Rk = m->cfg.mamba.dt_rank > 0 ? m->cfg.mamba.dt_rank : (m->cfg.hidden_size + 15) / 16;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "jamba[layer=%d]: %s failed\n", li, (w)); ok = false; } } while (0)

    /* in_proj -> x, z */
    mlx_array ipw = mlx_array_new(), ips = mlx_array_new(), ipb = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.in_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ipw, &ips, &ipb);
    mlx_array xz = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&xz, x, ipw, ips, ipb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj"); }
    mlx_array xq = mlx_array_new(), zz = mlx_array_new();
    if (ok) { int sec[1] = { Inner }; mlx_vector_array v = mlx_vector_array_new();
        ok = mlx_split_sections(&v, xz, sec, 1, -1, s) == 0;
        if (ok) ok = mlx_vector_array_get(&xq, v, 0) == 0 && mlx_vector_array_get(&zz, v, 1) == 0;
        mlx_vector_array_free(v); CHK(ok, "split xz"); }

    /* conv1d depthwise con cache [1,Kc-1,Inner] (v_cache) */
    mlx_array cw = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.conv1d.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cw);
    { mlx_array c32 = mlx_array_new(); if (mlx_astype(&c32, cw, MLX_FLOAT32, s) == 0) { mlx_array_free(cw); cw = c32; } else mlx_array_free(c32); }
    if (mlx_array_ndim(cw) == 3 && mlx_array_shape(cw)[1] == 1) {
        mlx_array cwt = mlx_array_new(); int ax[3] = {0, 2, 1};
        if (mlx_transpose_axes(&cwt, cw, ax, 3, s) == 0) { mlx_array_free(cw); cw = cwt; } else mlx_array_free(cwt);
    }
    mlx_array cb = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.conv1d.bias", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cb);
    mlx_array cstate = mlx_array_new();
    if (ok) {
        if (has_cache && mlx_array_size(*v_cache) > 0) mlx_array_set(&cstate, *v_cache);
        else { int cs[3] = { 1, Kc - 1, Inner }; mlx_zeros(&cstate, cs, 3, MLX_FLOAT32, s); }
    }
    mlx_array cf = mlx_array_new();
    if (ok) { mlx_array arr[2] = { cstate, xq }; mlx_vector_array v = mlx_vector_array_new_data(arr, 2);
        ok = mlx_concatenate_axis(&cf, v, 1, s) == 0; mlx_vector_array_free(v); CHK(ok, "conv concat"); }
    if (ok) { int ss[3] = {0, n_tokens, 0}, se[3] = {1, n_tokens + (Kc - 1), Inner}, st[3] = {1,1,1};
        mlx_array sl = mlx_array_new(); ok = mlx_slice(&sl, cf, ss, 3, se, 3, st, 3, s) == 0; if (ok) mlx_array_set(v_cache, sl); mlx_array_free(sl); }
    mlx_array cout = mlx_array_new();
    if (ok) { int str[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, id[1]={1};
        ok = mlx_conv_general(&cout, cf, cw, str,1,pl,1,ph,1,kd,1,id,1, Inner, false, s) == 0; CHK(ok, "conv1d"); }
    if (ok && cb.ctx != NULL && mlx_array_size(cb) > 0) { mlx_array c2 = mlx_array_new(); ok = mlx_add(&c2, cout, cb, s) == 0; if (ok) mlx_array_set(&cout, c2); mlx_array_free(c2); CHK(ok, "conv+bias"); }
    mlx_array xc = mlx_array_new();
    if (ok) { mlx_array sig = mlx_array_new(); ok = mlx_sigmoid(&sig, cout, s) == 0 && mlx_multiply(&xc, cout, sig, s) == 0; mlx_array_free(sig); CHK(ok, "silu"); }

    /* x_proj ; dt_proj (+bias lineare) ; A_log ; D */
    mlx_array xpw = mlx_array_new(), xps = mlx_array_new(), xpb = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.x_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &xpw, &xps, &xpb);
    mlx_array dtw = mlx_array_new(), dtsc = mlx_array_new(), dtb = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.dt_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &dtw, &dtsc, &dtb);
    mlx_array dtblin = mlx_array_new();
    { char lb[256]; tname(m->cfg.layer_prefix, li, "mamba.dt_proj.bias", lb, sizeof(lb));
      multishard_get(&m->weights, lb, &dtblin); }
    mlx_array dtlw = mlx_array_new(), blw = mlx_array_new(), clw = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.dt_layernorm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &dtlw);
    tname(m->cfg.layer_prefix, li, "mamba.b_layernorm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &blw);
    tname(m->cfg.layer_prefix, li, "mamba.c_layernorm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &clw);
    /* A_log / D: in jamba.py sono COSTANTI calcolate (A_log=log(arange(1,St+1)); D=ones),
     * non parametri -> potrebbero non essere nel checkpoint. Si caricano se presenti,
     * altrimenti si calcolano (equivalente). */
    mlx_array Alog = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.A_log", nb, sizeof(nb));
    multishard_get(&m->weights, nb, &Alog);
    if (ok && Alog.ctx == NULL) {
        mlx_array ar=mlx_array_new(), ar2=mlx_array_new(), lg=mlx_array_new(), z=mlx_array_new();
        int s1[1]={St}, s2[2]={1,St}, zsh[2]={Inner,1};
        ok = mlx_arange(&ar,1.0,(double)St+1.0,1.0,MLX_FLOAT32,s)==0
             && mlx_reshape(&ar2,ar,s2,2,s)==0 && mlx_log(&lg,ar2,s)==0
             && mlx_zeros(&z,zsh,2,MLX_FLOAT32,s)==0 && mlx_add(&Alog,lg,z,s)==0;
        (void)s1; mlx_array_free(ar); mlx_array_free(ar2); mlx_array_free(lg); mlx_array_free(z);
        CHK(ok,"A_log compute");
    }
    mlx_array Dv = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.D", nb, sizeof(nb));
    multishard_get(&m->weights, nb, &Dv);
    if (ok && Dv.ctx == NULL) { int dsh[1]={Inner}; ok = mlx_ones(&Dv,dsh,1,MLX_FLOAT32,s)==0; CHK(ok,"D compute"); }
    mlx_array Aneg = mlx_array_new();
    if (ok) { mlx_array ea = mlx_array_new(); ok = mlx_exp(&ea, Alog, s) == 0 && mlx_negative(&Aneg, ea, s) == 0; mlx_array_free(ea); }

    /* state [1,Inner,St] in k_cache */
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && mlx_array_size(*k_cache) > 0) mlx_array_set(&state, *k_cache);
        else { int st3[3] = { 1, Inner, St }; mlx_zeros(&state, st3, 3, MLX_FLOAT32, s); }
    }
    mlx_array ystack = mlx_array_new(); bool yset = false;
    for (int t = 0; t < n_tokens && ok; t++) {
        mlx_array xt = mlx_array_new();
        { int ss[3]={0,t,0}, se[3]={1,t+1,Inner}, st[3]={1,1,1};
          ok = mlx_slice(&xt, xc, ss,3,se,3,st,3,s) == 0; if (ok) ok = mlx_squeeze_axis(&xt, xt, 1, s) == 0; }
        mlx_array dbc = mlx_array_new();
        if (ok) ok = quantized_linear_cfg(&dbc, xt, xpw, xps, xpb, gsz, bts, m->cfg.quant_mode, s);
        mlx_array dlt = mlx_array_new(), Bv = mlx_array_new(), Cv = mlx_array_new();
        if (ok) { int sec[2] = { Rk, Rk + St }; mlx_vector_array v = mlx_vector_array_new();
            ok = mlx_split_sections(&v, dbc, sec, 2, -1, s) == 0;
            if (ok) ok = mlx_vector_array_get(&dlt,v,0)==0 && mlx_vector_array_get(&Bv,v,1)==0 && mlx_vector_array_get(&Cv,v,2)==0;
            mlx_vector_array_free(v); }
        /* RMSNorm su delta/B/C (dt_layernorm / b_layernorm / c_layernorm) */
        mlx_array dltn = mlx_array_new(), Bvn = mlx_array_new(), Cvn = mlx_array_new();
        if (ok) ok = mlx_fast_rms_norm(&dltn, dlt, dtlw, m->cfg.rms_norm_eps, s) == 0
                   && mlx_fast_rms_norm(&Bvn, Bv, blw, m->cfg.rms_norm_eps, s) == 0
                   && mlx_fast_rms_norm(&Cvn, Cv, clw, m->cfg.rms_norm_eps, s) == 0;
        if (ok) mlx_array_set(&dlt, dltn);
        if (ok) mlx_array_set(&Bv, Bvn);
        if (ok) mlx_array_set(&Cv, Cvn);
        /* delta = softplus(dt_proj(delta)) */
        mlx_array dlin = mlx_array_new();
        if (ok) ok = quantized_linear_cfg(&dlin, dlt, dtw, dtsc, dtb, gsz, bts, m->cfg.quant_mode, s);
        if (ok && dtblin.ctx != NULL && mlx_array_size(dtblin) > 0) { mlx_array dl2 = mlx_array_new(); ok = mlx_add(&dl2, dlin, dtblin, s) == 0; if (ok) mlx_array_set(&dlin, dl2); mlx_array_free(dl2); }
        if (ok) { int zs[1] = {1}; mlx_array zero = mlx_array_new(); mlx_zeros(&zero, zs, 1, MLX_FLOAT32, s);
            mlx_array sp = mlx_array_new(); ok = mlx_logaddexp(&sp, dlin, zero, s) == 0; if (ok) mlx_array_set(&dlt, sp);
            mlx_array_free(zero); mlx_array_free(sp); }
        /* scan S6: new_state = (delta*x)(x)B + state*exp(delta*A) ; y = new_state@C + D*x */
        mlx_array de = mlx_array_new(), xe = mlx_array_new(), Be = mlx_array_new(), Ce = mlx_array_new();
        mlx_expand_dims(&de, dlt, -1, s); mlx_expand_dims(&xe, xt, -1, s); mlx_expand_dims(&Be, Bv, 1, s); mlx_expand_dims(&Ce, Cv, -1, s);
        mlx_array ns1 = mlx_array_new(); mlx_multiply(&ns1, de, xe, s);
        mlx_array ns = mlx_array_new(); mlx_multiply(&ns, ns1, Be, s);
        mlx_array dA = mlx_array_new(); mlx_multiply(&dA, de, Aneg, s);
        mlx_array eA = mlx_array_new(); mlx_exp(&eA, dA, s);
        mlx_array st2 = mlx_array_new(); mlx_multiply(&st2, state, eA, s);
        mlx_array nsf = mlx_array_new(); mlx_add(&nsf, ns, st2, s);
        mlx_array_set(&state, nsf);
        mlx_array yp = mlx_array_new(); mlx_matmul(&yp, nsf, Ce, s);
        mlx_array ys = mlx_array_new(); mlx_squeeze_axis(&ys, yp, 2, s);
        mlx_array dx = mlx_array_new(); mlx_multiply(&dx, Dv, xt, s);
        mlx_array yt = mlx_array_new(); mlx_add(&yt, ys, dx, s);
        mlx_array ye = mlx_array_new(); mlx_expand_dims(&ye, yt, 1, s);
        if (!yset) { mlx_array_set(&ystack, ye); yset = true; }
        else { mlx_array cc = mlx_array_new(); mlx_array arr[2] = { ystack, ye };
               mlx_vector_array v = mlx_vector_array_new_data(arr,2); mlx_concatenate_axis(&cc, v, 1, s);
               mlx_vector_array_free(v); mlx_array_set(&ystack, cc); mlx_array_free(cc); }
        mlx_array_free(xt); mlx_array_free(dbc); mlx_array_free(dlt); mlx_array_free(Bv); mlx_array_free(Cv);
        mlx_array_free(dltn); mlx_array_free(Bvn); mlx_array_free(Cvn);
        mlx_array_free(dlin); mlx_array_free(de); mlx_array_free(xe); mlx_array_free(Be); mlx_array_free(Ce);
        mlx_array_free(ns1); mlx_array_free(ns); mlx_array_free(dA); mlx_array_free(eA); mlx_array_free(st2);
        mlx_array_free(nsf); mlx_array_free(yp); mlx_array_free(ys); mlx_array_free(dx); mlx_array_free(yt); mlx_array_free(ye);
    }
    if (ok) mlx_array_set(k_cache, state);

    /* out = out_proj(swiglu(z, y)) ; swiglu(a,g) = a*silu(g) */
    mlx_array ow = mlx_array_new(), ows = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, li, "mamba.out_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &ows, &ob);
    mlx_array sw = mlx_array_new();
    if (ok) { mlx_array sg = mlx_array_new(), sl = mlx_array_new();
        ok = mlx_sigmoid(&sg, zz, s)==0 && mlx_multiply(&sl, zz, sg, s)==0 && mlx_multiply(&sw, ystack, sl, s)==0;
        mlx_array_free(sg); mlx_array_free(sl); }
    mlx_array oo = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&oo, sw, ow, ows, ob, gsz, bts, m->cfg.quant_mode, s);
    if (ok) ok = mlx_array_set(out, oo) == 0;

    mlx_array_free(ipw); mlx_array_free(ips); mlx_array_free(ipb);
    mlx_array_free(xz); mlx_array_free(xq); mlx_array_free(zz); mlx_array_free(cw); mlx_array_free(cstate);
    mlx_array_free(cf); mlx_array_free(cout); mlx_array_free(xc); mlx_array_free(cb);
    mlx_array_free(xpw); mlx_array_free(xps); mlx_array_free(xpb); mlx_array_free(dtw); mlx_array_free(dtsc); mlx_array_free(dtb); mlx_array_free(dtblin);
    mlx_array_free(dtlw); mlx_array_free(blw); mlx_array_free(clw);
    mlx_array_free(Alog); mlx_array_free(Aneg); mlx_array_free(Dv); mlx_array_free(state); mlx_array_free(ystack);
    mlx_array_free(ow); mlx_array_free(ows); mlx_array_free(ob); mlx_array_free(sw); mlx_array_free(oo);
    return ok;
#undef CHK
}

/* ---- GQA, senza RoPE ---- */
static bool jamba_attn(LlmMlxModel* m, int li, int cache_offset, int n_tokens, mlx_array x,
                       mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out) {
    (void)cache_offset;
    bool ok=true; const int gsz=m->cfg.quant_group_size,bts=m->cfg.quant_bits;
    const int nh=m->cfg.n_heads,nkv=m->cfg.n_kv_heads,hd=m->cfg.head_dim;
    char nb[256]; mlx_stream s=m->stream;
#define CHK(c,w) do{ if(!(c)){ fprintf(stderr,"jamba[layer=%d]: %s failed\n",li,(w)); ok=false; } }while(0)
    mlx_array qw=mlx_array_new(),qs=mlx_array_new(),qb=mlx_array_new(),kw=mlx_array_new(),ksc=mlx_array_new(),kb=mlx_array_new(),vw=mlx_array_new(),vsc=mlx_array_new(),vb=mlx_array_new(),oq=mlx_array_new(),os=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix,li,"self_attn.q_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&qw,&qs,&qb);
    tname(m->cfg.layer_prefix,li,"self_attn.k_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&kw,&ksc,&kb);
    tname(m->cfg.layer_prefix,li,"self_attn.v_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&vw,&vsc,&vb);
    tname(m->cfg.layer_prefix,li,"self_attn.o_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&oq,&os,&ob);
    mlx_array ql=mlx_array_new(),kl=mlx_array_new(),vl=mlx_array_new();
    if(ok){ ok=quantized_linear_cfg(&ql,x,qw,qs,qb,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"q_proj"); }
    if(ok){ ok=quantized_linear_cfg(&kl,x,kw,ksc,kb,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"k_proj"); }
    if(ok){ ok=quantized_linear_cfg(&vl,x,vw,vsc,vb,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"v_proj"); }
    mlx_array q4=mlx_array_new(),k4=mlx_array_new(),v4=mlx_array_new(),qt=mlx_array_new(),kt=mlx_array_new(),vt=mlx_array_new();
    if(ok){ int qsh[4]={1,n_tokens,nh,hd},ksh[4]={1,n_tokens,nkv,hd}; ok=mlx_reshape(&q4,ql,qsh,4,s)==0&&mlx_reshape(&k4,kl,ksh,4,s)==0&&mlx_reshape(&v4,vl,ksh,4,s)==0; CHK(ok,"reshape"); }
    if(ok){ const int p[4]={0,2,1,3}; ok=mlx_transpose_axes(&qt,q4,p,4,s)==0&&mlx_transpose_axes(&kt,k4,p,4,s)==0&&mlx_transpose_axes(&vt,v4,p,4,s)==0; CHK(ok,"transpose"); }
    mlx_array aout=mlx_array_new();
    if(ok){
        if(has_cache&&mlx_array_size(*k_cache)>0&&mlx_array_ndim(*k_cache)==4){
            mlx_array kn=mlx_array_new(),vn=mlx_array_new(); mlx_vector_array v=mlx_vector_array_new();
            mlx_vector_array_append_value(v,*k_cache); mlx_vector_array_append_value(v,kt); ok=mlx_concatenate_axis(&kn,v,2,s)==0; mlx_vector_array_free(v);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,vt); ok=ok&&mlx_concatenate_axis(&vn,vv,2,s)==0; mlx_vector_array_free(vv);
            if(ok){ mlx_array_free(*k_cache);*k_cache=kn; mlx_array_free(*v_cache);*v_cache=vn; } else { mlx_array_free(kn);mlx_array_free(vn); }
        } else { mlx_array_free(*k_cache);mlx_array_free(*v_cache); ok=mlx_array_set(k_cache,kt)==0&&mlx_array_set(v_cache,vt)==0; }
        CHK(ok,"kv concat");
        mlx_array kk=*k_cache, vv=*v_cache; float sc=1.0f/sqrtf((float)hd);
        mlx_array at=mlx_array_new(), nzm=mlx_array_new(), nzs=mlx_array_new();
        ok = mlx_fast_scaled_dot_product_attention(&at,qt,kk,vv,sc,"causal",nzm,nzs,s)==0;
        mlx_array_free(nzm); mlx_array_free(nzs);
        CHK(ok,"sdpa");
        mlx_array at2=mlx_array_new();
        if(ok){ const int p[4]={0,2,1,3}; ok=mlx_transpose_axes(&at2,at,p,4,s)==0; CHK(ok,"attn transpose"); }
        mlx_array at3=mlx_array_new(); if(ok){ int sh[3]={1,n_tokens,nh*hd}; ok=mlx_reshape(&at3,at2,sh,3,s)==0; CHK(ok,"attn reshape"); }
        if(ok){ ok=quantized_linear_cfg(&aout,at3,oq,os,ob,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"o_proj"); }
        mlx_array_free(at); mlx_array_free(at2); mlx_array_free(at3);
    }
    if(ok) ok = mlx_array_set(out,aout)==0;
    mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);mlx_array_free(kw);mlx_array_free(ksc);mlx_array_free(kb);mlx_array_free(vw);mlx_array_free(vsc);mlx_array_free(vb);mlx_array_free(oq);mlx_array_free(os);mlx_array_free(ob);mlx_array_free(ql);mlx_array_free(kl);mlx_array_free(vl);mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v4);mlx_array_free(qt);mlx_array_free(kt);mlx_array_free(vt);mlx_array_free(aout);
    return ok;
#undef CHK
}

bool jamba_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                      mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok=true; char nb[256]; mlx_stream s=m->stream;
    mlx_array nw=mlx_array_new(); tname(m->cfg.layer_prefix,layer_idx,"input_layernorm.weight",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&nw);
    mlx_array h=mlx_array_new(); if(ok) ok=mlx_fast_rms_norm(&h,x,nw,m->cfg.rms_norm_eps,s)==0;
    mlx_array mh=mlx_array_new();
    if(ok){
        int period=m->cfg.attn_layer_period;
        bool is_attn = (period>0) ? ((layer_idx % period)==m->cfg.attn_layer_offset) : true;
        if(is_attn) ok=jamba_attn(m,layer_idx,cache_offset,n_tokens,h,k_cache,v_cache,has_cache,&mh);
        else ok=jamba_mixer(m,layer_idx,n_tokens,h,k_cache,v_cache,has_cache,&mh);
    }
    if(ok) ok=mlx_add(&*out_h,x,mh,s)==0;
    mlx_array_free(nw);mlx_array_free(h);mlx_array_free(mh);
    return ok;
}

bool jamba_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok=true; const int gsz=m->cfg.quant_group_size,bts=m->cfg.quant_bits;
    char nb[256]; mlx_stream s=m->stream;
    mlx_array nw=mlx_array_new(); tname(m->cfg.layer_prefix,layer_idx,"pre_ff_layernorm.weight",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&nw);
    mlx_array hn=mlx_array_new(); if(ok) ok=mlx_fast_rms_norm(&hn,h,nw,m->cfg.rms_norm_eps,s)==0;
    mlx_array gw=mlx_array_new(),gs=mlx_array_new(),gb=mlx_array_new(),uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new(),dw=mlx_array_new(),ds=mlx_array_new(),db=mlx_array_new();
    tname(m->cfg.layer_prefix,layer_idx,"feed_forward.gate_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&gw,&gs,&gb);
    tname(m->cfg.layer_prefix,layer_idx,"feed_forward.up_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&uw,&us,&ub);
    tname(m->cfg.layer_prefix,layer_idx,"feed_forward.down_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&dw,&ds,&db);
    mlx_array g=mlx_array_new(),u=mlx_array_new(),d=mlx_array_new();
    if(ok) ok=quantized_linear_cfg(&g,hn,gw,gs,gb,gsz,bts,m->cfg.quant_mode,s);
    if(ok) ok=quantized_linear_cfg(&u,hn,uw,us,ub,gsz,bts,m->cfg.quant_mode,s);
    mlx_array sw=mlx_array_new();
    if(ok){ mlx_array sg=mlx_array_new(),sl=mlx_array_new(); ok=mlx_sigmoid(&sg,g,s)==0&&mlx_multiply(&sl,g,sg,s)==0&&mlx_multiply(&sw,u,sl,s)==0; mlx_array_free(sg);mlx_array_free(sl); }
    if(ok) ok=quantized_linear_cfg(&d,sw,dw,ds,db,gsz,bts,m->cfg.quant_mode,s);
    if(ok) ok=mlx_add(&*out,h,d,s)==0;
    mlx_array_free(nw);mlx_array_free(hn);mlx_array_free(gw);mlx_array_free(gs);mlx_array_free(gb);mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);mlx_array_free(dw);mlx_array_free(ds);mlx_array_free(db);mlx_array_free(g);mlx_array_free(u);mlx_array_free(d);mlx_array_free(sw);
    return ok;
}
