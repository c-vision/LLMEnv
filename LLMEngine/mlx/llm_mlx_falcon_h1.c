/* ============================================================================
 * llm_mlx_falcon_h1.c — dominio Falcon-H1 (ibrido attenzione + Mamba2).
 *   Layer:  r=h; h=input_layernorm(h); h = r + mamba(h) + self_attn(h);
 *           r=h; h=pre_ff_layernorm(h); h = r + mlp(h)
 *   mamba  = FalconH1Mixer (identico a Mamba2 ma chiude con swiglu(gate,y): mamba_rms_norm=false)
 *   attn   = GQA + RoPE (no q/k-norm)
 *   mlp    = swiglu(gate_proj, up_proj) -> down_proj
 * I muP multipliers sono GIA' folded nei pesi MLX dalla sanitize di mlx_lm -> NON riapplicati.
 * Prefix layer = "model.layers." (default). Output tied: final_step; fattore lm_head/embedding
 * applicato nel loader via cfg.falcon_h1_out_scale.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool falcon_h1_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool falcon_h1_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* ---- mixer (Mamba2 + swiglu finale) ---- */
static bool fh1_mamba(LlmMlxModel* m, int li, int n_tokens, mlx_array x, mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out) {
    bool ok = true;
    const int gsz=m->cfg.quant_group_size, bts=m->cfg.quant_bits;
    const int H=m->cfg.mamba.fh1_heads>0?m->cfg.mamba.fh1_heads:1, Dh=m->cfg.mamba.fh1_head_dim>0?m->cfg.mamba.fh1_head_dim:64;
    const int Inner=H*Dh, G=m->cfg.n_group>0?m->cfg.n_group:1, Ds=m->cfg.mamba.state_size;
    const int Kc=m->cfg.linear_conv_kernel_dim>0?m->cfg.linear_conv_kernel_dim:4;
    const int Hpg=H/G, ConvDim=Inner+2*G*Ds;
    const float ts_lo=m->cfg.mamba.ts_lo, ts_hi=m->cfg.mamba.ts_hi;
    char nb[256]; mlx_stream s=m->stream;
#define CHK(c,w) do{ if(!(c)){ fprintf(stderr,"falcon_h1[layer=%d]: %s failed\n",li,(w)); ok=false; } }while(0)
    mlx_array ipw=mlx_array_new(),ips=mlx_array_new(),ipb=mlx_array_new();
    tname(m->cfg.layer_prefix,li,"mamba.in_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&ipw,&ips,&ipb);
    mlx_array proj=mlx_array_new();
    if(ok){ ok=quantized_linear_cfg(&proj,x,ipw,ips,ipb,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"in_proj"); }
    mlx_array gate=mlx_array_new(),conv_in=mlx_array_new(),dt=mlx_array_new();
    if(ok){ int sec[2]={Inner,Inner+ConvDim}; mlx_vector_array v=mlx_vector_array_new();
        ok=mlx_split_sections(&v,proj,sec,2,-1,s)==0;
        if(ok) ok=mlx_vector_array_get(&gate,v,0)==0&&mlx_vector_array_get(&conv_in,v,1)==0&&mlx_vector_array_get(&dt,v,2)==0;
        mlx_vector_array_free(v); CHK(ok,"split proj"); }
    mlx_array cw=mlx_array_new();
    tname(m->cfg.layer_prefix,li,"mamba.conv1d.weight",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&cw);
    { mlx_array c32=mlx_array_new(); if(mlx_astype(&c32,cw,MLX_FLOAT32,s)==0){mlx_array_free(cw);cw=c32;}else mlx_array_free(c32); }
    if(mlx_array_ndim(cw)==3&&mlx_array_shape(cw)[1]==1){int ax[3]={0,2,1};mlx_array t=mlx_array_new();if(mlx_transpose_axes(&t,cw,ax,3,s)==0){mlx_array_free(cw);cw=t;}else mlx_array_free(t);}
    mlx_array cb=mlx_array_new();
    tname(m->cfg.layer_prefix,li,"mamba.conv1d.bias",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&cb);
    mlx_array cst=mlx_array_new();
    if(ok){ if(has_cache&&mlx_array_size(*v_cache)>0) mlx_array_set(&cst,*v_cache); else {int cs[3]={1,Kc-1,ConvDim};mlx_zeros(&cst,cs,3,MLX_FLOAT32,s);} }
    mlx_array cf=mlx_array_new();
    if(ok){ mlx_array a[2]={cst,conv_in}; mlx_vector_array v=mlx_vector_array_new_data(a,2); ok=mlx_concatenate_axis(&cf,v,1,s)==0; mlx_vector_array_free(v); CHK(ok,"conv concat"); }
    if(ok){ int ss[3]={0,n_tokens,0},se[3]={1,n_tokens+(Kc-1),ConvDim},st[3]={1,1,1}; mlx_array sl=mlx_array_new(); ok=mlx_slice(&sl,cf,ss,3,se,3,st,3,s)==0; if(ok) mlx_array_set(v_cache,sl); mlx_array_free(sl); }
    mlx_array cout=mlx_array_new();
    if(ok){ int str[1]={1},pl[1]={0},ph[1]={0},kd[1]={1},id[1]={1}; ok=mlx_conv_general(&cout,cf,cw,str,1,pl,1,ph,1,kd,1,id,1,ConvDim,false,s)==0; CHK(ok,"conv1d"); }
    if(ok&&cb.ctx!=NULL&&mlx_array_size(cb)>0){ mlx_array c2=mlx_array_new(); ok=mlx_add(&c2,cout,cb,s)==0; if(ok) mlx_array_set(&cout,c2); mlx_array_free(c2); }
    mlx_array xc=mlx_array_new();
    if(ok){ mlx_array sg=mlx_array_new(); ok=mlx_sigmoid(&sg,cout,s)==0&&mlx_multiply(&xc,cout,sg,s)==0; mlx_array_free(sg); CHK(ok,"silu"); }
    mlx_array hid=mlx_array_new(),Bv=mlx_array_new(),Cv=mlx_array_new();
    if(ok){ int sec[2]={Inner,Inner+G*Ds}; mlx_vector_array v=mlx_vector_array_new(); ok=mlx_split_sections(&v,xc,sec,2,-1,s)==0;
        if(ok) ok=mlx_vector_array_get(&hid,v,0)==0&&mlx_vector_array_get(&Bv,v,1)==0&&mlx_vector_array_get(&Cv,v,2)==0; mlx_vector_array_free(v); CHK(ok,"split conv"); }
    mlx_array Al=mlx_array_new(),Dv=mlx_array_new(),dtb=mlx_array_new();
    tname(m->cfg.layer_prefix,li,"mamba.A_log",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&Al);
    tname(m->cfg.layer_prefix,li,"mamba.D",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&Dv);
    tname(m->cfg.layer_prefix,li,"mamba.dt_bias",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&dtb);
    mlx_array An=mlx_array_new(); if(ok){ mlx_array e=mlx_array_new(); ok=mlx_exp(&e,Al,s)==0&&mlx_negative(&An,e,s)==0; mlx_array_free(e); }
    mlx_array hd=mlx_array_new(),bg=mlx_array_new(),cg=mlx_array_new();
    if(ok){ int sh[4]={1,n_tokens,H,Dh}; ok=mlx_reshape(&hd,hid,sh,4,s)==0; }
    if(ok){ int sh[4]={1,n_tokens,G,Ds}; ok=mlx_reshape(&bg,Bv,sh,4,s)==0; int sc[4]={1,n_tokens,G,Ds}; ok=ok&&mlx_reshape(&cg,Cv,sc,4,s)==0; }
    mlx_array Bt=mlx_array_new(),Ct=mlx_array_new();
    if(ok){ int reps[4]={1,1,Hpg,1}; ok=mlx_tile(&Bt,bg,reps,4,s)==0&&mlx_tile(&Ct,cg,reps,4,s)==0; }
    mlx_array state=mlx_array_new();
    if(ok){ if(has_cache&&mlx_array_size(*k_cache)>0) mlx_array_set(&state,*k_cache); else {int st[4]={1,H,Dh,Ds};mlx_zeros(&state,st,4,MLX_FLOAT32,s);} }
    mlx_array ystack=mlx_array_new(); bool yset=false;
    for(int t=0;t<n_tokens&&ok;t++){
        mlx_array xt=mlx_array_new(),Bt1=mlx_array_new(),Ct1=mlx_array_new(),dt1=mlx_array_new();
        { int a[4]={0,t,0,0},b[4]={1,t+1,H,Dh},st[4]={1,1,1,1}; ok=mlx_slice(&xt,hd,a,4,b,4,st,4,s)==0; if(ok) ok=mlx_squeeze_axis(&xt,xt,1,s)==0; }
        { int a[4]={0,t,0,0},b[4]={1,t+1,H,Ds},st[4]={1,1,1,1}; ok=ok&&mlx_slice(&Bt1,Bt,a,4,b,4,st,4,s)==0&&mlx_slice(&Ct1,Ct,a,4,b,4,st,4,s)==0; if(ok) ok=mlx_squeeze_axis(&Bt1,Bt1,1,s)==0&&mlx_squeeze_axis(&Ct1,Ct1,1,s)==0; }
        { int a[3]={0,t,0},b[3]={1,t+1,H},st[3]={1,1,1}; ok=ok&&mlx_slice(&dt1,dt,a,3,b,3,st,3,s)==0; if(ok) ok=mlx_squeeze_axis(&dt1,dt1,0,s)==0; }
        mlx_array ds=mlx_array_new(),dsp=mlx_array_new(),dcl=mlx_array_new(),db3=mlx_array_new();
        { int sh[2]={1,H}; ok=ok&&mlx_reshape(&db3,dtb,sh,2,s)==0; }
        ok=ok&&mlx_add(&ds,dt1,db3,s)==0;
        { mlx_array z=mlx_array_new(); int zs[1]={1}; mlx_zeros(&z,zs,1,MLX_FLOAT32,s); ok=ok&&mlx_logaddexp(&dsp,ds,z,s)==0; mlx_array_free(z); }
        { mlx_array lo=mlx_array_new_float32(ts_lo),hi=mlx_array_new_float32(ts_hi); ok=ok&&mlx_clip(&dcl,dsp,lo,hi,s)==0; mlx_array_free(lo); mlx_array_free(hi); }
        mlx_array a3=mlx_array_new(); { int sh[2]={1,H}; ok=ok&&mlx_reshape(&a3,An,sh,2,s)==0; }
        mlx_array dm=mlx_array_new(),de=mlx_array_new(); ok=ok&&mlx_multiply(&dm,a3,dcl,s)==0&&mlx_exp(&de,dm,s)==0;
        mlx_array xde=mlx_array_new(),Bde=mlx_array_new(),Bde2=mlx_array_new(),outer=mlx_array_new(),outerd=mlx_array_new();
        ok=ok&&mlx_expand_dims(&xde,xt,-1,s)==0&&mlx_expand_dims(&Bde,Bt1,-1,s)==0;
        ok=ok&&mlx_swapaxes(&Bde2,Bde,2,3,s)==0&&mlx_multiply(&outer,xde,Bde2,s)==0;
        mlx_array de4=mlx_array_new(),dcl4=mlx_array_new(),t1=mlx_array_new(),t2=mlx_array_new();
        ok=ok&&mlx_expand_dims(&t1,de,-1,s)==0&&mlx_expand_dims(&de4,t1,-1,s)==0;
        ok=ok&&mlx_expand_dims(&t2,dcl,-1,s)==0&&mlx_expand_dims(&dcl4,t2,-1,s)==0;
        mlx_array stA=mlx_array_new(),nst=mlx_array_new();
        ok=ok&&mlx_multiply(&stA,state,de4,s)==0&&mlx_multiply(&outerd,outer,dcl4,s)==0&&mlx_add(&nst,stA,outerd,s)==0; if(ok) mlx_array_set(&state,nst);
        mlx_array Cde=mlx_array_new(),yp=mlx_array_new(),ys=mlx_array_new(); ok=ok&&mlx_expand_dims(&Cde,Ct1,-1,s)==0&&mlx_matmul(&yp,state,Cde,s)==0&&mlx_squeeze_axis(&ys,yp,3,s)==0;
        mlx_array D3=mlx_array_new(),D3b=mlx_array_new(),dx=mlx_array_new(),yt=mlx_array_new();
        { int sh[2]={1,H}; ok=ok&&mlx_reshape(&D3,Dv,sh,2,s)==0; ok=ok&&mlx_expand_dims(&D3b,D3,-1,s)==0; }
        ok=ok&&mlx_multiply(&dx,xt,D3b,s)==0&&mlx_add(&yt,ys,dx,s)==0;
        if(!yset){ mlx_array ye=mlx_array_new(); ok=ok&&mlx_expand_dims(&ye,yt,1,s)==0; if(ok){mlx_array_set(&ystack,ye);yset=true;} mlx_array_free(ye); }
        else { mlx_array ye=mlx_array_new(),cc=mlx_array_new(); ok=ok&&mlx_expand_dims(&ye,yt,1,s)==0; mlx_array a[2]={ystack,ye}; mlx_vector_array v=mlx_vector_array_new_data(a,2);
               ok=ok&&mlx_concatenate_axis(&cc,v,1,s)==0; mlx_vector_array_free(v); if(ok) mlx_array_set(&ystack,cc); mlx_array_free(ye); mlx_array_free(cc); }
        mlx_array_free(xt);mlx_array_free(Bt1);mlx_array_free(Ct1);mlx_array_free(dt1);mlx_array_free(ds);mlx_array_free(dsp);mlx_array_free(dcl);mlx_array_free(db3);mlx_array_free(a3);mlx_array_free(dm);mlx_array_free(de);mlx_array_free(xde);mlx_array_free(Bde);mlx_array_free(Bde2);mlx_array_free(outer);mlx_array_free(outerd);mlx_array_free(de4);mlx_array_free(dcl4);mlx_array_free(t1);mlx_array_free(t2);mlx_array_free(stA);mlx_array_free(nst);mlx_array_free(Cde);mlx_array_free(yp);mlx_array_free(ys);mlx_array_free(D3);mlx_array_free(D3b);mlx_array_free(dx);mlx_array_free(yt);
    }
    mlx_array_free(state);
    mlx_array ow=mlx_array_new(),ows=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix,li,"mamba.out_proj",nb,sizeof(nb)); ok=ok&&multishard_get_triple(&m->weights,nb,&ow,&ows,&ob);
    mlx_array yf=mlx_array_new();
    if(ok){ int sh[3]={n_tokens,H,Dh}; mlx_array ys=mlx_array_new(); ok=mlx_reshape(&ys,ystack,sh,3,s)==0; int s2[2]={n_tokens,Inner}; ok=ok&&mlx_reshape(&yf,ys,s2,2,s)==0; mlx_array_free(ys); }
    mlx_array sw=mlx_array_new();
    if(ok){ mlx_array sg=mlx_array_new(),sl=mlx_array_new(); ok=mlx_sigmoid(&sg,gate,s)==0&&mlx_multiply(&sl,gate,sg,s)==0&&mlx_multiply(&sw,yf,sl,s)==0; mlx_array_free(sg); mlx_array_free(sl); }
    if(ok){ ok=quantized_linear_cfg(out,sw,ow,ows,ob,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"out_proj"); }
    mlx_array_free(ipw);mlx_array_free(ips);mlx_array_free(ipb);mlx_array_free(proj);mlx_array_free(gate);mlx_array_free(conv_in);mlx_array_free(dt);mlx_array_free(cw);mlx_array_free(cb);mlx_array_free(cst);mlx_array_free(cf);mlx_array_free(cout);mlx_array_free(xc);mlx_array_free(hid);mlx_array_free(Bv);mlx_array_free(Cv);mlx_array_free(Al);mlx_array_free(An);mlx_array_free(Dv);mlx_array_free(dtb);mlx_array_free(hd);mlx_array_free(bg);mlx_array_free(cg);mlx_array_free(Bt);mlx_array_free(Ct);mlx_array_free(ystack);mlx_array_free(yf);mlx_array_free(sw);mlx_array_free(ow);mlx_array_free(ows);mlx_array_free(ob);
    return ok;
#undef CHK
}

/* ---- GQA + RoPE ---- */
static bool fh1_attn(LlmMlxModel* m, int li, int cache_offset, int n_tokens, mlx_array x, mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out) {
    bool ok=true; const int gsz=m->cfg.quant_group_size,bts=m->cfg.quant_bits;
    const int nh=m->cfg.n_heads,nkv=m->cfg.n_kv_heads,hd=m->cfg.head_dim;
    char nb[256]; mlx_stream s=m->stream;
#define CHK(c,w) do{ if(!(c)){ fprintf(stderr,"falcon_h1[layer=%d]: %s failed\n",li,(w)); ok=false; } }while(0)
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
    mlx_array qr=mlx_array_new(),kr=mlx_array_new();
    if(ok){ mlx_optional_float base={m->cfg.rope_theta,true}; mlx_array nf=mlx_array_new(); ok=mlx_fast_rope(&qr,qt,hd,false,base,1.0f,cache_offset,nf,s)==0&&mlx_fast_rope(&kr,kt,hd,false,base,1.0f,cache_offset,nf,s)==0; mlx_array_free(nf); CHK(ok,"rope"); }
    mlx_array aout=mlx_array_new();
    if(ok){
        if(has_cache&&mlx_array_size(*k_cache)>0&&mlx_array_ndim(*k_cache)==4){
            mlx_array kn=mlx_array_new(),vn=mlx_array_new(); mlx_vector_array v=mlx_vector_array_new();
            mlx_vector_array_append_value(v,*k_cache); mlx_vector_array_append_value(v,kr); ok=mlx_concatenate_axis(&kn,v,2,s)==0; mlx_vector_array_free(v);
            mlx_vector_array vv=mlx_vector_array_new(); mlx_vector_array_append_value(vv,*v_cache); mlx_vector_array_append_value(vv,vt); ok=ok&&mlx_concatenate_axis(&vn,vv,2,s)==0; mlx_vector_array_free(vv);
            if(ok){ mlx_array_free(*k_cache);*k_cache=kn; mlx_array_free(*v_cache);*v_cache=vn; } else { mlx_array_free(kn);mlx_array_free(vn); }
        } else { mlx_array_free(*k_cache);mlx_array_free(*v_cache); ok=mlx_array_set(k_cache,kr)==0&&mlx_array_set(v_cache,vt)==0; }
        CHK(ok,"kv concat");
        mlx_array kk=*k_cache, vv=*v_cache; float sc=1.0f/sqrtf((float)hd);
        mlx_array at=mlx_array_new(), nzm=mlx_array_new(), nzs=mlx_array_new();
        ok = mlx_fast_scaled_dot_product_attention(&at,qr,kk,vv,sc,"causal",nzm,nzs,s)==0;
        mlx_array_free(nzm); mlx_array_free(nzs);
        CHK(ok,"sdpa");
        mlx_array at2=mlx_array_new();
        if(ok){ const int p[4]={0,2,1,3}; ok=mlx_transpose_axes(&at2,at,p,4,s)==0; CHK(ok,"attn transpose"); }
        mlx_array at3=mlx_array_new(); if(ok){ int sh[3]={1,n_tokens,nh*hd}; ok=mlx_reshape(&at3,at2,sh,3,s)==0; CHK(ok,"attn reshape"); }
        if(ok){ ok=quantized_linear_cfg(&aout,at3,oq,os,ob,gsz,bts,m->cfg.quant_mode,s); CHK(ok,"o_proj"); }
        mlx_array_free(at); mlx_array_free(at2); mlx_array_free(at3);
    }
    if(ok) ok = mlx_array_set(out,aout)==0;
    mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);mlx_array_free(kw);mlx_array_free(ksc);mlx_array_free(kb);mlx_array_free(vw);mlx_array_free(vsc);mlx_array_free(vb);mlx_array_free(oq);mlx_array_free(os);mlx_array_free(ob);mlx_array_free(ql);mlx_array_free(kl);mlx_array_free(vl);mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v4);mlx_array_free(qt);mlx_array_free(kt);mlx_array_free(vt);mlx_array_free(qr);mlx_array_free(kr);mlx_array_free(aout);
    return ok;
#undef CHK
}

bool falcon_h1_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok=true; char nb[256]; mlx_stream s=m->stream;
    mlx_array nw=mlx_array_new(); tname(m->cfg.layer_prefix,layer_idx,"input_layernorm.weight",nb,sizeof(nb)); ok=ok&&multishard_get(&m->weights,nb,&nw);
    mlx_array h=mlx_array_new(); if(ok) ok=mlx_fast_rms_norm(&h,x,nw,m->cfg.rms_norm_eps,s)==0;
    if(!ok) fprintf(stderr,"falcon_h1[%d] input_norm fail\n",layer_idx);
    mlx_array mh=mlx_array_new(),ah=mlx_array_new();
    /* Il mixer Mamba2 ha la PROPRIA cache (stato SSM+conv), distinta dalla KV dell'attenzione
     * (il reference usa CacheList(ArraysCache, KVCache)). Qui: cache locale fresca per la
     * chiamata (corretta per il prefill: tutti i token in una sola invocazione). */
    mlx_array mk=mlx_array_new(), mv=mlx_array_new();
    if(ok) ok=fh1_mamba(m,layer_idx,n_tokens,h,&mk,&mv,false,&mh);
    mlx_array_free(mk); mlx_array_free(mv);
    if(ok) ok=fh1_attn(m,layer_idx,cache_offset,n_tokens,h,k_cache,v_cache,has_cache,&ah);
    if(ok){ mlx_array t=mlx_array_new(); ok=mlx_add(&t,x,mh,s)==0&&mlx_add(&*out_h,t,ah,s)==0; mlx_array_free(t); }
    mlx_array_free(nw);mlx_array_free(h);mlx_array_free(mh);mlx_array_free(ah);
    return ok;
}

bool falcon_h1_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
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
    if(ok){ ok=mlx_add(&*out,h,d,s)==0; }
    mlx_array_free(nw);mlx_array_free(hn);mlx_array_free(gw);mlx_array_free(gs);mlx_array_free(gb);mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);mlx_array_free(dw);mlx_array_free(ds);mlx_array_free(db);mlx_array_free(g);mlx_array_free(u);mlx_array_free(d);mlx_array_free(sw);
    return ok;
}
