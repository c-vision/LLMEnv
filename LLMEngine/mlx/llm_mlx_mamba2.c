/* ============================================================================
 * llm_mlx_mamba2.c — dominio Mamba2 (SSD), arch ricorrente.
 *   ResidualBlock:  out = Mamba2Block(RMSNorm(x)) + x
 *   Mamba2Block:    projected = in_proj(x); split(gate, conv_input, dt);
 *                   conv = silu(conv1d(conv_input, state));   (depthwise su conv_dim)
 *                   split(conv) -> (hidden_states, B, C);
 *                   y = ssm(hidden_states, B, C, dt);         (SSD: ricorrenza per-token, vettorizzata sulle teste)
 *                   y = rms_norm(silu(gate) * y, mixer.norm.weight, eps);  (MambaRMSNormGated)
 *                   out = out_proj(y)
 * Lo slot attn fa norm+mixer+residuo; lo slot ffn e' IDENTITA'.
 * Modellato su mlx_lm/models/mamba2.py + ssm.py (la ricorrenza per-token e' equivalente
 * alla forma chunked `ssm_attn`; correctness-first).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool mamba2_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                         mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool mamba2_ffn_identity(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

bool mamba2_ffn_identity(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    (void)m; (void)layer_idx; (void)n_tokens;
    return mlx_array_set(out, h) == 0;
}

bool mamba2_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                         mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int Hid = m->cfg.hidden_size;
    const int H = m->cfg.n_heads > 0 ? m->cfg.n_heads : 1; /* num_heads */
    const int Dh = m->cfg.head_dim;                    /* head_dim */
    const int Inner = H * Dh;                          /* intermediate_size */
    const int G = m->cfg.n_group > 0 ? m->cfg.n_group : 1;
    const int Ds = m->cfg.mamba_state_size > 0 ? m->cfg.mamba_state_size : 16;
    const int Kc = m->cfg.linear_conv_kernel_dim > 0 ? m->cfg.linear_conv_kernel_dim : 4;
    const int Hpg = H / G;                             /* heads_per_group */
    const int ConvDim = Inner + 2 * G * Ds;
    const float ts_lo = m->cfg.mamba_ts_lo, ts_hi = m->cfg.mamba_ts_hi;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "mamba2[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    /* norm */
    mlx_array nw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &nw);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, nw, m->cfg.rms_norm_eps, s); CHK(ok, "norm"); }

    /* in_proj -> gate, conv_input, dt */
    mlx_array ipw = mlx_array_new(), ips = mlx_array_new(), ipb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.in_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ipw, &ips, &ipb);
    mlx_array proj = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&proj, h, ipw, ips, ipb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj"); }
    mlx_array gate = mlx_array_new(), conv_in = mlx_array_new(), dt = mlx_array_new();
    if (ok) { int sec[2] = { Inner, Inner + ConvDim }; mlx_vector_array v = mlx_vector_array_new();
        ok = mlx_split_sections(&v, proj, sec, 2, -1, s) == 0;
        if (ok) ok = mlx_vector_array_get(&gate,v,0)==0 && mlx_vector_array_get(&conv_in,v,1)==0 && mlx_vector_array_get(&dt,v,2)==0;
        mlx_vector_array_free(v); CHK(ok, "split proj"); }

    /* conv1d depthwise su conv_dim (cache [1,Kc-1,ConvDim] in v_cache) */
    mlx_array cw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.conv1d.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cw);
    { mlx_array c32 = mlx_array_new(); if (mlx_astype(&c32, cw, MLX_FLOAT32, s) == 0) { mlx_array_free(cw); cw = c32; } else mlx_array_free(c32); }
    if (mlx_array_ndim(cw) == 3 && mlx_array_shape(cw)[1] == 1) { int ax[3] = {0,2,1};
        mlx_array cwt = mlx_array_new(); if (mlx_transpose_axes(&cwt, cw, ax, 3, s) == 0) { mlx_array_free(cw); cw = cwt; } else mlx_array_free(cwt); }
    mlx_array cbias = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.conv1d.bias", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cbias);
    mlx_array cstate = mlx_array_new();
    if (ok) { if (has_cache && mlx_array_size(*v_cache) > 0) mlx_array_set(&cstate, *v_cache);
        else { int cs[3] = { 1, Kc-1, ConvDim }; mlx_zeros(&cstate, cs, 3, MLX_FLOAT32, s); } }
    mlx_array cf = mlx_array_new();
    if (ok) { mlx_array arr[2] = { cstate, conv_in }; mlx_vector_array v = mlx_vector_array_new_data(arr, 2);
        ok = mlx_concatenate_axis(&cf, v, 1, s) == 0; mlx_vector_array_free(v); CHK(ok, "conv concat"); }
    if (ok) { int ss[3]={0,n_tokens,0}, se[3]={1,n_tokens+(Kc-1),ConvDim}, st[3]={1,1,1};
        mlx_array sl=mlx_array_new(); ok = mlx_slice(&sl, cf, ss,3,se,3,st,3,s)==0; if (ok) mlx_array_set(v_cache, sl); mlx_array_free(sl); }
    mlx_array cout = mlx_array_new();
    if (ok) { int str[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, id[1]={1};
        ok = mlx_conv_general(&cout, cf, cw, str,1,pl,1,ph,1,kd,1,id,1, ConvDim, false, s) == 0; CHK(ok, "conv1d"); }
    if (ok && cbias.ctx != NULL && mlx_array_size(cbias) > 0) { mlx_array c2=mlx_array_new(); ok = mlx_add(&c2, cout, cbias, s)==0; if (ok) mlx_array_set(&cout, c2); mlx_array_free(c2); }
    mlx_array xc = mlx_array_new();
    if (ok) { mlx_array sig=mlx_array_new(); ok = mlx_sigmoid(&sig, cout, s)==0 && mlx_multiply(&xc, cout, sig, s)==0; mlx_array_free(sig); CHK(ok, "silu"); }
    mlx_array hidden = mlx_array_new(), Bv = mlx_array_new(), Cv = mlx_array_new();
    if (ok) { int sec[2] = { Inner, Inner + G*Ds }; mlx_vector_array v = mlx_vector_array_new();
        ok = mlx_split_sections(&v, xc, sec, 2, -1, s) == 0;
        if (ok) ok = mlx_vector_array_get(&hidden,v,0)==0 && mlx_vector_array_get(&Bv,v,1)==0 && mlx_vector_array_get(&Cv,v,2)==0;
        mlx_vector_array_free(v); CHK(ok, "split conv"); }

    /* A_log [H], D [H], dt_bias [H] */
    mlx_array Alog=mlx_array_new(), Dv=mlx_array_new(), dtb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.A_log", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &Alog);
    tname(m->cfg.layer_prefix, layer_idx, "mixer.D", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &Dv);
    tname(m->cfg.layer_prefix, layer_idx, "mixer.dt_bias", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &dtb);
    mlx_array Aneg = mlx_array_new();
    if (ok) { mlx_array ea=mlx_array_new(); ok = mlx_exp(&ea, Alog, s)==0 && mlx_negative(&Aneg, ea, s)==0; mlx_array_free(ea); }
    /* B/C [1,T,G*Ds] -> [1,T,G,Ds] -> tile su H -> [1,T,H,Ds]; hidden -> [1,T,H,Dh] */
    mlx_array hd=mlx_array_new(), bg=mlx_array_new(), cg=mlx_array_new();
    if (ok) { int sh[4]={1,n_tokens,H,Dh}; ok = mlx_reshape(&hd, hidden, sh, 4, s)==0; }
    if (ok) { int sh[4]={1,n_tokens,G,Ds}; ok = mlx_reshape(&bg, Bv, sh, 4, s)==0; int sc[4]={1,n_tokens,G,Ds}; ok = ok && mlx_reshape(&cg, Cv, sc, 4, s)==0; }
    mlx_array Bt=mlx_array_new(), Ct=mlx_array_new();
    if (ok) { int reps[4]={1,1,Hpg,1}; ok = mlx_tile(&Bt, bg, reps, 4, s)==0 && mlx_tile(&Ct, cg, reps, 4, s)==0; }

    /* state [1,H,Dh,Ds] in k_cache */
    mlx_array state = mlx_array_new();
    if (ok) { if (has_cache && mlx_array_size(*k_cache) > 0) mlx_array_set(&state, *k_cache);
        else { int st4[4] = { 1, H, Dh, Ds }; mlx_zeros(&state, st4, 4, MLX_FLOAT32, s); } }
    /* y accumulato [1,T,H,Dh] */
    mlx_array ystack = mlx_array_new(); bool yset = false;
    /* Ricorrenza SSD (vettorizzata sulle teste) per-token:
     *   dt_ = clip(softplus(dt+dt_bias), lo, hi)          [1,H]
     *   dA  = exp(A * dt_)                                 [1,H]
     *   state = dA*state + dt_ * (x[:,h] ⊗ B[:,h])         [1,H,Dh,Ds]
     *   y = (state @ C[:,h]) + D*x                         [1,H,Dh]                     */
    for (int t = 0; t < n_tokens && ok; t++) {
        /* xt [1,H,Dh], dt_t [1,H], B_t/C_t [1,H,Ds] */
        mlx_array xt=mlx_array_new(), Btt=mlx_array_new(), Ctt=mlx_array_new(), dtt=mlx_array_new();
        { int ss[4]={0,t,0,0}, se[4]={1,t+1,H,Dh}, st[4]={1,1,1,1}; ok = mlx_slice(&xt, hd, ss,4,se,4,st,4,s)==0; if (ok) ok = mlx_squeeze_axis(&xt,xt,1,s)==0; }
        { int ss[4]={0,t,0,0}, se[4]={1,t+1,H,Ds}, st[4]={1,1,1,1}; ok = ok && mlx_slice(&Btt, Bt, ss,4,se,4,st,4,s)==0 && mlx_slice(&Ctt, Ct, ss,4,se,4,st,4,s)==0; if (ok) ok = mlx_squeeze_axis(&Btt,Btt,1,s)==0 && mlx_squeeze_axis(&Ctt,Ctt,1,s)==0; }
        { int ss[3]={0,t,0}, se[3]={1,t+1,H}, st[3]={1,1,1}; ok = ok && mlx_slice(&dtt, dt, ss,3,se,3,st,3,s)==0; if (ok) ok = mlx_squeeze_axis(&dtt,dtt,0,s)==0; }
        /* dt_ = clip(softplus(dtt + dtb), lo, hi) : dtt [1,1,H], dtb [H] broadcast */
        mlx_array dsum=mlx_array_new(), dsp=mlx_array_new(), dcl=mlx_array_new();
        mlx_array db3=mlx_array_new(); { int sh[2]={1,H}; ok = ok && mlx_reshape(&db3, dtb, sh, 2, s)==0; }
        ok = ok && mlx_add(&dsum, dtt, db3, s)==0;
        { mlx_array zero=mlx_array_new(); int zs[1]={1}; mlx_zeros(&zero,zs,1,MLX_FLOAT32,s);
          ok = ok && mlx_logaddexp(&dsp, dsum, zero, s)==0; mlx_array_free(zero); }
        { mlx_array lo=mlx_array_new_float32(ts_lo), hi=mlx_array_new_float32(ts_hi);
          ok = ok && mlx_clip(&dcl, dsp, lo, hi, s)==0; mlx_array_free(lo); mlx_array_free(hi); }
        mlx_array_free(db3);
        /* Aneg [H] -> [1,1,H]; dA = exp(A*dcl) */
        mlx_array a3=mlx_array_new(); { int sh[2]={1,H}; ok = ok && mlx_reshape(&a3, Aneg, sh, 2, s)==0; }
        mlx_array dAm=mlx_array_new(), dAe=mlx_array_new();
        ok = ok && mlx_multiply(&dAm, a3, dcl, s)==0 && mlx_exp(&dAe, dAm, s)==0;
        /* state = dA*state + dcl*(x⊗B) */
        mlx_array xde=mlx_array_new(), Bde=mlx_array_new(), outer=mlx_array_new(), outerd=mlx_array_new();
        ok = ok && mlx_expand_dims(&xde, xt, -1, s)==0 && mlx_expand_dims(&Bde, Btt, -1, s)==0; /* xde[1,H,Dh,1], Bde[1,H,Ds,1] */
        { mlx_array Bde2=mlx_array_new(); int ax[2]={2,3}; ok = ok && mlx_swapaxes(&Bde2, Bde, 2,3, s)==0; /* [1,H,1,Ds] */
          ok = ok && mlx_multiply(&outer, xde, Bde2, s)==0; mlx_array_free(Bde2); }  /* [1,H,Dh,Ds] */
        mlx_array dA4=mlx_array_new(), dcl4=mlx_array_new();
        { mlx_array t=mlx_array_new(); ok = ok && mlx_expand_dims(&t, dAe, -1, s)==0 && mlx_expand_dims(&dA4, t, -1, s)==0; mlx_array_free(t); }
        { mlx_array t=mlx_array_new(); ok = ok && mlx_expand_dims(&t, dcl, -1, s)==0 && mlx_expand_dims(&dcl4, t, -1, s)==0; mlx_array_free(t); }
        { mlx_array stA=mlx_array_new(); ok = ok && mlx_multiply(&stA, state, dA4, s)==0;
          ok = ok && mlx_multiply(&outerd, outer, dcl4, s)==0;
          mlx_array nst=mlx_array_new(); ok = ok && mlx_add(&nst, stA, outerd, s)==0; if (ok) mlx_array_set(&state, nst);
          mlx_array_free(stA); mlx_array_free(nst); }
        /* y = (state @ C) + D*x : Ctt [1,H,Ds] -> [1,H,Ds,1]; state[1,H,Dh,Ds] -> [1,H,Dh,1] */
        mlx_array Cde=mlx_array_new(); ok = ok && mlx_expand_dims(&Cde, Ctt, -1, s)==0; /* [1,H,Ds,1] */
        mlx_array yp=mlx_array_new(), ys=mlx_array_new();
        ok = ok && mlx_matmul(&yp, state, Cde, s)==0 && mlx_squeeze_axis(&ys, yp, 3, s)==0; /* [1,H,Dh] */
        mlx_array D3=mlx_array_new(); { int sh[2]={1,H}; ok = ok && mlx_reshape(&D3, Dv, sh, 2, s)==0; mlx_array D3b=mlx_array_new(); ok = ok && mlx_expand_dims(&D3b, D3, -1, s)==0; /* [1,H,1] */
          mlx_array dx=mlx_array_new(); ok = ok && mlx_multiply(&dx, xt, D3b, s)==0;
          mlx_array yt=mlx_array_new(); ok = ok && mlx_add(&yt, ys, dx, s)==0;
          /* accumula [1,T,H,Dh] */
          if (!yset) { mlx_array ye=mlx_array_new(); ok = ok && mlx_expand_dims(&ye, yt, 1, s)==0; if (ok) { mlx_array_set(&ystack, ye); yset = true; } mlx_array_free(ye); }
          else { mlx_array cc=mlx_array_new(); mlx_array yte=mlx_array_new(); ok = ok && mlx_expand_dims(&yte,yt,1,s)==0; mlx_array arr[2]={ystack,yte}; mlx_vector_array vv=mlx_vector_array_new_data(arr,2);
                 ok = ok && mlx_concatenate_axis(&cc, vv, 1, s)==0; mlx_vector_array_free(vv); if (ok) mlx_array_set(&ystack, cc); mlx_array_free(cc); mlx_array_free(yte); }
          mlx_array_free(dx); mlx_array_free(yt); mlx_array_free(D3b); }
        mlx_array_free(D3);
        mlx_array_free(Cde); mlx_array_free(yp); mlx_array_free(ys);
        mlx_array_free(dA4); mlx_array_free(dcl4); mlx_array_free(outer); mlx_array_free(outerd);
        mlx_array_free(xde); mlx_array_free(Bde); mlx_array_free(dAm); mlx_array_free(dAe); mlx_array_free(a3);
        mlx_array_free(dsum); mlx_array_free(dsp); mlx_array_free(dcl);
        mlx_array_free(xt); mlx_array_free(Btt); mlx_array_free(Ctt); mlx_array_free(dtt);
    }
    mlx_array_free(state);

    /* MambaRMSNormGated: rms_norm(silu(gate)*y, mixer.norm.weight, eps=layer_norm_epsilon); out_proj; + residuo */
    mlx_array gn=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &gn);
    { mlx_array g32=mlx_array_new(); if (mlx_astype(&g32, gn, MLX_FLOAT32, s)==0) { mlx_array_free(gn); gn=g32; } else mlx_array_free(g32); }
    mlx_array ow=mlx_array_new(), ows=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.out_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &ows, &ob);
    mlx_array yflat=mlx_array_new();
    if (ok) { int sh[3]={n_tokens, H, Dh}; mlx_array ysh=mlx_array_new(); ok = mlx_reshape(&ysh, ystack, sh, 3, s)==0; CHK(ok,"reshape y");
        int sh2[2]={n_tokens, Inner}; ok = ok && mlx_reshape(&yflat, ysh, sh2, 2, s)==0; mlx_array_free(ysh); }
    mlx_array sw=mlx_array_new();
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new();
        ok = mlx_sigmoid(&sg, gate, s)==0 && mlx_multiply(&sl, gate, sg, s)==0 && mlx_multiply(&sw, yflat, sl, s)==0;
        mlx_array_free(sg); mlx_array_free(sl); }
    if (ok) { mlx_array nn=mlx_array_new(); ok = mlx_fast_rms_norm(&nn, sw, gn, m->cfg.rms_norm_eps, s)==0; if (ok) mlx_array_set(&sw, nn); mlx_array_free(nn); CHK(ok,"rms gated"); }
    mlx_array oo = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&oo, sw, ow, ows, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok,"out_proj"); }
    if (ok) { ok = mlx_add(&*out_h, x, oo, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(nw); mlx_array_free(h); mlx_array_free(ipw); mlx_array_free(ips); mlx_array_free(ipb);
    mlx_array_free(proj); mlx_array_free(gate); mlx_array_free(conv_in); mlx_array_free(dt); mlx_array_free(cw);
    mlx_array_free(cbias); mlx_array_free(cstate); mlx_array_free(cf); mlx_array_free(cout); mlx_array_free(xc);
    mlx_array_free(hidden); mlx_array_free(Bv); mlx_array_free(Cv); mlx_array_free(Alog); mlx_array_free(Aneg);
    mlx_array_free(Dv); mlx_array_free(dtb); mlx_array_free(hd); mlx_array_free(bg); mlx_array_free(cg);
    mlx_array_free(Bt); mlx_array_free(Ct); mlx_array_free(ystack); mlx_array_free(gn); mlx_array_free(yflat);
    mlx_array_free(sw); mlx_array_free(ow); mlx_array_free(ows); mlx_array_free(ob); mlx_array_free(oo);
    return ok;
#undef CHK
}
