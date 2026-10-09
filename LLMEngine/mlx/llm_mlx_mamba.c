/* ============================================================================
 * llm_mlx_mamba.c — dominio Mamba (SSM selettivo), arch ricorrente.
 *   ResidualBlock:  out = MambaBlock(RMSNorm(x)) + x
 *   MambaBlock:     xz = in_proj(x); x,z = split; conv1d(depthwise, K) con cache;
 *                   x = silu(conv); scan selettivo per-token; out = out_proj(swiglu(z,y)).
 * Modellato 1:1 su mlx_lm/models/mamba.py. Lo slot attn fa norm+mixer+residuo,
 * lo slot ffn e' IDENTITA'.
 * NOTA PERF: lo scan e' per-token (T iterazioni di grafo) -> correttezza prima;
 * vettorizzazione/kernel in un secondo momento.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool mamba_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool mamba_ffn_identity(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

bool mamba_ffn_identity(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    (void)m; (void)layer_idx; (void)n_tokens;
    return mlx_array_set(out, h) == 0;
}


bool mamba_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int Hid = m->cfg.hidden_size;
    const int Inner = m->cfg.moe_intermediate_size > 0 ? m->cfg.moe_intermediate_size : 2 * Hid;
    const int St = m->cfg.mamba_state_size > 0 ? m->cfg.mamba_state_size : 16;
    const int Kc = m->cfg.linear_conv_kernel_dim > 0 ? m->cfg.linear_conv_kernel_dim : 4;
    const int Rk = m->cfg.mamba_dt_rank > 0 ? m->cfg.mamba_dt_rank : (Hid + 15) / 16;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "mamba[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    /* norm */
    mlx_array nw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &nw);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, nw, m->cfg.rms_norm_eps, s); CHK(ok, "norm"); }

    /* in_proj -> x, z */
    mlx_array ipw = mlx_array_new(), ips = mlx_array_new(), ipb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.in_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ipw, &ips, &ipb);
    mlx_array xz = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&xz, h, ipw, ips, ipb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj"); }
    mlx_array xq = mlx_array_new(), zz = mlx_array_new();
    if (ok) { int sec[1] = { Inner }; mlx_vector_array v = mlx_vector_array_new();
        ok = mlx_split_sections(&v, xz, sec, 1, -1, s) == 0;
        if (ok) ok = mlx_vector_array_get(&xq, v, 0) == 0 && mlx_vector_array_get(&zz, v, 1) == 0;
        mlx_vector_array_free(v); CHK(ok, "split xz"); }

    /* conv1d depthwise con cache [1,Kc-1,Inner] (v_cache) */
    mlx_array cw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.conv1d.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cw);
    { mlx_array c32 = mlx_array_new(); if (mlx_astype(&c32, cw, MLX_FLOAT32, s) == 0) { mlx_array_free(cw); cw = c32; } else mlx_array_free(c32); }
    /* MLX conv e' channels-last: kernel atteso (out, K, in/groups). Due layout nei checkpoint:
     *   - HF/mamba-130m: (d_inner, 1, K)     -> trasponi a (d_inner, K, 1);
     *   - falcon-mamba:  gia' (d_inner, K, 1) -> nessuna trasposizione. */
    if (mlx_array_ndim(cw) == 3 && mlx_array_shape(cw)[1] == 1) {
        mlx_array cwt = mlx_array_new(); int ax[3] = {0, 2, 1};
        if (mlx_transpose_axes(&cwt, cw, ax, 3, s) == 0) { mlx_array_free(cw); cw = cwt; } else mlx_array_free(cwt);
    }
    mlx_array cb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.conv1d.bias", nb, sizeof(nb));
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
    if (ok) { mlx_array c2 = mlx_array_new(); ok = mlx_add(&c2, cout, cb, s) == 0; if (ok) mlx_array_set(&cout, c2); mlx_array_free(c2); CHK(ok, "conv+bias"); }
    mlx_array xc = mlx_array_new();
    if (ok) { mlx_array sig = mlx_array_new(); ok = mlx_sigmoid(&sig, cout, s) == 0 && mlx_multiply(&xc, cout, sig, s) == 0; mlx_array_free(sig); CHK(ok, "silu"); }

    /* x_proj ; dt_proj ; A_log ; D */
    mlx_array xpw = mlx_array_new(), xps = mlx_array_new(), xpb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.x_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &xpw, &xps, &xpb);
    mlx_array dtw = mlx_array_new(), dtsc = mlx_array_new(), dtb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.dt_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &dtw, &dtsc, &dtb);
    mlx_array Alog = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.A_log", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &Alog);
    mlx_array Dv = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.D", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &Dv);
    /* A = -exp(A_log)  [Inner,St] */
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
        /* xt = xc[:, t]  [1,Inner] */
        mlx_array xt = mlx_array_new();
        { int ss[3]={0,t,0}, se[3]={1,t+1,Inner}, st[3]={1,1,1};
          ok = mlx_slice(&xt, xc, ss,3,se,3,st,3,s) == 0; if (ok) ok = mlx_squeeze_axis(&xt, xt, 1, s) == 0; }
        /* dbc = x_proj(xt) [1, Rk+2St] -> delta [1,Rk], B [1,St], C [1,St] */
        mlx_array dbc = mlx_array_new();
        if (ok) ok = quantized_linear_cfg(&dbc, xt, xpw, xps, xpb, gsz, bts, m->cfg.quant_mode, s);
        mlx_array dlt = mlx_array_new(), Bv = mlx_array_new(), Cv = mlx_array_new();
        if (ok) { int sec[2] = { Rk, Rk + St }; mlx_vector_array v = mlx_vector_array_new();
            ok = mlx_split_sections(&v, dbc, sec, 2, -1, s) == 0;
            if (ok) ok = mlx_vector_array_get(&dlt,v,0)==0 && mlx_vector_array_get(&Bv,v,1)==0 && mlx_vector_array_get(&Cv,v,2)==0;
            mlx_vector_array_free(v); }
        /* delta = softplus(dt_proj(dlt)) */
        mlx_array dlin = mlx_array_new();
        if (ok) ok = quantized_linear_cfg(&dlin, dlt, dtw, dtsc, dtb, gsz, bts, m->cfg.quant_mode, s);
        if (ok) { int zs[1] = {1}; mlx_array zero = mlx_array_new(); mlx_zeros(&zero, zs, 1, MLX_FLOAT32, s);
            mlx_array sp = mlx_array_new(); ok = mlx_logaddexp(&sp, dlin, zero, s) == 0; if (ok) mlx_array_set(&dlt, sp);
            mlx_array_free(zero); mlx_array_free(sp); }
        /* new_state = (dlt(x)xt)(x)B + state*exp(dlt(x)A) ; y = new_state@C + D*xt */
        mlx_array de = mlx_array_new(), xe = mlx_array_new(), Be = mlx_array_new(), Ce = mlx_array_new();
        mlx_expand_dims(&de, dlt, -1, s); mlx_expand_dims(&xe, xt, -1, s); mlx_expand_dims(&Be, Bv, 1, s); mlx_expand_dims(&Ce, Cv, -1, s);
        mlx_array ns1 = mlx_array_new(); mlx_multiply(&ns1, de, xe, s);
        mlx_array ns = mlx_array_new(); mlx_multiply(&ns, ns1, Be, s);
        mlx_array dA = mlx_array_new(); mlx_multiply(&dA, de, Aneg, s);
        mlx_array eA = mlx_array_new(); mlx_exp(&eA, dA, s);
        mlx_array st2 = mlx_array_new(); mlx_multiply(&st2, state, eA, s);
        mlx_array nsf = mlx_array_new(); mlx_add(&nsf, ns, st2, s);
        mlx_array_set(&state, nsf);
        mlx_array yp = mlx_array_new(); mlx_matmul(&yp, nsf, Ce, s);          /* [1,Inner,1] */
        mlx_array ys = mlx_array_new(); mlx_squeeze_axis(&ys, yp, 2, s);       /* [1,Inner] */
        mlx_array dx = mlx_array_new(); mlx_multiply(&dx, Dv, xt, s);
        mlx_array yt = mlx_array_new(); mlx_add(&yt, ys, dx, s);
        mlx_array ye = mlx_array_new(); mlx_expand_dims(&ye, yt, 1, s);        /* [1,1,Inner] */
        if (!yset) { mlx_array_set(&ystack, ye); yset = true; }
        else { mlx_array cc = mlx_array_new(); mlx_array arr[2] = { ystack, ye };
               mlx_vector_array v = mlx_vector_array_new_data(arr,2); mlx_concatenate_axis(&cc, v, 1, s);
               mlx_vector_array_free(v); mlx_array_set(&ystack, cc); mlx_array_free(cc); }
        mlx_array_free(xt); mlx_array_free(dbc); mlx_array_free(dlt); mlx_array_free(Bv); mlx_array_free(Cv);
        mlx_array_free(dlin); mlx_array_free(de); mlx_array_free(xe); mlx_array_free(Be); mlx_array_free(Ce);
        mlx_array_free(ns1); mlx_array_free(ns); mlx_array_free(dA); mlx_array_free(eA); mlx_array_free(st2);
        mlx_array_free(nsf); mlx_array_free(yp); mlx_array_free(ys); mlx_array_free(dx); mlx_array_free(yt); mlx_array_free(ye);
    }
    if (ok) mlx_array_set(k_cache, state);

    /* out = out_proj(swiglu(z, y)); swiglu(a,g)=a*silu(g) ; return out + x */
    mlx_array ow = mlx_array_new(), ows = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.out_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &ows, &ob);
    mlx_array sw = mlx_array_new();
    /* mixer(gate) = out_proj(y * silu(z))  (mlx_lm: nn.silu(z) NON silu(y)) */
    if (ok) { mlx_array sg = mlx_array_new(), sl = mlx_array_new();
        ok = mlx_sigmoid(&sg, zz, s)==0 && mlx_multiply(&sl, zz, sg, s)==0 && mlx_multiply(&sw, ystack, sl, s)==0;
        mlx_array_free(sg); mlx_array_free(sl); }
    mlx_array oo = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&oo, sw, ow, ows, ob, gsz, bts, m->cfg.quant_mode, s);
    if (ok) { ok = mlx_add(&*out_h, x, oo, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(nw); mlx_array_free(h); mlx_array_free(ipw); mlx_array_free(ips); mlx_array_free(ipb);
    mlx_array_free(xz); mlx_array_free(xq); mlx_array_free(zz); mlx_array_free(cw); mlx_array_free(cstate);
    mlx_array_free(cf); mlx_array_free(cout); mlx_array_free(xc); mlx_array_free(cb);
    mlx_array_free(xpw); mlx_array_free(xps); mlx_array_free(xpb); mlx_array_free(dtw); mlx_array_free(dtsc); mlx_array_free(dtb);
    mlx_array_free(Alog); mlx_array_free(Aneg); mlx_array_free(Dv); mlx_array_free(state); mlx_array_free(ystack);
    mlx_array_free(ow); mlx_array_free(ows); mlx_array_free(ob); mlx_array_free(sw); mlx_array_free(oo);
    return ok;
#undef CHK
}
