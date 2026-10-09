/* ============================================================================
 * llm_mlx_mamba.c — dominio Mamba (SSM selettivo), arch ricorrente.
 *   ResidualBlock:  out = MambaBlock(RMSNorm(x)) + x
 *   MambaBlock:     xz = in_proj(x); x,z = split; conv1d(depthwise, K) con cache;
 *                   x = silu(conv); scan selettivo per-token (delta/B/C da x_proj,
 *                   dt = softplus(dt_proj(delta)), A=-exp(A_log), D);
 *                   out = out_proj(swiglu(z, y)).
 * Modellato 1:1 su mlx_lm/models/mamba.py. Il "layer" e' tutto ricorrente: lo slot
 * attn fa norm+mixer+residuo, lo slot ffn e' IDENTITA'.
 * NB: da validare (richiede un mamba HF/safetensors; mamba-130m originale e' .bin).
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
    const int H = m->cfg.hidden_size, I = (m->cfg.moe_intermediate_size > 0 ? m->cfg.moe_intermediate_size : 2 * H);
    const int Ds = (m->cfg.mamba_state_size > 0 ? m->cfg.mamba_state_size : 16);
    const int K = (m->cfg.linear_conv_kernel_dim > 0 ? m->cfg.linear_conv_kernel_dim : 4);
    const int R = (m->cfg.mamba_dt_rank > 0 ? m->cfg.mamba_dt_rank : (H + 15) / 16);
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "mamba[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    /* norm (ResidualBlock.norm) */
    mlx_array nw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "norm.weight", nb, sizeof(nb));
    if (!multishard_get(&m->weights, nb, &nw)) { tname(m->cfg.layer_prefix, layer_idx, "mixer.norm.weight", nb, sizeof(nb)); multishard_get(&m->weights, nb, &nw); }
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, nw, m->cfg.rms_norm_eps, s); CHK(ok, "norm"); }

    mlx_array ipw = mlx_array_new(), ips = mlx_array_new(), ipb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.in_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ipw, &ips, &ipb);
    mlx_array xz = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&xz, h, ipw, ips, ipb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj"); }
    mlx_array xq = mlx_array_new(), z = mlx_array_new();
    if (ok) { int sec[1] = { I }; mlx_vector_array v = mlx_vector_array_new();
        ok = mlx_split_sections(&v, xz, sec, 1, -1, s) == 0;
        if (ok) ok = mlx_vector_array_get(&xq, v, 0) == 0 && mlx_vector_array_get(&z, v, 1) == 0;
        mlx_vector_array_free(v); CHK(ok, "split xz"); }

    /* conv1d depthwise con cache [B,K-1,I] (v_cache) */
    mlx_array cw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.conv1d.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &cw);
    { mlx_array c32 = mlx_array_new(); if (mlx_astype(&c32, cw, MLX_FLOAT32, s) == 0) { mlx_array_free(cw); cw = c32; } else mlx_array_free(c32); }
    mlx_array cstate = mlx_array_new();
    if (ok) {
        if (has_cache && mlx_array_size(*v_cache) > 0) mlx_array_set(&cstate, *v_cache);
        else { int cs[3] = { 1, K - 1, I }; mlx_zeros(&cstate, cs, 3, MLX_FLOAT32, s); }
    }
    mlx_array cf = mlx_array_new();
    if (ok) { mlx_array arr[2] = { cstate, xq }; mlx_vector_array v = mlx_vector_array_new_data(arr, 2);
        ok = mlx_concatenate_axis(&cf, v, 1, s) == 0; mlx_vector_array_free(v); CHK(ok, "conv concat"); }
    if (ok) { int ss[3] = {0, n_tokens, 0}, se[3] = {1, n_tokens + (K - 1), I}, st[3] = {1,1,1};
        mlx_array sl = mlx_array_new(); ok = mlx_slice(&sl, cf, ss, 3, se, 3, st, 3, s) == 0; if (ok) mlx_array_set(v_cache, sl); mlx_array_free(sl); }
    mlx_array cout = mlx_array_new();
    if (ok) { int str[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, id[1]={1};
        ok = mlx_conv_general(&cout, cf, cw, str,1,pl,1,ph,1,kd,1,id,1, I, false, s) == 0; CHK(ok, "conv1d"); }
    mlx_array xc = mlx_array_new();
    if (ok) { mlx_array sig = mlx_array_new(); ok = mlx_sigmoid(&sig, cout, s) == 0 && mlx_multiply(&xc, cout, sig, s) == 0; mlx_array_free(sig); CHK(ok, "silu"); }

    /* x_proj -> delta, B, C ; dt_proj ; A_log ; D */
    mlx_array xpw = mlx_array_new(), xps = mlx_array_new(), xpb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.x_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &xpw, &xps, &xpb);
    mlx_array dtw = mlx_array_new(), dtb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.dt_proj.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &dtw);
    tname(m->cfg.layer_prefix, layer_idx, "mixer.dt_proj.bias", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &dtb);
    mlx_array Alog = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.A_log", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &Alog);
    mlx_array Dv = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.D", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &Dv);

    /* scan per-token (state [1,I,Ds] in k_cache) */
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && mlx_array_size(*k_cache) > 0) mlx_array_set(&state, *k_cache);
        else { int st3[3] = { 1, I, Ds }; mlx_zeros(&state, st3, 3, MLX_FLOAT32, s); }
    }
    mlx_array ystack = mlx_array_new();
    for (int t = 0; t < n_tokens && ok; t++) {
        mlx_array xt = mlx_array_new(); { int ss[3]={0,t,0}, se[3]={1,t+1,I}, st[3]={1,1,1}; ok = mlx_slice(&xt, xc, ss,3,se,3,st,3,s) == 0; if (ok) ok = mlx_squeeze_axis(&xt, xt, 1, s) == 0; }
        mlx_array dbc = mlx_array_new();
        if (ok) { ok = quantized_linear_cfg(&dbc, xt, xpw, xps, xpb, gsz, bts, m->cfg.quant_mode, s); }
        mlx_array delta = mlx_array_new(), Bv = mlx_array_new(), Cv = mlx_array_new();
        if (ok) { int sec[2] = {R, R+Ds}; mlx_vector_array v = mlx_vector_array_new();
            ok = mlx_split_sections(&v, dbc, sec, 2, -1, s) == 0;
            if (ok) ok = mlx_vector_array_get(&delta,v,0)==0 && mlx_vector_array_get(&Bv,v,1)==0 && mlx_vector_array_get(&Cv,v,2)==0;
            mlx_vector_array_free(v); }
        mlx_array dlin = mlx_array_new();
        if (ok) { ok = quantized_linear_cfg(&dlin, delta, dtw, NULL, dtb, gsz, bts, m->cfg.quant_mode, s); }
        if (ok) {   /* delta = softplus(dt_proj(delta)) = logaddexp(x, 0) */
            mlx_array zero = mlx_array_new(); mlx_zeros(&zero, (int[]){1}, 1, MLX_FLOAT32, s);
            mlx_array sp = mlx_array_new(); ok = mlx_logaddexp(&sp, dlin, zero, s) == 0;
            if (ok) mlx_array_set(&delta, sp);
            mlx_array_free(zero); mlx_array_free(sp);
        }
        /* new_state = delta*x ⊗ B + state*exp(delta ⊗ A); y = new_state@C + D*x  (scan basico) */
        mlx_array Aneg = mlx_array_new(); if (ok) { mlx_array ea = mlx_array_new(); ok = mlx_exp(&ea, Alog, s)==0 && mlx_negative(&Aneg, ea, s)==0; mlx_array_free(ea); }
        mlx_array ys = mlx_array_new();
        if (ok) {   /* espansioni: delta [1,I]->[1,I,1]; B [1,Ds]->[1,1,Ds]; C [1,Ds]->[1,Ds,1] */
            mlx_array d_e = mlx_array_new(), B_e = mlx_array_new(), C_e = mlx_array_new();
            mlx_expand_dims(&d_e, delta, -1, s); mlx_expand_dims(&B_e, Bv, 1, s); mlx_expand_dims(&C_e, Cv, -1, s);
            mlx_array x_e = mlx_array_new(); mlx_expand_dims(&x_e, xt, -1, s);
            mlx_array ns1 = mlx_array_new(); mlx_multiply(&ns1, d_e, x_e, s);
            mlx_array ns = mlx_array_new(); mlx_multiply(&ns, ns1, B_e, s);
            mlx_array dA = mlx_array_new(); mlx_multiply(&dA, d_e, Aneg, s); mlx_array eA = mlx_array_new(); mlx_exp(&eA, dA, s);
            mlx_array st2 = mlx_array_new(); mlx_multiply(&st2, state, eA, s);
            mlx_array nsfin = mlx_array_new(); mlx_add(&nsfin, ns, st2, s);
            mlx_array_set(&state, nsfin);
            mlx_array yp = mlx_array_new(); mlx_matmul(&yp, nsfin, C_e, s);
            mlx_array ysq = mlx_array_new(); mlx_squeeze_axis(&ysq, yp, 2, s);
            mlx_array dxx = mlx_array_new(); mlx_multiply(&dxx, Dv, xt, s);
            mlx_add(&ys, ysq, dxx, s);
            mlx_array_free(d_e); mlx_array_free(B_e); mlx_array_free(C_e); mlx_array_free(x_e);
            mlx_array_free(ns1); mlx_array_free(ns); mlx_array_free(dA); mlx_array_free(eA); mlx_array_free(st2);
            mlx_array_free(nsfin); mlx_array_free(yp); mlx_array_free(ysq); mlx_array_free(dxx);
        }
        mlx_array ye = mlx_array_new(); mlx_expand_dims(&ye, ys, 1, s);
        if (t == 0) mlx_array_set(&ystack, ye); else { mlx_array cc = mlx_array_new(); mlx_array a[2] = { ystack, ye }; mlx_vector_array v = mlx_vector_array_new_data(a,2); mlx_concatenate_axis(&cc, v, 1, s); mlx_vector_array_free(v); mlx_array_set(&ystack, cc); mlx_array_free(cc); }
        mlx_array_free(xt); mlx_array_free(dbc); mlx_array_free(delta); mlx_array_free(Bv); mlx_array_free(Cv);
        mlx_array_free(dlin); mlx_array_free(Aneg); mlx_array_free(ys); mlx_array_free(ye);
    }
    if (ok) mlx_array_set(k_cache, state);

    /* out = out_proj(swiglu(z, y)) ; return out + x  (swiglu(a,g)=a*silu(g)) */
    mlx_array ow = mlx_array_new(), ows = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mixer.out_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &ows, &ob);
    mlx_array sw = mlx_array_new();
    if (ok) { mlx_array sig = mlx_array_new(), sil = mlx_array_new(); ok = mlx_sigmoid(&sig, ystack, s)==0 && mlx_multiply(&sil, ystack, sig, s)==0 && mlx_multiply(&sw, z, sil, s)==0; mlx_array_free(sig); mlx_array_free(sil); }
    mlx_array oo = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&oo, sw, ow, ows, ob, gsz, bts, m->cfg.quant_mode, s); }
    if (ok) { ok = mlx_add(&*out_h, x, oo, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(nw); mlx_array_free(h); mlx_array_free(ipw); mlx_array_free(ips); mlx_array_free(ipb);
    mlx_array_free(xz); mlx_array_free(xq); mlx_array_free(z); mlx_array_free(cw); mlx_array_free(cstate);
    mlx_array_free(cf); mlx_array_free(cout); mlx_array_free(xc);
    mlx_array_free(xpw); mlx_array_free(xps); mlx_array_free(xpb); mlx_array_free(dtw); mlx_array_free(dtb);
    mlx_array_free(Alog); mlx_array_free(Dv); mlx_array_free(state); mlx_array_free(ystack);
    mlx_array_free(ow); mlx_array_free(ows); mlx_array_free(ob); mlx_array_free(sw); mlx_array_free(oo);
    return ok;
#undef CHK
}
