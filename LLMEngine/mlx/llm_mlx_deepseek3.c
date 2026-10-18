/* ============================================================================
 * llm_mlx_deepseek3.c — dominio DeepSeek-V3 / Kimi-K2 (MLA ASSORBITA, "deepseek_v3"/"kimi_k2").
 *
 * mlx_lm `deepseek_v3` usa l'MLA **assorbita**: `q_lora` (q_a_proj->q_a_layernorm->q_b_proj),
 * `kv_a_proj_with_mqa` -> compressed(kvl)/k_pe(qr), `kv_a_layernorm`, poi ASSORBIMENTO con
 * `embed_q`/`unembed_out` (MultiLinear per-head [H,out,in]) + `pe_scores` additivi + RoPE yarn
 * (traditional=true, dims=qk_rope) su q_pe/k_pe. Cache: kv_latent [1,1,S,kvl] in k_cache,
 * k_pe [1,1,S,qr] in aux_cache[layer]. (Contrapposto al deepseek_v2 NON assorbito, kv_b_proj.)
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

static bool mla_linear3d(mlx_array* res, mlx_array x, mlx_array w, mlx_array sc, mlx_array bi,
                         bool transpose, int gsz, int bts, const char* mode, mlx_stream st) {
    if (sc.ctx != NULL) {
        mlx_optional_int o1 = { gsz, true }, o2 = { bts, true };
        return mlx_quantized_matmul(res, x, w, sc, bi, transpose, o1, o2, mode, st) == 0;
    }
    if (!transpose) return mlx_matmul(res, x, w, st) == 0;
    int nd = (int)mlx_array_ndim(w); int perm[8]; for (int i=0;i<nd;i++) perm[i]=i;
    int t = perm[nd-1]; perm[nd-1] = perm[nd-2]; perm[nd-2] = t;
    mlx_array wt = mlx_array_new();
    bool ok = mlx_transpose_axes(&wt, w, perm, nd, st) == 0 && mlx_matmul(res, x, wt, st) == 0;
    mlx_array_free(wt);
    return ok;
}

bool deepseek_v3_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens,
                                 mlx_array x, mlx_array* k_cache, mlx_array* v_cache,
                                 bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int H = m->cfg.n_heads;
    const int qn = m->cfg.qk_nope_head_dim, qr = m->cfg.qk_rope_head_dim;
    const int vhd = m->cfg.v_head_dim, kvl = m->cfg.kv_lora_rank;
    const int qh = qn + qr;
    const char* mode = m->cfg.quant_mode;
    const float scale = 1.0f / sqrtf((float)qh);
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "dsv3[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array ln=mlx_array_new(), aw=mlx_array_new(), as=mlx_array_new(), ab=mlx_array_new(), aln=mlx_array_new();
    mlx_array ew=mlx_array_new(), es=mlx_array_new(), eb=mlx_array_new();
    mlx_array uw=mlx_array_new(), us=mlx_array_new(), ub=mlx_array_new();
    mlx_array ow=mlx_array_new(), os=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &ln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_proj_with_mqa", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &aw,&as,&ab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &aln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.embed_q", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ew,&es,&eb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.unembed_out", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &uw,&us,&ub);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os,&ob);
    CHK(ok, "lookup");

    mlx_array normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&normed, x, ln, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm");

    /* Q: q_lora (q_a_proj -> q_a_layernorm -> q_b_proj) oppure q_proj. */
    mlx_array q=mlx_array_new();
    if (ok && m->cfg.q_lora_rank > 0) {
        mlx_array qaw=mlx_array_new(), qas=mlx_array_new(), qab=mlx_array_new(), qaln=mlx_array_new();
        mlx_array qbw=mlx_array_new(), qbs=mlx_array_new(), qbb=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_a_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qaw,&qas,&qab);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_a_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &qaln);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qbw,&qbs,&qbb);
        if (ok) { mlx_array qa=mlx_array_new(), qan=mlx_array_new();
                  ok = quantized_linear_cfg(&qa, normed, qaw,qas,qab, gsz,bts, mode, s) &&
                       rms_norm_cfg(&qan, qa, qaln, m->cfg.rms_norm_eps, s) &&
                       quantized_linear_cfg(&q, qan, qbw,qbs,qbb, gsz,bts, mode, s);
                  mlx_array_free(qa); mlx_array_free(qan); }
        mlx_array_free(qaw);mlx_array_free(qas);mlx_array_free(qab);mlx_array_free(qaln);
        mlx_array_free(qbw);mlx_array_free(qbs);mlx_array_free(qbb);
    } else if (ok) {
        mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
        if (ok) ok = quantized_linear_cfg(&q, normed, qw,qs,qb, gsz,bts, mode, s);
        mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);
    }
    CHK(ok, "q proj");

    mlx_array q4=mlx_array_new(), qt=mlx_array_new(), q_nope=mlx_array_new(), q_pe=mlx_array_new();
    if (ok) { int sh[4]={1,n_tokens,H,qh}; ok = mlx_reshape(&q4, q, sh, 4, s)==0; }
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&qt, q4, perm, 4, s)==0; }
    if (ok) { int _sec[1]={qn}; mlx_vector_array sp=mlx_vector_array_new();
              ok = mlx_split_sections(&sp, qt, _sec, 1, -1, s)==0 &&
                   mlx_vector_array_get(&q_nope, sp, 0)==0 && mlx_vector_array_get(&q_pe, sp, 1)==0;
              mlx_vector_array_free(sp); }
    CHK(ok, "q split");

    /* KV: compressed [1,L,kvl] / k_pe [1,1,L,qr]; kv_latent = rms(compressed). */
    mlx_array ckv=mlx_array_new(), ckv3=mlx_array_new(), compressed=mlx_array_new(), kpe3=mlx_array_new(), kpe=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&ckv, normed, aw,as,ab, gsz,bts, mode, s);
    if (ok) { int sh[3]={1,n_tokens,kvl+qr}; ok = mlx_reshape(&ckv3, ckv, sh, 3, s)==0; }
    if (ok) { int _sec[1]={kvl}; mlx_vector_array sp=mlx_vector_array_new();
              ok = mlx_split_sections(&sp, ckv3, _sec, 1, -1, s)==0 &&
                   mlx_vector_array_get(&compressed, sp, 0)==0 && mlx_vector_array_get(&kpe3, sp, 1)==0;
              mlx_vector_array_free(sp); }
    if (ok) { int sh[4]={1,1,n_tokens,qr}; ok = mlx_reshape(&kpe, kpe3, sh, 4, s)==0; }
    mlx_array kvl_lat=mlx_array_new(), kvl4=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&kvl_lat, compressed, aln, m->cfg.rms_norm_eps, s);
    if (ok) { int sh[4]={1,1,n_tokens,kvl}; ok = mlx_reshape(&kvl4, kvl_lat, sh, 4, s)==0; }
    CHK(ok, "kv_a");

    /* RoPE yarn (traditional=true, dims=qr) su q_pe/k_pe. */
    mlx_array q_pe_r=mlx_array_new(), kpe_r=mlx_array_new();
    if (ok) {
        mlx_array freqs = llm_mlx_yarn_rope_freqs_cached(qr, m->cfg.rope_theta, m->cfg.yarn_factor,
                                                          m->cfg.yarn_orig_max, m->cfg.yarn_beta_fast, m->cfg.yarn_beta_slow);
        mlx_optional_float no_base = {0.0f, false};
        ok = freqs.ctx != NULL &&
             mlx_fast_rope(&q_pe_r, q_pe, qr, true, no_base, 1.0f, cache_offset, freqs, s)==0 &&
             mlx_fast_rope(&kpe_r, kpe, qr, true, no_base, 1.0f, cache_offset, freqs, s)==0;
        float num = llm_mlx_yarn_mscale(m->cfg.yarn_factor, m->cfg.yarn_mscale);
        float den = llm_mlx_yarn_mscale(m->cfg.yarn_factor, m->cfg.yarn_mscale_all_dim);
        float msc = (den > 1e-9f) ? num/den : 1.0f;
        if (ok && msc != 1.0f) {
            mlx_array sc = mlx_array_new_float32(msc), q2=mlx_array_new(), k2=mlx_array_new();
            ok = mlx_multiply(&q2, q_pe_r, sc, s)==0 && mlx_multiply(&k2, kpe_r, sc, s)==0;
            if (ok) { mlx_array_set(&q_pe_r, q2); mlx_array_set(&kpe_r, k2); }
            mlx_array_free(sc); mlx_array_free(q2); mlx_array_free(k2);
        }
    }
    CHK(ok, "rope");

    /* Cache: kv_latent in k_cache, k_pe in aux_cache[layer]. */
    mlx_array k_full=mlx_array_new(), aux_full=mlx_array_new();
    if (ok) {
        if (has_cache && (*k_cache).ctx != NULL) { mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*k_cache); mlx_vector_array_append_value(v,kvl4);
            ok = mlx_concatenate_axis(&k_full, v, 2, s)==0; mlx_vector_array_free(v); }
        else ok = mlx_array_set(&k_full, kvl4)==0;
    }
    mlx_array* ac = (m->aux_cache && layer_idx < m->cfg.n_layers) ? &m->aux_cache[layer_idx] : NULL;
    if (ok) {
        if (ac && (*ac).ctx != NULL) { mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*ac); mlx_vector_array_append_value(v,kpe_r);
            ok = mlx_concatenate_axis(&aux_full, v, 2, s)==0; mlx_vector_array_free(v); }
        else ok = mlx_array_set(&aux_full, kpe_r)==0;
        if (ok && ac) ok = mlx_array_set(ac, aux_full)==0;
    }
    if (ok) ok = mlx_array_set(k_cache, k_full)==0;
    CHK(ok, "cache");

    /* pe_scores = (q_pe*scale) @ k_pe^T + maschera causale additiva. */
    mlx_array spe=mlx_array_new(), spe_m=mlx_array_new();
    if (ok) {
        mlx_array kc = (ac && (*ac).ctx != NULL) ? *ac : kpe_r;
        int nd=(int)mlx_array_ndim(kc); int perm[8]; for(int i=0;i<nd;i++)perm[i]=i; int t=perm[nd-1];perm[nd-1]=perm[nd-2];perm[nd-2]=t;
        mlx_array kpe_ct=mlx_array_new();
        ok = mlx_transpose_axes(&kpe_ct, kc, perm, nd, s)==0;
        mlx_array sf = mlx_array_new_float32(scale), sc = mlx_array_new(), qpes = mlx_array_new();
        if (ok) ok = mlx_astype(&sc, sf, mlx_array_dtype(q_pe_r), s)==0;
        if (ok) ok = mlx_multiply(&qpes, q_pe_r, sc, s)==0;
        if (ok) ok = mlx_matmul(&spe, qpes, kpe_ct, s)==0;
        mlx_array_free(sf); mlx_array_free(sc); mlx_array_free(qpes); mlx_array_free(kpe_ct);
        /* maschera additiva [1,1,L,S] (0 se j<=cache_offset+i, else -inf) */
        int S = (int)mlx_array_shape(*k_cache)[2];
        mlx_array iq=mlx_array_new(), ik=mlx_array_new(), iqe=mlx_array_new(), ike=mlx_array_new(), cmp=mlx_array_new(), addmask=mlx_array_new();
        mlx_arange(&iq, 0, n_tokens, 1, MLX_INT32, s);
        mlx_arange(&ik, 0, S, 1, MLX_INT32, s);
        mlx_array coff = mlx_array_new_int(cache_offset), ip=mlx_array_new();
        if (ok) ok = mlx_add(&ip, iq, coff, s)==0; mlx_array_free(coff);
        if (ok) { int sh[2]={n_tokens,1}; ok = mlx_reshape(&iqe, ip, sh, 2, s)==0; }
        if (ok) { int sh[2]={1,S}; ok = mlx_reshape(&ike, ik, sh, 2, s)==0; }
        if (ok) ok = mlx_greater_equal(&cmp, iqe, ike, s)==0;
        mlx_array zerof=mlx_array_new_float32(0.0f), ninff=mlx_array_new_float32(-INFINITY), zero=mlx_array_new(), ninf=mlx_array_new();
        if (ok) ok = mlx_astype(&zero, zerof, mlx_array_dtype(spe), s)==0 && mlx_astype(&ninf, ninff, mlx_array_dtype(spe), s)==0;
        if (ok) ok = mlx_where(&addmask, cmp, zero, ninf, s)==0;
        mlx_array_free(zerof);mlx_array_free(ninff);mlx_array_free(zero);mlx_array_free(ninf);
        if (ok) { int sh[4]={1,1,n_tokens,S}; ok = mlx_reshape(&addmask, addmask, sh, 4, s)==0; }
        if (ok) ok = mlx_add(&spe_m, spe, addmask, s)==0;
        mlx_array_free(iq);mlx_array_free(ik);mlx_array_free(iqe);mlx_array_free(ike);mlx_array_free(cmp);mlx_array_free(ip);mlx_array_free(addmask);
    }
    CHK(ok, "pe_scores");

    /* SDPA assorbita. */
    mlx_array attn=mlx_array_new(), ka=mlx_array_new(), va=mlx_array_new(), out_attn=mlx_array_new();
    mlx_array no_sinks = mlx_array_new();
    if (ok && n_tokens == 1) {
        ok = mla_linear3d(&ka, q_nope, ew, es, eb, true, gsz, bts, mode, s);
        if (ok) ok = mlx_fast_scaled_dot_product_attention(&attn, ka, *k_cache, *k_cache, scale, "", spe_m, no_sinks, s)==0;
        if (ok) ok = mla_linear3d(&out_attn, attn, uw, us, ub, true, gsz, bts, mode, s);
    } else if (ok) {
        ok = mla_linear3d(&ka, kvl4, ew, es, eb, false, gsz, bts, mode, s);
        if (ok) ok = mla_linear3d(&va, kvl4, uw, us, ub, true, gsz, bts, mode, s);
        if (ok) ok = mlx_fast_scaled_dot_product_attention(&attn, q_nope, ka, va, scale, "", spe_m, no_sinks, s)==0;
        if (ok) ok = mlx_array_set(&out_attn, attn)==0;
    }
    mlx_array_free(no_sinks);
    CHK(ok, "sdpa absorbed");

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new(), proj=mlx_array_new();
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, out_attn, perm, 4, s)==0; }
    if (ok) { int sh[3]={1,n_tokens,H*vhd}; ok = mlx_reshape(&attn_flat, attn_bt, sh, 3, s)==0; }
    if (ok) ok = quantized_linear_cfg(&proj, attn_flat, ow,os,ob, gsz,bts, mode, s);
    if (ok) ok = mlx_add(out_h, x, proj, s)==0;
    CHK(ok, "o_proj");

    mlx_array all[] = { ln,aw,as,ab,aln, ew,es,eb, uw,us,ub, ow,os,ob, normed, q, q4,qt, q_nope,q_pe,
                        ckv,ckv3, compressed,kpe3,kpe, kvl_lat,kvl4, q_pe_r,kpe_r, k_full, aux_full,
                        spe, spe_m, attn, ka, va, out_attn, attn_bt, attn_flat, proj };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
