/* ============================================================================
 * llm_mlx_hunyuan.c — dominio Hunyuan v1 DENSE ("hunyuan_v1_dense").
 *
 * Arch: RMSNorm + attenzione GQA con q/k/v/o BIAS + RoPE (traditional=false) + qk-norm
 * (`query_layernorm`/`key_layernorm`, RMSNorm su head_dim) applicata DOPO il rope.
 * FFN = dense_swiglu_ffn (input_layernorm/post_attention_layernorm + mlp.{gate,up,down}_proj),
 * residual STANDARD. Naming canonico `model.layers.*` / `model.embed_tokens` / `model.norm`.
 *
 * TRAP: quantized_linear_cfg / llm_mlx_gqa_sdpa ritornano bool TRUE=ok -> MAI `== 0`.
 * TRAP bias: `.bias` e' caricato dallo slot `biases` della triple (fallback) e applicato da
 * quantized_linear_cfg (mode none) -> NON ri-aggiungerlo.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool hunyuan_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "hunyuan[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    mlx_array kw=mlx_array_new(), ks=mlx_array_new(), kb=mlx_array_new();
    mlx_array vw=mlx_array_new(), vs=mlx_array_new(), vb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &kw,&ks,&kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &vw,&vs,&vb);
    mlx_array q0=mlx_array_new(), k0=mlx_array_new(), v0=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q0, normed, qw,qs,qb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "q_proj"); }
    if (ok) { ok = quantized_linear_cfg(&k0, normed, kw,ks,kb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "k_proj"); }
    if (ok) { ok = quantized_linear_cfg(&v0, normed, vw,vs,vb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "v_proj"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        const int qs4[4]={1,n_tokens,nh,hd}, kvs4[4]={1,n_tokens,nkv,hd};
        mlx_array qh=mlx_array_new(), kh=mlx_array_new(), vh=mlx_array_new();
        ok = mlx_reshape(&qh, q0, qs4,4, s)==0 && mlx_reshape(&kh, k0, kvs4,4, s)==0 && mlx_reshape(&vh, v0, kvs4,4, s)==0;
        if (ok) { const int perm[4]={0,2,1,3};
                  ok = mlx_transpose_axes(&q_t, qh, perm,4, s)==0 && mlx_transpose_axes(&k_t, kh, perm,4, s)==0 && mlx_transpose_axes(&v_t, vh, perm,4, s)==0; }
        mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh);
        CHK(ok, "head reshape/transpose");
    }

    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, hd, /*traditional=*/false, base, 1.0f, cache_offset, no_freqs, s)==0
          && mlx_fast_rope(&k_rope, k_t, hd, false, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs); CHK(ok, "rope");
    }

    /* qk-norm DOPO il rope (query_layernorm/key_layernorm, RMSNorm su head_dim). */
    mlx_array q_rope_n=mlx_array_new(), k_rope_n=mlx_array_new();
    if (ok) ok = mlx_array_set(&q_rope_n, q_rope)==0 && mlx_array_set(&k_rope_n, k_rope)==0;
    if (ok) {
        mlx_array qln=mlx_array_new(), kln=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.query_layernorm.weight", nb, sizeof(nb));
        bool have_qln = multishard_get(&m->weights, nb, &qln);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.key_layernorm.weight", nb, sizeof(nb));
        bool have_kln = multishard_get(&m->weights, nb, &kln);
        if (have_qln && have_kln) {
            mlx_array qn=mlx_array_new(), kn=mlx_array_new();
            ok = rms_norm_cfg(&qn, q_rope, qln, m->cfg.rms_norm_eps, s) && rms_norm_cfg(&kn, k_rope, kln, m->cfg.rms_norm_eps, s);
            if (ok) { mlx_array_set(&q_rope_n, qn); mlx_array_set(&k_rope_n, kn); }
            mlx_array_free(qn); mlx_array_free(kn);
            CHK(ok, "qk_norm");
        }
        mlx_array_free(qln); mlx_array_free(kln);
    }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_rope_n);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
            CHK(ok, "kv concat");
        } else { ok = mlx_array_set(&k_full, k_rope_n)==0 && mlx_array_set(&v_full, v_t)==0; CHK(ok, "kv init"); }
    }
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope_n, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&osp,&osb);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj"); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, normed, qw,qs,qb, kw,ks,kb, vw,vs,vb, q0, k0, v0, q_t, k_t, v_t,
                        q_rope, k_rope, q_rope_n, k_rope_n, k_full, v_full, attn, attn_bt, attn_flat,
                        ow, osp, osb, attn_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
