/* ============================================================================
 * llm_mlx_phi3.c — dominio Phi-3 (model_type "phi3").
 *
 * Struttura (mlx_lm `phi3`):
 *  - attn: `self_attn.qkv_proj` FUSO (op_size = n_heads*hd + 2*n_kv_heads*hd, layout [Q|K|V])
 *    + `self_attn.o_proj` (entrambi SENZA bias). RoPE **parziale** (traditional=cfg.rope_traditional,
 *    dims=int(hd*partial_rotary_factor)); SuScaledRoPE (longrope/su) NON coperto qui.
 *  - mlp: gate/up/down (swiglu) -> `dense_swiglu_ffn`.
 *  - norm: `input_layernorm` (attn), `post_attention_layernorm` (ffn) — RMSNorm.
 *  - residuo seriale: h = x + attn(ln(x)); out = h + mlp(post_ln(h)).
 * NB: `partial_rotary_factor` in quantized_linear_cfg e' 1.0 (riuso da `rotary_pct`) -> qui il
 * default cfg e' 1.0 (=rope pieno). TRAP: quantized_linear_cfg/llm_mlx_gqa_sdpa ritornano TRUE=ok.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool phi3_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "phi3_attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.qkv_proj", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "qkv_proj"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        mlx_vector_array parts = mlx_vector_array_new();
        int sec[2] = { nh*hd, nh*hd + nkv*hd };
        ok = mlx_split_sections(&parts, qkv, sec, 2, -1, s) == 0;
        if (ok && mlx_vector_array_size(parts) == 3) {
            mlx_array q=mlx_array_new(), k=mlx_array_new(), v=mlx_array_new();
            mlx_vector_array_get(&q, parts, 0); mlx_vector_array_get(&k, parts, 1); mlx_vector_array_get(&v, parts, 2);
            const int qs4[4]={1,n_tokens,nh,hd}, kvs4[4]={1,n_tokens,nkv,hd};
            mlx_array qh=mlx_array_new(), kh=mlx_array_new(), vh=mlx_array_new();
            ok = mlx_reshape(&qh, q, qs4,4, s)==0 && mlx_reshape(&kh, k, kvs4,4, s)==0 && mlx_reshape(&vh, v, kvs4,4, s)==0;
            if (ok) { const int perm[4]={0,2,1,3};
                      ok = mlx_transpose_axes(&q_t, qh, perm,4, s)==0 && mlx_transpose_axes(&k_t, kh, perm,4, s)==0 && mlx_transpose_axes(&v_t, vh, perm,4, s)==0; }
            mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh);
            mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
        } else ok = false;
        mlx_vector_array_free(parts);
        CHK(ok, "qkv split");
    }

    /* RoPE PARZIALE (dims=int(hd*partial_rotary_factor), traditional=cfg.rope_traditional). */
    mlx_array q_r=mlx_array_new(), k_r=mlx_array_new();
    if (ok) {
        float prf = m->cfg.partial_rotary_factor;
        int rd = (prf > 0.0f && prf < 1.0f) ? (int)(hd * prf) : hd;
        mlx_optional_float base = { m->cfg.rope_theta, true };
        mlx_array nf = mlx_array_new();
        ok = mlx_fast_rope(&q_r, q_t, rd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, s)==0 &&
             mlx_fast_rope(&k_r, k_t, rd, m->cfg.rope_traditional, base, 1.0f, cache_offset, nf, s)==0;
        mlx_array_free(nf); CHK(ok, "rope");
    }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache) llm_mlx_kv_trim(k_cache, v_cache, cache_offset, s);
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_r);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
            CHK(ok, "kv concat");
        } else { ok = mlx_array_set(&k_full, k_r)==0 && mlx_array_set(&v_full, v_t)==0; CHK(ok, "kv init"); }
    }
    if (ok && k_cache && v_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_r, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &osp, &osb);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj"); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, normed, qw,qs,qb, qkv, q_t,k_t,v_t, q_r,k_r, k_full,v_full, attn,
                        attn_bt, attn_flat, ow,osp,osb, attn_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool phi3_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "phi3_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "post_attention_layernorm"); }

    /* MLP fuso: gate_up_proj [2*inter] -> split gate/up -> swiglu -> down_proj */
    mlx_array gw=mlx_array_new(), gs=mlx_array_new(), gb=mlx_array_new(), gu=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_up_proj", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &gw, &gs, &gb);
    if (ok) { ok = quantized_linear_cfg(&gu, normed, gw, gs, gb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "gate_up_proj"); }
    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (ok) { int half = (int)mlx_array_shape(gu)[mlx_array_ndim(gu)-1] / 2;
              int sec[1] = { half }; mlx_vector_array sp=mlx_vector_array_new();
              ok = mlx_split_sections(&sp, gu, sec, 1, -1, s)==0;
              if (ok) ok = mlx_vector_array_get(&gate, sp, 0)==0 && mlx_vector_array_get(&up, sp, 1)==0;
              mlx_vector_array_free(sp); CHK(ok, "split gate_up"); }
    mlx_array sw=mlx_array_new(), dn=mlx_array_new();
    if (ok) { mlx_array sig=mlx_array_new(), silu=mlx_array_new();
              ok = mlx_sigmoid(&sig,gate,s)==0 && mlx_multiply(&silu,gate,sig,s)==0 && mlx_multiply(&sw,silu,up,s)==0;
              mlx_array_free(sig); mlx_array_free(silu); CHK(ok, "swiglu"); }
    mlx_array dw=mlx_array_new(), ds=mlx_array_new(), db=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &dw, &ds, &db);
    if (ok) { ok = quantized_linear_cfg(&dn, sw, dw, ds, db, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "down_proj"); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok, "residual (ffn)"); }

    mlx_array all[] = { lnw, normed, gw,gs,gb, gu, gate, up, sw, dn, dw,ds,db };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
