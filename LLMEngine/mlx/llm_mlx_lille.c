/* ============================================================================
 * llm_mlx_lille.c — dominio Lille-130m (Apple, "lille-130m").
 *
 * Arch: RMSNorm + ATTENZIONE a QKV FUSO (`attention.qkv_proj`) + RoPE TRADITIONAL +
 * GQA; FFN SwiGLU con norm dedicata (`feed_forward.norm`).
 * Naming: `transformer.layers.{l}.{attention,feed_forward}.*`; embed = transformer.tok_embeddings;
 * norm finale = transformer.norm (tie). Ref: mlx_lm/models/lille-130m.py.
 *
 * TRAP: `quantized_linear_cfg`/`rms_norm_cfg`/`llm_mlx_gqa_sdpa` ritornano bool TRUE=ok:
 * MAI confrontarli con `== 0` (inverte il risultato).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool lille_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "lille_attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "attention.norm"); }

    /* QKV fuso: un solo linear -> [1, T, (nh+2*nkv)*hd]. */
    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.qkv_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "qkv_proj matmul"); }

    /* Split dell'OUTPUT q|k|v lungo l'ultimo asse (sezioni disuguali). */
    const int q_size = nh * hd, kv_size = nkv * hd;
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) {
        int idxs[2] = { q_size, q_size + kv_size };
        mlx_vector_array parts = mlx_vector_array_new();
        ok = mlx_split_sections(&parts, qkv, idxs, 2, -1, s) == 0;
        if (ok && mlx_vector_array_size(parts) == 3) {
            mlx_vector_array_get(&q, parts, 0);
            mlx_vector_array_get(&k, parts, 1);
            mlx_vector_array_get(&v, parts, 2);
        } else ok = false;
        mlx_vector_array_free(parts);
        CHK(ok, "qkv split");
    }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        const int qs4[4]={1,n_tokens,nh,hd}, kvs4[4]={1,n_tokens,nkv,hd};
        mlx_array qh=mlx_array_new(), kh=mlx_array_new(), vh=mlx_array_new();
        ok = mlx_reshape(&qh, q, qs4,4, s)==0 && mlx_reshape(&kh, k, kvs4,4, s)==0 && mlx_reshape(&vh, v, kvs4,4, s)==0;
        if (ok) { const int perm[4]={0,2,1,3};
                  ok = mlx_transpose_axes(&q_t, qh, perm,4, s)==0 && mlx_transpose_axes(&k_t, kh, perm,4, s)==0 && mlx_transpose_axes(&v_t, vh, perm,4, s)==0; }
        mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh);
        CHK(ok, "head reshape/transpose");
    }

    /* RoPE TRADITIONAL (rotate_half) su head_dim pieno. */
    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, hd, /*traditional=*/true, base, 1.0f, cache_offset, no_freqs, s)==0
          && mlx_fast_rope(&k_rope, k_t, hd, true, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs);
        CHK(ok, "rope");
    }

    /* KV cache (concat; append-only). */
    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_rope);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
            CHK(ok, "kv concat");
        } else {
            ok = mlx_array_set(&k_full, k_rope)==0 && mlx_array_set(&v_full, v_t)==0;
            CHK(ok, "kv init");
        }
    }
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, 1.0f/sqrtf((float)hd), /*window=*/0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), os=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.out_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &os, &ob);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, os, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "out_proj"); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, normed, qw, qs, qb, qkv, q, k, v, q_t, k_t, v_t, q_rope, k_rope,
                        k_full, v_full, attn, attn_bt, attn_flat, ow, os, ob, attn_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool lille_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "lille_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "feed_forward.norm"); }

    mlx_array gw=mlx_array_new(),gs=mlx_array_new(),gb=mlx_array_new();
    mlx_array uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new();
    mlx_array dw=mlx_array_new(),ds=mlx_array_new(),db=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.gate_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &gw,&gs,&gb);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.up_proj", nb, sizeof(nb));   ok = ok && multishard_get_triple(&m->weights, nb, &uw,&us,&ub);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.down_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &dw,&ds,&db);
    CHK(ok, "ffn lookup");

    mlx_array g=mlx_array_new(), u=mlx_array_new(), sw=mlx_array_new(), dn=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&g, normed, gw,gs,gb, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "gate");
    if (ok) ok = quantized_linear_cfg(&u, normed, uw,us,ub, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "up");
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok = mlx_sigmoid(&sg,g,s)==0 && mlx_multiply(&sl,g,sg,s)==0 && mlx_multiply(&sw,sl,u,s)==0;
              mlx_array_free(sg); mlx_array_free(sl); }
    if (ok) ok = quantized_linear_cfg(&dn, sw, dw,ds,db, gsz,bts, m->cfg.quant_mode, s); CHK(ok, "down");
    if (ok) ok = mlx_add(out, h, dn, s)==0; CHK(ok, "residual (ffn)");

    mlx_array all[] = { lnw, normed, gw,gs,gb, uw,us,ub, dw,ds,db, g,u,sw,dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
