/* ============================================================================
 * llm_mlx_openelm.c — dominio OpenELM ("openelm").
 *
 * Arch: RMSNorm + attenzione GQA con QKV FUSO (`attn.qkv_proj`, layout per-head
 * [n_heads+2*n_kv_heads, head_dim] -> transpose -> split su asse 1) + qk-norm opzionale
 * (`attn.q_norm`/`k_norm`, RMSNorm su head_dim) + RoPE (traditional=false, base=rope_freq_constant);
 * MLP a 2 proiezioni FUSE (`ffn.proj_1` [dim,2*inter] -> split gate/up -> silu(gate)*up -> `ffn.proj_2`).
 * **n_heads/n_kv_heads sono PER-LAYER** (liste num_query_heads/num_kv_heads in cfg.layer_n_heads/kv).
 * Residual STANDARD. Naming `transformer.layers.*`; embed `transformer.token_embeddings`; norm
 * `transformer.norm`.
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

bool openelm_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh  = (m->cfg.layer_n_heads    && layer_idx < m->cfg.n_layers) ? m->cfg.layer_n_heads[layer_idx]    : m->cfg.n_heads;
    const int nkv = (m->cfg.layer_n_kv_heads && layer_idx < m->cfg.n_layers) ? m->cfg.layer_n_kv_heads[layer_idx] : m->cfg.n_kv_heads;
    const int hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "openelm[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn_norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "attn_norm"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn.qkv_proj", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "qkv_proj"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        const int sh4[4] = {1, n_tokens, nh + 2*nkv, hd};
        mlx_array qkv4 = mlx_array_new(), qkv_t = mlx_array_new();
        ok = mlx_reshape(&qkv4, qkv, sh4, 4, s)==0;
        if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&qkv_t, qkv4, perm,4, s)==0; }
        if (ok) {
            int idxs[2] = { nh, nh + nkv };
            mlx_vector_array parts = mlx_vector_array_new();
            ok = mlx_split_sections(&parts, qkv_t, idxs, 2, 1, s) == 0;
            if (ok && mlx_vector_array_size(parts) == 3) {
                mlx_vector_array_get(&q_t, parts, 0);
                mlx_vector_array_get(&k_t, parts, 1);
                mlx_vector_array_get(&v_t, parts, 2);
            } else ok = false;
            mlx_vector_array_free(parts);
        }
        mlx_array_free(qkv4); mlx_array_free(qkv_t);
        CHK(ok, "qkv reshape/split");
    }

    /* qk-norm opzionale (RMSNorm su head_dim) PRIMA del rope. */
    mlx_array q_n=mlx_array_new(), k_n=mlx_array_new();
    if (ok) ok = mlx_array_set(&q_n, q_t)==0 && mlx_array_set(&k_n, k_t)==0;
    if (ok) {
        mlx_array qln=mlx_array_new(), kln=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "attn.q_norm.weight", nb, sizeof(nb)); bool hq = multishard_get(&m->weights, nb, &qln);
        tname(m->cfg.layer_prefix, layer_idx, "attn.k_norm.weight", nb, sizeof(nb)); bool hk = multishard_get(&m->weights, nb, &kln);
        if (hq && hk) {
            mlx_array qq=mlx_array_new(), kk=mlx_array_new();
            ok = rms_norm_cfg(&qq, q_t, qln, m->cfg.rms_norm_eps, s) && rms_norm_cfg(&kk, k_t, kln, m->cfg.rms_norm_eps, s);
            if (ok) { mlx_array_set(&q_n, qq); mlx_array_set(&k_n, kk); }
            mlx_array_free(qq); mlx_array_free(kk);
            CHK(ok, "qk_norm");
        }
        mlx_array_free(qln); mlx_array_free(kln);
    }

    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_n, hd, /*traditional=*/false, base, 1.0f, cache_offset, no_freqs, s)==0
          && mlx_fast_rope(&k_rope, k_n, hd, false, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs); CHK(ok, "rope");
    }

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache) llm_mlx_kv_trim(k_cache, v_cache, cache_offset, s);
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_rope);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
            CHK(ok, "kv concat");
        } else { ok = mlx_array_set(&k_full, k_rope)==0 && mlx_array_set(&v_full, v_t)==0; CHK(ok, "kv init"); }
    }
    if (ok && k_cache && v_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn.out_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&osp,&osb);
    mlx_array o = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "out_proj"); }
    if (ok) { ok = mlx_add(out_h, x, o, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, normed, qw,qs,qb, qkv, q_t,k_t,v_t, q_n,k_n, q_rope,k_rope,
                        k_full,v_full, attn, attn_bt,attn_flat, ow,osp,osb, o };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool openelm_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "openelm_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ffn_norm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "ffn_norm"); }

    mlx_array p1w=mlx_array_new(), p1s=mlx_array_new(), p1b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ffn.proj_1", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &p1w,&p1s,&p1b);
    mlx_array p1 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&p1, normed, p1w, p1s, p1b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "proj_1"); }

    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (ok) {
        mlx_vector_array parts = mlx_vector_array_new();
        ok = mlx_split(&parts, p1, 2, -1, s) == 0;
        if (ok && mlx_vector_array_size(parts) == 2) {
            mlx_vector_array_get(&gate, parts, 0);
            mlx_vector_array_get(&up, parts, 1);
        } else ok = false;
        mlx_vector_array_free(parts);
        CHK(ok, "proj_1 split");
    }
    mlx_array act = mlx_array_new();
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok = mlx_sigmoid(&sg,gate,s)==0 && mlx_multiply(&sl,gate,sg,s)==0 && mlx_multiply(&act,sl,up,s)==0;
              mlx_array_free(sg); mlx_array_free(sl); CHK(ok, "swiglu"); }

    mlx_array p2w=mlx_array_new(), p2s=mlx_array_new(), p2b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ffn.proj_2", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &p2w,&p2s,&p2b);
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, act, p2w, p2s, p2b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "proj_2"); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok, "residual (ffn)"); }

    mlx_array all[] = { lnw, normed, p1w,p1s,p1b, p1, gate, up, act, p2w,p2s,p2b, dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
