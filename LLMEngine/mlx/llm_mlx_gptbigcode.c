/* ============================================================================
 * llm_mlx_gptbigcode.c — dominio GPT-BigCode / StarCoder ("gpt_bigcode").
 *
 * Arch: LayerNorm (bias) + attenzione a QKV FUSO (`attn.c_attn`, bias) con MQA (n_kv=1),
 * split a `[dim, dim+kv_dim]` (q | k | v), SENZA RoPE (position embeddings APPRESE `wpe` sommate
 * nel forward); MLP `c_fc` + GELU ESATTO (erf) + `c_proj` (bias). Residual STANDARD.
 * Prefisso layer `transformer.h.`; embed `transformer.wte`; wpe `transformer.wpe`; norm finale
 * `transformer.ln_f`; lm_head TIED (default).
 *
 * TRAP: quantized_linear_cfg / llm_mlx_gqa_sdpa ritornano bool TRUE=ok -> MAI `== 0`.
 * TRAP bias: `.bias` e' caricato dallo slot `biases` della triple -> applicato da
 * quantized_linear_cfg -> NON ri-aggiungerlo.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* gelu ESATTO: 0.5*x*(1+erf(x/sqrt(2))) — come mlx_lm nn.gelu. */
static bool gbc_gelu(mlx_array* out, mlx_array x, mlx_stream s) {
    mlx_array inv = mlx_array_new_float32(0.7071067811865476f), one = mlx_array_new_float32(1.0f);
    mlx_array half = mlx_array_new_float32(0.5f);
    mlx_array xs=mlx_array_new(), e=mlx_array_new(), op=mlx_array_new(), g=mlx_array_new();
    bool ok = mlx_multiply(&xs, x, inv, s) == 0;
    if (ok) ok = mlx_erf(&e, xs, s) == 0;
    if (ok) ok = mlx_add(&op, e, one, s) == 0;
    if (ok) ok = mlx_multiply(&g, x, op, s) == 0;
    if (ok) ok = mlx_multiply(out, g, half, s) == 0;
    mlx_array_free(inv); mlx_array_free(one); mlx_array_free(half);
    mlx_array_free(xs); mlx_array_free(e); mlx_array_free(op); mlx_array_free(g);
    return ok;
}

bool gptbigcode_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const int dim = nh * hd, kv_dim = nkv * hd;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "bigcode[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw=mlx_array_new(), lnb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ln_1.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &lnw);
    tname(m->cfg.layer_prefix, layer_idx, "ln_1.bias", nb, sizeof(nb));   ok = ok && multishard_get(&m->weights, nb, &lnb);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = mlx_fast_layer_norm(&normed, x, lnw, lnb, m->cfg.rms_norm_eps, s) == 0; CHK(ok, "ln_1"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn.c_attn", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "c_attn"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        int idxs[2] = { dim, dim + kv_dim };
        mlx_vector_array parts = mlx_vector_array_new();
        ok = mlx_split_sections(&parts, qkv, idxs, 2, -1, s) == 0;
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

    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    if (ok) {
        if (has_cache && (*k_cache).ctx != NULL && mlx_array_ndim(*k_cache)==4) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_t);
            ok = mlx_concatenate_axis(&k_full, kv, 2, s)==0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&v_full, vv, 2, s)==0; mlx_vector_array_free(vv);
            CHK(ok, "kv concat");
        } else { ok = mlx_array_set(&k_full, k_t)==0 && mlx_array_set(&v_full, v_t)==0; CHK(ok, "kv init"); }
    }
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_t, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,dim}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn.c_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&osp,&osb);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "c_proj"); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, lnb, normed, qw, qs, qb, qkv, q_t, k_t, v_t, k_full, v_full,
                        attn, attn_bt, attn_flat, ow, osp, osb, attn_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool gptbigcode_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "bigcode_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw=mlx_array_new(), lnb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ln_2.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &lnw);
    tname(m->cfg.layer_prefix, layer_idx, "ln_2.bias", nb, sizeof(nb));   ok = ok && multishard_get(&m->weights, nb, &lnb);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = mlx_fast_layer_norm(&normed, h, lnw, lnb, m->cfg.rms_norm_eps, s) == 0; CHK(ok, "ln_2"); }

    mlx_array fw=mlx_array_new(), fs=mlx_array_new(), fb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.c_fc", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &fw,&fs,&fb);
    mlx_array act = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&act, normed, fw, fs, fb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "c_fc"); }
    mlx_array g = mlx_array_new();
    if (ok) { ok = gbc_gelu(&g, act, s); CHK(ok, "gelu"); }

    mlx_array pw=mlx_array_new(), ps=mlx_array_new(), pb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.c_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &pw,&ps,&pb);
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, g, pw, ps, pb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "c_proj"); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok, "residual (ffn)"); }

    mlx_array all[] = { lnw, lnb, normed, fw, fs, fb, act, g, pw, ps, pb, dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
