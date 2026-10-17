/* ============================================================================
 * llm_mlx_qwen.c — dominio Qwen (v1/v1.5, "qwen").
 *
 * Arch: RMSNorm + attenzione a QKV FUSO (`attn.c_attn`, bias) con split in 3 parti UGUALI
 * (all-q | all-k | all-v) + RoPE PIENO (traditional=false) + MHA; MLP a 2 proiezioni
 * (`mlp.w1` = up, `mlp.w2` = gate, `mlp.c_proj`): out = c_proj(silu(w2(x)) * w1(x)).
 * Residual STANDARD (ln1 -> attn -> +x; ln2 -> mlp -> +h).
 * Prefisso layer `transformer.h.`; embed `transformer.wte`; norm finale `transformer.ln_f`;
 * lm_head separato `lm_head`.
 *
 * TRAP: quantized_linear_cfg / llm_mlx_gqa_sdpa ritornano bool TRUE=ok -> MAI `== 0`.
 * TRAP bias: il `.bias` di un Linear e' caricato dallo slot `biases` della triple (fallback
 * weights_io) e applicato da quantized_linear_cfg (mode none) -> NON ri-aggiungerlo.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool qwen_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "qwen_attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ln_1.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "ln_1"); }

    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn.c_attn", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "c_attn"); }

    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        mlx_vector_array parts = mlx_vector_array_new();
        ok = mlx_split(&parts, qkv, 3, -1, s) == 0;
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

    /* RoPE PIENO (traditional=false). */
    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, hd, /*traditional=*/false, base, 1.0f, cache_offset, no_freqs, s)==0
          && mlx_fast_rope(&k_rope, k_t, hd, false, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs); CHK(ok, "rope");
    }

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
        } else { ok = mlx_array_set(&k_full, k_rope)==0 && mlx_array_set(&v_full, v_t)==0; CHK(ok, "kv init"); }
    }
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn.c_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&osp,&osb);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "c_proj"); }
    if (ok) { ok = mlx_add(out_h, x, attn_out, s)==0; CHK(ok, "residual (attn)"); }

    mlx_array all[] = { lnw, normed, qw, qs, qb, qkv, q_t, k_t, v_t, q_rope, k_rope,
                        k_full, v_full, attn, attn_bt, attn_flat, ow, osp, osb, attn_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

bool qwen_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char nb[256]; mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "qwen_ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array lnw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ln_2.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &lnw);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, lnw, m->cfg.rms_norm_eps, s); CHK(ok, "ln_2"); }

    /* gate = w2, up = w1: swiglu(w2, w1) = silu(w2) * w1. */
    mlx_array w2w=mlx_array_new(), w2s=mlx_array_new(), w2b=mlx_array_new();
    mlx_array w1w=mlx_array_new(), w1s=mlx_array_new(), w1b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.w2", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &w2w,&w2s,&w2b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.w1", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &w1w,&w1s,&w1b);
    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gate, normed, w2w, w2s, w2b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w2 gate"); }
    if (ok) { ok = quantized_linear_cfg(&up,   normed, w1w, w1s, w1b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w1 up"); }
    mlx_array act = mlx_array_new();
    if (ok) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok = mlx_sigmoid(&sg,gate,s)==0 && mlx_multiply(&sl,gate,sg,s)==0 && mlx_multiply(&act,sl,up,s)==0;
              mlx_array_free(sg); mlx_array_free(sl); CHK(ok, "swiglu"); }

    mlx_array cw=mlx_array_new(), cs=mlx_array_new(), cb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.c_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &cw,&cs,&cb);
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, act, cw, cs, cb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "mlp c_proj"); }
    if (ok) { ok = mlx_add(out, h, dn, s)==0; CHK(ok, "residual (ffn)"); }

    mlx_array all[] = { lnw, normed, w2w,w2s,w2b, w1w,w1s,w1b, gate, up, act, cw,cs,cb, dn };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}
