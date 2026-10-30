/* ============================================================================
 * llm_mlx_gptneox.c — dominio GPT-NeoX / Pythia ("gpt_neox").
 *
 * Arch: LayerNorm (con bias) + attenzione a QKV FUSO (`attention.query_key_value`, bias)
 * con layout per-head [n_heads, 3*head_dim] -> split -> q|k|v; RoPE PARZIALE
 * (dims = head_dim * rotary_pct, traditional=false); MHA (n_kv = n_heads); MLP GELU-approx
 * (dense_h_to_4h / dense_4h_to_h, bias). `use_parallel_residual` (default True):
 *   out = attn(ln1(x)) + mlp(ln2(x)) + x   (altrimenti sequenziale).
 *
 * Per il parallel residual il blocco INTERO vive in gpt_neox_attention_block (l'engine passa
 * l'output dell'attn alla ffn, quindi il parallel non e' esprimibile come attn->ffn separati);
 * gpt_neox_ffn_block e' l'IDENTITA'.
 * Prefisso layer: `model.h.`; embed `model.embed_in`; norm finale `model.final_layer_norm` (LayerNorm);
 * lm_head separato `model.embed_out` (non tied).
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

/* gelu_approx(x) = 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3))) — come mlx_lm nn.gelu_approx. */
static bool gptneox_gelu_approx(mlx_array* out, mlx_array x, mlx_stream s) {
    mlx_array three = mlx_array_new_float32(3.0f), c = mlx_array_new_float32(0.044715f);
    mlx_array s2p = mlx_array_new_float32(0.7978845608028654f), one = mlx_array_new_float32(1.0f);
    mlx_array half = mlx_array_new_float32(0.5f);
    mlx_array x3 = mlx_array_new(), x3s = mlx_array_new(), inner = mlx_array_new();
    mlx_array inscale = mlx_array_new(), th = mlx_array_new(), op1 = mlx_array_new(), g = mlx_array_new();
    bool ok = mlx_power(&x3, x, three, s) == 0;
    if (ok) ok = mlx_multiply(&x3s, x3, c, s) == 0;
    if (ok) ok = mlx_add(&inner, x, x3s, s) == 0;
    if (ok) ok = mlx_multiply(&inscale, inner, s2p, s) == 0;
    if (ok) ok = mlx_tanh(&th, inscale, s) == 0;
    if (ok) ok = mlx_add(&op1, th, one, s) == 0;
    if (ok) ok = mlx_multiply(&g, x, op1, s) == 0;
    if (ok) ok = mlx_multiply(out, g, half, s) == 0;
    mlx_array_free(three); mlx_array_free(c); mlx_array_free(s2p); mlx_array_free(one); mlx_array_free(half);
    mlx_array_free(x3); mlx_array_free(x3s); mlx_array_free(inner); mlx_array_free(inscale);
    mlx_array_free(th); mlx_array_free(op1); mlx_array_free(g);
    return ok;
}

bool gpt_neox_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                              mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    const float eps = m->cfg.rms_norm_eps;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "gptneox[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    /* input_layernorm (LayerNorm con bias). */
    mlx_array ln1w=mlx_array_new(), ln1b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &ln1w);
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.bias", nb, sizeof(nb));   ok = ok && multishard_get(&m->weights, nb, &ln1b);
    mlx_array ln1 = mlx_array_new();
    if (ok) { ok = mlx_fast_layer_norm(&ln1, x, ln1w, ln1b, eps, s) == 0; CHK(ok, "input_layernorm"); }

    /* QKV fuso + bias: il bias F32 vive in `.bias` ed e' caricato dallo slot `biases` della
     * triple (fallback weights_io) e applicato da quantized_linear_cfg (mode none). */
    mlx_array qkvw=mlx_array_new(), qkvbp=mlx_array_new(), qkvbb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.query_key_value", nb, sizeof(nb));
    ok = ok && multishard_get_triple(&m->weights, nb, &qkvw, &qkvbp, &qkvbb);
    mlx_array qkv = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&qkv, ln1, qkvw, qkvbp, qkvbb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "qkv linear"); }

    /* Layout per-head: [1,T,nh,3*hd] -> split(3,-1) -> q,k,v [1,T,nh,hd] -> transpose [1,nh,T,hd]. */
    mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
    if (ok) {
        int qs4[4] = {1, n_tokens, nh, 3*hd};
        mlx_array qkv4 = mlx_array_new();
        ok = mlx_reshape(&qkv4, qkv, qs4, 4, s) == 0;
        mlx_vector_array parts = mlx_vector_array_new();
        if (ok) ok = mlx_split(&parts, qkv4, 3, -1, s) == 0;
        if (ok && mlx_vector_array_size(parts) == 3) {
            mlx_array q=mlx_array_new(), k=mlx_array_new(), v=mlx_array_new();
            mlx_vector_array_get(&q, parts, 0); mlx_vector_array_get(&k, parts, 1); mlx_vector_array_get(&v, parts, 2);
            const int perm[4] = {0,2,1,3};
            ok = mlx_transpose_axes(&q_t, q, perm,4, s)==0 && mlx_transpose_axes(&k_t, k, perm,4, s)==0 && mlx_transpose_axes(&v_t, v, perm,4, s)==0;
            mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
        } else ok = false;
        mlx_vector_array_free(parts); mlx_array_free(qkv4);
        CHK(ok, "qkv reshape/split");
    }

    /* RoPE PARZIALE (dims = int(hd*partial_rotary_factor), traditional=false). */
    mlx_array q_rope=mlx_array_new(), k_rope=mlx_array_new();
    if (ok) {
        float prf = m->cfg.partial_rotary_factor;
        int rope_dims = (prf > 0.0f && prf < 1.0f) ? (int)(hd * prf) : hd;
        if (rope_dims <= 0) rope_dims = hd;
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, rope_dims, /*traditional=*/false, base, 1.0f, cache_offset, no_freqs, s)==0
          && mlx_fast_rope(&k_rope, k_t, rope_dims, false, base, 1.0f, cache_offset, no_freqs, s)==0;
        mlx_array_free(no_freqs); CHK(ok, "rope");
    }

    /* KV cache (concat). */
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
    if (ok && has_cache) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; CHK(ok, "kv store"); }

    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, 1.0f/sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new();
    if (ok) { const int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm,4, s)==0; CHK(ok, "attn transpose"); }
    if (ok) { const int fl[3]={1,n_tokens,nh*hd}; ok = mlx_reshape(&attn_flat, attn_bt, fl,3, s)==0; CHK(ok, "attn flatten"); }

    mlx_array ow=mlx_array_new(), osp=mlx_array_new(), osb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attention.dense", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&osp,&osb);
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, osp, osb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "attn dense"); }

    /* MLP GELU-approx. Input della norm: x (parallel) oppure x+attn_out (sequenziale). */
    mlx_array resid_in = mlx_array_new();
    mlx_array agg = mlx_array_new();   /* x + attn_out */
    if (ok) { ok = mlx_add(&agg, x, attn_out, s)==0; CHK(ok, "residual attn"); }
    if (ok) { ok = mlx_array_set(&resid_in, m->cfg.parallel_residual ? x : agg)==0; }

    mlx_array ln2w=mlx_array_new(), ln2b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb)); ok = ok && multishard_get(&m->weights, nb, &ln2w);
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.bias", nb, sizeof(nb));   ok = ok && multishard_get(&m->weights, nb, &ln2b);
    mlx_array ln2 = mlx_array_new();
    if (ok) { ok = mlx_fast_layer_norm(&ln2, resid_in, ln2w, ln2b, eps, s) == 0; CHK(ok, "post_attention_layernorm"); }

    mlx_array h4w=mlx_array_new(), h4sp=mlx_array_new(), h4sb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.dense_h_to_4h", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &h4w,&h4sp,&h4sb);
    mlx_array h4 = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&h4, ln2, h4w, h4sp, h4sb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "mlp h4"); }
    mlx_array g4 = mlx_array_new();
    if (ok) { ok = gptneox_gelu_approx(&g4, h4, s); CHK(ok, "gelu"); }

    mlx_array dw=mlx_array_new(), dsp=mlx_array_new(), dsb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.dense_4h_to_h", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &dw,&dsp,&dsb);
    mlx_array mlp_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&mlp_out, g4, dw, dsp, dsb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "mlp down"); }

    /* out = x + attn_out + mlp_out (parallel)  |  (x+attn_out) + mlp_out (sequenziale). */
    if (ok) {
        mlx_array pre = mlx_array_new();
        ok = mlx_add(&pre, agg, mlp_out, s) == 0;
        if (ok) mlx_array_set(out_h, pre);
        mlx_array_free(pre);
        CHK(ok, "final residual");
    }

    mlx_array all[] = { ln1w, ln1b, ln1, qkvw, qkvbp, qkvbb, qkv, q_t, k_t, v_t, q_rope, k_rope,
                        k_full, v_full, attn, attn_bt, attn_flat, ow, osp, osb, attn_out, agg, resid_in,
                        ln2w, ln2b, ln2, h4w, h4sp, h4sb, h4, g4, dw, dsp, dsb, mlp_out };
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef CHK
    return ok;
}

/* FFN identity: il blocco intero e' gia' stato calcolato dall'attn (parallel residual). */
bool gpt_neox_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    (void)m; (void)layer_idx; (void)n_tokens;
    return mlx_array_set(out, h) == 0;
}
