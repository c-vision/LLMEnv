/* ============================================================================
 * llm_mlx_lfm2.c — dominio LFM2 (Liquid Foundation Model 2), arch ibrida:
 *   layer a piena attenzione (GQA + q/k RMSNorm per-testina + RoPE) E layer a
 *   "ShortConv" (conv1d depthwise causale su B*x, con gate C e stato in cache).
 * Modellato 1:1 su mlx_lm/models/lfm2.py (riferimento) e sui pattern dei blocchi
 * esistenti (qwen2_attention_block, dense_swiglu_ffn, gdn_attention_block).
 *
 * LAYOUT: x -> operator_norm -> (attn | shortconv) -> +x  (out_h)
 *         h -> ffn_norm -> feed_forward(w1,w3,w2 swiglu) -> +h  (out)
 * Il dispatch attn-vs-conv usa is_linear_layer[] (conv=true) dal loader.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



bool lfm2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool lfm2_conv_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool lfm2_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool lfm2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* ---------- attn (GQA + q/k RMSNorm per-testina + RoPE) ---------- */
bool lfm2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char namebuf[256];
    mlx_stream s = m->stream;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "lfm2_attn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array opn_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "operator_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &opn_w);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, opn_w, m->cfg.rms_norm_eps, s); CHK(ok, "operator_norm"); }

    mlx_array qw = mlx_array_new(), qsc = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ksc = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vsc = mlx_array_new(), vb = mlx_array_new();
    mlx_array ow = mlx_array_new(), osc = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &qw, &qsc, &qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &kw, &ksc, &kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &vw, &vsc, &vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.out_proj", namebuf, sizeof(namebuf));  /* LFM2 usa out_proj (non o_proj) */
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &osc, &ob);
    CHK(ok, "q/k/v/o_proj lookup");

    mlx_array qln = mlx_array_new(), kln = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &qln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &kln);
    CHK(ok, "q/k layernorm lookup");

    mlx_array q_lin = mlx_array_new(), k_lin = mlx_array_new(), v_lin = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q_lin, normed, qw, qsc, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "q_proj"); }
    if (ok) { ok = quantized_linear_cfg(&k_lin, normed, kw, ksc, kb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "k_proj"); }
    if (ok) { ok = quantized_linear_cfg(&v_lin, normed, vw, vsc, vb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "v_proj"); }

    mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new();
    mlx_array q_n = mlx_array_new(), k_n = mlx_array_new();
    if (ok) {
        int qsh[4] = {1, n_tokens, nh, hd}, ksh[4] = {1, n_tokens, nkv, hd};
        ok = mlx_reshape(&q4, q_lin, qsh, 4, s) == 0 && mlx_reshape(&k4, k_lin, ksh, 4, s) == 0 &&
             mlx_reshape(&v4, v_lin, ksh, 4, s) == 0;
        CHK(ok, "head reshape");
    }
    if (ok) { ok = rms_norm_cfg(&q_n, q4, qln, m->cfg.rms_norm_eps, s) &&
                   rms_norm_cfg(&k_n, k4, kln, m->cfg.rms_norm_eps, s); CHK(ok, "q/k layernorm"); }

    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_n, perm, 4, s) == 0 &&
             mlx_transpose_axes(&k_t, k_n, perm, 4, s) == 0 &&
             mlx_transpose_axes(&v_t, v4, perm, 4, s) == 0;
        CHK(ok, "transpose");
    }
    mlx_array q_r = mlx_array_new(), k_r = mlx_array_new();
    if (ok) {
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array nf = mlx_array_new();
        ok = mlx_fast_rope(&q_r, q_t, hd, false, base, 1.0f, cache_offset, nf, s) == 0 &&
             mlx_fast_rope(&k_r, k_t, hd, false, base, 1.0f, cache_offset, nf, s) == 0;
        mlx_array_free(nf);
        CHK(ok, "rope");
    }
    mlx_array attn = mlx_array_new();
    if (ok) {   /* KV cache: concat su asse 2 (BHTD). Semplice/corretto (non ottimizzato paging). */
        if (has_cache && mlx_array_size(*k_cache) > 0 && mlx_array_ndim(*k_cache) == 4) {
            mlx_array kn = mlx_array_new(), vn = mlx_array_new();
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, k_r);
            ok = mlx_concatenate_axis(&kn, kv, 2, s) == 0; mlx_vector_array_free(kv);
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, v_t);
            ok = ok && mlx_concatenate_axis(&vn, vv, 2, s) == 0; mlx_vector_array_free(vv);
            if (ok) { mlx_array_free(*k_cache); *k_cache = kn; mlx_array_free(*v_cache); *v_cache = vn; }
            else { mlx_array_free(kn); mlx_array_free(vn); }
        } else {
            mlx_array_free(*k_cache); mlx_array_free(*v_cache);
            ok = mlx_array_set(k_cache, k_r) == 0 && mlx_array_set(v_cache, v_t) == 0;
        }
        CHK(ok, "kv concat");
    }
    if (ok) {
        ok = llm_mlx_gqa_sdpa(&attn, q_r, *k_cache, *v_cache, 1.0f / sqrtf((float)hd), 0, s);
        CHK(ok, "sdpa");
    }
    mlx_array attn_bld = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) { const int p[4] = {0, 2, 1, 3}; ok = mlx_transpose_axes(&attn_bld, attn, p, 4, s) == 0; }
    if (ok) { int r3[3] = {1, n_tokens, nh * hd}; ok = mlx_reshape(&attn_flat, attn_bld, r3, 3, s) == 0; CHK(ok, "reshape attn"); }
    mlx_array o_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o_out, attn_flat, ow, osc, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj"); }

    if (ok) { ok = mlx_add(&*out_h, x, o_out, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(opn_w); mlx_array_free(normed);
    mlx_array_free(qw); mlx_array_free(qsc); mlx_array_free(qb);
    mlx_array_free(kw); mlx_array_free(ksc); mlx_array_free(kb);
    mlx_array_free(vw); mlx_array_free(vsc); mlx_array_free(vb);
    mlx_array_free(ow); mlx_array_free(osc); mlx_array_free(ob);
    mlx_array_free(qln); mlx_array_free(kln);
    mlx_array_free(q_lin); mlx_array_free(k_lin); mlx_array_free(v_lin);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_n); mlx_array_free(k_n); mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
    mlx_array_free(q_r); mlx_array_free(k_r); mlx_array_free(attn); mlx_array_free(attn_bld); mlx_array_free(attn_flat); mlx_array_free(o_out);
    return ok;
#undef CHK
}

/* ---------- ShortConv: operator_norm -> in_proj(3H) -> (B,C,x) -> Bx=B*x -> conv1d -> y=C*conv -> out_proj ---------- */
bool lfm2_conv_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    const int H = m->cfg.hidden_size;
    const int Lc = m->cfg.linear_conv_kernel_dim;   /* conv_L_cache */
    char namebuf[256];
    mlx_stream s = m->stream;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "lfm2_conv[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array opn_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "operator_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &opn_w);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, opn_w, m->cfg.rms_norm_eps, s); CHK(ok, "operator_norm"); }

    mlx_array ipw = mlx_array_new(), ips = mlx_array_new(), ipb = mlx_array_new();
    mlx_array opw = mlx_array_new(), ops = mlx_array_new(), opb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "conv.in_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ipw, &ips, &ipb);
    tname(m->cfg.layer_prefix, layer_idx, "conv.out_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &opw, &ops, &opb);
    CHK(ok, "conv in/out_proj lookup");
    mlx_array cw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "conv.conv.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &cw);
    CHK(ok, "conv.conv.weight lookup");
    if (ok) {   /* conv1d.weight e' BF16 (plain): mlx_conv_general richiede un dtype float coerente (come il GDN). */
        mlx_array cw32 = mlx_array_new();
        if (mlx_astype(&cw32, cw, MLX_FLOAT32, s) == 0) { mlx_array_free(cw); cw = cw32; } else mlx_array_free(cw32);
    }

    mlx_array bcx = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&bcx, normed, ipw, ips, ipb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj"); }
    mlx_array bB = mlx_array_new(), bC = mlx_array_new(), bX = mlx_array_new();
    if (ok) {
        int sec[2] = { H, 2 * H };
        mlx_vector_array v = mlx_vector_array_new();
        ok = mlx_split_sections(&v, bcx, sec, 2, -1, s) == 0;
        if (ok) ok = mlx_vector_array_get(&bB, v, 0) == 0 && mlx_vector_array_get(&bC, v, 1) == 0 &&
                      mlx_vector_array_get(&bX, v, 2) == 0;
        mlx_vector_array_free(v);
        CHK(ok, "split B/C/x");
    }
    mlx_array bx = mlx_array_new();
    if (ok) { ok = mlx_multiply(&bx, bB, bX, s) == 0; CHK(ok, "B*x"); }

    /* stato conv [1, Lc-1, H] in v_cache[layer] */
    mlx_array conv_state = mlx_array_new();
    if (ok) {
        if (has_cache && mlx_array_size(*v_cache) > 0) { ok = mlx_array_set(&conv_state, *v_cache) == 0; }
        else { int cs[3] = { 1, Lc - 1, H }; ok = mlx_zeros(&conv_state, cs, 3, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0; }
        CHK(ok, "conv_state");
    }
    mlx_array cinput = mlx_array_new();
    if (ok) {
        mlx_array arr[2] = { conv_state, bx };
        mlx_vector_array v = mlx_vector_array_new_data(arr, 2);
        ok = mlx_concatenate_axis(&cinput, v, 1, s) == 0;
        mlx_vector_array_free(v);
        CHK(ok, "concat state");
    }
    if (ok) {   /* aggiorna lo stato: ultimi Lc-1 token */
        int nkeep = Lc - 1;
        int pos2[3] = { 1, n_tokens, 1 };
        (void)pos2;
        mlx_array sl = mlx_array_new();
        /* slice [:, -nkeep:, :] */
        int st0 = n_tokens;   /* conv_input ha n_tokens + nkeep righe; prendi le ultime nkeep */
        int ss[3] = { 0, n_tokens, 0 };
        int se[3] = { 1, n_tokens + nkeep, H };
        (void)st0;
        int sst[3] = {1, 1, 1};
        ok = mlx_slice(&sl, cinput, ss, 3, se, 3, sst, 3, s) == 0;
        if (ok) { mlx_array_set(v_cache, sl); }
        mlx_array_free(sl);
    }
    /* conv1d depthwise (groups=H, kernel Lc, padding 0) */
    mlx_array cout = mlx_array_new();
    if (ok) {
        int str[1] = {1}, pl[1] = {0}, ph[1] = {0}, kd[1] = {1}, idil[1] = {1};
        ok = mlx_conv_general(&cout, cinput, cw, str, 1, pl, 1, ph, 1, kd, 1, idil, 1, H, false, s) == 0;
        CHK(ok, "conv1d");
    }
    mlx_array y = mlx_array_new();
    if (ok) { ok = mlx_multiply(&y, bC, cout, s) == 0; CHK(ok, "C*conv"); }
    mlx_array o_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o_out, y, opw, ops, opb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "out_proj"); }
    if (ok) { ok = mlx_add(&*out_h, x, o_out, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(opn_w); mlx_array_free(normed);
    mlx_array_free(ipw); mlx_array_free(ips); mlx_array_free(ipb);
    mlx_array_free(opw); mlx_array_free(ops); mlx_array_free(opb);
    mlx_array_free(cw); mlx_array_free(bcx); mlx_array_free(bB); mlx_array_free(bC); mlx_array_free(bX);
    mlx_array_free(bx); mlx_array_free(conv_state); mlx_array_free(cinput); mlx_array_free(cout);
    mlx_array_free(y); mlx_array_free(o_out);
    return ok;
#undef CHK
}

bool lfm2_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool conv = (m->cfg.is_linear_layer && layer_idx >= 0 && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx]);
    if (conv) return lfm2_conv_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    return lfm2_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
}

/* ---------- ffn: ffn_norm -> feed_forward.w1/w3 -> swiglu -> w2 -> +h ---------- */
bool lfm2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
    mlx_stream s = m->stream;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "lfm2_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array fn_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ffn_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &fn_w);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, fn_w, m->cfg.rms_norm_eps, s); CHK(ok, "ffn_norm"); }

    mlx_array w1 = mlx_array_new(), w1s = mlx_array_new(), w1b = mlx_array_new();
    mlx_array w2 = mlx_array_new(), w2s = mlx_array_new(), w2b = mlx_array_new();
    mlx_array w3 = mlx_array_new(), w3s = mlx_array_new(), w3b = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.w1", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &w1, &w1s, &w1b);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.w2", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &w2, &w2s, &w2b);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.w3", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &w3, &w3s, &w3b);
    CHK(ok, "feed_forward lookup");

    mlx_array g = mlx_array_new(), u = mlx_array_new(), sw = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&g, normed, w1, w1s, w1b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w1"); }
    if (ok) { ok = quantized_linear_cfg(&u, normed, w3, w3s, w3b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w3"); }
    if (ok) {   /* swiglu = silu(g) * u  (inline: silu(x)=x*sigmoid(x)) */
        mlx_array sig = mlx_array_new(), silu = mlx_array_new();
        ok = mlx_sigmoid(&sig, g, s) == 0 && mlx_multiply(&silu, g, sig, s) == 0 &&
             mlx_multiply(&sw, silu, u, s) == 0;
        mlx_array_free(sig); mlx_array_free(silu);
        CHK(ok, "swiglu");
    }
    mlx_array d = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&d, sw, w2, w2s, w2b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "w2"); }
    if (ok) { ok = mlx_add(out, h, d, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(fn_w); mlx_array_free(normed);
    mlx_array_free(w1); mlx_array_free(w1s); mlx_array_free(w1b);
    mlx_array_free(w2); mlx_array_free(w2s); mlx_array_free(w2b);
    mlx_array_free(w3); mlx_array_free(w3s); mlx_array_free(w3b);
    mlx_array_free(g); mlx_array_free(u); mlx_array_free(sw); mlx_array_free(d);
    return ok;
#undef CHK
}

/* ============================================================================
 * lfm2_moe_ffn_block — FFN di LFM2-MoE (Liquid). I primi `first_k_dense_replace` layer hanno MLP
 * denso (riusa lfm2_ffn_block); dal resto FFN MoE: router SIGMOID + `expert_bias` applicato SOLO
 * alla SELEZIONE top-k (non ai pesi), `norm_topk_prob`, `routed_scaling_factor`, esperti SwiGLU
 * stacked (`feed_forward.switch_mlp.{gate,up,down}_proj`, prodotti da stack_moe_experts).
 * Rif: transformers/models/lfm2_moe/Lfm2MoeTopKRouter + Lfm2MoeSparseMoeBlock.
 * ============================================================================ */
bool lfm2_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    if (layer_idx < m->cfg.first_k_dense_replace) {
        /* Layer denso (LFM2-MoE): ffn_norm -> feed_forward.{gate,up,down}_proj -> swiglu -> +h.
         * Nomi `*_proj` (convenzione MLX), NON w1/w2/w3 come `lfm2` denso. */
        bool d = true;
        char nb[256];
        const int dgsz = m->cfg.quant_group_size, dbts = m->cfg.quant_bits;
        mlx_array dln = mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "ffn_norm.weight", nb, sizeof nb);
        d = d && multishard_get(&m->weights, nb, &dln);
        mlx_array dx = mlx_array_new();
        if (d) d = rms_norm_cfg(&dx, h, dln, m->cfg.rms_norm_eps, m->stream);
        mlx_array dw1=mlx_array_new(),dw1s=mlx_array_new(),dw1b=mlx_array_new();
        mlx_array dw2=mlx_array_new(),dw2s=mlx_array_new(),dw2b=mlx_array_new();
        mlx_array dw3=mlx_array_new(),dw3s=mlx_array_new(),dw3b=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "feed_forward.gate_proj", nb, sizeof nb);
        d = d && multishard_get_triple(&m->weights, nb, &dw1,&dw1s,&dw1b);
        tname(m->cfg.layer_prefix, layer_idx, "feed_forward.up_proj", nb, sizeof nb);
        d = d && multishard_get_triple(&m->weights, nb, &dw3,&dw3s,&dw3b);
        tname(m->cfg.layer_prefix, layer_idx, "feed_forward.down_proj", nb, sizeof nb);
        d = d && multishard_get_triple(&m->weights, nb, &dw2,&dw2s,&dw2b);
        mlx_array dg=mlx_array_new(), du=mlx_array_new(), dsw=mlx_array_new(), ddo=mlx_array_new();
        if (d) { d = quantized_linear_cfg(&dg, dx, dw1,dw1s,dw1b, dgsz,dbts, m->cfg.quant_mode, m->stream); }
        if (d) { d = quantized_linear_cfg(&du, dx, dw3,dw3s,dw3b, dgsz,dbts, m->cfg.quant_mode, m->stream); }
        if (d) { mlx_array sg=mlx_array_new(), sl=mlx_array_new();
                 d = mlx_sigmoid(&sg, dg, m->stream)==0 && mlx_multiply(&sl, dg, sg, m->stream)==0 && mlx_multiply(&dsw, sl, du, m->stream)==0;
                 mlx_array_free(sg); mlx_array_free(sl); }
        if (d) { d = quantized_linear_cfg(&ddo, dsw, dw2,dw2s,dw2b, dgsz,dbts, m->cfg.quant_mode, m->stream); }
        if (d) { d = mlx_add(out, h, ddo, m->stream)==0; }
        mlx_array dall[]={dln,dx,dw1,dw1s,dw1b,dw2,dw2s,dw2b,dw3,dw3s,dw3b,dg,du,dsw,ddo};
        for (size_t i=0;i<sizeof dall/sizeof dall[0];i++) mlx_array_free(dall[i]);
        return d;
    }

    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char namebuf[256];
    mlx_stream s = m->stream;
#define MCHK(cond, what) do { if (!(cond)) { fprintf(stderr, "lfm2_moe_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ffn_norm.weight", namebuf, sizeof namebuf);
    ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, ln_w, m->cfg.rms_norm_eps, s); MCHK(ok, "ffn_norm"); }

    /* router: feed_forward.gate -> sigmoid. Il gate e' QUANTIZZATO (nel 4-bit e' 8-bit). */
    mlx_array gw = mlx_array_new(), gws = mlx_array_new(), gwb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.gate", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gw, &gws, &gwb);
    mlx_array gates = mlx_array_new();
    if (ok) {
        if (gws.ctx) {
            int rgg = gsz, rgb = bts; char rgm[16];
            llm_mlx_quant_derive(gw, gws, gsz, bts, "affine", &rgg, &rgb, rgm, sizeof rgm);
            ok = quantized_linear_cfg(&gates, normed, gw, gws, gwb, rgg, rgb, rgm, s);
        } else {
            ok = plain_linear(&gates, normed, gw, s);
        }
        MCHK(ok, "router matmul");
    }
    mlx_array scores = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&scores, gates, s) == 0; MCHK(ok, "router sigmoid"); }

    /* selezione = scores + expert_bias (il bias NON entra nei pesi) */
    mlx_array sel = mlx_array_new();
    if (ok && m->cfg.use_expert_bias) {
        mlx_array eb = mlx_array_new(), eb2 = mlx_array_new(), sum = mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "feed_forward.expert_bias", namebuf, sizeof namebuf);
        ok = ok && multishard_get(&m->weights, namebuf, &eb);
        if (ok) ok = mlx_astype(&eb2, eb, mlx_array_dtype(scores), s) == 0;
        if (ok) ok = mlx_add(&sum, scores, eb2, s) == 0;
        if (ok) mlx_array_set(&sel, sum); else mlx_array_free(sum);
        mlx_array_free(eb); mlx_array_free(eb2);
        MCHK(ok, "expert_bias");
    } else if (ok) { ok = mlx_array_set(&sel, scores) == 0; }

    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, sel, -m->cfg.top_k, -1, s) == 0; MCHK(ok, "argpartition");
        if (ok) {
            const int ne = (int)mlx_array_dim(part, 2);
            int st3[3]={0,0,ne-m->cfg.top_k}, sp3[3]={1,n_tokens,ne}, sv3[3]={1,1,1};
            mlx_array view = mlx_array_new();
            ok = mlx_slice(&view, part, st3,3, sp3,3, sv3,3, s) == 0;
            if (ok) ok = mlx_array_set(&top_inds, view) == 0;
            mlx_array_free(view);
        }
    }
    mlx_array weights = mlx_array_new();
    if (ok) {
        mlx_array tv = mlx_array_new();
        ok = mlx_take_along_axis(&tv, scores, top_inds, -1, s) == 0;
        if (ok && m->cfg.norm_topk_prob && m->cfg.top_k > 1) {
            mlx_array den = mlx_array_new(), tvn = mlx_array_new();
            ok = mlx_sum_axis(&den, tv, -1, true, s) == 0;
            if (ok) ok = mlx_divide(&tvn, tv, den, s) == 0;
            mlx_array_free(den);
            if (ok) mlx_array_set(&weights, tvn); else mlx_array_free(tvn);
            mlx_array_free(tv);
        } else { mlx_array_set(&weights, tv); mlx_array_free(tv); }
        if (ok && m->cfg.routed_scaling_factor != 1.0f) {
            mlx_array scf = mlx_array_new_float32(m->cfg.routed_scaling_factor), sc = mlx_array_new(), wsc = mlx_array_new();
            if (ok) ok = mlx_astype(&sc, scf, mlx_array_dtype(weights), s) == 0;
            ok = ok && mlx_multiply(&wsc, weights, sc, s) == 0;
            mlx_array_free(scf); mlx_array_free(sc);
            if (ok) mlx_array_set(&weights, wsc); else mlx_array_free(wsc);
        }
    }
    MCHK(ok, "top-k weights");

    mlx_array gp_w=mlx_array_new(),gp_s=mlx_array_new(),gp_b=mlx_array_new();
    mlx_array up_w=mlx_array_new(),up_s=mlx_array_new(),up_b=mlx_array_new();
    mlx_array dn_w=mlx_array_new(),dn_s=mlx_array_new(),dn_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.switch_mlp.gate_proj", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gp_w,&gp_s,&gp_b);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.switch_mlp.up_proj", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &up_w,&up_s,&up_b);
    tname(m->cfg.layer_prefix, layer_idx, "feed_forward.switch_mlp.down_proj", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &dn_w,&dn_s,&dn_b);
    int xg=gsz, xb=bts; char xm[16];
    llm_mlx_quant_derive(gp_w, gp_s, gsz, bts, "affine", &xg, &xb, xm, sizeof xm);
    MCHK(ok, "switch_mlp lookup");
    mlx_optional_int xgs={xg,true}, xbs={xb,true};

    mlx_array no_lhs = mlx_array_new(), inds = mlx_array_new();
    if (ok) {
        mlx_array tu = mlx_array_new();
        ok = mlx_astype(&tu, top_inds, MLX_UINT32, s) == 0;
        if (ok) { const int ts2[2]={n_tokens, m->cfg.top_k}; ok = mlx_reshape(&inds, tu, ts2, 2, s) == 0; }
        mlx_array_free(tu);
    }
    mlx_array x1 = mlx_array_new();
    if (ok) { const int x1s[4]={n_tokens,1,1,m->cfg.hidden_size}; ok = mlx_reshape(&x1, normed, x1s, 4, s) == 0; }

    mlx_array gate_out=mlx_array_new(), up_out=mlx_array_new(), sig=mlx_array_new(), silu=mlx_array_new(), act=mlx_array_new();
    mlx_array down_out=mlx_array_new(), w_arr=mlx_array_new(), scaled=mlx_array_new(), summed=mlx_array_new(), moe_out=mlx_array_new();
    if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, x1, gp_w,gp_s,gp_b, no_lhs, inds, true, xgs,xbs,xm, false, s)==0;
    if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out,   x1, up_w,up_s,up_b, no_lhs, inds, true, xgs,xbs,xm, false, s)==0;
    if (ok) ok = mlx_sigmoid(&sig, gate_out, s)==0 && mlx_multiply(&silu, gate_out, sig, s)==0 && mlx_multiply(&act, silu, up_out, s)==0;
    if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, dn_w,dn_s,dn_b, no_lhs, inds, true, xgs,xbs,xm, false, s)==0;
    MCHK(ok, "routed experts");
    if (ok) { const int ws[4]={n_tokens, m->cfg.top_k, 1, 1}; ok = mlx_reshape(&w_arr, weights, ws, 4, s)==0; }
    if (ok) ok = mlx_multiply(&scaled, down_out, w_arr, s)==0;
    if (ok) ok = mlx_sum_axis(&summed, scaled, 1, false, s)==0;
    if (ok) { const int o3[3]={1,n_tokens,m->cfg.hidden_size}; ok = mlx_reshape(&moe_out, summed, o3, 3, s)==0; }
    if (ok) ok = mlx_add(out, h, moe_out, s)==0;
    MCHK(ok, "final residual");

    mlx_array all[]={ln_w,normed,gw,gws,gwb,gates,scores,sel,part,top_inds,weights,
                     gp_w,gp_s,gp_b,up_w,up_s,up_b,dn_w,dn_s,dn_b,no_lhs,inds,x1,
                     gate_out,up_out,sig,silu,act,down_out,w_arr,scaled,summed,moe_out};
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
    return ok;
#undef MCHK
}
