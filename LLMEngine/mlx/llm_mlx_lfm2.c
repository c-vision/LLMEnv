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
