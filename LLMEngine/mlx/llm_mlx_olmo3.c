/* ============================================================================
 * llm_mlx_olmo3.c — dominio OLMo-3 (AllenAI), dense: GQA + q/k RMSNorm FLAT
 * (su n_heads*head_dim, applicata su q_proj/k_proj PRIMA del reshape) e layer
 * POST-NORM:  out_h = x + post_attention_layernorm(attn(x))
 *             out   = h + post_feedforward_layernorm(mlp(h))
 * Modellato 1:1 su mlx_lm/models/olmo3.py. Naming HF (model.layers.N.self_attn.*,
 * post_attention_layernorm, post_feedforward_layernorm, model.embed_tokens/norm/lm_head).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool olmo3_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool olmo3_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* ---------- attn (GQA + q/k RMSNorm FLAT + RoPE) + post_attention_layernorm + residuo ---------- */
bool olmo3_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    const int nh = m->cfg.n_heads, nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    char namebuf[256];
    mlx_stream s = m->stream;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "olmo3_attn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

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
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &osc, &ob);
    CHK(ok, "q/k/v/o_proj lookup");

    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &qn_w);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &kn_w);
    CHK(ok, "q/k_norm lookup");

    mlx_array q_lin = mlx_array_new(), k_lin = mlx_array_new(), v_lin = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&q_lin, x, qw, qsc, qb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "q_proj"); }
    if (ok) { ok = quantized_linear_cfg(&k_lin, x, kw, ksc, kb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "k_proj"); }
    if (ok) { ok = quantized_linear_cfg(&v_lin, x, vw, vsc, vb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "v_proj"); }

    /* q_norm/k_norm FLAT: su [.., nh*hd] PRIMA del reshape per testa. */
    mlx_array q_n = mlx_array_new(), k_n = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&q_n, q_lin, qn_w, m->cfg.rms_norm_eps, s) &&
                   rms_norm_cfg(&k_n, k_lin, kn_w, m->cfg.rms_norm_eps, s); CHK(ok, "q/k_norm"); }

    mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new();
    if (ok) {
        int qsh[4] = {1, n_tokens, nh, hd}, ksh[4] = {1, n_tokens, nkv, hd};
        ok = mlx_reshape(&q4, q_n, qsh, 4, s) == 0 && mlx_reshape(&k4, k_n, ksh, 4, s) == 0 &&
             mlx_reshape(&v4, v_lin, ksh, 4, s) == 0;
        CHK(ok, "head reshape");
    }
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q4, perm, 4, s) == 0 &&
             mlx_transpose_axes(&k_t, k4, perm, 4, s) == 0 &&
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
    if (ok) {   /* KV cache: concat su asse 2 (BHTD), semplice. */
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
    mlx_array attn = mlx_array_new();
    if (ok) { ok = llm_mlx_gqa_sdpa(&attn, q_r, *k_cache, *v_cache, 1.0f / sqrtf((float)hd), 0, s); CHK(ok, "sdpa"); }
    mlx_array attn_bld = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) { const int p[4] = {0, 2, 1, 3}; ok = mlx_transpose_axes(&attn_bld, attn, p, 4, s) == 0; }
    if (ok) { int r3[3] = {1, n_tokens, nh * hd}; ok = mlx_reshape(&attn_flat, attn_bld, r3, 3, s) == 0; CHK(ok, "reshape attn"); }
    mlx_array o_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&o_out, attn_flat, ow, osc, ob, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "o_proj"); }

    /* POST-norm: out_h = x + post_attention_layernorm(o_out) */
    mlx_array pan_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &pan_w);
    mlx_array pan = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&pan, o_out, pan_w, m->cfg.rms_norm_eps, s); CHK(ok, "post_attention_layernorm"); }
    if (ok) { ok = mlx_add(&*out_h, x, pan, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(qw); mlx_array_free(qsc); mlx_array_free(qb);
    mlx_array_free(kw); mlx_array_free(ksc); mlx_array_free(kb);
    mlx_array_free(vw); mlx_array_free(vsc); mlx_array_free(vb);
    mlx_array_free(ow); mlx_array_free(osc); mlx_array_free(ob);
    mlx_array_free(qn_w); mlx_array_free(kn_w);
    mlx_array_free(q_lin); mlx_array_free(k_lin); mlx_array_free(v_lin);
    mlx_array_free(q_n); mlx_array_free(k_n);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
    mlx_array_free(q_r); mlx_array_free(k_r); mlx_array_free(attn); mlx_array_free(attn_bld); mlx_array_free(attn_flat);
    mlx_array_free(o_out); mlx_array_free(pan_w); mlx_array_free(pan);
    return ok;
#undef CHK
}

/* ---------- ffn: out = h + post_feedforward_layernorm(swiglu(gate/up->down)) ---------- */
bool olmo3_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
    mlx_stream s = m->stream;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "olmo3_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array gw = mlx_array_new(), gs_ = mlx_array_new(), gb = mlx_array_new();
    mlx_array uw = mlx_array_new(), us = mlx_array_new(), ub = mlx_array_new();
    mlx_array dw = mlx_array_new(), ds = mlx_array_new(), db = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gw, &gs_, &gb);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.up_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &uw, &us, &ub);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &dw, &ds, &db);
    CHK(ok, "mlp lookup");

    mlx_array g = mlx_array_new(), u = mlx_array_new(), sw = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&g, h, gw, gs_, gb, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "gate"); }
    if (ok) { ok = quantized_linear_cfg(&u, h, uw, us, ub, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "up"); }
    if (ok) {
        mlx_array sig = mlx_array_new(), silu = mlx_array_new();
        ok = mlx_sigmoid(&sig, g, s) == 0 && mlx_multiply(&silu, g, sig, s) == 0 && mlx_multiply(&sw, silu, u, s) == 0;
        mlx_array_free(sig); mlx_array_free(silu);
        CHK(ok, "swiglu");
    }
    mlx_array dn = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&dn, sw, dw, ds, db, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "down"); }

    mlx_array pfn_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_feedforward_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &pfn_w);
    mlx_array pfn = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&pfn, dn, pfn_w, m->cfg.rms_norm_eps, s); CHK(ok, "post_feedforward_layernorm"); }
    if (ok) { ok = mlx_add(out, h, pfn, s) == 0; CHK(ok, "residual"); }

    mlx_array_free(gw); mlx_array_free(gs_); mlx_array_free(gb);
    mlx_array_free(uw); mlx_array_free(us); mlx_array_free(ub);
    mlx_array_free(dw); mlx_array_free(ds); mlx_array_free(db);
    mlx_array_free(g); mlx_array_free(u); mlx_array_free(sw); mlx_array_free(dn);
    mlx_array_free(pfn_w); mlx_array_free(pfn);
    return ok;
#undef CHK
}
