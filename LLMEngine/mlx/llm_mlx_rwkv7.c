/* ============================================================================
 * llm_mlx_rwkv7.c — dominio RWKV7 (fla-hub/rwkv7-*), arch INTERAMENTE ricorrente.
 *
 * Modellato 1:1 su mlx_lm/models/rwkv7.py (l'oracolo di parita').
 *   Layer (Rwkv7Layer):  if layer_idx==0: x = pre_norm(x)          [LayerNorm D]
 *                        h, v_first = attn(attn_norm(x), v_first)  [time-mixing]
 *                        h = x + h
 *                        out = h + ffn(ffn_norm(h))                 [channel-mixing]
 *   Time-mixing  (attn): token-shift -> x_r/x_w/x_k/x_v/x_a/x_g (addcmul) -> r/k/v proj
 *                        -> LoRA a/w/g/(v se layer>0) -> iclr/decay/kk -> WKV7 scan
 *                        -> per-head LayerNorm (g_norm) -> o_proj(out*gate) -> +residuo.
 *   Channel-mix  (ffn):  token-shift (stato SEPARATO) -> xx = h + (x_prev-h)*x_k
 *                        -> out = value(relu(key(xx))^2) -> +residuo.
 *
 * Stato di sessione per layer (3 slot):
 *   k_cache[layer]  = stato WKV7 [1, H, Dh, Dh]                 (cache[1] in mlx_lm)
 *   v_cache[layer]  = token-shift del time-mixing [1,1,D]        (cache[0])
 *   m->aux_cache[layer] = token-shift del channel-mixing [1,1,D] (cache[2])  -- ffn non
 *                         riceve cache, quindi lo stato ausiliario passa da aux_cache.
 * v_first (carry cross-layer, prodotto dal layer 0 e usato dai successivi) vive in
 * m->rwkv_v_first: scritto dal layer 0, letto dai layer >0 nello STESSO forward.
 *
 * NB: le proiezioni LoRA hanno un bias LINEARE separato (.lora.2.bias) che, in caso di
 * peso plain, multishard_get_triple carica nello slot biases MA quantized_linear_cfg
 * instrada a plain_linear (che lo IGNORA). Lo aggiungiamo quindi esplicitamente.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool rwkv7_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                      mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool rwkv7_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

#define RWK_DECAY_SCALE 0.606531f   /* exp(-0.606531*decay) come mlx_lm */
#define RWK_GNORM_EPS   64e-5f      /* LayerNormPerHead eps=64e-5 (hardcoded in mlx_lm) */

/* LayerNorm completo (mean+var) via fast::layer_norm; weight/bias possono essere NULL. */
static bool rw_ln(mlx_array* res, mlx_array x, mlx_array w, mlx_array b, float eps, mlx_stream s) {
    return mlx_fast_layer_norm(res, x, w, b, eps, s) == 0;
}

/* l2_norm su ultimo asse: x / max(norm, 1e-7) (mlx_lm l2_norm). */
static bool rw_l2norm(mlx_array* res, mlx_array x, mlx_stream s) {
    mlx_array sq = mlx_array_new(), sm = mlx_array_new(), mx = mlx_array_new();
    mlx_array sq2 = mlx_array_new(), cl = mlx_array_new(), r = mlx_array_new();
    mlx_array eps = mlx_array_new_float32(1e-7f);
    int ax[1] = { -1 };
    bool ok = mlx_square(&sq, x, s) == 0 && mlx_sum_axes(&sm, sq, ax, 1, true, s) == 0
           && mlx_maximum(&mx, sm, eps, s) == 0 && mlx_sqrt(&cl, mx, s) == 0
           && mlx_divide(&r, x, cl, s) == 0;
    if (ok) mlx_array_set(res, r);
    mlx_array_free(sq); mlx_array_free(sm); mlx_array_free(mx); mlx_array_free(sq2);
    mlx_array_free(cl); mlx_array_free(r); mlx_array_free(eps);
    return ok;
}

/* true se lo slot cache e' presente e non vuoto (mlx_array_size su array vuoto lancia). */
static bool rw_slot_has(mlx_array* slot) { return slot != NULL && slot->ctx != NULL && mlx_array_size(*slot) > 0; }

/* Fetch per-layer: name = <layer_prefix><idx>.<suf>; carica il tensore (plain o .weight). */
static bool rw_get(LlmMlxModel* m, int layer_idx, const char* suf, mlx_array* out) {
    char nb[320];
    tname(m->cfg.layer_prefix, layer_idx, suf, nb, sizeof(nb));
    return multishard_get(&m->weights, nb, out);
}

/* Lineare: se il peso e' PLAIN (scales assenti) usa plain_linear (quantized_linear_cfg in
 * mode "none" fallisce con pesi bf16 plain), altrimenti la qmm. Il bias lineare (.bias) e'
 * aggiunto a parte dal chiamante (plain_linear non lo applica). */
static bool rw_lin(mlx_array* res, LlmMlxModel* m, mlx_array x, mlx_array w, mlx_array sc, mlx_array bb,
                   int gsz, int bts, mlx_stream s) {
    if (sc.ctx == NULL || mlx_array_size(sc) == 0) return plain_linear(res, x, w, s);
    return quantized_linear_cfg(res, x, w, sc, bb, gsz, bts, m->cfg.quant_mode, s) == 0;
}

/* Lineare di un LoRA: lora.2(lora.1(lora.0(x))), activation 0=none 1=sigmoid 2=tanh.
 * base es. "attn.a_lora" -> usa "attn.a_lora.lora.0" e "attn.a_lora.lora.2". */
static bool rw_lora(mlx_array* res, LlmMlxModel* m, int layer_idx, const char* base, mlx_array x,
                    int act, int gsz, int bts, mlx_stream s) {
    char suf[192], nb[320];
    bool ok = true;
    /* lora.0 (bias=False) */
    mlx_array w0 = mlx_array_new(), s0 = mlx_array_new(), b0 = mlx_array_new();
    snprintf(suf, sizeof(suf), "%s.lora.0", base);
    { char n0[320]; tname(m->cfg.layer_prefix, layer_idx, suf, n0, sizeof(n0));
      if (!multishard_get_triple(&m->weights, n0, &w0, &s0, &b0)) ok = false; }
    mlx_array h1 = mlx_array_new(), empty = mlx_array_new();
    if (ok) ok = rw_lin(&h1, m, x, w0, s0, empty, gsz, bts, s);
    if (ok && act == 1) { mlx_array g = mlx_array_new(); ok = mlx_sigmoid(&g, h1, s) == 0; if (ok) mlx_array_set(&h1, g); mlx_array_free(g); }
    else if (ok && act == 2) { mlx_array g = mlx_array_new(); ok = mlx_tanh(&g, h1, s) == 0; if (ok) mlx_array_set(&h1, g); mlx_array_free(g); }
    /* lora.2 (bias=False su g_lora; separato per gli altri) */
    mlx_array w2 = mlx_array_new(), s2 = mlx_array_new(), b2 = mlx_array_new();
    if (ok) { snprintf(suf, sizeof(suf), "%s.lora.2", base); tname(m->cfg.layer_prefix, layer_idx, suf, nb, sizeof(nb));
              ok = multishard_get_triple(&m->weights, nb, &w2, &s2, &b2); }
    mlx_array h2 = mlx_array_new();
    if (ok) ok = rw_lin(&h2, m, h1, w2, s2, b2, gsz, bts, s);
    /* bias lineare esplicito (.lora.2.bias), NON coperto da quantized_linear_cfg (plain). */
    if (ok) {
        char sb[200], nbias[344]; snprintf(sb, sizeof(sb), "%s.lora.2.bias", base);
        tname(m->cfg.layer_prefix, layer_idx, sb, nbias, sizeof(nbias));
        mlx_array lb = mlx_array_new();
        if (multishard_get(&m->weights, nbias, &lb) && mlx_array_size(lb) > 0) {
            mlx_array h3 = mlx_array_new();
            if (mlx_add(&h3, h2, lb, s) == 0) mlx_array_set(&h2, h3);
            mlx_array_free(h3);
        }
        mlx_array_free(lb);
    }
    if (ok) mlx_array_set(res, h2);
    mlx_array_free(w0); mlx_array_free(s0); mlx_array_free(b0); mlx_array_free(h1); mlx_array_free(empty);
    mlx_array_free(w2); mlx_array_free(s2); mlx_array_free(b2); mlx_array_free(h2);
    return ok;
}

/* Token-shift (mlx_lm TokenShift): x_prev = [stato(1 token), x[:, :-1, :]]; aggiorna lo slot. */
static bool rw_shift(mlx_array* xprev, mlx_array hin, mlx_array* slot, bool have, int Hid, mlx_stream s) {
    int L = mlx_array_shape(hin)[1];
    mlx_array prev = mlx_array_new();
    if (have && rw_slot_has(slot)) mlx_array_set(&prev, *slot);
    else { int zs[3] = { 1, 1, Hid }; mlx_zeros(&prev, zs, 3, MLX_FLOAT32, s); }
    bool ok = true;
    if (L == 1) {
        mlx_array_set(xprev, prev);
    } else {
        int ss[3] = { 0, 0, 0 }, se[3] = { 1, L - 1, Hid }, stp[3] = { 1, 1, 1 };
        mlx_array body = mlx_array_new();
        ok = mlx_slice(&body, hin, ss, 3, se, 3, stp, 3, s) == 0;
        if (ok) { mlx_array arr[2] = { prev, body }; mlx_vector_array v = mlx_vector_array_new_data(arr, 2);
                  ok = mlx_concatenate_axis(xprev, v, 1, s) == 0; mlx_vector_array_free(v); }
        mlx_array_free(body);
    }
    if (ok && slot) {
        int ss[3] = { 0, L - 1, 0 }, se[3] = { 1, L, Hid }, stp[3] = { 1, 1, 1 };
        mlx_array last = mlx_array_new();
        if (mlx_slice(&last, hin, ss, 3, se, 3, stp, 3, s) == 0) mlx_array_set(slot, last);
        mlx_array_free(last);
    }
    mlx_array_free(prev);
    return ok;
}

/* slice [1,L,H,Dh] -> [1,H,Dh] al tempo t. */
static bool rw_tok(mlx_array* out, mlx_array a, int t, int H, int Dh, mlx_stream s) {
    int ss[4] = { 0, t, 0, 0 }, se[4] = { 1, t + 1, H, Dh }, st[4] = { 1, 1, 1, 1 };
    mlx_array sl = mlx_array_new();
    bool ok = mlx_slice(&sl, a, ss, 4, se, 4, st, 4, s) == 0 && mlx_squeeze_axis(out, sl, 1, s) == 0;
    mlx_array_free(sl);
    return ok;
}

bool rwkv7_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                      mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int Hid = m->cfg.hidden_size;
    const int H = m->cfg.n_heads > 0 ? m->cfg.n_heads : 1;
    const int Dh = m->cfg.head_dim > 0 ? m->cfg.head_dim : (Hid / (H > 0 ? H : 1));
    const float eps = m->cfg.rms_norm_eps;
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "rwkv7.attn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    /* layer 0: pre_norm; la base del residuo e' x (post pre_norm). */
    mlx_array xbase = mlx_array_new();
    if (layer_idx == 0) {
        mlx_array pw = mlx_array_new(), pb = mlx_array_new();
        rw_get(m, 0, "pre_norm.weight", &pw); rw_get(m, 0, "pre_norm.bias", &pb);
        ok = rw_ln(&xbase, x, pw, pb, eps, s); CHK(ok, "pre_norm");
        mlx_array_free(pw); mlx_array_free(pb);
    } else mlx_array_set(&xbase, x);

    /* attn_norm */
    mlx_array anw = mlx_array_new(), anb = mlx_array_new();
    rw_get(m, layer_idx, "attn_norm.weight", &anw); rw_get(m, layer_idx, "attn_norm.bias", &anb);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rw_ln(&h, xbase, anw, anb, eps, s); CHK(ok, "attn_norm"); }

    /* token-shift del time-mixing (v_cache) */
    mlx_array xprev = mlx_array_new();
    if (ok) { ok = rw_shift(&xprev, h, v_cache, has_cache, Hid, s); CHK(ok, "tokshift"); }
    mlx_array xx = mlx_array_new();
    if (ok) { ok = mlx_subtract(&xx, xprev, h, s) == 0; CHK(ok, "xx"); }

    /* x_r/x_w/x_k/x_v/x_a/x_g (addcmul: base + xx*coef) */
    const char* xn[6] = { "attn.x_r", "attn.x_w", "attn.x_k", "attn.x_v", "attn.x_a", "attn.x_g" };
    mlx_array xc[6] = { mlx_array_new(), mlx_array_new(), mlx_array_new(), mlx_array_new(), mlx_array_new(), mlx_array_new() };
    for (int q = 0; q < 6 && ok; q++) {
        mlx_array coef = mlx_array_new();
        ok = rw_get(m, layer_idx, xn[q], &coef) && mlx_array_size(coef) > 0;
        if (ok) { mlx_array t = mlx_array_new(); ok = mlx_multiply(&t, xx, coef, s) == 0; if (ok) ok = mlx_add(&xc[q], h, t, s) == 0; mlx_array_free(t); }
        mlx_array_free(coef);
    }

    /* proiezioni r/k/v -> [1,L,H,Dh] */
    mlx_array key = mlx_array_new(), value = mlx_array_new(), rej = mlx_array_new();
    mlx_array iclr = mlx_array_new(), gate = mlx_array_new();
    if (ok) {
        int sh[4] = { 1, (int)mlx_array_shape(h)[1], H, Dh };
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new(), lin = mlx_array_new();
        char b[340]; tname(m->cfg.layer_prefix, layer_idx, "attn.k_proj", b, sizeof(b));
        ok = multishard_get_triple(&m->weights, b, &w, &sc, &bb)
             && rw_lin(&lin, m, xc[2], w, sc, bb, gsz, bts, s)
             && mlx_reshape(&key, lin, sh, 4, s) == 0;
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb); mlx_array_free(lin);
    }
    CHK(ok, "k_proj");
    if (ok) {
        int sh[4] = { 1, (int)mlx_array_shape(h)[1], H, Dh };
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new(), lin = mlx_array_new();
        char b[340]; tname(m->cfg.layer_prefix, layer_idx, "attn.v_proj", b, sizeof(b));
        ok = multishard_get_triple(&m->weights, b, &w, &sc, &bb)
             && rw_lin(&lin, m, xc[3], w, sc, bb, gsz, bts, s)
             && mlx_reshape(&value, lin, sh, 4, s) == 0;
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb); mlx_array_free(lin);
    }
    CHK(ok, "v_proj");
    if (ok) {
        int sh[4] = { 1, (int)mlx_array_shape(h)[1], H, Dh };
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new(), lin = mlx_array_new();
        char b[340]; tname(m->cfg.layer_prefix, layer_idx, "attn.r_proj", b, sizeof(b));
        ok = multishard_get_triple(&m->weights, b, &w, &sc, &bb)
             && rw_lin(&lin, m, xc[0], w, sc, bb, gsz, bts, s)
             && mlx_reshape(&rej, lin, sh, 4, s) == 0;
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb); mlx_array_free(lin);
    }
    CHK(ok, "r_proj");

    /* iclr = sigmoid(a_lora(xa)) -> [1,L,H,Dh]; gate = g_lora(xg) -> [1,L,D] */
    if (ok) {
        int sh[4] = { 1, (int)mlx_array_shape(h)[1], H, Dh };
        mlx_array la = mlx_array_new();
        ok = rw_lora(&la, m, layer_idx, "attn.a_lora", xc[4], 0, gsz, bts, s);
        if (ok) { mlx_array sg = mlx_array_new(); ok = mlx_sigmoid(&sg, la, s) == 0; if (ok) ok = mlx_reshape(&iclr, sg, sh, 4, s) == 0; mlx_array_free(sg); }
        mlx_array_free(la);
    }
    if (ok) ok = rw_lora(&gate, m, layer_idx, "attn.g_lora", xc[5], 1, gsz, bts, s);

    /* v_first / v_lora */
    if (ok) {
        if (layer_idx == 0) {
            mlx_array_set(&m->rwkv_v_first, value);
        } else {
            int sh[4] = { 1, (int)mlx_array_shape(h)[1], H, Dh };
            mlx_array lv = mlx_array_new(), vv = mlx_array_new();
            ok = rw_lora(&lv, m, layer_idx, "attn.v_lora", xc[3], 0, gsz, bts, s);
            if (ok) { mlx_array sg = mlx_array_new(); ok = mlx_sigmoid(&sg, lv, s) == 0; if (ok) ok = mlx_reshape(&vv, sg, sh, 4, s) == 0; mlx_array_free(sg); }
            /* value = value + (v_first - value) * vv */
            if (ok) {
                mlx_array d = mlx_array_new(), m1 = mlx_array_new(), m2 = mlx_array_new();
                ok = mlx_subtract(&d, m->rwkv_v_first, value, s) == 0 && mlx_multiply(&m1, d, vv, s) == 0
                     && mlx_add(&m2, value, m1, s) == 0;
                if (ok) mlx_array_set(&value, m2);
                mlx_array_free(d); mlx_array_free(m1); mlx_array_free(m2);
            }
            mlx_array_free(lv); mlx_array_free(vv);
        }
    }

    /* decay = exp(-0.606531 * sigmoid(w_lora(xw))) in f32, poi back a dtype */
    mlx_array decay = mlx_array_new();
    if (ok) {
        int sh[4] = { 1, (int)mlx_array_shape(h)[1], H, Dh };
        mlx_array lw = mlx_array_new(), sg = mlx_array_new(), rs = mlx_array_new();
        ok = rw_lora(&lw, m, layer_idx, "attn.w_lora", xc[1], 2, gsz, bts, s);
        if (ok) { mlx_array sg2 = mlx_array_new(); ok = mlx_sigmoid(&sg2, lw, s) == 0; if (ok) ok = mlx_reshape(&sg, sg2, sh, 4, s) == 0; mlx_array_free(sg2); }
        if (ok) { mlx_array f = mlx_array_new(); ok = mlx_astype(&f, sg, MLX_FLOAT32, s) == 0;
            if (ok) { mlx_array c = mlx_array_new_float32(-RWK_DECAY_SCALE); mlx_array p = mlx_array_new();
                ok = mlx_multiply(&p, f, c, s) == 0; if (ok) ok = mlx_exp(&rs, p, s) == 0;
                mlx_array_free(c); mlx_array_free(p); }
            mlx_array_free(f); }
        if (ok) { mlx_array dt = mlx_array_new(); ok = mlx_astype(&dt, rs, mlx_array_dtype(value), s) == 0; if (ok) mlx_array_set(&decay, dt); mlx_array_free(dt); }
        mlx_array_free(lw); mlx_array_free(sg); mlx_array_free(rs);
    }

    /* kk = l2_norm(key * k_k); key = key * (1 + (iclr-1)*k_a); a=-kk, b=kk*iclr */
    mlx_array av = mlx_array_new(), bvk = mlx_array_new();
    if (ok) {
        mlx_array kk_w = mlx_array_new(), ka_w = mlx_array_new(), kkk = mlx_array_new();
        ok = rw_get(m, layer_idx, "attn.k_k", &kk_w) && rw_get(m, layer_idx, "attn.k_a", &ka_w);
        /* k_k/k_a sono [H*Dh] nel safetensors -> [H,Dh] (mlx_lm sanitize). */
        if (ok) { int shk[2] = { H, Dh };
            mlx_array t1 = mlx_array_new(), t2 = mlx_array_new();
            if (mlx_reshape(&t1, kk_w, shk, 2, s) == 0) mlx_array_set(&kk_w, t1);
            if (mlx_reshape(&t2, ka_w, shk, 2, s) == 0) mlx_array_set(&ka_w, t2);
            mlx_array_free(t1); mlx_array_free(t2); }
        if (ok) { mlx_array t = mlx_array_new(); ok = mlx_multiply(&t, key, kk_w, s) == 0; if (ok) ok = rw_l2norm(&kkk, t, s); mlx_array_free(t); }
        /* key = key*(1+(iclr-1)*ka) = key*ka_adj */
        if (ok) {
            mlx_array one = mlx_array_new_float32(1.0f), im1 = mlx_array_new(), t = mlx_array_new(), t2 = mlx_array_new();
            ok = mlx_subtract(&im1, iclr, one, s) == 0 && mlx_multiply(&t, im1, ka_w, s) == 0
                 && mlx_add(&t2, t, one, s) == 0;
            if (ok) { mlx_array nk = mlx_array_new(); ok = mlx_multiply(&nk, key, t2, s) == 0; if (ok) mlx_array_set(&key, nk); mlx_array_free(nk); }
            mlx_array_free(one); mlx_array_free(im1); mlx_array_free(t); mlx_array_free(t2);
        }
        if (ok) { mlx_array ng = mlx_array_new(); ok = mlx_negative(&ng, kkk, s) == 0; if (ok) mlx_array_set(&av, ng); mlx_array_free(ng); }
        if (ok) { mlx_array p = mlx_array_new(); ok = mlx_multiply(&p, kkk, iclr, s) == 0; if (ok) mlx_array_set(&bvk, p); mlx_array_free(p); }
        mlx_array_free(kk_w); mlx_array_free(ka_w); mlx_array_free(kkk);
    }

    /* WKV7 scan: state[B,H,Dh,Dh]; per t: sab=(state@a)@b; state=state*w + v@k + sab; y=state@r */
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && mlx_array_size(*k_cache) > 0) mlx_array_set(&state, *k_cache);
        else { int st4[4] = { 1, H, Dh, Dh }; mlx_zeros(&state, st4, 4, MLX_FLOAT32, s); }
    }
    mlx_array ystack = mlx_array_new(); bool yset = false;
    for (int t = 0; t < n_tokens && ok; t++) {
        mlx_array r = mlx_array_new(), w = mlx_array_new(), k = mlx_array_new();
        mlx_array v = mlx_array_new(), a = mlx_array_new(), b = mlx_array_new();
        ok = rw_tok(&r, rej, t, H, Dh, s) && rw_tok(&w, decay, t, H, Dh, s) && rw_tok(&k, key, t, H, Dh, s)
          && rw_tok(&v, value, t, H, Dh, s) && rw_tok(&a, av, t, H, Dh, s) && rw_tok(&b, bvk, t, H, Dh, s);
        mlx_array aE = mlx_array_new(), bE = mlx_array_new(), wE = mlx_array_new();
        mlx_array vE = mlx_array_new(), kE = mlx_array_new(), rE = mlx_array_new();
        if (ok) { ok = mlx_expand_dims(&aE, a, -1, s) == 0 && mlx_expand_dims(&bE, b, 2, s) == 0
                && mlx_expand_dims(&wE, w, 2, s) == 0 && mlx_expand_dims(&vE, v, -1, s) == 0
                && mlx_expand_dims(&kE, k, 2, s) == 0 && mlx_expand_dims(&rE, r, -1, s) == 0; }
        mlx_array sa = mlx_array_new(), sab = mlx_array_new(), sw = mlx_array_new();
        mlx_array vk = mlx_array_new(), n1 = mlx_array_new(), nsf = mlx_array_new();
        if (ok) { ok = mlx_matmul(&sa, state, aE, s) == 0 && mlx_matmul(&sab, sa, bE, s) == 0
                && mlx_multiply(&sw, state, wE, s) == 0 && mlx_matmul(&vk, vE, kE, s) == 0
                && mlx_add(&n1, sw, vk, s) == 0 && mlx_add(&nsf, n1, sab, s) == 0; }
        if (ok) mlx_array_set(&state, nsf);
        mlx_array yp = mlx_array_new(), ys = mlx_array_new(), ye = mlx_array_new();
        if (ok) { ok = mlx_matmul(&yp, state, rE, s) == 0 && mlx_squeeze_axis(&ys, yp, 3, s) == 0
                && mlx_expand_dims(&ye, ys, 1, s) == 0; }
        if (ok) {
            if (!yset) { mlx_array_set(&ystack, ye); yset = true; }
            else { mlx_array cc = mlx_array_new(); mlx_array arr[2] = { ystack, ye };
                   mlx_vector_array vv = mlx_vector_array_new_data(arr, 2);
                   ok = mlx_concatenate_axis(&cc, vv, 1, s) == 0; mlx_vector_array_free(vv);
                   if (ok) mlx_array_set(&ystack, cc); mlx_array_free(cc); }
        }
        mlx_array_free(r); mlx_array_free(w); mlx_array_free(k); mlx_array_free(v); mlx_array_free(a); mlx_array_free(b);
        mlx_array_free(aE); mlx_array_free(bE); mlx_array_free(wE); mlx_array_free(vE); mlx_array_free(kE); mlx_array_free(rE);
        mlx_array_free(sa); mlx_array_free(sab); mlx_array_free(sw); mlx_array_free(vk); mlx_array_free(n1); mlx_array_free(nsf);
        mlx_array_free(yp); mlx_array_free(ys); mlx_array_free(ye);
    }
    if (ok) mlx_array_set(k_cache, state);   /* out [1,L,H,Dh] in ystack */

    /* g_norm (per-head LayerNorm weightless) * g_norm.weight + g_norm.bias */
    mlx_array gout = mlx_array_new();
    if (ok && yset) {
        mlx_array gw = mlx_array_new(), gb = mlx_array_new();
        rw_get(m, layer_idx, "attn.g_norm.weight", &gw);
        rw_get(m, layer_idx, "attn.g_norm.bias", &gb);
        /* g_norm.weight/bias sono [H*Dh] -> [H,Dh] (mlx_lm sanitize). */
        { int shg[2] = { H, Dh }; mlx_array t1 = mlx_array_new(), t2 = mlx_array_new();
          if (mlx_reshape(&t1, gw, shg, 2, s) == 0) mlx_array_set(&gw, t1);
          if (mlx_reshape(&t2, gb, shg, 2, s) == 0) mlx_array_set(&gb, t2);
          mlx_array_free(t1); mlx_array_free(t2); }
        mlx_array ln = mlx_array_new(), e1 = mlx_array_new(), e2 = mlx_array_new();
        mlx_array nul1 = mlx_array_new(), nul2 = mlx_array_new();
        ok = rw_ln(&ln, ystack, nul1, nul2, RWK_GNORM_EPS, s);
        if (ok) { ok = mlx_multiply(&e1, ln, gw, s) == 0 && mlx_add(&e2, e1, gb, s) == 0; if (ok) mlx_array_set(&gout, e2); }
        mlx_array_free(gw); mlx_array_free(gb); mlx_array_free(ln); mlx_array_free(e1); mlx_array_free(e2);
        mlx_array_free(nul1); mlx_array_free(nul2);
    }

    /* out = (g + sum(receptance*key*r_k, -1, keepdim)*value).reshape(1,L,D) ; o_proj(out*gate) */
    mlx_array oo = mlx_array_new();
    if (ok && yset) {
        int L = (int)mlx_array_shape(h)[1];
        mlx_array r_k = mlx_array_new();
        ok = rw_get(m, layer_idx, "attn.r_k", &r_k);
        /* s = sum(rej*key*r_k, -1, keepdim) */
        mlx_array t1 = mlx_array_new(), t2 = mlx_array_new(), t3 = mlx_array_new();
        if (ok) { int ax1[1] = { -1 }; ok = mlx_multiply(&t1, rej, key, s) == 0 && mlx_multiply(&t2, t1, r_k, s) == 0
                && mlx_sum_axes(&t3, t2, ax1, 1, true, s) == 0; }
        mlx_array t4 = mlx_array_new(), t5 = mlx_array_new();
        if (ok) { ok = mlx_multiply(&t4, t3, value, s) == 0 && mlx_add(&t5, gout, t4, s) == 0; }
        mlx_array rsh = mlx_array_new(); int sh3[3] = { 1, L, Hid };
        if (ok) { ok = mlx_reshape(&rsh, t5, sh3, 3, s) == 0; }
        mlx_array gated = mlx_array_new(), attn_out = mlx_array_new();
        if (ok) { ok = mlx_multiply(&gated, rsh, gate, s) == 0; }
        if (ok) {
            mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new();
            char b[340]; tname(m->cfg.layer_prefix, layer_idx, "attn.o_proj", b, sizeof(b));
            ok = multishard_get_triple(&m->weights, b, &w, &sc, &bb)
                 && rw_lin(&attn_out, m, gated, w, sc, bb, gsz, bts, s);
            if (ok) ok = mlx_add(&oo, xbase, attn_out, s) == 0;
            mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb);
        }
        mlx_array_free(r_k); mlx_array_free(t1); mlx_array_free(t2); mlx_array_free(t3); mlx_array_free(t4); mlx_array_free(t5);
        mlx_array_free(rsh); mlx_array_free(gated); mlx_array_free(attn_out);
    }
    if (ok) mlx_array_set(out_h, oo); else CHK(false, "output");

    mlx_array_free(xbase); mlx_array_free(anw); mlx_array_free(anb); mlx_array_free(h);
    mlx_array_free(xprev); mlx_array_free(xx);
    for (int q = 0; q < 6; q++) mlx_array_free(xc[q]);
    mlx_array_free(key); mlx_array_free(value); mlx_array_free(rej); mlx_array_free(iclr); mlx_array_free(gate);
    mlx_array_free(decay); mlx_array_free(av); mlx_array_free(bvk); mlx_array_free(state);
    mlx_array_free(ystack); mlx_array_free(gout); mlx_array_free(oo);
    return ok;
#undef CHK
}

bool rwkv7_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int Hid = m->cfg.hidden_size;
    const float eps = m->cfg.rms_norm_eps;
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "rwkv7.ffn[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array fnw = mlx_array_new(), fnb = mlx_array_new();
    rw_get(m, layer_idx, "ffn_norm.weight", &fnw); rw_get(m, layer_idx, "ffn_norm.bias", &fnb);
    mlx_array hn = mlx_array_new();
    if (ok) { ok = rw_ln(&hn, h, fnw, fnb, eps, s); CHK(ok, "ffn_norm"); }

    /* token-shift del channel-mixing (aux_cache[layer]) */
    mlx_array* slot = (m->aux_cache && layer_idx < m->cfg.n_layers) ? &m->aux_cache[layer_idx] : NULL;
    bool have = rw_slot_has(slot);
    mlx_array xprev = mlx_array_new();
    if (ok) { ok = rw_shift(&xprev, hn, slot, have, Hid, s); CHK(ok, "tokshift ffn"); }

    /* xx = hn + (xprev - hn) * x_k */
    mlx_array xk = mlx_array_new();
    rw_get(m, layer_idx, "ffn.x_k", &xk);
    mlx_array d = mlx_array_new(), dm = mlx_array_new(), xx = mlx_array_new();
    if (ok) { ok = mlx_subtract(&d, xprev, hn, s) == 0 && mlx_multiply(&dm, d, xk, s) == 0
            && mlx_add(&xx, hn, dm, s) == 0; CHK(ok, "xx ffn"); }

    /* kk = key(xx); relu2 = relu(kk)^2; out_ffn = value(relu2) */
    mlx_array kv = mlx_array_new();
    if (ok) {
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new();
        char b[340]; tname(m->cfg.layer_prefix, layer_idx, "ffn.key", b, sizeof(b));
        ok = multishard_get_triple(&m->weights, b, &w, &sc, &bb)
             && rw_lin(&kv, m, xx, w, sc, bb, gsz, bts, s);
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb);
    }
    mlx_array r2 = mlx_array_new();
    if (ok) {
        mlx_array zero = mlx_array_new_float32(0.0f), rl = mlx_array_new();
        ok = mlx_maximum(&rl, kv, zero, s) == 0 && mlx_square(&r2, rl, s) == 0;
        mlx_array_free(zero); mlx_array_free(rl);
    }
    mlx_array fout = mlx_array_new();
    if (ok) {
        mlx_array w = mlx_array_new(), sc = mlx_array_new(), bb = mlx_array_new();
        char b[340]; tname(m->cfg.layer_prefix, layer_idx, "ffn.value", b, sizeof(b));
        ok = multishard_get_triple(&m->weights, b, &w, &sc, &bb)
             && rw_lin(&fout, m, r2, w, sc, bb, gsz, bts, s);
        mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bb);
    }
    if (ok) { ok = mlx_add(out, h, fout, s) == 0; CHK(ok, "residual ffn"); }

    mlx_array_free(fnw); mlx_array_free(fnb); mlx_array_free(hn); mlx_array_free(xprev);
    mlx_array_free(xk); mlx_array_free(d); mlx_array_free(dm); mlx_array_free(xx);
    mlx_array_free(kv); mlx_array_free(r2); mlx_array_free(fout);
    return ok;
#undef CHK
}
