/* ============================================================================
 * llm_mlx_train.c — dominio training/DPO differenziabile (R3.3): forward in-graph + masked CE,
 * value_and_grad w.r.t. LoRA A/B, loss DPO. Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

/* §D R3.3 — TRAINING FORWARD DIFFERENZIABILE (path denso). La closure imposta
 * m->lora_A/B[li] ai tensori del grafo (input del transform) e calcola il forward INTERAMENTE
 * in-graph (nessun readback host): embed -> N layer -> final_norm -> lm_head -> masked CE
 * (mean sul batch). value_and_grad w.r.t. [A,B] -> dA/dB. Verifica: finite differences. */

/* Forward in-graph + masked CE. sum_mode=false -> MEDIA sui token mascherati (SFT); true ->
 * SOMMA dei CE mascherati (= -sum logp, per DPO). mask NULL = tutte le posizioni. */
mlx_array llm_mlx_train_ce_impl(LlmMlxModel* m, const int32_t* toks, const int32_t* tgt,
                              const uint8_t* mask, int T, bool sum_mode, bool* ok_out) {
    OutputTensors t;
    bool ok = load_output_tensors(m, &t);
    int nl = m->cfg.n_layers;
    mlx_array* kc = (mlx_array*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(mlx_array));
    mlx_array* vc = (mlx_array*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(mlx_array));
    bool* hc = (bool*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(bool));
    for (int i = 0; i < nl; i++) { kc[i] = mlx_array_new(); vc[i] = mlx_array_new(); }
    mlx_array h = mlx_array_new(), normed = mlx_array_new(), logits = mlx_array_new();
    mlx_array lf = mlx_array_new(), mx = mlx_array_new(), cen = mlx_array_new();
    mlx_array ex = mlx_array_new(), sx = mlx_array_new(), lse = mlx_array_new();
    mlx_array tg = mlx_array_new(), picked = mlx_array_new(), ce = mlx_array_new();
    mlx_array cef = mlx_array_new(), num = mlx_array_new(), den = mlx_array_new(), loss = mlx_array_new();
    if (ok) ok = forward_tokens_batch(m, NULL, t.embed_w, t.embed_s, t.embed_b, toks, T, 0, kc, vc, hc, &h);
    if (ok) ok = llm_mlx_final_norm(m, h, t.final_norm_w, &normed);
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                      m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    if (ok) {
        int V = (int)mlx_array_shape(logits)[2];
        int s2[2] = { T, V };
        ok = mlx_reshape(&lf, logits, s2, 2, m->stream) == 0;
        if (ok) ok = mlx_max_axis(&mx, lf, 1, true, m->stream) == 0;
        if (ok) ok = mlx_subtract(&cen, lf, mx, m->stream) == 0;
        if (ok) ok = mlx_exp(&ex, cen, m->stream) == 0;
        if (ok) ok = mlx_sum_axis(&sx, ex, 1, true, m->stream) == 0;
        if (ok) ok = mlx_log(&lse, sx, m->stream) == 0;
        if (ok) ok = mlx_add(&lse, lse, mx, m->stream) == 0;
        int ts[2] = { T, 1 };
        if (ok) { tg = mlx_array_new_data(tgt, ts, 2, MLX_INT32); ok = tg.ctx != NULL; }
        if (ok) ok = mlx_take_along_axis(&picked, lf, tg, 1, m->stream) == 0;
        if (ok) ok = mlx_subtract(&ce, lse, picked, m->stream) == 0;
        { int s1[1] = { T }; if (ok) ok = mlx_reshape(&cef, ce, s1, 1, m->stream) == 0; }
        if (ok && mask) {
            float* mf = (float*)malloc((size_t)T * 4);
            for (int i = 0; i < T; i++) mf[i] = mask[i] ? 1.0f : 0.0f;
            int ms[1] = { T };
            mlx_array mk = mlx_array_new_data(mf, ms, 1, MLX_FLOAT32); free(mf);
            mlx_array cefm = mlx_array_new();
            ok = mk.ctx != NULL && mlx_multiply(&cefm, cef, mk, m->stream) == 0 &&
                 mlx_sum_axis(&num, cefm, 0, false, m->stream) == 0;
            if (ok && !sum_mode) {   /* media: den = somma delle mask */
                mlx_array mk2 = mlx_array_new(), dmax = mlx_array_new(), mone = mlx_array_new_float(1e-6f);
                ok = mlx_sum_axis(&mk2, mk, 0, false, m->stream) == 0 &&
                     mlx_maximum(&dmax, mk2, mone, m->stream) == 0;
                if (ok) mlx_array_set(&den, dmax);
                mlx_array_free(mk2); mlx_array_free(dmax); mlx_array_free(mone);
            }
            mlx_array_free(mk); mlx_array_free(cefm);
        } else if (ok) {
            ok = mlx_sum_axis(&num, cef, 0, false, m->stream) == 0;
        }
        if (ok && !sum_mode) { if (den.ctx == NULL) { den = mlx_array_new_float((float)T); ok = den.ctx != NULL; } }
        if (ok) {
            if (sum_mode) { ok = mlx_array_set(&loss, num) == 0; }
            else ok = mlx_divide(&loss, num, den, m->stream) == 0;
        }
    }
    mlx_array_free(h); mlx_array_free(normed); mlx_array_free(logits);
    mlx_array_free(lf); mlx_array_free(mx); mlx_array_free(cen); mlx_array_free(ex); mlx_array_free(sx);
    mlx_array_free(lse); mlx_array_free(tg); mlx_array_free(picked); mlx_array_free(ce);
    mlx_array_free(cef); mlx_array_free(num); mlx_array_free(den);
    for (int i = 0; i < nl; i++) { mlx_array_free(kc[i]); mlx_array_free(vc[i]); }
    free(kc); free(vc); free(hc);
    free_output_tensors(&t);
    if (ok_out) *ok_out = ok;
    if (!ok) { mlx_array_free(loss); return mlx_array_new(); }
    return loss;
}

static int llm_train_loss_fn(mlx_vector_array* res, const mlx_vector_array input, void* payload) {
    GtCtx* c = (GtCtx*)payload;
    mlx_array A = mlx_array_new(), B = mlx_array_new();
    mlx_vector_array_get(&A, input, 0);
    mlx_vector_array_get(&B, input, 1);
    mlx_array_set(&c->m->lora_A[c->li], A);
    mlx_array_set(&c->m->lora_B[c->li], B);
    bool ok = true;
    mlx_array loss = llm_mlx_train_ce_impl(c->m, c->toks, c->tgt, c->mask, c->T, /*sum_mode=*/false, &ok);
    if (ok) mlx_vector_array_set_value(res, loss);
    mlx_array_free(loss);
    return ok ? 0 : 1;
}

/* R3.3: SDPA EAGER differenziabile (matmul+softmax+matmul causale) — sostituisce fast_sdpa
 * (CustomKernel senza VJP) nel forward di training, via llm_train_sdpa_hook. */
bool llm_mlx_train_sdpa_eager_impl(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                 float scale, void* ctx, mlx_stream st) {
    (void)ctx;
    int B = (int)mlx_array_shape(q)[0], nh = (int)mlx_array_shape(q)[1];
    int Lq = (int)mlx_array_shape(q)[2], D = (int)mlx_array_shape(q)[3];
    int nkv = (int)mlx_array_shape(k)[1], Lk = (int)mlx_array_shape(k)[2];
    if (nkv <= 0 || nh % nkv != 0 || D <= 0) return false;
    { static int _p = 0; if (!_p) { fprintf(stderr, "[train] eager sdpa hook usato (B=%d nh=%d Lq=%d Lk=%d)\n", B, nh, Lq, Lk); _p = 1; } }
    int g = nh / nkv;
    mlx_array kr = mlx_array_new(), vr = mlx_array_new();
    bool ok;
    if (g > 1) {
        /* GQA: reshape+broadcast (repeat_axis ha VJP sospetto) [B,nkv,L,D]->[B,nh,L,D]. */
        int e5[5] = { B, nkv, 1, Lk, D }, b5[5] = { B, nkv, g, Lk, D }, r4[4] = { B, nh, Lk, D };
        mlx_array ke = mlx_array_new(), kb = mlx_array_new(), ve = mlx_array_new(), vb = mlx_array_new();
        ok = mlx_reshape(&ke, k, e5, 5, st) == 0 && mlx_broadcast_to(&kb, ke, b5, 5, st) == 0 && mlx_reshape(&kr, kb, r4, 4, st) == 0 &&
             mlx_reshape(&ve, v, e5, 5, st) == 0 && mlx_broadcast_to(&vb, ve, b5, 5, st) == 0 && mlx_reshape(&vr, vb, r4, 4, st) == 0;
        mlx_array_free(ke); mlx_array_free(kb); mlx_array_free(ve); mlx_array_free(vb);
    } else {
        ok = mlx_array_set(&kr, k) == 0 && mlx_array_set(&vr, v) == 0;
    }
    mlx_array kt = mlx_array_new();
    if (ok) { int perm[4] = { 0, 1, 3, 2 }; ok = mlx_transpose_axes(&kt, kr, perm, 4, st) == 0; }
    mlx_array sc = mlx_array_new(), sca = mlx_array_new(), scl = mlx_array_new();
    if (ok) ok = mlx_matmul(&sc, q, kt, st) == 0;                        /* [B,nh,Lq,Lk] */
    if (ok) { scl = mlx_array_new_float(scale); ok = scl.ctx != NULL; }
    if (ok) ok = mlx_multiply(&sca, sc, scl, st) == 0;
    mlx_array mask = mlx_array_new(), msc = mlx_array_new();
    if (ok) {
        int off = Lk - Lq;
        float* md = (float*)malloc((size_t)Lq * (size_t)Lk * sizeof(float));
        if (!md) ok = false;
        else {
            for (int i = 0; i < Lq; i++) for (int j = 0; j < Lk; j++)
                md[(size_t)i * Lk + j] = (j <= off + i) ? 0.0f : -INFINITY;
            int ms[4] = { 1, 1, Lq, Lk };
            mask = mlx_array_new_data(md, ms, 4, MLX_FLOAT32);
            free(md);
            ok = mask.ctx != NULL;
        }
    }
    if (ok) ok = mlx_add(&msc, sca, mask, st) == 0;
    mlx_array sm = mlx_array_new();
    if (ok) ok = mlx_softmax_axis(&sm, msc, -1, /*precise=*/true, st) == 0;
    if (ok) ok = mlx_matmul(out, sm, vr, st) == 0;
    mlx_array_free(kr); mlx_array_free(vr); mlx_array_free(kt); mlx_array_free(sc);
    mlx_array_free(sca); mlx_array_free(scl); mlx_array_free(mask); mlx_array_free(msc); mlx_array_free(sm);
    return ok;
}

/* R3.3: loss (masked CE) + gradiente LoRA (A,B) su (layer, proj) per il path denso.
 * `out_dA`/`out_dB` NULL -> solo loss. Ritorna 0 ok. */
int llm_ai_mlx_train_lora_loss_grad_mask(llm_ai_mlx_model_handle mh,
                                         const int32_t* toks, const int32_t* targets, const uint8_t* mask, int T,
                                         int layer, const char* proj,
                                    const float* A, int Ar, int Ac,
                                    const float* B, int Br, int Bc,
                                    float* out_loss, float* out_dA, float* out_dB) {
    if (!mh || !toks || !targets || T <= 0 || !A || !B || Ac <= 0) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    const char* pr = proj ? proj : "v_proj";
    /* R3.3: l'autograd MLX non ha il VJP per i CustomKernel (kernel Metal fusi) ne' per
     * mlx_compile -> per il forward di training si disabilitano i gate (prima del primo forward,
     * quando le cache statiche non sono ancora state risolte). */
    /* default-ON (opt-out con "0"): swiglu Metal fuso, swiglu compiled, compile-fuse. */
    /* attivazioni F32: la loss in bf16 quantizza il FD (gradiente piccolo). */
    setenv("LLM_BF16_DENSE", "fp32", 1);
    { const char* z[] = { "LLMMETAL_FUSE_SWIGLU_METAL","LLM_FUSE_SWIGLU_COMPILE","LLM_COMPILE_FUSE" };
      for (size_t i = 0; i < sizeof(z)/sizeof(z[0]); i++) setenv(z[i], "0", 1); }
    /* presence-gated (default OFF): remove del tutto. */
    { const char* off[] = { "LLM_FUSE_ROPE","LLM_FUSE_ROPE_VERIFY","LLM_FUSE_SWIGLU_PREFILL",
        "LLMMETAL_FUSE_GEGLU_METAL","LLM_FUSE_GEGLU","LLM_FUSE_GATEUP","LLM_GU_FUSE",
        "LLMENGINE_MOE_FUSE_MOE_DECODE","LLMENGINE_MOE_FUSE_MOE_DECODE_VERIFY","LLMENGINE_MOE_FUSE_MOE_MXFP4","LLM_GEMVQ4",
        "LLM_GEMVQ4_VERIFY","LLMMETAL_KVMETAL","LLMMETAL_KVMETAL_AUTH","LLMMETAL_KVMETAL_VERIFY",
        "LLMMETAL_GPTOSS_SWIGLU_METAL","LLMENGINE_MOE_GATEUP_FUSE","LLM_PACKED_QKV" };
      for (size_t i = 0; i < sizeof(off)/sizeof(off[0]); i++) unsetenv(off[i]); }
    llm_ai_mlx_model_lora_clear(mh);
    if (llm_ai_mlx_model_lora_add(mh, layer, pr, Ac, 1.0f, A, B, Ar, Br) != 0) return -1;
    llm_ai_mlx_model_lora_set_enabled(mh, 1);
    int li = llm_mlx_model_lora_get(m, layer, pr);
    if (li < 0) return -1;
    GtCtx ctx = { m, toks, T, targets, li, mask };
    llm_train_sdpa_hook.fn = llm_mlx_train_sdpa_eager_impl; llm_train_sdpa_hook.ctx = NULL;  /* R3.3 */
    mlx_closure cl = mlx_closure_new_func_payload(llm_train_loss_fn, &ctx, NULL);
    int sA[2] = { Ar, Ac }, sB[2] = { Br, Bc };
    mlx_array Aa = mlx_array_new_data(A, sA, 2, MLX_FLOAT32);
    mlx_array Ba = mlx_array_new_data(B, sB, 2, MLX_FLOAT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, Aa);
    mlx_vector_array_append_value(in, Ba);
    int rc = -1;
    if (out_dA || out_dB) {
        int argnums[2] = { 0, 1 };
        mlx_closure_value_and_grad vg = mlx_closure_value_and_grad_new();
        mlx_vector_array vals = mlx_vector_array_new(), grads = mlx_vector_array_new();
        if (mlx_value_and_grad(&vg, cl, argnums, 2) == 0 &&
            mlx_closure_value_and_grad_apply(&vals, &grads, vg, in) == 0) {
            mlx_array lv = mlx_array_new(), gA = mlx_array_new(), gB = mlx_array_new();
            mlx_vector_array_get(&lv, vals, 0);
            mlx_vector_array_get(&gA, grads, 0);
            mlx_vector_array_get(&gB, grads, 1);
            mlx_array lvc = mlx_array_new(), gAc = mlx_array_new(), gBc = mlx_array_new();
            /* NB: il gradiente del VJP e' LAZY/non contiguo -> mlx_contiguous (gatage mlx-c). */
            bool eok = mlx_contiguous(&lvc, lv, false, m->stream) == 0 &&
                       mlx_contiguous(&gAc, gA, false, m->stream) == 0 &&
                       mlx_contiguous(&gBc, gB, false, m->stream) == 0 &&
                       mlx_array_eval(lvc) == 0 && mlx_array_eval(gAc) == 0 && mlx_array_eval(gBc) == 0;
            if (eok) {
                if (out_loss) *out_loss = mlx_array_data_float32(lvc)[0];
                if (out_dA) memcpy(out_dA, mlx_array_data_float32(gAc), (size_t)Ar * (size_t)Ac * 4);
                if (out_dB) memcpy(out_dB, mlx_array_data_float32(gBc), (size_t)Br * (size_t)Bc * 4);
                rc = 0;
            }
            mlx_array_free(lvc); mlx_array_free(gAc); mlx_array_free(gBc);
            mlx_array_free(lv); mlx_array_free(gA); mlx_array_free(gB);
        }
        mlx_vector_array_free(vals); mlx_vector_array_free(grads);
        mlx_closure_value_and_grad_free(vg);
    } else {
        mlx_vector_array vals = mlx_vector_array_new();
        if (mlx_closure_apply(&vals, cl, in) == 0 && mlx_vector_array_size(vals) >= 1) {
            mlx_array lv = mlx_array_new();
            mlx_vector_array_get(&lv, vals, 0);
            mlx_array lvc = mlx_array_new();
            if (mlx_contiguous(&lvc, lv, false, m->stream) == 0 && mlx_array_eval(lvc) == 0) {
                if (out_loss) *out_loss = mlx_array_data_float32(lvc)[0];
                rc = 0;
            }
            mlx_array_free(lvc); mlx_array_free(lv);
        }
        mlx_vector_array_free(vals);
    }
    mlx_vector_array_free(in);
    mlx_closure_free(cl);
    llm_train_sdpa_hook.fn = NULL; llm_train_sdpa_hook.ctx = NULL;   /* R3.3: ripristina */
    return rc;
}

/* Wrapper retro-compatibile (nessuna mask: CE media su tutte le posizioni). */
int llm_ai_mlx_train_lora_loss_grad(llm_ai_mlx_model_handle mh,
                                    const int32_t* toks, const int32_t* targets, int T,
                                    int layer, const char* proj,
                                    const float* A, int Ar, int Ac,
                                    const float* B, int Br, int Bc,
                                    float* out_loss, float* out_dA, float* out_dB) {
    return llm_ai_mlx_train_lora_loss_grad_mask(mh, toks, targets, NULL, T, layer, proj,
                                                A, Ar, Ac, B, Br, Bc, out_loss, out_dA, out_dB);
}

/* §D R7 — DPO: closure che calcola la loss di preference in-graph.
 * z = beta * ((lp_w - ref_w) - (lp_l - ref_l)) con lp = -ce_sum; loss = softplus(-z).
 * I passi "ref" girano con l'adapter OFF -> nessun gradiente verso A/B. */

static int llm_dpo_loss_fn(mlx_vector_array* res, const mlx_vector_array input, void* payload) {
    GdpoCtx* c = (GdpoCtx*)payload;
    LlmMlxModel* m = c->m;
    mlx_array A = mlx_array_new(), B = mlx_array_new();
    mlx_vector_array_get(&A, input, 0);
    mlx_vector_array_get(&B, input, 1);
    mlx_array_set(&m->lora_A[c->li], A);
    mlx_array_set(&m->lora_B[c->li], B);

    bool ok = true;
    mlx_array ce_w = llm_mlx_train_ce_impl(m, c->tw, c->tgw, c->mw, c->Tw, /*sum_mode=*/true, &ok);
    mlx_array ce_l = ok ? llm_mlx_train_ce_impl(m, c->tl, c->tgl, c->ml, c->Tl, true, &ok) : mlx_array_new();
    m->lora_enabled = false;          /* REFERENCE: adapter OFF (costante, fuori dal grafo) */
    mlx_array rc_w = ok ? llm_mlx_train_ce_impl(m, c->tw, c->tgw, c->mw, c->Tw, true, &ok) : mlx_array_new();
    mlx_array rc_l = ok ? llm_mlx_train_ce_impl(m, c->tl, c->tgl, c->ml, c->Tl, true, &ok) : mlx_array_new();
    m->lora_enabled = true;
    if (!ok) { m->lora_enabled = true; mlx_array_free(ce_w); mlx_array_free(ce_l); mlx_array_free(rc_w); mlx_array_free(rc_l); return 1; }

    mlx_array a1 = mlx_array_new(), a2 = mlx_array_new(), a = mlx_array_new(), z = mlx_array_new();
    mlx_array nz = mlx_array_new(), e = mlx_array_new(), e1 = mlx_array_new(), one = mlx_array_new();
    mlx_array sp = mlx_array_new(), b = mlx_array_new_float(c->beta);
    ok = mlx_subtract(&a1, rc_w, ce_w, m->stream) == 0 &&
         mlx_subtract(&a2, rc_l, ce_l, m->stream) == 0 &&
         mlx_subtract(&a, a1, a2, m->stream) == 0 &&
         mlx_multiply(&z, a, b, m->stream) == 0 &&
         mlx_negative(&nz, z, m->stream) == 0 &&
         mlx_exp(&e, nz, m->stream) == 0;
    one = mlx_array_new_float(1.0f);
    if (ok) ok = mlx_add(&e1, e, one, m->stream) == 0;
    if (ok) ok = mlx_log(&sp, e1, m->stream) == 0;      /* softplus(-z) = log(1+exp(-z)) */
    if (ok) mlx_vector_array_set_value(res, sp);
    mlx_array_free(ce_w); mlx_array_free(ce_l); mlx_array_free(rc_w); mlx_array_free(rc_l);
    mlx_array_free(a1); mlx_array_free(a2); mlx_array_free(a); mlx_array_free(z); mlx_array_free(nz);
    mlx_array_free(e); mlx_array_free(e1); mlx_array_free(one); mlx_array_free(sp); mlx_array_free(b);
    return ok ? 0 : 1;
}

/* §D R7 — DPO step: Preference(θ) sui log-prob di chosen (w) / rejected (l), ref = adapter OFF.
 * Ritorna loss DPO + dLoss/dA,dB (il gradiente passa SOLO dai passi policy). 0 ok / -1. */
int llm_ai_mlx_train_dpo_step(llm_ai_mlx_model_handle mh,
                              const int32_t* tw, const int32_t* tgw, const uint8_t* mw, int Tw,
                              const int32_t* tl, const int32_t* tgl, const uint8_t* ml, int Tl,
                              int layer, const char* proj,
                              const float* A, int Ar, int Ac, const float* B, int Br, int Bc,
                              float beta, float* out_loss, float* out_dA, float* out_dB) {
    if (!mh || !tw || !tgw || !tl || !tgl || Tw <= 0 || Tl <= 0 || !A || !B || Ac <= 0) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    const char* pr = proj ? proj : "v_proj";
    unsetenv("LLMMETAL_FUSE_SWIGLU_METAL");   /* R3.3: kernel Metal fusi non hanno VJP */
    setenv("LLMMETAL_FUSE_SWIGLU_METAL", "0", 1); setenv("LLM_BF16_DENSE", "fp32", 1);
    llm_ai_mlx_model_lora_clear(mh);
    if (llm_ai_mlx_model_lora_add(mh, layer, pr, Ac, 1.0f, A, B, Ar, Br) != 0) return -1;
    llm_ai_mlx_model_lora_set_enabled(mh, 1);
    int li = llm_mlx_model_lora_get(m, layer, pr);
    if (li < 0) return -1;
    GdpoCtx ctx = { m, tw, tgw, tl, tgl, mw, ml, Tw, Tl, li, beta };
    llm_train_sdpa_hook.fn = llm_mlx_train_sdpa_eager_impl; llm_train_sdpa_hook.ctx = NULL;
    mlx_closure cl = mlx_closure_new_func_payload(llm_dpo_loss_fn, &ctx, NULL);
    int sA[2] = { Ar, Ac }, sB[2] = { Br, Bc };
    mlx_array Aa = mlx_array_new_data(A, sA, 2, MLX_FLOAT32);
    mlx_array Ba = mlx_array_new_data(B, sB, 2, MLX_FLOAT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, Aa);
    mlx_vector_array_append_value(in, Ba);
    int rc = -1;
    if (out_dA || out_dB) {
        int argnums[2] = { 0, 1 };
        mlx_closure_value_and_grad vg = mlx_closure_value_and_grad_new();
        mlx_vector_array vals = mlx_vector_array_new(), grads = mlx_vector_array_new();
        if (mlx_value_and_grad(&vg, cl, argnums, 2) == 0 &&
            mlx_closure_value_and_grad_apply(&vals, &grads, vg, in) == 0) {
            mlx_array lv = mlx_array_new(), gA = mlx_array_new(), gB = mlx_array_new();
            mlx_vector_array_get(&lv, vals, 0); mlx_vector_array_get(&gA, grads, 0); mlx_vector_array_get(&gB, grads, 1);
            mlx_array lvc = mlx_array_new(), gAc = mlx_array_new(), gBc = mlx_array_new();
            bool eok = mlx_contiguous(&lvc, lv, false, m->stream) == 0 &&
                       mlx_contiguous(&gAc, gA, false, m->stream) == 0 &&
                       mlx_contiguous(&gBc, gB, false, m->stream) == 0 &&
                       mlx_array_eval(lvc) == 0 && mlx_array_eval(gAc) == 0 && mlx_array_eval(gBc) == 0;
            if (eok) {
                if (out_loss) *out_loss = mlx_array_data_float32(lvc)[0];
                if (out_dA) memcpy(out_dA, mlx_array_data_float32(gAc), (size_t)Ar * (size_t)Ac * 4);
                if (out_dB) memcpy(out_dB, mlx_array_data_float32(gBc), (size_t)Br * (size_t)Bc * 4);
                rc = 0;
            }
            mlx_array_free(lvc); mlx_array_free(gAc); mlx_array_free(gBc);
            mlx_array_free(lv); mlx_array_free(gA); mlx_array_free(gB);
        }
        mlx_vector_array_free(vals); mlx_vector_array_free(grads);
        mlx_closure_value_and_grad_free(vg);
    } else {   /* solo loss (nessun gradiente) */
        mlx_vector_array vals = mlx_vector_array_new();
        if (mlx_closure_apply(&vals, cl, in) == 0 && mlx_vector_array_size(vals) >= 1) {
            mlx_array lv = mlx_array_new(), lvc = mlx_array_new();
            mlx_vector_array_get(&lv, vals, 0);
            if (mlx_contiguous(&lvc, lv, false, m->stream) == 0 && mlx_array_eval(lvc) == 0) {
                if (out_loss) *out_loss = mlx_array_data_float32(lvc)[0];
                rc = 0;
            }
            mlx_array_free(lvc); mlx_array_free(lv);
        }
        mlx_vector_array_free(vals);
    }
    mlx_vector_array_free(in);
    mlx_closure_free(cl);
    llm_train_sdpa_hook.fn = NULL; llm_train_sdpa_hook.ctx = NULL;
    return rc;
}
