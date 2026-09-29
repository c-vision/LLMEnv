/* llm_mlx_cb.c — dominio CORE-BENCH / callback (llm_cb_*). Estratto da llmengine_mlx.c
 * (split fase 2), codice VERBATIM. Contiene llm_ai_mlx_core_bench (entry pubblica, decl. in
 * llmengine_mlx.h) + le funzioni static llm_cb_* usate da esso. Dipende dai CORE shared
 * (rms_norm_cfg/quantized_linear_cfg/plain_linear/llm_mlx_mode_set/...) ora extern.
 */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>   /* CLOCK_MONOTONIC */
#include "llm_mlx_geglu_kernel.h"
#include "llm_mlx_kv_quant_sdpa_kernel.h"
#include "llm_mlx_splash_store.h"
#include "llm_mlx_splash_attn.h"
#include "llm_mlx_moe_fused_kernel.h"
#include "llm_mlx_swiglu_kernel.h"
#include "llm_mlx_swiglu_gptoss_kernel.h"


static uint32_t g_cb_rng = 0x12345678u;
static uint32_t llm_cb_rand(void) {
    g_cb_rng ^= g_cb_rng << 13; g_cb_rng ^= g_cb_rng >> 17; g_cb_rng ^= g_cb_rng << 5;
    return g_cb_rng;
}
static mlx_array llm_cb_rand_arr(const int* shape, int ndim, mlx_dtype dt, mlx_stream s) {
    size_t n = 1; for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) return mlx_array_new();
    for (size_t i = 0; i < n; i++) buf[i] = ((float)((llm_cb_rand() >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
    mlx_array a = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
    free(buf);
    if (dt == MLX_FLOAT32) return a;
    mlx_array b = mlx_array_new();
    if (mlx_astype(&b, a, dt, s) != 0) mlx_array_set(&b, a);
    mlx_array_free(a);
    return b;
}
typedef bool (*llm_cb_fn)(mlx_array* out, void* ctx);
static double llm_cb_run(llm_cb_fn fn, void* ctx, mlx_array* result, int iters, mlx_stream s) {
    mlx_array acc = mlx_array_new();
    for (int i = 0; i < 3; i++) { if (fn(&acc, ctx)) { mlx_array_eval(acc); } }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        if (fn(&acc, ctx)) {
            /* async (no sync per-call): misura il THROUGHPUT del pipeline, il regime del
             * loop di decode. Il drain finale attende la GPU. La sync per-call (~0.15 ms)
             * avrebbe reso il rapporto inutilizzabile (±5% sul rumore, anche per op 1:1). */
            mlx_vector_array v = mlx_vector_array_new();
            mlx_vector_array_append_value(v, acc);
            mlx_async_eval(v);
            mlx_vector_array_free(v);
        }
    }
    if (acc.ctx != NULL) mlx_array_eval(acc); /* drain */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    mlx_array_set(result, acc); mlx_array_free(acc);
    return ((double)(t1.tv_sec - t0.tv_sec) * 1e3) + ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);
}
/* Encoding throughput: costruisce K grafi SEMPRE senza sync (l'ultimo graph scarta il
 * precedente senza valutarlo -> nessuna crescita di memoria), un solo eval finale.
 * Misura il costo CPU di ENCODING per op (dispatch del grafo), distinto dal costo GPU. */
static double llm_cb_run_enc(llm_cb_fn fn, void* ctx, mlx_array* result, int iters, mlx_stream s) {
    mlx_array acc = mlx_array_new();
    for (int i = 0; i < 3; i++) { fn(&acc, ctx); }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) { fn(&acc, ctx); }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    mlx_array_eval(acc);
    mlx_array_set(result, acc); mlx_array_free(acc);
    return (((double)(t1.tv_sec - t0.tv_sec) * 1e6) + ((double)(t1.tv_nsec - t0.tv_nsec) / 1e3)) / (double)iters;
}
static double llm_cb_maxdiff(mlx_array a, mlx_array b, mlx_stream s) {
    mlx_array af = mlx_array_new(), bf = mlx_array_new();
    double md = -1.0;
    if (mlx_astype(&af, a, MLX_FLOAT32, s) == 0 && mlx_astype(&bf, b, MLX_FLOAT32, s) == 0 &&
        mlx_array_eval(af) == 0 && mlx_array_eval(bf) == 0) {
        const float* ad = mlx_array_data_float32(af);
        const float* bd = mlx_array_data_float32(bf);
        size_t n = mlx_array_size(af);
        if (ad && bd && mlx_array_size(bf) == n) {
            md = 0.0;
            for (size_t i = 0; i < n; i++) { double d = fabs((double)ad[i] - (double)bd[i]); if (d > md) md = d; }
        }
    }
    mlx_array_free(af); mlx_array_free(bf);
    return md;
}
static void llm_cb_emit(char* out, size_t cap, const char* s) {
    size_t used = strlen(out);
    if (used + 1 < cap) strncat(out, s, cap - used - 1);
}

static bool llm_cb_rms_raw(mlx_array* o, void* c) { llm_cb_rms_ctx* p = (llm_cb_rms_ctx*)c; return mlx_fast_rms_norm(o, p->x, p->w, p->eps, p->s) == 0; }
static bool llm_cb_rms_llm(mlx_array* o, void* c) { llm_cb_rms_ctx* p = (llm_cb_rms_ctx*)c; return rms_norm_cfg(o, p->x, p->w, p->eps, p->s); }

static bool llm_cb_ql_raw(mlx_array* o, void* c) {
    llm_cb_ql_ctx* p = (llm_cb_ql_ctx*)c;
    mlx_optional_int g = {p->gs, true}, b = {p->bits, true};
    return mlx_quantized_matmul(o, p->x, p->wq, p->sc, p->bi, true, g, b, "affine", p->s) == 0;
}
static bool llm_cb_ql_llm(mlx_array* o, void* c) {
    llm_cb_ql_ctx* p = (llm_cb_ql_ctx*)c;
    return quantized_linear_cfg(o, p->x, p->wq, p->sc, p->bi, p->gs, p->bits, "affine", p->s);
}

static bool llm_cb_sd_raw(mlx_array* o, void* c) {
    llm_cb_sd_ctx* p = (llm_cb_sd_ctx*)c;
    return mlx_fast_scaled_dot_product_attention(o, p->q, p->k, p->v, p->scale, "",
                                                 g_mlx_empty_arr, g_mlx_empty_arr, p->s) == 0;
}
static bool llm_cb_sd_llm(mlx_array* o, void* c) {
    llm_cb_sd_ctx* p = (llm_cb_sd_ctx*)c;
    return llm_mlx_gqa_sdpa(o, p->q, p->k, p->v, p->scale, /*window=*/0, p->s);
}

/* kvq_sdpa — dequant-on-read (nuovo kernel) vs path corrente dequant+concat+SDPA. */
static bool llm_cb_kvq_raw(mlx_array* o, void* c) {
    llm_cb_kvq_ctx* p = (llm_cb_kvq_ctx*)c;
    mlx_array kd = mlx_array_new(), vd = mlx_array_new();
    mlx_array k16 = mlx_array_new(), v16 = mlx_array_new();
    mlx_array kc = mlx_array_new(), vc = mlx_array_new();
    mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new(), att = mlx_array_new();
    mlx_vector_array catk = mlx_vector_array_new(), catv = mlx_vector_array_new();
    int nh = mlx_array_shape(p->q)[0], hd = mlx_array_shape(p->q)[1];
    mlx_dtype T = mlx_array_dtype(p->q);
    bool ok = llm_mlx_kv_dequantize(&kd, p->Kc_w, p->Kc_s, p->Kc_b, p->bits, p->gs, p->s) &&
              llm_mlx_kv_dequantize(&vd, p->Vc_w, p->Vc_s, p->Vc_b, p->vbits, p->gs, p->s);
    if (ok) ok = mlx_astype(&k16, kd, T, p->s) == 0;
    if (ok) ok = mlx_astype(&v16, vd, T, p->s) == 0;
    if (ok) {
        mlx_vector_array_append_value(catk, k16); mlx_vector_array_append_value(catk, p->Kr);
        mlx_vector_array_append_value(catv, v16); mlx_vector_array_append_value(catv, p->Vr);
        ok = mlx_concatenate_axis(&kc, catk, 1, p->s) == 0 &&
             mlx_concatenate_axis(&vc, catv, 1, p->s) == 0;
    }
    if (ok) {
        int nkv = mlx_array_shape(kc)[0], S = mlx_array_shape(kc)[1];
        int qsh[4] = {1, nh, 1, hd}, ksh[4] = {1, nkv, S, hd};
        ok = mlx_reshape(&q4, p->q, qsh, 4, p->s) == 0 &&
             mlx_reshape(&k4, kc, ksh, 4, p->s) == 0 &&
             mlx_reshape(&v4, vc, ksh, 4, p->s) == 0;
    }
    if (ok) ok = llm_mlx_gqa_sdpa(&att, q4, k4, v4, p->scale, /*window=*/0, p->s);
    if (ok) { int osh[2] = {nh, hd}; ok = mlx_reshape(o, att, osh, 2, p->s) == 0; }
    mlx_array_free(kd); mlx_array_free(vd); mlx_array_free(k16); mlx_array_free(v16);
    mlx_array_free(kc); mlx_array_free(vc); mlx_array_free(q4); mlx_array_free(k4);
    mlx_array_free(v4); mlx_array_free(att);
    mlx_vector_array_free(catk); mlx_vector_array_free(catv);
    return ok;
}
static bool llm_cb_kvq_llm(mlx_array* o, void* c) {
    llm_cb_kvq_ctx* p = (llm_cb_kvq_ctx*)c;
    return llm_mlx_kvq_sdpa_decode(o, p->q, p->Kc_w, p->Kc_s, p->Kc_b, p->Vc_w, p->Vc_s, p->Vc_b,
                                   p->Kr, p->Vr, 0, p->bits, p->vbits, p->scale, p->s);
}

static mlx_array llm_cb_scalar_bf16(float v, mlx_stream s) {
    mlx_array t = mlx_array_new_float32(v);
    mlx_array b = mlx_array_new();
    if (mlx_astype(&b, t, MLX_BFLOAT16, s) != 0) mlx_array_set(&b, t);
    mlx_array_free(t);
    return b;
}

static bool llm_cb_sw_eager(mlx_array* o, void* c) {
    llm_cb_sw_ctx* p = (llm_cb_sw_ctx*)c;
    mlx_array sig = mlx_array_new(), s2 = mlx_array_new();
    bool ok = mlx_sigmoid(&sig, p->gate, p->s) == 0 &&
              mlx_multiply(&s2, sig, p->gate, p->s) == 0 &&
              mlx_multiply(o, s2, p->up, p->s) == 0;
    mlx_array_free(sig); mlx_array_free(s2);
    return ok;
}
static bool llm_cb_sw_metal(mlx_array* o, void* c) {
    llm_cb_sw_ctx* p = (llm_cb_sw_ctx*)c;
    return llm_mlx_swiglu_metal(p->gate, p->up, o, p->s);
}
static bool llm_cb_gos_eager(mlx_array* o, void* c) {
    llm_cb_sw_ctx* p = (llm_cb_sw_ctx*)c;
    mlx_array seven = llm_cb_scalar_bf16(7.0f, p->s), mseven = llm_cb_scalar_bf16(-7.0f, p->s);
    mlx_array one = llm_cb_scalar_bf16(1.0f, p->s), alpha = llm_cb_scalar_bf16(1.702f, p->s);
    mlx_array cg = mlx_array_new(), cu = mlx_array_new(), ag = mlx_array_new(), sig = mlx_array_new(), u1 = mlx_array_new();
    bool ok = mlx_minimum(&cg, p->gate, seven, p->s) == 0 &&
              mlx_clip(&cu, p->up, mseven, seven, p->s) == 0 &&
              mlx_multiply(&ag, cg, alpha, p->s) == 0 &&
              mlx_sigmoid(&sig, ag, p->s) == 0 &&
              mlx_multiply(&sig, sig, cg, p->s) == 0 &&
              mlx_add(&u1, cu, one, p->s) == 0 &&
              mlx_multiply(o, sig, u1, p->s) == 0;
    mlx_array_free(seven); mlx_array_free(mseven); mlx_array_free(one); mlx_array_free(alpha);
    mlx_array_free(cg); mlx_array_free(cu); mlx_array_free(ag); mlx_array_free(sig); mlx_array_free(u1);
    return ok;
}
static bool llm_cb_gos_metal(mlx_array* o, void* c) {
    llm_cb_sw_ctx* p = (llm_cb_sw_ctx*)c;
    return llm_mlx_swiglu_gptoss_metal(p->gate, p->up, o, p->s);
}
static bool llm_cb_gg_eager(mlx_array* o, void* c) {
    llm_cb_sw_ctx* p = (llm_cb_sw_ctx*)c;
    mlx_array three = llm_cb_scalar_bf16(3.0f, p->s), c447 = llm_cb_scalar_bf16(0.044715f, p->s);
    mlx_array s2pi = llm_cb_scalar_bf16(0.7978845608028654f, p->s), one = llm_cb_scalar_bf16(1.0f, p->s);
    mlx_array half = llm_cb_scalar_bf16(0.5f, p->s);
    mlx_array g3 = mlx_array_new(), g3s = mlx_array_new(), inner = mlx_array_new();
    mlx_array isc = mlx_array_new(), th = mlx_array_new(), opt = mlx_array_new(), go = mlx_array_new();
    bool ok = mlx_power(&g3, p->gate, three, p->s) == 0 &&
              mlx_multiply(&g3s, g3, c447, p->s) == 0 &&
              mlx_add(&inner, p->gate, g3s, p->s) == 0 &&
              mlx_multiply(&isc, inner, s2pi, p->s) == 0 &&
              mlx_tanh(&th, isc, p->s) == 0 &&
              mlx_add(&opt, th, one, p->s) == 0 &&
              mlx_multiply(&go, p->gate, opt, p->s) == 0 &&
              mlx_multiply(&go, go, half, p->s) == 0 &&
              mlx_multiply(o, go, p->up, p->s) == 0;
    mlx_array_free(three); mlx_array_free(c447); mlx_array_free(s2pi); mlx_array_free(one); mlx_array_free(half);
    mlx_array_free(g3); mlx_array_free(g3s); mlx_array_free(inner); mlx_array_free(isc);
    mlx_array_free(th); mlx_array_free(opt); mlx_array_free(go);
    return ok;
}
static bool llm_cb_gg_metal(mlx_array* o, void* c) {
    llm_cb_sw_ctx* p = (llm_cb_sw_ctx*)c;
    return llm_mlx_geglu_metal(p->gate, p->up, o, p->s);
}

static bool llm_cb_ls_raw(mlx_array* o, void* c) {
    llm_cb_ls_ctx* p = (llm_cb_ls_ctx*)c;
    mlx_array cap = mlx_array_new_float32(p->cap), sh = mlx_array_new(), t = mlx_array_new();
    bool ok = mlx_divide(&sh, p->logits, cap, p->s) == 0 &&
              mlx_tanh(&t, sh, p->s) == 0 &&
              mlx_multiply(o, t, cap, p->s) == 0;
    mlx_array_free(cap); mlx_array_free(sh); mlx_array_free(t);
    return ok;
}
static bool llm_cb_ls_llm(mlx_array* o, void* c) {
    llm_cb_ls_ctx* p = (llm_cb_ls_ctx*)c;
    return llm_mlx_logit_softcap(o, p->logits, p->cap, p->s);
}

static bool llm_cb_emb_raw(mlx_array* o, void* c) {
    llm_cb_emb_ctx* p = (llm_cb_emb_ctx*)c;
    mlx_array wr = mlx_array_new(), sr = mlx_array_new(), br = mlx_array_new(), flat = mlx_array_new();
    mlx_optional_int g = {p->gs, true}, b = {p->bits, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    bool ok = mlx_take_axis(&wr, p->emb_w, p->tid, 0, p->s) == 0 &&
              mlx_take_axis(&sr, p->emb_s, p->tid, 0, p->s) == 0 &&
              mlx_take_axis(&br, p->emb_b, p->tid, 0, p->s) == 0 &&
              mlx_dequantize(&flat, wr, sr, br, g, b, p->mode, g_mlx_empty_arr, dt, p->s) == 0;
    if (ok) { const int sh[3] = {1, 1, p->hidden}; ok = mlx_reshape(o, flat, sh, 3, p->s) == 0; }
    mlx_array_free(wr); mlx_array_free(sr); mlx_array_free(br); mlx_array_free(flat);
    return ok;
}
static bool llm_cb_emb_llm(mlx_array* o, void* c) {
    llm_cb_emb_ctx* p = (llm_cb_emb_ctx*)c;
    LlmMlxModel* fm = (LlmMlxModel*)calloc(1, sizeof(LlmMlxModel));
    if (!fm) return false;
    fm->cfg.hidden_size = p->hidden;
    fm->cfg.quant_group_size = p->gs;
    fm->cfg.quant_bits = p->bits;
    llm_mlx_mode_set(fm->cfg.quant_mode, sizeof(fm->cfg.quant_mode), p->mode);
    fm->cfg.bfloat16_input = false;
    fm->cfg.embed_scale = 0.0f;
    fm->stream = p->s;
    bool ok = llm_mlx_embed_row_from_idx(fm, p->emb_w, p->emb_s, p->emb_b, p->tid, o);
    free(fm);
    return ok;
}

static bool llm_cb_kv_raw(mlx_array* o, void* c) {
    llm_cb_kv_ctx* p = (llm_cb_kv_ctx*)c;
    int nkv = mlx_array_shape(p->kc)[1], hd = mlx_array_shape(p->kc)[3];
    int s0[4] = {0,0,0,0}, sp[4] = {1,nkv,p->off,hd}, st[4] = {1,1,1,1};
    mlx_array used = mlx_array_new();
    mlx_vector_array v = mlx_vector_array_new();
    bool ok = mlx_slice(&used, p->kc, s0, 4, sp, 4, st, 4, p->s) == 0;
    if (ok) {
        mlx_vector_array_append_value(v, used);
        mlx_vector_array_append_value(v, p->kn);
        ok = mlx_concatenate_axis(o, v, 2, p->s) == 0;
    }
    mlx_vector_array_free(v); mlx_array_free(used);
    return ok;
}
static bool llm_cb_kv_llm(mlx_array* o, void* c) {
    llm_cb_kv_ctx* p = (llm_cb_kv_ctx*)c;
    mlx_array kv = mlx_array_new(), vv = mlx_array_new();
    bool ok = llm_mlx_kv_append_inplace(&p->kc, &p->vc, p->kn, p->vn, p->off, 1, 256, &kv, &vv, p->s);
    if (ok) mlx_array_set(o, kv);
    mlx_array_free(kv); mlx_array_free(vv);
    return ok;
}


static bool llm_cb_moe_fused(mlx_array* o, void* c) {
    llm_cb_moe_ctx* p = (llm_cb_moe_ctx*)c;
    return llm_mlx_moe_fused_decode_mxfp4(o, p->x, p->idx, p->gw, p->gsc, p->glb,
                                          p->uw, p->usc, p->ulb, p->dw, p->dsc, p->dlb,
                                          p->sc, p->din, p->dff, p->k, p->s);
}

/* Eager reference = stesso percorso di gpt_oss_ffn_block (unsort): 3 gather_qmm mxfp4,
 * bias lineari per-esperto, SwiGLU gpt-oss, scale per score, sum sui K. */
static bool llm_cb_moe_eager(mlx_array* o, void* c) {
    llm_cb_moe_ctx* p = (llm_cb_moe_ctx*)c;
    mlx_optional_int g = {32, true}, b = {4, true};
    mlx_array no_lhs = mlx_array_new(), x2 = mlx_array_new(), xb = mlx_array_new(), x_perm = mlx_array_new();
    mlx_array graw = mlx_array_new(), graw_r = mlx_array_new(), gt = mlx_array_new(), glbg = mlx_array_new();
    mlx_array uraw = mlx_array_new(), uraw_r = mlx_array_new(), ut = mlx_array_new(), ulbg = mlx_array_new();
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new(), act = mlx_array_new(), actx = mlx_array_new();
    mlx_array draw = mlx_array_new(), draw_r = mlx_array_new(), dt = mlx_array_new(), dlbg = mlx_array_new();
    mlx_array down_out = mlx_array_new(), w_arr = mlx_array_new(), scaled = mlx_array_new(), summed = mlx_array_new();
    bool ok = true;
    const int s2[2] = {1, p->din}, sb[2] = {p->k, p->din}, sp[3] = {p->k, 1, p->din};
    const int r4[4] = {1, p->k, 1, p->dff}, d4[4] = {1, p->k, 1, p->din};
    const int ax[3] = {p->k, 1, p->dff}, w4[4] = {1, p->k, 1, 1}, o1[1] = {p->din};
    ok = ok && mlx_reshape(&x2, p->x, s2, 2, p->s) == 0;
    ok = ok && mlx_broadcast_to(&xb, x2, sb, 2, p->s) == 0;
    ok = ok && mlx_reshape(&x_perm, xb, sp, 3, p->s) == 0;
    ok = ok && mlx_gather_qmm(&graw, x_perm, p->gw, p->gsc, g_mlx_empty_arr, no_lhs, p->idx,
                              true, g, b, "mxfp4", false, p->s) == 0;
    ok = ok && mlx_gather_qmm(&uraw, x_perm, p->uw, p->usc, g_mlx_empty_arr, no_lhs, p->idx,
                              true, g, b, "mxfp4", false, p->s) == 0;
    ok = ok && mlx_reshape(&graw_r, graw, r4, 4, p->s) == 0;
    ok = ok && mlx_reshape(&uraw_r, uraw, r4, 4, p->s) == 0;
    ok = ok && mlx_take_axis(&gt, p->glb, p->idx, 0, p->s) == 0;
    ok = ok && mlx_take_axis(&ut, p->ulb, p->idx, 0, p->s) == 0;
    ok = ok && mlx_reshape(&glbg, gt, r4, 4, p->s) == 0;
    ok = ok && mlx_reshape(&ulbg, ut, r4, 4, p->s) == 0;
    ok = ok && mlx_add(&gate_out, graw_r, glbg, p->s) == 0;
    ok = ok && mlx_add(&up_out, uraw_r, ulbg, p->s) == 0;
    ok = ok && llm_mlx_swiglu_gptoss_metal(gate_out, up_out, &act, p->s);
    ok = ok && mlx_reshape(&actx, act, ax, 3, p->s) == 0;
    ok = ok && mlx_gather_qmm(&draw, actx, p->dw, p->dsc, g_mlx_empty_arr, no_lhs, p->idx,
                              true, g, b, "mxfp4", false, p->s) == 0;
    ok = ok && mlx_reshape(&draw_r, draw, d4, 4, p->s) == 0;
    ok = ok && mlx_take_axis(&dt, p->dlb, p->idx, 0, p->s) == 0;
    ok = ok && mlx_reshape(&dlbg, dt, d4, 4, p->s) == 0;
    ok = ok && mlx_add(&down_out, draw_r, dlbg, p->s) == 0;
    ok = ok && mlx_reshape(&w_arr, p->sc, w4, 4, p->s) == 0;
    ok = ok && mlx_multiply(&scaled, down_out, w_arr, p->s) == 0;
    ok = ok && mlx_sum_axis(&summed, scaled, 1, false, p->s) == 0;
    if (ok) ok = mlx_reshape(o, summed, o1, 1, p->s) == 0;
    mlx_array_free(no_lhs); mlx_array_free(x2); mlx_array_free(xb); mlx_array_free(x_perm);
    mlx_array_free(graw); mlx_array_free(graw_r); mlx_array_free(gt); mlx_array_free(glbg);
    mlx_array_free(uraw); mlx_array_free(uraw_r); mlx_array_free(ut); mlx_array_free(ulbg);
    mlx_array_free(gate_out); mlx_array_free(up_out); mlx_array_free(act); mlx_array_free(actx);
    mlx_array_free(draw); mlx_array_free(draw_r); mlx_array_free(dt); mlx_array_free(dlbg);
    mlx_array_free(down_out); mlx_array_free(w_arr); mlx_array_free(scaled); mlx_array_free(summed);
    return ok;
}

static bool llm_cb_mxfp4_stacked(mlx_array wf2, int E, int out, int in, mlx_array* wq, mlx_array* sc, mlx_stream s) {
    mlx_vector_array qv = mlx_vector_array_new();
    mlx_optional_int g = {32, true}, b = {4, true};
    bool ok = mlx_quantize(&qv, wf2, g, b, "mxfp4", g_mlx_empty_arr, s) == 0 &&
              mlx_vector_array_size(qv) == 2;
    mlx_array w0 = mlx_array_new(), s0 = mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&w0, qv, 0) == 0 && mlx_vector_array_get(&s0, qv, 1) == 0;
    if (ok) { const int wsh[3] = {E, out, in / 8}; ok = mlx_reshape(wq, w0, wsh, 3, s) == 0; }
    if (ok) { const int ssh[3] = {E, out, in / 32}; ok = mlx_reshape(sc, s0, ssh, 3, s) == 0; }
    mlx_array_free(w0); mlx_array_free(s0); mlx_vector_array_free(qv);
    return ok;
}

/* ---- Ondata A: linear/norm/attivazioni/lora (raw = sequenza MLX inline equivalente) ---- */
static bool llm_cb_pl_raw(mlx_array* o, void* c) {
    llm_cb_pl_ctx* p = (llm_cb_pl_ctx*)c;
    mlx_dtype xd = mlx_array_dtype(p->x);
    mlx_array wc = mlx_array_new(), wt = mlx_array_new();
    bool ok = (mlx_array_dtype(p->w) == xd) ? (mlx_array_set(&wc, p->w) == 0)
                                            : (mlx_astype(&wc, p->w, xd, p->s) == 0);
    if (ok) ok = mlx_transpose(&wt, wc, p->s) == 0;
    if (ok) ok = mlx_matmul(o, p->x, wt, p->s) == 0;
    mlx_array_free(wc); mlx_array_free(wt);
    return ok;
}
static bool llm_cb_pl_llm(mlx_array* o, void* c) { llm_cb_pl_ctx* p = (llm_cb_pl_ctx*)c; return plain_linear(o, p->x, p->w, p->s); }
static bool llm_cb_q4pl_llm(mlx_array* o, void* c) { llm_cb_pl_ctx* p = (llm_cb_pl_ctx*)c; return llm_mlx_qwen4exp_plain_linear(o, p->x, p->w, p->s); }

static bool llm_cb_qm_raw(mlx_array* o, void* c) {
    llm_cb_qm_ctx* p = (llm_cb_qm_ctx*)c;
    mlx_optional_int g = {p->gs, true}, b = {p->bits, true};
    return mlx_quantized_matmul(o, p->x, p->xw, p->wsc, g_mlx_empty_arr, true, g, b, "mxfp4", p->s) == 0;
}
static bool llm_cb_qm_llm(mlx_array* o, void* c) {
    llm_cb_qm_ctx* p = (llm_cb_qm_ctx*)c;
    return quantized_linear_cfg(o, p->x, p->xw, p->wsc, g_mlx_empty_arr, p->gs, p->bits, "mxfp4", p->s);
}

static bool llm_cb_silu_raw(mlx_array* o, void* c) {
    llm_cb_u_ctx* p = (llm_cb_u_ctx*)c;
    mlx_array sg = mlx_array_new();
    bool ok = mlx_sigmoid(&sg, p->x, p->s) == 0 && mlx_multiply(o, p->x, sg, p->s) == 0;
    mlx_array_free(sg);
    return ok;
}
static bool llm_cb_silu_llm(mlx_array* o, void* c) { llm_cb_u_ctx* p = (llm_cb_u_ctx*)c; return llm_mlx_qwen4exp_silu(o, p->x, p->s); }

static bool llm_cb_zc_llm(mlx_array* o, void* c) {
    llm_cb_zc_ctx* p = (llm_cb_zc_ctx*)c;
    return llm_mlx_zero_centered_rmsnorm(o, p->x, p->w, p->gs, p->eps, p->s);
}
static bool llm_cb_zc_raw(mlx_array* o, void* c) {
    llm_cb_zc_ctx* p = (llm_cb_zc_ctx*)c;
    mlx_array xf = mlx_array_new(), sq = mlx_array_new(), mn = mlx_array_new(), eps = mlx_array_new_float32(p->eps);
    mlx_array mne = mlx_array_new(), rs = mlx_array_new(), nm = mlx_array_new(), wf = mlx_array_new();
    mlx_array one = mlx_array_new_float32(1.0f), wone = mlx_array_new(), scaled = mlx_array_new();
    bool ok = mlx_astype(&xf, p->x, MLX_FLOAT32, p->s) == 0 &&
              mlx_square(&sq, xf, p->s) == 0 &&
              mlx_mean_axis(&mn, sq, -1, true, p->s) == 0 &&
              mlx_add(&mne, mn, eps, p->s) == 0 &&
              mlx_rsqrt(&rs, mne, p->s) == 0 &&
              mlx_multiply(&nm, xf, rs, p->s) == 0 &&
              mlx_astype(&wf, p->w, MLX_FLOAT32, p->s) == 0 &&
              mlx_add(&wone, wf, one, p->s) == 0 &&
              mlx_multiply(&scaled, nm, wone, p->s) == 0 &&
              mlx_astype(o, scaled, mlx_array_dtype(p->x), p->s) == 0;
    mlx_array_free(xf); mlx_array_free(sq); mlx_array_free(mn); mlx_array_free(eps); mlx_array_free(mne);
    mlx_array_free(rs); mlx_array_free(nm); mlx_array_free(wf); mlx_array_free(one); mlx_array_free(wone);
    mlx_array_free(scaled);
    return ok;
}

static bool llm_cb_sg_llm(mlx_array* o, void* c) {
    llm_cb_sg_ctx* p = (llm_cb_sg_ctx*)c;
    return llm_mlx_sigmoid_rmsnorm_gated(o, p->h, p->w, p->gate, p->use_silu, p->eps, p->s);
}
static bool llm_cb_sg_raw(mlx_array* o, void* c) {
    llm_cb_sg_ctx* p = (llm_cb_sg_ctx*)c;
    mlx_array normed = mlx_array_new(), nf = mlx_array_new(), gf = mlx_array_new(), sig = mlx_array_new(), gated = mlx_array_new();
    bool ok = mlx_fast_rms_norm(&normed, p->h, p->w, p->eps, p->s) == 0 &&
              mlx_astype(&nf, normed, MLX_FLOAT32, p->s) == 0 &&
              mlx_astype(&gf, p->gate, MLX_FLOAT32, p->s) == 0;
    if (ok && p->use_silu) ok = llm_mlx_qwen4exp_silu(&sig, gf, p->s);
    else if (ok) ok = mlx_sigmoid(&sig, gf, p->s) == 0;
    ok = ok && mlx_multiply(&gated, nf, sig, p->s) == 0 &&
              mlx_astype(o, gated, mlx_array_dtype(p->h), p->s) == 0;
    mlx_array_free(normed); mlx_array_free(nf); mlx_array_free(gf); mlx_array_free(sig); mlx_array_free(gated);
    return ok;
}

static bool llm_cb_l2_llm(mlx_array* o, void* c) {
    llm_cb_l2_ctx* p = (llm_cb_l2_ctx*)c;
    mlx_array kk = mlx_array_new();
    bool ok = llm_mlx_qwen4exp_qk_l2norm(o, &kk, p->q, p->k, p->scale, p->s);
    mlx_array_free(kk);
    return ok;
}
static bool llm_cb_l2_raw(mlx_array* o, void* c) {
    llm_cb_l2_ctx* p = (llm_cb_l2_ctx*)c;
    mlx_array eps = mlx_array_new_float32(1e-6f);
    mlx_array sq = mlx_array_new(), sm = mlx_array_new(), sme = mlx_array_new(), rs = mlx_array_new(), qn = mlx_array_new(), sc = mlx_array_new_float32(p->scale);
    mlx_array ksq = mlx_array_new(), ksm = mlx_array_new(), ksme = mlx_array_new(), krs = mlx_array_new(), kn = mlx_array_new(), kk = mlx_array_new();
    bool ok = mlx_square(&sq, p->q, p->s) == 0 && mlx_sum_axis(&sm, sq, -1, true, p->s) == 0 &&
              mlx_add(&sme, sm, eps, p->s) == 0 && mlx_rsqrt(&rs, sme, p->s) == 0 &&
              mlx_multiply(&qn, p->q, rs, p->s) == 0 && mlx_multiply(o, qn, sc, p->s) == 0 &&
              mlx_square(&ksq, p->k, p->s) == 0 && mlx_sum_axis(&ksm, ksq, -1, true, p->s) == 0 &&
              mlx_add(&ksme, ksm, eps, p->s) == 0 && mlx_rsqrt(&krs, ksme, p->s) == 0 &&
              mlx_multiply(&kn, p->k, krs, p->s) == 0 && mlx_array_set(&kk, kn) == 0;
    mlx_array_free(eps); mlx_array_free(sq); mlx_array_free(sm); mlx_array_free(sme);
    mlx_array_free(rs); mlx_array_free(qn); mlx_array_free(sc);
    mlx_array_free(ksq); mlx_array_free(ksm); mlx_array_free(ksme); mlx_array_free(krs);
    mlx_array_free(kn); mlx_array_free(kk);
    return ok;
}

static bool llm_cb_lo_llm(mlx_array* o, void* c) {
    llm_cb_lo_ctx* p = (llm_cb_lo_ctx*)c;
    mlx_array r = mlx_array_new();
    bool ok = llm_mlx_lora_apply_mlx(&r, p->x, p->A, p->B, p->scale, p->s);
    if (ok) mlx_array_set(o, r);
    mlx_array_free(r);
    return ok;
}
static bool llm_cb_lo_raw(mlx_array* o, void* c) {
    llm_cb_lo_ctx* p = (llm_cb_lo_ctx*)c;
    mlx_array xa = mlx_array_new(), xab = mlx_array_new(), sc = mlx_array_new_float32(p->scale), scaled = mlx_array_new();
    bool ok = mlx_matmul(&xa, p->x, p->A, p->s) == 0 && mlx_matmul(&xab, xa, p->B, p->s) == 0 &&
              mlx_multiply(&scaled, xab, sc, p->s) == 0;
    if (ok) mlx_array_set(o, scaled);
    mlx_array_free(xa); mlx_array_free(xab); mlx_array_free(sc); mlx_array_free(scaled);
    return ok;
}

/* ---- Op `layer`: blocco denso completo (Llama-like) — GATE "mai indietro".
 * Due varianti IDENTICHE per struttura e pesi:
 *   - llm: wrapper del host (rms_norm_cfg, quantized_linear_cfg, llm_mlx_gqa_sdpa,
 *          llm_mlx_swiglu_metal);
 *   - raw: sequenza MLX minimale stile mlx_lm (mlx_fast_rms_norm, mlx_quantized_matmul,
 *          mlx_fast_rope, mlx_fast_scaled_dot_product_attention, silu eager).
 * Tempi ~ms (come il modello) -> il rapporto e' affidabile <1% (a differenza delle op
 * singole, 0.01-0.08 ms). Decode: un token, KV-cache [1,nkv,S,hd]. */

typedef bool (*llm_cb_qmm_fn)(mlx_array*, mlx_array, mlx_array, mlx_array, mlx_array, int, mlx_stream);
static bool llm_cb_raw_qmm(mlx_array* o, mlx_array x, mlx_array w, mlx_array sc, mlx_array bi, int gs, mlx_stream s) {
    mlx_optional_int g = {gs, true}, b = {8, true};
    return mlx_quantized_matmul(o, x, w, sc, bi, true, g, b, "affine", s) == 0;
}
static bool llm_cb_llm_qmm(mlx_array* o, mlx_array x, mlx_array w, mlx_array sc, mlx_array bi, int gs, mlx_stream s) {
    return quantized_linear_cfg(o, x, w, sc, bi, gs, 8, "affine", s);
}
static bool llm_cb_layer_body(llm_cb_layer_ctx* p, llm_cb_qmm_fn qmm, bool llm, bool metal_act, mlx_array* o) {
    mlx_stream s = p->s;
    mlx_array h1 = mlx_array_new(), q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    mlx_array qh = mlx_array_new(), kh = mlx_array_new(), vh = mlx_array_new();
    mlx_array qr = mlx_array_new(), kr = mlx_array_new(), attn = mlx_array_new(), af = mlx_array_new();
    mlx_array opr = mlx_array_new(), x2 = mlx_array_new(), h2 = mlx_array_new();
    mlx_array ga = mlx_array_new(), ua = mlx_array_new(), act = mlx_array_new(), dn = mlx_array_new();
    bool ok = true;
    const int qsh[4] = {1, 1, p->nh, p->hd}, kvsh[4] = {1, 1, p->nkv, p->hd};
    const int perm[4] = {0, 2, 1, 3};
    const int aflat[3] = {1, 1, p->nh * p->hd};
    /* norm + qkv */
    ok = ok && (llm ? rms_norm_cfg(&h1, p->x, p->ln1, p->eps, s) : mlx_fast_rms_norm(&h1, p->x, p->ln1, p->eps, s) == 0);
    ok = ok && qmm(&q, h1, p->qw, p->qs, p->qb, 64, s);
    ok = ok && qmm(&k, h1, p->kw, p->ks, p->kb, 64, s);
    ok = ok && qmm(&v, h1, p->vw, p->vs, p->vb, 64, s);
    ok = ok && mlx_reshape(&qh, q, qsh, 4, s) == 0 && mlx_transpose_axes(&qh, qh, perm, 4, s) == 0;
    ok = ok && mlx_reshape(&kh, k, kvsh, 4, s) == 0 && mlx_transpose_axes(&kh, kh, perm, 4, s) == 0;
    ok = ok && mlx_reshape(&vh, v, kvsh, 4, s) == 0 && mlx_transpose_axes(&vh, vh, perm, 4, s) == 0;
    /* rope: q sul token nuovo (offset), k sull'intera cache [1,nkv,S,hd] */
    { mlx_optional_float base = {p->rope_theta, true};
      ok = ok && mlx_fast_rope(&qr, qh, p->hd, false, base, 1.0f, p->rope_off, g_mlx_empty_arr, s) == 0;
      ok = ok && mlx_fast_rope(&kr, p->kcache, p->hd, false, base, 1.0f, 0, g_mlx_empty_arr, s) == 0; }
    /* attenzione GQA: q [1,nh,1,hd], k=kr [1,nkv,S,hd], v=cache [1,nkv,S,hd] */
    { float sc = 1.0f / sqrtf((float)p->hd);
      if (llm) ok = ok && llm_mlx_gqa_sdpa(&attn, qr, kr, p->vcache, sc, 0, s);
      else ok = ok && mlx_fast_scaled_dot_product_attention(&attn, qr, kr, p->vcache, sc, "",
                                                            g_mlx_empty_arr, g_mlx_empty_arr, s) == 0; }
    ok = ok && mlx_transpose_axes(&attn, attn, perm, 4, s) == 0 && mlx_reshape(&af, attn, aflat, 3, s) == 0;
    ok = ok && qmm(&opr, af, p->ow, p->os, p->ob, 64, s);
    ok = ok && mlx_add(&x2, p->x, opr, s) == 0;
    /* mlp */
    ok = ok && (llm ? rms_norm_cfg(&h2, x2, p->ln2, p->eps, s) : mlx_fast_rms_norm(&h2, x2, p->ln2, p->eps, s) == 0);
    ok = ok && qmm(&ga, h2, p->gw, p->gs, p->gb, 64, s);
    ok = ok && qmm(&ua, h2, p->uw, p->us, p->ub, 64, s);
    if (metal_act) { ok = ok && llm_mlx_swiglu_metal(ga, ua, &act, s); }
    else { mlx_array sg = mlx_array_new(); ok = ok && mlx_sigmoid(&sg, ga, s) == 0 && mlx_multiply(&sg, sg, ga, s) == 0 && mlx_multiply(&act, sg, ua, s) == 0; mlx_array_free(sg); }
    ok = ok && qmm(&dn, act, p->dw, p->ds, p->db, 64, s);
    if (ok) ok = mlx_add(o, x2, dn, s) == 0;
    if (getenv("LLM_CB_DEBUG")) fprintf(stderr, "[layer] llm=%d act=%d dtype(h1)=%d dtype(q)=%d dtype(gate)=%d dtype(act)=%d\n",
                                        (int)llm, (int)metal_act, (int)mlx_array_dtype(h1), (int)mlx_array_dtype(q),
                                        (int)mlx_array_dtype(ga), (int)mlx_array_dtype(act));
    mlx_array_free(h1); mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
    mlx_array_free(qh); mlx_array_free(kh); mlx_array_free(vh);
    mlx_array_free(qr); mlx_array_free(kr); mlx_array_free(attn); mlx_array_free(af);
    mlx_array_free(opr); mlx_array_free(x2); mlx_array_free(h2);
    mlx_array_free(ga); mlx_array_free(ua); mlx_array_free(act); mlx_array_free(dn);
    if (!ok && getenv("LLM_CB_DEBUG")) fprintf(stderr, "[layer] body failed (llm=%d)\n", (int)llm);
    return ok;
}
static bool llm_cb_layer_llm(mlx_array* o, void* c) { return llm_cb_layer_body((llm_cb_layer_ctx*)c, llm_cb_llm_qmm, true, true, o); }
static bool llm_cb_layer_raw(mlx_array* o, void* c) { return llm_cb_layer_body((llm_cb_layer_ctx*)c, llm_cb_raw_qmm, false, false, o); }
static bool llm_cb_layer_act(mlx_array* o, void* c) { return llm_cb_layer_body((llm_cb_layer_ctx*)c, llm_cb_raw_qmm, false, true, o); }
static bool llm_cb_layer_qmm(mlx_array* o, void* c) { return llm_cb_layer_body((llm_cb_layer_ctx*)c, llm_cb_llm_qmm, true, false, o); }

static bool llm_cb_q3(mlx_array* w, mlx_array* sc, mlx_array* bi, int out, int in, mlx_stream s) {
    mlx_array wf = llm_cb_rand_arr((int[]){out, in}, 2, MLX_FLOAT32, s);
    mlx_vector_array qv = mlx_vector_array_new();
    mlx_optional_int og = {64, true}, ob = {8, true};
    bool ok = mlx_quantize(&qv, wf, og, ob, "affine", g_mlx_empty_arr, s) == 0 && mlx_vector_array_size(qv) >= 3;
    if (ok) ok = mlx_vector_array_get(w, qv, 0) == 0 && mlx_vector_array_get(sc, qv, 1) == 0 && mlx_vector_array_get(bi, qv, 2) == 0;
    mlx_vector_array_free(qv);
    mlx_array_free(wf);
    return ok;
}

static void llm_cb_report(char* out, size_t cap, const char* name, const char* shape,
                          int iters, llm_cb_fn raw, llm_cb_fn llm, void* ctx, mlx_stream s) {
    mlx_array ra = mlx_array_new(), ga = mlx_array_new();
    /* min-di-3 round (best-case, meno conteso) per togliere il rumore da macchina condivisa. */
    double raw_ms = 1e18, llm_ms = 1e18;
    for (int r = 0; r < 3; r++) {
        double rm = llm_cb_run(raw, ctx, &ra, iters, s);
        double gm = llm_cb_run(llm, ctx, &ga, iters, s);
        if (rm < raw_ms) raw_ms = rm;
        if (gm < llm_ms) llm_ms = gm;
    }
    double raw_enc = llm_cb_run_enc(raw, ctx, &ra, iters, s);
    double llm_enc = llm_cb_run_enc(llm, ctx, &ga, iters, s);
    double md = llm_cb_maxdiff(ra, ga, s);
    if (getenv("LLM_CB_DEBUG")) {
        mlx_array fa = mlx_array_new(), fb = mlx_array_new();
        double sa = 0, sb = 0, ma = 0, mb = 0;
        if (mlx_astype(&fa, ra, MLX_FLOAT32, s) == 0 && mlx_astype(&fb, ga, MLX_FLOAT32, s) == 0 &&
            mlx_array_eval(fa) == 0 && mlx_array_eval(fb) == 0) {
            const float* da = mlx_array_data_float32(fa);
            const float* db = mlx_array_data_float32(fb);
            size_t n = mlx_array_size(fa);
            if (da && db && mlx_array_size(fb) == n) {
                for (size_t i = 0; i < n; i++) { sa += da[i]; sb += db[i];
                    if (fabs((double)da[i]) > ma) ma = fabs((double)da[i]);
                    if (fabs((double)db[i]) > mb) mb = fabs((double)db[i]); }
                fprintf(stderr, "[cb] %s raw sum=%.4g max=%.4g | llm sum=%.4g max=%.4g\n", name, sa, ma, sb, mb);
            }
        }
        mlx_array_free(fa); mlx_array_free(fb);
    }
    char line[640];
    snprintf(line, sizeof(line),
             "OP %-15s shape=%-26s raw_ms=%.4f llm_ms=%.4f ratio=%.3f enc_us=%.1f/%.1f maxdiff=%.3g\n",
             name, shape, raw_ms / iters, llm_ms / iters, raw_ms > 0 ? llm_ms / raw_ms : 0.0,
             raw_enc, llm_enc, md);
    llm_cb_emit(out, cap, line);
    mlx_array_free(ra); mlx_array_free(ga);
}

int llm_ai_mlx_core_bench(const char* op, int iters, char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    if (iters < 1) iters = 200;
    mlx_stream s = mlx_default_gpu_stream_new();
    int done = 0;
    const bool all = (op == NULL || strcmp(op, "all") == 0);

    if (all || strcmp(op, "rms_norm") == 0) {
        const int H = 2880;
        llm_cb_rms_ctx c; c.eps = 1e-5f; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, H}, 2, MLX_BFLOAT16, s);
        c.w = llm_cb_rand_arr((int[]){H}, 1, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "rms_norm", "[1,2880]bf16", iters, llm_cb_rms_raw, llm_cb_rms_llm, &c, s);
        mlx_array_free(c.x); mlx_array_free(c.w);
        done++;
    }

    if (all || strcmp(op, "qlinear") == 0) {
        const int H = 2880;
        const int gs = 64, bits = 8;
        llm_cb_ql_ctx c; c.gs = gs; c.bits = bits; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, H}, 2, MLX_BFLOAT16, s);
        mlx_array wf = llm_cb_rand_arr((int[]){H, H}, 2, MLX_FLOAT32, s);
        mlx_vector_array qv = mlx_vector_array_new();
        mlx_optional_int og = {gs, true}, ob = {bits, true};
        c.wq = mlx_array_new(); c.sc = mlx_array_new(); c.bi = mlx_array_new();
        if (mlx_quantize(&qv, wf, og, ob, "affine", g_mlx_empty_arr, s) == 0 &&
            mlx_vector_array_size(qv) >= 3 &&
            mlx_vector_array_get(&c.wq, qv, 0) == 0 &&
            mlx_vector_array_get(&c.sc, qv, 1) == 0 &&
            mlx_vector_array_get(&c.bi, qv, 2) == 0) {
            llm_cb_report(out, cap, "qlinear", "[1,2880]x[2880,2880]g64b8", iters,
                          llm_cb_ql_raw, llm_cb_ql_llm, &c, s);
            done++;
        } else {
            llm_cb_emit(out, cap, "OP qlinear FAILED (mlx_quantize)\n");
        }
        mlx_vector_array_free(qv);
        mlx_array_free(wf); mlx_array_free(c.x); mlx_array_free(c.wq); mlx_array_free(c.sc); mlx_array_free(c.bi);
    }

    if (all || strcmp(op, "sdpa") == 0 || strcmp(op, "sdpa_big") == 0) {
        const int nh = 32, nkv = 8, hd = 128;
        const int S = strcmp(op, "sdpa_big") == 0 ? 8192 : 2048;
        llm_cb_sd_ctx c; c.scale = 1.0f / sqrtf((float)hd); c.s = s;
        c.q = llm_cb_rand_arr((int[]){1, nh, 1, hd}, 4, MLX_BFLOAT16, s);
        c.k = llm_cb_rand_arr((int[]){1, nkv, S, hd}, 4, MLX_BFLOAT16, s);
        c.v = llm_cb_rand_arr((int[]){1, nkv, S, hd}, 4, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, S == 8192 ? "sdpa_big" : "sdpa",
                      S == 8192 ? "[1,32,1,128] GQA 8x S=8192" : "[1,32,1,128] GQA 8x S=2048",
                      iters, llm_cb_sd_raw, llm_cb_sd_llm, &c, s);
        mlx_array_free(c.q); mlx_array_free(c.k); mlx_array_free(c.v);
        done++;
    }

    if (all || strcmp(op, "swiglu") == 0) {
        llm_cb_sw_ctx c; c.s = s;
        c.gate = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.up   = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "swiglu", "[1,2880]bf16 metal vs eager", iters,
                      llm_cb_sw_eager, llm_cb_sw_metal, &c, s);
        mlx_array_free(c.gate); mlx_array_free(c.up);
        done++;
    }
    if (all || strcmp(op, "swiglu_prefill") == 0) {
        llm_cb_sw_ctx c; c.s = s;
        c.gate = llm_cb_rand_arr((int[]){2048, 8192}, 2, MLX_BFLOAT16, s);
        c.up   = llm_cb_rand_arr((int[]){2048, 8192}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "swiglu_prefill", "[2048,8192]bf16 metal vs eager", iters,
                      llm_cb_sw_eager, llm_cb_sw_metal, &c, s);
        mlx_array_free(c.gate); mlx_array_free(c.up);
        done++;
    }
    if (all || strcmp(op, "gptoss_swiglu") == 0) {
        llm_cb_sw_ctx c; c.s = s;
        c.gate = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.up   = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "gptoss_swiglu", "[1,2880]bf16 metal vs eager", iters,
                      llm_cb_gos_eager, llm_cb_gos_metal, &c, s);
        mlx_array_free(c.gate); mlx_array_free(c.up);
        done++;
    }
    if (all || strcmp(op, "geglu") == 0) {
        llm_cb_sw_ctx c; c.s = s;
        c.gate = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.up   = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "geglu", "[1,2880]bf16 metal vs eager", iters,
                      llm_cb_gg_eager, llm_cb_gg_metal, &c, s);
        mlx_array_free(c.gate); mlx_array_free(c.up);
        done++;
    }
    if (all || strcmp(op, "rms_norm_prefill") == 0) {
        const int H = 2880, T = 2048;
        llm_cb_rms_ctx c; c.eps = 1e-5f; c.s = s;
        c.x = llm_cb_rand_arr((int[]){T, H}, 2, MLX_BFLOAT16, s);
        c.w = llm_cb_rand_arr((int[]){H}, 1, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "rms_norm_prefill", "[2048,2880]bf16", iters,
                      llm_cb_rms_raw, llm_cb_rms_llm, &c, s);
        mlx_array_free(c.x); mlx_array_free(c.w);
        done++;
    }
    if (all || strcmp(op, "qlinear_prefill") == 0) {
        const int H = 2880, T = 512, gs = 64, bits = 8;
        llm_cb_ql_ctx c; c.gs = gs; c.bits = bits; c.s = s;
        c.x = llm_cb_rand_arr((int[]){T, H}, 2, MLX_BFLOAT16, s);
        mlx_array wf = llm_cb_rand_arr((int[]){H, H}, 2, MLX_FLOAT32, s);
        mlx_vector_array qv = mlx_vector_array_new();
        mlx_optional_int og = {gs, true}, ob = {bits, true};
        c.wq = mlx_array_new(); c.sc = mlx_array_new(); c.bi = mlx_array_new();
        if (mlx_quantize(&qv, wf, og, ob, "affine", g_mlx_empty_arr, s) == 0 &&
            mlx_vector_array_size(qv) >= 3 &&
            mlx_vector_array_get(&c.wq, qv, 0) == 0 &&
            mlx_vector_array_get(&c.sc, qv, 1) == 0 &&
            mlx_vector_array_get(&c.bi, qv, 2) == 0) {
            llm_cb_report(out, cap, "qlinear_prefill", "[512,2880]x[2880,2880]", iters,
                          llm_cb_ql_raw, llm_cb_ql_llm, &c, s);
            done++;
        }
        mlx_vector_array_free(qv);
        mlx_array_free(wf); mlx_array_free(c.x); mlx_array_free(c.wq); mlx_array_free(c.sc); mlx_array_free(c.bi);
    }

    if (all || strcmp(op, "logit_softcap") == 0) {
        llm_cb_ls_ctx c; c.cap = 30.0f; c.s = s;
        c.logits = llm_cb_rand_arr((int[]){1, 32768}, 2, MLX_FLOAT32, s);
        llm_cb_report(out, cap, "logit_softcap", "[1,32768]f32 cap=30", iters,
                      llm_cb_ls_raw, llm_cb_ls_llm, &c, s);
        mlx_array_free(c.logits);
        done++;
    }
    if (all || strcmp(op, "embed") == 0) {
        const int V = 8192, H = 2880, gs = 64, bits = 8;
        llm_cb_emb_ctx c; c.hidden = H; c.gs = gs; c.bits = bits; c.mode = "affine"; c.s = s;
        mlx_array wf = llm_cb_rand_arr((int[]){V, H}, 2, MLX_FLOAT32, s);
        mlx_vector_array qv = mlx_vector_array_new();
        mlx_optional_int og = {gs, true}, ob = {bits, true};
        c.emb_w = mlx_array_new(); c.emb_s = mlx_array_new(); c.emb_b = mlx_array_new();
        int32_t tidv = 1234; const int tsh[1] = {1};
        c.tid = mlx_array_new_data(&tidv, tsh, 1, MLX_UINT32);
        if (mlx_quantize(&qv, wf, og, ob, "affine", g_mlx_empty_arr, s) == 0 &&
            mlx_vector_array_size(qv) >= 3 &&
            mlx_vector_array_get(&c.emb_w, qv, 0) == 0 &&
            mlx_vector_array_get(&c.emb_s, qv, 1) == 0 &&
            mlx_vector_array_get(&c.emb_b, qv, 2) == 0) {
            llm_cb_report(out, cap, "embed", "[V=8192,2880]g64b8 row-take", iters,
                          llm_cb_emb_raw, llm_cb_emb_llm, &c, s);
            done++;
        }
        mlx_vector_array_free(qv);
        mlx_array_free(wf); mlx_array_free(c.emb_w); mlx_array_free(c.emb_s); mlx_array_free(c.emb_b); mlx_array_free(c.tid);
    }

    if (all || strcmp(op, "kv_append") == 0) {
        const int nkv = 8, hd = 128, cap = 4096, off = 2048;
        llm_cb_kv_ctx c; c.off = off; c.s = s;
        c.kc = llm_cb_rand_arr((int[]){1, nkv, cap, hd}, 4, MLX_BFLOAT16, s);
        c.vc = llm_cb_rand_arr((int[]){1, nkv, cap, hd}, 4, MLX_BFLOAT16, s);
        c.kn = llm_cb_rand_arr((int[]){1, nkv, 1, hd}, 4, MLX_BFLOAT16, s);
        c.vn = llm_cb_rand_arr((int[]){1, nkv, 1, hd}, 4, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "kv_append", "inplace vs concat off=2048", iters,
                      llm_cb_kv_raw, llm_cb_kv_llm, &c, s);
        mlx_array_free(c.kc); mlx_array_free(c.vc); mlx_array_free(c.kn); mlx_array_free(c.vn);
        done++;
    }

    if (all || strcmp(op, "kvq_sdpa") == 0) {
        int bits = 4, vbits = 4;
        const int nh = 32, nkv = 8, hd = 128, NC = 2048, NR = 128, gs = 64;
        { const char* kbv = llm_ai_mlx_opt_env("LLM_CB_KVBITS"); if (kbv) { int a=4,b2=4; if (sscanf(kbv,"%d:%d",&a,&b2)==2) { bits=a; vbits=b2; } } }
        llm_cb_kvq_ctx c; c.scale = 1.0f / sqrtf((float)hd); c.bits = bits; c.vbits = vbits; c.gs = gs; c.s = s;
        c.q = llm_cb_rand_arr((int[]){nh, hd}, 2, MLX_FLOAT32, s);
        c.Kr = llm_cb_rand_arr((int[]){nkv, NR, hd}, 3, MLX_FLOAT32, s);
        c.Vr = llm_cb_rand_arr((int[]){nkv, NR, hd}, 3, MLX_FLOAT32, s);
        mlx_array coldk = llm_cb_rand_arr((int[]){nkv, NC, hd}, 3, MLX_FLOAT32, s);
        mlx_array coldv = llm_cb_rand_arr((int[]){nkv, NC, hd}, 3, MLX_FLOAT32, s);
        mlx_vector_array qk = mlx_vector_array_new(), qv2 = mlx_vector_array_new();
        mlx_optional_int og = {gs, true}, ob = {bits, true}, obv = {vbits, true};
        c.Kc_w = mlx_array_new(); c.Kc_s = mlx_array_new(); c.Kc_b = mlx_array_new();
        c.Vc_w = mlx_array_new(); c.Vc_s = mlx_array_new(); c.Vc_b = mlx_array_new();
        bool ok = mlx_quantize(&qk, coldk, og, ob, "affine", g_mlx_empty_arr, s) == 0 &&
                  mlx_vector_array_size(qk) >= 3 &&
                  mlx_vector_array_get(&c.Kc_w, qk, 0) == 0 &&
                  mlx_vector_array_get(&c.Kc_s, qk, 1) == 0 &&
                  mlx_vector_array_get(&c.Kc_b, qk, 2) == 0 &&
                  mlx_quantize(&qv2, coldv, og, obv, "affine", g_mlx_empty_arr, s) == 0 &&
                  mlx_vector_array_size(qv2) >= 3 &&
                  mlx_vector_array_get(&c.Vc_w, qv2, 0) == 0 &&
                  mlx_vector_array_get(&c.Vc_s, qv2, 1) == 0 &&
                  mlx_vector_array_get(&c.Vc_b, qv2, 2) == 0;
        if (ok && llm_ai_mlx_opt_env("LLM_KVQ_MAN")) {
            mlx_dtype T = mlx_array_dtype(c.q);
            mlx_array kd=mlx_array_new(), vd=mlx_array_new(), krf=mlx_array_new(), vrf=mlx_array_new(), qf=mlx_array_new();
            mlx_array kcat=mlx_array_new(), vcat=mlx_array_new();
            mlx_array krep=mlx_array_new(), vrep=mlx_array_new(), kT=mlx_array_new();
            mlx_array q4=mlx_array_new(), v4=mlx_array_new(), sc=mlx_array_new();
            mlx_array scl=mlx_array_new(), sm=mlx_array_new(), om=mlx_array_new(), man=mlx_array_new(), ko=mlx_array_new();
            bool mok = llm_mlx_kv_dequantize(&kd, c.Kc_w, c.Kc_s, c.Kc_b, bits, gs, s) &&
                       llm_mlx_kv_dequantize(&vd, c.Vc_w, c.Vc_s, c.Vc_b, bits, gs, s);
            if (mok) mok = (T==MLX_FLOAT32) ? (mlx_array_set(&krf,c.Kr)==0 && mlx_array_set(&vrf,c.Vr)==0 && mlx_array_set(&qf,c.q)==0)
                                            : (mlx_astype(&krf,c.Kr,MLX_FLOAT32,s)==0 && mlx_astype(&vrf,c.Vr,MLX_FLOAT32,s)==0 && mlx_astype(&qf,c.q,MLX_FLOAT32,s)==0);
            int groups = nh / nkv;
            if (mok) { mlx_vector_array a=mlx_vector_array_new(), b=mlx_vector_array_new();
                mlx_vector_array_append_value(a,kd); mlx_vector_array_append_value(a,krf);
                mlx_vector_array_append_value(b,vd); mlx_vector_array_append_value(b,vrf);
                mok = mlx_concatenate_axis(&kcat,a,1,s)==0 && mlx_concatenate_axis(&vcat,b,1,s)==0;
                mlx_vector_array_free(a); mlx_vector_array_free(b); }
            int S = mok ? mlx_array_shape(kcat)[1] : 0;
            if (mok) { int k4s[4]={1,nkv,S,hd}; mlx_array k4=mlx_array_new(), vv4=mlx_array_new();
                mok = mlx_reshape(&k4,kcat,k4s,4,s)==0 && mlx_reshape(&vv4,vcat,k4s,4,s)==0;
                if (mok) mok = mlx_repeat_axis(&krep,k4,groups,1,s)==0 && mlx_repeat_axis(&vrep,vv4,groups,1,s)==0;
                mlx_array_free(k4); mlx_array_free(vv4); }
            if (mok) { int p4[4]={0,1,3,2}; mok = mlx_transpose_axes(&kT,krep,p4,4,s)==0; }
            if (mok) { int qsh[4]={1,nh,1,hd}, vsh[4]={1,nh,S,hd};
                mok = mlx_reshape(&q4,qf,qsh,4,s)==0 && mlx_reshape(&v4,vrep,vsh,4,s)==0; }
            if (mok) mok = mlx_matmul(&sc,q4,kT,s)==0;
            if (mok) { mlx_array sca=mlx_array_new_float(c.scale); mok = mlx_multiply(&scl,sc,sca,s)==0; mlx_array_free(sca); }
            if (mok) mok = mlx_softmax_axis(&sm,scl,-1,true,s)==0;
            if (mok) mok = mlx_matmul(&om,sm,v4,s)==0;
            if (mok) { int o2[2]={nh,hd}; mok = mlx_reshape(&man,om,o2,2,s)==0; }
            if (mok) mok = llm_mlx_kvq_sdpa_decode(&ko, c.q, c.Kc_w, c.Kc_s, c.Kc_b, c.Vc_w, c.Vc_s, c.Vc_b, c.Kr, c.Vr, 0, 4, 4, c.scale, s);
            double dk=-1, dr=-1;
            if (mok) { mlx_array mf=mlx_array_new(), kf=mlx_array_new();
                if (mlx_astype(&mf,man,MLX_FLOAT32,s)==0 && mlx_astype(&kf,ko,MLX_FLOAT32,s)==0 && mlx_array_eval(mf)==0 && mlx_array_eval(kf)==0) {
                    const float* a=mlx_array_data_float32(mf); const float* b=mlx_array_data_float32(kf);
                    if (a&&b&&mlx_array_size(mf)==mlx_array_size(kf)) { dk=0; for (size_t i=0;i<mlx_array_size(mf);i++){double d=fabs((double)a[i]-(double)b[i]); if(d>dk)dk=d;} }
                    dr=0; for (size_t i=0;i<mlx_array_size(mf);i++){double d=fabs((double)a[i]); if(d>dr)dr=d;}
                }
                mlx_array_free(mf); mlx_array_free(kf); }
            double drk=-1;
            if (mok) { mlx_array rr=mlx_array_new();
                if (llm_cb_kvq_raw(&rr,&c)) { mlx_array mf=mlx_array_new(), rf=mlx_array_new(), kf2=mlx_array_new();
                    if (mlx_astype(&mf,man,MLX_FLOAT32,s)==0 && mlx_astype(&rf,rr,MLX_FLOAT32,s)==0 && mlx_astype(&kf2,ko,MLX_FLOAT32,s)==0 &&
                        mlx_array_eval(mf)==0 && mlx_array_eval(rf)==0 && mlx_array_eval(kf2)==0) {
                        const float* a=mlx_array_data_float32(mf); const float* b=mlx_array_data_float32(rf); const float* cc=mlx_array_data_float32(kf2);
                        size_t n=mlx_array_size(mf);
                        if (a&&b&&cc&&mlx_array_size(rf)==n&&mlx_array_size(kf2)==n) {
                            drk=0; double dmr=0;
                            for (size_t i=0;i<n;i++){double d1=fabs((double)a[i]-(double)cc[i]); if(d1>drk)drk=d1;
                                                    double d2=fabs((double)a[i]-(double)b[i]); if(d2>dmr)dmr=d2;}
                            fprintf(stderr, "[man] maxdiff(kernel,raw)=%.3g maxdiff(man,raw)=%.3g\n", drk, dmr);
                        }
                    }
                    mlx_array_free(mf); mlx_array_free(rf); mlx_array_free(kf2);
                } else fprintf(stderr, "[man] raw failed\n");
                mlx_array_free(rr); }
            if (mok) { mlx_array mf=mlx_array_new(), kf=mlx_array_new();
                if (mlx_astype(&mf,man,MLX_FLOAT32,s)==0 && mlx_astype(&kf,ko,MLX_FLOAT32,s)==0 && mlx_array_eval(mf)==0 && mlx_array_eval(kf)==0) {
                    const float* a=mlx_array_data_float32(mf); const float* b=mlx_array_data_float32(kf);
                    fprintf(stderr, "[man] h0: man[0..3]=%.5g,%.5g,%.5g,%.5g  kern[0..3]=%.5g,%.5g,%.5g,%.5g\n",
                            a?a[0]:0,a?a[1]:0,a?a[2]:0,a?a[3]:0, b?b[0]:0,b?b[1]:0,b?b[2]:0,b?b[3]:0);
                    if (a && b) { size_t nn=mlx_array_size(mf), bi=0; double bd=0;
                        for (size_t i=0;i<nn;i++){double d=fabs((double)a[i]-(double)b[i]); if(d>bd){bd=d;bi=i;}}
                        size_t nb=mlx_array_size(kf);
                        fprintf(stderr, "[man] worst idx=%zu (h=%zu d=%zu) man=%.6g kern=%.6g diff=%.4g | n=%zu/%zu\n",
                                bi, bi%hd, bi%hd, bi<nb?a[bi]:0, bi<nb?b[bi]:0, bd, nn, nb);
                    }
                }
                mlx_array_free(mf); mlx_array_free(kf); }
            fprintf(stderr, "[man] NC=%d NR=%d hd=%d S=%d groups=%d max|man|=%.4g maxdiff(man,kernel)=%.3g ok=%d\n", NC, NR, hd, S, groups, dr, dk, (int)mok);
            mlx_array_free(kd); mlx_array_free(vd); mlx_array_free(krf); mlx_array_free(vrf); mlx_array_free(qf);
            mlx_array_free(kcat); mlx_array_free(vcat); mlx_array_free(krep); mlx_array_free(vrep); mlx_array_free(kT);
            mlx_array_free(q4); mlx_array_free(v4); mlx_array_free(sc); mlx_array_free(scl); mlx_array_free(sm);
            mlx_array_free(om); mlx_array_free(man); mlx_array_free(ko);
        }
        if (ok) {
            llm_cb_report(out, cap, "kvq_sdpa", "nh32 nkv8 hd128 NC2048 NR128 b4g64", iters,
                          llm_cb_kvq_raw, llm_cb_kvq_llm, &c, s);
            done++;
        } else {
            llm_cb_emit(out, cap, "OP kvq_sdpa FAILED (quantize)\n");
        }
        mlx_vector_array_free(qk); mlx_vector_array_free(qv2);
        mlx_array_free(coldk); mlx_array_free(coldv);
        mlx_array_free(c.q); mlx_array_free(c.Kr); mlx_array_free(c.Vr);
        mlx_array_free(c.Kc_w); mlx_array_free(c.Kc_s); mlx_array_free(c.Kc_b);
        mlx_array_free(c.Vc_w); mlx_array_free(c.Vc_s); mlx_array_free(c.Vc_b);
    }

    if (all || strcmp(op, "kvq_hetero") == 0) {
        const int nh = 32, nkv = 8, hd = 128, NC = 2048, NR = 128, kbits = 8, vbits = 4, gs = 64;
        llm_cb_kvq_ctx c; c.scale = 1.0f / sqrtf((float)hd); c.bits = kbits; c.vbits = vbits; c.gs = gs; c.s = s;
        c.q = llm_cb_rand_arr((int[]){nh, hd}, 2, MLX_FLOAT32, s);
        c.Kr = llm_cb_rand_arr((int[]){nkv, NR, hd}, 3, MLX_FLOAT32, s);
        c.Vr = llm_cb_rand_arr((int[]){nkv, NR, hd}, 3, MLX_FLOAT32, s);
        mlx_array coldk = llm_cb_rand_arr((int[]){nkv, NC, hd}, 3, MLX_FLOAT32, s);
        mlx_array coldv = llm_cb_rand_arr((int[]){nkv, NC, hd}, 3, MLX_FLOAT32, s);
        mlx_vector_array qk = mlx_vector_array_new(), qv2 = mlx_vector_array_new();
        mlx_optional_int ogk = {gs, true}, obk = {kbits, true}, ogv = {gs, true}, obv = {vbits, true};
        c.Kc_w = mlx_array_new(); c.Kc_s = mlx_array_new(); c.Kc_b = mlx_array_new();
        c.Vc_w = mlx_array_new(); c.Vc_s = mlx_array_new(); c.Vc_b = mlx_array_new();
        bool ok = mlx_quantize(&qk, coldk, ogk, obk, "affine", g_mlx_empty_arr, s) == 0 &&
                  mlx_vector_array_size(qk) >= 3 &&
                  mlx_vector_array_get(&c.Kc_w, qk, 0) == 0 &&
                  mlx_vector_array_get(&c.Kc_s, qk, 1) == 0 &&
                  mlx_vector_array_get(&c.Kc_b, qk, 2) == 0 &&
                  mlx_quantize(&qv2, coldv, ogv, obv, "affine", g_mlx_empty_arr, s) == 0 &&
                  mlx_vector_array_size(qv2) >= 3 &&
                  mlx_vector_array_get(&c.Vc_w, qv2, 0) == 0 &&
                  mlx_vector_array_get(&c.Vc_s, qv2, 1) == 0 &&
                  mlx_vector_array_get(&c.Vc_b, qv2, 2) == 0;
        if (ok) {
            llm_cb_report(out, cap, "kvq_hetero", "K8V4 nh32 hd128 NC2048 NR128", iters,
                          llm_cb_kvq_raw, llm_cb_kvq_llm, &c, s);
            done++;
        } else {
            llm_cb_emit(out, cap, "OP kvq_hetero FAILED (quantize)\n");
        }
        mlx_vector_array_free(qk); mlx_vector_array_free(qv2);
        mlx_array_free(coldk); mlx_array_free(coldv);
        mlx_array_free(c.q); mlx_array_free(c.Kr); mlx_array_free(c.Vr);
        mlx_array_free(c.Kc_w); mlx_array_free(c.Kc_s); mlx_array_free(c.Kc_b);
        mlx_array_free(c.Vc_w); mlx_array_free(c.Vc_s); mlx_array_free(c.Vc_b);
    }

    if (all || strcmp(op, "moe_mxfp4") == 0) {
        const int E = 32, din = 2880, dff = 2880, k = 4;
        llm_cb_moe_ctx c; c.din = din; c.dff = dff; c.k = k; c.s = s;
        c.gw = mlx_array_new(); c.gsc = mlx_array_new();
        c.uw = mlx_array_new(); c.usc = mlx_array_new();
        c.dw = mlx_array_new(); c.dsc = mlx_array_new();
        mlx_array gwf = llm_cb_rand_arr((int[]){E * dff, din}, 2, MLX_FLOAT32, s);
        mlx_array uwf = llm_cb_rand_arr((int[]){E * dff, din}, 2, MLX_FLOAT32, s);
        mlx_array dwf = llm_cb_rand_arr((int[]){E * din, dff}, 2, MLX_FLOAT32, s);
        bool ok = llm_cb_mxfp4_stacked(gwf, E, dff, din, &c.gw, &c.gsc, s) &&
                  llm_cb_mxfp4_stacked(uwf, E, dff, din, &c.uw, &c.usc, s) &&
                  llm_cb_mxfp4_stacked(dwf, E, din, dff, &c.dw, &c.dsc, s);
        c.x = llm_cb_rand_arr((int[]){din}, 1, MLX_BFLOAT16, s);
        c.glb = llm_cb_rand_arr((int[]){E, dff}, 2, MLX_BFLOAT16, s);
        c.ulb = llm_cb_rand_arr((int[]){E, dff}, 2, MLX_BFLOAT16, s);
        c.dlb = llm_cb_rand_arr((int[]){E, din}, 2, MLX_BFLOAT16, s);
        c.sc = llm_cb_rand_arr((int[]){k}, 1, MLX_BFLOAT16, s);
        int32_t idxv[4] = {3, 11, 20, 29}; const int ish[1] = {k};
        c.idx = mlx_array_new_data(idxv, ish, 1, MLX_UINT32);
        if (ok) {
            llm_cb_report(out, cap, "moe_mxfp4", "E32 K4 din/dff2880", iters,
                          llm_cb_moe_eager, llm_cb_moe_fused, &c, s);
            done++;
        } else {
            llm_cb_emit(out, cap, "OP moe_mxfp4 FAILED (quantize/rescale)\n");
        }
        mlx_array_free(gwf); mlx_array_free(uwf); mlx_array_free(dwf);
        mlx_array_free(c.x); mlx_array_free(c.idx); mlx_array_free(c.glb); mlx_array_free(c.ulb); mlx_array_free(c.dlb); mlx_array_free(c.sc);
        mlx_array_free(c.gw); mlx_array_free(c.gsc); mlx_array_free(c.uw); mlx_array_free(c.usc); mlx_array_free(c.dw); mlx_array_free(c.dsc);
    }

    if (all || strcmp(op, "plain_linear") == 0 || strcmp(op, "q4_plain_linear") == 0) {
        llm_cb_pl_ctx c; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.w = llm_cb_rand_arr((int[]){2880, 2880}, 2, MLX_BFLOAT16, s);
        bool q4 = strcmp(op, "q4_plain_linear") == 0;
        llm_cb_report(out, cap, q4 ? "q4_plain_linear" : "plain_linear", "[1,2880]x[2880,2880]bf16", iters,
                      llm_cb_pl_raw, q4 ? llm_cb_q4pl_llm : llm_cb_pl_llm, &c, s);
        mlx_array_free(c.x); mlx_array_free(c.w);
        done++;
    }
    if (all || strcmp(op, "qlinear_mxfp4") == 0) {
        llm_cb_qm_ctx c; c.gs = 32; c.bits = 4; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        mlx_array wf = llm_cb_rand_arr((int[]){2880, 2880}, 2, MLX_FLOAT32, s);
        mlx_vector_array qv = mlx_vector_array_new();
        mlx_optional_int og = {32, true}, ob = {4, true};
        c.xw = mlx_array_new(); c.wsc = mlx_array_new();
        if (mlx_quantize(&qv, wf, og, ob, "mxfp4", g_mlx_empty_arr, s) == 0 &&
            mlx_vector_array_size(qv) >= 2 &&
            mlx_vector_array_get(&c.xw, qv, 0) == 0 &&
            mlx_vector_array_get(&c.wsc, qv, 1) == 0) {
            llm_cb_report(out, cap, "qlinear_mxfp4", "[1,2880]x[2880,2880]g32b4", iters,
                          llm_cb_qm_raw, llm_cb_qm_llm, &c, s);
            done++;
        }
        mlx_vector_array_free(qv);
        mlx_array_free(wf); mlx_array_free(c.x); mlx_array_free(c.xw); mlx_array_free(c.wsc);
    }
    if (all || strcmp(op, "silu") == 0) {
        llm_cb_u_ctx c; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "silu", "[1,2880]bf16", iters, llm_cb_silu_raw, llm_cb_silu_llm, &c, s);
        mlx_array_free(c.x);
        done++;
    }
    if (all || strcmp(op, "zero_centered_rmsnorm") == 0) {
        llm_cb_zc_ctx c; c.gs = 0; c.eps = 1e-6f; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.w = llm_cb_rand_arr((int[]){2880}, 1, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "zero_centered_rmsnorm", "[1,2880]bf16 gs=0", iters,
                      llm_cb_zc_raw, llm_cb_zc_llm, &c, s);
        mlx_array_free(c.x); mlx_array_free(c.w);
        done++;
    }
    if (all || strcmp(op, "sigmoid_rmsnorm_gated") == 0) {
        llm_cb_sg_ctx c; c.eps = 1e-6f; c.s = s;
        c.h = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.w = llm_cb_rand_arr((int[]){2880}, 1, MLX_BFLOAT16, s);
        c.gate = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "sigmoid_rmsnorm_gated", "[1,2880]bf16", iters,
                      llm_cb_sg_raw, llm_cb_sg_llm, &c, s);
        mlx_array_free(c.h); mlx_array_free(c.w); mlx_array_free(c.gate);
        done++;
    }
    if (all || strcmp(op, "qk_l2norm") == 0) {
        llm_cb_l2_ctx c; c.scale = 1.0f / sqrtf(128.0f); c.s = s;
        c.q = llm_cb_rand_arr((int[]){1, 32, 1, 128}, 4, MLX_BFLOAT16, s);
        c.k = llm_cb_rand_arr((int[]){1, 32, 1, 128}, 4, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "qk_l2norm", "[1,32,1,128]bf16", iters, llm_cb_l2_raw, llm_cb_l2_llm, &c, s);
        mlx_array_free(c.q); mlx_array_free(c.k);
        done++;
    }
    if (all || strcmp(op, "lora_apply") == 0) {
        llm_cb_lo_ctx c; c.scale = 0.5f; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, 2880}, 2, MLX_BFLOAT16, s);
        c.A = llm_cb_rand_arr((int[]){2880, 16}, 2, MLX_BFLOAT16, s);
        c.B = llm_cb_rand_arr((int[]){16, 2880}, 2, MLX_BFLOAT16, s);
        llm_cb_report(out, cap, "lora_apply", "[1,2880]xA[2880,16]xB[16,2880]", iters,
                      llm_cb_lo_raw, llm_cb_lo_llm, &c, s);
        mlx_array_free(c.x); mlx_array_free(c.A); mlx_array_free(c.B);
        done++;
    }

    if (all || strcmp(op, "layer") == 0) {
        const int H = 2048, nh = 32, nkv = 8, hd = 64, dff = 8192, S = 256;
        llm_cb_layer_ctx c; memset(&c, 0, sizeof(c));
        c.H = H; c.nh = nh; c.nkv = nkv; c.hd = hd; c.dff = dff; c.S = S; c.rope_off = S;
        c.eps = 1e-5f; c.rope_theta = 500000.0f; c.s = s;
        c.x = llm_cb_rand_arr((int[]){1, H}, 2, MLX_BFLOAT16, s);
        c.ln1 = llm_cb_rand_arr((int[]){H}, 1, MLX_BFLOAT16, s);
        c.ln2 = llm_cb_rand_arr((int[]){H}, 1, MLX_BFLOAT16, s);
        c.qw = mlx_array_new(); c.qs = mlx_array_new(); c.qb = mlx_array_new();
        c.kw = mlx_array_new(); c.ks = mlx_array_new(); c.kb = mlx_array_new();
        c.vw = mlx_array_new(); c.vs = mlx_array_new(); c.vb = mlx_array_new();
        c.ow = mlx_array_new(); c.os = mlx_array_new(); c.ob = mlx_array_new();
        c.gw = mlx_array_new(); c.gs = mlx_array_new(); c.gb = mlx_array_new();
        c.uw = mlx_array_new(); c.us = mlx_array_new(); c.ub = mlx_array_new();
        c.dw = mlx_array_new(); c.ds = mlx_array_new(); c.db = mlx_array_new();
        bool cok = llm_cb_q3(&c.qw, &c.qs, &c.qb, nh * hd, H, s) &&
                   llm_cb_q3(&c.kw, &c.ks, &c.kb, nkv * hd, H, s) &&
                   llm_cb_q3(&c.vw, &c.vs, &c.vb, nkv * hd, H, s) &&
                   llm_cb_q3(&c.ow, &c.os, &c.ob, H, nh * hd, s) &&
                   llm_cb_q3(&c.gw, &c.gs, &c.gb, dff, H, s) &&
                   llm_cb_q3(&c.uw, &c.us, &c.ub, dff, H, s) &&
                   llm_cb_q3(&c.dw, &c.ds, &c.db, H, dff, s);
        c.kcache = llm_cb_rand_arr((int[]){1, nkv, S, hd}, 4, MLX_BFLOAT16, s);
        c.vcache = llm_cb_rand_arr((int[]){1, nkv, S, hd}, 4, MLX_BFLOAT16, s);
        if (cok) {
            bool sb = g_bf16_all, sa = g_act16;
            g_bf16_all = true; g_act16 = MLX_BFLOAT16;
            llm_cb_report(out, cap, "layer", "Llama-like H2048 dff8192 S256", iters,
                          llm_cb_layer_raw, llm_cb_layer_llm, &c, s);
            llm_cb_report(out, cap, "layer_act", "raw qmm + metal act", iters,
                          llm_cb_layer_raw, llm_cb_layer_act, &c, s);
            llm_cb_report(out, cap, "layer_qmm", "llm qmm + eager act", iters,
                          llm_cb_layer_raw, llm_cb_layer_qmm, &c, s);
            g_bf16_all = sb; g_act16 = sa;
            done += 3;
        } else {
            llm_cb_emit(out, cap, "OP layer FAILED (quantize)\n");
        }
        mlx_array_free(c.x); mlx_array_free(c.ln1); mlx_array_free(c.ln2);
        mlx_array_free(c.qw); mlx_array_free(c.qs); mlx_array_free(c.qb);
        mlx_array_free(c.kw); mlx_array_free(c.ks); mlx_array_free(c.kb);
        mlx_array_free(c.vw); mlx_array_free(c.vs); mlx_array_free(c.vb);
        mlx_array_free(c.ow); mlx_array_free(c.os); mlx_array_free(c.ob);
        mlx_array_free(c.gw); mlx_array_free(c.gs); mlx_array_free(c.gb);
        mlx_array_free(c.uw); mlx_array_free(c.us); mlx_array_free(c.ub);
        mlx_array_free(c.dw); mlx_array_free(c.ds); mlx_array_free(c.db);
        mlx_array_free(c.kcache); mlx_array_free(c.vcache);
    }

    (void)done;
    return done;
}


/* Self-test del port Splash (store/quant in scrittura, splash_store_kv_row).
 * Ritorna 0 se kq/vq/ks/  corrispondono al riferimento C; stampa il max diff. */
int llm_ai_mlx_splash_store_selftest(char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    const int nkv = 4, n = 8, hd = 64;
    const int N = nkv * n * hd;
    float* kh = (float*)malloc((size_t)N * sizeof(float));
    float* vh = (float*)malloc((size_t)N * sizeof(float));
    int8_t* kref = (int8_t*)malloc((size_t)N);
    float* ksref = (float*)malloc((size_t)(nkv * n) * sizeof(float));
    for (int i = 0; i < N; i++) { kh[i] = (float)((i * 37 % 200) - 100) * 0.031f; vh[i] = (float)((i * 53 % 160) - 80) * 0.027f; }
    /* riferimento (fedele alla matematica di Splash) */
    for (int h = 0; h < nkv; h++) for (int t = 0; t < n; t++) {
        const size_t off = ((size_t)h * n + t) * hd;
        float mx = 0.0f;
        for (int d = 0; d < hd; d++) { float a = fabsf(kh[off + d]), b = fabsf(vh[off + d]); if (a > mx) mx = a; if (b > mx) mx = b; }
        ksref[h * n + t] = mx == 0.0f ? 0.0f : mx / 127.0f;
        for (int d = 0; d < hd; d++) {
            long q = (mx == 0.0f) ? 0L : lrintf((kh[off + d] * 127.0f) / mx);
            if (q > 127) q = 127; if (q < -127) q = -127;
            kref[off + d] = (int8_t)q;
        }
    }
    mlx_stream st = mlx_default_gpu_stream_new();
    int shp[3] = {nkv, n, hd};
    mlx_array k = mlx_array_new_data(kh, shp, 3, MLX_FLOAT32);
    mlx_array v = mlx_array_new_data(vh, shp, 3, MLX_FLOAT32);
    mlx_array kq = mlx_array_new(), vq = mlx_array_new(), ks = mlx_array_new(), vs = mlx_array_new();
    int rc = llm_splash_store_run(&kq, &vq, &ks, &vs, k, v, st);
    int bad = 0; float maxdiff = 0.0f;
    if (rc == 0) {
        mlx_array_eval(kq); mlx_array_eval(ks);
        const int8_t* kqp = (const int8_t*)mlx_array_data_int8(kq);
        const float* ksp = (const float*)mlx_array_data_float32(ks);
        for (int i = 0; i < N && kqp; i++) if (kqp[i] != kref[i]) bad++;
        for (int i = 0; i < nkv * n && ksp; i++) { float d = fabsf(ksp[i] - ksref[i]); if (d > maxdiff) maxdiff = d; }
        snprintf(out, cap, "splash_store selftest: rc=0 int8_mismatch=%d/%d scale_maxdiff=%.6g %s",
                 bad, N, (double)maxdiff, (bad == 0 && maxdiff < 1e-6f) ? "OK" : "FAIL");
    } else snprintf(out, cap, "splash_store selftest: rc=%d (kernel build/apply FAIL)", rc);
    mlx_array_free(k); mlx_array_free(v); mlx_array_free(kq); mlx_array_free(vq); mlx_array_free(ks); mlx_array_free(vs);
    free(kh); free(vh); free(kref); free(ksref);
    return (rc == 0 && bad == 0) ? 0 : 1;
}

/* Self-test del pool int8 Splash: store -> scatter agli indici Splash -> verifica. */
int llm_ai_mlx_splash_pool_selftest(char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    const int nkv = 4, n = 8, hd = 64, cap_t = 32;
    const int N = nkv * n * hd;
    float* kh = (float*)malloc((size_t)N * sizeof(float));
    float* vh = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) { kh[i] = (float)((i*37%200)-100)*0.031f; vh[i] = (float)((i*53%160)-80)*0.027f; }
    mlx_stream st = mlx_default_gpu_stream_new();
    int ksh[3] = {nkv,n,hd};
    mlx_array k = mlx_array_new_data(kh, ksh, 3, MLX_FLOAT32);
    mlx_array v = mlx_array_new_data(vh, ksh, 3, MLX_FLOAT32);
    mlx_array kq = mlx_array_new(), vq = mlx_array_new(), ks = mlx_array_new(), vs = mlx_array_new();
    int rc = llm_splash_store_run(&kq, &vq, &ks, &vs, k, v, st);
    int psh[3] = {nkv,cap_t,hd}, vsh[3] = {nkv,hd,cap_t}, ssh[3] = {nkv,cap_t,1};
    mlx_array kpool = mlx_array_new(), vpool = mlx_array_new(), spool = mlx_array_new();
    if (rc == 0) { rc = mlx_zeros(&kpool, psh,3,MLX_INT8,st)==0 && mlx_zeros(&vpool, vsh,3,MLX_INT8,st)==0 && mlx_zeros(&spool, ssh,3,MLX_FLOAT32,st)==0 ? 0 : -1; }
    if (rc == 0) rc = llm_splash_pool_scatter(kpool, vpool, spool, kq, vq, ks, 0, n, nkv, hd, st);
    int bad = 0;
    if (rc == 0) {
        mlx_array_eval(kpool); mlx_array_eval(vpool); mlx_array_eval(spool);
        const int8_t* kp = (const int8_t*)mlx_array_data_int8(kpool);
        const int8_t* vp = (const int8_t*)mlx_array_data_int8(vpool);
        const float* sp = (const float*)mlx_array_data_float32(spool);
        /* riferimento int8 V (per il check del layout dim-major) */
        for (int h = 0; h < nkv; h++) for (int t = 0; t < n; t++) {
            const size_t off = ((size_t)h*n+t)*hd; float mx = 0.0f;
            for (int d = 0; d < hd; d++) { float a=fabsf(kh[off+d]), b=fabsf(vh[off+d]); if(a>mx)mx=a; if(b>mx)mx=b; }
            for (int d = 0; d < hd; d++) {
                long rk = mx==0.0f?0:lrintf(kh[off+d]*127.0f/mx); if(rk>127)rk=127; if(rk<-127)rk=-127;
                long rv = mx==0.0f?0:lrintf(vh[off+d]*127.0f/mx); if(rv>127)rv=127; if(rv<-127)rv=-127;
                int kidx = (h*cap_t + t)*hd + d;               /* K token-major */
                int vidx = (h*hd + d)*cap_t + t;               /* V dim-major */
                if (kp && kp[kidx] != (int8_t)rk) bad++;
                if (vp && vp[vidx] != (int8_t)rv) bad++;
                if (sp && t == 0 && d == 0 && fabsf(sp[h*cap_t+t] - (mx==0.0f?0.0f:mx/127.0f)) > 1e-6f) bad++;
            }
        }
        snprintf(out, cap, "splash_pool selftest: rc=0 mismatch=%d %s", bad, bad==0 ? "OK" : "FAIL");
    } else snprintf(out, cap, "splash_pool selftest: rc=%d FAIL", rc);
    mlx_array_free(k); mlx_array_free(v); mlx_array_free(kq); mlx_array_free(vq); mlx_array_free(ks); mlx_array_free(vs);
    mlx_array_free(kpool); mlx_array_free(vpool); mlx_array_free(spool);
    free(kh); free(vh);
    return (rc == 0 && bad == 0) ? 0 : 1;
}

/* Probe: MPP (MetalPerformancePrimitives) e' compilabile sotto metal_kernel di MLX? */
int llm_ai_mlx_splash_mpp_probe(char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    const char* hdr = "#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                      "using namespace metal;\nusing namespace mpp::tensor_ops;\n";
    const char* src = "  auto d = matmul2d_descriptor(8,8,8,false,true,false, matmul2d_descriptor::mode::multiply);\n"
                      "  x[0] = (float)d.get_capacity();\n";   /* tocca MPP nel corpo */
    mlx_vector_string iv = mlx_vector_string_new(); mlx_vector_string_append_value(iv, "a");
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, "x");
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new("mpp_probe", iv, ov, src, hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    if (k.ctx == NULL) { snprintf(out, cap, "MPP probe: build FALLITA (MPP non disponibile sotto mlx_fast_metal_kernel)"); return 1; }
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    int osh[1] = {1}; mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 1, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, 1, 1, 1); mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_stream s = mlx_default_gpu_stream_new();
    int ash[1] = {1}; mlx_array a = mlx_array_new_data((float[]){0.0f}, ash, 1, MLX_FLOAT32);
    mlx_vector_array in = mlx_vector_array_new(); mlx_vector_array_append_value(in, a);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, in, cfg, s);
    snprintf(out, cap, "MPP probe: build OK, apply rc=%d", rc);
    mlx_vector_array_free(in); mlx_vector_array_free(outs); mlx_array_free(a);
    mlx_fast_metal_kernel_config_free(cfg);
    return 0;
}

/* Self-test del read paged int8 Splash vs riferimento C (dequant-folded). */
int llm_ai_mlx_splash_attn_selftest(char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    const int nh = 4, nkv = 2, hd = 64, n = 8, cap_t = 32;
    const int N = nkv * n * hd;
    float* kh = (float*)malloc((size_t)N * sizeof(float));
    float* vh = (float*)malloc((size_t)N * sizeof(float));
    float* qh = (float*)malloc((size_t)(nh*hd) * sizeof(float));
    for (int i = 0; i < N; i++) { kh[i]=(float)((i*37%200)-100)*0.031f; vh[i]=(float)((i*53%160)-80)*0.027f; }
    for (int i = 0; i < nh*hd; i++) qh[i]=(float)((i*29%120)-60)*0.041f;
    mlx_stream st = mlx_default_gpu_stream_new();
    int ksh[3]={nkv,n,hd};
    mlx_array k=mlx_array_new_data(kh,ksh,3,MLX_FLOAT32), v=mlx_array_new_data(vh,ksh,3,MLX_FLOAT32);
    mlx_array kq=mlx_array_new(),vq=mlx_array_new(),ks=mlx_array_new(),vs=mlx_array_new();
    int rc = llm_splash_store_run(&kq,&vq,&ks,&vs,k,v,st);
    int psh[3]={nkv,cap_t,hd}, vsh[3]={nkv,hd,cap_t}, ssh[3]={nkv,cap_t,1};
    mlx_array kpool=mlx_array_new(),vpool=mlx_array_new(),spool=mlx_array_new();
    if (rc==0) rc = (mlx_zeros(&kpool,psh,3,MLX_INT8,st)==0 && mlx_zeros(&vpool,vsh,3,MLX_INT8,st)==0 && mlx_zeros(&spool,ssh,3,MLX_FLOAT32,st)==0)?0:-1;
    if (rc==0) rc = llm_splash_pool_scatter(kpool,vpool,spool,kq,vq,ks,0,n,nkv,hd,st);
    int qsh[2]={nh,hd}; mlx_array q=mlx_array_new_data(qh,qsh,2,MLX_FLOAT32);
    mlx_array attn=mlx_array_new();
    float attn_scale = 1.0f / sqrtf((float)hd);
    if (rc==0) rc = llm_splash_attn_run(&attn,q,kpool,vpool,ks,vs,n,cap_t,attn_scale,st);
    int bad=0; float maxdiff=0.0f;
    if (rc==0) {
        mlx_array_eval(attn);
        const float* ap=(const float*)mlx_array_data_float32(attn);
        const float* ksp=(const float*)mlx_array_data_float32(ks);
        const int8_t* kpp=(const int8_t*)mlx_array_data_int8(kpool);
        const int8_t* vpp=(const int8_t*)mlx_array_data_int8(vpool);
        for (int h=0; h<nh; h++) {
            int khh = h*nkv/nh;
            float sc[64]; float mxs=-1e30f;
            for (int t=0;t<n;t++){ float dot=0; for(int d=0;d<hd;d++) dot += qh[h*hd+d]*(float)kpp[(khh*cap_t+t)*hd+d]; float s=dot*ksp[khh*cap_t+t]*attn_scale; sc[t]=s; if(s>mxs)mxs=s; }
            float sm=0; for(int t=0;t<n;t++){ sc[t]=expf(sc[t]-mxs); sm+=sc[t]; }
            for (int d=0; d<hd; d++){ const int8_t* vpd; (void)vpd; float a=0; for(int t=0;t<n;t++) a += (sc[t]/sm)*ksp[khh*cap_t+t]*(float)vpp[(khh*hd+d)*cap_t+t];
                float diff=fabsf(a-ap[h*hd+d]); if(diff>maxdiff)maxdiff=diff; if(diff>1e-3f)bad++; }
        }
        snprintf(out,cap,"splash_attn selftest: rc=0 bad=%d/%d maxdiff=%.3g %s",bad,nh*hd,(double)maxdiff,bad==0?"OK":"FAIL");
    } else snprintf(out,cap,"splash_attn selftest: rc=%d FAIL",rc);
    mlx_array_free(k);mlx_array_free(v);mlx_array_free(kq);mlx_array_free(vq);mlx_array_free(ks);mlx_array_free(vs);
    mlx_array_free(kpool);mlx_array_free(vpool);mlx_array_free(spool);mlx_array_free(q);mlx_array_free(attn);
    free(kh);free(vh);free(qh);
    return (rc==0&&bad==0)?0:1;
}

/* Probe: una matmul2d MPP reale (M=32,N=32,K=64) sotto mlx_fast_metal_kernel. */
int llm_ai_mlx_splash_mpp_matmul_probe(char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    const char* hdr = "#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                      "using namespace metal;\nusing namespace mpp::tensor_ops;\n";
    const char* src =
        "  constexpr int M=32,N=32,K=64;\n"
        "  auto A = tensor(a, dextents<int,2>{K,M}, array<int,2>{1,K});\n"
        "  auto B = tensor(b, dextents<int,2>{K,N}, array<int,2>{1,K});\n"
        "  constexpr auto desc = matmul2d_descriptor(M,N,K,false,true,false, matmul2d_descriptor::mode::multiply);\n"
        "  matmul2d<desc, execution_simdgroups<4>> mm;\n"
        "  auto a0 = A.slice<K,M>(0,0); auto b0 = B.slice<K,N>(0,0);\n"
        "  auto d = mm.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n"
        "  mm.run(a0, b0, d);\n"
        "  auto C = tensor(c, dextents<int,2>{N,M}, array<int,2>{1,N});\n"
        "  d.store(C.slice<N,M>(0,0));\n";
    mlx_vector_string iv = mlx_vector_string_new();
    mlx_vector_string_append_value(iv, "a"); mlx_vector_string_append_value(iv, "b");
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, "c");
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new("mpp_mm_probe", iv, ov, src, hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    if (k.ctx == NULL) { snprintf(out, cap, "MPP matmul probe: build FALLITA"); return 1; }
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    int csh[2] = {32,32}; mlx_fast_metal_kernel_config_add_output_arg(cfg, csh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, 128, 1, 1); mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1);
    mlx_stream s = mlx_default_gpu_stream_new();
    int ash[2] = {64,32}, bsh[2] = {32,32};
    mlx_array a = mlx_array_new_data((float[]){0}, ash, 2, MLX_FLOAT32);   /* non inizializzato nel probe */
    mlx_array b = mlx_array_new_data((float[]){0}, bsh, 2, MLX_FLOAT32);
    mlx_vector_array in = mlx_vector_array_new(); mlx_vector_array_append_value(in, a); mlx_vector_array_append_value(in, b);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, in, cfg, s);
    snprintf(out, cap, "MPP matmul probe: build OK, apply rc=%d", rc);
    mlx_vector_array_free(in); mlx_vector_array_free(outs); mlx_array_free(a); mlx_array_free(b);
    mlx_fast_metal_kernel_config_free(cfg);
    return 0;
}

/* Probe MPP matmul2d v2 — pattern di Splash (tensor espliciti + .template). */
int llm_ai_mlx_splash_mpp_matmul2(char* out, size_t cap) {
    if (out == NULL || cap == 0) return -1;
    out[0] = '\0';
    const char* hdr = "#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                      "using namespace metal;\nusing namespace mpp::tensor_ops;\n";
    const char* src =
        "  constexpr int M=8,N=32,K=64;\n"
        "  auto a = tensor(x, dextents<int,2>{K,M}, array<int,2>{1,K});\n"
        "  auto a0 = a.slice<K,M>(0,0);\n"
        "  tensor<threadgroup half, dextents<int,2>, tensor_inline> bt0(stage, dextents<int,2>{K,N}, array<int,2>{1,K});\n"
        "  auto b0 = bt0.slice<K,N>(0,0);\n"
        "  constexpr auto desc = matmul2d_descriptor(M,N,K,false,true,false, matmul2d_descriptor::mode::multiply);\n"
        "  matmul2d<desc, execution_simdgroups<1>> op;\n"
        "  auto d = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n"
        "  op.run(a0, b0, d);\n"
        "  auto c = tensor(y, dextents<int,2>{N,M}, array<int,2>{1,N});\n"
        "  d.store(c.slice<N,M>(0,0));\n"
        "  threadgroup half stage[K*N];\n";
    mlx_vector_string iv = mlx_vector_string_new(); mlx_vector_string_append_value(iv, "x");
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, "y");
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new("mpp_mm2", iv, ov, src, hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    if (k.ctx == NULL) { snprintf(out, cap, "MPP matmul2: build FALLITA"); return 1; }
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    int csh[2] = {32,8}; mlx_fast_metal_kernel_config_add_output_arg(cfg, csh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, 32, 1, 1); mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_stream s = mlx_default_gpu_stream_new();
    int xsh[2] = {64,8}; mlx_array x = mlx_array_new_data((float[]){0}, xsh, 2, MLX_FLOAT32);
    mlx_vector_array in = mlx_vector_array_new(); mlx_vector_array_append_value(in, x);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, in, cfg, s);
    snprintf(out, cap, "MPP matmul2: build OK, apply rc=%d", rc);
    mlx_vector_array_free(in); mlx_vector_array_free(outs); mlx_array_free(x);
    mlx_fast_metal_kernel_config_free(cfg);
    return 0;
}
