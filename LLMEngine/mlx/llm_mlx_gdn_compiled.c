/* ============================================================================
 * Dominio GDN compiled: llm_mlx_gdn_kernel_dbg_enabled, gdn_inproj_fn, llm_mlx_gdn_inproj_compiled, llm_mlx_gdn_gate_metal_enabled, llm_mlx_m1_block_enabled, gdn_block_fn, llm_mlx_gdn_block_compiled (closure mlx_compile di decode GDN).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_gdn_gate_kernel.h"
#include "llm_mlx_gdn_kernel.h"
#include "llm_mlx_swiglu_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* LLM_GDN_KERNEL_DBG (default OFF, cached): branch-first sul GDN — dice QUALE ramo gira
 * (kernel Metal fuso vs loop per-token), una riga per layer. Zero overhead se non impostata.
 * Serve alla verifica del todo MLX-Decode-Composizione §12 (il kernel fuso deve partire sulle
 * shape reali: se qui compare "LOOP per-token", il kernel e' ineleggibile e va corretto). */
static int g_gdn_kernel_dbg_state = -1;
bool llm_mlx_gdn_kernel_dbg_enabled(void) {
    if (g_gdn_kernel_dbg_state < 0) g_gdn_kernel_dbg_state = (getenv("LLM_GDN_KERNEL_DBG") != NULL) ? 1 : 0;
    return g_gdn_kernel_dbg_state == 1;
}
/* M1 iter-1 (decode compilato): closure mlx_compile per le 4 proj in_proj del blocco GDN.
 * Pure (solo h + pesi catturati), shape-stabile a M=1 (decode). Il gate arch
 * (LLM_COMPILE_QWEN35) da solo NON agganciava; qui si compila la sub-graph delle proiezioni
 * (la classe di nodi maggiore: QuantizedMatmul). */
bool llm_mlx_compile_enabled(void);
bool llm_mlx_compile_qwen35_enabled(void);

static GdnInProjPayload* g_gdn_ip_pay[256];
static mlx_closure g_gdn_ip_cl[256];
static bool g_gdn_ip_valid[256];

static int gdn_inproj_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    GdnInProjPayload* p = (GdnInProjPayload*)payload;
    mlx_array h = mlx_array_new(); mlx_vector_array_get(&h, in, 0);
    mlx_stream st = mlx_default_gpu_stream_new();
    mlx_array o0=mlx_array_new(), o1=mlx_array_new(), o2=mlx_array_new(), o3=mlx_array_new();
    bool ok = quantized_linear_cfg(&o0, h, p->qkv_w, p->qkv_sc, p->qkv_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = quantized_linear_cfg(&o1, h, p->z_w, p->z_sc, p->z_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = quantized_linear_cfg(&o2, h, p->a_w, p->a_sc, p->a_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = quantized_linear_cfg(&o3, h, p->b_w, p->b_sc, p->b_b, p->gsz, p->bts, p->mode, st);
    int rc = 1;
    if (ok) { mlx_array arr[4] = { o0,o1,o2,o3 }; mlx_vector_array b = mlx_vector_array_new_data(arr,4);
              rc = (mlx_vector_array_set(out, b) == 0) ? 0 : 1; mlx_vector_array_free(b); }
    mlx_array_free(o0); mlx_array_free(o1); mlx_array_free(o2); mlx_array_free(o3);
    mlx_array_free(h); mlx_stream_free(st);
    return rc;
}

bool llm_mlx_gdn_inproj_compiled(int li, mlx_array h,
        mlx_array* qkv, mlx_array* z, mlx_array* av, mlx_array* bv,
        mlx_array qkv_w, mlx_array qkv_sc, mlx_array qkv_b,
        mlx_array z_w, mlx_array z_sc, mlx_array z_b,
        mlx_array a_w, mlx_array a_sc, mlx_array a_b,
        mlx_array b_w, mlx_array b_sc, mlx_array b_b,
        int gsz, int bts, const char* mode) {
    if (li < 0 || li >= 256) return false;
    if (!g_gdn_ip_valid[li]) {
        GdnInProjPayload* pl = (GdnInProjPayload*)calloc(1, sizeof(*pl));
        if (pl == NULL) return false;
        pl->qkv_w=qkv_w; pl->qkv_sc=qkv_sc; pl->qkv_b=qkv_b;
        pl->z_w=z_w; pl->z_sc=z_sc; pl->z_b=z_b;
        pl->a_w=a_w; pl->a_sc=a_sc; pl->a_b=a_b;
        pl->b_w=b_w; pl->b_sc=b_sc; pl->b_b=b_b;
        pl->gsz=gsz; pl->bts=bts; pl->mode=mode;
        g_gdn_ip_pay[li] = pl;
        mlx_closure raw = mlx_closure_new_func_payload(gdn_inproj_fn, pl, NULL);
        mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
        int rcc = mlx_compile(&g_gdn_ip_cl[li], raw, /*shapeless=*/false);
        mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
        mlx_closure_free(raw);
        if (rcc != 0) { free(pl); g_gdn_ip_pay[li] = NULL; if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1dbg] mlx_compile FAIL li=%d rc=%d\n", li, rcc); return false; }
        if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1dbg] mlx_compile OK li=%d\n", li);
        g_gdn_ip_valid[li] = true;
    }
    mlx_vector_array in = mlx_vector_array_new_data(&h, 1);
    mlx_vector_array res = mlx_vector_array_new();
    int rcc = mlx_closure_apply(&res, g_gdn_ip_cl[li], in);
    bool ok = (rcc == 0) && mlx_vector_array_size(res) == 4;
    if (ok) ok = mlx_vector_array_get(qkv,res,0)==0 && mlx_vector_array_get(z,res,1)==0 &&
                 mlx_vector_array_get(av,res,2)==0 && mlx_vector_array_get(bv,res,3)==0;
    mlx_vector_array_free(in); mlx_vector_array_free(res);
    return ok;
}

/* S5 (2026-09-21): cattura dello stato ricorrente GDN PER-TOKEN durante un forward
 * BATCH del verify speculativo -> su rifiuto si ripristina lo stato dopo `accepted`
 * token con un ALIAS O(1), senza il re-forward di 38ms (niente "row-state" compilato,
 * la stessa idea in software). Attiva/disattiva da verify_draft_batch. */
#define LLM_SPEC_MAXL 256
mlx_array g_spec_snap_k[LLM_SPEC_MAXL];  /* state_seq [B,T,Hv,Dv,Dk] (stato dopo ogni t) */
mlx_array g_spec_snap_v[LLM_SPEC_MAXL];  /* conv_input [B, cstate_len+T, conv_dim] */
int g_spec_capture = 0;

/* LLMMETAL_GDN_GATE_METAL: fonde in UN dispatch Metal la catena elementwise del gate GDN
 * (beta=sigmoid(b) + softplus + g=exp(-exp(A_log)*softplus(a+dt_bias))), che altrimenti sono
 * 8 op di grafo per layer x 48 layer. Il verify speculativo e' OP-COUNT-bound: questo e' il
 * primo pezzo della fusione del layer GDN. Default OFF (verificare parita' greedy prima). */
static int g_gdn_gate_metal_state = -1;
bool llm_mlx_gdn_gate_metal_enabled(void) {
    if (g_gdn_gate_metal_state < 0)
        g_gdn_gate_metal_state = (llm_ai_mlx_opt_env("LLMMETAL_GDN_GATE_METAL") != NULL) ? 1 : 0;
    return g_gdn_gate_metal_state == 1;
}

/* ===================== M1 vero: closure dell'INTERO blocco GDN di decode ==================
 * (Todo "(open) - Decode-Compilato-Shape-Stabile-M1-M2", M1; il suo rischio "primitivo custom
 * compatibile con mlx_compile" e' stato verificato in positivo: tools/mlx_compile_custom_probe.c.)
 *
 * Perche': il costo del decode e' costruire/encodare il grafo (~2811 nodi/token, ~2,3us/nodo =
 * l'86% del wall). I 36/48 layer GDN sono ~56% di quei nodi; collassarli in UNA closure
 * compilata per layer e' la leva (il passo compilato diventa UN nodo `Compiled`).
 *
 * Forma: la closure e' PURA — input {x, stato S, conv_state}, output {out_h, stato S', conv'}.
 * Lo stato e' ARGOMENTO I/O e dentro non si muta nulla in-place (la donation romperebbe il
 * cache-hit). I pesi sono catturati come costanti nel payload (cattura SICURA: nuovo handle +
 * mlx_array_set, perche' il motore libera gli handle dei pesi a fine blocco).
 *
 * Contenuto (specchia il blocco eager riga per riga, S=1): input_layernorm, le 4 in_proj, il
 * conv1d depthwise + silu, lo split q/k/v, rms_norm+inv_scale su q/k, il gate (kernel fuso
 * llm_mlx_gdn_gate o catena MLX, come deciso dal motore), la ricorrenza col kernel fuso
 * llm_mlx_gdn_prefill, l'aggiornamento del conv_state, RMSNormGated e la out_proj + residual.
 *
 * Gate: LLM_M1_BLOCK=1 (default OFF) + LLM_COMPILE=1 + LLM_COMPILE_QWEN35=1, n_tokens==1,
 * niente spec-capture, niente LoRA, niente LLMMETAL_OQ_A8_MINTOK (il path A8 a M=1 divergerebbe da
 * questa closure, che usa quantized_linear_cfg), e kernel GDN fuso attivo.
 * Verifica obbligatoria: IDS greedy BIT-IDENTICI vs eager.
 * ========================================================================================= */
#define LLM_M1_BLK_MAXL 256


static GdnBlkPayload* g_gdn_blk_pay[LLM_M1_BLK_MAXL];
static mlx_closure g_gdn_blk_cl[LLM_M1_BLK_MAXL];
static bool g_gdn_blk_valid[LLM_M1_BLK_MAXL];
static int g_m1_block_state = -1;

bool llm_mlx_m1_block_enabled(void) {
    if (g_m1_block_state < 0) g_m1_block_state = (llm_ai_mlx_opt_env("LLM_M1_BLOCK") != NULL) ? 1 : 0;
    return g_m1_block_state == 1;
}

static int gdn_block_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    GdnBlkPayload* p = (GdnBlkPayload*)payload;
    mlx_stream st = mlx_default_gpu_stream_new();
    const int B = 1, S = 1;
    const int Hk = p->Hk, Hv = p->Hv, Dk = p->Dk, Dv = p->Dv;
    const int key_dim = p->key_dim, value_dim = p->value_dim, conv_dim = p->conv_dim;
    const int cstate_len = p->cstate_len, repeat_factor = p->repeat_factor;
    bool ok = true;
    bool gate_fused = false;

    mlx_array x = mlx_array_new(), state = mlx_array_new(), conv0 = mlx_array_new();
    mlx_array h = mlx_array_new(), qkv = mlx_array_new(), z = mlx_array_new();
    mlx_array av = mlx_array_new(), bv = mlx_array_new();
    mlx_array conv_input = mlx_array_new(), conv_out = mlx_array_new();
    mlx_array sig = mlx_array_new(), silu = mlx_array_new();
    mlx_array q_part = mlx_array_new(), k_part = mlx_array_new(), v_part = mlx_array_new();
    mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new();
    mlx_array q_n = mlx_array_new(), k_n = mlx_array_new();
    mlx_array q_s = mlx_array_new(), k_s = mlx_array_new();
    mlx_array q_rep = mlx_array_new(), k_rep = mlx_array_new();
    mlx_array beta = mlx_array_new(), g = mlx_array_new();
    mlx_array a_plus_dt = mlx_array_new(), sp_exp = mlx_array_new(), sp_1p = mlx_array_new();
    mlx_array sp_log = mlx_array_new(), neg_exp = mlx_array_new(), neg_g = mlx_array_new();
    mlx_array y_stack = mlx_array_new(), state_new = mlx_array_new(), conv_new = mlx_array_new();
    mlx_array z4 = mlx_array_new(), y_norm = mlx_array_new(), z_sig = mlx_array_new();
    mlx_array silu_z = mlx_array_new(), gated = mlx_array_new(), gated_flat = mlx_array_new();
    mlx_array r = mlx_array_new(), out_h = mlx_array_new();
    mlx_array zero_arr = llm_mlx_act_scalar(0.0f);
    mlx_array inv2 = llm_mlx_act_scalar((float)(p->inv_scale * p->inv_scale));
    mlx_array inv1 = llm_mlx_act_scalar((float)p->inv_scale);
    mlx_vector_array kv_split = mlx_vector_array_new();

    if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1blk] fn call (trace/replay)\n");
    if (mlx_vector_array_get(&x, in, 0) != 0 ||
        mlx_vector_array_get(&state, in, 1) != 0 ||
        mlx_vector_array_get(&conv0, in, 2) != 0) ok = false;

    if (ok) ok = rms_norm_cfg(&h, x, p->ln_w, p->eps, st);
    if (ok) ok = quantized_linear_cfg(&qkv, h, p->qkv_w, p->qkv_sc, p->qkv_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = quantized_linear_cfg(&z, h, p->z_w, p->z_sc, p->z_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = quantized_linear_cfg(&av, h, p->a_w, p->a_sc, p->a_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = quantized_linear_cfg(&bv, h, p->b_w, p->b_sc, p->b_b, p->gsz, p->bts, p->mode, st);

    if (ok) { mlx_array cv_arr[2] = { conv0, qkv }; mlx_vector_array cv = mlx_vector_array_new_data(cv_arr, 2);
              ok = mlx_concatenate_axis(&conv_input, cv, /*axis=*/1, st) == 0; mlx_vector_array_free(cv); }
    if (ok) { int c_stride[1] = {1}, c_pl[1] = {0}, c_ph[1] = {0}, c_kd[1] = {1}, c_id[1] = {1};
              ok = mlx_conv_general(&conv_out, conv_input, p->conv_w, c_stride, 1, c_pl, 1, c_ph, 1,
                                    c_kd, 1, c_id, 1, /*groups=*/conv_dim, /*flip=*/false, st) == 0; }
    if (ok) ok = llm_mlx_silu_parity(&silu, &sig, conv_out, st) == 0;

    if (ok) { int secs[2] = { key_dim, 2 * key_dim };
              ok = mlx_split_sections(&kv_split, silu, secs, 2, /*axis=*/-1, st) == 0; }
    if (ok) ok = mlx_vector_array_get(&q_part, kv_split, 0) == 0 &&
                 mlx_vector_array_get(&k_part, kv_split, 1) == 0 &&
                 mlx_vector_array_get(&v_part, kv_split, 2) == 0;
    if (ok) { int q_shape[4] = { B, S, Hk, Dk }, v_shape[4] = { B, S, Hv, Dv };
              ok = mlx_reshape(&q4, q_part, q_shape, 4, st) == 0 &&
                   mlx_reshape(&k4, k_part, q_shape, 4, st) == 0 &&
                   mlx_reshape(&v4, v_part, v_shape, 4, st) == 0; }
    if (ok) ok = mlx_fast_rms_norm_no_scale(&q_n, q4, Dk, 1e-6f, st) &&
                 mlx_fast_rms_norm_no_scale(&k_n, k4, Dk, 1e-6f, st);
    if (ok) ok = mlx_multiply(&q_s, q_n, inv2, st) == 0 && mlx_multiply(&k_s, k_n, inv1, st) == 0;

    if (ok && p->gate_metal && mlx_array_dtype(bv) == mlx_array_dtype(av) &&
        mlx_array_dtype(p->eAlog) == MLX_FLOAT32 &&
        mlx_array_dtype(p->dt_bias) == mlx_array_dtype(av))
        gate_fused = (llm_mlx_gdn_gate(&g, &beta, av, bv, p->eAlog, p->dt_bias, st) == 0);
    if (ok && !gate_fused) {
        if (ok) ok = mlx_sigmoid(&beta, bv, st) == 0;
        if (ok) ok = mlx_add(&a_plus_dt, av, p->dt_bias, st) == 0;
        if (ok) ok = mlx_logaddexp(&sp_log, a_plus_dt, zero_arr, st) == 0;  /* nn.softplus = logaddexp(x,0) */
        if (ok) ok = mlx_multiply(&neg_exp, p->eAlog, sp_log, st) == 0;
        if (ok) ok = mlx_negative(&neg_g, neg_exp, st) == 0;
        if (ok) ok = mlx_exp(&g, neg_g, st) == 0;
    }

    if (ok) {
        if (repeat_factor > 1)
            ok = mlx_repeat_axis(&q_rep, q_s, repeat_factor, /*axis=*/2, st) == 0 &&
                 mlx_repeat_axis(&k_rep, k_s, repeat_factor, /*axis=*/2, st) == 0;
        else
            ok = mlx_array_set(&q_rep, q_s) == 0 && mlx_array_set(&k_rep, k_s) == 0;
    }

    if (ok) ok = llm_mlx_gdn_prefill(&y_stack, &state_new, q_s, k_s, v4, g, beta, state, S,
                                     /*g_vector=*/0, st) == 0;

    if (ok) { int cs0[3] = {0, S, 0}, cs1[3] = {1, S + cstate_len, conv_dim}, csd[3] = {1, 1, 1};
              ok = mlx_slice(&conv_new, conv_input, cs0, 3, cs1, 3, csd, 3, st) == 0; }

    if (ok) { int z_shape[4] = { B, S, Hv, Dv }; ok = mlx_reshape(&z4, z, z_shape, 4, st) == 0; }
    if (ok) ok = rms_norm_cfg(&y_norm, y_stack, p->norm_w, p->eps, st);
    if (ok) ok = mlx_sigmoid(&z_sig, z4, st) == 0;
    if (ok) ok = mlx_multiply(&silu_z, z4, z_sig, st) == 0;
    if (ok) ok = mlx_multiply(&gated, y_norm, silu_z, st) == 0;
    if (ok) { int f_shape[3] = { B, S, value_dim }; ok = mlx_reshape(&gated_flat, gated, f_shape, 3, st) == 0; }
    if (ok) ok = quantized_linear_cfg(&r, gated_flat, p->out_w, p->out_sc, p->out_b, p->gsz, p->bts, p->mode, st);
    if (ok) ok = mlx_add(&out_h, x, r, st) == 0;

    int rc = 1;
    if (ok) {
        mlx_array arr[3] = { out_h, state_new, conv_new };
        mlx_vector_array b = mlx_vector_array_new_data(arr, 3);
        rc = (mlx_vector_array_set(out, b) == 0) ? 0 : 1;
        mlx_vector_array_free(b);
    }

    mlx_vector_array_free(kv_split);
    mlx_array_free(x); mlx_array_free(state); mlx_array_free(conv0);
    mlx_array_free(h); mlx_array_free(qkv); mlx_array_free(z);
    mlx_array_free(av); mlx_array_free(bv);
    mlx_array_free(conv_input); mlx_array_free(conv_out);
    mlx_array_free(sig); mlx_array_free(silu);
    mlx_array_free(q_part); mlx_array_free(k_part); mlx_array_free(v_part);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_n); mlx_array_free(k_n); mlx_array_free(q_s); mlx_array_free(k_s);
    mlx_array_free(q_rep); mlx_array_free(k_rep);
    mlx_array_free(beta); mlx_array_free(g);
    mlx_array_free(a_plus_dt); mlx_array_free(sp_exp); mlx_array_free(sp_1p);
    mlx_array_free(sp_log); mlx_array_free(neg_exp); mlx_array_free(neg_g);
    mlx_array_free(y_stack); mlx_array_free(state_new); mlx_array_free(conv_new);
    mlx_array_free(z4); mlx_array_free(y_norm); mlx_array_free(z_sig);
    mlx_array_free(silu_z); mlx_array_free(gated); mlx_array_free(gated_flat);
    mlx_array_free(r); mlx_array_free(out_h);
    mlx_array_free(zero_arr); mlx_array_free(inv2); mlx_array_free(inv1);
    mlx_stream_free(st);
    return rc;
}

/* Costruisce (una volta per layer) e applica la closure del blocco GDN. Vedi la doc sopra. */
bool llm_mlx_gdn_block_compiled(LlmMlxModel* m, int li, mlx_array x,
        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    if (li < 0 || li >= LLM_M1_BLK_MAXL) return false;
    char namebuf[256];
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int B = 1, Hk = m->cfg.linear_num_key_heads, Hv = m->cfg.linear_num_value_heads;
    const int Dk = m->cfg.linear_key_head_dim, Dv = m->cfg.linear_value_head_dim;
    const int key_dim = Hk * Dk, value_dim = Hv * Dv, conv_dim = 2 * key_dim + value_dim;
    const int cstate_len = m->cfg.linear_conv_kernel_dim - 1;
    const int repeat_factor = (Hk > 0) ? (Hv / Hk) : 1;

    if (!g_gdn_blk_valid[li]) {
        GdnBlkPayload* pl = (GdnBlkPayload*)calloc(1, sizeof(*pl));
        if (pl == NULL) return false;
        if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1blk] build start li=%d\n", li);
        /* cattura SICURA: nuovo handle + mlx_array_set (il buffer resta vivo: il motore
         * libera gli handle dei pesi a fine blocco). */
#define GDNBLK_CAP(dst, src) do { dst = mlx_array_new(); \
        if (mlx_array_set(&(dst), (src)) != 0) { return false; } } while (0)
        mlx_array t = mlx_array_new();
        tname(m->cfg.layer_prefix, li, "input_layernorm.weight", namebuf, sizeof(namebuf));
        if (!multishard_get(&m->weights, namebuf, &t)) { mlx_array_free(t); return false; }
        GDNBLK_CAP(pl->ln_w, t); mlx_array_free(t);
        tname(m->cfg.layer_prefix, li, "linear_attn.in_proj_qkv", namebuf, sizeof(namebuf));
        if (!multishard_get_triple(&m->weights, namebuf, &pl->qkv_w, &pl->qkv_sc, &pl->qkv_b)) return false;
        tname(m->cfg.layer_prefix, li, "linear_attn.in_proj_z", namebuf, sizeof(namebuf));
        if (!multishard_get_triple(&m->weights, namebuf, &pl->z_w, &pl->z_sc, &pl->z_b)) return false;
        tname(m->cfg.layer_prefix, li, "linear_attn.in_proj_a", namebuf, sizeof(namebuf));
        if (!multishard_get_triple(&m->weights, namebuf, &pl->a_w, &pl->a_sc, &pl->a_b)) return false;
        tname(m->cfg.layer_prefix, li, "linear_attn.in_proj_b", namebuf, sizeof(namebuf));
        if (!multishard_get_triple(&m->weights, namebuf, &pl->b_w, &pl->b_sc, &pl->b_b)) return false;
        tname(m->cfg.layer_prefix, li, "linear_attn.out_proj", namebuf, sizeof(namebuf));
        if (!multishard_get_triple(&m->weights, namebuf, &pl->out_w, &pl->out_sc, &pl->out_b)) return false;
        t = mlx_array_new();   /* handle NUOVO: il precedente e' stato liberato (mlx_array_set_ scriverebbe su un ctx dangling) */
        tname(m->cfg.layer_prefix, li, "linear_attn.norm.weight", namebuf, sizeof(namebuf));
        if (!multishard_get(&m->weights, namebuf, &t)) { mlx_array_free(t); return false; }
        GDNBLK_CAP(pl->norm_w, t); mlx_array_free(t);
        t = mlx_array_new();
        tname(m->cfg.layer_prefix, li, "linear_attn.A_log", namebuf, sizeof(namebuf));
        if (!multishard_get(&m->weights, namebuf, &t)) { mlx_array_free(t); return false; }
        GDNBLK_CAP(pl->A_log, t); mlx_array_free(t);
        {   /* dt_bias e conv_w: dalla cache del motore se presente (stesso valore del path eager) */
            mlx_array* _dt = llm_mlx_gdn_dt_bias(m, li);
            if (_dt != NULL) { GDNBLK_CAP(pl->dt_bias, *_dt); }
            else { t = mlx_array_new();
                   tname(m->cfg.layer_prefix, li, "linear_attn.dt_bias", namebuf, sizeof(namebuf));
                   if (!multishard_get(&m->weights, namebuf, &t)) { mlx_array_free(t); return false; }
                   GDNBLK_CAP(pl->dt_bias, t); mlx_array_free(t); }
            mlx_array* _cw = llm_mlx_gdn_conv_cached(m, li);
            if (_cw != NULL) { GDNBLK_CAP(pl->conv_w, *_cw); }
            else { t = mlx_array_new();
                   tname(m->cfg.layer_prefix, li, "linear_attn.conv1d.weight", namebuf, sizeof(namebuf));
                   if (!multishard_get(&m->weights, namebuf, &t)) { mlx_array_free(t); return false; }
                   mlx_array cw32 = mlx_array_new();
                   if (mlx_astype(&cw32, t, llm_mlx_act_dt(m->cfg.bfloat16_input), m->stream) != 0) {
                       mlx_array_free(t); mlx_array_free(cw32); return false; }
                   mlx_array_free(t); GDNBLK_CAP(pl->conv_w, cw32); mlx_array_free(cw32); }
        }
        {   /* eAlog: dalla cache se c'e', altrimenti exp(A_log) UNA volta (non per token) */
            mlx_array* _ae = llm_mlx_gdn_alog_exp(m, li);
            if (_ae != NULL) { GDNBLK_CAP(pl->eAlog, *_ae); }
            else { mlx_array e = mlx_array_new();
                   if (mlx_exp(&e, pl->A_log, m->stream) != 0) { mlx_array_free(e); return false; }
                   GDNBLK_CAP(pl->eAlog, e); mlx_array_free(e); }
        }
        if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1blk] capture done li=%d\n", li);
#undef GDNBLK_CAP
        pl->Hk = Hk; pl->Hv = Hv; pl->Dk = Dk; pl->Dv = Dv;
        pl->key_dim = key_dim; pl->value_dim = value_dim; pl->conv_dim = conv_dim;
        pl->cstate_len = cstate_len; pl->repeat_factor = repeat_factor;
        pl->gsz = gsz; pl->bts = bts; pl->mode = m->cfg.quant_mode;
        pl->eps = m->cfg.rms_norm_eps;
        pl->act_dt = llm_mlx_act_dt(m->cfg.bfloat16_input);
        pl->gate_metal = llm_mlx_gdn_gate_metal_enabled() ? 1 : 0;
        pl->inv_scale = pow((double)Dk, -0.5);
        g_gdn_blk_pay[li] = pl;
        mlx_closure raw = mlx_closure_new_func_payload(gdn_block_fn, pl, NULL);
        mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
        int rcc = mlx_compile(&g_gdn_blk_cl[li], raw, /*shapeless=*/false);
        mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
        mlx_closure_free(raw);
        if (rcc != 0) {
            if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1blk] mlx_compile FAIL li=%d rc=%d\n", li, rcc);
            g_gdn_blk_pay[li] = NULL;
            return false;
        }
        if (getenv("LLM_M1_DBG")) fprintf(stderr, "[m1blk] mlx_compile OK li=%d\n", li);
        g_gdn_blk_valid[li] = true;
    }

    mlx_array stt = mlx_array_new(), cvv = mlx_array_new();
    int state_shape[4] = { B, Hv, Dv, Dk };
    int cshape[3] = { B, cstate_len, conv_dim };
    bool iok = true;
    if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache) > 0)
        iok = mlx_array_set(&stt, *k_cache) == 0;
    else
        iok = mlx_zeros(&stt, state_shape, 4, MLX_FLOAT32, m->stream) == 0;
    if (iok) {
        if (has_cache && v_cache && v_cache->ctx != NULL && mlx_array_ndim(*v_cache) > 0)
            iok = mlx_array_set(&cvv, *v_cache) == 0;
        else
            iok = mlx_zeros(&cvv, cshape, 3, llm_mlx_act_dt(m->cfg.bfloat16_input), m->stream) == 0;
    }
    bool okk = false;
    if (iok) {
        mlx_array ins[3] = { x, stt, cvv };
        mlx_vector_array in = mlx_vector_array_new_data(ins, 3);
        mlx_vector_array res = mlx_vector_array_new();
        int rcc = mlx_closure_apply(&res, g_gdn_blk_cl[li], in);
        okk = (rcc == 0) && mlx_vector_array_size(res) == 3;
        if (okk) {
            mlx_array o = mlx_array_new(), sn = mlx_array_new(), cn = mlx_array_new();
            okk = mlx_vector_array_get(&o, res, 0) == 0 &&
                  mlx_vector_array_get(&sn, res, 1) == 0 &&
                  mlx_vector_array_get(&cn, res, 2) == 0;
            if (okk) {
                mlx_array_set(out_h, o);
                mlx_array_set(k_cache, sn);
                mlx_array_set(v_cache, cn);
            }
            mlx_array_free(o); mlx_array_free(sn); mlx_array_free(cn);
        }
        mlx_vector_array_free(in); mlx_vector_array_free(res);
    }
    mlx_array_free(stt); mlx_array_free(cvv);
    return okk;
}
