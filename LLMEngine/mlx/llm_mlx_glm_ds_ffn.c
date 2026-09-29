/* ============================================================================
 * llm_mlx_glm_ds_ffn.c — dominio FFN MoE di GLM-4.5 e DeepSeek-V2: gate sorted
 * (GLM/DS), deepseek_v2_ffn_block, glm4_moe_ffn_block. Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_router_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GLM-4.5-Air (glm4_moe) FFN: primi `first_k_dense_replace` layer densi (SwiGLU,
 * riusa dense_swiglu_ffn per i nomi mlp.gate/up/down_proj); gli altri MoE:
 * router mlp.gate (BF16 piatto) + e_score_correction_bias -> sigmoid scores ->
 * top-k -> rinormalizza -> esperti mlp.switch_mlp (4-bit group 64, gather_qmm) ->
 * somma pesata -> + shared_experts (dense SwiGLU, intermediate 1408) -> residuo.
 * Reference: mlx_lm/models/glm4_moe.py (MoEGate sigmoid noaux_tc, MoE, MLP). */
static int g_glm_sorted_state = -1;
/* Gate del path SORTED per il dispatch esperti del GLM (SwitchGLU reference):
 * quando M=n_tokens*top_k>=64 la reference (mlx_lm switch_layers.SwitchGLU) fa
 * argsort + gather ordinato + unsort per avere accesso per-esperto contiguo;
 * il path legacy fa gather_qmm unsorted sempre. DEFAULT sorted (parita' greedy
 * verificata 2026-09-06: identica a LLM_GLM_SORTED=0); LLM_GLM_SORTED=0 forza
 * il path legacy. */
bool llm_glm_sorted_enabled(void) {
    if (g_glm_sorted_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_GLM_SORTED");
        if (e != NULL && strcmp(e, "0") == 0) g_glm_sorted_state = 0;
        else g_glm_sorted_state = 1; /* default sorted */
    }
    return g_glm_sorted_state == 1;
}
/* DeepSeek-V2 (MLA) FFN: primi `first_k_dense_replace` layer densi (SwiGLU), poi DeepSeekMoE
 * (router softmax + top-k greedy; n_group>1 -> group_limited_greedy con max-per-gruppo; esperti
 * routed `switch_mlp` via gather_qmm + `shared_experts`). Rif: mlx_lm/models/deepseek_v2.py. */

/* DeepSeek-V2 MoE: gate del path SORTED (SwitchGLU 1:1 quando M = n_tokens*top_k >= 64).
 * Default ON (come qwen3_moe/glm4_moe); LLM_DS_SORTED=0 per l'A/B. */
static int g_ds_sorted_state = -1;
bool llm_ds_sorted_enabled(void) {
    if (g_ds_sorted_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_DS_SORTED");
        g_ds_sorted_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_ds_sorted_state == 1;
}

bool deepseek_v2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    if (layer_idx < m->cfg.first_k_dense_replace)
        return dense_swiglu_ffn(m, layer_idx, n_tokens, h, out);
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    char namebuf[256], nb1[256];
#define DCHK(cond, what) do { if (!(cond)) { fprintf(stderr, "deepseek_v2_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array ln2_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof namebuf);
    ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); DCHK(ok, "post_attention_layernorm"); }

    mlx_array gw = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate.weight", namebuf, sizeof namebuf);
    ok = ok && multishard_get(&m->weights, namebuf, &gw);
    mlx_array gates = mlx_array_new();
    if (ok) { ok = plain_linear(&gates, normed2, gw, m->stream); DCHK(ok, "router matmul"); }
    mlx_array gates_f = mlx_array_new(), scores = mlx_array_new();
    /* Router softmax nel dtype di attivazione (come mlx_lm) invece di f32: il f32 del gate
     * si propagava a TUTTO il combine MoE -> residuo f32 -> i rms_norm e i quantized_matmul
     * castavano per-token (misurato V2-Lite: 156 AsType/token, 0 con questa scelta).
     * Opt-out LLM_ROUTER_F16=0 (torna al f32). IDS greedy: 116,68,207 identico; A/B +2,6/+2,0/+1%
     * decode @512/2048/8192. Il softmax resta precise=true (piu' preciso dello standard). */
    if (ok) {
        static int rf16 = -1; if (rf16 < 0) rf16 = llm_ai_mlx_opt_env("LLM_ROUTER_F16") != NULL ? (strcmp(llm_ai_mlx_opt_env("LLM_ROUTER_F16"), "0") != 0) : 1;
        if (rf16 && mlx_array_dtype(gates) == MLX_FLOAT16)
            ok = mlx_array_set(&gates_f, gates) == 0;
        else
            ok = mlx_astype(&gates_f, gates, MLX_FLOAT32, m->stream) == 0;
        DCHK(ok, "router f32");
    }
    if (ok && layer_idx == 1 && n_tokens > 1) {
    }
    if (ok) { ok = mlx_softmax_axis(&scores, gates_f, /*axis=*/-1, /*precise=*/true, m->stream) == 0; DCHK(ok, "router softmax"); }
    if (ok && layer_idx == 1 && n_tokens > 1) {
    }

    /* group_limited_greedy (solo n_group>1): max per gruppo, tieni i topk_group. V2-Lite: n_group=1 -> skip. */
    mlx_array pruned = mlx_array_new();
    if (ok && m->cfg.n_group > 1 && m->cfg.n_experts > 0 && m->cfg.n_experts % m->cfg.n_group == 0 &&
        m->cfg.topk_group > 0 && m->cfg.topk_group < m->cfg.n_group) {
        const int ng = m->cfg.n_group, epg = m->cfg.n_experts / ng;
        int us4[4] = {1, n_tokens, ng, epg};
        mlx_array bu = mlx_array_new();
        ok = mlx_reshape(&bu, scores, us4, 4, m->stream) == 0; DCHK(ok, "group reshape");
        mlx_array gmax = mlx_array_new();
        if (ok) ok = mlx_max_axis(&gmax, bu, -1, true, m->stream) == 0;
        if (ok) {
            const int k = ng - m->cfg.topk_group;
            mlx_array gpart = mlx_array_new();
            ok = mlx_argpartition_axis(&gpart, gmax, k - 1, 2, m->stream) == 0;
            if (ok) {
                int st4[4]={0,0,0,0}, sp4[4]={1,n_tokens,k,1}, sv4[4]={1,1,1,1};
                mlx_array gview = mlx_array_new(), gidx = mlx_array_new();
                ok = mlx_slice(&gview, gpart, st4,4, sp4,4, sv4,4, m->stream) == 0;
                if (ok) ok = mlx_contiguous(&gidx, gview, false, m->stream) == 0;
                mlx_array_free(gview);
                if (ok) {
                    mlx_array zero = mlx_array_new_float32(0.0f), bu2 = mlx_array_new();
                    ok = mlx_put_along_axis(&bu2, bu, gidx, zero, 2, m->stream) == 0;
                    mlx_array_free(zero);
                    if (ok) { mlx_array_free(bu); bu = bu2; } else mlx_array_free(bu2);
                }
                mlx_array_free(gidx);
            }
            mlx_array_free(gpart);
        }
        mlx_array_free(gmax);
        if (ok) { int fs3[3]={1,n_tokens,m->cfg.n_experts}; ok = mlx_reshape(&pruned, bu, fs3, 3, m->stream) == 0; }
        mlx_array_free(bu);
    } else if (ok) mlx_array_set(&pruned, scores);
    DCHK(ok, "group pruning");

    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, pruned, -m->cfg.top_k, -1, m->stream) == 0; DCHK(ok, "argpartition");
        if (ok) {
            const int ne = (int)mlx_array_dim(part, 2);
            int st3[3]={0,0,ne-m->cfg.top_k}, sp3[3]={1,n_tokens,ne}, sv3[3]={1,1,1};
            mlx_array view = mlx_array_new();
            ok = mlx_slice(&view, part, st3,3, sp3,3, sv3,3, m->stream) == 0;
            /* LLMENGINE_MOE_NOCONTIG (default ON = salta): `take_along_axis` accetta la vista
             * (mlx_lm passa la slice di argpartition direttamente, 0 Contiguous). Era un
             * workaround difensivo: 1 nodo/layer/token. Opt-out LLMENGINE_MOE_NOCONTIG=0. */
            static int nocontig = -1; if (nocontig < 0) nocontig = llm_ai_mlx_opt_env("LLMENGINE_MOE_NOCONTIG") != NULL ? (strcmp(llm_ai_mlx_opt_env("LLMENGINE_MOE_NOCONTIG"), "0") != 0) : 1;
            if (ok && !nocontig) ok = mlx_contiguous(&top_inds, view, false, m->stream) == 0;
            else if (ok) ok = mlx_array_set(&top_inds, view) == 0;
            mlx_array_free(view);
        }
    }
    mlx_array weights = mlx_array_new();
    if (ok) {
        mlx_array tv = mlx_array_new();
        ok = mlx_take_along_axis(&tv, scores, top_inds, -1, m->stream) == 0;
        if (ok && m->cfg.norm_topk_prob && m->cfg.top_k > 1) {
            mlx_array den = mlx_array_new(), tvn = mlx_array_new();
            ok = mlx_sum_axis(&den, tv, -1, true, m->stream) == 0;
            if (ok) ok = mlx_divide(&tvn, tv, den, m->stream) == 0;
            mlx_array_free(den);
            if (ok) mlx_array_set(&weights, tvn); else mlx_array_free(tvn);
            mlx_array_free(tv);
        } else { mlx_array_set(&weights, tv); mlx_array_free(tv); }
        if (ok && m->cfg.routed_scaling_factor != 1.0f) {
            mlx_array scf = mlx_array_new_float32(m->cfg.routed_scaling_factor), sc = mlx_array_new(), wsc = mlx_array_new();
            if (ok) ok = mlx_astype(&sc, scf, mlx_array_dtype(weights), m->stream) == 0;   /* weak scalar: resta nel dtype dei pesi (bf16) */
            ok = ok && mlx_multiply(&wsc, weights, sc, m->stream) == 0;
            mlx_array_free(scf); mlx_array_free(sc);
            if (ok) mlx_array_set(&weights, wsc); else mlx_array_free(wsc);
        }
    }
    DCHK(ok, "top-k weights");
    if (ok && layer_idx == 1 && n_tokens > 1) {
        mlx_array wf=mlx_array_new(), tf=mlx_array_new();
        mlx_array_free(wf);
        mlx_array_free(tf);
    }

    mlx_array gp_w=mlx_array_new(),gp_s=mlx_array_new(),gp_b=mlx_array_new();
    mlx_array up_w=mlx_array_new(),up_s=mlx_array_new(),up_b=mlx_array_new();
    mlx_array dn_w=mlx_array_new(),dn_s=mlx_array_new(),dn_b=mlx_array_new();
    int xp_group=gsz, xp_bits=bts; char xp_mode[16]; /* xp_mode e' SEMPRE scritto dalla llm_mlx_quant_derive sotto: niente copia iniziale (dead code) */
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gp_w, &gp_s, &gp_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &up_w, &up_s, &up_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj", namebuf, sizeof namebuf);
    ok = ok && multishard_get_triple(&m->weights, namebuf, &dn_w, &dn_s, &dn_b);
    llm_mlx_quant_derive(gp_w, gp_s, gsz, bts, "affine", &xp_group, &xp_bits, xp_mode, sizeof xp_mode);
    DCHK(ok, "switch_mlp lookup");
    if (ok && llm_ai_mlx_opt_env("LLMENGINE_MOE_DT")) fprintf(stderr, "[moe-dt] layer=%d h=%d normed2=%d gp_w=%d gp_s=%d gp_b=%d (F32=%d BF16=%d) mode=%s\n",
        layer_idx, (int)mlx_array_dtype(h), (int)mlx_array_dtype(normed2),
        (int)mlx_array_dtype(gp_w), (int)mlx_array_dtype(gp_s), (int)mlx_array_dtype(gp_b),
        (int)MLX_FLOAT32, (int)MLX_BFLOAT16, xp_mode);
    mlx_optional_int xp_gsz={xp_group,true}, xp_bts={xp_bits,true};
    mlx_array no_lhs = mlx_array_new(), top_inds_arr = mlx_array_new();
    if (ok) {
        mlx_array tu = mlx_array_new();
        ok = mlx_astype(&tu, top_inds, MLX_UINT32, m->stream) == 0;
        if (ok) { const int ts[2]={n_tokens,m->cfg.top_k}; mlx_array tv2=mlx_array_new();
                  ok = mlx_reshape(&tv2, tu, ts, 2, m->stream)==0;
                  /* `tu` e' un astype (dense): il reshape non aggiunge stride, la vista e' gia'
                   * contigua -> niente copia (mlx_lm: 0 Contiguous). Vedi LLMENGINE_MOE_NOCONTIG sopra. */
                  static int nocontig2 = -1; if (nocontig2 < 0) nocontig2 = llm_ai_mlx_opt_env("LLMENGINE_MOE_NOCONTIG") != NULL ? (strcmp(llm_ai_mlx_opt_env("LLMENGINE_MOE_NOCONTIG"), "0") != 0) : 1;
                  if (ok && !nocontig2) ok = mlx_contiguous(&top_inds_arr, tv2, false, m->stream)==0;
                  else if (ok) ok = mlx_array_set(&top_inds_arr, tv2) == 0;
                  mlx_array_free(tv2); }
        mlx_array_free(tu);
    }
    mlx_array x1 = mlx_array_new();
    /* A/B 2026-09-23: lhs 5D (mlx_lm expand_dims(-2,-3)) vs 4D storica -> NESSUNA differenza
     * (5D ~18.0 vs 4D ~17.7 tok/s), parita' IDS ok. Tenuta la 4D storica. LLM_GLM_X1_5D=1 per il 5D. */
    static int g_x1_5d = -1; if (g_x1_5d < 0) g_x1_5d = (llm_ai_mlx_opt_env("LLM_GLM_X1_5D") != NULL) ? 1 : 0;
    if (ok && g_x1_5d) { const int x1s[5]={1,n_tokens,1,1,m->cfg.hidden_size}; ok = mlx_reshape(&x1, normed2, x1s, 5, m->stream)==0; }
    else if (ok) { const int x1s[4]={n_tokens,1,1,m->cfg.hidden_size}; ok = mlx_reshape(&x1, normed2, x1s, 4, m->stream)==0; }
    if (ok && llm_ai_mlx_opt_env("LLMENGINE_MOE_DT")) fprintf(stderr, "[moe-dt] layer=%d h=%d normed2=%d x1=%d gp_w=%d gp_s=%d gp_b=%d (F32=%d BF16=%d)\n",
        layer_idx, (int)mlx_array_dtype(h), (int)mlx_array_dtype(normed2), (int)mlx_array_dtype(x1),
        (int)mlx_array_dtype(gp_w), (int)mlx_array_dtype(gp_s), (int)mlx_array_dtype(gp_b),
        (int)MLX_FLOAT32, (int)MLX_BFLOAT16);
    mlx_array gate_out=mlx_array_new(), up_out=mlx_array_new(), sig=mlx_array_new(), silu=mlx_array_new(), act=mlx_array_new();
    mlx_array down_out=mlx_array_new(), w_arr=mlx_array_new(), scaled=mlx_array_new(), summed=mlx_array_new(), moe_out=mlx_array_new();
    /* Path SORTED (1:1 mlx_lm switch_layers.SwitchGLU quando indices.size >= 64). DeepSeek-V2
     * era l'UNICO MoE a usare sempre il gather UNSORTED: misurato (LLM_PHASE) FFN = 91% del
     * prefill V2-Lite e ~5x sotto mlx_lm. Stessa sequenza del branch sorted di qwen3_moe/glm4_moe
     * (commit 3c667c9b/0699c2ec): argsort esperti, righe x ordinate, gather_qmm sorted_indices=true,
     * unsort. Combine invariata (renorm, routed scaling, somma pesata, shared_experts). */
    bool do_sort = ok && llm_ds_sorted_enabled() &&
                   ((int64_t)n_tokens * (int64_t)m->cfg.top_k) >= 64;
    /* Q2-b: imatrix (path sorted) — esperto per riga + riga sorted -> riga token-major. */
    uint32_t* im_expert = NULL; uint32_t* im_src = NULL; int im_rows = 0;
    if (ok && do_sort) {
        const int M = n_tokens * m->cfg.top_k;
        mlx_array tt = mlx_array_new();
        { const int ts[1] = {M}; if (ok) ok = mlx_reshape(&tt, top_inds_arr, ts, 1, m->stream) == 0; }
        mlx_array order = mlx_array_new();
        if (ok) ok = mlx_argsort(&order, tt, m->stream) == 0;
        mlx_array inv = mlx_array_new();
        if (ok) ok = mlx_argsort(&inv, order, m->stream) == 0;
        mlx_array df = mlx_array_new();
        { mlx_array kk = mlx_array_new_int(m->cfg.top_k);
          if (ok) ok = mlx_floor_divide(&df, order, kk, m->stream) == 0; mlx_array_free(kk); }
        mlx_array dv = mlx_array_new();
        if (ok) ok = mlx_astype(&dv, df, MLX_UINT32, m->stream) == 0;
        mlx_array idxs = mlx_array_new();
        if (ok) ok = mlx_take(&idxs, tt, order, m->stream) == 0;
        mlx_array dvm4 = mlx_array_new();
        { const int ms[4] = {M, 1, 1, 1}; if (ok) ok = mlx_reshape(&dvm4, dv, ms, 4, m->stream) == 0; }
        mlx_array rn = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&rn, x1, dvm4, 0, m->stream) == 0;
        mlx_array rnc = mlx_array_new();
        if (ok) ok = mlx_contiguous(&rnc, rn, /*allow_col_major=*/false, m->stream) == 0;
        mlx_array xs = mlx_array_new();
        { const int xs3[3] = {M, 1, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&xs, rnc, xs3, 3, m->stream) == 0; }
        /* NB: la fusione gate_up (LLMENGINE_MOE_GATEUP_FUSE) NON va usata qui: su V2-Lite misurata
         * -18% (la split produce viste non-contigue che degradano il gather successivo). */
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, xs, gp_w, gp_s, gp_b, no_lhs, idxs,
                                    true, xp_gsz, xp_bts, xp_mode, /*sorted=*/true, m->stream) == 0;
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, xs, up_w, up_s, up_b, no_lhs, idxs,
                                    true, xp_gsz, xp_bts, xp_mode, /*sorted=*/true, m->stream) == 0;
        if (ok && llm_mlx_fuse_swiglu_prefill_enabled()) {
            ok = llm_mlx_swiglu_compiled(gate_out, up_out, &act, m->stream);
        } else if (ok) {
            if (llm_env_isset("LLM_SIG_TRACE")) fprintf(stderr, "[SIG] layer=%d SORTED eager\n", layer_idx);
            ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0;
            if (ok) ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0;
            if (ok) ok = mlx_multiply(&act, silu, up_out, m->stream) == 0;
        }
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, dn_w, dn_s, dn_b, no_lhs, idxs,
                                    true, xp_gsz, xp_bts, xp_mode, /*sorted=*/true, m->stream) == 0;
        /* Q2-b: readback idxs (esperto per riga) + inv (riga sorted -> token-major), solo in
         * calibrazione (come qwen3_moe_ffn_block). Senza path sorted non si accumula. */
        if (ok && m->im.active) {
            const int Mi = n_tokens * m->cfg.top_k; im_rows = Mi;
            if (mlx_array_eval(idxs) == 0 && mlx_array_eval(inv) == 0 &&
                mlx_array_data_uint32(idxs) != NULL && mlx_array_data_uint32(inv) != NULL) {
                im_expert = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
                im_src = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
                if (im_expert && im_src) {
                    memcpy(im_expert, mlx_array_data_uint32(idxs), (size_t)Mi * sizeof(uint32_t));
                    memcpy(im_src, mlx_array_data_uint32(inv), (size_t)Mi * sizeof(uint32_t));
                }
            }
        }
        mlx_array invm = mlx_array_new();
        { const int is3[3] = {M, 1, 1}; if (ok) ok = mlx_reshape(&invm, inv, is3, 3, m->stream) == 0; }
        mlx_array duns = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&duns, down_out, invm, 0, m->stream) == 0;
        mlx_array dh3 = mlx_array_new();
        { const int d3[3] = {n_tokens, m->cfg.top_k, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&dh3, duns, d3, 3, m->stream) == 0; }
        mlx_array w3 = mlx_array_new();
        { const int w3s[3] = {n_tokens, m->cfg.top_k, 1};
          if (ok) ok = mlx_reshape(&w3, weights, w3s, 3, m->stream) == 0; }
        if (ok) ok = mlx_multiply(&scaled, dh3, w3, m->stream) == 0;
        if (ok) ok = mlx_sum_axis(&summed, scaled, 1, /*keepdims=*/false, m->stream) == 0;
        { const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&moe_out, summed, o3, 3, m->stream) == 0; }
        mlx_array_free(tt); mlx_array_free(order); mlx_array_free(inv); mlx_array_free(df);
        mlx_array_free(dv); mlx_array_free(idxs); mlx_array_free(dvm4);
        mlx_array_free(rn); mlx_array_free(rnc); mlx_array_free(xs);
        mlx_array_free(invm); mlx_array_free(duns); mlx_array_free(dh3); mlx_array_free(w3);
    } else {
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, x1, gp_w, gp_s, gp_b, no_lhs, top_inds_arr, true, xp_gsz, xp_bts, xp_mode, false, m->stream)==0;
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, x1, up_w, up_s, up_b, no_lhs, top_inds_arr, true, xp_gsz, xp_bts, xp_mode, false, m->stream)==0;
        if (ok) { if (llm_env_isset("LLM_SIG_TRACE")) fprintf(stderr, "[SIG] layer=%d ROUTED eager\n", layer_idx); ok = mlx_sigmoid(&sig, gate_out, m->stream)==0 && mlx_multiply(&silu, gate_out, sig, m->stream)==0 && mlx_multiply(&act, silu, up_out, m->stream)==0; }
        /* (2026-09-13) RIMOSSO qui il ramo compilato: questo e' il decode unsorted (n_tokens
         * piccolo, M<64) e mlx_closure_apply ha un costo fisso CPU per applicazione che a
         * n_tokens==1 non si ammortizza. A/B V2-Lite: eager 169,1 vs compiled 163,9 tok/s
         * (+3,2%), IDS greedy identici. La closure resta nel ramo sorted (prefill). */
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, dn_w, dn_s, dn_b, no_lhs, top_inds_arr, true, xp_gsz, xp_bts, xp_mode, false, m->stream)==0;
        if (ok) { const int ws[4]={n_tokens,m->cfg.top_k,1,1}; ok = mlx_reshape(&w_arr, weights, ws, 4, m->stream)==0; }
        if (ok) ok = mlx_multiply(&scaled, down_out, w_arr, m->stream)==0;
        if (ok) ok = mlx_sum_axis(&summed, scaled, 1, false, m->stream)==0;
        if (ok) { const int o3[3]={1,n_tokens,m->cfg.hidden_size}; ok = mlx_reshape(&moe_out, summed, o3, 3, m->stream)==0; }
    }
    DCHK(ok, "routed experts");

    if (ok && m->cfg.n_shared_experts > 0) {
        mlx_array sgw=mlx_array_new(),sgs=mlx_array_new(),sgb=mlx_array_new();
        mlx_array suw=mlx_array_new(),sus=mlx_array_new(),sub=mlx_array_new();
        mlx_array sdw=mlx_array_new(),sds=mlx_array_new(),sdb=mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_experts.gate_proj", namebuf, sizeof namebuf);
        ok = ok && multishard_get_triple(&m->weights, namebuf, &sgw, &sgs, &sgb);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_experts.up_proj", namebuf, sizeof namebuf);
        ok = ok && multishard_get_triple(&m->weights, namebuf, &suw, &sus, &sub);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_experts.down_proj", namebuf, sizeof namebuf);
        ok = ok && multishard_get_triple(&m->weights, namebuf, &sdw, &sds, &sdb);
        DCHK(ok, "shared lookup");
        mlx_array sgo=mlx_array_new(),suo=mlx_array_new(),ssig=mlx_array_new(),ssilu=mlx_array_new(),sact=mlx_array_new(),sd_out=mlx_array_new();
        if (ok) { ok = quantized_linear_cfg(&sgo, normed2, sgw,sgs,sgb, gsz,bts, m->cfg.quant_mode, m->stream); DCHK(ok, "shared gate"); }
        if (ok) { ok = quantized_linear_cfg(&suo, normed2, suw,sus,sub, gsz,bts, m->cfg.quant_mode, m->stream); DCHK(ok, "shared up"); }
        if (ok && n_tokens > 1 && llm_mlx_fuse_swiglu_prefill_enabled()) { ok = llm_mlx_swiglu_compiled(sgo, suo, &sact, m->stream); DCHK(ok, "shared swiglu compiled"); }
        else if (ok) { if (llm_env_isset("LLM_SIG_TRACE")) fprintf(stderr, "[SIG] layer=%d SHARED eager\n", layer_idx); ok = mlx_sigmoid(&ssig, sgo, m->stream)==0 && mlx_multiply(&ssilu, sgo, ssig, m->stream)==0 && mlx_multiply(&sact, ssilu, suo, m->stream)==0; DCHK(ok, "shared swiglu"); }
        if (ok) { ok = quantized_linear_cfg(&sd_out, sact, sdw,sds,sdb, gsz,bts, m->cfg.quant_mode, m->stream); DCHK(ok, "shared down"); }
        if (ok) { ok = mlx_add(&moe_out, moe_out, sd_out, m->stream)==0; DCHK(ok, "shared add"); }
        mlx_array sh[]={sgw,sgs,sgb,suw,sus,sub,sdw,sds,sdb,sgo,suo,ssig,ssilu,sact,sd_out};
        for (size_t i=0;i<sizeof sh/sizeof sh[0];i++) mlx_array_free(sh[i]);
    }
    DCHK(ok, "shared_experts");

    if (ok) ok = mlx_add(out, h, moe_out, m->stream)==0;
    DCHK(ok, "final residual");
    if (ok && layer_idx == 1 && n_tokens > 1) {
        mlx_array mf=mlx_array_new(), of=mlx_array_new();
        mlx_array_free(mf);
        mlx_array_free(of);
    }

    /* Q2-b: imatrix — energia di gate/up (input normed2) e down (riga SwiGLU). Prima dei free
     * di `act` (che e' nella lista `all[]`). */
    if (m->im.active && ok && im_expert && im_src && im_rows > 0 && act.ctx != NULL)
        llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, im_expert, im_src, im_rows);
    free(im_expert); free(im_src);
    mlx_array all[]={ln2_w,normed2,gw,gates,gates_f,scores,pruned,part,top_inds,weights,
                     gp_w,gp_s,gp_b,up_w,up_s,up_b,dn_w,dn_s,dn_b,no_lhs,top_inds_arr,x1,
                     gate_out,up_out,sig,silu,act,down_out,w_arr,scaled,summed,moe_out};
    for (size_t i=0;i<sizeof all/sizeof all[0];i++) mlx_array_free(all[i]);
#undef DCHK
    return ok;
}

bool glm4_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    /* Layer denso (primi first_k_dense_replace): riusa dense_swiglu_ffn */
    if (layer_idx < m->cfg.first_k_dense_replace)
        return dense_swiglu_ffn(m, layer_idx, n_tokens, h, out);

    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "glm4_moe_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln2_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }

    /* Router: mlp.gate.weight — GLM-4.5 lo ha in BF16 piatto (NON quantizzato); Kimi Linear
     * lo ha QUANTIZZATO (scales/biases, override per-tensore 8-bit). Triple + fallback a
     * plain_linear quando le scale sono assenti. */
    mlx_array gw = mlx_array_new(), gs = mlx_array_new(), gb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gw, &gs, &gb);
    CHK(ok, "router lookup");
    /* DIAG (LLM_ROUTER_TIME=1): tempo del router (gate matmul + group_expert_select) per layer.
     * Eval dei soli output del router: attribuzione, non misura di throughput. */
    mlx_array gates = mlx_array_new();
    if (ok) {
        if (gs.ctx != NULL && mlx_array_size(gs) > 0)
            ok = quantized_linear_cfg(&gates, normed2, gw, gs, gb, m->cfg.quant_group_size,
                                      m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
        else
            ok = plain_linear(&gates, normed2, gw, m->stream);
        CHK(ok, "router matmul");
    }
    mlx_array_free(gs); mlx_array_free(gb);
    mlx_array escb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.e_score_correction_bias", namebuf, sizeof(namebuf));
    if (!multishard_get(&m->weights, namebuf, &escb)) {
        /* GLM-4.5 lo mette sotto `mlp.gate.`; Kimi Linear a livello `mlp.` (stesso ruolo). */
        tname(m->cfg.layer_prefix, layer_idx, "mlp.gate.e_score_correction_bias", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &escb);
    }
    mlx_array gates_bf = mlx_array_new();
    mlx_array escb_use = mlx_array_new();
    /* glm4_moe (reference): sigmoid(gates.astype(f32)) + e_score_correction_bias f32.
     * kimi_linear (reference _group_expert_select): sigmoid(gates) nel dtype dei gates e
     * `bias.astype(scores.dtype)` -> il ROUTER resta bf16 (era il leak che rendeva f32
     * tutto l'hidden dal primo layer MoE in poi). */
    const bool kimi_router_bf16 = (strcmp(m->cfg.model_type, "kimi_linear") == 0);
    if (ok) {
        if (kimi_router_bf16) {
            ok = mlx_array_set(&gates_bf, gates) == 0;
            if (ok && escb.ctx != NULL)
                ok = (mlx_array_dtype(escb) == mlx_array_dtype(gates))
                         ? (mlx_array_set(&escb_use, escb) == 0)
                         : (mlx_astype(&escb_use, escb, mlx_array_dtype(gates), m->stream) == 0);
        } else {
            ok = mlx_astype(&gates_bf, gates, MLX_FLOAT32, m->stream) == 0;
            if (ok && escb.ctx != NULL) ok = mlx_array_set(&escb_use, escb) == 0;
        }
        CHK(ok, "router dtype");
    }
    /* FIX MoE f32 (LLMENGINE_MOE_BF16=1): se normed2 e' f32 (norm weight promosso a f32) ma l'hidden e'
     * bf16, il gather esperti gira in f32 e le scale vengono promosse per-op -> AsType + GatherQMM
     * f32 = ~47 ms su GLM (PEROP), il costo #1. Riporta l'input del MoE al dtype dell'hidden. */
    static int g_moe_bf16 = -1; if (g_moe_bf16 < 0) g_moe_bf16 = (llm_ai_mlx_opt_env("LLMENGINE_MOE_BF16") != NULL) ? 1 : 0;
    if (ok && g_moe_bf16 && mlx_array_dtype(normed2) == MLX_FLOAT32 && mlx_array_dtype(h) != MLX_FLOAT32) {
        mlx_array n2b = mlx_array_new();
        if (mlx_astype(&n2b, normed2, mlx_array_dtype(h), m->stream) == 0) mlx_array_set(&normed2, n2b);
        mlx_array_free(n2b);
    }
    /* Router Metal fuso (LLMMETAL_GLM_ROUTER_METAL=1): un dispatch invece di ~10 (sigmoid, add bias,
     * argpartition, slice, contiguous, take_along_axis, sum, divide, multiply, astype). Solo
     * n_group==1, gates f32 (GLM) e scaling 1.0. La strada closure/compile e' documentata
     * controproducente (geglu 14x) -> kernel Metal diretto. PARITA' GREEDY da verificare. OFF di default. */
    static int g_glm_router_metal_state = -1;
    if (g_glm_router_metal_state < 0) g_glm_router_metal_state = (llm_ai_mlx_opt_env("LLMMETAL_GLM_ROUTER_METAL") != NULL) ? 1 : 0;
    mlx_array top_inds = mlx_array_new(), weights = mlx_array_new();
    bool router_done = false;
    if (ok && g_glm_router_metal_state == 1 && m->cfg.n_group <= 1 && !kimi_router_bf16 &&
        escb_use.ctx != NULL && m->cfg.routed_scaling_factor == 1.0f && n_tokens <= 4096) {
        router_done = llm_mlx_router_metal(gates_bf, escb_use, m->cfg.top_k, &top_inds, &weights, m->stream);
    }
    /* Dichiarati FUORI dal ramo eager: la cleanup finale (array `all[]`) li libera anche nel
     * path Metal (dove restano array vuoti). */
    mlx_array scores = mlx_array_new(), biased = mlx_array_new(), pruned = mlx_array_new(), part = mlx_array_new();
    if (!router_done) {
    /* scores = sigmoid(gates) ; biased = scores + e_score_correction_bias ; orig_scores per il softmax */
    if (ok) { ok = mlx_sigmoid(&scores, gates_bf, m->stream) == 0; CHK(ok, "gate sigmoid"); }
    if (ok && escb_use.ctx != NULL) { ok = mlx_add(&biased, scores, escb_use, m->stream) == 0; CHK(ok, "gate bias"); }
    else if (ok) mlx_array_set(&biased, scores);

    /* group-pruning (copia 1:1 mlx_lm glm4_moe.py:141-171 group_expert_select):
     * su biased, se n_group>1 zero gli esperti dei (n_group-topk_group) gruppi con
     * group_score (=somma top2 del gruppo) piu' bassa. Tutto su device. */
    if (ok && m->cfg.n_group > 1 && m->cfg.n_experts > 0 &&
        m->cfg.n_experts % m->cfg.n_group == 0 && m->cfg.topk_group > 0 &&
        m->cfg.topk_group < m->cfg.n_group) {
        const int ng = m->cfg.n_group, epg = m->cfg.n_experts / ng;
        int us4[4] = {1, n_tokens, ng, epg};
        mlx_array bu = mlx_array_new();
        ok = mlx_reshape(&bu, biased, us4, 4, m->stream) == 0;
        CHK(ok, "group reshape");
        mlx_array gs2 = mlx_array_new();
        if (ok) {
            mlx_array gtop = mlx_array_new();
            ok = mlx_topk_axis(&gtop, bu, 2, -1, m->stream) == 0;
            if (ok) ok = mlx_sum_axis(&gs2, gtop, -1, /*keepdims=*/true, m->stream) == 0;
            mlx_array_free(gtop);
            CHK(ok, "group topk");
        }
        if (ok) {
            const int k = ng - m->cfg.topk_group;
            mlx_array gpart = mlx_array_new();
            ok = mlx_argpartition_axis(&gpart, gs2, k - 1, /*axis=*/2, m->stream) == 0;
            CHK(ok, "group argpartition");
            if (ok) {
                int st4[4] = {0, 0, 0, 0}, sp4[4] = {1, n_tokens, k, 1}, sv4[4] = {1, 1, 1, 1};
                mlx_array gview = mlx_array_new();
                ok = mlx_slice(&gview, gpart, st4, 4, sp4, 4, sv4, 4, m->stream) == 0;
                mlx_array gidx = mlx_array_new();
                if (ok) ok = mlx_contiguous(&gidx, gview, /*allow_col_major=*/false, m->stream) == 0;
                mlx_array_free(gview);
                if (ok) {
                    mlx_array zero = mlx_array_new_float32(0.0f);
                    mlx_array bu2 = mlx_array_new();
                    ok = mlx_put_along_axis(&bu2, bu, gidx, zero, /*axis=*/2, m->stream) == 0;
                    mlx_array_free(zero);
                    mlx_array_free(bu);
                    bu = bu2;
                }
                mlx_array_free(gidx);
            }
            mlx_array_free(gpart);
        }
        mlx_array_free(gs2);
        if (ok) {
            int fs3[3] = {1, n_tokens, m->cfg.n_experts};
            ok = mlx_reshape(&pruned, bu, fs3, 3, m->stream) == 0;
            CHK(ok, "group flatten");
        }
        mlx_array_free(bu);
    } else if (ok) {
        mlx_array_set(&pruned, biased);
    }

    /* top-k: argpartition su pruned (i top_k piu' GRANDI in coda), poi top-k valori da ORIG scores */
    if (ok) {
        ok = mlx_argpartition_axis(&part, pruned, /*kth=*/-m->cfg.top_k, /*axis=*/-1, m->stream) == 0;
        CHK(ok, "router argpartition");
        if (ok) {
            const int ne = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, ne - m->cfg.top_k};
            const int stop[3] = {1, n_tokens, ne};
            const int strides[3] = {1, 1, 1};
            mlx_array view = mlx_array_new();
            ok = mlx_slice(&view, part, start, 3, stop, 3, strides, 3, m->stream) == 0;
            CHK(ok, "top-k slice");
            if (ok) ok = mlx_contiguous(&top_inds, view, false, m->stream) == 0;
            mlx_array_free(view);
        }
    }
    /* pesi top-k PER-TOKEN (copia mlx_lm glm4_moe.py:161-169): take ORIG scores su
     * top_inds; se norm_topk_prob dividi per la somma per-token (keepdims); poi
     * routed_scaling_factor. Tutto su device, nessun readback host per layer. */
    /* weights gia' dichiarato sopra (router Metal / eager) */
    if (ok) {
        mlx_array tv = mlx_array_new();
        ok = mlx_take_along_axis(&tv, scores, top_inds, -1, m->stream) == 0;
        CHK(ok, "top-k values gather");
        if (ok && m->cfg.norm_topk_prob && m->cfg.top_k > 1) {
            mlx_array den = mlx_array_new(), tvn = mlx_array_new();
            ok = mlx_sum_axis(&den, tv, -1, /*keepdims=*/true, m->stream) == 0;
            if (ok) ok = mlx_divide(&tvn, tv, den, m->stream) == 0;
            mlx_array_free(den);
            if (ok) mlx_array_set(&weights, tvn);
            mlx_array_free(tvn);
            mlx_array_free(tv);
        } else {
            mlx_array_set(&weights, tv);
            mlx_array_free(tv);
        }
        if (ok && m->cfg.routed_scaling_factor != 1.0f) {
            mlx_array scf = mlx_array_new_float32(m->cfg.routed_scaling_factor), sc = mlx_array_new();
            if (ok) ok = mlx_astype(&sc, scf, mlx_array_dtype(weights), m->stream) == 0;
            mlx_array_free(scf);
            mlx_array wsc = mlx_array_new();
            ok = mlx_multiply(&wsc, weights, sc, m->stream) == 0;
            mlx_array_free(sc);
            if (ok) mlx_array_set(&weights, wsc);
            mlx_array_free(wsc);
        }
        CHK(ok, "top-k weights");
    }
    } /* fine eager router (!router_done) */

    /* top_inds per la gather: cast u32 + reshape device [n_tokens, top_k].
     * Host readback SOLO per l'imatrix collector (calibration, mai nel forward). */
    uint32_t* top_inds_host = NULL;
    if (ok && m->im.active) {
        if (mlx_array_eval(top_inds) == 0 && mlx_array_data_uint32(top_inds) != NULL) {
            size_t tn = (size_t)n_tokens * (size_t)m->cfg.top_k;
            top_inds_host = (uint32_t*)malloc(tn * sizeof(uint32_t));
            if (top_inds_host) memcpy(top_inds_host, mlx_array_data_uint32(top_inds), tn * sizeof(uint32_t));
        }
    }

    /* Esperti routed: mlp.switch_mlp.{gate,up,down}_proj 3D [128,1408,512], 4-bit g64 */
    mlx_array gp_w = mlx_array_new(), gp_s = mlx_array_new(), gp_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_s = mlx_array_new(), up_b = mlx_array_new();
    mlx_array dn_w = mlx_array_new(), dn_s = mlx_array_new(), dn_b = mlx_array_new();
    int xp_group = gsz, xp_bits = bts; char xp_mode[16]; /* xp_mode scritto sotto (fq->gmode o derive): niente copia iniziale */
    char nb1[256];
    /* Cache pesi esperti (P1/P3): router/shared_experts GLM restano legacy (naming diverso). */
    GmoeFfnQuant* fq = llm_mlx_moe_ffn_quant(m, layer_idx);
    if (fq != NULL) {
        ok = ok && mlx_array_set(&gp_w, fq->gw) == 0 && mlx_array_set_optional(&gp_s, fq->gs) &&
                   mlx_array_set_optional(&gp_b, fq->gb);
        ok = ok && mlx_array_set(&up_w, fq->uw) == 0 && mlx_array_set_optional(&up_s, fq->us) &&
                   mlx_array_set_optional(&up_b, fq->ub);
        ok = ok && mlx_array_set(&dn_w, fq->dw) == 0 && mlx_array_set_optional(&dn_s, fq->ds) &&
                   mlx_array_set_optional(&dn_b, fq->db);
        xp_group = fq->gg; xp_bits = fq->gbts; llm_mlx_mode_set(xp_mode, sizeof(xp_mode), fq->gmode);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj.weight", namebuf, sizeof(namebuf));
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj.scales", nb1, sizeof(nb1));
        ok = ok && multishard_get(&m->weights, namebuf, &gp_w) && multishard_get(&m->weights, nb1, &gp_s);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj.biases", nb1, sizeof(nb1));
        ok = ok && multishard_get(&m->weights, nb1, &gp_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj.weight", namebuf, sizeof(namebuf));
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj.scales", nb1, sizeof(nb1));
        ok = ok && multishard_get(&m->weights, namebuf, &up_w) && multishard_get(&m->weights, nb1, &up_s);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj.biases", nb1, sizeof(nb1));
        ok = ok && multishard_get(&m->weights, nb1, &up_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj.weight", namebuf, sizeof(namebuf));
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj.scales", nb1, sizeof(nb1));
        ok = ok && multishard_get(&m->weights, namebuf, &dn_w) && multishard_get(&m->weights, nb1, &dn_s);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj.biases", nb1, sizeof(nb1));
        ok = ok && multishard_get(&m->weights, nb1, &dn_b);
        llm_mlx_quant_derive(gp_w, gp_s, gsz, bts, "affine", &xp_group, &xp_bits, xp_mode, sizeof(xp_mode));
    }
    CHK(ok, "switch_mlp lookup");
    mlx_optional_int xp_gsz = {xp_group, true};
    mlx_optional_int xp_bts = {xp_bits, true};
    mlx_array no_lhs = mlx_array_new();
    mlx_array top_inds_arr = mlx_array_new();
    if (ok) {
        /* top_inds [1,n_tokens,top_k] (uint32) -> [n_tokens,top_k] device per gather_qmm */
        mlx_array tu = mlx_array_new();
        ok = mlx_astype(&tu, top_inds, MLX_UINT32, m->stream) == 0;
        if (ok) {
            const int ti_shape[2] = {n_tokens, m->cfg.top_k};
            mlx_array tv2 = mlx_array_new();
            ok = mlx_reshape(&tv2, tu, ti_shape, 2, m->stream) == 0;
            if (ok) ok = mlx_contiguous(&top_inds_arr, tv2, /*allow_col_major=*/false, m->stream) == 0;
            mlx_array_free(tv2);
        }
        mlx_array_free(tu);
        CHK(ok, "top_inds device reshape");
    }
    mlx_array x1 = mlx_array_new();
    if (ok) {
        const int x1_shape[4] = {n_tokens, 1, 1, m->cfg.hidden_size};
        ok = mlx_reshape(&x1, normed2, x1_shape, 4, m->stream) == 0;
        CHK(ok, "moe x1 reshape");
        /* FIX (LLMENGINE_MOE_BF16=1): se l'hidden e' f32 ma il peso e' bf16, il gather esperti gira in
         * f32 (costoso). Riporta x1 al dtype dei pesi (bf16). */
        /* DEFAULT ON (2026-09-23): l'hidden che entra nel MoE deve stare nel dtype del modello
         * (bf16), come mlx_lm. Senza, va in f32 (deviazione+3x piu' lento). Opt-out LLMENGINE_MOE_BF16=0. */
        static int g_moe_bf16 = -1; if (g_moe_bf16 < 0) { const char* e = llm_ai_mlx_opt_env("LLMENGINE_MOE_BF16"); g_moe_bf16 = !(e && strcmp(e, "0") == 0); }
        if (g_moe_bf16 && mlx_array_dtype(x1) == MLX_FLOAT32 && mlx_array_dtype(gp_w) != MLX_FLOAT32) {
            mlx_array x1b = mlx_array_new();
            if (mlx_astype(&x1b, x1, MLX_BFLOAT16, m->stream) == 0) mlx_array_set(&x1, x1b);
            mlx_array_free(x1b);
        }
        if (llm_ai_mlx_opt_env("LLMENGINE_MOE_DT")) fprintf(stderr, "[moe-dt] layer=%d h=%d normed2=%d x1=%d gp_w=%d gp_s=%d gp_b=%d (F32=%d BF16=%d)\n",
            layer_idx, (int)mlx_array_dtype(h), (int)mlx_array_dtype(normed2), (int)mlx_array_dtype(x1),
            (int)mlx_array_dtype(gp_w), (int)mlx_array_dtype(gp_s), (int)mlx_array_dtype(gp_b),
            (int)MLX_FLOAT32, (int)MLX_BFLOAT16);
    }
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    mlx_array down_out = mlx_array_new();
    mlx_array scaled = mlx_array_new();
    mlx_array summed = mlx_array_new();
    mlx_array moe_out = mlx_array_new();
    bool do_sort = ok && llm_glm_sorted_enabled() &&
                   ((int64_t)n_tokens * (int64_t)m->cfg.top_k) >= 64;
    if (ok && do_sort) {
        /* Path SORTED (1:1 mlx_lm switch_layers.SwitchGLU quando M>=64): argsort
         * degli esperti top-k, righe di x ordinate per esperto, gather_qmm con
         * sorted_indices=true (accesso per-esperto contiguo), poi unsort del
         * risultato. Stessa sequenza del branch sorted gia' verificato per
         * qwen3_moe (commit 3c667c9b); combine GLM invariata (group-pruning a
         * monte, rinorm, routed scaling, somma pesata, shared_experts). */
        const int M = n_tokens * m->cfg.top_k;
        mlx_array tt = mlx_array_new();
        { const int ts[1] = {M}; if (ok) ok = mlx_reshape(&tt, top_inds_arr, ts, 1, m->stream) == 0; }
        mlx_array order = mlx_array_new();
        if (ok) ok = mlx_argsort(&order, tt, m->stream) == 0;
        mlx_array inv = mlx_array_new();
        if (ok) ok = mlx_argsort(&inv, order, m->stream) == 0;
        mlx_array df = mlx_array_new();
        { mlx_array kk = mlx_array_new_int(m->cfg.top_k);
          if (ok) ok = mlx_floor_divide(&df, order, kk, m->stream) == 0; mlx_array_free(kk); }
        mlx_array dv = mlx_array_new();
        if (ok) ok = mlx_astype(&dv, df, MLX_UINT32, m->stream) == 0;
        mlx_array idxs = mlx_array_new();
        if (ok) ok = mlx_take(&idxs, tt, order, m->stream) == 0;
        mlx_array dvm4 = mlx_array_new();
        { const int ms[4] = {M, 1, 1, 1}; if (ok) ok = mlx_reshape(&dvm4, dv, ms, 4, m->stream) == 0; }
        mlx_array rn = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&rn, x1, dvm4, 0, m->stream) == 0;
        mlx_array rnc = mlx_array_new();
        if (ok) ok = mlx_contiguous(&rnc, rn, /*allow_col_major=*/false, m->stream) == 0;
        mlx_array xs = mlx_array_new();
        { const int xs3[3] = {M, 1, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&xs, rnc, xs3, 3, m->stream) == 0; }
        if (ok && !llm_mlx_moe_gateup_gather(fq, xs, no_lhs, idxs, /*sorted=*/true, xp_gsz, xp_bts,
                                             xp_mode, &gate_out, &up_out, m->stream)) {
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, xs, gp_w, gp_s, gp_b, no_lhs, idxs,
                                        true, xp_gsz, xp_bts, xp_mode, /*sorted=*/true, m->stream) == 0;
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, xs, up_w, up_s, up_b, no_lhs, idxs,
                                        true, xp_gsz, xp_bts, xp_mode, /*sorted=*/true, m->stream) == 0;
        }
        CHK(ok, "sorted gather_qmm gate/up");
        if (ok) ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0;
        if (ok) ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0;
        if (ok) ok = mlx_multiply(&act, silu, up_out, m->stream) == 0;
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, dn_w, dn_s, dn_b, no_lhs, idxs,
                                    true, xp_gsz, xp_bts, xp_mode, /*sorted=*/true, m->stream) == 0;
        CHK(ok, "sorted gather_qmm down");
        mlx_array invm = mlx_array_new();
        { const int is3[3] = {M, 1, 1}; if (ok) ok = mlx_reshape(&invm, inv, is3, 3, m->stream) == 0; }
        mlx_array duns = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&duns, down_out, invm, 0, m->stream) == 0;
        /* act per l'imatrix (calibrazione): serve in ordine naturale [T,K,1,M];
         * nel forward normale si unsorta solo down_out. */
        if (ok && m->im.active) {
            mlx_array auns = mlx_array_new();
            if (ok) ok = mlx_take_along_axis(&auns, act, invm, 0, m->stream) == 0;
            mlx_array a3 = mlx_array_new();
            { const int a3s[4] = {n_tokens, m->cfg.top_k, 1, m->cfg.moe_intermediate_size};
              if (ok) ok = mlx_reshape(&a3, auns, a3s, 4, m->stream) == 0; }
            mlx_array_free(auns);
            if (ok) mlx_array_set(&act, a3); else mlx_array_free(a3);
        }
        mlx_array_free(invm);
        mlx_array dh3 = mlx_array_new();
        { const int d3[3] = {n_tokens, m->cfg.top_k, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&dh3, duns, d3, 3, m->stream) == 0; }
        mlx_array w3 = mlx_array_new();
        { const int w3s[3] = {n_tokens, m->cfg.top_k, 1};
          if (ok) ok = mlx_reshape(&w3, weights, w3s, 3, m->stream) == 0; }
        if (ok) ok = mlx_multiply(&scaled, dh3, w3, m->stream) == 0;
        if (ok) ok = mlx_sum_axis(&summed, scaled, 1, /*keepdims=*/false, m->stream) == 0;
        { const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&moe_out, summed, o3, 3, m->stream) == 0; }
        CHK(ok, "sorted moe combine");
        mlx_array_free(tt); mlx_array_free(order); mlx_array_free(inv); mlx_array_free(df);
        mlx_array_free(dv); mlx_array_free(idxs); mlx_array_free(dvm4);
        mlx_array_free(rn); mlx_array_free(rnc); mlx_array_free(xs);
        mlx_array_free(duns); mlx_array_free(dh3); mlx_array_free(w3);
    } else
    {
    /* Path legacy UNSORTED (dispatch per (token,slot), gather_qmm non ordinato).
     * E' il percorso pre-sorted (parita' storica vs mlx_lm). */
    if (ok && !llm_mlx_moe_gateup_gather(fq, x1, no_lhs, top_inds_arr, /*sorted=*/false, xp_gsz, xp_bts,
                                         xp_mode, &gate_out, &up_out, m->stream)) {
    if (ok) {
        ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, x1, gp_w, gp_s, gp_b, no_lhs, top_inds_arr,
                             true, xp_gsz, xp_bts, xp_mode, false, m->stream) == 0;
        CHK(ok, "gather_qmm gate_proj");
    }
    if (ok) {
        ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, x1, up_w, up_s, up_b, no_lhs, top_inds_arr,
                             true, xp_gsz, xp_bts, xp_mode, false, m->stream) == 0;
        CHK(ok, "gather_qmm up_proj");
    }
    }
    /* SwiGLU silu(gate)*up */
    if (ok) { ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0 && mlx_multiply(&silu, gate_out, sig, m->stream) == 0 && mlx_multiply(&act, silu, up_out, m->stream) == 0; CHK(ok, "swiglu"); }
    if (ok) {
        ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, dn_w, dn_s, dn_b, no_lhs, top_inds_arr,
                             true, xp_gsz, xp_bts, xp_mode, false, m->stream) == 0;
        CHK(ok, "gather_qmm down_proj");
    }
    /* somma pesata top-k */
    if (ok) {
        const int ws_shape[4] = {n_tokens, m->cfg.top_k, 1, 1};
        mlx_array w_arr = mlx_array_new();
        ok = mlx_reshape(&w_arr, weights, ws_shape, 4, m->stream) == 0;
        if (ok) ok = mlx_multiply(&scaled, down_out, w_arr, m->stream) == 0;
        mlx_array_free(w_arr);
        CHK(ok, "moe score scale");
    }
    if (ok) { ok = mlx_sum_axis(&summed, scaled, 1, false, m->stream) == 0; CHK(ok, "moe sum over top_k"); }
    if (ok) {
        const int out_shape[3] = {1, n_tokens, m->cfg.hidden_size};
        ok = mlx_reshape(&moe_out, summed, out_shape, 3, m->stream) == 0;
        CHK(ok, "moe_out reshape");
    }
    }

    /* shared_experts: MLP denso SwiGLU, intermediate = moe_intermediate * n_shared */
    if (ok && m->cfg.n_shared_experts > 0) {
        mlx_array sgw = mlx_array_new(), sgs = mlx_array_new(), sgb = mlx_array_new();
        mlx_array suw = mlx_array_new(), sus = mlx_array_new(), sub = mlx_array_new();
        mlx_array sdw = mlx_array_new(), sds = mlx_array_new(), sdb = mlx_array_new();
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_experts.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &sgw, &sgs, &sgb);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_experts.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &suw, &sus, &sub);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_experts.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &sdw, &sds, &sdb);
        CHK(ok, "shared_experts lookup");
        mlx_array sgo = mlx_array_new(), suo = mlx_array_new();
        if (ok) { ok = quantized_linear_cfg(&sgo, normed2, sgw, sgs, sgb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "shared gate"); }
        if (ok) { ok = quantized_linear_cfg(&suo, normed2, suw, sus, sub, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "shared up"); }
        mlx_array ssig = mlx_array_new(), ssilu = mlx_array_new(), sact = mlx_array_new();
        if (ok) { ok = mlx_sigmoid(&ssig, sgo, m->stream) == 0 && mlx_multiply(&ssilu, sgo, ssig, m->stream) == 0 && mlx_multiply(&sact, ssilu, suo, m->stream) == 0; CHK(ok, "shared swiglu"); }
        mlx_array sd_out = mlx_array_new();
        if (ok) { ok = quantized_linear_cfg(&sd_out, sact, sdw, sds, sdb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "shared down"); }
        if (ok) { ok = mlx_add(&moe_out, moe_out, sd_out, m->stream) == 0; CHK(ok, "shared add"); }
        mlx_array all_sh[] = {sgw,sgs,sgb,suw,sus,sub,sdw,sds,sdb,sgo,suo,ssig,ssilu,sact,sd_out};
        for (size_t i = 0; i < sizeof(all_sh)/sizeof(all_sh[0]); i++) mlx_array_free(all_sh[i]);
    }

    if (ok) { ok = mlx_add(out, h, moe_out, m->stream) == 0; CHK(ok, "final residual add"); }

    /* Imatrix collector (calibration): accumula l'energia per-(layer, expert routato, colonna)
     * sull'input FFN-normato (gate/up) e sulla riga SwiGLU (down). Solo se attivo via API. */
    if (m->im.active) llm_mlx_imatrix_accumulate(m, layer_idx, normed2, act, top_inds_host, n_tokens);

    free(top_inds_host);
    mlx_array all[] = {
        ln2_w, normed2, gw, gates, escb, gates_bf, scores, biased, pruned, part, top_inds,
        weights, no_lhs, top_inds_arr, x1, gate_out, up_out,
        sig, silu, act, down_out, scaled, summed, moe_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
