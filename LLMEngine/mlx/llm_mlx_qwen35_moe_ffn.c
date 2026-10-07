/* ============================================================================
 * Dominio MoE FFN Qwen3.5/Next + Qwen4-Exp: qwen3_5_moe_ffn_block (MoE + shared_expert), qwen4_exp_moe_block (hyper-connection, senza norm/residuo interno).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_swiglu_kernel.h"
#include "llm_mlx_router_kernel.h"
#include "llm_mlx_moe_fused_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
bool qwen3_5_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen3_5_moe_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    /* Cache pesi per-layer (P1/P3): norm e router risolti una volta al load; fallback legacy. */
    GmoeFfnQuant* fq = llm_mlx_moe_ffn_quant(m, layer_idx);
    mlx_array ln2_w = mlx_array_new();
    if (fq != NULL) {
        ok = ok && mlx_array_set(&ln2_w, fq->ln2) == 0;
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
    }
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }

    /* ── router MoE (gate) + switch_mlp (esperti) ── (stesso pattern qwen3_moe_ffn_block) */
    mlx_array gate_w = mlx_array_new(), gate_s = mlx_array_new(), gate_b = mlx_array_new();
    if (fq != NULL) {
        ok = ok && mlx_array_set(&gate_w, fq->rw) == 0 && mlx_array_set_optional(&gate_s, fq->rs) &&
                   mlx_array_set_optional(&gate_b, fq->rb);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "mlp.gate", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_w, &gate_s, &gate_b);
    }
    mlx_array gates = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gates, normed2, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "router matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "mlp.gate"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&gates, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array gates_sm = mlx_array_new();
    /* bfloat16 refactor: mlx_lm calcola i logit del router in bfloat16 (input bf16). Il softmax
     * precise in float32 sui logit float32 amplifica la differenza di selezione esperti vs la
     * reference (root cause della divergenza MoE Qwen3.6-35B-A3B, §9). Se bfloat16_input e' attivo,
     * abbassa i logit a bf16 PRIMA del softmax per allineare la selezione. */
    if (ok && m->cfg.bfloat16_input)
        ok = (mlx_array_dtype(gates) == g_act16) || (mlx_astype(&gates, gates, g_act16, m->stream) == 0);
    if (ok) { ok = mlx_softmax_axis(&gates_sm, gates, -1, /*precise=*/true, m->stream) == 0; CHK(ok, "router softmax"); }

    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, gates_sm, /*kth=*/-m->cfg.top_k, /*axis=*/-1, m->stream) == 0;
        if (ok) {
            const int num_experts = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, num_experts - m->cfg.top_k};
            const int stop[3] = {1, n_tokens, num_experts};
            const int strides[3] = {1, 1, 1};
            mlx_array top_inds_view = mlx_array_new();
            ok = mlx_slice(&top_inds_view, part, start, 3, stop, 3, strides, 3, m->stream) == 0;
            if (ok) ok = mlx_contiguous(&top_inds, top_inds_view, /*allow_col_major=*/false, m->stream) == 0;
            mlx_array_free(top_inds_view);
        }
    }
    /* Leva A: router e pesi softmax restano su device (niente readback host per layer).
     * top_inds_arr = top_inds lazy reshape [T,K] astype uint32; probs = take_along. */
    mlx_array gate_proj_w = mlx_array_new(), gate_proj_s = mlx_array_new(), gate_proj_b = mlx_array_new();
    mlx_array up_proj_w = mlx_array_new(), up_proj_s = mlx_array_new(), up_proj_b = mlx_array_new();
    mlx_array down_proj_w = mlx_array_new(), down_proj_s = mlx_array_new(), down_proj_b = mlx_array_new();
    int eg_group = m->cfg.quant_group_size, eg_bits = m->cfg.quant_bits; char eg_mode[16];
    int eu_group = m->cfg.quant_group_size, eu_bits = m->cfg.quant_bits; char eu_mode[16];
    int ed_group = m->cfg.quant_group_size, ed_bits = m->cfg.quant_bits; char ed_mode[16];
    /* eg/eu/ed_mode sono SEMPRE scritti sotto (fq->*mode nel ramo cache, oppure
     * llm_mlx_quant_derive che azzera e riscrive out_mode): il vecchio
     * snprintf(..., m->cfg.quant_mode) era DEAD CODE e pagava __vfprintf a ogni layer. */
    /* Bug P1/P3: triple esperti + quant per-tensore (fq risolta sopra; alias refcount sicuri). */
    if (fq != NULL) {
        /* handle LOCALI indipendenti (mai alias di struct: la free a fine funzione decrementa
         * il refcount; la cache tiene i propri riferimenti vivi). */
        ok = ok && mlx_array_set(&gate_proj_w, fq->gw) == 0 && mlx_array_set_optional(&gate_proj_s, fq->gs) &&
                   mlx_array_set_optional(&gate_proj_b, fq->gb);
        ok = ok && mlx_array_set(&up_proj_w, fq->uw) == 0 && mlx_array_set_optional(&up_proj_s, fq->us) &&
                   mlx_array_set_optional(&up_proj_b, fq->ub);
        ok = ok && mlx_array_set(&down_proj_w, fq->dw) == 0 && mlx_array_set_optional(&down_proj_s, fq->ds) &&
                   mlx_array_set_optional(&down_proj_b, fq->db);
        eg_group = fq->gg; eg_bits = fq->gbts; llm_mlx_mode_set(eg_mode, sizeof(eg_mode), fq->gmode);
        eu_group = fq->ug; eu_bits = fq->ubts; llm_mlx_mode_set(eu_mode, sizeof(eu_mode), fq->umode);
        ed_group = fq->dg; ed_bits = fq->dbts; llm_mlx_mode_set(ed_mode, sizeof(ed_mode), fq->dmode);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_proj_w, &gate_proj_s, &gate_proj_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_proj_w, &up_proj_s, &up_proj_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &down_proj_w, &down_proj_s, &down_proj_b);
        /* §quant per-tensore esperti: gli esperti possono avere bits/mode diverso dal globale
         * (Qwen3.6-122B: switch_mlp mxfp4 32/4 ratio 4, SENZA biases; Qwen3.6: switch_mlp
         * affine bits4). llm_mlx_quant_derive deriva (group,bits,mode) dagli esperti stacked
         * [n_exp, out, wcols] / scales [n_exp, out, scols] (shape[-1] di ciascuno). */
        llm_mlx_quant_derive(gate_proj_w, gate_proj_s, eg_group, eg_bits, "affine", &eg_group, &eg_bits, eg_mode, sizeof(eg_mode));
        llm_mlx_quant_derive(up_proj_w, up_proj_s, eu_group, eu_bits, "affine", &eu_group, &eu_bits, eu_mode, sizeof(eu_mode));
        llm_mlx_quant_derive(down_proj_w, down_proj_s, ed_group, ed_bits, "affine", &ed_group, &ed_bits, ed_mode, sizeof(ed_mode));
    }
    mlx_array x1 = mlx_array_new();
    mlx_array moe_out = mlx_array_new();
    mlx_array no_lhs = mlx_array_new();
    mlx_array top_inds_arr = mlx_array_new();
    uint32_t* im_expert = NULL; uint32_t* im_src = NULL; int im_rows = 0; /* §E2 imatrix collector */
    if (ok) {
        const int ti2_shape[2] = {n_tokens, m->cfg.top_k};
        mlx_array tt = mlx_array_new();
        ok = mlx_reshape(&tt, top_inds, ti2_shape, 2, m->stream) == 0;
        if (ok) ok = mlx_astype(&top_inds_arr, tt, MLX_UINT32, m->stream) == 0;
        mlx_array_free(tt);
        CHK(ok, "top_inds device reshape/astype");
    }
    if (ok) {
        const int x1_shape[4] = {n_tokens, 1, 1, m->cfg.hidden_size};
        ok = mlx_reshape(&x1, normed2, x1_shape, 4, m->stream) == 0;
    }
    bool has_moe_down_lora = false;
    for (int li = 0; li < m->lora_n && !has_moe_down_lora; li++)
        if (m->lora_layer[li] == layer_idx && m->lora_proj[li] && strcmp(m->lora_proj[li], "moe_down") == 0 && m->lora_expert[li] >= 0)
            has_moe_down_lora = true;
    /* Leva f16-attivazione mxfp4 pura (LLM_MXFP4_ACT_F16, default OFF; vedi helper in testa):
     * casta la lhs delle gather ESPERTI a f16 quando i pesi sono mxfp4 4/g32 E l'attivazione
     * arriva fp32 (want_bf16 falso -> rms_norm non riduce). Copre il percorso fp32 di
     * 122B-A10B (model_type qwen3_5_moe) e Coder-Next (qwen3_next, STESSO ffn block). gu =
     * gate/up (lhs xs nel ramo sorted, x1 nel ramo decode), dn = down (lhs act). Fallback
     * silenzioso a fp32 se l'astype fallisce. NON tocca il denso/shared_expert. */
    const bool mxfp4_f32act = llm_mlx_mxfp4_act_f16_enabled()
        && mlx_array_dtype(normed2) == MLX_FLOAT32;
    const bool mxfp4_gu = mxfp4_f32act
        && ((strcmp(eg_mode, "mxfp4") == 0 && eg_bits == 4 && eg_group == 32)
         || (strcmp(eu_mode, "mxfp4") == 0 && eu_bits == 4 && eu_group == 32));
    const bool mxfp4_dn = mxfp4_f32act
        && strcmp(ed_mode, "mxfp4") == 0 && ed_bits == 4 && ed_group == 32;
    bool do_sort = ok && ((int64_t)n_tokens * (int64_t)m->cfg.top_k) >= 64 && !has_moe_down_lora;
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    mlx_array down_out = mlx_array_new();
    mlx_array probs = mlx_array_new();
    mlx_array scores_arr = mlx_array_new(), scaled = mlx_array_new();
    mlx_array summed = mlx_array_new();
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
        mlx_array dvm = mlx_array_new();
        { const int ms[3] = {1, M, 1}; if (ok) ok = mlx_reshape(&dvm, dv, ms, 3, m->stream) == 0; }
        mlx_array rn = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&rn, normed2, dvm, 1, m->stream) == 0;
        mlx_array rnc = mlx_array_new();
        if (ok) ok = mlx_contiguous(&rnc, rn, false, m->stream) == 0;
        mlx_array xs = mlx_array_new();
        { const int xs3[3] = {M, 1, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&xs, rnc, xs3, 3, m->stream) == 0; }
        if (mxfp4_gu) llm_mlx_cast_f16_inplace(&xs, m->stream);
        /* NB (todo MLX-Decode-Composizione §7): la fusione gate+up NON è cablata qui di
         * proposito: (a) il prepare condiviso `llm_mlx_moe_ffn_quant` richiede
         * `post_attention_layernorm.weight` (assente in qwen4_exp: il blocco usa l'HC mixer),
         * quindi la cache non si costruirebbe; (b) il costo memoria della concat su 512 esperti
         * × 48 layer è stimato ~40 GB, non giustificato sul modello da 97 GB. Se servirà, va
         * fatto un prepare dedicato (senza ln2) + un budget di memoria. */
        /* LLMENGINE_MOE_GATEUP_FUSE (todo MLX-Decode-Composizione §7): un gather_qmm sul peso
         * gate+up fuso (preparato al load) invece di due; fallback esatto se il flag e' OFF
         * o la fusione non e' disponibile. */
        if (ok && !llm_mlx_moe_gateup_gather(fq, xs, no_lhs, idxs, /*sorted=*/true,
                                             (mlx_optional_int){eg_group,true}, (mlx_optional_int){eg_bits,true},
                                             eg_mode, &gate_out, &up_out, m->stream)) {
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, xs, gate_proj_w, gate_proj_s, gate_proj_b, no_lhs, idxs,
                                        true, (mlx_optional_int){eg_group,true}, (mlx_optional_int){eg_bits,true}, eg_mode, true, m->stream) == 0;
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, xs, up_proj_w, up_proj_s, up_proj_b, no_lhs, idxs,
                                        true, (mlx_optional_int){eu_group,true}, (mlx_optional_int){eu_bits,true}, eu_mode, true, m->stream) == 0;
        }
        if (ok) ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0;
        if (ok) ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0;
        if (ok) ok = mlx_multiply(&act, silu, up_out, m->stream) == 0;
        if (mxfp4_dn) llm_mlx_cast_f16_inplace(&act, m->stream);
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b, no_lhs, idxs,
                                    true, (mlx_optional_int){ed_group,true}, (mlx_optional_int){ed_bits,true}, ed_mode, /*sorted=*/true, m->stream) == 0;
        /* §E2 imatrix collector (path SORTED): idxs (esperto per riga) + inv (riga sorted -> token-major) */
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
        mlx_array duns = mlx_array_new();
        mlx_array invm = mlx_array_new();
        { const int is3[3] = {M, 1, 1}; if (ok) ok = mlx_reshape(&invm, inv, is3, 3, m->stream) == 0; }
        if (ok) ok = mlx_take_along_axis(&duns, down_out, invm, 0, m->stream) == 0;
        mlx_array_free(invm);
        mlx_array dh3 = mlx_array_new();
        { const int d3[3] = {n_tokens, m->cfg.top_k, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&dh3, duns, d3, 3, m->stream) == 0; }
        if (ok) ok = mlx_take_along_axis(&probs, gates_sm, top_inds, -1, m->stream) == 0;
        if (ok && m->cfg.norm_topk_prob) {
            mlx_array psum = mlx_array_new();
            if (ok) ok = mlx_sum_axis(&psum, probs, 2, true, m->stream) == 0;
            if (ok) ok = mlx_divide(&probs, probs, psum, m->stream) == 0;
            mlx_array_free(psum);
        }
        {
            mlx_array pr = mlx_array_new();
            const int pr2[2] = {n_tokens, m->cfg.top_k};
            if (ok) ok = mlx_reshape(&pr, probs, pr2, 2, m->stream) == 0;
            mlx_array sc3 = mlx_array_new();
            const int sc3s[3] = {n_tokens, m->cfg.top_k, 1};
            if (ok) ok = mlx_reshape(&sc3, pr, sc3s, 3, m->stream) == 0;
            mlx_array scaled3 = mlx_array_new();
            if (ok) ok = mlx_multiply(&scaled3, dh3, sc3, m->stream) == 0;
            mlx_array sumt = mlx_array_new();
            if (ok) ok = mlx_sum_axis(&sumt, scaled3, 1, false, m->stream) == 0;
            const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
            if (ok) ok = mlx_reshape(&moe_out, sumt, o3, 3, m->stream) == 0;
            mlx_array_free(pr); mlx_array_free(sc3); mlx_array_free(scaled3); mlx_array_free(sumt);
        }
        mlx_array_free(tt); mlx_array_free(order); mlx_array_free(inv); mlx_array_free(idxs); mlx_array_free(dvm);
        mlx_array_free(rn); mlx_array_free(rnc); mlx_array_free(xs); mlx_array_free(duns);
        mlx_array_free(dh3);
        if (!ok) fprintf(stderr, "qwen3_5_moe_ffn[layer=%d]: sorted MoE branch failed\n", layer_idx);
    } else
    {
    /* Score per (token,esperto), spostato qui (prima serviva solo dopo down_out) perche' il
     * kernel fuso sotto ne ha bisogno subito -- stesso schema di qwen3_moe_ffn_block. */
    if (ok) ok = mlx_take_along_axis(&probs, gates_sm, top_inds, /*axis=*/-1, m->stream) == 0;
    if (ok && m->cfg.norm_topk_prob) {
        mlx_array psum = mlx_array_new();
        if (ok) ok = mlx_sum_axis(&psum, probs, /*axis=*/2, /*keepdims=*/true, m->stream) == 0;
        if (ok) ok = mlx_divide(&probs, probs, psum, m->stream) == 0;
        mlx_array_free(psum);
    }

    /* Kernel Metal MoE decode fuso (LLMENGINE_MOE_FUSE_MOE_DECODE, sperimentale, default OFF) --
     * stesso kernel di qwen3_moe_ffn_block, qui con bits/group_size di gate/up e down
     * eventualmente DIVERSI (quant per-tensore, eg/eu/ed_*) e requisito aggiuntivo: solo
     * modalita' "affine" (mxfp4/altro non supportati da questo kernel), gate e up con
     * bits/group condivisi (kernel A li processa insieme), niente LoRA moe_down attivo
     * (il kernel fuso non espone un `act`/`down_out` intermedio su cui il LoRA per-esperto
     * possa agganciarsi). Fallback silenzioso al path gather_qmm sotto in ogni altro caso. */
    bool moe_fused_done = false;
    /* STREAM-AWARE: nessuna guardia per-modello. Il kernel fuso riceve i tensori giusti in
     * entrambe le modalità (completi se OFF, stacked degli esperti attivi se ON) tramite
     * `llm_mlx_moe_fused_stream_aware`. */
    if (ok && n_tokens == 1 && !has_moe_down_lora && llm_mlx_fuse_moe_decode_enabled()
        && strcmp(m->cfg.model_type, "qwen3_5_moe") != 0  /* NON bit-fedele per qwen3_5_moe (bf16): il kernel accumula in f32 mentre eager/gather (e mlx_lm) arrotondano a bf16 -> diff ~0.5% che flipa i near-tie. Bug: Bugs/(open) - MLX-Qwen3.5-MoE-Parita-Greedy-Diverge.md */
        && strcmp(eg_mode, "affine") == 0 && strcmp(eu_mode, "affine") == 0 && strcmp(ed_mode, "affine") == 0
        && eg_bits == eu_bits && eg_group == eu_group
        && (eg_bits == 4 || eg_bits == 8) && (ed_bits == 4 || ed_bits == 8)) {
        const int din = m->cfg.hidden_size, dff = m->cfg.moe_intermediate_size;
        const int k = m->cfg.top_k;
        mlx_array x_flat = mlx_array_new(), idx_flat = mlx_array_new();
        mlx_array pr1 = mlx_array_new(), sc_flat = mlx_array_new();
        const int xsh[1] = {din};
        bool prep = mlx_reshape(&x_flat, x1, xsh, 1, m->stream) == 0;
        const int ksh[1] = {k};
        prep = prep && mlx_reshape(&idx_flat, top_inds_arr, ksh, 1, m->stream) == 0;
        prep = prep && mlx_reshape(&pr1, probs, ksh, 1, m->stream) == 0;
        prep = prep && mlx_astype(&sc_flat, pr1, MLX_FLOAT32, m->stream) == 0;
        if (prep) {
            mlx_array fused_summed = mlx_array_new();
            bool fok = llm_mlx_moe_fused_stream_aware(&m->cfg, layer_idx, x_flat, top_inds_arr,
                gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
                down_proj_w, down_proj_s, down_proj_b,
                eg_bits, eg_group, ed_bits, ed_group, /*act=SwiGLU*/0, sc_flat, &fused_summed, m->stream);
            if (fok) {
                mlx_array cast = mlx_array_new();
                fok = mlx_astype(&cast, fused_summed, mlx_array_dtype(h), m->stream) == 0;
                if (fok) {
                    const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
                    fok = mlx_reshape(&moe_out, cast, o3, 3, m->stream) == 0;
                }
                mlx_array_free(cast);
            }
            if (fok && llm_mlx_fuse_moe_decode_verify_enabled() && !llm_mlx_moe_swap_on()) {
                /* _VERIFY: non usato per calcolare il risultato, solo per stampare il diff --
                 * zero rischio. Stesso schema di qwen3_moe_ffn_block, con bits/group per
                 * proiezione (eg/eu/ed_*) invece del gsz/bts globale. */
                mlx_array ref_gate = mlx_array_new(), ref_up = mlx_array_new(), ref_act = mlx_array_new();
                mlx_array ref_down = mlx_array_new(), ref_sig = mlx_array_new();
                bool rok = mlx_gather_qmm(&ref_gate, x1, gate_proj_w, gate_proj_s, gate_proj_b,
                        no_lhs, top_inds_arr, true, (mlx_optional_int){eg_group,true}, (mlx_optional_int){eg_bits,true}, eg_mode, false, m->stream) == 0 &&
                    mlx_gather_qmm(&ref_up, x1, up_proj_w, up_proj_s, up_proj_b,
                        no_lhs, top_inds_arr, true, (mlx_optional_int){eu_group,true}, (mlx_optional_int){eu_bits,true}, eu_mode, false, m->stream) == 0;
                if (rok) rok = mlx_sigmoid(&ref_sig, ref_gate, m->stream) == 0;
                mlx_array ref_silu = mlx_array_new();
                if (rok) rok = mlx_multiply(&ref_silu, ref_gate, ref_sig, m->stream) == 0;
                if (rok) rok = mlx_multiply(&ref_act, ref_silu, ref_up, m->stream) == 0;
                if (rok) rok = mlx_gather_qmm(&ref_down, ref_act, down_proj_w, down_proj_s, down_proj_b,
                        no_lhs, top_inds_arr, true, (mlx_optional_int){ed_group,true}, (mlx_optional_int){ed_bits,true}, ed_mode, false, m->stream) == 0;
                mlx_array ref_scaled = mlx_array_new(), ref_scores4 = mlx_array_new(), ref_summed = mlx_array_new();
                if (rok) { const int sc4[4] = {1, k, 1, 1}; rok = mlx_reshape(&ref_scores4, pr1, sc4, 4, m->stream) == 0; }
                if (rok) rok = mlx_multiply(&ref_scaled, ref_down, ref_scores4, m->stream) == 0;
                if (rok) rok = mlx_sum_axis(&ref_summed, ref_scaled, 1, false, m->stream) == 0;
                if (rok) {
                    mlx_array af = mlx_array_new(), bf = mlx_array_new(), ref_summed_flat = mlx_array_new();
                    const int rsh[1] = {din};
                    if (mlx_reshape(&ref_summed_flat, ref_summed, rsh, 1, m->stream) == 0 &&
                        mlx_astype(&af, fused_summed, MLX_FLOAT32, m->stream) == 0 &&
                        mlx_astype(&bf, ref_summed_flat, MLX_FLOAT32, m->stream) == 0 &&
                        mlx_array_eval(af) == 0 && mlx_array_eval(bf) == 0) {
                        const float* ad = mlx_array_data_float32(af);
                        const float* bd = mlx_array_data_float32(bf);
                        if (ad && bd) {
                            double maxdiff = 0, sumsq = 0, refsq = 0;
                            for (int vi = 0; vi < din; vi++) {
                                double d = fabs((double)ad[vi] - (double)bd[vi]);
                                if (d > maxdiff) maxdiff = d;
                                sumsq += d * d; refsq += (double)bd[vi] * (double)bd[vi];
                            }
                            double rms = refsq > 0 ? sqrt(sumsq / refsq) : sqrt(sumsq);
                            fprintf(stderr, "[fuse-moe-verify] qwen3_5_moe layer=%d max_abs_diff=%.6g rel_rms=%.6g\n",
                                    layer_idx, maxdiff, rms);
                        }
                    }
                    mlx_array_free(af); mlx_array_free(bf); mlx_array_free(ref_summed_flat);
                }
                mlx_array_free(ref_gate); mlx_array_free(ref_up); mlx_array_free(ref_act);
                mlx_array_free(ref_down); mlx_array_free(ref_sig); mlx_array_free(ref_silu);
                mlx_array_free(ref_scaled); mlx_array_free(ref_scores4); mlx_array_free(ref_summed);
            }
            moe_fused_done = fok;
            mlx_array_free(fused_summed);
        }
        mlx_array_free(x_flat); mlx_array_free(idx_flat); mlx_array_free(pr1); mlx_array_free(sc_flat);
    }

    if (mxfp4_gu) llm_mlx_cast_f16_inplace(&x1, m->stream);
    if (ok && !moe_fused_done && !llm_mlx_moe_gateup_gather(fq, x1, no_lhs, top_inds_arr, /*sorted=*/false,
                                                            (mlx_optional_int){eg_group,true}, (mlx_optional_int){eg_bits,true},
                                                            eg_mode, &gate_out, &up_out, m->stream)) {
        if (ok && !moe_fused_done) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, x1, gate_proj_w, gate_proj_s, gate_proj_b, no_lhs, top_inds_arr, true, (mlx_optional_int){eg_group,true}, (mlx_optional_int){eg_bits,true}, eg_mode, false, m->stream) == 0;
        if (ok && !moe_fused_done) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, x1, up_proj_w, up_proj_s, up_proj_b, no_lhs, top_inds_arr, true, (mlx_optional_int){eu_group,true}, (mlx_optional_int){eu_bits,true}, eu_mode, false, m->stream) == 0;
    }
    bool moe_swiglu_done = false;
    if (ok && !moe_fused_done && llm_mlx_fuse_swiglu_metal_enabled())
        moe_swiglu_done = llm_mlx_swiglu_metal(gate_out, up_out, &act, m->stream);
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0; }
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0; }
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_multiply(&act, silu, up_out, m->stream) == 0; }
    if (mxfp4_dn) llm_mlx_cast_f16_inplace(&act, m->stream);
    /* §E2 imatrix collector (path UNSORTED): row_expert = top_inds_arr, row_src = identita'. */
    if (ok && m->im.active && !do_sort && top_inds_arr.ctx != NULL) {
        const int Mi = n_tokens * m->cfg.top_k; im_rows = Mi;
        if (mlx_array_eval(top_inds_arr) == 0 && mlx_array_data_uint32(top_inds_arr) != NULL) {
            im_expert = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
            im_src = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
            if (im_expert && im_src) {
                memcpy(im_expert, mlx_array_data_uint32(top_inds_arr), (size_t)Mi * sizeof(uint32_t));
                for (int r = 0; r < Mi; r++) im_src[r] = (uint32_t)r;
            }
        }
    }
    if (ok && !moe_fused_done) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b, no_lhs, top_inds_arr, true, (mlx_optional_int){ed_group,true}, (mlx_optional_int){ed_bits,true}, ed_mode, false, m->stream) == 0;
    if (ok && !moe_fused_done) {
        /* down_out esce dalla gather come [n_tokens*top_k, 1, H]: normalizza a [n_tokens, top_k, H]
         * cosi' il broadcast con scores [n_tokens, top_k, 1, 1] e la somma su axis=1 sono corretti
         * per OGNI n_tokens. Senza questa normalizza il ramo UNSORTED (n_tokens in 2..7) moltiplica
         * [T*K,1,H] x [T,K,1,1] -> mismatch (T*K != K) -> fallisce; per n_tokens==1 il flatten
         * coincide con top_k e funzionava per caso (il ramo SORTED, n_tokens>=8, gia' normalizza). */
        mlx_array d3 = mlx_array_new();
        const int dsh[3] = { n_tokens, m->cfg.top_k, m->cfg.hidden_size };
        if (mlx_reshape(&d3, down_out, dsh, 3, m->stream) == 0) mlx_array_set(&down_out, d3);
        else ok = false;
        mlx_array_free(d3);
    }
    /* LoRA per-esperto MoE (fase 4): per ogni modulo su (layer, "moe_down", expert=j),
     * aggiunge contrib = act @ A_j @ B_j * scale SOLO alle posizioni dove l'esperto j e'
     * nel top-k (mask da top_inds_host). Adapter per-esperto del down_proj degli esperti.
     * moe_fused_done implica has_moe_down_lora==false (condizione di eleggibilita' sopra),
     * quindi questo loop non trova mai moduli quando il kernel fuso e' stato usato. */
    if (ok && !moe_fused_done) {
        for (int li = 0; li < m->lora_n; li++) {
            if (m->lora_layer[li] != layer_idx || strcmp(m->lora_proj[li], "moe_down") != 0 || m->lora_expert[li] < 0) continue;
            int ej = m->lora_expert[li];
            mlx_array contrib = mlx_array_new();
            bool okc = llm_mlx_lora_apply_mlx(&contrib, act, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream);
            if (!okc) { mlx_array_free(contrib); ok = false; break; }
            /* mask su device: top_inds == ej -> [1,T,K] -> [T,K,1,1] */
            mlx_array ej_a = mlx_array_new_data(&ej, (const int[]){1}, 1, MLX_INT32);
            mlx_array eqv = mlx_array_new(), mf = mlx_array_new(), mask = mlx_array_new();
            bool okw = false;
            if (ok) okw = mlx_equal(&eqv, top_inds, ej_a, m->stream) == 0;
            if (okw) okw = mlx_astype(&mf, eqv, MLX_FLOAT32, m->stream) == 0;
            if (okw) { int mk[4] = {n_tokens, m->cfg.top_k, 1, 1}; okw = mlx_reshape(&mask, mf, mk, 4, m->stream) == 0; }
            mlx_array zero = mlx_array_new_float32(0.0f);
            mlx_array masked = mlx_array_new();
            if (okw) okw = mlx_where(&masked, mask, contrib, zero, m->stream) == 0;
            if (okw) okw = mlx_add(&down_out, down_out, masked, m->stream) == 0;
            mlx_array_free(contrib); mlx_array_free(ej_a); mlx_array_free(eqv); mlx_array_free(mf);
            mlx_array_free(mask); mlx_array_free(zero); mlx_array_free(masked);
            if (!okw) { ok = false; break; }
        }
    }

    /* score (probs) gia' calcolato piu' sopra, prima del tentativo di kernel fuso -- non
     * ricalcolarlo qui. */
    if (ok && !moe_fused_done) {
        const int sc_shape[3] = {n_tokens, m->cfg.top_k, 1};
        if (llm_mlx_reshape_canonical_enabled()) {
            /* `probs` contiene gia' T*K elementi: il reshape intermedio [T,K] non
             * modifica ordine o dtype e aggiungeva una primitive per layer/token. */
            ok = mlx_reshape(&scores_arr, probs, sc_shape, 3, m->stream) == 0;
        } else {
            ok = mlx_reshape(&scores_arr, probs, sc_shape, 3, m->stream) == 0;   /* canonico: [T,K] intermedio ridondante */
        }
        if (ok) ok = mlx_multiply(&scaled, down_out, scores_arr, m->stream) == 0;
    }
    if (ok && !moe_fused_done) { ok = mlx_sum_axis(&summed, scaled, /*axis=*/1, /*keepdims=*/false, m->stream) == 0; }
    if (ok && !moe_fused_done) {
        const int out_shape[3] = {1, n_tokens, m->cfg.hidden_size};
        ok = mlx_reshape(&moe_out, summed, out_shape, 3, m->stream) == 0;
    }

    /* ── shared_expert (MLP denso SwiGLU con shared_expert_gate) ── */    }

    mlx_array se_w = mlx_array_new(), se_s = mlx_array_new(), se_b = mlx_array_new();
    mlx_array su_w = mlx_array_new(), su_s = mlx_array_new(), su_b = mlx_array_new();
    mlx_array sd_w = mlx_array_new(), sd_s = mlx_array_new(), sd_b = mlx_array_new();
    mlx_array seg_w = mlx_array_new(), seg_s = mlx_array_new(), seg_b = mlx_array_new();
    if (fq != NULL && fq->has_shared) {
        ok = ok && mlx_array_set(&se_w, fq->se_w) == 0 && mlx_array_set_optional(&se_s, fq->se_s) && mlx_array_set_optional(&se_b, fq->se_b);
        ok = ok && mlx_array_set(&su_w, fq->su_w) == 0 && mlx_array_set_optional(&su_s, fq->su_s) && mlx_array_set_optional(&su_b, fq->su_b);
        ok = ok && mlx_array_set(&sd_w, fq->sd_w) == 0 && mlx_array_set_optional(&sd_s, fq->sd_s) && mlx_array_set_optional(&sd_b, fq->sd_b);
        ok = ok && mlx_array_set(&seg_w, fq->sg_w) == 0 && mlx_array_set_optional(&seg_s, fq->sg_s) && mlx_array_set_optional(&seg_b, fq->sg_b);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &se_w, &se_s, &se_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &su_w, &su_s, &su_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &sd_w, &sd_s, &sd_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert_gate", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &seg_w, &seg_s, &seg_b);
    }
    CHK(ok, "shared_expert lookup");

    mlx_array se_gate = mlx_array_new(), se_up = mlx_array_new();
    /* NOTA: oMLX condivide la Stage A anche sullo shared_expert (oq_a8_mlp), ma MISURATO sul
     * 35B-A3B: la condivisione qui COSTA ~4% (SE -8% vs NO_SE -4.4%): l'MLP shared e' piccolo,
     * la prim multi-uscita costa piu' della Stage A risparmiata. Tenuta NON condivisa. */
    if (ok) { ok = quantized_linear_cfg(&se_gate, normed2, se_w, se_s, se_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_gate_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&se_gate, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = quantized_linear_cfg(&se_up, normed2, su_w, su_s, su_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_up_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&se_up, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array se_sig = mlx_array_new(), se_silu = mlx_array_new(), se_act = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&se_sig, se_gate, m->stream) == 0; }
    if (ok) { ok = mlx_multiply(&se_silu, se_gate, se_sig, m->stream) == 0; }
    if (ok) { ok = mlx_multiply(&se_act, se_silu, se_up, m->stream) == 0; }
    mlx_array se_down = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&se_down, se_act, sd_w, sd_s, sd_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_down_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&se_down, se_act, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    /* shared_expert_gate: [B,S,1] -> sigmoid, poi * se_down */
    mlx_array seg = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&seg, normed2, seg_w, seg_s, seg_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_gate"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&seg, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array seg_sig = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&seg_sig, seg, m->stream) == 0; }
    mlx_array shared_out = mlx_array_new();
    if (ok) { ok = mlx_multiply(&shared_out, se_down, seg_sig, m->stream) == 0; }

    /* out = h + moe_out + shared_out (graph lazy: l'add annidato dipende da mid; gli array
     * sono refcounted in mlx-c, quindi la free dei temporanei qui sotto non invalida il
     * grafo — l'eval avviene una sola volta a fine token (final_step), come in qwen3_moe. */
    mlx_array mid = mlx_array_new();
    if (ok) { ok = mlx_add(&mid, h, moe_out, m->stream) == 0; CHK(ok, "mid add"); }
    if (ok) { ok = mlx_add(out, mid, shared_out, m->stream) == 0; CHK(ok, "final residual add"); }

    /* §E2 imatrix collector (calibrazione): accumula (layer, expert routato, colonna) sia sorted sia unsorted. */
    if (m->im.active && ok && im_expert && im_src && im_rows > 0 && act.ctx != NULL)
        llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, im_expert, im_src, im_rows);
    free(im_expert); free(im_src);

    mlx_array all[] = {
        ln2_w, normed2, gate_w, gate_s, gate_b, gates, gates_sm, part, top_inds,
        gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
        down_proj_w, down_proj_s, down_proj_b, moe_out, no_lhs, top_inds_arr, x1,
        gate_out, up_out, sig, silu, act, down_out, scores_arr, scaled, summed, probs,
        se_w, se_s, se_b, su_w, su_s, su_b, sd_w, sd_s, sd_b, seg_w, seg_s, seg_b,
        se_gate, se_up, se_sig, se_silu, se_act, se_down, seg, seg_sig, shared_out, mid,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}

/* SparseMoeBlock di Qwen4-Exp (qwen4_exp.py righe 303-329) -- STESSO pattern esatto di
 * qwen3_5_moe_ffn_block sopra (softmax top-k + switch_mlp + shared_expert con gate
 * sigmoid separato, nomi pesi IDENTICI: mlp.gate/mlp.switch_mlp.{gate,up,down}_proj/
 * mlp.shared_expert.{gate,up,down}_proj/mlp.shared_expert_gate -- verificato riga per
 * riga contro il reference, nessuna differenza). Le uniche due differenze, entrambe per
 * via dell'hyper-connection del chiamante (qwen4_exp_ffn_dispatch): (1) NESSUNA norma
 * qui dentro -- `h` arriva GIA' normalizzato/mixato (mlp_hyper_connection), a differenza
 * di ogni altro *_ffn_block che fa il proprio post_attention_layernorm; (2) NESSUN
 * residuo qui dentro -- ritorna `moe_out+shared_out` puro, il residuo/injection lo fa
 * il chiamante via llm_mlx_hc_combine (schema hyper-connection, diverso da "h+moe_out"). */
bool qwen4_exp_moe_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen4_exp_moe[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array normed2 = mlx_array_new();
    ok = mlx_array_set(&normed2, h) == 0;

    mlx_array gate_w = mlx_array_new(), gate_s = mlx_array_new(), gate_b = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.gate", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_w, &gate_s, &gate_b);
    mlx_array gates = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gates, normed2, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "router matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "mlp.gate"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&gates, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array gates_sm = mlx_array_new();
    if (ok && m->cfg.bfloat16_input && mlx_array_dtype(gates) != g_act16) ok = mlx_astype(&gates, gates, g_act16, m->stream) == 0;
    if (ok) { ok = mlx_softmax_axis(&gates_sm, gates, -1, /*precise=*/true, m->stream) == 0; CHK(ok, "router softmax"); }

    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, gates_sm, /*kth=*/-m->cfg.top_k, /*axis=*/-1, m->stream) == 0;
        if (ok) {
            const int num_experts = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, num_experts - m->cfg.top_k};
            const int stop[3] = {1, n_tokens, num_experts};
            const int strides[3] = {1, 1, 1};
            mlx_array top_inds_view = mlx_array_new();
            ok = mlx_slice(&top_inds_view, part, start, 3, stop, 3, strides, 3, m->stream) == 0;
            if (ok) ok = mlx_contiguous(&top_inds, top_inds_view, /*allow_col_major=*/false, m->stream) == 0;
            mlx_array_free(top_inds_view);
        }
    }
    mlx_array gate_proj_w = mlx_array_new(), gate_proj_s = mlx_array_new(), gate_proj_b = mlx_array_new();
    mlx_array up_proj_w = mlx_array_new(), up_proj_s = mlx_array_new(), up_proj_b = mlx_array_new();
    mlx_array down_proj_w = mlx_array_new(), down_proj_s = mlx_array_new(), down_proj_b = mlx_array_new();
    int eg_group = m->cfg.quant_group_size, eg_bits = m->cfg.quant_bits; char eg_mode[16];
    int eu_group = m->cfg.quant_group_size, eu_bits = m->cfg.quant_bits; char eu_mode[16];
    int ed_group = m->cfg.quant_group_size, ed_bits = m->cfg.quant_bits; char ed_mode[16];
    /* eg/eu/ed_mode sono SEMPRE scritti sotto (fq->*mode nel ramo cache, oppure
     * llm_mlx_quant_derive che azzera e riscrive out_mode): il vecchio
     * snprintf(..., m->cfg.quant_mode) era DEAD CODE e pagava __vfprintf a ogni layer. */
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_proj_w, &gate_proj_s, &gate_proj_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &up_proj_w, &up_proj_s, &up_proj_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &down_proj_w, &down_proj_s, &down_proj_b);
    llm_mlx_quant_derive(gate_proj_w, gate_proj_s, eg_group, eg_bits, "affine", &eg_group, &eg_bits, eg_mode, sizeof(eg_mode));
    llm_mlx_quant_derive(up_proj_w, up_proj_s, eu_group, eu_bits, "affine", &eu_group, &eu_bits, eu_mode, sizeof(eu_mode));
    llm_mlx_quant_derive(down_proj_w, down_proj_s, ed_group, ed_bits, "affine", &ed_group, &ed_bits, ed_mode, sizeof(ed_mode));

    mlx_array x1 = mlx_array_new();
    mlx_array moe_out = mlx_array_new();
    mlx_array no_lhs = mlx_array_new();
    mlx_array top_inds_arr = mlx_array_new();
    uint32_t* im_expert = NULL; uint32_t* im_src = NULL; int im_rows = 0; /* §E2 imatrix collector */
    if (ok) {
        const int ti2_shape[2] = {n_tokens, m->cfg.top_k};
        mlx_array tt = mlx_array_new();
        ok = mlx_reshape(&tt, top_inds, ti2_shape, 2, m->stream) == 0;
        if (ok) ok = mlx_astype(&top_inds_arr, tt, MLX_UINT32, m->stream) == 0;
        mlx_array_free(tt);
        CHK(ok, "top_inds device reshape/astype");
    }
    if (ok) {
        const int x1_shape[4] = {n_tokens, 1, 1, m->cfg.hidden_size};
        ok = mlx_reshape(&x1, normed2, x1_shape, 4, m->stream) == 0;
    }
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    mlx_array down_out = mlx_array_new();
    mlx_array probs = mlx_array_new();
    mlx_array scores_arr = mlx_array_new(), scaled = mlx_array_new();
    mlx_array summed = mlx_array_new();
    /* SORTED (fix performance 2026-09-18, profilo LLM_DISPATCH): l'UNSORTED broadcasta i token
     * sui top_k esperti -> 215 nodi/layer e 8.3 GB/token (Broadcast/Multiply/Reshape shape
     * (T,top_k,1,hidden)). Il path sorted (argsort -> take dei token ordinati -> gather
     * sorted=true -> unsort) NON broadcasta: e' quello di qwen3_5_moe/qwen3_moe. Semantica
     * identica (stessa somma pesata sui top_k). */
    const int M = n_tokens * m->cfg.top_k;
    mlx_array tt = mlx_array_new(), order = mlx_array_new(), inv = mlx_array_new();
    mlx_array df = mlx_array_new(), dv = mlx_array_new(), idxs = mlx_array_new();
    mlx_array dvm4 = mlx_array_new(), rn = mlx_array_new(), rnc = mlx_array_new(), xs = mlx_array_new();
    if (ok) { const int ts[1] = {M}; ok = mlx_reshape(&tt, top_inds_arr, ts, 1, m->stream) == 0; }
    if (ok) ok = mlx_argsort(&order, tt, m->stream) == 0;
    if (ok) ok = mlx_argsort(&inv, order, m->stream) == 0;
    { mlx_array kk = mlx_array_new_int(m->cfg.top_k);
      if (ok) ok = mlx_floor_divide(&df, order, kk, m->stream) == 0; mlx_array_free(kk); }
    if (ok) ok = mlx_astype(&dv, df, MLX_UINT32, m->stream) == 0;
    if (ok) ok = mlx_take(&idxs, tt, order, m->stream) == 0;
    { const int ms[4] = {M, 1, 1, 1}; if (ok) ok = mlx_reshape(&dvm4, dv, ms, 4, m->stream) == 0; }
    if (ok) ok = mlx_take_along_axis(&rn, x1, dvm4, 0, m->stream) == 0;
    if (ok) ok = mlx_contiguous(&rnc, rn, false, m->stream) == 0;
    { const int xs3[3] = {M, 1, m->cfg.hidden_size}; if (ok) ok = mlx_reshape(&xs, rnc, xs3, 3, m->stream) == 0; }
    /* NB (todo MLX-Decode-Composizione §7): la fusione gate+up NON è cablata qui di
     * proposito: (a) il prepare condiviso `llm_mlx_moe_ffn_quant` richiede
     * `post_attention_layernorm.weight` (assente in qwen4_exp: il blocco usa l'HC mixer),
     * quindi la cache non si costruirebbe; (b) il costo memoria della concat su 512 esperti
     * × 48 layer è stimato ~40 GB, non giustificato sul modello da 97 GB. Se servirà, va
     * fatto un prepare dedicato (senza ln2) + un budget di memoria. */
    if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, xs, gate_proj_w, gate_proj_s, gate_proj_b, no_lhs, idxs, true, (mlx_optional_int){eg_group,true}, (mlx_optional_int){eg_bits,true}, eg_mode, /*sorted=*/true, m->stream) == 0;
    if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, xs, up_proj_w, up_proj_s, up_proj_b, no_lhs, idxs, true, (mlx_optional_int){eu_group,true}, (mlx_optional_int){eu_bits,true}, eu_mode, /*sorted=*/true, m->stream) == 0;
    if (ok) { ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0; }
    if (ok) { ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0; }
    if (ok) { ok = mlx_multiply(&act, silu, up_out, m->stream) == 0; }
    if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b, no_lhs, idxs, true, (mlx_optional_int){ed_group,true}, (mlx_optional_int){ed_bits,true}, ed_mode, /*sorted=*/true, m->stream) == 0;
    /* §E2 imatrix collector (path SORTED): idxs + inv. */
    if (ok && m->im.active) {
        const int Mi = M; im_rows = Mi;
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
    mlx_array invm = mlx_array_new(), duns = mlx_array_new(), dh3 = mlx_array_new();
    { const int is3[3] = {M, 1, 1}; if (ok) ok = mlx_reshape(&invm, inv, is3, 3, m->stream) == 0; }
    if (ok) ok = mlx_take_along_axis(&duns, down_out, invm, 0, m->stream) == 0;
    { const int d3[3] = {n_tokens, m->cfg.top_k, m->cfg.hidden_size}; if (ok) ok = mlx_reshape(&dh3, duns, d3, 3, m->stream) == 0; }
    if (ok) ok = mlx_take_along_axis(&probs, gates_sm, top_inds, /*axis=*/-1, m->stream) == 0;
    if (ok && m->cfg.norm_topk_prob) {
        mlx_array psum = mlx_array_new();
        if (ok) ok = mlx_sum_axis(&psum, probs, /*axis=*/2, /*keepdims=*/true, m->stream) == 0;
        if (ok) ok = mlx_divide(&probs, probs, psum, m->stream) == 0;
        mlx_array_free(psum);
    }
    {
        mlx_array pr = mlx_array_new(), sc3 = mlx_array_new(), scaled3 = mlx_array_new(), sumt = mlx_array_new();
        const int pr2[2] = {n_tokens, m->cfg.top_k};
        const int sc3s[3] = {n_tokens, m->cfg.top_k, 1};
        if (ok) ok = mlx_reshape(&pr, probs, pr2, 2, m->stream) == 0;
        if (ok) ok = mlx_reshape(&sc3, pr, sc3s, 3, m->stream) == 0;
        if (ok) ok = mlx_multiply(&scaled3, dh3, sc3, m->stream) == 0;
        if (ok) ok = mlx_sum_axis(&sumt, scaled3, 1, false, m->stream) == 0;
        const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
        if (ok) ok = mlx_reshape(&moe_out, sumt, o3, 3, m->stream) == 0;
        mlx_array_free(pr); mlx_array_free(sc3); mlx_array_free(scaled3); mlx_array_free(sumt);
    }
    mlx_array_free(tt); mlx_array_free(order); mlx_array_free(inv); mlx_array_free(df);
    mlx_array_free(dv); mlx_array_free(idxs); mlx_array_free(dvm4); mlx_array_free(rn);
    mlx_array_free(rnc); mlx_array_free(xs); mlx_array_free(invm); mlx_array_free(duns); mlx_array_free(dh3);

    mlx_array se_w = mlx_array_new(), se_s = mlx_array_new(), se_b = mlx_array_new();
    mlx_array su_w = mlx_array_new(), su_s = mlx_array_new(), su_b = mlx_array_new();
    mlx_array sd_w = mlx_array_new(), sd_s = mlx_array_new(), sd_b = mlx_array_new();
    mlx_array seg_w = mlx_array_new(), seg_s = mlx_array_new(), seg_b = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.gate_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &se_w, &se_s, &se_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.up_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &su_w, &su_s, &su_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.down_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &sd_w, &sd_s, &sd_b);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert_gate", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &seg_w, &seg_s, &seg_b);
    CHK(ok, "shared_expert lookup");

    mlx_array se_gate = mlx_array_new(), se_up = mlx_array_new();
    /* NOTA: oMLX condivide la Stage A anche sullo shared_expert (oq_a8_mlp), ma MISURATO sul
     * 35B-A3B: la condivisione qui COSTA ~4% (SE -8% vs NO_SE -4.4%): l'MLP shared e' piccolo,
     * la prim multi-uscita costa piu' della Stage A risparmiata. Tenuta NON condivisa. */
    if (ok) { ok = quantized_linear_cfg(&se_gate, normed2, se_w, se_s, se_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_gate_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&se_gate, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { ok = quantized_linear_cfg(&se_up, normed2, su_w, su_s, su_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_up_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&se_up, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array se_sig = mlx_array_new(), se_silu = mlx_array_new(), se_act = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&se_sig, se_gate, m->stream) == 0; }
    if (ok) { ok = mlx_multiply(&se_silu, se_gate, se_sig, m->stream) == 0; }
    if (ok) { ok = mlx_multiply(&se_act, se_silu, se_up, m->stream) == 0; }
    mlx_array se_down = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&se_down, se_act, sd_w, sd_s, sd_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_down_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&se_down, se_act, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array seg = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&seg, normed2, seg_w, seg_s, seg_b, gsz, bts, m->cfg.quant_mode, m->stream); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "shared_expert_gate"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&seg, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    mlx_array seg_sig = mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&seg_sig, seg, m->stream) == 0; }
    mlx_array shared_out = mlx_array_new();
    if (ok) { ok = mlx_multiply(&shared_out, se_down, seg_sig, m->stream) == 0; }

    /* NIENTE residuo qui: out = moe_out + shared_out puro (il residuo/injection e'
     * responsabilita' del chiamante, llm_mlx_hc_combine). */
    if (ok) { ok = mlx_add(out, moe_out, shared_out, m->stream) == 0; CHK(ok, "moe_out+shared_out"); }

    /* §E2 imatrix collector (calibrazione): accumula (layer, expert routato, colonna). */
    if (m->im.active && ok && im_expert && im_src && im_rows > 0 && act.ctx != NULL)
        llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, im_expert, im_src, im_rows);
    free(im_expert); free(im_src);

    mlx_array all[] = {
        normed2, gate_w, gate_s, gate_b, gates, gates_sm, part, top_inds,
        gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
        down_proj_w, down_proj_s, down_proj_b, moe_out, no_lhs, top_inds_arr, x1,
        gate_out, up_out, sig, silu, act, down_out, scores_arr, scaled, summed, probs,
        se_w, se_s, se_b, su_w, su_s, su_b, sd_w, sd_s, sd_b, seg_w, seg_s, seg_b,
        se_gate, se_up, se_sig, se_silu, se_act, se_down, seg, seg_sig, shared_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
