/* ============================================================================
 * llm_mlx_mixtral.c — dominio Mixtral: FFN MoE (ordine di normalizzazione router diverso
 * da Qwen3-MoE: softmax sui soli top-k logit). Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_moe_fused_kernel.h"
#include "llm_mlx_swiglu_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* FASE 1b.3 (Mixtral): FFN MoE con ordine di normalizzazione DIVERSO da Qwen3-MoE --
 * verificato leggendo MixtralSparseMoeBlock.__call__ in mlx_lm/models/mixtral.py: qui
 * il softmax si applica SOLO sui top_k logit GIA' selezionati (non su tutti gli esperti
 * prima della selezione, come invece fa qwen3_moe_ffn_block). La selezione stessa
 * (argpartition) non cambia in base a dove si applica il softmax, perché softmax è una
 * trasformazione monotona (non altera QUALI esperti vengono scelti, solo il valore
 * numerico del punteggio finale con cui si combinano i loro output) -- ma il PUNTEGGIO
 * risultante è diverso fra le due famiglie, quindi le due formule non sono
 * intercambiabili: mai assumere l'ordine di normalizzazione di una famiglia MoE da
 * un'altra senza verificare il sorgente reale (vedi anche Knowledge Base §5.3).
 * Router (`block_sparse_moe.gate`) NON quantizzato nel modello reale
 * (Mixtral-8x22B-4bit: solo ".weight", nessuna ".scales"/".biases" accanto, a
 * differenza di ogni altra proiezione del layer) -- richiede plain_linear invece di
 * quantized_linear_cfg. I pesi degli esperti (gate_proj/up_proj/down_proj) sono già
 * stati impilati una volta a caricamento (stack_mixtral_moe_experts) sotto
 * "block_sparse_moe.switch_mlp.*" -- da qui in poi stesso identico schema gather_qmm
 * di qwen3_moe_ffn_block (SwiGLU, non GeGLU: hidden_act="silu" nel config reale). */
bool mixtral_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "mixtral_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)
    bool gmt = llm_env_mixtral_time();
    bool gmem = llm_env_memprof();
    struct timespec _t0, _t1;
    #define TICK(label) do { if (gmt && layer_idx == 0) { clock_gettime(CLOCK_MONOTONIC, &_t1); fprintf(stderr, "  [mx] %-18s %8.1f ms\n", (label), (double)((_t1.tv_sec-_t0.tv_sec)+(_t1.tv_nsec-_t0.tv_nsec)/1e9)*1e3); clock_gettime(CLOCK_MONOTONIC, &_t0); } \
        if (gmem && layer_idx == 0) { size_t _a=0; mlx_get_active_memory(&_a); fprintf(stderr, "  [mxm] %-18s active=%.1f MB\n", (label), _a/1048576.0); fflush(stderr); } } while(0)
    if (gmt && layer_idx == 0) clock_gettime(CLOCK_MONOTONIC, &_t0);

    mlx_array ln2_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }

    mlx_array gate_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "block_sparse_moe.gate.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &gate_w);
    mlx_array gates = mlx_array_new();
    if (ok) { ok = plain_linear(&gates, normed2, gate_w, m->stream); CHK(ok, "router matmul (non quantizzato)"); }
    if ((gmt||gmem) && layer_idx == 0) { mlx_array_eval(gates); TICK("rms+router"); }

    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        /* argpartition direttamente sui logit grezzi -- softmax è monotona, non
         * cambia quali indici finiscono nel top_k (vedi commento sopra). */
        ok = mlx_argpartition_axis(&part, gates, /*kth=*/-m->cfg.top_k, /*axis=*/-1, m->stream) == 0;
        CHK(ok, "router argpartition");
        if (ok) {
            const int num_experts = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, num_experts - m->cfg.top_k};
            const int stop[3] = {1, n_tokens, num_experts};
            const int strides[3] = {1, 1, 1};
            mlx_array top_inds_view = mlx_array_new();
            ok = mlx_slice(&top_inds_view, part, start, 3, stop, 3, strides, 3, m->stream) == 0;
            CHK(ok, "top-k slice");
            if (ok) {
                /* Stessa view non contigua di qwen3_moe_ffn_block per n_tokens>1 --
                 * mlx_contiguous forza una copia densa prima del memcpy in host memory. */
                ok = mlx_contiguous(&top_inds, top_inds_view, /*allow_col_major=*/false, m->stream) == 0;
                CHK(ok, "top-k contiguous copy");
            }
            mlx_array_free(top_inds_view);
        }
    }

    /* Leva A: niente readback host del router per layer (era il collo decode di Mixtral, 0.3 tok/s).
     * top_inds_arr su device (reshape+astype di top_inds), pesi top-k via take_along+softmax. */

    mlx_array gate_proj_w = mlx_array_new(), gate_proj_s = mlx_array_new(), gate_proj_b = mlx_array_new();
    mlx_array up_proj_w = mlx_array_new(), up_proj_s = mlx_array_new(), up_proj_b = mlx_array_new();
    mlx_array down_proj_w = mlx_array_new(), down_proj_s = mlx_array_new(), down_proj_b = mlx_array_new();
    /* Cache pesi esperti (P1/P3): risolti una volta; fallback legacy (LLM_WCACHE=0). */
    GmoeFfnQuant* fq = llm_mlx_moe_ffn_quant(m, layer_idx);
    if (fq != NULL) {
        ok = ok && mlx_array_set(&gate_proj_w, fq->gw) == 0 && mlx_array_set_optional(&gate_proj_s, fq->gs) &&
                   mlx_array_set_optional(&gate_proj_b, fq->gb);
        ok = ok && mlx_array_set(&up_proj_w, fq->uw) == 0 && mlx_array_set_optional(&up_proj_s, fq->us) &&
                   mlx_array_set_optional(&up_proj_b, fq->ub);
        ok = ok && mlx_array_set(&down_proj_w, fq->dw) == 0 && mlx_array_set_optional(&down_proj_s, fq->ds) &&
                   mlx_array_set_optional(&down_proj_b, fq->db);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "block_sparse_moe.switch_mlp.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_proj_w, &gate_proj_s, &gate_proj_b);
        tname(m->cfg.layer_prefix, layer_idx, "block_sparse_moe.switch_mlp.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_proj_w, &up_proj_s, &up_proj_b);
        tname(m->cfg.layer_prefix, layer_idx, "block_sparse_moe.switch_mlp.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &down_proj_w, &down_proj_s, &down_proj_b);
    }
    CHK(ok, "switch_mlp lookup (stacked a caricamento da stack_mixtral_moe_experts)");
    if ((gmt||gmem) && layer_idx == 0) {
        const int* shg = mlx_array_shape(gate_proj_w); int ng = mlx_array_ndim(gate_proj_w);
        const int* shs = mlx_array_shape(gate_proj_s);
        fprintf(stderr, "  [mx] gate_proj_w ndim=%d shape=[%d,%d,%d] scales=[%d,%d,%d] ndim_s=%d\n",
                ng, ng>0?shg[0]:-1, ng>1?shg[1]:-1, ng>2?shg[2]:-1,
                (int)mlx_array_shape(gate_proj_s)[0], (int)mlx_array_shape(gate_proj_s)[1],
                (int)mlx_array_shape(gate_proj_s)[2], (int)mlx_array_ndim(gate_proj_s));
        fflush(stderr);
    }

    mlx_array x1 = mlx_array_new();
    mlx_array moe_out = mlx_array_new();
    mlx_array no_lhs = mlx_array_new();
    mlx_array top_inds_arr = mlx_array_new();
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
        CHK(ok, "moe x1 reshape");
    }

    mlx_optional_int gsz_opt = {gsz, true};
    mlx_optional_int bts_opt = {bts, true};
    bool do_sort = ok && ((int64_t)n_tokens * (int64_t)m->cfg.top_k) >= 64;
    /* Q2-b: imatrix (calibrazione) — esperto per riga sorted + riga sorted -> token-major. */
    uint32_t* im_expert = NULL; uint32_t* im_src = NULL; int im_rows = 0;
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
        mlx_array dvm4 = mlx_array_new();
        { const int ms[4] = {M, 1, 1, 1}; if (ok) ok = mlx_reshape(&dvm4, dv, ms, 4, m->stream) == 0; }
        mlx_array rn = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&rn, x1, dvm4, 0, m->stream) == 0;
        mlx_array rnc = mlx_array_new();
        if (ok) ok = mlx_contiguous(&rnc, rn, false, m->stream) == 0;
        mlx_array xs = mlx_array_new();
        { const int xs3[3] = {M, 1, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&xs, rnc, xs3, 3, m->stream) == 0; }
        if ((gmt||gmem) && layer_idx == 0) { mlx_array_eval(xs); TICK("pref sort+xs"); }
        if (ok && !llm_mlx_moe_gateup_gather(fq, xs, no_lhs, idxs, /*sorted=*/true, gsz_opt, bts_opt,
                                             "affine", &gate_out, &up_out, m->stream)) {
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, xs, gate_proj_w, gate_proj_s, gate_proj_b, no_lhs, idxs,
                                        true, gsz_opt, bts_opt, "affine", /*sorted=*/true, m->stream) == 0;
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, xs, up_proj_w, up_proj_s, up_proj_b, no_lhs, idxs,
                                        true, gsz_opt, bts_opt, "affine", /*sorted=*/true, m->stream) == 0;
        }
        if ((gmt||gmem) && layer_idx == 0) { mlx_array_eval(gate_out); TICK("pref gate/up"); }
        if (ok) ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0;
        if (ok) ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0;
        if (ok) ok = mlx_multiply(&act, silu, up_out, m->stream) == 0;
        if ((gmt||gmem) && layer_idx == 0) { mlx_array_eval(act); TICK("pref swiglu"); }
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b, no_lhs, idxs,
                                    true, gsz_opt, bts_opt, "affine", /*sorted=*/true, m->stream) == 0;
        /* Q2-b: readback idxs (esperto per riga) + inv (riga sorted -> token-major), solo in
         * calibrazione (stesso schema di qwen3_moe_ffn_block). Senza path sorted non si accumula. */
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
        if ((gmt||gmem) && layer_idx == 0) { mlx_array_eval(down_out); TICK("pref down"); }
        mlx_array invm = mlx_array_new();
        { const int is3[3] = {M, 1, 1}; if (ok) ok = mlx_reshape(&invm, inv, is3, 3, m->stream) == 0; }
        mlx_array duns = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&duns, down_out, invm, 0, m->stream) == 0;
        mlx_array_free(invm);
        mlx_array dh3 = mlx_array_new();
        { const int d3[3] = {n_tokens, m->cfg.top_k, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&dh3, duns, d3, 3, m->stream) == 0; }
        if (ok) ok = mlx_take_along_axis(&probs, gates, top_inds, -1, m->stream) == 0;
        if (ok) ok = mlx_softmax_axis(&probs, probs, -1, true, m->stream) == 0;
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
        mlx_array_free(tt); mlx_array_free(order); mlx_array_free(inv); mlx_array_free(idxs); mlx_array_free(dvm4);
        mlx_array_free(rn); mlx_array_free(rnc); mlx_array_free(xs); mlx_array_free(duns);
        mlx_array_free(dh3);
    } else
    {

    /* Punteggio (softmax sui soli top_k logit grezzi selezionati) MOVED UP qui: prima
     * veniva calcolato dopo down_out, ma il kernel fuso sotto ne ha bisogno subito. */
    if (ok) ok = mlx_take_along_axis(&probs, gates, top_inds, /*axis=*/-1, m->stream) == 0;
    if (ok) ok = mlx_softmax_axis(&probs, probs, -1, /*precise=*/true, m->stream) == 0;
    CHK(ok, "moe topk softmax");
    if ((gmt||gmem) && layer_idx == 0) { mlx_array_eval(probs); TICK("topk+softmax"); }

    /* Kernel Metal MoE decode fuso (LLMENGINE_MOE_FUSE_MOE_DECODE, default ON): sostituisce le 2-3
     * gather_qmm + sigmoid/multiply con due dispatch fusi (gate+up+SwiGLU, poi down pesata
     * per score). Solo n_tokens==1 e bits in {4,8}; il kernel rifiuta (false) scale/bias non
     * affine o dtype mismatch -> fallback silenzioso al path gather_qmm qui sotto. Stesso
     * schema di qwen3_moe_ffn_block (vedi llm_mlx_moe_fused_kernel.h). */
    bool moe_fused_done = false;
    if (ok && n_tokens == 1 && (bts == 4 || bts == 8) && llm_mlx_fuse_moe_decode_enabled()) {
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
            bool fok = llm_mlx_moe_fused_decode(&fused_summed, x_flat, idx_flat,
                gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
                down_proj_w, down_proj_s, down_proj_b, sc_flat,
                din, dff, k, bts, gsz, bts, gsz, /*act=SwiGLU*/0, m->stream);
            if (fok) {
                mlx_array cast = mlx_array_new();
                fok = mlx_astype(&cast, fused_summed, mlx_array_dtype(h), m->stream) == 0;
                if (fok) {
                    const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
                    fok = mlx_reshape(&moe_out, cast, o3, 3, m->stream) == 0;
                }
                mlx_array_free(cast);
            }
            if (fok && llm_mlx_fuse_moe_decode_verify_enabled()) {
                /* _VERIFY: non usato per calcolare moe_out, solo per stampare il diff. */
                mlx_array ref_gate = mlx_array_new(), ref_up = mlx_array_new(), ref_act = mlx_array_new();
                mlx_array ref_down = mlx_array_new(), ref_sig = mlx_array_new();
                bool rok = mlx_gather_qmm(&ref_gate, x1, gate_proj_w, gate_proj_s, gate_proj_b,
                        no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0 &&
                    mlx_gather_qmm(&ref_up, x1, up_proj_w, up_proj_s, up_proj_b,
                        no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
                if (rok) rok = mlx_sigmoid(&ref_sig, ref_gate, m->stream) == 0;
                mlx_array ref_silu = mlx_array_new();
                if (rok) rok = mlx_multiply(&ref_silu, ref_gate, ref_sig, m->stream) == 0;
                if (rok) rok = mlx_multiply(&ref_act, ref_silu, ref_up, m->stream) == 0;
                if (rok) rok = mlx_gather_qmm(&ref_down, ref_act, down_proj_w, down_proj_s, down_proj_b,
                        no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
                mlx_array ref_scaled = mlx_array_new(), ref_scores4 = mlx_array_new(), ref_summed = mlx_array_new();
                if (rok) { const int sc4[4] = {1, k, 1, 1}; rok = mlx_reshape(&ref_scores4, pr1, sc4, 4, m->stream) == 0; }
                if (rok) rok = mlx_multiply(&ref_scaled, ref_down, ref_scores4, m->stream) == 0;
                if (rok) rok = mlx_sum_axis(&ref_summed, ref_scaled, 1, false, m->stream) == 0;
                if (rok) {
                    mlx_array af = mlx_array_new(), bf = mlx_array_new();
                    mlx_array ref_summed_flat = mlx_array_new();
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
                            fprintf(stderr, "[fuse-moe-verify] layer=%d max_abs_diff=%.6g rel_rms=%.6g\n",
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

    if (ok && !moe_fused_done) {
        /* Copia rapidMLX moe_fusion.py: se i pesi gate+up sono stati fusi a
         * load (gate_up_proj, shard sintetico), UNA gather_qmm [E,2*inter,in]
         * invece di due; poi split lungo l'ultimo asse. Bit-identico (quant
         * affine per-riga). Altrimenti path storico (due gather separate). */
        mlx_array gu_w = mlx_array_new(), gu_s = mlx_array_new(), gu_b = mlx_array_new();
        char gubase[256];
        tname(m->cfg.layer_prefix, layer_idx, "block_sparse_moe.switch_mlp.gate_up_proj", gubase, sizeof(gubase));
        bool have_gu = getenv("LLMENGINE_MOE_MLX_MOE_SWAP_DIR") == NULL && multishard_get_triple(&m->weights, gubase, &gu_w, &gu_s, &gu_b) && (gu_s.ctx != NULL);
        if (have_gu) {
            mlx_array gu_out = mlx_array_new();
            ok = llm_mlx_moe_gather_mm(&m->cfg, &gu_out, x1, gu_w, gu_s, gu_b, no_lhs, top_inds_arr,
                                /*transpose=*/true, gsz_opt, bts_opt, "affine",
                                /*sorted_indices=*/false, m->stream) == 0;
            CHK(ok, "gather_qmm gate_up fused");
            if (ok) {
                mlx_vector_array parts = mlx_vector_array_new();
                ok = mlx_split(&parts, gu_out, 2, -1, m->stream) == 0;
                if (ok) {
                    mlx_vector_array_get(&gate_out, parts, 0);
                    mlx_vector_array_get(&up_out, parts, 1);
                    /* Le due meta' della split sono VISTE non-contigue: la gather_qmm
                     * (down) successiva su input non-contiguo degradava ~40x (misurato
                     * 2026-09-03). Forza la densita' qui, prima di sigmoid/act. */
                    mlx_array gc = mlx_array_new(), uc = mlx_array_new();
                    ok = mlx_contiguous(&gc, gate_out, false, m->stream) == 0 &&
                         mlx_contiguous(&uc, up_out, false, m->stream) == 0;
                    if (ok) { mlx_array_set(&gate_out, gc); mlx_array_set(&up_out, uc); }
                    mlx_array_free(gc); mlx_array_free(uc);
                }
                mlx_vector_array_free(parts);
                CHK(ok, "gate_up split");
            }
            mlx_array_free(gu_out);
        } else if (!llm_mlx_moe_gateup_gather(fq, x1, no_lhs, top_inds_arr, /*sorted=*/false, gsz_opt, bts_opt,
                                              "affine", &gate_out, &up_out, m->stream)) {
            ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, x1, gate_proj_w, gate_proj_s, gate_proj_b,
                                 no_lhs, top_inds_arr, /*transpose=*/true, gsz_opt, bts_opt,
                                 "affine", /*sorted_indices=*/false, m->stream) == 0;
            CHK(ok, "gather_qmm gate_proj");
            if (ok) {
                ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, x1, up_proj_w, up_proj_s, up_proj_b,
                                     no_lhs, top_inds_arr, true, gsz_opt, bts_opt,
                                     "affine", false, m->stream) == 0;
                CHK(ok, "gather_qmm up_proj");
            }
        }
        mlx_array_free(gu_w); mlx_array_free(gu_s); mlx_array_free(gu_b);
    }

    /* Replica della vittoria densa sulla MoE: kernel Metal fuso silu(g)*u (bit-fedele,
     * 1 dispatch invece di 3) gated da LLMMETAL_FUSE_SWIGLU_METAL (default ON); fallback 3-op. */
    bool moe_swiglu_done = false;
    if (ok && !moe_fused_done && llm_mlx_fuse_swiglu_metal_enabled())
        moe_swiglu_done = llm_mlx_swiglu_metal(gate_out, up_out, &act, m->stream);
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0; CHK(ok, "moe sigmoid"); }
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0; CHK(ok, "moe silu"); }
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_multiply(&act, silu, up_out, m->stream) == 0; CHK(ok, "moe swiglu"); }

    if (ok && !moe_fused_done) {
        ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b,
                             no_lhs, top_inds_arr, true, gsz_opt, bts_opt,
                             "affine", false, m->stream) == 0;
        CHK(ok, "gather_qmm down_proj");
    }
    if ((gmt||gmem) && layer_idx == 0 && !moe_fused_done) { mlx_array_eval(down_out); TICK("gate/up/down gather"); }

    /* probs (score top-k rinormalizzato via softmax) gia' calcolato in cima a questo ramo. */
    if (ok && !moe_fused_done) {
        const int pr_shape[2] = {n_tokens, m->cfg.top_k};
        const int sc_shape[4] = {n_tokens, m->cfg.top_k, 1, 1};
        ok = mlx_reshape(&scores_arr, probs, sc_shape, 4, m->stream) == 0;   /* canonico: [T,K] intermedio ridondante */
        if (ok) ok = mlx_multiply(&scaled, down_out, scores_arr, m->stream) == 0;
        CHK(ok, "moe score scale");
    }

    if (ok && !moe_fused_done) { ok = mlx_sum_axis(&summed, scaled, /*axis=*/1, /*keepdims=*/false, m->stream) == 0; CHK(ok, "moe sum over top_k"); }
    if (ok && !moe_fused_done) {
        const int out_shape[3] = {1, n_tokens, m->cfg.hidden_size};
        ok = mlx_reshape(&moe_out, summed, out_shape, 3, m->stream) == 0;
        CHK(ok, "moe_out reshape");
    }    }


    if (ok) { ok = mlx_add(out, h, moe_out, m->stream) == 0; CHK(ok, "final residual add"); }

    /* Q2-b: accumulo imatrix — energia di gate/up (input normed2) e down (riga `act`). */
    if (m->im.active && ok && im_expert && im_src && im_rows > 0 && act.ctx != NULL)
        llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, im_expert, im_src, im_rows);
    free(im_expert); free(im_src);
    mlx_array all[] = {
        ln2_w, normed2, gate_w, gates, part, top_inds,
        gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
        down_proj_w, down_proj_s, down_proj_b, moe_out,
        no_lhs, top_inds_arr, x1, gate_out, up_out, sig, silu, act, down_out,
        scores_arr, scaled, summed, probs,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
