/* llm_mlx_qwen4_hc_packed.h — hyper-connection qwen4 su pesi PACKED zero-copy.
 * Stessa formula di qwen4_exp_gated_residual_q, con down/up/inject letti dai descrittori.
 * 0 = calcolato; 1 = non gestito (il chiamante usa il percorso MLX con gli array). */
#ifndef LLM_MLX_QWEN4_HC_PACKED_H
#define LLM_MLX_QWEN4_HC_PACKED_H

#include "llm_mlx_internal.h"
#include "llm_mlx_weights_backend.h"
#include "llm_mlx_qwen4_moe_packed.h"   /* q4mp_shard_of/q4mp_eval */

static int q4hc_linear(LlmMlxModel* m, int layer_idx, const char* hc, const char* leaf,
                       mlx_array x, mlx_array* out) {
    const int g = m->cfg.quant_group_size;
    char base[256], path[1600];
    { size_t n=strlen(m->cfg.layer_prefix);
      if (n && m->cfg.layer_prefix[n-1]=='.') snprintf(base,sizeof base,"%s%d.%s.%s", m->cfg.layer_prefix, layer_idx, hc, leaf);
      else snprintf(base,sizeof base,"%s.%d.%s.%s", m->cfg.layer_prefix, layer_idx, hc, leaf); }
    q4mp_shard_of(m, base, path, sizeof path);
    LlmMlxPackedLinear L;
    if (llm_mlx_linear_packed_open(path, base, 4, g, &L) != 0) { fprintf(stderr, "[hc-packed] open FALLITO %s (path=%s)\n", base, path); return 1; }
    int rc = llm_mlx_linear_packed_run(&L, x, out, m->stream);
    llm_mlx_linear_packed_close(&L);
    return rc == 0 ? 0 : 1;
}

/* out_injection opzionale: passa NULL per saltarla (come ctx==NULL nel percorso MLX). */
static int qwen4_hc_packed_try(LlmMlxModel* m, int layer_idx, const char* hc,
                               mlx_array hyper_input, mlx_array hc_norm_w,
                               mlx_array* out_mixed, mlx_array* out_residual, mlx_array* out_injection) {
    if (!m || !hc || !out_mixed) return 1;
    if (!llm_mlx_weights_backend_packed() || m->cfg.quant_bits != 4) return 1;
    if (!m->cfg.model_type[0] || !strstr(m->cfg.model_type, "qwen4")) return 1;
    const int nd = mlx_array_ndim(hyper_input);
    if (nd != 2 && nd != 3) return 1;
    const int hc_count = m->cfg.hc_count, hidden = m->cfg.hidden_size;
    const int* hs = mlx_array_shape(hyper_input);
    if (!hs || hs[nd-1] != hc_count * hidden) return 1;
    const int h = hidden, hcn = hc_count;
    int nrow = 1; for (int d = 0; d < nd-1; d++) nrow *= hs[d];

    mlx_array flat = mlx_array_new();
    if (nd == 3) { int sh2[2] = {nrow, hcn*h}; if (mlx_reshape(&flat, hyper_input, sh2, 2, m->stream) != 0) { mlx_array_free(flat); return 1; } }
    else if (mlx_array_set(&flat, hyper_input) != 0) { mlx_array_free(flat); return 1; }
    mlx_array normed = mlx_array_new();
    if (!llm_mlx_zero_centered_rmsnorm(&normed, flat, hc_norm_w, h, m->cfg.rms_norm_eps, m->stream)) { mlx_array_free(flat); mlx_array_free(normed); return 1; }
    mlx_array down = mlx_array_new();
    if (q4hc_linear(m, layer_idx, hc, "input_mix_weight_down", normed, &down) != 0) { mlx_array_free(normed); mlx_array_free(down); return 1; }
    mlx_array hcf = mlx_array_new_float32((float)hcn);
    mlx_array dsc = mlx_array_new();
    if (mlx_divide(&dsc, down, hcf, m->stream) != 0) goto fail;
    mlx_array dsilu = mlx_array_new();
    if (!llm_mlx_qwen4exp_silu(&dsilu, dsc, m->stream)) { mlx_array_free(dsilu); goto fail; }
    mlx_array up = mlx_array_new();
    if (q4hc_linear(m, layer_idx, hc, "input_mix_weight_up", dsilu, &up) != 0) { mlx_array_free(dsilu); mlx_array_free(up); goto fail; }
    mlx_array sig = mlx_array_new();
    if (mlx_sigmoid(&sig, up, m->stream) != 0) goto fail_sig;

    /* [1, hc*hidden] -> [1, hc, hidden] per mix e normed, poi media sull'asse hc. */
    { mlx_array mix_r = mlx_array_new(), normed_r = mlx_array_new(), prod = mlx_array_new();
      int sh[3] = {nrow, hcn, h};
      int rc = mlx_reshape(&mix_r, sig, sh, 3, m->stream);
      if (rc == 0) rc = mlx_reshape(&normed_r, normed, sh, 3, m->stream);
      if (rc == 0) rc = mlx_multiply(&prod, mix_r, normed_r, m->stream);
      mlx_array mean2 = mlx_array_new();
      if (rc == 0) rc = mlx_mean_axis(&mean2, prod, -2, false, m->stream);
      if (rc == 0) {
          if (nd == 3) { int osh[3] = {hs[0], hs[1], h}; rc = mlx_reshape(out_mixed, mean2, osh, 3, m->stream); }
          else rc = mlx_array_set(out_mixed, mean2);
      }
      mlx_array_free(mean2);
      mlx_array_free(mix_r); mlx_array_free(normed_r); mlx_array_free(prod);
      if (rc != 0) goto fail_sig;
    }

    if (out_residual && out_injection) {
        mlx_array inj = mlx_array_new();
        if (q4hc_linear(m, layer_idx, hc, "block_inject_weight", normed, &inj) == 0) {
            mlx_array isc = mlx_array_new(), isig = mlx_array_new(), two = mlx_array_new_float32(2.0f);
            int rc = mlx_divide(&isc, inj, hcf, m->stream);
            if (rc == 0) rc = mlx_sigmoid(&isig, isc, m->stream);
            if (rc == 0) rc = mlx_multiply(out_injection, isig, two, m->stream);
            if (rc == 0) rc = mlx_array_set(out_residual, hyper_input);
            mlx_array_free(isc); mlx_array_free(isig); mlx_array_free(two);
            if (rc != 0) { mlx_array_free(inj); goto fail_sig; }
        }
        mlx_array_free(inj);
    }

    mlx_array_free(flat); mlx_array_free(normed); mlx_array_free(down); mlx_array_free(hcf);
    mlx_array_free(dsc); mlx_array_free(dsilu); mlx_array_free(up); mlx_array_free(sig);
    return 0;
fail_sig:
    mlx_array_free(sig);
fail:
    mlx_array_free(flat); mlx_array_free(normed); mlx_array_free(down); mlx_array_free(hcf);
    mlx_array_free(dsc); mlx_array_free(dsilu); mlx_array_free(up);
    return 1;
}

#endif
