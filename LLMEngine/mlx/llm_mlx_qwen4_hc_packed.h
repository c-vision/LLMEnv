/* llm_mlx_qwen4_hc_packed.h — hyper-connection qwen4 su pesi PACKED zero-copy.
 * Stessa formula di qwen4_exp_gated_residual_q, con down/up/inject letti dai descrittori.
 * 0 = calcolato; 1 = non gestito (il chiamante usa il percorso MLX con gli array). */
#ifndef LLM_MLX_QWEN4_HC_PACKED_H
#define LLM_MLX_QWEN4_HC_PACKED_H

#include "llm_mlx_internal.h"
#include "llm_mlx_weights_backend.h"
#include "llm_mlx_linear_packed_model.h"
#include "llm_mlx_hc_fused.h"
#include <pthread.h>

static int q4hc_linear(LlmMlxModel* m, int layer_idx, const char* hc, const char* leaf,
                       mlx_array x, mlx_array* out) {
    char base[256];
    { size_t n=strlen(m->cfg.layer_prefix);
      if (n && m->cfg.layer_prefix[n-1]=='.') snprintf(base,sizeof base,"%s%d.%s.%s", m->cfg.layer_prefix, layer_idx, hc, leaf);
      else snprintf(base,sizeof base,"%s.%d.%s.%s", m->cfg.layer_prefix, layer_idx, hc, leaf); }
    /* registry + nodo lazy (nessun eval, nessuna riapertura del file a ogni chiamata) */
    return llm_mlx_linear_packed_model(&m->packed, m->model_dir, base, 4, m->cfg.quant_group_size, x, out, m->stream) == 0 ? 0 : 1;
}

/* Cache f32 dei pesi hc_norm per (layer, hc): evita un astype per chiamata (l'array valutato resta riusabile). */
typedef struct { int layer; char hc[32]; mlx_array w32; } Q4hcW;
static Q4hcW* q4hc_wtab; static int q4hc_wn, q4hc_wcap;
static pthread_mutex_t q4hc_mu = PTHREAD_MUTEX_INITIALIZER;

static int q4hc_w32(LlmMlxModel* m, int layer_idx, const char* hc, mlx_array w, mlx_array* out) {
    pthread_mutex_lock(&q4hc_mu);
    for (int i = 0; i < q4hc_wn; i++) if (q4hc_wtab[i].layer == layer_idx && strcmp(q4hc_wtab[i].hc, hc) == 0) { int r = mlx_array_set(out, q4hc_wtab[i].w32); pthread_mutex_unlock(&q4hc_mu); return r; }
    mlx_array w32 = mlx_array_new();
    if (mlx_astype(&w32, w, MLX_FLOAT32, m->stream) != 0) { mlx_array_free(w32); pthread_mutex_unlock(&q4hc_mu); return -1; }
    { mlx_vector_array ev = mlx_vector_array_new(); mlx_vector_array_append_value(ev, w32); mlx_eval(ev); mlx_vector_array_free(ev); }
    if (q4hc_wn >= q4hc_wcap) { int nc = q4hc_wcap ? q4hc_wcap * 2 : 128; Q4hcW* nt = (Q4hcW*)realloc(q4hc_wtab, (size_t)nc * sizeof *nt); if (!nt) { mlx_array_free(w32); pthread_mutex_unlock(&q4hc_mu); return -1; } q4hc_wtab = nt; q4hc_wcap = nc; }
    q4hc_wtab[q4hc_wn].layer = layer_idx; snprintf(q4hc_wtab[q4hc_wn].hc, sizeof q4hc_wtab[q4hc_wn].hc, "%s", hc); q4hc_wtab[q4hc_wn].w32 = w32; q4hc_wn++;
    int r = mlx_array_set(out, w32);
    pthread_mutex_unlock(&q4hc_mu);
    return r;
}

static LlmMlxPackedEntry* q4hc_entry(LlmMlxModel* m, int layer_idx, const char* hc, const char* leaf) {
    char base[256];
    size_t n = strlen(m->cfg.layer_prefix);
    if (n && m->cfg.layer_prefix[n-1] == '.') snprintf(base, sizeof base, "%s%d.%s.%s", m->cfg.layer_prefix, layer_idx, hc, leaf);
    else snprintf(base, sizeof base, "%s.%d.%s.%s", m->cfg.layer_prefix, layer_idx, hc, leaf);
    LlmMlxPackedEntry* e = llm_mlx_packed_reg_entry_cfg(&m->packed, m->model_dir, base, m->cfg.quant_bits, m->cfg.quant_group_size);
    if (!e || (e->w.bits != 4 && e->w.bits != 2) || llm_mlx_lazy_build(&e->lazy, &e->w) != 0) return NULL;
    return e;
}

/* Combine precedente in sospeso: la HC successiva ne ricostruisce l'uscita (residual + sub*inj) dentro il proprio kernel invece di aspettare il
 * nodo di combine, che resta fuori dal percorso critico. Identita' = stesso handle (ctx) + sequenza (layer, tipo) attesa. */
typedef struct { int valid, next_layer, next_kind; void* out_ctx; mlx_array sub, res, inj; } Q4hcPend;
static Q4hcPend g_q4hc_pend;

/* Percorso fuso (3 nodi lazy). 0 = fatto; 1 = non gestito (nessun effetto). */
static int q4hc_try_fused(LlmMlxModel* m, int layer_idx, const char* hc, mlx_array hyper_input, mlx_array hc_norm_w,
                          mlx_array* out_mixed, mlx_array* out_residual, mlx_array* out_injection) {
    if (m->cfg.bfloat16_input || m->cfg.hc_count != 4 || mlx_array_dtype(hyper_input) != MLX_FLOAT32) return 1;
    const int nd = mlx_array_ndim(hyper_input), hcn = m->cfg.hc_count, h = m->cfg.hidden_size, W = hcn * h;
    if ((nd != 2 && nd != 3) || (int)mlx_array_dim(hyper_input, nd - 1) != W) return 1;
    int nt = 1; for (int d = 0; d < nd - 1; d++) nt *= (int)mlx_array_dim(hyper_input, d);
    LlmMlxPackedEntry *ed = q4hc_entry(m, layer_idx, hc, "input_mix_weight_down"), *eu = q4hc_entry(m, layer_idx, hc, "input_mix_weight_up"),
                      *ei = q4hc_entry(m, layer_idx, hc, "block_inject_weight");
    if (!ed || !eu || !ei) return 1;
    const int kind = (strcmp(hc, "mlp_hyper_connection") == 0) ? 1 : 0;
    mlx_array pb = mlx_array_new(), ps = mlx_array_new(), pi = mlx_array_new();
    int pend = 0;
    pthread_mutex_lock(&q4hc_mu);
    if (g_q4hc_pend.valid && g_q4hc_pend.next_layer == layer_idx && g_q4hc_pend.next_kind == kind && g_q4hc_pend.out_ctx == hyper_input.ctx) {
        pend = (mlx_array_set(&pb, g_q4hc_pend.res) == 0 && mlx_array_set(&ps, g_q4hc_pend.sub) == 0 && mlx_array_set(&pi, g_q4hc_pend.inj) == 0);
    }
    if (g_q4hc_pend.valid) { mlx_array_free(g_q4hc_pend.sub); mlx_array_free(g_q4hc_pend.res); mlx_array_free(g_q4hc_pend.inj); g_q4hc_pend.valid = 0; }   // consumata o scaduta
    pthread_mutex_unlock(&q4hc_mu);
    mlx_array hw32 = mlx_array_new(), flat = mlx_array_new(), mixed = mlx_array_new(), inj = mlx_array_new();
    int rc = 1;
    int sh2[2] = { nt, W };
    if (q4hc_w32(m, layer_idx, hc, hc_norm_w, &hw32) != 0 || mlx_reshape(&flat, hyper_input, sh2, 2, m->stream) != 0) goto out;
    if (llm_hc_mix_fused(&mixed, &inj, pend ? pb : flat, pend ? ps : flat, pend ? pi : flat, pend, nt, hcn, h, hw32, m->cfg.rms_norm_eps,
                         &ed->lazy, &ed->w, &eu->lazy, &eu->w, &ei->lazy, &ei->w, m->stream) != 0) goto out;
    if (nd == 3) { int osh[3] = { (int)mlx_array_dim(hyper_input, 0), (int)mlx_array_dim(hyper_input, 1), h }; if (mlx_reshape(out_mixed, mixed, osh, 3, m->stream) != 0) goto out; }
    else if (mlx_array_set(out_mixed, mixed) != 0) goto out;
    if (out_residual && mlx_array_set(out_residual, hyper_input) != 0) goto out;
    if (out_injection && mlx_array_set(out_injection, inj) != 0) goto out;
    rc = 0;
out:
    mlx_array_free(hw32); mlx_array_free(flat); mlx_array_free(mixed); mlx_array_free(inj);
    mlx_array_free(pb); mlx_array_free(ps); mlx_array_free(pi);
    return rc;
}

/* residual + sub (x) injection in un kernel. 0 = fatto; 1 = non gestito. */
static int llm_mlx_hc_combine_packed(mlx_array* out, mlx_array sub, mlx_array residual, mlx_array injection, LlmMlxModel* m, int layer_idx, int after_ffn) {
    if (!llm_mlx_weights_backend_packed() || m->cfg.bfloat16_input || m->cfg.hc_count != 4) return 1;
    if (mlx_array_dtype(sub) != MLX_FLOAT32 || mlx_array_dtype(residual) != MLX_FLOAT32 || mlx_array_dtype(injection) != MLX_FLOAT32) return 1;
    const int hcn = m->cfg.hc_count, h = m->cfg.hidden_size, W = hcn * h;
    const size_t inj_n = mlx_array_size(injection), res_n = mlx_array_size(residual), sub_n = mlx_array_size(sub);
    if (inj_n == 0 || inj_n % 4 != 0) return 1;
    const int nt = (int)(inj_n / 4);
    if (res_n != (size_t)nt * (size_t)W || sub_n != (size_t)nt * (size_t)h) return 1;
    mlx_array tmp = mlx_array_new();
    int rc = llm_hc_combine_fused(&tmp, sub, residual, injection, nt, hcn, h, m->stream);
    if (rc == 0) {
        const int nd = mlx_array_ndim(residual); int sh[8];
        for (int d = 0; d < nd && d < 8; d++) sh[d] = (int)mlx_array_dim(residual, d);
        rc = mlx_reshape(out, tmp, sh, nd, m->stream);
        if (rc == 0) {   // in sospeso per la HC successiva (dopo attn: mlp dello stesso layer; dopo ffn: attn del layer+1)
            pthread_mutex_lock(&q4hc_mu);
            if (g_q4hc_pend.valid) { mlx_array_free(g_q4hc_pend.sub); mlx_array_free(g_q4hc_pend.res); mlx_array_free(g_q4hc_pend.inj); g_q4hc_pend.valid = 0; }
            g_q4hc_pend.sub = mlx_array_new(); g_q4hc_pend.res = mlx_array_new(); g_q4hc_pend.inj = mlx_array_new();
            if (mlx_array_set(&g_q4hc_pend.sub, sub) == 0 && mlx_array_set(&g_q4hc_pend.res, residual) == 0 && mlx_array_set(&g_q4hc_pend.inj, injection) == 0) {
                g_q4hc_pend.next_layer = after_ffn ? layer_idx + 1 : layer_idx; g_q4hc_pend.next_kind = after_ffn ? 0 : 1; g_q4hc_pend.out_ctx = out->ctx; g_q4hc_pend.valid = 1;
            } else { mlx_array_free(g_q4hc_pend.sub); mlx_array_free(g_q4hc_pend.res); mlx_array_free(g_q4hc_pend.inj); }
            pthread_mutex_unlock(&q4hc_mu);
        }
    }
    mlx_array_free(tmp);
    return rc == 0 ? 0 : 1;
}

/* out_injection opzionale: passa NULL per saltarla (come ctx==NULL nel percorso MLX). */
static int qwen4_hc_packed_try(LlmMlxModel* m, int layer_idx, const char* hc,
                               mlx_array hyper_input, mlx_array hc_norm_w,
                               mlx_array* out_mixed, mlx_array* out_residual, mlx_array* out_injection) {
    if (!m || !hc || !out_mixed) return 1;
    if (!llm_mlx_weights_backend_packed() || (m->cfg.quant_bits != 4 && m->cfg.quant_bits != 2)) return 1;
    if (!m->cfg.model_type[0] || !strstr(m->cfg.model_type, "qwen4")) return 1;
    const int nd = mlx_array_ndim(hyper_input);
    if (nd != 2 && nd != 3) return 1;
    if (q4hc_try_fused(m, layer_idx, hc, hyper_input, hc_norm_w, out_mixed, out_residual, out_injection) == 0) return 0;
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
        if (q4hc_linear(m, layer_idx, hc, "block_inject_weight", normed, &inj) != 0) { mlx_array_free(inj); goto fail_sig; }
        {
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
