/* llm_dsv4_gpu.h — DeepSeek V4 Flash (DS4) GPU: blocco mHC (hyper-connection) su MLX,
 * header-only, GPU attiva di default (nessuna variabile d'ambiente; fallback al path C se assente).
 *
 * REGOLA §0bis (una sola sorgente): la semantica mHC/Sinkhorn/RMSNorm di DS4 e' IDENTICA a
 * DeepSeek V4.1 — lo dichiara gia' `llm_dsv4_block.h` ("riusa la matematica di deepseek_v41").
 * Qui NON si reimplementa nulla: si chiamano le primitive MLX gia' esistenti di
 * `llm_dsv41_mlx.h` (`llm_dsv41_mlx_hc_mixes`, `llm_dsv41_mlx_hc_pre_norm`, `llm_dsv41_mlx_hc_post`)
 * e i kernel Metal opt-in `llm_dsv41_metal_*` (sinkhorn hc==4), esattamente come il path v4.1.
 *
 * Fallback sicuro: ogni funzione ritorna `false` se la GPU e' disabilitata o se una primitiva
 * fallisce (nessuno stub, nessun risultato silenzioso: il chiamante prosegue col path C).
 *
 * F1 (questo file): mHC pre/post del layer. F2: attenzione (CSA/HCA+indexer). F3: MoE
 * (esperti IQ2_XXS/Q2_K -> kernel Metal via mlx_fast_metal_kernel). F4: sessione+gating.
 */
#ifndef LLM_DSV4_GPU_H
#define LLM_DSV4_GPU_H

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <mlx/c/mlx.h>

#include "llm_dsv4.h"          /* LlmDsv4Config */
#include "llm_dsv41_mlx.h"     /* primitive MLX v4.1 riusate (mHC/Sinkhorn/RMSNorm) */
#include "llm_engine_opts.h"   /* parametro `gpu` dal registro (niente env) */

static inline int llm_dsv4_gpu_enabled(void) {
    /* GPU AUTOMATICA: attiva se Metal e' disponibile. Nessun flag/parametro utente. */
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool av = false;
    mlx_device_is_available(&av, gpu);
    mlx_device_free(gpu);
    return av ? 1 : 0;
}

/* Shim: LlmDsv41Config con i SOLI campi letti da llm_dsv41_mlx_hc_* (mHC identico v4/v4.1). */
static inline LlmDsv41Config llm_dsv4_gpu_c41(const LlmDsv4Config* c) {
    LlmDsv41Config r;
    memset(&r, 0, sizeof r);
    r.hc_mult = c->n_hc;
    r.norm_eps = c->rms_eps;                       /* v4 usa rms_eps nel rms del mHC */
    r.hc_eps = c->hc_eps;
    r.hc_sinkhorn_iters = c->n_hc_sinkhorn_iter;
    return r;
}

/* mHC pre (GPU) — equivalente di `llm_dsv4_block_pre` (llm_dsv4_block.h).
 * residual_hc [HC*D], fn [SIZE,HCD], scale [3], base [SIZE], norm_w [D]  (SIZE=(2+HC)*HC, HCD=HC*D)
 * Output: x [D], post [HC], comb [HC*HC]. Ritorna false -> il chiamante usa il path C. */
static inline bool llm_dsv4_gpu_block_pre(const LlmDsv4Config* c, const float* fn,
        const float* scale, const float* base, const float* norm_w, const float* residual_hc,
        mlx_stream s, float* out_x, float* post_out, float* comb_out) {
    if (!llm_dsv4_gpu_enabled()) return false;
    const int HC = c->n_hc, D = c->n_embd, SIZE = (2 + HC) * HC, HCD = HC * D;
    LlmDsv41Config c41 = llm_dsv4_gpu_c41(c);
    const int hsh[4] = { 1, 1, HC, D };
    const int fsh[2] = { SIZE, HCD };
    const int ssh[1] = { 3 };
    const int bsh[1] = { SIZE };
    const int wsh[1] = { D };
    bool ok = true;
    mlx_array h   = mlx_array_new_data(residual_hc, hsh, 4, MLX_FLOAT32);
    mlx_array fna = mlx_array_new_data(fn, fsh, 2, MLX_FLOAT32);
    mlx_array sca = mlx_array_new_data(scale, ssh, 1, MLX_FLOAT32);
    mlx_array bsa = mlx_array_new_data(base, bsh, 1, MLX_FLOAT32);
    mlx_array wa  = mlx_array_new_data(norm_w, wsh, 1, MLX_FLOAT32);
    mlx_array pre = mlx_array_new(), post = mlx_array_new(), comb = mlx_array_new();
    mlx_array x = mlx_array_new();
    llm_dsv41_mlx_hc_mixes(h, fna, sca, bsa, &c41, &pre, &post, &comb, s, &ok);
    if (ok) { mlx_array_free(x); x = llm_dsv41_mlx_hc_pre_norm(h, pre, wa, c->rms_eps, s, &ok); }
    if (ok) {
        mlx_array_eval(x); mlx_array_eval(post); mlx_array_eval(comb);
        const float* xp = mlx_array_data_float32(x);
        const float* pp = mlx_array_data_float32(post);
        const float* cp = mlx_array_data_float32(comb);
        if (xp) for (int i = 0; i < D; i++) out_x[i] = xp[i]; else ok = false;
        if (pp) for (int i = 0; i < HC; i++) post_out[i] = pp[i]; else ok = false;
        if (cp) for (int i = 0; i < HC * HC; i++) comb_out[i] = cp[i]; else ok = false;
    }
    mlx_array_free(h); mlx_array_free(fna); mlx_array_free(sca); mlx_array_free(bsa);
    mlx_array_free(wa); mlx_array_free(pre); mlx_array_free(post); mlx_array_free(comb);
    mlx_array_free(x);
    return ok;
}

/* mHC post (GPU) — equivalente di `llm_dsv4_block_post` (llm_dsv4_block.h).
 * block_out [D], residual_hc [HC*D], post [HC], comb [HC*HC] -> out_hc [HC*D]. */
static inline bool llm_dsv4_gpu_block_post(const LlmDsv4Config* c, const float* block_out,
        const float* residual_hc, const float* post, const float* comb,
        mlx_stream s, float* out_hc) {
    if (!llm_dsv4_gpu_enabled()) return false;
    const int HC = c->n_hc, D = c->n_embd;
    const int xsh[3] = { 1, 1, D };
    const int rsh[4] = { 1, 1, HC, D };
    const int psh[3] = { 1, 1, HC };
    const int csh[4] = { 1, 1, HC, HC };
    bool ok = true;
    mlx_array x   = mlx_array_new_data(block_out, xsh, 3, MLX_FLOAT32);
    mlx_array res = mlx_array_new_data(residual_hc, rsh, 4, MLX_FLOAT32);
    mlx_array po  = mlx_array_new_data(post, psh, 3, MLX_FLOAT32);
    mlx_array co  = mlx_array_new_data(comb, csh, 4, MLX_FLOAT32);
    mlx_array out = llm_dsv41_mlx_hc_post(x, res, po, co, s, &ok);
    if (ok) {
        mlx_array_eval(out);
        const float* op = mlx_array_data_float32(out);
        if (op) for (int i = 0; i < HC * D; i++) out_hc[i] = op[i]; else ok = false;
    }
    mlx_array_free(x); mlx_array_free(res); mlx_array_free(po); mlx_array_free(co);
    mlx_array_free(out);
    return ok;
}

#endif /* LLM_DSV4_GPU_H */
