/* llm_mlx_dflash2.c — dominio DFlash2 (forward nativo del drafter block-diffusion per
 * Qwen3.8-27B-DFlash2). Estratto VERBATIM da llmengine_mlx.c (split fase 2).
 * Contiene: helper (df2_linear/rms/dbg/dump/convolve), DFlashAttention (df2_attn_layer),
 * DFlash2DecoderLayer (df2_layer), backbone, CandidateSelector (df2_selector) e i timer
 * g_df2_* condivisi con l'API esportata rimasta nel monolite. */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "llm_mlx_swiglu_kernel.h"
/* ============================================================================
 * DFlash2 (Qwen3.8-27B-DFlash2) -- forward nativo del drafter block-diffusion.
 *
 * Sezione aggiunta in coda al translation unit: implementa op-per-op il
 * reference python
 *   Rapid-MLX-NextGen vllm_mlx/speculative/dflash/drafters/qwen3_dflash.py
 * (classi DFlashAttention, GroupedDynamicCausalConv, DFlash2DecoderLayer,
 * CandidateSelector, DFlashDraftModel).
 *
 * Dtype: i pesi draft sono BF16 e NON vengono mai castati; embed/lm_head sono
 * del TARGET (quant affine 8-bit) e l'embed produce BF16 (come il reference:
 * dequant con optional_dtype BF16). Gli hidden del draft girano in BF16 per
 * tutto il backbone (stesse op MLX di Python => stessi bit). Proiezioni draft:
 * sempre df2_linear (x @ w.T senza astype), MAI quantized_linear_cfg/plain.
 *
 * Reference "round": block = [anchor] + mask*(block_size-1); la cache ctx per
 * layer accumula le S righe di x_ctx (kv-injection) e ctx_pos avanza di S.
 * ============================================================================ */

/* x @ w.T SENZA astype: i pesi BF16 del draft non si castano (df2_linear e'
 * diverso da plain_linear, che promuove w a float32). */
static bool df2_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_stream stream) {
    mlx_array wt = mlx_array_new();
    bool ok = mlx_transpose(&wt, w, stream) == 0;
    if (ok) ok = mlx_matmul(res, x, wt, stream) == 0;
    mlx_array_free(wt);
    return ok;
}

/* RMSNorm diretta (mlx_fast_rms_norm, MAI rms_norm_cfg che casta a bf16/f32). */
static bool df2_rms(mlx_array* res, mlx_array x, mlx_array w, float eps, mlx_stream stream) {
    return mlx_fast_rms_norm(res, x, w, eps, stream) == 0;
}

/* Debug temporaneo (validazione): stampa min/max/mean + primi 8 valori fp32. */
static void df2_dbg(mlx_array a, const char* tag, mlx_stream st) {
    if (getenv("DF2_DEBUG") == NULL) return;

    mlx_array h = mlx_array_new();
    if (mlx_astype(&h, a, MLX_FLOAT32, st) != 0) { mlx_array_free(h); return; }
    if (mlx_array_eval(h) != 0) { mlx_array_free(h); return; }
    const float* d = mlx_array_data_float32(h);
    if (!d) { mlx_array_free(h); return; }
    size_t n = mlx_array_size(h);
    if (n > 65536) n = 65536;
    double mn = 1e30, mx = -1e30, mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (d[i] < mn) mn = d[i];
        if (d[i] > mx) mx = d[i];
        mean += d[i];
    }
    fprintf(stderr, "[df2dbg] %-24s n=%zu min=%g max=%g mean=%g |", tag, mlx_array_size(h), mn, mx, mean / (n ? (double)n : 1));
    size_t p = mlx_array_size(h) > 8 ? 8 : mlx_array_size(h);
    for (size_t i = 0; i < p; i++) fprintf(stderr, " %.5g", d[i]);
    fprintf(stderr, "\n");
    mlx_array_free(h);
}

/* Dump binario temporaneo (DF2_DUMP=/tmp/df2dir): scrive l'array fp32 raw per
 * confronto bit-exact col reference python (stesso tag, file c_<tag>.bin). */
static void df2_dump(mlx_array a, const char* tag, mlx_stream st) {
    const char* dir = getenv("DF2_DUMP");
    if (dir == NULL) return;
    mlx_array h = mlx_array_new();
    if (mlx_astype(&h, a, MLX_FLOAT32, st) != 0) { mlx_array_free(h); return; }
    if (mlx_array_eval(h) != 0) { mlx_array_free(h); return; }
    const float* d = mlx_array_data_float32(h);
    if (!d) { mlx_array_free(h); return; }
    char path[1024];
    snprintf(path, sizeof(path), "%s/c_%s.bin", dir, tag);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(d, sizeof(float), mlx_array_size(h), f); fclose(f); }
    mlx_array_free(h);
}

/* Grouped dynamic causal conv a due tap -- reference `_grouped_dynamic_convolve`:
 *   hidden : [1, L, H]
 *   dyn    : [1, L, kern, groups]     (dinamico gia' proiettato)
 *   base   : [kern, H]                (base_kernel[0] per prepare, [1] per finish)
 *   output = base[off]*x[t-off] + dyn[t,off]*x[t-off], per tap, causale. */
static bool df2_convolve(mlx_array* res, mlx_array hidden, mlx_array dyn,
                         mlx_array base, int group_size, mlx_stream stream) {
    /* FUSIONE (default ON): il path eager costruisce ~18 op MLX per chiamata e questa funzione
     * e' chiamata 4x per layer x 5 layer = 20x per round. Il round DFlash2 e' latency-bound
     * (GPU 7%: 42% del CPU in attesa GPU) -> un solo dispatch al posto di ~18. */
    bool ok = true;
    int L = mlx_array_shape(hidden)[1];
    int H = mlx_array_shape(hidden)[2];
    int kern = mlx_array_shape(base)[0];
    int groups = H / group_size;
    static int conv_dump_done = 0;
    const bool conv_dbg = getenv("DF2_DUMP") != NULL && !conv_dump_done;

    /* blocks = hidden.reshape(B,L,groups,group) */
    mlx_array blocks = mlx_array_new();
    {
        const int sh[4] = {1, L, groups, group_size};
        ok = mlx_reshape(&blocks, hidden, sh, 4, stream) == 0;
    }
    /* dynamic = dyn.reshape(B,L,kern,groups,1) */
    mlx_array dyn5 = mlx_array_new();
    if (ok) {
        const int sh[5] = {1, L, kern, groups, 1};
        ok = mlx_reshape(&dyn5, dyn, sh, 5, stream) == 0;
    }
    mlx_array output = mlx_array_new();
    if (ok) ok = mlx_zeros_like(&output, blocks, stream) == 0;
    if (conv_dbg) {
        df2_dump(blocks, "cv_blocks", stream);
        df2_dump(dyn5, "cv_dyn5", stream);
        df2_dump(base, "cv_base", stream);
    }

    for (int off = 0; ok && off < kern; off++) {
        /* values = blocks shiftate a destra di off lungo l'asse temporale */
        mlx_array values = mlx_array_new();
        if (off == 0) {
            ok = mlx_array_set(&values, blocks) == 0;
        } else {
            mlx_array zfull = mlx_array_new(), head0 = mlx_array_new(), tail = mlx_array_new();
            const int s0[4] = {0, 0, 0, 0}, t0[4] = {1, off, groups, group_size};
            /* tail = blocks[:, :-off] (righe 0..L-off-1), NON blocks[:, off:] -- lo
             * shift causale sposta gli elementi PRECEDENTI (x[t-offset]) in avanti. */
            const int s1[4] = {0, 0, 0, 0}, t1[4] = {1, L - off, groups, group_size};
            const int ones[4] = {1, 1, 1, 1};
            ok = mlx_zeros_like(&zfull, blocks, stream) == 0;
            if (ok) ok = mlx_slice(&head0, zfull, s0, 4, t0, 4, ones, 4, stream) == 0;
            if (ok) ok = mlx_slice(&tail, blocks, s1, 4, t1, 4, ones, 4, stream) == 0;
            if (ok) {
                mlx_vector_array vec = mlx_vector_array_new();
                mlx_vector_array_append_value(vec, head0);
                mlx_vector_array_append_value(vec, tail);
                ok = mlx_concatenate_axis(&values, vec, 1, stream) == 0;
                mlx_vector_array_free(vec);
            }
            mlx_array_free(zfull); mlx_array_free(head0); mlx_array_free(tail);
        }
        /* kernel = base[off].reshape(1,1,groups,group) (astype a hidden.dtype = no-op bf16) */
        mlx_array kernel = mlx_array_new();
        if (ok) {
            const int k0[2] = {off, 0}, k1[2] = {off + 1, H}, ko[2] = {1, 1};
            ok = mlx_slice(&kernel, base, k0, 2, k1, 2, ko, 2, stream) == 0;
            if (ok) {
                const int sh[4] = {1, 1, groups, group_size};
                ok = mlx_reshape(&kernel, kernel, sh, 4, stream) == 0;
            }
        }
        /* dynamic[:, :, off] -> [1,L,groups,1] */
        mlx_array tap = mlx_array_new();
        if (ok) {
            const int t0[5] = {0, 0, off, 0, 0}, t1[5] = {1, L, off + 1, groups, 1};
            const int ones[5] = {1, 1, 1, 1, 1};
            ok = mlx_slice(&tap, dyn5, t0, 5, t1, 5, ones, 5, stream) == 0;
            if (ok) {
                const int sh[4] = {1, L, groups, 1};
                ok = mlx_reshape(&tap, tap, sh, 4, stream) == 0;
            }
        }
        mlx_array kv = mlx_array_new(), dv = mlx_array_new();
        if (ok) ok = mlx_multiply(&kv, kernel, values, stream) == 0;
        if (ok) ok = mlx_multiply(&dv, tap, values, stream) == 0;
        /* output = ((output + kernel*values) + dynamic*values) -- ordine Python */
        if (ok) ok = mlx_add(&output, output, kv, stream) == 0;
        if (ok) ok = mlx_add(&output, output, dv, stream) == 0;
        if (conv_dbg) {
            char tag[64];
            snprintf(tag, sizeof(tag), "cv_k%d", off);
            df2_dump(kernel, tag, stream);
            snprintf(tag, sizeof(tag), "cv_dyn%d", off);
            df2_dump(tap, tag, stream);
            snprintf(tag, sizeof(tag), "cv_values%d", off);
            df2_dump(values, tag, stream);
            snprintf(tag, sizeof(tag), "cv_out%d", off);
            df2_dump(output, tag, stream);
        }
        mlx_array_free(values); mlx_array_free(kernel); mlx_array_free(tap);
        mlx_array_free(kv); mlx_array_free(dv);
    }
    if (ok) {
        const int sh[3] = {1, L, H};
        ok = mlx_reshape(res, output, sh, 3, stream) == 0;
    }
    if (conv_dbg) conv_dump_done = 1;
    mlx_array_free(blocks);
    mlx_array_free(dyn5);
    mlx_array_free(output);
    return ok;
}


/* Scambia i globali bf16 sul modo del draft e ritorna lo stato precedente nei tre out. */
void df2_mode_enter_draft(DFlash2Runtime* rt, bool* sv_all, mlx_dtype* sv_act) {
    *sv_all = g_bf16_all; *sv_act = g_act16;
    g_bf16_all = rt->drf_bf16_all; g_act16 = rt->drf_act16;
    rt->draft->cfg.bfloat16_input = rt->drf_bf16_input;
}
void df2_mode_leave_draft(DFlash2Runtime* rt, bool sv_all, mlx_dtype sv_act) {
    g_bf16_all = sv_all; g_act16 = sv_act;
    rt->target->cfg.bfloat16_input = rt->tgt_bf16_input;
}

/* Fetch embed_tokens e lm_head del target (nomi language_model.*). */
bool dflash_fetch_target_tensors(DFlash2Runtime* rt) {
    LlmMlxModel* m = rt->target;
    char name[288];
    bool ok = true;
    rt->embed_w = mlx_array_new(); rt->embed_s = mlx_array_new(); rt->embed_b = mlx_array_new();
    llm_mlx_mode_set(name, sizeof(name), "language_model.model.embed_tokens");
    ok = multishard_get_triple(&m->weights, name, &rt->embed_w, &rt->embed_s, &rt->embed_b);
    if (!ok) {
        char plain[320];
        snprintf(plain, sizeof(plain), "%s.weight", name);
        ok = multishard_get(&m->weights, plain, &rt->embed_w);
    }
    if (!ok) { fprintf(stderr, "dflash: embed_tokens target non trovato\n"); return false; }

    rt->lm_w = mlx_array_new(); rt->lm_s = mlx_array_new(); rt->lm_b = mlx_array_new();
    llm_mlx_mode_set(name, sizeof(name), "language_model.lm_head");
    rt->have_lm_head = multishard_get_triple(&m->weights, name, &rt->lm_w, &rt->lm_s, &rt->lm_b);
    if (!rt->have_lm_head) {
        char plain[320];
        snprintf(plain, sizeof(plain), "%s.weight", name);
        rt->have_lm_head = multishard_get(&m->weights, plain, &rt->lm_w);
        if (!rt->have_lm_head) {
            mlx_array_free(rt->lm_w); mlx_array_free(rt->lm_s); mlx_array_free(rt->lm_b);
            rt->lm_w = mlx_array_new(); rt->lm_s = mlx_array_new(); rt->lm_b = mlx_array_new();
        }
    }
    if (!rt->have_lm_head) { fprintf(stderr, "dflash: lm_head target non trovato\n"); return false; }
    return true;
}

/* Embed BF16 di n token dal target (dequant con optional_dtype BF16, come il
 * reference). Variante locale di embed_tokens_batch: dtype FORZATO BF16.
 * out: [1, n, hidden_target]. */
static bool df2_embed_batch(DFlash2Runtime* rt, const int32_t* token_ids, int n, mlx_array* out_x) {
    LlmMlxModel* m = rt->target;
    const bool quantized = (rt->embed_s.ctx != NULL && mlx_array_size(rt->embed_s) > 0);
    int e_group = m->cfg.quant_group_size, e_bits = m->cfg.quant_bits;
    char e_mode[16];
    llm_mlx_mode_set(e_mode, sizeof(e_mode), m->cfg.quant_mode);
    if (quantized)
        llm_mlx_quant_derive(rt->embed_w, rt->embed_s, e_group, e_bits, e_mode, &e_group, &e_bits, e_mode, sizeof(e_mode));
    const int idx_shape[1] = {n};
    mlx_array indices = mlx_array_new_data(token_ids, idx_shape, 1, MLX_INT32);
    mlx_array w_row = mlx_array_new(), s_row = mlx_array_new(), b_row = mlx_array_new();
    bool ok = mlx_take_axis(&w_row, rt->embed_w, indices, 0, m->stream) == 0;
    if (ok && quantized) {
        ok = mlx_take_axis(&s_row, rt->embed_s, indices, 0, m->stream) == 0;
        if (ok && rt->embed_b.ctx != NULL && mlx_array_size(rt->embed_b) > 0)
            ok = mlx_take_axis(&b_row, rt->embed_b, indices, 0, m->stream) == 0;
    }
    mlx_array x_flat = mlx_array_new();
    if (ok) {
        if (quantized) {
            mlx_optional_int gsz = {e_group, true};
            mlx_optional_int bts = {e_bits, true};
            mlx_optional_dtype dt = {MLX_BFLOAT16, true};
            ok = mlx_dequantize(&x_flat, w_row, s_row, b_row, gsz, bts, e_mode, g_mlx_empty_arr, dt, m->stream) == 0;
        } else {
            ok = mlx_astype(&x_flat, w_row, MLX_BFLOAT16, m->stream) == 0;
        }
    }
    if (ok) {
        const int x_shape[3] = {1, n, m->cfg.hidden_size};
        ok = mlx_reshape(out_x, x_flat, x_shape, 3, m->stream) == 0;
    }
    mlx_array_free(indices); mlx_array_free(w_row); mlx_array_free(s_row); mlx_array_free(b_row); mlx_array_free(x_flat);
    return ok;
}

/* ============================================================================
 * DFlashAttention di un layer draft. x e' l'input post conv-prepare del layer
 * ([1,L,H]); x_ctx la feature di contesto [1,S,H]. Aggiorna la cache ctx per
 * layer (rt->kcache/vcache). Ritorna l'output di o_proj [1,L,H] (PRE residual).
 * Reference DFlashAttention.__call__.
 * ============================================================================ */
static bool df2_attn_layer(DFlash2Runtime* rt, int layer_idx, int cur,
                           int L, int S, mlx_array x, mlx_array x_ctx,
                           mlx_array* out_o) {
    LlmMlxModel* d = rt->draft;
    const ModelConfig* c = &d->cfg;
    mlx_stream st = d->stream;
    char namebuf[256];
    bool ok = true;
    const int n_heads = c->n_heads, n_kv = c->n_kv_heads, hd = c->head_dim;
    const int rope_off_q = cur + S;

#define CHK_DF2(cond, what) do { if (!(cond)) { fprintf(stderr, "df2_attn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    mlx_array qw = mlx_array_new(), kw = mlx_array_new(), vw = mlx_array_new(), ow = mlx_array_new();
    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    tname(c->layer_prefix, layer_idx, "self_attn.q_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &qw);
    tname(c->layer_prefix, layer_idx, "self_attn.k_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &kw);
    tname(c->layer_prefix, layer_idx, "self_attn.v_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &vw);
    tname(c->layer_prefix, layer_idx, "self_attn.o_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &ow);
    tname(c->layer_prefix, layer_idx, "self_attn.q_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &qn_w);
    tname(c->layer_prefix, layer_idx, "self_attn.k_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &kn_w);
    if (!ok) CHK_DF2(false, "lookup pesi attenzione");

    mlx_array q = mlx_array_new(), k_ctx = mlx_array_new(), v_ctx = mlx_array_new();
    mlx_array k_prop = mlx_array_new(), v_prop = mlx_array_new();
    if (ok) ok = df2_linear(&q, x, qw, st);
    if (ok) ok = df2_linear(&k_ctx, x_ctx, kw, st);
    if (ok) ok = df2_linear(&v_ctx, x_ctx, vw, st);
    if (ok) ok = df2_linear(&k_prop, x, kw, st);
    if (ok) ok = df2_linear(&v_prop, x, vw, st);
    CHK_DF2(ok, "proiezioni q/k/v");

    mlx_array q_h = mlx_array_new(), kc_h = mlx_array_new(), vc_h = mlx_array_new();
    mlx_array kp_h = mlx_array_new(), vp_h = mlx_array_new();
    if (ok) {
        const int qsh[4] = {1, L, n_heads, hd};
        const int kvsh[4] = {1, S, n_kv, hd};
        const int kvpl[4] = {1, L, n_kv, hd};
        ok = mlx_reshape(&q_h, q, qsh, 4, st) == 0 &&
             mlx_reshape(&kc_h, k_ctx, kvsh, 4, st) == 0 &&
             mlx_reshape(&vc_h, v_ctx, kvsh, 4, st) == 0 &&
             mlx_reshape(&kp_h, k_prop, kvpl, 4, st) == 0 &&
             mlx_reshape(&vp_h, v_prop, kvpl, 4, st) == 0;
    }
    mlx_array qn = mlx_array_new(), kcn = mlx_array_new(), kpn = mlx_array_new();
    if (ok) ok = df2_rms(&qn, q_h, qn_w, c->rms_norm_eps, st);
    if (ok) ok = df2_rms(&kcn, kc_h, kn_w, c->rms_norm_eps, st);
    if (ok) ok = df2_rms(&kpn, kp_h, kn_w, c->rms_norm_eps, st);
    CHK_DF2(ok, "qk-norm");

    mlx_array q_t = mlx_array_new(), kc_t = mlx_array_new(), vc_t = mlx_array_new();
    mlx_array kp_t = mlx_array_new(), vp_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, qn, perm, 4, st) == 0 &&
             mlx_transpose_axes(&kc_t, kcn, perm, 4, st) == 0 &&
             mlx_transpose_axes(&vc_t, vc_h, perm, 4, st) == 0 &&
             mlx_transpose_axes(&kp_t, kpn, perm, 4, st) == 0 &&
             mlx_transpose_axes(&vp_t, vp_h, perm, 4, st) == 0;
    }
    mlx_array q_r = mlx_array_new(), kc_r = mlx_array_new(), kp_r = mlx_array_new();
    if (ok) {
        mlx_optional_float base = {c->rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_r, q_t, hd, /*traditional=*/false, base, /*scale=*/1.0f, rope_off_q, no_freqs, st) == 0 &&
             mlx_fast_rope(&kc_r, kc_t, hd, false, base, 1.0f, cur, no_freqs, st) == 0 &&
             mlx_fast_rope(&kp_r, kp_t, hd, false, base, 1.0f, rope_off_q, no_freqs, st) == 0;
        mlx_array_free(no_freqs);
    }
    CHK_DF2(ok, "rope");
    if (getenv("DF2_DEBUG") && layer_idx == 0) {
        df2_dbg(q, "attn.q", st);
        df2_dbg(qn, "attn.qn", st);
        df2_dbg(q_r, "attn.q_rope", st);
        df2_dbg(kc_r, "attn.kc_rope", st);
        df2_dbg(kp_r, "attn.kp_rope", st);
    }
    if (getenv("DF2_DUMP") && layer_idx == 0) {
        df2_dump(qn, "l0_qnorm", st);
        df2_dump(q_r, "l0_qrope", st);
        df2_dump(kc_r, "l0_kcrope", st);
        df2_dump(kp_r, "l0_kprope", st);
    }

    /* append ctx keys/values alla cache per layer (positions [cur, cur+S)) */
    mlx_array cache_full = mlx_array_new();
    if (ok) {
        if (cur == 0) {
            ok = mlx_array_set(&cache_full, kc_r) == 0;
        } else {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, rt->ctx[rt->cur_ctx].kcache[layer_idx]);
            mlx_vector_array_append_value(kv, kc_r);
            ok = mlx_concatenate_axis(&cache_full, kv, 2, st) == 0;
            mlx_vector_array_free(kv);
        }
    }
    mlx_array cachev_full = mlx_array_new();
    if (ok) {
        if (cur == 0) {
            ok = mlx_array_set(&cachev_full, vc_t) == 0;
        } else {
            mlx_vector_array vv = mlx_vector_array_new();
            mlx_vector_array_append_value(vv, rt->ctx[rt->cur_ctx].vcache[layer_idx]);
            mlx_vector_array_append_value(vv, vc_t);
            ok = mlx_concatenate_axis(&cachev_full, vv, 2, st) == 0;
            mlx_vector_array_free(vv);
        }
    }
    if (ok) ok = mlx_array_set(&rt->ctx[rt->cur_ctx].kcache[layer_idx], cache_full) == 0;
    if (ok) ok = mlx_array_set(&rt->ctx[rt->cur_ctx].vcache[layer_idx], cachev_full) == 0;

    mlx_array keys_all = mlx_array_new(), vals_all = mlx_array_new();
    if (ok) {
        mlx_vector_array kv = mlx_vector_array_new();
        mlx_vector_array_append_value(kv, cache_full);
        mlx_vector_array_append_value(kv, kp_r);
        ok = mlx_concatenate_axis(&keys_all, kv, 2, st) == 0;
        mlx_vector_array_free(kv);
    }
    if (ok) {
        mlx_vector_array vv = mlx_vector_array_new();
        mlx_vector_array_append_value(vv, cachev_full);
        mlx_vector_array_append_value(vv, vp_t);
        ok = mlx_concatenate_axis(&vals_all, vv, 2, st) == 0;
        mlx_vector_array_free(vv);
    }
    CHK_DF2(ok, "concat keys/values");

    /* SDPA NON causale (mask_mode=""): denoisa tutto il blocco in parallelo */
    mlx_array attn = mlx_array_new();
    if (ok) {
        float scale = 1.0f / sqrtf((float)hd);
        mlx_array no_mask = mlx_array_new();
        mlx_array no_sinks = mlx_array_new();
        ok = mlx_fast_scaled_dot_product_attention(&attn, q_r, keys_all, vals_all,
                                                   scale, "", no_mask, no_sinks, st) == 0;
        mlx_array_free(no_mask); mlx_array_free(no_sinks);
    }
    CHK_DF2(ok, "scaled_dot_product_attention");
    if (getenv("DF2_DEBUG") && layer_idx == 0) {
        df2_dbg(keys_all, "attn.keys_all", st);
        df2_dbg(vals_all, "attn.vals_all", st);
        df2_dbg(attn, "attn.sdpa_out", st);
    }
    if (getenv("DF2_DUMP") && layer_idx == 0) {
        df2_dump(keys_all, "l0_keys", st);
        df2_dump(vals_all, "l0_vals", st);
        df2_dump(attn, "l0_sdpa", st);
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, st) == 0;
        if (ok) {
            const int fsh[3] = {1, L, n_heads * hd};
            ok = mlx_reshape(&attn_flat, attn_bt, fsh, 3, st) == 0;
        }
    }
    if (getenv("DF2_DUMP") && layer_idx == 0) {
        df2_dump(attn_flat, "l0_attn_flat", st);
        df2_dump(ow, "l0_ow", st);
    }
    if (ok) ok = df2_linear(out_o, attn_flat, ow, st);
    CHK_DF2(ok, "o_proj");
    if (getenv("DF2_DEBUG") && layer_idx == 0) {
        df2_dbg(attn_flat, "op_attn_flat", st);
        df2_dbg(*out_o, "op_bf16", st);
        /* matmul fp32 esplicito (peso astype f32) per verificare il risultato bf16 */
        mlx_array wf = mlx_array_new(), wtf = mlx_array_new(), of = mlx_array_new();
        if (ok) ok = mlx_astype(&wf, ow, MLX_FLOAT32, st) == 0;
        if (ok) ok = mlx_transpose(&wtf, wf, st) == 0;
        if (ok) ok = mlx_matmul(&of, attn_flat, wtf, st) == 0;
        df2_dbg(of, "op_fp32", st);
        mlx_array_free(wf); mlx_array_free(wtf); mlx_array_free(of);
    }

    mlx_array all[] = {
        qw, kw, vw, ow, qn_w, kn_w, q, k_ctx, v_ctx, k_prop, v_prop,
        q_h, kc_h, vc_h, kp_h, vp_h, qn, kcn, kpn,
        q_t, kc_t, vc_t, kp_t, vp_t, q_r, kc_r, kp_r,
        cache_full, cachev_full, keys_all, vals_all, attn, attn_bt, attn_flat,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK_DF2
    return ok;
}

/* MLP denso SwiGLU (Qwen3MLP, pesi plain bf16): silu(gate)*up -> down. */
static bool df2_mlp_swiglu(DFlash2Runtime* rt, int layer_idx, mlx_array h, mlx_array* out) {
    LlmMlxModel* d = rt->draft;
    mlx_stream st = d->stream;
    char namebuf[256];
    bool ok = true;
    mlx_array gate_w = mlx_array_new(), up_w = mlx_array_new(), down_w = mlx_array_new();
    tname(d->cfg.layer_prefix, layer_idx, "mlp.gate_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &gate_w);
    tname(d->cfg.layer_prefix, layer_idx, "mlp.up_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &up_w);
    tname(d->cfg.layer_prefix, layer_idx, "mlp.down_proj.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &down_w);
    if (!ok) { fprintf(stderr, "df2_mlp[layer=%d]: lookup pesi mlp fallito\n", layer_idx); }

    const bool _mt = (getenv("DF2_PHASE_TIME") != NULL);
    struct timespec _m0,_m1; double _tg=0,_tu=0,_ts=0,_td=0;
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    if (ok) { if(_mt) clock_gettime(CLOCK_MONOTONIC,&_m0); ok = df2_linear(&gate_out, h, gate_w, st); }
    if (ok) { if(_mt) clock_gettime(CLOCK_MONOTONIC,&_m1); _tg=((_m1.tv_sec-_m0.tv_sec)*1e3+(_m1.tv_nsec-_m0.tv_nsec)/1e6); }
    if (ok) { if(_mt) clock_gettime(CLOCK_MONOTONIC,&_m0); ok = df2_linear(&up_out, h, up_w, st); }
    if (ok) { if(_mt) clock_gettime(CLOCK_MONOTONIC,&_m1); _tu=((_m1.tv_sec-_m0.tv_sec)*1e3+(_m1.tv_nsec-_m0.tv_nsec)/1e6); }
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    /* SWIGLU FUSO (default ON, kill-switch LLMMETAL_DF2_SWIGLU_METAL=0): il kernel Metal gia' in
     * produzione per l'FFN denso (bit-exact) al posto di sigmoid+mul+mul (3 dispatch -> 1).
     * Selezionato con la metrica deterministica --per-op: Sigmoid 687 + Multiply 718 dispatch. */
    bool sw_fused = false;
    { static int _sw = -1;
      if (_sw < 0) { const char* e = llm_ai_mlx_opt_env("LLMMETAL_DF2_SWIGLU_METAL"); _sw = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1; }
      if (_sw == 1) sw_fused = llm_mlx_swiglu_metal(gate_out, up_out, &act, st); }
    if (ok && !sw_fused) {
        if (ok) ok = mlx_sigmoid(&sig, gate_out, st) == 0;
        if (ok) ok = mlx_multiply(&silu, gate_out, sig, st) == 0;
        if (ok) ok = mlx_multiply(&act, silu, up_out, st) == 0;
    }
    mlx_array down_out = mlx_array_new();
    if (ok) { if(_mt) clock_gettime(CLOCK_MONOTONIC,&_m0); ok = df2_linear(&down_out, act, down_w, st); }
    if (ok) { if(_mt) clock_gettime(CLOCK_MONOTONIC,&_m1); _td=((_m1.tv_sec-_m0.tv_sec)*1e3+(_m1.tv_nsec-_m0.tv_nsec)/1e6); }
    if (_mt && layer_idx==2) { static int _o=0; if(_o++<8) fprintf(stderr,"[df2mlp] gate=%.2fms up=%.2fms down=%.2fms (costruzione lazy)\n",_tg,_tu,_td); }
    if (ok) ok = mlx_array_set(out, down_out) == 0;

    mlx_array all[] = {gate_w, up_w, down_w, gate_out, up_out, sig, silu, act, down_out};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);
    return ok;
}

/* ============================================================================
 * DFlash2DecoderLayer + GroupedDynamicCausalConv (prepare/finish su attn e MLP):
 *   xnorm = input_layernorm(x);  h,dyn = conv_prepare(xnorm)
 *   h = x + conv_finish(self_attn(h, x_ctx), dyn)
 *   hnorm = post_attention_layernorm(h);  h2,dyn2 = conv_prepare(hnorm)
 *   out = h + conv_finish(mlp(hnorm), dyn2)
 * ============================================================================ */
static bool df2_layer(DFlash2Runtime* rt, int layer_idx, int cur, int L, int S,
                      mlx_array x, mlx_array x_ctx, mlx_array* out) {
    LlmMlxModel* d = rt->draft;
    const ModelConfig* c = &d->cfg;
    mlx_stream st = d->stream;
    char namebuf[256];
    const int H = c->hidden_size;
    const int kern = c->dflash_conv_kernel;
    const int group_size = c->dflash_conv_group;
    const int groups = H / group_size;
    bool ok = true;
    const bool _ptime_l = (getenv("DF2_PHASE_TIME") != NULL); static int _once_l = 0;

#define CHK_DF2L(cond, what) do { if (!(cond)) { fprintf(stderr, "df2_layer[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)

    /* ---- pesi: norm, conv attenzione, conv mlp ---- */
    mlx_array ln_w = mlx_array_new(), ln2_w = mlx_array_new();
    tname(c->layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &ln_w);
    tname(c->layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &ln2_w);

    mlx_array ac_w = mlx_array_new(), ac_b = mlx_array_new();   /* attention_conv */
    mlx_array mc_w = mlx_array_new(), mc_b = mlx_array_new();   /* mlp_conv */
    tname(c->layer_prefix, layer_idx, "attention_conv.kernel_projection.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &ac_w);
    tname(c->layer_prefix, layer_idx, "attention_conv.base_kernel", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &ac_b);
    tname(c->layer_prefix, layer_idx, "mlp_conv.kernel_projection.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &mc_w);
    tname(c->layer_prefix, layer_idx, "mlp_conv.base_kernel", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&d->weights, namebuf, &mc_b);
    if (!ok) CHK_DF2L(false, "lookup pesi conv/norm");

    /* base[0] e base[1] per ciascun conv ([kern,H] ciascuno) */
    mlx_array ac_b0 = mlx_array_new(), ac_b1 = mlx_array_new();
    mlx_array mc_b0 = mlx_array_new(), mc_b1 = mlx_array_new();
    if (ok) {
        const int st0[3] = {0, 0, 0}, sp0[3] = {1, kern, H};
        const int st1[3] = {1, 0, 0}, sp1[3] = {2, kern, H};
        const int one[3] = {1, 1, 1};
        ok = mlx_slice(&ac_b0, ac_b, st0, 3, sp0, 3, one, 3, st) == 0;
        if (ok) ok = mlx_slice(&ac_b1, ac_b, st1, 3, sp1, 3, one, 3, st) == 0;
        if (ok) ok = mlx_slice(&mc_b0, mc_b, st0, 3, sp0, 3, one, 3, st) == 0;
        if (ok) ok = mlx_slice(&mc_b1, mc_b, st1, 3, sp1, 3, one, 3, st) == 0;
        if (ok) {
            const int sh[2] = {kern, H};
            ok = mlx_reshape(&ac_b0, ac_b0, sh, 2, st) == 0 &&
                 mlx_reshape(&ac_b1, ac_b1, sh, 2, st) == 0 &&
                 mlx_reshape(&mc_b0, mc_b0, sh, 2, st) == 0 &&
                 mlx_reshape(&mc_b1, mc_b1, sh, 2, st) == 0;
        }
    }

    /* ---- attenzione: conv.prepare + DFlashAttention + conv.finish ---- */
    mlx_array xnorm = mlx_array_new();
    if (ok) ok = df2_rms(&xnorm, x, ln_w, c->rms_norm_eps, st);
    /* dynamic = ac_w(xnorm) -> [1,L,2,kern,groups];  dyn_pre = [..,0], dyn_sv = [..,1] */
    mlx_array dyn_l = mlx_array_new(), dyn_pre = mlx_array_new(), dyn_sv = mlx_array_new();
    if (ok) ok = df2_linear(&dyn_l, xnorm, ac_w, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(dyn_l); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] lin_attn=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    if (ok) {
        const int sh[5] = {1, L, 2, kern, groups};
        ok = mlx_reshape(&dyn_l, dyn_l, sh, 5, st) == 0;
    }
    if (ok) {
        const int sA[5] = {0, 0, 0, 0, 0}, sB[5] = {1, L, 1, kern, groups};
        const int sA1[5] = {0, 0, 1, 0, 0}, sB1[5] = {1, L, 2, kern, groups};
        const int one[5] = {1, 1, 1, 1, 1};
        ok = mlx_slice(&dyn_pre, dyn_l, sA, 5, sB, 5, one, 5, st) == 0;
        if (ok) ok = mlx_slice(&dyn_sv, dyn_l, sA1, 5, sB1, 5, one, 5, st) == 0;
        if (ok) {
            const int sh4[4] = {1, L, kern, groups};
            ok = mlx_reshape(&dyn_pre, dyn_pre, sh4, 4, st) == 0 &&
                 mlx_reshape(&dyn_sv, dyn_sv, sh4, 4, st) == 0;
        }
    }
    mlx_array h_pre = mlx_array_new();
    if (ok) ok = df2_convolve(&h_pre, xnorm, dyn_pre, ac_b0, group_size, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(h_pre); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] conv_hpre=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    CHK_DF2L(ok, "conv prepare attenzione");
    if (getenv("DF2_DEBUG") && layer_idx == 0) {
        df2_dbg(xnorm, "l0.xnorm", st);
        df2_dbg(dyn_pre, "l0.dyn_pre", st);
        df2_dbg(h_pre, "l0.h_pre", st);
    }
    if (getenv("DF2_DUMP") && layer_idx == 0) {
        df2_dump(xnorm, "l0_xnorm", st);
        df2_dump(dyn_pre, "l0_dyn_pre", st);
        df2_dump(dyn_sv, "l0_dyn_sv", st);
        df2_dump(ac_b0, "l0_ac_b0", st);
        df2_dump(h_pre, "l0_h_pre", st);
    }

    mlx_array attn_o = mlx_array_new();
    if (ok) ok = df2_attn_layer(rt, layer_idx, cur, L, S, h_pre, x_ctx, &attn_o);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(attn_o); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] attn=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    CHK_DF2L(ok, "attenzione");
    if (getenv("DF2_DUMP") && layer_idx == 0) {
        df2_dump(attn_o, "l0_attn_o", st);
        df2_dump(dyn_sv, "l0_dyn_sv", st);
        df2_dump(ac_b1, "l0_ac_b1", st);
    }
    mlx_array h_fin = mlx_array_new();
    if (ok) ok = df2_convolve(&h_fin, attn_o, dyn_sv, ac_b1, group_size, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(h_fin); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] conv_hfin=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    mlx_array h = mlx_array_new();
    if (ok) ok = mlx_add(&h, x, h_fin, st) == 0;   /* residual 1 */
    if (getenv("DF2_DEBUG") && layer_idx == 0) {
        df2_dbg(attn_o, "l0.attn_o", st);
        df2_dbg(h_fin, "l0.h_fin", st);
        df2_dbg(h, "l0.h_post_attn", st);
    }

    /* ---- MLP: conv.prepare(post_ln) + SwiGLU + conv.finish ---- */
    mlx_array hnorm = mlx_array_new();
    if (ok) ok = df2_rms(&hnorm, h, ln2_w, c->rms_norm_eps, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(hnorm); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] rms2=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    mlx_array dyn2_l = mlx_array_new(), dyn2_pre = mlx_array_new(), dyn2_sv = mlx_array_new();
    if (ok) ok = df2_linear(&dyn2_l, hnorm, mc_w, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(dyn2_l); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] lin_mlp=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    if (ok) {
        const int sh[5] = {1, L, 2, kern, groups};
        ok = mlx_reshape(&dyn2_l, dyn2_l, sh, 5, st) == 0;
    }
    if (ok) {
        const int sA[5] = {0, 0, 0, 0, 0}, sB[5] = {1, L, 1, kern, groups};
        const int sA1[5] = {0, 0, 1, 0, 0}, sB1[5] = {1, L, 2, kern, groups};
        const int one[5] = {1, 1, 1, 1, 1};
        ok = mlx_slice(&dyn2_pre, dyn2_l, sA, 5, sB, 5, one, 5, st) == 0;
        if (ok) ok = mlx_slice(&dyn2_sv, dyn2_l, sA1, 5, sB1, 5, one, 5, st) == 0;
        if (ok) {
            const int sh4[4] = {1, L, kern, groups};
            ok = mlx_reshape(&dyn2_pre, dyn2_pre, sh4, 4, st) == 0 &&
                 mlx_reshape(&dyn2_sv, dyn2_sv, sh4, 4, st) == 0;
        }
    }
    mlx_array m_pre = mlx_array_new();
    if (ok) ok = df2_convolve(&m_pre, hnorm, dyn2_pre, mc_b0, group_size, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(m_pre); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] conv_mpre=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    CHK_DF2L(ok, "conv prepare mlp");

    mlx_array mlp_o = mlx_array_new();
    if (ok) ok = df2_mlp_swiglu(rt, layer_idx, m_pre, &mlp_o);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(mlp_o); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] mlp=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    mlx_array m_fin = mlx_array_new();
    if (ok) ok = df2_convolve(&m_fin, mlp_o, dyn2_sv, mc_b1, group_size, st);
    if (ok && _ptime_l) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a); mlx_array_eval(m_fin); clock_gettime(CLOCK_MONOTONIC,&_b); if (layer_idx==2 && _once_l<10) { _once_l++; fprintf(stderr,"[df2lt] conv_mfin=%.2fms\n", ((_b.tv_sec-_a.tv_sec)*1e3+(_b.tv_nsec-_a.tv_nsec)/1e6)); } }
    if (ok) ok = mlx_add(out, h, m_fin, st) == 0;   /* residual 2 */
    CHK_DF2L(ok, "layer output");
    if (getenv("DF2_DEBUG") && layer_idx == 0) {
        df2_dbg(hnorm, "l0.hnorm", st);
        df2_dbg(m_pre, "l0.m_pre", st);
        df2_dbg(mlp_o, "l0.mlp_o", st);
        df2_dbg(*out, "l0.out", st);
    }
    if (getenv("DF2_DUMP") && layer_idx == 0) {
        df2_dump(*out, "l0_out", st);
        df2_dump(h_fin, "l0_h_fin", st);
        df2_dump(h, "l0_h_post_attn", st);
    }

    mlx_array all[] = {
        ln_w, ln2_w, ac_w, ac_b, mc_w, mc_b, ac_b0, ac_b1, mc_b0, mc_b1,
        xnorm, dyn_l, dyn_pre, dyn_sv, h_pre, attn_o, h_fin, h,
        hnorm, dyn2_l, dyn2_pre, dyn2_sv, m_pre, mlp_o, m_fin,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK_DF2L
    return ok;
}

/* ============================================================================
 * Backbone draft (DFlashDraftModel._hidden): embed blocco (bf16 dal target),
 * fc+hidden_norm sul ctx, 5 layer, norm finale. Ritorna h [1,L,H].
 * Se layer_out != NULL (array di n_layers handle), ogni layer_out[l] riceve
 * una copia dell'output del layer l (per il dump di validazione).
 * ============================================================================ */
bool df2_backbone(DFlash2Runtime* rt, const int32_t* block_ids, int L,
                         mlx_array x_ctx /* [1,S,H] bf16 */, int S,
                         mlx_array* out_h, mlx_array* layer_out) {
    LlmMlxModel* d = rt->draft;
    const ModelConfig* c = &d->cfg;
    mlx_stream st = d->stream;
    char namebuf[256];
    bool ok = true;

    /* embed blocco [1,L,H] bf16 dal target */
    const bool _ptime = (getenv("DF2_PHASE_TIME") != NULL);
    mlx_array x = mlx_array_new();
    if (ok) ok = df2_embed_batch(rt, block_ids, L, &x);
    if (ok && _ptime) { struct timespec _t0,_t1; clock_gettime(CLOCK_MONOTONIC,&_t0); mlx_array_eval(x); clock_gettime(CLOCK_MONOTONIC,&_t1); fprintf(stderr,"[df2bb] embed=%.2fms\n",((_t1.tv_sec-_t0.tv_sec)*1e3+(_t1.tv_nsec-_t0.tv_nsec)/1e6)); }
    df2_dbg(x, "backbone.x_embed", st);

    /* h_ctx = hidden_norm(fc(x_ctx)): fc.weight [H, n_ctx*H] */
    mlx_array fc_w = mlx_array_new();
    mlx_array hn_w = mlx_array_new();
    llm_mlx_mode_set(namebuf, sizeof(namebuf), "fc.weight");
    ok = ok && multishard_get(&d->weights, namebuf, &fc_w);
    llm_mlx_mode_set(namebuf, sizeof(namebuf), "hidden_norm.weight");
    ok = ok && multishard_get(&d->weights, namebuf, &hn_w);
    if (!ok) { fprintf(stderr, "dflash: fc/hidden_norm.weight non trovati\n"); }
    mlx_array h_ctx = mlx_array_new();
    if (ok) ok = df2_linear(&h_ctx, x_ctx, fc_w, st);
    if (ok) ok = df2_rms(&h_ctx, h_ctx, hn_w, c->rms_norm_eps, st);
    if (ok && _ptime) { struct timespec _t0,_t1; clock_gettime(CLOCK_MONOTONIC,&_t0); mlx_array_eval(h_ctx); clock_gettime(CLOCK_MONOTONIC,&_t1); fprintf(stderr,"[df2bb] fc+norm=%.2fms (x_ctx [1,%d,%d])\n",((_t1.tv_sec-_t0.tv_sec)*1e3+(_t1.tv_nsec-_t0.tv_nsec)/1e6), S, c->hidden_size); }
    df2_dbg(h_ctx, "backbone.h_ctx", st);

    /* n layer */
    mlx_array h = mlx_array_new();
    if (ok) ok = mlx_array_set(&h, x) == 0;
    int cur = rt->ctx[rt->cur_ctx].ctx_pos;
    for (int l = 0; ok && l < c->n_layers; l++) {
        mlx_array lout = mlx_array_new();
        struct timespec _l0,_l1; if (_ptime) clock_gettime(CLOCK_MONOTONIC,&_l0);
        ok = df2_layer(rt, l, cur, L, S, h, h_ctx, &lout);
        if (ok && _ptime) { mlx_array_eval(lout); clock_gettime(CLOCK_MONOTONIC,&_l1); static int _once=0; if(_once++<10) fprintf(stderr,"[df2bb] layer%d=%.2fms\n", l, ((_l1.tv_sec-_l0.tv_sec)*1e3+(_l1.tv_nsec-_l0.tv_nsec)/1e6)); }
        if (ok && layer_out != NULL) ok = mlx_array_set(&layer_out[l], lout) == 0;
        mlx_array_free(h);
        h = lout;
    }
    /* norm finale */
    mlx_array norm_w = mlx_array_new();
    llm_mlx_mode_set(namebuf, sizeof(namebuf), "norm.weight");
    ok = ok && multishard_get(&d->weights, namebuf, &norm_w);
    if (!ok) { fprintf(stderr, "dflash: norm.weight non trovato\n"); }
    if (ok) ok = df2_rms(out_h, h, norm_w, c->rms_norm_eps, st);

    mlx_array_free(x); mlx_array_free(fc_w); mlx_array_free(hn_w); mlx_array_free(h_ctx); mlx_array_free(norm_w);
    return ok;
}

/* ============================================================================
 * CandidateSelector greedy (reference select, T=0): lattice top-k + greedy walk.
 *   hidden_post : [1, K(=L-1), H] (hidden dopo la norm, escluso l'anchor)
 *   logits      : [1, K, vocab]   gia' calcolati (lm_head del target)
 *   anchor      : token verificato
 *   out_path    : [K] token del path greedy
 * ============================================================================ */
double g_df2_t_ckpt = 0, g_df2_t_fwd = 0, g_df2_t_arg = 0, g_df2_t_roll = 0;
double g_df2_t_prop = 0, g_df2_t_sel = 0, g_df2_t_bb = 0, g_df2_t_wget = 0, g_df2_t_argp = 0;
double g_df2_ts_argp = 0, g_df2_ts_unary = 0, g_df2_ts_hproj = 0, g_df2_ts_succ = 0, g_df2_ts_ast = 0;
int g_df2_n_rounds = 0, g_df2_n_roll = 0;
double g_df2_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
bool g_df2_time_on(void) { static int v = -1; if (v < 0) v = (getenv("DF2_TIME") != NULL) ? 1 : 0; return v == 1; }

bool df2_selector(DFlash2Runtime* rt, mlx_array hidden_post, mlx_array logits,
                         int K, int32_t anchor, int32_t* out_path) {
    LlmMlxModel* d = rt->draft;
    const ModelConfig* c = &d->cfg;
    mlx_stream st = d->stream;
    const int top_k = c->dflash_selector_top_k;
    const int rank = c->dflash_selector_rank;
    char namebuf[256];
    bool ok = true;
    double _st0 = g_df2_time_on() ? g_df2_now() : 0.0;

#define CHK_DFS(cond, what) do { if (!(cond)) { fprintf(stderr, "df2_selector: %s failed\n", (what)); ok = false; } } while (0)

    /* candidate indices: argpartition(logits, -top_k)[..., -top_k:] */
    mlx_array part = mlx_array_new();
    double _ap0 = g_df2_time_on() ? g_df2_now() : 0.0;
    if (ok) ok = mlx_argpartition_axis(&part, logits, -top_k, -1, st) == 0;
    if (g_df2_time_on()) g_df2_t_argp += g_df2_now() - _ap0;
    mlx_array cand = mlx_array_new();
    if (ok) {
        int V = mlx_array_shape(logits)[2];
        const int sA[3] = {0, 0, V - top_k}, sB[3] = {1, K, V};
        const int one[3] = {1, 1, 1};
        ok = mlx_slice(&cand, part, sA, 3, sB, 3, one, 3, st) == 0;
    }
    mlx_array cand32 = mlx_array_new();
    if (ok) ok = mlx_astype(&cand32, cand, MLX_INT32, st) == 0;
    mlx_array unary = mlx_array_new();
    if (ok) ok = mlx_take_along_axis(&unary, logits, cand32, -1, st) == 0;
    mlx_array_free(part); mlx_array_free(cand);

    /* P1: carica i pesi del selector UNA sola volta (cache in rt). */
    if (ok && !rt->sel_w_loaded) {
        rt->sel_hp_w = mlx_array_new(); rt->sel_cs_w = mlx_array_new(); rt->sel_cp_w = mlx_array_new();
        llm_mlx_mode_set(namebuf, sizeof(namebuf), "candidate_selector.hidden_projection.weight");
        ok = ok && multishard_get(&d->weights, namebuf, &rt->sel_hp_w);
        if (ok) { llm_mlx_mode_set(namebuf, sizeof(namebuf), "candidate_selector.successor_codebook");
                  ok = ok && multishard_get(&d->weights, namebuf, &rt->sel_cs_w); }
        if (ok) { llm_mlx_mode_set(namebuf, sizeof(namebuf), "candidate_selector.predecessor_codebook");
                  ok = ok && multishard_get(&d->weights, namebuf, &rt->sel_cp_w); }
        if (ok) { mlx_vector_array wv = mlx_vector_array_new();
                  mlx_vector_array_append_value(wv, rt->sel_hp_w);
                  mlx_vector_array_append_value(wv, rt->sel_cs_w);
                  mlx_vector_array_append_value(wv, rt->sel_cp_w);
                  ok = mlx_eval(wv) == 0; mlx_vector_array_free(wv); }
        rt->sel_w_loaded = ok;
    }
    mlx_array hp_w = rt->sel_hp_w;
    if (!ok) CHK_DFS(false, "pesi selector non trovati");
    /* hidden_proj = hidden_projection(hidden_post) [1,K,rank] */
    mlx_array hidden_proj = mlx_array_new();
    if (ok) ok = df2_linear(&hidden_proj, hidden_post, hp_w, st);

    /* successors = successor_codebook(candidates) */
    mlx_array cs_w = rt->sel_cs_w;
    mlx_array cand_flat = mlx_array_new();
    if (ok) {
        const int sh[1] = {K * top_k};
        ok = mlx_reshape(&cand_flat, cand32, sh, 1, st) == 0;
    }
    mlx_array succ_rows = mlx_array_new();   /* [K*top_k, rank] */
    if (ok) ok = mlx_take_axis(&succ_rows, cs_w, cand_flat, 0, st) == 0;
    mlx_array succ = mlx_array_new();        /* [1,K,top_k,rank] */
    if (ok) {
        const int sh[4] = {1, K, top_k, rank};
        ok = mlx_reshape(&succ, succ_rows, sh, 4, st) == 0;
    }

    /* predecessor codebook (per il greedy walk) */
    mlx_array cp_w = rt->sel_cp_w;

    /* ---- P1 (2026-09-19): UNA eval per round + greedy-walk su CPU ----
     * Prima: un mlx_array_eval dentro il loop per-posizione -> K sync GPU/round su op minuscole.
     * Ora: si materializzano una volta i tensori del walk (fp32 su host) e si cammina su CPU
     * (O(K*top_k*rank)); il predecessor_codebook si copia su host una sola volta (cache in rt). */
    mlx_array cand_h = mlx_array_new();
    mlx_array unary32 = mlx_array_new(), hproj32 = mlx_array_new(), succ32 = mlx_array_new();
    if (ok) ok = mlx_array_set(&cand_h, cand32) == 0;
    if (ok) ok = mlx_astype(&unary32, unary, MLX_FLOAT32, st) == 0;
    if (ok) ok = mlx_astype(&hproj32, hidden_proj, MLX_FLOAT32, st) == 0;
    if (ok) ok = mlx_astype(&succ32, succ, MLX_FLOAT32, st) == 0;
    /* P1: UNA sola eval per round (prima erano 4 mlx_array_eval separate = 4 sync GPU). */
    const bool build_cpw = (rt->sel_cpw_host == NULL);
    mlx_array cp32 = mlx_array_new();
    if (ok && build_cpw) ok = mlx_astype(&cp32, cp_w, MLX_FLOAT32, st) == 0;
    /* DF2_SEL_TIME: valuta i sottografi SEPARATAMENTE per trovare il collo (diagnostica). */
    if (ok && getenv("DF2_SEL_TIME") != NULL) {
        double a;
        a = g_df2_now(); mlx_array_eval(cand_h);            g_df2_ts_argp  += g_df2_now() - a;
        a = g_df2_now(); mlx_array_eval(unary);             g_df2_ts_unary += g_df2_now() - a;
        a = g_df2_now(); mlx_array_eval(hidden_proj);       g_df2_ts_hproj += g_df2_now() - a;
        a = g_df2_now(); mlx_array_eval(succ);              g_df2_ts_succ  += g_df2_now() - a;
        a = g_df2_now(); mlx_array_eval(unary32); mlx_array_eval(hproj32); mlx_array_eval(succ32);
        g_df2_ts_ast += g_df2_now() - a;
    }
    if (ok) {
        mlx_vector_array v = mlx_vector_array_new();
        mlx_vector_array_append_value(v, cand_h);
        mlx_vector_array_append_value(v, unary32);
        mlx_vector_array_append_value(v, hproj32);
        mlx_vector_array_append_value(v, succ32);
        if (build_cpw) mlx_vector_array_append_value(v, cp32);
        ok = mlx_eval(v) == 0;
        mlx_vector_array_free(v);
    }
    if (ok && build_cpw) {
        if (mlx_array_ndim(cp32) != 2 || mlx_array_data_float32(cp32) == NULL) ok = false;
        else {
            int rows = (int)mlx_array_shape(cp32)[0], rr = (int)mlx_array_shape(cp32)[1];
            rt->sel_cpw_host = (float*)malloc((size_t)rows * (size_t)rr * sizeof(float));
            if (rt->sel_cpw_host == NULL) ok = false;
            else { memcpy(rt->sel_cpw_host, mlx_array_data_float32(cp32), (size_t)rows * (size_t)rr * sizeof(float));
                   rt->sel_cpw_rows = rows; rt->sel_cpw_rank = rr; }
        }
    }
    mlx_array_free(cp32);
    if (ok && rt->sel_cpw_rank != rank) ok = false;
    const int32_t* cand_host = ok ? mlx_array_data_int32(cand_h) : NULL;
    const float* U = ok ? mlx_array_data_float32(unary32) : NULL;
    const float* HP = ok ? mlx_array_data_float32(hproj32) : NULL;
    const float* SU = ok ? mlx_array_data_float32(succ32) : NULL;
    if (ok && (cand_host == NULL || U == NULL || HP == NULL || SU == NULL)) ok = false;
    int32_t predecessor = anchor;
    for (int p = 0; ok && p < K; p++) {
        if (predecessor < 0 || predecessor >= rt->sel_cpw_rows) { ok = false; break; }
        const float* A = rt->sel_cpw_host + (size_t)predecessor * rank;
        const float* Hp = HP + (size_t)p * rank;
        const float* B = SU + ((size_t)p * top_k) * rank;
        const float* Up = U + (size_t)p * top_k;
        float best = -INFINITY; int sel = -1;
        for (int cc = 0; cc < top_k; cc++) {
            const float* Bc = B + (size_t)cc * rank;
            float e = 0.0f;
            for (int r = 0; r < rank; r++) e += (A[r] * Hp[r]) * Bc[r];
            float sc = Up[cc] + e;
            if (sc > best) { best = sc; sel = cc; }
        }
        if (sel < 0) { ok = false; break; }
        predecessor = cand_host[(size_t)p * top_k + sel];
        out_path[p] = predecessor;
    }
    mlx_array_free(cand_h); mlx_array_free(unary32); mlx_array_free(hproj32); mlx_array_free(succ32);

    mlx_array_free(cand32);
    mlx_array_free(unary); mlx_array_free(hidden_proj);
    mlx_array_free(cand_flat); mlx_array_free(succ_rows); mlx_array_free(succ);
    if (g_df2_time_on()) g_df2_t_sel += g_df2_now() - _st0;
    if (getenv("DF2_SEL_TIME") != NULL) {
        static int pr = 0;
        if ((pr++ % 8) == 7)
            fprintf(stderr, "[df2seltime] argp=%.1f unary=%.1f hproj=%.1f succ=%.1f ast=%.1f ms (cumulativi)\n",
                    g_df2_ts_argp*1000, g_df2_ts_unary*1000, g_df2_ts_hproj*1000, g_df2_ts_succ*1000, g_df2_ts_ast*1000);
    }
#undef CHK_DFS
    return ok;
}
