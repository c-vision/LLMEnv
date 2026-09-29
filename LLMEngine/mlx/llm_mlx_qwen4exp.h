/* llm_mlx_qwen4exp.h — primitive per Qwen3.8-Flash-Next (qwen4_exp).
 *
 * Porting fedele da rapidMLX (unica reference reale disponibile in locale --
 * oMLX 0.6.4 installato non contiene questa architettura, verificato):
 * ~/Documents/Projects/Rapid-MLX-NextGen/vllm_mlx/models/qwen4_exp.py
 * ~/Documents/Projects/Rapid-MLX-NextGen/vllm_mlx/models/qwen4_exp_cache.py
 *
 * Vedi Todo/(open) - Piano-Correzioni-Modelli-MLX.md sezione C17 per la
 * scomposizione completa e l'ordine di implementazione. Questo file contiene
 * SOLO le primitive standalone (fase 1 del piano): ZeroCenteredRMSNorm e il
 * RoPE a posizioni esplicite (rotate-half, non-interleaved, rotary parziale).
 * Nessuna dipendenza da pesi di un modello reale -- verificate con dati
 * sintetici in native/llmengine_mlx/tests (vedi commento in fondo al file).
 */
#ifndef LLM_MLX_QWEN4EXP_H
#define LLM_MLX_QWEN4EXP_H

#include "mlx/c/mlx.h"
#include "llm_mlx_ops.h" /* llm_mlx_gated_delta_step -- passo ricorrente GDN, riusato identico */
#include <stdlib.h>
#include <stdint.h>

/* Linear non quantizzato (y = x @ astype(w,fp32)^T, bias=False) -- stessa logica di
 * plain_linear in llmengine_mlx.c (nome diverso per evitare collisione quando questo
 * header viene incluso li'; i pesi hyper-connection sono piccoli e tipicamente NON
 * quantizzati nel checkpoint reale, da confermare quando si collega il loader). */
static bool llm_mlx_qwen4exp_plain_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_stream stream) {
    mlx_array w32 = mlx_array_new(), wt = mlx_array_new();
    bool ok = mlx_astype(&w32, w, MLX_FLOAT32, stream) == 0;
    if (ok) ok = mlx_transpose(&wt, w32, stream) == 0;
    if (ok) ok = mlx_matmul(res, x, wt, stream) == 0;
    mlx_array_free(w32);
    mlx_array_free(wt);
    return ok;
}

static bool llm_mlx_qwen4exp_silu(mlx_array* out, mlx_array x, mlx_stream stream) {
    mlx_array sig = mlx_array_new();
    bool ok = mlx_sigmoid(&sig, x, stream) == 0;
    if (ok) ok = mlx_multiply(out, x, sig, stream) == 0;
    mlx_array_free(sig);
    return ok;
}

/* ZeroCenteredRMSNorm (qwen4_exp.py righe 202-226): RMSNorm il cui peso di
 * checkpoint rappresenta un DELTA da 1 (normalized * (1+weight)), non il
 * fattore diretto (normalized * weight) come la RMSNorm "normale" del resto
 * del motore (rms_norm_cfg). group_size>0 raggruppa l'ultimo asse prima di
 * normalizzare (usato per hc_norm/norm_key/norm_query/norm_conv su tensori
 * hc_width=4*hidden_size, con group=hidden_size); group_size<=0 = nessun
 * raggruppamento (usato per q_norm/k_norm/q_layernorm/k_layernorm su
 * head_dim). weight: mlx_array 1-D di lunghezza (ultimo_asse di x se
 * group_size<=0, altrimenti ultimo_asse di x). Calcolo SEMPRE in float32
 * (fedele alla reference), risultato ricast al dtype originale di x. */
static bool llm_mlx_zero_centered_rmsnorm(mlx_array* out, mlx_array x, mlx_array weight,
                                           int group_size, float eps, mlx_stream stream) {
    mlx_dtype dtype = mlx_array_dtype(x);
    const int* xshape = mlx_array_shape(x);
    int xndim = mlx_array_ndim(x);
    if (xshape == NULL || xndim <= 0) return false;

    mlx_array xr = mlx_array_new(), wr = mlx_array_new();
    bool ok = true;
    if (group_size > 0) {
        int last = xshape[xndim - 1];
        if (last % group_size != 0) return false;
        int ngroups = last / group_size;
        int* newshape = (int*)malloc((size_t)(xndim + 1) * sizeof(int));
        if (newshape == NULL) return false;
        for (int i = 0; i < xndim - 1; i++) newshape[i] = xshape[i];
        newshape[xndim - 1] = ngroups;
        newshape[xndim] = group_size;
        ok = mlx_reshape(&xr, x, newshape, xndim + 1, stream) == 0;
        free(newshape);
        const int wshape[2] = {ngroups, group_size};
        if (ok) ok = mlx_reshape(&wr, weight, wshape, 2, stream) == 0;
    } else {
        ok = mlx_array_set(&xr, x) == 0 && mlx_array_set(&wr, weight) == 0;
    }

    mlx_array xf = mlx_array_new();
    if (ok) ok = mlx_astype(&xf, xr, MLX_FLOAT32, stream) == 0;
    mlx_array sq = mlx_array_new();
    if (ok) ok = mlx_square(&sq, xf, stream) == 0;
    mlx_array mn = mlx_array_new();
    if (ok) ok = mlx_mean_axis(&mn, sq, -1, /*keepdims=*/true, stream) == 0;
    mlx_array eps_arr = mlx_array_new_float32(eps);
    mlx_array mne = mlx_array_new();
    if (ok) ok = mlx_add(&mne, mn, eps_arr, stream) == 0;
    mlx_array rs = mlx_array_new();
    if (ok) ok = mlx_rsqrt(&rs, mne, stream) == 0;
    mlx_array normed = mlx_array_new();
    if (ok) ok = mlx_multiply(&normed, xf, rs, stream) == 0;
    mlx_array wf = mlx_array_new();
    if (ok) ok = mlx_astype(&wf, wr, MLX_FLOAT32, stream) == 0;
    mlx_array one = mlx_array_new_float32(1.0f);
    mlx_array wone = mlx_array_new();
    if (ok) ok = mlx_add(&wone, wf, one, stream) == 0;
    mlx_array scaled = mlx_array_new();
    if (ok) ok = mlx_multiply(&scaled, normed, wone, stream) == 0;
    mlx_array casted = mlx_array_new();
    if (ok) ok = mlx_astype(&casted, scaled, dtype, stream) == 0;
    if (ok) ok = mlx_reshape(out, casted, xshape, xndim, stream) == 0;

    mlx_array_free(xr); mlx_array_free(wr); mlx_array_free(xf); mlx_array_free(sq);
    mlx_array_free(mn); mlx_array_free(eps_arr); mlx_array_free(mne); mlx_array_free(rs);
    mlx_array_free(normed); mlx_array_free(wf); mlx_array_free(one); mlx_array_free(wone);
    mlx_array_free(scaled); mlx_array_free(casted);
    return ok;
}

/* apply_rotary_positions (qwen4_exp.py righe 440-471): RoPE "rotate-half"
 * NON interleaved a posizioni ESPLICITE (necessario per QSA: le chiavi
 * compresse ruotano alla posizione del PRIMO token del loro gruppo, mai
 * quella corrente -- non e' esprimibile con un offset scalare singolo).
 * x: [batch, tokens, heads, dim]. positions: [tokens] o [batch, tokens].
 * Applica la rotazione SOLO ai primi rotary_dim canali, il resto passa
 * invariato (concatenato in coda). rotary_dim==0 -> ritorna x invariato. */
static bool llm_mlx_qwen4exp_rope_positions(mlx_array* out, mlx_array x, mlx_array positions,
                                             int rotary_dim, float base, mlx_stream stream) {
    if (rotary_dim == 0) return mlx_array_set(out, x) == 0;
    if (rotary_dim % 2 != 0) return false;
    const int* xshape = mlx_array_shape(x);
    int xndim = mlx_array_ndim(x);
    if (xshape == NULL || xndim != 4) return false;
    const int dim = xshape[3];

    mlx_array pos2 = mlx_array_new();
    bool ok;
    if (mlx_array_ndim(positions) == 1) {
        const int pshape[2] = {1, mlx_array_dim(positions, 0)};
        mlx_array tmp = mlx_array_new();
        ok = mlx_reshape(&tmp, positions, pshape, 2, stream) == 0;
        if (ok) ok = mlx_array_set(&pos2, tmp) == 0;
        mlx_array_free(tmp);
    } else {
        ok = mlx_array_set(&pos2, positions) == 0;
    }

    /* inverse_frequency = 1 / base^(arange(0,rotary_dim,2)/rotary_dim) : [rotary_dim/2] */
    const int half = rotary_dim / 2;
    mlx_array idx = mlx_array_new();
    if (ok) {
        const int shp0[1] = {half};
        mlx_array ar = mlx_array_new();
        mlx_optional_dtype fdt = {MLX_FLOAT32, true};
        ok = mlx_arange(&ar, 0.0, (double)rotary_dim, 2.0, MLX_FLOAT32, stream) == 0;
        (void)shp0; (void)fdt;
        if (ok) ok = mlx_array_set(&idx, ar) == 0;
        mlx_array_free(ar);
    }
    mlx_array rd = mlx_array_new_float32((float)rotary_dim);
    mlx_array exponent = mlx_array_new();
    if (ok) ok = mlx_divide(&exponent, idx, rd, stream) == 0;
    mlx_array base_arr = mlx_array_new_float32(base);
    mlx_array powbase = mlx_array_new();
    if (ok) ok = mlx_power(&powbase, base_arr, exponent, stream) == 0;
    mlx_array inv_freq = mlx_array_new();
    mlx_array one = mlx_array_new_float32(1.0f);
    if (ok) ok = mlx_divide(&inv_freq, one, powbase, stream) == 0;

    /* angles = positions[...,None] * inv_freq -> [B,T,half]; poi concat[angles,angles] su
     * ultimo asse -> [B,T,rotary_dim]; poi [:, :, None, :] -> [B,T,1,rotary_dim] per
     * broadcast sull'asse head. */
    mlx_array posf = mlx_array_new();
    if (ok) ok = mlx_astype(&posf, pos2, MLX_FLOAT32, stream) == 0;
    mlx_array pos3 = mlx_array_new();
    if (ok) {
        const int b = mlx_array_dim(posf, 0), t = mlx_array_dim(posf, 1);
        const int ps[3] = {b, t, 1};
        ok = mlx_reshape(&pos3, posf, ps, 3, stream) == 0;
    }
    mlx_array angles_h = mlx_array_new();
    if (ok) ok = mlx_multiply(&angles_h, pos3, inv_freq, stream) == 0;
    mlx_array angles_full = mlx_array_new();
    if (ok) {
        mlx_vector_array av = mlx_vector_array_new();
        mlx_vector_array_append_value(av, angles_h);
        mlx_vector_array_append_value(av, angles_h);
        ok = mlx_concatenate_axis(&angles_full, av, -1, stream) == 0;
        mlx_vector_array_free(av);
    }
    mlx_array angles = mlx_array_new();
    if (ok) {
        const int b = mlx_array_dim(angles_full, 0), t = mlx_array_dim(angles_full, 1);
        const int as4[4] = {b, t, 1, rotary_dim};
        ok = mlx_reshape(&angles, angles_full, as4, 4, stream) == 0;
    }
    mlx_array cosine = mlx_array_new(), sine = mlx_array_new();
    if (ok) ok = mlx_cos(&cosine, angles, stream) == 0 && mlx_sin(&sine, angles, stream) == 0;

    /* rotary = x[..., :rotary_dim]; half-split: rotated_half = concat(-rotary[...,half:],
     * rotary[...,:half]); rotated = rotary*cos + rotated_half*sin */
    mlx_array rotary = mlx_array_new();
    if (ok) {
        const int start[4] = {0, 0, 0, 0};
        const int stop[4] = {xshape[0], xshape[1], xshape[2], rotary_dim};
        const int strd[4] = {1, 1, 1, 1};
        ok = mlx_slice(&rotary, x, start, 4, stop, 4, strd, 4, stream) == 0;
    }
    mlx_array first_half = mlx_array_new(), second_half = mlx_array_new();
    if (ok) {
        const int s1[4] = {0, 0, 0, 0}, e1[4] = {xshape[0], xshape[1], xshape[2], half};
        const int s2[4] = {0, 0, 0, half}, e2[4] = {xshape[0], xshape[1], xshape[2], rotary_dim};
        const int strd[4] = {1, 1, 1, 1};
        ok = mlx_slice(&first_half, rotary, s1, 4, e1, 4, strd, 4, stream) == 0 &&
             mlx_slice(&second_half, rotary, s2, 4, e2, 4, strd, 4, stream) == 0;
    }
    mlx_array neg_second = mlx_array_new();
    if (ok) ok = mlx_negative(&neg_second, second_half, stream) == 0;
    mlx_array rotated_half = mlx_array_new();
    if (ok) {
        mlx_vector_array rv = mlx_vector_array_new();
        mlx_vector_array_append_value(rv, neg_second);
        mlx_vector_array_append_value(rv, first_half);
        ok = mlx_concatenate_axis(&rotated_half, rv, -1, stream) == 0;
        mlx_vector_array_free(rv);
    }
    mlx_array term1 = mlx_array_new(), term2 = mlx_array_new(), rotated = mlx_array_new();
    if (ok) ok = mlx_multiply(&term1, rotary, cosine, stream) == 0;
    if (ok) ok = mlx_multiply(&term2, rotated_half, sine, stream) == 0;
    if (ok) ok = mlx_add(&rotated, term1, term2, stream) == 0;
    mlx_array rotated_cast = mlx_array_new();
    if (ok) ok = mlx_astype(&rotated_cast, rotated, mlx_array_dtype(x), stream) == 0;

    if (ok && rotary_dim == dim) {
        ok = mlx_array_set(out, rotated_cast) == 0;
    } else if (ok) {
        mlx_array pass = mlx_array_new();
        const int sp[4] = {0, 0, 0, rotary_dim};
        const int ep[4] = {xshape[0], xshape[1], xshape[2], dim};
        const int strd[4] = {1, 1, 1, 1};
        ok = mlx_slice(&pass, x, sp, 4, ep, 4, strd, 4, stream) == 0;
        if (ok) {
            mlx_vector_array ov = mlx_vector_array_new();
            mlx_vector_array_append_value(ov, rotated_cast);
            mlx_vector_array_append_value(ov, pass);
            ok = mlx_concatenate_axis(out, ov, -1, stream) == 0;
            mlx_vector_array_free(ov);
        }
        mlx_array_free(pass);
    }

    mlx_array_free(pos2); mlx_array_free(idx); mlx_array_free(rd); mlx_array_free(exponent);
    mlx_array_free(base_arr); mlx_array_free(powbase); mlx_array_free(inv_freq);
    mlx_array_free(posf); mlx_array_free(pos3); mlx_array_free(angles_h);
    mlx_array_free(angles_full); mlx_array_free(angles); mlx_array_free(cosine);
    mlx_array_free(sine); mlx_array_free(rotary); mlx_array_free(first_half);
    mlx_array_free(second_half); mlx_array_free(neg_second); mlx_array_free(rotated_half);
    mlx_array_free(term1); mlx_array_free(term2); mlx_array_free(rotated);
    mlx_array_free(rotated_cast); mlx_array_free(one);
    return ok;
}

/* apply_qwen4_exp_rope (qwen4_exp.py righe 542-575, solo il fallback puro-MLX --
 * il kernel Metal dedicato della reference e' un'ottimizzazione, non necessaria alla
 * correttezza): x arriva come [B,H,T,D] (layout standard di questo motore dopo
 * transpose), trasposto a [B,T,H,D] per llm_mlx_qwen4exp_rope_positions, ritrasposto
 * all'uscita. */
static bool llm_mlx_qwen4exp_rope(mlx_array* out, mlx_array x_bhtd, mlx_array positions,
                                   int rotary_dim, float base, mlx_stream stream) {
    const int perm[4] = {0, 2, 1, 3};
    mlx_array x_bthd = mlx_array_new();
    bool ok = mlx_transpose_axes(&x_bthd, x_bhtd, perm, 4, stream) == 0;
    mlx_array rot = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_rope_positions(&rot, x_bthd, positions, rotary_dim, base, stream);
    if (ok) ok = mlx_transpose_axes(out, rot, perm, 4, stream) == 0;
    mlx_array_free(x_bthd);
    mlx_array_free(rot);
    return ok;
}

/* GatedResidual / "hyper-connection mixer" (qwen4_exp.py righe 255-289) -- il cuore
 * architetturale, usato ovunque (embedding iniziale via tile, prima di ogni attn/mlp,
 * mixer finale). hyper_input ha ultimo asse hc_count*hidden_size (l'intero modello
 * lavora "tileato" hc_count volte). Se inject_w.ctx==NULL (use_combine=False, SOLO il
 * mixer finale del modello) ritorna solo *out_mixed, out_residual/out_injection non
 * toccati -- il chiamante deve passare NULL per quei due in quel caso.
 *
 *   normed = ZeroCenteredRMSNorm(hc_width, group=hidden_size)(hyper_input)
 *   mix = sigmoid(up(silu(down(normed) / hc_count)))
 *   mixed = mean(reshape(mix,[...,hc_count,hidden]) * reshape(normed,[...,hc_count,hidden]), axis=-2)
 *   se use_combine: injection = 2*sigmoid(inject(normed) / hc_count); out_residual = hyper_input (NON normed)
 *
 * ASSUNZIONE DA VERIFICARE sul checkpoint reale (nessun modello caricato in questa
 * sessione): down_w/up_w/inject_w sono usati qui via llm_mlx_qwen4exp_plain_linear
 * (matmul PIENO, non quantizzato) -- CORRETTO per inject_w (verificato sul checkpoint
 * reale: block_inject_weight ha solo ".weight"), MA SBAGLIATO per down_w/up_w
 * (input_mix_weight_down/up SONO quantizzati, .scales/.biases presenti sul checkpoint
 * reale). Per questo motivo questa funzione NON e' piu' chiamata dal wiring reale --
 * llmengine_mlx.c usa invece qwen4_exp_gated_residual_q (stessa formula, down/up via
 * quantized_linear_cfg), aggiunta 2026-09-06 dopo aver trovato il bug ispezionando
 * model.safetensors.index.json (commit 31580069). QUESTA funzione resta -- deliberatamente,
 * non e' un refuso da pulire -- per i test standalone con pesi densi sintetici verificati
 * contro numpy (stesso principio di llm_mlx_qwen4exp_qsa_indexer_project/
 * _qsa_attention_core sotto: "wrapper plain" tenuto per la testabilita' in isolamento,
 * il motore reale usa sempre la variante quantization-aware). Un'implementazione
 * indipendente della stessa hyper-connection scritta in una sessione precedente
 * (q4_gated_residual/q4_zc_rmsnorm/q4_linear_helper, mai chiamata da nessuno) faceva la
 * STESSA assunzione sbagliata su down/up -- rimossa 2026-09-06 come duplicato (vedi
 * llmengine_mlx.c, commento sopra q4_sigmoid_rmsnorm_gated), MAI corretta perche' mai
 * arrivata a un load reale. */
static bool llm_mlx_gated_residual(mlx_array* out_mixed, mlx_array* out_residual, mlx_array* out_injection,
                                    mlx_array hyper_input, mlx_array hc_norm_w,
                                    mlx_array down_w, mlx_array up_w, mlx_array inject_w,
                                    int hc_count, int hidden_size, float eps, mlx_stream stream) {
    const int* hshape = mlx_array_shape(hyper_input);
    int hndim = mlx_array_ndim(hyper_input);
    if (hshape == NULL || hndim <= 0) return false;
    if (hshape[hndim - 1] != hc_count * hidden_size) return false;

    mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_zero_centered_rmsnorm(&normed, hyper_input, hc_norm_w, hidden_size, eps, stream);
    mlx_array down_out = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&down_out, normed, down_w, stream);
    mlx_array hcf = mlx_array_new_float32((float)hc_count);
    mlx_array down_scaled = mlx_array_new();
    if (ok) ok = mlx_divide(&down_scaled, down_out, hcf, stream) == 0;
    mlx_array down_silu = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_silu(&down_silu, down_scaled, stream);
    mlx_array up_out = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&up_out, down_silu, up_w, stream);
    mlx_array mix = mlx_array_new();
    if (ok) ok = mlx_sigmoid(&mix, up_out, stream) == 0;

    int* mshape = NULL;
    mlx_array mix_r = mlx_array_new(), normed_r = mlx_array_new();
    if (ok) {
        mshape = (int*)malloc((size_t)(hndim + 1) * sizeof(int));
        if (mshape == NULL) { ok = false; }
    }
    if (ok) {
        for (int i = 0; i < hndim - 1; i++) mshape[i] = hshape[i];
        mshape[hndim - 1] = hc_count;
        mshape[hndim] = hidden_size;
        ok = mlx_reshape(&mix_r, mix, mshape, hndim + 1, stream) == 0 &&
             mlx_reshape(&normed_r, normed, mshape, hndim + 1, stream) == 0;
    }
    mlx_array prod = mlx_array_new();
    if (ok) ok = mlx_multiply(&prod, mix_r, normed_r, stream) == 0;
    if (ok) ok = mlx_mean_axis(out_mixed, prod, -2, /*keepdims=*/false, stream) == 0;

    if (ok && inject_w.ctx != NULL && out_residual != NULL && out_injection != NULL) {
        mlx_array inj_out = mlx_array_new();
        ok = llm_mlx_qwen4exp_plain_linear(&inj_out, normed, inject_w, stream);
        mlx_array inj_scaled = mlx_array_new();
        if (ok) ok = mlx_divide(&inj_scaled, inj_out, hcf, stream) == 0;
        mlx_array inj_sig = mlx_array_new();
        if (ok) ok = mlx_sigmoid(&inj_sig, inj_scaled, stream) == 0;
        mlx_array two = mlx_array_new_float32(2.0f);
        if (ok) ok = mlx_multiply(out_injection, inj_sig, two, stream) == 0;
        if (ok) ok = mlx_array_set(out_residual, hyper_input) == 0;
        mlx_array_free(inj_out); mlx_array_free(inj_scaled); mlx_array_free(inj_sig);
        mlx_array_free(two);
    }

    if (mshape) free(mshape);
    mlx_array_free(normed); mlx_array_free(down_out); mlx_array_free(hcf);
    mlx_array_free(down_scaled); mlx_array_free(down_silu); mlx_array_free(up_out);
    mlx_array_free(mix); mlx_array_free(mix_r); mlx_array_free(normed_r); mlx_array_free(prod);
    return ok;
}

/* DecoderLayer._combine (qwen4_exp.py righe 1196-1203): applica l'injection per-stream
 * all'output del sotto-layer (attn o mlp) e lo somma al residuo NON normalizzato.
 *   injected = output[...,None,:] * injection[...,:,None]   -- [*, hc_count, hidden]
 *   return residual + flatten(injected, -2)                 -- [*, hc_count*hidden]
 * output: [*, hidden_size] (hc_count=1 implicito, l'output del sotto-layer NON e' tileato).
 * injection: [*, hc_count]. residual: [*, hc_count*hidden_size] (hyper_input originale). */
static bool llm_mlx_hc_combine(mlx_array* out, mlx_array output, mlx_array residual,
                                mlx_array injection, int hc_count, int hidden_size, mlx_stream stream) {
    const int* oshape = mlx_array_shape(output);
    int ondim = mlx_array_ndim(output);
    if (oshape == NULL || ondim <= 0) return false;

    int* o_exp_shape = (int*)malloc((size_t)(ondim + 1) * sizeof(int));
    int* i_exp_shape = (int*)malloc((size_t)(ondim + 1) * sizeof(int));
    if (o_exp_shape == NULL || i_exp_shape == NULL) { free(o_exp_shape); free(i_exp_shape); return false; }
    for (int i = 0; i < ondim - 1; i++) { o_exp_shape[i] = oshape[i]; i_exp_shape[i] = oshape[i]; }
    o_exp_shape[ondim - 1] = 1; o_exp_shape[ondim] = hidden_size;
    i_exp_shape[ondim - 1] = hc_count; i_exp_shape[ondim] = 1;

    mlx_array output_e = mlx_array_new(), injection_e = mlx_array_new();
    bool ok = mlx_reshape(&output_e, output, o_exp_shape, ondim + 1, stream) == 0 &&
              mlx_reshape(&injection_e, injection, i_exp_shape, ondim + 1, stream) == 0;
    mlx_array injected = mlx_array_new();
    if (ok) ok = mlx_multiply(&injected, output_e, injection_e, stream) == 0;
    int* flat_shape = (int*)malloc((size_t)ondim * sizeof(int));
    mlx_array injected_flat = mlx_array_new();
    if (ok && flat_shape != NULL) {
        for (int i = 0; i < ondim - 1; i++) flat_shape[i] = oshape[i];
        flat_shape[ondim - 1] = hc_count * hidden_size;
        ok = mlx_reshape(&injected_flat, injected, flat_shape, ondim, stream) == 0;
    } else if (flat_shape == NULL) { ok = false; }
    if (ok) ok = mlx_add(out, residual, injected_flat, stream) == 0;

    free(o_exp_shape); free(i_exp_shape); free(flat_shape);
    mlx_array_free(output_e); mlx_array_free(injection_e); mlx_array_free(injected);
    mlx_array_free(injected_flat);
    return ok;
}

/* ==== N-gram/PLE: hashing deterministico (qwen4_exp.py righe 893-931) ====
 * Pura aritmetica intera a 64 bit, NESSUNA dipendenza da mlx -- deve produrre
 * ESATTAMENTE gli stessi numeri del checkpoint di riferimento (i pesi delle tabelle
 * di embedding sono indicizzati da questi valori: un mismatch anche di un bit rompe
 * silenziosamente la corrispondenza pesi<->hash). uint64_t in C avvolge a 2^64
 * automaticamente su add/mul, stessa semantica di "& mask" esplicito in Python. */
static uint64_t llm_mlx_qwen4exp_splitmix64(uint64_t value) {
    value = value + 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

/* out_multipliers: buffer di ngram_size elementi (chiamante alloca). */
static void llm_mlx_qwen4exp_build_layer_multipliers(int64_t* out_multipliers,
                                                      int64_t unigram_vocab_size,
                                                      int ngram_size, int ple_layer_index,
                                                      int64_t seed) {
    int64_t vocab = unigram_vocab_size > 0 ? unigram_vocab_size : 1;
    uint64_t multiplier_max = 0x7FFFFFFFFFFFFFFFULL / (uint64_t)vocab; /* (2^63)-1, come Python */
    uint64_t half_bound = multiplier_max / 2; if (half_bound < 1) half_bound = 1;
    uint64_t base_seed = (uint64_t)seed + 10007ULL * (uint64_t)ple_layer_index;
    for (int index = 0; index < ngram_size; index++) {
        uint64_t v = base_seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(index + 1);
        out_multipliers[index] = (int64_t)(2ULL * (llm_mlx_qwen4exp_splitmix64(v) % half_bound) + 1ULL);
    }
}

static bool llm_mlx_qwen4exp_is_prime(int64_t value) {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    for (int64_t d = 3; d * d <= value; d += 2) {
        if (value % d == 0) return false;
    }
    return true;
}

/* Trova il conteggio-esimo numero primo strettamente dopo `start` (1-based: count=1 ->
 * il primo primo dopo start). Usato per dimensionare le tabelle di embedding n-gram --
 * ogni "testa" ottiene una dimensione-tabella primo-numero deterministica dal proprio
 * indice, cosi' che due teste non condividano mai un periodo di hashing comune. */
static int64_t llm_mlx_qwen4exp_find_nth_prime_after(int64_t start, int64_t count) {
    int64_t value = start;
    for (int64_t i = 0; i < count; i++) {
        value++;
        while (!llm_mlx_qwen4exp_is_prime(value)) value++;
    }
    return value;
}

/* ==== GDN Qwen4-Exp: normalizzazione q/k e gate d'uscita ====
 * Il passo ricorrente vero e proprio (llm_mlx_gated_delta_step, gia' in
 * llm_mlx_ops.h, usato da qwen3_next/qwen3_5) e' RIUSATO IDENTICO -- stessa
 * ricorrenza esatta, nessuna modifica. Queste due funzioni sono le uniche parti
 * DAVVERO diverse rispetto a qwen3_next_linear_block (verificato leggendo
 * llmengine_mlx.c righe ~6620-6790): qwen3_next normalizza q/k con
 * mlx_fast_rms_norm_no_scale (RMS, media dei quadrati) e usa un gate SILU in
 * uscita; Qwen4-Exp usa una norma L2 diversa (epsilon FUORI dalla somma, non
 * dentro una media) e un gate SIGMOID -- riusare per errore la normalizzazione
 * di qwen3_next qui sarebbe esattamente il bug che la reference stessa segnala
 * di aver gia' preso una volta ("la forma RMS-normalizzata precedente
 * divergeva sul checkpoint reale", qwen4_exp.py righe 410-414). */

/* qwen4_exp.py righe 415-420: query = query * rsqrt(sum(query^2,-1,keepdims)+1e-6),
 * poi query *= head_dim**-0.5 (SOLO su query, mai su key); key = key * rsqrt(sum(key^2,
 * -1,keepdims)+1e-6), nessuna scala aggiuntiva. q,k: [B,T,H,Dk] (stesso layout non
 * trasposto della reference in questo punto -- il transpose ad [B,H,T,Dk] avviene DOPO
 * in QSAAttention/GatedDeltaNet, qui non serve). */
static bool llm_mlx_qwen4exp_qk_l2norm(mlx_array* out_q, mlx_array* out_k,
                                        mlx_array q, mlx_array k, float head_dim_neg_half_base,
                                        mlx_stream stream) {
    mlx_array eps = mlx_array_new_float32(1e-6f);
    mlx_array qsq = mlx_array_new(), qsum = mlx_array_new(), qsume = mlx_array_new(), qrs = mlx_array_new();
    bool ok = mlx_square(&qsq, q, stream) == 0;
    if (ok) ok = mlx_sum_axis(&qsum, qsq, -1, /*keepdims=*/true, stream) == 0;
    if (ok) ok = mlx_add(&qsume, qsum, eps, stream) == 0;
    if (ok) ok = mlx_rsqrt(&qrs, qsume, stream) == 0;
    mlx_array qn = mlx_array_new();
    if (ok) ok = mlx_multiply(&qn, q, qrs, stream) == 0;
    mlx_array scale = mlx_array_new_float32(head_dim_neg_half_base);
    if (ok) ok = mlx_multiply(out_q, qn, scale, stream) == 0;

    mlx_array ksq = mlx_array_new(), ksum = mlx_array_new(), ksume = mlx_array_new(), krs = mlx_array_new();
    if (ok) ok = mlx_square(&ksq, k, stream) == 0;
    if (ok) ok = mlx_sum_axis(&ksum, ksq, -1, /*keepdims=*/true, stream) == 0;
    if (ok) ok = mlx_add(&ksume, ksum, eps, stream) == 0;
    if (ok) ok = mlx_rsqrt(&krs, ksume, stream) == 0;
    if (ok) ok = mlx_multiply(out_k, k, krs, stream) == 0;

    mlx_array_free(eps); mlx_array_free(qsq); mlx_array_free(qsum); mlx_array_free(qsume);
    mlx_array_free(qrs); mlx_array_free(qn); mlx_array_free(scale);
    mlx_array_free(ksq); mlx_array_free(ksum); mlx_array_free(ksume); mlx_array_free(krs);
    return ok;
}

/* SigmoidRMSNormGated (qwen4_exp.py righe 229-252): confine di arrotondamento
 * ESPLICITAMENTE preservato dalla reference -- rms_norm PRIMA, poi cast fp32, poi
 * sigmoid(gate) in fp32, poi moltiplica, poi ricast al dtype originale. Qwen4-Exp usa
 * SOLO la variante sigmoid (validato a monte nel config, mai silu) -- a differenza del
 * gate SILU usato da qwen3_next per lo stesso punto strutturale (non riusare quel path). */
static bool llm_mlx_sigmoid_rmsnorm_gated(mlx_array* out, mlx_array hidden, mlx_array weight,
                                           mlx_array gate, int use_silu, float eps, mlx_stream stream) {
    mlx_dtype dtype = mlx_array_dtype(hidden);
    mlx_array normed = mlx_array_new();
    bool ok = mlx_fast_rms_norm(&normed, hidden, weight, eps, stream) == 0;
    mlx_array normed_f = mlx_array_new();
    if (ok) ok = mlx_astype(&normed_f, normed, MLX_FLOAT32, stream) == 0;
    mlx_array gate_f = mlx_array_new();
    if (ok) ok = mlx_astype(&gate_f, gate, MLX_FLOAT32, stream) == 0;
    mlx_array gate_sig = mlx_array_new();
    /* Reference: activation = output_gate_type or hidden_act; sigmoid SE == "sigmoid", altrimenti silu. */
    if (ok && use_silu) ok = llm_mlx_qwen4exp_silu(&gate_sig, gate_f, stream);
    else if (ok) ok = mlx_sigmoid(&gate_sig, gate_f, stream) == 0;
    mlx_array gated = mlx_array_new();
    if (ok) ok = mlx_multiply(&gated, normed_f, gate_sig, stream) == 0;
    if (ok) ok = mlx_astype(out, gated, dtype, stream) == 0;
    mlx_array_free(normed); mlx_array_free(normed_f); mlx_array_free(gate_f);
    mlx_array_free(gate_sig); mlx_array_free(gated);
    return ok;
}

/* ==== Blocco GDN Qwen4-Exp completo (qwen4_exp.py righe 332-437, GatedDeltaNet.__call__) ====
 * Assembla: 4 proiezioni SEPARATE (a differenza di qwen3_next, merged+split -- qui gia'
 * pulite), conv1d depthwise + silu (pattern IDENTICO a qwen3_next_linear_block, riusato),
 * norma L2 q/k NUOVA (llm_mlx_qwen4exp_qk_l2norm), formula g/beta IDENTICA a qwen3_next,
 * ripetizione teste Hk->Hv IDENTICA, loop ricorrente per-token via llm_mlx_gated_delta_step
 * IDENTICO, gate d'uscita sigmoid NUOVO (llm_mlx_sigmoid_rmsnorm_gated), out_proj.
 *
 * Solo il caso "prefill puro" (nessuna continuazione di cache, stato iniziale zero) --
 * l'integrazione nel motore con continuazione conv_state/recurrent state fra chiamate
 * successive (decode) segue lo stesso schema esatto di qwen3_next_linear_block (§13.5,
 * k_cache/v_cache riusati come slot di stato) e va aggiunta al momento del wiring reale,
 * non prima (nessun modello reale da cui verificarla ancora in questa sessione).
 *
 * Tutti i pesi sono mlx_array gia' pronti (non quantizzati qui — la reference li tratta
 * come nn.Linear pieni; se il checkpoint reale li quantizza andra' rivisto al wiring).
 * w_qkv/w_z/w_b/w_a: [out,in] (convenzione nn.Linear). conv_w: [conv_dim, kernel, 1]
 * (convenzione mlx_conv_general per depthwise, groups=conv_dim). */
static bool llm_mlx_qwen4exp_gdn_block(mlx_array* out, mlx_array x,
                                        mlx_array w_qkv, mlx_array w_z, mlx_array w_b, mlx_array w_a,
                                        mlx_array conv_w, mlx_array dt_bias, mlx_array A_log,
                                        mlx_array norm_w, mlx_array w_out,
                                        int num_k_heads, int num_v_heads,
                                        int head_k_dim, int head_v_dim, int conv_kernel_size,
                                        int use_silu, float eps, mlx_stream stream) {
    const int* xshape = mlx_array_shape(x);
    if (xshape == NULL || mlx_array_ndim(x) != 3) return false;
    const int B = xshape[0], T = xshape[1];
    const int R = num_v_heads / num_k_heads;
    const int key_dim = head_k_dim * num_k_heads;
    const int value_dim = head_v_dim * num_v_heads;
    const int conv_dim = 2 * key_dim + value_dim;

    mlx_array mixed = mlx_array_new(), z_flat = mlx_array_new();
    mlx_array beta_lin = mlx_array_new(), alpha_lin = mlx_array_new();
    bool ok = llm_mlx_qwen4exp_plain_linear(&mixed, x, w_qkv, stream);
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&z_flat, x, w_z, stream);
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&beta_lin, x, w_b, stream);
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&alpha_lin, x, w_a, stream);
    mlx_array z = mlx_array_new();
    if (ok) { const int zs[4] = {B, T, num_v_heads, head_v_dim}; ok = mlx_reshape(&z, z_flat, zs, 4, stream) == 0; }

    /* conv1d depthwise: concat([zeros(kernel-1), mixed], axis=1), conv valid, silu.
     * Stato iniziale zero (nessuna continuazione cache in questa funzione, vedi commento). */
    const int cstate_len = conv_kernel_size - 1;
    mlx_array conv_state = mlx_array_new();
    if (ok) { const int cs[3] = {B, cstate_len, conv_dim}; ok = mlx_zeros(&conv_state, cs, 3, MLX_FLOAT32, stream) == 0; }
    mlx_array conv_input = mlx_array_new();
    if (ok) {
        mlx_array parts[2] = {conv_state, mixed};
        mlx_vector_array cv = mlx_vector_array_new_data(parts, 2);
        ok = mlx_concatenate_axis(&conv_input, cv, /*axis=*/1, stream) == 0;
        mlx_vector_array_free(cv);
    }
    mlx_array conv_out = mlx_array_new();
    if (ok) {
        const int c_stride[1] = {1}, c_pad[1] = {0}, c_dil[1] = {1};
        ok = mlx_conv_general(&conv_out, conv_input, conv_w, c_stride, 1, c_pad, 1, c_pad, 1,
                               c_dil, 1, c_dil, 1, /*groups=*/conv_dim, /*flip=*/false, stream) == 0;
    }
    mlx_array convolved = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_silu(&convolved, conv_out, stream);
    

    mlx_array q_part = mlx_array_new(), k_part = mlx_array_new(), v_part = mlx_array_new();
    if (ok) {
        const int secs[2] = {key_dim, 2 * key_dim};
        mlx_vector_array sv = mlx_vector_array_new();
        ok = mlx_split_sections(&sv, convolved, secs, 2, /*axis=*/-1, stream) == 0;
        if (ok) ok = mlx_vector_array_get(&q_part, sv, 0) == 0 &&
                     mlx_vector_array_get(&k_part, sv, 1) == 0 &&
                     mlx_vector_array_get(&v_part, sv, 2) == 0;
        mlx_vector_array_free(sv);
    }
    mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new();
    if (ok) {
        const int qs[4] = {B, T, num_k_heads, head_k_dim}, vs[4] = {B, T, num_v_heads, head_v_dim};
        ok = mlx_reshape(&q4, q_part, qs, 4, stream) == 0 &&
             mlx_reshape(&k4, k_part, qs, 4, stream) == 0 &&
             mlx_reshape(&v4, v_part, vs, 4, stream) == 0;
    }

    mlx_array q_n = mlx_array_new(), k_n = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_qk_l2norm(&q_n, &k_n, q4, k4, powf((float)head_k_dim, -0.5f), stream);
    

    /* g/beta: IDENTICO a qwen3_next -- beta=sigmoid(b); g=exp(-exp(A_log)*softplus(a+dt_bias)) */
    mlx_array beta = mlx_array_new();
    if (ok) ok = mlx_sigmoid(&beta, beta_lin, stream) == 0;
    mlx_array a_plus_dt = mlx_array_new();
    if (ok) ok = mlx_add(&a_plus_dt, alpha_lin, dt_bias, stream) == 0;
    mlx_array sp_exp = mlx_array_new(), one_arr = mlx_array_new_float32(1.0f), sp_1p = mlx_array_new(), sp_log = mlx_array_new();
    if (ok) ok = mlx_exp(&sp_exp, a_plus_dt, stream) == 0;
    if (ok) ok = mlx_add(&sp_1p, sp_exp, one_arr, stream) == 0;
    if (ok) ok = mlx_log(&sp_log, sp_1p, stream) == 0;
    mlx_array eAlog = mlx_array_new(), neg_exp = mlx_array_new(), neg_g = mlx_array_new(), g = mlx_array_new();
    if (ok) ok = mlx_exp(&eAlog, A_log, stream) == 0;
    if (ok) ok = mlx_multiply(&neg_exp, eAlog, sp_log, stream) == 0;
    if (ok) ok = mlx_negative(&neg_g, neg_exp, stream) == 0;
    if (ok) ok = mlx_exp(&g, neg_g, stream) == 0;

    mlx_array q_rep = mlx_array_new(), k_rep = mlx_array_new();
    if (ok) {
        if (R > 1) ok = mlx_repeat_axis(&q_rep, q_n, R, /*axis=*/2, stream) == 0 &&
                        mlx_repeat_axis(&k_rep, k_n, R, /*axis=*/2, stream) == 0;
        else ok = mlx_array_set(&q_rep, q_n) == 0 && mlx_array_set(&k_rep, k_n) == 0;
    }

    mlx_array state = mlx_array_new();
    if (ok) { const int ss[4] = {B, num_v_heads, head_v_dim, head_k_dim}; ok = mlx_zeros(&state, ss, 4, MLX_FLOAT32, stream) == 0; }

    /* loop ricorrente per-token -- IDENTICO a qwen3_next_linear_block righe 6720-6763 */
    mlx_vector_array ys = mlx_vector_array_new();
    for (int t = 0; ok && t < T; t++) {
        const int st0[4] = {0, t, 0, 0}, ones4[4] = {1, 1, 1, 1};
        const int stk[4] = {B, t + 1, num_v_heads, head_k_dim}, stv[4] = {B, t + 1, num_v_heads, head_v_dim};
        const int gt0[3] = {0, t, 0}, gt1[3] = {B, t + 1, num_v_heads}, ones3[3] = {1, 1, 1};
        mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
        mlx_array g_t = mlx_array_new(), beta_t = mlx_array_new();
        ok = mlx_slice(&q_t, q_rep, st0, 4, stk, 4, ones4, 4, stream) == 0 &&
             mlx_slice(&k_t, k_rep, st0, 4, stk, 4, ones4, 4, stream) == 0 &&
             mlx_slice(&v_t, v4, st0, 4, stv, 4, ones4, 4, stream) == 0 &&
             mlx_slice(&g_t, g, gt0, 3, gt1, 3, ones3, 3, stream) == 0 &&
             mlx_slice(&beta_t, beta, gt0, 3, gt1, 3, ones3, 3, stream) == 0;
        mlx_array q_sq = mlx_array_new(), k_sq = mlx_array_new(), v_sq = mlx_array_new();
        mlx_array g_sq = mlx_array_new(), beta_sq = mlx_array_new();
        if (ok) ok = mlx_squeeze_axis(&q_sq, q_t, 1, stream) == 0 &&
                     mlx_squeeze_axis(&k_sq, k_t, 1, stream) == 0 &&
                     mlx_squeeze_axis(&v_sq, v_t, 1, stream) == 0 &&
                     mlx_squeeze_axis(&g_sq, g_t, 1, stream) == 0 &&
                     mlx_squeeze_axis(&beta_sq, beta_t, 1, stream) == 0;
        mlx_array y_t = mlx_array_new();
        if (ok) ok = llm_mlx_gated_delta_step(&y_t, &state, q_sq, k_sq, v_sq, g_sq, beta_sq, stream);
        if (ok) ok = mlx_vector_array_append_value(ys, y_t) == 0;
        mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
        mlx_array_free(g_t); mlx_array_free(beta_t);
        mlx_array_free(q_sq); mlx_array_free(k_sq); mlx_array_free(v_sq);
        mlx_array_free(g_sq); mlx_array_free(beta_sq); mlx_array_free(y_t);
    }
    mlx_array y_stack = mlx_array_new();
    if (ok) ok = mlx_stack_axis(&y_stack, ys, /*axis=*/1, stream) == 0;
    

    mlx_array gated = mlx_array_new();
    if (ok) ok = llm_mlx_sigmoid_rmsnorm_gated(&gated, y_stack, norm_w, z, use_silu, eps, stream);
    
    mlx_array gated_flat = mlx_array_new();
    if (ok) { const int fs[3] = {B, T, value_dim}; ok = mlx_reshape(&gated_flat, gated, fs, 3, stream) == 0; }
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(out, gated_flat, w_out, stream);

    mlx_vector_array_free(ys);
    mlx_array_free(mixed); mlx_array_free(z_flat); mlx_array_free(beta_lin); mlx_array_free(alpha_lin);
    mlx_array_free(z); mlx_array_free(conv_state); mlx_array_free(conv_input); mlx_array_free(conv_out);
    mlx_array_free(convolved); mlx_array_free(q_part); mlx_array_free(k_part); mlx_array_free(v_part);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4); mlx_array_free(q_n); mlx_array_free(k_n);
    mlx_array_free(beta); mlx_array_free(a_plus_dt); mlx_array_free(sp_exp); mlx_array_free(one_arr);
    mlx_array_free(sp_1p); mlx_array_free(sp_log); mlx_array_free(eAlog); mlx_array_free(neg_exp);
    mlx_array_free(neg_g); mlx_array_free(g); mlx_array_free(q_rep); mlx_array_free(k_rep);
    mlx_array_free(state); mlx_array_free(y_stack); mlx_array_free(gated); mlx_array_free(gated_flat);
    return ok;
}

/* ==== N-gram/PLE: parte rimanente (qwen4_exp.py righe 933-1085) ====
 * L'hashing deterministico (_splitmix64/build_layer_multipliers/find_nth_prime_after) e'
 * gia' verificato sopra. Qui: lo shift "ignora EOS" (un token oltre un confine di segmento
 * non vede token del segmento precedente), il mixing xor pesato, il lookup su embedding
 * a shard. Nota: eos_token_id qui e' un SINGOLO int64 (la reference supporta una lista di
 * piu' id EOS -- da estendere con piu' OR se un checkpoint reale lo richiede, non
 * osservato finora). */

/* qwen4_exp.py righe 1018-1041. token_ids: [B,L] int64. shift==0 -> copia identica. */
static bool llm_mlx_qwen4exp_shift_right_ignore_eos(mlx_array* out, mlx_array token_ids,
                                                     int shift, int64_t eos_token_id, mlx_stream stream) {
    if (shift == 0) return mlx_array_set(out, token_ids) == 0;
    const int* shape = mlx_array_shape(token_ids);
    if (shape == NULL || mlx_array_ndim(token_ids) != 2) return false;
    const int B = shape[0], L = shape[1];

    mlx_array positions = mlx_array_new();
    bool ok = mlx_arange(&positions, 0.0, (double)L, 1.0, MLX_INT64, stream) == 0;
    mlx_array positions_row = mlx_array_new();
    if (ok) { const int ps[2] = {1, L}; ok = mlx_reshape(&positions_row, positions, ps, 2, stream) == 0; }

    mlx_array eos_val = mlx_array_new_data(&eos_token_id, NULL, 0, MLX_INT64);
    mlx_array is_eos = mlx_array_new();
    if (ok) ok = mlx_equal(&is_eos, token_ids, eos_val, stream) == 0;
    mlx_array neg1 = mlx_array_new_data(&(int64_t){-1}, NULL, 0, MLX_INT64);
    mlx_array eos_positions = mlx_array_new();
    if (ok) ok = mlx_where(&eos_positions, is_eos, positions_row, neg1, stream) == 0;
    mlx_array prev_incl = mlx_array_new();
    if (ok) ok = mlx_cummax(&prev_incl, eos_positions, /*axis=*/1, /*reverse=*/false, /*inclusive=*/true, stream) == 0;

    /* prev_eos = concat([full((B,1),-1), prev_incl[:, :-1]], axis=1) */
    mlx_array head = mlx_array_new();
    if (ok) { const int hs[2] = {B, 1}; ok = mlx_full(&head, hs, 2, neg1, MLX_INT64, stream) == 0; }
    mlx_array tail = mlx_array_new();
    if (ok) {
        const int t0[2] = {0, 0}, t1[2] = {B, L - 1}, ts[2] = {1, 1};
        ok = mlx_slice(&tail, prev_incl, t0, 2, t1, 2, ts, 2, stream) == 0;
    }
    mlx_array prev_eos = mlx_array_new();
    if (ok) {
        mlx_array parts[2] = {head, tail};
        mlx_vector_array pv = mlx_vector_array_new_data(parts, 2);
        ok = mlx_concatenate_axis(&prev_eos, pv, /*axis=*/1, stream) == 0;
        mlx_vector_array_free(pv);
    }

    /* position_in_segment = positions_row - prev_eos - 1 */
    mlx_array one_i = mlx_array_new_data(&(int64_t){1}, NULL, 0, MLX_INT64);
    mlx_array pis_tmp = mlx_array_new(), pis = mlx_array_new();
    if (ok) ok = mlx_subtract(&pis_tmp, positions_row, prev_eos, stream) == 0;
    if (ok) ok = mlx_subtract(&pis, pis_tmp, one_i, stream) == 0;

    /* source_positions = positions - shift (1-D, poi confrontato/broadcast come riga) */
    mlx_array shift_arr = mlx_array_new_data(&(int64_t){shift}, NULL, 0, MLX_INT64);
    mlx_array source_positions = mlx_array_new();
    if (ok) ok = mlx_subtract(&source_positions, positions_row, shift_arr, stream) == 0;
    mlx_array zero_i = mlx_array_new_data(&(int64_t){0}, NULL, 0, MLX_INT64);
    mlx_array gather_positions_row = mlx_array_new();
    if (ok) ok = mlx_maximum(&gather_positions_row, source_positions, zero_i, stream) == 0;
    mlx_array gather_positions = mlx_array_new();
    if (ok) {
        /* broadcast [1,L] -> [B,L] esplicito (take_along_axis richiede stesso ndim/shape
         * broadcastabile per indices -- costruiamo la forma esplicita con moltiplicazione
         * per un array di uni [B,1], piu' sicuro di affidarsi al broadcast implicito di
         * take_along_axis, non documentato per mlx-c). */
        mlx_array ones_b1 = mlx_array_new();
        const int b1[2] = {B, 1};
        mlx_array one_val = mlx_array_new_data(&(int64_t){1}, NULL, 0, MLX_INT64);
        ok = mlx_full(&ones_b1, b1, 2, one_val, MLX_INT64, stream) == 0;
        if (ok) ok = mlx_multiply(&gather_positions, gather_positions_row, ones_b1, stream) == 0;
        mlx_array_free(ones_b1); mlx_array_free(one_val);
    }
    mlx_array shifted = mlx_array_new();
    if (ok) ok = mlx_take_along_axis(&shifted, token_ids, gather_positions, /*axis=*/1, stream) == 0;

    mlx_array shift_i64 = mlx_array_new_data(&(int64_t){shift}, NULL, 0, MLX_INT64);
    mlx_array valid1 = mlx_array_new(), valid2 = mlx_array_new(), valid = mlx_array_new();
    if (ok) ok = mlx_greater_equal(&valid1, pis, shift_i64, stream) == 0;
    if (ok) ok = mlx_greater_equal(&valid2, source_positions, zero_i, stream) == 0;
    if (ok) ok = mlx_logical_and(&valid, valid1, valid2, stream) == 0;
    if (ok) ok = mlx_where(out, valid, shifted, eos_val, stream) == 0;

    mlx_array_free(positions); mlx_array_free(positions_row); mlx_array_free(eos_val);
    mlx_array_free(is_eos); mlx_array_free(neg1); mlx_array_free(eos_positions);
    mlx_array_free(prev_incl); mlx_array_free(head); mlx_array_free(tail); mlx_array_free(prev_eos);
    mlx_array_free(one_i); mlx_array_free(pis_tmp); mlx_array_free(pis); mlx_array_free(shift_arr);
    mlx_array_free(source_positions); mlx_array_free(zero_i); mlx_array_free(gather_positions_row);
    mlx_array_free(gather_positions); mlx_array_free(shifted); mlx_array_free(shift_i64);
    mlx_array_free(valid1); mlx_array_free(valid2); mlx_array_free(valid);
    return ok;
}

/* qwen4_exp.py righe 1043-1081 (senza continuazione di cache -- "previous" e' passato
 * esplicitamente, gia' concatenato dal chiamante come per il caso "fresh"/primo prefill;
 * la continuazione decode va aggiunta al wiring reale). history = concat([previous,
 * input_ids], axis=1), gia' fatto dal chiamante e passato qui come `history`.
 * layer_multipliers: ngram_size int64 valori (da llm_mlx_qwen4exp_build_layer_multipliers).
 * vocab_sizes/vocab_offsets: array int64 [ngram_heads] (da find_nth_prime_after +
 * somma cumulativa, precalcolati dal chiamante al setup del layer, non per-token).
 * out: [B, input_len, ngram_heads] int64 (id per ogni testa, pronto per il lookup).
 * eos_token_id: l'id EOS vero del modello (obbligatorio, nessun default silenzioso --
 * un valore sbagliato qui corrompe la segmentazione senza errori visibili). */
static bool llm_mlx_qwen4exp_ngram_compute_ids(mlx_array* out, mlx_array history, int input_len,
                                                int ngram_size, int heads_per_ngram,
                                                const int64_t* layer_multipliers,
                                                mlx_array vocab_sizes, mlx_array vocab_offsets,
                                                int64_t eos_token_id, mlx_stream stream) {
    mlx_array* shifted = (mlx_array*)malloc((size_t)ngram_size * sizeof(mlx_array));
    if (shifted == NULL) return false;
    bool ok = true;
    for (int sh = 0; sh < ngram_size && ok; sh++) {
        shifted[sh] = mlx_array_new();
        ok = llm_mlx_qwen4exp_shift_right_ignore_eos(&shifted[sh], history, sh, eos_token_id, stream);
    }

    const int n_ngrams = ngram_size - 1; /* bigram..ngram_size-gram */
    mlx_array* blocks = (mlx_array*)malloc((size_t)n_ngrams * sizeof(mlx_array));
    if (blocks == NULL) { ok = false; }
    for (int g = 0; ok && g < n_ngrams; g++) {
        const int ngram = g + 2;
        const int start = g * heads_per_ngram;
        mlx_array mult0 = mlx_array_new_data(&layer_multipliers[0], NULL, 0, MLX_INT64);
        mlx_array mixed = mlx_array_new();
        ok = mlx_multiply(&mixed, shifted[0], mult0, stream) == 0;
        mlx_array_free(mult0);
        for (int pos = 1; ok && pos < ngram; pos++) {
            mlx_array multp = mlx_array_new_data(&layer_multipliers[pos], NULL, 0, MLX_INT64);
            mlx_array term = mlx_array_new();
            ok = mlx_multiply(&term, shifted[pos], multp, stream) == 0;
            mlx_array_free(multp);
            if (ok) {
                mlx_array xored = mlx_array_new();
                ok = mlx_bitwise_xor(&xored, mixed, term, stream) == 0;
                mlx_array_free(mixed);
                mixed = xored;
            }
            mlx_array_free(term);
        }
        mlx_array mixed_e = mlx_array_new();
        if (ok) { const int es[3] = {mlx_array_dim(mixed,0), mlx_array_dim(mixed,1), 1};
                  ok = mlx_reshape(&mixed_e, mixed, es, 3, stream) == 0; }
        mlx_array_free(mixed);
        mlx_array sizes_g = mlx_array_new(), offsets_g = mlx_array_new();
        if (ok) {
            const int s0[1] = {start}, s1[1] = {start + heads_per_ngram}, ss[1] = {1};
            ok = mlx_slice(&sizes_g, vocab_sizes, s0, 1, s1, 1, ss, 1, stream) == 0 &&
                 mlx_slice(&offsets_g, vocab_offsets, s0, 1, s1, 1, ss, 1, stream) == 0;
        }
        mlx_array rem = mlx_array_new();
        if (ok) ok = mlx_remainder(&rem, mixed_e, sizes_g, stream) == 0;
        mlx_array_free(mixed_e);
        if (ok) ok = mlx_add(&blocks[g], rem, offsets_g, stream) == 0;
        mlx_array_free(rem); mlx_array_free(sizes_g); mlx_array_free(offsets_g);
    }
    for (int sh = 0; sh < ngram_size; sh++) mlx_array_free(shifted[sh]);
    free(shifted);

    mlx_array full = mlx_array_new();
    if (ok) {
        mlx_vector_array bv = mlx_vector_array_new_data(blocks, (size_t)n_ngrams);
        ok = mlx_concatenate_axis(&full, bv, /*axis=*/-1, stream) == 0;
        mlx_vector_array_free(bv);
    }
    for (int g = 0; g < n_ngrams; g++) mlx_array_free(blocks[g]);
    free(blocks);

    if (ok) {
        const int total_len = mlx_array_dim(full, 1);
        const int s0[3] = {0, total_len - input_len, 0};
        const int s1[3] = {mlx_array_dim(full,0), total_len, mlx_array_dim(full,2)};
        const int ss[3] = {1, 1, 1};
        ok = mlx_slice(out, full, s0, 3, s1, 3, ss, 3, stream) == 0;
    }
    mlx_array_free(full);
    return ok;
}

/* qwen4_exp.py righe 933-963 (ShardedEmbedding). ids: [B,L,ngram_heads] int64 (o
 * qualunque forma -- generico sull'ultimo asse). shard_weights: array di n_shards
 * mlx_array [rows_per_shard, dim]. Richiede un readback host degli id (come la
 * reference: `.tolist()`) per sapere quali shard sono usati -- e' un round-trip per
 * TOKEN (o per chunk di prefill), non per elemento, e la reference stessa lo fa cosi'
 * per evitare di materializzare l'intera tabella da 51 miliardi di parametri. */
static bool llm_mlx_qwen4exp_sharded_embedding_lookup(mlx_array* out, mlx_array ids,
                                                       mlx_array* shard_weights, int n_shards,
                                                       int rows_per_shard, mlx_stream stream) {
    mlx_array ids_i32 = mlx_array_new();
    bool ok = mlx_astype(&ids_i32, ids, MLX_INT32, stream) == 0;
    if (ok) ok = mlx_array_eval(ids_i32) == 0;
    const int32_t* idata = ok ? mlx_array_data_int32(ids_i32) : NULL;
    size_t n = ok ? mlx_array_size(ids_i32) : 0;
    if (!ok || idata == NULL) { mlx_array_free(ids_i32); return false; }

    bool* used = (bool*)calloc((size_t)n_shards, sizeof(bool));
    if (used == NULL) { mlx_array_free(ids_i32); return false; }
    for (size_t i = 0; i < n; i++) {
        int32_t sid = idata[i] / rows_per_shard;
        if (sid >= 0 && sid < n_shards) used[sid] = true;
    }

    mlx_array rows_per_shard_arr = mlx_array_new_data(&rows_per_shard, NULL, 0, MLX_INT32);
    mlx_array output = mlx_array_new();
    bool have_output = false;
    for (int sid = 0; ok && sid < n_shards; sid++) {
        if (!used[sid]) continue;
        mlx_array sid_arr = mlx_array_new_data(&sid, NULL, 0, MLX_INT32);
        mlx_array shard_off = mlx_array_new();
        ok = mlx_multiply(&shard_off, sid_arr, rows_per_shard_arr, stream) == 0;
        mlx_array local = mlx_array_new();
        if (ok) ok = mlx_subtract(&local, ids_i32, shard_off, stream) == 0;
        mlx_array local_c = mlx_array_new();
        mlx_array rmax = mlx_array_new_data(&(int32_t){rows_per_shard - 1}, NULL, 0, MLX_INT32);
        mlx_array zero32 = mlx_array_new_data(&(int32_t){0}, NULL, 0, MLX_INT32);
        if (ok) ok = mlx_clip(&local_c, local, zero32, rmax, stream) == 0;
        mlx_array gathered = mlx_array_new();
        if (ok) ok = mlx_take_axis(&gathered, shard_weights[sid], local_c, /*axis=*/0, stream) == 0;
        mlx_array mask = mlx_array_new(), shard_ids_full = mlx_array_new();
        if (ok) ok = mlx_floor_divide(&shard_ids_full, ids_i32, rows_per_shard_arr, stream) == 0;
        if (ok) ok = mlx_equal(&mask, shard_ids_full, sid_arr, stream) == 0;
        mlx_array mask_e = mlx_array_new();
        if (ok) {
            int mnd = mlx_array_ndim(mask);
            int* ms = (int*)malloc((size_t)(mnd + 1) * sizeof(int));
            if (ms == NULL) { ok = false; }
            else {
                const int* mshape = mlx_array_shape(mask);
                for (int i = 0; i < mnd; i++) ms[i] = mshape[i];
                ms[mnd] = 1;
                ok = mlx_reshape(&mask_e, mask, ms, mnd + 1, stream) == 0;
                free(ms);
            }
        }
        mlx_array gathered_m = mlx_array_new(), zerof = mlx_array_new_float32(0.0f);
        if (ok) ok = mlx_where(&gathered_m, mask_e, gathered, zerof, stream) == 0;
        if (ok) {
            if (!have_output) { ok = mlx_array_set(&output, gathered_m) == 0; have_output = true; }
            else {
                mlx_array sum = mlx_array_new();
                ok = mlx_add(&sum, output, gathered_m, stream) == 0;
                mlx_array_free(output); output = sum;
            }
        }
        mlx_array_free(sid_arr); mlx_array_free(shard_off); mlx_array_free(local);
        mlx_array_free(local_c); mlx_array_free(rmax); mlx_array_free(zero32);
        mlx_array_free(gathered); mlx_array_free(mask); mlx_array_free(shard_ids_full);
        mlx_array_free(mask_e); mlx_array_free(gathered_m); mlx_array_free(zerof);
    }
    free(used);
    mlx_array_free(ids_i32); mlx_array_free(rows_per_shard_arr);
    if (ok && have_output) { ok = mlx_array_set(out, output) == 0; mlx_array_free(output); return ok; }
    mlx_array_free(output);
    return false; /* nessuno shard usato: caso degenere, non atteso con input reali */
}

/* PLELayer (qwen4_exp.py righe 1088-1172) -- usa gli embedding n-gram (gia' calcolati
 * dal chiamante via ngram_compute_ids + sharded_embedding_lookup + flatten, vedi sopra)
 * per iniettare un termine additivo in hidden_states, sui soli layer elencati in
 * ple_layer_ids. mask (padding) NON gestito qui (nessuna richiesta multi-batch in
 * LLMEnv, single-stream) -- se servisse, applicare `where(mask,...)` come la reference
 * prima di chiamare questa funzione.
 * hidden_states: [B,L,hc_width]. key_lin/values: GIA' proiettati dal chiamante (plain O
 * quantizzato, stesso principio di qsa_indexer_postproject/qsa_attention_postproject --
 * key_proj/value_proj sono quantizzati sul checkpoint reale, verificato 2026-09-06).
 * key_lin: [B,L,hc_width] (embeddings @ key_proj_w.T). values: [B,L,hidden_size].
 * conv_state_in: [B,(ple_conv_kernel_size-1)*ngram_size,hc_width] o ctx==NULL (fresh,
 * equivalente a zeri -- primo prefill del layer). conv_state_out: SEMPRE popolato (gli
 * ultimi (ple_conv_kernel_size-1)*ngram_size passi di `normalized`), da salvare dal
 * chiamante per la chiamata successiva (stesso schema esatto di k_cache/v_cache/GDN
 * altrove in questo motore). out: [B,L,hc_width] (il chiamante lo SOMMA a hidden_states,
 * non sostituisce). */
static bool llm_mlx_qwen4exp_ple_layer(mlx_array* out, mlx_array* conv_state_out,
                                        mlx_array hidden_states, mlx_array key_lin, mlx_array values,
                                        mlx_array norm_key_w, mlx_array norm_query_w, mlx_array norm_conv_w,
                                        mlx_array conv_w, mlx_array conv_state_in, int hc_count, int hidden_size,
                                        int ple_conv_kernel_size, int ngram_size, float eps, mlx_stream stream) {
    const int* hshape = mlx_array_shape(hidden_states);
    if (hshape == NULL || mlx_array_ndim(hidden_states) != 3) return false;
    const int B = hshape[0], L = hshape[1], hc_width = hshape[2];

    mlx_array keys_n = mlx_array_new();
    bool ok = llm_mlx_zero_centered_rmsnorm(&keys_n, key_lin, norm_key_w, hidden_size, eps, stream);
    mlx_array keys = mlx_array_new();
    if (ok) { const int ks[4] = {B, L, hc_count, hidden_size}; ok = mlx_reshape(&keys, keys_n, ks, 4, stream) == 0; }

    mlx_array values_e = mlx_array_new();
    if (ok) { const int vs[4] = {B, L, 1, hidden_size}; ok = mlx_reshape(&values_e, values, vs, 4, stream) == 0; }

    mlx_array queries_n = mlx_array_new(), queries = mlx_array_new();
    if (ok) ok = llm_mlx_zero_centered_rmsnorm(&queries_n, hidden_states, norm_query_w, hidden_size, eps, stream);
    if (ok) { const int qs[4] = {B, L, hc_count, hidden_size}; ok = mlx_reshape(&queries, queries_n, qs, 4, stream) == 0; }

    mlx_array kq = mlx_array_new(), gate_sum = mlx_array_new();
    if (ok) ok = mlx_multiply(&kq, keys, queries, stream) == 0;
    if (ok) ok = mlx_sum_axis(&gate_sum, kq, -1, /*keepdims=*/true, stream) == 0;
    mlx_array sqrt_hidden = mlx_array_new_float32(sqrtf((float)hidden_size));
    mlx_array gate0 = mlx_array_new();
    if (ok) ok = mlx_divide(&gate0, gate_sum, sqrt_hidden, stream) == 0;
    mlx_array gate_abs = mlx_array_new(), eps_arr = mlx_array_new_float32(1e-6f), gate_abs_clamped = mlx_array_new();
    if (ok) ok = mlx_abs(&gate_abs, gate0, stream) == 0;
    if (ok) ok = mlx_maximum(&gate_abs_clamped, gate_abs, eps_arr, stream) == 0;
    mlx_array gate_sqrt = mlx_array_new(), gate_sign = mlx_array_new(), gate = mlx_array_new();
    if (ok) ok = mlx_sqrt(&gate_sqrt, gate_abs_clamped, stream) == 0;
    if (ok) ok = mlx_sign(&gate_sign, gate0, stream) == 0;
    if (ok) ok = mlx_multiply(&gate, gate_sqrt, gate_sign, stream) == 0;
    mlx_array gate_sig = mlx_array_new();
    if (ok) ok = mlx_sigmoid(&gate_sig, gate, stream) == 0;
    
    

    mlx_array gated4 = mlx_array_new();
    if (ok) ok = mlx_multiply(&gated4, gate_sig, values_e, stream) == 0;
    mlx_array gated_flat = mlx_array_new();
    if (ok) { const int gs[3] = {B, L, hc_width}; ok = mlx_reshape(&gated_flat, gated4, gs, 3, stream) == 0; }

    mlx_array normalized = mlx_array_new();
    if (ok) ok = llm_mlx_zero_centered_rmsnorm(&normalized, gated_flat, norm_conv_w, hidden_size, eps, stream);

    const int conv_state_len = (ple_conv_kernel_size - 1) * ngram_size;
    mlx_array conv_state = mlx_array_new();
    if (ok) {
        if (conv_state_in.ctx != NULL) ok = mlx_array_set(&conv_state, conv_state_in) == 0;
        else { const int cs[3] = {B, conv_state_len, hc_width}; ok = mlx_zeros(&conv_state, cs, 3, MLX_FLOAT32, stream) == 0; }
    }
    mlx_array conv_input = mlx_array_new();
    if (ok) {
        mlx_array parts[2] = {conv_state, normalized};
        mlx_vector_array cv = mlx_vector_array_new_data(parts, 2);
        ok = mlx_concatenate_axis(&conv_input, cv, /*axis=*/1, stream) == 0;
        mlx_vector_array_free(cv);
    }
    mlx_array conv_out = mlx_array_new();
    if (ok) {
        const int c_stride[1] = {1}, c_pad[1] = {0}, c_idil[1] = {1}, c_kdil[1] = {ngram_size};
        ok = mlx_conv_general(&conv_out, conv_input, conv_w, c_stride, 1, c_pad, 1, c_pad, 1,
                               c_kdil, 1, c_idil, 1, /*groups=*/hc_width, /*flip=*/false, stream) == 0;
    }
    mlx_array conv_silu = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_silu(&conv_silu, conv_out, stream);
    if (ok) ok = mlx_add(out, gated_flat, conv_silu, stream) == 0;
    
    /* Stato per la chiamata successiva: ultimi conv_state_len passi di conv_input, stesso
     * schema esatto di qwen3_next_linear_block/qwen4_linear_gdn_block per il conv1d GDN. */
    if (ok && conv_state_out != NULL) {
        const int total_len = L + conv_state_len;
        const int cst0[3] = {0, total_len - conv_state_len, 0};
        const int cst1[3] = {B, total_len, hc_width};
        const int cstr[3] = {1, 1, 1};
        ok = mlx_slice(conv_state_out, conv_input, cst0, 3, cst1, 3, cstr, 3, stream) == 0;
    }

    mlx_array_free(keys_n); mlx_array_free(keys);
    mlx_array_free(values_e); mlx_array_free(queries_n); mlx_array_free(queries);
    mlx_array_free(kq); mlx_array_free(gate_sum); mlx_array_free(sqrt_hidden); mlx_array_free(gate0);
    mlx_array_free(gate_abs); mlx_array_free(eps_arr); mlx_array_free(gate_abs_clamped);
    mlx_array_free(gate_sqrt); mlx_array_free(gate_sign); mlx_array_free(gate); mlx_array_free(gate_sig);
    mlx_array_free(gated4); mlx_array_free(gated_flat); mlx_array_free(normalized);
    mlx_array_free(conv_state); mlx_array_free(conv_input); mlx_array_free(conv_out); mlx_array_free(conv_silu);
    return ok;
}

/* ==== QSA (Qwen Sparse Attention) -- fase 4, la piu' rischiosa (qwen4_exp.py righe
 * 578-890 + qwen4_exp_cache.py). Iniziamo dalla cache dell'indexer, semplificata per il
 * caso single-stream di LLMEnv (B=1, nessun left-padding/merge/filter multi-richiesta --
 * quella complessita' nella reference serve SOLO al serving concorrente, mai necessaria
 * qui). ====
 *
 * LlmQwen4ExpQSACache: ring buffer raw di `compress_ratio` posizioni + lista growable di
 * chiavi compresse (cresciuta a step di 256, come la reference). Un token alla volta
 * scritto nel ring; ogni volta che il ring si riempie, la MEDIA del gruppo viene
 * normalizzata (k_layernorm) e ruotata (RoPE alla posizione del PRIMO token del gruppo,
 * non quella corrente -- per questo serve llm_mlx_qwen4exp_rope_positions a posizioni
 * esplicite, non un offset scalare) e appesa. */
typedef struct {
    mlx_array ring;       /* [1, compress_ratio, dim] -- ctx==NULL finche' non inizializzato */
    mlx_array compressed; /* [1, capacity, dim] -- ctx==NULL finche' vuoto */
    int offset;
    int compressed_count;
    int compress_ratio;
    int dim;
} LlmQwen4ExpQSACache;

static void llm_mlx_qwen4exp_qsa_cache_init(LlmQwen4ExpQSACache* c, int compress_ratio, int dim) {
    c->ring = mlx_array_new();
    c->compressed = mlx_array_new();
    c->offset = 0;
    c->compressed_count = 0;
    c->compress_ratio = compress_ratio;
    c->dim = dim;
}

static void llm_mlx_qwen4exp_qsa_cache_free(LlmQwen4ExpQSACache* c) {
    mlx_array_free(c->ring);
    mlx_array_free(c->compressed);
}

/* raw_keys: [1,L,dim] float (gia' proiettate, PRIMA di q/k_layernorm -- la normalizzazione
 * si applica solo alla MEDIA del gruppo completato, mai ai singoli raw token). Aggiorna
 * `c` in-place; *out_compressed riceve [1, compressed_count_aggiornato, dim] (puo' essere
 * [1,0,dim] se nessun gruppo ancora completo -- il chiamante deve trattarlo come "nessuna
 * chiave compressa disponibile", non come un errore). */
static bool llm_mlx_qwen4exp_qsa_cache_update(LlmQwen4ExpQSACache* c, mlx_array raw_keys,
                                               mlx_array k_layernorm_w, int rotary_dim,
                                               float rope_theta, mlx_array* out_compressed,
                                               mlx_stream stream) {
    const int L = mlx_array_dim(raw_keys, 1);
    const int dim = c->dim;
    bool ok = true;
    if (c->ring.ctx == NULL) {
        const int rs[3] = {1, c->compress_ratio, dim};
        ok = mlx_zeros(&c->ring, rs, 3, MLX_FLOAT32, stream) == 0;
    }
    mlx_array* new_groups = (mlx_array*)malloc((size_t)(L > 0 ? L : 1) * sizeof(mlx_array));
    if (new_groups == NULL) return false;
    int n_new = 0;

    for (int t = 0; ok && t < L; t++) {
        const int position = c->offset + t;
        const int slot = position % c->compress_ratio;
        mlx_array token_row = mlx_array_new();
        { const int s0[3] = {0, t, 0}, s1[3] = {1, t + 1, dim}, ss[3] = {1, 1, 1};
          ok = mlx_slice(&token_row, raw_keys, s0, 3, s1, 3, ss, 3, stream) == 0; }
        if (ok) {
            const int u0[3] = {0, slot, 0}, u1[3] = {1, slot + 1, dim}, us[3] = {1, 1, 1};
            mlx_array new_ring = mlx_array_new();
            ok = mlx_slice_update(&new_ring, c->ring, token_row, u0, 3, u1, 3, us, 3, stream) == 0;
            if (ok) { mlx_array_free(c->ring); c->ring = new_ring; } else mlx_array_free(new_ring);
        }
        mlx_array_free(token_row);

        if (ok && (position + 1) % c->compress_ratio == 0) {
            mlx_array pooled = mlx_array_new();
            ok = mlx_mean_axis(&pooled, c->ring, /*axis=*/1, /*keepdims=*/false, stream) == 0;
            mlx_array normed = mlx_array_new();
            if (ok) ok = llm_mlx_zero_centered_rmsnorm(&normed, pooled, k_layernorm_w, /*group=*/0, 1e-6f, stream);
            mlx_array normed4 = mlx_array_new();
            if (ok) { const int ns[4] = {1, 1, 1, dim}; ok = mlx_reshape(&normed4, normed, ns, 4, stream) == 0; }
            const int64_t group_start = (int64_t)(position + 1 - c->compress_ratio);
            mlx_array pos_arr = mlx_array_new_data(&group_start, (const int[]){1, 1}, 2, MLX_INT64);
            mlx_array rotated = mlx_array_new();
            if (ok) ok = llm_mlx_qwen4exp_rope_positions(&rotated, normed4, pos_arr, rotary_dim, rope_theta, stream);
            mlx_array group_final = mlx_array_new();
            if (ok) { const int fs[3] = {1, 1, dim}; ok = mlx_reshape(&group_final, rotated, fs, 3, stream) == 0; }
            if (ok) new_groups[n_new++] = group_final; else mlx_array_free(group_final);
            mlx_array_free(pooled); mlx_array_free(normed); mlx_array_free(normed4);
            mlx_array_free(pos_arr); mlx_array_free(rotated);
        }
    }
    c->offset += L;

    if (ok && n_new > 0) {
        const int old_count = c->compressed_count;
        const int new_count = old_count + n_new;
        const int old_capacity = (c->compressed.ctx != NULL) ? mlx_array_dim(c->compressed, 1) : 0;
        const int step = 256;
        const int new_capacity = ((new_count + step - 1) / step) * step;
        if (new_capacity > old_capacity) {
            mlx_array expanded = mlx_array_new();
            const int es[3] = {1, new_capacity, dim};
            ok = mlx_zeros(&expanded, es, 3, MLX_FLOAT32, stream) == 0;
            if (ok && c->compressed.ctx != NULL && old_capacity > 0) {
                const int u0[3] = {0, 0, 0}, u1[3] = {1, old_capacity, dim}, us[3] = {1, 1, 1};
                mlx_array merged = mlx_array_new();
                ok = mlx_slice_update(&merged, expanded, c->compressed, u0, 3, u1, 3, us, 3, stream) == 0;
                if (ok) { mlx_array_free(expanded); expanded = merged; } else mlx_array_free(merged);
            }
            if (ok) { mlx_array_free(c->compressed); c->compressed = expanded; } else mlx_array_free(expanded);
        }
        for (int g = 0; ok && g < n_new; g++) {
            const int u0[3] = {0, old_count + g, 0}, u1[3] = {1, old_count + g + 1, dim}, us[3] = {1, 1, 1};
            mlx_array merged = mlx_array_new();
            ok = mlx_slice_update(&merged, c->compressed, new_groups[g], u0, 3, u1, 3, us, 3, stream) == 0;
            if (ok) { mlx_array_free(c->compressed); c->compressed = merged; } else mlx_array_free(merged);
        }
        if (ok) c->compressed_count = new_count;
    }
    for (int g = 0; g < n_new; g++) mlx_array_free(new_groups[g]);
    free(new_groups);

    if (ok) {
        if (c->compressed_count == 0) {
            const int zs[3] = {1, 0, dim};
            ok = mlx_zeros(out_compressed, zs, 3, MLX_FLOAT32, stream) == 0;
        } else {
            const int s0[3] = {0, 0, 0}, s1[3] = {1, c->compressed_count, dim}, ss[3] = {1, 1, 1};
            ok = mlx_slice(out_compressed, c->compressed, s0, 3, s1, 3, ss, 3, stream) == 0;
        }
    }
    return ok;
}

/* QSAIndexer -- proiezione query/raw_key (qwen4_exp.py righe ~600-650), INDIPENDENTE
 * dalla q/k/v della QSAAttention principale (pesi propri index_qk_proj/q_layernorm/
 * k_layernorm). num_kv_heads dell'indexer e' sempre 1 nella reference. hidden_states:
 * [1,L,hidden]. index_qk_proj_w: [(num_heads+1)*head_dim, hidden]. cache_offset:
 * posizione assoluta del PRIMO token di questa chiamata (stessa convenzione di
 * qsa_attention_core). *out_query_bhtd: [1,H,L,D] gia' normalizzato+ruotato, da usare
 * per lo scoring in qsa_select_mask. *out_raw_key: [1,L,D] (kv head singolo, squeezed),
 * da passare a llm_mlx_qwen4exp_qsa_cache_update (che applica k_layernorm+RoPE SOLO
 * alla media del gruppo completato, mai al raw token). */
static bool llm_mlx_qwen4exp_qsa_indexer_postproject(
    mlx_array* out_query_bhtd, mlx_array* out_raw_key, mlx_array projected,
    mlx_array q_layernorm_w, int num_heads, int head_dim,
    int rotary_dim, float rope_theta, int cache_offset, mlx_stream stream) {
    const int* pshape = mlx_array_shape(projected);
    if (pshape == NULL || mlx_array_ndim(projected) != 3) return false;
    const int B = pshape[0], L = pshape[1];
    const int query_width = num_heads * head_dim;
    bool ok = true;

    mlx_vector_array split = mlx_vector_array_new();
    mlx_array query0 = mlx_array_new(), raw_key0 = mlx_array_new();
    if (ok) {
        const int secs[1] = {query_width};
        ok = mlx_split_sections(&split, projected, secs, 1, /*axis=*/-1, stream) == 0;
        if (ok) ok = mlx_vector_array_get(&query0, split, 0) == 0 &&
                     mlx_vector_array_get(&raw_key0, split, 1) == 0;
    }
    mlx_vector_array_free(split);

    mlx_array query_r = mlx_array_new();
    if (ok) { const int qs[4] = {B, L, num_heads, head_dim}; ok = mlx_reshape(&query_r, query0, qs, 4, stream) == 0; }
    mlx_array raw_key_sq = mlx_array_new();
    if (ok) { const int rs[3] = {B, L, head_dim}; ok = mlx_reshape(&raw_key_sq, raw_key0, rs, 3, stream) == 0; }

    mlx_array query_n = mlx_array_new();
    if (ok) ok = llm_mlx_zero_centered_rmsnorm(&query_n, query_r, q_layernorm_w, /*group=*/0, 1e-6f, stream);

    const int perm[4] = {0, 2, 1, 3};
    mlx_array query_bhtd = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&query_bhtd, query_n, perm, 4, stream) == 0;

    mlx_array positions = mlx_array_new();
    if (ok) {
        mlx_array ar = mlx_array_new();
        ok = mlx_arange(&ar, (double)cache_offset, (double)(cache_offset + L), 1.0, MLX_INT64, stream) == 0;
        if (ok) { const int prs[1] = {L}; ok = mlx_reshape(&positions, ar, prs, 1, stream) == 0; }
        mlx_array_free(ar);
    }
    mlx_array query_rope = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_rope(&query_rope, query_bhtd, positions, rotary_dim, rope_theta, stream);

    if (ok) ok = mlx_array_set(out_query_bhtd, query_rope) == 0 && mlx_array_set(out_raw_key, raw_key_sq) == 0;

    mlx_array_free(query0); mlx_array_free(raw_key0);
    mlx_array_free(query_r); mlx_array_free(raw_key_sq); mlx_array_free(query_n);
    mlx_array_free(query_bhtd); mlx_array_free(positions); mlx_array_free(query_rope);
    return ok;
}

/* Wrapper usato dai test standalone (proiezione PLAIN, non quantizzata): calcola
 * `projected` con un matmul semplice e delega a qsa_indexer_postproject. Il wiring reale
 * nel motore (pesi quantizzati) chiama invece quantized_linear_cfg per ottenere
 * `projected` e passa direttamente a qsa_indexer_postproject (vedi qwen4_full_qsa_block
 * in llmengine_mlx.c) -- STESSA funzione di post-processing, stesso risultato
 * verificato, la sola differenza e' come si ottiene `projected`. */
static bool llm_mlx_qwen4exp_qsa_indexer_project(
    mlx_array* out_query_bhtd, mlx_array* out_raw_key, mlx_array hidden_states,
    mlx_array index_qk_proj_w, mlx_array q_layernorm_w, int num_heads, int head_dim,
    int rotary_dim, float rope_theta, int cache_offset, mlx_stream stream) {
    mlx_array projected = mlx_array_new();
    bool ok = llm_mlx_qwen4exp_plain_linear(&projected, hidden_states, index_qk_proj_w, stream);
    if (ok) ok = llm_mlx_qwen4exp_qsa_indexer_postproject(out_query_bhtd, out_raw_key, projected,
                                                           q_layernorm_w, num_heads, head_dim,
                                                           rotary_dim, rope_theta, cache_offset, stream);
    mlx_array_free(projected);
    return ok;
}

/* QSAIndexer -- decisione budget + selezione top-k a blocchi (qwen4_exp.py righe
 * ~680-764). E' il pezzo che DA' IL NOME a "QSA": senza questo, l'attenzione resta
 * densa/causale come qualunque altro modello. Ritorna in *out_mask una maschera
 * ADDITIVA [1,1,L,physical_kv_length] (0.0 = permesso, -inf = bloccato) da passare a
 * una SDPA con maschera esplicita (schema di llm_mlx_gqa_sdpa_prefix), oppure
 * out_mask->ctx==NULL se il contesto e' ancora SOTTO indexer_budget (nessuna
 * sparsificazione: il chiamante deve usare l'attenzione causale piena, invariata).
 * query_bhtd: [1,H,L,D] gia' normalizzato+ruotato (output di qsa_indexer_project).
 * compressed_keys: [1,available_blocks,D] (ctx puo' essere NULL, available_blocks==0).
 * cache_offset: posizione assoluta del PRIMO token di questa chiamata, PRIMA
 * dell'update della cache indexer (stessa convenzione di qsa_attention_core: la
 * reference legge cache._offsets PRIMA di cache.update()).
 * Selezione calcolata su HOST: available_blocks e block_topk sono piccoli (decine-
 * centinaia, mai l'intero contesto) -- stesso pattern gia' usato in
 * llm_mlx_qwen4exp_sharded_embedding_lookup per un motivo analogo (evitare di
 * materializzare argpartition su device per un problema minuscolo). Semplificazione
 * single-stream di LLMEnv: niente left_padding/input_start/valid_length (sempre
 * 0/0/L) -- la reference li porta solo per il serving multi-richiesta concorrente. */
static bool llm_mlx_qwen4exp_qsa_select_mask_host(
    mlx_array* out_mask, mlx_array query_bhtd, mlx_array compressed_keys,
    int compress_ratio, int indexer_budget, int cache_offset, int physical_kv_length,
    mlx_stream stream) {
    const int block_topk = indexer_budget / compress_ratio;
    if (physical_kv_length / compress_ratio <= block_topk) return true; /* sotto budget: out_mask resta ctx==NULL */

    const int* qshape = mlx_array_shape(query_bhtd);
    if (qshape == NULL || mlx_array_ndim(query_bhtd) != 4) return false;
    const int H = qshape[1], L = qshape[2], D = qshape[3];
    const int available_blocks = (compressed_keys.ctx != NULL) ? mlx_array_dim(compressed_keys, 1) : 0;

    mlx_array q_eval = mlx_array_new();
    bool ok = mlx_astype(&q_eval, query_bhtd, MLX_FLOAT32, stream) == 0;
    if (ok) ok = mlx_array_eval(q_eval) == 0;
    const float* qdata = ok ? mlx_array_data_float32(q_eval) : NULL;
    if (!ok || qdata == NULL) { mlx_array_free(q_eval); return false; }

    mlx_array k_eval = mlx_array_new();
    const float* kdata = NULL;
    if (available_blocks > 0) {
        ok = mlx_astype(&k_eval, compressed_keys, MLX_FLOAT32, stream) == 0;
        if (ok) ok = mlx_array_eval(k_eval) == 0;
        kdata = ok ? mlx_array_data_float32(k_eval) : NULL;
        if (!ok || kdata == NULL) { mlx_array_free(q_eval); mlx_array_free(k_eval); return false; }
    }

    float* scores = NULL;
    if (available_blocks > block_topk) {
        scores = (float*)malloc((size_t)L * (size_t)available_blocks * sizeof(float));
        if (scores == NULL) { mlx_array_free(q_eval); mlx_array_free(k_eval); return false; }
        const float inv_sqrt_d = 1.0f / sqrtf((float)D);
        for (int l = 0; l < L; l++) {
            for (int b = 0; b < available_blocks; b++) {
                float acc = 0.0f;
                for (int h = 0; h < H; h++) {
                    const float* qp = qdata + ((size_t)h * L + (size_t)l) * D;
                    const float* kp = kdata + (size_t)b * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qp[d] * kp[d];
                    if (dot > 0.0f) acc += dot;
                }
                scores[l * available_blocks + b] = acc * inv_sqrt_d;
            }
        }
    }
    mlx_array_free(q_eval); mlx_array_free(k_eval);

    bool* selected = (bool*)calloc((size_t)L * (size_t)physical_kv_length, sizeof(bool));
    int* block_idx = (int*)malloc((size_t)block_topk * sizeof(int));
    bool* taken = (bool*)malloc((size_t)(available_blocks > 0 ? available_blocks : 1) * sizeof(bool));
    bool ok2 = (selected != NULL && block_idx != NULL && taken != NULL);

    for (int l = 0; ok2 && l < L; l++) {
        const int logical_position = cache_offset + l;
        const long complete_counts_score = (long)(logical_position + 1) / compress_ratio;
        const long complete_counts = (long)(logical_position + 1) / compress_ratio;
        const bool use_selected = (available_blocks > block_topk) && (complete_counts > block_topk);

        if (use_selected) {
            long cc = complete_counts_score; if (cc > available_blocks) cc = available_blocks;
            for (int b = 0; b < available_blocks; b++) taken[b] = false;
            for (int k = 0; k < block_topk; k++) {
                int best = -1; float best_score = -INFINITY;
                for (int b = 0; b < cc; b++) {
                    if (taken[b]) continue;
                    if (scores[l * available_blocks + b] > best_score) { best_score = scores[l * available_blocks + b]; best = b; }
                }
                block_idx[k] = best;
                if (best >= 0) taken[best] = true;
            }
        } else {
            for (int k = 0; k < block_topk; k++) block_idx[k] = k;
        }

        for (int k = 0; k < block_topk; k++) {
            if (k >= complete_counts) continue; /* block_valid = k < min(complete_counts, block_topk) */
            const int blk = block_idx[k];
            if (blk < 0) continue;
            for (int r = 0; r < compress_ratio; r++) {
                long tok = (long)blk * compress_ratio + r;
                if (tok < 0) tok = 0; else if (tok >= physical_kv_length) tok = physical_kv_length - 1;
                selected[(size_t)l * physical_kv_length + tok] = true;
            }
        }
        const long tail_start = complete_counts * compress_ratio;
        const long tail_counts = (logical_position + 1) - tail_start;
        for (int r = 0; r < compress_ratio; r++) {
            if (r >= tail_counts) continue;
            long tok = tail_start + r;
            if (tok < 0) tok = 0; else if (tok >= physical_kv_length) tok = physical_kv_length - 1;
            selected[(size_t)l * physical_kv_length + tok] = true;
        }
    }
    free(block_idx); free(taken);
    if (scores) free(scores);
    if (!ok2) { free(selected); return false; }

    float* mask_data = (float*)malloc((size_t)L * (size_t)physical_kv_length * sizeof(float));
    if (mask_data == NULL) { free(selected); return false; }
    for (size_t i = 0; i < (size_t)L * (size_t)physical_kv_length; i++)
        mask_data[i] = selected[i] ? 0.0f : -INFINITY;
    free(selected);

    const int mshape[4] = {1, 1, L, physical_kv_length};
    mlx_array mask = mlx_array_new_data(mask_data, mshape, 4, MLX_FLOAT32);
    free(mask_data);
    ok = mlx_array_set(out_mask, mask) == 0;
    mlx_array_free(mask);
    return ok;
}

static bool llm_mlx_qwen4exp_qsa_select_mask(
    mlx_array* out_mask, mlx_array query_bhtd, mlx_array compressed_keys,
    int compress_ratio, int indexer_budget, int cache_offset, int physical_kv_length,
    mlx_stream stream) {
    const int block_topk = indexer_budget / compress_ratio;
    if (physical_kv_length / compress_ratio <= block_topk) return true; /* sotto budget: out_mask resta ctx==NULL */

    const int* qshape = mlx_array_shape(query_bhtd);
    if (qshape == NULL || mlx_array_ndim(query_bhtd) != 4) return false;
    const int H = qshape[1], L = qshape[2], D = qshape[3];
    const int available_blocks = (compressed_keys.ctx != NULL) ? mlx_array_dim(compressed_keys, 1) : 0;

    /* SCORING + TOP-K ON-DEVICE (fix 2026-09-18): niente readback di q/k, niente loop CPU
     * L*B*H*D, niente sync per layer. scores[L,B] = sum_h max(q_h . k_b, 0)/sqrt(D);
     * top-k via argpartition; si rileggono SOLO gli indici scelti [L, topk] (piccoli). */
    int* block_idx = (int*)malloc((size_t)L * (size_t)block_topk * sizeof(int));
    bool* selected = (bool*)calloc((size_t)L * (size_t)physical_kv_length, sizeof(bool));
    bool ok2 = (selected != NULL && block_idx != NULL);
    bool ok = ok2;

    mlx_array qf = mlx_array_new(), q2 = mlx_array_new(), kf = mlx_array_new(), kT = mlx_array_new();
    mlx_array prod = mlx_array_new(), prod3 = mlx_array_new();
    mlx_array zf = mlx_array_new_float32(0.0f), sdv = mlx_array_new_float32(sqrtf((float)D));
    mlx_array pos = mlx_array_new(), ssum = mlx_array_new(), sc = mlx_array_new();
    mlx_array ar = mlx_array_new(), ccv = mlx_array_new(), bidx = mlx_array_new();
    mlx_array bidx2 = mlx_array_new(), cc2 = mlx_array_new(), valid = mlx_array_new();
    mlx_array ninf = mlx_array_new_float32(-INFINITY), scm = mlx_array_new();
    mlx_array part = mlx_array_new(), sel = mlx_array_new();
    const int B_av = (available_blocks > 0) ? available_blocks : 1;
    if (ok) ok = mlx_astype(&qf, query_bhtd, MLX_FLOAT32, stream) == 0;
    if (ok) { const int s2[2] = {H * L, D}; ok = mlx_reshape(&q2, qf, s2, 2, stream) == 0; }
    if (ok && available_blocks > 0) ok = mlx_astype(&kf, compressed_keys, MLX_FLOAT32, stream) == 0;
    if (ok && available_blocks > 0) { const int s2[2] = {available_blocks, D}; ok = mlx_reshape(&kf, kf, s2, 2, stream) == 0; }
    if (ok && available_blocks > 0) { const int pm[2] = {1, 0}; ok = mlx_transpose_axes(&kT, kf, pm, 2, stream) == 0; }
    if (ok && available_blocks > 0) ok = mlx_matmul(&prod, q2, kT, stream) == 0;
    if (ok && available_blocks > 0) { const int s3[3] = {H, L, available_blocks}; ok = mlx_reshape(&prod3, prod, s3, 3, stream) == 0; }
    if (ok && available_blocks > 0) ok = mlx_maximum(&pos, prod3, zf, stream) == 0;
    if (ok && available_blocks > 0) ok = mlx_sum_axis(&ssum, pos, 0, false, stream) == 0;
    if (ok && available_blocks > 0) ok = mlx_divide(&sc, ssum, sdv, stream) == 0;
    /* complete_counts[l] = (cache_offset + l + 1)/compress_ratio ; valid: b < min(cc,B) */
    if (ok) { const int a0[1] = {L}; ok = mlx_arange(&ar, 0, L, 1, MLX_INT32, stream) == 0; (void)a0; }
    if (ok) { mlx_array c0 = mlx_array_new_int(cache_offset + 1); ok = mlx_add(&ccv, ar, c0, stream) == 0; mlx_array_free(c0); }
    if (ok) { mlx_array cr0 = mlx_array_new_int(compress_ratio); ok = mlx_floor_divide(&ccv, ccv, cr0, stream) == 0; mlx_array_free(cr0); }
    if (ok) { const int b0[1] = {available_blocks}; ok = mlx_arange(&bidx, 0, available_blocks, 1, MLX_INT32, stream) == 0; (void)b0; }
    if (ok) { const int e2[2] = {1, available_blocks}; ok = mlx_reshape(&bidx2, bidx, e2, 2, stream) == 0; }
    if (ok) { const int e2[2] = {L, 1}; ok = mlx_reshape(&cc2, ccv, e2, 2, stream) == 0; }
    if (ok && available_blocks > 0) ok = mlx_less(&valid, bidx2, cc2, stream) == 0;
    if (ok && available_blocks > 0) ok = mlx_where(&scm, valid, sc, ninf, stream) == 0;
    if (ok && available_blocks > 0) {
        /* argpartition(scc, kth=-topk) -> gli ultimi topk (insieme = i top-k, ordine non garantito) */
        ok = mlx_argpartition_axis(&part, scm, -block_topk, -1, stream) == 0;
        if (ok) {
            const int* psh = mlx_array_shape(part);
            int s0[2] = {0, psh[1] - block_topk}, s1[2] = {L, psh[1]}, ss[2] = {1, 1};
            ok = mlx_slice(&sel, part, s0, 2, s1, 2, ss, 2, stream) == 0;
        }
        if (ok) ok = mlx_array_eval(sel) == 0;   /* readback SOLO degli indici [L,topk] */
        if (ok) {
            const int32_t* sidx = mlx_array_data_int32(sel);
            for (int l = 0; l < L && ok; l++)
                for (int k = 0; k < block_topk; k++) {
                    int b = (sidx != NULL) ? (int)sidx[l * block_topk + k] : -1;
                    /* stesso filtro della reference/host: solo blocchi < complete_counts */
                    const long cc_l = (long)(cache_offset + l + 1) / compress_ratio;
                    block_idx[l * block_topk + k] = (b >= 0 && b < cc_l && b < available_blocks) ? b : -1;
                }
        }
    }
    mlx_array_free(qf); mlx_array_free(q2); mlx_array_free(kf); mlx_array_free(kT);
    mlx_array_free(prod); mlx_array_free(prod3); mlx_array_free(zf); mlx_array_free(sdv);
    mlx_array_free(pos); mlx_array_free(ssum); mlx_array_free(sc);
    mlx_array_free(ar); mlx_array_free(ccv); mlx_array_free(bidx); mlx_array_free(bidx2);
    mlx_array_free(cc2); mlx_array_free(valid); mlx_array_free(ninf); mlx_array_free(scm);
    mlx_array_free(part); mlx_array_free(sel);
    if (!ok) { free(selected); free(block_idx); return false; }

    for (int l = 0; ok2 && l < L; l++) {
        const int logical_position = cache_offset + l;
        const long complete_counts = (long)(logical_position + 1) / compress_ratio;
        const bool use_selected = (available_blocks > block_topk) && (complete_counts > block_topk);
        if (!use_selected) { for (int k = 0; k < block_topk; k++) block_idx[l * block_topk + k] = k; }

        for (int k = 0; k < block_topk; k++) {
            if (k >= complete_counts) continue; /* block_valid = k < min(complete_counts, block_topk) */
            const int blk = block_idx[l * block_topk + k];
            if (blk < 0) continue;
            for (int r = 0; r < compress_ratio; r++) {
                long tok = (long)blk * compress_ratio + r;
                if (tok < 0) tok = 0; else if (tok >= physical_kv_length) tok = physical_kv_length - 1;
                selected[(size_t)l * physical_kv_length + tok] = true;
            }
        }
        const long tail_start = complete_counts * compress_ratio;
        const long tail_counts = (logical_position + 1) - tail_start;
        for (int r = 0; r < compress_ratio; r++) {
            if (r >= tail_counts) continue;
            long tok = tail_start + r;
            if (tok < 0) tok = 0; else if (tok >= physical_kv_length) tok = physical_kv_length - 1;
            selected[(size_t)l * physical_kv_length + tok] = true;
        }
    }
    free(block_idx);
    if (!ok2) { free(selected); return false; }

    float* mask_data = (float*)malloc((size_t)L * (size_t)physical_kv_length * sizeof(float));
    if (mask_data == NULL) { free(selected); return false; }
    for (size_t i = 0; i < (size_t)L * (size_t)physical_kv_length; i++)
        mask_data[i] = selected[i] ? 0.0f : -INFINITY;
    free(selected);

    const int mshape[4] = {1, 1, L, physical_kv_length};
    mlx_array mask = mlx_array_new_data(mask_data, mshape, 4, MLX_FLOAT32);
    free(mask_data);
    ok = mlx_array_set(out_mask, mask) == 0;
    mlx_array_free(mask);
    return ok;
}

/* QSAAttention -- percorso PRINCIPALE (qwen4_exp.py righe 808-890). Copre ORA sia il
 * caso "sotto budget" (attenzione causale piena via llm_mlx_gqa_sdpa) SIA il caso
 * sparso (maschera esplicita da qsa_select_mask via llm_mlx_gqa_sdpa_prefix-style SDPA
 * con mask, mode ""). L'indexer (qsa_indexer_project + qsa_cache_update +
 * qsa_select_mask) va chiamato dal layer chiamante PRIMA di questa funzione, e il
 * risultato (mask, ctx==NULL se sotto budget) passato in `sparse_mask`. Questa funzione
 * copre inoltre: la
 * doppia proiezione q_proj (query+gate insieme, split e gate sigmoid sull'output),
 * ZeroCenteredRMSNorm su q/k, RoPE parziale a posizioni esplicite, crescita KV-cache
 * (concat, stesso schema del resto del motore), o_proj.
 * x: [1,L,hidden]. k_cache/v_cache: [1,Hkv,prevL,head_dim] (ctx==NULL se has_cache=false).
 * cache_offset: posizione logica assoluta del primo token di questa chiamata.
 * sparse_mask: [1,1,L,Nk] additiva (0/-inf), ctx==NULL = nessuna sparsificazione
 * (causale piena). Nk deve combaciare con k_full (prevL+L) -- responsabilita' del
 * chiamante, che la costruisce con llm_mlx_qwen4exp_qsa_select_mask usando lo stesso
 * physical_kv_length. */
/* Post-proiezione di QSAAttention: prende q_lin/k_lin/v_lin GIA' densi (l'aritmetica di
 * proiezione -- plain o quantizzata -- resta a carico del chiamante, stesso principio
 * gia' applicato a qsa_indexer_postproject) e produce `gated` = output*sigmoid(gate),
 * PRIMA di o_proj (anch'esso a carico del chiamante: puo' essere quantizzato nel
 * wiring reale). Permette a llm_mlx_qwen4exp_qsa_attention_core di restare un wrapper
 * plain-linear per i test standalone, mentre il wiring reale in llmengine_mlx.c
 * (qwen4_full_qsa_block) chiama direttamente questa con proiezioni quantized_linear_cfg. */
static bool llm_mlx_qwen4exp_qsa_attention_postproject(
    mlx_array* out_gated, mlx_array* k_cache, mlx_array* v_cache,
    mlx_array q_lin, mlx_array k_lin, mlx_array v_lin,
    mlx_array q_norm_w, mlx_array k_norm_w, bool has_cache, int cache_offset,
    int num_heads, int num_kv_heads, int head_dim, int rotary_dim, float rope_theta,
    float eps, mlx_array sparse_mask, mlx_stream stream) {
    const int* qlshape = mlx_array_shape(q_lin);
    if (qlshape == NULL || mlx_array_ndim(q_lin) != 3) return false;
    const int B = qlshape[0], L = qlshape[1];
    const float scale = 1.0f / sqrtf((float)head_dim);

    mlx_array projected = mlx_array_new();
    bool ok = true;
    { const int ps[4] = {B, L, num_heads, 2 * head_dim}; ok = mlx_reshape(&projected, q_lin, ps, 4, stream) == 0; }
    mlx_vector_array split_qg = mlx_vector_array_new();
    mlx_array queries0 = mlx_array_new(), gate0 = mlx_array_new();
    if (ok) {
        ok = mlx_split(&split_qg, projected, 2, /*axis=*/-1, stream) == 0;
        if (ok) ok = mlx_vector_array_get(&queries0, split_qg, 0) == 0 &&
                     mlx_vector_array_get(&gate0, split_qg, 1) == 0;
    }
    mlx_array gate = mlx_array_new();
    if (ok) { const int gs[3] = {B, L, num_heads * head_dim}; ok = mlx_reshape(&gate, gate0, gs, 3, stream) == 0; }

    mlx_array keys0 = mlx_array_new(), values0 = mlx_array_new();
    if (ok) {
        const int ks[4] = {B, L, num_kv_heads, head_dim};
        ok = mlx_reshape(&keys0, k_lin, ks, 4, stream) == 0 && mlx_reshape(&values0, v_lin, ks, 4, stream) == 0;
    }

    mlx_array queries_n = mlx_array_new(), keys_n = mlx_array_new();
    if (ok) ok = llm_mlx_zero_centered_rmsnorm(&queries_n, queries0, q_norm_w, /*group=*/0, eps, stream);
    if (ok) ok = llm_mlx_zero_centered_rmsnorm(&keys_n, keys0, k_norm_w, /*group=*/0, eps, stream);

    const int perm[4] = {0, 2, 1, 3};
    mlx_array q_bhtd = mlx_array_new(), k_bhtd = mlx_array_new(), v_bhtd = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&q_bhtd, queries_n, perm, 4, stream) == 0 &&
                 mlx_transpose_axes(&k_bhtd, keys_n, perm, 4, stream) == 0 &&
                 mlx_transpose_axes(&v_bhtd, values0, perm, 4, stream) == 0;

    mlx_array positions = mlx_array_new();
    if (ok) {
        mlx_array ar = mlx_array_new();
        ok = mlx_arange(&ar, (double)cache_offset, (double)(cache_offset + L), 1.0, MLX_INT64, stream) == 0;
        if (ok) { const int prs[1] = {L}; ok = mlx_reshape(&positions, ar, prs, 1, stream) == 0; }
        mlx_array_free(ar);
    }
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_rope(&q_rope, q_bhtd, positions, rotary_dim, rope_theta, stream);
    if (ok) ok = llm_mlx_qwen4exp_rope(&k_rope, k_bhtd, positions, rotary_dim, rope_theta, stream);

    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    if (ok) {
        if (has_cache && k_cache->ctx != NULL) {
            mlx_array kparts[2] = {*k_cache, k_rope};
            mlx_vector_array kv = mlx_vector_array_new_data(kparts, 2);
            ok = mlx_concatenate_axis(&k_full, kv, /*axis=*/2, stream) == 0;
            mlx_vector_array_free(kv);
            if (ok) {
                mlx_array vparts[2] = {*v_cache, v_bhtd};
                mlx_vector_array vv = mlx_vector_array_new_data(vparts, 2);
                ok = mlx_concatenate_axis(&v_full, vv, /*axis=*/2, stream) == 0;
                mlx_vector_array_free(vv);
            }
        } else {
            ok = mlx_array_set(&k_full, k_rope) == 0 && mlx_array_set(&v_full, v_bhtd) == 0;
        }
    }
    if (ok) { ok = mlx_array_set(k_cache, k_full) == 0 && mlx_array_set(v_cache, v_full) == 0; }

    mlx_array attn = mlx_array_new();
    if (ok && sparse_mask.ctx != NULL) {
        mlx_array mask_c = mlx_array_new(), no_sinks = mlx_array_new();
        bool mok = (mlx_array_dtype(sparse_mask) == mlx_array_dtype(q_rope))
            ? mlx_array_set(&mask_c, sparse_mask) == 0
            : mlx_astype(&mask_c, sparse_mask, mlx_array_dtype(q_rope), stream) == 0;
        ok = mok && mlx_fast_scaled_dot_product_attention(&attn, q_rope, k_full, v_full, scale, "", mask_c, no_sinks, stream) == 0;
        mlx_array_free(mask_c); mlx_array_free(no_sinks);
    } else if (ok) {
        ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_full, v_full, scale, /*window=*/0, stream);
    }
    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, stream) == 0;
    if (ok) { const int fs[3] = {B, L, num_heads * head_dim}; ok = mlx_reshape(&attn_flat, attn_bt, fs, 3, stream) == 0; }
    mlx_array gate_sig = mlx_array_new(), gated = mlx_array_new();
    if (ok) ok = mlx_sigmoid(&gate_sig, gate, stream) == 0;
    if (ok) ok = mlx_multiply(&gated, attn_flat, gate_sig, stream) == 0;
    if (ok) ok = mlx_array_set(out_gated, gated) == 0;

    mlx_array_free(projected); mlx_vector_array_free(split_qg);
    mlx_array_free(queries0); mlx_array_free(gate0); mlx_array_free(gate);
    mlx_array_free(keys0); mlx_array_free(values0);
    mlx_array_free(queries_n); mlx_array_free(keys_n);
    mlx_array_free(q_bhtd); mlx_array_free(k_bhtd); mlx_array_free(v_bhtd);
    mlx_array_free(positions); mlx_array_free(q_rope); mlx_array_free(k_rope);
    mlx_array_free(k_full); mlx_array_free(v_full); mlx_array_free(attn);
    mlx_array_free(attn_bt); mlx_array_free(attn_flat); mlx_array_free(gate_sig); mlx_array_free(gated);
    return ok;
}

/* Wrapper plain-linear (proiezioni non quantizzate) per i test standalone e per chiamanti
 * che gia' hanno pesi densi -- STESSA firma/comportamento di prima del refactor. Il
 * wiring reale nel motore (pesi quantizzati) chiama invece direttamente
 * qsa_attention_postproject dopo aver fatto le proiezioni con quantized_linear_cfg
 * (vedi qwen4_full_qsa_block in llmengine_mlx.c). */
static bool llm_mlx_qwen4exp_qsa_attention_core(
    mlx_array* out_h, mlx_array* k_cache, mlx_array* v_cache, mlx_array x,
    mlx_array q_proj_w, mlx_array k_proj_w, mlx_array v_proj_w, mlx_array o_proj_w,
    mlx_array q_norm_w, mlx_array k_norm_w, bool has_cache, int cache_offset,
    int num_heads, int num_kv_heads, int head_dim, int rotary_dim, float rope_theta,
    float eps, mlx_array sparse_mask, mlx_stream stream) {
    mlx_array q_lin = mlx_array_new(), k_lin = mlx_array_new(), v_lin = mlx_array_new();
    bool ok = llm_mlx_qwen4exp_plain_linear(&q_lin, x, q_proj_w, stream);
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&k_lin, x, k_proj_w, stream);
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(&v_lin, x, v_proj_w, stream);
    mlx_array gated = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_qsa_attention_postproject(&gated, k_cache, v_cache, q_lin, k_lin, v_lin,
                                                             q_norm_w, k_norm_w, has_cache, cache_offset,
                                                             num_heads, num_kv_heads, head_dim, rotary_dim,
                                                             rope_theta, eps, sparse_mask, stream);
    if (ok) ok = llm_mlx_qwen4exp_plain_linear(out_h, gated, o_proj_w, stream);
    mlx_array_free(q_lin); mlx_array_free(k_lin); mlx_array_free(v_lin); mlx_array_free(gated);
    return ok;
}

#endif /* LLM_MLX_QWEN4EXP_H */
