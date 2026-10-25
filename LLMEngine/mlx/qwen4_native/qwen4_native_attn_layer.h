/* qwen4_native_attn_layer.h — P7b: wiring di un layer full-attention (QSA).
 * Ordine ds4 `qwen4_ref_attention` + `qwen4_ref_select`:
 *   qg = attn_q@x (2*H*D, q + gate), k = attn_k@x, v = attn_v@x,
 *   qi = indexer_q@x, ik = indexer_k@x;
 *   prep_q (RMS gqn + RoPE, gate = 2a meta'), prep_k (RMS gkn + RoPE) -> cache K,
 *   cache V (grezza), cache idx (grezza);
 *   idx_q (RMS gq + RoPE) -> idx_scr (blocchi ratio) -> idx_sel -> attn (gated)
 *   -> attn_out@o.
 * Cache KV/idx sono host (float*) con cap; `pos` e' la posizione corrente.
 * RoPE testo: pos3 = pos; freq = base^(-2i/n_rot) (n_rot = 64 nel modello).
 */
#ifndef LLM_QWEN4_NATIVE_ATTN_LAYER_H
#define LLM_QWEN4_NATIVE_ATTN_LAYER_H

#include "mlx/c/mlx.h"
#include <string.h>
#include <stdlib.h>
#include "qwen4_native_layer.h"
#include "qwen4_native_attn_kernel.h"
#include "qwen4_native_indexer_kernel.h"

typedef struct {
    const float* attn_q;      /* [2*H*D][E] */
    const float* attn_k;      /* [Hkv*D][E] */
    const float* attn_v;      /* [Hkv*D][E] */
    const float* attn_q_norm; /* [D] */
    const float* attn_k_norm; /* [D] */
    const float* attn_out;    /* [E][H*D] */
    const float* idx_q;       /* [Hi*Di][E] */
    const float* idx_k;       /* [Di][E] */
    const float* idx_q_norm;  /* [Di] */
    const float* idx_k_norm;  /* [Di] */
    int H, Hkv, D, Hi, Di;
} Q4nAttnW;

static int qwen4_native_attn_layer(mlx_array* out, mlx_array x,
                                   float* k_cache, float* v_cache, float* idx_cache,
                                   int cap, int pos,
                                   const Q4nAttnW* w, const float* rope_freq, int n_rot,
                                   float eps, Q4nMvFn mv, void* mvctx, mlx_stream s) {
    if (!out || !w || !mv || !k_cache || !v_cache || !idx_cache || !rope_freq) return -1;
    if (mlx_array_ndim(x) != 2) return -1;
    const int E = mlx_array_shape(x)[1];
    const int H = w->H, Hkv = w->Hkv, D = w->D, Hi = w->Hi, Di = w->Di;
    if (H <= 0 || Hkv <= 0 || D <= 0 || Hi <= 0 || Di <= 0 || H % Hkv != 0) return -1;
    if (pos < 0 || pos >= cap) return -1;
    const int qd = H * D, kvd = Hkv * D, nvis = pos + 1;
    const float* xh = mlx_array_data_float32(x);
    float* qg = malloc((size_t)2 * qd * sizeof(float));
    float* kr = malloc((size_t)kvd * sizeof(float));
    float* vr = malloc((size_t)kvd * sizeof(float));
    float* qi = malloc((size_t)Hi * Di * sizeof(float));
    float* ik = malloc((size_t)Di * sizeof(float));
    float* op = malloc((size_t)E * sizeof(float));
    if (!qg || !kr || !vr || !qi || !ik || !op) { free(qg); free(kr); free(vr); free(qi); free(ik); free(op); return -1; }
    int ok = mv(mvctx, w->attn_q, xh, qg, E, 2 * qd) == 0 &&
             mv(mvctx, w->attn_k, xh, kr, E, kvd) == 0 &&
             mv(mvctx, w->attn_v, xh, vr, E, kvd) == 0 &&
             mv(mvctx, w->idx_q, xh, qi, E, Hi * Di) == 0 &&
             mv(mvctx, w->idx_k, xh, ik, E, Di) == 0;
    if (!ok) { free(qg); free(kr); free(vr); free(qi); free(ik); free(op); return -1; }

    int shqg[2] = {1, 2 * qd}, shk[2] = {1, kvd}, shdi[2] = {1, Hi * Di}, shik[2] = {1, Di};
    int shd[1] = {D}, shdi1[1] = {Di}, shf[1] = {n_rot / 2}, shkv[2] = {nvis, kvd}, shidx[2] = {nvis, Di};
    mlx_array aqg = mlx_array_new_data(qg, shqg, 2, MLX_FLOAT32);
    mlx_array akr = mlx_array_new_data(kr, shk, 2, MLX_FLOAT32);
    mlx_array avr = mlx_array_new_data(vr, shk, 2, MLX_FLOAT32);
    mlx_array aqi = mlx_array_new_data(qi, shdi, 2, MLX_FLOAT32);
    mlx_array aik = mlx_array_new_data(ik, shik, 2, MLX_FLOAT32);
    mlx_array agqn = mlx_array_new_data(w->attn_q_norm, shd, 1, MLX_FLOAT32);
    mlx_array agkn = mlx_array_new_data(w->attn_k_norm, shd, 1, MLX_FLOAT32);
    mlx_array agq = mlx_array_new_data(w->idx_q_norm, shdi1, 1, MLX_FLOAT32);
    mlx_array agk = mlx_array_new_data(w->idx_k_norm, shdi1, 1, MLX_FLOAT32);
    mlx_array af = mlx_array_new_data(rope_freq, shf, 1, MLX_FLOAT32);

    mlx_array q = mlx_array_new(), gate = mlx_array_new(), kp = mlx_array_new();
    mlx_array qip = mlx_array_new();
    if (llm_q4n_prep_q(&q, &gate, aqg, agqn, af, n_rot, pos, 1.0f, eps, s) != 0) goto fail;
    if (llm_q4n_prep_k(&kp, akr, agkn, af, n_rot, pos, 1.0f, eps, s) != 0) goto fail;
    if (llm_q4n_idx_q(&qip, aqi, agq, af, n_rot, pos, 1.0f, eps, s) != 0) goto fail;
    {
        mlx_vector_array ev3 = mlx_vector_array_new();
        mlx_vector_array_append_value(ev3, kp);
        mlx_vector_array_append_value(ev3, qip);
        mlx_eval(ev3); mlx_vector_array_free(ev3);
        memcpy(k_cache + (size_t)pos * kvd, mlx_array_data_float32(kp), (size_t)kvd * sizeof(float));
        memcpy(v_cache + (size_t)pos * kvd, vr, (size_t)kvd * sizeof(float));
        memcpy(idx_cache + (size_t)pos * Di, ik, (size_t)Di * sizeof(float));
    }
    mlx_array aK = mlx_array_new_data(k_cache, shkv, 2, MLX_FLOAT32);
    mlx_array aV = mlx_array_new_data(v_cache, shkv, 2, MLX_FLOAT32);
    mlx_array aIK = mlx_array_new_data(idx_cache, shidx, 2, MLX_FLOAT32);
    const int nb = nvis / 4, kb = 128;   /* ratio 4, top-512 blocchi (k_blocks) */
    mlx_array score = mlx_array_new(), sel = mlx_array_new(), nsel = mlx_array_new();
    int* selh = NULL;
    if (nb <= kb) {
        /* come ds4: sotto soglia si usano tutti i token visibili (niente scoring) */
        selh = malloc((size_t)nvis * sizeof(int));
        if (!selh) goto fail2;
        for (int i = 0; i < nvis; i++) selh[i] = i;
        int selsh[1] = {nvis};
        mlx_array_free(sel);
        sel = mlx_array_new_data(selh, selsh, 1, MLX_INT32);
    } else {
        if (llm_q4n_idx_scr(&score, qip, aIK, agk, af, 4, n_rot, 1.0f, eps, s) != 0) goto fail2;
        if (llm_q4n_idx_sel(&sel, &nsel, score, nvis, 4, kb, s) != 0) goto fail2;
        {
            mlx_vector_array evs = mlx_vector_array_new();
            mlx_vector_array_append_value(evs, sel);
            mlx_vector_array_append_value(evs, nsel);
            mlx_eval(evs); mlx_vector_array_free(evs);
            const int n = mlx_array_data_int32(nsel)[0];
            const int* sv = mlx_array_data_int32(sel);
            if (n <= 0 || n > nvis) goto fail2;
            selh = malloc((size_t)n * sizeof(int));
            if (!selh) goto fail2;
            for (int i = 0; i < n; i++) selh[i] = sv[i];
            int selsh[1] = {n};
            mlx_array_free(sel);
            sel = mlx_array_new_data(selh, selsh, 1, MLX_INT32);
        }
    }
    mlx_array o = mlx_array_new();
    if (llm_q4n_attn(&o, q, gate, aK, aV, sel, H, nvis, s) != 0) goto fail3;
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, o);
    mlx_eval(ev); mlx_vector_array_free(ev);
    {
        const float* oh = mlx_array_data_float32(o);
        ok = mv(mvctx, w->attn_out, oh, op, qd, E) == 0;
    }
    if (!ok) { mlx_array_free(o); goto fail3; }
    {
        int osh[2] = {1, E};
        *out = mlx_array_new_data(op, osh, 2, MLX_FLOAT32);
    }
    mlx_array_free(o); mlx_array_free(score); mlx_array_free(sel); mlx_array_free(nsel);
    free(selh);
    mlx_array_free(aK); mlx_array_free(aV); mlx_array_free(aIK);
    mlx_array_free(q); mlx_array_free(gate); mlx_array_free(kp); mlx_array_free(qip);
    mlx_array_free(aqg); mlx_array_free(akr); mlx_array_free(avr); mlx_array_free(aqi);
    mlx_array_free(aik); mlx_array_free(agqn); mlx_array_free(agkn); mlx_array_free(agq);
    mlx_array_free(agk); mlx_array_free(af);
    free(qg); free(kr); free(vr); free(qi); free(ik); free(op);
    return 0;
fail3:
    mlx_array_free(o);
fail2:
    free(selh);
    mlx_array_free(score); mlx_array_free(sel); mlx_array_free(nsel);
    mlx_array_free(aK); mlx_array_free(aV); mlx_array_free(aIK);
fail:
    mlx_array_free(q); mlx_array_free(gate); mlx_array_free(kp); mlx_array_free(qip);
    mlx_array_free(aqg); mlx_array_free(akr); mlx_array_free(avr); mlx_array_free(aqi);
    mlx_array_free(aik); mlx_array_free(agqn); mlx_array_free(agkn); mlx_array_free(agq);
    mlx_array_free(agk); mlx_array_free(af);
    free(qg); free(kr); free(vr); free(qi); free(ik); free(op);
    return -1;
}

#endif /* LLM_QWEN4_NATIVE_ATTN_LAYER_H */
