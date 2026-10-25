/* qwen4_native_trunk.h — P7d: HC mix + composizione di un layer del trunk.
 * Ordine ds4 `qwen4_ref_layer` (senza PLE, gestito a parte):
 *   hc_mix(R, attn) -> block -> hc_combine
 *   hc_mix(R1, ffn) -> block -> hc_combine
 * I due block sono callback (GDN/attention e MoE); il test usa identita' per
 * verificare solo mix/combine e l'ordine.
 * Convenzione hc_norm: pesi HF zero-centered -> (1+gamma) (vedi P1).
 */
#ifndef LLM_QWEN4_NATIVE_TRUNK_H
#define LLM_QWEN4_NATIVE_TRUNK_H

#include "mlx/c/mlx.h"
#include <stdlib.h>
#include "qwen4_native_hc_kernel.h"
#include "qwen4_native_layer.h"

typedef struct {
    const float* norm;    /* [hc*E] */
    const float* down;    /* [rank][hc*E] */
    const float* up;      /* [hc*E][rank] */
    const float* inject;  /* [hc][hc*E] */
    int rank;
} Q4nHcW;

typedef int (*Q4nBlockFn)(void* ctx, mlx_array x, mlx_array* out, mlx_stream s);

/* R:[1][hc*E] -> *mixed [1][E], *inj [1][hc] (grezza; hc_combine applica 2*sigmoid(x/hc)). */
static int qwen4_native_hc_mix(mlx_array* mixed, mlx_array* inj, mlx_array R,
                               const Q4nHcW* w, int E, int hc, float eps,
                               Q4nMvFn mv, void* mvctx, mlx_stream s) {
    if (!mixed || !inj || !w || !mv || mlx_array_ndim(R) != 2) return -1;
    if (mlx_array_shape(R)[1] != hc * E || hc <= 0 || E <= 0 || w->rank <= 0) return -1;
    int nsh[1] = {hc * E};
    mlx_array aNorm = mlx_array_new_data(w->norm, nsh, 1, MLX_FLOAT32);
    mlx_array xn = mlx_array_new();
    if (llm_q4n_hc_norm(&xn, R, aNorm, hc, E, eps, s) != 0) {
        mlx_array_free(xn); mlx_array_free(aNorm); return -1;
    }
    mlx_array_free(aNorm);
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, xn);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const float* xh = mlx_array_data_float32(xn);
    float* lo = malloc((size_t)w->rank * sizeof(float));
    float* iv = malloc((size_t)hc * sizeof(float));
    if (!lo || !iv) { free(lo); free(iv); mlx_array_free(xn); return -1; }
    int ok = mv(mvctx, w->down, xh, lo, hc * E, w->rank) == 0;
    if (ok) {
        if (w->inject) ok = mv(mvctx, w->inject, xh, iv, hc * E, hc) == 0;
        else memset(iv, 0, (size_t)hc * sizeof(float));
    }
    if (!ok) { free(lo); free(iv); mlx_array_free(xn); return -1; }
    int lsh[2] = {1, w->rank}, ish[2] = {1, hc};
    mlx_array aLo = mlx_array_new_data(lo, lsh, 2, MLX_FLOAT32);
    mlx_array aInj = mlx_array_new_data(iv, ish, 2, MLX_FLOAT32);
    mlx_array aUp = mlx_array_new_data(w->up, (int[]){hc * E, w->rank}, 2, MLX_FLOAT32);
    int rc = llm_q4n_hc_gate_mix(mixed, xn, aLo, aUp, hc, E, s);
    if (rc == 0) {
        /* eval PRIMA di liberare gli input del kernel (graph lazy) */
        mlx_vector_array evm = mlx_vector_array_new();
        mlx_vector_array_append_value(evm, *mixed);
        mlx_eval(evm); mlx_vector_array_free(evm);
        *inj = aInj;
    } else mlx_array_free(aInj);
    mlx_array_free(aLo); mlx_array_free(aUp); mlx_array_free(xn);
    free(lo); free(iv);
    return rc;
}

static int qwen4_native_trunk_layer(mlx_array* R_out, mlx_array R,
                                    const Q4nHcW* attn_hc, const Q4nHcW* ffn_hc,
                                    Q4nBlockFn blk_attn, void* cattn,
                                    Q4nBlockFn blk_ffn, void* cffn,
                                    int E, int hc, float eps,
                                    Q4nMvFn mv, void* mvctx, mlx_stream s) {
    if (!R_out || !attn_hc || !ffn_hc || !blk_attn || !blk_ffn) return -1;
    mlx_array m1 = mlx_array_new(), i1 = mlx_array_new();
    if (qwen4_native_hc_mix(&m1, &i1, R, attn_hc, E, hc, eps, mv, mvctx, s) != 0) goto fail;
    mlx_array b1 = mlx_array_new();
    if (blk_attn(cattn, m1, &b1, s) != 0) goto fail1;
    mlx_array R1 = mlx_array_new();
    if (llm_q4n_hc_combine(&R1, R, b1, i1, s) != 0) goto fail2;
    mlx_array m2 = mlx_array_new(), i2 = mlx_array_new();
    if (qwen4_native_hc_mix(&m2, &i2, R1, ffn_hc, E, hc, eps, mv, mvctx, s) != 0) goto fail3;
    mlx_array b2 = mlx_array_new();
    if (blk_ffn(cffn, m2, &b2, s) != 0) goto fail4;
    if (llm_q4n_hc_combine(R_out, R1, b2, i2, s) != 0) goto fail5;
    mlx_array_free(b2); mlx_array_free(m2); mlx_array_free(i2);
    mlx_array_free(R1); mlx_array_free(b1); mlx_array_free(m1); mlx_array_free(i1);
    return 0;
fail5:
    mlx_array_free(b2);
fail4:
    mlx_array_free(m2); mlx_array_free(i2);
fail3:
    mlx_array_free(R1);
fail2:
    mlx_array_free(b1);
fail1:
    mlx_array_free(m1); mlx_array_free(i1);
fail:
    return -1;
}

#endif /* LLM_QWEN4_NATIVE_TRUNK_H */
