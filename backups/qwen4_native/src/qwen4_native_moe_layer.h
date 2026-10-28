/* qwen4_native_moe_layer.h — P7c: wiring del blocco MoE (router + esperti + shared).
 * Semantica qwen4_ref_moe: softmax 512 -> top-10 rinormalizzato; per esperto
 * silu(gate)*up -> down, accumulo pesato; shared expert con gate sigmoid.
 * Le matvec sono un callback (gemv GPU nel motore, CPU nel test).
 */
#ifndef LLM_QWEN4_NATIVE_MOE_LAYER_H
#define LLM_QWEN4_NATIVE_MOE_LAYER_H

#include "mlx/c/mlx.h"
#include <math.h>
#include <stdlib.h>
#include "qwen4_native_layer.h"
#include "qwen4_native_router_kernel.h"

typedef struct {
    const float* gate_inp;     /* [NE][E] */
    const float* gate_exps;    /* [NE*F][E] */
    const float* up_exps;      /* [NE*F][E] */
    const float* down_exps;    /* [NE*E][F] */
    const float* shexp_gate;   /* [F][E] */
    const float* shexp_up;     /* [F][E] */
    const float* shexp_down;   /* [E][F] */
    const float* gate_inp_sh;  /* [1][E] */
    int E, NE, K, F;
} Q4nMoeW;

static int qwen4_native_moe_layer(mlx_array* out, mlx_array x,
                                  const Q4nMoeW* w, Q4nMvFn mv, void* mvctx, mlx_stream s) {
    if (!out || !w || !mv || mlx_array_ndim(x) != 2) return -1;
    const int E = w->E, NE = w->NE, K = w->K, F = w->F;
    if (E <= 0 || NE <= 0 || K <= 0 || K > 16 || F <= 0) return -1;
    if (mlx_array_shape(x)[0] != 1) return -1;
    const float* xh = mlx_array_data_float32(x);
    float* logits = malloc((size_t)NE * sizeof(float));
    float* g = malloc((size_t)F * sizeof(float));
    float* u = malloc((size_t)F * sizeof(float));
    float* h = malloc((size_t)F * sizeof(float));
    float* y = malloc((size_t)E * sizeof(float));
    float* op = malloc((size_t)E * sizeof(float));
    if (!logits || !g || !u || !h || !y || !op) { free(logits); free(g); free(u); free(h); free(y); free(op); return -1; }
    if (mv(mvctx, w->gate_inp, xh, logits, E, NE) != 0) goto fail;

    int lsh[2] = {1, NE}, xsh[2] = {1, E}, wsh[1] = {E};
    mlx_array aL = mlx_array_new_data(logits, lsh, 2, MLX_FLOAT32);
    mlx_array aX = mlx_array_new_data(xh, xsh, 2, MLX_FLOAT32);
    mlx_array aW = mlx_array_new_data(w->gate_inp_sh, wsh, 1, MLX_FLOAT32);
    mlx_array aSel = mlx_array_new(), aWt = mlx_array_new(), aSG = mlx_array_new();
    if (llm_q4n_router_topk(&aSel, &aWt, &aSG, aL, aX, aW, K, s) != 0) { mlx_array_free(aL); mlx_array_free(aX); mlx_array_free(aW); goto fail; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aSel);
    mlx_vector_array_append_value(ev, aWt);
    mlx_vector_array_append_value(ev, aSG);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const int* sel = mlx_array_data_int32(aSel);
    const float* wt = mlx_array_data_float32(aWt);
    const float sg = mlx_array_data_float32(aSG)[0];
    for (int d = 0; d < E; d++) op[d] = 0.0f;
    for (int k = 0; k < K; k++) {
        const int e = sel[k];
        if (e < 0 || e >= NE) continue;
        if (mv(mvctx, w->gate_exps + (size_t)e * F * E, xh, g, E, F) != 0 ||
            mv(mvctx, w->up_exps + (size_t)e * F * E, xh, u, E, F) != 0) goto fail_ev;
        for (int f = 0; f < F; f++) {
            const float gv = g[f];
            const float sv = gv >= 0.0f ? 1.0f / (1.0f + expf(-gv)) : expf(gv) / (1.0f + expf(gv));
            h[f] = gv * sv * u[f];
        }
        if (mv(mvctx, w->down_exps + (size_t)e * E * F, h, y, F, E) != 0) goto fail_ev;
        for (int d = 0; d < E; d++) op[d] += wt[k] * y[d];
    }
    /* shared expert */
    if (mv(mvctx, w->shexp_gate, xh, g, E, F) != 0 ||
        mv(mvctx, w->shexp_up, xh, u, E, F) != 0) goto fail_ev;
    for (int f = 0; f < F; f++) {
        const float gv = g[f];
        const float sv = gv >= 0.0f ? 1.0f / (1.0f + expf(-gv)) : expf(gv) / (1.0f + expf(gv));
        h[f] = gv * sv * u[f];
    }
    if (mv(mvctx, w->shexp_down, h, y, F, E) != 0) goto fail_ev;
    {
        const float sv = sg >= 0.0f ? 1.0f / (1.0f + expf(-sg)) : expf(sg) / (1.0f + expf(sg));
        for (int d = 0; d < E; d++) op[d] += sv * y[d];
    }
    {
        int osh[2] = {1, E};
        *out = mlx_array_new_data(op, osh, 2, MLX_FLOAT32);
    }
    mlx_array_free(aSel); mlx_array_free(aWt); mlx_array_free(aSG);
    mlx_array_free(aL); mlx_array_free(aX); mlx_array_free(aW);
    free(logits); free(g); free(u); free(h); free(y); free(op);
    return 0;
fail_ev:
    mlx_array_free(aSel); mlx_array_free(aWt); mlx_array_free(aSG);
    mlx_array_free(aL); mlx_array_free(aX); mlx_array_free(aW);
fail:
    free(logits); free(g); free(u); free(h); free(y); free(op);
    return -1;
}

#endif /* LLM_QWEN4_NATIVE_MOE_LAYER_H */
