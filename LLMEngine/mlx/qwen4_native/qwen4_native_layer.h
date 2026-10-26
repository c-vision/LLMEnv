/* qwen4_native_layer.h — P7 (primo passo): wiring di un layer linear/GDN.
 * Ordine ds4 `qwen4_ref_linear`:
 *   qkv/z/b/a = matvec(x) -> conv depthwise K=4 con silu -> L2norm q/k (+1/sqrt(D))
 *   -> gdn_prep (g,beta) -> gdn_scan (stato [Hv][D][D]) -> gdn_out (RMS * sigmoid(z))
 *   -> matvec lin_out.
 * Le matvec sono un callback (il motore usera' i gemv GPU; il test una versione CPU).
 * Stato (history conv e stato ricorrente) e' [in/out] tra le chiamate.
 */
#ifndef LLM_QWEN4_NATIVE_LAYER_H
#define LLM_QWEN4_NATIVE_LAYER_H

#include "mlx/c/mlx.h"
#include "qwen4_native_gdn_kernel.h"
#include "qwen4_native_gpu.h"

/* W e' [n_out][n_in] f32 row-major; y = W*x. 0 = ok. */
typedef int (*Q4nMvFn)(void* ctx, const float* W, const float* x, float* y, int n_in, int n_out);

typedef struct {
    Q4nMat lin_qkv, lin_gate, lin_beta, lin_alpha, lin_out;
    const float* lin_conv;
    const float* lin_a;
    const float* lin_dt;
    const float* lin_norm;
    int conv_dim, v_dim, Hk, Hv, D;
} Q4nGdnW;

static int q4n_default_mv(void* ctx, const float* W, const float* x, float* y, int n_in, int n_out) {
    (void)ctx;
    if (!W || !x || !y || n_in <= 0 || n_out <= 0) return -1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, W, n_in, x, 1, 0.0f, y, 1);
    return 0;
}

/* x:[T][E] -> *out [T][E]. conv_hist:[K-1][conv_dim], lin_state:[Hv][D][D] (in/out). */
static int qwen4_native_gdn_layer(mlx_array* out, mlx_array x,
                                  mlx_array* conv_hist, mlx_array* lin_state,
                                  const Q4nGdnW* w, Q4nMatMvFn mv, void* mvctx, mlx_stream s) {
    if (!out || !w || !mv || !conv_hist || !lin_state) return -1;
    if (mlx_array_ndim(x) != 2) return -1;
    const int T = mlx_array_shape(x)[0], E = mlx_array_shape(x)[1];
    const int cd = w->conv_dim, vd = w->v_dim, Hk = w->Hk, Hv = w->Hv, D = w->D;
    if (cd != 2 * Hk * D + vd || vd != Hv * D || cd <= 0 || E <= 0) return -1;
    if (mlx_array_ndim(*conv_hist) != 2 || mlx_array_shape(*conv_hist)[0] != 3 ||
        mlx_array_shape(*conv_hist)[1] != cd) return -1;
    if (mlx_array_ndim(*lin_state) != 3 || mlx_array_shape(*lin_state)[0] != Hv ||
        mlx_array_shape(*lin_state)[1] != D || mlx_array_shape(*lin_state)[2] != D) return -1;

    const float* xh = mlx_array_data_float32(x);
    float* qkv = malloc((size_t)T * cd * sizeof(float));
    float* z = malloc((size_t)T * vd * sizeof(float));
    float* a = malloc((size_t)T * Hv * sizeof(float));
    float* b = malloc((size_t)T * Hv * sizeof(float));
    float* op = malloc((size_t)T * E * sizeof(float));
    if (!qkv || !z || !a || !b || !op) { free(qkv); free(z); free(a); free(b); free(op); return -1; }
    for (int t = 0; t < T; t++) {
        const float* xr = xh + (size_t)t * E;
        if (mv(mvctx, &w->lin_qkv, xr, qkv + (size_t)t * cd, E, cd) != 0 ||
            mv(mvctx, &w->lin_gate, xr, z + (size_t)t * vd, E, vd) != 0 ||
            mv(mvctx, &w->lin_beta, xr, b + (size_t)t * Hv, E, Hv) != 0 ||
            mv(mvctx, &w->lin_alpha, xr, a + (size_t)t * Hv, E, Hv) != 0) {
            free(qkv); free(z); free(a); free(b); free(op); return -1;
        }
    }
    int shq[2] = {T, cd}, shz[2] = {T, vd}, shv[2] = {T, Hv};
    int shw[2] = {cd, 4}, shs[1] = {Hv}, shd[1] = {D};
    mlx_array aqkv = mlx_array_new_data(qkv, shq, 2, MLX_FLOAT32);
    mlx_array az = mlx_array_new_data(z, shz, 2, MLX_FLOAT32);
    mlx_array aa = mlx_array_new_data(a, shv, 2, MLX_FLOAT32);
    mlx_array ab = mlx_array_new_data(b, shv, 2, MLX_FLOAT32);
    mlx_array aw = mlx_array_new_data(w->lin_conv, shw, 2, MLX_FLOAT32);
    mlx_array asa = mlx_array_new_data(w->lin_a, shs, 1, MLX_FLOAT32);
    mlx_array adt = mlx_array_new_data(w->lin_dt, shs, 1, MLX_FLOAT32);
    mlx_array anw = mlx_array_new_data(w->lin_norm, shd, 1, MLX_FLOAT32);

    mlx_array conv = mlx_array_new(), hist = mlx_array_new();
    if (llm_q4n_conv_stream(&conv, &hist, aqkv, *conv_hist, aw, 4, 1, s) != 0) goto fail0;
    mlx_array qp = mlx_array_new(), ga = mlx_array_new(), gb = mlx_array_new();
    if (llm_q4n_gdn_prep(&qp, &ga, &gb, conv, aa, ab, asa, adt, Hk, Hv, D, s) != 0) goto fail1;
    mlx_array st2 = mlx_array_new(), o = mlx_array_new();
    if (llm_q4n_gdn_scan(&st2, &o, qp, ga, gb, *lin_state, Hk, Hv, D, s) != 0) goto fail2;
    mlx_array o2 = mlx_array_new();
    if (llm_q4n_gdn_out(&o2, o, az, anw, Hv, D, 1e-6f, s) != 0) goto fail3;
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, o2);
    mlx_vector_array_append_value(ev, hist);
    mlx_vector_array_append_value(ev, st2);
    mlx_eval(ev);
    mlx_vector_array_free(ev);
    {
        const float* oh = mlx_array_data_float32(o2);
        for (int t = 0; t < T; t++)
            if (mv(mvctx, &w->lin_out, oh + (size_t)t * vd, op + (size_t)t * E, vd, E) != 0) {
                goto fail4;
            }
    }
    mlx_array_set(conv_hist, hist);
    mlx_array_set(lin_state, st2);
    {
        int osh[2] = {T, E};
        *out = mlx_array_new_data(op, osh, 2, MLX_FLOAT32);
    }
    mlx_array_free(o2); mlx_array_free(st2); mlx_array_free(o); mlx_array_free(qp);
    mlx_array_free(ga); mlx_array_free(gb); mlx_array_free(conv); mlx_array_free(hist);
    mlx_array_free(aqkv); mlx_array_free(az); mlx_array_free(aa); mlx_array_free(ab);
    mlx_array_free(aw); mlx_array_free(asa); mlx_array_free(adt); mlx_array_free(anw);
    free(qkv); free(z); free(a); free(b); free(op);
    return 0;
fail4:
    mlx_array_free(o2);
fail3:
    mlx_array_free(st2); mlx_array_free(o);
fail2:
    mlx_array_free(qp); mlx_array_free(ga); mlx_array_free(gb);
fail1:
    mlx_array_free(conv); mlx_array_free(hist);
fail0:
    mlx_array_free(aqkv); mlx_array_free(az); mlx_array_free(aa); mlx_array_free(ab);
    mlx_array_free(aw); mlx_array_free(asa); mlx_array_free(adt); mlx_array_free(anw);
    free(qkv); free(z); free(a); free(b); free(op);
    return -1;
}

#endif /* LLM_QWEN4_NATIVE_LAYER_H */
