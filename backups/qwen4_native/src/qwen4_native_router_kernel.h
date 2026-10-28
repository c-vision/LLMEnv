/* qwen4_native_router_kernel.h — P4: router top-k + shared gate nativo
 * (port semplificato di ds4 `kernel_qwen4_router_topk`, 1 thread per token).
 *
 * Semantica (qwen4_ref_moe): softmax sui 512 logits, top-K (10) per probabilita',
 * pesi rinormalizzati su somma dei top-K, shared gate = dot(w_gate, x) (logit grezzo).
 * Le matvec dei router/esperti restano ai gemv del motore (quantizzati).
 * gate_type f32 (il motore dequantizza la riga se serve).
 */
#ifndef LLM_QWEN4_NATIVE_ROUTER_KERNEL_H
#define LLM_QWEN4_NATIVE_ROUTER_KERNEL_H

#include "mlx/c/mlx.h"

static const char llm_q4n_router_header[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n";

static const char llm_q4n_router_topk_src[] =
    "\n"
    "uint tok = uint(thread_position_in_grid.x);\n"
    "uint NE = uint(nexpert), K = uint(nused), X = uint(indim);\n"
    "if (tok >= uint(ntok) || K == 0u || K > 16u) return;\n"
    "ulong lbase = (ulong)tok*NE;\n"
    "float mx = -3.0e38f;\n"
    "for (uint e = 0; e < NE; e++) mx = metal::max(mx, logits[lbase + e]);\n"
    "float sum = 0.f;\n"
    "for (uint e = 0; e < NE; e++) sum += metal::exp(logits[lbase + e] - mx);\n"
    "float inv = 1.f/sum;\n"
    "int sel[16];\n"
    "float wsum = 0.f;\n"
    "for (uint r = 0; r < K; r++) {\n"
    "  float bp = -1.f; int bi = -1;\n"
    "  for (uint e = 0; e < NE; e++) {\n"
    "    bool used = false;\n"
    "    for (uint j = 0; j < r; j++) used = used || (sel[j] == (int)e);\n"
    "    if (used) continue;\n"
    "    float p = metal::exp(logits[lbase + e] - mx)*inv;\n"
    "    if (p > bp) { bp = p; bi = (int)e; }\n"
    "  }\n"
    "  if (bi < 0) return;\n"
    "  sel[r] = bi; wsum += bp;\n"
    "  selected[(ulong)tok*K + r] = bi;\n"
    "  weights[(ulong)tok*K + r] = bp;\n"
    "}\n"
    "if (wsum > 0.f) for (uint r = 0; r < K; r++) weights[(ulong)tok*K + r] /= wsum;\n"
    "float sg = 0.f;\n"
    "if (X > 0u) {\n"
    "  for (uint i = 0; i < X; i++) sg += w_gate[i]*x[(ulong)tok*X + i];\n"
    "}\n"
    "shared_gate[tok] = sg;\n";

static mlx_fast_metal_kernel g_q4n_router_topk;
static int g_q4n_router_topk_built = 0;

static int llm_q4n_router_topk_build(void) {
    if (g_q4n_router_topk_built) return 0;
    const char* in_names[]  = {"logits","x","w_gate","ntok","nexpert","nused","indim"};
    const char* out_names[] = {"selected","weights","shared_gate"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 7; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 3; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_router_topk = mlx_fast_metal_kernel_new(
        "qwen4_router_topk_native", iv, ov, llm_q4n_router_topk_src, llm_q4n_router_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_router_topk_built = (g_q4n_router_topk.ctx != NULL) ? 1 : 0;
    return g_q4n_router_topk_built ? 0 : -1;
}

/* logits:[T][NE] f32, x:[T][X] f32 (puo' essere [T][0] se X=0), w_gate:[X] f32.
 * Out: selected [T][K] i32, weights [T][K] f32, shared_gate [T] f32. */
static int llm_q4n_router_topk(mlx_array* selected, mlx_array* weights, mlx_array* shared_gate,
                               mlx_array logits, mlx_array x, mlx_array w_gate,
                               int K, mlx_stream s) {
    if (!selected || !weights || !shared_gate) return -1;
    if (mlx_array_ndim(logits) != 2 || mlx_array_ndim(x) != 2) return -1;
    const int* ls = mlx_array_shape(logits);
    const int* xs = mlx_array_shape(x);
    const int T = ls[0], NE = ls[1], X = xs[1];
    if (T <= 0 || NE <= 0 || K <= 0 || K > 16 || K > NE || xs[0] != T) return -1;
    if (X > 0) {
        if (mlx_array_ndim(w_gate) != 1) return -1;
        const int* ws = mlx_array_shape(w_gate);
        if (ws[0] != X) return -1;
        if (mlx_array_dtype(w_gate) != MLX_FLOAT32) return -1;
    }
    if (mlx_array_dtype(logits) != MLX_FLOAT32 || mlx_array_dtype(x) != MLX_FLOAT32) return -1;
    if (llm_q4n_router_topk_build() != 0) return -1;
    int ntok = T, nexpert = NE, nused = K, indim = X;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ane = mlx_array_new_data(&nexpert, NULL, 0, MLX_INT32);
    mlx_array ak = mlx_array_new_data(&nused, NULL, 0, MLX_INT32);
    mlx_array ax = mlx_array_new_data(&indim, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, logits);
    mlx_vector_array_append_value(in, x);
    mlx_vector_array_append_value(in, w_gate);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ane);
    mlx_vector_array_append_value(in, ak);
    mlx_vector_array_append_value(in, ax);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int selsh[2] = {T, K}, wsh[2] = {T, K}, gsh[1] = {T};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, selsh, 2, MLX_INT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, wsh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, gsh, 1, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 1, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_router_topk, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 3) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new(), r2 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 &&
            mlx_vector_array_get(&r1, outs, 1) == 0 &&
            mlx_vector_array_get(&r2, outs, 2) == 0) {
            mlx_array_set(selected, r0); mlx_array_set(weights, r1); mlx_array_set(shared_gate, r2);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1); mlx_array_free(r2);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ane); mlx_array_free(ak); mlx_array_free(ax);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_ROUTER_KERNEL_H */
