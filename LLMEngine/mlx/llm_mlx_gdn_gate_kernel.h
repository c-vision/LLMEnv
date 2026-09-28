/* llm_mlx_gdn_gate_kernel.h — KERNEL METAL FUSO per il "gate" del blocco GDN.
 *
 * Sostituisce la catena elementwise (9 nodi MLX, 8 op di grafo + le letture dei pesi):
 *     beta = sigmoid(b)
 *     a_plus_dt = a + dt_bias
 *     sp = log(1 + exp(a_plus_dt))            (softplus)
 *     g = exp(-exp(A_log) * sp)
 * con UN solo dispatch Metal (`mlx_fast_metal_kernel`, stesso meccanismo del kernel GDN fuso
 * e di llm_mlx_swiglu_metal).
 *
 * PERCHE': il verify speculativo e' OP-COUNT-bound (misurato: ~49 op/layer GDN x 48 layer =
 * ~2352 op per forward; a 0,02-0,05 ms/op = 47-118 ms = il costo del verify). `mlx_compile`
 * NON aiuta (non riduce i dispatch reali, misurato 14x peggio) -> serve il kernel diretto.
 *
 * PARITA' numerica: la catena di riferimento arrotonda a InT a OGNI passo (MLX applica op
 * per op nel dtype dell'attivazione). Il kernel replica lo stesso arrotondamento intermedio,
 * quindi il risultato e' identico bit-per-bit alla catena MLX (a meno dell'ordine di eval).
 *
 * Layout: a,b: [B,S,Hv] (dtype act); A_log_exp, dt_bias: [Hv]; g, beta: [B,S,Hv] (dtype act).
 */
#ifndef LLM_MLX_GDN_GATE_KERNEL_H
#define LLM_MLX_GDN_GATE_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kernel_fallback.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char llm_gdn_gate_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n";

static const char llm_gdn_gate_src[] =
    "\n"
    "    uint i = thread_position_in_grid.x;\n"
    "    if (i >= uint(n)) return;\n"
    "    int hv = int(i % uint(HV));\n"
    "    /* PARITA' BIT-ESATTA con le op MLX: MLX applica ogni unary nel dtype T\n"
    "     * (bfloat16_t) e usa metal::precise::exp/log (vedi unary_ops.h), mentre\n"
    "     * Sigmoid e' `y = 1/(1+exp(|x|)); x<0 ? y : 1-y` con metal::exp (NON precise).\n"
    "     * Qui si replica la stessa aritmetica in T, non in float. */\n"
    "    /* Struttura dei TIPI della catena di riferimento MLX (verificata a runtime):\n"
    "     *   a (bf16) + dt_bias (bf16) -> bf16;  exp/log/add in bf16 (precise);\n"
    "     *   eAlog (f32) * sp_log (bf16) -> f32;  negate/exp in f32.\n"
    "     *   beta = sigmoid(b) in bf16 con la formula esatta di MLX. */\n"
    "    InT x = a[i] + dt[hv];\n"
    "    InT e = metal::precise::exp(x);\n"
    "    InT p = e + InT(1.0f);\n"
    "    InT l = metal::precise::log(p);\n"
    "    float m = alog[hv] * float(l);\n"
    "    g[i] = metal::precise::exp(-m);\n"
    "    InT bv = b[i];\n"
    "    InT ys = InT(1.0f) / (InT(1.0f) + metal::exp(metal::abs(bv)));\n"
    "    beta[i] = (bv < InT(0)) ? ys : InT(1.0f) - ys;\n";

typedef struct {
    mlx_fast_metal_kernel k;
    int built;
    mlx_dtype dt;
    int hv;
} llm_gdn_gate_cache;

static llm_gdn_gate_cache llm_gdn_gate_singleton;

static int llm_mlx_gdn_gate_build(mlx_dtype dt, int hv) {
    if (llm_gdn_gate_singleton.built && llm_gdn_gate_singleton.dt == dt &&
        llm_gdn_gate_singleton.hv == hv)
        return 0;
    if (llm_gdn_gate_singleton.built) {
        mlx_fast_metal_kernel_free(llm_gdn_gate_singleton.k);
        llm_gdn_gate_singleton.built = false;
    }
    const char* in_names[] = {"a", "b", "alog", "dt", "n"};
    const char* out_names[] = {"g", "beta"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 5; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    /* il nome DEVE includere i template arg (cache keyed per nome in mlx_fast_metal_kernel). */
    char name[96];
    snprintf(name, sizeof(name), "llm_gdn_gate_hv%d_dt%d", hv, (int)dt);
    llm_gdn_gate_singleton.k = mlx_fast_metal_kernel_new(
        name, iv, ov, llm_gdn_gate_src, llm_gdn_gate_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_gdn_gate_singleton.dt = dt;
    llm_gdn_gate_singleton.hv = hv;
    llm_gdn_gate_singleton.built = true;
    return 0;
}

/* g_out, beta_out: [B,S,Hv]; a,b: [B,S,Hv]; alog,dt_bias: [Hv]. 0 su successo. */
static int llm_mlx_gdn_gate(mlx_array* g_out, mlx_array* beta_out,
                            mlx_array a, mlx_array b, mlx_array alog, mlx_array dt_bias,
                            mlx_stream stream) {
    if (!g_out || !beta_out) return -1;
    if (mlx_array_ndim(a) != 3 || mlx_array_ndim(b) != 3) return -1;
    const int* as = mlx_array_shape(a);
    const int* bs = mlx_array_shape(b);
    const int B = as[0], S = as[1], Hv = as[2];
    if (bs[0] != B || bs[1] != S || bs[2] != Hv) return -1;
    if (mlx_array_size(alog) != (size_t)Hv || mlx_array_size(dt_bias) != (size_t)Hv) { LLM_KFALLBACK("gdn_gate", "alog/dt_bias size != Hv"); return -1; }
    mlx_dtype dt = mlx_array_dtype(a);
    if (mlx_array_dtype(b) != dt) { LLM_KFALLBACK("gdn_gate", "dtype(b) != dtype(a)"); return -1; }
    long tot = (long)B * S * Hv;
    if (tot <= 0 || tot > 0x7fffffffL) return -1;
    if (llm_mlx_gdn_gate_build(dt, Hv) != 0) { LLM_KFALLBACK("gdn_gate", "build del kernel fallita (dtype/Hv)"); return -1; }
    int n = (int)tot;
    mlx_array narr = mlx_array_new_data(&n, NULL, 0, MLX_INT32);
    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, a);
    mlx_vector_array_append_value(inputs, b);
    mlx_vector_array_append_value(inputs, alog);
    mlx_vector_array_append_value(inputs, dt_bias);
    mlx_vector_array_append_value(inputs, narr);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HV", Hv);
    int oshape[3] = {B, S, Hv};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, oshape, 3, MLX_FLOAT32);   /* g: f32 (promozione a+dt_bias) */
    mlx_fast_metal_kernel_config_add_output_arg(cfg, oshape, 3, dt);            /* beta: dtype attivazione */
    mlx_fast_metal_kernel_config_set_grid(cfg, n, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_gdn_gate_singleton.k, inputs, cfg, stream);
    size_t on = mlx_vector_array_size(outputs);
    if (rc == 0 && on == 2) {
        mlx_array g0 = mlx_array_new(), b0 = mlx_array_new();
        int r0 = mlx_vector_array_get(&g0, outputs, 0);
        int r1 = mlx_vector_array_get(&b0, outputs, 1);
        if (r0 == 0 && r1 == 0) {
            mlx_array_set(g_out, g0);
            mlx_array_set(beta_out, b0);
        } else rc = -1;
        mlx_array_free(g0);
        mlx_array_free(b0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(narr);
    return rc == 0 ? 0 : -1;
}

#endif
