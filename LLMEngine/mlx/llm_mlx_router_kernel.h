/* llm_mlx_router_kernel.h — kernel Metal fuso per il ROUTER MoE (group_expert_select).
 *
 * PERCHE'. In `glm4_moe_ffn_block` il router (gate matmul + group_expert_select) e' il ~19,6%
 * del FFN (misurato con LLM_ROUTER_TIME=1: 0,373 ms/layer su 1,906). Il port eager fa ~10
 * dispatch MLX per layer (sigmoid, add bias, argpartition, slice, contiguous, take_along_axis,
 * sum, divide, multiply, astype). mlx_lm lo fa con `@mx.compile` (1 dispatch).
 *
 * PERCHE' NON `mlx_compile`/closure. Provato e misurato CONTROPRODUCENTE per GeGLU (vedi
 * llm_mlx_geglu_kernel.h: iokit_user_client_trap 129ms -> 1855ms = 14x): il meccanismo di
 * chiusura compilata via mlx-c ha un costo di marshalling per chiamata che supera il risparmio.
 * Quindi si usa la strada diretta: UN kernel Metal (mlx_fast_metal_kernel), un dispatch.
 *
 * CONTRATTO (caso n_group == 1, GLM/Kimi):
 *   in : gates [N, E] f32, escb [E] f32, N/E/K scalari int32
 *   out: top_inds [N, K] uint32, weights [N, K] f32
 *   scores = sigmoid(gates); biased = scores + escb; top-K su biased; pesi da scores (orig),
 *   rinormalizzati (norm_topk_prob). Un threadgroup per token, E thread.
 * NB: l'ORDINE dei K indici puo' differire dall'argpartition eager -> la somma pesata finale
 * puo' differire negli ultimi bit. PARITA' GREEDY OBBLIGATORIA prima di qualunque misura.
 */
#ifndef LLM_MLX_ROUTER_KERNEL_H
#define LLM_MLX_ROUTER_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kernel_fallback.h"

static const char llm_router_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
;

static const char llm_router_kernel_src[] =
    "\n"
    "    uint tgid = threadgroup_position_in_grid.x;\n"   /* token */
    "    uint tid  = thread_position_in_threadgroup.x;\n" /* esperto */
    "    if (tgid >= (uint)N || tid >= (uint)E) return;\n"
    "    threadgroup float sc[256];\n"
    "    threadgroup float orig[256];\n"
    "    float g = float(gates[tgid * (uint)E + tid]);\n"
    "    float o = 1.0f / (1.0f + exp(-g));\n"            /* sigmoid */
    "    orig[tid] = o;\n"
    "    sc[tid] = o + float(escb[tid]);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid == 0) {\n"
    "        uint picked[64];\n"
    "        float sum = 0.0f;\n"
    "        for (uint k = 0; k < (uint)K; k++) {\n"
    "            float best = -3.4e38f; uint bi = 0;\n"
    "            for (uint i = 0; i < (uint)E; i++) {\n"
    "                if (sc[i] > best) { best = sc[i]; bi = i; }\n"
    "            }\n"
    "            sc[bi] = -3.4e38f;\n"
    "            picked[k] = bi;\n"
    "            sum += orig[bi];\n"
    "        }\n"
    "        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;\n"
    "        for (uint k = 0; k < (uint)K; k++) {\n"
    "            top_inds[tgid * (uint)K + k] = picked[k];\n"
    "            weights[tgid * (uint)K + k] = orig[picked[k]] * inv;\n"
    "        }\n"
    "    }\n"
;

typedef struct {
    mlx_fast_metal_kernel k;
    bool built;
} llm_router_kernel_cache;

static llm_router_kernel_cache llm_router_kernel_singleton = { {0}, false };

static int llm_mlx_router_kernel_build(void) {
    if (llm_router_kernel_singleton.built) return 0;
    const char* in_names[] = {"gates", "escb", "N", "E", "K"};
    const char* out_names[] = {"top_inds", "weights"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 5; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    llm_router_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llm_router_fused", iv, ov, llm_router_kernel_src, llm_router_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_router_kernel_singleton.built = true;
    return 0;
}

/* gates [N,E] f32, escb [E] f32, top_k. Ritorna top_inds [N,K] u32 e weights [N,K] f32.
 * false su errore (il chiamante fa fallback al path eager). n_group>1 NON supportato. */
static bool llm_mlx_router_metal(mlx_array gates, mlx_array escb, int top_k,
                                 mlx_array* out_inds, mlx_array* out_w, mlx_stream stream) {
    if (llm_mlx_router_kernel_build() != 0) { LLM_KFALLBACK("router", "build del kernel fallita"); return false; }
    if (mlx_array_ndim(gates) != 2) { LLM_KFALLBACK("router", "gates non 2D"); return false; }
    if (top_k < 1 || top_k > 64) { LLM_KFALLBACK("router", "top_k fuori range"); return false; }
    const int* gs = mlx_array_shape(gates);
    const int N = gs[0], E = gs[1];
    if (E > 256) { LLM_KFALLBACK("router", "E>256 (threadgroup)"); return false; }
    if (mlx_array_dtype(gates) != MLX_FLOAT32) { LLM_KFALLBACK("router", "gates non f32"); return false; }

    int32_t n32 = N, e32 = E, k32 = top_k;
    mlx_array narr = mlx_array_new_data(&n32, NULL, 0, MLX_INT32);
    mlx_array earr = mlx_array_new_data(&e32, NULL, 0, MLX_INT32);
    mlx_array karr = mlx_array_new_data(&k32, NULL, 0, MLX_INT32);

    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, gates);
    mlx_vector_array_append_value(inputs, escb);
    mlx_vector_array_append_value(inputs, narr);
    mlx_vector_array_append_value(inputs, earr);
    mlx_vector_array_append_value(inputs, karr);

    int ishape[2] = {N, top_k};
    int wshape[2] = {N, top_k};

    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ishape, 2, MLX_UINT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, wshape, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)N, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, (size_t)E, 1, 1);

    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_router_kernel_singleton.k, inputs, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 2;
    if (ok) {
        mlx_array a0 = mlx_array_new(), a1 = mlx_array_new();
        ok = mlx_vector_array_get(&a0, outputs, 0) == 0 && mlx_vector_array_get(&a1, outputs, 1) == 0;
        if (ok) { mlx_array_set(out_inds, a0); mlx_array_set(out_w, a1); }
        mlx_array_free(a0); mlx_array_free(a1);
    }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(narr); mlx_array_free(earr); mlx_array_free(karr);
    return ok;
}

#endif /* LLM_MLX_ROUTER_KERNEL_H */
