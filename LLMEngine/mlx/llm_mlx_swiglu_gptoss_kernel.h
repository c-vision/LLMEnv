/* llm_mlx_swiglu_gptoss_kernel.h — kernel Metal fuso per l'attivazione SwiGLU di GPT-OSS.
 *
 * Gemello di llm_mlx_swiglu_kernel.h (densi), ma con la variante gpt-oss (mlx_lm
 * models/gpt_oss.py `swiglu`, alpha=1.702, limit=7.0):
 *   clip(gate,None,7) * sigmoid(1.702*clip(gate,None,7)) * (clip(up,-7,7) + 1)
 * La eager in gpt_oss_ffn_block la calcola con 7 op MLX separate (Minimum/Maximum/
 * Clip/Multiply/Sigmoid/Add/Multiply) + ~5 Broadcast di scalari: ~13 dispatch e
 * ~1,5 GB/token di intermedi per layer sul prefill. mlx_lm la compila in UN nodo
 * (`CompiledBroadcast...`). Questo kernel la collassa in UN dispatch, senza
 * mlx_compile/closure (che in questo motore ha dato iokit trap, vedi
 * llm_mlx_geglu_kernel.h).
 *
 * FEDELTA': si arrotonda a InT dopo ogni passo come il path eager (che arrotonda a
 * bf16 a ogni op), incluso l'uso delle costanti nel dtype di attivazione
 * ((InT)1.702f), per avvicinarsi il piu' possibile al bit. La parita' che conta e'
 * comunque il gate greedy IDS, verificato in sessione.
 *
 * Contratto: gate, up stesso shape/dtype [*, H] (tipicamente [T,K,1,H]), out = act
 * stesso shape/dtype. Fallback (false) -> il chiamante usa il path eager.
 */
#ifndef LLM_MLX_SWIGLU_GPTOSS_KERNEL_H
#define LLM_MLX_SWIGLU_GPTOSS_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kernel_fallback.h"

static const char llm_swiglu_gptoss_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
;

/* Un thread per elemento. N (conteggio totale) passato come input scalare int32 per
 * il bound-check (il grid e' arrotondato al threadgroup). */
static const char llm_swiglu_gptoss_kernel_src[] =
    "\n"
    "    uint gid = thread_position_in_grid.x;\n"
    "    if (gid >= (uint)N) return;\n"
    "    InT g = gate[gid];\n"
    "    InT u = up[gid];\n"
    "    InT cg = (InT)min((float)g, 7.0f);\n"                 /* min(gate, 7) */
    "    InT cu = (InT)max(min((float)u, 7.0f), -7.0f);\n"      /* clip(up,-7,7) */
    "    InT alpha = (InT)1.702f;\n"
    "    InT ag = (InT)((float)cg * (float)alpha);\n"
    "    InT s  = (InT)(1.0f / (1.0f + exp(-(float)ag)));\n"    /* sigmoid */
    "    InT sg = (InT)((float)s * (float)cg);\n"
    "    InT u1 = (InT)((float)cu + 1.0f);\n"
    "    act[gid] = (InT)((float)sg * (float)u1);\n"
;

typedef struct {
    mlx_fast_metal_kernel k;
    bool built;
} llm_swiglu_gptoss_kernel_cache;

static llm_swiglu_gptoss_kernel_cache llm_swiglu_gptoss_singleton = { {0}, false };

static int llm_mlx_swiglu_gptoss_kernel_build(void) {
    if (llm_swiglu_gptoss_singleton.built) return 0;
    const char* in_names[] = {"gate", "up", "N"};
    const char* out_names[] = {"act"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 3; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_swiglu_gptoss_singleton.k = mlx_fast_metal_kernel_new(
        "llm_swiglu_gptoss_fused", iv, ov, llm_swiglu_gptoss_kernel_src,
        llm_swiglu_gptoss_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_swiglu_gptoss_singleton.built = true;
    return 0;
}

/* gate/up: stesso shape/dtype. out_act: nuovo mlx_array (mlx_array_set), stesso
 * shape/dtype di gate. Ritorna false su qualunque errore (fallback eager). */
static bool llm_mlx_swiglu_gptoss_metal(mlx_array gate, mlx_array up,
                                        mlx_array* out_act, mlx_stream stream) {
    if (llm_mlx_swiglu_gptoss_kernel_build() != 0) { LLM_KFALLBACK("swiglu_gptoss", "build del kernel fallita"); return false; }
    mlx_dtype dt = mlx_array_dtype(gate);
    const int* shp = mlx_array_shape(gate);
    int ndim = mlx_array_ndim(gate);
    if (shp == NULL || ndim <= 0) { LLM_KFALLBACK("swiglu_gptoss", "shape non valida"); return false; }
    size_t total = mlx_array_size(gate);
    if (total == 0) { LLM_KFALLBACK("swiglu_gptoss", "totale elementi 0"); return false; }

    int32_t n32 = (int32_t)total;
    mlx_array narr = mlx_array_new_data(&n32, NULL, 0, MLX_INT32);

    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, gate);
    mlx_vector_array_append_value(inputs, up);
    mlx_vector_array_append_value(inputs, narr);

    const int tg = (total < 256) ? (int)total : 256;
    const int grid = ((int)((total + (size_t)tg - 1) / (size_t)tg)) * tg;

    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, shp, (size_t)ndim, dt);
    mlx_fast_metal_kernel_config_set_grid(cfg, grid, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);

    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_swiglu_gptoss_singleton.k, inputs, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        ok = mlx_vector_array_get(&a0, outputs, 0) == 0;
        if (ok) mlx_array_set(out_act, a0);
        mlx_array_free(a0);
    }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(narr);
    return ok;
}

#endif /* LLM_MLX_SWIGLU_GPTOSS_KERNEL_H */
