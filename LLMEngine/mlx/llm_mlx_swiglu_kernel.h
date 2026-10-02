/* llm_mlx_swiglu_kernel.h — kernel Metal fuso per l'attivazione SwiGLU (densi).
 *
 * Gemello di llm_mlx_geglu_kernel.h (gemma4): dense_swiglu_ffn calcola
 * silu(gate)*up = (gate*sigmoid(gate))*up con 3 dispatch MLX separati per layer
 * (mlx_sigmoid, mlx_multiply, mlx_multiply — :6300-6302). Questo modulo li collassa in
 * UN dispatch via mlx_fast_metal_kernel (stesso meccanismo del kernel GeGLU/GDN gia' in
 * produzione): un comando invece di 3, thread-per-elemento, nessun compile/closure
 * (LLM_FUSE_GEGLU via mlx_compile fu PROVATO CONTROPRODUCENTE — 14x iokit trap — vedi
 * commento in llmengine_mlx.c:7625; qui non si ripete quell'errore).
 *
 * Bersaglio L7 (Sintesi §5 B): i modelli DENSi (llama/qwen3/qwen3_5/qwen2 = DeepSeek)
 * usano dense_swiglu_ffn; e' l'unica attivazione fusa che MANCA davvero (GeGLU ha gia'
 * il suo kernel). NON tocca i blocchi MoE (lì l'attivazione e' dentro il gate/up fuse).
 *
 * Contratto: gate, up stesso shape/dtype [*, H] (tipicamente [1, n_tokens, H]),
 * out = silu(gate) * up, stesso dtype. Il kernel usa la STESSA formula bf16 del Sigmoid metal
 * di MLX (`y = 1/(1+exp(|x|)); x<0 ? y : 1-y`) con aritmetica in InT, cosi' e' bit-identico
 * al `swiglu`/`nn.silu` compilati di mlx-lm (verificato model-free su 16M valori, 4 scale).
 * Nessun clamp necessario: exp(|x|) -> +inf da' y=0, sigmoid ben definita per ogni x finito.
 */
#ifndef LLM_MLX_SWIGLU_KERNEL_H
#define LLM_MLX_SWIGLU_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kernel_fallback.h"

static const char llm_swiglu_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
;

/* Un thread per elemento. N (conteggio totale) passato come input scalare int32
 * per il bound-check — il grid e' arrotondato al multiplo del threadgroup, quindi
 * gli ultimi thread di un dispatch possono eccedere N senza il guard. */
static const char llm_swiglu_kernel_src[] =
    "\n"
    "    uint gid = thread_position_in_grid.x;\n"
    "    if (gid >= (uint)N) return;\n"
    /* PARITA' BIT-ESATTA con mlx-lm (`swiglu(gate,x) = nn.silu(gate)*x`, @mx.compile fuso):
     * stessa formula del Sigmoid metal di MLX (`y = 1/(1+exp(|x|)); x<0 ? y : 1-y`, unary_ops.h),
     * tutto in InT. La vecchia `1/(1+exp(-x))` in float scartava >1 ULP bf16 sul ~17% degli
     * elementi (misurato model-free su 16M valori; 0 scarti con questa formula). */
    "    InT x = gate[gid];\n"
    "    InT y = 1 / (1 + metal::exp(metal::abs(x)));\n"
    "    InT s = (x < 0) ? y : 1 - y;\n"
    "    InT sx = x * s;\n"
    "    act[gid] = sx * up[gid];\n"
;

typedef struct {
    mlx_fast_metal_kernel k;
    bool built;
} llm_swiglu_kernel_cache;

static llm_swiglu_kernel_cache llm_swiglu_kernel_singleton = { {0}, false };

static int llm_mlx_swiglu_kernel_build(void) {
    if (llm_swiglu_kernel_singleton.built) return 0;
    const char* in_names[] = {"gate", "up", "N"};
    const char* out_names[] = {"act"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 3; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_swiglu_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llm_swiglu_fused", iv, ov, llm_swiglu_kernel_src, llm_swiglu_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_swiglu_kernel_singleton.built = true;
    return 0;
}

/* gate/up: stesso shape/dtype. out_act: nuovo mlx_array (mlx_array_set), stesso
 * shape/dtype di gate. Ritorna false su qualunque errore (chiamante fa fallback
 * al path eager a 3 op). */
static bool llm_mlx_swiglu_metal(mlx_array gate, mlx_array up, mlx_array* out_act, mlx_stream stream) {
    if (llm_mlx_swiglu_kernel_build() != 0) { LLM_KFALLBACK("swiglu", "build del kernel fallita"); return false; }
    mlx_dtype dt = mlx_array_dtype(gate);
    const int* shp = mlx_array_shape(gate);
    int ndim = mlx_array_ndim(gate);
    if (shp == NULL || ndim <= 0) { LLM_KFALLBACK("swiglu", "shape non valida"); return false; }
    size_t total = mlx_array_size(gate);
    if (total == 0) { LLM_KFALLBACK("swiglu", "totale elementi 0"); return false; }

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
    if (getenv("LLMMETAL_SWIGLU_METAL_VERBOSE")) mlx_fast_metal_kernel_config_set_verbose(cfg, true);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);

    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_swiglu_kernel_singleton.k, inputs, cfg, stream);
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

/* silu(x) = x*sigmoid(x), bit-esatto con nn.silu di mlx-lm (compilato fuso; vedi nota sopra).
 * Sostituisce mlx_sigmoid+mlx_multiply (2 round bf16, ~0,07% di scarti a scala 4). */
static const char llm_silu_kernel_src[] =
    "\n"
    "    uint gid = thread_position_in_grid.x;\n"
    "    if (gid >= (uint)N) return;\n"
    "    InT x = inp[gid];\n"
    "    InT y = 1 / (1 + metal::exp(metal::abs(x)));\n"
    "    InT s = (x < 0) ? y : 1 - y;\n"
    "    outp[gid] = x * s;\n"
;

static llm_swiglu_kernel_cache llm_silu_kernel_singleton = { {0}, false };

static bool llm_mlx_silu_metal(mlx_array x, mlx_array* out, mlx_stream stream) {
    if (!llm_silu_kernel_singleton.built) {
        const char* in_names[] = {"inp", "N"};
        mlx_vector_string iv = mlx_vector_string_new();
        for (int i = 0; i < 2; i++) mlx_vector_string_append_value(iv, in_names[i]);
        mlx_vector_string ov = mlx_vector_string_new();
        mlx_vector_string_append_value(ov, "outp");
        llm_silu_kernel_singleton.k = mlx_fast_metal_kernel_new(
            "llm_silu_fused", iv, ov, llm_silu_kernel_src, llm_swiglu_kernel_header,
            /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
        mlx_vector_string_free(iv);
        mlx_vector_string_free(ov);
        llm_silu_kernel_singleton.built = true;
    }
    mlx_dtype dt = mlx_array_dtype(x);
    if (dt != MLX_BFLOAT16 && dt != MLX_FLOAT16 && dt != MLX_FLOAT32) return false;
    const int* shp = mlx_array_shape(x);
    int ndim = mlx_array_ndim(x);
    size_t total = mlx_array_size(x);
    if (shp == NULL || ndim <= 0 || total == 0) { LLM_KFALLBACK("silu", "shape non valida"); return false; }
    int32_t n32 = (int32_t)total;
    mlx_array narr = mlx_array_new_data(&n32, NULL, 0, MLX_INT32);
    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, x);
    mlx_vector_array_append_value(inputs, narr);
    const int tg = (total < 256) ? (int)total : 256;
    const int grid = ((int)((total + (size_t)tg - 1) / (size_t)tg)) * tg;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, shp, (size_t)ndim, dt);
    mlx_fast_metal_kernel_config_set_grid(cfg, grid, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);
    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_silu_kernel_singleton.k, inputs, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 1;
    if (ok) {
        mlx_array a0 = mlx_array_new();
        ok = mlx_vector_array_get(&a0, outputs, 0) == 0;
        if (ok) mlx_array_set(out, a0);
        mlx_array_free(a0);
    }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(narr);
    return ok;
}

extern bool llm_mlx_fuse_swiglu_metal_enabled(void);

/* silu(x) con parita' bit-esatta a nn.silu: kernel Metal se abilitato (non in training), altrimenti
 * catena eager sigmoid+multiply (tmp: scratch per il sigmoid, liberato dal chiamante). 0 = ok. */
static int llm_mlx_silu_parity(mlx_array* out, mlx_array* tmp, mlx_array x, mlx_stream stream) {
    if (llm_mlx_fuse_swiglu_metal_enabled() && llm_mlx_silu_metal(x, out, stream)) return 0;
    if (mlx_sigmoid(tmp, x, stream) != 0) return -1;
    return mlx_multiply(out, x, *tmp, stream);
}

#endif /* LLM_MLX_SWIGLU_KERNEL_H */
