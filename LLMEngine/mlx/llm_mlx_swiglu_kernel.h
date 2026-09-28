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
 * out = silu(gate) * up, stesso dtype. Il kernel replica i TRE arrotondamenti a InT del
 * path eager (sigmoid->InT, gate*sigmoid->InT, *up->InT) per essere BIT-FEDELE: una
 * variante con UN solo round finale (accumulo f32) era scartata >1 ULP e flip-ava l'argmax
 * su posizioni con logit in quasi-parita' — DIVERGENZA reale su testo vero (verificata
 * 2026-09-10: eager-vs-eager e kernel-vs-kernel deterministici, eager-vs-kernel divergeva).
 * Sigmoid non richiede clamp dell'argomento (a differenza del tanh di GeGLU): exp(-x)
 * overflowa a +inf solo per x molto NEGATIVO, e 1/(1+inf)=0 -> silu->0, nessun NaN; per x
 * molto positivo exp(-x)->0 -> sigmoid->1. Il percorso e' numericamente chiuso per input finito.
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
    /* Bit-fedele al path eager a 3 op: eager arrotonda a InT DOPO ogni op (sigmoid->InT,
     * gate*sigmoid->InT, *up->InT). Un accumulo f32 con UN solo round finale SCARTA >1 ULP
     * e flip-a l'argmax su posizioni con logit in quasi-parita' (divergenza reale su testo
     * vero, verificata 2026-09-10). Replichiamo i TRE round intermedi: (InT) dopo ogni step. */
    "    InT x = gate[gid];\n"
    "    InT s = (InT)(1.0f / (1.0f + exp(-float(x))));\n"   /* sigmoid, round a InT */
    "    InT sx = (InT)(float(x) * float(s));\n"             /* gate*sigmoid, round a InT */
    "    InT u = up[gid];\n"
    "    act[gid] = (InT)(float(sx) * float(u));\n"          /* *up, round a InT */
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

#endif /* LLM_MLX_SWIGLU_KERNEL_H */
