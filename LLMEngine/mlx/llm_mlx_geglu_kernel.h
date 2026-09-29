/* llm_mlx_geglu_kernel.h — kernel Metal fuso per l'attivazione GeGLU (gemma4).
 *
 * §2026-09-06: dense_geglu_ffn calcola gelu_approx(gate)*up con 9 dispatch MLX
 * separati per layer. Un primo tentativo di fusione via mlx_compile/mlx_closure_apply
 * (LLM_FUSE_GEGLU) e' risultato CONTROPRODUCENTE, verificato col profiler
 * (Time Profiler / xctrace): iokit_user_client_trap passa da 129ms a 1855ms (14x) --
 * il meccanismo generico di chiusura compilata ha un costo di marshalling per
 * chiamata che supera il risparmio. Questo modulo prova la strada diretta: un vero
 * kernel Metal via mlx_fast_metal_kernel (stesso meccanismo gia' in produzione per
 * llm_mlx_gdn_kernel.h, ~2x sul prefill GDN), che dispatcha UN comando invece di 9,
 * senza passare da compile/closure. Elementwise puro, nessuno stato/riduzione fra
 * thread (a differenza del kernel GDN, ricorrente) -- un thread per elemento.
 *
 * Contratto: gate, up stesso shape/dtype [*, H] (tipicamente [1, n_tokens, H]),
 * out = 0.5*gate*(1+tanh(sqrt(2/pi)*(gate+0.044715*gate^3)))*up, stesso dtype.
 */
#ifndef LLM_MLX_GEGLU_KERNEL_H
#define LLM_MLX_GEGLU_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kernel_fallback.h"

static const char llm_geglu_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
;

/* Un thread per elemento. N (conteggio totale) passato come input scalare int32
 * per il bound-check -- il grid e' arrotondato al multiplo del threadgroup, quindi
 * gli ultimi thread di un dispatch possono eccedere N senza il guard. */
static const char llm_geglu_kernel_src[] =
    "\n"
    "    uint gid = thread_position_in_grid.x;\n"
    "    if (gid >= (uint)N) return;\n"
    "    float x = float(gate[gid]);\n"
    "    float x3 = x * x * x;\n"
    "    float inner = 0.7978845608028654f * (x + 0.044715f * x3);\n"
    /* BUG REALE trovato dal vivo (2026-09-06, gemma-4-31b-it-4bit, layer 0: gate=13.47 ->
     * inner=97.9, per niente un input estremo): il tanh() di Metal calcola internamente
     * via exp(2*inner), che overflowa a +inf per |inner| grande -> inf/inf = NaN. tanhf()
     * su CPU (libc) satura correttamente a +-1 per lo stesso input -- lo stesso identico
     * kernel dava NaN su 2 elementi su 1.37M nel primo layer, che poi si propagavano
     * all'intero tensore dal layer 1 in poi (matmul: un NaN in un input contamina l'intera
     * riga di output). Fix: clampare l'ARGOMENTO di tanh (non x, ne' x^3 -- un tentativo
     * precedente su x non risolveva, il collo era qui) -- tanh(20) e' gia' 1.0 in float32,
     * clampare a +-20 e' innocuo per qualunque input realistico e elimina l'overflow. */
    "    float t = tanh(clamp(inner, -20.0f, 20.0f));\n"
    "    float gelu = 0.5f * x * (1.0f + t);\n"
    "    float u = float(up[gid]);\n"
    "    act[gid] = (InT)(gelu * u);\n"
;

typedef struct {
    mlx_fast_metal_kernel k;
    bool built;
} llm_geglu_kernel_cache;

static llm_geglu_kernel_cache llm_geglu_kernel_singleton = { {0}, false };

static int llm_mlx_geglu_kernel_build(void) {
    if (llm_geglu_kernel_singleton.built) return 0;
    const char* in_names[] = {"gate", "up", "N"};
    const char* out_names[] = {"act"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 3; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_geglu_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llm_geglu_fused", iv, ov, llm_geglu_kernel_src, llm_geglu_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_geglu_kernel_singleton.built = true;
    return 0;
}

/* gate/up: stesso shape/dtype. out_act: nuovo mlx_array (mlx_array_set), stesso
 * shape/dtype di gate. Ritorna false su qualunque errore (chiamante fa fallback
 * al path eager a 9 op). */
static bool llm_mlx_geglu_metal(mlx_array gate, mlx_array up, mlx_array* out_act, mlx_stream stream) {
    if (llm_mlx_geglu_kernel_build() != 0) { LLM_KFALLBACK("geglu", "build del kernel fallita"); return false; }
    mlx_dtype dt = mlx_array_dtype(gate);
    const int* shp = mlx_array_shape(gate);
    int ndim = mlx_array_ndim(gate);
    if (shp == NULL || ndim <= 0) { LLM_KFALLBACK("geglu", "shape non valida"); return false; }
    size_t total = mlx_array_size(gate);
    if (total == 0) { LLM_KFALLBACK("geglu", "totale elementi 0"); return false; }

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
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_geglu_kernel_singleton.k, inputs, cfg, stream);
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

#endif /* LLM_MLX_GEGLU_KERNEL_H */
