/* llm_kernelcache_probe.c — verifica se `mlx_fast_metal_kernel` cache-a per NOME ignorando i
 * template args (dtype), la trappola I4 del todo Ishizuki. Model-free, MLX puro.
 *
 * Test: UN kernel a nome fisso con template InT; lo si applica PRIMA con InT=float16, POI con
 * InT=float32 su un valore che DISTINGUE i due (0.0001: in f16 x+1 perde il termine, in f32 no).
 *  - se il secondo apply RICOMPILA (dtype nei template args = parte della chiave) -> y=1.0001;
 *  - se RIUSA il build f16 (chiave solo il nome) -> y=1.0 (o garbage) => trappola CONFERMATA.
 *
 * Build: bash LLMEngine/tools/build_kernelcache_probe.sh
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>

static const char* HDR = "#include <metal_stdlib>\nusing namespace metal;\n";
/* MLX genera `template<typename InT> kernel void <name>(device const InT* x, device InT* y, ...) { <SRC> }`:
 * SRC e' il CORPO (come llm_gdn_gate_src), non un kernel completo. */
static const char* SRC =
    "    uint tid = thread_position_in_grid.x;\n"
    "    y[tid] = x[tid] + InT(1.0);\n";

static int apply1(mlx_fast_metal_kernel k, mlx_dtype dt, float val, mlx_stream s, double* out) {
    int sh[1] = {1};
    mlx_array x, y, xc, yc;
    if (dt == MLX_FLOAT16) { _Float16 v = (_Float16)val; x = mlx_array_new_data(&v, sh, 1, dt); }
    else { float v = val; x = mlx_array_new_data(&v, sh, 1, dt); }
    y = mlx_array_new();
    mlx_vector_array in = mlx_vector_array_new(); mlx_vector_array_append_value(in, x);
    mlx_vector_array outs = mlx_vector_array_new();
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, sh, 1, dt);
    mlx_fast_metal_kernel_config_set_grid(cfg, 1, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    int rc = mlx_fast_metal_kernel_apply(&outs, k, in, cfg, s);
    int ok = (rc == 0 && mlx_vector_array_size(outs) == 1);
    if (ok) { mlx_vector_array_get(&y, outs, 0); xc = mlx_array_new(); yc = mlx_array_new();
        ok = mlx_astype(&yc, y, MLX_FLOAT32, s) == 0 && mlx_array_eval(yc) == 0;
        if (ok) *out = (double)mlx_array_data_float32(yc)[0]; else ok = 0; mlx_array_free(xc); mlx_array_free(yc); }
    mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(in); mlx_vector_array_free(outs);
    mlx_array_free(x); mlx_array_free(y);
    return ok ? 0 : -1;
}

int main(void) {
    mlx_device d = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(d);
    mlx_stream s = mlx_stream_new(); mlx_get_default_stream(&s, d);
    const char* in_names[] = {"x"}; const char* out_names[] = {"y"};
    mlx_vector_string iv = mlx_vector_string_new(); mlx_vector_string_append_value(iv, in_names[0]);
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, out_names[0]);
    /* NOME FISSO, dtype NON nel nome (esattamente il pattern dei nostri kernel) */
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new("probe_add_fixed", iv, ov, SRC, HDR, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    if (k.ctx == NULL) { fprintf(stderr, "build kernel fallita\n"); return 1; }

    double y16 = -1, y32 = -1;
    if (apply1(k, MLX_FLOAT16, 0.0001f, s, &y16)) { fprintf(stderr, "apply f16 fallito\n"); return 1; }
    if (apply1(k, MLX_FLOAT32, 0.0001f, s, &y32)) { fprintf(stderr, "apply f32 fallito\n"); return 1; }
    printf("f16: 0.0001+1 -> %.6f (atteso 1.000000 in f16)\n", y16);
    printf("f32: 0.0001+1 -> %.6f (atteso 1.000100 in f32)\n", y32);
    printf("VERDETTO: %s\n", (y32 > 1.00005) ? "RICOMPILA per dtype (tipo nei template args) -> NO trappola"
                                             : "RIUSA il build f16 -> TRAPPOLA I4 CONFERMATA (dtype non nella chiave)");
    return 0;
}
