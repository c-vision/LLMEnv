/* smoke P7h: MoE quantizzato/f16 su un layer reale (slice 3:4). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_gguf.h"
#include "qwen4_native/qwen4_native_moe_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv) {
    if (argc < 3) { printf("uso: %s <slice.gguf> <il>\n", argv[0]); return 2; }
    const int il = atoi(argv[2]);
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    LlmEngineMoeGguf g; void* map = NULL; size_t len = 0;
    if (!llm_gguf_open_file(argv[1], &g, &map, &len)) { printf("FAIL open\n"); return 1; }
    Q4nHp hp; char err[256] = {0};
    if (q4n_hparams(&g, &hp, err, sizeof err) != 0) { printf("FAIL hp\n"); return 1; }
    Q4nLayerRefs r; memset(&r, 0, sizeof r);
    char lk[64]; snprintf(lk, sizeof lk, "blk.%d.attn_qkv.weight", il);
    int is_linear = llm_gguf_find_tensor(&g, lk) >= 0;
    if (q4n_layer_load(&g, il, is_linear, false, &r, err, sizeof err) != 0) { printf("FAIL layer: %s\n", err); return 1; }
    float* x = malloc((size_t)hp.E * sizeof(float));
    for (int d = 0; d < hp.E; d++) x[d] = sinf(0.01f * d);
    int xsh[2] = {1, hp.E};
    mlx_array ax = mlx_array_new_data(x, xsh, 2, MLX_FLOAT32);
    mlx_array ao = mlx_array_new();
    if (qwen4_native_moe_quant(&ao, ax, &g, il, &r, hp.E, hp.NE, hp.K, hp.F, s) != 0) { printf("FAIL moe\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, ao);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const float* o = mlx_array_data_float32(ao);
    double mx = 0; int nf = 0;
    for (int d = 0; d < hp.E; d++) { if (!isfinite(o[d])) nf++; if (fabs(o[d]) > mx) mx = fabs((double)o[d]); }
    printf("moe absmax=%.4g nonfin=%d\n", mx, nf);
    printf("%s\n", (nf || !(mx < 1e3)) ? "P7h MOE SMOKE: FALLITO" : "P7h MOE SMOKE: OK");
    return (nf || !(mx < 1e3));
}
