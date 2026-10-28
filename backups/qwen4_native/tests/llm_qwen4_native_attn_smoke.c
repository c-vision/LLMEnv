/* smoke P7g: layer full-attention nativo su un layer reale (slice 3:4). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_gguf.h"
#include "qwen4_native/qwen4_native_attn_layer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) { printf("uso: %s <slice_l3.gguf> <il>\n", argv[0]); return 2; }
    const int il = atoi(argv[2]);
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(argv[1], &g, &map, &maplen)) { printf("FAIL open\n"); return 1; }
    Q4nHp hp; char err[256] = {0};
    if (q4n_hparams(&g, &hp, err, sizeof err) != 0) { printf("FAIL hp: %s\n", err); return 1; }
    Q4nLayerRefs r; memset(&r, 0, sizeof r);
    if (q4n_layer_load(&g, il, false, false, &r, err, sizeof err) != 0) { printf("FAIL layer: %s\n", err); return 1; }
    const float* iq = r.idx_q_proj;
    const float* ik = r.idx_k_proj;
    if (!iq && r.idx_qk) { iq = r.idx_qk; ik = r.idx_qk + (size_t)hp.Hi * hp.Di * hp.E; }
    Q4nAttnW w = {r.attn_q, r.attn_k, r.attn_v, r.attn_q_norm, r.attn_k_norm, r.attn_out,
                  iq, ik, r.idx_q_norm, r.idx_k_norm, hp.H, hp.Hkv, hp.D, hp.Hi, hp.Di};
    float* freq = malloc((size_t)(hp.n_rot / 2) * sizeof(float));
    for (int i = 0; i < hp.n_rot / 2; i++) freq[i] = (float)pow((double)hp.rope_base, -2.0 * i / hp.n_rot);
    const int cap = 8;
    float* kc = calloc((size_t)cap * hp.Hkv * hp.D, sizeof(float));
    float* vc = calloc((size_t)cap * hp.Hkv * hp.D, sizeof(float));
    float* ic = calloc((size_t)cap * hp.Di, sizeof(float));
    float* x = malloc((size_t)hp.E * sizeof(float));
    int bad = 0;
    for (int pos = 0; pos < 3; pos++) {
        for (int d = 0; d < hp.E; d++) x[d] = sinf(0.01f * (float)(pos * hp.E + d));
        int xsh[2] = {1, hp.E};
        mlx_array ax = mlx_array_new_data(x, xsh, 2, MLX_FLOAT32);
        mlx_array ao = mlx_array_new();
        if (qwen4_native_attn_layer(&ao, ax, kc, vc, ic, cap, pos, &w, freq, hp.n_rot, hp.eps,
                                    q4n_default_mv, NULL, s) != 0) { printf("FAIL attn pos %d\n", pos); return 1; }
        mlx_vector_array ev = mlx_vector_array_new();
        mlx_vector_array_append_value(ev, ao);
        mlx_eval(ev); mlx_vector_array_free(ev);
        const float* o = mlx_array_data_float32(ao);
        double mx = 0; int nf = 0;
        for (int d = 0; d < hp.E; d++) { if (!isfinite(o[d])) nf++; if (fabs(o[d]) > mx) mx = fabs((double)o[d]); }
        printf("pos %d absmax=%.4g nonfin=%d\n", pos, mx, nf);
        if (nf || !(mx < 1e6)) bad = 1;
        mlx_array_free(ao); mlx_array_free(ax);
    }
    printf("%s\n", bad ? "P7g ATTN SMOKE: FALLITO" : "P7g ATTN SMOKE: OK");
    return bad;
}
