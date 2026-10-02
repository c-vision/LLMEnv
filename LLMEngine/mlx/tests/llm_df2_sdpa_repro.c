/* llm_df2_sdpa_repro.c — riproduce la SDPA del drafter DFlash2 a ctx grande SENZA l'LLM.
 * Uso: repro <ctx_S> [window_W]
 *   q [1,32,4096,128], k/v [1,8,S+4096,128] bf16, SDPA NON causale (come df2_attn_layer).
 * Con W>0 tronca le chiavi alla finestra [S+L-W, S+L) e ripete (per validare il fix).
 * Shape dal config del drafter: nh=32 nkv=8 hd=128 L=4096 sliding_window=2048. */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int S = argc > 1 ? atoi(argv[1]) : 252892;
    int W = argc > 2 ? atoi(argv[2]) : 0;
    const int nh = 32, nkv = 8, hd = 128, L = 4096;
    mlx_stream st = mlx_default_gpu_stream_new();
    int qsh[4] = {1, nh, L, hd};
    int K = S + L;
    if (W > 0 && W < K) K = W;   /* fix: solo la finestra recente */
    int ksh[4] = {1, nkv, K, hd};
    printf("[repro] nh=%d nkv=%d hd=%d L=%d S=%d W=%d -> K=%d (kv %.2fGB x2)\n",
           nh, nkv, hd, L, S, W, K, (double)nkv*K*hd*2/1e9);
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (mlx_zeros(&q, qsh, 4, MLX_BFLOAT16, st) != 0) { printf("FAIL q alloc\n"); return 1; }
    if (mlx_zeros(&k, ksh, 4, MLX_BFLOAT16, st) != 0) { printf("FAIL k alloc\n"); return 1; }
    if (mlx_zeros(&v, ksh, 4, MLX_BFLOAT16, st) != 0) { printf("FAIL v alloc\n"); return 1; }
    printf("[repro] alloc ok, eval k/v...\n");
    if (mlx_array_eval(k) != 0 || mlx_array_eval(v) != 0) { printf("FAIL eval k/v (Metal?)\n"); return 2; }
    printf("[repro] kv materializzate, SDPA su %d chiavi...\n", K);
    mlx_array out = mlx_array_new(), nm = mlx_array_new(), ns = mlx_array_new();
    float scale = 1.0f / sqrtf((float)hd);
    int rc = mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "", nm, ns, st);
    printf("[repro] sdpa rc=%d\n", rc);
    if (rc == 0) {
        int e = mlx_array_eval(out);
        printf("[repro] eval out rc=%d -> %s\n", e, e == 0 ? "OK" : "FAIL (Metal)");
    }
    return rc == 0 ? 0 : 3;
}
