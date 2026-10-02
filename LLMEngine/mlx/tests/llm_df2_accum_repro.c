/* llm_df2_accum_repro.c — simula l'ACCUMULO della cache-ctx del drafter DFlash2 (streaming a
 * span + propose) SENZA l'LLM, per riprodurre il crash (encoder Metal NULL) a ctx grande.
 * Come df2_attn_layer: cache = concat(cache, span) ; SDPA NON causale (cache+block). Ripetuto
 * per 5 layer. Uso: accum <n_span> <span_ctx> <block_L>. */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int nspan = argc > 1 ? atoi(argv[1]) : 63;      /* span dello streaming */
    int sc    = argc > 2 ? atoi(argv[2]) : 4096;    /* token per span */
    int L     = argc > 3 ? atoi(argv[3]) : 4096;    /* tail del propose */
    const int nh = 32, nkv = 8, hd = 128, nl = 5;
    mlx_stream st = mlx_default_gpu_stream_new();
    float scale = 1.0f / sqrtf((float)hd);
    /* cache per layer [1,nkv,0,hd] -> cresce via concat (come rt->ctx[].kcache) */
    mlx_array cache = mlx_array_new();
    int c0[4] = {1, nkv, 0, hd};
    if (mlx_zeros(&cache, c0, 4, MLX_BFLOAT16, st) != 0) { printf("FAIL cache init\n"); return 1; }
    for (int i = 0; i < nspan; i++) {
        int sh[4] = {1, nkv, sc, hd};
        mlx_array span = mlx_array_new();
        if (mlx_zeros(&span, sh, 4, MLX_BFLOAT16, st) != 0) { printf("FAIL span alloc @%d\n", i); return 1; }
        mlx_vector_array kv = mlx_vector_array_new();
        mlx_vector_array_append_value(kv, cache);
        mlx_vector_array_append_value(kv, span);
        mlx_array ncache = mlx_array_new();
        if (mlx_concatenate_axis(&ncache, kv, 2, st) != 0) { printf("FAIL concat @%d\n", i); return 1; }
        mlx_vector_array_free(kv);
        mlx_array_set(&cache, ncache);
        mlx_array_free(ncache); mlx_array_free(span);
        if (getenv("DF2_EVAL")) mlx_array_eval(cache);   /* eval per-span (async): collassa la catena concat */
        int S = (i + 1) * sc;
        for (int l = 0; l < nl; l++) {              /* 5 layer: la SDPA su cache+block */
            int qsh[4] = {1, nh, L, hd}, ksh[4] = {1, nkv, S + L, hd};
            mlx_array q = mlx_array_new(), kk = mlx_array_new(), vv = mlx_array_new();
            if (mlx_zeros(&q, qsh, 4, MLX_BFLOAT16, st) != 0 || mlx_zeros(&kk, ksh, 4, MLX_BFLOAT16, st) != 0 ||
                mlx_zeros(&vv, ksh, 4, MLX_BFLOAT16, st) != 0) { printf("FAIL qkv @%d/%d\n", i, l); return 1; }
            mlx_array out = mlx_array_new(), nm = mlx_array_new(), ns = mlx_array_new();
            int rc = mlx_fast_scaled_dot_product_attention(&out, q, kk, vv, scale, "", nm, ns, st);
            if (rc != 0) { printf("[accum] SDPA rc=%d @span=%d layer=%d S=%d -> FAIL\n", rc, i, l, S); return 3; }
            /* LAZY: NON eval per-iterazione (come l'engine). Se accumula -> memoria/risorse.
             * Con DF2_EVAL=1 evita per-iterazione (per confronto). */
            if (getenv("DF2_EVAL")) { int e = mlx_array_eval(out);
                if (e != 0) { printf("[accum] eval rc=%d @span=%d layer=%d S=%d -> FAIL(Metal)\n", e, i, l, S); return 4; } }
            mlx_array_free(q); mlx_array_free(kk); mlx_array_free(vv);
            mlx_array_free(out); mlx_array_free(nm); mlx_array_free(ns);
        }
        if ((i % 10) == 0 || i == nspan - 1) { size_t act = 0; mlx_get_active_memory(&act);
            printf("[accum] span %d/%d S=%d active=%.2fGB%s\n", i + 1, nspan, S, act/1e9, getenv("DF2_EVAL")?" (eval)":""); }
    }
    printf("[accum] completato OK (nspan=%d S=%d)\n", nspan, nspan * sc);
    return 0;
}
