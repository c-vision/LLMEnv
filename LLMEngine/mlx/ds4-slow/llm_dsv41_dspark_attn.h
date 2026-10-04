/* llm_dsv41_dspark_attn.h — DeepSeek V4.1 DSpark: context cache (ring fisico) + attenzione draft.
 * Port di `dspark.py DSparkAttention` e `mlx_lm_mtp/deepseek_v4_dspark.py DSparkContextCache`.
 * Riusa i primitivi MLX di `llm_dsv41_mlx.h`. Verifica: llm_dsv41_dspark_attn_test.c. */
#ifndef LLM_DSV41_DSPARK_ATTN_H
#define LLM_DSV41_DSPARK_ATTN_H

#include "llm_dsv41_mlx.h"

typedef struct {
    int max_size, offset;
    mlx_array keys;   /* [1,1,cap,D] nel layout fisico */
    bool has;
} LlmDsv41DSparkCache;

/* _chronological: espone il ring dall'oldest al newest (se pieno). */
static inline mlx_array llm_dspark_chrono(mlx_array v, int max_size, int offset,
                                          mlx_stream s, bool* ok) {
    int w = (int)mlx_array_shape(v)[2];
    if (w < max_size || offset <= max_size) return v;
    int cutoff = offset % max_size;
    if (cutoff == 0) return v;
    mlx_array a = llm_dsv41_mlx_slice_axis(v, 2, cutoff, w, s, ok);
    mlx_array b = llm_dsv41_mlx_slice_axis(v, 2, 0, cutoff, s, ok);
    mlx_vector_array ops = mlx_vector_array_new();
    mlx_vector_array_append_value(ops, a);
    mlx_vector_array_append_value(ops, b);
    mlx_array out = mlx_array_new();
    if (mlx_concatenate_axis(&out, ops, 2, s) != 0) *ok = false;
    mlx_vector_array_free(ops);
    return out;
}
/* _physical: colloca le entry cronologiche negli slot assoluti. */
static inline mlx_array llm_dspark_physical(mlx_array v, int max_size, int offset,
                                            mlx_stream s, bool* ok) {
    int w = (int)mlx_array_shape(v)[2];
    if (w < max_size) return v;
    int cutoff = offset % max_size;
    if (cutoff == 0) return v;
    int split = max_size - cutoff;
    mlx_array a = llm_dsv41_mlx_slice_axis(v, 2, split, w, s, ok);
    mlx_array b = llm_dsv41_mlx_slice_axis(v, 2, 0, split, s, ok);
    mlx_vector_array ops = mlx_vector_array_new();
    mlx_vector_array_append_value(ops, a);
    mlx_vector_array_append_value(ops, b);
    mlx_array out = mlx_array_new();
    if (mlx_concatenate_axis(&out, ops, 2, s) != 0) *ok = false;
    mlx_vector_array_free(ops);
    return out;
}

static inline void llm_dspark_cache_append(LlmDsv41DSparkCache* c, mlx_array keys,
                                           mlx_stream s, bool* ok) {
    int L = (int)mlx_array_shape(keys)[2];
    mlx_array next;
    if (!c->has) {
        next = keys;
    } else {
        mlx_array chr = llm_dspark_chrono(c->keys, c->max_size, c->offset, s, ok);
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, chr);
        mlx_vector_array_append_value(ops, keys);
        next = mlx_array_new();
        if (mlx_concatenate_axis(&next, ops, 2, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    int no = c->offset + L;
    int w = (int)mlx_array_shape(next)[2];
    if (w > c->max_size) next = llm_dsv41_mlx_slice_axis(next, 2, w - c->max_size, w, s, ok);
    c->keys = llm_dspark_physical(next, c->max_size, no, s, ok);
    c->offset = no;
    c->has = true;
}

/* DSparkAttention.__call__(x [1,L,dim], cache). */
static inline mlx_array llm_dsv41_dspark_attention(mlx_array x, LlmDsv41DSparkCache* cache,
                                                   const LlmDsv41Config* c,
                                                   mlx_array wq_a, mlx_array q_norm, mlx_array wq_b,
                                                   mlx_array wkv, mlx_array kv_norm,
                                                   mlx_array wo_a, mlx_array wo_b, mlx_array sink,
                                                   mlx_stream s, bool* ok) {
    const int H = c->n_heads, D = c->head_dim, OG = c->o_groups, OL = c->o_lora_rank;
    const int GRP = H * D / OG;
    int L = (int)mlx_array_shape(x)[1];
    mlx_array positions = mlx_array_new();
    if (mlx_arange(&positions, (double)cache->offset, (double)(cache->offset + L), 1.0, MLX_INT32, s) != 0) {
        *ok = false; return mlx_array_new();
    }
    mlx_array q = llm_dsv41_mlx_linear(x, wq_a, s, ok);
    q = llm_dsv41_mlx_rmsnorm(q, q_norm, c->norm_eps, s, ok);
    q = llm_dsv41_mlx_linear(q, wq_b, s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, q, q, (const int[]){ 1, L, H, D }, 4);
    q = llm_dsv41_mlx_rope(q, positions, c, 0, 0, s, ok);

    mlx_array dk = llm_dsv41_mlx_linear(x, wkv, s, ok);
    dk = llm_dsv41_mlx_rmsnorm(dk, kv_norm, c->norm_eps, s, ok);
    dk = llm_dsv41_mlx_rope(dk, positions, c, 0, 0, s, ok);
    dk = llm_dsv41_mlx_fp8_roundtrip(dk, D, s, ok);         /* quantize_activation default */

    mlx_array ctx = llm_dsv41_mlx_slice_axis(cache->keys, 1, 0, 1, s, ok);  /* [1,1,cap,D] */
    LLM_DSV41_MLX_RET(mlx_reshape, ctx, ctx, (const int[]){ 1, -1, D }, 3);
    mlx_array kv = mlx_array_new();
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, ctx);
        mlx_vector_array_append_value(ops, dk);
        if (mlx_concatenate_axis(&kv, ops, 1, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    int N = (int)mlx_array_shape(kv)[1];
    mlx_array scores = mlx_array_new();
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, q);
        mlx_vector_array_append_value(ops, kv);
        if (mlx_einsum(&scores, "bihd,bjd->bhij", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    mlx_array sc = llm_dsv41_mlx_scalar(1.0f / sqrtf((float)D));
    LLM_DSV41_MLX_RET(mlx_multiply, scores, scores, sc);
    mlx_array sk = mlx_array_new(), skb = mlx_array_new(), cat = mlx_array_new(), probs = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, sk, sink, (const int[]){ 1, H, 1, 1 }, 4);
    LLM_DSV41_MLX_RET(mlx_broadcast_to, skb, sk, (const int[]){ 1, H, L, 1 }, 4);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, scores);
        mlx_vector_array_append_value(ops, skb);
        if (mlx_concatenate_axis(&cat, ops, 3, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_softmax_axis, probs, cat, 3, false);
    probs = llm_dsv41_mlx_slice_last(probs, 0, N, s, ok);
    mlx_array out = mlx_array_new();
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, probs);
        mlx_vector_array_append_value(ops, kv);
        if (mlx_einsum(&out, "bhij,bjd->bihd", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    out = llm_dsv41_mlx_rope(out, positions, c, 0, 1, s, ok);
    mlx_array grp = mlx_array_new(), woa = mlx_array_new(), y = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, grp, out, (const int[]){ 1, L, OG, GRP }, 4);
    LLM_DSV41_MLX_RET(mlx_reshape, woa, wo_a, (const int[]){ OG, OL, GRP }, 3);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, grp);
        mlx_vector_array_append_value(ops, woa);
        if (mlx_einsum(&y, "bsgd,grd->bsgr", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, y, y, (const int[]){ 1, L, OG * OL }, 3);
    return llm_dsv41_mlx_linear(y, wo_b, s, ok);
}

#endif /* LLM_DSV41_DSPARK_ATTN_H */
