/* llm_dsv41_vision.h — DeepSeek V4.1: ViT + Aligner (vision.py), header-only, API MLX C.
 * Attention full su un'immagine, 2D RoPE (half-split, non interleaved), patch embed, GeLU aligner
 * con downsample. Port fedele di `vision.py`; verifica vs `llm_dsv41_vision_ref.h`. */
#ifndef LLM_DSV41_VISION_H
#define LLM_DSV41_VISION_H

#include <math.h>
#include <stdlib.h>

#include <mlx/c/mlx.h>

#include "llm_dsv41_tensors.h"

#define LLM_VIS_NORM_EPS 1e-6f
#define LLM_VIS_MAX_BLOCKS 64   /* ViT reale V4.1: 32 (vision_config.num_hidden_layers) */

#define LLM_VIS_RET(_op, _res, ...) \
    do { if ((_op)(&(_res), __VA_ARGS__, s) != 0) { *ok = false; return mlx_array_new(); } } while (0)

typedef struct {
    mlx_array n1, wqkv, wqkvb, wo, wob, n2, w1, w2;
} LlmDsv41VisionBlock;

typedef struct {
    mlx_array patch_w, patch_b;
    LlmDsv41VisionBlock blk[LLM_VIS_MAX_BLOCKS];
    int n_blocks;
    mlx_array final_norm;
} LlmDsv41VisionWeights;

typedef struct {
    mlx_array w1, w1b, w2, w2b;
    int ratio;
} LlmDsv41AlignerWeights;

/* slice sull'ultimo asse */
static inline mlx_array llm_vis_slice(mlx_array a, int lo, int hi, mlx_stream s, bool* ok) {
    int nd = (int)mlx_array_ndim(a);
    const int* sh = mlx_array_shape(a);
    int* st = (int*)calloc((size_t)nd, sizeof(int));
    int* sp = (int*)calloc((size_t)nd, sizeof(int));
    int* sd = (int*)malloc((size_t)nd * sizeof(int));
    if (!st || !sp || !sd) { free(st); free(sp); free(sd); *ok = false; return mlx_array_new(); }
    for (int i = 0; i < nd; i++) { sp[i] = sh[i]; sd[i] = 1; }
    st[nd - 1] = lo; sp[nd - 1] = hi;
    mlx_array out = mlx_array_new();
    if (mlx_slice(&out, a, st, (size_t)nd, sp, (size_t)nd, sd, (size_t)nd, s) != 0) *ok = false;
    free(st); free(sp); free(sd);
    return out;
}

static inline mlx_array llm_vis_linear(mlx_array x, mlx_array w, mlx_array b, mlx_stream s, bool* ok) {
    mlx_array wt = mlx_array_new(), y = mlx_array_new(), y2 = mlx_array_new();
    LLM_VIS_RET(mlx_transpose, wt, w);
    LLM_VIS_RET(mlx_matmul, y, x, wt);
    LLM_VIS_RET(mlx_add, y2, y, b);
    return y2;
}

static inline mlx_array llm_vis_norm(mlx_array x, mlx_array w, mlx_stream s, bool* ok) {
    mlx_array xf = mlx_array_new(), sq = mlx_array_new(), mn = mlx_array_new();
    mlx_array rs = mlx_array_new(), y = mlx_array_new();
    mlx_array e = mlx_array_new_float(LLM_VIS_NORM_EPS);
    LLM_VIS_RET(mlx_astype, xf, x, MLX_FLOAT32);
    LLM_VIS_RET(mlx_square, sq, xf);
    LLM_VIS_RET(mlx_mean_axis, mn, sq, (int)mlx_array_ndim(x) - 1, true);
    LLM_VIS_RET(mlx_add, mn, mn, e);
    LLM_VIS_RET(mlx_rsqrt, rs, mn);
    LLM_VIS_RET(mlx_multiply, y, xf, rs);
    LLM_VIS_RET(mlx_multiply, y, y, w);
    return y;
}

/* 2D RoPE half-split: x [N,H,D]; cos/sin [N,rope_dim] con rope_dim=D/2. */
static inline mlx_array llm_vis_rotary(mlx_array x, mlx_array cosv, mlx_array sinv, mlx_stream s, bool* ok) {
    int D = (int)mlx_array_shape(x)[2];
    mlx_array x1 = llm_vis_slice(x, 0, D / 2, s, ok);
    mlx_array x2 = llm_vis_slice(x, D / 2, D, s, ok);
    mlx_array ce = mlx_array_new(), se = mlx_array_new();
    LLM_VIS_RET(mlx_expand_dims, ce, cosv, 1);
    LLM_VIS_RET(mlx_expand_dims, se, sinv, 1);
    mlx_array a = mlx_array_new(), b = mlx_array_new(), c = mlx_array_new(), d = mlx_array_new();
    LLM_VIS_RET(mlx_multiply, a, x1, ce);
    LLM_VIS_RET(mlx_multiply, b, x2, se);
    LLM_VIS_RET(mlx_subtract, a, a, b);
    LLM_VIS_RET(mlx_multiply, c, x2, ce);
    LLM_VIS_RET(mlx_multiply, d, x1, se);
    LLM_VIS_RET(mlx_add, c, c, d);
    mlx_vector_array v = mlx_vector_array_new();
    mlx_vector_array_append_value(v, a);
    mlx_vector_array_append_value(v, c);
    mlx_array out = mlx_array_new();
    LLM_VIS_RET(mlx_concatenate_axis, out, v, 2);
    mlx_vector_array_free(v);
    return out;
}

/* get_vision_cos_sin: cos/sin [N, rope_dim], N=n_h*n_w. */
static inline void llm_vis_cos_sin(int n_h, int n_w, int rope_dim, float theta,
                                   mlx_array* cos_out, mlx_array* sin_out, mlx_stream s) {
    int N = n_h * n_w, hd2 = rope_dim / 2;
    float* freqs = (float*)malloc((size_t)N * (size_t)rope_dim * sizeof(float));
    for (int i = 0; i < N; i++) {
        int h = i / n_w, w = i % n_w;
        for (int j = 0; j < hd2; j++) {
            float inv = 1.0f / powf(theta, (2.0f * j) / (float)rope_dim);
            freqs[i * rope_dim + j] = (float)h * inv;
            freqs[i * rope_dim + hd2 + j] = (float)w * inv;
        }
    }
    mlx_array fr = llm_dsv41_array_f32(freqs, 2, (const int[]){ N, rope_dim });
    free(freqs);
    mlx_array c = mlx_array_new(), sn = mlx_array_new();
    mlx_cos(&c, fr, s);
    mlx_sin(&sn, fr, s);
    *cos_out = c; *sin_out = sn;
}

static inline mlx_array llm_vis_attention(mlx_array x, mlx_array wqkv, mlx_array bqkv,
                                          mlx_array wo, mlx_array wob, mlx_array cosv, mlx_array sinv,
                                          int heads, int hd, float scale, mlx_stream s, bool* ok) {
    int N = (int)mlx_array_shape(x)[0], D = heads * hd;
    mlx_array qkv = llm_vis_linear(x, wqkv, bqkv, s, ok);       /* [N,3D] */
    mlx_array q = llm_vis_slice(qkv, 0, D, s, ok);
    mlx_array k = llm_vis_slice(qkv, D, 2 * D, s, ok);
    mlx_array v = llm_vis_slice(qkv, 2 * D, 3 * D, s, ok);
    LLM_VIS_RET(mlx_reshape, q, q, (const int[]){ N, heads, hd }, 3);
    LLM_VIS_RET(mlx_reshape, k, k, (const int[]){ N, heads, hd }, 3);
    LLM_VIS_RET(mlx_reshape, v, v, (const int[]){ N, heads, hd }, 3);
    q = llm_vis_rotary(q, cosv, sinv, s, ok);
    k = llm_vis_rotary(k, cosv, sinv, s, ok);
    mlx_array qt = mlx_array_new(), kt = mlx_array_new(), vt = mlx_array_new(), ktt = mlx_array_new();
    LLM_VIS_RET(mlx_transpose_axes, qt, q, (const int[]){ 1, 0, 2 }, 3);
    LLM_VIS_RET(mlx_transpose_axes, kt, k, (const int[]){ 1, 0, 2 }, 3);
    LLM_VIS_RET(mlx_transpose_axes, vt, v, (const int[]){ 1, 0, 2 }, 3);
    LLM_VIS_RET(mlx_transpose_axes, ktt, kt, (const int[]){ 0, 2, 1 }, 3);
    mlx_array scores = mlx_array_new(), ss = mlx_array_new_float(scale);
    LLM_VIS_RET(mlx_matmul, scores, qt, ktt);
    LLM_VIS_RET(mlx_multiply, scores, scores, ss);
    LLM_VIS_RET(mlx_softmax_axis, scores, scores, 2, false);
    mlx_array out = mlx_array_new();
    LLM_VIS_RET(mlx_matmul, out, scores, vt);                   /* [H,N,hd] */
    mlx_array ot = mlx_array_new(), of = mlx_array_new();
    LLM_VIS_RET(mlx_transpose_axes, ot, out, (const int[]){ 1, 0, 2 }, 3);
    LLM_VIS_RET(mlx_reshape, of, ot, (const int[]){ N, D }, 2);
    return llm_vis_linear(of, wo, wob, s, ok);
}

static inline mlx_array llm_vis_mlp(mlx_array x, mlx_array w1, mlx_array w2, mlx_stream s, bool* ok) {
    mlx_array wx = mlx_array_new();
    {   /* w1 senza bias */
        mlx_array wt = mlx_array_new();
        LLM_VIS_RET(mlx_transpose, wt, w1);
        LLM_VIS_RET(mlx_matmul, wx, x, wt);
    }
    int d = (int)mlx_array_shape(wx)[1];
    mlx_array gate = llm_vis_slice(wx, 0, d / 2, s, ok);
    mlx_array up = llm_vis_slice(wx, d / 2, d, s, ok);
    mlx_array sig = mlx_array_new(), y = mlx_array_new();
    LLM_VIS_RET(mlx_sigmoid, sig, gate);
    LLM_VIS_RET(mlx_multiply, y, gate, sig);
    LLM_VIS_RET(mlx_multiply, y, y, up);
    mlx_array wt2 = mlx_array_new(), out = mlx_array_new();
    LLM_VIS_RET(mlx_transpose, wt2, w2);
    LLM_VIS_RET(mlx_matmul, out, y, wt2);
    return out;
}

/* ViT forward: patches [N, 3*patch^2] -> [N, vision_dim]. */
static inline mlx_array llm_dsv41_vision_vit(const LlmDsv41VisionWeights* w, mlx_array patches,
                                             int n_h, int n_w, int dim, int heads, float theta,
                                             mlx_stream s, bool* ok) {
    mlx_array x = llm_vis_linear(patches, w->patch_w, w->patch_b, s, ok);
    int hd = dim / heads;
    mlx_array cosv = mlx_array_new(), sinv = mlx_array_new();
    llm_vis_cos_sin(n_h, n_w, hd / 2, theta, &cosv, &sinv, s);
    float scale = 1.0f / sqrtf((float)hd);
    for (int b = 0; b < w->n_blocks; b++) {
        const LlmDsv41VisionBlock* B = &w->blk[b];
        mlx_array xn = llm_vis_norm(x, B->n1, s, ok);
        mlx_array a = llm_vis_attention(xn, B->wqkv, B->wqkvb, B->wo, B->wob, cosv, sinv, heads, hd, scale, s, ok);
        mlx_array x1 = mlx_array_new();
        LLM_VIS_RET(mlx_add, x1, x, a);
        x = x1;
        mlx_array xn2 = llm_vis_norm(x, B->n2, s, ok);
        mlx_array f = llm_vis_mlp(xn2, B->w1, B->w2, s, ok);
        mlx_array x2 = mlx_array_new();
        LLM_VIS_RET(mlx_add, x2, x, f);
        x = x2;
    }
    return llm_vis_norm(x, w->final_norm, s, ok);
}

/* Aligner forward: x [N, vision_dim] -> [out_h*out_w, dim]. */
static inline mlx_array llm_dsv41_vision_aligner(const LlmDsv41AlignerWeights* a, mlx_array x,
                                                 int n_h, int n_w, int vis_dim, int out_dim,
                                                 mlx_stream s, bool* ok) {
    int ratio = a->ratio;
    int padh = (ratio - n_h % ratio) % ratio;
    int padw = (ratio - n_w % ratio) % ratio;
    LLM_VIS_RET(mlx_reshape, x, x, (const int[]){ n_h, n_w, vis_dim }, 3);
    if (padh || padw) {
        mlx_array pv = mlx_array_new_float(0.0f), padded = mlx_array_new();
        int axes[2] = { 0, 1 }, lo[2] = { 0, 0 }, hi[2] = { padh, padw };
        if (mlx_pad(&padded, x, axes, 2, lo, 2, hi, 2, pv, "constant", s) != 0) { *ok = false; return mlx_array_new(); }
        x = padded;
    }
    int nh = n_h + padh, nw = n_w + padw;
    int oh = nh / ratio, ow = nw / ratio;
    LLM_VIS_RET(mlx_reshape, x, x, (const int[]){ oh, ratio, ow, ratio, vis_dim }, 5);
    mlx_array t = mlx_array_new();
    LLM_VIS_RET(mlx_transpose_axes, t, x, (const int[]){ 0, 2, 4, 1, 3 }, 5);
    LLM_VIS_RET(mlx_reshape, t, t, (const int[]){ oh * ow, vis_dim * ratio * ratio }, 2);
    /* gelu(w1(t)) -> w2 */
    mlx_array y = llm_vis_linear(t, a->w1, a->w1b, s, ok);
    mlx_array r2 = mlx_array_new_float(0.70710678f), yg = mlx_array_new(), erf = mlx_array_new();
    LLM_VIS_RET(mlx_multiply, yg, y, r2);
    LLM_VIS_RET(mlx_erf, erf, yg);
    {
        mlx_array one = mlx_array_new_float(1.0f), half = mlx_array_new_float(0.5f), t1 = mlx_array_new(), g = mlx_array_new();
        LLM_VIS_RET(mlx_add, t1, erf, one);
        LLM_VIS_RET(mlx_multiply, g, y, t1);
        LLM_VIS_RET(mlx_multiply, g, g, half);
        y = g;
    }
    return llm_vis_linear(y, a->w2, a->w2b, s, ok);
}

#endif /* LLM_DSV41_VISION_H */
