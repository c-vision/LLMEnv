/* Packed affine QKV, owned by one layer. No global tensor cache. Experimental.
 * Like llm_mlx_ops.h this is a composable C module included by the native engine.
 * Keep the original weights for fallback/LoRA; materialize the packed copy ONCE,
 * outside generation, so a lazy concatenate cannot retain a new graph per token.
 */
#ifndef LLM_MLX_PACKED_QKV_H
#define LLM_MLX_PACKED_QKV_H
#include "mlx/c/mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    mlx_array w, s, b;
    int rows[3];
    bool ready;
    unsigned reported;
} LlmMlxPackedQkv;

typedef bool (*LlmMlxQkvLinear)(mlx_array*, mlx_array, mlx_array, mlx_array,
                              mlx_array, int, int, const char*, mlx_stream);

static bool llm_mlx_qkv_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        /* DEFAULT ON (2026-09-13): fonde q/k/v proj in UNA GEMV/GEMM (3 dispatch -> 1).
         * Misurato su Llama-1B: prefill +2..+14% (N=2048 il piu' pulito), decode ~+1,6%,
         * LLM_PACKED_QKV=0 kill-switch. IDS greedy identici. Applicato solo dove il packing
         * e' popolato al load (llama/qwen3 affine), altrove fallback alla proiezione singola. */
        const char* flag = llm_ai_mlx_opt_env("LLM_PACKED_QKV");
        enabled = (flag != NULL && strcmp(flag, "0") == 0) ? 0 : 1;
    }
    return enabled != 0;
}

static void llm_mlx_qkv_destroy(LlmMlxPackedQkv* p) {
    mlx_array_free(p->w); mlx_array_free(p->s); mlx_array_free(p->b);
    memset(p, 0, sizeof(*p));
}

static bool llm_mlx_qkv_concat(mlx_array* out, const mlx_array a[3], mlx_stream st) {
    mlx_vector_array v = mlx_vector_array_new_data(a, 3);
    bool ok = mlx_concatenate_axis(out, v, 0, st) == 0;
    mlx_vector_array_free(v);
    return ok;
}

/* Reject mixed quantization, plain weights, missing affine biases, and casts
 * introduced by concatenate. Equal column geometry preserves per-tensor derive.
 * Caller restricts architecture and quant mode before invoking this function. */
static bool llm_mlx_qkv_pack(LlmMlxPackedQkv* p, const mlx_array w[3],
                            const mlx_array s[3], const mlx_array b[3], mlx_stream st) {
    if (p->ready) return true;
    int total = 0;
    for (int i = 0; i < 3; ++i) {
        if (!w[i].ctx || !s[i].ctx || !b[i].ctx ||
            mlx_array_ndim(w[i]) != 2 || mlx_array_ndim(s[i]) != 2 ||
            mlx_array_ndim(b[i]) != 2 || mlx_array_dtype(w[i]) != MLX_UINT32)
            return false;
        const int* ws = mlx_array_shape(w[i]);
        const int* ss = mlx_array_shape(s[i]);
        const int* bs = mlx_array_shape(b[i]);
        if (ws[0] <= 0 || ws[1] <= 0 || ss[1] <= 0 || total > INT_MAX - ws[0] ||
            ss[0] != ws[0] || bs[0] != ws[0] || bs[1] != ss[1] ||
            ws[1] != mlx_array_shape(w[0])[1] || ss[1] != mlx_array_shape(s[0])[1] ||
            mlx_array_dtype(s[i]) != mlx_array_dtype(s[0]) ||
            mlx_array_dtype(b[i]) != mlx_array_dtype(b[0])) return false;
        p->rows[i] = ws[0];
        total += ws[0];
    }
    bool ok = llm_mlx_qkv_concat(&p->w, w, st) &&
              llm_mlx_qkv_concat(&p->s, s, st) &&
              llm_mlx_qkv_concat(&p->b, b, st);
    if (ok) {
        mlx_array arrays[3] = {p->w, p->s, p->b};
        mlx_vector_array v = mlx_vector_array_new_data(arrays, 3);
        ok = mlx_eval(v) == 0;
        mlx_vector_array_free(v);
    }
    if (!ok) { llm_mlx_qkv_destroy(p); return false; }
    p->ready = true;
    return true;
}

/* One linear operation, three views. LoRA and real additive projection biases
 * remain in each architecture after this call. Failure publishes no output. */
static bool llm_mlx_qkv_apply(LlmMlxPackedQkv* p, mlx_array x,
                             mlx_array* q, mlx_array* k, mlx_array* v,
                             int group, int bits, const char* mode, mlx_stream st,
                             LlmMlxQkvLinear linear) {
    if (!p || !p->ready || mlx_array_ndim(x) != 3) return false;
    mlx_array y = mlx_array_new();
    bool ok = linear(&y, x, p->w, p->s, p->b, group, bits, mode, st);
    /* Split q/k/v in UNA primitiva (Split, 1 nodo con 3 output) invece di 3 mlx_slice:
     * 3 nodi -> 1 per layer (verificato: -32 nodi/token su Llama-1B decode). Viste
     * equivalenti -> bit-identico. */
    if (ok) {
        mlx_vector_array sec = mlx_vector_array_new();
        int cuts[2] = {p->rows[0], p->rows[0] + p->rows[1]};
        ok = mlx_split_sections(&sec, y, cuts, 2, 2, st) == 0 &&
             mlx_vector_array_size(sec) == 3 &&
             mlx_vector_array_get(q, sec, 0) == 0 &&
             mlx_vector_array_get(k, sec, 1) == 0 &&
             mlx_vector_array_get(v, sec, 2) == 0;
        mlx_vector_array_free(sec);
    }
    mlx_array_free(y);
    if (!ok) fprintf(stderr, "[packed-qkv] projection failed; using separate Q/K/V\n");
    return ok;
}
#endif
