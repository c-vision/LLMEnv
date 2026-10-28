/* llm_mlx_linear_packed.h — M2b: proiezione lineare eseguita su PESI PACKED ZERO-COPY.
 *
 * API di integrazione: dato x [T][in] (mlx_array) e un descrittore LlmMlxPackedW, produce
 * out [T][rows] (mlx_array f32) eseguendo il GEMV affine-packed (pesi non materializzati).
 * v1: T=1 (decode), bits=4 (kernel Q4 del bridge). Nessun fallback silenzioso: ritorna !=0 e il
 * chiamante decide (in "packed" il fallback e' esplicito a MLX). */
#ifndef LLM_MLX_LINEAR_PACKED_H
#define LLM_MLX_LINEAR_PACKED_H

#include "mlx/c/mlx.h"
#include "llm_mlx_wpacked.h"
#include "llm_gemvq4_bridge.h"
#include <stdlib.h>
#include <string.h>

typedef struct { LlmGemvQ4Ctx* g; LlmMlxPackedW w; } LlmMlxPackedLinear;

/* Apre un tensore packed come "layer lineare" pronto al decode (carica i pesi nel ctx). */
static inline int llm_mlx_linear_packed_open(const char* st_path, const char* base, int bits, int group,
                                             LlmMlxPackedLinear* L) {
    if (!L) return -1;
    L->g = NULL;
    if (llm_mlx_packed_open(st_path, base, bits, group, &L->w) != 0) return -1;
    L->g = llm_gemvq4_create();
    if (!L->g) { llm_mlx_packed_close(&L->w); return -1; }
    if (llm_gemvq4_load_weights_wonly_nocopy(L->g, (const uint8_t*)L->w.w, L->w.scales, L->w.biases,
                                             L->w.cols, L->w.rows, (uint32_t)L->w.group) != 0) {
        llm_gemvq4_destroy(L->g); L->g = NULL; llm_mlx_packed_close(&L->w); return -1;
    }
    llm_gemvq4_set_scale_bf16(L->g, L->w.scale_bf16);
    return 0;
}

static inline void llm_mlx_linear_packed_close(LlmMlxPackedLinear* L) {
    if (!L) return;
    if (L->g) llm_gemvq4_destroy(L->g);
    llm_mlx_packed_close(&L->w);
    L->g = NULL;
}

/* x [1][in] f32 -> out [1][rows] f32. 0 ok. */
static inline int llm_mlx_linear_packed_run(const LlmMlxPackedLinear* L, mlx_array x, mlx_array* out,
                                            mlx_stream s) {
    if (!L || !L->g || !out || mlx_array_ndim(x) != 2) return -1;
    const int nrow = mlx_array_dim(x, 0);
    if (nrow < 1 || mlx_array_dim(x, 1) != (int)L->w.cols) return -1;
    if (L->w.bits != 4) return -2;   /* v1: solo Q4 */
    { mlx_vector_array ev = mlx_vector_array_new(); mlx_vector_array_append_value(ev, x);
      mlx_eval(ev); mlx_vector_array_free(ev); }
    const float* xf = mlx_array_data_float32(x);
    if (!xf) return -1;
    uint16_t* xh = malloc((size_t)L->w.cols * 2);
    uint16_t* yh = malloc((size_t)L->w.rows * 2);
    float* yf = malloc((size_t)L->w.rows * (size_t)nrow * 4);
    if (!xh || !yh || !yf) { free(xh); free(yh); free(yf); return -1; }
    int rc = 0;
    for (int t = 0; t < nrow && rc == 0; t++) {
        const float* xt = xf + (size_t)t * L->w.cols;
        for (uint32_t c = 0; c < L->w.cols; c++) { _Float16 h = (_Float16)xt[c]; memcpy(&xh[c], &h, 2); }
        rc = llm_gemvq4_run(L->g, xh, yh);
        if (rc == 0) for (uint32_t r = 0; r < L->w.rows; r++) { _Float16 h; memcpy(&h, &yh[r], 2); yf[(size_t)t*L->w.rows + r] = (float)h; }
    }
    if (rc == 0) { int sh[2] = {nrow, (int)L->w.rows}; *out = mlx_array_new_data(yf, sh, 2, MLX_FLOAT32); }
    free(xh); free(yh);
    (void)s;
    return rc;
}

#endif
