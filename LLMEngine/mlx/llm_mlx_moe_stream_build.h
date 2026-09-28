/* llm_mlx_moe_stream_build.h — costruisce i tensori STACKED [k, d1, d2] dei k esperti unici
 * a partire dalle SLICE dell'ingest canonico safetensors (`llmengine_moe_safetensors_ingest.h`).
 *
 * NON un formato nuovo: il descrittore `(layer,expert,proj,kind) -> slice` e' UNO (l'ingest);
 * qui c'e' solo il consumo lato MLX (costruzione degli mlx_array per gather_qmm). I bundle sono
 * copiati in buffer RESIDENTI (mlock) -> Metal non fa fault (no Command buffer). */
#ifndef LLM_MLX_MOE_STREAM_BUILD_H
#define LLM_MLX_MOE_STREAM_BUILD_H
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "../moe/include/llmengine_moe_safetensors_ingest.h"

/* Costruisce+materializza [k, d1, d2] per (layer, proj, kind) dagli esperti `uniq[]`.
 * kind: 0=weight(U32) 1=scales(BF16) 2=biases(BF16). Ritorna 0 e *out, oppure -1. */
static inline int llm_mlx_xs_stack(const LlmEngineMoeTsIngest* s, int layer, int proj, int kind,
                                   const int* uniq, int k, mlx_stream str, mlx_array* out) {
    if (k <= 0 || !out) return -1;
    const LlmEngineMoeTsSlice* t = llmengine_moe_ts_slice(s, layer, proj, kind);
    if (!t || t->d1 <= 0 || t->d2 <= 0) return -1;
    mlx_array* arrs = (mlx_array*)calloc((size_t)k, sizeof(mlx_array));
    uint8_t** bufs = (uint8_t**)calloc((size_t)k, sizeof(uint8_t*));
    if (!arrs || !bufs) { free(arrs); free(bufs); return -1; }
    const mlx_dtype dt = (kind == 0) ? MLX_UINT32 : MLX_BFLOAT16;
    int shp[2] = { t->d1, t->d2 };
    int ok = 0;
    for (int j = 0; j < k && ok == 0; j++) {
        uint32_t bytes = 0;
        const uint8_t* p = llmengine_moe_ts_get(s, layer, proj, kind, uniq[j], &bytes);
        if (!p) { ok = -1; break; }
        bufs[j] = (uint8_t*)malloc(bytes);
        if (!bufs[j]) { ok = -1; break; }
        memcpy(bufs[j], p, bytes);
        (void)mlock(bufs[j], bytes); /* best-effort: buffer comunque residente */
        arrs[j] = mlx_array_new_data(bufs[j], shp, 2, dt);
    }
    mlx_array tmp = mlx_array_new();
    if (ok == 0) {
        mlx_vector_array v = mlx_vector_array_new_data(arrs, (size_t)k);
        if (mlx_stack(&tmp, v, str) != 0) ok = -1;
        else if (mlx_array_eval(tmp) != 0) ok = -1; /* materializza: i buf non servono piu' */
    }
    for (int j = 0; j < k; j++) {
        if (arrs[j].ctx != NULL) mlx_array_free(arrs[j]);
        if (bufs[j]) { munlock(bufs[j], t->bytes); free(bufs[j]); }
    }
    free(arrs); free(bufs);
    if (ok != 0) { mlx_array_free(tmp); return -1; }
    *out = tmp;
    return 0;
}

/* I 9 tensori di un layer (gate/up/down × weight/scales/biases) per gli esperti `uniq[]`. */
typedef struct {
    mlx_array w[LLMENGINE_MOE_TS_NPROJ]; /* 0=gate 1=up 2=down, weight  */
    mlx_array s[LLMENGINE_MOE_TS_NPROJ]; /* scales */
    mlx_array b[LLMENGINE_MOE_TS_NPROJ]; /* biases */
} LlmMlxStreamTriples;

/* Prepara i 9 stacked [k,d1,d2] per (layer, uniq[], k). 0 ok / -1. */
static inline int llm_mlx_moe_swap_prepare(const LlmEngineMoeTsIngest* ing, int layer,
                                           const int* uniq, int k, mlx_stream str,
                                           LlmMlxStreamTriples* out) {
    for (int p = 0; p < LLMENGINE_MOE_TS_NPROJ; p++) {
        for (int kd = 0; kd < LLMENGINE_MOE_TS_NKIND; kd++) {
            mlx_array a = mlx_array_new();
            if (llm_mlx_xs_stack(ing, layer, p, kd, uniq, k, str, &a) != 0) return -1;
            if (kd == 0) out->w[p] = a; else if (kd == 1) out->s[p] = a; else out->b[p] = a;
        }
    }
    return 0;
}

#endif /* LLM_MLX_MOE_STREAM_BUILD_H */
