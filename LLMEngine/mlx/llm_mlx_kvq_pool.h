/* llm_mlx_kvq_pool.h — KV page pool int8 come STORAGE PRIMARIO del cold.
 *
 * Architettura (port Splash `runtime/ops/PageStorage.*` + `PagedKv.hpp`):
 * NON e' un add-on sopra la cache fp. Il cold della KV vive SOLO qui, in pagine
 * int8 preallocate; `k_cache`/`v_cache` contengono SOLO la finestra recente (W
 * token, dimensione FISSA). Cosi' la memoria fp e' O(W) e il cold e' int8 in
 * pagine (nessun Concat O(pos), nessuno shrink da liberare -> niente +1266 MB:
 * la cache fp del cold NON esiste mai).
 *
 * Layout pagina (Splash, page = LLM_KVQ_PAGE_TOKENS=32):
 *   keyData   int8  [page][kv-head][token][hd]   (token-major)
 *   valueData int8  [page][kv-head][hd][token]   (dimension-major)
 *   keyScales/valueScales f32 [page][kv-head][token] (1 scala simmetrica per token-head)
 *
 * Capacita': cresce per POTENZE DI 2 (un "extent" = N pagine). L'append scrive
 * sul posto (slice_update), mai un Concat.
 */
#ifndef LLM_MLX_KVQ_POOL_H
#define LLM_MLX_KVQ_POOL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kvq_splash.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Stato del pool per (sessione, layer). Un array per campo, [layer]. */
typedef struct {
    int layers;
    int kv_heads;
    int head_dim;
    llm_kvq_fmt fmt;
    /* Capacita' (token) e array, per layer. */
    int  cap_tokens;          /* capacita' allocata (multiplo di page_tokens) */
    int  committed;           /* token scritti (logici) */
    mlx_array* keyData;       /* int8  [nkv, cap, hd]  (token-major, vedi nota sotto) */
    mlx_array* keyScales;     /* f32   [nkv, cap] */
    mlx_array* valueData;     /* int8  [nkv, hd, cap]  (dimension-major) */
    mlx_array* valueScales;   /* f32   [nkv, cap] */
} llm_kvq_pool;

/* Nota: per semplicita' dell'host usiamo [nkv, cap, hd] (K) e [nkv, hd, cap] (V)
 * invece del nesting per-page; gli INDICI per-page di Splash restano validi
 * perche' page 0..P e' un prefisso contiguo della capacita'. */

static inline int llm_kvq_pool_bytes_per_token(llm_kvq_pool* p) {
    const int d = p->fmt == LLM_KVQ_FMT_INT8 ? 1 : 2;
    return 2 * p->kv_heads * p->head_dim * d                 /* K + V data */
         + 2 * p->kv_heads * (int)sizeof(float);             /* scale K + V */
}

/* Alloca il pool per un modello. cap_tokens multiplo di page_tokens. */
static inline bool llm_kvq_pool_init(llm_kvq_pool* p, int layers, int kv_heads, int head_dim,
                                     llm_kvq_fmt fmt, int cap_tokens, mlx_stream s) {
    if (!p || layers <= 0 || kv_heads <= 0 || head_dim <= 0) return false;
    memset(p, 0, sizeof *p);
    p->layers = layers; p->kv_heads = kv_heads; p->head_dim = head_dim; p->fmt = fmt;
    if (cap_tokens < LLM_KVQ_PAGE_TOKENS) cap_tokens = LLM_KVQ_PAGE_TOKENS;
    cap_tokens = ((cap_tokens + LLM_KVQ_PAGE_TOKENS - 1) / LLM_KVQ_PAGE_TOKENS) * LLM_KVQ_PAGE_TOKENS;
    p->cap_tokens = cap_tokens;
    p->keyData   = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->keyScales = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->valueData = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->valueScales = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    if (!p->keyData || !p->keyScales || !p->valueData || !p->valueScales) return false;
    const mlx_dtype edt = (fmt == LLM_KVQ_FMT_INT8) ? MLX_INT8 : MLX_BFLOAT16;
    for (int l = 0; l < layers; l++) {
        int ksh[3] = {kv_heads, cap_tokens, head_dim};
        int vsh[3] = {kv_heads, head_dim, cap_tokens};
        int ssh[2] = {kv_heads, cap_tokens};
        p->keyData[l]   = mlx_array_new(); p->valueData[l]   = mlx_array_new();
        p->keyScales[l] = mlx_array_new(); p->valueScales[l] = mlx_array_new();
        if (mlx_zeros(&p->keyData[l], ksh, 3, edt, s) != 0) return false;
        if (mlx_zeros(&p->valueData[l], vsh, 3, edt, s) != 0) return false;
        if (mlx_zeros(&p->keyScales[l], ssh, 2, MLX_FLOAT32, s) != 0) return false;
        if (mlx_zeros(&p->valueScales[l], ssh, 2, MLX_FLOAT32, s) != 0) return false;
    }
    return true;
}

/* Raddoppia la capacita' (rialloca + copia il prefisso committed). Ritorna false su errore. */
static inline bool llm_kvq_pool_grow(llm_kvq_pool* p, mlx_stream s) {
    if (!p) return false;
    const int ncap = p->cap_tokens * 2;
    const mlx_dtype edt = (p->fmt == LLM_KVQ_FMT_INT8) ? MLX_INT8 : MLX_BFLOAT16;
    for (int l = 0; l < p->layers; l++) {
        int ksh[3] = {p->kv_heads, ncap, p->head_dim};
        int vsh[3] = {p->kv_heads, p->head_dim, ncap};
        int ssh[2] = {p->kv_heads, ncap};
        mlx_array nk = mlx_array_new(), nv = mlx_array_new(), nks = mlx_array_new(), nvs = mlx_array_new();
        bool ok = mlx_zeros(&nk, ksh, 3, edt, s) == 0 &&
                  mlx_zeros(&nv, vsh, 3, edt, s) == 0 &&
                  mlx_zeros(&nks, ssh, 2, MLX_FLOAT32, s) == 0 &&
                  mlx_zeros(&nvs, ssh, 2, MLX_FLOAT32, s) == 0;
        if (ok && p->committed > 0) {
            /* copia il prefisso scritto: K [nkv, committed, hd], V [nkv, hd, committed] */
            int ks0[3] = {0,0,0}, ks1[3] = {p->kv_heads, p->committed, p->head_dim}, kst[3] = {1,1,1};
            int vs0[3] = {0,0,0}, vs1[3] = {p->kv_heads, p->head_dim, p->committed}, vst[3] = {1,1,1};
            mlx_array ksrc = mlx_array_new(), vsrc = mlx_array_new();
            mlx_array kssrc = mlx_array_new(), vssrc = mlx_array_new();
            ok = ok && mlx_slice(&ksrc, p->keyData[l], ks0, 3, ks1, 3, kst, 3, s) == 0 &&
                       mlx_slice(&vsrc, p->valueData[l], vs0, 3, vs1, 3, vst, 3, s) == 0;
            int ss0[2] = {0,0}, ss1[2] = {p->kv_heads, p->committed}, sst[2] = {1,1};
            ok = ok && mlx_slice(&kssrc, p->keyScales[l], ss0, 2, ss1, 2, sst, 2, s) == 0 &&
                       mlx_slice(&vssrc, p->valueScales[l], ss0, 2, ss1, 2, sst, 2, s) == 0;
            if (ok) { /* scrivi il prefisso in testa ai nuovi buffer */
                mlx_array ku = mlx_array_new(), vu = mlx_array_new(), ksu = mlx_array_new(), vsu = mlx_array_new();
                int ku0[3] = {0,0,0}, ku1[3] = {p->kv_heads, p->committed, p->head_dim}, ust[3] = {1,1,1};
                int vu0[3] = {0,0,0}, vu1[3] = {p->kv_heads, p->head_dim, p->committed};
                ok = mlx_slice_update(&ku, nk, ksrc, ku0, 3, ku1, 3, ust, 3, s) == 0 &&
                     mlx_slice_update(&vu, nv, vsrc, vu0, 3, vu1, 3, ust, 3, s) == 0;
                if (ok) mlx_array_set(&nk, ku);
                if (ok) mlx_array_set(&nv, vu);
                int su0[2] = {0,0}, su1[2] = {p->kv_heads, p->committed}, sust[2] = {1,1};
                ok = ok && mlx_slice_update(&ksu, nks, kssrc, su0, 2, su1, 2, sust, 2, s) == 0 &&
                           mlx_slice_update(&vsu, nvs, vssrc, su0, 2, su1, 2, sust, 2, s) == 0;
                if (ok) mlx_array_set(&nks, ksu);
                if (ok) mlx_array_set(&nvs, vsu);
                mlx_array_free(ku); mlx_array_free(vu); mlx_array_free(ksu); mlx_array_free(vsu);
            }
            mlx_array_free(ksrc); mlx_array_free(vsrc); mlx_array_free(kssrc); mlx_array_free(vssrc);
        }
        if (!ok) { mlx_array_free(nk); mlx_array_free(nv); mlx_array_free(nks); mlx_array_free(nvs); return false; }
        mlx_array_free(p->keyData[l]); mlx_array_free(p->valueData[l]);
        mlx_array_free(p->keyScales[l]); mlx_array_free(p->valueScales[l]);
        p->keyData[l] = nk; p->valueData[l] = nv; p->keyScales[l] = nks; p->valueScales[l] = nvs;
    }
    p->cap_tokens = ncap;
    return true;
}

static inline void llm_kvq_pool_free(llm_kvq_pool* p) {
    if (!p || !p->keyData) return;
    for (int l = 0; l < p->layers; l++) {
        mlx_array_free(p->keyData[l]); mlx_array_free(p->valueData[l]);
        mlx_array_free(p->keyScales[l]); mlx_array_free(p->valueScales[l]);
    }
    free(p->keyData); free(p->keyScales); free(p->valueData); free(p->valueScales);
    memset(p, 0, sizeof *p);
}

#endif /* LLM_MLX_KVQ_POOL_H */
