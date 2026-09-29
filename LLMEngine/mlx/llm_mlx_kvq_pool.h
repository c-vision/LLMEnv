/* llm_mlx_kvq_pool.h — KV page pool int8 come STORAGE PRIMARIO del cold.
 *
 * Architettura (port Splash `runtime/ops/PageStorage.*` + `PagedKv.hpp`):
 * NON e' un add-on sopra la cache fp. Il cold della KV vive SOLO qui, in pagine
 * int8 preallocate; `k_cache`/`v_cache` contengono SOLO la finestra recente (W
 * token, dimensione FISSA). Cosi' la memoria fp e' O(W) e il cold e' int8 in
 * pagine (nessun Concat O(pos), nessuno shrink da liberare).
 *
 * Layout pagina (Splash, page = LLM_KVQ_PAGE_TOKENS=32):
 *   keyData   int8  [kv-head][token][hd]   (token-major)
 *   valueData int8  [kv-head][token][hd]   (token-major, come il kernel read)
 *   keyScales/valueScales f32 [kv-head][token] (1 scala simmetrica per token-head)
 *
 * Capacita': cresce per POTENZE DI 2. L'append scrive sul posto (slice_update), mai Concat.
 * LE ALLOCAZIONI SONO PER-LAYER E LAZY: i modelli ibridi (linear_attention + full_attention,
 * es. Qwen3.5/Qwen3.6) non allocano il pool per i layer senza KV.
 */
#ifndef LLM_MLX_KVQ_POOL_H
#define LLM_MLX_KVQ_POOL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kvq_splash.h"
#include "llm_mlx_splash_store.h"   /* store int8 verificato (splash_store.h) */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Stato del pool per (sessione, modello). Un array per campo, [layer]; allocazione lazy. */
typedef struct {
    int layers;
    int kv_heads;
    int head_dim;
    llm_kvq_fmt fmt;
    int  cap_tokens;          /* capacita' allocata (multiplo di page_tokens) */
    int* committed_l;         /* token scritti PER LAYER */
    mlx_array* keyData;       /* int8  [nkv, cap, hd]  (token-major) */
    mlx_array* keyScales;     /* f32   [nkv, cap, 1] */
    mlx_array* valueData;     /* int8  [nkv, cap, hd]  (token-major) */
    mlx_array* valueScales;   /* f32   [nkv, cap, 1] */
    mlx_array* bias;          /* f32 ZERO [nkv, cap, 1] per layer: Splash e' simmetrico (q*scala) */
} llm_kvq_pool;

static inline int llm_kvq_pool_bytes_per_token(llm_kvq_pool* p) {
    const int d = p->fmt == LLM_KVQ_FMT_INT8 ? 1 : 2;
    return 2 * p->kv_heads * p->head_dim * d                 /* K + V data */
         + 2 * p->kv_heads * (int)sizeof(float);             /* scale K + V */
}

/* Alloca gli array di UN layer alla capacita' corrente. */
static inline bool llm_kvq_pool_alloc_layer(llm_kvq_pool* p, int l, int cap, mlx_stream s) {
    const mlx_dtype edt = (p->fmt == LLM_KVQ_FMT_INT8) ? MLX_INT8 : MLX_BFLOAT16;
    int ksh[3] = {p->kv_heads, cap, p->head_dim};
    int ssh[3] = {p->kv_heads, cap, 1};
    p->keyData[l] = mlx_array_new(); p->valueData[l] = mlx_array_new();
    p->keyScales[l] = mlx_array_new(); p->valueScales[l] = mlx_array_new();
    p->bias[l] = mlx_array_new();
    if (mlx_zeros(&p->keyData[l], ksh, 3, edt, s) != 0) return false;
    if (mlx_zeros(&p->valueData[l], ksh, 3, edt, s) != 0) return false;
    if (mlx_zeros(&p->keyScales[l], ssh, 3, MLX_FLOAT32, s) != 0) return false;
    if (mlx_zeros(&p->valueScales[l], ssh, 3, MLX_FLOAT32, s) != 0) return false;
    if (mlx_zeros(&p->bias[l], ssh, 3, MLX_FLOAT32, s) != 0) return false;
    return true;
}

static inline bool llm_kvq_pool_ensure(llm_kvq_pool* p, int l, mlx_stream s) {
    if (l < 0 || l >= p->layers) return false;
    if (p->keyData[l].ctx != NULL) return true;
    return llm_kvq_pool_alloc_layer(p, l, p->cap_tokens, s);
}

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
    p->bias      = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->committed_l = (int*)calloc((size_t)layers, sizeof(int));
    if (!p->keyData || !p->keyScales || !p->valueData || !p->valueScales || !p->bias || !p->committed_l) return false;
    return true;   /* NESSUNA allocazione per-layer qui: lazy in ensure() */
}

/* Raddoppia la capacita' ricopiando solo i layer GIA' allocati. */
static inline bool llm_kvq_pool_grow(llm_kvq_pool* p, mlx_stream s) {
    if (!p) return false;
    const int ncap = p->cap_tokens * 2;
    const mlx_dtype edt = (p->fmt == LLM_KVQ_FMT_INT8) ? MLX_INT8 : MLX_BFLOAT16;
    for (int l = 0; l < p->layers; l++) {
        if (p->keyData[l].ctx == NULL) continue;   /* non allocato -> resta lazy */
        int committed = p->committed_l[l];
        int ksh[3] = {p->kv_heads, ncap, p->head_dim};
        int ssh[3] = {p->kv_heads, ncap, 1};
        mlx_array nk = mlx_array_new(), nv = mlx_array_new(), nks = mlx_array_new(), nvs = mlx_array_new(), nb = mlx_array_new();
        bool ok = mlx_zeros(&nk, ksh, 3, edt, s) == 0 &&
                  mlx_zeros(&nv, ksh, 3, edt, s) == 0 &&
                  mlx_zeros(&nks, ssh, 3, MLX_FLOAT32, s) == 0 &&
                  mlx_zeros(&nvs, ssh, 3, MLX_FLOAT32, s) == 0 &&
                  mlx_zeros(&nb, ssh, 3, MLX_FLOAT32, s) == 0;
        if (ok && committed > 0) {
            int c0[3] = {0,0,0}, c1[3] = {p->kv_heads, committed, p->head_dim}, st[3] = {1,1,1};
            int s0[3] = {0,0,0}, s1[3] = {p->kv_heads, committed, 1};
            mlx_array ksrc = mlx_array_new(), vsrc = mlx_array_new(), kssrc = mlx_array_new(), vssrc = mlx_array_new();
            ok = mlx_slice(&ksrc, p->keyData[l], c0,3,c1,3,st,3, s) == 0 &&
                 mlx_slice(&vsrc, p->valueData[l], c0,3,c1,3,st,3, s) == 0 &&
                 mlx_slice(&kssrc, p->keyScales[l], s0,3,s1,3,st,3, s) == 0 &&
                 mlx_slice(&vssrc, p->valueScales[l], s0,3,s1,3,st,3, s) == 0;
            if (ok) {
                mlx_array ku = mlx_array_new(), vu = mlx_array_new(), ksu = mlx_array_new(), vsu = mlx_array_new();
                ok = mlx_slice_update(&ku, nk, ksrc, c0,3,c1,3,st,3, s) == 0 &&
                     mlx_slice_update(&vu, nv, vsrc, c0,3,c1,3,st,3, s) == 0 &&
                     mlx_slice_update(&ksu, nks, kssrc, s0,3,s1,3,st,3, s) == 0 &&
                     mlx_slice_update(&vsu, nvs, vssrc, s0,3,s1,3,st,3, s) == 0;
                if (ok) { mlx_array_set(&nk, ku); mlx_array_set(&nv, vu);
                          mlx_array_set(&nks, ksu); mlx_array_set(&nvs, vsu); }
                mlx_array_free(ku); mlx_array_free(vu); mlx_array_free(ksu); mlx_array_free(vsu);
            }
            mlx_array_free(ksrc); mlx_array_free(vsrc); mlx_array_free(kssrc); mlx_array_free(vssrc);
        }
        if (!ok) { mlx_array_free(nk); mlx_array_free(nv); mlx_array_free(nks); mlx_array_free(nvs); mlx_array_free(nb); return false; }
        mlx_array_free(p->keyData[l]); mlx_array_free(p->valueData[l]);
        mlx_array_free(p->keyScales[l]); mlx_array_free(p->valueScales[l]); mlx_array_free(p->bias[l]);
        p->keyData[l] = nk; p->valueData[l] = nv; p->keyScales[l] = nks; p->valueScales[l] = nvs; p->bias[l] = nb;
    }
    p->cap_tokens = ncap;
    return true;
}

/* Store n token (fp [1,nkv,n,hd] o [nkv,n,hd]) nel pool alla posizione `pos`. */
static inline bool llm_kvq_pool_store(llm_kvq_pool* p, int layer,
                                      mlx_array k, mlx_array v, int pos, int n, mlx_stream s) {
    if (!p || layer < 0 || layer >= p->layers || n <= 0) return false;
    if (!llm_kvq_pool_ensure(p, layer, s)) return false;
    while (pos + n > p->cap_tokens) { if (!llm_kvq_pool_grow(p, s)) return false; }
    int sh3[3] = {p->kv_heads, n, p->head_dim};
    mlx_array k3 = mlx_array_new(), v3 = mlx_array_new();
    bool ok = mlx_reshape(&k3, k, sh3, 3, s) == 0 && mlx_reshape(&v3, v, sh3, 3, s) == 0;
    mlx_array kq = mlx_array_new(), vq = mlx_array_new(), ks = mlx_array_new(), vs = mlx_array_new();
    if (ok) ok = llm_splash_store_run(&kq, &vq, &ks, &vs, k3, v3, s) == 0;   /* store VERIFICATO */
    mlx_array_free(k3); mlx_array_free(v3);
    mlx_array sc2 = mlx_array_new(), vsc2 = mlx_array_new(), vqt = mlx_array_new();
    if (ok) { int s2[3] = {p->kv_heads, n, 1};
              ok = mlx_reshape(&sc2, ks, s2, 3, s) == 0 && mlx_reshape(&vsc2, vs, s2, 3, s) == 0; }
    if (ok) ok = mlx_array_set(&vqt, vq) == 0;   /* V token-major */
    if (ok) {
        mlx_array nk = mlx_array_new(), nv = mlx_array_new(), nks = mlx_array_new(), nvs = mlx_array_new();
        int k0[3] = {0,pos,0},  k1[3] = {p->kv_heads,pos+n,p->head_dim}, kst[3] = {1,1,1};
        int s0[3] = {0,pos,0},  s1[3] = {p->kv_heads,pos+n,1}, sst[3] = {1,1,1};
        ok = mlx_slice_update(&nk, p->keyData[layer], kq, k0,3,k1,3,kst,3, s) == 0 &&
             mlx_slice_update(&nv, p->valueData[layer], vqt, k0,3,k1,3,kst,3, s) == 0 &&
             mlx_slice_update(&nks, p->keyScales[layer], sc2, s0,3,s1,3,sst,3, s) == 0 &&
             mlx_slice_update(&nvs, p->valueScales[layer], vsc2, s0,3,s1,3,sst,3, s) == 0;
        if (ok) { mlx_array_set(&p->keyData[layer], nk); mlx_array_set(&p->valueData[layer], nv);
                  mlx_array_set(&p->keyScales[layer], nks); mlx_array_set(&p->valueScales[layer], nvs); }
        mlx_array_free(nk); mlx_array_free(nv); mlx_array_free(nks); mlx_array_free(nvs);
    }
    if (ok) p->committed_l[layer] = pos + n;
    mlx_array_free(sc2); mlx_array_free(vsc2); mlx_array_free(vqt);
    mlx_array_free(kq); mlx_array_free(vq); mlx_array_free(ks); mlx_array_free(vs);
    return ok;
}

static inline void llm_kvq_pool_free(llm_kvq_pool* p) {
    if (!p || !p->keyData) return;
    for (int l = 0; l < p->layers; l++) {
        mlx_array_free(p->keyData[l]); mlx_array_free(p->valueData[l]);
        mlx_array_free(p->keyScales[l]); mlx_array_free(p->valueScales[l]);
        if (p->bias) mlx_array_free(p->bias[l]);
    }
    free(p->keyData); free(p->keyScales); free(p->valueData); free(p->valueScales);
    free(p->bias); free(p->committed_l);
    memset(p, 0, sizeof *p);
}

/* ── Registry: un pool per (modello, sessione), creato lazy. ───────────────── */
typedef struct { void* model; void* sess; llm_kvq_pool p; } llm_kvq_pool_entry;
static llm_kvq_pool_entry g_kvq_pools[8];
static int g_kvq_pools_n = 0;

static inline llm_kvq_pool* llm_kvq_pool_for(void* m, void* sess,
                                             int layers, int nkv, int hd, llm_kvq_fmt fmt,
                                             int cap_tokens, mlx_stream s) {
    for (int i = 0; i < g_kvq_pools_n; i++)
        if (g_kvq_pools[i].model == m && g_kvq_pools[i].sess == sess) return &g_kvq_pools[i].p;
    if (g_kvq_pools_n >= 8) return NULL;
    llm_kvq_pool_entry* e = &g_kvq_pools[g_kvq_pools_n];
    memset(e, 0, sizeof *e);
    e->model = m; e->sess = sess;
    if (!llm_kvq_pool_init(&e->p, layers, nkv, hd, fmt, cap_tokens, s)) return NULL;
    g_kvq_pools_n++;
    return &e->p;
}

#endif /* LLM_MLX_KVQ_POOL_H */
