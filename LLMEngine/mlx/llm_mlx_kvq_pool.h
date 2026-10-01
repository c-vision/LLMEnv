/* llm_mlx_kvq_pool.h — KV page pool int8 PAGED (storage primario del cold).
 *
 * Port dell'architettura `PageStorage`/`PagedKv.hpp` di Splash: il cold della KV vive in
 * pagine fisiche int8; i blocchi logici (32 token) sono mappati su pagine fisiche da una
 * **page table** (`llm_mlx_kvpage_store.h`), quindi i blocchi contigui del prefisso possono
 * stare in pagine NON contigue (basi per prefix-cache/eviction/tier). La cache fp contiene
 * SOLO la finestra recente (via layer_forward).
 *
 * Layout per pagina (token-major, [nkv, cap, hd]): la pagina fisica `phys` occupa i token
 * [phys*32, phys*32+32); il token logico `t` sta in `ptab[t/32]*32 + (t%32)`.
 * Allocazione LAZY per-layer (i modelli ibridi non allocano i layer senza KV).
 */
#ifndef LLM_MLX_KVQ_POOL_H
#define LLM_MLX_KVQ_POOL_H

#include "mlx/c/mlx.h"
#include "llm_mlx_kvq_splash.h"
#include "llm_mlx_splash_store.h"    /* store int8 verificato */
#include "llm_mlx_kvpage_store.h"    /* page table + allocator (Splash-like) */
#include "llm_mlx_memgov.h"          /* B9: memory governor */
#include "llm_mlx_kvdisk.h"          /* B9: tier SSD per spill/fault pagine */
#include "llm_engine_opts.h"         /* registro opzioni (memgov_*) */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int layers;
    int kv_heads;
    int head_dim;
    llm_kvq_fmt fmt;
    int  cap_tokens;          /* capacita' FISICA (multiplo di page_tokens) */
    int  page_tokens;         /* token per pagina (32) */
    int* committed_l;         /* token logici scritti PER LAYER */
    mlx_array* keyData;       /* int8  [nkv, cap, hd]  (token-major, physical) */
    mlx_array* keyScales;     /* f32   [nkv, cap, 1] */
    mlx_array* valueData;     /* int8  [nkv, cap, hd]  (token-major, physical) */
    mlx_array* valueScales;   /* f32   [nkv, cap, 1] */
    mlx_array* bias;          /* f32 ZERO [nkv, cap, 1] per layer */
    LlmKvPageTable pt;        /* page table logical->physical (condivisa fra i layer) */
    LlmMemGov gov;            /* B9: budget memoria (reserve/grow/timeout) */
    KvDisk*   disk;          /* B9: tier SSD per spill/fault delle pagine (NULL = off) */
    uint64_t  key_base;      /* B9: base chiave distinta per questo pool */
    mlx_array ptab;           /* int32 [phys_pages] mirror della page table (per il kernel) */
} llm_kvq_pool;

static inline int llm_kvq_pool_bytes_per_token(llm_kvq_pool* p) {
    const int d = p->fmt == LLM_KVQ_FMT_INT8 ? 1 : 2;
    return 2 * p->kv_heads * p->head_dim * d + 2 * p->kv_heads * (int)sizeof(float);
}

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

/* Riallinea l'array `ptab` (int32) alla page table corrente. */
static inline bool llm_kvq_pool_export_ptab(llm_kvq_pool* p, mlx_stream s) {
    int np = p->pt.phys_capacity;
    int32_t* buf = (int32_t*)malloc((size_t)np * sizeof(int32_t));
    if (!buf) return false;
    llm_kv_pt_export(&p->pt, buf, np);
    int sh[1] = {np};
    mlx_array a = mlx_array_new_data(buf, sh, 1, MLX_INT32);
    free(buf);
    if (a.ctx == NULL) return false;
    if (p->ptab.ctx != NULL) mlx_array_free(p->ptab);
    p->ptab = mlx_array_new();
    bool ok = mlx_array_set(&p->ptab, a) == 0;
    mlx_array_free(a);
    (void)s;
    return ok;
}

static inline bool llm_kvq_pool_init(llm_kvq_pool* p, int layers, int kv_heads, int head_dim,
                                     llm_kvq_fmt fmt, int cap_tokens, mlx_stream s) {
    if (!p || layers <= 0 || kv_heads <= 0 || head_dim <= 0) return false;
    memset(p, 0, sizeof *p);
    p->layers = layers; p->kv_heads = kv_heads; p->head_dim = head_dim; p->fmt = fmt;
    p->page_tokens = (int)LLM_KV_PAGE_TOKENS;
    p->gov = llm_memgov_from_registry();
    if (cap_tokens < (int)LLM_KVQ_PAGE_TOKENS) cap_tokens = (int)LLM_KVQ_PAGE_TOKENS;
    cap_tokens = ((cap_tokens + (int)LLM_KVQ_PAGE_TOKENS - 1) / (int)LLM_KVQ_PAGE_TOKENS) * (int)LLM_KVQ_PAGE_TOKENS;
    p->cap_tokens = cap_tokens;
    int nphys = cap_tokens / p->page_tokens;
    p->pt = llm_kv_pt_new(nphys, nphys);
    if (!p->pt.table) return false;
    p->keyData   = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->keyScales = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->valueData = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->valueScales = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->bias      = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    p->committed_l = (int*)calloc((size_t)layers, sizeof(int));
    if (!p->keyData || !p->keyScales || !p->valueData || !p->valueScales || !p->bias || !p->committed_l) return false;
    p->ptab = mlx_array_new();
    if (!llm_kvq_pool_export_ptab(p, s)) return false;
    return true;
}

/* Raddoppia la capacita' fisica ricopiando i layer allocati e allargando la page table. */
/* B9: true se qualche pagina logica e' stata spilliata (non residente). */
static inline bool llm_kvq_pool_offloaded(const llm_kvq_pool* p) {
    if (!p) return false;
    for (int lp = 0; lp < p->pt.n_logical; lp++) if (p->pt.table[lp] < 0) return true;
    return false;
}
/* B9: sotto pressione, spilla su disco le pagine dei pool INATTIVI (non `p`). */
static inline void llm_kvq_pool_pressure_pool(llm_kvq_pool* p, mlx_stream s);

static inline bool llm_kvq_pool_grow(llm_kvq_pool* p, mlx_stream s) {
    if (!p) return false;
    llm_kvq_pool_pressure_pool(p, s);   /* B9: libera RAM spillando i pool inattivi */
    /* B9: prima di crescere, chiedi memoria al governor (attende -> resource_timeout). */
    { uint64_t grew = llm_kv_page_bytes(p->fmt == LLM_KVQ_FMT_INT8, (uint32_t)p->kv_heads, (uint32_t)p->head_dim)
                    * (uint64_t)(p->pt.phys_capacity);  /* pagine aggiunte (doppio) */
      if (llm_memgov_wait(&p->gov, grew) != 0) return false; }
    /* CRESCITA 1.25x (2026-10-01): il raddoppio portava la cap a ~2x l'usato -> il pool
     * int8 costava ~66KB/token (misurato +4.5GB@71k, picco 44 vs 39.6 con kvquant=off).
     * Con +1/8 l'over-allocazione scende a ~1.125x; le riallocazioni restano O(log). */
    const int ngrow = (p->pt.phys_capacity / 8) > 1 ? (p->pt.phys_capacity / 8) : 1;
    if (!llm_kv_pt_grow(&p->pt, p->pt.phys_capacity + ngrow, p->pt.phys_capacity + ngrow)) return false;
    const int ncap = p->cap_tokens + (p->cap_tokens / 8 > (int)LLM_KVQ_PAGE_TOKENS ? p->cap_tokens / 8 : (int)LLM_KVQ_PAGE_TOKENS);
    const mlx_dtype edt = (p->fmt == LLM_KVQ_FMT_INT8) ? MLX_INT8 : MLX_BFLOAT16;
    for (int l = 0; l < p->layers; l++) {
        if (p->keyData[l].ctx == NULL) continue;
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
    return llm_kvq_pool_export_ptab(p, s);
}

/* Store n token (fp [1,nkv,n,hd] o [nkv,n,hd]) nel pool alla posizione LOGICA assoluta `pos`.
 * Scrive per segmenti di pagina, agli indici FISICI dati dalla page table. */
static inline bool llm_kvq_pool_store(llm_kvq_pool* p, int layer,
                                      mlx_array k, mlx_array v, int pos, int n, mlx_stream s) {
    if (!p || layer < 0 || layer >= p->layers || n <= 0) return false;
    if (!llm_kvq_pool_ensure(p, layer, s)) return false;
    while (pos + n > p->cap_tokens) { if (!llm_kvq_pool_grow(p, s)) return false; }
    int sh3[3] = {p->kv_heads, n, p->head_dim};
    mlx_array k3 = mlx_array_new(), v3 = mlx_array_new();
    bool ok = mlx_reshape(&k3, k, sh3, 3, s) == 0 && mlx_reshape(&v3, v, sh3, 3, s) == 0;
    mlx_array kq = mlx_array_new(), vq = mlx_array_new(), ks = mlx_array_new(), vs = mlx_array_new();
    if (ok) ok = llm_splash_store_run(&kq, &vq, &ks, &vs, k3, v3, s) == 0;
    mlx_array_free(k3); mlx_array_free(v3);
    mlx_array sc2 = mlx_array_new(), vsc2 = mlx_array_new(), vqt = mlx_array_new();
    if (ok) { int s2[3] = {p->kv_heads, n, 1};
              ok = mlx_reshape(&sc2, ks, s2, 3, s) == 0 && mlx_reshape(&vsc2, vs, s2, 3, s) == 0; }
    if (ok) ok = mlx_array_set(&vqt, vq) == 0;
    /* scrittura COALESCENTE: token logico `pos+j` -> fisico `ptab[lp]*32 + r`. Se le pagine
     * fisiche consecutive sono contigue (caso default = identita'), l'intera corsa e' UN solo
     * slice_update; si spezza solo quando la page table rende le pagine non contigue. */
    int j = 0;
    while (ok && j < n) {
        int abs = pos + j, lp = abs / p->page_tokens, r = abs % p->page_tokens;
        int phys = llm_kv_pt_map(&p->pt, lp);
        if (phys < 0) { ok = false; break; }
        int ptok = phys * p->page_tokens + r;
        int seg = n - j; if (seg > p->page_tokens - r) seg = p->page_tokens - r;
        {   /* estendi la corsa finche' le pagine fisiche successive sono contigue */
            int lp2 = lp + 1, j2 = j + seg;
            while (j2 < n && (pos + j2) % p->page_tokens == 0) {
                int lpA = (pos + j2) / p->page_tokens;
                int physA = llm_kv_pt_map(&p->pt, lpA);
                if (physA != phys + (lpA - lp)) break;
                int seg2 = n - j2; if (seg2 > p->page_tokens) seg2 = p->page_tokens;
                seg += seg2; j2 += seg2; lp2++;
            }
        }
        mlx_array ks_seg = mlx_array_new(), kq_seg = mlx_array_new();
        mlx_array vq_seg = mlx_array_new(), sc_seg = mlx_array_new(), vsc_seg = mlx_array_new();
        int a0[3] = {0,j,0}, a1[3] = {p->kv_heads, j+seg, p->head_dim}, ast[3] = {1,1,1};
        int c0[3] = {0,ptok,0}, c1[3] = {p->kv_heads, ptok+seg, p->head_dim}, cst[3] = {1,1,1};
        int d0[3] = {0,j,0}, d1[3] = {p->kv_heads, j+seg, 1};
        ok = mlx_slice(&kq_seg, kq, a0,3,a1,3,ast,3, s) == 0 &&
             mlx_slice(&vq_seg, vqt, a0,3,a1,3,ast,3, s) == 0 &&
             mlx_slice(&sc_seg, sc2, d0,3,d1,3,ast,3, s) == 0 &&
             mlx_slice(&vsc_seg, vsc2, d0,3,d1,3,ast,3, s) == 0;
        if (ok) {
            mlx_array nk = mlx_array_new(), nv = mlx_array_new(), nks = mlx_array_new(), nvs = mlx_array_new();
            ok = mlx_slice_update(&nk, p->keyData[layer], kq_seg, c0,3,c1,3,cst,3, s) == 0 &&
                 mlx_slice_update(&nv, p->valueData[layer], vq_seg, c0,3,c1,3,cst,3, s) == 0 &&
                 mlx_slice_update(&nks, p->keyScales[layer], sc_seg, c0,3,c1,3,cst,3, s) == 0 &&
                 mlx_slice_update(&nvs, p->valueScales[layer], vsc_seg, c0,3,c1,3,cst,3, s) == 0;
            if (ok) { mlx_array_set(&p->keyData[layer], nk); mlx_array_set(&p->valueData[layer], nv);
                      mlx_array_set(&p->keyScales[layer], nks); mlx_array_set(&p->valueScales[layer], nvs); }
            mlx_array_free(nk); mlx_array_free(nv); mlx_array_free(nks); mlx_array_free(nvs);
        }
        mlx_array_free(kq_seg); mlx_array_free(vq_seg); mlx_array_free(sc_seg); mlx_array_free(vsc_seg);
        j += seg;
    }
    if (ok) p->committed_l[layer] = pos + n;
    mlx_array_free(sc2); mlx_array_free(vsc2); mlx_array_free(vqt);
    mlx_array_free(kq); mlx_array_free(vq); mlx_array_free(ks); mlx_array_free(vs);
    if (ok) ok = llm_kvq_pool_export_ptab(p, s);
    return ok;
}

/* ── Spill/fault di UNA pagina logica (TUTTI i layer) <-> tier SSD (riuso kvdisk) ──
 * La page table (e quindi la pagina fisica) e' CONDIVISA fra i layer: una pagina logica
 * corrisponde a un blob che contiene la pagina di OGNI layer (K/V/scale). Spill rilascia la
 * pagina fisica e la marca non-residente; fault-in la ripristina in una pagina fisica nuova. */
#include "llm_mlx_kvdisk_io.h"
static inline int llm_kvq_pool_spill_page(llm_kvq_pool* p, int lp, KvDisk* d, uint64_t key, mlx_stream s) {
    if (!p || !d) return -1;
    int phys = llm_kv_pt_phys(&p->pt, lp);
    if (phys < 0) return -1;
    int PT = p->page_tokens, nkv = p->kv_heads, hd = p->head_dim, t0 = phys * PT;
    int nl = 0; for (int l = 0; l < p->layers; l++) if (p->keyData[l].ctx) nl++;
    uint8_t* buf = NULL; size_t len = 0, cap = 0;
    uint32_t magic = LLM_KVDISK_BLOB_MAGIC, ver = LLM_KVDISK_BLOB_VER, nt = (uint32_t)(3*nl);
    if (llm_kvdisk_buf_reserve(&buf, &cap, 0, 12) != 0) return -1;
    memcpy(buf, &magic, 4); memcpy(buf+4, &ver, 4); memcpy(buf+8, &nt, 4); len = 12;
    bool ok = true;
    for (int l = 0; l < p->layers && ok; l++) {
        if (!p->keyData[l].ctx) continue;
        mlx_array K = mlx_array_new(), V = mlx_array_new(), S = mlx_array_new();
        int a0[3] = {0,t0,0}, a1[3] = {nkv,t0+PT,hd}, st[3] = {1,1,1};
        int s0[3] = {0,t0,0}, s1[3] = {nkv,t0+PT,1};
        ok = mlx_slice(&K, p->keyData[l], a0,3,a1,3,st,3, s) == 0 &&
             mlx_slice(&V, p->valueData[l], a0,3,a1,3,st,3, s) == 0 &&
             mlx_slice(&S, p->keyScales[l], s0,3,s1,3,st,3, s) == 0;
        mlx_array Kc = mlx_array_new(), Vc = mlx_array_new(), Sc = mlx_array_new();
        if (ok) ok = mlx_contiguous(&Kc, K, false, s) == 0 &&
                     mlx_contiguous(&Vc, V, false, s) == 0 &&
                     mlx_contiguous(&Sc, S, false, s) == 0;
        if (ok) ok = llm_kvdisk_blob_append_tensor(&buf, &len, &cap, Kc, l, 0) == 0 &&
             llm_kvdisk_blob_append_tensor(&buf, &len, &cap, Vc, l, 1) == 0 &&
             llm_kvdisk_blob_append_tensor(&buf, &len, &cap, Sc, l, 2) == 0;
        mlx_array_free(Kc); mlx_array_free(Vc); mlx_array_free(Sc);
        mlx_array_free(K); mlx_array_free(V); mlx_array_free(S);
    }
    if (ok) ok = llm_kv_disk_put(d, key, buf, len) == 0;
    free(buf);
    if (ok) { llm_kv_pt_release_phys(&p->pt, phys); p->pt.table[lp] = -1; ok = llm_kvq_pool_export_ptab(p, s); }
    return ok ? 0 : -1;
}

static inline int llm_kvq_pool_faultin_page(llm_kvq_pool* p, int lp, KvDisk* d, uint64_t key, mlx_stream s) {
    if (!p || !d) return -1;
    if (llm_kv_pt_phys(&p->pt, lp) >= 0) return 0;
    size_t need = 0;
    if (!llm_kv_disk_has(d, key, &need) || need < 12) return -1;
    uint8_t* buf = (uint8_t*)malloc(need); size_t got = 0;
    if (!buf || llm_kv_disk_get(d, key, buf, need, &got) != 0 || got < 12) { free(buf); return -1; }
    uint32_t magic = 0, ver = 0, nt = 0;
    memcpy(&magic, buf, 4); memcpy(&ver, buf+4, 4); memcpy(&nt, buf+8, 4);
    bool ok = (magic == LLM_KVDISK_BLOB_MAGIC && ver == LLM_KVDISK_BLOB_VER);
    int phys = ok ? llm_kv_pt_map(&p->pt, lp) : -1;
    if (phys < 0) ok = false;
    int PT = p->page_tokens, nkv = p->kv_heads, hd = p->head_dim, t0 = phys * PT;
    size_t off = 12;
    for (uint32_t i = 0; i < nt && ok; i++) {
        mlx_array a = mlx_array_new(); int l = -1, k = -1;
        ok = llm_kvdisk_blob_read_tensor(buf, got, &off, &a, &l, &k) == 0;
        if (ok && l >= 0 && l < p->layers && p->keyData[l].ctx) {
            int a0[3] = {0,t0,0}, a1[3] = {nkv,t0+PT,hd}, st[3] = {1,1,1};
            int s0[3] = {0,t0,0}, s1[3] = {nkv,t0+PT,1};
            mlx_array nb = mlx_array_new();
            if (k == 0) { ok = mlx_slice_update(&nb, p->keyData[l], a, a0,3,a1,3,st,3, s) == 0; if (ok) mlx_array_set(&p->keyData[l], nb); }
            else if (k == 1) { ok = mlx_slice_update(&nb, p->valueData[l], a, a0,3,a1,3,st,3, s) == 0; if (ok) mlx_array_set(&p->valueData[l], nb); }
            else if (k == 2) { ok = mlx_slice_update(&nb, p->keyScales[l], a, s0,3,s1,3,st,3, s) == 0; if (ok) mlx_array_set(&p->keyScales[l], nb); }
            mlx_array_free(nb);
        }
        mlx_array_free(a);
    }
    if (ok) ok = llm_kvq_pool_export_ptab(p, s);
    free(buf);
    return ok ? 0 : -1;
}

/* Spill/fault dell'INTERO cold (tutte le pagine logiche) — usato dalla policy B9 quando il
 * governor segnala pressione (sessione inattiva) e al resume. Chiave per pagina = key_base ^ lp. */
static inline int llm_kvq_pool_spill_all(llm_kvq_pool* p, KvDisk* d, uint64_t key_base, mlx_stream s) {
    if (!p || !d) return -1;
    for (int lp = 0; lp < p->pt.n_logical; lp++)
        if (llm_kv_pt_phys(&p->pt, lp) >= 0)
            if (llm_kvq_pool_spill_page(p, lp, d, key_base + (uint64_t)lp, s) != 0) return -1;
    return 0;
}
static inline int llm_kvq_pool_faultin_all(llm_kvq_pool* p, KvDisk* d, uint64_t key_base, mlx_stream s) {
    if (!p || !d) return -1;
    for (int lp = 0; lp < p->pt.n_logical; lp++)
        if (llm_kv_pt_phys(&p->pt, lp) < 0)
            if (llm_kvq_pool_faultin_page(p, lp, d, key_base + (uint64_t)lp, s) != 0) return -1;
    return 0;
}

static inline void llm_kvq_pool_free(llm_kvq_pool* p) {
    if (!p || !p->keyData) return;
    for (int l = 0; l < p->layers; l++) {
        mlx_array_free(p->keyData[l]); mlx_array_free(p->valueData[l]);
        mlx_array_free(p->keyScales[l]); mlx_array_free(p->valueScales[l]);
        if (p->bias) mlx_array_free(p->bias[l]);
    }
    free(p->keyData); free(p->keyScales); free(p->valueData); free(p->valueScales); free(p->bias);
    free(p->committed_l);
    if (p->ptab.ctx) mlx_array_free(p->ptab);
    llm_kv_pt_free(&p->pt);
    memset(p, 0, sizeof *p);
}

/* ── Registry: un pool per (modello, sessione), creato lazy. ───────────────── */
typedef struct { void* model; void* sess; llm_kvq_pool p; } llm_kvq_pool_entry;
static llm_kvq_pool_entry g_kvq_pools[8];
static int g_kvq_pools_n = 0;
static uint64_t g_kvq_key_seq = 1;

static inline llm_kvq_pool* llm_kvq_pool_for(void* m, void* sess,
                                             int layers, int nkv, int hd, llm_kvq_fmt fmt,
                                             int cap_tokens, KvDisk* disk, mlx_stream s) {
    for (int i = 0; i < g_kvq_pools_n; i++)
        if (g_kvq_pools[i].model == m && g_kvq_pools[i].sess == sess) return &g_kvq_pools[i].p;
    if (g_kvq_pools_n >= 8) return NULL;
    llm_kvq_pool_entry* e = &g_kvq_pools[g_kvq_pools_n];
    memset(e, 0, sizeof *e);
    e->model = m; e->sess = sess;
    if (!llm_kvq_pool_init(&e->p, layers, nkv, hd, fmt, cap_tokens, s)) return NULL;
    e->p.disk = disk; e->p.key_base = (g_kvq_key_seq++ << 40);
    g_kvq_pools_n++;
    return &e->p;
}

static inline void llm_kvq_pool_pressure_pool(llm_kvq_pool* p, mlx_stream s) {
    if (!p || !p->disk) return;
    uint64_t f = llm_memgov_system_free_bytes();
    if (!llm_memgov_should_spill(&p->gov, f)) return;
    for (int i = 0; i < g_kvq_pools_n; i++) {
        llm_kvq_pool* q = &g_kvq_pools[i].p;
        if (q == p || !q->disk || llm_kvq_pool_offloaded(q)) continue;
        llm_kvq_pool_spill_all(q, q->disk, q->key_base, s);
    }
}

#endif /* LLM_MLX_KVQ_POOL_H */
