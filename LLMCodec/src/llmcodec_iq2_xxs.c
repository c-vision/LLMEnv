/* llmcodec_iq2_xxs.c — encoder IQ2_XXS e IQ2_XS (riscrittura propria a partire dal FORMATO dei blocchi
 * e dagli algoritmi di ggml; nessun codice copiato). La "grid" dei valori + le tabelle kgrid
 * (indici) stanno in llmcodec_ggml_iq_grids.{c,h} (dati di formato generati). L'init costruisce, per
 * famiglia, la mappa codice->indice-grid e le liste di vicini (neighbours) usate dal quantizzatore. */
#include "llmcodec_iq2_xxs.h"
#include "llmcodec_fp16.h"
#include "llmcodec_ggml_iq_grids.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#define IQ2_QK_K 256
#define IQ2_KMAP_SIZE 43692
#define IQ2_GROUP_MAX_EPS 1e-15f

/* f32 -> f16 (round-to-nearest); llmcodec_ggml_quants.c ha un equivalente statico. */
static int llmcodec_iq2_nearest_int(float fval) {
    float v = fval + 12582912.0f; int i; memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}

/* dati per famiglia: 0=IQ2_XXS, 1=IQ2_XS, 2=IQ1_S/IQ1_M, 3=IQ2_S (come ggml). */
typedef struct { uint64_t* grid; int* map; uint16_t* neigh; int grid_size; int inited; } GgmlIq2Data;
static GgmlIq2Data s_iq2[4];

static int llmcodec_iq2_cmp(const void* l, const void* r) {
    const int* a = (const int*)l; const int* b = (const int*)r;
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0;
}

static void llmcodec_iq2_init_family(int fam) {
    if (s_iq2[fam].inited) return;
    static const int grid_sz[4] = { 256, 512, 2048, 1024 };
    static const uint16_t* kgrids[4] = { kgrid_2bit_256, kgrid_2bit_512, kgrid_1bit_2048, kgrid_2bit_1024 };
    static const int nwants[4] = { 2, 2, 3, 1 };
    const int grid_size = grid_sz[fam], nwant = nwants[fam], kmap_size = IQ2_KMAP_SIZE;
    const uint16_t* kgrid = kgrids[fam];

    uint64_t* grid = (uint64_t*)malloc((size_t)grid_size * 8);
    for (int k = 0; k < grid_size; ++k) {
        int8_t* pos = (int8_t*)(grid + k);
        for (int i = 0; i < 8; ++i) { int l = (kgrid[k] >> (2*i)) & 3; pos[i] = (int8_t)(2*l + 1); }
    }
    int* map = (int*)malloc((size_t)kmap_size * sizeof(int));
    for (int i = 0; i < kmap_size; ++i) map[i] = -1;
    for (int i = 0; i < grid_size; ++i) {
        uint64_t a = grid[i]; uint8_t* a8 = (uint8_t*)&a; uint16_t index = 0;
        for (int k = 0; k < 8; ++k) { uint16_t q = (uint16_t)((a8[k] - 1) / 2); index |= (uint16_t)(q << (2*k)); }
        map[index] = i;
    }
    int* n_per = (int*)malloc((size_t)kmap_size * sizeof(int));
    int* d2 = (int*)malloc((size_t)2 * grid_size * sizeof(int));
    int8_t pos[8];
    int ntot = 0, nnot = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (map[i] >= 0) { n_per[i] = 0; continue; }
        ++nnot;
        for (int k = 0; k < 8; ++k) { int l = (i >> (2*k)) & 3; pos[k] = (int8_t)(2*l + 1); }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t* pg = (const int8_t*)(grid + j);
            int dd = 0; for (int k = 0; k < 8; ++k) dd += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            d2[2*j] = dd; d2[2*j+1] = j;
        }
        qsort(d2, (size_t)grid_size, 2*sizeof(int), llmcodec_iq2_cmp);
        int n = 0, dd = d2[0], nh = 1;
        for (int j = 0; j < grid_size; ++j) { if (d2[2*j] > dd) { if (nh == nwant) break; dd = d2[2*j]; ++nh; } ++n; }
        n_per[i] = n; ntot += n;
    }
    uint16_t* neigh = (uint16_t*)malloc((size_t)(ntot + nnot) * sizeof(uint16_t));
    int* offs = (int*)malloc((size_t)kmap_size * sizeof(int));
    int c = 0;
    for (int i = 0; i < kmap_size; ++i) { if (map[i] >= 0) { offs[i] = -1; continue; } offs[i] = c; c += 1 + n_per[i]; }
    for (int i = 0; i < kmap_size; ++i) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 8; ++k) { int l = (i >> (2*k)) & 3; pos[k] = (int8_t)(2*l + 1); }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t* pg = (const int8_t*)(grid + j);
            int dd = 0; for (int k = 0; k < 8; ++k) dd += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            d2[2*j] = dd; d2[2*j+1] = j;
        }
        qsort(d2, (size_t)grid_size, 2*sizeof(int), llmcodec_iq2_cmp);
        int lc = offs[i]; map[i] = -(lc + 1);
        int dd = d2[0]; uint16_t* start = &neigh[lc++]; int n = 0, nh = 1;
        for (int j = 0; j < grid_size; ++j) { if (d2[2*j] > dd) { if (nh == nwant) break; dd = d2[2*j]; ++nh; } neigh[lc++] = (uint16_t)d2[2*j+1]; ++n; }
        *start = (uint16_t)n;
    }
    free(d2); free(offs); free(n_per);
    s_iq2[fam].grid = grid; s_iq2[fam].map = map; s_iq2[fam].neigh = neigh;
    s_iq2[fam].grid_size = grid_size; s_iq2[fam].inited = 1;
}

float llmcodec_make_qp_quants(int n, int nmax, const float* x, uint8_t* L, const float* quant_weights) {
    float mx = 0;
    for (int i = 0; i < n; ++i) mx = mx > x[i] ? mx : x[i];
    if (mx < IQ2_GROUP_MAX_EPS) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = nmax / mx;
    for (int i = 0; i < n; ++i) L[i] = (uint8_t)llmcodec_iq2_nearest_int(iscale * x[i]);
    float scale = 1/iscale, best_mse = 0;
    for (int i = 0; i < n; ++i) { float diff = x[i] - scale*L[i]; float w = quant_weights[i]; best_mse += w*diff*diff; }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float isc = (0.1f*is + nmax)/mx, sc = 1/isc, mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = llmcodec_iq2_nearest_int(isc*x[i]); l = l > nmax ? nmax : l;
            float diff = x[i] - sc*l; float w = quant_weights[i]; mse += w*diff*diff;
        }
        if (mse < best_mse) { best_mse = mse; iscale = isc; }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = llmcodec_iq2_nearest_int(iscale * x[i]); l = l > nmax ? nmax : l; L[i] = (uint8_t)l;
        float w = quant_weights[i]; sumlx += w*x[i]*l; suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int nch = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i], sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int nl = llmcodec_iq2_nearest_int(x[i] * sl2 / slx); nl = nl > nmax ? nmax : nl;
                if (nl != L[i]) {
                    slx += w*x[i]*nl; sl2 += w*nl*nl;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) { L[i] = (uint8_t)nl; sumlx = slx; suml2 = sl2; ++nch; }
                }
            }
        }
        if (!nch) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static int llmcodec_iq2_find_best_neighbour(const uint16_t* neigh, const uint64_t* grid,
        const float* xval, const float* weight, float scale, int8_t* L) {
    int nn = neigh[0]; float best = 3.4e38f; int gi = -1;
    for (int j = 1; j <= nn; ++j) {
        const int8_t* pg = (const int8_t*)(grid + neigh[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) { float q = pg[i]; float diff = scale*q - xval[i]; d2 += weight[i]*diff*diff; }
        if (d2 < best) { best = d2; gi = neigh[j]; }
    }
    if (gi < 0) gi = 0;
    { const int8_t* pg = (const int8_t*)(grid + gi); for (int i = 0; i < 8; ++i) L[i] = (int8_t)((pg[i] - 1)/2); }
    return gi;
}

/* ── IQ2_XXS (fam 0): 4 gruppi da 8 per 32-value, 16-bit code + 7-bit segno + 4-bit scala. ── */
void llmcodec_iq2_xxs_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                          int64_t quant_weights_len, int64_t row_len) {
    llmcodec_iq2_init_family(0);
    const int kMaxQ = 3;
    const int64_t nbl = n / IQ2_QK_K;
    const uint64_t* grid = s_iq2[0].grid; const int* map = s_iq2[0].map; const uint16_t* neigh = s_iq2[0].neigh;
    float* wbuf = NULL;
    if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; quant_weights_len=n; }
    const float* quant_weights = quant_weights_in ? quant_weights_in : wbuf;
    const int64_t nrows = row_len > 0 ? n/row_len : 1;
    const int64_t weight_sets = row_len > 0 ? quant_weights_len/row_len : 1;
    const int64_t rows_per_set = weight_sets > 0 ? nrows/weight_sets : 0;
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*66;
        float scales[8], weight[32], xval[32], waux[32];
        int8_t L[32], Laux[32];
        uint8_t block_signs[4];
        uint32_t q2[16];
        uint16_t dh0 = llmcodec_f16_from_f32(0.f); memcpy(blk, &dh0, 2);
        memset(q2, 0, 64);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            const float* xb = xbl + 32*ib;
            const int64_t off = IQ2_QK_K*ibl + 32*ib;
            const int64_t row = row_len > 0 ? off/row_len : 0;
            const int64_t col = row_len > 0 ? off%row_len : off;
            const int64_t set = rows_per_set > 0 ? row/rows_per_set : 0;
            const float* qw = quant_weights + set*row_len + col;
            for (int i=0;i<32;++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i=0;i<32;++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0; uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k+i] >= 0) xval[8*k+i] = xb[8*k+i];
                    else { xval[8*k+i] = -xb[8*k+i]; ++nflip; s |= (uint8_t)(1 << i); }
                }
                if (nflip % 2) {
                    int imin = 0; float mn = weight[8*k]*xb[8*k]*xb[8*k];
                    for (int i=1;i<8;++i) { float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i]; if (ax < mn) { mn = ax; imin = i; } }
                    xval[8*k+imin] = -xval[8*k+imin]; s ^= (uint8_t)(1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float mx = xval[0]; for (int i=1;i<32;++i) mx = xval[i] > mx ? xval[i] : mx;
            if (mx < IQ2_GROUP_MAX_EPS) { scales[ib] = 0; memset(L, 0, 32); continue; }
            float scale = llmcodec_make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
            float eff_max = scale*kMaxQ;
            if (eff_max <= 0) { scales[ib] = 0; memset(L, 0, 32); continue; }
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/eff_max, this_scale = 1/id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[8*k+i]-1)); Laux[8*k+i] = (int8_t)(l > kMaxQ-1 ? kMaxQ-1 : (l < 0 ? 0 : l)); }
                    uint16_t u = 0; for (int i=0;i<8;++i) u |= (uint16_t)(Laux[8*k+i] << 2*i);
                    int gi = map[u];
                    if (gi < 0) gi = llmcodec_iq2_find_best_neighbour(neigh - map[u] - 1, grid, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i=0;i<32;++i) { float w = weight[i]; float q = 2*Laux[i]+1; sumqx += w*xval[i]*q; sumq2 += w*q*q; }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { scale = sumqx/sumq2; best = scale*sumqx; memcpy(L, Laux, 32); }
            }
            if (scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i=0;i<8;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[8*k+i]-1)); l = l > kMaxQ-1 ? kMaxQ-1 : (l < 0 ? 0 : l); u |= (uint16_t)(l << 2*i); }
                    int gi = map[u];
                    if (gi < 0) gi = llmcodec_iq2_find_best_neighbour(neigh - map[u] - 1, grid, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    const int8_t* pg = (const int8_t*)(grid + gi);
                    for (int i=0;i<8;++i) L[8*k+i] = (int8_t)((pg[i] - 1)/2);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i=0;i<32;++i) { float w = weight[i]; float q = 2*L[i]+1; sumqx += w*xval[i]*q; sumq2 += w*q*q; }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) { scale = -scale; for (int k=0;k<4;++k) block_signs[k] = (uint8_t)((~block_signs[k]) & 127); }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0; for (int i=0;i<8;++i) u |= (uint16_t)(L[8*k+i] << 2*i);
                int gi = map[u]; if (gi < 0) gi = 0;
                q2[2*ib+0] |= ((uint32_t)gi << 8*k);
                q2[2*ib+1] |= ((uint32_t)block_signs[k] << 7*k);
            }
            scales[ib] = scale; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) { memset(blk + 2, 0, 64); continue; }
        float d = max_scale/31; uint16_t dhh = llmcodec_f16_from_f32(d); memcpy(blk, &dhh, 2);
        float id = 1/d;
        for (int ib = 0; ib < 8; ++ib) { int l = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l = l > 15 ? 15 : (l < 0 ? 0 : l); q2[2*ib+1] |= ((uint32_t)l << 28); }
        memcpy(blk + 2, q2, 64);
    }
    free(wbuf);
}

/* ── IQ2_XS (fam 1): 16 sub-blocchi da 16, 2 gruppi da 8; qs uint16 = grid(9 bit)|segno(7 bit). ── */
void llmcodec_iq2_xs_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                         int64_t quant_weights_len, int64_t row_len) {
    (void)row_len; (void)quant_weights_len;
    llmcodec_iq2_init_family(1);
    const int kMaxQ = 3;
    const int64_t nbl = n / IQ2_QK_K;
    const uint64_t* grid = s_iq2[1].grid; const int* map = s_iq2[1].map; const uint16_t* neigh = s_iq2[1].neigh;
    float* wbuf = NULL;
    if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; }
    const float* quant_weights = quant_weights_in ? quant_weights_in : wbuf;
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*74;   /* d(2) + qs[32*2=64] + scales[8] = 74 */
        float scales[16], weight[16], xval[16], waux[16];
        int8_t L[16], Laux[16];
        bool on_grid[2], on_grid_aux[2];
        uint8_t block_signs[2];
        uint16_t q2[32];
        uint16_t d0 = llmcodec_f16_from_f32(0.f); memcpy(blk, &d0, 2);
        memset(q2, 0, 64); memset(blk + 66, 0, 8);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 16; ++ib) {
            const float* xb = xbl + 16*ib;
            const float* qw = quant_weights + IQ2_QK_K*ibl + 16*ib;
            for (int i=0;i<16;++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i=0;i<16;++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                int nflip = 0; uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k+i] >= 0) xval[8*k+i] = xb[8*k+i];
                    else { xval[8*k+i] = -xb[8*k+i]; ++nflip; s |= (uint8_t)(1 << i); }
                }
                if (nflip % 2) {
                    int imin = 0; float mn = weight[8*k]*xb[8*k]*xb[8*k];
                    for (int i=1;i<8;++i) { float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i]; if (ax < mn) { mn = ax; imin = i; } }
                    xval[8*k+imin] = -xval[8*k+imin]; s ^= (uint8_t)(1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float mx = xval[0]; for (int i=1;i<16;++i) mx = mx > xval[i] ? mx : xval[i];
            memset(L, 0, 16);
            if (mx < IQ2_GROUP_MAX_EPS) { scales[ib] = 0; continue; }
            float best = 0;
            float scale = mx/(2*kMaxQ-1);
            on_grid[0] = on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/mx, this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i=0;i<8;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[8*k+i]-1)); Laux[8*k+i] = (int8_t)(l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l)); }
                    uint16_t u = 0; for (int i=0;i<8;++i) u |= (uint16_t)(Laux[8*k+i] << 2*i);
                    int gi = map[u]; on_grid_aux[k] = true;
                    if (gi < 0) { on_grid_aux[k] = false; gi = llmcodec_iq2_find_best_neighbour(neigh - map[u] - 1, grid, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k); }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i=0;i<16;++i) { float w = weight[i]; float q = 2*Laux[i]+1; sumqx += w*xval[i]*q; sumq2 += w*q*q; }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i=0;i<16;++i) L[i] = Laux[i];
                    for (int k=0;k<2;++k) on_grid[k] = on_grid_aux[k];
                }
            }
            int not_og = 0; for (int k=0;k<2;++k) if (!on_grid[k]) ++not_og;
            if (not_og > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i=0;i<8;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[8*k+i]-1)); l = l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l); u |= (uint16_t)(l << 2*i); L[8*k+i] = (int8_t)l; }
                    int gi = map[u];
                    if (gi < 0) gi = llmcodec_iq2_find_best_neighbour(neigh - map[u] - 1, grid, xval + 8*k, waux + 8*k, scale, L + 8*k);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i=0;i<16;++i) { float w = weight[i]; float q = 2*L[i]+1; sumqx += w*xval[i]*q; sumq2 += w*q*q; }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) { scale = -scale; for (int k=0;k<2;++k) block_signs[k] = (uint8_t)((~block_signs[k]) & 127); }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0; for (int i=0;i<8;++i) u |= (uint16_t)(L[8*k+i] << 2*i);
                int gi = map[u]; if (gi < 0) gi = 0;
                q2[2*ib+k] = (uint16_t)(gi | (block_signs[k] << 9));
            }
            scales[ib] = scale; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) { memset(blk + 2, 0, 64); continue; }
        float d = max_scale/31; uint16_t dhh = llmcodec_f16_from_f32(d); memcpy(blk, &dhh, 2);
        float id = 1/d;
        for (int ib = 0; ib < 16; ++ib) {
            int l = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l = l < 0 ? 0 : (l > 15 ? 15 : l);
            if (ib % 2 == 0) blk[66 + ib/2] = (uint8_t)l;
            else blk[66 + ib/2] |= (uint8_t)(l << 4);
        }
        memcpy(blk + 2, q2, 64);
    }
    free(wbuf);
}

/* ── IQ2_S (fam 3): grid 1024 (indice 10-bit), 16 sub-blocchi da 16; qh per i bit alti, signs in qs. ── */
void llmcodec_iq2_s_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                        int64_t quant_weights_len, int64_t row_len) {
    (void)row_len; (void)quant_weights_len;
    llmcodec_iq2_init_family(3);
    const int kMaxQ = 3;
    const int64_t nbl = n / IQ2_QK_K;
    const uint64_t* grid = s_iq2[3].grid; const int* map = s_iq2[3].map; const uint16_t* neigh = s_iq2[3].neigh;
    float* wbuf = NULL;
    if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; }
    const float* quant_weights = quant_weights_in ? quant_weights_in : wbuf;
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*82;
        float scales[16], weight[16], xval[16], waux[16];
        int8_t L[16], Laux[16];
        bool on_grid[2], on_grid_aux[2];
        uint8_t block_signs[2];
        memset(blk, 0, 82);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 16; ++ib) {
            const float* xb = xbl + 16*ib;
            if (quant_weights) { const float* qw = quant_weights + IQ2_QK_K*ibl + 16*ib; for (int i=0;i<16;++i) weight[i] = qw[i]*sqrtf(sigma2 + xb[i]*xb[i]); }
            else { for (int i=0;i<16;++i) weight[i] = 0.25f*sigma2 + xb[i]*xb[i]; }
            for (int i=0;i<16;++i) waux[i] = sqrtf(weight[i]);
            for (int k=0;k<2;++k) {
                uint8_t s = 0;
                for (int i=0;i<8;++i) { if (xb[8*k+i]>=0) xval[8*k+i]=xb[8*k+i]; else { xval[8*k+i]=-xb[8*k+i]; s |= (uint8_t)(1<<i); } }
                block_signs[k] = s;
            }
            float mx = xval[0]; for (int i=1;i<16;++i) mx = mx > xval[i] ? mx : xval[i];
            memset(L, 0, 16);
            if (mx < 1e-8f) { scales[ib] = 0; continue; }
            float best = 0, scale = mx/(2*kMaxQ-1);
            on_grid[0] = on_grid[1] = true;
            for (int is=-9; is<=9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/mx, this_scale = 1/id;
                for (int k=0;k<2;++k) {
                    for (int i=0;i<8;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[8*k+i]-1)); Laux[8*k+i] = (int8_t)(l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l)); }
                    uint16_t u = 0; for (int i=0;i<8;++i) u |= (uint16_t)(Laux[8*k+i] << (2*i));
                    int gi = map[u]; on_grid_aux[k] = true;
                    if (gi < 0) { on_grid_aux[k] = false; gi = llmcodec_iq2_find_best_neighbour(neigh - map[u] - 1, grid, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k); }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i=0;i<16;++i) { float w = weight[i]; float q = 2*Laux[i]+1; sumqx += w*xval[i]*q; sumq2 += w*q*q; }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i=0;i<16;++i) L[i] = Laux[i];
                    for (int k=0;k<2;++k) on_grid[k] = on_grid_aux[k];
                }
            }
            int not_og = 0; for (int k=0;k<2;++k) if (!on_grid[k]) ++not_og;
            if (not_og > 0 && scale > 0) {
                float id = 1/scale;
                for (int k=0;k<2;++k) {
                    if (on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i=0;i<8;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[8*k+i]-1)); l = l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l); u |= (uint16_t)(l << (2*i)); L[8*k+i] = (int8_t)l; }
                    int gi = map[u];
                    if (gi < 0) gi = llmcodec_iq2_find_best_neighbour(neigh - map[u] - 1, grid, xval + 8*k, waux + 8*k, scale, L + 8*k);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i=0;i<16;++i) { float w = weight[i]; float q = 2*L[i]+1; sumqx += w*xval[i]*q; sumq2 += w*q*q; }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) { scale = -scale; for (int k=0;k<2;++k) block_signs[k] = (uint8_t)(~block_signs[k]); }
            for (int k=0;k<2;++k) {
                uint16_t u = 0; for (int i=0;i<8;++i) u |= (uint16_t)(L[8*k+i] << (2*i));
                int gi = map[u]; if (gi < 0) gi = 0;
                const int i8 = 2*ib + k;
                blk[2 + i8] = (uint8_t)(gi & 255);
                blk[2 + 64 + (i8/4)] |= (uint8_t)((gi >> 8) << (2*(i8%4)));
                blk[2 + IQ2_QK_K/8 + i8] = block_signs[k];
            }
            scales[ib] = scale; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) continue;
        float d = max_scale/31; uint16_t dh = llmcodec_f16_from_f32(d * 0.9875f); memcpy(blk, &dh, 2);
        float id = 1/d;
        for (int ib=0; ib<16; ++ib) { int l = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l = l < 0 ? 0 : (l > 15 ? 15 : l); if (ib%2==0) blk[74 + ib/2] = (uint8_t)l; else blk[74 + ib/2] |= (uint8_t)(l << 4); }
    }
    free(wbuf);
}

/* ── IQ1_S (fam 2): ternario -1/0/1, minimizzazione SSD esaustiva + sort + split search. ── */
static int llmcodec_iq1_sort_helper(const void* l, const void* r) { const float* a=l; const float* b=r; return *a<*b?-1:*a>*b?1:0; }
static int llmcodec_iq1_find_best_neighbour2(const uint16_t* neigh, const uint64_t* grid, const float* xval,
        const float* weight, float scale, const float* xg, int8_t* L, int ngrid) {
    (void)ngrid;
    int nn = neigh[0]; float best = 3.4e38f; int gi = -1;
    for (int j = 1; j <= nn; ++j) {
        const int8_t* pg = (const int8_t*)(grid + neigh[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) { float q = xg[(pg[i] - 1)/2]; float w = weight[i]; float diff = scale*q - xval[i]; d2 += w*diff*diff; }
        if (d2 < best) { best = d2; gi = neigh[j]; }
    }
    if (gi >= 0) { const int8_t* pg = (const int8_t*)(grid + gi); for (int i = 0; i < 8; ++i) L[i] = pg[i]; }
    return gi;
}
void llmcodec_iq1_s_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                        int64_t quant_weights_len, int64_t row_len) {
    (void)row_len; (void)quant_weights_len;
    llmcodec_iq2_init_family(2);
    const int64_t nbl = n / IQ2_QK_K;
    const uint64_t* grid = s_iq2[2].grid; const int* map = s_iq2[2].map; const uint16_t* neigh = s_iq2[2].neigh;
    float* wbuf = NULL;
    if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; }
    const float* qw0 = quant_weights_in ? quant_weights_in : wbuf;
    const float x_p[3] = { -1.0f + 0.125f, 0.125f, 1.0f + 0.125f };
    const float x_m[3] = { -1.0f - 0.125f, -0.125f, 1.0f - 0.125f };
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*50;   /* d(2) + qs[32] + qh[8 uint16=16] */
        float scales[8], weight[32], sumx[33], sumw[33], pairs[64];
        int8_t L[32], shifts[8];
        uint16_t index[4];
        memset(blk, 0, 50);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            const float* xb = xbl + 32*ib;
            const float* qw = qw0 + IQ2_QK_K*ibl + 32*ib;
            for (int i=0;i<32;++i) weight[i] = qw[i]*sqrtf(sigma2 + xb[i]*xb[i]);
            float mx = fabsf(xb[0]); for (int i=1;i<32;++i) { float a = fabsf(xb[i]); if (a > mx) mx = a; }
            if (mx < 1e-12f) { scales[ib] = 0; shifts[ib] = 1; memset(L, 1, 32); continue; }
            for (int j=0;j<32;++j) { pairs[2*j] = xb[j]; ((int*)pairs)[2*j+1] = j; }
            qsort(pairs, 32, 2*sizeof(float), llmcodec_iq1_sort_helper);
            int* idx = (int*)(pairs + 1);
            sumx[0] = sumw[0] = 0;
            for (int j=0;j<32;++j) { int i = idx[2*j]; sumx[j+1] = sumx[j] + weight[i]*xb[i]; sumw[j+1] = sumw[j] + weight[i]; }
            float best_score = -3.4e38f, scale = mx; int besti1 = -1, besti2 = -1, best_shift = 0;
            for (int i1=0;i1<=32;++i1) for (int i2=i1;i2<=32;++i2) {
                float sx = (sumx[i1]-sumx[0])*x_p[0] + (sumx[i2]-sumx[i1])*x_p[1] + (sumx[32]-sumx[i2])*x_p[2];
                float s2 = (sumw[i1]-sumw[0])*x_p[0]*x_p[0] + (sumw[i2]-sumw[i1])*x_p[1]*x_p[1] + (sumw[32]-sumw[i2])*x_p[2]*x_p[2];
                if (s2 > 0 && sx*sx > best_score*s2) { scale = sx/s2; best_score = scale*sx; besti1 = i1; besti2 = i2; best_shift = 1; }
                sx = (sumx[i1]-sumx[0])*x_m[0] + (sumx[i2]-sumx[i1])*x_m[1] + (sumx[32]-sumx[i2])*x_m[2];
                s2 = (sumw[i1]-sumw[0])*x_m[0]*x_m[0] + (sumw[i2]-sumw[i1])*x_m[1]*x_m[1] + (sumw[32]-sumw[i2])*x_m[2]*x_m[2];
                if (s2 > 0 && sx*sx > best_score*s2) { scale = sx/s2; best_score = scale*sx; besti1 = i1; besti2 = i2; best_shift = -1; }
            }
            if (besti1 < 0 || besti2 < 0 || best_shift == 0) { scales[ib] = 0; shifts[ib] = 1; memset(L, 1, 32); continue; }
            for (int j=0;j<besti1;++j) L[idx[2*j]] = 0;
            for (int j=besti1;j<besti2;++j) L[idx[2*j]] = 1;
            for (int j=besti2;j<32;++j) L[idx[2*j]] = 2;
            if (scale < 0) { for (int j=0;j<32;++j) L[j] = (int8_t)(2 - L[j]); scale = -scale; best_shift = -best_shift; }
            bool all_on_grid = true; const float* xx = best_shift == 1 ? x_p : x_m;
            for (int k=0;k<4;++k) {
                uint16_t u = 0; for (int j=0;j<8;++j) u |= (uint16_t)(L[8*k+j] << (2*j));
                int gi = map[u];
                if (gi < 0) { all_on_grid = false; gi = llmcodec_iq1_find_best_neighbour2(neigh - map[u] - 1, grid, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, 2048); }
                index[k] = (uint16_t)gi;
            }
            if (!all_on_grid) {
                float sx = 0, s2 = 0;
                for (int k=0;k<4;++k) { const int8_t* pg = (const int8_t*)(grid + index[k]);
                    for (int j=0;j<8;++j) { float w = weight[8*k+j]; float q = xx[(pg[j]-1)/2]; sx += w*q*xb[8*k+j]; s2 += w*q*q; } }
                if (sx > 0 && s2 > 0) scale = sx/s2;
            }
            uint16_t h = 0;
            for (int k=0;k<4;++k) { blk[2 + 4*ib + k] = (uint8_t)(index[k] & 255); h |= (uint16_t)((index[k] >> 8) << (3*k)); }
            blk[34 + 2*ib] = (uint8_t)(h & 255); blk[34 + 2*ib + 1] = (uint8_t)(h >> 8);
            scales[ib] = scale; shifts[ib] = (int8_t)best_shift; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) continue;
        float d = max_scale/15; uint16_t dh = llmcodec_f16_from_f32(d * 1.125f); memcpy(blk, &dh, 2);
        float id = 1/d;
        for (int ib=0; ib<8; ++ib) {
            int l = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l = l < 0 ? 0 : (l > 7 ? 7 : l);
            if (shifts[ib] == -1) l |= 8;
            uint16_t h = (uint16_t)(blk[34 + 2*ib] | (blk[34 + 2*ib + 1] << 8));
            h |= (uint16_t)(l << 12);
            blk[34 + 2*ib] = (uint8_t)(h & 255); blk[34 + 2*ib + 1] = (uint8_t)(h >> 8);
        }
    }
    free(wbuf);
}

/* accumulo per IQ1_M: 4 combinazioni di segno (maschere per le due meta'). */
static void llmcodec_iq1m_acc(float* sumqx, float* sumq2, float w, float xb, float xp, float xm, int i, int bs) {
    if (i < bs/2) {
        sumqx[0] += w*xp*xb; sumqx[1] += w*xp*xb; sumqx[2] += w*xm*xb; sumqx[3] += w*xm*xb;
        sumq2[0] += w*xp*xp; sumq2[1] += w*xp*xp; sumq2[2] += w*xm*xm; sumq2[3] += w*xm*xm;
    } else {
        sumqx[0] += w*xp*xb; sumqx[2] += w*xp*xb; sumqx[1] += w*xm*xb; sumqx[3] += w*xm*xb;
        sumq2[0] += w*xp*xp; sumq2[2] += w*xp*xp; sumq2[1] += w*xm*xm; sumq2[3] += w*xm*xm;
    }
}
/* ── IQ1_M (fam 2): ternario, block_size=16, 4 combinazioni di segno, 3-bit scales. ── */
void llmcodec_iq1_m_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                        int64_t quant_weights_len, int64_t row_len) {
    (void)row_len; (void)quant_weights_len;
    llmcodec_iq2_init_family(2);
    const int bs = 16;
    const int64_t nbl = n / IQ2_QK_K;
    const uint64_t* grid = s_iq2[2].grid; const int* map = s_iq2[2].map; const uint16_t* neigh = s_iq2[2].neigh;
    float* wbuf = NULL; if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; }
    const float* qw0 = quant_weights_in ? quant_weights_in : wbuf;
    const float x_p[3] = { -1.0f + 0.125f, 0.125f, 1.0f + 0.125f };
    const float x_m[3] = { -1.0f - 0.125f, -0.125f, 1.0f - 0.125f };
    const uint8_t masks[4] = { 0x00, 0x80, 0x08, 0x88 };
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*56;   /* qs[32] + qh[16] + scales[8] */
        float scales[16], weight[16], pairs[32], sumqx[4], sumq2[4];
        int8_t L[16], shifts[16];
        uint16_t index[2];
        memset(blk, 0, 56);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 16; ++ib) {
            const float* xb = xbl + bs*ib;
            const float* qw = qw0 + IQ2_QK_K*ibl + bs*ib;
            for (int i=0;i<bs;++i) weight[i] = qw[i]*sqrtf(sigma2 + xb[i]*xb[i]);
            float mx = fabsf(xb[0]); for (int i=1;i<bs;++i) { float a = fabsf(xb[i]); if (a > mx) mx = a; }
            if (mx < 1e-7f) { scales[ib] = 0; shifts[ib] = 0; memset(L, 1, bs); continue; }
            for (int j=0;j<bs;++j) { pairs[2*j] = xb[j]; ((int*)pairs)[2*j+1] = j; }
            qsort(pairs, bs, 2*sizeof(float), llmcodec_iq1_sort_helper);
            int* idx = (int*)(pairs + 1);
            float best_score = -3.4e38f, scale = mx; int besti1 = -1, besti2 = -1, best_k = -1;
            for (int i1=0;i1<=bs;++i1) for (int i2=i1;i2<=bs;++i2) {
                for (int kk=0;kk<4;++kk){ sumqx[kk]=0; sumq2[kk]=0; }
                for (int j=0;j<i1;++j){ int i=idx[2*j]; llmcodec_iq1m_acc(sumqx,sumq2,weight[i],xb[i],x_p[0],x_m[0],i,bs); }
                for (int j=i1;j<i2;++j){ int i=idx[2*j]; llmcodec_iq1m_acc(sumqx,sumq2,weight[i],xb[i],x_p[1],x_m[1],i,bs); }
                for (int j=i2;j<bs;++j){ int i=idx[2*j]; llmcodec_iq1m_acc(sumqx,sumq2,weight[i],xb[i],x_p[2],x_m[2],i,bs); }
                for (int kk=0;kk<4;++kk) if (sumq2[kk] > 0 && sumqx[kk]*sumqx[kk] > best_score*sumq2[kk]) { scale = sumqx[kk]/sumq2[kk]; best_score = scale*sumqx[kk]; besti1 = i1; besti2 = i2; best_k = kk; }
            }
            if (besti1 < 0 || besti2 < 0 || best_k < 0) { scales[ib] = 0; shifts[ib] = 0; memset(L, 1, bs); continue; }
            for (int j=0;j<besti1;++j) L[idx[2*j]] = 0;
            for (int j=besti1;j<besti2;++j) L[idx[2*j]] = 1;
            for (int j=besti2;j<bs;++j) L[idx[2*j]] = 2;
            if (scale < 0) { for (int j=0;j<bs;++j) L[j] = (int8_t)(2 - L[j]); scale = -scale; best_k = best_k==0?3:best_k==1?2:best_k==2?1:0; }
            bool all = true; const float* xx;
            for (int k=0;k<2;++k) {
                xx = (k == 0) ? (best_k < 2 ? x_p : x_m) : (best_k % 2 == 0 ? x_p : x_m);
                uint16_t u = 0; for (int j=0;j<8;++j) u |= (uint16_t)(L[8*k+j] << (2*j));
                int gi = map[u];
                if (gi < 0) { all = false; gi = llmcodec_iq1_find_best_neighbour2(neigh - map[u] - 1, grid, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, 2048); }
                index[k] = (uint16_t)gi;
            }
            if (!all) {
                float sx = 0, s2 = 0;
                for (int k=0;k<2;++k) {
                    xx = (k == 0) ? (best_k < 2 ? x_p : x_m) : (best_k % 2 == 0 ? x_p : x_m);
                    const int8_t* pg = (const int8_t*)(grid + index[k]);
                    for (int j=0;j<8;++j) { float w = weight[8*k+j]; float q = xx[(pg[j]-1)/2]; sx += w*q*xb[8*k+j]; s2 += w*q*q; }
                }
                if (sx > 0 && s2 > 0) scale = sx/s2;
            }
            blk[2*ib + 0] = (uint8_t)(index[0] & 255);
            blk[2*ib + 1] = (uint8_t)(index[1] & 255);
            blk[32 + ib] = (uint8_t)((index[0] >> 8) | ((index[1] >> 8) << 4));
            scales[ib] = scale; shifts[ib] = (int8_t)best_k; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) continue;
        uint16_t* sc = (uint16_t*)(blk + 48);
        float d = max_scale/15; float id = 1/d, sqx = 0, sq2 = 0;
        for (int ib=0; ib<16; ++ib) {
            int l = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l = l < 0 ? 0 : (l > 7 ? 7 : l);
            sc[ib/4] |= (uint16_t)(l << (3*(ib%4)));
            blk[32 + ib] |= masks[shifts[ib]];
            const float* xb = xbl + bs*ib;
            const float* qw = qw0 + IQ2_QK_K*ibl + bs*ib;
            for (int i=0;i<bs;++i) weight[i] = qw[i]*sqrtf(sigma2 + xb[i]*xb[i]);
            for (int k=0;k<2;++k) {
                const float* xx = (k == 0) ? (shifts[ib] < 2 ? x_p : x_m) : (shifts[ib] % 2 == 0 ? x_p : x_m);
                const int8_t* pg = (const int8_t*)(grid + blk[2*ib+k] + ((blk[32+ib] << (8 - 4*k)) & 0x700));
                for (int j=0;j<8;++j) { float w = weight[8*k+j]; float q = xx[(pg[j]-1)/2]*(2*l+1); sqx += w*q*xb[8*k+j]; sq2 += w*q*q; }
            }
        }
        if (sq2 > 0) d = sqx/sq2;
        uint16_t h = llmcodec_f16_from_f32(d * 1.1125f);
        sc[0] |= (uint16_t)((h & 0x000f) << 12);
        sc[1] |= (uint16_t)((h & 0x00f0) <<  8);
        sc[2] |= (uint16_t)((h & 0x0f00) <<  4);
        sc[3] |= (uint16_t)((h & 0xf000) <<  0);
    }
    free(wbuf);
}

/* ── IQ3 (XXS grid 256 / S grid 512): 4 valori per entry, code 12-bit (kmap 4096). ── */
typedef struct { uint32_t* grid; int* map; uint16_t* neigh; int grid_size; int inited; } GgmlIq3Data;
static GgmlIq3Data s_iq3[2];
static void llmcodec_iq3_init(int grid_size) {
    const int fam = grid_size == 256 ? 0 : 1;
    if (s_iq3[fam].inited) return;
    const int nwant = grid_size == 256 ? 2 : 3, kmap_size = 4096;
    const uint16_t* kgrid = grid_size == 256 ? kgrid_256 : kgrid_512;
    uint32_t* grid = (uint32_t*)malloc((size_t)grid_size*4);
    for (int k = 0; k < grid_size; ++k) {
        int8_t* pos = (int8_t*)(grid + k);
        for (int i = 0; i < 4; ++i) { int l = (kgrid[k] >> (3*i)) & 7; pos[i] = (int8_t)(2*l + 1); }
    }
    int* map = (int*)malloc((size_t)kmap_size*sizeof(int));
    for (int i = 0; i < kmap_size; ++i) map[i] = -1;
    for (int i = 0; i < grid_size; ++i) {
        uint32_t a = grid[i]; uint8_t* a8 = (uint8_t*)&a; uint16_t index = 0;
        for (int k = 0; k < 4; ++k) { uint16_t q = (uint16_t)((a8[k] - 1)/2); index |= (uint16_t)(q << (3*k)); }
        map[index] = i;
    }
    int* nper = (int*)malloc((size_t)kmap_size*sizeof(int));
    int* d2 = (int*)malloc((size_t)2*grid_size*sizeof(int));
    int8_t pos[4]; int ntot = 0, nnot = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (map[i] >= 0) { nper[i] = 0; continue; }
        ++nnot;
        for (int k = 0; k < 4; ++k) { int l = (i >> (3*k)) & 7; pos[k] = (int8_t)(2*l + 1); }
        for (int j = 0; j < grid_size; ++j) { const int8_t* pg = (const int8_t*)(grid + j); int dd = 0; for (int k = 0; k < 4; ++k) dd += (pg[k]-pos[k])*(pg[k]-pos[k]); d2[2*j] = dd; d2[2*j+1] = j; }
        qsort(d2, (size_t)grid_size, 2*sizeof(int), llmcodec_iq2_cmp);
        int n = 0, dd = d2[0], nh = 1;
        for (int j = 0; j < grid_size; ++j) { if (d2[2*j] > dd) { if (nh == nwant) break; dd = d2[2*j]; ++nh; } ++n; }
        nper[i] = n; ntot += n;
    }
    uint16_t* neigh = (uint16_t*)malloc((size_t)(ntot+nnot)*sizeof(uint16_t));
    int* offs = (int*)malloc((size_t)kmap_size*sizeof(int));
    int c = 0;
    for (int i = 0; i < kmap_size; ++i) { if (map[i] >= 0) { offs[i] = -1; continue; } offs[i] = c; c += 1 + nper[i]; }
    for (int i = 0; i < kmap_size; ++i) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 4; ++k) { int l = (i >> (3*k)) & 7; pos[k] = (int8_t)(2*l + 1); }
        for (int j = 0; j < grid_size; ++j) { const int8_t* pg = (const int8_t*)(grid + j); int dd = 0; for (int k = 0; k < 4; ++k) dd += (pg[k]-pos[k])*(pg[k]-pos[k]); d2[2*j] = dd; d2[2*j+1] = j; }
        qsort(d2, (size_t)grid_size, 2*sizeof(int), llmcodec_iq2_cmp);
        int lc = offs[i]; map[i] = -(lc + 1);
        int dd = d2[0]; uint16_t* start = &neigh[lc++]; int n = 0, nh = 1;
        for (int j = 0; j < grid_size; ++j) { if (d2[2*j] > dd) { if (nh == nwant) break; dd = d2[2*j]; ++nh; } neigh[lc++] = (uint16_t)d2[2*j+1]; ++n; }
        *start = (uint16_t)n;
    }
    free(d2); free(offs); free(nper);
    s_iq3[fam].grid = grid; s_iq3[fam].map = map; s_iq3[fam].neigh = neigh; s_iq3[fam].grid_size = grid_size; s_iq3[fam].inited = 1;
}
static int llmcodec_iq3_find_best_neighbour(const uint16_t* neigh, const uint32_t* grid, const float* xval,
        const float* weight, float scale, int8_t* L) {
    int nn = neigh[0]; float best = 3.4e38f; int gi = -1;
    for (int j = 1; j <= nn; ++j) {
        const int8_t* pg = (const int8_t*)(grid + neigh[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) { float q = pg[i]; float diff = scale*q - xval[i]; d2 += weight[i]*diff*diff; }
        if (d2 < best) { best = d2; gi = neigh[j]; }
    }
    if (gi < 0) gi = 0;
    { const int8_t* pg = (const int8_t*)(grid + gi); for (int i = 0; i < 4; ++i) L[i] = (int8_t)((pg[i]-1)/2); }
    return gi;
}
/* ── IQ3_XXS (grid 256): block = d(F16) + qs[96] (64 grid-index + 8 uint32 scales_and_signs). ── */
void llmcodec_iq3_xxs_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                          int64_t quant_weights_len, int64_t row_len) {
    (void)row_len; (void)quant_weights_len;
    llmcodec_iq3_init(256);
    const int kMaxQ = 8;
    const int64_t nbl = n / IQ2_QK_K;
    const uint32_t* grid = s_iq3[0].grid; const int* map = s_iq3[0].map; const uint16_t* neigh = s_iq3[0].neigh;
    float* wbuf = NULL; if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; }
    const float* qw0 = quant_weights_in ? quant_weights_in : wbuf;
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*98;
        uint8_t q3[104];
        float scales[8], weight[32], xval[32], waux[32];
        int8_t L[32], Laux[32];
        bool og[8], oga[8];
        uint8_t block_signs[8];
        uint32_t* ss = (uint32_t*)(q3 + 64);
        memset(blk, 0, 98); memset(q3, 0, 104);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            const float* xb = xbl + 32*ib;
            const float* qw = qw0 + IQ2_QK_K*ibl + 32*ib;
            for (int i=0;i<32;++i) weight[i] = qw[i]*sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i=0;i<32;++i) waux[i] = sqrtf(weight[i]);
            for (int k=0;k<4;++k) {
                int nflip = 0; uint8_t sg = 0;
                for (int i=0;i<8;++i) { if (xb[8*k+i] >= 0) xval[8*k+i] = xb[8*k+i]; else { xval[8*k+i] = -xb[8*k+i]; ++nflip; sg |= (uint8_t)(1<<i); } }
                if (nflip % 2) { int imin = 0; float mn = weight[8*k]*xb[8*k]*xb[8*k];
                    for (int i=1;i<8;++i) { float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i]; if (ax < mn) { mn = ax; imin = i; } }
                    xval[8*k+imin] = -xval[8*k+imin]; sg ^= (uint8_t)(1<<imin); }
                block_signs[k] = sg & 127;
            }
            float mx = xval[0]; for (int i=1;i<32;++i) mx = mx > xval[i] ? mx : xval[i];
            memset(L, 0, 32);
            if (mx < 1e-8f) { scales[ib] = 0; continue; }
            float best = 0, scale = mx/(2*kMaxQ-1);
            for (int k=0;k<8;++k) og[k] = true;
            for (int is=-15; is<=15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/mx, this_scale = 1/id;
                for (int k=0;k<8;++k) {
                    for (int i=0;i<4;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[4*k+i]-1)); Laux[4*k+i] = (int8_t)(l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l)); }
                    uint16_t u = 0; for (int i=0;i<4;++i) u |= (uint16_t)(Laux[4*k+i] << (3*i));
                    int gi = map[u]; oga[k] = true;
                    if (gi < 0) { oga[k] = false; gi = llmcodec_iq3_find_best_neighbour(neigh - map[u] - 1, grid, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k); }
                }
                float sqx = 0, sq2 = 0;
                for (int i=0;i<32;++i) { float w = weight[i]; float q = 2*Laux[i]+1; sqx += w*xval[i]*q; sq2 += w*q*q; }
                if (sq2 > 0 && sqx*sqx > best*sq2) { scale = sqx/sq2; best = scale*sqx; for (int i=0;i<32;++i) L[i]=Laux[i]; for (int k=0;k<8;++k) og[k]=oga[k]; }
            }
            int nog = 0; for (int k=0;k<8;++k) if (!og[k]) ++nog;
            if (nog > 0 && scale > 0) {
                float id = 1/scale;
                for (int k=0;k<8;++k) {
                    if (og[k]) continue;
                    uint16_t u = 0; for (int i=0;i<4;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[4*k+i]-1)); l = l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l); u |= (uint16_t)(l << (3*i)); }
                    int gi = map[u];
                    if (gi < 0) gi = llmcodec_iq3_find_best_neighbour(neigh - map[u] - 1, grid, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    const int8_t* pg = (const int8_t*)(grid + gi);
                    for (int i=0;i<4;++i) L[4*k+i] = (int8_t)((pg[i]-1)/2);
                }
                float sqx = 0, sq2 = 0;
                for (int i=0;i<32;++i) { float w = weight[i]; float q = 2*L[i]+1; sqx += w*xval[i]*q; sq2 += w*q*q; }
                if (sq2 > 0) scale = sqx/sq2;
            }
            if (scale < 0) { scale = -scale; for (int k=0;k<4;++k) block_signs[k] = (uint8_t)((~block_signs[k]) & 127); }
            for (int k=0;k<8;++k) { uint16_t u = 0; for (int i=0;i<4;++i) u |= (uint16_t)(L[4*k+i] << (3*i)); int gi = map[u]; if (gi < 0) gi = 0; q3[8*ib+k] = (uint8_t)gi; }
            ss[ib] = (uint32_t)(block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21));
            scales[ib] = scale; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) { memset(blk + 2, 0, 96); continue; }
        float d = max_scale/31; uint16_t dh = llmcodec_f16_from_f32(d * 1.0125f); memcpy(blk, &dh, 2);
        float id = 1/d;
        for (int ib=0; ib<8; ++ib) { int l = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l = l < 0 ? 0 : (l > 15 ? 15 : l); ss[ib] |= ((uint32_t)l << 28); }
        memcpy(blk + 2, q3, 96);
    }
    free(wbuf);
}

/* ── IQ3_S (grid 512): block = d(F16) + qs[64] + qh[8] + signs[32] + scales[4] = 110 byte. ── */
void llmcodec_iq3_s_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights_in,
                        int64_t quant_weights_len, int64_t row_len) {
    (void)row_len; (void)quant_weights_len;
    llmcodec_iq3_init(512);
    const int kMaxQ = 8, bs4 = 8, bs8 = 4;
    const int64_t nbl = n / IQ2_QK_K;
    const uint32_t* grid = s_iq3[1].grid; const int* map = s_iq3[1].map; const uint16_t* neigh = s_iq3[1].neigh;
    float* wbuf = NULL; if (quant_weights_in == NULL) { wbuf = (float*)malloc((size_t)n*sizeof(float)); for (int64_t i=0;i<n;++i) wbuf[i]=1.0f; }
    const float* qw0 = quant_weights_in ? quant_weights_in : wbuf;
    for (int64_t ibl = 0; ibl < nbl; ++ibl) {
        uint8_t* blk = out + ibl*110;
        uint8_t* qs = blk + 2; uint8_t* qh = blk + 66; uint8_t* signs = blk + 74; uint8_t* sc = blk + 106;
        float scales[8], weight[32], xval[32], waux[32];
        int8_t L[32], Laux[32];
        bool og[8], oga[8];
        uint8_t block_signs[4];
        memset(blk, 0, 110);
        float max_scale = 0;
        const float* xbl = x + IQ2_QK_K*ibl;
        float sumx2 = 0; for (int i=0;i<IQ2_QK_K;++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/IQ2_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            const float* xb = xbl + 32*ib;
            const float* qw = qw0 + IQ2_QK_K*ibl + 32*ib;
            for (int i=0;i<32;++i) weight[i] = qw[i]*sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i=0;i<32;++i) waux[i] = sqrtf(weight[i]);
            for (int k=0;k<bs8;++k) { uint8_t sg = 0; for (int i=0;i<8;++i) { if (xb[8*k+i] >= 0) xval[8*k+i] = xb[8*k+i]; else { xval[8*k+i] = -xb[8*k+i]; sg |= (uint8_t)(1<<i); } } block_signs[k] = sg; }
            float mx = xval[0]; for (int i=1;i<32;++i) mx = mx > xval[i] ? mx : xval[i];
            memset(L, 0, 32);
            if (!mx) { scales[ib] = 0; continue; }
            float best = 0, scale = mx/(2*kMaxQ-1);
            for (int k=0;k<bs4;++k) og[k] = false;
            for (int is=-9; is<=9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/mx, this_scale = 1/id;
                for (int k=0;k<bs4;++k) {
                    for (int i=0;i<4;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[4*k+i]-1)); Laux[4*k+i] = (int8_t)(l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l)); }
                    uint16_t u = 0; for (int i=0;i<4;++i) u |= (uint16_t)(Laux[4*k+i] << (3*i));
                    int gi = map[u]; oga[k] = true;
                    if (gi < 0) { oga[k] = false; gi = llmcodec_iq3_find_best_neighbour(neigh - map[u] - 1, grid, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k); }
                }
                float sqx = 0, sq2 = 0;
                for (int i=0;i<32;++i) { float w = weight[i]; float q = 2*Laux[i]+1; sqx += w*xval[i]*q; sq2 += w*q*q; }
                if (sq2 > 0 && sqx*sqx > best*sq2) { scale = sqx/sq2; best = scale*sqx; for (int i=0;i<32;++i) L[i]=Laux[i]; for (int k=0;k<bs4;++k) og[k]=oga[k]; }
            }
            int nog = 0; for (int k=0;k<bs4;++k) if (!og[k]) ++nog;
            if (nog > 0 && scale > 0) {
                float id = 1/scale;
                for (int k=0;k<bs4;++k) {
                    uint16_t u = 0; for (int i=0;i<4;++i) { int l = llmcodec_iq2_nearest_int(0.5f*(id*xval[4*k+i]-1)); l = l < 0 ? 0 : (l > kMaxQ-1 ? kMaxQ-1 : l); u |= (uint16_t)(l << (3*i)); }
                    int gi = map[u];
                    if (gi < 0) gi = llmcodec_iq3_find_best_neighbour(neigh - map[u] - 1, grid, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    const int8_t* pg = (const int8_t*)(grid + gi);
                    for (int i=0;i<4;++i) L[4*k+i] = (int8_t)((pg[i]-1)/2);
                }
                float sqx = 0, sq2 = 0;
                for (int i=0;i<32;++i) { float w = weight[i]; float q = 2*L[i]+1; sqx += w*xval[i]*q; sq2 += w*q*q; }
                if (sq2 > 0) scale = sqx/sq2;
            }
            if (scale < 0) { scale = -scale; for (int k=0;k<bs8;++k) block_signs[k] = (uint8_t)(~block_signs[k]); }
            for (int k=0;k<bs4;++k) {
                uint16_t u = 0; for (int i=0;i<4;++i) u |= (uint16_t)(L[4*k+i] << (3*i));
                int gi = map[u]; if (gi < 0) gi = 0;
                qs[k] = (uint8_t)(gi & 255);
                qh[(ib*bs4+k)/8] |= (uint8_t)((gi >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k=0;k<bs8;++k) signs[k] = block_signs[k];
            signs += bs8;
            scales[ib] = scale; max_scale = scale > max_scale ? scale : max_scale;
        }
        if (!max_scale) continue;
        float d = max_scale/31; uint16_t dh = llmcodec_f16_from_f32(d * 1.033f); memcpy(blk, &dh, 2);
        float id = 1/d;
        for (int ib=0; ib<8; ib+=2) {
            int l1 = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib]-1)); l1 = l1 < 0 ? 0 : (l1 > 15 ? 15 : l1);
            int l2 = llmcodec_iq2_nearest_int(0.5f*(id*scales[ib+1]-1)); l2 = l2 < 0 ? 0 : (l2 > 15 ? 15 : l2);
            sc[ib/2] = (uint8_t)(l1 | (l2 << 4));
        }
    }
    free(wbuf);
}
