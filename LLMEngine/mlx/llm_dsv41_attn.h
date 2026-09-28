/* llm_dsv41_attn.h — DeepSeek V4.1: CSA2 attention semantics (Level-3 port, step 3), header-only.
 *
 * Port C della SEMANTICA di `omlx/patches/deepseek_v41/language.py` + `kernels.py`:
 *   - SWA local window indices           (Attention: `local` / `valid` / `idx`)
 *   - Compressor con cache/remainder      (Compressor.__call__)
 *   - Indexer top-k                       (packed_index_topk: top-k + sort cronologico)
 *   - Sparse attention con attention-sink  (packed_sparse_attention)
 *
 * NON e' il forward completo (cache per-layer MLX, kernel fusi, MoE, Engram, vision, MTP):
 * v. `(open) - DeepSeek-V41-Port.md`. Queste sono le funzioni verificabili offline (nessun device).
 * Verifica: `llm_dsv41_attn_test.c` vs valori di riferimento MLX (`tools/gen_dsv41_math_reference.py`).
 *
 * Limiti dichiarati: gli scratch sono a capacita' fissa (indicata ai limiti delle funzioni) perche'
 * l'integrazione reale userebbe array MLX; qui servono solo alla verifica della semantica.
 */
#ifndef LLM_DSV41_ATTN_H
#define LLM_DSV41_ATTN_H

#include <math.h>
#include "llm_dsv41_math.h"

#define LLM_DSV41_ATTN_MAXLEN 4096   /* capacita' scratch [posizioni] */
#define LLM_DSV41_ATTN_MAXH    512   /* capacita' scratch [head] */

/* SWA local window indices (Attention.__call__).
 * idx: [length x W], W = (start==0 ? min(length,window) : window). -1 = posizione invalida.
 * Ritorna W (0 se input invalido). */
static inline int llm_dsv41_swa_indices(int start, int length, int window, int old_len, int* idx) {
    if (length <= 0 || window <= 0) return 0;
    int W = (start == 0) ? (length < window ? length : window) : window;
    int lo = start - old_len; if (lo < 0) lo = 0;
    for (int i = 0; i < length; i++) {
        int pos = start + i;
        if (start == 0) {
            int base = pos - window + 1; if (base < 0) base = 0;
            for (int j = 0; j < W; j++) {
                int local = base + j;
                idx[i * W + j] = (local >= 0 && local <= pos) ? local : -1;
            }
        } else {
            for (int j = 0; j < W; j++) {
                int local = pos - window + 1 + j;
                int valid = (local >= lo && local <= pos);
                idx[i * W + j] = valid ? (local - (start - old_len)) : -1;
            }
        }
    }
    return W;
}

/* Compressor: concat cache[:rem] + input, cutoff=r*floor(len/r), resto in cache, pooling sul tagliato.
 *  kv_in/gate_in: [L x H]; cache_kv/cache_gate: [cache_len x H] (rem = start % r);
 *  pooled: [m x H] (m = cutoff/r); new_cache_*: [cache_new x H]. Ritorna m. */
static inline int llm_dsv41_compressor_step(const float* kv_in, const float* gate_in, int L, int H,
                                            int r, int start,
                                            const float* cache_kv, const float* cache_gate, int cache_len,
                                            float* pooled, float* new_cache_kv, float* new_cache_gate,
                                            int* new_cache_len) {
    if (L < 0 || H <= 0 || r <= 0) return 0;
    int rem = start % r;
    int keep = (rem && cache_kv && cache_gate) ? (rem < cache_len ? rem : cache_len) : 0;
    if ((keep + L) * H > LLM_DSV41_ATTN_MAXLEN * LLM_DSV41_ATTN_MAXH) return 0; /* oltre capacity */
    static float kv[LLM_DSV41_ATTN_MAXLEN * LLM_DSV41_ATTN_MAXH];
    static float gate[LLM_DSV41_ATTN_MAXLEN * LLM_DSV41_ATTN_MAXH];
    int off = 0;
    if (keep) { for (int i = 0; i < keep * H; i++) { kv[i] = cache_kv[i]; gate[i] = cache_gate[i]; } off = keep * H; }
    for (int i = 0; i < L * H; i++) { kv[off + i] = kv_in[i]; gate[off + i] = gate_in[i]; }
    int total = keep + L;
    int cutoff = (total / r) * r;
    int cnew = total - cutoff;
    for (int i = 0; i < cnew * H; i++) { new_cache_kv[i] = kv[cutoff * H + i]; new_cache_gate[i] = gate[cutoff * H + i]; }
    *new_cache_len = cnew;
    int m = cutoff / r;
    llm_dsv41_compressor_pool(kv, gate, r, H, m, pooled);
    return m;
}

/* Indexer top-k (packed_index_topk): prende i k migliori score (desc, tie: indice minore), poi li
 * riordina in ordine cronologico crescente; slot invalidi (score == -inf) -> -1.
 * Ritorna il numero di indici validi scritti (<= k). idx_out capacita' >= k. */
static inline int llm_dsv41_index_topk(const float* scores, int n, int k, int* idx_out) {
    if (n <= 0 || k <= 0) return 0;
    if (k > 256) k = 256;
    int chosen[256], nch = 0;
    char used[LLM_DSV41_ATTN_MAXLEN]; for (int i = 0; i < n && i < LLM_DSV41_ATTN_MAXLEN; i++) used[i] = 0;
    for (int t = 0; t < k; t++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (used[i] || scores[i] == -INFINITY) continue;
            if (best < 0 || scores[i] > scores[best]) best = i;   /* tie: i minore (strict >) */
        }
        if (best < 0) break;
        used[best] = 1; chosen[nch++] = best;
    }
    /* ordina gli indici scelti in ordine crescente (cronologico) */
    for (int i = 1; i < nch; i++) { int x = chosen[i], j = i - 1; while (j >= 0 && chosen[j] > x) { chosen[j + 1] = chosen[j]; j--; } chosen[j + 1] = x; }
    for (int i = 0; i < k; i++) idx_out[i] = -1;
    for (int i = 0; i < nch; i++) idx_out[i] = chosen[i];
    return nch;
}

/* Candidate block pre-filtering (language.py candidate_block_ids), riga singola.
 *  scores[W]; lengths>0; raggruppa in blocchi di `block` (pad a -inf); forza a +inf il blocco che
 *  contiene l'ultimo token valido ((lengths-1)/block); prende i `count` blocchi migliori e li
 *  riordina crescenti; slot invalidi -> -1. Ritorna il numero di blocchi scritti (min(count,nblk)). */
static inline int llm_dsv41_candidate_block_ids(const float* scores, int W, int lengths,
                                                int count, int block, int* out) {
    if (W <= 0 || block <= 0 || count <= 0) return 0;
    int pad = (block - (W % block)) % block;
    int padded = W + pad;
    int nblk = padded / block;
    if (nblk > 256) nblk = 256;
    float grouped[256];
    for (int b = 0; b < nblk; b++) {
        float mx = -INFINITY;
        for (int k = 0; k < block; k++) {
            int idx = b * block + k;
            float v = (idx < W) ? scores[idx] : -INFINITY;
            if (v > mx) mx = v;
        }
        grouped[b] = mx;
    }
    if (lengths > 0) {
        int pos = (lengths - 1) / block;
        if (pos >= 0 && pos < nblk) grouped[pos] = INFINITY;
    }
    int nn = count < nblk ? count : nblk;
    int chosen[256], nch = 0;
    char used[256]; for (int b = 0; b < nblk; b++) used[b] = 0;
    for (int t = 0; t < nn; t++) {
        int best = -1;
        for (int b = 0; b < nblk; b++) {
            if (used[b]) continue;
            if (best < 0 || grouped[b] > grouped[best]) best = b;
        }
        used[best] = 1; chosen[nch++] = best;
    }
    for (int i = 1; i < nch; i++) { int x = chosen[i], j = i - 1; while (j >= 0 && chosen[j] > x) { chosen[j + 1] = chosen[j]; j--; } chosen[j + 1] = x; }
    for (int t = 0; t < nn; t++) out[t] = (grouped[chosen[t]] > -INFINITY) ? chosen[t] : -1;
    return nn;
}

/* Sparse attention con attention-sink (packed_sparse_attention).
 *  q: [heads x dim]; keys: [M x heads x dim]; indices[M] (>=0 valido); sink[heads]; scale.
 *  out: [heads x dim].  softmax su [score_m (validi), sink[head]]. */
static inline void llm_dsv41_sdpa_sink(const float* q, const float* keys, const int* indices, int M,
                                       int heads, int dim, const float* sink, float scale, float* out) {
    float sc[LLM_DSV41_ATTN_MAXLEN];
    for (int h = 0; h < heads; h++) {
        float mxv = sink[h];
        for (int m = 0; m < M; m++) {
            if (indices[m] < 0) { sc[m] = -INFINITY; continue; }
            float dot = 0.0f;
            for (int e = 0; e < dim; e++) dot += q[h * dim + e] * keys[(m * heads + h) * dim + e];
            sc[m] = dot * scale;
            if (sc[m] > mxv) mxv = sc[m];
        }
        float ssum = 0.0f, w[LLM_DSV41_ATTN_MAXLEN];
        for (int m = 0; m < M; m++) { w[m] = (indices[m] < 0) ? 0.0f : expf(sc[m] - mxv); ssum += w[m]; }
        ssum += expf(sink[h] - mxv);
        for (int e = 0; e < dim; e++) out[h * dim + e] = 0.0f;
        for (int m = 0; m < M; m++) {
            if (indices[m] < 0) continue;
            float wm = w[m] / ssum;
            for (int e = 0; e < dim; e++) out[h * dim + e] += wm * keys[(m * heads + h) * dim + e];
        }
    }
}

#endif /* LLM_DSV41_ATTN_H */
