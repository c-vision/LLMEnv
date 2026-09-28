/* llm_dsv41_math.h — DeepSeek V4.1: core math (Level-3 port, step 2), header-only.
 *
 * Port C delle formule VERBIFICABILI di `omlx/patches/deepseek_v41/`:
 *   - hyper_connection.py : sinkhorn_reference (mHC, 4-stream + Sinkhorn)
 *   - language.py         : _hc_mix_weights, _hc_mixes, hc_pre, hc_post, Compressor pooling
 *   - kernels.py          : packed_index_scores -> score = sum_h relu(dot(q_h,k_h)) * w_h
 *
 * NON e' il forward completo di DeepSeek V4.1 (CSA2/cache/MoE/Engram/vision/MTP): v. il todo
 * `(open) - DeepSeek-V41-Port.md`. Queste funzioni operano su array piccoli (float), senza MLX e
 * senza device: sono la parte verificabile offline.
 *
 * Verifica: `llm_dsv41_math_test.c` confronta con i valori di riferimento generati da
 * `tools/gen_dsv41_math_reference.py` (che usa MLX) -> `llm_dsv41_math_ref.h`.
 * Layout: row-major. n = hc_mult (4). D = dim. size = (2+n)*n.
 */
#ifndef LLM_DSV41_MATH_H
#define LLM_DSV41_MATH_H

#include <math.h>

static inline float llm_dsv41_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float llm_dsv41_relu(float x) { return x > 0.0f ? x : 0.0f; }

/* sinkhorn in-place su comb (n x n), replica di sinkhorn_reference(comb, eps, iters).
 * -1: dividi per la somma sulle RIGHE (axis -2): le colonne sommano 1 (+eps)
 * poi (iters-1)x: dividi per la somma sulle COLONNE (axis -1), poi sulle RIGHE (axis -2). */
static inline void llm_dsv41_sinkhorn(float* comb, int n, float eps, int iters) {
    for (int c = 0; c < n; c++) {
        float s = eps; for (int r = 0; r < n; r++) s += comb[r * n + c];
        for (int r = 0; r < n; r++) comb[r * n + c] /= s;
    }
    for (int it = 1; it < iters; it++) {
        for (int r = 0; r < n; r++) {
            float s = eps; for (int c = 0; c < n; c++) s += comb[r * n + c];
            for (int c = 0; c < n; c++) comb[r * n + c] /= s;
        }
        for (int c = 0; c < n; c++) {
            float s = eps; for (int r = 0; r < n; r++) s += comb[r * n + c];
            for (int r = 0; r < n; r++) comb[r * n + c] /= s;
        }
    }
}

/* Softmax lungo l'ultima dimensione (rows x cols), riga per riga. */
static inline void llm_dsv41_softmax_rows(float* m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float mxv = m[i * cols];
        for (int j = 1; j < cols; j++) if (m[i * cols + j] > mxv) mxv = m[i * cols + j];
        float s = 0.0f;
        for (int j = 0; j < cols; j++) { float e = expf(m[i * cols + j] - mxv); m[i * cols + j] = e; s += e; }
        for (int j = 0; j < cols; j++) m[i * cols + j] /= s;
    }
}

/* _hc_mix_weights(mixes[3n], scale[3], base[3n], n, hc_eps, iters) -> pre[n], post[n], comb[n*n] */
static inline void llm_dsv41_hc_mix_weights(const float* mixes, const float* scale,
                                            const float* base, int n, float hc_eps, int iters,
                                            float* pre, float* post, float* comb) {
    for (int i = 0; i < n; i++) pre[i] = llm_dsv41_sigmoid(mixes[i] * scale[0] + base[i]) + hc_eps;
    for (int i = 0; i < n; i++) post[i] = 2.0f * llm_dsv41_sigmoid(mixes[n + i] * scale[1] + base[n + i]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            comb[i * n + j] = mixes[2 * n + i * n + j] * scale[2] + base[2 * n + i * n + j];
    llm_dsv41_softmax_rows(comb, n, n);
    for (int i = 0; i < n * n; i++) comb[i] += hc_eps;
    llm_dsv41_sinkhorn(comb, n, hc_eps, iters);
}

/* _hc_mixes(x[n*D], fn[size*(n*D)], size, scale[3], base[size], n, norm_eps, hc_eps, iters)
 * -> mixes[size], pre[n], post[n], comb[n*n].  (size = (2+n)*n) */
static inline void llm_dsv41_hc_mixes(const float* x, int nD, const float* fn, int size,
                                      const float* scale, const float* base, int n,
                                      float norm_eps, float hc_eps, int iters,
                                      float* mixes, float* pre, float* post, float* comb) {
    float sq = 0.0f; for (int j = 0; j < nD; j++) sq += x[j] * x[j];
    float inv = 1.0f / sqrtf(sq / (float)nD + norm_eps);
    for (int k = 0; k < size; k++) {
        float dot = 0.0f;
        for (int j = 0; j < nD; j++) dot += x[j] * fn[k * nD + j];
        mixes[k] = dot * inv;
    }
    llm_dsv41_hc_mix_weights(mixes, scale, base, n, hc_eps, iters, pre, post, comb);
}

/* hc_pre(x[n*D], pre[n], D) -> out[D]:  sum_i x[i,c]*pre[i] */
static inline void llm_dsv41_hc_pre(const float* x, const float* pre, int n, int D, float* out) {
    for (int d = 0; d < D; d++) {
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += x[i * D + d] * pre[i];
        out[d] = s;
    }
}

/* hc_post(x[D], residual[n*D], post[n], comb[n*n], n, D) -> out[n*D]:
 * out[j,d] = post[j]*x[d] + sum_i comb[i,j]*residual[i,d] */
static inline void llm_dsv41_hc_post(const float* x, int D, const float* residual,
                                     const float* post, const float* comb, int n, float* out) {
    for (int j = 0; j < n; j++)
        for (int d = 0; d < D; d++) {
            float s = 0.0f;
            for (int i = 0; i < n; i++) s += comb[i * n + j] * residual[i * D + d];
            out[j * D + d] = post[j] * x[d] + s;
        }
}

/* Compressor pooling: kv/gate [(m*r) x H] -> pooled [m x H].
 * pooled[i,d] = sum_k kv[i,k,d] * softmax_k(gate[i,k,d]).  (language.py Compressor, ratio>1) */
static inline void llm_dsv41_compressor_pool(const float* kv, const float* gate, int r, int H,
                                             int m, float* pooled) {
    for (int i = 0; i < m; i++)
        for (int d = 0; d < H; d++) {
            float mxv = gate[(i * r) * H + d];
            for (int k = 1; k < r; k++) { float g = gate[(i * r + k) * H + d]; if (g > mxv) mxv = g; }
            float s = 0.0f, w[256];
            for (int k = 0; k < r; k++) { w[k] = expf(gate[(i * r + k) * H + d] - mxv); s += w[k]; }
            float acc = 0.0f;
            for (int k = 0; k < r; k++) acc += kv[(i * r + k) * H + d] * (w[k] / s);
            pooled[i * H + d] = acc;
        }
}

/* RMSNorm (language.py norm): out[d] = x[d] * rsqrt(mean(x^2)+eps) * w[d] */
static inline void llm_dsv41_rms_norm(const float* x, const float* w, int D, float eps, float* out) {
    float sq = 0.0f; for (int d = 0; d < D; d++) sq += x[d] * x[d];
    float inv = 1.0f / sqrtf(sq / (float)D + eps);
    for (int d = 0; d < D; d++) out[d] = x[d] * inv * w[d];
}

/* Indexer score (kernels.py packed_index_scores): sum_h relu(dot(q_h,k_h)) * w_h */
static inline float llm_dsv41_index_score(const float* q, const float* key, const float* weights,
                                          int heads, int dim) {
    float s = 0.0f;
    for (int h = 0; h < heads; h++) {
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) dot += q[h * dim + d] * key[h * dim + d];
        s += llm_dsv41_relu(dot) * weights[h];
    }
    return s;
}

#endif /* LLM_DSV41_MATH_H */
