/* llm_dsv41_engram.h — DeepSeek V4.1: Engram (n-gram hash + gate), step 5, header-only.
 *
 * Port C delle parti VERIFICABILI offline di `omlx/patches/deepseek_v41/engram.py`:
 *   - NgramHash.__call__ : hashing n-gram (XOR dei prodotti token*fingerprint) % primi + offset,
 *     con lookback che si BLOCCA su -1 (posizioni immagine/padding) e nuovo history.
 *   - Engram.__call__    : gate (cosine-like su h/key) e iniezione nel residual.
 *
 * NON include: `build_compressed_token_map` (richiede il tokenizer Rust) ne' le TABELLE
 * (`nn.Embedding`): richiedono il modello. La COSTRUZIONE delle tabelle (primi/multipliers/offsets)
 * usa il RNG NumPy PCG64: qui le tabelle sono INPUT (v. `llm_dsv41_engram_ref.h`, generate da
 * `tools/gen_dsv41_engram_reference.py`); il porting del RNG e' un todo dedicato (loader).
 *
 * Aritmetica: come NumPy int64 (wrap-around in two's complement) e modulo "floored" (risultato
 * non-negativo) per uguagliare Python.
 */
#ifndef LLM_DSV41_ENGRAM_H
#define LLM_DSV41_ENGRAM_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Modulo "floored" (come Python): risultato sempre >= 0, b > 0. */
static inline int64_t llm_dsv41_fmod(int64_t a, int64_t b) {
    int64_t r = a % b; if (r < 0) r += b; return r;
}

/* NgramHash per UNA sequenza (B=1): tokens[L] (id compressi), history_in[depth].
 *  multipliers[depth+1]; primes[depth*n_heads]; offsets[depth*n_heads].
 *  hashes_out[L*depth*n_heads]; history_out[depth]. */
static inline void llm_dsv41_engram_hash(const int64_t* tokens, int L,
                                         const int64_t* history_in, int depth,
                                         int64_t pad_id,
                                         const int64_t* multipliers,
                                         const int64_t* primes, int n_heads,
                                         const int64_t* offsets,
                                         int64_t* hashes_out, int64_t* history_out) {
    int total = depth + L;
    /* joined = history ++ tokens */
    int64_t* joined = (int64_t*)malloc((size_t)total * sizeof(int64_t));
    if (joined == NULL) return;
    for (int j = 0; j < depth; j++) joined[j] = history_in[j];
    for (int i = 0; i < L; i++) joined[depth + i] = tokens[i];

    for (int i = 0; i < L; i++) {
        int blocked = 0;
        int64_t rolling = 0;
        for (int shift = 0; shift <= depth; shift++) {
            int idx = (i + depth) - shift;             /* positions[i] - shift, >= 0 */
            int64_t src = joined[idx];
            if (src == -1) blocked = 1;                /* lookback si blocca su -1 (cumulativo) */
            int64_t look = blocked ? pad_id : src;
            int64_t prod = (int64_t)((uint64_t)look * (uint64_t)multipliers[shift]); /* wrap */
            if (shift == 0) {
                rolling = prod;
            } else {
                rolling = (int64_t)((uint64_t)rolling ^ (uint64_t)prod);
                for (int h = 0; h < n_heads; h++) {
                    int k = (shift - 1) * n_heads + h;
                    hashes_out[(size_t)i * depth * n_heads + k] =
                        llm_dsv41_fmod(rolling, primes[k]) + offsets[k];
                }
            }
        }
    }
    for (int j = 0; j < depth; j++) history_out[j] = joined[L + j];
    free(joined);
}

/* Engram.__call__ (gate + iniezione). h:/key: [hc_mult x dim]; value:[dim]; q_weight/k_weight:
 * [hc_mult x dim]; out:[hc_mult x dim]. (image_mask applicato dal chiamante) */
static inline void llm_dsv41_engram_gate(const float* h, int hc, int dim,
                                         const float* key, const float* value,
                                         const float* q_weight, const float* k_weight,
                                         float norm_eps, float* out) {
    for (int s = 0; s < hc; s++) {
        float sqx = 0.0f, sqk = 0.0f, dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            float x = h[s * dim + d], k = key[s * dim + d];
            sqx += x * x; sqk += k * k;
            dot += x * q_weight[s * dim + d] * k_weight[s * dim + d] * k;
        }
        float inv = (1.0f / sqrtf(sqx / (float)dim + norm_eps)) *
                    (1.0f / sqrtf(sqk / (float)dim + norm_eps));
        dot = dot * inv * (1.0f / sqrtf((float)dim));
        float g;
        if (dot == 0.0f) {
            g = 1.0f / (1.0f + expf(-0.001f));           /* copysign(+sqrt, 0) e' positivo */
        } else {
            float a = fabsf(dot); if (a < 1e-6f) a = 1e-6f;
            float root = sqrtf(a);
            float sgn = dot < 0.0f ? -1.0f : 1.0f;
            g = 1.0f / (1.0f + expf(-sgn * root));
        }
        for (int d = 0; d < dim; d++) out[s * dim + d] = h[s * dim + d] + g * value[d];
    }
}

#endif /* LLM_DSV41_ENGRAM_H */
