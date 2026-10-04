/* llm_dsv41_engram_params.h — DeepSeek V4.1: costruzione dei parametri di hash Engram
 * (primes + offsets) come `omlx .../engram.py NgramHash.__init__`, senza il RNG (i multipliers,
 * che usano NumPy PCG64, restano separati). Il `seen` e' condiviso tra i layer engram, in ordine.
 *
 * I primes partono da `engram_vocab_size - 1` e avanzano saltando i non-primi e i gia' visti;
 * `offsets` = somma prefissa esclusiva dei primes; `flat.sum` per layer DEVE uguagliare
 * `engram_num_embeddings[layer]` (asserzione della reference). C puro, nessun device. */
#ifndef LLM_DSV41_ENGRAM_PARAMS_H
#define LLM_DSV41_ENGRAM_PARAMS_H

#include <stdint.h>
#include <stdbool.h>

#include "llm_dsv41_numpy_rng.h"

static inline bool llm_dsv41_is_prime(int64_t n) {
    if (n < 2) return false;
    for (int64_t d = 2; d * d <= n; d++) if (n % d == 0) return false;
    return true;
}

/* Genera `depth*heads` primes/offsets per UN layer. `seen`/`seen_n` sono condivisi (in/out).
 * Ritorna la somma dei primes (== offset totale == num_embeddings atteso). */
static inline int64_t llm_dsv41_engram_scan(int vocab_size, int depth, int heads,
                                            int64_t* seen, int* seen_n, int seen_cap,
                                            int64_t* primes_out, int64_t* offsets_out) {
    int k = 0;
    int64_t off = 0;
    for (int d = 0; d < depth; d++) {
        int64_t cur = (int64_t)vocab_size - 1;
        for (int h = 0; h < heads; h++) {
            cur++;
            bool used = true;
            while (used) {
                used = false;
                if (!llm_dsv41_is_prime(cur)) {
                    used = true;
                } else {
                    for (int i = 0; i < *seen_n; i++) if (seen[i] == cur) { used = true; break; }
                }
                if (used) cur++;
            }
            if (*seen_n < seen_cap) seen[(*seen_n)++] = cur;
            primes_out[k] = cur;
            offsets_out[k] = off;
            off += cur;
            k++;
        }
    }
    return off;
}

/* multipliers: rng.integers(0, bound, max_ngram, int64)*2+1 con seed 10007*layer_id
 * (NumPy PCG64). NB: la reference usa il **vocab compresso** (token_map.max()+1), NON
 * engram_vocab_size: `bound = max(1, (INT64_MAX // compressed_vocab) // 2)`. */
static inline void llm_dsv41_engram_multipliers(int compressed_vocab, int layer_id, int max_ngram,
                                                int64_t* mult_out) {
    int64_t bound = (INT64_MAX / (int64_t)compressed_vocab) / 2;
    if (bound < 1) bound = 1;
    llm_dsv41_numpy_integers_i64((int64_t)10007 * (int64_t)layer_id, bound, mult_out, max_ngram);
    for (int i = 0; i < max_ngram; i++) mult_out[i] = mult_out[i] * 2 + 1;
}

#endif /* LLM_DSV41_ENGRAM_PARAMS_H */
