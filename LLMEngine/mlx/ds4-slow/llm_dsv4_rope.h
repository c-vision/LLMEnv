/* llm_dsv4_rope.h — port C di `kernel_dsv4_head_rms_norm_rope_tail_f32` (metal/dsv4_rope.metal):
 * RMSNorm per-head su head_dim + RoPE sulla coda (ultimi n_dims, coppie consecutive).
 * Qui senza YaRN (original_seq_len=0); riusa llm_dsv41_rope per la rotazione.
 * Header-only. Test: llm_dsv4_rope_test.c. */
#ifndef LLM_DSV4_ROPE_H
#define LLM_DSV4_ROPE_H

#include "llm_dsv41_rope.h"

/* x [n_head*head_dim] in-place. RoPE sulla coda di `n_dims` dim, base `freq_base`, posizione `pos`. */
static inline void llm_dsv4_head_rms_norm_rope(float* x, int n_head, int head_dim, int n_dims,
                                               float pos, float freq_base, float eps, int inverse) {
    for (int h = 0; h < n_head; h++) {
        float* xh = x + (long)h * head_dim;
        float sum = 0.0f;
        for (int i = 0; i < head_dim; i++) sum += xh[i] * xh[i];
        float scale = 1.0f / sqrtf(sum / (float)head_dim + eps);
        for (int i = 0; i < head_dim; i++) xh[i] *= scale;
        if (n_dims > 0) llm_dsv41_rope(xh, head_dim, pos, n_dims, freq_base, 0, 32.0f, 1.0f, 1.0f, 0, inverse);
    }
}

#endif /* LLM_DSV4_ROPE_H */
