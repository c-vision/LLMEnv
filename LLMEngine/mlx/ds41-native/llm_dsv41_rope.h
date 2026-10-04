/* llm_dsv41_rope.h — DeepSeek V4.1: RoPE con YaRN (step 6-ish, offline), header-only.
 *
 * Port C di `omlx/patches/deepseek_v41/language.py` `_rope`/`rope`:
 *   - freq_j = base^(-2j/d); se `compressed` e `original_seq_len>0` -> correzione YaRN
 *     (low/high da beta_fast/beta_slow, `smooth`, blend con rope_factor).
 *   - rotazione a coppie consecutive sull'ULTIMA d dimensione; `inverse` nega l'angolo.
 *   - base = `compress_rope_theta` se compressed, altrimenti `rope_theta`.
 *
 * Verifica: `llm_dsv41_rope_test.c` vs `tools/gen_dsv41_rope_reference.py`. Nessun peso/device.
 */
#ifndef LLM_DSV41_ROPE_H
#define LLM_DSV41_ROPE_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* In-place su x[head_dim]; ruota le ultime d dimensioni. d pari, d <= head_dim. */
static inline void llm_dsv41_rope(float* x, int head_dim, float pos, int d, float base,
                                  int original_seq_len, float beta_fast, float beta_slow,
                                  float rope_factor, int compressed, int inverse) {
    if (d <= 0 || (d & 1) || d > head_dim) return;
    int hd2 = d / 2;
    if (hd2 > 256) return;
    float freq[256];
    for (int j = 0; j < hd2; j++)
        freq[j] = 1.0f / powf(base, (2.0f * (float)j) / (float)d);

    if (compressed && original_seq_len > 0) {
        float lnb = logf(base);
        float cfast = (float)d * logf((float)original_seq_len / (beta_fast * 2.0f * (float)M_PI)) / (2.0f * lnb);
        float cslow = (float)d * logf((float)original_seq_len / (beta_slow * 2.0f * (float)M_PI)) / (2.0f * lnb);
        int low = (int)floorf(cfast); if (low < 0) low = 0;
        int high = (int)ceilf(cslow); if (high > d - 1) high = d - 1;
        float denom = (float)(high - low); if (denom < 1e-3f) denom = 1e-3f;
        for (int j = 0; j < hd2; j++) {
            float sm = 1.0f - fminf(fmaxf(((float)j - (float)low) / denom, 0.0f), 1.0f);
            freq[j] = freq[j] / rope_factor * (1.0f - sm) + freq[j] * sm;
        }
    }

    int off = head_dim - d;
    for (int j = 0; j < hd2; j++) {
        float ang = pos * freq[j];
        if (inverse) ang = -ang;
        float c = cosf(ang), s = sinf(ang);
        float a = x[off + 2 * j], b = x[off + 2 * j + 1];
        x[off + 2 * j] = a * c - b * s;
        x[off + 2 * j + 1] = a * s + b * c;
    }
}

#endif /* LLM_DSV41_ROPE_H */
