/* llmcodec_quant_nf4.c — decoder bitsandbytes NF4. Vedi include/llmcodec_quant_nf4.h per la spec.
 *
 * Il double-quant e' la parte che conta per i checkpoint `bnb_4bit_use_double_quant=true`:
 * l'absmax per blocco di 64 pesi NON e' un F32, ma un indice U8 in `nested_quant_map`
 * (codebook a 256 livelli della mappa dinamica 8-bit), scalato da `nested_absmax` ogni 256
 * blocchi e traslato di `nested_offset`. La formula e' quella di
 * `bitsandbytes.functional.dequantize_blockwise(absmax, state2) + state.offset`. */
#include "llmcodec_quant_nf4.h"
#include <stdlib.h>

/* get_4bit_type("nf4") — create_normal_map(offset=0.9677083, use_extra_value=True). */
static const float LLMCODEC_NF4_CODE[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

int llmcodec_nf4_dequant(const uint8_t* packed, const void* absmax,
                    const float* nested_absmax, const float* nested_code,
                    const float* code, float offset,
                    int blocksize, int nested_blocksize, size_t n, float* dst) {
    if (packed == NULL || absmax == NULL || dst == NULL) return -1;
    if (blocksize <= 0 || n == 0) return -1;
    const float* map = (code != NULL) ? code : LLMCODEC_NF4_CODE;
    const size_t nblocks = (n + (size_t)blocksize - 1) / (size_t)blocksize;

    /* absmax per blocco, gia' dequantizzato (F32). */
    float* scale = (float*)malloc(nblocks * sizeof(float));
    if (scale == NULL) return -1;
    if (nested_code != NULL) {
        if (nested_absmax == NULL || nested_blocksize <= 0) { free(scale); return -1; }
        const uint8_t* idx = (const uint8_t*)absmax;
        for (size_t b = 0; b < nblocks; b++) {
            const float ns = nested_absmax[b / (size_t)nested_blocksize];
            scale[b] = nested_code[idx[b]] * ns + offset;
        }
    } else {
        const float* ab = (const float*)absmax;
        for (size_t b = 0; b < nblocks; b++) scale[b] = ab[b];
    }

    for (size_t j = 0; j < n; j++) {
        const uint8_t byte = packed[j >> 1];
        /* nibble ALTO = elemento pari (bitsandbytes: (idx[j]<<4)|idx[j+1]). */
        const int nib = (j & 1u) ? (int)(byte & 0x0Fu) : (int)(byte >> 4);
        dst[j] = map[nib] * scale[j / (size_t)blocksize];
    }
    free(scale);
    return 0;
}
