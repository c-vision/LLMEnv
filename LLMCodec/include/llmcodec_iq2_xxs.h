/* llmcodec_iq2_xxs.h — APACHE: solo `llmcodec_make_qp_quants` (usata dall'encode K-quant).
 * L'IQ-ENCODE e' PUL (gmbllm/include/quant). */
#ifndef LLMCODEC_IQ2_XXS_H
#define LLMCODEC_IQ2_XXS_H
#include <stdint.h>
float llmcodec_make_qp_quants(int n, int nmax, const float* x, uint8_t* L, const float* quant_weights);
#endif
