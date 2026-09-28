/* llmcodec_iq2_xxs.h — quantizzatori IQ (2-3 bit) portati da llama.cpp ggml-quants.c:
 * IQ2_XXS/XS/S, IQ1_S/M, IQ3_XXS/S, con `llmcodec_make_qp_quants` (codebook lattice) e
 * `*_quantize` (x -> byte packed, con quant_weights imatrix opzionali). */
#ifndef LLMCODEC_IQ2_XXS_H
#define LLMCODEC_IQ2_XXS_H
#include <stdint.h>
/* IQ2_XXS (2.0625 bpw, super-blocco 256 = d(F16) + qs[32] uint16 = 66 byte).
 * Port fedele da llama.cpp ggml-quants.c. quant_weights (imatrix) NULL -> pesi uniformi 1.0f. */
float llmcodec_make_qp_quants(int n, int nmax, const float* x, uint8_t* L, const float* quant_weights);
void llmcodec_iq2_xxs_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                          int64_t quant_weights_len, int64_t row_len);
/* IQ2_XS (2.3125 bpw, super-blocco 256 = d(F16) + qs[32] uint16 + scales[8] = 74 byte). */
void llmcodec_iq2_xs_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                         int64_t quant_weights_len, int64_t row_len);
/* IQ2_S (2.5625 bpw, super-blocco 256 = d(F16) + qs[64] + qh[8] + scales[8] = 82 byte). */
void llmcodec_iq2_s_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                        int64_t quant_weights_len, int64_t row_len);
/* IQ1_S (1.5625 bpw, super-blocco 256 = d(F16) + qs[32] + qh[8] uint16 = 50 byte). */
void llmcodec_iq1_s_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                        int64_t quant_weights_len, int64_t row_len);
/* IQ1_M (1.75 bpw, super-blocco 256 = qs[32] + qh[16] + scales[8] = 56 byte, scala f16 in testa). */
void llmcodec_iq1_m_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                        int64_t quant_weights_len, int64_t row_len);
/* IQ3_XXS (3.0625 bpw, super-blocco 256 = d(F16) + qs[96] = 98 byte). */
void llmcodec_iq3_xxs_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                          int64_t quant_weights_len, int64_t row_len);
/* IQ3_S (3.4375 bpw, super-blocco 256 = d(F16) + qs[64] + qh[8] + signs[32] + scales[4] = 110 byte). */
void llmcodec_iq3_s_quantize(const float* x, int64_t n, uint8_t* out, const float* quant_weights,
                        int64_t quant_weights_len, int64_t row_len);
#endif
