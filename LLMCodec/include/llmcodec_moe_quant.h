/* llmcodec_moe_quant.h — quantizzatore AFFINE RTN (round-to-nearest) per pesi, formato MLX affine.
 *
 * Produce lo STESSO layout che i motori già leggono (MLX affine: `u32` packed + `scales` + `biases`
 * per gruppo) → è lo specchio del dequant esistente (`llmcodec_moe_q3_gemv`, ggml affine). Nessuna
 * calibrazione: per ogni gruppo di `group` colonne, `scale=(max-min)/(2^bits-1)`, `bias=min`,
 * `q=clamp(round((x-bias)/scale), 0, 2^bits-1)`; il dequant è `q*scale+bias`.
 *
 * bits 1..8 (flusso contiguo LSB-first, larghezza = cols*bits/32). scales/biases scritti in F16
 * (dtype segnalato dal flag `sc_f16` nei consumatori, es. `MLX` header[28] bit0). Vedi §E1 del todo. */
#ifndef LLMCODEC_MOE_QUANT_H
#define LLMCODEC_MOE_QUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quantizza x[rows*cols] (row-major) in `w` (u32 packed, rows*ceil(cols/per)), `scales`/`biases`
 * (F16, rows*cols/group). Ritorna 0 su successo, <0 se i parametri non sono validi
 * (bits 1..8, cols e rows multipli di group). */
int llmcodec_moe_affine_quantize(const float* x, uint32_t rows, uint32_t cols,
                            uint32_t group, uint32_t bits,
                            uint32_t* w_out, uint16_t* scales_out, uint16_t* biases_out);

/* Dequant (inverso esatto della formula sopra): out[rows*cols] = q*scale + bias.
 * `sc_f16` = 1 se scales/biases sono F16, 0 se BF16. Per la verifica dell'errore. */
void llmcodec_moe_affine_dequantize(const uint32_t* w, const uint16_t* scales, const uint16_t* biases,
                               uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                               int sc_f16, float* out);

/* Byte necessari per i buffer di un tensore rows×cols: w (u32), scales, biases (F16). */
size_t llmcodec_moe_affine_w_bytes(uint32_t rows, uint32_t cols, uint32_t bits);
size_t llmcodec_moe_affine_sc_bytes(uint32_t rows, uint32_t cols, uint32_t group);

/* Quantizzatore affine PESATO dall'IMATRIX (DS4-Q1.b §3.5): come llmcodec_moe_affine_quantize ma
 * l'errore e' pesato per-colonna da `imp` (`cols` energie imatrix; NULL = fallback RTN identico).
 * Stesso formato di output (U32 packed + scales/biases F16). */
int llmcodec_moe_affine_quantize_weighted(const float* x, const float* imp, uint32_t rows, uint32_t cols,
                                      uint32_t group, uint32_t bits,
                                      uint32_t* w_out, uint16_t* scales_out, uint16_t* biases_out);

/* Variante GROUP-level (un blocco di `n` colonne): scala/bias pesati + codici (uint8, non
 * impaccati) per il gruppo. Usata dal collector/smoketest del motore. */
void llmcodec_moe_weighted_affine_group(const float* x, const float* imp, uint32_t n, uint32_t bits,
                                   float* out_scale, float* out_bias, uint8_t* codes_out);

#ifdef __cplusplus
}
#endif

#endif /* LLMCODEC_MOE_QUANT_H */
