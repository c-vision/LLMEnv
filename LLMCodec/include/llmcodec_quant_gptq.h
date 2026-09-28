/* llmcodec_quant_gptq.h — decoder GPTQ int2/int4/int8 (formato AutoGPTQ/gptqmodel HF).
 *
 * SPECCHIO del decoder di riferimento AutoGPTQ (non copiato: la formula e' pubblica):
 *  - `auto_gptq/nn_modules/qlinear/qlinear_cuda.py` → `QuantLinear.pack` (packing) e
 *    `QuantLinear.forward` ramo CPU (dequant) — identico a `qlinear_triton.py`.
 *  - `auto_gptq/nn_modules/qlinear/qlinear_exllama.py` (stesso packing di `make_q4`).
 *
 * Layout del checkpoint (bits=4, group_size=128, in=896, out=896):
 *   qweight I32 [in/per, out]       per = 32/bits; nibble lungo IN, shift = bits*(i%per)
 *   qzeros  I32 [in/group, out/per] nibble lungo OUT, shift = bits*(o%per)
 *   scales  F16 [in/group, out]
 *   g_idx   I32 [in]                opzionale: gruppo di scala di ogni riga input
 * Il packing AutoGPTQ salva `zeros - 1`; il dequant ripristina `+1`:
 *   w[o,i] = ( (qweight[i/per,o] >> bits*(i%per)) & mask
 *              - ( (qzeros[g,o/per] >> bits*(o%per)) & mask ) - 1 ) * scales[g, o]
 * con `g = g_idx[i]` (o `i/group` se g_idx assente). Nessun offset ulteriore (sym=True ha
 * comunque zero point codificato in qzeros).
 */
#ifndef LLMCODEC_QUANT_GPTQ_H
#define LLMCODEC_QUANT_GPTQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantizza un peso GPTQ in `dst` come matrice [out, in] row-major (dst[o*in + i]).
 *   qweight: [in/per, out]        I32 (uint32 letto come packed), per = 32/bits
 *   qzeros : [in/group, out/per]  I32
 *   scales : [in/group, out]      F16
 *   g_idx  : [in]                 I32, puo' essere NULL -> si usa i/group
 * `bits` in {2,4,8}; `group_size` divide `in`; `in` e `out` multipli di `per`.
 * Ritorna 0 su successo, <0 su parametri invalidi. */
int llmcodec_gptq_dequant(const uint32_t* qweight, const uint32_t* qzeros,
                     const uint16_t* scales, const uint32_t* g_idx,
                     int bits, int group_size, int out, int in, float* dst);

#ifdef __cplusplus
}
#endif
#endif /* LLMCODEC_QUANT_GPTQ_H */
