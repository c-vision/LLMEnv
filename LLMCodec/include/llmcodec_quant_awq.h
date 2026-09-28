/* llmcodec_quant_awq.h — decoder AWQ (AutoAWQ GEMM) per i pesi int4 dei checkpoint HF.
 *
 * SPECCHIO del decoder di riferimento AutoAWQ (non copiato: la formula e' pubblica):
 *  - `awq/utils/packing_utils.py` → `unpack_awq` + `reverse_awq_order` + `dequantize_gemm`
 *  - `awq/modules/linear/gemm.py` → `WQLinear_GEMM.from_linear` (packing con `order_map`)
 * Per il formato GEMM (`quantization_config.quant_method="awq"`, `version="gemm"`):
 *   qweight  I32 [in,        out/8]   nibble impaccati LSB-first
 *   qzeros   I32 [in/group,  out/8]   idem
 *   scales   F16 [in/group,  out  ]
 * L'ordine dei nibble NON e' sequenziale: AutoAWQ impacca, per ogni colonna di 8 output,
 *   qweight[:, col] |= intweight[:, col*8 + order_map[i]] << (4*i),  order_map=[0,2,4,6,1,3,5,7]
 * quindi il canale output `o = col*8 + r` vive nel nibble `AWQ_REVERSE_ORDER[r]`
 * ([0,4,1,5,2,6,3,7] = posizione di r in order_map). Il dequant e'
 *   w[o,i] = (qweight[i, o/8] >> 4*AWQ_REVERSE_ORDER[o%8] & 0xF
 *             - qzeros[i/group, o/8] >> 4*AWQ_REVERSE_ORDER[o%8] & 0xF) * scales[i/group, o]
 * (nessun offset sul zero-point per il GEMM: `dequantize_gemm` fa direttamente (q - z) * s).
 */
#ifndef LLMCODEC_QUANT_AWQ_H
#define LLMCODEC_QUANT_AWQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantizza un peso AWQ GEMM int4 in `dst` come matrice [out, in] row-major
 * (convenzione `nn.Linear`, dst[o*in + i]) — pronta per `x @ w.T`.
 *   qweight: [in, out/8]        I32 (uint32 letto come packed)
 *   qzeros : [in/group, out/8]  I32
 *   scales : [in/group, out]    F16
 * `bits` deve essere 4 (unico supportato da AutoAWQ GEMM); `group_size` divide `in`.
 * Ritorna 0 su successo, <0 su parametri invalidi. */
int llmcodec_awq_dequant(const uint32_t* qweight, const uint32_t* qzeros,
                    const uint16_t* scales, int bits, int group_size,
                    int out, int in, float* dst);

#ifdef __cplusplus
}
#endif
#endif /* LLMCODEC_QUANT_AWQ_H */
