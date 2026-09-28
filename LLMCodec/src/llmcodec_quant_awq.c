/* llmcodec_quant_awq.c — decoder AWQ int4 GEMM (AutoAWQ). Vedi include/llmcodec_quant_awq.h per la spec.
 *
 * AutoAWQ `WQLinear_GEMM.from_linear` (awq/modules/linear/gemm.py) impacca i nibble in ordine
 * NON sequenziale: per ogni colonna di 8 canali output, `qweight[:, col]` tiene
 * `intweight[:, col*8 + order_map[i]]` nel nibble i, con order_map=[0,2,4,6,1,3,5,7].
 * `dequantize_gemm` (awq/utils/packing_utils.py) fa l'inverso (`AWQ_REVERSE_ORDER`,
 * [0,4,1,5,2,6,3,7]) sia su qweight sia su qzeros, maschera a 4 bit e applica (q - z) * s.
 * Nessun offset ulteriore sul zero-point. */
#include "llmcodec_quant_awq.h"
#include "llmcodec_fp16.h"

/* AWQ_REVERSE_ORDER[r] = indice del nibble che contiene il canale output (col*8 + r):
 * e' la posizione di r dentro order_map=[0,2,4,6,1,3,5,7]. */
static const int AWQ_REVERSE_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

int llmcodec_awq_dequant(const uint32_t* qweight, const uint32_t* qzeros,
                    const uint16_t* scales, int bits, int group_size,
                    int out, int in, float* dst) {
    if (qweight == NULL || qzeros == NULL || scales == NULL || dst == NULL) return -1;
    if (bits != 4) return -1;                       /* AutoAWQ GEMM: solo int4 */
    if (group_size <= 0 || in <= 0 || out <= 0) return -1;
    if ((in % group_size) != 0 || (out % 8) != 0) return -1;

    const int op = out / 8;                          /* colonne di qweight/qzeros (out/8) */
    const uint32_t mask = (1u << bits) - 1u;         /* 0xF */

    for (int o = 0; o < out; o++) {
        const int col = o >> 3;                      /* colonna packed */
        const int sh = 4 * AWQ_REVERSE_ORDER[o & 7]; /* shift in bit del canale o */
        float* drow = dst + (size_t)o * (size_t)in;
        for (int i = 0; i < in; i++) {
            const uint32_t q = (qweight[(size_t)i * op + col] >> sh) & mask;
            const uint32_t z = (qzeros[(size_t)(i / group_size) * op + col] >> sh) & mask;
            const float s = llmcodec_f16_to_f32(scales[(size_t)(i / group_size) * out + o]);
            drow[i] = (float)((int)q - (int)z) * s;
        }
    }
    return 0;
}
