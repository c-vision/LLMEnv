/* llmcodec_quant_gptq.c — decoder GPTQ (AutoGPTQ/gptqmodel). Vedi include/llmcodec_quant_gptq.h.
 *
 * AutoGPTQ `QuantLinear.pack` (qlinear_cuda.py / qlinear_triton.py) impacca:
 *   qweight: per ogni riga di `per = 32/bits` valori input consecutivi,
 *            qweight[i/per, o] |= intweight[i, o] << bits*(i%per)   -> [in/per, out]
 *   qzeros : per ogni riga di `per` canali output consecutivi,
 *            qzeros[g, o/per] |= (zeros[g, o]-1) << bits*(o%per)     -> [in/group, out/per]
 *   scales : [in/group, out] (F16)
 * Il dequant CPU (`forward`, ramo else) fa:
 *   zeros = ((qzeros >> wf) & mask) + 1;  weight = (qweight >> wf) & mask;
 *   w = scales[g_idx] * (weight - zeros[g_idx])  con g_idx[i] = gruppo della riga i.
 * Qui lo replichiamo in C su `dst[o*in+i]` (layout nn.Linear, pronto per x @ w.T). */
#include "llmcodec_quant_gptq.h"
#include "llmcodec_fp16.h"

int llmcodec_gptq_dequant(const uint32_t* qweight, const uint32_t* qzeros,
                     const uint16_t* scales, const uint32_t* g_idx,
                     int bits, int group_size, int out, int in, float* dst) {
    if (qweight == NULL || qzeros == NULL || scales == NULL || dst == NULL) return -1;
    if (bits != 2 && bits != 4 && bits != 8) return -1;
    if (group_size <= 0 || in <= 0 || out <= 0) return -1;
    if ((in % group_size) != 0) return -1;
    const int per = 32 / bits;                   /* valori packed per int32 */
    if ((in % per) != 0 || (out % per) != 0) return -1;
    const int op = out / per;                    /* colonne di qzeros (= out/per) */
    const int ng = in / group_size;              /* gruppi di scala */
    const uint32_t mask = (1u << bits) - 1u;

    for (int o = 0; o < out; o++) {
        float* drow = dst + (size_t)o * (size_t)in;
        const int zc = o / per;                  /* colonna packed di qzeros */
        const int zsh = bits * (o % per);        /* shift del canale o in qzeros */
        for (int i = 0; i < in; i++) {
            /* qweight e' [in/per, out]: lo stride di riga e' `out`, non `op`. */
            const uint32_t q =
                (qweight[(size_t)(i / per) * (size_t)out + o] >> (bits * (i % per))) & mask;
            int gi = (g_idx != NULL) ? (int)g_idx[i] : (i / group_size);
            if (gi < 0 || gi >= ng) gi = i / group_size;
            const uint32_t z =
                ((qzeros[(size_t)gi * op + zc] >> zsh) & mask) + 1u;  /* AutoGPTQ: zeros+1 */
            const float s = llmcodec_f16_to_f32(scales[(size_t)gi * out + o]);
            drow[i] = (float)((int64_t)q - (int64_t)z) * s;
        }
    }
    return 0;
}
