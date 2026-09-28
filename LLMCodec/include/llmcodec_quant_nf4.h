/* llmcodec_quant_nf4.h — decoder bitsandbytes NF4 (bnb-4bit, con double-quant opzionale).
 *
 * SPECCHIO del decoder bitsandbytes (formula pubblica, non copiata):
 *  - codebook NF4: `bitsandbytes/functional.py` → `get_4bit_type("nf4")` /
 *    `create_normal_map(offset=0.9677083, use_extra_value=True)` (16 livelli quantili di N(0,1)).
 *  - double-quant: `quantize_4bit(..., compress_statistics=True)` e `dequantize_4bit`:
 *        offset = absmax.mean(); qabsmax = quantize_blockwise(absmax - offset, blocksize=256)
 *        dequant: absmax = dequantize_blockwise(qabsmax, state2) + offset
 *    con `state2.code = nested_quant_map` (256 valori, mappa dinamica 8-bit salvata nel ckpt).
 *  - packing: 2 nibble per byte; il PRIMO elemento (indice pari) sta nel nibble ALTO
 *    (`out[j/2] = (idx[j] << 4) | idx[j+1]`).
 *
 * Layout del checkpoint (`<base>.weight` U8 [numel/2, 1] + siblings):
 *   <base>.weight                        U8   nibble packed (N=out*in valori, high-first)
 *   <base>.weight.absmax                 U8   [nblocks]   indici nella nested_quant_map
 *   <base>.weight.nested_absmax          F32  [ceil(nblocks/256)]
 *   <base>.weight.nested_quant_map       F32  [256]
 *   <base>.weight.quant_map              F32  [16]  (codebook NF4)
 *   <base>.weight.quant_state.bitsandbytes__nf4  U8 JSON packed:
 *        {"shape":[out,in],"blocksize":64,"nested_blocksize":256,"nested_offset":f,...}
 *
 * Dequant (elemento j nell'ordine flat row-major [out,in]):
 *   b = j / blocksize
 *   absmax[b] = nested_quant_map[absmax_u8[b]] * nested_absmax[b / nested_blocksize] + offset
 *   nib = (j&1) ? (packed[j/2] & 0xF) : (packed[j/2] >> 4)
 *   w[j] = quant_map[nib] * absmax[b]
 */
#ifndef LLMCODEC_QUANT_NF4_H
#define LLMCODEC_QUANT_NF4_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantizza NF4 da `packed` (N elementi flat row-major) in `dst[N]`.
 *   packed          : ceil(N/2) byte, 2 nibble/byte, nibble ALTO = elemento pari
 *   absmax          : se `nested_code != NULL` e' const uint8_t* (indici 0..255);
 *                     altrimenti e' const float* (absmax F32 diretto, niente double-quant)
 *   nested_absmax   : F32 [ceil(nblocks/nested_blocksize)], usata solo con double-quant
 *   nested_code     : F32 [256], mappa degli assi quantizzati; NULL = niente double-quant
 *   code            : F32 [16] codebook NF4; NULL = tabella NF4 built-in
 *   offset          : nested_offset del checkpoint (0 se niente double-quant)
 *   blocksize       : 64 (o 32/128/...); nested_blocksize: 256
 * Ritorna 0 su successo, <0 su parametri invalidi. */
int llmcodec_nf4_dequant(const uint8_t* packed, const void* absmax,
                    const float* nested_absmax, const float* nested_code,
                    const float* code, float offset,
                    int blocksize, int nested_blocksize, size_t n, float* dst);

#ifdef __cplusplus
}
#endif
#endif /* LLMCODEC_QUANT_NF4_H */
