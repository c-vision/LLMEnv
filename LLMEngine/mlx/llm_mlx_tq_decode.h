/* llm_mlx_tq_decode.h — decodifica TurboQuant (PolarQuant-MLX) dei pesi: UNA SOLA SORGENTE.
 *
 * Condivisa tra il path RESIDENTE (`turboquant_synthesize_weights` in llm_mlx_weights_io.c, che
 * dequantizza a load) e il path SWAP (`llm_mlx_moe_stream_build.h`, che dequantizza on-the-fly i
 * soli esperti attivi). La matematica e' quella validata identica alla libreria
 * github.com/manjunathshiva/turboquant-mlx.
 *
 * Formato: `<base>.weight` U32 (indici impacchettati) + `.scales` F16 [out, in/64] + `.signs`
 * F16 [in] + `.codebook` F16. I pesi sono in dominio RUOTATO `W_rot = Hadamard(W·signs)`.
 * `codebook` e' auto-descrittivo: 3 voci = ternary trit {0,1,2} base-3 (20/U32, ~1.6 bpw);
 * 8/4/16 voci = 3/2/4-bit bit-packed (10/16/8 per U32).
 *
 * Decode per riga (out-row), nell'ordine ESATTO della libreria: indici -> codebook -> scala ->
 * Hadamard inversa -> signs. NOTA: l'inversa NON e' `rotate_weight` (che moltiplica i segni
 * PRIMA della Hadamard): si fa Hadamard e POI signs, perche' H e diag(signs) non commutano. */
#ifndef LLM_MLX_TQ_DECODE_H
#define LLM_MLX_TQ_DECODE_H
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "llmcodec_fp16.h"

static inline int llm_mlx_tq_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

/* Hadamard normalizzata (n = 2^k), in-place. Stessa implementazione del path residente,
 * verificata identica alla libreria di riferimento. */
static inline void llm_mlx_tq_ihadamard(float* x, int n) {
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += (h << 1))
            for (int j = 0; j < h; j++) { float a = x[i + j], b = x[i + j + h]; x[i + j] = a + b; x[i + j + h] = a - b; }
    float inv = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* Decodifica UNA riga di peso tq: `wrow` packed U32, `scrow` F16 [ngrp], `cbf` codebook F32
 * (ncode voci), `sgf` signs F32 [in_dim]. Scrive `in_dim` float in `orow`. `trit`=base-3 (cbn==3).
 *
 * HOT PATH (chiamata out_dim volte per esperto): (1) i trit di una parola si estraggono con 20
 * divisioni `%3`/`/3` UNA VOLTA per parola (non `3^(c%20)` per elemento); (2) scale/codebook/signs
 * arrivano gia' F32 (hoist dal chiamante) -> nessuna conversione f16 per elemento; (3) la scala di
 * gruppo si carica una volta per gruppo. */
static inline void llm_mlx_tq_decode_row(const uint32_t* wrow, const uint16_t* scrow,
                                         const float* cbf, const float* sgf,
                                         int in_dim, int ngrp, int trit, int bits, float* orow) {
    const int perw = trit ? 20 : (32 / bits);
    const int gs = (ngrp > 0) ? (in_dim / ngrp) : 64;   /* group size derivato dal formato */
    const uint32_t mask = (1u << bits) - 1u;
    int g = 0, in_group = gs; float cur_sc = 0.0f;
    uint32_t word = 0; int pos = trit ? 20 : perw;
    uint8_t tr[20];
    for (int c = 0; c < in_dim; c++) {
        if (in_group == gs) { cur_sc = llmcodec_f16_to_f32(scrow[g]); g++; in_group = 0; }
        in_group++;
        int idx;
        if (trit) {
            if (pos == 20) { word = wrow[c / 20]; for (int t = 0; t < 20; t++) { tr[t] = (uint8_t)(word % 3u); word /= 3u; } pos = 0; }
            idx = tr[pos++];
        } else {
            if (pos == perw) { word = wrow[c / perw]; pos = 0; }
            idx = (int)((word >> (bits * pos)) & mask); pos++;
        }
        orow[c] = cbf[idx] * cur_sc;
    }
    llm_mlx_tq_ihadamard(orow, in_dim);
    for (int i = 0; i < in_dim; i++) orow[i] *= sgf[i];
}

#endif /* LLM_MLX_TQ_DECODE_H */
