/* quant_diff_ggml.c — llama.cpp/ggml come TESTIMONE del gate differenziale del codec.
 *
 * Perché esiste: la regola di copertura di LLMCodec dice che un tipo entra solo se esiste un
 * SECONDO codice indipendente con cui confrontarlo bit a bit. C è un codice che non condivide
 * una riga con il codec e che gira in un altro mondo.
 *
 * Cosa è verificato qui DENTRO, non assunto:
 *  - i `_Static_assert` sui valori di enum: la ns. tabella keyed per NUMERO deve dare lo
 *    stesso numero di ggml per ogni tipo. Se uno dei due ha rinumerato, questa unità non
 *    compila e lo dice col nome del tipo;
 *  - `ggc_row_bytes` passa da `ggml_row_size`: la dimensione in byte del blocco è letta dalla
 *    TAVOLA DI GGML, non ricopiata qui — così il gate confronta anche i byte, non solo i float.
 *
 * Non si compila `ggml-quants.c` in questo eseguibile: si linka la libreria già costruita
 * (es. `llamacpp/build/bin/libggml-base.*.dylib`), che esporta le `dequantize_row_*` nominate
 * qui sotto. Il gate non è un target CMake: è uno strumento di confronto.
 */

#include "ggml-quants.h" /* dequantize_row_*, block_q*, QK* */
#include "ggml.h"        /* ggml_row_size */

#include <stdint.h>
#include <string.h>

int ggc_block_vals(int type);
int ggc_dequant_row(int type, const uint8_t *bytes, int64_t n, float *dst);
uint64_t ggc_row_bytes(int type, int64_t n);

#define GG_ASSERT_NUM(ggname, ourn) _Static_assert((int)(ggname) == (ourn), #ggname " != " #ourn)
GG_ASSERT_NUM(GGML_TYPE_Q4_0, 2);
GG_ASSERT_NUM(GGML_TYPE_Q4_1, 3);
GG_ASSERT_NUM(GGML_TYPE_Q5_0, 6);
GG_ASSERT_NUM(GGML_TYPE_Q5_1, 7);
GG_ASSERT_NUM(GGML_TYPE_Q8_0, 8);
GG_ASSERT_NUM(GGML_TYPE_Q2_K, 10);
GG_ASSERT_NUM(GGML_TYPE_Q3_K, 11);
GG_ASSERT_NUM(GGML_TYPE_Q4_K, 12);
GG_ASSERT_NUM(GGML_TYPE_Q5_K, 13);
GG_ASSERT_NUM(GGML_TYPE_Q6_K, 14);
GG_ASSERT_NUM(GGML_TYPE_Q8_K, 15);
GG_ASSERT_NUM(GGML_TYPE_IQ2_XXS, 16);
GG_ASSERT_NUM(GGML_TYPE_IQ2_XS, 17);
GG_ASSERT_NUM(GGML_TYPE_IQ3_XXS, 18);
GG_ASSERT_NUM(GGML_TYPE_IQ1_S, 19);
GG_ASSERT_NUM(GGML_TYPE_IQ4_NL, 20);
GG_ASSERT_NUM(GGML_TYPE_IQ3_S, 21);
GG_ASSERT_NUM(GGML_TYPE_IQ2_S, 22);
GG_ASSERT_NUM(GGML_TYPE_IQ4_XS, 23);
GG_ASSERT_NUM(GGML_TYPE_IQ1_M, 29);
GG_ASSERT_NUM(GGML_TYPE_TQ1_0, 34);
GG_ASSERT_NUM(GGML_TYPE_TQ2_0, 35);
GG_ASSERT_NUM(GGML_TYPE_MXFP4, 39);
GG_ASSERT_NUM(GGML_TYPE_NVFP4, 40);
GG_ASSERT_NUM(GGML_TYPE_Q1_0, 41);

/* Elementi per blocco SECONDO GGML (i suoi macro). Il gate lo confronta con `block_vals` del
 * codec e, se divergono, NON chiama il dequant (ogni `dequantize_row_*` fa `assert(k % qk == 0)`). */
int ggc_block_vals(int type) {
    switch (type) {
    case 2:  return QK4_0;
    case 3:  return QK4_1;
    case 6:  return QK5_0;
    case 7:  return QK5_1;
    case 8:  return QK8_0;
    case 10: case 11: case 12: case 13: case 14: case 15:
    case 16: case 17: case 18: case 19:
    case 21: case 22: case 23: case 29: case 34: case 35: return QK_K;
    case 20: return QK4_NL;
    case 39: return QK_MXFP4;
    case 40: return QK_NVFP4;
    case 41: return QK1_0;
    default: return 0;
    }
}

/* 1 = il testimone conosce `type` e ha scritto n float in dst. F32/F16/BF16 hanno una copia
 * con conversione (le `ggml_*_to_fp32_row`), gli altri un `dequantize_row_*`. */
int ggc_dequant_row(int type, const uint8_t *bytes, int64_t n, float *dst) {
    int bv;
    switch (type) {
    case 0:
        if (n <= 0) return 0;
        memcpy(dst, bytes, (size_t)n * sizeof(float));
        return 1;
    case 1:
        if (n <= 0) return 0;
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)bytes, dst, n);
        return 1;
    case 30:
        if (n <= 0) return 0;
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)bytes, dst, n);
        return 1;
    default: break;
    }
    bv = ggc_block_vals(type);
    if (bv == 0) return 0;
    if (n % bv) return 0;
    switch (type) {
    case 2:  dequantize_row_q4_0((const block_q4_0 *)bytes, dst, n); return 1;
    case 3:  dequantize_row_q4_1((const block_q4_1 *)bytes, dst, n); return 1;
    case 6:  dequantize_row_q5_0((const block_q5_0 *)bytes, dst, n); return 1;
    case 7:  dequantize_row_q5_1((const block_q5_1 *)bytes, dst, n); return 1;
    case 8:  dequantize_row_q8_0((const block_q8_0 *)bytes, dst, n); return 1;
    case 10: dequantize_row_q2_K((const block_q2_K *)bytes, dst, n); return 1;
    case 11: dequantize_row_q3_K((const block_q3_K *)bytes, dst, n); return 1;
    case 12: dequantize_row_q4_K((const block_q4_K *)bytes, dst, n); return 1;
    case 13: dequantize_row_q5_K((const block_q5_K *)bytes, dst, n); return 1;
    case 14: dequantize_row_q6_K((const block_q6_K *)bytes, dst, n); return 1;
    case 15: dequantize_row_q8_K((const block_q8_K *)bytes, dst, n); return 1;
    case 16: dequantize_row_iq2_xxs((const block_iq2_xxs *)bytes, dst, n); return 1;
    case 17: dequantize_row_iq2_xs((const block_iq2_xs *)bytes, dst, n); return 1;
    case 18: dequantize_row_iq3_xxs((const block_iq3_xxs *)bytes, dst, n); return 1;
    case 19: dequantize_row_iq1_s((const block_iq1_s *)bytes, dst, n); return 1;
    case 20: dequantize_row_iq4_nl((const block_iq4_nl *)bytes, dst, n); return 1;
    case 21: dequantize_row_iq3_s((const block_iq3_s *)bytes, dst, n); return 1;
    case 22: dequantize_row_iq2_s((const block_iq2_s *)bytes, dst, n); return 1;
    case 23: dequantize_row_iq4_xs((const block_iq4_xs *)bytes, dst, n); return 1;
    case 29: dequantize_row_iq1_m((const block_iq1_m *)bytes, dst, n); return 1;
    case 34: dequantize_row_tq1_0((const block_tq1_0 *)bytes, dst, n); return 1;
    case 35: dequantize_row_tq2_0((const block_tq2_0 *)bytes, dst, n); return 1;
    case 39: dequantize_row_mxfp4((const block_mxfp4 *)bytes, dst, n); return 1;
    case 40: dequantize_row_nvfp4((const block_nvfp4 *)bytes, dst, n); return 1;
    case 41: dequantize_row_q1_0((const block_q1_0 *)bytes, dst, n); return 1;
    default: return 0;
    }
}

uint64_t ggc_row_bytes(int type, int64_t n) {
    size_t r;
    if (ggc_block_vals(type) == 0) return 0;
    if (n % ggc_block_vals(type)) return 0;
    r = ggml_row_size((enum ggml_type)type, n);
    return (uint64_t)r;
}

/* Quanti valori di enum ggml conosce QUESTA libreria. */
int ggc_type_count(void) { return (int)GGML_TYPE_COUNT; }

/* Byte per `n` elementi di un tipo QUALSIASI noto a ggml (0 se il valore non le è noto). */
uint64_t ggc_row_bytes_any(int type, int64_t n) {
    if (type < 0 || type >= (int)GGML_TYPE_COUNT || n <= 0) return 0;
    return (uint64_t)ggml_row_size((enum ggml_type)type, n);
}

/* La conversione f16->f32 che ggml usa nei suoi dequant: la funzione ESPORTATA, non una copia. */
uint32_t ggc_f16_bits(uint16_t h) {
    float f = ggml_fp16_to_fp32(h);
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return bits;
}
