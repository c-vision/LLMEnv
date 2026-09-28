/* llmcodec_ggml_quants.h — codec GGML (blocco ↔ f32) per i formati dei blocchi ggml.
 *
 * Decoder (blocco→f32): serve a leggere QUALSIASI .gguf. Encoder (f32→blocco): a scriverlo.
 * Il layout dei blocchi è un FORMATO PUBBLICO (definito dagli header ggml); qui è REIMPLEMENTATO
 * in proprio dalla definizione del formato (non è copiato codice di terzi — nessuna questione di
 * licenza). Riferimenti di formato/verifica (usati come spec, non copiati):
 *  - `native/llmengine/build/_deps/llama_cpp-src/ggml/src/ggml-common.h` (layout blocchi).
 *  - `ds4/gguf-tools/quants.c` (comportamento atteso di encode per q4_k/q2_k/iq2_xxs).
 *
 * Implementazione INCREMENTALE, famiglia per famiglia, con gate di parità byte-exact.
 * Oggi: F32/F16/BF16 + Q4_0/Q4_1/Q5_0/Q5_1/Q8_0. Prossime: Q2_K..Q6_K/Q8_K, TQ1_0/TQ2_0/Q1_0,
 * MXFP4/NVFP4, IQ1_S/M..IQ4_XS.
 */
#ifndef LLMCODEC_GGML_QUANTS_H
#define LLMCODEC_GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tipi GGML (sottoinsieme; numerazione = ggml.h). */
enum {
    LLMCODEC_GGML_T_F32 = 0, LLMCODEC_GGML_T_F16 = 1, LLMCODEC_GGML_T_Q4_0 = 2, LLMCODEC_GGML_T_Q4_1 = 3,
    LLMCODEC_GGML_T_Q5_0 = 6, LLMCODEC_GGML_T_Q5_1 = 7, LLMCODEC_GGML_T_Q8_0 = 8, LLMCODEC_GGML_T_Q8_1 = 9,
    LLMCODEC_GGML_T_I8 = 24, LLMCODEC_GGML_T_I16 = 25, LLMCODEC_GGML_T_I32 = 26, LLMCODEC_GGML_T_I64 = 27,
    LLMCODEC_GGML_T_Q2_K = 10, LLMCODEC_GGML_T_Q3_K = 11, LLMCODEC_GGML_T_Q4_K = 12,
    LLMCODEC_GGML_T_Q5_K = 13, LLMCODEC_GGML_T_Q6_K = 14, LLMCODEC_GGML_T_Q8_K = 15,
    LLMCODEC_GGML_T_IQ2_XXS = 16, LLMCODEC_GGML_T_IQ2_XS = 17, LLMCODEC_GGML_T_IQ3_XXS = 18, LLMCODEC_GGML_T_IQ1_S = 19,
    LLMCODEC_GGML_T_IQ4_NL = 20, LLMCODEC_GGML_T_IQ3_S = 21, LLMCODEC_GGML_T_IQ2_S = 22, LLMCODEC_GGML_T_IQ4_XS = 23,
    LLMCODEC_GGML_T_IQ1_M = 29,
    LLMCODEC_GGML_T_BF16 = 30,
    LLMCODEC_GGML_T_TQ1_0 = 34, LLMCODEC_GGML_T_TQ2_0 = 35,
    LLMCODEC_GGML_T_MXFP4 = 39, LLMCODEC_GGML_T_NVFP4 = 40, LLMCODEC_GGML_T_Q1_0 = 41, LLMCODEC_GGML_T_Q2_0 = 42,
};

/* Decode: riempie dst[0..n) dai byte packed `src` del tipo `ggml_type`.
 * Ritorna 0 su successo, -1 se il tipo non è gestito o `n` non è multiplo del blocco. */
int llmcodec_ggml_dequant_row(int ggml_type, const void* src, float* dst, int64_t n);

/* Encode: quantizza src[0..n) in `dst` (packed) del tipo `ggml_type`.
 * `imatrix` (opzionale, n pesi) = errore pesato; NULL = RTN. Ritorna 0 / -1. */
int llmcodec_ggml_quant_row(int ggml_type, const float* src, void* dst, int64_t n, const float* imatrix);

/* Valori per blocco del tipo (0 se ignoto). */
int64_t llmcodec_ggml_block_vals(int ggml_type);

/* Helper PER-VALORE (esposti: usati anche dal compute dell'engine DS4). */
void  llmcodec_ggml_q4_K_scale_min(uint32_t j, const uint8_t* q, uint8_t* sc, uint8_t* m);
float llmcodec_ggml_q2_K_value(const uint8_t* xb, uint32_t idx);
float llmcodec_ggml_iq2_xxs_value(const uint8_t* blk, uint32_t idx);


/* Helper K-quant (ggml) esposti: UNA SOLA SORGENTE (§0bis) — anche llm_runtime li usa. */
int    llmcodec_nearest_int(float f);
float  llmcodec_make_qx_quants(int n, int nmax, const float* x, int8_t* L, int rmse_type, const float* qw);
float  llmcodec_make_q3_quants(int n, int nmax, const float* x, int8_t* L, int do_rmse);
float  llmcodec_make_qkx2_quants(int n, int nmax, const float* x, const float* w, uint8_t* L,
                            float* the_min, uint8_t* Laux, float rmse_type, float pos, int nstep, int use_mad);

#ifdef __cplusplus
}
#endif

int llmcodec_ggml_type_name(int ggml_type, char* out, size_t cap);
/* Nome-fmt dalla .llmarch (es. "q4_k","iq2_xxs","f16") -> id tipo GGML, o -1. */
int llmcodec_ggml_type_of_fmt(const char* fmt);
#endif /* LLMCODEC_GGML_QUANTS_H */
