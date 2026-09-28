/* llm_ggml.h — tipi ggml (GGUF) usati da DS4: UNA SOLA SORGENTE nel CODEC (§0bis). Wrapper sottili
 * su llmcodec_ggml_quants.h (tabella tipi + dequant) e llmcodec_fp16.h (atomi fp16/bf16). L'engine tiene SOLO
 * l'f32→f16 NEON, che è una specializzazione di PERFORMANCE (non logica duplicata). */
#ifndef LLM_GGML_H
#define LLM_GGML_H

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "llmcodec_ggml_quants.h"   /* enum canonico LLMCODEC_GGML_T_* + block_vals/type_name/dequant_row */
#include "llmcodec_gguf_read.h"     /* llmcodec_ggml_type_bytes */
#include "llmcodec_fp16.h"
#include "llm_iq2_tables.h"   /* grid iq2 usate dal compute DS4 (actquant/packed) */          /* llmcodec_f16_to_f32 / llmcodec_bf16_to_f32 / llmcodec_f16_from_f32 */

/* Alias ai nomi storici (UPPERCASE) → enum canonico del codec. */
#define LLM_GGML_TYPE_F32     LLMCODEC_GGML_T_F32
#define LLM_GGML_TYPE_F16     LLMCODEC_GGML_T_F16
#define LLM_GGML_TYPE_Q4_0    LLMCODEC_GGML_T_Q4_0
#define LLM_GGML_TYPE_Q4_1    LLMCODEC_GGML_T_Q4_1
#define LLM_GGML_TYPE_Q5_0    LLMCODEC_GGML_T_Q5_0
#define LLM_GGML_TYPE_Q5_1    LLMCODEC_GGML_T_Q5_1
#define LLM_GGML_TYPE_Q8_0    LLMCODEC_GGML_T_Q8_0
#define LLM_GGML_TYPE_Q8_1    LLMCODEC_GGML_T_Q8_1
#define LLM_GGML_TYPE_Q2_K    LLMCODEC_GGML_T_Q2_K
#define LLM_GGML_TYPE_Q3_K    LLMCODEC_GGML_T_Q3_K
#define LLM_GGML_TYPE_Q4_K    LLMCODEC_GGML_T_Q4_K
#define LLM_GGML_TYPE_Q5_K    LLMCODEC_GGML_T_Q5_K
#define LLM_GGML_TYPE_Q6_K    LLMCODEC_GGML_T_Q6_K
#define LLM_GGML_TYPE_Q8_K    LLMCODEC_GGML_T_Q8_K
#define LLM_GGML_TYPE_IQ2_XXS LLMCODEC_GGML_T_IQ2_XXS
#define LLM_GGML_TYPE_IQ2_XS  LLMCODEC_GGML_T_IQ2_XS
#define LLM_GGML_TYPE_I8      LLMCODEC_GGML_T_I8
#define LLM_GGML_TYPE_I16     LLMCODEC_GGML_T_I16
#define LLM_GGML_TYPE_I32     LLMCODEC_GGML_T_I32
#define LLM_GGML_TYPE_BF16    LLMCODEC_GGML_T_BF16

static inline const char* llm_ggml_type_name_cstr(int t) {
    static _Thread_local char b[32];
    if (llmcodec_ggml_type_name(t, b, sizeof b) != 0) return "?";
    return b;
}
/* elementi per blocco (1 = non bloccato). */
static inline int llm_ggml_blck_size(int t) { int64_t q = llmcodec_ggml_block_vals(t); return q > 0 ? (int)q : 1; }
/* byte per blocco. */
static inline int llm_ggml_type_size(int t) {
    uint32_t bs = (uint32_t)llm_ggml_blck_size(t); uint64_t b = 0;
    return llmcodec_ggml_type_bytes(t, bs, &b) == 0 ? (int)b : -1;
}
/* byte totali per `n` elementi (n multiplo del blocco). */
static inline int64_t llm_ggml_nbytes(int t, int64_t n) {
    uint64_t b = 0;
    if (n < 0 || llmcodec_ggml_type_bytes(t, (uint64_t)n, &b) != 0) return -1;
    return (int64_t)b;
}

/* f32 -> f16 (NEON del motore: specializzazione di PERFORMANCE; il canonico scalare è in llmcodec_fp16.h). */
static inline uint16_t llm_ggml_f32_to_f16(float f) {
#if defined(__ARM_NEON)
    float16x4_t hv = vcvt_f16_f32(vdupq_n_f32(f));
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
    return llmcodec_f16_from_f32(f);
#endif
}
static inline float llm_ggml_f16_round(float x) { return llmcodec_f16_to_f32(llm_ggml_f32_to_f16(x)); }

/* Il DEQUANT è quello canonico del codec: `int llmcodec_ggml_dequant_row(int t, const void* src, float* dst, int64_t n)`
 * (declarato in llmcodec_ggml_quants.h). I chiamanti dell'engine usano la STESSA firma/ordine. */

#endif /* LLM_GGML_H */
