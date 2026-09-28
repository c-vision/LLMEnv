/* llmengine_moe_ggml.h — dequantizzazione dei tipi ggml (GGUF) verso f32.
 *
 * Primo slice dell'ingest GGUF: F16/F32 (non quantizzati) + Q8_0/Q4_0 (blocchi a 32 elementi).
 * I tipi non implementati ritornano un errore ESPLICITO (mai un risultato silenzioso): la
 * detection del tipo avviene comunque (type_size/blck_size), così il chiamante può rifiutare
 * con un messaggio chiaro. I K-quant (Q4_K/Q5_K/Q6_K) arriveranno in uno slice successivo.
 * Implementazione in C, mai Python. Vedi `Todo/(open) - Model-Architecture-Auto-Discovery.md`. */
#ifndef LLMENGINE_MOE_GGML_H
#define LLMENGINE_MOE_GGML_H

#include <stddef.h>
#include <stdint.h>
#include "llmcodec_ggml_quants.h"   /* ENUM CANONICO ggml-type (CODEC, §0bis) */

#ifdef __cplusplus
extern "C" {
#endif

/* Alias ai nomi storici del moe → enum canonico del codec (una sola tabella). */
#define LLMENGINE_MOE_GGML_F32     LLMCODEC_GGML_T_F32
#define LLMENGINE_MOE_GGML_F16     LLMCODEC_GGML_T_F16
#define LLMENGINE_MOE_GGML_Q4_0    LLMCODEC_GGML_T_Q4_0
#define LLMENGINE_MOE_GGML_Q4_1    LLMCODEC_GGML_T_Q4_1
#define LLMENGINE_MOE_GGML_Q5_0    LLMCODEC_GGML_T_Q5_0
#define LLMENGINE_MOE_GGML_Q5_1    LLMCODEC_GGML_T_Q5_1
#define LLMENGINE_MOE_GGML_Q8_0    LLMCODEC_GGML_T_Q8_0
#define LLMENGINE_MOE_GGML_Q2_K    LLMCODEC_GGML_T_Q2_K
#define LLMENGINE_MOE_GGML_Q3_K    LLMCODEC_GGML_T_Q3_K
#define LLMENGINE_MOE_GGML_Q4_K    LLMCODEC_GGML_T_Q4_K
#define LLMENGINE_MOE_GGML_Q5_K    LLMCODEC_GGML_T_Q5_K
#define LLMENGINE_MOE_GGML_Q6_K    LLMCODEC_GGML_T_Q6_K
#define LLMENGINE_MOE_GGML_Q8_K    LLMCODEC_GGML_T_Q8_K
#define LLMENGINE_MOE_GGML_IQ2_XXS LLMCODEC_GGML_T_IQ2_XXS

const char* llmengine_moe_ggml_type_name(uint32_t type);
/* Elementi per blocco (1 per F16/F32, 32 per Q4_0/Q8_0, 256 per i K-quant). 0 se ignoto. */
uint32_t llmengine_moe_ggml_blck_size(uint32_t type);
/* Byte per blocco. 0 se ignoto. */
size_t llmengine_moe_ggml_type_size(uint32_t type);
/* Dequantizza `n_elem` elementi da `src` (blocchi contigui) in `dst` (f32).
 * `n_elem` deve essere multiplo di blck_size. Ritorna 0 su successo, <0 se il tipo non è
 * supportato o i parametri sono invalidi. */
int llmengine_moe_ggml_dequant_row(uint32_t type, const void* src, float* dst, size_t n_elem);

/* Byte occupati da `n_elem` elementi (n_elem/blck*type_size). 0 se tipo ignoto o non allineato. */
size_t llmengine_moe_ggml_row_bytes(uint32_t type, size_t n_elem);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_GGML_H */
