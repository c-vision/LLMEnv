/* llmengine_moe_gguf.h — lettore del formato GGUF (header + metadati + tensor-info), solo METADATI.
 *
 * Serve all'auto-discovery da file `.gguf` (la maggior parte dei modelli locali): dimensioni,
 * varianti (rope/router/norm), nomi/ruoli dei tensori e tipo quant, SENZA leggere i dati tensore.
 * Non è un dequantizzatore: per l'ingest runtime (M6) i tipi ggml verranno dequantizzati a parte.
 * Implementazione in C, mai Python. Vedi `Todo/(open) - Model-Architecture-Auto-Discovery.md`. */
#ifndef LLMENGINE_MOE_GGUF_H
#define LLMENGINE_MOE_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "llmengine_moe_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tipi valore metadati GGUF. */
enum {
    LLMCODEC_GGUF_UINT8 = 0, LLMCODEC_GGUF_INT8, LLMCODEC_GGUF_UINT16, LLMCODEC_GGUF_INT16,
    LLMCODEC_GGUF_UINT32, LLMCODEC_GGUF_INT32, LLMCODEC_GGUF_FLOAT32, LLMCODEC_GGUF_BOOL,
    LLMCODEC_GGUF_STRING, LLMCODEC_GGUF_ARRAY, LLMCODEC_GGUF_UINT64, LLMCODEC_GGUF_INT64, LLMCODEC_GGUF_FLOAT64
};

typedef struct {
    char     key[128];
    uint32_t type;       /* LLMCODEC_GGUF_* */
    uint32_t arr_type;   /* se ARRAY */
    uint64_t arr_count;
    double   num;        /* tipi numerici e bool */
    char     str[256];   /* STRING */
} LlmEngineMoeGgufKv;

typedef struct {
    char     name[128];
    uint32_t n_dims;
    uint64_t dims[4];    /* ggml: ne0 (piu` veloce) .. ne3 */
    uint32_t type;       /* ggml_type */
    uint64_t offset;
} LlmCodecGgufTensor;

typedef struct {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    LlmEngineMoeGgufKv*     kvs;
    uint64_t       n_kv;
    LlmCodecGgufTensor* tensors;
    uint64_t       n_tensors;
    uint64_t alignment;
    uint64_t data_offset;   /* inizio del blocco dati tensori (allineato) */
} LlmEngineMoeGguf;

/* Legge header + metadati + tensor-info (NON i dati). Ritorna LLMENGINE_MOE_OK o un errore. */
LlmEngineMoeStatus llmengine_moe_gguf_read(const char* path, LlmEngineMoeGguf* out);
void         llmengine_moe_gguf_free(LlmEngineMoeGguf* g);

/* Cerca un metadato per chiave esatta. NULL se assente. */
const LlmEngineMoeGgufKv* llmengine_moe_gguf_find(const LlmEngineMoeGguf* g, const char* key);
/* Comodità: valore numerico (0 se assente). */
double llmengine_moe_gguf_num(const LlmEngineMoeGguf* g, const char* key, double def);

/* ── dati tensore (slice 1: F16/F32/Q8_0/Q4_0 via llmengine_moe_ggml) ──
 * `f` è il file GGUF riaperto dal chiamante (llmengine_moe_gguf_read lo chiude). Dequantizza l'INTERO
 * tensore `t` in `out` (capacity >= n. elementi). Ritorna LLMENGINE_MOE_OK o un errore esplicito
 * (es. tipo non supportato). */
LlmEngineMoeStatus llmengine_moe_gguf_tensor_f32(FILE* f, const LlmEngineMoeGguf* g, const LlmCodecGgufTensor* t, float* out, size_t out_cap);

/* Esperto `e` di un tensore 3D stacked [ne0, ne1, ne2] (e < ne2): dequant di ne0*ne1 elementi
 * (l'esperto è il dim più lento → blocco contiguo). Per il dequant-on-read per-esperto. */
LlmEngineMoeStatus llmengine_moe_gguf_expert_f32(FILE* f, const LlmEngineMoeGguf* g, const LlmCodecGgufTensor* t, uint32_t e, float* out, size_t out_cap);

/* Riga `row` di un tensore 2D [ne0, ne1]: dequant di ne0 elementi (ne0 è il dim più veloce
 * → la riga è contigua). Usato per embedding e lm_head. */
LlmEngineMoeStatus llmengine_moe_gguf_row_f32(FILE* f, const LlmEngineMoeGguf* g, const LlmCodecGgufTensor* t, uint64_t row, float* out, size_t out_cap);

/* Cerca la tensor-info per nome. NULL se assente. */
const LlmCodecGgufTensor* llmengine_moe_gguf_tensor(const LlmEngineMoeGguf* g, const char* name);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_GGUF_H */
