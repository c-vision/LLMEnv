/* llmcodec_safetensors.h — EMITTER safetensors (writer + header reader) per il model compiler LLM. */
#ifndef LLMCODEC_SAFETENSORS_H
#define LLMCODEC_SAFETENSORS_H
#include <stdint.h>

typedef struct {
    const char* name;
    const char* dtype;   /* "F64","F32","F16","BF16","I64","I32","I16","I8","U8","BOOL" */
    const int*  shape;
    int         ndim;
    const void* data;
    int64_t     _offset; /* interno: scritto da llmcodec_safetensors_write */
} LlmCodecStTensor;

/* Scrive `n` tensori in `path` (formato safetensors). 0 ok / -1 errore. */
int llmcodec_safetensors_write(const char* path, LlmCodecStTensor* tensors, int n);

/* Legge l'header; ritorna il n. di tensori (>=0) o -1. Se out_json_dup != NULL,
 * assegna (malloc) il JSON dell'header al chiamante. */
int llmcodec_safetensors_read_header(const char* path, char** out_json_dup);

/* ---- READER (componente 3-bis): tabella tensori dall'header JSON ---- */
typedef struct {
    char     name[256];
    char     dtype[16];
    int      ndim;
    uint64_t shape[8];
    uint64_t begin;   /* data_offsets[0] (relativo alla data section) */
    uint64_t end;     /* data_offsets[1] */
} LlmCodecStTensorInfo;

/* Legge la tabella tensori + l'offset assoluto della data section. 0 ok / -1.
 * Alloca out_tensors (liberare con llmcodec_safetensors_info_free). */
int llmcodec_safetensors_read(const char* path, LlmCodecStTensorInfo** out_tensors, int* out_n,
                         uint64_t* out_data_base);
void llmcodec_safetensors_info_free(LlmCodecStTensorInfo* t);

#endif
