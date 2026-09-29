/* llmcodec_gguf_read.h — READER GGUF canonico (metadata KV + tabella tensori + data_start).
 * Separato da llmcodec_gguf_writer.h per NON esporre l'enum ggml / LlmCodecGgufTensor del writer (che
 * collidevano coi tipi omonimi del runtime MoE). UNA SOLA SORGENTE per il parsing GGUF (§0bis). */
#ifndef LLMCODEC_GGUF_READ_H
#define LLMCODEC_GGUF_READ_H
#include <stdint.h>
#include <stddef.h>
#include "llmcodec_gguf_status.h"   /* TASSONOMIA D'ERRORE (contratto stabile): LlmCodecGgufStatus */

#ifdef __cplusplus
extern "C" {
#endif

/* Tipi valore metadata GGUF (sottoinsieme; numerazione = GGUF spec). */
enum {
    LLMCODEC_GGUF_MT_U32 = 4, LLMCODEC_GGUF_MT_I32 = 5, LLMCODEC_GGUF_MT_F32 = 6,
    LLMCODEC_GGUF_MT_BOOL = 7, LLMCODEC_GGUF_MT_STRING = 8, LLMCODEC_GGUF_MT_ARRAY = 9, LLMCODEC_GGUF_MT_U64 = 10,
    LLMCODEC_GGUF_MT_I64 = 11, LLMCODEC_GGUF_MT_F64 = 12
};

typedef struct {
    const char* key;
    int         type;                     /* LLMCODEC_GGUF_MT_* / GGUF spec */
    uint32_t    u32;
    int32_t     i32;
    float       f32;
    uint64_t    u64;
    int         b;
    const char* str;
    const char* const* arr; int arr_n;
    int arr_type;
    const void* arr_raw; uint64_t arr_raw_bytes; uint64_t arr_count;
} LlmCodecGgufMetaKV;

/* Dimensione in byte dei `numel` valori di un tipo GGML (0/-1). */
int llmcodec_ggml_type_bytes(int ggml_type, uint64_t numel, uint64_t* out);

/* Legge magic/version/counts. 0 ok / -1. */
int llmcodec_gguf_read_header(const char* path, uint32_t* out_version,
                         uint64_t* out_n_ts, uint64_t* out_n_kv);

typedef struct {
    char     name[256];
    int      ggml_type;
    uint64_t dims[8];
    int      ndim;
    uint64_t offset;   /* relativo alla data section */
} LlmCodecGgufTensorInfo;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    LlmCodecGgufTensorInfo* tensors;   /* calloc(n_tensors) — liberare con llmcodec_gguf_free_info */
    uint64_t data_start;          /* offset assoluto della data section nel file */
    uint64_t alignment;
    uint64_t file_size;
} LlmCodecGgufFileInfo;

/* Legge metadata(scan)+tensor-infos. 0 ok / -1. Alloca out->tensors. */
int llmcodec_gguf_read_info(const char* path, LlmCodecGgufFileInfo* out);
/* Variante con TASSONOMIA d'errore: 0 = LLMCODEC_GGUF_STATUS_OK, altrimenti un codice
 * distinto (vedi llmcodec_gguf_status.h). I fallimenti di I/O/parse sono codificati, non
 * appiattiti su -1. */
LlmCodecGgufStatus llmcodec_gguf_read_info_status(const char* path, LlmCodecGgufFileInfo* out);
void llmcodec_gguf_free_info(LlmCodecGgufFileInfo* info);

typedef struct {
    char*    key;                 /* malloc */
    int      type;                /* GGUF spec */
    uint32_t u32; int32_t i32; float f32; uint64_t u64; int64_t i64; double f64; int b;
    char*    str;                 /* STRING: malloc (o NULL) */
    char**   arr; int arr_n;      /* ARRAY di stringhe: malloc (o NULL) */
    int arr_type; void* arr_raw; uint64_t arr_raw_bytes; uint64_t arr_count;
} LlmCodecGgufMetaKVOwned;

/* PARSER canonico buffer-based (UNA SORGENTE): read_meta/read_info lo chiamano su mmap. */
int llmcodec_gguf_parse_mem(const uint8_t* buf, size_t size, LlmCodecGgufMetaKVOwned** out_kv, int* out_n,
                       LlmCodecGgufFileInfo* out_info);
/* Variante con TASSONOMIA d'errore (stesso parser): ritorna LLMCODEC_GGUF_STATUS_*, 0 = OK.
 * È la porta che espone i codici distinti (TRUNCATED/OVERLAP/NAME_DUP/DATA_SHORT/...). */
LlmCodecGgufStatus llmcodec_gguf_parse_mem_status(const uint8_t* buf, size_t size,
                       LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info);

/* Legge i KV + le tensor-info (out_info opzionale). 0 ok / -1. */
int llmcodec_gguf_read_meta(const char* path, LlmCodecGgufMetaKVOwned** out_kv, int* out_n,
                       LlmCodecGgufFileInfo* out_info);
/* Variante con TASSONOMIA d'errore (mmappa il file e chiama parse_mem_status). */
LlmCodecGgufStatus llmcodec_gguf_read_meta_status(const char* path,
                       LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info);
void llmcodec_gguf_meta_free(LlmCodecGgufMetaKVOwned* kv, int n);

/* Serializza i KV posseduti nel formato del writer (per il round-trip/template). */
void llmcodec_gguf_meta_to_writer(const LlmCodecGgufMetaKVOwned* src, LlmCodecGgufMetaKV* dst);

#ifdef __cplusplus
}
#endif

#endif
