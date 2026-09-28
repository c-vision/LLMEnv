/* llmcodec_gguf_writer.h — EMITTER GGUF (writer). Il READER canonico (metadata KV + tabella tensori) è
 * in `llmcodec_gguf_read.h` (separato per non esporre l'enum ggml / LlmCodecGgufTensor del writer). */
#ifndef LLMCODEC_GGUF_WRITER_H
#define LLMCODEC_GGUF_WRITER_H
#include <stdint.h>
#include "llmcodec_gguf_read.h"

/* GGML tensor types (sottoinsieme). */
enum { LLMCODEC_GGML_F32 = 0, LLMCODEC_GGML_F16 = 1, LLMCODEC_GGML_Q4_0 = 2, LLMCODEC_GGML_Q8_0 = 8 };

typedef struct {
    const char*     name;
    int             ggml_type;            /* LLMCODEC_GGML_F32/F16 */
    const uint64_t* dims;
    int             ndim;
    const void*     data;
    uint64_t        _offset;              /* interno */
} LlmCodecGgufTensor;

/* Scrive un GGUF v3 valido. 0 ok / -1 errore. `alignment` 0 -> 32. */
int llmcodec_gguf_write(const char* path, const LlmCodecGgufMetaKV* kv, int n_kv,
                   LlmCodecGgufTensor* ts, int n_ts, uint32_t alignment);

#endif
