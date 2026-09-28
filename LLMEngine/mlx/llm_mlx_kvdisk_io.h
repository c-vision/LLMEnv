/* llm_mlx_kvdisk_io.h — serializzazione di array MLX <-> blob per il tier SSD KV (§G).
 *
 * Formato blob (little-endian, version 1):
 *   u32 magic '1DSKVDB0' -> 0x3042444B
 *   u32 version (1)
 *   i32 cache_offset
 *   i32 token_len
 *   u32 n_tensors
 *   ripetuto n_tensors:
 *     i32 layer ; u8 kind (0=k,1=v) ; u8 dtype_code ; u8 ndim ; u8 pad
 *     i32 shape[4] ; u64 nbytes ; u8 data[nbytes]
 *
 * dtype_code: 1=FLOAT16, 2=BFLOAT16, 3=FLOAT32 (altri -> non serializzabile).
 * `mlx_array_eval` PRIMA di leggere i dati; `mlx_array_new_data` COPIA il buffer
 * (quindi il blob temporaneo puo' essere liberato subito dopo il restore).
 * Dipende da mlx-c (quindi non e' puro); vedi llm_mlx_kvdisk.h per il tier disco puro.
 */
#ifndef LLM_MLX_KVDISK_IO_H
#define LLM_MLX_KVDISK_IO_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mlx/c/mlx.h"
#include "llm_mlx_kvdisk.h"

#define LLM_KVDISK_BLOB_MAGIC 0x3042444Bu  /* "KDB0" */
#define LLM_KVDISK_BLOB_VER   1u

static inline int llm_kvdisk_dtype_code(mlx_dtype d) {
    switch (d) {
        case MLX_FLOAT16:  return 1;
        case MLX_BFLOAT16: return 2;
        case MLX_FLOAT32:  return 3;
        case MLX_INT32:    return 4;
        default:           return 0;
    }
}
static inline mlx_dtype llm_kvdisk_dtype_from_code(int c) {
    switch (c) {
        case 1: return MLX_FLOAT16;
        case 2: return MLX_BFLOAT16;
        case 3: return MLX_FLOAT32;
        case 4: return MLX_INT32;
        default: return MLX_FLOAT32;
    }
}

/* Riserva `need` byte in piu' nel buffer (crescita geometrica). 0 ok / -1 OOM. */
static inline int llm_kvdisk_buf_reserve(uint8_t** buf, size_t* cap, size_t len, size_t need) {
    if (len + need <= *cap) return 0;
    size_t ncap = *cap ? *cap : 4096;
    while (ncap < len + need) ncap *= 2;
    uint8_t* nb = (uint8_t*)realloc(*buf, ncap);
    if (!nb) return -1;
    *buf = nb; *cap = ncap;
    return 0;
}

#define LLM_KVDISK_APPEND(dst, n, src) do { \
    if (llm_kvdisk_buf_reserve(buf, cap, *len, (n)) != 0) return -1; \
    memcpy(*buf + *len, (src), (n)); *len += (n); \
} while (0)

/* Appende (record + dati) di un tensore al blob. Ritorna 0 / -1 (dtype non serializzabile, OOM). */
static inline int llm_kvdisk_blob_append_tensor(uint8_t** buf, size_t* len, size_t* cap,
                                                mlx_array a, int layer, int kind) {
    int nd = mlx_array_ndim(a);
    if (nd < 1 || nd > 4) return -1;
    int dc = llm_kvdisk_dtype_code(mlx_array_dtype(a));
    if (dc == 0) return -1;
    if (mlx_array_eval(a) != 0) return -1;
    const int* sh = mlx_array_shape(a);
    const size_t n = mlx_array_size(a);
    const void* ptr = NULL;
    switch (dc) {
        case 1: ptr = (const void*)mlx_array_data_float16(a);  break;
        case 2: ptr = (const void*)mlx_array_data_bfloat16(a); break;
        case 3: ptr = (const void*)mlx_array_data_float32(a);  break;
        case 4: ptr = (const void*)mlx_array_data_int32(a);    break;
        default: return -1;
    }
    if (!ptr) return -1;
    size_t itemsize = (dc == 1 || dc == 2) ? 2u : 4u;
    uint64_t nbytes = (uint64_t)n * (uint64_t)itemsize;
    if (nbytes == 0) return -1;
    int32_t hdr[4] = { (int32_t)layer, (int32_t)kind, (int32_t)dc, (int32_t)nd };
    int32_t shape4[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < nd; i++) shape4[i] = (int32_t)sh[i];
    LLM_KVDISK_APPEND(buf, 16, hdr);
    LLM_KVDISK_APPEND(buf, 16, shape4);
    LLM_KVDISK_APPEND(buf, 8, &nbytes);
    LLM_KVDISK_APPEND(buf, (size_t)nbytes, ptr);
    return 0;
}

/* Legge il tensore all'offset *off, crea l'array (che COPIA i dati). Avanza *off. 0/-1. */
static inline int llm_kvdisk_blob_read_tensor(const uint8_t* buf, size_t len, size_t* off,
                                              mlx_array* out, int* layer, int* kind) {
    if (*off + 40 > len) return -1;
    int32_t hdr[4]; memcpy(hdr, buf + *off, 16); *off += 16;
    *layer = hdr[0]; *kind = hdr[1];
    int dc = hdr[2], ndim = hdr[3];
    int32_t shape4[4]; memcpy(shape4, buf + *off, 16); *off += 16;
    uint64_t nbytes = 0; memcpy(&nbytes, buf + *off, 8); *off += 8;
    if (*off + (size_t)nbytes > len) return -1;
    if (ndim < 1 || ndim > 4 || llm_kvdisk_dtype_code(llm_kvdisk_dtype_from_code(dc)) != dc) return -1;
    int shape[4] = { shape4[0], shape4[1], shape4[2], shape4[3] };
    *out = mlx_array_new_data(buf + *off, shape, ndim, llm_kvdisk_dtype_from_code(dc));
    *off += (size_t)nbytes;
    return 0;
}

#endif /* LLM_MLX_KVDISK_IO_H */
