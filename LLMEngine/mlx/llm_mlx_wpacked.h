/* llm_mlx_wpacked.h — PESI PACKED ZERO-COPY (M1, branch feat/mlx-packed-weights).
 *
 * Rappresenta un tensore quantizzato affine MLX (weight U32 packed + scales/biases F16) come
 * PUNTATORI al mmap della safetensors: nessun array MLX, nessuna copia. I kernel affine-packed
 * (M2) leggono questi blocchi direttamente.
 *
 * Contratto: `open` mappa il file una volta per shard e risolve <name>.weight/.scales/.biases;
 * i puntatori restano validi fino a `close`. Nessuna decodifica qui (solo layout + bounds). */
#ifndef LLM_MLX_WPACKED_H
#define LLM_MLX_WPACKED_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    const uint32_t* w;        /* packed [rows][cols*bits/32] */
    const uint16_t* scales;   /* [rows][cols/group] F16 */
    const uint16_t* biases;   /* [rows][cols/group] F16 (NULL se assenti) */
    uint32_t rows, cols;
    int bits, group;
    size_t w_bytes, s_bytes, b_bytes;
    void*  map;               /* mmap base (proprietario) */
    size_t map_bytes;
} LlmMlxPackedW;

/* 0 = ok; <0 errore (file/header/tensori mancanti o incoerenti). */
int  llm_mlx_packed_open(const char* safetensors_path, const char* base_name,
                         int bits, int group, LlmMlxPackedW* out);
void llm_mlx_packed_close(LlmMlxPackedW* w);

#endif
