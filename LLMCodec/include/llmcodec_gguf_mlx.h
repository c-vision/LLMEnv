/* llmcodec_gguf_mlx.h — conversioni GGML ↔ MLX affine (primitive per-tensore).
 *
 * GGML → MLX: dequantizza un tensore GGML (any llmcodec_ggml_dequant_row type) in f32 e lo
 * ri-quantizza nel formato MLX affine (U32 packed + scales/biases F16).
 * MLX → GGML: dequantizza l'affine MLX in f32 e lo ri-quantizza in un tipo GGML.
 *
 * Sono le primitive del converter di formato; il converter a livello di MODELLO (per-dir) le usa. */
#ifndef LLMCODEC_GGUF_MLX_H
#define LLMCODEC_GGUF_MLX_H

#include <stdint.h>
#include "llmcodec_rolemap.h"
#include "llmcodec_gguf_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGML → MLX affine. `src` = byte packed del tipo `ggml_type` (rows×cols valori).
 * Scrive `w_out` (U32 packed, rows*ceil(cols*bits/32)), `sc_out`/`bs_out` (F16, rows*cols/group).
 * Ritorna 0 / -1 (tipo GGML non gestito, o parametri affine invalidi). */
int llmcodec_conv_ggml_to_mlx(int ggml_type, const void* src, uint32_t rows, uint32_t cols,
                         uint32_t group, uint32_t bits,
                         uint32_t* w_out, uint16_t* sc_out, uint16_t* bs_out);

/* MLX affine → GGML. `w` (U32 packed) + `sc`/`bs` (F16) = tensore rows×cols affine.
 * Scrive `dst` (byte packed del tipo `ggml_type`). Ritorna 0 / -1. */
int llmcodec_conv_mlx_to_ggml(const uint32_t* w, const uint16_t* sc, const uint16_t* bs,
                         uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                         int ggml_type, void* dst);

/* A2 — modello MLX (una dir con model.safetensors + config.json) -> GGUF.
 * `ggml_type` = tipo GGML per i pesi 2D affine; gli altri tensori sono copiati (F16/BF16/F32).
 * Ritorna 0 / -1. I pesi U32 affine sono accoppiati ai loro `.scales`/`.biases`. */
int llmcodec_mlx_to_gguf(const char* mlx_dir, const char* out_path, int ggml_type,
                    uint32_t group, uint32_t bits);
/* Variante role-aware: usa la mappa ruoli (dalla `.llmarch`) per arch/metadata e
 * per risolvere i suffissi di scala. `rm` puo' essere NULL (comportamento di default).
 * `tm` = tokenizer pre-digerito dal progetto tokenizer (NULL = nessun KV tokenizer). */
int llmcodec_mlx_to_gguf_ex(const char* mlx_dir, const char* out_path, int ggml_type,
                       uint32_t group, uint32_t bits, const LlmCodecRoleMap* rm, const LlmCodecTokenizerMeta* tm);
/* Come sopra, ma carica da se' la `.llmarch` (path esplicito). */
int llmcodec_conv_mlx_to_gguf_arch(const char* mlx_dir, const char* out_path, const char* llmarch_path,
                              int ggml_type, uint32_t group, uint32_t bits);
/* Come sopra, con tokenizer pre-digerito. */
int llmcodec_conv_mlx_to_gguf_arch_ex(const char* mlx_dir, const char* out_path, const char* llmarch_path,
                                 int ggml_type, uint32_t group, uint32_t bits, const LlmCodecTokenizerMeta* tm);

/* A1 — GGUF -> modello MLX (dir di output: model.safetensors + config.json con `quantization`).
 * I tensori 2D `.weight` sono ri-quantizzati affine (group/bits); il resto copiato F32/F16. */
int llmcodec_gguf_to_mlx(const char* gguf_path, const char* out_dir, uint32_t group, uint32_t bits);
/* Variante role-aware: scrive `architectures`/`model_type` dal role map nel config.json. */
int llmcodec_gguf_to_mlx_ex(const char* gguf_path, const char* out_dir, uint32_t group, uint32_t bits, const LlmCodecRoleMap* rm);
int llmcodec_conv_gguf_to_mlx_arch(const char* gguf_path, const char* out_dir, const char* llmarch_path,
                              uint32_t group, uint32_t bits);

#ifdef __cplusplus
}
#endif
#endif /* LLMCODEC_GGUF_MLX_H */
