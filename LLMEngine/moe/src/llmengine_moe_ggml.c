/* llmengine_moe_ggml.c — type/dequant ggml→f32: UNA SOLA SORGENTE nel CODEC (§0bis).
 * Prima reimplementava qui le tabelle (blck_size/type_size/row_bytes) e il dequant di ogni tipo
 * (F32/F16/Q8_0/Q4_0/Q6_K/Q4_K/Q5_K/Q2_K/IQ2_XXS). Ora delega a `llmcodec_ggml_*` del codec, che è
 * verificato byte-exact vs ggml. */
#include "llmengine_moe_ggml.h"
#include "llmcodec_ggml_quants.h"
#include "llmcodec_gguf_read.h"   /* llmcodec_ggml_type_bytes */

#include <string.h>

const char* llmengine_moe_ggml_type_name(uint32_t type) {
    static _Thread_local char b[32];
    if (llmcodec_ggml_type_name((int)type, b, sizeof b) != 0) return "unknown";
    return b;
}
uint32_t llmengine_moe_ggml_blck_size(uint32_t type) {
    int64_t qk = llmcodec_ggml_block_vals((int)type);
    return qk > 0 ? (uint32_t)qk : 0;
}
size_t llmengine_moe_ggml_type_size(uint32_t type) {
    uint32_t bs = llmengine_moe_ggml_blck_size(type);
    uint64_t b = 0;
    if (bs == 0 || llmcodec_ggml_type_bytes((int)type, (uint64_t)bs, &b) != 0) return 0;
    return (size_t)b;
}
size_t llmengine_moe_ggml_row_bytes(uint32_t type, size_t n_elem) {
    uint64_t b = 0;
    if (llmcodec_ggml_type_bytes((int)type, (uint64_t)n_elem, &b) != 0) return 0;
    return (size_t)b;
}
int llmengine_moe_ggml_dequant_row(uint32_t type, const void* src, float* dst, size_t n_elem) {
    return llmcodec_ggml_dequant_row((int)type, src, dst, (int64_t)n_elem);
}
