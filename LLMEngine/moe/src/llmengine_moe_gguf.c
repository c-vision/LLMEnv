/* llmengine_moe_gguf.c — lettore GGUF (metadati + tensor-info). Vedi include/llmengine_moe_gguf.h. */
#include "llmengine_moe_gguf.h"
#include "llmcodec_gguf_read.h"   /* parsing GGUF canonico (CODEC, §0bis) */
#include "llmengine_moe_ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LlmEngineMoeStatus llmengine_moe_gguf_read(const char* path, LlmEngineMoeGguf* out) {
    /* UNA SOLA SORGENTE (§0bis): il PARSING GGUF vive nel CODEC (llmcodec_gguf_read_meta); qui si adatta
     * al formato del runtime MoE (LlmEngineMoeGguf/LlmEngineMoeGgufKv). Prima era reimplementato qui. */
    if (path == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    LlmCodecGgufMetaKVOwned* kv = NULL; int n = 0; LlmCodecGgufFileInfo info; memset(&info, 0, sizeof info);
    if (llmcodec_gguf_read_meta(path, &kv, &n, &info) != 0) return LLMENGINE_MOE_ERR_FORMAT;
    out->version = info.version;
    out->tensor_count = info.n_tensors; out->kv_count = (uint64_t)n;
    out->alignment = info.alignment ? info.alignment : 32;
    out->data_offset = info.data_start;
    if (n > 0) {
        out->kvs = (LlmEngineMoeGgufKv*)calloc((size_t)n, sizeof(LlmEngineMoeGgufKv));
        if (out->kvs == NULL) { llmcodec_gguf_meta_free(kv, n); llmcodec_gguf_free_info(&info); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        for (int i = 0; i < n; ++i) {
            LlmEngineMoeGgufKv* dd = &out->kvs[i]; LlmCodecGgufMetaKVOwned* ss = &kv[i];
            snprintf(dd->key, sizeof dd->key, "%s", ss->key ? ss->key : "");
            dd->type = (uint32_t)ss->type; dd->arr_type = (uint32_t)ss->arr_type; dd->arr_count = ss->arr_count;
            switch (ss->type) {
                case 8:  if (ss->str) snprintf(dd->str, sizeof dd->str, "%s", ss->str); break;   /* STRING */
                case 1: case 3: case 5: dd->num = (double)ss->i32; break;                        /* I8/I16/I32 */
                case 11: dd->num = (double)ss->i64; break;                                       /* I64 */
                case 6:  dd->num = (double)ss->f32; break;                                       /* F32 */
                case 12: dd->num = ss->f64; break;                                               /* F64 */
                case 7:  dd->num = (double)ss->b; break;                                         /* BOOL */
                case 10: dd->num = (double)ss->u64; break;                                       /* U64 */
                default: dd->num = (double)ss->u32; break;                                       /* U8/I16/U32/ARRAY */
            }
            out->n_kv++;
        }
    }
    if (info.n_tensors > 0) {
        out->tensors = (LlmCodecGgufTensor*)calloc((size_t)info.n_tensors, sizeof(LlmCodecGgufTensor));
        if (out->tensors == NULL) { llmcodec_gguf_meta_free(kv, n); llmcodec_gguf_free_info(&info); llmengine_moe_gguf_free(out); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        for (uint64_t i = 0; i < info.n_tensors; ++i) {
            LlmCodecGgufTensor* dd = &out->tensors[i]; LlmCodecGgufTensorInfo* ss = &info.tensors[i];
            snprintf(dd->name, sizeof dd->name, "%s", ss->name);
            dd->n_dims = (uint32_t)ss->ndim;
            for (int q = 0; q < ss->ndim && q < 4; ++q) dd->dims[q] = ss->dims[q];
            dd->type = (uint32_t)ss->ggml_type; dd->offset = ss->offset;
            out->n_tensors++;
        }
    }
    llmcodec_gguf_meta_free(kv, n); llmcodec_gguf_free_info(&info);
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_gguf_free(LlmEngineMoeGguf* g) {
    if (g == NULL) return;
    free(g->kvs); free(g->tensors);
    g->kvs = NULL; g->tensors = NULL;
    g->n_kv = 0; g->n_tensors = 0;
}

const LlmEngineMoeGgufKv* llmengine_moe_gguf_find(const LlmEngineMoeGguf* g, const char* key) {
    if (g == NULL || key == NULL) return NULL;
    for (uint64_t i = 0; i < g->n_kv; ++i)
        if (strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    return NULL;
}

double llmengine_moe_gguf_num(const LlmEngineMoeGguf* g, const char* key, double def) {
    const LlmEngineMoeGgufKv* kv = llmengine_moe_gguf_find(g, key);
    if (kv == NULL) return def;
    if (kv->type == LLMCODEC_GGUF_STRING) return strtod(kv->str, NULL);
    return kv->num;
}

const LlmCodecGgufTensor* llmengine_moe_gguf_tensor(const LlmEngineMoeGguf* g, const char* name) {
    if (g == NULL || name == NULL) return NULL;
    for (uint64_t i = 0; i < g->n_tensors; ++i)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

static uint64_t tensor_n_elem(const LlmCodecGgufTensor* t) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) n *= t->dims[d];
    return n;
}

LlmEngineMoeStatus llmengine_moe_gguf_tensor_f32(FILE* f, const LlmEngineMoeGguf* g, const LlmCodecGgufTensor* t, float* out, size_t out_cap) {
    if (f == NULL || g == NULL || t == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    uint64_t n = tensor_n_elem(t);
    if (n == 0 || n > out_cap) return LLMENGINE_MOE_ERR_INVALID;
    size_t nbytes = llmengine_moe_ggml_row_bytes(t->type, (size_t)n);
    if (nbytes == 0) return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;   /* tipo non supportato/blocco non allineato */
    uint8_t* buf = (uint8_t*)malloc(nbytes);
    if (buf == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    if (fseek(f, (long)(g->data_offset + t->offset), SEEK_SET) != 0 ||
        fread(buf, 1, nbytes, f) != nbytes) { free(buf); return LLMENGINE_MOE_ERR_IO; }
    int rc = llmengine_moe_ggml_dequant_row(t->type, buf, out, (size_t)n);
    free(buf);
    return rc == 0 ? LLMENGINE_MOE_OK : LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
}

LlmEngineMoeStatus llmengine_moe_gguf_expert_f32(FILE* f, const LlmEngineMoeGguf* g, const LlmCodecGgufTensor* t, uint32_t e, float* out, size_t out_cap) {
    if (f == NULL || g == NULL || t == NULL || out == NULL || t->n_dims < 3) return LLMENGINE_MOE_ERR_INVALID;
    uint64_t per = t->dims[0] * t->dims[1];
    if (e >= t->dims[2] || per == 0 || per > out_cap) return LLMENGINE_MOE_ERR_INVALID;
    size_t ebytes = llmengine_moe_ggml_row_bytes(t->type, (size_t)per);
    if (ebytes == 0) return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    uint8_t* buf = (uint8_t*)malloc(ebytes);
    if (buf == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    if (fseek(f, (long)(g->data_offset + t->offset + (uint64_t)e * ebytes), SEEK_SET) != 0 ||
        fread(buf, 1, ebytes, f) != ebytes) { free(buf); return LLMENGINE_MOE_ERR_IO; }
    int rc = llmengine_moe_ggml_dequant_row(t->type, buf, out, (size_t)per);
    free(buf);
    return rc == 0 ? LLMENGINE_MOE_OK : LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
}

LlmEngineMoeStatus llmengine_moe_gguf_row_f32(FILE* f, const LlmEngineMoeGguf* g, const LlmCodecGgufTensor* t, uint64_t row, float* out, size_t out_cap) {
    if (f == NULL || g == NULL || t == NULL || out == NULL || t->n_dims < 1) return LLMENGINE_MOE_ERR_INVALID;
    uint64_t cols = t->dims[0];
    if (t->n_dims >= 2 && row >= t->dims[1]) return LLMENGINE_MOE_ERR_INVALID;
    if (cols == 0 || cols > out_cap) return LLMENGINE_MOE_ERR_INVALID;
    size_t rbytes = llmengine_moe_ggml_row_bytes(t->type, (size_t)cols);
    if (rbytes == 0) return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    uint8_t* buf = (uint8_t*)malloc(rbytes);
    if (buf == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    if (fseek(f, (long)(g->data_offset + t->offset + row * rbytes), SEEK_SET) != 0 ||
        fread(buf, 1, rbytes, f) != rbytes) { free(buf); return LLMENGINE_MOE_ERR_IO; }
    int rc = llmengine_moe_ggml_dequant_row(t->type, buf, out, (size_t)cols);
    free(buf);
    return rc == 0 ? LLMENGINE_MOE_OK : LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
}
