/* llm_engine_container.c — vedi llm_engine_container.h. Orchestra il forward leggendo un GGUF
 * (reader `llm_gguf.h`) e dequantando coi decoder del CODEC; nessun dequant di formato qui (§0bis). */
#include "llm_engine_container.h"
#include "llmcodec_ggml_quants.h"   /* llmcodec_ggml_dequant_row / llmcodec_ggml_type_bytes */

#include <stdio.h>
#include <string.h>

LlmEngineMoeStatus llm_engine_read_dense_f32(const LlmEngineMoeGguf* g, const char* name,
                                       float* out, size_t cap, size_t* n_out) {
    if (n_out) *n_out = 0;
    if (g == NULL || name == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    int i = llm_gguf_find_tensor(g, name);
    if (i < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    int64_t n = 0;
    const uint8_t* p = llm_gguf_tensor_data(g, i, &n);
    if (p == NULL || n <= 0) return LLMENGINE_MOE_ERR_FORMAT;
    if ((size_t)n > cap) return LLMENGINE_MOE_ERR_NO_MEMORY;
    if (llmcodec_ggml_dequant_row(g->ti[i].type, p, out, n) != 0) return LLMENGINE_MOE_ERR_FORMAT;
    if (n_out) *n_out = (size_t)n;
    return LLMENGINE_MOE_OK;
}

/* Esperto `expert` di un tensore 3D `[ne0(cols), ne1(rows), ne2(E)]` (l'esperto è il dim più lento
 * → blocco contiguo): dequant di ne0*ne1 valori all'offset expert * type_bytes(ne0*ne1). */
static LlmEngineMoeStatus eng_expert_mat(const LlmEngineMoeGguf* g, const char* name, uint32_t expert,
                                   float* out, size_t cap, uint32_t* rows, uint32_t* cols) {
    int i = llm_gguf_find_tensor(g, name);
    if (i < 0) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    const LlmCodecGgufTensor* t = &g->ti[i];
    if (t->ndim < 3) return LLMENGINE_MOE_ERR_FORMAT;
    uint64_t n0 = (uint64_t)t->dims[0], n1 = (uint64_t)t->dims[1], ne = (uint64_t)t->dims[2];
    if ((uint64_t)expert >= ne) return LLMENGINE_MOE_ERR_INVALID;
    uint64_t nv = n0 * n1, per = 0;
    if (llmcodec_ggml_type_bytes(t->type, nv, &per) != 0 || per == 0) return LLMENGINE_MOE_ERR_FORMAT;
    int64_t tot = 0; const uint8_t* base = llm_gguf_tensor_data(g, i, &tot);
    if (base == NULL) return LLMENGINE_MOE_ERR_FORMAT;
    if ((uint64_t)expert * per + per > (uint64_t)tot) return LLMENGINE_MOE_ERR_FORMAT;
    if (nv > cap) return LLMENGINE_MOE_ERR_NO_MEMORY;
    if (llmcodec_ggml_dequant_row(t->type, base + (uint64_t)expert * per, out, (int64_t)nv) != 0) return LLMENGINE_MOE_ERR_FORMAT;
    if (rows) *rows = (uint32_t)n1;
    if (cols) *cols = (uint32_t)n0;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llm_engine_read_expert_f32(const LlmEngineMoeGguf* g, int layer, uint32_t expert,
                                        float* gate, size_t gate_cap,
                                        float* up, size_t up_cap,
                                        float* down, size_t down_cap,
                                        uint32_t* rows_out, uint32_t* cols_out) {
    if (g == NULL) return LLMENGINE_MOE_ERR_INVALID;
    char nm[128]; LlmEngineMoeStatus s;
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", layer);
    s = eng_expert_mat(g, nm, expert, gate, gate_cap, rows_out, cols_out); if (s != LLMENGINE_MOE_OK) return s;
    snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", layer);
    s = eng_expert_mat(g, nm, expert, up, up_cap, NULL, NULL);             if (s != LLMENGINE_MOE_OK) return s;
    snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", layer);
    s = eng_expert_mat(g, nm, expert, down, down_cap, NULL, NULL);         if (s != LLMENGINE_MOE_OK) return s;
    return LLMENGINE_MOE_OK;
}
