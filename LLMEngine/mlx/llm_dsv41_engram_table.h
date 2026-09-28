/* llm_dsv41_engram_table.h — lettura a RIGHE della tabella Engram (`engram.embed.weight` fp8
 * e4m3 + `engram.embed.scale` ue8m0) da un file safetensors, SENZA caricare l'intero tensore
 * (~200 GB nel release reale). Parsa l'header safetensors col cJSON e legge le righe richieste
 * via fseek/fread; dequantizza con `llmcodec_fp8_e4m3_to_f32` (e4m3) * 2^(byte-127) (ue8m0).
 *
 * Verifica: `llm_dsv41_engram_table_test.c` su un safetensors sintetico.
 */
#ifndef LLM_DSV41_ENGRAM_TABLE_H
#define LLM_DSV41_ENGRAM_TABLE_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <cJSON.h>

#include "llm_dsv41_fp8.h"

typedef struct {
    FILE* fp;
    size_t w_off;      /* offset assoluto dei dati di weight */
    size_t s_off;      /* offset assoluto dei dati di scale */
    int rows, hd, block, s_per_row;
} LlmDsv41EngramTable;

static inline uint64_t llm_dsv41_le_u64(const uint8_t* p) {
    uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

/* Apre la tabella: cerca `wname` (fp8, [rows,hd]) e `sname` (ue8m0, [rows,hd/block]). */
static inline bool llm_dsv41_engram_table_open(LlmDsv41EngramTable* t, const char* path,
                                               const char* wname, const char* sname, int block) {
    memset(t, 0, sizeof *t);
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint8_t hl[8];
    if (fread(hl, 1, 8, f) != 8) { fclose(f); return false; }
    uint64_t hlen = llm_dsv41_le_u64(hl);
    char* hbuf = (char*)malloc((size_t)hlen + 1);
    if (!hbuf) { fclose(f); return false; }
    if (fread(hbuf, 1, (size_t)hlen, f) != (size_t)hlen) { free(hbuf); fclose(f); return false; }
    hbuf[hlen] = '\0';
    cJSON* root = cJSON_Parse(hbuf);
    free(hbuf);
    if (!root) { fclose(f); return false; }
    cJSON* jw = cJSON_GetObjectItemCaseSensitive(root, wname);
    cJSON* js = cJSON_GetObjectItemCaseSensitive(root, sname);
    if (!cJSON_IsObject(jw) || !cJSON_IsObject(js)) { cJSON_Delete(root); fclose(f); return false; }
    cJSON* jws = cJSON_GetObjectItemCaseSensitive(jw, "shape");
    cJSON* jss = cJSON_GetObjectItemCaseSensitive(js, "shape");
    cJSON* jwo = cJSON_GetObjectItemCaseSensitive(jw, "data_offsets");
    cJSON* jso = cJSON_GetObjectItemCaseSensitive(js, "data_offsets");
    if (!jws || !jss || !jwo || !jso) { cJSON_Delete(root); fclose(f); return false; }
    int rows = (int)cJSON_GetArrayItem(jws, 0)->valuedouble;
    int hd = (int)cJSON_GetArrayItem(jws, 1)->valuedouble;
    int srows = (int)cJSON_GetArrayItem(jss, 0)->valuedouble;
    int scols = (int)cJSON_GetArrayItem(jss, 1)->valuedouble;
    size_t base = (size_t)(8 + hlen);
    t->w_off = base + (size_t)cJSON_GetArrayItem(jwo, 0)->valuedouble;
    t->s_off = base + (size_t)cJSON_GetArrayItem(jso, 0)->valuedouble;
    cJSON_Delete(root);
    if (rows != srows || scols != (hd / block)) { fclose(f); return false; }
    t->fp = f; t->rows = rows; t->hd = hd; t->block = block; t->s_per_row = scols;
    return true;
}

static inline void llm_dsv41_engram_table_close(LlmDsv41EngramTable* t) { if (t->fp) fclose(t->fp); t->fp = NULL; }

/* Dequant riga `row`: out[hd] = fp8(w[i]) * 2^(scale[i/block]-127). */
static inline bool llm_dsv41_engram_table_row(LlmDsv41EngramTable* t, int row, float* out) {
    if (!t->fp || row < 0 || row >= t->rows) return false;
    uint8_t* w = (uint8_t*)malloc((size_t)t->hd);
    uint8_t* s = (uint8_t*)malloc((size_t)t->s_per_row);
    if (!w || !s) { free(w); free(s); return false; }
    if (fseek(t->fp, (long)(t->w_off + (size_t)row * t->hd), SEEK_SET) != 0 ||
        fread(w, 1, (size_t)t->hd, t->fp) != (size_t)t->hd ||
        fseek(t->fp, (long)(t->s_off + (size_t)row * t->s_per_row), SEEK_SET) != 0 ||
        fread(s, 1, (size_t)t->s_per_row, t->fp) != (size_t)t->s_per_row) {
        free(w); free(s); return false;
    }
    for (int i = 0; i < t->hd; i++) {
        float sc = llmcodec_e8m0_to_f32(s[i / t->block]);
        out[i] = llmcodec_fp8_e4m3_to_f32(w[i]) * sc;
    }
    free(w); free(s);
    return true;
}

#endif /* LLM_DSV41_ENGRAM_TABLE_H */
