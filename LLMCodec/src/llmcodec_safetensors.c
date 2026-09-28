/* llmcodec_safetensors.c — EMITTER (writer) safetensors per il model compiler (namespace LLM).
 *
 * Primo componente dell'EMITTER (§4bis del piano LLM-IR): scrive un artefatto **MLX/safetensors**
 * valido (header JSON + blob allineati). Formato: 8 byte u64 LE = lunghezza header, poi l'header
 * JSON {"nome":{"dtype":"F16","shape":[..],"data_offsets":[a,b]},...}, poi i byte dei tensori.
 * È la base per `LLM MODEL SAVE` (round-trip e, dopo le transform, emissione del modello nuovo).
 *
 * §0bis: implementazione reale (scrive byte), niente stub. Nessuna dipendenza esterna.
 * Self-check: `llmcodec_safetensors_selfcheck()` (round-trip write→read su un tensore noto).
 */
#include "llmcodec_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int st_dtype_size(const char* dtype) {
    if (strcmp(dtype, "F64") == 0) return 8;
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "F16") == 0) return 2;
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "I64") == 0) return 8;
    if (strcmp(dtype, "I32") == 0 || strcmp(dtype, "U32") == 0) return 4;
    if (strcmp(dtype, "I16") == 0) return 2;
    if (strcmp(dtype, "I8") == 0 || strcmp(dtype, "U8") == 0 || strcmp(dtype, "BOOL") == 0) return 1;
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E5M2") == 0 || strcmp(dtype, "F8_E8M0") == 0) return 1;
    return -1;
}

static int64_t st_shape_numel(const int* shape, int ndim) {
    if (ndim <= 0) return 1;
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= (shape[i] > 0 ? shape[i] : 1);
    return n;
}

/* Scrive `n` tensori in `path`. Ritorna 0 su successo, -1 su errore. */
int llmcodec_safetensors_write(const char* path, LlmCodecStTensor* tensors, int n) {
    if (path == NULL || tensors == NULL || n <= 0) return -1;

    /* 1. Calcola offset e valida i tensori. */
    int64_t off = 0;
    for (int i = 0; i < n; i++) {
        int es = st_dtype_size(tensors[i].dtype);
        if (es < 0 || tensors[i].data == NULL || tensors[i].name == NULL) return -1;
        tensors[i]._offset = off;                                  /* muta la copia locale */
        int64_t bytes = st_shape_numel(tensors[i].shape, tensors[i].ndim) * es;
        off += bytes;
    }

    /* 2. Costruisce l'header JSON. */
    size_t cap = (size_t)off + 4096 + (size_t)n * 256;
    char* json = (char*)malloc(cap);
    if (json == NULL) return -1;
    size_t p = 0;
    p += (size_t)snprintf(json + p, cap - p, "{");
    for (int i = 0; i < n; i++) {
        int es = st_dtype_size(tensors[i].dtype);
        int64_t numel = st_shape_numel(tensors[i].shape, tensors[i].ndim);
        p += (size_t)snprintf(json + p, cap - p, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                              i ? "," : "", tensors[i].name, tensors[i].dtype);
        for (int d = 0; d < tensors[i].ndim; d++)
            p += (size_t)snprintf(json + p, cap - p, "%s%d", d ? "," : "", tensors[i].shape[d]);
        p += (size_t)snprintf(json + p, cap - p, "],\"data_offsets\":[%lld,%lld]}",
                              (long long)tensors[i]._offset,
                              (long long)(tensors[i]._offset + numel * es));
    }
    p += (size_t)snprintf(json + p, cap - p, "}");

    /* 3. Allinea l'header a 8 byte (formato safetensors). */
    while ((p % 8) != 0) json[p++] = ' ';

    FILE* f = fopen(path, "wb");
    if (f == NULL) { free(json); return -1; }
    uint64_t hlen = (uint64_t)p;
    if (fwrite(&hlen, 1, 8, f) != 8) { fclose(f); free(json); return -1; }
    if (fwrite(json, 1, p, f) != p) { fclose(f); free(json); return -1; }
    for (int i = 0; i < n; i++) {
        int es = st_dtype_size(tensors[i].dtype);
        int64_t bytes = st_shape_numel(tensors[i].shape, tensors[i].ndim) * es;
        if (fwrite(tensors[i].data, 1, (size_t)bytes, f) != (size_t)bytes) {
            fclose(f); free(json); return -1;
        }
    }
    fclose(f);
    free(json);
    return 0;
}

/* ---- READER (componente 3-bis) ---- */
static int st_parse_value(const char* v, const char* vend, LlmCodecStTensorInfo* ti) {
    const char* d = strstr(v, "\"dtype\"");
    if (d == NULL || d >= vend) return -1;
    d = strchr(d, ':'); if (d == NULL) return -1;
    d = strchr(d, '"'); if (d == NULL) return -1;
    d++;
    const char* de = strchr(d, '"');
    if (de == NULL || de >= vend) return -1;
    size_t dl = (size_t)(de - d);
    if (dl >= sizeof(ti->dtype)) return -1;
    memcpy(ti->dtype, d, dl); ti->dtype[dl] = 0;

    const char* s = strstr(v, "\"shape\"");
    if (s == NULL || s >= vend) return -1;
    s = strchr(s, '['); if (s == NULL) return -1;
    s++;
    ti->ndim = 0;
    while (s < vend && *s && *s != ']') {
        while (*s == ' ' || *s == ',' || *s == '\n') s++;
        if (*s == ']') break;
        char* e = NULL;
        unsigned long long val = strtoull(s, &e, 10);
        if (e == s) break;
        if (ti->ndim < 8) ti->shape[ti->ndim++] = (uint64_t)val;
        s = e;
    }
    const char* o = strstr(v, "\"data_offsets\"");
    if (o == NULL || o >= vend) return -1;
    o = strchr(o, '['); if (o == NULL) return -1;
    o++;
    char* e = NULL;
    ti->begin = (uint64_t)strtoull(o, &e, 10); o = e;
    while (*o == ' ' || *o == ',') o++;
    ti->end = (uint64_t)strtoull(o, &e, 10);
    return 0;
}

int llmcodec_safetensors_read(const char* path, LlmCodecStTensorInfo** out_tensors, int* out_n,
                         uint64_t* out_data_base) {
    if (path == NULL || out_tensors == NULL || out_n == NULL) return -1;
    *out_tensors = NULL; *out_n = 0;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    uint64_t hlen = 0;
    if (fread(&hlen, 1, 8, f) != 8 || hlen == 0 || hlen > (64u << 20)) { fclose(f); return -1; }
    char* js = (char*)malloc((size_t)hlen + 1);
    if (js == NULL) { fclose(f); return -1; }
    if (fread(js, 1, (size_t)hlen, f) != (size_t)hlen) { free(js); fclose(f); return -1; }
    js[hlen] = 0;
    fclose(f);
    if (out_data_base) *out_data_base = 8 + hlen;

    int cap = 16, n = 0;
    LlmCodecStTensorInfo* arr = (LlmCodecStTensorInfo*)calloc((size_t)cap, sizeof(*arr));
    if (arr == NULL) { free(js); return -1; }

    const char* p = js;
    const char* end = js + hlen;
    if (p < end && *p == '{') p++;
    while (p < end && *p && *p != '}') {
        while (p < end && (*p == ' ' || *p == ',' || *p == '\n')) p++;
        if (*p != '"') break;
        const char* ks = ++p;
        const char* ke = strchr(ks, '"');
        if (ke == NULL) break;
        size_t kl = (size_t)(ke - ks);
        char name[256];
        if (kl >= sizeof(name)) break;
        memcpy(name, ks, kl); name[kl] = 0;
        p = ke + 1;
        while (p < end && *p != '{') { if (*p == '}') break; p++; }
        if (p >= end || *p != '{') break;
        /* trova la '}' di chiusura (con nesting) */
        const char* vs = p;
        int depth = 0;
        const char* ve = p;
        for (; ve < end; ve++) {
            if (*ve == '{') depth++;
            else if (*ve == '}') { depth--; if (depth == 0) break; }
        }
        if (strcmp(name, "__metadata__") != 0) {
            if (n >= cap) { cap *= 2; arr = (LlmCodecStTensorInfo*)realloc(arr, (size_t)cap * sizeof(*arr)); if (arr == NULL) { free(js); return -1; } }
            memset(&arr[n], 0, sizeof(arr[n]));
            snprintf(arr[n].name, sizeof(arr[n].name), "%s", name);
            if (st_parse_value(vs, ve, &arr[n]) == 0) n++;
        }
        p = (ve < end) ? ve + 1 : end;
    }
    free(js);
    *out_tensors = arr; *out_n = n;
    return 0;
}

void llmcodec_safetensors_info_free(LlmCodecStTensorInfo* t) { free(t); }

/* Legge l'header e verifica che il file sia un safetensors valido (self-check).
 * Ritorna il numero di tensori trovati (>=0) o -1. */
int llmcodec_safetensors_read_header(const char* path, char** out_json_dup) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    uint64_t hlen = 0;
    if (fread(&hlen, 1, 8, f) != 8 || hlen == 0 || hlen > (64u << 20)) { fclose(f); return -1; }
    char* json = (char*)malloc((size_t)hlen + 1);
    if (json == NULL) { fclose(f); return -1; }
    if (fread(json, 1, (size_t)hlen, f) != (size_t)hlen) { free(json); fclose(f); return -1; }
    json[hlen] = 0;
    fclose(f);
    /* conta i tensori = numero di "data_offsets" */
    int cnt = 0;
    for (const char* q = json; (q = strstr(q, "\"data_offsets\"")) != NULL; q++) cnt++;
    if (out_json_dup) *out_json_dup = json; else free(json);
    return cnt;
}
