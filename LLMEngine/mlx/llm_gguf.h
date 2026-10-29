/* llm_gguf.h — lettore GGUF (v3) per DS4/DeepSeek V4 Flash: header, key/value (scalari+stringhe+
 * array), tensor info. Data non copiata (puntatore al mmap). Header-only, C puro.
 *
 * IL PARSING è UNA SOLA SORGENTE (§0bis): `llm_gguf_parse` delega a `llmcodec_gguf_parse_mem` del CODEC
 * (`llmengine_moe_gguf_read.h`) e mappa nel formato dell'engine (LlmGgufVal tipizzati, array fissi, mmap).
 * Verifica: llm_gguf_test.c. */
#ifndef LLM_GGUF_H
#define LLM_GGUF_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "llm_ggml.h"
#include "llmcodec_gguf_read.h"   /* PARSER GGUF canonico del CODEC (llmcodec_gguf_parse_mem) */

#define LLM_GGUF_MAXKV   1024
#define LLM_GGUF_MAXTI   4096
#define LLM_GGUF_KEYSZ   128
#define LLM_GGUF_NAMESZ  160
#define LLM_GGUF_MAXDIM  4

/* Perché l'apertura ha detto "no". */
enum {
    LLMCODEC_GGUF_ERR_NONE = 0,
    LLMCODEC_GGUF_ERR_OPEN_FAILED,
    LLMCODEC_GGUF_ERR_STAT_FAILED,
    LLMCODEC_GGUF_ERR_MMAP_FAILED,
    LLMCODEC_GGUF_ERR_BAD_MAGIC,
    LLMCODEC_GGUF_ERR_UNSUPPORTED_VERSION,
    LLMCODEC_GGUF_ERR_TOO_MANY_KV,
    LLMCODEC_GGUF_ERR_TOO_MANY_TENSORS,
    LLMCODEC_GGUF_ERR_KEY_TOO_LONG,
    LLMCODEC_GGUF_ERR_NAME_TOO_LONG,
    LLMCODEC_GGUF_ERR_BAD_NDIM,
    LLMCODEC_GGUF_ERR_TRUNCATED,
    LLMCODEC_GGUF_ERR_DATA_PAST_EOF
};
typedef int LlmGgufErr;

static inline const char* llm_gguf_err_str(LlmGgufErr e, uint64_t found, char* buf, size_t n) {
    if (buf == NULL || n == 0) return "";
    switch (e) {
        case LLMCODEC_GGUF_ERR_NONE: snprintf(buf, n, "nessun errore"); break;
        case LLMCODEC_GGUF_ERR_OPEN_FAILED: snprintf(buf, n, "open fallita (errno %d)", (int)found); break;
        case LLMCODEC_GGUF_ERR_STAT_FAILED: snprintf(buf, n, "fstat fallita o file di dimensione nulla"); break;
        case LLMCODEC_GGUF_ERR_MMAP_FAILED: snprintf(buf, n, "mmap fallita (errno %d)", (int)found); break;
        case LLMCODEC_GGUF_ERR_BAD_MAGIC: snprintf(buf, n, "magic non GGUF"); break;
        case LLMCODEC_GGUF_ERR_UNSUPPORTED_VERSION:
            snprintf(buf, n, "versione GGUF non supportata: %llu", (unsigned long long)found); break;
        case LLMCODEC_GGUF_ERR_TOO_MANY_KV:
            snprintf(buf, n, "trovate %llu chiavi KV, massimo %d", (unsigned long long)found, LLM_GGUF_MAXKV); break;
        case LLMCODEC_GGUF_ERR_TOO_MANY_TENSORS:
            snprintf(buf, n, "trovati %llu tensori, massimo %d", (unsigned long long)found, LLM_GGUF_MAXTI); break;
        case LLMCODEC_GGUF_ERR_KEY_TOO_LONG:
            snprintf(buf, n, "chiave di %llu byte, massimo %d", (unsigned long long)found, LLM_GGUF_KEYSZ - 1); break;
        case LLMCODEC_GGUF_ERR_NAME_TOO_LONG:
            snprintf(buf, n, "nome di tensore di %llu byte, massimo %d", (unsigned long long)found, LLM_GGUF_NAMESZ - 1); break;
        case LLMCODEC_GGUF_ERR_BAD_NDIM:
            snprintf(buf, n, "ndim=%llu fuori da 0..%d", (unsigned long long)found, LLM_GGUF_MAXDIM); break;
        case LLMCODEC_GGUF_ERR_TRUNCATED: snprintf(buf, n, "header troncato"); break;
        case LLMCODEC_GGUF_ERR_DATA_PAST_EOF: snprintf(buf, n, "data section oltre la fine del file"); break;
        default: snprintf(buf, n, "errore sconosciuto (%d)", e); break;
    }
    return buf;
}

/* GGUF value types (numerazione = GGUF spec, la stessa del codec). */
enum { GGU8=0, GGI8=1, GGU16=2, GGI16=3, GGU32=4, GGI32=5, GGF32=6, GGBOOL=7,
       GGSTR=8, GGARR=9, GGU64=10, GGI64=11, GGF64=12 };

typedef struct {
    int type;
    uint64_t u;
    double f;
    const char* s;   /* stringa (owned dal reader; NUL-terminata) */
    size_t slen;
} LlmGgufVal;

typedef struct {
    char name[LLM_GGUF_NAMESZ];
    int ndim;
    int64_t dims[LLM_GGUF_MAXDIM];
    int type;             /* ggml_type */
    uint64_t offset;      /* offset nel data blob */
} LlmCodecGgufTensor;

typedef struct {
    uint32_t version;
    uint64_t n_tensors, n_kv;
    char keys[LLM_GGUF_MAXKV][LLM_GGUF_KEYSZ];
    LlmGgufVal vals[LLM_GGUF_MAXKV];
    int nkv;
    LlmCodecGgufTensor ti[LLM_GGUF_MAXTI];
    int nti;
    LlmGgufErr err;
    uint64_t err_found;
    uint64_t alignment;   /* `general.alignment` del file (default 32) */
    const uint8_t* data;  /* base del blob dati (dentro il buffer) */
    const uint8_t* buf;
    size_t size;
    LlmCodecGgufMetaKVOwned* _kvown;   /* KV owned del parser (per stringhe/array) — liberati alla chiusura */
    int _kvn;
} LlmEngineMoeGguf;

static inline bool llm_gguf_fail(LlmEngineMoeGguf* g, LlmGgufErr e, uint64_t found) {
    g->err = e; g->err_found = found; return false;
}
static inline const char* llm_gguf_errmsg(const LlmEngineMoeGguf* g, char* buf, size_t n) {
    return llm_gguf_err_str(g->err, g->err_found, buf, n);
}

/* PARSING (UNA SOLA SORGENTE): delega al parser del CODEC e mappa in LlmEngineMoeGguf. */
static inline bool llm_gguf_parse(LlmEngineMoeGguf* g, const uint8_t* buf, size_t size) {
    memset(g, 0, sizeof *g);
    LlmCodecGgufMetaKVOwned* kv = NULL; int n = 0; LlmCodecGgufFileInfo info; memset(&info, 0, sizeof info);
    if (llmcodec_gguf_parse_mem(buf, size, &kv, &n, &info) != 0) return llm_gguf_fail(g, LLMCODEC_GGUF_ERR_BAD_MAGIC, 0);
    if (info.n_kv > LLM_GGUF_MAXKV) { llmcodec_gguf_meta_free(kv, n); llmcodec_gguf_free_info(&info); return llm_gguf_fail(g, LLMCODEC_GGUF_ERR_TOO_MANY_KV, info.n_kv); }
    if (info.n_tensors > LLM_GGUF_MAXTI) { llmcodec_gguf_meta_free(kv, n); llmcodec_gguf_free_info(&info); return llm_gguf_fail(g, LLMCODEC_GGUF_ERR_TOO_MANY_TENSORS, info.n_tensors); }
    g->version = info.version; g->n_kv = info.n_kv; g->n_tensors = info.n_tensors;
    g->alignment = info.alignment ? info.alignment : 32;
    g->_kvown = kv; g->_kvn = n;
    for (int i = 0; i < n; i++) {
        LlmCodecGgufMetaKVOwned* s = &kv[i]; LlmGgufVal* v = &g->vals[i];
        snprintf(g->keys[i], LLM_GGUF_KEYSZ, "%s", s->key ? s->key : "");
        v->type = s->type;
        switch (s->type) {
            case GGU8: case GGU16: case GGU32: v->u = s->u32; break;
            case GGI8: case GGI16: case GGI32: v->u = (uint64_t)(int64_t)s->i32; break;
            case GGF32: v->f = (double)s->f32; break;
            case GGBOOL: v->u = (uint64_t)s->b; break;
            case GGSTR: v->s = s->str; v->slen = s->str ? strlen(s->str) : 0; break;
            case GGU64: v->u = s->u64; break;
            case GGI64: v->u = (uint64_t)s->i64; break;
            case GGF64: v->f = s->f64; break;
            default: break;   /* GGARR: campi lasciati a 0 (letti via kv_array_*) */
        }
        g->nkv = i + 1;
    }
    for (uint64_t i = 0; i < info.n_tensors; i++) {
        LlmCodecGgufTensorInfo* s = &info.tensors[i]; LlmCodecGgufTensor* t = &g->ti[i];
        snprintf(t->name, LLM_GGUF_NAMESZ, "%s", s->name);
        t->ndim = s->ndim;
        for (int d = 0; d < s->ndim && d < LLM_GGUF_MAXDIM; d++) t->dims[d] = (int64_t)s->dims[d];
        t->type = s->ggml_type; t->offset = s->offset;
        g->nti = (int)i + 1;
    }
    llmcodec_gguf_free_info(&info);
    size_t blob = (size_t)info.data_start;
    if (blob > size) { return llm_gguf_fail(g, LLMCODEC_GGUF_ERR_DATA_PAST_EOF, (uint64_t)blob); }
    g->buf = buf; g->size = size; g->data = buf + blob;
    return true;
}

/* KV array di stringhe per nome (i puntatori sono owned dal reader: validi fino alla chiusura).
 * Alloca strs e lens (da liberare). Ritorna il numero di elementi (>=0) o -1. */
static inline long llm_gguf_kv_array_str(const LlmEngineMoeGguf* g, const char* key,
                                         const char*** strs, size_t** lens) {
    if (!g->_kvown) return -1;
    for (int i = 0; i < g->nkv; i++) {
        if (strcmp(g->keys[i], key) != 0) continue;
        const LlmCodecGgufMetaKVOwned* kv = &g->_kvown[i];
        if (kv->type != GGARR || kv->arr_type != GGSTR) return -1;
        long n = kv->arr_n;
        const char** ss = (const char**)malloc(sizeof(char*) * (size_t)n);
        size_t* ll = (size_t*)malloc(sizeof(size_t) * (size_t)n);
        if ((!ss || !ll) && n) { free(ss); free(ll); return -1; }
        for (long j = 0; j < n; j++) { ss[j] = kv->arr[j]; ll[j] = kv->arr[j] ? strlen(kv->arr[j]) : 0; }
        *strs = ss; *lens = ll;
        return n;
    }
    return -1;
}

/* KV array di interi (GGI32), per nome. Alloca out (da liberare). n. elementi (>=0) o -1. */
static inline long llm_gguf_kv_array_i32(const LlmEngineMoeGguf* g, const char* key, int32_t** out) {
    if (!g->_kvown) return -1;
    for (int i = 0; i < g->nkv; i++) {
        if (strcmp(g->keys[i], key) != 0) continue;
        const LlmCodecGgufMetaKVOwned* kv = &g->_kvown[i];
        if (kv->type != GGARR || kv->arr_type != GGI32 || kv->arr_raw == NULL) return -1;
        long n = (long)kv->arr_count;
        int32_t* a = (int32_t*)malloc(sizeof(int32_t) * (size_t)n);
        if (!a && n) return -1;
        memcpy(a, kv->arr_raw, sizeof(int32_t) * (size_t)n);
        *out = a;
        return n;
    }
    return -1;
}

/* KV array di float32 (GGF32), per nome (p.es. tokenizer.ggml.scores). Alloca out (da liberare). n. elementi (>=0) o -1. */
static inline long llm_gguf_kv_array_f32(const LlmEngineMoeGguf* g, const char* key, float** out) {
    if (!g->_kvown) return -1;
    for (int i = 0; i < g->nkv; i++) {
        if (strcmp(g->keys[i], key) != 0) continue;
        const LlmCodecGgufMetaKVOwned* kv = &g->_kvown[i];
        if (kv->type != GGARR || kv->arr_type != GGF32 || kv->arr_raw == NULL) return -1;
        long n = (long)kv->arr_count;
        float* a = (float*)malloc(sizeof(float) * (size_t)n);
        if (!a && n) return -1;
        memcpy(a, kv->arr_raw, sizeof(float) * (size_t)n);
        *out = a;
        return n;
    }
    return -1;
}

static inline const LlmGgufVal* llm_gguf_find_val(const LlmEngineMoeGguf* g, const char* key) {
    for (int i = 0; i < g->nkv; i++) if (strcmp(g->keys[i], key) == 0) return &g->vals[i];
    return NULL;
}
static inline const LlmGgufVal* llm_gguf_find2(const LlmEngineMoeGguf* g, const char* a, const char* b) {
    const LlmGgufVal* v = llm_gguf_find_val(g, a);
    return v ? v : (b ? llm_gguf_find_val(g, b) : NULL);
}
static inline uint64_t llm_gguf_u(const LlmEngineMoeGguf* g, const char* k, uint64_t def, const char* k2) {
    const LlmGgufVal* v = llm_gguf_find2(g, k, k2);
    return v ? v->u : def;
}
static inline double llm_gguf_f(const LlmEngineMoeGguf* g, const char* k, double def, const char* k2) {
    const LlmGgufVal* v = llm_gguf_find2(g, k, k2);
    return v ? (v->type == GGF32 || v->type == GGF64 ? v->f : (double)v->u) : def;
}

/* Puntatore ai dati di un tensore + n. elementi; NULL se fuori dal blob o tipo ignoto. */
static inline const uint8_t* llm_gguf_tensor_data(const LlmEngineMoeGguf* g, int i, int64_t* n_out) {
    if (i < 0 || i >= g->nti) return NULL;
    const LlmCodecGgufTensor* t = &g->ti[i];
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->dims[d];
    int64_t nb = llm_ggml_nbytes(t->type, n);
    if (nb < 0) return NULL;
    size_t blob = (size_t)(g->data - g->buf);
    if (t->offset + (uint64_t)nb > (uint64_t)(g->size - blob)) return NULL;
    if (n_out) *n_out = n;
    return g->data + t->offset;
}

static inline bool llm_gguf_dequant_tensor(const LlmEngineMoeGguf* g, int i, float* dst) {
    int64_t n = 0;
    const uint8_t* p = llm_gguf_tensor_data(g, i, &n);
    if (!p) return false;
    return llmcodec_ggml_dequant_row(g->ti[i].type, p, dst, n) == 0;
}

static inline int llm_gguf_find_tensor(const LlmEngineMoeGguf* g, const char* name) {
    for (int i = 0; i < g->nti; i++) if (strcmp(g->ti[i].name, name) == 0) return i;
    return -1;
}

static inline float* llm_gguf_load_f32(const LlmEngineMoeGguf* g, const char* name, int64_t* n_out) {
    int i = llm_gguf_find_tensor(g, name);
    if (i < 0) return NULL;
    int64_t n = 0;
    const uint8_t* p = llm_gguf_tensor_data(g, i, &n);
    if (!p || n <= 0) return NULL;
    float* dst = (float*)malloc((size_t)n * sizeof(float));
    if (!dst) return NULL;
    if (llmcodec_ggml_dequant_row(g->ti[i].type, p, dst, n) != 0) { free(dst); return NULL; }
    if (n_out) *n_out = n;
    return dst;
}

static inline int64_t llm_gguf_tensor_nbytes(const LlmEngineMoeGguf* g, int i) {
    if (i < 0 || i >= g->nti) return -1;
    const LlmCodecGgufTensor* t = &g->ti[i];
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->dims[d];
    return llm_ggml_nbytes(t->type, n);
}

/* Mappa in sola lettura un file GGUF e lo parsa. `map`/`maplen` per la munmap. */
static inline bool llm_gguf_open_file(const char* path, LlmEngineMoeGguf* g, void** map, size_t* maplen) {
    memset(g, 0, sizeof *g); *map = NULL; *maplen = 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { g->err = LLMCODEC_GGUF_ERR_OPEN_FAILED; g->err_found = (uint64_t)errno; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        int e = errno; close(fd);
        g->err = LLMCODEC_GGUF_ERR_STAT_FAILED; g->err_found = (uint64_t)e; return false;
    }
    size_t sz = (size_t)st.st_size;
    void* m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { g->err = LLMCODEC_GGUF_ERR_MMAP_FAILED; g->err_found = (uint64_t)errno; return false; }
    if (!llm_gguf_parse(g, (const uint8_t*)m, sz)) { munmap(m, sz); return false; }
    *map = m; *maplen = sz;
    return true;
}
static inline void llm_gguf_close_file(LlmEngineMoeGguf* g, void* map, size_t maplen) {
    if (g->_kvown) { llmcodec_gguf_meta_free(g->_kvown, g->_kvn); g->_kvown = NULL; g->_kvn = 0; }
    if (map) munmap(map, maplen);
    g->buf = NULL; g->data = NULL;
}

#endif /* LLM_GGUF_H */
