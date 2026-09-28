/* llmcodec_gguf_parse.c — PARSER GGUF canonico (BUFFER-based): UNA SOLA SORGENTE (§0bis).
 * `llmcodec_gguf_parse_mem(buf,size,…)` è l'unico parser; `llmcodec_gguf_read_meta/read_info(path)` mmappano
 * il file e lo chiamano; l'engine DS4 (che mmappa già) lo chiama direttamente. Le stringhe/KV
 * sono COPIATE (owned): il chiamante può liberare il buffer dopo. */
#include "llmcodec_gguf_read.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct { const uint8_t* p; const uint8_t* end; int ok; } Cur;
static int cur_have(Cur* c, size_t n) { if (!c->ok || (size_t)(c->end - c->p) < n) { c->ok = 0; return 0; } return 1; }
static uint64_t cur_u(Cur* c, int nb) { uint64_t v = 0; if (!cur_have(c, (size_t)nb)) return 0; for (int i = 0; i < nb; i++) v |= (uint64_t)c->p[i] << (8*i); c->p += nb; return v; }
static const uint8_t* cur_bytes(Cur* c, size_t n) { if (!cur_have(c, n)) return NULL; const uint8_t* r = c->p; c->p += n; return r; }
static int mem_str(Cur* c, char** out) {
    uint64_t n = cur_u(c, 8); if (!c->ok || n > (1u << 24)) return -1;
    char* s = (char*)malloc((size_t)n + 1); if (!s) return -1;
    if (n) { const uint8_t* p = cur_bytes(c, (size_t)n); if (!p) { free(s); return -1; } memcpy(s, p, (size_t)n); }
    s[n] = 0; *out = s; return 0;
}
static void kv_free_fields(LlmCodecGgufMetaKVOwned* kv) {
    free(kv->key); free(kv->str);
    if (kv->arr) { for (int a = 0; a < kv->arr_n; a++) free(kv->arr[a]); free(kv->arr); }
    free(kv->arr_raw);
}
static int mem_value(Cur* c, uint32_t t, LlmCodecGgufMetaKVOwned* kv) {
    switch (t) {
        case 0: case 1: kv->u32 = (uint32_t)cur_u(c, 1); return c->ok ? 0 : -1;
        case 2: case 3: kv->u32 = (uint32_t)cur_u(c, 2); return c->ok ? 0 : -1;
        case LLMCODEC_GGUF_MT_U32: kv->u32 = (uint32_t)cur_u(c, 4); return c->ok ? 0 : -1;
        case LLMCODEC_GGUF_MT_I32: kv->i32 = (int32_t)cur_u(c, 4); return c->ok ? 0 : -1;
        case LLMCODEC_GGUF_MT_F32: { uint32_t u = (uint32_t)cur_u(c, 4); memcpy(&kv->f32, &u, 4); return c->ok ? 0 : -1; }
        case LLMCODEC_GGUF_MT_U64: kv->u64 = cur_u(c, 8); return c->ok ? 0 : -1;
        case LLMCODEC_GGUF_MT_I64: kv->i64 = (int64_t)cur_u(c, 8); return c->ok ? 0 : -1;
        case LLMCODEC_GGUF_MT_F64: { uint64_t u = cur_u(c, 8); memcpy(&kv->f64, &u, 8); return c->ok ? 0 : -1; }
        case LLMCODEC_GGUF_MT_BOOL: kv->b = cur_u(c, 1) ? 1 : 0; return c->ok ? 0 : -1;
        case LLMCODEC_GGUF_MT_STRING: return mem_str(c, &kv->str);
        case LLMCODEC_GGUF_MT_ARRAY: {
            uint32_t et = (uint32_t)cur_u(c, 4); uint64_t n = cur_u(c, 8);
            if (!c->ok || n > (1u << 20)) return -1;
            kv->arr_type = (int)et;
            if (et != LLMCODEC_GGUF_MT_STRING) {
                uint64_t es = (et == LLMCODEC_GGUF_MT_U32 || et == LLMCODEC_GGUF_MT_I32 || et == LLMCODEC_GGUF_MT_F32) ? 4 :
                              (et == LLMCODEC_GGUF_MT_U64 || et == LLMCODEC_GGUF_MT_I64 || et == LLMCODEC_GGUF_MT_F64) ? 8 :
                              (et <= 3) ? ((et == 2 || et == 3) ? 2 : 1) : 0;
                if (es == 0) return -1;
                uint64_t bytes = n * es;
                const uint8_t* p = cur_bytes(c, (size_t)bytes); if (!p) return -1;
                void* raw = malloc((size_t)bytes ? (size_t)bytes : 1); if (!raw) return -1;
                memcpy(raw, p, (size_t)bytes);
                kv->arr_raw = raw; kv->arr_raw_bytes = bytes; kv->arr_count = n; return 0;
            }
            kv->arr = (char**)calloc((size_t)n, sizeof(char*)); if (!kv->arr && n) return -1;
            kv->arr_n = (int)n;
            for (uint64_t i = 0; i < n; i++) if (mem_str(c, &kv->arr[i]) != 0) return -1;
            return 0;
        }
        default: return -1;
    }
}

int llmcodec_gguf_parse_mem(const uint8_t* buf, size_t size, LlmCodecGgufMetaKVOwned** out_kv, int* out_n,
                       LlmCodecGgufFileInfo* out_info) {
    if (out_kv) *out_kv = NULL;
    if (out_n) *out_n = 0;
    if (out_info) memset(out_info, 0, sizeof *out_info);
    if (buf == NULL || size < 24) return -1;
    Cur c = { buf, buf + size, 1 };
    uint32_t magic = (uint32_t)cur_u(&c, 4), ver = (uint32_t)cur_u(&c, 4);
    uint64_t n_ts = cur_u(&c, 8), n_kv = cur_u(&c, 8);
    if (!c.ok || magic != 0x46554747u || ver < 2 || ver > 3) return -1;
    if (n_kv > (1u << 20) || n_ts > (1u << 20)) return -1;
    LlmCodecGgufMetaKVOwned* kv = out_kv ? (LlmCodecGgufMetaKVOwned*)calloc((size_t)(n_kv ? n_kv : 1), sizeof(*kv)) : NULL;
    if (out_kv && kv == NULL) return -1;
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        char* key = NULL;
        if (mem_str(&c, &key) != 0) { free(key); if (kv) llmcodec_gguf_meta_free(kv, (int)i); return -1; }
        uint32_t t = (uint32_t)cur_u(&c, 4);
        if (!c.ok) { free(key); if (kv) llmcodec_gguf_meta_free(kv, (int)i); return -1; }
        LlmCodecGgufMetaKVOwned tmp; memset(&tmp, 0, sizeof tmp);
        if (mem_value(&c, t, &tmp) != 0) { free(key); kv_free_fields(&tmp); if (kv) llmcodec_gguf_meta_free(kv, (int)i); return -1; }
        tmp.key = key; tmp.type = (int)t;
        if (strcmp(key, "general.alignment") == 0) {
            uint64_t v = 0;
            if (t == 10) v = tmp.u64; else if (t == 11) v = (uint64_t)tmp.i64;
            else if (t == 6) v = 0; else v = tmp.u32;   /* u8/i8/u16/i16/u32/i32 */
            if (v != 0 && (v & (v - 1)) == 0) alignment = v;
        }
        if (kv) kv[i] = tmp; else kv_free_fields(&tmp);
    }
    if (out_info) {
        out_info->version = ver; out_info->n_kv = n_kv; out_info->n_tensors = n_ts;
        out_info->alignment = alignment;
        if (n_ts) {
            out_info->tensors = (LlmCodecGgufTensorInfo*)calloc((size_t)n_ts, sizeof(LlmCodecGgufTensorInfo));
            if (!out_info->tensors) { if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return -1; }
            for (uint64_t i = 0; i < n_ts; i++) {
                LlmCodecGgufTensorInfo* ti = &out_info->tensors[i];
                char* nm = NULL;
                if (mem_str(&c, &nm) != 0) { free(nm); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return -1; }
                snprintf(ti->name, sizeof ti->name, "%s", nm); free(nm);
                uint32_t nd = (uint32_t)cur_u(&c, 4);
                if (!c.ok || nd > 8) { llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return -1; }
                ti->ndim = (int)nd;
                for (uint32_t d = 0; d < nd; d++) ti->dims[d] = cur_u(&c, 8);
                ti->ggml_type = (int)cur_u(&c, 4); ti->offset = cur_u(&c, 8);
                if (!c.ok) { llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return -1; }
            }
        }
        size_t hdr_off = (size_t)(c.p - buf);
        out_info->data_start = ((uint64_t)hdr_off + alignment - 1) / alignment * alignment;
        out_info->file_size = size;
    }
    if (out_kv) *out_kv = kv;
    if (out_n) *out_n = (int)n_kv;
    return 0;
}

/* Mappa il file in sola lettura e chiama il parser (parse_mem copia le stringhe -> si può smappare). */
int llmcodec_gguf_read_meta(const char* path, LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info) {
    if (path == NULL) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return -1; }
    size_t sz = (size_t)st.st_size;
    void* m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { close(fd); return -1; }
    int rc = llmcodec_gguf_parse_mem((const uint8_t*)m, sz, out_kv, out_n, out_info);
    munmap(m, sz); close(fd);
    return rc;
}

int llmcodec_gguf_read_info(const char* path, LlmCodecGgufFileInfo* out) {
    if (path == NULL || out == NULL) return -1;
    return llmcodec_gguf_read_meta(path, NULL, NULL, out);
}

int llmcodec_gguf_read_header(const char* path, uint32_t* out_version, uint64_t* out_n_ts, uint64_t* out_n_kv) {
    LlmCodecGgufFileInfo info;
    if (llmcodec_gguf_read_info(path, &info) != 0) return -1;
    if (out_version) *out_version = info.version;
    if (out_n_ts) *out_n_ts = info.n_tensors;
    if (out_n_kv) *out_n_kv = info.n_kv;
    llmcodec_gguf_free_info(&info);
    return 0;
}
