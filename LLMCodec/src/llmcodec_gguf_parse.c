/* llmcodec_gguf_parse.c — PARSER GGUF canonico (BUFFER-based): UNA SOLA SORGENTE (§0bis).
 * `llmcodec_gguf_parse_mem(buf,size,…)` è l'unico parser; `llmcodec_gguf_read_meta/read_info(path)` mmappano
 * il file e lo chiamano; l'engine DS4 (che mmappa già) lo chiama direttamente. Le stringhe/KV
 * sono COPIATE (owned): il chiamante può liberare il buffer dopo.
 *
 * Dal recupero dell'asset di lane_gguf (piano LLMEnv §5) il parser porta la TASSONOMIA D'ERRORE:
 * ogni rifiuto ha un codice distinto (`llmcodec_gguf_status.h`), esposto da
 * `llmcodec_gguf_parse_mem_status` / `llmcodec_gguf_read_{meta,info}_status`. Le vecchie funzioni
 * restano (0/-1) e sono wrapper sulla variante con codici. */
#include "llmcodec_gguf_read.h"
#include "llmcodec_gguf_status.h"
#include "llmcodec_ggml_quants.h"   /* llmcodec_ggml_type_bytes / block_vals per la validazione dei tipi */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct { const uint8_t* p; const uint8_t* end; int ok; LlmCodecGgufStatus st; } Cur;
static int cur_have(Cur* c, size_t n) {
    if (!c->ok || (size_t)(c->end - c->p) < n) { c->ok = 0; if (c->st == LLMCODEC_GGUF_STATUS_OK) c->st = LLMCODEC_GGUF_STATUS_TRUNCATED; return 0; }
    return 1;
}
static uint64_t cur_u(Cur* c, int nb) { uint64_t v = 0; if (!cur_have(c, (size_t)nb)) return 0; for (int i = 0; i < nb; i++) v |= (uint64_t)c->p[i] << (8*i); c->p += nb; return v; }
static const uint8_t* cur_bytes(Cur* c, size_t n) { if (!cur_have(c, n)) return NULL; const uint8_t* r = c->p; c->p += n; return r; }
static int mem_str(Cur* c, char** out) {
    uint64_t n = cur_u(c, 8);
    if (!c->ok) return -1;
    if (n > (1u << 24)) { c->ok = 0; if (c->st == LLMCODEC_GGUF_STATUS_OK) c->st = LLMCODEC_GGUF_STATUS_IMPLAUSIBLE; return -1; }
    char* s = (char*)malloc((size_t)n + 1);
    if (!s) { c->ok = 0; c->st = LLMCODEC_GGUF_STATUS_NO_MEM; return -1; }
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
            if (!c->ok || n > (1u << 20)) {
                if (c->ok) { c->ok = 0; c->st = LLMCODEC_GGUF_STATUS_IMPLAUSIBLE; }
                return -1;
            }
            kv->arr_type = (int)et;
            if (et != LLMCODEC_GGUF_MT_STRING) {
                uint64_t es = (et == LLMCODEC_GGUF_MT_U32 || et == LLMCODEC_GGUF_MT_I32 || et == LLMCODEC_GGUF_MT_F32) ? 4 :
                              (et == LLMCODEC_GGUF_MT_U64 || et == LLMCODEC_GGUF_MT_I64 || et == LLMCODEC_GGUF_MT_F64) ? 8 :
                              (et <= 3) ? ((et == 2 || et == 3) ? 2 : 1) : 0;
                if (es == 0) { c->ok = 0; c->st = LLMCODEC_GGUF_STATUS_KV_TYPE; return -1; }
                uint64_t bytes = n * es;
                const uint8_t* p = cur_bytes(c, (size_t)bytes); if (!p) return -1;
                void* raw = malloc((size_t)bytes ? (size_t)bytes : 1); if (!raw) { c->ok = 0; c->st = LLMCODEC_GGUF_STATUS_NO_MEM; return -1; }
                memcpy(raw, p, (size_t)bytes);
                kv->arr_raw = raw; kv->arr_raw_bytes = bytes; kv->arr_count = n; return 0;
            }
            kv->arr = (char**)calloc((size_t)n, sizeof(char*)); if (!kv->arr && n) { c->ok = 0; c->st = LLMCODEC_GGUF_STATUS_NO_MEM; return -1; }
            kv->arr_n = (int)n;
            for (uint64_t i = 0; i < n; i++) if (mem_str(c, &kv->arr[i]) != 0) return -1;
            return 0;
        }
        default: c->ok = 0; c->st = LLMCODEC_GGUF_STATUS_KV_TYPE; return -1;
    }
}

/* Validazione del tipo ggml di un tensore: usa la tabella del codec (llmcodec_ggml_block_vals /
 * llmcodec_ggml_type_bytes). Ritorna il codice e scrive i byte del tensore in `*nbytes`. */
static LlmCodecGgufStatus check_tensor_type(int t, uint64_t numel, uint64_t* nbytes) {
    uint64_t nb = 0;
    *nbytes = 0;
    if (t == 4 || t == 5) return LLMCODEC_GGUF_STATUS_TYPE_REMOVED;
    int64_t bv = llmcodec_ggml_block_vals(t);
    if (bv > 1 && (numel % (uint64_t)bv) != 0) return LLMCODEC_GGUF_STATUS_NUMEL_BLOCK;
    if (llmcodec_ggml_type_bytes(t, numel, &nb) != 0) return LLMCODEC_GGUF_STATUS_TYPE_UNKNOWN;
    *nbytes = nb;
    return LLMCODEC_GGUF_STATUS_OK;
}

/* Confronti per nome e per offset (sweep di sovrapposizione/duplicati). */
typedef struct { const char* name; uint64_t start, end; } Seg;
static int seg_cmp_offset(const void* a, const void* b) {
    const Seg* x = (const Seg*)a; const Seg* y = (const Seg*)b;
    if (x->start != y->start) return x->start < y->start ? -1 : 1;
    if (x->end != y->end) return x->end < y->end ? -1 : 1;
    return strcmp(x->name ? x->name : "", y->name ? y->name : "");
}
static int seg_cmp_name(const void* a, const void* b) {
    const Seg* x = (const Seg*)a; const Seg* y = (const Seg*)b;
    return strcmp(x->name ? x->name : "", y->name ? y->name : "");
}

LlmCodecGgufStatus llmcodec_gguf_parse_mem_status(const uint8_t* buf, size_t size,
        LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info) {
    if (out_kv) *out_kv = NULL;
    if (out_n) *out_n = 0;
    if (out_info) memset(out_info, 0, sizeof *out_info);
    if (buf == NULL || size < 24) return LLMCODEC_GGUF_STATUS_TRUNCATED;
    Cur c = { buf, buf + size, 1, LLMCODEC_GGUF_STATUS_OK };
    uint32_t magic = (uint32_t)cur_u(&c, 4);
    uint32_t ver = (uint32_t)cur_u(&c, 4);
    if (!c.ok) return c.st;
    if (magic != 0x46554747u) return LLMCODEC_GGUF_STATUS_NOT_GGUF;
    if (ver < 2 || ver > 4) return LLMCODEC_GGUF_STATUS_VERSION;
    int wide = (ver >= 3);
    uint64_t n_ts = cur_u(&c, wide ? 8 : 4);
    uint64_t n_kv = cur_u(&c, wide ? 8 : 4);
    if (!c.ok) return c.st;
    /* Non un cap: il minimo perché il conteggio possa essere vero (tensor-info >= 32 B, KV >= 13 B). */
    {
        size_t remaining = size - (size_t)(c.p - buf);
        if (n_ts > remaining / 32 + 1 || n_kv > remaining / 13 + 1) return LLMCODEC_GGUF_STATUS_COUNT;
    }
    LlmCodecGgufMetaKVOwned* kv = out_kv ? (LlmCodecGgufMetaKVOwned*)calloc((size_t)(n_kv ? n_kv : 1), sizeof(*kv)) : NULL;
    if (out_kv && kv == NULL) return LLMCODEC_GGUF_STATUS_NO_MEM;
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        char* key = NULL;
        if (mem_str(&c, &key) != 0) { free(key); if (kv) llmcodec_gguf_meta_free(kv, (int)i); return c.st; }
        uint32_t t = (uint32_t)cur_u(&c, 4);
        if (!c.ok) { free(key); if (kv) llmcodec_gguf_meta_free(kv, (int)i); return c.st; }
        LlmCodecGgufMetaKVOwned tmp; memset(&tmp, 0, sizeof tmp);
        if (mem_value(&c, t, &tmp) != 0) { free(key); kv_free_fields(&tmp); if (kv) llmcodec_gguf_meta_free(kv, (int)i); return c.st; }
        tmp.key = key; tmp.type = (int)t;
        if (strcmp(key, "general.alignment") == 0) {
            uint64_t v;
            if (t == LLMCODEC_GGUF_MT_ARRAY || t == LLMCODEC_GGUF_MT_STRING) {
                kv_free_fields(&tmp);          /* tmp.key == key: libera anche la chiave */
                if (kv) llmcodec_gguf_meta_free(kv, (int)i);
                return LLMCODEC_GGUF_STATUS_BAD_ALIGN;
            }
            if (t == 10) v = tmp.u64; else if (t == 11) v = (uint64_t)tmp.i64;
            else if (t == 6) v = 0; else v = tmp.u32;   /* u8/i8/u16/i16/u32/i32 */
            if (v == 0 || (v & (v - 1)) != 0) {
                kv_free_fields(&tmp);          /* tmp.key == key */
                if (kv) llmcodec_gguf_meta_free(kv, (int)i);
                return LLMCODEC_GGUF_STATUS_BAD_ALIGN;
            }
            alignment = v;
        }
        if (kv) kv[i] = tmp; else kv_free_fields(&tmp);
    }
    Seg* sg = NULL;
    if (out_info) {
        out_info->version = ver; out_info->n_kv = n_kv; out_info->n_tensors = n_ts;
        out_info->alignment = alignment;
        if (n_ts) {
            out_info->tensors = (LlmCodecGgufTensorInfo*)calloc((size_t)n_ts, sizeof(LlmCodecGgufTensorInfo));
            if (!out_info->tensors) { if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_NO_MEM; }
            sg = (Seg*)calloc((size_t)n_ts, sizeof(Seg));
            if (!sg) { llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_NO_MEM; }
            for (uint64_t i = 0; i < n_ts; i++) {
                LlmCodecGgufTensorInfo* ti = &out_info->tensors[i];
                char* nm = NULL;
                if (mem_str(&c, &nm) != 0) { free(nm); free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return c.st; }
                snprintf(ti->name, sizeof ti->name, "%s", nm); free(nm);
                uint32_t nd = (uint32_t)cur_u(&c, 4);
                if (!c.ok) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return c.st; }
                if (nd == 0 || nd > 8) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_NDIMS; }
                ti->ndim = (int)nd;
                uint64_t numel = 1;
                for (uint32_t d = 0; d < nd; d++) {
                    ti->dims[d] = cur_u(&c, 8);
                    if (!c.ok) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return c.st; }
                    if (ti->dims[d] == 0) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_NEG_DIM; }
                    numel *= ti->dims[d];
                }
                ti->ggml_type = (int)cur_u(&c, 4);
                ti->offset = cur_u(&c, 8);
                if (!c.ok) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return c.st; }
                uint64_t nbytes = 0;
                LlmCodecGgufStatus ts = check_tensor_type(ti->ggml_type, numel, &nbytes);
                if (ts != LLMCODEC_GGUF_STATUS_OK) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return ts; }
                if (ti->offset > (uint64_t)-1 - nbytes) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_IMPLAUSIBLE; }
                sg[i].name = ti->name; sg[i].start = ti->offset; sg[i].end = ti->offset + nbytes;
            }
            size_t hdr_off = (size_t)(c.p - buf);
            out_info->data_start = ((uint64_t)hdr_off + alignment - 1) / alignment * alignment;
            out_info->file_size = size;
            /* OVERLAP: gli offset non hanno ordine significativo in GGUF, la SOVRAPPOSIZIONE sì. */
            qsort(sg, (size_t)n_ts, sizeof(Seg), seg_cmp_offset);
            for (uint64_t i = 1; i < n_ts; i++) {
                if (sg[i].start < sg[i-1].end) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_OVERLAP; }
            }
            /* NAME_DUP: sullo stesso array ordinato per nome. */
            qsort(sg, (size_t)n_ts, sizeof(Seg), seg_cmp_name);
            for (uint64_t i = 1; i < n_ts; i++) {
                if (strcmp(sg[i].name, sg[i-1].name) == 0) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_NAME_DUP; }
            }
            /* DATA_SHORT: i dati dichiarati devono stare nel file. */
            for (uint64_t i = 0; i < n_ts; i++) {
                if (out_info->data_start > size || sg[i].end > size - out_info->data_start) { free(sg); llmcodec_gguf_free_info(out_info); if (kv) llmcodec_gguf_meta_free(kv, (int)n_kv); return LLMCODEC_GGUF_STATUS_DATA_SHORT; }
            }
            free(sg);
        }
    }
    if (out_kv) *out_kv = kv;
    if (out_n) *out_n = (int)n_kv;
    return LLMCODEC_GGUF_STATUS_OK;
}

int llmcodec_gguf_parse_mem(const uint8_t* buf, size_t size, LlmCodecGgufMetaKVOwned** out_kv, int* out_n,
                       LlmCodecGgufFileInfo* out_info) {
    return llmcodec_gguf_parse_mem_status(buf, size, out_kv, out_n, out_info) == LLMCODEC_GGUF_STATUS_OK ? 0 : -1;
}

/* Mappa il file in sola lettura e chiama il parser (parse_mem copia le stringhe -> si può smappare). */
/* ============ DIR MLX/HF -> GGUF-equivalente (read-only) ============
 * I consumer `LLM TENSOR` devono funzionare su un file .gguf *o* una directory MLX/HF
 * (config.json + *.safetensors) indifferentemente. Se `path` e' una directory, si sintetizza
 * un `LlmCodecGgufFileInfo` + metadata KV dall'header dei safetensors + config.json. La
 * `data_start`/offset non sono usati dai consumer report-only (PROFILE/MEMORY/COST/...). */
#include "llmcodec_safetensors.h"
#include <dirent.h>
static int ggml_type_of_dtype(const char* dt) {
    if (!dt) return LLMCODEC_GGML_T_F32;
    if (strcasecmp(dt, "F32") == 0) return LLMCODEC_GGML_T_F32;
    if (strcasecmp(dt, "F16") == 0) return LLMCODEC_GGML_T_F16;
    if (strcasecmp(dt, "BF16") == 0) return LLMCODEC_GGML_T_BF16;
    if (strcasecmp(dt, "I8") == 0 || strcasecmp(dt, "U8") == 0) return LLMCODEC_GGML_T_I8;
    if (strcasecmp(dt, "I16") == 0) return LLMCODEC_GGML_T_I16;
    if (strcasecmp(dt, "I32") == 0) return LLMCODEC_GGML_T_I32;
    if (strcasecmp(dt, "I64") == 0) return LLMCODEC_GGML_T_I64;
    return LLMCODEC_GGML_T_F16;
}
static char* mlx_read_file(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) == (size_t)n) b[n] = 0; else { free(b); b = NULL; }
    fclose(f); return b;
}
static int mlx_json_int(const char* j, const char* key, long* out) {
    if (!j) return 0; char pat[128]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(j, pat); if (!p) return 0; p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == ':' || *p == '\r' || *p == '\n') p++;
    if (!(*p == '-' || (*p >= '0' && *p <= '9'))) return 0; *out = strtol(p, NULL, 10); return 1;
}
static int mlx_json_int_any(const char* j, const char** keys, int nk, long dflt) { long v; for (int i = 0; i < nk; i++) if (mlx_json_int(j, keys[i], &v)) return v; return dflt; }
static char* mlx_json_str(const char* j, const char* key) {
    if (!j) return NULL; char pat[128]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(j, pat); if (!p) return NULL; p = strchr(p + strlen(pat), ':'); if (!p) return NULL;
    p = strchr(p, '"'); if (!p) return NULL; const char* q = strchr(p + 1, '"'); if (!q) return NULL;
    return strndup(p + 1, (size_t)(q - p - 1));
}
static int llmcodec_gguf_read_meta_mlx(const char* dir, LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info) {
    if (out_kv) *out_kv = NULL; if (out_n) *out_n = 0; if (out_info) memset(out_info, 0, sizeof *out_info);
    char cp[2048]; snprintf(cp, sizeof cp, "%s/config.json", dir);
    char* cfg = mlx_read_file(cp);
    /* shard *.safetensors */
    char** shards = NULL; int ns = 0;
    DIR* dp = opendir(dir); if (!dp) { free(cfg); return -1; }
    struct dirent* e;
    while ((e = readdir(dp)) != NULL) { size_t L = strlen(e->d_name);
        if (L > 12 && strcmp(e->d_name + L - 12, ".safetensors") == 0) { shards = (char**)realloc(shards, sizeof(char*) * (ns + 1)); char sp[2048]; snprintf(sp, sizeof sp, "%s/%s", dir, e->d_name); shards[ns++] = strdup(sp); } }
    closedir(dp);
    if (!out_info) { for (int i = 0; i < ns; i++) free(shards[i]); free(shards); free(cfg); return 0; }
    uint64_t cap = 0; for (int s = 0; s < ns; s++) { LlmCodecStTensorInfo* ti = NULL; int nt = 0; uint64_t base = 0; if (llmcodec_safetensors_read(shards[s], &ti, &nt, &base) == 0) { cap += nt; llmcodec_safetensors_info_free(ti); } }
    out_info->version = 3; out_info->n_tensors = 0; out_info->tensors = (LlmCodecGgufTensorInfo*)calloc(cap ? cap : 1, sizeof(LlmCodecGgufTensorInfo));
    for (int s = 0; s < ns; s++) {
        LlmCodecStTensorInfo* ti = NULL; int nt = 0; uint64_t base = 0;
        if (llmcodec_safetensors_read(shards[s], &ti, &nt, &base) != 0) continue;
        for (int i = 0; i < nt; i++) {
            LlmCodecGgufTensorInfo* t = &out_info->tensors[out_info->n_tensors++];
            snprintf(t->name, sizeof t->name, "%s", ti[i].name);
            t->ggml_type = ggml_type_of_dtype(ti[i].dtype);
            t->ndim = ti[i].ndim; for (int d = 0; d < t->ndim && d < 8; d++) t->dims[d] = ti[i].shape[d];
            t->offset = 0;
            t->size_bytes = (ti[i].end > ti[i].begin) ? (ti[i].end - ti[i].begin) : 0;  /* REALE (packed) */
        }
        llmcodec_safetensors_info_free(ti);
    }
    /* metadata KV (dai campi config.json; nomi = quelli GGUF attesi dai consumer) */
    if (out_kv) {
        const char* k_lay[] = {"num_hidden_layers","n_layer","num_layers"};
        const char* k_hd[]  = {"num_attention_heads","n_head","num_heads"};
        const char* k_hkv[] = {"num_key_value_heads","n_head_kv","num_kv_heads"};
        const char* k_en[]  = {"hidden_size","n_embd","d_model"};
        const char* k_hdim[]= {"head_dim","key_length"};
        const char* k_ne[]  = {"n_routed_experts","num_experts","n_experts","num_local_experts"};
        const char* k_tk[]  = {"num_experts_per_tok","num_experts_per_token","top_k"};
        long lay = mlx_json_int_any(cfg,k_lay,3,-1), hd = mlx_json_int_any(cfg,k_hd,3,-1), hkv = mlx_json_int_any(cfg,k_hkv,3,-1);
        long en = mlx_json_int_any(cfg,k_en,3,-1), hdim = mlx_json_int_any(cfg,k_hdim,2,-1), ne = mlx_json_int_any(cfg,k_ne,4,-1), tk = mlx_json_int_any(cfg,k_tk,3,-1);
        char* mt = mlx_json_str(cfg, "model_type");
        int nk = 1 + (lay>0) + (hd>0) + (hkv>0) + (en>0) + (hdim>0) + (ne>0) + (tk>0);
        LlmCodecGgufMetaKVOwned* kv = (LlmCodecGgufMetaKVOwned*)calloc((size_t)nk, sizeof *kv);
        int i = 0;
        if (mt) { kv[i].key = strdup("general.architecture"); kv[i].type = LLMCODEC_GGUF_MT_STRING; kv[i].str = mt; i++; }
        #define MKI(KEY,VAL) if ((VAL)>0) { kv[i].key = strdup(KEY); kv[i].type = LLMCODEC_GGUF_MT_U32; kv[i].u32 = (uint32_t)(VAL); i++; }
        MKI("block_count", lay) MKI("attention.head_count", hd) MKI("attention.head_count_kv", hkv)
        MKI("embedding_length", en) MKI("attention.key_length", hdim) MKI("attention.value_length", hdim)
        MKI("expert_count", ne) MKI("expert_used_count", tk)
        #undef MKI
        *out_kv = kv; *out_n = i;
    }
    for (int i = 0; i < ns; i++) free(shards[i]); free(shards); free(cfg);
    return 0;
}
LlmCodecGgufStatus llmcodec_gguf_read_meta_status(const char* path, LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info) {
    if (path == NULL) return LLMCODEC_GGUF_STATUS_IO;
    { struct stat dst; if (stat(path, &dst) == 0 && S_ISDIR(dst.st_mode))
        return llmcodec_gguf_read_meta_mlx(path, out_kv, out_n, out_info) == 0 ? LLMCODEC_GGUF_STATUS_OK : LLMCODEC_GGUF_STATUS_IO; }
    if (out_kv) *out_kv = NULL;
    if (out_n) *out_n = 0;
    if (out_info) memset(out_info, 0, sizeof *out_info);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return LLMCODEC_GGUF_STATUS_IO;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return LLMCODEC_GGUF_STATUS_IO; }
    size_t sz = (size_t)st.st_size;
    void* m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { close(fd); return LLMCODEC_GGUF_STATUS_IO; }
    LlmCodecGgufStatus rc = llmcodec_gguf_parse_mem_status((const uint8_t*)m, sz, out_kv, out_n, out_info);
    munmap(m, sz); close(fd);
    return rc;
}

int llmcodec_gguf_read_meta(const char* path, LlmCodecGgufMetaKVOwned** out_kv, int* out_n, LlmCodecGgufFileInfo* out_info) {
    return llmcodec_gguf_read_meta_status(path, out_kv, out_n, out_info) == LLMCODEC_GGUF_STATUS_OK ? 0 : -1;
}

LlmCodecGgufStatus llmcodec_gguf_read_info_status(const char* path, LlmCodecGgufFileInfo* out) {
    if (out == NULL) return LLMCODEC_GGUF_STATUS_IO;
    return llmcodec_gguf_read_meta_status(path, NULL, NULL, out);
}

int llmcodec_gguf_read_info(const char* path, LlmCodecGgufFileInfo* out) {
    if (path == NULL || out == NULL) return -1;
    return llmcodec_gguf_read_info_status(path, out) == LLMCODEC_GGUF_STATUS_OK ? 0 : -1;
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
