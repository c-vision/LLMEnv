/* llmcodec_gguf_writer.c — EMITTER (writer) GGUF v3 per il model compiler (namespace LLM).
 *
 * Componente 2 dell'EMITTER: scrive un artefatto **GGUF** valido (magic/version/counts +
 * metadata KV + tensor-infos + padding all'allineamento + dati). È il formato del mondo DS4/
 * llama.cpp (metodo antirez: metadata/tokenizer dal template, byte dei tensori rigenerati).
 * §0bis: scrittura reale, niente stub. Nessuna dipendenza esterna.
 *
 * GGUF spec (v3): magic u32 "GGUF" | version u32 | n_tensors u64 | n_kv u64 |
 *   KV* (key string, type u32, value) | tensor_info* (name string, n_dims u32, dims u64[],
 *   type u32, offset u64) | padding a `alignment` | dati dei tensori.
 * string = u64 len + bytes. array = u32 elem_type, u64 n, elementi.
 */
#include "llmcodec_gguf_writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Tipi valore metadata GGUF. */
#define LLMCODEC_GGUF_MT_U8     0
#define LLMCODEC_GGUF_MT_I8     1
#define LLMCODEC_GGUF_MT_U16    2
#define LLMCODEC_GGUF_MT_I16    3
#define LLMCODEC_GGUF_MT_U32    4
#define LLMCODEC_GGUF_MT_I32    5
#define LLMCODEC_GGUF_MT_F32    6
#define LLMCODEC_GGUF_MT_BOOL   7
#define LLMCODEC_GGUF_MT_STRING 8
#define LLMCODEC_GGUF_MT_ARRAY  9
#define LLMCODEC_GGUF_MT_U64    10
#define LLMCODEC_GGUF_MT_I64    11
#define LLMCODEC_GGUF_MT_F64    12

static uint64_t align_up(uint64_t x, uint64_t a) { return a > 1 ? ((x + a - 1) / a) * a : x; }

static int wr(FILE* f, const void* p, size_t n) { return fwrite(p, 1, n, f) == n ? 0 : -1; }
static int wr_u32(FILE* f, uint32_t v) { return wr(f, &v, 4); }
static int wr_u64(FILE* f, uint64_t v) { return wr(f, &v, 8); }
static int wr_i32(FILE* f, int32_t v) { return wr(f, &v, 4); }
static int wr_f32(FILE* f, float v) { return wr(f, &v, 4); }
static int wr_str(FILE* f, const char* s) {
    uint64_t n = s ? (uint64_t)strlen(s) : 0;
    if (wr_u64(f, n) != 0) return -1;
    return n ? wr(f, s, (size_t)n) : 0;
}

/* block_bytes/block_vals dei tipi GGML — per leggere/copiare QUALUNQUE tipo.
 *
 * I valori NON sono memoria di chi scrive: sono la tabella `{block_vals, block_bytes}` di
 * `ds4/gguf-tools/quants.c:39-74`, e `tools/llmcodec_ggml_type_bytes_probe.c` confronta OGNI riga
 * contro quella sorgente indipendente (la numerazione dell'enum è in `ds4/gguf-tools/quants.h:20-54`
 * e coincide con `ggml.h`). Due righe, qui, erano errate per anni e il probe le nomina: IQ3_XXS
 * (18) 34 byte invece di 98 e IQ1_S (19) 144 invece di 50 — un GGUF scritto con quei tipi era
 * tagliato o gonfiato senza che nessuno se ne accorgesse. Un tipo che non sta qui NON è un caso
 * teorico: I32 (26) è in `ds4flash.gguf` e BF16 (30) in centinaia di tensori dei PaddleOCR-VL. */
static uint64_t ts_bytes(const LlmCodecGgufTensor* t) {
    uint64_t numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= t->dims[d];
    /* tabella unificata dei tipi GGML: copre QUALUNQUE tipo (Q4_K/Q6_K/...), non solo i nostrani */
    uint64_t b = 0;
    return llmcodec_ggml_type_bytes(t->ggml_type, numel, &b) == 0 ? b : 0;
}

static int gguf_is_align_kv(const LlmCodecGgufMetaKV* k) {
    return k->key && strcmp(k->key, "general.alignment") == 0;
}

int llmcodec_gguf_write(const char* path, const LlmCodecGgufMetaKV* kv, int n_kv,
                   LlmCodecGgufTensor* ts, int n_ts, uint32_t alignment) {
    if (path == NULL || n_ts <= 0 || ts == NULL) return -1;
    if (alignment == 0) alignment = 32;
    /* Potenza di 2 obbligatoria. `align_up` qui sopra divide, quindi 48 "funzionerebbe" e il file
     * sarebbe scritto bene; ma TUTTI e tre i lettori di casa (e llama.cpp) ricadono su 32 quando il
     * `general.alignment` dichiarato non e' una potenza di 2 — accettare 48 qui avrebbe significato
     * produrre un file che NESSUNO legge dal punto giusto, con i pesi spostati di 16 byte e nessun
     * errore. I chiamanti del repo passano tutti 32 o 0: la guardia non spegne nulla. */
    if (alignment & (alignment - 1)) return -1;

    /* `general.alignment` e' del WRITER, non di chi passa i KV: descrive dove il writer ha MESSO
     * i dati. Se arrivasse dai metadata (es. un template copiato da un file allineato 64 e riversato
     * a 32) mentirebbe, e nessun reader — llama.cpp, `llmcodec_moe_gguf.c`, `llmcodec_gguf.h` — potrebbe
     * trovare la data section. Quindi: si scarta quello in ingresso e, solo se diverso dal default
     * 32, si scrive quello effettivo. Con 32 niente chiave: i file gia' prodotti restano
     * byte-identici. */
    uint64_t kv_out = 0;
    for (int i = 0; i < n_kv; i++) if (!gguf_is_align_kv(&kv[i])) kv_out++;
    if (alignment != 32) kv_out++;   /* la chiave effettiva, escrita dopo il giro KV */

    /* offset dei tensori (allineati) */
    uint64_t off = 0;
    for (int i = 0; i < n_ts; i++) {
        uint64_t b = ts_bytes(&ts[i]);
        if (b == 0) return -1;                       /* tipo non supportato */
        ts[i]._offset = align_up(off, alignment);
        off = ts[i]._offset + b;
    }

    FILE* f = fopen(path, "wb");
    if (f == NULL) return -1;
    int rc = 0;
    rc |= wr_u32(f, 0x46554747u);                    /* magic "GGUF" LE */
    rc |= wr_u32(f, 3);                              /* version */
    rc |= wr_u64(f, (uint64_t)n_ts);
    rc |= wr_u64(f, kv_out);

    for (int i = 0; i < n_kv && rc == 0; i++) {
        if (gguf_is_align_kv(&kv[i])) continue;      /* sostituita da quella effettiva, sotto */
        if (getenv("LLMCODEC_GGUF_DEBUG")) fprintf(stderr, "[wr-kv] i=%d key=%s type=%d arrn=%d arrraw=%p\n", i, kv[i].key?kv[i].key:"(null)", kv[i].type, kv[i].arr_n, kv[i].arr_raw);
        rc |= wr_str(f, kv[i].key);
        rc |= wr_u32(f, (uint32_t)kv[i].type);
        switch (kv[i].type) {
            case LLMCODEC_GGUF_MT_U32:    rc |= wr_u32(f, (uint32_t)kv[i].u32); break;
            case LLMCODEC_GGUF_MT_I32:    rc |= wr_i32(f, kv[i].i32); break;
            case LLMCODEC_GGUF_MT_F32:    rc |= wr_f32(f, kv[i].f32); break;
            case LLMCODEC_GGUF_MT_U64:    rc |= wr_u64(f, kv[i].u64); break;
            case 0: case 1: { uint8_t b = (uint8_t)kv[i].u32; rc |= wr(f, &b, 1); } break;   /* u8/i8 */
            case 2: case 3: { uint16_t v = (uint16_t)kv[i].u32; rc |= wr(f, &v, 2); } break; /* u16/i16 */
            case 11: case 12: rc |= wr_u64(f, kv[i].u64); break;   /* i64/f64 (bit-copy in u64) */
            case LLMCODEC_GGUF_MT_BOOL:   { uint8_t b = kv[i].b ? 1 : 0; rc |= wr(f, &b, 1); } break;
            case LLMCODEC_GGUF_MT_STRING: rc |= wr_str(f, kv[i].str); break;
            case LLMCODEC_GGUF_MT_ARRAY:
                if (kv[i].arr_raw != NULL) {                 /* array non-stringa: raw */
                    rc |= wr_u32(f, (uint32_t)kv[i].arr_type);
                    rc |= wr_u64(f, kv[i].arr_count);
                    if (kv[i].arr_raw_bytes) rc |= wr(f, kv[i].arr_raw, (size_t)kv[i].arr_raw_bytes);
                } else {                                     /* array di stringhe */
                    rc |= wr_u32(f, LLMCODEC_GGUF_MT_STRING);
                    rc |= wr_u64(f, (uint64_t)kv[i].arr_n);
                    for (int a = 0; a < kv[i].arr_n && rc == 0; a++) rc |= wr_str(f, kv[i].arr[a]);
                }
                break;
            default: rc = -1; break;
        }
    }
    if (rc == 0 && alignment != 32) {
        rc |= wr_str(f, "general.alignment");
        rc |= wr_u32(f, (uint32_t)LLMCODEC_GGUF_MT_U32);
        rc |= wr_u32(f, alignment);
    }
    for (int i = 0; i < n_ts && rc == 0; i++) {
        if (getenv("LLMCODEC_GGUF_DEBUG")) fprintf(stderr, "[wr-ts] i=%d name=%s ndim=%d type=%d\n", i, ts[i].name?ts[i].name:"(null)", ts[i].ndim, ts[i].ggml_type);
        rc |= wr_str(f, ts[i].name);
        rc |= wr_u32(f, (uint32_t)ts[i].ndim);
        for (int d = 0; d < ts[i].ndim && rc == 0; d++) rc |= wr_u64(f, ts[i].dims[d]);
        rc |= wr_u32(f, (uint32_t)ts[i].ggml_type);
        rc |= wr_u64(f, ts[i]._offset);
    }

    /* padding fino alla data section (header allineato) */
    if (rc == 0) {
        long pos = ftell(f);
        uint64_t pad = align_up((uint64_t)pos, alignment) - (uint64_t)pos;
        for (uint64_t i = 0; i < pad && rc == 0; i++) { char z = 0; rc |= wr(f, &z, 1); }
    }
    /* Data section: scrivi i tensori ai loro offset (padding nei gap d'allineamento). */
    if (rc == 0) {
        const uint64_t data_start = (uint64_t)ftell(f);
        for (int i = 0; i < n_ts && rc == 0; i++) {
            uint64_t written = (uint64_t)ftell(f) - data_start;
            while (written < ts[i]._offset && rc == 0) { char z = 0; rc |= wr(f, &z, 1); written++; }
            uint64_t b = ts_bytes(&ts[i]);
            if (getenv("LLMCODEC_GGUF_DEBUG")) fprintf(stderr, "[wr] i=%d name=%s type=%d b=%llu off=%llu pos=%lld\n", i, ts[i].name, ts[i].ggml_type, (unsigned long long)b, (unsigned long long)ts[i]._offset, (long)ftell(f));
            if (!ts[i].data || fwrite(ts[i].data, 1, (size_t)b, f) != (size_t)b) { if (getenv("LLMCODEC_GGUF_DEBUG")) fprintf(stderr, "[wr] FWRITE FAIL i=%d data=%p\n", i, (void*)ts[i].data); rc = -1; }
        }
    }
    fclose(f);
    return rc == 0 ? 0 : -1;
}

/* ---- READER (componente 3) ---- */
static int rd(FILE* f, void* p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }
static int rd_u64(FILE* f, uint64_t* v) { return rd(f, v, 8); }
static int rd_skip_str(FILE* f) { uint64_t n = 0; if (rd_u64(f, &n)) return -1; return fseek(f, (long)n, SEEK_CUR) == 0 ? 0 : -1; }

/* Una SOLA tabella di larghezze per i valori scalari di un KV. Prima era riscritta tre volte
 * (`rd_skip_value`, `rd_skip_array`, `rd_align_value`) e la copia usata da `llmcodec_gguf_read_info`
 * dimenticava u8/i8/u16/i16: lo STESSO file era interpretato da `llmcodec_gguf_read_meta` (che quei
 * tipi li legge, `case 0: case 1:` / `case 2: case 3:` in `rd_value`) e RIFIUTATO da
 * `llmcodec_gguf_read_info` con un -1 senza motivo. Ritorna 0 = non-scalare (stringa/array) o sconosciuto. */
static int gguf_kv_int_bytes(uint32_t t) {
    switch (t) {
        case LLMCODEC_GGUF_MT_U8: case LLMCODEC_GGUF_MT_I8: case LLMCODEC_GGUF_MT_BOOL: return 1;
        case LLMCODEC_GGUF_MT_U16: case LLMCODEC_GGUF_MT_I16: return 2;
        case LLMCODEC_GGUF_MT_U32: case LLMCODEC_GGUF_MT_I32: return 4;
        case LLMCODEC_GGUF_MT_U64: case LLMCODEC_GGUF_MT_I64: return 8;
        default: return 0;
    }
}
static int gguf_kv_scalar_bytes(uint32_t t) {
    int nb = gguf_kv_int_bytes(t);
    if (nb) return nb;
    return t == (uint32_t)LLMCODEC_GGUF_MT_F32 ? 4 : t == (uint32_t)LLMCODEC_GGUF_MT_F64 ? 8 : 0;
}

static int rd_skip_array(FILE* f) {
    uint32_t et = 0; uint64_t n = 0;
    if (rd(f, &et, 4) || rd_u64(f, &n)) return -1;
    if (et == (uint32_t)LLMCODEC_GGUF_MT_STRING) {
        for (uint64_t i = 0; i < n; i++) if (rd_skip_str(f)) return -1;
        return 0;
    }
    int es = gguf_kv_scalar_bytes(et);
    if (es == 0) return -1;
    for (uint64_t i = 0; i < n; i++) if (fseek(f, es, SEEK_CUR)) return -1;
    return 0;
}
static int rd_skip_value(FILE* f, uint32_t t) {
    switch (t) {
        case LLMCODEC_GGUF_MT_STRING: return rd_skip_str(f);
        case LLMCODEC_GGUF_MT_ARRAY:  return rd_skip_array(f);
        default: {
            int nb = gguf_kv_scalar_bytes(t);
            return nb == 0 ? -1 : (fseek(f, nb, SEEK_CUR) == 0 ? 0 : -1);
        }
    }
}

/* Legge una chiave di nome nel buffer (0 se non entra o se il file finisce). Serve al giro KV di
 * `llmcodec_gguf_read_info`, che deve RICONOSCERE `general.alignment`, non saltarlo alla cieca. */
static int rd_str_into(FILE* f, char* out, size_t cap) {
    uint64_t n = 0;
    if (rd_u64(f, &n)) return 0;
    if (n >= cap) {                       /* chiave piu' lunga del buffer: consumata e NON match,
                                             come faceva `rd_skip_str` prima — non deve diventare
                                             un "file illeggibile" solo perche' la chiave e' lunga */
        if (fseek(f, (long)n, SEEK_CUR)) return 0;
        out[0] = '\0';
        return 1;
    }
    if (n && rd(f, out, (size_t)n)) return 0;
    out[n] = '\0';
    return 1;
}

/* `general.alignment` arriva come u8/i8/u16/i16/u32/i32/u64/i64 a seconda di chi scrive (i float
 * non sono allineamenti legittimi): ritorna 0 se il tipo non e' intero e lascia decidere al chiamante. */
static int rd_align_value(FILE* f, uint32_t t, uint64_t* v) {
    int nb = gguf_kv_int_bytes(t);          /* i float NON sono allineamenti legittimi */
    if (nb == 0) return 0;
    uint64_t raw = 0;
    if (rd(f, &raw, nb)) return 0;
    *v = raw;
    return 1;
}



void llmcodec_gguf_free_info(LlmCodecGgufFileInfo* info) {
    if (info) { free(info->tensors); info->tensors = NULL; }
}

/* ---- TEMPLATE (componente 4) ---- */
static int rd_value(FILE* f, uint32_t t, LlmCodecGgufMetaKVOwned* kv) {
    switch (t) {
        case 0: case 1: { uint8_t v = 0; if (rd(f, &v, 1)) return -1; kv->u32 = v; return 0; }
        case 2: case 3: { uint16_t v = 0; if (rd(f, &v, 2)) return -1; kv->u32 = v; return 0; }
        case LLMCODEC_GGUF_MT_U32: return rd(f, &kv->u32, 4);
        case LLMCODEC_GGUF_MT_I32: return rd(f, &kv->i32, 4);
        case LLMCODEC_GGUF_MT_F32: return rd(f, &kv->f32, 4);
        case LLMCODEC_GGUF_MT_U64: return rd(f, &kv->u64, 8);
        case LLMCODEC_GGUF_MT_I64: return rd(f, &kv->i64, 8);
        case LLMCODEC_GGUF_MT_F64: return rd(f, &kv->f64, 8);
        case LLMCODEC_GGUF_MT_BOOL: { uint8_t b = 0; if (rd(f, &b, 1)) return -1; kv->b = b ? 1 : 0; return 0; }
        case LLMCODEC_GGUF_MT_STRING: {
            uint64_t n = 0;
            if (rd_u64(f, &n) || n > (1u << 24)) return -1;
            kv->str = (char*)malloc((size_t)n + 1);
            if (kv->str == NULL) return -1;
            if (n && rd(f, kv->str, (size_t)n)) return -1;
            kv->str[n] = 0;
            return 0;
        }
        case LLMCODEC_GGUF_MT_ARRAY: {
            uint32_t et = 0; uint64_t n = 0;
            if (rd(f, &et, 4) || rd_u64(f, &n) || n > (1u << 20)) return -1;
            kv->arr_type = (int)et;
            if (et != LLMCODEC_GGUF_MT_STRING) {
                uint64_t es = (et == LLMCODEC_GGUF_MT_U32 || et == LLMCODEC_GGUF_MT_I32 || et == LLMCODEC_GGUF_MT_F32) ? 4 :
                              (et == LLMCODEC_GGUF_MT_U64 || et == LLMCODEC_GGUF_MT_I64 || et == LLMCODEC_GGUF_MT_F64) ? 8 :
                              (et <= 3) ? ((et == 2 || et == 3) ? 2 : 1) : 0;
                if (es == 0) return -1;
                uint64_t bytes = n * es;
                void* raw = malloc((size_t)bytes ? (size_t)bytes : 1);
                if (raw == NULL) return -1;
                if (bytes && rd(f, raw, (size_t)bytes)) { free(raw); return -1; }
                kv->arr_raw = raw; kv->arr_raw_bytes = bytes; kv->arr_count = n;
                return 0;
            }
            kv->arr = (char**)calloc((size_t)n, sizeof(char*));
            if (kv->arr == NULL && n) return -1;
            kv->arr_n = (int)n;
            for (uint64_t i = 0; i < n; i++) {
                uint64_t l = 0;
                if (rd_u64(f, &l) || l > (1u << 24)) return -1;
                kv->arr[i] = (char*)malloc((size_t)l + 1);
                if (kv->arr[i] == NULL) return -1;
                if (l && rd(f, kv->arr[i], (size_t)l)) return -1;
                kv->arr[i][l] = 0;
            }
            return 0;
        }
        default: return -1;
    }
}



void llmcodec_gguf_meta_free(LlmCodecGgufMetaKVOwned* kv, int n) {
    if (kv == NULL) return;
    for (int i = 0; i < n; i++) {
        free(kv[i].key);
        free(kv[i].str);
        if (kv[i].arr) { for (int a = 0; a < kv[i].arr_n; a++) free(kv[i].arr[a]); free(kv[i].arr); }
        free(kv[i].arr_raw);
    }
    free(kv);
}

void llmcodec_gguf_meta_to_writer(const LlmCodecGgufMetaKVOwned* src, LlmCodecGgufMetaKV* dst) {
    memset(dst, 0, sizeof(*dst));
    dst->key = src->key;
    dst->type = src->type;
    dst->u32 = src->u32; dst->i32 = src->i32; dst->f32 = src->f32;
    dst->u64 = (src->type == 11) ? (uint64_t)src->i64 : src->u64;
    if (src->type == 12) memcpy(&dst->u64, &src->f64, 8);
    dst->b = src->b;
    dst->str = src->str;
    dst->arr = (const char* const*)src->arr;
    dst->arr_n = src->arr_n;
    dst->arr_type = src->arr_type;
    dst->arr_raw = src->arr_raw;
    dst->arr_raw_bytes = src->arr_raw_bytes;
    dst->arr_count = src->arr_count;
}


