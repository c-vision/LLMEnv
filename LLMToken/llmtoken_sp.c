/* ============================================================================
 * llmtoken_sp.c
 * Loader NATIVO (C, nessun Python) per tokenizer SentencePiece: legge il file
 * `tokenizer.model` (protobuf ModelProto di sentencepiece) e costruisce un
 * tokenizer per l'engine. Chiude i modelli che hanno SOLO il tokenizer slow
 * SentencePiece e nessun `tokenizer.json` HF-fast (es. ERNIE-4.5, Gemma-1/2,
 * Llama-1/2 originali in formato .model).
 *
 * Percorso: parse protobuf minimale (varint + length-delimited) -> pieces
 * (piece/score/type), trainer_spec.byte_fallback, normalizer_spec
 * (add_dummy_prefix) -> derivazione dei `merges` BPE dal vocabolario (stessa
 * funzione `generate_merges` di HF: per ogni piece, split in ogni punto dove
 * entrambi i lati sono nel vocab; ordina per (id_del_piece, len(l), len(r))) ->
 * emissione di un `tokenizer.json` in memoria -> `llmtoken_load_json`.
 *
 * Riusa INTEGRALMENTE il percorso BPE-m metaspace gia' validato in llmtoken.c
 * (nessuna logica di encode duplicata): qui si produce solo l'input JSON.
 *
 * Limite noto (documentato): `remove_extra_whitespaces` NON e' replicato (il
 * normalizer dell'engine applica solo Replace letterali); il testo normale
 * coincide, divergono solo i run di spazio multipli/spazi in testa -- stesso
 * limite del converter HF ufficiale (SpmConverter).
 * ============================================================================ */
#include "llmtoken.h"
#include "sds.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------- lettore protobuf minimale ---------------------- */
typedef struct { const uint8_t* p; const uint8_t* end; } PbReader;

static int pb_varint(PbReader* r, uint64_t* out) {
    uint64_t v = 0; int shift = 0;
    while (r->p < r->end) {
        uint8_t b = *r->p++;
        v |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) { *out = v; return 1; }
        shift += 7;
        if (shift > 63) return 0;
    }
    return 0;
}
static int pb_skip(PbReader* r, uint32_t wire) {
    switch (wire) {
        case 0: { uint64_t t; return pb_varint(r, &t); }
        case 1: if ((size_t)(r->end - r->p) < 8) return 0; r->p += 8; return 1;
        case 2: { uint64_t n; if (!pb_varint(r, &n)) return 0;
                  if ((uint64_t)(r->end - r->p) < n) return 0; r->p += n; return 1; }
        case 5: if ((size_t)(r->end - r->p) < 4) return 0; r->p += 4; return 1;
        default: return 0;
    }
}

/* ----------------------------- hash stringa -> int32 ---------------------- */
typedef struct { char** keys; uint32_t* hashes; int32_t* vals; size_t cap; } StrMap;

static uint32_t sp_hash(const char* s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h ? h : 1u;
}
static int sm_init(StrMap* m, size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    m->cap = cap;
    m->keys = (char**)calloc(cap, sizeof(char*));
    m->hashes = (uint32_t*)calloc(cap, sizeof(uint32_t));
    m->vals = (int32_t*)calloc(cap, sizeof(int32_t));
    return m->keys && m->hashes && m->vals;
}
static void sm_free(StrMap* m) { free(m->keys); free(m->hashes); free(m->vals); }
static void sm_put(StrMap* m, const char* k, size_t len, int32_t v) {
    uint32_t h = sp_hash(k, len);
    size_t mask = m->cap - 1, i = h & mask;
    while (m->keys[i]) {
        if (m->hashes[i] == h && strlen(m->keys[i]) == len && memcmp(m->keys[i], k, len) == 0) {
            m->vals[i] = v; return;
        }
        i = (i + 1) & mask;
    }
    m->keys[i] = (char*)k; m->hashes[i] = h; m->vals[i] = v;
}
static int sm_get(const StrMap* m, const char* k, size_t len, int32_t* out) {
    uint32_t h = sp_hash(k, len);
    size_t mask = m->cap - 1, i = h & mask;
    while (m->keys[i]) {
        if (m->hashes[i] == h && strlen(m->keys[i]) == len && memcmp(m->keys[i], k, len) == 0) {
            *out = m->vals[i]; return 1;
        }
        i = (i + 1) & mask;
    }
    return 0;
}

/* ------------------------------- candidate merge -------------------------- */
typedef struct { int32_t piece; int32_t split; } SpCand;
static int sp_cmp_cand(const void* a, const void* b) {
    const SpCand* x = (const SpCand*)a; const SpCand* y = (const SpCand*)b;
    if (x->piece != y->piece) return x->piece < y->piece ? -1 : 1;
    if (x->split != y->split) return x->split < y->split ? -1 : 1;
    return 0;
}

/* ---------------------------------- escaping ------------------------------ */
static void json_escape(sds* out, const char* s, size_t len) {
    *out = sdscatlen(*out, "\"", 1);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  *out = sdscatlen(*out, "\\\"", 2); break;
            case '\\': *out = sdscatlen(*out, "\\\\", 2); break;
            case '\n': *out = sdscatlen(*out, "\\n", 2); break;
            case '\r': *out = sdscatlen(*out, "\\r", 2); break;
            case '\t': *out = sdscatlen(*out, "\\t", 2); break;
            case '\b': *out = sdscatlen(*out, "\\b", 2); break;
            case '\f': *out = sdscatlen(*out, "\\f", 2); break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); *out = sdscatlen(*out, b, 6); }
                else *out = sdscatlen(*out, (const char*)&s[i], 1);
        }
    }
    *out = sdscatlen(*out, "\"", 1);
}

/* ------------------------------- main loader ------------------------------ */
llmtoken_handle llmtoken_load_sp(const char* model_path) {
    if (!model_path || !*model_path) return NULL;
    FILE* f = fopen(model_path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long fsz = ftell(f);
    if (fsz <= 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)fsz);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)fsz, f);
    fclose(f);
    if (got != (size_t)fsz) { free(buf); return NULL; }

    /* pieces */
    char** pieces = NULL; float* scores = NULL; uint8_t* types = NULL;
    size_t n = 0, cap = 0;
    int byte_fallback = 0, add_dummy_prefix = 1, model_type = 1;
    int ok = 1;

    PbReader r = { buf, buf + (size_t)fsz };
    while (r.p < r.end && ok) {
        uint64_t tag;
        if (!pb_varint(&r, &tag)) { ok = 0; break; }
        uint32_t field = (uint32_t)(tag >> 3), wire = (uint32_t)(tag & 7);
        if (field == 1 && wire == 2) {                 /* repeated SentencePiece pieces */
            uint64_t plen;
            if (!pb_varint(&r, &plen) || (uint64_t)(r.end - r.p) < plen) { ok = 0; break; }
            const uint8_t* pend = r.p + plen;
            PbReader pr = { r.p, pend };
            const char* piece = NULL; size_t piece_len = 0; float score = 0.f; int ptype = 1;
            while (pr.p < pr.end) {
                uint64_t t2;
                if (!pb_varint(&pr, &t2)) break;
                uint32_t f2 = (uint32_t)(t2 >> 3), w2 = (uint32_t)(t2 & 7);
                if (f2 == 1 && w2 == 2) {              /* piece */
                    uint64_t l2;
                    if (!pb_varint(&pr, &l2) || (uint64_t)(pr.end - pr.p) < l2) break;
                    piece = (const char*)pr.p; piece_len = (size_t)l2; pr.p += l2;
                } else if (f2 == 2 && w2 == 5) {       /* score (float) */
                    if ((size_t)(pr.end - pr.p) < 4) break;
                    uint32_t bits = (uint32_t)pr.p[0] | ((uint32_t)pr.p[1] << 8)
                                  | ((uint32_t)pr.p[2] << 16) | ((uint32_t)pr.p[3] << 24);
                    memcpy(&score, &bits, 4); pr.p += 4;
                } else if (f2 == 3 && w2 == 0) {       /* type (enum) */
                    uint64_t v;
                    if (!pb_varint(&pr, &v)) break;
                    ptype = (int)v;
                } else {
                    if (!pb_skip(&pr, w2)) break;
                }
            }
            if (piece) {
                if (n == cap) { cap = cap ? cap * 2 : 1024;
                    pieces = (char**)realloc(pieces, cap * sizeof(char*));
                    scores = (float*)realloc(scores, cap * sizeof(float));
                    types  = (uint8_t*)realloc(types, cap * sizeof(uint8_t)); }
                char* cp = (char*)malloc(piece_len + 1);
                if (!cp) { ok = 0; break; }
                memcpy(cp, piece, piece_len); cp[piece_len] = 0;
                pieces[n] = cp; scores[n] = score; types[n] = (uint8_t)ptype; n++;
            }
            r.p = pend;
        } else if (field == 2 && wire == 2) {          /* trainer_spec */
            uint64_t l;
            if (!pb_varint(&r, &l) || (uint64_t)(r.end - r.p) < l) { ok = 0; break; }
            PbReader tr = { r.p, r.p + l };
            while (tr.p < tr.end) {
                uint64_t t2;
                if (!pb_varint(&tr, &t2)) break;
                uint32_t f2 = (uint32_t)(t2 >> 3), w2 = (uint32_t)(t2 & 7);
                if (w2 == 0) {
                    uint64_t v; if (!pb_varint(&tr, &v)) break;
                    if (f2 == 3) model_type = (int)v;
                    else if (f2 == 35) byte_fallback = (int)v;
                } else if (!pb_skip(&tr, w2)) break;
            }
            r.p += l;
        } else if (field == 3 && wire == 2) {          /* normalizer_spec */
            uint64_t l;
            if (!pb_varint(&r, &l) || (uint64_t)(r.end - r.p) < l) { ok = 0; break; }
            PbReader nr = { r.p, r.p + l };
            while (nr.p < nr.end) {
                uint64_t t2;
                if (!pb_varint(&nr, &t2)) break;
                uint32_t f2 = (uint32_t)(t2 >> 3), w2 = (uint32_t)(t2 & 7);
                if (w2 == 0) {
                    uint64_t v; if (!pb_varint(&nr, &v)) break;
                    if (f2 == 3) add_dummy_prefix = (int)v;
                } else if (!pb_skip(&nr, w2)) break;
            }
            r.p += l;
        } else {
            if (!pb_skip(&r, wire)) { ok = 0; break; }
        }
    }
    free(buf);
    if (!ok || n == 0) {
        for (size_t i = 0; i < n; i++) free(pieces[i]);
        free(pieces); free(scores); free(types);
        return NULL;
    }
    if (model_type != 2) {   /* solo BPE supportato (l'engine e' un encoder BPE) */
        for (size_t i = 0; i < n; i++) free(pieces[i]);
        free(pieces); free(scores); free(types);
        return NULL;
    }

    /* vocab map piece -> id */
    StrMap map; int map_ok = sm_init(&map, n);
    if (!map_ok) {
        for (size_t i = 0; i < n; i++) free(pieces[i]);
        free(pieces); free(scores); free(types);
        return NULL;
    }
    for (size_t i = 0; i < n; i++) sm_put(&map, pieces[i], strlen(pieces[i]), (int32_t)i);

    /* unk token (primo piece di tipo UNKNOWN=2) */
    const char* unk = NULL;
    for (size_t i = 0; i < n; i++) if (types[i] == 2) { unk = pieces[i]; break; }
    if (!unk) unk = "<unk>";

    /* merges: candidate (piece, split) dove entrambi i lati sono nel vocab */
    SpCand* cand = (SpCand*)malloc(n * 8 * sizeof(SpCand));
    size_t ncand = 0, cand_cap = n ? n * 8 : 0;
    for (size_t i = 0; i < n && cand; i++) {
        size_t pl = strlen(pieces[i]);
        for (size_t k = 1; k < pl; k++) {
            int32_t dummy;
            if (sm_get(&map, pieces[i], k, &dummy) && sm_get(&map, pieces[i] + k, pl - k, &dummy)) {
                if (ncand == cand_cap) { cand_cap = cand_cap ? cand_cap * 2 : 4096;
                    cand = (SpCand*)realloc(cand, cand_cap * sizeof(SpCand)); }
                cand[ncand].piece = (int32_t)i; cand[ncand].split = (int32_t)k; ncand++;
            }
        }
    }
    qsort(cand, ncand, sizeof(SpCand), sp_cmp_cand);

    /* emette il tokenizer.json in memoria */
    sds out = sdsempty();
    out = sdscat(out, "{\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":");
    { sds t = sdsempty(); json_escape(&t, unk, strlen(unk)); out = sdscatlen(out, t, sdslen(t)); sdsfree(t); }
    out = sdscat(out, byte_fallback ? ",\"byte_fallback\":true,\"vocab\":{" : ",\"byte_fallback\":false,\"vocab\":{");
    for (size_t i = 0; i < n; i++) {
        if (i) out = sdscatlen(out, ",", 1);
        sds k = sdsempty(); json_escape(&k, pieces[i], strlen(pieces[i]));
        out = sdscatlen(out, k, sdslen(k)); sdsfree(k);
        char nb[16]; int nl = snprintf(nb, sizeof nb, ":%zu", i);
        out = sdscatlen(out, nb, (size_t)nl);
    }
    out = sdscat(out, "},\"merges\":[");
    for (size_t i = 0; i < ncand; i++) {
        const char* p = pieces[cand[i].piece]; size_t pl = strlen(p); size_t k = (size_t)cand[i].split;
        if (i) out = sdscatlen(out, ",", 1);
        out = sdscatlen(out, "[", 1);
        sds l = sdsempty(); json_escape(&l, p, k); out = sdscatlen(out, l, sdslen(l)); sdsfree(l);
        out = sdscatlen(out, ",", 1);
        sds rr = sdsempty(); json_escape(&rr, p + k, pl - k); out = sdscatlen(out, rr, sdslen(rr)); sdsfree(rr);
        out = sdscatlen(out, "]", 1);
    }
    out = sdscat(out, "]}");

    /* normalizer: Prepend (se add_dummy_prefix) + Replace(" " -> U+2581) */
    out = sdscat(out, ",\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":["
                      "{\"type\":\"Strip\",\"strip_left\":false,\"strip_right\":true}");
    if (add_dummy_prefix)
        out = sdscat(out, ",{\"type\":\"Prepend\",\"prepend\":\"\xe2\x96\x81\"}");
    out = sdscat(out, ",{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"\xe2\x96\x81\"}]}");

    out = sdscat(out, ",\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"\xe2\x96\x81\",\"prepend_scheme\":\"never\"}");
    out = sdscat(out, ",\"decoder\":{\"type\":\"Metaspace\",\"replacement\":\"\xe2\x96\x81\",\"prepend_scheme\":\"never\"}");

    /* added_tokens: type CONTROL(3) o USER_DEFINED(4) */
    out = sdscat(out, ",\"added_tokens\":[");
    int first_at = 1;
    for (size_t i = 0; i < n; i++) {
        if (types[i] != 3 && types[i] != 4) continue;
        if (!first_at) out = sdscatlen(out, ",", 1);
        first_at = 0;
        char nb[32]; int nl = snprintf(nb, sizeof nb, "{\"id\":%zu,\"content\":", i);
        out = sdscatlen(out, nb, (size_t)nl);
        sds c = sdsempty(); json_escape(&c, pieces[i], strlen(pieces[i])); out = sdscatlen(out, c, sdslen(c)); sdsfree(c);
        out = sdscat(out, types[i] == 3
            ? ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}"
            : ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false}");
    }
    out = sdscat(out, "]}");

    llmtoken_handle h = llmtoken_load_json(out, sdslen(out));

    sdsfree(out);
    sm_free(&map);
    for (size_t i = 0; i < n; i++) free(pieces[i]);
    free(pieces); free(scores); free(types); free(cand);
    return h;
}
