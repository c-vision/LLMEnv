/* ============================================================================
 * llmtoken_wordpiece_selfcheck.c
 * Conformance del tokenizer BERT WordPiece C (llmtoken_wordpiece.c)
 * contro l'ORACOLO ESTERNO versionato samples/Fixtures/tokenizer/tokenizer_golden.json
 * (generato da tools/gen_tokenizer_golden.py con la libreria HF `tokenizers`).
 *
 * Specchia tests/AI.Tests/BertWordPieceTokenizerGoldenTests.cs (il C#): confronta
 * id-per-id BasicTokenize, Encode, EncodePair e troncamento. Uno scanner JSON
 * minimale locale legge proprio il file golden (stessa fixture del test C#),
 * cosi' parita' = comportamento, non un'altra lettura della spec.
 *
 * Uso: llmtoken_wordpiece_selfcheck [dir_fixture]
 *   dir_fixture default: <repo>/samples/Fixtures/tokenizer (derivata da argv/CWD).
 * Exit 0 = conforme, 1 = qualche divergenza (stampa atteso/ottenuto).
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "llmtoken_wordpiece.h"

static int main_argc;
static char** main_argv;

/* ===========================================================================
 * Scanner JSON minimale (solo il sottoinsieme del golden: obj/array/string/num).
 * =========================================================================== */
typedef enum { JT_NULL, JT_BOOL, JT_NUM, JT_STR, JT_ARR, JT_OBJ } jtype;
typedef struct jval jval;
typedef struct { char* key; jval* v; } jpair;

struct jval {
    jtype type;
    union {
        bool b;
        double num;
        char* s;
        struct { jval** items; size_t n; } arr;
        struct { jpair* items; size_t n; } obj;
    } u;
};

static void jskip(const char** p) {
    while (**p && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) (*p)++;
}

static const char* jstr(const char** p) {
    if (**p != '"') return NULL;
    (*p)++;
    size_t cap = 32, n = 0;
    char* out = (char*)malloc(cap);
    while (**p && **p != '"') {
        if (**p == '\\') {
            (*p)++;
            char c = **p;
            if (c == 'n') { if (n + 1 >= cap) { cap *= 2; out = (char*)realloc(out, cap); } out[n++] = '\n'; (*p)++; continue; }
            if (c == 't') { if (n + 1 >= cap) { cap *= 2; out = (char*)realloc(out, cap); } out[n++] = '\t'; (*p)++; continue; }
            if (c == 'r') { if (n + 1 >= cap) { cap *= 2; out = (char*)realloc(out, cap); } out[n++] = '\r'; (*p)++; continue; }
            if (c == 'u') {
                unsigned int cp = 0;
                (*p)++;
                for (int i = 0; i < 4; i++) {
                    char h = **p; cp = cp * 16 + (unsigned int)((h >= '0' && h <= '9') ? h - '0' : ((h >= 'a' && h <= 'f') ? h - 'a' + 10 : h - 'A' + 10));
                    (*p)++;
                }
                char tmp[4]; int tn;
                if (cp < 0x80) { tmp[0] = (char)cp; tn = 1; }
                else if (cp < 0x800) { tmp[0] = (char)(0xC0 | (cp >> 6)); tmp[1] = (char)(0x80 | (cp & 0x3F)); tn = 2; }
                else { tmp[0] = (char)(0xE0 | (cp >> 12)); tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[2] = (char)(0x80 | (cp & 0x3F)); tn = 3; }
                if (n + (size_t)tn + 1 >= cap) { cap = cap * 2 > n + (size_t)tn + 1 ? cap * 2 : n + (size_t)tn + 1; out = (char*)realloc(out, cap); }
                memcpy(out + n, tmp, (size_t)tn); n += (size_t)tn;
                continue;
            }
            if (n + 1 >= cap) { cap *= 2; out = (char*)realloc(out, cap); }
            out[n++] = c; (*p)++;
            continue;
        }
        if (n + 1 >= cap) { cap *= 2; out = (char*)realloc(out, cap); }
        out[n++] = **p; (*p)++;
    }
    if (**p == '"') (*p)++;
    out[n] = '\0';
    return out;
}

static jval* jparse(const char** p) {
    jskip(p);
    jval* v = (jval*)calloc(1, sizeof(jval));
    if (**p == '{') {
        v->type = JT_OBJ; (*p)++; jskip(p);
        if (**p == '}') { (*p)++; return v; }
        while (1) {
            jskip(p);
            const char* key = jstr(p);
            jskip(p);
            if (**p == ':') (*p)++;
            jval* val = jparse(p);
            v->u.obj.items = (jpair*)realloc(v->u.obj.items, (v->u.obj.n + 1) * sizeof(jpair));
            v->u.obj.items[v->u.obj.n].key = (char*)key;
            v->u.obj.items[v->u.obj.n].v = val;
            v->u.obj.n++;
            jskip(p);
            if (**p == ',') { (*p)++; continue; }
            if (**p == '}') { (*p)++; break; }
        }
    } else if (**p == '[') {
        v->type = JT_ARR; (*p)++; jskip(p);
        if (**p == ']') { (*p)++; return v; }
        while (1) {
            jval* el = jparse(p);
            v->u.arr.items = (jval**)realloc(v->u.arr.items, (v->u.arr.n + 1) * sizeof(jval*));
            v->u.arr.items[v->u.arr.n] = el;
            v->u.arr.n++;
            jskip(p);
            if (**p == ',') { (*p)++; continue; }
            if (**p == ']') { (*p)++; break; }
        }
    } else if (**p == '"') {
        v->type = JT_STR;
        v->u.s = (char*)jstr(p);
    } else if (strncmp(*p, "null", 4) == 0) {
        v->type = JT_NULL;
        *p += 4;
    } else if (strncmp(*p, "true", 4) == 0) {
        v->type = JT_BOOL; v->u.b = true;
        *p += 4;
    } else if (strncmp(*p, "false", 5) == 0) {
        v->type = JT_BOOL; v->u.b = false;
        *p += 5;
    } else {
        v->type = JT_NUM;
        v->u.num = strtod(*p, (char**)p);
    }
    return v;
}

static void jfree(jval* v) {
    if (!v) return;
    if (v->type == JT_STR) free(v->u.s);
    else if (v->type == JT_ARR) { for (size_t i = 0; i < v->u.arr.n; i++) jfree(v->u.arr.items[i]); free(v->u.arr.items); }
    else if (v->type == JT_OBJ) { for (size_t i = 0; i < v->u.obj.n; i++) { free(v->u.obj.items[i].key); jfree(v->u.obj.items[i].v); } free(v->u.obj.items); }
    free(v);
}

static jval* jobj_get(jval* obj, const char* key) {
    if (!obj || obj->type != JT_OBJ) return NULL;
    for (size_t i = 0; i < obj->u.obj.n; i++)
        if (strcmp(obj->u.obj.items[i].key, key) == 0) return obj->u.obj.items[i].v;
    return NULL;
}

/* ===========================================================================
 * Fail + formattazione
 * =========================================================================== */
static int g_failures = 0;

static void fail(const char* id, const char* what, const char* exp, const char* got) {
    g_failures++;
    fprintf(stderr, "FAIL [%s] %s\n  atteso  : %s\n  ottenuto: %s\n", id, what, exp, got);
}

static void fmt_ids(const int* a, int n, char* out, size_t cap) {
    size_t pos = 0; out[0] = '\0';
    for (int i = 0; i < n && pos < cap - 1; i++) {
        int w = snprintf(out + pos, cap - pos, "%s%d", i ? "," : "", a[i]);
        pos += (size_t)w;
    }
}

/* ===========================================================================
 * Collettori via callback
 * =========================================================================== */
typedef struct { char** items; int n, cap; } strvec;
typedef struct { int* items; int n, cap; } intvec;

static int collect_str(const char* tok, void* u) {
    strvec* v = (strvec*)u;
    if (v->n >= v->cap) { v->cap = v->cap ? v->cap * 2 : 16; v->items = (char**)realloc(v->items, (size_t)v->cap * sizeof(char*)); }
    v->items[v->n++] = strdup(tok);
    return 0;
}
static int collect_int(int id, void* u) {
    intvec* v = (intvec*)u;
    if (v->n >= v->cap) { v->cap = v->cap ? v->cap * 2 : 16; v->items = (int*)realloc(v->items, (size_t)v->cap * sizeof(int)); }
    v->items[v->n++] = id;
    return 0;
}

/* Encode completo (ids scollegati dal tokenizer) per il confronto: riusiamo
 * llmtoken_wp_encode che fa gia' tutto (basic + wordpiece + trunc + pad). */

/* ===========================================================================
 * Test
 * =========================================================================== */
static char* dir_fixture(void);
static char* vpath(const char* name) {
    static char b[4096]; snprintf(b, sizeof(b), "%s/%s", dir_fixture(), name); return b;
}

static llmtoken_wp_tok* load_vocab(const char* name) {
    static char vfile[4096];
    snprintf(vfile, sizeof(vfile), "%s/vocab_%s.txt", dir_fixture(), name);
    return llmtoken_wp_tok_load(vfile);
}

static void check_case(jval* c) {
    const char* id = jobj_get(c, "id")->u.s;
    const char* vocab = jobj_get(c, "vocab")->u.s;
    const char* text = jobj_get(c, "text")->u.s;
    jval* basic = jobj_get(c, "basic_tokens");
    jval* ids = jobj_get(c, "ids");
    int nids = (int)ids->u.arr.n;

    llmtoken_wp_tok* t = load_vocab(vocab);
    if (!t) { fail(id, "load vocab", vocab, "ERR"); return; }

    /* BasicTokenize */
    {
        strvec got = {0};
        int n = llmtoken_wp_basic_tokenize(text, collect_str, &got);
        if (n != (int)basic->u.arr.n) {
            fail(id, "basic_tokenize count", "N", "M");
        } else {
            int bad = 0;
            for (int i = 0; i < n; i++)
                if (strcmp(got.items[i], basic->u.arr.items[i]->u.s) != 0) { bad = 1; break; }
            if (bad) {
                char e[4096] = "", o[4096] = "";
                size_t ep = 0, op = 0;
                for (int i = 0; i < n; i++) { ep += (size_t)snprintf(e + ep, sizeof(e) - ep, "%s'%s'", i ? "," : "", basic->u.arr.items[i]->u.s); op += (size_t)snprintf(o + op, sizeof(o) - op, "%s'%s'", i ? "," : "", got.items[i]); }
                fail(id, "basic_tokenize", e, o);
            }
        }
        for (int i = 0; i < got.n; i++) free(got.items[i]);
        free(got.items);
    }

    /* Encode (max_len = lunghezza ids attesa; aspettiamo niente padding) */
    {
        int* exp = (int*)malloc((size_t)nids * sizeof(int));
        int* in = (int*)malloc((size_t)nids * sizeof(int));
        int* am = (int*)malloc((size_t)nids * sizeof(int));
        int* tt = (int*)malloc((size_t)nids * sizeof(int));
        for (int i = 0; i < nids; i++) exp[i] = (int)ids->u.arr.items[i]->u.num;
        llmtoken_wp_encode(t, text, nids, in, am, tt);
        char e[4096], o[4096];
        fmt_ids(exp, nids, e, sizeof(e));
        fmt_ids(in, nids, o, sizeof(o));
        if (memcmp(exp, in, (size_t)nids * sizeof(int)) != 0) fail(id, "encode ids", e, o);
        for (int i = 0; i < nids; i++) if (am[i] != 1) { fail(id, "encode mask (atteso tutto 1)", "all-1", o); break; }
        free(exp); free(in); free(am); free(tt);
    }

    llmtoken_wp_tok_free(t);
}

static void check_pair(jval* c) {
    const char* id = jobj_get(c, "id")->u.s;
    const char* vocab = jobj_get(c, "vocab")->u.s;
    const char* q = jobj_get(c, "query")->u.s;
    const char* d = jobj_get(c, "document")->u.s;
    jval* ids = jobj_get(c, "ids");
    jval* ttypes = jobj_get(c, "token_type_ids");
    int nids = (int)ids->u.arr.n;

    llmtoken_wp_tok* t = load_vocab(vocab);
    if (!t) { fail(id, "load vocab", vocab, "ERR"); return; }
    int* exp = (int*)malloc((size_t)nids * sizeof(int));
    int* expt = (int*)malloc((size_t)nids * sizeof(int));
    int* in = (int*)malloc((size_t)nids * sizeof(int));
    int* am = (int*)malloc((size_t)nids * sizeof(int));
    int* tt = (int*)malloc((size_t)nids * sizeof(int));
    for (int i = 0; i < nids; i++) { exp[i] = (int)ids->u.arr.items[i]->u.num; expt[i] = (int)ttypes->u.arr.items[i]->u.num; }
    llmtoken_wp_encode_pair(t, q, d, nids, in, am, tt);
    char e[4096], o[4096], et[4096], ot[4096];
    fmt_ids(exp, nids, e, sizeof(e)); fmt_ids(in, nids, o, sizeof(o));
    if (memcmp(exp, in, (size_t)nids * sizeof(int)) != 0) fail(id, "encode_pair ids", e, o);
    fmt_ids(expt, nids, et, sizeof(et)); fmt_ids(tt, nids, ot, sizeof(ot));
    if (memcmp(expt, tt, (size_t)nids * sizeof(int)) != 0) fail(id, "encode_pair token_type_ids", et, ot);
    free(exp); free(expt); free(in); free(am); free(tt);
    llmtoken_wp_tok_free(t);
}

static void check_trunc(jval* c) {
    const char* id = jobj_get(c, "id")->u.s;
    const char* vocab = jobj_get(c, "vocab")->u.s;
    const char* text = jobj_get(c, "text")->u.s;
    int max_len = (int)jobj_get(c, "max_length")->u.num;
    jval* ids = jobj_get(c, "ids");
    int nids = (int)ids->u.arr.n;

    llmtoken_wp_tok* t = load_vocab(vocab);
    if (!t) { fail(id, "load vocab", vocab, "ERR"); return; }
    int* exp = (int*)malloc((size_t)nids * sizeof(int));
    int* in = (int*)malloc((size_t)nids * sizeof(int));
    int* am = (int*)malloc((size_t)nids * sizeof(int));
    int* tt = (int*)malloc((size_t)nids * sizeof(int));
    for (int i = 0; i < nids; i++) exp[i] = (int)ids->u.arr.items[i]->u.num;
    llmtoken_wp_encode(t, text, max_len, in, am, tt);
    char e[4096], o[4096];
    fmt_ids(exp, nids, e, sizeof(e)); fmt_ids(in, nids, o, sizeof(o));
    if (memcmp(exp, in, (size_t)nids * sizeof(int)) != 0) fail(id, "trunc ids", e, o);
    if (in[0] != 2) fail(id, "trunc [CLS] preservato", "2", o);
    if (in[max_len - 1] != 3) fail(id, "trunc [SEP] preservato", "3", o);
    free(exp); free(in); free(am); free(tt);
    llmtoken_wp_tok_free(t);
}

/* casi degenerati D2 identici al C# (Encode_DegenerateMaxLengths_KeepMarkers). */
static void check_degenerate(void) {
    llmtoken_wp_tok* t = load_vocab("en_uncased");
    if (!t) { fail("degenerate", "load vocab", "en_uncased", "ERR"); return; }
    {
        int in[2], am[2], tt[2];
        llmtoken_wp_encode(t, "hello world rocket launch", 2, in, am, tt);
        if (in[0] != 2 || in[1] != 3) { char o[64]; fmt_ids(in, 2, o, sizeof(o)); fail("maxlen2", "[CLS][SEP]", "2,3", o); }
    }
    {
        int in[3], am[3], tt[3];
        llmtoken_wp_encode(t, "hello world rocket launch", 3, in, am, tt);
        if (in[0] != 2 || in[1] != 5 || in[2] != 3) { char o[64]; fmt_ids(in, 3, o, sizeof(o)); fail("maxlen3", "[CLS] hello [SEP]", "2,5,3", o); }
    }
    llmtoken_wp_tok_free(t);
}

/* ===========================================================================
 * main
 * =========================================================================== */
static char* dir_fixture(void) {
    static char dir[4096];
    if (main_argc > 1) { strncpy(dir, main_argv[1], sizeof(dir) - 1); dir[sizeof(dir) - 1] = 0; return dir; }
    snprintf(dir, sizeof(dir), "samples/Fixtures/tokenizer");
    return dir;
}
int main(int argc, char** argv) {
    main_argc = argc; main_argv = argv;
    const char* dir = dir_fixture();

    FILE* f = fopen(vpath("tokenizer_golden.json"), "r");
    if (!f) { fprintf(stderr, "ERR: golden non trovato in %s\n", dir); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* js = (char*)malloc((size_t)sz + 1);
    if (fread(js, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(js); return 1; }
    js[sz] = 0; fclose(f);
    const char* cur = js;
    jval* golden = jparse(&cur);
    free(js);
    if (!golden) { fprintf(stderr, "ERR: parse golden fallito\n"); return 1; }

    jval* cases = jobj_get(golden, "cases");
    for (size_t i = 0; i < cases->u.arr.n; i++) check_case(cases->u.arr.items[i]);
    jval* pairs = jobj_get(golden, "pairs");
    for (size_t i = 0; i < pairs->u.arr.n; i++) check_pair(pairs->u.arr.items[i]);
    jval* trunc = jobj_get(golden, "truncation");
    for (size_t i = 0; i < trunc->u.arr.n; i++) check_trunc(trunc->u.arr.items[i]);
    check_degenerate();

    jfree(golden);

    if (g_failures == 0) { printf("llmtoken_wordpiece_selfcheck OK\n"); return 0; }
    printf("llmtoken_wordpiece_selfcheck FAILED: %d divergenze\n", g_failures);
    return 1;
}
