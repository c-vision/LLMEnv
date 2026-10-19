/* llm_grammar.h — decodifica vincolata JSON/JSON-Schema con BITMASK sul vocabolario (header-only, C99, nessuna dipendenza).
 *
 * Contratto (ispirato a SGLang/xgrammar, vedi Knowledge/SGLang-Analisi-e-Materiale-LLMEngine.md, C3 stadio 1):
 *   g = llm_grammar_new_schema(schema_json, n, err, cap)   // oppure llm_grammar_new_json_object()
 *   llm_grammar_set_vocab(g, n_vocab, tok_bytes, tok_len, eos_ids, n_eos)   // una volta per modello/richiesta
 *   per ogni passo:  llm_grammar_fill_mask(g, words)  -> 1 bit per token ammesso (n_words = (n_vocab+31)/32)
 *                    ...il motore sceglie un token fra quelli ammessi...
 *                    llm_grammar_accept(g, token)     -> avanza la grammatica (0 ok / -1 illegale)
 *   llm_grammar_is_done(g): il documento e' completo (l'EOS e' ammesso).
 *
 * Riconoscitore: automa a pila a livello di BYTE. Un token e' ammesso se TUTTI i suoi byte sono accettati in sequenza
 * (simulazione su copia dello stato). Whitespace tra i token JSON consentito (max LLM_GRAMMAR_WS_MAX consecutivi).
 * Sottoinsieme di JSON Schema: type (anche array di tipi), properties/required (proprieta' emesse NELL'ORDINE dello schema,
 * le non-required sono opzionali), items/minItems/maxItems, enum/const, maxLength, anyOf/oneOf (scelta dal primo byte),
 * $ref locali (#/$defs/x, #/definitions/x). additionalProperties: vengono emesse solo le proprieta' dichiarate.
 * Limiti noti: pattern/format/minLength/min/max numerici ignorati; anyOf con rami dal primo byte identico sceglie il primo.
 *
 * Nessuna allocazione nel percorso caldo tranne la cache delle maschere. NON thread-safe per istanza. */
#ifndef LLM_GRAMMAR_H
#define LLM_GRAMMAR_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define LLM_GRAMMAR_WS_MAX 16
#define LLM_GRAMMAR_MAXD 40
#define LLM_GRAMMAR_MAXPROPS 64
#define LLM_GRAMMAR_ANY_DEPTH 6
#define LLM_GRAMMAR_CACHE 32

/* ───────────────────────── mini parser JSON (solo per lo schema) ───────────────────────── */
typedef struct LgJ {
    int t;                       /* 0 null, 1 bool, 2 num, 3 str, 4 arr, 5 obj */
    int b; double num;
    char* s; int sl;             /* stringa decodificata (UTF-8) */
    struct LgJ** kv; char** keys; int n;   /* arr: kv[i]; obj: keys[i]/kv[i] */
    const char* raw; int rawl;   /* testo grezzo del valore (per enum/const) */
} LgJ;

typedef struct { const char* p; const char* end; const char* base; int err; } LgP;

static void lg_ws(LgP* P) { while (P->p < P->end && (*P->p == ' ' || *P->p == '\n' || *P->p == '\t' || *P->p == '\r')) P->p++; }
static void lg_jfree(LgJ* j) {
    if (!j) return;
    for (int i = 0; i < j->n; i++) { lg_jfree(j->kv[i]); if (j->keys) free(j->keys[i]); }
    free(j->kv); free(j->keys); free(j->s); free(j);
}
static LgJ* lg_parse_val(LgP* P, int depth);
static char* lg_parse_str(LgP* P, int* outl) {
    if (P->p >= P->end || *P->p != '"') { P->err = 1; return NULL; }
    P->p++;
    size_t cap = 16, n = 0; char* o = (char*)malloc(cap); if (!o) { P->err = 1; return NULL; }
    while (P->p < P->end && *P->p != '"') {
        unsigned c = (unsigned char)*P->p++;
        if (c == '\\') {
            if (P->p >= P->end) { P->err = 1; break; }
            c = (unsigned char)*P->p++;
            switch (c) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break; case 'r': c = '\r'; break;
                case 'b': c = '\b'; break; case 'f': c = '\f'; break;
                case 'u': {
                    unsigned u = 0;
                    for (int k = 0; k < 4 && P->p < P->end; k++) { char h = *P->p++; u = u * 16 + (unsigned)((h >= 'a') ? h - 'a' + 10 : (h >= 'A') ? h - 'A' + 10 : h - '0'); }
                    if (n + 4 >= cap) { cap *= 2; o = (char*)realloc(o, cap); if (!o) { P->err = 1; return NULL; } }
                    if (u < 0x80) o[n++] = (char)u;
                    else if (u < 0x800) { o[n++] = (char)(0xC0 | (u >> 6)); o[n++] = (char)(0x80 | (u & 63)); }
                    else { o[n++] = (char)(0xE0 | (u >> 12)); o[n++] = (char)(0x80 | ((u >> 6) & 63)); o[n++] = (char)(0x80 | (u & 63)); }
                    continue; }
                default: break;
            }
        }
        if (n + 2 >= cap) { cap *= 2; o = (char*)realloc(o, cap); if (!o) { P->err = 1; return NULL; } }
        o[n++] = (char)c;
    }
    if (P->p >= P->end) { P->err = 1; free(o); return NULL; }
    P->p++; o[n] = 0; *outl = (int)n; return o;
}
static LgJ* lg_parse_val(LgP* P, int depth) {
    lg_ws(P);
    if (P->p >= P->end || depth > 64) { P->err = 1; return NULL; }
    LgJ* j = (LgJ*)calloc(1, sizeof *j); if (!j) { P->err = 1; return NULL; }
    j->raw = P->p;
    char c = *P->p;
    if (c == '{') {
        j->t = 5; P->p++; int cap = 0;
        lg_ws(P);
        if (P->p < P->end && *P->p == '}') { P->p++; }
        else for (;;) {
            lg_ws(P); int kl = 0; char* k = lg_parse_str(P, &kl); if (!k) { P->err = 1; break; }
            lg_ws(P); if (P->p >= P->end || *P->p != ':') { free(k); P->err = 1; break; } P->p++;
            LgJ* v = lg_parse_val(P, depth + 1); if (!v) { free(k); break; }
            if (j->n >= cap) { cap = cap ? cap * 2 : 4; j->kv = (LgJ**)realloc(j->kv, (size_t)cap * sizeof *j->kv); j->keys = (char**)realloc(j->keys, (size_t)cap * sizeof *j->keys); }
            j->keys[j->n] = k; j->kv[j->n++] = v;
            lg_ws(P); if (P->p < P->end && *P->p == ',') { P->p++; continue; }
            if (P->p < P->end && *P->p == '}') { P->p++; break; }
            P->err = 1; break;
        }
    } else if (c == '[') {
        j->t = 4; P->p++; int cap = 0;
        lg_ws(P);
        if (P->p < P->end && *P->p == ']') { P->p++; }
        else for (;;) {
            LgJ* v = lg_parse_val(P, depth + 1); if (!v) break;
            if (j->n >= cap) { cap = cap ? cap * 2 : 4; j->kv = (LgJ**)realloc(j->kv, (size_t)cap * sizeof *j->kv); }
            j->kv[j->n++] = v;
            lg_ws(P); if (P->p < P->end && *P->p == ',') { P->p++; continue; }
            if (P->p < P->end && *P->p == ']') { P->p++; break; }
            P->err = 1; break;
        }
    } else if (c == '"') { j->t = 3; j->s = lg_parse_str(P, &j->sl); }
    else if (c == 't' && P->end - P->p >= 4 && !memcmp(P->p, "true", 4)) { j->t = 1; j->b = 1; P->p += 4; }
    else if (c == 'f' && P->end - P->p >= 5 && !memcmp(P->p, "false", 5)) { j->t = 1; j->b = 0; P->p += 5; }
    else if (c == 'n' && P->end - P->p >= 4 && !memcmp(P->p, "null", 4)) { j->t = 0; P->p += 4; }
    else if (c == '-' || (c >= '0' && c <= '9')) {
        char* e = NULL; j->t = 2; j->num = strtod(P->p, &e); if (e == P->p) P->err = 1; else P->p = e;
    } else P->err = 1;
    j->rawl = (int)(P->p - j->raw);
    if (P->err) { lg_jfree(j); return NULL; }
    return j;
}
static LgJ* lg_get(const LgJ* o, const char* k) {
    if (!o || o->t != 5) return NULL;
    for (int i = 0; i < o->n; i++) if (!strcmp(o->keys[i], k)) return o->kv[i];
    return NULL;
}

/* ───────────────────────── nodi della grammatica ───────────────────────── */
enum { LGN_ANY, LGN_OBJ, LGN_ARR, LGN_STR, LGN_NUM, LGN_INT, LGN_ENUM, LGN_UNION, LGN_ANYOBJ, LGN_ANYARR };

typedef struct {
    uint8_t type;
    int po, np;              /* OBJ: indice/conteggio in props; UNION: indice/conteggio in branches; ENUM: lits */
    int item;                /* ARR: nodo item */
    int minc, maxc;          /* ARR: min/max items; STR: maxc = maxLength (0 = libero) */
    uint8_t first[32];       /* bitset dei primi byte ammessi */
} LgNode;

typedef struct { char* key; int kl; int node; int req; } LgProp;
typedef struct { char* s; int l; } LgLit;

typedef struct { uint16_t node; uint8_t ph; uint8_t aux; uint16_t idx; uint16_t cnt; uint64_t mask; } LgFrame;
typedef struct { int sp, ws, started; LgFrame f[LLM_GRAMMAR_MAXD]; } LgState;

typedef struct { LgState st; uint32_t* words; int cnt; int valid; } LgCacheEnt;

typedef struct LlmGrammar {
    LgNode* nodes; int nn, cn;
    LgProp* props; int np, cp;
    LgLit* lits; int nl, cl;
    int* br; int nb, cb;
    int root;
    int n_str_free, n_num, n_boolnull, n_any_obj, n_any_arr, n_any;   /* nodi canonici per ANY */
    LgState st;
    /* vocabolario */
    int n_vocab, n_words; uint8_t* vb; size_t* voff; int* vlen; uint8_t* plain; uint32_t* plain_words;
    int eos[8]; int n_eos;
    LgCacheEnt cache[LLM_GRAMMAR_CACHE]; int cache_next;
    int hits, misses;
} LlmGrammar;

static int lg_node_new(LlmGrammar* g, int type) {
    if (g->nn >= g->cn) { g->cn = g->cn ? g->cn * 2 : 32; g->nodes = (LgNode*)realloc(g->nodes, (size_t)g->cn * sizeof *g->nodes); if (!g->nodes) return -1; }
    memset(&g->nodes[g->nn], 0, sizeof(LgNode)); g->nodes[g->nn].type = (uint8_t)type;
    return g->nn++;
}
static int lg_prop_new(LlmGrammar* g) {
    if (g->np >= g->cp) { g->cp = g->cp ? g->cp * 2 : 16; g->props = (LgProp*)realloc(g->props, (size_t)g->cp * sizeof *g->props); if (!g->props) return -1; }
    memset(&g->props[g->np], 0, sizeof(LgProp)); return g->np++;
}
static int lg_lit_new(LlmGrammar* g, const char* s, int l) {
    if (g->nl >= g->cl) { g->cl = g->cl ? g->cl * 2 : 16; g->lits = (LgLit*)realloc(g->lits, (size_t)g->cl * sizeof *g->lits); if (!g->lits) return -1; }
    g->lits[g->nl].s = (char*)malloc((size_t)l + 1); if (!g->lits[g->nl].s) return -1;
    memcpy(g->lits[g->nl].s, s, (size_t)l); g->lits[g->nl].s[l] = 0; g->lits[g->nl].l = l; return g->nl++;
}
static int lg_br_new(LlmGrammar* g, int node) {
    if (g->nb >= g->cb) { g->cb = g->cb ? g->cb * 2 : 8; g->br = (int*)realloc(g->br, (size_t)g->cb * sizeof(int)); if (!g->br) return -1; }
    g->br[g->nb] = node; return g->nb++;
}
/* testo JSON di una stringa (con virgolette ed escape) */
static char* lg_json_quote(const char* s, int l, int* outl) {
    char* o = (char*)malloc((size_t)l * 6 + 3); if (!o) return NULL; int n = 0; o[n++] = '"';
    for (int i = 0; i < l; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { o[n++] = '\\'; o[n++] = (char)c; }
        else if (c == '\n') { o[n++] = '\\'; o[n++] = 'n'; } else if (c == '\t') { o[n++] = '\\'; o[n++] = 't'; } else if (c == '\r') { o[n++] = '\\'; o[n++] = 'r'; }
        else if (c < 0x20) n += snprintf(o + n, 8, "\\u%04x", c);
        else o[n++] = (char)c;
    }
    o[n++] = '"'; o[n] = 0; *outl = n; return o;
}
/* serializza un valore JSON in forma compatta (per i letterali enum/const) */
static int lg_ser(const LgJ* j, char** o, size_t* n, size_t* cap) {
    #define LG_PUT(str, l) do { if (*n + (size_t)(l) + 8 > *cap) { *cap = (*cap + (size_t)(l) + 8) * 2; *o = (char*)realloc(*o, *cap); if (!*o) return -1; } memcpy(*o + *n, (str), (size_t)(l)); *n += (size_t)(l); } while (0)
    if (j->t == 0) { LG_PUT("null", 4); }
    else if (j->t == 1) { if (j->b) { LG_PUT("true", 4); } else { LG_PUT("false", 5); } }
    else if (j->t == 2) { char b[40]; int l = snprintf(b, sizeof b, "%.17g", j->num); LG_PUT(b, l); }
    else if (j->t == 3) { int ql; char* q = lg_json_quote(j->s, j->sl, &ql); if (!q) return -1; LG_PUT(q, ql); free(q); }
    else if (j->t == 4) { LG_PUT("[", 1); for (int i = 0; i < j->n; i++) { if (i) LG_PUT(",", 1); if (lg_ser(j->kv[i], o, n, cap)) return -1; } LG_PUT("]", 1); }
    else { LG_PUT("{", 1); for (int i = 0; i < j->n; i++) { int ql; char* q = lg_json_quote(j->keys[i], (int)strlen(j->keys[i]), &ql); if (!q) return -1; if (i) LG_PUT(",", 1); LG_PUT(q, ql); free(q); LG_PUT(":", 1); if (lg_ser(j->kv[i], o, n, cap)) return -1; } LG_PUT("}", 1); }
    return 0;
    #undef LG_PUT
}
static int lg_enum_of(LlmGrammar* g, const LgJ* const* vals, int nv) {
    int nd = lg_node_new(g, LGN_ENUM); if (nd < 0) return -1;
    g->nodes[nd].po = g->nl; g->nodes[nd].np = nv;
    if (nv > LLM_GRAMMAR_MAXPROPS) return -1;
    for (int i = 0; i < nv; i++) {
        char* o = (char*)malloc(64); size_t n = 0, cap = 64; if (!o) return -1;
        if (lg_ser(vals[i], &o, &n, &cap)) { free(o); return -1; }
        int r = lg_lit_new(g, o, (int)n); free(o); if (r < 0) return -1;
    }
    return nd;
}
static int lg_enum_text(LlmGrammar* g, const char* const* txt, int nv) {
    int nd = lg_node_new(g, LGN_ENUM); if (nd < 0) return -1;
    g->nodes[nd].po = g->nl; g->nodes[nd].np = nv;
    for (int i = 0; i < nv; i++) if (lg_lit_new(g, txt[i], (int)strlen(txt[i])) < 0) return -1;
    return nd;
}

static const LgJ* lg_resolve_ref(const LgJ* root, const char* ref) {
    if (ref[0] != '#' || ref[1] != '/') return NULL;
    const LgJ* cur = root; const char* p = ref + 2;
    while (*p && cur) {
        char key[128]; int k = 0; while (*p && *p != '/' && k < 127) key[k++] = *p++; key[k] = 0; if (*p == '/') p++;
        cur = lg_get(cur, key);
    }
    return cur;
}

static int lg_compile(LlmGrammar* g, const LgJ* root, const LgJ* sc, int depth, char* err, size_t ec);
static int lg_compile_type(LlmGrammar* g, const LgJ* root, const LgJ* sc, const char* ty, int depth, char* err, size_t ec) {
    if (!strcmp(ty, "string")) {
        int nd = lg_node_new(g, LGN_STR); if (nd < 0) return -1;
        const LgJ* ml = lg_get(sc, "maxLength"); if (ml && ml->t == 2 && ml->num >= 1 && ml->num < 65000) g->nodes[nd].maxc = (int)ml->num;
        return nd;
    }
    if (!strcmp(ty, "integer")) return lg_node_new(g, LGN_INT);
    if (!strcmp(ty, "number")) return lg_node_new(g, LGN_NUM);
    if (!strcmp(ty, "boolean")) { const char* t[2] = { "true", "false" }; return lg_enum_text(g, t, 2); }
    if (!strcmp(ty, "null")) { const char* t[1] = { "null" }; return lg_enum_text(g, t, 1); }
    if (!strcmp(ty, "array")) {
        int nd = lg_node_new(g, LGN_ARR); if (nd < 0) return -1;
        const LgJ* it = lg_get(sc, "items");
        int item = it ? lg_compile(g, root, it, depth + 1, err, ec) : -1;
        if (!it) { item = lg_node_new(g, LGN_ANY); }
        if (item < 0) return -1;
        g->nodes[nd].item = item;
        const LgJ* mn = lg_get(sc, "minItems"); const LgJ* mx = lg_get(sc, "maxItems");
        g->nodes[nd].minc = (mn && mn->t == 2 && mn->num > 0) ? (int)mn->num : 0;
        g->nodes[nd].maxc = (mx && mx->t == 2 && mx->num >= 0) ? (int)mx->num : 0;
        if (mx && mx->t == 2 && mx->num == 0) g->nodes[nd].maxc = -1;   /* maxItems=0: array vuoto */
        return nd;
    }
    if (!strcmp(ty, "object")) {
        const LgJ* pr = lg_get(sc, "properties");
        if (!pr || pr->t != 5 || pr->n == 0) return lg_node_new(g, LGN_ANYOBJ);
        if (pr->n > LLM_GRAMMAR_MAXPROPS) { snprintf(err, ec, "troppe proprieta' (max %d)", LLM_GRAMMAR_MAXPROPS); return -1; }
        int nd = lg_node_new(g, LGN_OBJ); if (nd < 0) return -1;
        const LgJ* rq = lg_get(sc, "required");
        int base = g->np;
        g->nodes[nd].po = base; g->nodes[nd].np = pr->n;
        for (int i = 0; i < pr->n; i++) if (lg_prop_new(g) < 0) return -1;
        for (int i = 0; i < pr->n; i++) {
            int ql; char* q = lg_json_quote(pr->keys[i], (int)strlen(pr->keys[i]), &ql); if (!q) return -1;
            g->props[base + i].key = q + 1; g->props[base + i].kl = ql - 2;   /* senza virgolette (il key-match lavora sui byte interni) */
            g->props[base + i].req = 0;
            if (rq && rq->t == 4) for (int r = 0; r < rq->n; r++) if (rq->kv[r]->t == 3 && !strcmp(rq->kv[r]->s, pr->keys[i])) g->props[base + i].req = 1;
            int cn = lg_compile(g, root, pr->kv[i], depth + 1, err, ec); if (cn < 0) return -1;
            g->props[base + i].node = cn;
        }
        return nd;
    }
    snprintf(err, ec, "tipo non supportato: %s", ty); return -1;
}
static int lg_compile(LlmGrammar* g, const LgJ* root, const LgJ* sc, int depth, char* err, size_t ec) {
    if (depth > 12) return lg_node_new(g, LGN_ANY);
    if (!sc || sc->t != 5) return lg_node_new(g, LGN_ANY);   /* true / {} = qualsiasi valore */
    const LgJ* rf = lg_get(sc, "$ref");
    if (rf && rf->t == 3) { const LgJ* t = lg_resolve_ref(root, rf->s); if (!t) { snprintf(err, ec, "$ref non risolvibile: %s", rf->s); return -1; } return lg_compile(g, root, t, depth + 1, err, ec); }
    const LgJ* en = lg_get(sc, "enum"); const LgJ* cs = lg_get(sc, "const");
    if (en && en->t == 4 && en->n > 0) return lg_enum_of(g, (const LgJ* const*)en->kv, en->n);
    if (cs) { const LgJ* v[1] = { cs }; return lg_enum_of(g, v, 1); }
    const LgJ* uo = lg_get(sc, "anyOf"); if (!uo) uo = lg_get(sc, "oneOf");
    if (uo && uo->t == 4 && uo->n > 0) {
        int nd = lg_node_new(g, LGN_UNION); if (nd < 0) return -1;
        int ids[16]; int k = 0;
        for (int i = 0; i < uo->n && k < 16; i++) { int c = lg_compile(g, root, uo->kv[i], depth + 1, err, ec); if (c < 0) return -1; ids[k++] = c; }
        g->nodes[nd].po = g->nb; g->nodes[nd].np = k;
        for (int i = 0; i < k; i++) if (lg_br_new(g, ids[i]) < 0) return -1;
        return nd;
    }
    const LgJ* ty = lg_get(sc, "type");
    if (ty && ty->t == 4 && ty->n > 0) {   /* "type": ["string","null"] -> unione */
        int nd = lg_node_new(g, LGN_UNION); if (nd < 0) return -1;
        int ids[16]; int k = 0;
        for (int i = 0; i < ty->n && k < 16; i++) { if (ty->kv[i]->t != 3) continue; int c = lg_compile_type(g, root, sc, ty->kv[i]->s, depth + 1, err, ec); if (c < 0) return -1; ids[k++] = c; }
        g->nodes[nd].po = g->nb; g->nodes[nd].np = k;
        for (int i = 0; i < k; i++) if (lg_br_new(g, ids[i]) < 0) return -1;
        return nd;
    }
    if (ty && ty->t == 3) return lg_compile_type(g, root, sc, ty->s, depth, err, ec);
    if (lg_get(sc, "properties")) return lg_compile_type(g, root, sc, "object", depth, err, ec);
    if (lg_get(sc, "items")) return lg_compile_type(g, root, sc, "array", depth, err, ec);
    return lg_node_new(g, LGN_ANY);
}

static void lg_first_set(LlmGrammar* g, int nd, uint8_t* out, int depth) {
    if (depth > 20) return;
    LgNode* n = &g->nodes[nd];
    #define LG_SET(c) out[(uint8_t)(c) >> 3] |= (uint8_t)(1u << ((uint8_t)(c) & 7))
    switch (n->type) {
        case LGN_OBJ: case LGN_ANYOBJ: LG_SET('{'); break;
        case LGN_ARR: case LGN_ANYARR: LG_SET('['); break;
        case LGN_STR: LG_SET('"'); break;
        case LGN_NUM: case LGN_INT: LG_SET('-'); for (int c = '0'; c <= '9'; c++) LG_SET(c); break;
        case LGN_ENUM: for (int i = 0; i < n->np; i++) if (g->lits[n->po + i].l) LG_SET(g->lits[n->po + i].s[0]); break;
        case LGN_UNION: for (int i = 0; i < n->np; i++) lg_first_set(g, g->br[n->po + i], out, depth + 1); break;
        case LGN_ANY: LG_SET('{'); LG_SET('['); LG_SET('"'); LG_SET('-'); for (int c = '0'; c <= '9'; c++) LG_SET(c); LG_SET('t'); LG_SET('f'); LG_SET('n'); break;
    }
    #undef LG_SET
}

static void lg_free(LlmGrammar* g) {
    if (!g) return;
    for (int i = 0; i < g->np; i++) if (g->props[i].key) free(g->props[i].key - 1);
    for (int i = 0; i < g->nl; i++) free(g->lits[i].s);
    for (int i = 0; i < LLM_GRAMMAR_CACHE; i++) free(g->cache[i].words);
    free(g->nodes); free(g->props); free(g->lits); free(g->br);
    free(g->vb); free(g->voff); free(g->vlen); free(g->plain); free(g->plain_words);
    free(g);
}
#define llm_grammar_free lg_free

static LlmGrammar* lg_init_common(void) {
    LlmGrammar* g = (LlmGrammar*)calloc(1, sizeof *g); if (!g) return NULL;
    /* nodi canonici per i valori "qualsiasi" */
    g->n_str_free = lg_node_new(g, LGN_STR);
    g->n_num = lg_node_new(g, LGN_NUM);
    { const char* t[3] = { "true", "false", "null" }; g->n_boolnull = lg_enum_text(g, t, 3); }
    g->n_any_obj = lg_node_new(g, LGN_ANYOBJ);
    g->n_any_arr = lg_node_new(g, LGN_ANYARR);
    g->n_any = lg_node_new(g, LGN_ANY);
    if (g->n_any_arr < 0 || g->n_boolnull < 0) { lg_free(g); return NULL; }
    return g;
}
static void lg_finish(LlmGrammar* g) {
    for (int i = 0; i < g->nn; i++) memset(g->nodes[i].first, 0, 32);
    for (int i = 0; i < g->nn; i++) lg_first_set(g, i, g->nodes[i].first, 0);
    memset(&g->st, 0, sizeof g->st);
}
static LlmGrammar* llm_grammar_new_json_object(void) {
    LlmGrammar* g = lg_init_common(); if (!g) return NULL;
    g->root = g->n_any_obj; lg_finish(g); return g;
}
static LlmGrammar* llm_grammar_new_schema(const char* schema, size_t n, char* err, size_t ec) {
    LgP P = { schema, schema + n, schema, 0 };
    LgJ* root = lg_parse_val(&P, 0);
    if (!root) { if (ec) snprintf(err, ec, "schema JSON non valido"); return NULL; }
    LlmGrammar* g = lg_init_common(); if (!g) { lg_jfree(root); return NULL; }
    if (ec) err[0] = 0;
    char e2[160] = { 0 };
    int r = lg_compile(g, root, root, 0, e2, sizeof e2);
    lg_jfree(root);
    if (r < 0) { if (ec) snprintf(err, ec, "%s", e2[0] ? e2 : "schema non compilabile"); lg_free(g); return NULL; }
    g->root = r; lg_finish(g); return g;
}

/* ───────────────────────── riconoscitore (a byte) ───────────────────────── */
enum { LGR_REJ = -1, LGR_NOT = 0, LGR_CONS = 1, LGR_RE = 2, LGR_POPRE = 3 };
/* fasi */
enum { P_START = 0,
       O_FIRST, O_KEYQ, O_KEY, O_COLON, O_VALWAIT, O_AFTER,
       A_FIRST, A_AFTER, A_VALWAIT,
       S_BODY, S_ESC, S_U,
       NU_SIGN, NU_ZERO, NU_INT, NU_FRAC0, NU_FRAC, NU_EXP0, NU_EXP1, NU_EXP,
       E_BODY,
       AO_FIRST, AO_COLON, AO_VALWAIT, AO_AFTER, AO_KEYQ,
       AA_FIRST, AA_AFTER, AA_VALWAIT };

static int lg_isws(uint8_t c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
static int lg_push(LgState* s, int node, int aux) {
    if (s->sp >= LLM_GRAMMAR_MAXD) return 0;
    LgFrame* f = &s->f[s->sp++]; memset(f, 0, sizeof *f); f->node = (uint16_t)node; f->aux = (uint8_t)aux; return 1;
}
static void lg_pop(LgState* s) { s->sp--; }
static int lg_ws_eat(LgState* s) { if (s->ws >= LLM_GRAMMAR_WS_MAX) return LGR_REJ; s->ws++; return LGR_CONS; }

static int lg_feed(const LlmGrammar* g, LgState* s, uint8_t c) {
    LgFrame* f = &s->f[s->sp - 1];
    const LgNode* n = &g->nodes[f->node];
    switch (n->type) {
    case LGN_OBJ: {
        const LgProp* P = &g->props[n->po];
        switch (f->ph) {
        case P_START: if (c == '{') { f->ph = O_FIRST; return LGR_CONS; } return LGR_REJ;
        case O_FIRST: case O_KEYQ: {
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == '}' && f->ph == O_FIRST) {
                for (int i = f->cnt; i < n->np; i++) if (P[i].req) return LGR_REJ;
                lg_pop(s); return LGR_CONS;
            }
            if (c != '"') return LGR_REJ;
            uint64_t m = 0;
            for (int i = f->cnt; i < n->np; i++) { m |= (1ull << i); if (P[i].req) break; }
            if (!m) return LGR_REJ;
            f->mask = m; f->idx = 0; f->ph = O_KEY; return LGR_CONS;
        }
        case O_KEY: {
            if (c == '"') {
                for (int i = 0; i < n->np; i++) if ((f->mask >> i) & 1u) if (P[i].kl == f->idx) { f->aux = (uint8_t)i; f->cnt = (uint16_t)(i + 1); f->ph = O_COLON; return LGR_CONS; }
                return LGR_REJ;
            }
            uint64_t m = 0;
            for (int i = 0; i < n->np; i++) if (((f->mask >> i) & 1u) && f->idx < P[i].kl && (uint8_t)P[i].key[f->idx] == c) m |= (1ull << i);
            if (!m) return LGR_REJ;
            f->mask = m; f->idx++; return LGR_CONS;
        }
        case O_COLON: if (lg_isws(c)) return lg_ws_eat(s); if (c == ':') { f->ph = O_VALWAIT; return LGR_CONS; } return LGR_REJ;
        case O_VALWAIT: {
            if (lg_isws(c)) return lg_ws_eat(s);
            int child = P[f->aux].node; f->ph = O_AFTER;
            return lg_push(s, child, 0) ? LGR_RE : LGR_REJ;
        }
        case O_AFTER: {
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == ',') { if (f->cnt >= n->np) return LGR_REJ; f->ph = O_KEYQ; return LGR_CONS; }
            if (c == '}') { for (int i = f->cnt; i < n->np; i++) if (P[i].req) return LGR_REJ; lg_pop(s); return LGR_CONS; }
            return LGR_REJ;
        }
        }
        return LGR_REJ;
    }
    case LGN_ARR: {
        switch (f->ph) {
        case P_START: if (c == '[') { f->ph = A_FIRST; return LGR_CONS; } return LGR_REJ;
        case A_FIRST:
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == ']') { if (n->minc > 0) return LGR_REJ; lg_pop(s); return LGR_CONS; }
            if (n->maxc < 0) return LGR_REJ;
            f->cnt = 1; f->ph = A_AFTER; return lg_push(s, n->item, 0) ? LGR_RE : LGR_REJ;
        case A_AFTER:
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == ',') { if (n->maxc > 0 && f->cnt >= n->maxc) return LGR_REJ; if (f->cnt >= 4000) return LGR_REJ; f->ph = A_VALWAIT; return LGR_CONS; }
            if (c == ']') { if (f->cnt < n->minc) return LGR_REJ; lg_pop(s); return LGR_CONS; }
            return LGR_REJ;
        case A_VALWAIT:
            if (lg_isws(c)) return lg_ws_eat(s);
            f->cnt++; f->ph = A_AFTER; return lg_push(s, n->item, 0) ? LGR_RE : LGR_REJ;
        }
        return LGR_REJ;
    }
    case LGN_STR: {
        switch (f->ph) {
        case P_START: if (c == '"') { f->ph = S_BODY; return LGR_CONS; } return LGR_REJ;
        case S_BODY:
            if (c == '"') { lg_pop(s); return LGR_CONS; }
            if (c == '\\') { if (n->maxc) { if (f->cnt >= n->maxc) return LGR_REJ; f->cnt++; } f->ph = S_ESC; return LGR_CONS; }
            if (c < 0x20) return LGR_REJ;
            if (n->maxc && (c & 0xC0) != 0x80) { if (f->cnt >= n->maxc) return LGR_REJ; f->cnt++; }
            return LGR_CONS;
        case S_ESC:
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
                
                f->ph = S_BODY; return LGR_CONS;
            }
            if (c == 'u') { f->ph = S_U; f->aux = 4; return LGR_CONS; }
            return LGR_REJ;
        case S_U:
            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) { if (--f->aux == 0) f->ph = S_BODY; return LGR_CONS; }
            return LGR_REJ;
        }
        return LGR_REJ;
    }
    case LGN_NUM: case LGN_INT: {
        const int isint = n->type == LGN_INT;
        const int dig = (c >= '0' && c <= '9');
        switch (f->ph) {
        case P_START: if (c == '-') { f->ph = NU_SIGN; return LGR_CONS; } if (c == '0') { f->ph = NU_ZERO; return LGR_CONS; } if (dig) { f->ph = NU_INT; f->cnt = 1; return LGR_CONS; } return LGR_REJ;
        case NU_SIGN: if (c == '0') { f->ph = NU_ZERO; return LGR_CONS; } if (dig) { f->ph = NU_INT; f->cnt = 1; return LGR_CONS; } return LGR_REJ;
        case NU_ZERO: if (!isint && c == '.') { f->ph = NU_FRAC0; return LGR_CONS; } if (!isint && (c == 'e' || c == 'E')) { f->ph = NU_EXP0; return LGR_CONS; } lg_pop(s); return LGR_POPRE;
        case NU_INT:
            if (dig) { if (f->cnt >= 18) return LGR_REJ; f->cnt++; return LGR_CONS; }
            if (!isint && c == '.') { f->ph = NU_FRAC0; return LGR_CONS; }
            if (!isint && (c == 'e' || c == 'E')) { f->ph = NU_EXP0; return LGR_CONS; }
            lg_pop(s); return LGR_POPRE;
        case NU_FRAC0: if (dig) { f->ph = NU_FRAC; f->cnt = 1; return LGR_CONS; } return LGR_REJ;
        case NU_FRAC:
            if (dig) { if (f->cnt >= 18) return LGR_REJ; f->cnt++; return LGR_CONS; }
            if (c == 'e' || c == 'E') { f->ph = NU_EXP0; return LGR_CONS; }
            lg_pop(s); return LGR_POPRE;
        case NU_EXP0: if (c == '+' || c == '-') { f->ph = NU_EXP1; return LGR_CONS; } if (dig) { f->ph = NU_EXP; f->cnt = 1; return LGR_CONS; } return LGR_REJ;
        case NU_EXP1: if (dig) { f->ph = NU_EXP; f->cnt = 1; return LGR_CONS; } return LGR_REJ;
        case NU_EXP: if (dig) { if (f->cnt >= 4) return LGR_REJ; f->cnt++; return LGR_CONS; } lg_pop(s); return LGR_POPRE;
        }
        return LGR_REJ;
    }
    case LGN_ENUM: {
        const LgLit* L = &g->lits[n->po];
        uint64_t m;
        int idx;
        if (f->ph == P_START) { m = (n->np >= 64) ? ~0ull : ((1ull << n->np) - 1); idx = 0; f->ph = E_BODY; f->mask = m; f->idx = 0; }
        else { m = f->mask; idx = f->idx; }
        uint64_t nm = 0;
        for (int i = 0; i < n->np; i++) if (((m >> i) & 1u) && idx < L[i].l && (uint8_t)L[i].s[idx] == c) nm |= (1ull << i);
        if (!nm) {
            if (f->ph == E_BODY && f->idx > 0) { for (int i = 0; i < n->np; i++) if (((m >> i) & 1u) && L[i].l == idx) { lg_pop(s); return LGR_POPRE; } }
            return LGR_REJ;
        }
        f->mask = nm; f->idx = (uint16_t)(idx + 1);
        int longer = 0;
        for (int i = 0; i < n->np; i++) if (((nm >> i) & 1u) && L[i].l > f->idx) longer = 1;
        if (!longer) { lg_pop(s); return LGR_CONS; }
        return LGR_CONS;
    }
    case LGN_UNION: {
        for (int i = 0; i < n->np; i++) {
            int b = g->br[n->po + i];
            if ((g->nodes[b].first[c >> 3] >> (c & 7)) & 1u) { f->node = (uint16_t)b; f->ph = 0; return LGR_RE; }
        }
        return LGR_REJ;
    }
    case LGN_ANY: {
        const int d = f->aux;
        int tgt = -1, aux2 = 0;
        if (c == '{') { if (d >= LLM_GRAMMAR_ANY_DEPTH) return LGR_REJ; tgt = g->n_any_obj; aux2 = d + 1; }
        else if (c == '[') { if (d >= LLM_GRAMMAR_ANY_DEPTH) return LGR_REJ; tgt = g->n_any_arr; aux2 = d + 1; }
        else if (c == '"') tgt = g->n_str_free;
        else if (c == '-' || (c >= '0' && c <= '9')) tgt = g->n_num;
        else if (c == 't' || c == 'f' || c == 'n') tgt = g->n_boolnull;
        else return LGR_REJ;
        f->node = (uint16_t)tgt; f->ph = 0; f->aux = (uint8_t)aux2; return LGR_RE;
    }
    case LGN_ANYOBJ: {
        switch (f->ph) {
        case P_START: if (c == '{') { f->ph = AO_FIRST; return LGR_CONS; } return LGR_REJ;
        case AO_FIRST: case AO_KEYQ:
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == '}' && f->ph == AO_FIRST) { lg_pop(s); return LGR_CONS; }
            if (c == '"') { f->ph = AO_COLON; return lg_push(s, g->n_str_free, 0) ? LGR_RE : LGR_REJ; }
            return LGR_REJ;
        case AO_COLON: if (lg_isws(c)) return lg_ws_eat(s); if (c == ':') { f->ph = AO_VALWAIT; return LGR_CONS; } return LGR_REJ;
        case AO_VALWAIT: if (lg_isws(c)) return lg_ws_eat(s); { int d = f->aux; f->ph = AO_AFTER; return lg_push(s, g->n_any, d) ? LGR_RE : LGR_REJ; }
        case AO_AFTER:
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == ',') { f->ph = AO_KEYQ; return LGR_CONS; }
            if (c == '}') { lg_pop(s); return LGR_CONS; }
            return LGR_REJ;
        }
        return LGR_REJ;
    }
    case LGN_ANYARR: {
        switch (f->ph) {
        case P_START: if (c == '[') { f->ph = AA_FIRST; return LGR_CONS; } return LGR_REJ;
        case AA_FIRST:
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == ']') { lg_pop(s); return LGR_CONS; }
            { int d = f->aux; f->ph = AA_AFTER; return lg_push(s, g->n_any, d) ? LGR_RE : LGR_REJ; }
        case AA_AFTER:
            if (lg_isws(c)) return lg_ws_eat(s);
            if (c == ',') { f->ph = AA_VALWAIT; return LGR_CONS; }
            if (c == ']') { lg_pop(s); return LGR_CONS; }
            return LGR_REJ;
        case AA_VALWAIT:
            if (lg_isws(c)) return lg_ws_eat(s);
            { int d = f->aux; f->ph = AA_AFTER; return lg_push(s, g->n_any, d) ? LGR_RE : LGR_REJ; }
        }
        return LGR_REJ;
    }
    }
    return LGR_REJ;
}

/* Un byte nella pila: gestisce re-dispatch (push/replace) e pop lazy. 0 = ok, -1 = rifiutato. */
static int lg_step(const LlmGrammar* g, LgState* s, uint8_t c) {
    for (int guard = 0; guard < 4 * LLM_GRAMMAR_MAXD + 16; guard++) {
        if (s->sp == 0) {
            if (!s->started) {
                if (lg_isws(c)) { if (s->ws >= LLM_GRAMMAR_WS_MAX) return -1; s->ws++; return 0; }
                s->started = 1;
                if (!lg_push(s, g->root, 0)) return -1;
                continue;
            }
            if (lg_isws(c)) { if (s->ws >= LLM_GRAMMAR_WS_MAX) return -1; s->ws++; return 0; }   /* documento completo: solo whitespace */
            return -1;
        }
        int r = lg_feed(g, s, c);
        if (r == LGR_CONS) { if (!lg_isws(c)) s->ws = 0; return 0; }
        if (r == LGR_RE || r == LGR_POPRE) continue;
        return -1;
    }
    return -1;
}
static int lg_done(const LlmGrammar* g, const LgState* s) { (void)g; return s->started && s->sp == 0; }
/* Il documento puo' terminare qui? (anche con un numero in corso: pop lazy di frame terminali) */
static int lg_can_end(const LlmGrammar* g, const LgState* st) {
    LgState t = *st;
    if (!t.started) return 0;
    while (t.sp > 0) {
        const LgFrame* f = &t.f[t.sp - 1]; const LgNode* n = &g->nodes[f->node];
        int ok = 0;
        if (n->type == LGN_NUM || n->type == LGN_INT) ok = (f->ph == NU_ZERO || f->ph == NU_INT || f->ph == NU_FRAC || f->ph == NU_EXP);
        else if (n->type == LGN_ENUM && f->ph == E_BODY) { for (int i = 0; i < n->np; i++) if (((f->mask >> i) & 1u) && g->lits[n->po + i].l == f->idx) ok = 1; }
        if (!ok) return 0;
        t.sp--;
    }
    return 1;
}

/* ───────────────────────── API: vocabolario, maschera, accept ───────────────────────── */
static int llm_grammar_set_vocab(LlmGrammar* g, int n_vocab, const uint8_t* const* tok, const int* tlen, const int* eos, int n_eos) {
    free(g->vb); free(g->voff); free(g->vlen); free(g->plain); free(g->plain_words);
    g->n_vocab = n_vocab; g->n_words = (n_vocab + 31) / 32;
    size_t tot = 0; for (int i = 0; i < n_vocab; i++) tot += (size_t)(tlen[i] > 0 ? tlen[i] : 0);
    g->vb = (uint8_t*)malloc(tot + 1); g->voff = (size_t*)malloc(sizeof(size_t) * (size_t)(n_vocab + 1)); g->vlen = (int*)malloc(sizeof(int) * (size_t)n_vocab);
    g->plain = (uint8_t*)calloc((size_t)n_vocab, 1); g->plain_words = (uint32_t*)calloc((size_t)g->n_words, 4);
    if (!g->vb || !g->voff || !g->vlen || !g->plain || !g->plain_words) return -1;
    size_t o = 0;
    for (int i = 0; i < n_vocab; i++) {
        int l = tlen[i] > 0 ? tlen[i] : 0; g->voff[i] = o; g->vlen[i] = l;
        if (l) memcpy(g->vb + o, tok[i], (size_t)l);
        int plain = l > 0;
        for (int k = 0; k < l && plain; k++) { uint8_t b = tok[i][k]; if (b < 0x20 || b == '"' || b == '\\') plain = 0; }
        g->plain[i] = (uint8_t)plain; if (plain) g->plain_words[i >> 5] |= 1u << (i & 31);
        o += (size_t)l;
    }
    g->voff[n_vocab] = o;
    g->n_eos = n_eos > 8 ? 8 : n_eos; for (int i = 0; i < g->n_eos; i++) g->eos[i] = eos[i];
    for (int i = 0; i < LLM_GRAMMAR_CACHE; i++) { free(g->cache[i].words); g->cache[i].words = NULL; g->cache[i].valid = 0; }
    return 0;
}
static int lg_is_eos(const LlmGrammar* g, int id) { for (int i = 0; i < g->n_eos; i++) if (g->eos[i] == id) return 1; return 0; }

/* Riempie la maschera (n_words = (n_vocab+31)/32 uint32). Ritorna il numero di token ammessi (0 = grammatica morta). */
static int llm_grammar_fill_mask(LlmGrammar* g, uint32_t* words) {
    const LgState* st = &g->st;
    /* cache per stato (le posizioni ripetute -- stesso contesto -- non rifanno la scansione) */
    size_t ssz = offsetof(LgState, f) + (size_t)st->sp * sizeof(LgFrame);
    for (int i = 0; i < LLM_GRAMMAR_CACHE; i++) {
        LgCacheEnt* e = &g->cache[i];
        if (e->valid && e->st.sp == st->sp && e->st.ws == st->ws && e->st.started == st->started && !memcmp(&e->st, st, ssz)) {
            memcpy(words, e->words, (size_t)g->n_words * 4); g->hits++; return e->cnt;
        }
    }
    g->misses++;
    memset(words, 0, (size_t)g->n_words * 4);
    uint8_t first_ok[256];
    for (int b = 0; b < 256; b++) { LgState t = *st; first_ok[b] = (uint8_t)(lg_step(g, &t, (uint8_t)b) == 0); }
    int in_free_str = 0;
    if (st->sp > 0) { const LgFrame* f = &st->f[st->sp - 1]; const LgNode* n = &g->nodes[f->node]; in_free_str = (n->type == LGN_STR && f->ph == S_BODY && n->maxc == 0); }
    int cnt = 0;
    const size_t cp = offsetof(LgState, f) + (size_t)st->sp * sizeof(LgFrame);
    for (int t = 0; t < g->n_vocab; t++) {
        const int l = g->vlen[t];
        if (l == 0) continue;
        const uint8_t* b = g->vb + g->voff[t];
        if (!first_ok[b[0]]) continue;
        if (in_free_str && g->plain[t]) { words[t >> 5] |= 1u << (t & 31); cnt++; continue; }
        LgState tmp; memcpy(&tmp, st, cp);
        int ok = 1;
        for (int k = 0; k < l; k++) if (lg_step(g, &tmp, b[k]) != 0) { ok = 0; break; }
        if (ok) { words[t >> 5] |= 1u << (t & 31); cnt++; }
    }
    if (lg_can_end(g, st)) for (int i = 0; i < g->n_eos; i++) { int id = g->eos[i]; if (id >= 0 && id < g->n_vocab && !(words[id >> 5] & (1u << (id & 31)))) { words[id >> 5] |= 1u << (id & 31); cnt++; } }
    if (cnt == 0) for (int i = 0; i < g->n_eos; i++) { int id = g->eos[i]; if (id >= 0 && id < g->n_vocab) { words[id >> 5] |= 1u << (id & 31); cnt++; } }   /* grammatica morta: lascia chiudere */
    LgCacheEnt* e = &g->cache[g->cache_next]; g->cache_next = (g->cache_next + 1) % LLM_GRAMMAR_CACHE;
    free(e->words); e->words = (uint32_t*)malloc((size_t)g->n_words * 4);
    if (e->words) { memcpy(e->words, words, (size_t)g->n_words * 4); memcpy(&e->st, st, cp); e->st.sp = st->sp; e->st.ws = st->ws; e->st.started = st->started; e->cnt = cnt; e->valid = 1; }
    return cnt;
}
/* Avanza la grammatica con il token scelto. 0 = ok, -1 = illegale. L'EOS e' ammesso solo a documento completo. */
static int llm_grammar_accept(LlmGrammar* g, int id) {
    if (id < 0 || id >= g->n_vocab) return -1;
    if (lg_is_eos(g, id)) return lg_can_end(g, &g->st) ? 0 : -1;
    const int l = g->vlen[id]; if (l == 0) return -1;
    LgState t = g->st; const uint8_t* b = g->vb + g->voff[id];
    for (int k = 0; k < l; k++) if (lg_step(g, &t, b[k]) != 0) return -1;
    g->st = t; return 0;
}
static int llm_grammar_is_done(const LlmGrammar* g) { return lg_can_end(g, &g->st); }
static void llm_grammar_reset(LlmGrammar* g) { memset(&g->st, 0, sizeof g->st); }
#endif /* LLM_GRAMMAR_H */
