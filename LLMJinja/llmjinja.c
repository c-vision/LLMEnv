/* llmjinja.c — motore Jinja VERSIONE C, subset per i chat-template LLM locali.
 *
 * PORT DA GMBasic/native/gmb_jinja (stesso autore, Gaetano Minardi), ricopiato qui come
 * LLMJinja: semantica Jinja2 per i costrutti usati dai template HF, fail-loud (costrutto non
 * supportato -> errore esplicito, mai output approssimato), keep_trailing_newline=false.
 * Niente stub. API pubblica in llmjinja.h.
 *
 * Ownership: i valori INTERNI al render vivono in un'arena liberata a fine render; i valori
 * del chiamante (llmjinja_str/obj/arr/...) sono malloc e vengono copiati nell'arena
 * all'ingresso (env_from_vars).
 */
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct llmjinja_val jval;   /* tag condiviso con llmjinja.h */
typedef struct env env;
typedef struct node node;
typedef struct { node** p; size_t n, cap; } nlist;

typedef enum { J_UNDEF, J_NULL, J_BOOL, J_LONG, J_DOUBLE, J_STR, J_ARR, J_OBJ, J_VAR, J_FN, J_MACRO } jtype;
typedef jval* (*jfn)(int n, jval** a);

struct llmjinja_val;   /* fwd per field */

typedef struct { char** pname; char** pdef; size_t np; nlist body; env* closure; } jmacro;

struct llmjinja_val {
    jtype t;
    struct arena* A;       /* arena di appartenenza (NULL = malloc-based, del chiamante) */
    union {
        int b;
        long l;
        double d;
        char* s;   /* J_STR/J_VAR (owned) */
        struct { jval** items; size_t n, cap; } arr;
        struct { struct { char* k; jval* v; }* items; size_t n, cap; } obj;
        jfn f;
        jmacro* m;
    } u;
};
typedef struct llmjinja_val llmjinja_val;   /* alias nome-tipo (llmjinja.h) */
typedef jfn llmjinja_fn;

/* ================= error ================= */
static char g_err[512];
static void jerr_set(const char* fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(g_err, sizeof g_err, fmt, ap); va_end(ap); }
#define JERR(...) do { jerr_set(__VA_ARGS__); return NULL; } while (0)

/* ================= arena ================= */
typedef struct apnode { struct apnode* next; } apnode;
typedef struct arena { apnode* head; } arena;
static void* a_alloc(arena* A, size_t n) {
    if (!n) n = 1;
    apnode* p = (apnode*)malloc(sizeof(apnode) + n);
    p->next = A->head; A->head = p;
    return (char*)p + sizeof(apnode);
}
static char* a_strdup(arena* A, const char* s) {
    s = s ? s : "";
    size_t l = strlen(s);
    char* p = (char*)a_alloc(A, l + 1);
    memcpy(p, s, l + 1);
    return p;
}
static char* a_strndup(arena* A, const char* s, size_t l) {
    char* p = (char*)a_alloc(A, l + 1);
    memcpy(p, s, l); p[l] = 0;
    return p;
}
static void a_free(arena* A) { apnode* p = A->head; while (p) { apnode* n = p->next; free(p); p = n; } A->head = NULL; }

/* ================= sb ================= */
typedef struct { char* p; size_t n, cap; } sb;
static void sb_init(sb* b) { b->cap = 256; b->n = 0; b->p = (char*)malloc(b->cap); b->p[0] = 0; }
static void sb_grow(sb* b, size_t need) { while (b->n + need + 1 > b->cap) { b->cap *= 2; b->p = (char*)realloc(b->p, b->cap); } }
static void sb_put(sb* b, const char* s) { size_t l = strlen(s); sb_grow(b, l); memcpy(b->p + b->n, s, l); b->n += l; b->p[b->n] = 0; }
static void sb_putn(sb* b, const char* s, size_t l) { sb_grow(b, l); memcpy(b->p + b->n, s, l); b->n += l; b->p[b->n] = 0; }
static void sb_putc(sb* b, char c) { sb_grow(b, 1); b->p[b->n++] = c; b->p[b->n] = 0; }
static void sb_num(sb* b, long l) { char t[40]; snprintf(t, sizeof t, "%ld", l); sb_put(b, t); }
static void sb_free(sb* b) { free(b->p); b->p = NULL; b->n = b->cap = 0; }

/* ================= valori ================= */
static jval* v_new(arena* A, jtype t) { jval* v = A ? (jval*)a_alloc(A, sizeof *v) : (jval*)malloc(sizeof *v); memset(v, 0, sizeof *v); v->t = t; v->A = A; return v; }
static jval* v_undef(arena* A) { return v_new(A, J_UNDEF); }
static jval* v_null(arena* A) { return v_new(A, J_NULL); }
static jval* v_bool(arena* A, int b) { jval* v = v_new(A, J_BOOL); v->u.b = b; return v; }
static jval* v_long(arena* A, long l) { jval* v = v_new(A, J_LONG); v->u.l = l; return v; }
static jval* v_double(arena* A, double d) { jval* v = v_new(A, J_DOUBLE); v->u.d = d; return v; }
static jval* v_str(arena* A, const char* s) { jval* v = v_new(A, J_STR); v->u.s = A ? a_strdup(A, s) : strdup(s ? s : ""); return v; }
static jval* v_str_take(arena* A, char* s) { jval* v = v_str(A, s); free(s); return v; }
static jval* v_var(arena* A, const char* name) { jval* v = v_new(A, J_VAR); v->u.s = A ? a_strdup(A, name) : strdup(name ? name : ""); return v; }
static jval* v_arr(arena* A) { return v_new(A, J_ARR); }
static jval* v_obj(arena* A) { return v_new(A, J_OBJ); }
static jval* v_fn(arena* A, jfn f) { jval* v = v_new(A, J_FN); v->u.f = f; return v; }
static void arr_push(jval* a, jval* v) {
    if (a->u.arr.n == a->u.arr.cap) {
        size_t nc = a->u.arr.cap ? a->u.arr.cap * 2 : 4;
        if (a->A) {
            jval** ni = (jval**)a_alloc(a->A, sizeof(jval*) * nc);
            if (a->u.arr.n) memcpy(ni, a->u.arr.items, sizeof(jval*) * a->u.arr.n);
            a->u.arr.items = ni;
        } else {
            a->u.arr.items = (jval**)realloc(a->u.arr.items, sizeof(jval*) * nc);
        }
        a->u.arr.cap = nc;
    }
    a->u.arr.items[a->u.arr.n++] = v;
}
static void obj_set(jval* o, const char* k, jval* v) {
    for (size_t i = 0; i < o->u.obj.n; i++)
        if (strcmp(o->u.obj.items[i].k, k) == 0) { o->u.obj.items[i].v = v; return; }
    if (o->u.obj.n == o->u.obj.cap) {
        size_t nc = o->u.obj.cap ? o->u.obj.cap * 2 : 4;
        if (o->A) {
            jval** dummy = NULL; (void)dummy;
            void* ni = a_alloc(o->A, sizeof(*o->u.obj.items) * nc);
            if (o->u.obj.n) memcpy(ni, o->u.obj.items, sizeof(*o->u.obj.items) * o->u.obj.n);
            o->u.obj.items = (void*)ni;
        } else {
            o->u.obj.items = (void*)realloc(o->u.obj.items, sizeof(*o->u.obj.items) * nc);
        }
        o->u.obj.cap = nc;
    }
    o->u.obj.items[o->u.obj.n].k = o->A ? a_strdup(o->A, k) : strdup(k);
    o->u.obj.items[o->u.obj.n].v = v;
    o->u.obj.n++;
}
static void v_free(jval* v) {
    if (!v) return;
    if (v->A) return;   /* arena-owned: liberata in blocco */
    switch (v->t) {
        case J_STR: case J_VAR: free(v->u.s); break;
        case J_ARR: for (size_t i = 0; i < v->u.arr.n; i++) v_free(v->u.arr.items[i]); free(v->u.arr.items); break;
        case J_OBJ: for (size_t i = 0; i < v->u.obj.n; i++) { free(v->u.obj.items[i].k); v_free(v->u.obj.items[i].v); } free(v->u.obj.items); break;
        default: break;
    }
    free(v);
}
static int is_dict(const jval* v) { return v && v->t == J_OBJ; }
static int is_list(const jval* v) { return v && v->t == J_ARR; }
static int is_iterable(const jval* v) { return v && (v->t == J_STR || v->t == J_ARR || v->t == J_OBJ); }

static jval* obj_get(const jval* o, const char* k) {
    for (size_t i = 0; i < o->u.obj.n; i++) if (strcmp(o->u.obj.items[i].k, k) == 0) return o->u.obj.items[i].v;
    return NULL;
}
static int obj_has(const jval* o, const char* k) {
    for (size_t i = 0; i < o->u.obj.n; i++) if (strcmp(o->u.obj.items[i].k, k) == 0) return 1;
    return 0;
}

/* deep copy di un valore del chiamante nell'arena (per vars in ingresso) */
static jval* v_deepcopy(arena* A, const jval* v) {
    if (!v) return v_undef(A);
    switch (v->t) {
        case J_UNDEF: return v_undef(A);
        case J_NULL: return v_null(A);
        case J_BOOL: return v_bool(A, v->u.b);
        case J_LONG: return v_long(A, v->u.l);
        case J_DOUBLE: return v_double(A, v->u.d);
        case J_STR: return v_str(A, v->u.s);
        case J_VAR: return v_var(A, v->u.s);
        case J_FN: return v_fn(A, v->u.f);
        case J_MACRO: return (jval*)v;   /* i macro nascono solo internamente */
        case J_ARR: { jval* c = v_arr(A); for (size_t i = 0; i < v->u.arr.n; i++) arr_push(c, v_deepcopy(A, v->u.arr.items[i])); return c; }
        case J_OBJ: { jval* c = v_obj(A); for (size_t i = 0; i < v->u.obj.n; i++) obj_set(c, v->u.obj.items[i].k, v_deepcopy(A, v->u.obj.items[i].v)); return c; }
    }
    return v_undef(A);
}

/* ================= env ================= */
struct env { struct { char* k; jval* v; }* e; size_t n, cap; arena* A; };
static env* env_new(arena* A) { env* e = (env*)calloc(1, sizeof(env)); e->A = A; return e; }
static void env_set(env* e, const char* k, jval* v) {
    for (size_t i = 0; i < e->n; i++) if (strcmp(e->e[i].k, k) == 0) { e->e[i].v = v; return; }
    if (e->n == e->cap) { e->cap = e->cap ? e->cap * 2 : 8; e->e = (void*)realloc(e->e, sizeof(*e->e) * e->cap); }
    e->e[e->n].k = a_strdup(e->A, k);
    e->e[e->n].v = v;
    e->n++;
}
static int env_has(env* e, const char* k) { for (size_t i = 0; i < e->n; i++) if (strcmp(e->e[i].k, k) == 0) return 1; return 0; }
static jval* env_get(env* e, const char* k) { for (size_t i = 0; i < e->n; i++) if (strcmp(e->e[i].k, k) == 0) return e->e[i].v; return NULL; }
static env* env_child(env* p) {
    env* c = env_new(p->A);
    if (p->n) { c->e = (void*)malloc(sizeof(*c->e) * p->n); c->cap = p->n; memcpy(c->e, p->e, sizeof(*c->e) * p->n); c->n = p->n; }
    return c;
}
static void env_free(env* e) { if (!e) return; free(e->e); free(e); }
static env* env_from_vars(arena* A, const jval* vars) {
    env* e = env_new(A);
    if (vars && vars->t == J_OBJ)
        for (size_t i = 0; i < vars->u.obj.n; i++) env_set(e, vars->u.obj.items[i].k, v_deepcopy(A, vars->u.obj.items[i].v));
    return e;
}

/* helper stringa / numero */
static char* to_str_alloc(arena* A, const jval* v);
static long to_long(const jval* v) {
    switch (v->t) {
        case J_LONG: return v->u.l;
        case J_DOUBLE: return (long)v->u.d;
        default: jerr_set("atteso un intero"); return 0;
    }
}
static int truth(const jval* v) {
    if (!v) return 0;
    switch (v->t) {
        case J_UNDEF: return 0;
        case J_NULL: return 0;
        case J_BOOL: return v->u.b;
        case J_STR: return v->u.s[0] != 0;
        case J_LONG: return v->u.l != 0;
        case J_DOUBLE: return v->u.d != 0;
        case J_ARR: return v->u.arr.n > 0;
        case J_OBJ: return v->u.obj.n > 0;
        case J_VAR: return 0;
        default: return 1;
    }
}
static long len_of(const jval* v) {
    switch (v->t) {
        case J_STR: return (long)strlen(v->u.s);
        case J_ARR: return (long)v->u.arr.n;
        case J_OBJ: return (long)v->u.obj.n;
        default: jerr_set("|length su valore non iterabile"); return 0;
    }
}

/* ================= template tokenizer (specchio C#) ================= */
typedef struct { int kind; /*0 txt 1 out 2 stmt*/ char* s; } seg;
static void seglist_free(seg* a, size_t n) { for (size_t i = 0; i < n; i++) free(a[i].s); free(a); }
static char* trimdup(arena* A, const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    size_t l = strlen(s);
    while (l && isspace((unsigned char)s[l - 1])) l--;
    return A ? a_strndup(A, s, l) : strndup(s, l);
}

static seg* tokenize(const char* t, size_t* outn) {
    size_t L = strlen(t);
    seg* segs = NULL; size_t ns = 0, cap = 0;
    sb text; sb_init(&text);
    int trimNext = 0;
    size_t i = 0;
#define FLUSH() do { \
        if (text.n == 0) break; \
        if (trimNext) { size_t k = 0; while (k < text.n && isspace((unsigned char)text.p[k])) k++; if (k) { memmove(text.p, text.p + k, text.n - k); text.n -= k; text.p[text.n] = 0; } trimNext = 0; } \
        if (ns == cap) { cap = cap ? cap * 2 : 16; segs = (seg*)realloc(segs, sizeof(seg) * cap); } \
        segs[ns].kind = 0; segs[ns].s = strndup(text.p, text.n); ns++; \
        text.n = 0; text.p[0] = 0; \
    } while (0)
    while (i < L) {
        size_t o = (size_t)-1, p = (size_t)-1, cm = (size_t)-1;
        for (size_t j = i; j + 1 < L; j++) {
            if (t[j] == '{' && t[j + 1] == '{' && o == (size_t)-1) o = j;
            if (t[j] == '{' && t[j + 1] == '%' && p == (size_t)-1) p = j;
            if (t[j] == '{' && t[j + 1] == '#' && cm == (size_t)-1) cm = j;
        }
        size_t n = (size_t)-1;
        if (o != (size_t)-1) n = o;
        if (p != (size_t)-1 && p < n) n = p;
        if (cm != (size_t)-1 && cm < n) n = cm;
        if (n == (size_t)-1) { sb_put(&text, t + i); break; }
        sb_putn(&text, t + i, n - i);
        i = n;
        if (n == cm) { /* commento: scartato (come C#) */
            const char* cl = strstr(t + n + 2, "#}");
            if (!cl) { jerr_set("commento non chiuso"); sb_free(&text); seglist_free(segs, ns); *outn = 0; return NULL; }
            size_t ce = (size_t)(cl - t);
            int cm_tl = (n + 2 < L && t[n + 2] == '-');
            int cm_tr = (ce > n + 2 && t[ce - 1] == '-');
            if (cm_tl) { while (text.n && isspace((unsigned char)text.p[text.n - 1])) text.n--; text.p[text.n] = 0; }
            FLUSH();
            trimNext = cm_tr;
            i = ce + 2;
            continue;
        }
        int out_ = (n == o);
        size_t beg = n + 2;
        int tg_tl = (beg < L && t[beg] == '-');
        if (tg_tl) beg++;
        const char* cl = strstr(t + beg, out_ ? "}}" : "%}");
        if (!cl) { jerr_set(out_ ? "{{ non chiuso" : "{% non chiuso"); sb_free(&text); seglist_free(segs, ns); *outn = 0; return NULL; }
        size_t closepos = (size_t)(cl - t);
        size_t end = closepos;
        int tg_tr = (end > beg && t[end - 1] == '-');
        if (tg_tr) end--;
        char* content = strndup(t + beg, end - beg);
        if (tg_tl) { while (text.n && isspace((unsigned char)text.p[text.n - 1])) text.n--; text.p[text.n] = 0; }
        FLUSH();
        if (ns == cap) { cap = cap ? cap * 2 : 16; segs = (seg*)realloc(segs, sizeof(seg) * cap); }
        segs[ns].kind = out_ ? 1 : 2;
        segs[ns].s = trimdup(NULL, content);
        ns++;
        free(content);
        trimNext = tg_tr;
        i = closepos + 2;
    }
    FLUSH();
    sb_free(&text);
    seg* res = NULL; size_t nr = 0, rcap = 0;
    for (size_t s = 0; s < ns; s++) {
        if (segs[s].kind == 0 && nr && res[nr - 1].kind == 0) {
            size_t ol = strlen(res[nr - 1].s), nl = strlen(segs[s].s);
            res[nr - 1].s = (char*)realloc(res[nr - 1].s, ol + nl + 1);
            memcpy(res[nr - 1].s + ol, segs[s].s, nl + 1);
            free(segs[s].s);
        } else {
            if (nr == rcap) { rcap = rcap ? rcap * 2 : 16; res = (seg*)realloc(res, sizeof(seg) * rcap); }
            res[nr++] = segs[s];
        }
    }
    free(segs);
    *outn = nr;
    return res;
#undef FLUSH
}

/* ================= AST / parser (specchio C#) ================= */
struct node {
    int kind;                 /* 0 Txt 1 Out 2 Set 3 If 4 For 5 Macro 6 SetBlock */
    char* s;                  /* Txt:texto | Out:expr | Set:var | For:var | Macro:name */
    char* expr;               /* Out expr / Set expr / For expr */
    struct { char* cond; nlist body; }* br; size_t nb;   /* If */
    nlist body;               /* For/Macro/SetBlock body */
    char** pname; char** pdef; size_t np;                /* Macro params */
};
static void nlist_add(nlist* l, node* nd) { if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 8; l->p = (node**)realloc(l->p, sizeof(void*) * l->cap); } l->p[l->n++] = nd; }
static void nlist_free(nlist* l);
static node* n_new(int kind) { node* nd = (node*)calloc(1, sizeof *nd); nd->kind = kind; return nd; }
static void n_free(node* nd) {
    if (!nd) return;
    free(nd->s); free(nd->expr);
    for (size_t i = 0; i < nd->nb; i++) { free(nd->br[i].cond); nlist_free(&nd->br[i].body); }
    free(nd->br);
    nlist_free(&nd->body);
    for (size_t i = 0; i < nd->np; i++) { free(nd->pname[i]); free(nd->pdef[i]); }
    free(nd->pname); free(nd->pdef);
    free(nd);
}
static void nlist_free(nlist* l) { for (size_t i = 0; i < l->n; i++) n_free(l->p[i]); free(l->p); l->p = NULL; l->n = l->cap = 0; }

static const char* first_word(const char* s) {
    static char w[128];
    size_t k = 0;
    while (s[k] && !isspace((unsigned char)s[k]) && k < 127) k++;
    memcpy(w, s, k); w[k] = 0;
    return w;
}
static char* substr_trim(const char* s, size_t start) {
    size_t len = strlen(s);
    if (start > len) return strdup("");
    return trimdup(NULL, s + start);
}
static int stop_contains(const char* const* stop, size_t nstop, const char* w) {
    for (size_t i = 0; i < nstop; i++) if (strcmp(w, stop[i]) == 0) return 1;
    return 0;
}
/* split top-level (fuori da stringhe/parentesi) su sep; ritorna array arena */
static char** split_top_level(arena* A, const char* s, char sep, size_t* outn) {
    size_t cap = 4, n = 0;
    char** res = (char**)malloc(sizeof(char*) * cap);
    sb cur; sb_init(&cur);
    int depth = 0; char q = 0; int esc = 0;
    for (size_t i = 0; ; i++) {
        char c = s[i];
        if (q) {
            sb_putc(&cur, c);
            if (esc) esc = 0;
            else if (c == '\\') esc = 1;
            else if (c == q) q = 0;
        } else {
            if (c == '\'' || c == '"') { q = c; sb_putc(&cur, c); }
            else if (c == '(' || c == '[' || c == '{') { depth++; sb_putc(&cur, c); }
            else if (c == ')' || c == ']' || c == '}') { depth--; sb_putc(&cur, c); }
            else if ((c == sep && depth == 0) || c == 0) {
                if (n == cap) { cap *= 2; res = (char**)realloc(res, sizeof(char*) * cap); }
                res[n++] = A ? a_strdup(A, cur.p) : strdup(cur.p);
                sb_free(&cur);
                if (c == 0) break;
                sb_init(&cur);
            } else sb_putc(&cur, c);
        }
        if (!c) break;
    }
    *outn = n;
    return res;
}
static int top_level_index_of(const char* s, char target) {
    int depth = 0; char q = 0; int esc = 0;
    for (size_t i = 0; s[i]; i++) {
        char c = s[i];
        if (q) {
            if (esc) esc = 0;
            else if (c == '\\') esc = 1;
            else if (c == q) q = 0;
            continue;
        }
        if (c == '\'' || c == '"') { q = c; continue; }
        if (c == '(' || c == '[' || c == '{') depth++;
        else if (c == ')' || c == ']' || c == '}') depth--;
        else if (c == target && depth == 0) return (int)i;
    }
    return -1;
}

static int parse_until(seg* a, size_t n, size_t* idx, nlist* out, const char* const* stop, size_t nstop) {
    while (*idx < n) {
        seg s = a[*idx];
        if (s.kind == 0) { node* nd = n_new(0); nd->s = strdup(s.s); nlist_add(out, nd); (*idx)++; continue; }
        if (s.kind == 1) { node* nd = n_new(1); nd->expr = strdup(s.s); nlist_add(out, nd); (*idx)++; continue; }
        const char* w = first_word(s.s);
        if (stop_contains(stop, nstop, w)) return 0;
        if (strcmp(w, "set") == 0) {
            const char* code = s.s;
            size_t sp = 3; while (code[sp] && isspace((unsigned char)code[sp])) sp++;
            size_t ne = sp; while (code[ne] && !isspace((unsigned char)code[ne]) && code[ne] != '=') ne++;
            char* sname = trimdup(NULL, "");
            { char* raw = strndup(code + sp, ne - sp); char* t = trimdup(NULL, raw); free(raw); free(sname); sname = t; }
            const char* eq = code[ne] ? strchr(code + ne, '=') : NULL;
            if (eq) {
                node* nd = n_new(2);
                nd->s = sname;
                nd->expr = trimdup(NULL, eq + 1);
                nlist_add(out, nd);
                (*idx)++;
            } else {
                (*idx)++;
                node* nd = n_new(6);
                nd->s = sname;
                const char* endset[] = { "endset" };
                if (parse_until(a, n, idx, &nd->body, endset, 1)) { n_free(nd); return 1; }
                if (*idx >= n || strcmp(first_word(a[*idx].s), "endset") != 0) { jerr_set("set senza '=' e senza endset: %s", code); n_free(nd); return 1; }
                (*idx)++;
                nlist_add(out, nd);
            }
            continue;
        }
        if (strcmp(w, "macro") == 0) {
            const char* code = s.s;
            const char* lp = strchr(code, '(');
            if (!lp) { jerr_set("macro malformato: %s", code); return 1; }
            char* mname = trimdup(NULL, "");
            { char* raw = strndup(code + 5, (size_t)(lp - (code + 5))); char* t = trimdup(NULL, raw); free(raw); free(mname); mname = t; }
            char* paramstr = strndup(lp + 1, strlen(lp + 1));
            { size_t pl = strlen(paramstr); while (pl && isspace((unsigned char)paramstr[pl - 1])) paramstr[--pl] = 0; }
            size_t pl = strlen(paramstr);
            if (pl == 0 || paramstr[pl - 1] != ')') { jerr_set("macro malformato: %s", code); free(mname); free(paramstr); return 1; }
            paramstr[pl - 1] = 0;
            node* nd = n_new(5);
            nd->s = mname;
            size_t nps = 0;
            char** parts = split_top_level(NULL, paramstr, ',', &nps);
            nd->pname = (char**)calloc(nps ? nps : 1, sizeof(char*));
            nd->pdef = (char**)calloc(nps ? nps : 1, sizeof(char*));
            for (size_t k = 0; k < nps; k++) {
                char* p = trimdup(NULL, parts[k]);
                free(parts[k]);
                if (!*p) { free(p); continue; }
                int pe = top_level_index_of(p, '=');
                if (pe < 0) { nd->pname[nd->np] = p; nd->pdef[nd->np] = NULL; nd->np++; }
                else {
                    char* nm = strndup(p, (size_t)pe); char* t = trimdup(NULL, nm); free(nm);
                    char* dv = trimdup(NULL, p + pe + 1);
                    nd->pname[nd->np] = t; nd->pdef[nd->np] = dv; nd->np++;
                    free(p);
                }
            }
            free(parts); free(paramstr);
            (*idx)++;
            const char* endmacro[] = { "endmacro" };
            if (parse_until(a, n, idx, &nd->body, endmacro, 1)) { n_free(nd); return 1; }
            if (*idx >= n || strcmp(first_word(a[*idx].s), "endmacro") != 0) { jerr_set("macro senza endmacro"); n_free(nd); return 1; }
            (*idx)++;
            nlist_add(out, nd);
            continue;
        }
        if (strcmp(w, "for") == 0) {
            const char* in = strstr(s.s + 3, " in ");
            if (!in) { jerr_set("for senza 'in': %s", s.s); return 1; }
            node* nd = n_new(4);
            { char* raw = strndup(s.s + 3, (size_t)(in - (s.s + 3))); nd->s = trimdup(NULL, raw); free(raw); }
            nd->expr = trimdup(NULL, in + 4);
            (*idx)++;
            const char* endfor[] = { "endfor" };
            if (parse_until(a, n, idx, &nd->body, endfor, 1)) { n_free(nd); return 1; }
            if (*idx >= n || strcmp(first_word(a[*idx].s), "endfor") != 0) { jerr_set("for senza endfor"); n_free(nd); return 1; }
            (*idx)++;
            nlist_add(out, nd);
            continue;
        }
        if (strcmp(w, "if") == 0) {
            (*idx)++;
            node* nd = n_new(3);
            nd->br = (void*)calloc(1, sizeof(*nd->br));
            nd->nb = 1;
            nd->br[0].cond = substr_trim(s.s, 2);
            {
                const char* el[] = { "elif", "else", "endif" };
                if (parse_until(a, n, idx, &nd->br[0].body, el, 3)) { n_free(nd); return 1; }
            }
            int done = 0;
            while (*idx < n && !done) {
                const char* w2 = first_word(a[*idx].s);
                if (strcmp(w2, "elif") == 0) {
                    nd->br = (void*)realloc(nd->br, sizeof(*nd->br) * (nd->nb + 1));
                    size_t bi = nd->nb;
                    nd->br[bi].cond = substr_trim(a[*idx].s, 4);
                    nd->br[bi].body = (nlist) { NULL, 0, 0 };
                    (*idx)++;
                    const char* el[] = { "elif", "else", "endif" };
                    if (parse_until(a, n, idx, &nd->br[bi].body, el, 3)) { n_free(nd); return 1; }
                    nd->nb++;
                    continue;
                }
                if (strcmp(w2, "else") == 0) {
                    (*idx)++;
                    nd->br = (void*)realloc(nd->br, sizeof(*nd->br) * (nd->nb + 1));
                    size_t bi = nd->nb;
                    nd->br[bi].cond = NULL;
                    nd->br[bi].body = (nlist) { NULL, 0, 0 };
                    const char* endif_[] = { "endif" };
                    if (parse_until(a, n, idx, &nd->br[bi].body, endif_, 1)) { n_free(nd); return 1; }
                    nd->nb++;
                    continue;
                }
                if (strcmp(w2, "endif") == 0) { (*idx)++; done = 1; continue; }
                jerr_set("if malformato (tag: %s)", a[*idx].s);
                n_free(nd);
                return 1;
            }
            if (!done) { jerr_set("if senza endif"); n_free(nd); return 1; }
            nlist_add(out, nd);
            continue;
        }
        jerr_set("statement non supportato: %s", w);
        return 1;
    }
    return 0;
}
static int parse_all(seg* a, size_t n, nlist* root) {
    size_t idx = 0;
    int r = parse_until(a, n, &idx, root, NULL, 0);
    if (!r && idx < n) { jerr_set("tag di blocco senza apertura"); return 1; }
    return r;
}

/* ================= expr (specchio C# POr..PAtom) ================= */
typedef struct { char** t; size_t n, p; } toks;
static void toks_free(toks* tk) { for (size_t i = 0; i < tk->n; i++) free(tk->t[i]); free(tk->t); tk->t = NULL; tk->n = tk->p = 0; }
static void tok_add(toks* tk, const char* s, size_t len) { tk->t = (char**)realloc(tk->t, sizeof(char*) * (tk->n + 1)); tk->t[tk->n++] = strndup(s, len); }

static void lex_expr(const char* s, toks* tk) {
    tk->n = 0; tk->p = 0; tk->t = NULL;
    size_t i = 0, L = strlen(s);
    while (i < L) {
        char c = s[i];
        if (isspace((unsigned char)c)) { i++; continue; }
        if (c == '\'' || c == '"') {
            char q = c; i++;
            sb b; sb_init(&b);
            int closed = 0;
            while (i < L) {
                if (s[i] == q) { i++; closed = 1; break; }
                if (s[i] == '\\' && i + 1 < L) {
                    i++;
                    switch (s[i]) {
                        case 'n': sb_putc(&b, '\n'); break;
                        case 't': sb_putc(&b, '\t'); break;
                        case 'r': sb_putc(&b, '\r'); break;
                        case '\\': sb_putc(&b, '\\'); break;
                        case '"': sb_putc(&b, '"'); break;
                        case '\'': sb_putc(&b, '\''); break;
                        default: sb_putc(&b, s[i]); break;
                    }
                    i++;
                } else { sb_putc(&b, s[i]); i++; }
            }
            if (!closed) { jerr_set("stringa non chiusa"); sb_free(&b); toks_free(tk); return; }
            char* tok = (char*)malloc(b.n + 5); memcpy(tok, "STR:", 4); memcpy(tok + 4, b.p, b.n + 1);
            tk->t = (char**)realloc(tk->t, sizeof(char*) * (tk->n + 1)); tk->t[tk->n++] = tok;
            sb_free(&b);
            continue;
        }
        if (isdigit((unsigned char)c) || (c == '-' && i + 1 < L && isdigit((unsigned char)s[i + 1]))) {
            size_t st = i; if (c == '-') i++;
            while (i < L && (isdigit((unsigned char)s[i]) || s[i] == '.')) i++;
            tok_add(tk, s + st, i - st);
            continue;
        }
        if (isalpha((unsigned char)c) || c == '_') {
            size_t st = i;
            while (i < L && (isalnum((unsigned char)s[i]) || s[i] == '_')) i++;
            tok_add(tk, s + st, i - st);
            continue;
        }
        if (strncmp(s + i, "==", 2) == 0) { tok_add(tk, "==", 2); i += 2; continue; }
        if (strncmp(s + i, "!=", 2) == 0) { tok_add(tk, "!=", 2); i += 2; continue; }
        if (strncmp(s + i, "<=", 2) == 0) { tok_add(tk, "<=", 2); i += 2; continue; }
        if (strncmp(s + i, ">=", 2) == 0) { tok_add(tk, ">=", 2); i += 2; continue; }
        if (strncmp(s + i, "->", 2) == 0) { tok_add(tk, "->", 2); i += 2; continue; }
        if (c == '<') { tok_add(tk, "<", 1); i++; continue; }
        if (c == '>') { tok_add(tk, ">", 1); i++; continue; }
        if (strchr("|()[]:.,+-=~", c)) { tok_add(tk, s + i, 1); i++; continue; }
        jerr_set("token inatteso '%c'", c);
        toks_free(tk);
        return;
    }
}

static int t_is(toks* tk, const char* s) { return tk->p < tk->n && strcmp(tk->t[tk->p], s) == 0; }

/* prototipi valutazione espressione */
static jval* p_conditional(toks* tk, env* e);
static jval* p_or(toks* tk, env* e);
static jval* p_and(toks* tk, env* e);
static jval* p_not(toks* tk, env* e);
static jval* p_test(toks* tk, env* e);
static jval* p_concat(toks* tk, env* e);
static jval* p_unary(toks* tk, env* e);
static jval* p_filter(toks* tk, env* e);
static jval* p_postfix(toks* tk, env* e);
static jval* p_atom(toks* tk, env* e);
static jval* eval_expr_str(const char* expr, env* e);
static void json_ser(sb* b, const jval* v);
static jval* v_attr(arena* A, jval* v, const char* name);
static jval* v_index(arena* A, jval* v, jval* i);
static jval* v_slice(arena* A, jval* v, jval* a, jval* b, jval* step);
static jval* method_call(arena* A, jval* v, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw);
static jval* apply_filter(arena* A, const char* f, jval* v, jval** args, int na, jval** kwk, jval** kwv, int nkw);
static jval* call_fn(arena* A, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw, env* e);
static jval* items_arr(arena* A, jval* it);
static char* eval_to_string(arena* A, nlist* nodes, env* e);

static int is_name_token(const char* s) { return s[0] && (isalpha((unsigned char)s[0]) || s[0] == '_') && strncmp(s, "STR:", 4) != 0; }

static int parse_call_args(toks* tk, env* e, jval*** pos, int* npos, jval*** kwk, jval*** kwv, int* nkw) {
    *pos = NULL; *npos = 0; *kwk = NULL; *kwv = NULL; *nkw = 0;
    while (tk->p < tk->n && strcmp(tk->t[tk->p], ")") != 0) {
        if (is_name_token(tk->t[tk->p]) && tk->p + 1 < tk->n && strcmp(tk->t[tk->p + 1], "=") == 0) {
            const char* key = tk->t[tk->p]; tk->p += 2;
            jval* v = p_or(tk, e);
            if (!v) { for (int i = 0; i < *npos; i++) {} return 1; }
            *kwk = (jval**)realloc(*kwk, sizeof(jval*) * (*nkw + 1));
            *kwv = (jval**)realloc(*kwv, sizeof(jval*) * (*nkw + 1));
            (*kwk)[*nkw] = v_str(e->A, key); (*kwv)[*nkw] = v; (*nkw)++;
        } else {
            jval* v = p_or(tk, e);
            if (!v) return 1;
            *pos = (jval**)realloc(*pos, sizeof(jval*) * (*npos + 1));
            (*pos)[(*npos)++] = v;
        }
        if (tk->p < tk->n && strcmp(tk->t[tk->p], ",") == 0) tk->p++;
        else if (tk->p < tk->n && strcmp(tk->t[tk->p], ")") != 0) { jerr_set("argomenti funzione malformati"); return 1; }
    }
    if (tk->p >= tk->n) { jerr_set("')' mancante nella chiamata"); return 1; }
    tk->p++;
    return 0;
}

static jval* p_conditional(toks* tk, env* e) {
    jval* v = p_or(tk, e);
    if (!v) return NULL;
    if (t_is(tk, "if")) {
        tk->p++;
        jval* cond = p_or(tk, e);
        if (!cond) return NULL;
        if (!t_is(tk, "else")) { jerr_set("condizionale 'if' senza 'else'"); return NULL; }
        tk->p++;
        jval* alt = p_conditional(tk, e);
        if (!alt) return NULL;
        return truth(cond) ? v : alt;
    }
    return v;
}
static jval* p_or(toks* tk, env* e) {
    jval* v = p_and(tk, e);
    if (!v) return NULL;
    while (t_is(tk, "or")) { tk->p++; jval* r = p_and(tk, e); if (!r) return NULL; if (!truth(v)) v = r; }
    return v;
}
static jval* p_and(toks* tk, env* e) {
    jval* v = p_not(tk, e);
    if (!v) return NULL;
    while (t_is(tk, "and")) { tk->p++; jval* r = p_not(tk, e); if (!r) return NULL; if (truth(v)) v = r; }
    return v;
}
static jval* p_not(toks* tk, env* e) {
    if (t_is(tk, "not")) { tk->p++; jval* v = p_not(tk, e); if (!v) return NULL; return v_bool(e->A, !truth(v)); }
    return p_test(tk, e);
}

static int v_eq(const jval* a, const jval* b);
static int v_contains(const jval* container, const jval* item);
static int v_compare(const jval* a, const jval* b, int* ok);

static jval* p_test(toks* tk, env* e) {
    jval* lhs = p_concat(tk, e);
    if (!lhs) return NULL;
    if (tk->p >= tk->n) return lhs;
    const char* op = tk->t[tk->p];
    if (strcmp(op, "not") == 0 && tk->p + 1 < tk->n && strcmp(tk->t[tk->p + 1], "in") == 0) {
        tk->p += 2; jval* r = p_concat(tk, e); if (!r) return NULL;
        return v_bool(e->A, !v_contains(r, lhs));
    }
    if (strcmp(op, "==") == 0 || strcmp(op, "!=") == 0) {
        tk->p++; jval* r = p_concat(tk, e); if (!r) return NULL;
        int eq = v_eq(lhs, r);
        return v_bool(e->A, op[0] == '!' ? !eq : eq);
    }
    if (strcmp(op, "<") == 0 || strcmp(op, ">") == 0 || strcmp(op, "<=") == 0 || strcmp(op, ">=") == 0) {
        tk->p++; jval* r = p_concat(tk, e); if (!r) return NULL;
        int ok = 1; int c = v_compare(lhs, r, &ok);
        if (!ok) return NULL;
        if (strcmp(op, "<") == 0) return v_bool(e->A, c < 0);
        if (strcmp(op, ">") == 0) return v_bool(e->A, c > 0);
        if (strcmp(op, "<=") == 0) return v_bool(e->A, c <= 0);
        return v_bool(e->A, c >= 0);
    }
    if (strcmp(op, "in") == 0) {
        tk->p++; jval* r = p_concat(tk, e); if (!r) return NULL;
        return v_bool(e->A, v_contains(r, lhs));
    }
    if (strcmp(op, "is") == 0) {
        tk->p++;
        int neg = 0;
        if (t_is(tk, "not")) { neg = 1; tk->p++; }
        if (tk->p >= tk->n) { jerr_set("is senza test"); return NULL; }
        const char* test = tk->t[tk->p]; tk->p++;
        int res;
        if (strcmp(test, "defined") == 0) res = (lhs->t == J_UNDEF) ? 0 : (lhs->t == J_VAR) ? env_has(e, lhs->u.s) : 1;
        else if (strcmp(test, "undefined") == 0) res = (lhs->t == J_UNDEF) ? 1 : (lhs->t == J_VAR) ? !env_has(e, lhs->u.s) : 0;
        else if (strcmp(test, "none") == 0) res = (lhs->t == J_NULL);
        else if (strcmp(test, "mapping") == 0) res = is_dict(lhs);
        else if (strcmp(test, "iterable") == 0) res = is_iterable(lhs);
        else if (strcmp(test, "string") == 0) res = (lhs->t == J_STR);
        else if (strcmp(test, "boolean") == 0) res = (lhs->t == J_BOOL);
        else if (strcmp(test, "true") == 0) res = (lhs->t == J_BOOL && lhs->u.b);
        else if (strcmp(test, "false") == 0) res = (lhs->t == J_BOOL && !lhs->u.b);
        else if (strcmp(test, "number") == 0) res = (lhs->t == J_LONG || lhs->t == J_DOUBLE);
        else if (strcmp(test, "sequence") == 0) res = is_list(lhs) || (lhs->t == J_STR);
        else { jerr_set("test is non supportato: %s", test); return NULL; }
        return v_bool(e->A, neg ? !res : res);
    }
    return lhs;
}

static jval* v_concat(jval* a, jval* b);
static jval* p_concat(toks* tk, env* e) {
    jval* v = p_unary(tk, e);
    if (!v) return NULL;
    while (t_is(tk, "+") || t_is(tk, "-") || t_is(tk, "~")) {
        const char* op = tk->t[tk->p]; tk->p++;
        jval* r = p_unary(tk, e);
        if (!r) return NULL;
        if (op[0] == '+') v = v_concat(v, r);
        else if (op[0] == '-') {
            if (v->t == J_LONG && r->t == J_LONG) v = v_long(e->A, v->u.l - r->u.l);
            else if (v->t == J_LONG && r->t == J_DOUBLE) v = v_double(e->A, (double)v->u.l - r->u.d);
            else if (v->t == J_DOUBLE && r->t == J_LONG) v = v_double(e->A, v->u.d - (double)r->u.l);
            else if (v->t == J_DOUBLE && r->t == J_DOUBLE) v = v_double(e->A, v->u.d - r->u.d);
            else JERR("'-' non supportato tra questi tipi");
        } else {
            char* x = to_str_alloc(e->A, v); char* y = to_str_alloc(e->A, r);
            char* z = (char*)a_alloc(e->A, strlen(x) + strlen(y) + 1);
            strcpy(z, x); strcat(z, y);
            v = v_str(e->A, z);
        }
    }
    return v;
}
static jval* p_unary(toks* tk, env* e) {
    if (t_is(tk, "-")) {
        tk->p++; jval* v = p_unary(tk, e); if (!v) return NULL;
        if (v->t == J_LONG) return v_long(e->A, -v->u.l);
        if (v->t == J_DOUBLE) return v_double(e->A, -v->u.d);
        JERR("meno unario non numerico");
    }
    return p_filter(tk, e);
}

static jval* p_filter(toks* tk, env* e) {
    jval* v = p_postfix(tk, e);
    if (!v) return NULL;
    while (t_is(tk, "|")) {
        tk->p++;
        if (tk->p >= tk->n) { jerr_set("| senza filtro"); return NULL; }
        const char* f = tk->t[tk->p]; tk->p++;
        jval** args = NULL; int na = 0;
        jval** kwk = NULL; jval** kwv = NULL; int nkw = 0;
        if (t_is(tk, "(")) { tk->p++; if (parse_call_args(tk, e, &args, &na, &kwk, &kwv, &nkw)) return NULL; }
        v = apply_filter(e->A, f, v, args, na, kwk, kwv, nkw);
        if (!v) return NULL;
    }
    return v;
}

static jval* p_postfix(toks* tk, env* e) {
    jval* v = p_atom(tk, e);
    if (!v) return NULL;
    for (;;) {
        if (t_is(tk, ".")) {
            tk->p++;
            if (tk->p >= tk->n) JERR("'.' senza nome");
            const char* name = tk->t[tk->p]; tk->p++;
            if (t_is(tk, "(")) {
                tk->p++;
                jval** args = NULL; int na = 0; jval** kwk = NULL; jval** kwv = NULL; int nkw = 0;
                if (parse_call_args(tk, e, &args, &na, &kwk, &kwv, &nkw)) return NULL;
                v = method_call(e->A, v, name, args, na, kwk, kwv, nkw);
                if (!v) return NULL;
            } else {
                v = v_attr(e->A, v, name);
                if (!v) return NULL;
            }
            continue;
        }
        if (t_is(tk, "[")) {
            tk->p++;
            jval* start = NULL, * stop = NULL, * step = NULL;
            int slice = 0;
            if (t_is(tk, ":")) { slice = 1; tk->p++; }
            else {
                start = p_conditional(tk, e);
                if (!start) return NULL;
                if (t_is(tk, ":")) { slice = 1; tk->p++; }
            }
            if (slice) {
                if (tk->p < tk->n && strcmp(tk->t[tk->p], "]") != 0 && strcmp(tk->t[tk->p], ":") != 0) { stop = p_conditional(tk, e); if (!stop) return NULL; }
                if (t_is(tk, ":")) { tk->p++; if (tk->p < tk->n && strcmp(tk->t[tk->p], "]") != 0) { step = p_conditional(tk, e); if (!step) return NULL; } }
                if (!t_is(tk, "]")) JERR("']' mancante");
                tk->p++;
                v = v_slice(e->A, v, start, stop, step);
                if (!v) return NULL;
            } else {
                if (!t_is(tk, "]")) JERR("']' mancante");
                tk->p++;
                v = v_index(e->A, v, start);
                if (!v) return NULL;
            }
            continue;
        }
        break;
    }
    return v;
}

static jval* p_atom(toks* tk, env* e) {
    if (tk->p >= tk->n) JERR("espressione vuota");
    const char* tk0 = tk->t[tk->p];
    if (strncmp(tk0, "STR:", 4) == 0) { tk->p++; return v_str(e->A, tk0 + 4); }
    if (strcmp(tk0, "true") == 0) { tk->p++; return v_bool(e->A, 1); }
    if (strcmp(tk0, "false") == 0) { tk->p++; return v_bool(e->A, 0); }
    if (strcmp(tk0, "none") == 0 || strcmp(tk0, "None") == 0) { tk->p++; return v_null(e->A); }
    if (strcmp(tk0, "(") == 0) {
        tk->p++;
        jval* first = NULL; jval* tup = NULL; int cnt = 0;
        while (tk->p < tk->n && strcmp(tk->t[tk->p], ")") != 0) {
            jval* x = p_conditional(tk, e); if (!x) return NULL;
            if (cnt == 0) first = x;
            else { if (!tup) { tup = v_arr(e->A); arr_push(tup, first); } arr_push(tup, x); }
            cnt++;
            if (tk->p < tk->n && strcmp(tk->t[tk->p], ",") == 0) tk->p++;
        }
        if (tk->p >= tk->n) JERR("')' mancante");
        tk->p++;
        return (cnt == 1) ? first : (tup ? tup : v_arr(e->A));
    }
    if (strcmp(tk0, "[") == 0) {
        tk->p++;
        jval* lst = v_arr(e->A);
        while (tk->p < tk->n && strcmp(tk->t[tk->p], "]") != 0) {
            jval* x = p_or(tk, e); if (!x) return NULL;
            arr_push(lst, x);
            if (tk->p < tk->n && strcmp(tk->t[tk->p], ",") == 0) tk->p++;
        }
        if (tk->p >= tk->n) JERR("']' mancante");
        tk->p++;
        return lst;
    }
    {
        char* end = NULL;
        long lv = strtol(tk0, &end, 10);
        if (end && *end == 0 && tk0[0]) { tk->p++; return v_long(e->A, lv); }
    }
    {
        char* en2 = NULL;
        double dv = strtod(tk0, &en2);
        if (en2 && *en2 == 0 && tk0[0]) { tk->p++; return v_double(e->A, dv); }
    }
    const char* name = tk0; tk->p++;
    if (t_is(tk, "(")) {
        tk->p++;
        jval** args = NULL; int na = 0; jval** kwk = NULL; jval** kwv = NULL; int nkw = 0;
        if (parse_call_args(tk, e, &args, &na, &kwk, &kwv, &nkw)) return NULL;
        return call_fn(e->A, name, args, na, kwk, kwv, nkw, e);
    }
    if (env_has(e, name)) return env_get(e, name);
    return v_var(e->A, name);
}

/* ================= valore: attr/index/slice ================= */
static jval* v_attr(arena* A, jval* v, const char* name) {
    if (v->t == J_VAR || v->t == J_UNDEF) return v_undef(A);
    if (is_dict(v)) { jval* g = obj_get(v, name); return g ? g : v_undef(A); }
    jerr_set("attributo '%s' su valore non-mapping", name);
    return NULL;
}
static jval* v_index(arena* A, jval* v, jval* i) {
    if (i && i->t == J_VAR) JERR("indice su variabile non definita");
    if (v->t == J_UNDEF) return v_undef(A);
    if (is_dict(v) && i && i->t == J_STR) { jval* g = obj_get(v, i->u.s); return g ? g : v_undef(A); }
    if (v->t == J_ARR && i && i->t == J_LONG) {
        long idx = i->u.l;
        long n = idx < 0 ? (long)v->u.arr.n + idx : idx;
        if (n < 0 || (size_t)n >= v->u.arr.n) JERR("indice fuori range");
        return v->u.arr.items[n];
    }
    JERR("subscript non supportato");
}
static long clampi(long x, long lo, long hi) { return x < lo ? lo : x > hi ? hi : x; }
static jval* v_slice(arena* A, jval* v, jval* a, jval* b, jval* stepv) {
    if (v->t != J_ARR && v->t != J_STR) JERR("slice su valore non indicizzabile");
    long step = (stepv && stepv->t != J_NULL && stepv->t != J_UNDEF) ? to_long(stepv) : 1;
    if (g_err[0]) return NULL;
    if (step == 0) JERR("slice con step 0");
    long n = (v->t == J_ARR) ? (long)v->u.arr.n : (long)strlen(v->u.s);
    if (step > 0) {
        long lo = (a && a->t != J_NULL && a->t != J_UNDEF) ? to_long(a) : 0;
        if (g_err[0]) return NULL;
        if (lo < 0) lo += n; lo = clampi(lo, 0, n);
        long hi = (b && b->t != J_NULL && b->t != J_UNDEF) ? to_long(b) : n;
        if (g_err[0]) return NULL;
        if (hi < 0) hi += n; hi = clampi(hi, 0, n);
        if (v->t == J_ARR) { jval* r = v_arr(A); for (long k = lo; k < hi; k += step) arr_push(r, v->u.arr.items[k]); return r; }
        sb o; sb_init(&o);
        for (long k = lo; k < hi; k += step) sb_putc(&o, v->u.s[k]);
        jval* r = v_str(A, o.p); sb_free(&o); return r;
    } else {
        long lo = (a && a->t != J_NULL && a->t != J_UNDEF) ? to_long(a) : n - 1;
        if (g_err[0]) return NULL;
        if (lo < 0) lo += n; lo = clampi(lo, -1, n - 1);
        long hi;
        if (!b || b->t == J_NULL || b->t == J_UNDEF) hi = -1;
        else { hi = to_long(b); if (g_err[0]) return NULL; if (hi < 0) hi += n; hi = clampi(hi, -1, n - 1); }
        if (v->t == J_ARR) { jval* r = v_arr(A); for (long k = lo; k > hi; k += step) arr_push(r, v->u.arr.items[k]); return r; }
        sb o; sb_init(&o);
        for (long k = lo; k > hi; k += step) sb_putc(&o, v->u.s[k]);
        jval* r = v_str(A, o.p); sb_free(&o); return r;
    }
}

/* ================= iterazione / unpack ================= */
static jval* items_arr(arena* A, jval* it) {
    if (it->t == J_ARR) return it;
    if (it->t == J_STR) {
        jval* r = v_arr(A);
        for (size_t i = 0; it->u.s[i]; i++) { char c[2] = { it->u.s[i], 0 }; arr_push(r, v_str(A, c)); }
        return r;
    }
    if (it->t == J_OBJ) {
        jval* r = v_arr(A);
        for (size_t i = 0; i < it->u.obj.n; i++) {
            jval* pair = v_arr(A);
            arr_push(pair, v_str(A, it->u.obj.items[i].k));
            arr_push(pair, it->u.obj.items[i].v);
            arr_push(r, pair);
        }
        return r;
    }
    JERR("valore non iterabile");
}

/* ================= filtri ================= */
static jval* v_tojson(arena* A, const jval* v) { sb b; sb_init(&b); json_ser(&b, v); jval* r = v_str(A, b.p); sb_free(&b); return r; }
static jval* apply_by_name(arena* A, const char* fn, jval* x) {
    if (strcmp(fn, "upper") == 0) { char* s = to_str_alloc(A, x); for (char* p = s; *p; p++) *p = (char)toupper((unsigned char)*p); return v_str(A, s); }
    if (strcmp(fn, "lower") == 0) { char* s = to_str_alloc(A, x); for (char* p = s; *p; p++) *p = (char)tolower((unsigned char)*p); return v_str(A, s); }
    if (strcmp(fn, "trim") == 0) {
        if (x->t == J_STR) { const char* b = x->u.s; while (*b && isspace((unsigned char)*b)) b++; size_t l = strlen(b); while (l && isspace((unsigned char)b[l - 1])) l--; return v_str_take(A, strndup(b, l)); }
        return x;
    }
    if (strcmp(fn, "string") == 0) return v_str(A, to_str_alloc(A, x));
    if (strcmp(fn, "length") == 0) { long l = len_of(x); if (g_err[0]) return NULL; return v_long(A, l); }
    jerr_set("map: filtro non supportato '%s'", fn);
    return NULL;
}
static int pair_cmp(const void* pa, const void* pb) {
    const jval* a = *(const jval* const*)pa; const jval* b = *(const jval* const*)pb;
    const char* ka = (a->t == J_ARR && a->u.arr.n > 0 && a->u.arr.items[0]->t == J_STR) ? a->u.arr.items[0]->u.s : "";
    const char* kb = (b->t == J_ARR && b->u.arr.n > 0 && b->u.arr.items[0]->t == J_STR) ? b->u.arr.items[0]->u.s : "";
    return strcmp(ka, kb);
}
static jval* apply_filter(arena* A, const char* f, jval* v, jval** args, int na, jval** kwk, jval** kwv, int nkw) {
    if (v->t == J_UNDEF) {
        if (strcmp(f, "default") == 0) return na > 0 ? args[0] : v_str(A, "");
        if (strcmp(f, "length") == 0 || strcmp(f, "count") == 0) return v_long(A, 0);
        if (strcmp(f, "join") == 0) return v_str(A, "");
        if (strcmp(f, "list") == 0 || strcmp(f, "items") == 0 || strcmp(f, "dictsort") == 0 || strcmp(f, "map") == 0) return v_arr(A);
        if (strcmp(f, "tojson") == 0) return v_str(A, "null");
        if (strcmp(f, "safe") == 0) return v;
        return v_str(A, "");
    }
    if (strcmp(f, "trim") == 0) {
        if (v->t == J_STR) { const char* b = v->u.s; while (*b && isspace((unsigned char)*b)) b++; size_t l = strlen(b); while (l && isspace((unsigned char)b[l - 1])) l--; return v_str_take(A, strndup(b, l)); }
        return v;
    }
    if (strcmp(f, "length") == 0 || strcmp(f, "count") == 0) { long l = len_of(v); if (g_err[0]) return NULL; return v_long(A, l); }
    if (strcmp(f, "tojson") == 0) return v_tojson(A, v);
    if (strcmp(f, "safe") == 0) return v;
    if (strcmp(f, "string") == 0) return v_str(A, to_str_alloc(A, v));
    if (strcmp(f, "upper") == 0) { char* s = to_str_alloc(A, v); for (char* p = s; *p; p++) *p = (char)toupper((unsigned char)*p); return v_str(A, s); }
    if (strcmp(f, "lower") == 0) { char* s = to_str_alloc(A, v); for (char* p = s; *p; p++) *p = (char)tolower((unsigned char)*p); return v_str(A, s); }
    if (strcmp(f, "default") == 0) {
        int useDefault = (v->t == J_NULL || v->t == J_VAR || v->t == J_UNDEF);
        if (!useDefault && na > 1 && truth(args[1])) useDefault = !truth(v);
        return useDefault ? (na > 0 ? args[0] : v_str(A, "")) : v;
    }
    if (strcmp(f, "join") == 0) {
        const char* sep = na > 0 ? to_str_alloc(A, args[0]) : "";
        jval* list = items_arr(A, v); if (!list) return NULL;
        sb b; sb_init(&b);
        for (size_t i = 0; i < list->u.arr.n; i++) { if (i) sb_put(&b, sep); sb_put(&b, to_str_alloc(A, list->u.arr.items[i])); }
        jval* r = v_str(A, b.p); sb_free(&b); return r;
    }
    if (strcmp(f, "list") == 0) return items_arr(A, v);
    if (strcmp(f, "items") == 0) {
        if (!is_dict(v)) JERR("filtro |items su valore non-mapping");
        jval* r = v_arr(A);
        for (size_t i = 0; i < v->u.obj.n; i++) { jval* p = v_arr(A); arr_push(p, v_str(A, v->u.obj.items[i].k)); arr_push(p, v->u.obj.items[i].v); arr_push(r, p); }
        return r;
    }
    if (strcmp(f, "dictsort") == 0) {
        if (!is_dict(v)) JERR("filtro |dictsort su valore non-mapping");
        jval* r = v_arr(A);
        for (size_t i = 0; i < v->u.obj.n; i++) { jval* p = v_arr(A); arr_push(p, v_str(A, v->u.obj.items[i].k)); arr_push(p, v->u.obj.items[i].v); arr_push(r, p); }
        qsort(r->u.arr.items, r->u.arr.n, sizeof(jval*), pair_cmp);
        return r;
    }
    if (strcmp(f, "map") == 0) {
        jval* r = v_arr(A);
        jval* list = items_arr(A, v); if (!list) return NULL;
        const char* attr = NULL;
        for (int i = 0; i < nkw; i++) if (kwk[i]->t == J_STR && strcmp(kwk[i]->u.s, "attribute") == 0 && kwv[i]->t == J_STR) attr = kwv[i]->u.s;
        if (attr) { for (size_t i = 0; i < list->u.arr.n; i++) { jval* x = list->u.arr.items[i]; jval* g = is_dict(x) ? obj_get(x, attr) : NULL; arr_push(r, g ? g : v_undef(A)); } return r; }
        if (na > 0 && args[0]->t == J_STR) { for (size_t i = 0; i < list->u.arr.n; i++) { jval* x = apply_by_name(A, args[0]->u.s, list->u.arr.items[i]); if (!x) return NULL; arr_push(r, x); } return r; }
        JERR("map richiede attribute= o un filtro");
    }
    if (strcmp(f, "replace") == 0) {
        if (na < 2) JERR("replace richiede 2 argomenti");
        const char* s = to_str_alloc(A, v);
        const char* from = to_str_alloc(A, args[0]);
        const char* to = to_str_alloc(A, args[1]);
        sb b; sb_init(&b);
        size_t fl = strlen(from);
        if (fl == 0) { sb_put(&b, s); }
        else {
            const char* p = s;
            while (*p) {
                if (strncmp(p, from, fl) == 0) { sb_put(&b, to); p += fl; }
                else sb_putc(&b, *p++);
            }
        }
        jval* r = v_str(A, b.p); sb_free(&b); return r;
    }
    if (strcmp(f, "reject") == 0) {
        if (na < 2) JERR("reject richiede test e argomento");
        const char* test = to_str_alloc(A, args[0]);
        jval* target = args[1];
        jval* list = items_arr(A, v); if (!list) return NULL;
        jval* r = v_arr(A);
        for (size_t i = 0; i < list->u.arr.n; i++) {
            jval* x = list->u.arr.items[i];
            int hit;
            if (strcmp(test, "in") == 0) hit = v_contains(target, x);
            else if (strcmp(test, "eq") == 0 || strcmp(test, "equalto") == 0) hit = v_eq(x, target);
            else JERR("reject: test non supportato '%s'", test);
            if (!hit) arr_push(r, x);
        }
        return r;
    }
    jerr_set("filtro non supportato: %s", f);
    return NULL;
}

/* ================= metodi ================= */
static jval* split_string(arena* A, const char* s, jval** args, int na) {
    jval* r = v_arr(A);
    if (na == 0 || (args[0] && args[0]->t == J_UNDEF) || (args[0] && args[0]->t == J_NULL)) {
        size_t i = 0, L = strlen(s);
        while (i < L) {
            while (i < L && isspace((unsigned char)s[i])) i++;
            size_t st = i;
            while (i < L && !isspace((unsigned char)s[i])) i++;
            if (i > st) arr_push(r, v_str_take(A, strndup(s + st, i - st)));
        }
        return r;
    }
    const char* sep = to_str_alloc(A, args[0]);
    if (!sep[0]) JERR("split con separatore vuoto");
    int max = (na > 1 && args[1] && args[1]->t == J_LONG) ? (int)args[1]->u.l : -1;
    if (max < 0) {
        const char* p = s; const char* q;
        while ((q = strstr(p, sep))) { arr_push(r, v_str_take(A, strndup(p, (size_t)(q - p)))); p = q + strlen(sep); }
        arr_push(r, v_str(A, p));
    } else {
        const char* p = s; int cnt = 0;
        while (cnt < max) {
            const char* q = strstr(p, sep);
            if (!q) break;
            arr_push(r, v_str_take(A, strndup(p, (size_t)(q - p))));
            p = q + strlen(sep); cnt++;
        }
        arr_push(r, v_str(A, p));
    }
    return r;
}
static jval* py_strip(arena* A, const char* s, const char* chars, int left, int right) {
    size_t len = strlen(s), lo = 0, hi = len;
    if (left) while (lo < hi && (chars ? strchr(chars, s[lo]) != NULL : isspace((unsigned char)s[lo]))) lo++;
    if (right) while (hi > lo && (chars ? strchr(chars, s[hi - 1]) != NULL : isspace((unsigned char)s[hi - 1]))) hi--;
    return v_str_take(A, strndup(s + lo, hi - lo));
}
static jval* starts_ends(arena* A, const char* s, jval** args, int na, int start) {
    if (na == 0) return v_bool(A, 0);
    jval* a0 = args[0];
    if (a0->t == J_STR) return v_bool(A, start ? strncmp(s, a0->u.s, strlen(a0->u.s)) == 0 : (strlen(s) >= strlen(a0->u.s) && strcmp(s + strlen(s) - strlen(a0->u.s), a0->u.s) == 0));
    if (a0->t == J_ARR) {
        for (size_t i = 0; i < a0->u.arr.n; i++) {
            if (a0->u.arr.items[i]->t == J_STR) {
                const char* p = a0->u.arr.items[i]->u.s;
                if (start ? strncmp(s, p, strlen(p)) == 0 : (strlen(s) >= strlen(p) && strcmp(s + strlen(s) - strlen(p), p) == 0)) return v_bool(A, 1);
            }
        }
    }
    return v_bool(A, 0);
}
static jval* method_call(arena* A, jval* v, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw) {
    (void)kwk; (void)kwv; (void)nkw;
    if (v->t == J_UNDEF) return v_undef(A);
    if (strcmp(name, "get") == 0) {
        if (!is_dict(v)) JERR(".get() su valore non-mapping");
        if (na < 1) JERR(".get() richiede una chiave");
        const char* key = to_str_alloc(A, args[0]);
        jval* g = obj_get(v, key);
        if (!g) return (na > 1) ? args[1] : v_undef(A);
        return g;
    }
    if (strcmp(name, "items") == 0) {
        if (!is_dict(v)) JERR(".items() su valore non-mapping");
        jval* r = v_arr(A);
        for (size_t i = 0; i < v->u.obj.n; i++) { jval* p = v_arr(A); arr_push(p, v_str(A, v->u.obj.items[i].k)); arr_push(p, v->u.obj.items[i].v); arr_push(r, p); }
        return r;
    }
    if (strcmp(name, "keys") == 0 || strcmp(name, "values") == 0) {
        if (!is_dict(v)) { jerr_set(".%s() su valore non-mapping", name); return NULL; }
        jval* r = v_arr(A);
        for (size_t i = 0; i < v->u.obj.n; i++) arr_push(r, strcmp(name, "keys") == 0 ? v_str(A, v->u.obj.items[i].k) : v->u.obj.items[i].v);
        return r;
    }
    if (v->t == J_STR) {
        const char* s = v->u.s;
        if (strcmp(name, "split") == 0) return split_string(A, s, args, na);
        if (strcmp(name, "strip") == 0) return py_strip(A, s, (na > 0 && args[0]->t != J_NULL && args[0]->t != J_UNDEF) ? to_str_alloc(A, args[0]) : NULL, 1, 1);
        if (strcmp(name, "lstrip") == 0) return py_strip(A, s, (na > 0 && args[0]->t != J_NULL && args[0]->t != J_UNDEF) ? to_str_alloc(A, args[0]) : NULL, 1, 0);
        if (strcmp(name, "rstrip") == 0) return py_strip(A, s, (na > 0 && args[0]->t != J_NULL && args[0]->t != J_UNDEF) ? to_str_alloc(A, args[0]) : NULL, 0, 1);
        if (strcmp(name, "startswith") == 0) return starts_ends(A, s, args, na, 1);
        if (strcmp(name, "endswith") == 0) return starts_ends(A, s, args, na, 0);
        if (strcmp(name, "upper") == 0) { char* t = to_str_alloc(A, v); for (char* p = t; *p; p++) *p = (char)toupper((unsigned char)*p); return v_str(A, t); }
        if (strcmp(name, "lower") == 0) { char* t = to_str_alloc(A, v); for (char* p = t; *p; p++) *p = (char)tolower((unsigned char)*p); return v_str(A, t); }
    }
    jerr_set("metodo non supportato: .%s()", name);
    return NULL;
}

/* ================= funzioni ================= */
static jval* range_fn(arena* A, jval** args, int na) {
    long start, stop, step;
    if (na == 1) { start = 0; stop = to_long(args[0]); step = 1; }
    else if (na >= 2) { start = to_long(args[0]); stop = to_long(args[1]); step = (na >= 3) ? to_long(args[2]) : 1; }
    else JERR("range richiede almeno 1 argomento");
    if (g_err[0]) return NULL;
    if (step == 0) JERR("range con step 0");
    jval* r = v_arr(A);
    if (step > 0) for (long i = start; i < stop; i += step) arr_push(r, v_long(A, i));
    else for (long i = start; i > stop; i += step) arr_push(r, v_long(A, i));
    return r;
}
static jval* strftime_now(arena* A, jval** args, int na) {
    time_t t = time(NULL);
    struct tm tmv; localtime_r(&t, &tmv);
    const char* fmt = (na > 0) ? to_str_alloc(A, args[0]) : "";
    static const char* mon[] = { "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec" };
    static const char* mfull[] = { "January","February","March","April","May","June","July","August","September","October","November","December" };
    static const char* wday[] = { "Sun","Mon","Tue","Wed","Thu","Fri","Sat" };
    static const char* wdfull[] = { "Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday" };
    sb b; sb_init(&b);
    for (size_t i = 0; fmt[i]; i++) {
        if (fmt[i] != '%' || !fmt[i + 1]) { sb_putc(&b, fmt[i]); continue; }
        char c = fmt[++i];
        char tmp[64];
        switch (c) {
            case 'Y': snprintf(tmp, sizeof tmp, "%04d", tmv.tm_year + 1900); sb_put(&b, tmp); break;
            case 'y': snprintf(tmp, sizeof tmp, "%02d", (tmv.tm_year + 1900) % 100); sb_put(&b, tmp); break;
            case 'm': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_mon + 1); sb_put(&b, tmp); break;
            case 'd': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_mday); sb_put(&b, tmp); break;
            case 'H': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_hour); sb_put(&b, tmp); break;
            case 'I': { int h = tmv.tm_hour % 12; if (!h) h = 12; snprintf(tmp, sizeof tmp, "%02d", h); sb_put(&b, tmp); break; }
            case 'M': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_min); sb_put(&b, tmp); break;
            case 'S': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_sec); sb_put(&b, tmp); break;
            case 'p': sb_put(&b, tmv.tm_hour < 12 ? "AM" : "PM"); break;
            case 'b': sb_put(&b, mon[tmv.tm_mon]); break;
            case 'B': sb_put(&b, mfull[tmv.tm_mon]); break;
            case 'a': sb_put(&b, wday[tmv.tm_wday]); break;
            case 'A': sb_put(&b, wdfull[tmv.tm_wday]); break;
            case 'j': snprintf(tmp, sizeof tmp, "%03d", tmv.tm_yday + 1); sb_put(&b, tmp); break;
            case 'z': case 'Z': { size_t n = strftime(tmp, sizeof tmp, c == 'z' ? "%z" : "%Z", &tmv); if (n) sb_putn(&b, tmp, n); break; }
            case 'f': sb_put(&b, "000000"); break;
            case '%': sb_putc(&b, '%'); break;
            default: sb_putc(&b, '%'); sb_putc(&b, c); break;
        }
    }
    jval* r = v_str(A, b.p); sb_free(&b); return r;
}
static jval* eval_macro(arena* A, jmacro* m, jval** args, int na, jval** kwk, jval** kwv, int nkw) {
    env* me = env_child(m->closure);
    int argi = 0;
    for (size_t k = 0; k < m->np; k++) {
        jval* val = NULL;
        for (int i = 0; i < nkw; i++) if (kwk[i]->t == J_STR && strcmp(kwk[i]->u.s, m->pname[k]) == 0) { val = kwv[i]; break; }
        if (!val && argi < na) val = args[argi++];
        if (!val && m->pdef[k]) { val = eval_expr_str(m->pdef[k], me); if (!val) return NULL; }
        if (!val) val = v_null(A);
        env_set(me, m->pname[k], val);
    }
    return v_str(A, eval_to_string(A, &m->body, me));
}
static jval* call_fn(arena* A, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw, env* e) {
    if (strcmp(name, "raise_exception") == 0) { const char* m = na > 0 ? to_str_alloc(A, args[0]) : "raise_exception"; jerr_set("%s", m); return NULL; }
    if (strcmp(name, "namespace") == 0) {
        jval* o = v_obj(A);
        for (int i = 0; i < nkw; i++) obj_set(o, to_str_alloc(A, kwk[i]), kwv[i]);
        return o;
    }
    if (strcmp(name, "range") == 0) return range_fn(A, args, na);
    if (strcmp(name, "strftime_now") == 0) return strftime_now(A, args, na);
    jval* f = env_get(e, name);
    if (f && f->t == J_MACRO) return eval_macro(A, f->u.m, args, na, kwk, kwv, nkw);
    if (f && f->t == J_FN) return f->u.f(na, args);
    jerr_set("funzione non supportata: %s", name);
    return NULL;
}

/* ================= valore: stringify / json / confronti ================= */
static char* to_str_alloc(arena* A, const jval* v) {
    if (!v) return A ? a_strdup(A, "") : strdup("");
    switch (v->t) {
        case J_UNDEF: return A ? a_strdup(A, "") : strdup("");
        case J_NULL: return A ? a_strdup(A, "") : strdup("");
        case J_STR: return A ? a_strdup(A, v->u.s) : strdup(v->u.s);
        case J_BOOL: return A ? a_strdup(A, v->u.b ? "True" : "False") : strdup(v->u.b ? "True" : "False");
        case J_LONG: { char t[40]; snprintf(t, sizeof t, "%ld", v->u.l); return A ? a_strdup(A, t) : strdup(t); }
        case J_DOUBLE: { char t[64]; snprintf(t, sizeof t, "%.15g", v->u.d); return A ? a_strdup(A, t) : strdup(t); }
        case J_VAR: return A ? a_strdup(A, "") : strdup("");
        case J_ARR: {
            sb b; sb_init(&b); sb_putc(&b, '[');
            for (size_t i = 0; i < v->u.arr.n; i++) { if (i) sb_put(&b, ", "); char* s = to_str_alloc(NULL, v->u.arr.items[i]); sb_put(&b, s); free(s); }
            sb_putc(&b, ']');
            char* r = A ? a_strdup(A, b.p) : strdup(b.p); sb_free(&b); return r;
        }
        case J_OBJ: {
            sb b; sb_init(&b); sb_putc(&b, '{');
            for (size_t i = 0; i < v->u.obj.n; i++) { if (i) sb_put(&b, ", "); sb_put(&b, v->u.obj.items[i].k); sb_put(&b, ": "); char* s = to_str_alloc(NULL, v->u.obj.items[i].v); sb_put(&b, s); free(s); }
            sb_putc(&b, '}');
            char* r = A ? a_strdup(A, b.p) : strdup(b.p); sb_free(&b); return r;
        }
        default: return A ? a_strdup(A, "") : strdup("");
    }
}
static void json_ser(sb* b, const jval* v) {
    switch (v->t) {
        case J_UNDEF: case J_VAR: sb_put(b, "null"); break;
        case J_NULL: sb_put(b, "null"); break;
        case J_BOOL: sb_put(b, v->u.b ? "true" : "false"); break;
        case J_LONG: sb_num(b, v->u.l); break;
        case J_DOUBLE: { char t[64]; snprintf(t, sizeof t, "%.15g", v->u.d); sb_put(b, t); break; }
        case J_STR: {
            sb_putc(b, '"');
            for (const char* p = v->u.s; *p; p++) {
                unsigned char c = (unsigned char)*p;
                switch (c) {
                    case '"': sb_put(b, "\\\""); break;
                    case '\\': sb_put(b, "\\\\"); break;
                    case '\n': sb_put(b, "\\n"); break;
                    case '\r': sb_put(b, "\\r"); break;
                    case '\t': sb_put(b, "\\t"); break;
                    default:
                        if (c < 0x20) { char t[8]; snprintf(t, sizeof t, "\\u%04x", c); sb_put(b, t); }
                        else sb_putc(b, (char)c);
                }
            }
            sb_putc(b, '"');
            break;
        }
        case J_ARR: {
            sb_putc(b, '[');
            for (size_t i = 0; i < v->u.arr.n; i++) { if (i) sb_putc(b, ','); json_ser(b, v->u.arr.items[i]); }
            sb_putc(b, ']');
            break;
        }
        case J_OBJ: {
            sb_putc(b, '{');
            for (size_t i = 0; i < v->u.obj.n; i++) {
                if (i) sb_putc(b, ',');
                sb_putc(b, '"'); sb_put(b, v->u.obj.items[i].k); sb_put(b, "\":");
                json_ser(b, v->u.obj.items[i].v);
            }
            sb_putc(b, '}');
            break;
        }
        default: sb_put(b, "null"); break;
    }
}
static int v_eq(const jval* a, const jval* b) {
    if (a->t == J_VAR || b->t == J_VAR) return 0;
    if (a->t == J_UNDEF || b->t == J_UNDEF) return a->t == J_UNDEF && b->t == J_UNDEF;
    if (a->t == J_NULL || b->t == J_NULL) return a->t == J_NULL && b->t == J_NULL;
    if (a->t == J_STR && b->t == J_STR) return strcmp(a->u.s, b->u.s) == 0;
    if (a->t == J_BOOL && b->t == J_BOOL) return a->u.b == b->u.b;
    if (a->t == J_LONG && b->t == J_LONG) return a->u.l == b->u.l;
    if (a->t == J_LONG && b->t == J_DOUBLE) return (double)a->u.l == b->u.d;
    if (a->t == J_DOUBLE && b->t == J_LONG) return a->u.d == (double)b->u.l;
    if (a->t == J_DOUBLE && b->t == J_DOUBLE) return a->u.d == b->u.d;
    return a == b;
}
static int v_contains(const jval* container, const jval* item) {
    if (container->t == J_STR && item->t == J_STR) return strstr(container->u.s, item->u.s) != NULL;
    if (is_dict(container) && item->t == J_STR) return obj_has(container, item->u.s);
    if (container->t == J_ARR) { for (size_t i = 0; i < container->u.arr.n; i++) if (v_eq(container->u.arr.items[i], item)) return 1; return 0; }
    return 0;
}
static int v_compare(const jval* a, const jval* b, int* ok) {
    *ok = 1;
    if (a->t == J_LONG && b->t == J_LONG) return a->u.l < b->u.l ? -1 : a->u.l > b->u.l ? 1 : 0;
    if ((a->t == J_LONG || a->t == J_DOUBLE) && (b->t == J_LONG || b->t == J_DOUBLE)) {
        double da = a->t == J_LONG ? (double)a->u.l : a->u.d;
        double db = b->t == J_LONG ? (double)b->u.l : b->u.d;
        return da < db ? -1 : da > db ? 1 : 0;
    }
    if (a->t == J_STR && b->t == J_STR) { int c = strcmp(a->u.s, b->u.s); return c < 0 ? -1 : c > 0 ? 1 : 0; }
    if (a->t == J_VAR || b->t == J_VAR) { jerr_set("confronto con variabile non definita"); *ok = 0; return 0; }
    jerr_set("confronto tra tipi non supportati"); *ok = 0; return 0;
}
static jval* v_concat(jval* a, jval* b) {
    arena* A = a->A ? a->A : b->A;
    if (a->t == J_STR || b->t == J_STR) {
        char* x = to_str_alloc(NULL, a); char* y = to_str_alloc(NULL, b);
        char* z = (char*)malloc(strlen(x) + strlen(y) + 1); strcpy(z, x); strcat(z, y);
        jval* r = v_str(A, z); free(z); free(x); free(y); return r;
    }
    if (a->t == J_LONG && b->t == J_LONG) return v_long(A, a->u.l + b->u.l);
    if ((a->t == J_LONG || a->t == J_DOUBLE) && (b->t == J_LONG || b->t == J_DOUBLE)) {
        double da = a->t == J_LONG ? (double)a->u.l : a->u.d;
        double db = b->t == J_LONG ? (double)b->u.l : b->u.d;
        return v_double(A, da + db);
    }
    JERR("'+' non supportato tra questi tipi");
}

/* ================= eval ================= */
static jval* eval_expr_str(const char* expr, env* e) {
    toks tk; lex_expr(expr, &tk);
    if (g_err[0] || tk.t == NULL) { toks_free(&tk); return NULL; }
    jval* v = p_conditional(&tk, e);
    if (v && tk.p != tk.n) { jerr_set("espressione malformata: %s (token %s)", expr, tk.t[tk.p]); toks_free(&tk); return NULL; }
    toks_free(&tk);
    return v;
}
static int assign_target(const char* target, jval* val, env* e);
static int eval_list(nlist* nodes, env* e, sb* out);
static char* eval_to_string(arena* A, nlist* nodes, env* e) {
    sb out; sb_init(&out);
    if (eval_list(nodes, e, &out)) { sb_free(&out); return NULL; }
    char* r = a_strdup(A, out.p);
    sb_free(&out);
    return r;
}

static int eval_set_block(arena* A, node* nd, env* e) {
    sb out; sb_init(&out);
    if (eval_list(&nd->body, e, &out)) { sb_free(&out); return 1; }
    env_set(e, nd->s, v_str(A, out.p));
    sb_free(&out);
    return 0;
}
static jval* make_macro(arena* A, node* nd, env* e) {
    jmacro* m = (jmacro*)a_alloc(A, sizeof(jmacro));
    m->np = nd->np;
    m->pname = (char**)a_alloc(A, sizeof(char*) * (nd->np ? nd->np : 1));
    m->pdef = (char**)a_alloc(A, sizeof(char*) * (nd->np ? nd->np : 1));
    for (size_t i = 0; i < nd->np; i++) { m->pname[i] = a_strdup(A, nd->pname[i]); m->pdef[i] = nd->pdef[i] ? a_strdup(A, nd->pdef[i]) : NULL; }
    m->body = nd->body;
    m->closure = e;
    jval* v = v_new(A, J_MACRO); v->u.m = m; return v;
}
static size_t split_names(arena* A, const char* s, char*** out) {
    size_t n = 0; char** parts = split_top_level(NULL, s, ',', &n);
    char** names = (char**)a_alloc(A, sizeof(char*) * (n ? n : 1));
    size_t k = 0;
    for (size_t i = 0; i < n; i++) { char* t = trimdup(NULL, parts[i]); free(parts[i]); if (*t) { names[k++] = a_strdup(A, t); } free(t); }
    free(parts); *out = names; return k;
}
static int eval_list(nlist* nodes, env* e, sb* out) {
    arena* A = e->A;
    for (size_t i = 0; i < nodes->n; i++) {
        node* nd = nodes->p[i];
        switch (nd->kind) {
            case 0: sb_put(out, nd->s); break;
            case 1: { jval* v = eval_expr_str(nd->expr, e); if (!v) return 1; sb_put(out, to_str_alloc(A, v)); break; }
            case 2: { jval* v = eval_expr_str(nd->expr, e); if (!v) return 1; if (assign_target(nd->s, v, e)) return 1; break; }
            case 6: if (eval_set_block(A, nd, e)) return 1; break;
            case 5: env_set(e, nd->s, make_macro(A, nd, e)); break;
            case 3: {
                int took = 0;
                for (size_t k = 0; k < nd->nb && !took; k++) {
                    int take = 0;
                    if (!nd->br[k].cond) take = 1;
                    else { jval* c = eval_expr_str(nd->br[k].cond, e); if (!c) return 1; take = truth(c); }
                    if (take) { if (eval_list(&nd->br[k].body, e, out)) return 1; took = 1; }
                }
                break;
            }
            case 4: {
                jval* it = eval_expr_str(nd->expr, e);
                if (!it) return 1;
                jval* list = items_arr(A, it);
                if (!list) return 1;
                char** names = NULL; size_t nv = split_names(A, nd->s, &names);
                for (size_t idx = 0; idx < list->u.arr.n; idx++) {
                    env* le = env_child(e);
                    if (nv == 1) env_set(le, names[0], list->u.arr.items[idx]);
                    else {
                        jval* item = list->u.arr.items[idx];
                        jval* seq = (item->t == J_STR) ? NULL : items_arr(A, item);
                        if (!seq || seq->u.arr.n < nv) { jerr_set("'for %s ...' non puo' scompattare un valore non-pair", nd->s); return 1; }
                        for (size_t q = 0; q < nv; q++) env_set(le, names[q], seq->u.arr.items[q]);
                    }
                    jval* loop = v_obj(A);
                    obj_set(loop, "index", v_long(A, (long)idx + 1));
                    obj_set(loop, "index0", v_long(A, (long)idx));
                    obj_set(loop, "revindex", v_long(A, (long)list->u.arr.n - (long)idx));
                    obj_set(loop, "revindex0", v_long(A, (long)list->u.arr.n - (long)idx - 1));
                    obj_set(loop, "first", v_bool(A, idx == 0));
                    obj_set(loop, "last", v_bool(A, idx + 1 == list->u.arr.n));
                    obj_set(loop, "length", v_long(A, (long)list->u.arr.n));
                    obj_set(loop, "previtem", idx > 0 ? list->u.arr.items[idx - 1] : v_undef(A));
                    obj_set(loop, "nextitem", idx + 1 < list->u.arr.n ? list->u.arr.items[idx + 1] : v_undef(A));
                    env_set(le, "loop", loop);
                    int r = eval_list(&nd->body, le, out);
                    if (r) return 1;
                }
                break;
            }
            default: jerr_set("nodo non supportato (%d)", nd->kind); return 1;
        }
    }
    return 0;
}
static int assign_target(const char* target, jval* val, env* e) {
    arena* A = e->A;
    const char* dot = strchr(target, '.');
    const char* br = strchr(target, '[');
    const char* cut = NULL;
    if (dot && br) cut = dot < br ? dot : br; else if (dot) cut = dot; else if (br) cut = br;
    if (!cut || cut == target) { env_set(e, target, val); return 0; }
    char* baseName = trimdup(NULL, strndup(target, (size_t)(cut - target)));
    jval* obj = env_has(e, baseName) ? env_get(e, baseName) : v_undef(A);
    const char* rest = cut;
    if (rest[0] == '[') {
        const char* end = strchr(rest, ']');
        if (!end) { jerr_set("assegnazione con '[' non chiusa: %s", target); return 1; }
        char* keye = strndup(rest + 1, (size_t)(end - (rest + 1)));
        jval* key = eval_expr_str(keye, e);
        free(keye);
        if (!key) return 1;
        if (is_dict(obj)) { obj_set(obj, to_str_alloc(A, key), val); return 0; }
        jerr_set("assegnazione a indice su valore non-mapping: %s", target);
        return 1;
    }
    const char* attr = rest + 1;
    if (strchr(attr, '.') || strchr(attr, '[')) { jerr_set("assegnazione annidata non supportata: %s", target); return 1; }
    if (is_dict(obj)) { obj_set(obj, attr, val); return 0; }
    jerr_set("assegnazione a attributo su valore non-mapping: %s", target);
    return 1;
}

/* ================= render pubblico (specchio C# Jinja.Render) ================= */
char* llmjinja_render(const char* tmpl, const jval* vars, char* errbuf, size_t errcap) {
    g_err[0] = 0;
    if (!tmpl) { if (errbuf && errcap) snprintf(errbuf, errcap, "template nullo"); return NULL; }
    arena A; A.head = NULL;
    size_t ns = 0;
    seg* segs = tokenize(tmpl, &ns);
    if (!segs) {
        if (errbuf && errcap) snprintf(errbuf, errcap, "%s", g_err[0] ? g_err : "tokenize error");
        a_free(&A); return NULL;
    }
    nlist root = { NULL, 0, 0 };
    if (parse_all(segs, ns, &root)) {
        seglist_free(segs, ns); nlist_free(&root);
        if (errbuf && errcap) snprintf(errbuf, errcap, "%s", g_err[0] ? g_err : "parse error");
        a_free(&A); return NULL;
    }
    seglist_free(segs, ns);
    env* e = env_from_vars(&A, vars);
    sb out; sb_init(&out);
    if (eval_list(&root, e, &out)) {
        env_free(e); nlist_free(&root); sb_free(&out);
        if (errbuf && errcap) snprintf(errbuf, errcap, "%s", g_err[0] ? g_err : "eval error");
        a_free(&A); return NULL;
    }
    env_free(e);
    nlist_free(&root);
    size_t tl = strlen(tmpl);
    if (tl && tmpl[tl - 1] == '\n' && out.n && out.p[out.n - 1] == '\n') out.n--;
    out.p[out.n] = 0;
    char* result = strdup(out.p);
    sb_free(&out);
    a_free(&A);
    return result;
}

/* ================= API pubblica (llmjinja.h) ================= */
llmjinja_val* llmjinja_null(void) { return v_null(NULL); }
llmjinja_val* llmjinja_bool(int b) { return v_bool(NULL, b); }
llmjinja_val* llmjinja_long(long l) { return v_long(NULL, l); }
llmjinja_val* llmjinja_str(const char* s) { return v_str(NULL, s); }
llmjinja_val* llmjinja_arr(void) { return v_arr(NULL); }
void llmjinja_arr_push(llmjinja_val* arr, llmjinja_val* v) { arr_push(arr, v); }
llmjinja_val* llmjinja_obj(void) { return v_obj(NULL); }
void llmjinja_obj_set(llmjinja_val* obj, const char* key, llmjinja_val* v) { obj_set(obj, key, v); }
llmjinja_val* llmjinja_fn_val(llmjinja_fn f) { return v_fn(NULL, (jfn)f); }
void llmjinja_val_free(llmjinja_val* v) { v_free(v); }
int llmjinja_is_str(const llmjinja_val* v) { return v && v->t == J_STR; }
const char* llmjinja_as_str(const llmjinja_val* v) { return (v && v->t == J_STR) ? v->u.s : NULL; }

