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
#include "llmjinja_internal.h"
/* ================= error ================= */
char g_err[512];
void jerr_set(const char* fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(g_err, sizeof g_err, fmt, ap); va_end(ap); }

/* ================= arena ================= */
void* a_alloc(arena* A, size_t n) {
    if (!n) n = 1;
    apnode* p = (apnode*)malloc(sizeof(apnode) + n);
    p->next = A->head; A->head = p;
    return (char*)p + sizeof(apnode);
}
char* a_strdup(arena* A, const char* s) {
    s = s ? s : "";
    size_t l = strlen(s);
    char* p = (char*)a_alloc(A, l + 1);
    memcpy(p, s, l + 1);
    return p;
}
char* a_strndup(arena* A, const char* s, size_t l) {
    char* p = (char*)a_alloc(A, l + 1);
    memcpy(p, s, l); p[l] = 0;
    return p;
}
void a_free(arena* A) { apnode* p = A->head; while (p) { apnode* n = p->next; free(p); p = n; } A->head = NULL; }

/* ================= sb ================= */
void sb_init(sb* b) { b->cap = 256; b->n = 0; b->p = (char*)malloc(b->cap); b->p[0] = 0; }
void sb_grow(sb* b, size_t need) { while (b->n + need + 1 > b->cap) { b->cap *= 2; b->p = (char*)realloc(b->p, b->cap); } }
void jinja_sb_put(sb* b, const char* s) { size_t l = strlen(s); sb_grow(b, l); memcpy(b->p + b->n, s, l); b->n += l; b->p[b->n] = 0; }
void jinja_jinja_sb_putn(sb* b, const char* s, size_t l) { sb_grow(b, l); memcpy(b->p + b->n, s, l); b->n += l; b->p[b->n] = 0; }
void jinja_jinja_sb_putc(sb* b, char c) { sb_grow(b, 1); b->p[b->n++] = c; b->p[b->n] = 0; }
void jinja_sb_num(sb* b, long l) { char t[40]; snprintf(t, sizeof t, "%ld", l); jinja_sb_put(b, t); }
void sb_free(sb* b) { free(b->p); b->p = NULL; b->n = b->cap = 0; }

/* ================= valori ================= */
jval* v_new(arena* A, jtype t) { jval* v = A ? (jval*)a_alloc(A, sizeof *v) : (jval*)malloc(sizeof *v); memset(v, 0, sizeof *v); v->t = t; v->A = A; return v; }
jval* v_undef(arena* A) { return v_new(A, J_UNDEF); }
jval* v_null(arena* A) { return v_new(A, J_NULL); }
jval* v_bool(arena* A, int b) { jval* v = v_new(A, J_BOOL); v->u.b = b; return v; }
jval* v_long(arena* A, long l) { jval* v = v_new(A, J_LONG); v->u.l = l; return v; }
jval* v_double(arena* A, double d) { jval* v = v_new(A, J_DOUBLE); v->u.d = d; return v; }
jval* v_str(arena* A, const char* s) { jval* v = v_new(A, J_STR); v->u.s = A ? a_strdup(A, s) : strdup(s ? s : ""); return v; }
jval* v_str_take(arena* A, char* s) { jval* v = v_str(A, s); free(s); return v; }
jval* v_var(arena* A, const char* name) { jval* v = v_new(A, J_VAR); v->u.s = A ? a_strdup(A, name) : strdup(name ? name : ""); return v; }
jval* v_arr(arena* A) { return v_new(A, J_ARR); }
jval* v_obj(arena* A) { return v_new(A, J_OBJ); }
jval* v_fn(arena* A, jfn f) { jval* v = v_new(A, J_FN); v->u.f = f; return v; }
void arr_push(jval* a, jval* v) {
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
void obj_set(jval* o, const char* k, jval* v) {
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
void v_free(jval* v) {
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
int is_dict(const jval* v) { return v && v->t == J_OBJ; }
int is_list(const jval* v) { return v && v->t == J_ARR; }
int is_iterable(const jval* v) { return v && (v->t == J_STR || v->t == J_ARR || v->t == J_OBJ); }

jval* obj_get(const jval* o, const char* k) {
    for (size_t i = 0; i < o->u.obj.n; i++) if (strcmp(o->u.obj.items[i].k, k) == 0) return o->u.obj.items[i].v;
    return NULL;
}
int obj_has(const jval* o, const char* k) {
    for (size_t i = 0; i < o->u.obj.n; i++) if (strcmp(o->u.obj.items[i].k, k) == 0) return 1;
    return 0;
}

/* deep copy di un valore del chiamante nell'arena (per vars in ingresso) */
jval* v_deepcopy(arena* A, const jval* v) {
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
env* env_new(arena* A) { env* e = (env*)calloc(1, sizeof(env)); e->A = A; return e; }
void env_set(env* e, const char* k, jval* v) {
    for (size_t i = 0; i < e->n; i++) if (strcmp(e->e[i].k, k) == 0) { e->e[i].v = v; return; }
    if (e->n == e->cap) { e->cap = e->cap ? e->cap * 2 : 8; e->e = (void*)realloc(e->e, sizeof(*e->e) * e->cap); }
    e->e[e->n].k = a_strdup(e->A, k);
    e->e[e->n].v = v;
    e->n++;
}
int env_has(env* e, const char* k) { for (size_t i = 0; i < e->n; i++) if (strcmp(e->e[i].k, k) == 0) return 1; return 0; }
jval* env_get(env* e, const char* k) { for (size_t i = 0; i < e->n; i++) if (strcmp(e->e[i].k, k) == 0) return e->e[i].v; return NULL; }
env* env_child(env* p) {
    env* c = env_new(p->A);
    if (p->n) { c->e = (void*)malloc(sizeof(*c->e) * p->n); c->cap = p->n; memcpy(c->e, p->e, sizeof(*c->e) * p->n); c->n = p->n; }
    return c;
}
void env_free(env* e) { if (!e) return; free(e->e); free(e); }
env* env_from_vars(arena* A, const jval* vars) {
    env* e = env_new(A);
    if (vars && vars->t == J_OBJ)
        for (size_t i = 0; i < vars->u.obj.n; i++) env_set(e, vars->u.obj.items[i].k, v_deepcopy(A, vars->u.obj.items[i].v));
    return e;
}

/* helper stringa / numero */
char* to_str_alloc(arena* A, const jval* v);
long to_long(const jval* v) {
    switch (v->t) {
        case J_LONG: return v->u.l;
        case J_DOUBLE: return (long)v->u.d;
        default: jerr_set("atteso un intero"); return 0;
    }
}
int truth(const jval* v) {
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
long len_of(const jval* v) {
    switch (v->t) {
        case J_STR: return (long)strlen(v->u.s);
        case J_ARR: return (long)v->u.arr.n;
        case J_OBJ: return (long)v->u.obj.n;
        default: jerr_set("|length su valore non iterabile"); return 0;
    }
}

/* ================= template tokenizer (specchio C#) ================= */
void seglist_free(seg* a, size_t n) { for (size_t i = 0; i < n; i++) free(a[i].s); free(a); }
char* trimdup(arena* A, const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    size_t l = strlen(s);
    while (l && isspace((unsigned char)s[l - 1])) l--;
    return A ? a_strndup(A, s, l) : strndup(s, l);
}

seg* tokenize(const char* t, size_t* outn) {
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
        if (n == (size_t)-1) { jinja_sb_put(&text, t + i); break; }
        jinja_jinja_sb_putn(&text, t + i, n - i);
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
void nlist_add(nlist* l, node* nd) { if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 8; l->p = (node**)realloc(l->p, sizeof(void*) * l->cap); } l->p[l->n++] = nd; }
void nlist_free(nlist* l);
node* n_new(int kind) { node* nd = (node*)calloc(1, sizeof *nd); nd->kind = kind; return nd; }
void n_free(node* nd) {
    if (!nd) return;
    free(nd->s); free(nd->expr);
    for (size_t i = 0; i < nd->nb; i++) { free(nd->br[i].cond); nlist_free(&nd->br[i].body); }
    free(nd->br);
    nlist_free(&nd->body);
    for (size_t i = 0; i < nd->np; i++) { free(nd->pname[i]); free(nd->pdef[i]); }
    free(nd->pname); free(nd->pdef);
    free(nd);
}
void nlist_free(nlist* l) { for (size_t i = 0; i < l->n; i++) n_free(l->p[i]); free(l->p); l->p = NULL; l->n = l->cap = 0; }

const char* first_word(const char* s) {
    static char w[128];
    size_t k = 0;
    while (s[k] && !isspace((unsigned char)s[k]) && k < 127) k++;
    memcpy(w, s, k); w[k] = 0;
    return w;
}
char* substr_trim(const char* s, size_t start) {
    size_t len = strlen(s);
    if (start > len) return strdup("");
    return trimdup(NULL, s + start);
}
int stop_contains(const char* const* stop, size_t nstop, const char* w) {
    for (size_t i = 0; i < nstop; i++) if (strcmp(w, stop[i]) == 0) return 1;
    return 0;
}
/* split top-level (fuori da stringhe/parentesi) su sep; ritorna array arena */
char** split_top_level(arena* A, const char* s, char sep, size_t* outn) {
    size_t cap = 4, n = 0;
    char** res = (char**)malloc(sizeof(char*) * cap);
    sb cur; sb_init(&cur);
    int depth = 0; char q = 0; int esc = 0;
    for (size_t i = 0; ; i++) {
        char c = s[i];
        if (q) {
            jinja_jinja_sb_putc(&cur, c);
            if (esc) esc = 0;
            else if (c == '\\') esc = 1;
            else if (c == q) q = 0;
        } else {
            if (c == '\'' || c == '"') { q = c; jinja_jinja_sb_putc(&cur, c); }
            else if (c == '(' || c == '[' || c == '{') { depth++; jinja_jinja_sb_putc(&cur, c); }
            else if (c == ')' || c == ']' || c == '}') { depth--; jinja_jinja_sb_putc(&cur, c); }
            else if ((c == sep && depth == 0) || c == 0) {
                if (n == cap) { cap *= 2; res = (char**)realloc(res, sizeof(char*) * cap); }
                res[n++] = A ? a_strdup(A, cur.p) : strdup(cur.p);
                sb_free(&cur);
                if (c == 0) break;
                sb_init(&cur);
            } else jinja_jinja_sb_putc(&cur, c);
        }
        if (!c) break;
    }
    *outn = n;
    return res;
}
int top_level_index_of(const char* s, char target) {
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

int parse_until(seg* a, size_t n, size_t* idx, nlist* out, const char* const* stop, size_t nstop) {
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
int parse_all(seg* a, size_t n, nlist* root) {
    size_t idx = 0;
    int r = parse_until(a, n, &idx, root, NULL, 0);
    if (!r && idx < n) { jerr_set("tag di blocco senza apertura"); return 1; }
    return r;
}

/* ================= expr (specchio C# POr..PAtom) ================= */
void toks_free(toks* tk) { for (size_t i = 0; i < tk->n; i++) free(tk->t[i]); free(tk->t); tk->t = NULL; tk->n = tk->p = 0; }
void tok_add(toks* tk, const char* s, size_t len) { tk->t = (char**)realloc(tk->t, sizeof(char*) * (tk->n + 1)); tk->t[tk->n++] = strndup(s, len); }

void lex_expr(const char* s, toks* tk) {
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
                        case 'n': jinja_jinja_sb_putc(&b, '\n'); break;
                        case 't': jinja_jinja_sb_putc(&b, '\t'); break;
                        case 'r': jinja_jinja_sb_putc(&b, '\r'); break;
                        case '\\': jinja_jinja_sb_putc(&b, '\\'); break;
                        case '"': jinja_jinja_sb_putc(&b, '"'); break;
                        case '\'': jinja_jinja_sb_putc(&b, '\''); break;
                        default: jinja_jinja_sb_putc(&b, s[i]); break;
                    }
                    i++;
                } else { jinja_jinja_sb_putc(&b, s[i]); i++; }
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

int t_is(toks* tk, const char* s) { return tk->p < tk->n && strcmp(tk->t[tk->p], s) == 0; }

/* prototipi valutazione espressione */
jval* p_conditional(toks* tk, env* e);
jval* p_or(toks* tk, env* e);
jval* p_and(toks* tk, env* e);
jval* p_not(toks* tk, env* e);
jval* p_test(toks* tk, env* e);
jval* p_concat(toks* tk, env* e);
jval* p_unary(toks* tk, env* e);
jval* p_filter(toks* tk, env* e);
jval* p_postfix(toks* tk, env* e);
jval* p_atom(toks* tk, env* e);
jval* eval_expr_str(const char* expr, env* e);
void json_ser(sb* b, const jval* v);
jval* v_attr(arena* A, jval* v, const char* name);
jval* v_index(arena* A, jval* v, jval* i);
jval* v_slice(arena* A, jval* v, jval* a, jval* b, jval* step);
jval* method_call(arena* A, jval* v, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw);
jval* apply_filter(arena* A, const char* f, jval* v, jval** args, int na, jval** kwk, jval** kwv, int nkw);
jval* call_fn(arena* A, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw, env* e);
jval* items_arr(arena* A, jval* it);
char* eval_to_string(arena* A, nlist* nodes, env* e);

int is_name_token(const char* s) { return s[0] && (isalpha((unsigned char)s[0]) || s[0] == '_') && strncmp(s, "STR:", 4) != 0; }

int parse_call_args(toks* tk, env* e, jval*** pos, int* npos, jval*** kwk, jval*** kwv, int* nkw) {
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

jval* p_conditional(toks* tk, env* e) {
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
jval* p_or(toks* tk, env* e) {
    jval* v = p_and(tk, e);
    if (!v) return NULL;
    while (t_is(tk, "or")) { tk->p++; jval* r = p_and(tk, e); if (!r) return NULL; if (!truth(v)) v = r; }
    return v;
}
jval* p_and(toks* tk, env* e) {
    jval* v = p_not(tk, e);
    if (!v) return NULL;
    while (t_is(tk, "and")) { tk->p++; jval* r = p_not(tk, e); if (!r) return NULL; if (truth(v)) v = r; }
    return v;
}
jval* p_not(toks* tk, env* e) {
    if (t_is(tk, "not")) { tk->p++; jval* v = p_not(tk, e); if (!v) return NULL; return v_bool(e->A, !truth(v)); }
    return p_test(tk, e);
}

int v_eq(const jval* a, const jval* b);
int v_contains(const jval* container, const jval* item);
int v_compare(const jval* a, const jval* b, int* ok);

jval* p_test(toks* tk, env* e) {
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

jval* v_concat(jval* a, jval* b);
jval* p_concat(toks* tk, env* e) {
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
jval* p_unary(toks* tk, env* e) {
    if (t_is(tk, "-")) {
        tk->p++; jval* v = p_unary(tk, e); if (!v) return NULL;
        if (v->t == J_LONG) return v_long(e->A, -v->u.l);
        if (v->t == J_DOUBLE) return v_double(e->A, -v->u.d);
        JERR("meno unario non numerico");
    }
    return p_filter(tk, e);
}

jval* p_filter(toks* tk, env* e) {
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

jval* p_postfix(toks* tk, env* e) {
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

jval* p_atom(toks* tk, env* e) {
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

/* ================= eval ================= */
jval* eval_expr_str(const char* expr, env* e) {
    toks tk; lex_expr(expr, &tk);
    if (g_err[0] || tk.t == NULL) { toks_free(&tk); return NULL; }
    jval* v = p_conditional(&tk, e);
    if (v && tk.p != tk.n) { jerr_set("espressione malformata: %s (token %s)", expr, tk.t[tk.p]); toks_free(&tk); return NULL; }
    toks_free(&tk);
    return v;
}
int assign_target(const char* target, jval* val, env* e);
int eval_list(nlist* nodes, env* e, sb* out);
char* eval_to_string(arena* A, nlist* nodes, env* e) {
    sb out; sb_init(&out);
    if (eval_list(nodes, e, &out)) { sb_free(&out); return NULL; }
    char* r = a_strdup(A, out.p);
    sb_free(&out);
    return r;
}

int eval_set_block(arena* A, node* nd, env* e) {
    sb out; sb_init(&out);
    if (eval_list(&nd->body, e, &out)) { sb_free(&out); return 1; }
    env_set(e, nd->s, v_str(A, out.p));
    sb_free(&out);
    return 0;
}
jval* make_macro(arena* A, node* nd, env* e) {
    jmacro* m = (jmacro*)a_alloc(A, sizeof(jmacro));
    m->np = nd->np;
    m->pname = (char**)a_alloc(A, sizeof(char*) * (nd->np ? nd->np : 1));
    m->pdef = (char**)a_alloc(A, sizeof(char*) * (nd->np ? nd->np : 1));
    for (size_t i = 0; i < nd->np; i++) { m->pname[i] = a_strdup(A, nd->pname[i]); m->pdef[i] = nd->pdef[i] ? a_strdup(A, nd->pdef[i]) : NULL; }
    m->body = nd->body;
    m->closure = e;
    jval* v = v_new(A, J_MACRO); v->u.m = m; return v;
}
size_t split_names(arena* A, const char* s, char*** out) {
    size_t n = 0; char** parts = split_top_level(NULL, s, ',', &n);
    char** names = (char**)a_alloc(A, sizeof(char*) * (n ? n : 1));
    size_t k = 0;
    for (size_t i = 0; i < n; i++) { char* t = trimdup(NULL, parts[i]); free(parts[i]); if (*t) { names[k++] = a_strdup(A, t); } free(t); }
    free(parts); *out = names; return k;
}
int eval_list(nlist* nodes, env* e, sb* out) {
    arena* A = e->A;
    for (size_t i = 0; i < nodes->n; i++) {
        node* nd = nodes->p[i];
        switch (nd->kind) {
            case 0: jinja_sb_put(out, nd->s); break;
            case 1: { jval* v = eval_expr_str(nd->expr, e); if (!v) return 1; jinja_sb_put(out, to_str_alloc(A, v)); break; }
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
int assign_target(const char* target, jval* val, env* e) {
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

