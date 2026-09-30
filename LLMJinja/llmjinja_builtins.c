/* llmjinja_builtins.c — dominio BUILTINS (§0quater): valore attr/index/slice, iterazione/unpack,
 * filtri, metodi, funzioni, stringify/json e confronti. Tipi/prototipi in llmjinja_internal.h. */
#include "llmjinja_internal.h"
/* ================= valore: attr/index/slice ================= */
jval* v_attr(arena* A, jval* v, const char* name) {
    if (v->t == J_VAR || v->t == J_UNDEF) return v_undef(A);
    if (is_dict(v)) { jval* g = obj_get(v, name); return g ? g : v_undef(A); }
    jerr_set("attributo '%s' su valore non-mapping", name);
    return NULL;
}
jval* v_index(arena* A, jval* v, jval* i) {
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
long clampi(long x, long lo, long hi) { return x < lo ? lo : x > hi ? hi : x; }
jval* v_slice(arena* A, jval* v, jval* a, jval* b, jval* stepv) {
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
        for (long k = lo; k < hi; k += step) jinja_jinja_sb_putc(&o, v->u.s[k]);
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
        for (long k = lo; k > hi; k += step) jinja_jinja_sb_putc(&o, v->u.s[k]);
        jval* r = v_str(A, o.p); sb_free(&o); return r;
    }
}

/* ================= iterazione / unpack ================= */
jval* items_arr(arena* A, jval* it) {
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
jval* v_tojson(arena* A, const jval* v) { sb b; sb_init(&b); json_ser(&b, v); jval* r = v_str(A, b.p); sb_free(&b); return r; }
jval* apply_by_name(arena* A, const char* fn, jval* x) {
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
int pair_cmp(const void* pa, const void* pb) {
    const jval* a = *(const jval* const*)pa; const jval* b = *(const jval* const*)pb;
    const char* ka = (a->t == J_ARR && a->u.arr.n > 0 && a->u.arr.items[0]->t == J_STR) ? a->u.arr.items[0]->u.s : "";
    const char* kb = (b->t == J_ARR && b->u.arr.n > 0 && b->u.arr.items[0]->t == J_STR) ? b->u.arr.items[0]->u.s : "";
    return strcmp(ka, kb);
}
jval* apply_filter(arena* A, const char* f, jval* v, jval** args, int na, jval** kwk, jval** kwv, int nkw) {
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
        for (size_t i = 0; i < list->u.arr.n; i++) { if (i) jinja_sb_put(&b, sep); jinja_sb_put(&b, to_str_alloc(A, list->u.arr.items[i])); }
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
        if (fl == 0) { jinja_sb_put(&b, s); }
        else {
            const char* p = s;
            while (*p) {
                if (strncmp(p, from, fl) == 0) { jinja_sb_put(&b, to); p += fl; }
                else jinja_jinja_sb_putc(&b, *p++);
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
jval* split_string(arena* A, const char* s, jval** args, int na) {
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
jval* py_strip(arena* A, const char* s, const char* chars, int left, int right) {
    size_t len = strlen(s), lo = 0, hi = len;
    if (left) while (lo < hi && (chars ? strchr(chars, s[lo]) != NULL : isspace((unsigned char)s[lo]))) lo++;
    if (right) while (hi > lo && (chars ? strchr(chars, s[hi - 1]) != NULL : isspace((unsigned char)s[hi - 1]))) hi--;
    return v_str_take(A, strndup(s + lo, hi - lo));
}
jval* starts_ends(arena* A, const char* s, jval** args, int na, int start) {
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
jval* method_call(arena* A, jval* v, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw) {
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
jval* range_fn(arena* A, jval** args, int na) {
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
jval* strftime_now(arena* A, jval** args, int na) {
    time_t t = time(NULL);
    struct tm tmv; localtime_r(&t, &tmv);
    const char* fmt = (na > 0) ? to_str_alloc(A, args[0]) : "";
    static const char* mon[] = { "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec" };
    static const char* mfull[] = { "January","February","March","April","May","June","July","August","September","October","November","December" };
    static const char* wday[] = { "Sun","Mon","Tue","Wed","Thu","Fri","Sat" };
    static const char* wdfull[] = { "Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday" };
    sb b; sb_init(&b);
    for (size_t i = 0; fmt[i]; i++) {
        if (fmt[i] != '%' || !fmt[i + 1]) { jinja_jinja_sb_putc(&b, fmt[i]); continue; }
        char c = fmt[++i];
        char tmp[64];
        switch (c) {
            case 'Y': snprintf(tmp, sizeof tmp, "%04d", tmv.tm_year + 1900); jinja_sb_put(&b, tmp); break;
            case 'y': snprintf(tmp, sizeof tmp, "%02d", (tmv.tm_year + 1900) % 100); jinja_sb_put(&b, tmp); break;
            case 'm': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_mon + 1); jinja_sb_put(&b, tmp); break;
            case 'd': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_mday); jinja_sb_put(&b, tmp); break;
            case 'H': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_hour); jinja_sb_put(&b, tmp); break;
            case 'I': { int h = tmv.tm_hour % 12; if (!h) h = 12; snprintf(tmp, sizeof tmp, "%02d", h); jinja_sb_put(&b, tmp); break; }
            case 'M': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_min); jinja_sb_put(&b, tmp); break;
            case 'S': snprintf(tmp, sizeof tmp, "%02d", tmv.tm_sec); jinja_sb_put(&b, tmp); break;
            case 'p': jinja_sb_put(&b, tmv.tm_hour < 12 ? "AM" : "PM"); break;
            case 'b': jinja_sb_put(&b, mon[tmv.tm_mon]); break;
            case 'B': jinja_sb_put(&b, mfull[tmv.tm_mon]); break;
            case 'a': jinja_sb_put(&b, wday[tmv.tm_wday]); break;
            case 'A': jinja_sb_put(&b, wdfull[tmv.tm_wday]); break;
            case 'j': snprintf(tmp, sizeof tmp, "%03d", tmv.tm_yday + 1); jinja_sb_put(&b, tmp); break;
            case 'z': case 'Z': { size_t n = strftime(tmp, sizeof tmp, c == 'z' ? "%z" : "%Z", &tmv); if (n) jinja_jinja_sb_putn(&b, tmp, n); break; }
            case 'f': jinja_sb_put(&b, "000000"); break;
            case '%': jinja_jinja_sb_putc(&b, '%'); break;
            default: jinja_jinja_sb_putc(&b, '%'); jinja_jinja_sb_putc(&b, c); break;
        }
    }
    jval* r = v_str(A, b.p); sb_free(&b); return r;
}
jval* eval_macro(arena* A, jmacro* m, jval** args, int na, jval** kwk, jval** kwv, int nkw) {
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
jval* call_fn(arena* A, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw, env* e) {
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
char* to_str_alloc(arena* A, const jval* v) {
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
            sb b; sb_init(&b); jinja_jinja_sb_putc(&b, '[');
            for (size_t i = 0; i < v->u.arr.n; i++) { if (i) jinja_sb_put(&b, ", "); char* s = to_str_alloc(NULL, v->u.arr.items[i]); jinja_sb_put(&b, s); free(s); }
            jinja_jinja_sb_putc(&b, ']');
            char* r = A ? a_strdup(A, b.p) : strdup(b.p); sb_free(&b); return r;
        }
        case J_OBJ: {
            sb b; sb_init(&b); jinja_jinja_sb_putc(&b, '{');
            for (size_t i = 0; i < v->u.obj.n; i++) { if (i) jinja_sb_put(&b, ", "); jinja_sb_put(&b, v->u.obj.items[i].k); jinja_sb_put(&b, ": "); char* s = to_str_alloc(NULL, v->u.obj.items[i].v); jinja_sb_put(&b, s); free(s); }
            jinja_jinja_sb_putc(&b, '}');
            char* r = A ? a_strdup(A, b.p) : strdup(b.p); sb_free(&b); return r;
        }
        default: return A ? a_strdup(A, "") : strdup("");
    }
}
void json_ser(sb* b, const jval* v) {
    switch (v->t) {
        case J_UNDEF: case J_VAR: jinja_sb_put(b, "null"); break;
        case J_NULL: jinja_sb_put(b, "null"); break;
        case J_BOOL: jinja_sb_put(b, v->u.b ? "true" : "false"); break;
        case J_LONG: jinja_sb_num(b, v->u.l); break;
        case J_DOUBLE: { char t[64]; snprintf(t, sizeof t, "%.15g", v->u.d); jinja_sb_put(b, t); break; }
        case J_STR: {
            jinja_jinja_sb_putc(b, '"');
            for (const char* p = v->u.s; *p; p++) {
                unsigned char c = (unsigned char)*p;
                switch (c) {
                    case '"': jinja_sb_put(b, "\\\""); break;
                    case '\\': jinja_sb_put(b, "\\\\"); break;
                    case '\n': jinja_sb_put(b, "\\n"); break;
                    case '\r': jinja_sb_put(b, "\\r"); break;
                    case '\t': jinja_sb_put(b, "\\t"); break;
                    default:
                        if (c < 0x20) { char t[8]; snprintf(t, sizeof t, "\\u%04x", c); jinja_sb_put(b, t); }
                        else jinja_jinja_sb_putc(b, (char)c);
                }
            }
            jinja_jinja_sb_putc(b, '"');
            break;
        }
        case J_ARR: {
            jinja_jinja_sb_putc(b, '[');
            for (size_t i = 0; i < v->u.arr.n; i++) { if (i) jinja_jinja_sb_putc(b, ','); json_ser(b, v->u.arr.items[i]); }
            jinja_jinja_sb_putc(b, ']');
            break;
        }
        case J_OBJ: {
            jinja_jinja_sb_putc(b, '{');
            for (size_t i = 0; i < v->u.obj.n; i++) {
                if (i) jinja_jinja_sb_putc(b, ',');
                jinja_jinja_sb_putc(b, '"'); jinja_sb_put(b, v->u.obj.items[i].k); jinja_sb_put(b, "\":");
                json_ser(b, v->u.obj.items[i].v);
            }
            jinja_jinja_sb_putc(b, '}');
            break;
        }
        default: jinja_sb_put(b, "null"); break;
    }
}
int v_eq(const jval* a, const jval* b) {
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
int v_contains(const jval* container, const jval* item) {
    if (container->t == J_STR && item->t == J_STR) return strstr(container->u.s, item->u.s) != NULL;
    if (is_dict(container) && item->t == J_STR) return obj_has(container, item->u.s);
    if (container->t == J_ARR) { for (size_t i = 0; i < container->u.arr.n; i++) if (v_eq(container->u.arr.items[i], item)) return 1; return 0; }
    return 0;
}
int v_compare(const jval* a, const jval* b, int* ok) {
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
jval* v_concat(jval* a, jval* b) {
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

