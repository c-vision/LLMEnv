/* llmjinja_internal.h — tipi, macro e prototipi CONDIVISI fra i .c del motore Jinja.
 *
 * §0quater: `llmjinja.c` era un monolite di 1550 righe. Il codice è diviso per dominio in
 * `llmjinja.c` (helper: arena/sb/valori/env + parser/tokenizer + expr + eval + render/API) e
 * `llmjinja_builtins.c` (attr/index/slice, iterazione, filtri, metodi, funzioni, stringify/json,
 * confronti). Le funzioni usate da entrambi sono NON-static e dichiarate qui (non nel header
 * pubblico `llmjinja.h`, che resta l'ABI stabile del motore).
 */
#ifndef LLMJINJA_INTERNAL_H
#define LLMJINJA_INTERNAL_H

#include "llmjinja.h"

#include <ctype.h>
#include <stdarg.h>
#include <stddef.h>
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

typedef struct apnode { struct apnode* next; } apnode;
typedef struct arena { apnode* head; } arena;
typedef struct { char* p; size_t n, cap; } sb;
struct env { struct { char* k; jval* v; }* e; size_t n, cap; arena* A; };
typedef struct { int kind; /*0 txt 1 out 2 stmt*/ char* s; } seg;
struct node {
    int kind;                 /* 0 Txt 1 Out 2 Set 3 If 4 For 5 Macro 6 SetBlock */
    char* s;                  /* Txt:texto | Out:expr | Set:var | For:var | Macro:name */
    char* expr;               /* Out expr / Set expr / For expr */
    struct { char* cond; nlist body; }* br; size_t nb;   /* If */
    nlist body;               /* For/Macro/SetBlock body */
    char** pname; char** pdef; size_t np;                /* Macro params */
};
typedef struct { char** t; size_t n, p; } toks;

#define JERR(...) do { jerr_set(__VA_ARGS__); return NULL; } while (0)

/* ---- error / helper condivisi (definiti in llmjinja.c) ---- */
extern char g_err[512];
void jerr_set(const char* fmt, ...);
void* a_alloc(arena* A, size_t n);
char* a_strdup(arena* A, const char* s);
char* a_strndup(arena* A, const char* s, size_t l);
void a_free(arena* A);
void sb_init(sb* b);
void sb_grow(sb* b, size_t need);
void sb_put(sb* b, const char* s);
void sb_putn(sb* b, const char* s, size_t l);
void sb_putc(sb* b, char c);
void sb_num(sb* b, long l);
void sb_free(sb* b);
jval* v_new(arena* A, jtype t);
jval* v_undef(arena* A);
jval* v_null(arena* A);
jval* v_bool(arena* A, int b);
jval* v_long(arena* A, long l);
jval* v_double(arena* A, double d);
jval* v_str(arena* A, const char* s);
jval* v_str_take(arena* A, char* s);
jval* v_var(arena* A, const char* name);
jval* v_arr(arena* A);
jval* v_obj(arena* A);
jval* v_fn(arena* A, jfn f);
void arr_push(jval* a, jval* v);
void obj_set(jval* o, const char* k, jval* v);
void v_free(jval* v);
int is_dict(const jval* v);
int is_list(const jval* v);
int is_iterable(const jval* v);
jval* obj_get(const jval* o, const char* k);
int obj_has(const jval* o, const char* k);
jval* v_deepcopy(arena* A, const jval* v);
env* env_new(arena* A);
void env_set(env* e, const char* k, jval* v);
int env_has(env* e, const char* k);
jval* env_get(env* e, const char* k);
env* env_child(env* p);
void env_free(env* e);
env* env_from_vars(arena* A, const jval* vars);
long to_long(const jval* v);
int truth(const jval* v);
long len_of(const jval* v);

/* ---- parser / expr (definiti in llmjinja.c) ---- */
int parse_all(seg* a, size_t n, nlist* root);
void toks_free(toks* tk);
int t_is(toks* tk, const char* s);
int is_name_token(const char* s);
int parse_call_args(toks* tk, env* e, jval*** pos, int* npos, jval*** kwk, jval*** kwv, int* nkw);
jval* eval_expr_str(const char* expr, env* e);
char* eval_to_string(arena* A, nlist* nodes, env* e);

/* ---- builtins (definiti in llmjinja_builtins.c) ---- */
jval* v_attr(arena* A, jval* v, const char* name);
jval* v_index(arena* A, jval* v, jval* i);
jval* v_slice(arena* A, jval* v, jval* a, jval* b, jval* step);
jval* items_arr(arena* A, jval* it);
jval* apply_filter(arena* A, const char* f, jval* v, jval** args, int na, jval** kwk, jval** kwv, int nkw);
jval* method_call(arena* A, jval* v, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw);
jval* call_fn(arena* A, const char* name, jval** args, int na, jval** kwk, jval** kwv, int nkw, env* e);
jval* v_tojson(arena* A, const jval* v);
void json_ser(sb* b, const jval* v);
char* to_str_alloc(arena* A, const jval* v);
int v_eq(const jval* a, const jval* b);
int v_contains(const jval* container, const jval* item);
int v_compare(const jval* a, const jval* b, int* ok);
jval* v_concat(jval* a, jval* b);

#endif /* LLMJINJA_INTERNAL_H */
