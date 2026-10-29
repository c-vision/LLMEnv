/* llmjinja_selfcheck.c — comportamento dell'ambiente Jinja2 dei chat template Hugging Face (trim_blocks, lstrip_blocks,
 * keep_trailing_newline=false) e operatori aritmetici. I valori attesi sono quelli prodotti da jinja2 con
 * Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False).
 * Compilazione: clang -I LLMJinja LLMJinja/llmjinja_selfcheck.c LLMJinja/llmjinja.c LLMJinja/llmjinja_builtins.c -o /tmp/jinja_selfcheck */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llmjinja.h"

static int check(const char* name, const char* tmpl, const llmjinja_val* vars, const char* expect) {
    char err[256] = "";
    char* out = llmjinja_render(tmpl, vars, err, sizeof err);
    int ok = out != NULL && strcmp(out, expect) == 0;
    if (!ok) printf("FAIL %-14s atteso '%s' ottenuto '%s' %s\n", name, expect, out ? out : "(null)", err);
    free(out);
    return ok ? 0 : 1;
}

int main(void) {
    int bad = 0;
    llmjinja_val* v1 = llmjinja_obj(); llmjinja_obj_set(v1, "b", llmjinja_str("B"));
    bad += check("comment_nl", "{# c #}\n{{ b }}", v1, "B");
    llmjinja_val* xs = llmjinja_arr();
    llmjinja_arr_push(xs, llmjinja_long(1)); llmjinja_arr_push(xs, llmjinja_long(2));
    llmjinja_val* v2 = llmjinja_obj(); llmjinja_obj_set(v2, "xs", xs);
    bad += check("for_trim", "{% for m in xs %}\n{{ m }}\n{% endfor %}\n", v2, "1\n2\n");
    llmjinja_val* v3 = llmjinja_obj();
    bad += check("lstrip_if", "  {% if true %}\n  x\n  {% endif %}\n", v3, "  x\n");
    bad += check("arith", "{{ 7 % 3 }} {{ -7 % 3 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 * 3 }} {{ 7 / 2 }} {{ 2 + 3 * 4 }}", v3, "1 2 3 -4 21 3.5 14");
    bad += check("trailing_nl", "a\n", v3, "a");
    bad += check("expr_nl_kept", "{{ 'x\n' }}\n", v3, "x\n");
    llmjinja_val* xs3 = llmjinja_arr();
    for (int i = 1; i <= 3; i++) llmjinja_arr_push(xs3, llmjinja_long(i));
    llmjinja_val* v4 = llmjinja_obj(); llmjinja_obj_set(v4, "xs", xs3);
    bad += check("alternate", "{% for m in xs %}{{ loop.index0 % 2 == 0 }}{% endfor %}", v4, "TrueFalseTrue");
    printf("llmjinja_selfcheck: %s\n", bad ? "FALLITO" : "OK (7 casi)");
    return bad != 0;
}
