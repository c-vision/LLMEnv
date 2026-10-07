/* llmjinja.h — motore Jinja (subset per i chat-template LLM locali), VERSIONE C.
 *
 * Port di `il motore Jinja originale` (stesso autore, Gaetano Minardi). Contratto identico:
 * llmjinja_render(template, vars) con semantica Jinja2; costrutto non supportato -> errore
 * esplicito (fail-loud, mai output approssimato). Niente stub.
 *
 * Tipi di valore esposti (si costruiscono i vars come oggetto):
 *   llmjinja_null/bool/long/str/arr/obj + llmjinja_val_free per l'ownership dei valori
 *   creati dal chiamante. Una funzione di contesto (es. "strftime_now") si passa come
 *   valore J_FN: llmjinja_fn(nargs, args) deve RITORNARE un valore nuovo (l'engine libera
 *   args dopo la chiamata) e non trattenere i puntatori arg.
 *
 * La funzione ritorna stringa allocata (free() del chiamante) o NULL con messaggio in err.
 */
#ifndef LLMJINJA_H
#define LLMJINJA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llmjinja_val llmjinja_val;
typedef llmjinja_val *(*llmjinja_fn)(int nargs, llmjinja_val **args);

/* costruttori/ownership: llmjinja_val_free libera l'intero albero */
llmjinja_val *llmjinja_null(void);
llmjinja_val *llmjinja_bool(int b);
llmjinja_val *llmjinja_long(long l);
llmjinja_val *llmjinja_str(const char *s);
llmjinja_val *llmjinja_arr(void);
void llmjinja_arr_push(llmjinja_val *arr, llmjinja_val *v);
llmjinja_val *llmjinja_obj(void);
void llmjinja_obj_set(llmjinja_val *obj, const char *key, llmjinja_val *v);
llmjinja_val *llmjinja_fn_val(llmjinja_fn f);
void llmjinja_val_free(llmjinja_val *v);
/* accessor per le jfn di contesto (il valore è opaco fuori dall'engine) */
int llmjinja_is_str(const llmjinja_val *v);
const char *llmjinja_as_str(const llmjinja_val *v);

/* render del template con vars (oggetto, può contenere anche J_FN).
 * ritorna malloc'd oppure NULL con err[] valorizzato. */
char *llmjinja_render(const char *template, const llmjinja_val *vars, char *err, size_t errcap);

#ifdef __cplusplus
}
#endif
#endif /* LLMJINJA_H */
