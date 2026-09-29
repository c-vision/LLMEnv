/* llm_engine_opts.h — Registro di opzioni del motore (parametri SENZA variabili d'ambiente).
 *
 * Ogni parametro ha: nome canonico, tipo, default COMPILATO, valore corrente, set-by-user.
 * Si imposta via API (quindi da BASIC o dalla CLI) PRIMA di caricare/avviare il modello; il
 * motore lo legge (non getenv). Debug/dump restano via env (non sono parametri di funzionamento).
 */
#ifndef LLM_ENGINE_OPTS_H
#define LLM_ENGINE_OPTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Imposta un'opzione dal testo `value` (int/float/bool/string secondo il tipo).
 * Ritorna 0 se accettata, -1 se nome ignoto, -2 se valore non valido. */
int llm_ai_mlx_opt_set(const char* name, const char* value);

int llm_ai_mlx_opt_get_int(const char* name, int* out);
int llm_ai_mlx_opt_get_float(const char* name, double* out);
int llm_ai_mlx_opt_get_bool(const char* name, int* out);      /* 1/0 */
int llm_ai_mlx_opt_get_str(const char* name, char* out, int cap);

/* 1 se l'utente ha sovrascritto il default. */
int llm_ai_mlx_opt_is_set(const char* name);

/* Ponte verso il codice esistente (stessa forma di getenv): registro se impostato, altrimenti
 * getenv (fallback di TRANSIZIONE). Ritorna NULL se non impostato da nessuno dei due. */
const char* llm_ai_mlx_opt_env(const char* name);
/* Registrazione del setter TurboQuant da parte del modulo mlx (decoupling dei link). */
void llm_ai_mlx_opt_register_tq_setter(void (*fn)(int));

/* Enumerazione (per `AI OPTION LIST`): ritorna il numero di opzioni; `opt_at` riempie i campi. */
int llm_ai_mlx_opt_count(void);
int llm_ai_mlx_opt_at(int i, const char** name, const char** desc, char* value, int cap);

#ifdef __cplusplus
}
#endif

#endif /* LLM_ENGINE_OPTS_H */
