/* llmengine_moe_json.h — helper JSON minimale (config.json / index safetensors / metadati).
 *
 * Deliberatamente tollerante e senza allocazioni: sufficiente a leggere i campi piani del
 * `config.json` HF e l'oggetto `weight_map` dell'index safetensors. Non è un parser JSON
 * generale (niente escape unicode, niente numeri esotici). Condiviso da discovery/converter. */
#ifndef LLMENGINE_MOE_JSON_H
#define LLMENGINE_MOE_JSON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Legge un file intero in RAM, NUL-terminato. Ritorna NULL su errore. Da free() dal chiamante. */
char* llmengine_moe_slurp(const char* path, size_t* n_out);

/* Valore associato alla chiave `"key"` (posizionato dopo ':'), o NULL. */
const char* llmengine_moe_json_find(const char* s, const char* key);

/* Conversioni dal puntatore-valore. Ritornano 1 su successo, 0 su fallimento. */
int llmengine_moe_json_int(const char* p, long long* out);
int llmengine_moe_json_double(const char* p, double* out);
int llmengine_moe_json_str(const char* p, char* out, size_t cap);
int llmengine_moe_json_bool(const char* p, int* out);   /* true/false/0/1 */

/* Comodità: valore o default. */
long long llmengine_moe_json_num(const char* json, const char* key, long long def);
double    llmengine_moe_json_flt(const char* json, const char* key, double def);
int       llmengine_moe_json_flag(const char* json, const char* key, int def);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_JSON_H */
