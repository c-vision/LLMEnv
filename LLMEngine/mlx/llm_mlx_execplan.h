/* llm_mlx_execplan.h — §9.A M10 EXECUTION PLANNER (runtime manager).
 *
 * Generalizza il pipeline per-layer del `.llmarch` (llm_mlx_set_layer_ops) in un PIANO
 * ESECUTIVO per-layer ingerito a runtime: ogni layer porta la sequenza di op e/o una
 * decisione SKIP (pass-through). Consumato da `layer_forward` (llm_mlx_layer_forward.c).
 *
 * Il piano arriva dal compile-time (il producer PUL (`LLM TENSOR EXECPLAN`)) come DATO
 * JSON — nessuna dipendenza PUL: il motore Apache lo legge e lo applica.
 * Formato:
 *   {"version":1,
 *    "layers":[ {"ops":["attention","ffn"]},
 *               {"skip":true},
 *               {"ops":["attention","ffn"],"skip":false} ]}
 * Un layer assente dal piano usa la pipeline di default (`g_mlx_layer_ops` o [attention,ffn]).
 */
#ifndef LLM_MLX_EXECPLAN_H
#define LLM_MLX_EXECPLAN_H

#include <stddef.h>

#define LLM_MLX_EXECPLAN_MAX_OPS 16
#define LLM_MLX_EXECPLAN_OP_LEN  24

/* Carica il piano da file JSON. Ritorna 0 ok, -1 errore (file/parse). Sostituisce il precedente. */
int  llm_mlx_execplan_load(const char* path);
/* Come load ma da stringa (seam di test). */
int  llm_mlx_execplan_load_str(const char* json);
/* Azzera il piano (torna al default per-layer). */
void llm_mlx_execplan_reset(void);
/* 1 se un piano e' attivo. */
int  llm_mlx_execplan_active(void);
/* Numero di layer coperti dal piano (0 se nessuno). */
int  llm_mlx_execplan_nlayers(void);
/* Per il layer `layer`: copia fino a `max` op in `ops[i]` (NUL-terminate) e ritorna n op;
 * setta `*skip` (1 = SKIP il layer, pass-through). Ritorna -1 se il layer NON e' nel piano
 * (il chiamante usa il default). */
int  llm_mlx_execplan_layer(int layer, char ops[][LLM_MLX_EXECPLAN_OP_LEN], int max, int* skip);

#endif /* LLM_MLX_EXECPLAN_H */
