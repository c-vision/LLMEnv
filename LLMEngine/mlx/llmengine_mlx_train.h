/* ============================================================================
 * llmengine_mlx_train.h — ABI Apache-2.0 dei PRIMITIVI DI TRAINING dell'engine.
 *
 * Solo i PRIMITIVI (forward differenziabile in-graph + hook SDPA + setup fusions).
 * L'ORCHESTRAZIONE del training (loss CE/DPO, value_and_grad, optimizer, checkpoint)
 * vive nel progetto PUL `gmbllm/train/` (vedi ADR-Confine-Licenza-...): qui resta
 * l'ABI "strictly necessary" per far girare un forward differenziabile.
 *
 * Nota: espone tipi mlx-c (Apache-2.0) — il chiamante (PUL) costruisce le proprie
 * closure autograd sopra `llm_ai_mlx_train_forward_ce`.
 * ============================================================================ */
#ifndef LLMENGINE_MLX_TRAIN_H
#define LLMENGINE_MLX_TRAIN_H

#include "llmengine_mlx.h"
#include <mlx/c/array.h>
#include <mlx/c/vector.h>
#include <mlx/c/closure.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Prepara il modello al training differenziabile (disabilita le fusioni Metal/compile che non
 * hanno VJP e installa l'SDPA eager). Idempotente. Ritorna 0 ok. */
int llm_ai_mlx_train_setup(llm_ai_mlx_model_handle mh);

/* Imposta i tensori LoRA A/B come TENSORI DEL GRAFO (input del forward differenziabile) per
 * (layer, proj). `li` (indice adapter) viene registrato; A/B sono riferimenti (no copia host). */
int llm_ai_mlx_train_set_lora_mlx(llm_ai_mlx_model_handle mh, int layer, const char* proj,
                                  mlx_array A, mlx_array B, int* out_li);

/* Forward differenziabile in-graph + masked CE.
 *   `toks` [T], `tgt` [T], `mask` [T] (NULL = tutte); sum_mode=false -> media sui token
 *   mascherati (SFT), true -> somma (= -sum logp, per DPO).
 * Ritorna la loss (mlx_array) in-graph (lazy) oppure un array vuoto su errore; *ok_out = esito.
 * Il chiamante (PUL) la usa dentro una closure + mlx_value_and_grad. */
mlx_array llm_ai_mlx_train_forward_ce(llm_ai_mlx_model_handle mh,
                                      const int32_t* toks, const int32_t* tgt, const uint8_t* mask,
                                      int T, bool sum_mode, bool* ok_out);

/* Abilita/disabilita l'uso dell'SDPA eager (necessaria per il VJP). */
void llm_ai_mlx_train_set_eager_sdpa(int on);

#ifdef __cplusplus
}
#endif
#endif /* LLMENGINE_MLX_TRAIN_H */
