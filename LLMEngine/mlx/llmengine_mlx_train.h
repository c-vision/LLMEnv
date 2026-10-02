/* ============================================================================
 * llmengine_mlx_train.h — ABI Apache dei PRIMITIVI del forward DIFFERENZIABILE.
 *
 * PERIMETRO DI LICENZA (Apache-2.0). Questo header è il CONTRATTO pubblico che
 * l'engine MLX (Apache) espone per consentire a un consumer PUL (gmbllm, dentro
 * GMBasic) di costruire loss/grad/DPO/AdamW SENZA reimplementare il forward.
 *
 * Cosa fa questa ABI (INFERENZA, non training):
 *   - esegue il forward del modello su una sequenza di token, RESTITUENDO la
 *     cross-entropy (loss) come `mlx_array` DENTRO il grafo autograd di MLX;
 *   - inietta i parametri LoRA (A/B) come `mlx_array` nel grafo, così che
 *     `mlx_value_and_grad` possa differenziare rispetto ad A/B.
 *
 * Cosa NON fa (sta nel PUL, `gmbllm/src/train/llm_train.c`): comporre la loss
 * (DPO/masked-CE), `mlx_value_and_grad`, l'optimizer (AdamW), i dataset, i
 * checkpoint. Sono logiche che PRODUCONO/ALTERANO pesi → PUL.
 *
 * Dipendenza: PUL -> Apache (gmbllm linka l'engine), MAI l'inverso.
 * Vedi: GMBasic/Decisions/ADR-Confine-Licenza-Engine-Apache-Training-Quant-PUL.md
 * ============================================================================ */
#ifndef LLMENGINE_MLX_TRAIN_H
#define LLMENGINE_MLX_TRAIN_H

#include <stdbool.h>
#include <stdint.h>

#include <mlx/c/array.h>
#include <mlx/c/stream.h>

#include "llmengine_mlx.h"   /* llm_ai_mlx_model_handle */

#ifdef __cplusplus
extern "C" {
#endif

/* Prepara il motore per una passata con autograd (abilita i grafi differenziabili e
 * disattiva le cache/memoizzazioni non compatibili). Idempotente. */
int llm_ai_mlx_train_setup(llm_ai_mlx_model_handle mh);

/* Inietta i parametri LoRA A[in_dim,rank] / B[out_dim,rank] come ARRAY MLX nel grafo
 * del forward per (layer, proj). `extra` è riservato (passare NULL). Gli A/B restano
 * variabili del grafo: la loss restituita da forward_ce è differenziabile rispetto ad
 * essi (vedi mlx_value_and_grad lato PUL). */
int llm_ai_mlx_train_set_lora_mlx(llm_ai_mlx_model_handle mh, int layer, const char* proj,
                                  mlx_array A, mlx_array B, void* extra);

/* Esegue il forward su `T` token (`toks`), con target (`tgt`) e maschera opzionale
 * (`mask`, NULL = tutti attivi), e restituisce la cross-entropy come `mlx_array` nel
 * grafo. `sum_mode` = true somma la CE su tutti i token; false media. Se `ok` è non-NULL
 * viene posto a false in caso di errore (l'array è comunque valido/empty). L'array
 * ritornato è di proprietà del chiamante (`mlx_array_free`). */
mlx_array llm_ai_mlx_train_forward_ce(llm_ai_mlx_model_handle mh,
                                      const int32_t* toks, const int32_t* tgt,
                                      const uint8_t* mask, int T,
                                      bool sum_mode, bool* ok);

/* Attiva/disattiva l'attenzione "eager" (necessaria per differenziare la SDPA invece
 * del kernel fused, che non ha VJP in MLX-C). */
void llm_ai_mlx_train_set_eager_sdpa(int on);

/* Stream MLX associato al modello (usato per le closure/value_and_grad lato PUL). */
mlx_stream llm_ai_mlx_train_stream(llm_ai_mlx_model_handle mh);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MLX_TRAIN_H */
