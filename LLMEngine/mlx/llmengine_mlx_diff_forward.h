/* ============================================================================
 * llmengine_mlx_diff_forward.h — ABI Apache dei PRIMITIVI del forward DIFFERENZIABILE.
 *
 * PERIMETRO DI LICENZA (Apache-2.0). Contratto pubblico che l'engine MLX (Apache) espone a un
 * consumer PUL (gmbllm) per costruire loss/grad/DPO/AdamW SENZA reimplementare il forward.
 *
 * L'engine espone SOLO l'INFERENZA differenziabile: il forward (backbone+norm+lm_head) →
 * LOGITS, l'iniezione dei parametri LoRA (A/B) nel grafo, lo stream. NON espone la loss:
 * la cross-entropy (masked/DPO/...), `mlx_value_and_grad`, l'AdamW, i checkpoint e i dataset
 * sono PUL (`gmbllm/src/train/`) — PRODUCONO/ALTERANO pesi.
 * Dipendenza: PUL -> Apache (gmbllm linka l'engine), MAI l'inverso. ADR §4.
 * ============================================================================ */
#ifndef LLMENGINE_MLX_DIFF_FORWARD_H
#define LLMENGINE_MLX_DIFF_FORWARD_H

#include <stdbool.h>
#include <stdint.h>
#include <mlx/c/array.h>
#include <mlx/c/stream.h>
#include "llmengine_mlx.h"   /* llm_ai_mlx_model_handle, llm_ai_mlx_session_handle */

#ifdef __cplusplus
extern "C" {
#endif

/* Prepara il motore per una passata con autograd (grafi differenziabili, cache non compatibili
 * disattivate). Idempotente. */
int llm_ai_mlx_diff_setup(llm_ai_mlx_model_handle mh);

/* Inietta i parametri LoRA A[in,rank] / B[out,rank] come ARRAY MLX nel grafo per (layer, proj).
 * `extra` riservato (NULL). La loss costruita DAL PUL sopra i logits resta differenziabile
 * rispetto ad A/B. */
int llm_ai_mlx_diff_set_lora_mlx(llm_ai_mlx_model_handle mh, int layer, const char* proj,
                                 mlx_array A, mlx_array B, void* extra);

/* Forward DIFFERENZIABILE su `T` token → LOGITS [1,T,V] (applicando final_logit_softcap se
 * configurato). Se `sess_out` != NULL riceve la sessione (KV) creata, che il chiamante DEVE
 * distruggere dopo l'uso. `ok`=false su errore (l'array è comunque valido/empty, da free). */
mlx_array llm_ai_mlx_diff_forward_logits(llm_ai_mlx_model_handle mh, const int32_t* toks, int T,
                                         llm_ai_mlx_session_handle* sess_out, bool* ok);

/* Attenzione "eager"/composabile (no kernel Metal fused, che non ha VJP). */
void llm_ai_mlx_diff_set_eager_sdpa(int on);

/* Stream MLX del modello (per le closure/value_and_grad lato PUL). */
mlx_stream llm_ai_mlx_diff_stream(llm_ai_mlx_model_handle mh);

/* G6 — FULL-PARAMETER training: espone i pesi del modello come VETTORE di array MLX differenziabili
 * (modelli FP16/BF16, pesi PLAIN). `get` riempie `names[i]` (64 byte) + appende l'array i-esimo;
 * `set` rimette i pesi per NOME + invalida la wcache. Il PUL costruisce value_and_grad/optimizer. */
int llm_ai_mlx_diff_num_params(llm_ai_mlx_model_handle mh);
int llm_ai_mlx_diff_get_params(llm_ai_mlx_model_handle mh, mlx_vector_array* out, char names[][64], int cap);
int llm_ai_mlx_diff_set_params(llm_ai_mlx_model_handle mh, mlx_vector_array in, const char names[][64], int n);

#ifdef __cplusplus
}
#endif
#endif /* LLMENGINE_MLX_DIFF_FORWARD_H */
