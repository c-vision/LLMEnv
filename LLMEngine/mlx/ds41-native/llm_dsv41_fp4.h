/* llm_dsv41_fp4.h — DeepSeek V4.1: packing attivazioni FP4 E2M1 (+ FP8 con scala e4m3).
 *
 * Formati (group_size 16/32, scala potenza-di-2 oppure e4m3): pack/unpack sono FORMATO → vivono
 * nel CODEC (§0bis): llmcodec_pack_activation / llmcodec_unpack_activation (llmcodec_fp8_fp4.h), come gli atomi
 * e2m1 (llmcodec_fp4_e2m1_code/level). Questo header resta come include di comodo.
 *
 * Verifica: `llm_dsv41_fp4_test.c` vs `llm_dsv41_fp4_ref.h`. */
#ifndef LLM_DSV41_FP4_H
#define LLM_DSV41_FP4_H
#include "llmcodec_fp8_fp4.h"
#endif
