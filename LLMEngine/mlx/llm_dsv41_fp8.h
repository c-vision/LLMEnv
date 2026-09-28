/* llm_dsv41_fp8.h — DeepSeek V4.1: packing attivazioni FP8 E4M3FN (group 32, scala ue8m0).
 *
 * Il pack/unpack e' FORMATO → vive nel CODEC (§0bis), come gli atomi e4m3:
 *   llmcodec_pack_fp8 / llmcodec_unpack_fp8 / llmcodec_pack_activation / llmcodec_unpack_activation  (llmcodec_fp8_fp4.h)
 * Questo header resta come include di comodo per il codice d'engine.
 *
 * Verifica: `llm_dsv41_fp8_test.c` vs il riferimento MLX (tools/gen_dsv41_fp8_reference.py). */
#ifndef LLM_DSV41_FP8_H
#define LLM_DSV41_FP8_H
#include "llmcodec_fp8_fp4.h"
#endif
