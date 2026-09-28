/* llm_iq2_tables.h — le codebook grid iq2 sono UNICHE nel CODEC (§0bis). Alias ai simboli
 * canonici (definiti in llmcodec, plugin ggml). */
#ifndef LLM_IQ2_TABLES_H
#define LLM_IQ2_TABLES_H
#include "llmcodec_ggml_iq_grids.h"
#define llm_iq2_kmask  kmask_iq2xs
#define llm_iq2_ksigns ksigns_iq2xs
#define llm_iq2_grid   iq2xxs_grid
#endif
