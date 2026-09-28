/* llmengine_moe_adapter_gguf.h — adattatore per modelli MoE in formato GGUF (ingest diretto). */
#ifndef LLMENGINE_MOE_ADAPTER_GGUF_H
#define LLMENGINE_MOE_ADAPTER_GGUF_H

#include "llmengine_moe_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Vtable dell'adattatore GGUF (Qwen3-MoE in GGUF): legge i dense dal file GGUF (dequant a f32 al
 * load, con trasposizione [in,out]→[out,in] dove serve) e gli esperti dequant-on-read per
 * (layer, expert). Nessuna conversione offline. Primo slice: F16/F32/Q8_0/Q4_0. */
LlmEngineMoeModelAdapter llmengine_moe_gguf_adapter(void);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_ADAPTER_GGUF_H */
