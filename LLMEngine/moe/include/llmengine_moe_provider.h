/* llmengine_moe_provider.h — registro multi-modello del provider MoE (M14).
 *
 * Tiene PIU' modelli in memoria contemporaneamente, ognuno con la propria config (SWAP/RESIDENT,
 * cache, prefetch, device). Default = SWAP (streaming out-of-core); RESIDENT = precaricato in RAM
 * senza swap. E' la base del provider AI "embedded" lato BASIC (un modello per nome). */
#ifndef LLMENGINE_MOE_PROVIDER_H
#define LLMENGINE_MOE_PROVIDER_H

#include "llmengine_moe.h"

typedef struct LlmEngineMoeProvider LlmEngineMoeProvider;

LlmEngineMoeProvider* llmengine_moe_provider_create(void);
void llmengine_moe_provider_destroy(LlmEngineMoeProvider* p);

/* Carica (o sostituisce, se `name` esiste gia') un modello `GGUF`. */
LlmEngineMoeStatus llmengine_moe_provider_load(LlmEngineMoeProvider* p, const char* name, const char* path,
                                   const LlmEngineMoeRuntimeConfig* cfg);
LlmEngineMoeStatus llmengine_moe_provider_unload(LlmEngineMoeProvider* p, const char* name);

/* Runtime del modello (NULL se assente). */
LlmEngineMoeRuntime* llmengine_moe_provider_get(LlmEngineMoeProvider* p, const char* name);
LlmEngineMoeStatus llmengine_moe_provider_stats(LlmEngineMoeProvider* p, const char* name, LlmEngineMoeStreamStats* out);

/* M21 — Scheduler di residenza (registry condiviso, UNA implementazione per tutti i backend).
 * Il chiamante inietta la memoria disponibile (byte) PRIMA di load/promote: il provider, se serve
 * e la memoria non basta, DEMOTA i modelli ACTIVE meno usati (LRU) e poi li EVICTA, finché il
 * fabbisogno è coperto. 0 = nessuna informazione (scheduler disattivo). */
void llmengine_moe_provider_set_mem_available(LlmEngineMoeProvider* p, uint64_t bytes);
void llmengine_moe_provider_mark_used(LlmEngineMoeProvider* p, const char* name);   /* touch LRU */
/* M21 orchestrazione: priorità del modello (più alto = tenuto di più; evict dal più basso). */
void llmengine_moe_provider_set_priority(LlmEngineMoeProvider* p, const char* name, int priority);
LlmEngineMoeStatus llmengine_moe_provider_promote(LlmEngineMoeProvider* p, const char* name);  /* ACTIVE, con scheduler */
LlmEngineMoeStatus llmengine_moe_provider_demote(LlmEngineMoeProvider* p, const char* name);   /* WARM */

/* Riconfigura a caldo un modello caricato (applica device/cache/prefetch/async). */
LlmEngineMoeStatus llmengine_moe_provider_reconfigure(LlmEngineMoeProvider* p, const char* name,
                                          const LlmEngineMoeRuntimeConfig* cfg);

uint32_t llmengine_moe_provider_count(const LlmEngineMoeProvider* p);
const char* llmengine_moe_provider_name_at(const LlmEngineMoeProvider* p, uint32_t i);

#endif /* LLMENGINE_MOE_PROVIDER_H */
