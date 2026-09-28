/* llmengine_moe_backend_metal.h — backend Metal per il compute degli esperti (M8, spec §38).
 *
 * Nativo C/ObjC++ (nessun C#). Sola GPU: upload del blob esperto in un buffer device,
 * `expert_compute` = matvec `out[j] = Σ_i blob[j*K+i] * in[i]` (l'esperto e' una matrice M×K),
 * release. Il pool di slot (M8) decide QUALE esperto sta in QUALE slot; il backend esegue. */
#ifndef LLMENGINE_MOE_BACKEND_METAL_H
#define LLMENGINE_MOE_BACKEND_METAL_H

#include "llmengine_moe_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmEngineMoeMetalBackend LlmEngineMoeMetalBackend;

/* 1 se esiste un device Metal (altrimenti il runtime usa il CPU path). */
int llmengine_moe_metal_available(void);

/* Crea il backend (compila il kernel). `NULL`/`sorgente vuota` => kernel built-in. */
LlmEngineMoeStatus llmengine_moe_metal_create(LlmEngineMoeMetalBackend** out, const char* kernel_src);
void llmengine_moe_metal_destroy(LlmEngineMoeMetalBackend* b);

/* Copia `blob` in un buffer device; `addr_out` e' l'handle da passare a compute/release. */
LlmEngineMoeStatus llmengine_moe_metal_upload(LlmEngineMoeMetalBackend* b, const void* blob, size_t size, void** addr_out);
void llmengine_moe_metal_release(LlmEngineMoeMetalBackend* b, void* addr);

/* out[j] = Σ_i blob[j*in_size+i] * in[i], j in [0,out_size). Blob = out_size*in_size float32. */
LlmEngineMoeStatus llmengine_moe_metal_expert_compute(LlmEngineMoeMetalBackend* b, void* addr, size_t blob_bytes,
                                          const float* in, uint32_t in_size,
                                          float* out, uint32_t out_size);

/* Attende il completamento di tutto il lavoro GPU accodato. */
LlmEngineMoeStatus llmengine_moe_metal_sync(LlmEngineMoeMetalBackend* b);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_BACKEND_METAL_H */
