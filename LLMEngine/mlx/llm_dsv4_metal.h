/* llm_dsv4_metal.h — backend Metal per il compute degli esperti DS4 (IQ2_XXS/Q2_K), glue
 * dell'engine. Zero-copy: aggancia il mmap del GGUF con "view" page-aligned (come ds4_metal.m),
 * niente copia dei 97 GB. Kernel: out[j] = Σ_k dequant(W_row_j,k) * x[k]  (attivazioni f32,
 * parità con llm_dsv4_dot_tensor). ObjC++ (Metal), ABI C. */
#ifndef LLM_DSV4_METAL_H
#define LLM_DSV4_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmDsv4Metal LlmDsv4Metal;

/* Descrittore di un tensore quantizzato (puntatore raw nella regione mappata + metadati). */
typedef struct { const uint8_t* w; int type; size_t row_bytes; } LlmDsv4GpuView;

/* 1 se esiste un device Metal. */
int llm_dsv4_metal_available(void);

LlmDsv4Metal* llm_dsv4_metal_create(void);
void llm_dsv4_metal_destroy(LlmDsv4Metal* b);

/* View zero-copy su `[file_off, file_off+bytes)` del mmap `mmap_base` (base page-aligned).
 * Ritorna l'handle (da passare a gemv/release) e scrive in *out_view_off l'offset del tensore
 * dentro la view. NULL su errore (mmap non allineato / device). */
void* llm_dsv4_metal_nocopy(LlmDsv4Metal* b, const void* mmap_base, size_t file_off, size_t bytes,
                            size_t* out_view_off);
/* Come nocopy ma da un puntatore arbitrario nella regione mappata (allinea a pagina da solo). */
void* llm_dsv4_metal_nocopy_ptr(LlmDsv4Metal* b, const void* ptr, size_t bytes, size_t* out_view_off);
void llm_dsv4_metal_release(LlmDsv4Metal* b, void* handle);

/* out[j] = dot(W_row_j, x), j in [0,out_dim). `ggml_type`: 16=IQ2_XXS, 10=Q2_K, 8=Q8_0. */
int llm_dsv4_metal_gemv(LlmDsv4Metal* b, void* handle, size_t view_off, int ggml_type,
                        size_t row_bytes, uint32_t n, uint32_t out_dim,
                        const float* x, float* out);

/* Un lavoro GEMV per il batch (view zero-copy transitoria: creata ed eliminata per job,
 * NESSUNA residenza persistente del mmap). */
typedef struct {
    const void* w; size_t byte_off; int ggml_type; size_t row_bytes;
    uint32_t n, out_dim; const float* x; float* out;
} LlmDsv4GpuJob;

/* Esegue `n_jobs` GEMV in una command buffer (commit+wait una volta). 0 = ok. */
int llm_dsv4_metal_gemv_batch(LlmDsv4Metal* b, const LlmDsv4GpuJob* jobs, int n_jobs);

/* MoE fuso ds4 (pair): gate+up in un pass + swiglu*peso -> mid[ne0]. Ritorna 0 = ok.
 * `gate`/`up` = inizio dell'esperto (dati packed); `x` = attivazione f32 [ne00]. */
int llm_dsv4_metal_pair_swiglu(LlmDsv4Metal* b, const void* gate, const void* up,
                               const float* x, float* mid, float route_w, float clamp,
                               uint32_t ne00, uint32_t nb01, uint32_t ne0);

/* Batch: `n` esperti in UNA command buffer + UN wait (gate/up/mid/route per esperto). */
int llm_dsv4_metal_pair_swiglu_batch(LlmDsv4Metal* b, const void* const* gates, const void* const* ups,
                                     const float* x, float* const* mids, const float* rws, int n,
                                     float clamp, uint32_t ne00, uint32_t nb01, uint32_t ne0);

#ifdef __cplusplus
}
#endif

#endif /* LLM_DSV4_METAL_H */
