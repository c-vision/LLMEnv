/* qwen4_native_gpu.h — gemv quantizzato su Metal (kernel ds4) per il path qwen4.
 * Zero-copy sul mmap del GGUF; tipi supportati dal backend: 8=Q8_0, 10=Q2_K, 16=IQ2_XXS.
 * Nessun buffer grande: out/attivazioni f32 piccoli. */
#ifndef LLM_QWEN4_NATIVE_GPU_H
#define LLM_QWEN4_NATIVE_GPU_H

#include "ds4/llm_dsv4_metal.h"
#include "llm_ggml.h"
#include <Accelerate/Accelerate.h>

typedef struct { LlmDsv4Metal* b; } Q4nGpu;

static int q4n_gpu_init(Q4nGpu* g) {
    if (!g) return -1;
    g->b = llm_dsv4_metal_create();
    return g->b ? 0 : -1;
}
static void q4n_gpu_reset(Q4nGpu* g) { (void)g; }   /* le view sono cache interne ds4 */
static void q4n_gpu_free(Q4nGpu* g) {
    if (!g) return;
    llm_dsv4_metal_destroy(g->b);
    memset(g, 0, sizeof *g);
}

/* esegue out_dim dot (tipo `type`) da `w` (mmap) con x f32. 0 ok. */
static int q4n_gpu_gemv(Q4nGpu* g, const void* w, int type, size_t row_bytes,
                        uint32_t n, uint32_t out_dim, const float* x, float* out) {
    if (!g || !g->b) return -1;
    LlmDsv4GpuJob job = { .w = w, .byte_off = 0, .ggml_type = type, .row_bytes = row_bytes,
                          .n = n, .out_dim = out_dim, .x = x, .out = out };
    return llm_dsv4_metal_gemv_batch(g->b, &job, 1);
}

/* gate+up+swiglu fusi (Q8_0) per n esperti in una cb. */
static int q4n_gpu_pair_swiglu_q8(Q4nGpu* g, const void* const* gates, const void* const* ups,
                                  const float* x, float* const* mids, int n,
                                  uint32_t ne0, uint32_t ne00, uint32_t rb_g, uint32_t rb_u) {
    if (!g || !g->b) return -1;
    return llm_dsv4_metal_pair_swiglu_q8_wide_batch(g->b, gates, ups, x, mids, n, 0.0f, ne00, rb_g, rb_u, ne0);
}

static int q4n_gpu_gemv_q8_wide_batch(Q4nGpu* g, const LlmDsv4WideJob* jobs, int n) {
    if (!g || !g->b) return -1;
    return llm_dsv4_metal_gemv_q8_wide_batch(g->b, jobs, n);
}

/* batch di job (una command buffer, un wait). */
static int q4n_gpu_gemv_batch(Q4nGpu* g, const LlmDsv4GpuJob* jobs, int n) {
    if (!g || !g->b) return -1;
    return llm_dsv4_metal_gemv_batch(g->b, jobs, n);
}

/* Matrice peso quantizzata (ptr nel mmap) o f32 (type=-1). */
typedef struct { const void* w; int type; size_t row_bytes; } Q4nMat;
typedef int (*Q4nMatMvFn)(void* ctx, const Q4nMat* W, const float* x, float* y, int n_in, int n_out);

static int q4n_mat_mv_cpu(void* ctx, const Q4nMat* W, const float* x, float* y, int n_in, int n_out) {
    (void)ctx;
    if (!W || !W->w || !x || !y || n_in <= 0 || n_out <= 0) return -1;
    const float* w = (const float*)W->w;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, w, n_in, x, 1, 0.0f, y, 1);
    return 0;
}
static int q4n_mat_mv_gpu(void* ctx, const Q4nMat* W, const float* x, float* y, int n_in, int n_out) {
    Q4nGpu* g = (Q4nGpu*)ctx;
    if (!W || !W->w) return -1;
    if (W->type < 0) return q4n_mat_mv_cpu(NULL, W, x, y, n_in, n_out);
    if (!g || !g->b) return -1;
    return q4n_gpu_gemv(g, W->w, W->type, W->row_bytes, (uint32_t)n_in, (uint32_t)n_out, x, y);
}

#endif
