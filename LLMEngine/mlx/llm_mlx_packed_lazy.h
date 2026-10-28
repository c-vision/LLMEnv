/* llm_mlx_packed_lazy.h — proiezioni Q4 affine su pesi PACKED come NODI LAZY del grafo MLX.
 *
 * I pesi restano nel mmap delle safetensors: ogni componente (weight/scales/biases) e' avvolto in un mlx_array uint8 che usa
 * direttamente la memoria (MLX crea un buffer Metal zero-copy su memoria host page-aligned). Il gemv e' un mlx_fast_metal_kernel:
 * nessun eval, nessun command buffer dedicato, nessuna attesa per proiezione — il token intero resta un unico grafo valutato da MLX.
 * Il kernel legge scale/bias a byte (gli offset nelle safetensors non sono allineati a 2) e accetta scale bf16/half. */
#ifndef LLM_MLX_PACKED_LAZY_H
#define LLM_MLX_PACKED_LAZY_H

#include "mlx/c/mlx.h"
#include "llm_mlx_wpacked.h"
#include "llm_mlx_q4_msl.h"
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int ready;
    mlx_array w, s, b, p;   /* viste uint8 zero-copy + parametri int32 [H,M,group,lead_w,lead_s,lead_b,bf16] */
} LlmMlxPackedLazy;

static void llm_mlx_lazy_noop(void* p) { (void)p; }

/* Avvolge [p, p+len) in un mlx_array uint8 sulla pagina che lo contiene (zero-copy). lead = offset di p nella pagina. */
static inline mlx_array llm_mlx_lazy_view(const void* p, size_t len, int* lead) {
    const size_t page = (size_t)getpagesize();
    const uintptr_t base = (uintptr_t)p & ~(uintptr_t)(page - 1);
    const size_t ld = (size_t)((uintptr_t)p - base), tot = ((ld + len + page - 1) / page) * page;
    *lead = (int)ld;
    int shape[1] = { (int)tot };
    return mlx_array_new_data_managed((void*)base, shape, 1, MLX_UINT8, llm_mlx_lazy_noop);
}

static inline int llm_mlx_lazy_build(LlmMlxPackedLazy* z, const LlmMlxPackedW* w) {
    if (z->ready) return 0;
    if (!w || w->bits != 4 || !w->biases) return -1;
    const size_t wb = (size_t)w->rows * (w->cols / 2), sb = (size_t)w->rows * (w->cols / (uint32_t)w->group) * 2;
    int lw = 0, ls = 0, lb = 0;
    z->w = llm_mlx_lazy_view(w->w, wb, &lw);
    z->s = llm_mlx_lazy_view(w->scales, sb, &ls);
    z->b = llm_mlx_lazy_view(w->biases, sb, &lb);
    int32_t pv[8] = { (int32_t)w->cols, (int32_t)w->rows, w->group, lw, ls, lb, w->scale_bf16, 0 };
    int psh[1] = { 8 };
    z->p = mlx_array_new_data(pv, psh, 1, MLX_INT32);
    z->ready = 1;
    return 0;
}

static inline void llm_mlx_lazy_free(LlmMlxPackedLazy* z) {
    if (!z->ready) return;
    mlx_array_free(z->w); mlx_array_free(z->s); mlx_array_free(z->b); mlx_array_free(z->p);
    z->ready = 0;
}

static const char llm_mlx_lazy_gemv_src[] =
"uint tg = threadgroup_position_in_grid.x; uint t = threadgroup_position_in_grid.y;\n"
"uint sg = simdgroup_index_in_threadgroup; uint lane = thread_index_in_simdgroup;\n"
"uint H = P[0], M = P[1], gs = P[2]; ulong lw = P[3], ls = P[4], lb = P[5]; uint bf = P[6];\n"
"uint r0 = (tg * 8 + sg) * 4;\n"
"if (r0 >= M) return;\n"
"uint nr = min(4u, M - r0);\n"
"uint r1 = min(r0 + 1u, M - 1u), r2 = min(r0 + 2u, M - 1u), r3 = min(r0 + 3u, M - 1u);\n"
"ulong rw = (ulong)(H / 2), sr = (ulong)(H / gs) * 2;\n"
"const device uchar* wp = W + lw; const device uchar* sp = S + ls; const device uchar* bp = B + lb;\n"
"float o[4];\n"
"q4_dotn(o, nr, wp + r0 * rw, wp + r1 * rw, wp + r2 * rw, wp + r3 * rw, sp + r0 * sr, sp + r1 * sr, sp + r2 * sr, sp + r3 * sr,\n"
"        bp + r0 * sr, bp + r1 * sr, bp + r2 * sr, bp + r3 * sr, (uint)(lw & 3), X + (ulong)t * H, H, gs, bf, lane);\n"
"if (lane == 0) for (uint i = 0; i < nr; i++) OUT[(ulong)t * M + r0 + i] = o[i];\n";

static mlx_fast_metal_kernel g_llm_mlx_lazy_gemv;
static int g_llm_mlx_lazy_gemv_ready;
static pthread_mutex_t g_llm_mlx_lazy_mu = PTHREAD_MUTEX_INITIALIZER;

static inline mlx_fast_metal_kernel llm_mlx_lazy_kernel(const char* name, const char* const* ins, int nin, const char* const* outs, int nout,
                                                         const char* src) {
    mlx_vector_string vi = mlx_vector_string_new(), vo = mlx_vector_string_new();
    for (int i = 0; i < nin; i++) mlx_vector_string_append_value(vi, ins[i]);
    for (int i = 0; i < nout; i++) mlx_vector_string_append_value(vo, outs[i]);
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new(name, vi, vo, src, llm_q4_msl_helpers, true, false);
    mlx_vector_string_free(vi); mlx_vector_string_free(vo);
    return k;
}

/* y = x . W^T per le righe di x. x: [..., H] (qualsiasi dtype); out: [..., M] float32 (lazy). 0 = ok. */
static inline int llm_mlx_lazy_linear(const LlmMlxPackedLazy* z, const LlmMlxPackedW* w, mlx_array x, mlx_array* out, mlx_stream s) {
    if (!z || !z->ready || !w || !out) return -1;
    pthread_mutex_lock(&g_llm_mlx_lazy_mu);
    if (!g_llm_mlx_lazy_gemv_ready) {
        static const char* const ins[] = { "W", "S", "B", "X", "P" }; static const char* const outs[] = { "OUT" };
        g_llm_mlx_lazy_gemv = llm_mlx_lazy_kernel("llm_packed_gemv_q4", ins, 5, outs, 1, llm_mlx_lazy_gemv_src);
        g_llm_mlx_lazy_gemv_ready = 1;
    }
    mlx_fast_metal_kernel k = g_llm_mlx_lazy_gemv;
    pthread_mutex_unlock(&g_llm_mlx_lazy_mu);

    const int nd = mlx_array_ndim(x);
    if (nd < 1 || (int)mlx_array_dim(x, nd - 1) != (int)w->cols) return -1;
    int nt = 1; for (int d = 0; d < nd - 1; d++) nt *= (int)mlx_array_dim(x, d);
    mlx_array x32 = mlx_array_new(), x2 = mlx_array_new();
    int rc = -1;
    int sh2[2] = { nt, (int)w->cols };
    if (mlx_astype(&x32, x, MLX_FLOAT32, s) != 0 || mlx_reshape(&x2, x32, sh2, 2, s) != 0) goto done;
    {
        mlx_vector_array ins = mlx_vector_array_new();
        mlx_vector_array_append_value(ins, z->w); mlx_vector_array_append_value(ins, z->s); mlx_vector_array_append_value(ins, z->b);
        mlx_vector_array_append_value(ins, x2); mlx_vector_array_append_value(ins, z->p);
        mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
        int osh[2] = { nt, (int)w->rows };
        mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
        mlx_fast_metal_kernel_config_set_grid(cfg, (((int)w->rows + 31) / 32) * 256, nt, 1);
        mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
        mlx_vector_array outs = mlx_vector_array_new();
        int ar = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
        if (ar == 0) {
            mlx_array y = mlx_array_new();
            mlx_vector_array_get(&y, outs, 0);
            int osh_n[8]; for (int d = 0; d < nd - 1; d++) osh_n[d] = (int)mlx_array_dim(x, d);
            osh_n[nd - 1] = (int)w->rows;
            rc = (nd == 2) ? mlx_array_set(out, y) : mlx_reshape(out, y, osh_n, nd, s);
            mlx_array_free(y);
        }
        mlx_vector_array_free(outs); mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(ins);
    }
done:
    mlx_array_free(x32); mlx_array_free(x2);
    return rc == 0 ? 0 : -1;
}

#endif
