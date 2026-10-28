/* llm_mlx_hc_fused.h — HYPER-CONNECTION qwen4 come 3 nodi lazy + 1 di combine (al posto di ~25 op MLX per hyper-connection).
 *
 *   HA hc_norm   : rmsnorm "zero-centered" per gruppo (hc gruppi da h): normed = x * rsqrt(mean(x^2)+eps) * (1+w)
 *   HB hc_down   : down = normed . Wd^T -> silu(down/hc) ; in un TG extra: injection = sigmoid((normed . Wi^T)/hc) * 2
 *   HC hc_upmix  : up = dsilu . Wu^T (4 righe per simdgroup: la stessa colonna dei hc gruppi) -> mixed[i] = mean_g sigmoid(up[g,i]) * normed[g,i]
 *   HD hc_combine: out[g,i] = residual[g,i] + sub[i] * injection[g]
 * Vincoli: f32, hc_count = 4 (4 righe per simdgroup in HC), up con colonne = righe di down (multiplo di 32). Fuori da qui il
 * chiamante usa il percorso a op. I pesi sono viste mmap zero-copy come per le altre proiezioni packed. */
#ifndef LLM_MLX_HC_FUSED_H
#define LLM_MLX_HC_FUSED_H

#include "mlx/c/mlx.h"
#include "llm_mlx_packed_lazy.h"
#include "llm_mlx_q4_msl.h"
#include <pthread.h>

static const char llm_hc_src_norm[] =
"uint tid = thread_position_in_threadgroup.x; uint t = threadgroup_position_in_grid.y;\n"
"uint hcn = HP[0], h = HP[1]; uint W = hcn * h;\n"
"threadgroup float part[4][32];\n"
"const device float* x = X + (ulong)t * W;\n"
"float sq[4] = { 0.0f, 0.0f, 0.0f, 0.0f };\n"
"for (uint i = tid; i < W; i += 1024) { float v = x[i]; sq[i / h] += v * v; }\n"
"uint lane = thread_index_in_simdgroup, sgid = simdgroup_index_in_threadgroup;\n"
"for (uint g = 0; g < 4; g++) { float s = simd_sum(sq[g]); if (lane == 0) part[g][sgid] = s; }\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"float rs[4];\n"
"for (uint g = 0; g < 4; g++) { float s = 0.0f; for (uint k = 0; k < 32; k++) s += part[g][k]; rs[g] = rsqrt(s / (float)h + EPS[0]); }\n"
"for (uint i = tid; i < W; i += 1024) NORMED[(ulong)t * W + i] = x[i] * rs[i / h] * (1.0f + HW[i]);\n";

static const char llm_hc_src_down[] =
"uint tg = threadgroup_position_in_grid.x; uint t = threadgroup_position_in_grid.y;\n"
"uint sg = simdgroup_index_in_threadgroup; uint lane = thread_index_in_simdgroup;\n"
"uint H = PD[0], M = PD[1]; float inv_hc = 1.0f / (float)HP[0];\n"
"const device float* x = NORMED + (ulong)t * H;\n"
"uint nd_tg = (M + 31) / 32;\n"
"if (tg < nd_tg) {\n"
"    uint r0 = (tg * 8 + sg) * 4; if (r0 >= M) return;\n"
"    uint nr = min(4u, M - r0); uint r1 = min(r0 + 1u, M - 1u), r2 = min(r0 + 2u, M - 1u), r3 = min(r0 + 3u, M - 1u);\n"
"    ulong lw = PD[3], ls = PD[4], lb = PD[5]; uint gs = PD[2], bf = PD[6];\n"
"    ulong rw = (ulong)(H / 2), sr = (ulong)(H / gs) * 2;\n"
"    const device uchar* wp = DW + lw; const device uchar* sp = DS + ls; const device uchar* bp = DB + lb;\n"
"    float o[4];\n"
"    q4_dotn(o, nr, wp + r0 * rw, wp + r1 * rw, wp + r2 * rw, wp + r3 * rw, sp + r0 * sr, sp + r1 * sr, sp + r2 * sr, sp + r3 * sr,\n"
"            bp + r0 * sr, bp + r1 * sr, bp + r2 * sr, bp + r3 * sr, (uint)(lw & 3), x, H, gs, bf, lane);\n"
"    if (lane == 0) for (uint i = 0; i < nr; i++) { float v = o[i] * inv_hc; DSILU[(ulong)t * M + r0 + i] = v / (1.0f + exp(-v)); }\n"
"} else if (sg == 0) {\n"
"    uint MI = PI[1]; ulong lw = PI[3], ls = PI[4], lb = PI[5]; uint gs = PI[2], bf = PI[6];\n"
"    ulong rw = (ulong)(H / 2), sr = (ulong)(H / gs) * 2;\n"
"    const device uchar* wp = IW + lw; const device uchar* sp = IS + ls; const device uchar* bp = IB + lb;\n"
"    uint r1 = min(1u, MI - 1u), r2 = min(2u, MI - 1u), r3 = min(3u, MI - 1u);\n"
"    float o[4];\n"
"    q4_dotn(o, min(4u, MI), wp, wp + r1 * rw, wp + r2 * rw, wp + r3 * rw, sp, sp + r1 * sr, sp + r2 * sr, sp + r3 * sr,\n"
"            bp, bp + r1 * sr, bp + r2 * sr, bp + r3 * sr, (uint)(lw & 3), x, H, gs, bf, lane);\n"
"    if (lane == 0) for (uint i = 0; i < min(4u, MI); i++) INJ[(ulong)t * 4 + i] = 2.0f / (1.0f + exp(-(o[i] * inv_hc)));\n"
"}\n";

static const char llm_hc_src_upmix[] =
"uint tg = threadgroup_position_in_grid.x; uint t = threadgroup_position_in_grid.y;\n"
"uint sg = simdgroup_index_in_threadgroup; uint lane = thread_index_in_simdgroup;\n"
"uint hcn = HP[0], h = HP[1]; uint W = hcn * h; uint H = PU[0], M = PU[1];\n"
"uint i = tg * 8 + sg; if (i >= h) return;\n"
"ulong lw = PU[3], ls = PU[4], lb = PU[5]; uint gs = PU[2], bf = PU[6];\n"
"ulong rw = (ulong)(H / 2), sr = (ulong)(H / gs) * 2;\n"
"const device uchar* wp = UW + lw; const device uchar* sp = US + ls; const device uchar* bp = UB + lb;\n"
"float o[4];\n"
"q4_dotn(o, 4u, wp + (ulong)i * rw, wp + (ulong)(i + h) * rw, wp + (ulong)(i + 2 * h) * rw, wp + (ulong)(i + 3 * h) * rw,\n"
"        sp + (ulong)i * sr, sp + (ulong)(i + h) * sr, sp + (ulong)(i + 2 * h) * sr, sp + (ulong)(i + 3 * h) * sr,\n"
"        bp + (ulong)i * sr, bp + (ulong)(i + h) * sr, bp + (ulong)(i + 2 * h) * sr, bp + (ulong)(i + 3 * h) * sr,\n"
"        (uint)(lw & 3), DSILU + (ulong)t * H, H, gs, bf, lane);\n"
"if (lane == 0) {\n"
"    const device float* nm = NORMED + (ulong)t * W;\n"
"    float acc = 0.0f;\n"
"    for (uint g = 0; g < 4; g++) acc += (1.0f / (1.0f + exp(-o[g]))) * nm[g * h + i];\n"
"    MIXED[(ulong)t * h + i] = acc * (1.0f / (float)hcn);\n"
"}\n";

static const char llm_hc_src_comb[] =
"uint e = thread_position_in_grid.x; uint t = thread_position_in_grid.y;\n"
"uint hcn = HP[0], h = HP[1]; uint W = hcn * h; if (e >= W) return;\n"
"OUT[(ulong)t * W + e] = RES[(ulong)t * W + e] + SUB[(ulong)t * h + (e % h)] * INJ[(ulong)t * 4 + (e / h)];\n";

static mlx_fast_metal_kernel g_llm_hc_k[4]; static int g_llm_hc_ready; static pthread_mutex_t g_llm_hc_mu = PTHREAD_MUTEX_INITIALIZER;

static inline void llm_hc_kernels_init(void) {
    if (g_llm_hc_ready) return;
    static const char* const i0[] = { "X", "HW", "HP", "EPS" };              static const char* const o0[] = { "NORMED" };
    static const char* const i1[] = { "NORMED", "DW","DS","DB","PD", "IW","IS","IB","PI", "HP" }; static const char* const o1[] = { "DSILU", "INJ" };
    static const char* const i2[] = { "NORMED", "DSILU", "UW","US","UB","PU", "HP" };         static const char* const o2[] = { "MIXED" };
    static const char* const i3[] = { "SUB", "RES", "INJ", "HP" };           static const char* const o3[] = { "OUT" };
    const char* const* ins[4] = { i0, i1, i2, i3 }; const int nin[4] = { 4, 10, 7, 4 };
    const char* const* outs[4] = { o0, o1, o2, o3 }; const int nout[4] = { 1, 2, 1, 1 };
    const char* srcs[4] = { llm_hc_src_norm, llm_hc_src_down, llm_hc_src_upmix, llm_hc_src_comb };
    const char* names[4] = { "llm_hc_norm", "llm_hc_down", "llm_hc_upmix", "llm_hc_combine" };
    for (int k = 0; k < 4; k++) {
        mlx_vector_string vi = mlx_vector_string_new(), vo = mlx_vector_string_new();
        for (int i = 0; i < nin[k]; i++) mlx_vector_string_append_value(vi, ins[k][i]);
        for (int i = 0; i < nout[k]; i++) mlx_vector_string_append_value(vo, outs[k][i]);
        g_llm_hc_k[k] = mlx_fast_metal_kernel_new(names[k], vi, vo, srcs[k], llm_q4_msl_helpers, true, false);
        mlx_vector_string_free(vi); mlx_vector_string_free(vo);
    }
    g_llm_hc_ready = 1;
}

static inline int llm_hc_apply(int k, const mlx_array* in, int nin, const int (*shape)[2], const int* ndim, const mlx_dtype* dt, int nout,
                               int g0, int g1, int t0, mlx_array* outs, mlx_stream s) {
    mlx_vector_array vi = mlx_vector_array_new();
    for (int i = 0; i < nin; i++) mlx_vector_array_append_value(vi, in[i]);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    for (int o = 0; o < nout; o++) mlx_fast_metal_kernel_config_add_output_arg(cfg, shape[o], (size_t)ndim[o], dt[o]);
    mlx_fast_metal_kernel_config_set_grid(cfg, g0, g1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, t0, 1, 1);
    mlx_vector_array vo = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&vo, g_llm_hc_k[k], vi, cfg, s);
    if (rc == 0) for (int o = 0; o < nout; o++) mlx_vector_array_get(&outs[o], vo, (size_t)o);
    mlx_vector_array_free(vo); mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(vi);
    return rc;
}

/* Lato "mix" della hyper-connection. x: [nt, hcn*h] f32. wnorm1: pesi hc_norm gia' (f32), HW = 1+w si applica nel kernel (qui si passa w).
 * dz/up/inj: lazy dei tre tensori. Uscite: mixed [nt,h], injection [nt,4]. 0 = ok. */
static inline int llm_hc_mix_fused(mlx_array* mixed, mlx_array* inj_out, mlx_array x2, int nt, int hcn, int h, mlx_array hw32, float eps,
                                   const LlmMlxPackedLazy* dz, const LlmMlxPackedW* wd,
                                   const LlmMlxPackedLazy* up, const LlmMlxPackedW* wu,
                                   const LlmMlxPackedLazy* ij, const LlmMlxPackedW* wi, mlx_stream s) {
    if (hcn != 4 || (int)wd->cols != hcn * h || (int)wu->rows != hcn * h || (int)wu->cols != (int)wd->rows || (int)wi->cols != hcn * h ||
        (int)wi->rows != 4 || (int)wd->rows % 32 != 0 || h > 65535) return -1;
    pthread_mutex_lock(&g_llm_hc_mu); llm_hc_kernels_init(); pthread_mutex_unlock(&g_llm_hc_mu);
    int32_t hp[4] = { hcn, h, 0, 0 }; int hps[1] = { 4 }; float ep[1] = { eps }; int eps_[1] = { 1 };
    mlx_array HP = mlx_array_new_data(hp, hps, 1, MLX_INT32), EP = mlx_array_new_data(ep, eps_, 1, MLX_FLOAT32);
    mlx_array normed = mlx_array_new(), dsilu = mlx_array_new(), inj = mlx_array_new(), mx = mlx_array_new(), tmp[2];
    tmp[0] = mlx_array_new(); tmp[1] = mlx_array_new();
    int rc = -1; const int W = hcn * h, MD = (int)wd->rows;
    int sh[2][2]; int nd[2]; mlx_dtype dt[2];
    { mlx_array in[4] = { x2, hw32, HP, EP }; sh[0][0] = nt; sh[0][1] = W; nd[0] = 2; dt[0] = MLX_FLOAT32;
      if (llm_hc_apply(0, in, 4, sh, nd, dt, 1, 1024, nt, 1024, &normed, s) != 0) goto done; }
    { mlx_array in[10] = { normed, dz->w, dz->s, dz->b, dz->p, ij->w, ij->s, ij->b, ij->p, HP };
      sh[0][0] = nt; sh[0][1] = MD; nd[0] = 2; dt[0] = MLX_FLOAT32; sh[1][0] = nt; sh[1][1] = 4; nd[1] = 2; dt[1] = MLX_FLOAT32;
      if (llm_hc_apply(1, in, 10, sh, nd, dt, 2, (((MD + 31) / 32) + 1) * 256, nt, 256, tmp, s) != 0) goto done;
      mlx_array_set(&dsilu, tmp[0]); mlx_array_set(&inj, tmp[1]); }
    { mlx_array in[7] = { normed, dsilu, up->w, up->s, up->b, up->p, HP };
      sh[0][0] = nt; sh[0][1] = h; nd[0] = 2; dt[0] = MLX_FLOAT32;
      if (llm_hc_apply(2, in, 7, sh, nd, dt, 1, ((h + 7) / 8) * 256, nt, 256, &mx, s) != 0) goto done; }
    mlx_array_set(mixed, mx); mlx_array_set(inj_out, inj);
    rc = 0;
done:
    mlx_array_free(HP); mlx_array_free(EP); mlx_array_free(normed); mlx_array_free(dsilu); mlx_array_free(inj); mlx_array_free(mx);
    mlx_array_free(tmp[0]); mlx_array_free(tmp[1]);
    return rc;
}

/* out = residual + sub (x) injection. sub: [nt*h] (f32), residual: [nt*hcn*h], injection: [nt*4]. 0 = ok. */
static inline int llm_hc_combine_fused(mlx_array* out, mlx_array sub, mlx_array residual, mlx_array injection, int nt, int hcn, int h, mlx_stream s) {
    if (hcn != 4) return -1;
    pthread_mutex_lock(&g_llm_hc_mu); llm_hc_kernels_init(); pthread_mutex_unlock(&g_llm_hc_mu);
    int32_t hp[4] = { hcn, h, 0, 0 }; int hps[1] = { 4 };
    mlx_array HP = mlx_array_new_data(hp, hps, 1, MLX_INT32), o = mlx_array_new();
    const int W = hcn * h; int sh[1][2] = { { nt, W } }; int nd[1] = { 2 }; mlx_dtype dt[1] = { MLX_FLOAT32 };
    mlx_array in[4] = { sub, residual, injection, HP };
    int rc = llm_hc_apply(3, in, 4, sh, nd, dt, 1, ((W + 255) / 256) * 256, nt, 256, &o, s);
    if (rc == 0) rc = mlx_array_set(out, o);
    mlx_array_free(HP); mlx_array_free(o);
    return rc;
}

#endif
