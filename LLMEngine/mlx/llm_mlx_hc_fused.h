/* llm_mlx_hc_fused.h — HYPER-CONNECTION qwen4 in 2 nodi lazy (+1 di combine fuori dal percorso critico).
 *
 * Su GPU il costo e' la LATENZA della catena di kernel dipendenti (~30 us ciascuno), non il lavoro: per questo la catena e' stata
 * accorciata da norm -> down -> upmix (+combine) a:
 *   HB hc_down   : statistiche della norma "zero-centered" (rsqrt(mean(x^2)+eps) per gruppo, ricalcolate in ogni threadgroup) +
 *                  down = norm(x) . Wd^T -> silu(down/hc) ; in un TG extra injection = sigmoid((norm(x) . Wi^T)/hc) * 2 ;
 *                  il TG 0 scrive anche le statistiche rs[g]. x puo' essere ricostruito al volo come residual + sub*injP
 *                  (la combine del sotto-layer precedente): cosi' il kernel NON dipende dal nodo di combine, che resta in parallelo.
 *   HC hc_upmix  : up = dsilu . Wu^T (4 righe per simdgroup: la stessa colonna dei hc gruppi) -> mixed[i] = mean_g sigmoid(up[g,i]) * norm(x)[g,i]
 *   HD hc_combine: out[g,i] = residual[g,i] + sub[i] * injection[g]   (scrive il flusso residuo per i consumatori successivi)
 * Vincoli: f32, hc_count = 4, up con colonne = righe di down (multiplo di 32). Fuori da qui il chiamante usa il percorso a op. */
#ifndef LLM_MLX_HC_FUSED_H
#define LLM_MLX_HC_FUSED_H

#include "mlx/c/mlx.h"
#include "llm_mlx_packed_lazy.h"
#include "llm_mlx_q4_msl.h"
#include <pthread.h>

/* x effettivo del gruppo (32 colonne): ((base + pend*sub*injP[g]) * rs[g] * (1+w)) , 8 float4 in registri; xs = somma. */
static const char llm_hc_msl_helpers[] =
"inline float hc_xc(const device float* base, const device float* sub, thread const float* injp, uint pend, uint h, uint i) {\n"
"    float v = base[i]; if (pend) v += sub[i % h] * injp[i / h]; return v;\n"
"}\n"
"inline void q4_dotn_hc(thread float* out, uint nr,\n"
"                    const device uchar* rp0, const device uchar* rp1, const device uchar* rp2, const device uchar* rp3,\n"
"                    const device uchar* rs0, const device uchar* rs1, const device uchar* rs2, const device uchar* rs3,\n"
"                    const device uchar* rb0, const device uchar* rb1, const device uchar* rb2, const device uchar* rb3,\n"
"                    uint m, const device float* base, const device float* sub, thread const float* injp, uint pend,\n"
"                    const device float* hw, thread const float* rsg, uint h, uint H, uint gs, uint bf, uint lane) {\n"
"    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
"    const uint ngroups = H / 32;\n"
"    for (uint gi = lane; gi < ngroups; gi += 32) {\n"
"        const uint col0 = gi * 32u; const uint g = col0 / h; const uint ii = col0 - g * h;\n"
"        const device float4* b4 = (const device float4*)(base + col0); const device float4* w4 = (const device float4*)(hw + col0);\n"
"        const device float4* s4 = (const device float4*)(sub + ii);\n"
"        const float ij = pend ? injp[g] : 0.0f; const float r = rsg[g];\n"
"        float4 xv0 = (b4[0] + s4[0] * ij) * r * (1.0f + w4[0]), xv1 = (b4[1] + s4[1] * ij) * r * (1.0f + w4[1]);\n"
"        float4 xv2 = (b4[2] + s4[2] * ij) * r * (1.0f + w4[2]), xv3 = (b4[3] + s4[3] * ij) * r * (1.0f + w4[3]);\n"
"        float4 xv4 = (b4[4] + s4[4] * ij) * r * (1.0f + w4[4]), xv5 = (b4[5] + s4[5] * ij) * r * (1.0f + w4[5]);\n"
"        float4 xv6 = (b4[6] + s4[6] * ij) * r * (1.0f + w4[6]), xv7 = (b4[7] + s4[7] * ij) * r * (1.0f + w4[7]);\n"
"        const float xs = dot(xv0 + xv1 + xv2 + xv3 + xv4 + xv5 + xv6 + xv7, float4(1.0f));\n"
"        const uint sidx = (gi * 32u) / gs;\n"
"        acc0 += q4_row(rp0, rs0, rb0, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"        if (nr > 1) acc1 += q4_row(rp1, rs1, rb1, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"        if (nr > 2) acc2 += q4_row(rp2, rs2, rb2, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"        if (nr > 3) acc3 += q4_row(rp3, rs3, rb3, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"    }\n"
"    out[0] = simd_sum(acc0); out[1] = simd_sum(acc1); out[2] = simd_sum(acc2); out[3] = simd_sum(acc3);\n"
"}\n";

static const char llm_hc_src_down[] =
"uint tg = threadgroup_position_in_grid.x; uint t = threadgroup_position_in_grid.y;\n"
"uint tid = thread_position_in_threadgroup.x; uint sg = simdgroup_index_in_threadgroup; uint lane = thread_index_in_simdgroup;\n"
"uint hcn = HP[0], h = HP[1], pend = HP[2]; uint W = hcn * h;\n"
"uint H = PD[0], M = PD[1];\n"
"const device float* base = BASE + (ulong)t * W; const device float* sub = SUB + (ulong)t * h;\n"
"float injp[4]; for (uint g = 0; g < 4; g++) injp[g] = pend ? INJP[(ulong)t * 4 + g] : 0.0f;   // INJP (4 float) e' `constant` per MLX: copia in registri\n"
"threadgroup float part[4][8];\n"
"float sq[4] = { 0.0f, 0.0f, 0.0f, 0.0f };\n"
"for (uint i = tid; i < W; i += 256) { float v = hc_xc(base, sub, injp, pend, h, i); sq[i / h] += v * v; }\n"
"for (uint g = 0; g < 4; g++) { float s = simd_sum(sq[g]); if (lane == 0) part[g][sg] = s; }\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"float rsg[4];\n"
"for (uint g = 0; g < 4; g++) { float s = 0.0f; for (uint k = 0; k < 8; k++) s += part[g][k]; rsg[g] = rsqrt(s / (float)h + EPS[0]); }\n"
"if (tg == 0 && tid < 4) RSO[(ulong)t * 4 + tid] = rsg[tid];\n"
"float inv_hc = 1.0f / (float)hcn;\n"
"uint nd_tg = (M + 31) / 32;\n"
"if (tg < nd_tg) {\n"
"    uint r0 = (tg * 8 + sg) * 4; if (r0 >= M) return;\n"
"    uint nr = min(4u, M - r0); uint r1 = min(r0 + 1u, M - 1u), r2 = min(r0 + 2u, M - 1u), r3 = min(r0 + 3u, M - 1u);\n"
"    ulong lw = PD[3], ls = PD[4], lb = PD[5]; uint gs = PD[2], bf = PD[6];\n"
"    ulong rw = (ulong)(H / 2), sr = (ulong)(H / gs) * 2;\n"
"    const device uchar* wp = DW + lw; const device uchar* sp = DS + ls; const device uchar* bp = DB + lb;\n"
"    float o[4];\n"
"    q4_dotn_hc(o, nr, wp + r0 * rw, wp + r1 * rw, wp + r2 * rw, wp + r3 * rw, sp + r0 * sr, sp + r1 * sr, sp + r2 * sr, sp + r3 * sr,\n"
"               bp + r0 * sr, bp + r1 * sr, bp + r2 * sr, bp + r3 * sr, (uint)(lw & 3), base, sub, injp, pend, HW, rsg, h, H, gs, bf, lane);\n"
"    if (lane == 0) for (uint i = 0; i < nr; i++) { float v = o[i] * inv_hc; DSILU[(ulong)t * M + r0 + i] = v / (1.0f + exp(-v)); }\n"
"} else if (sg == 0) {\n"
"    uint MI = PI[1]; ulong lw = PI[3], ls = PI[4], lb = PI[5]; uint gs = PI[2], bf = PI[6];\n"
"    ulong rw = (ulong)(H / 2), sr = (ulong)(H / gs) * 2;\n"
"    const device uchar* wp = IW + lw; const device uchar* sp = IS + ls; const device uchar* bp = IB + lb;\n"
"    uint r1 = min(1u, MI - 1u), r2 = min(2u, MI - 1u), r3 = min(3u, MI - 1u);\n"
"    float o[4];\n"
"    q4_dotn_hc(o, min(4u, MI), wp, wp + r1 * rw, wp + r2 * rw, wp + r3 * rw, sp, sp + r1 * sr, sp + r2 * sr, sp + r3 * sr,\n"
"               bp, bp + r1 * sr, bp + r2 * sr, bp + r3 * sr, (uint)(lw & 3), base, sub, injp, pend, HW, rsg, h, H, gs, bf, lane);\n"
"    if (lane == 0) for (uint i = 0; i < min(4u, MI); i++) INJ[(ulong)t * 4 + i] = 2.0f / (1.0f + exp(-(o[i] * inv_hc)));\n"
"}\n";

static const char llm_hc_src_upmix[] =
"uint tg = threadgroup_position_in_grid.x; uint t = threadgroup_position_in_grid.y;\n"
"uint sg = simdgroup_index_in_threadgroup; uint lane = thread_index_in_simdgroup;\n"
"uint hcn = HP[0], h = HP[1], pend = HP[2]; uint W = hcn * h; uint H = PU[0];\n"
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
"    const device float* base = BASE + (ulong)t * W; const device float* sub = SUB + (ulong)t * h;\n"
"    float acc = 0.0f;\n"
"    for (uint g = 0; g < 4; g++) {\n"
"        float xc = base[g * h + i]; if (pend) xc += sub[i] * INJP[(ulong)t * 4 + g];\n"
"        acc += (1.0f / (1.0f + exp(-o[g]))) * (xc * RSI[(ulong)t * 4 + g] * (1.0f + HW[g * h + i]));\n"
"    }\n"
"    MIXED[(ulong)t * h + i] = acc * (1.0f / (float)hcn);\n"
"}\n";

static const char llm_hc_src_comb[] =
"uint e = thread_position_in_grid.x; uint t = thread_position_in_grid.y;\n"
"uint hcn = HP[0], h = HP[1]; uint W = hcn * h; if (e >= W) return;\n"
"OUT[(ulong)t * W + e] = RES[(ulong)t * W + e] + SUB[(ulong)t * h + (e % h)] * INJ[(ulong)t * 4 + (e / h)];\n";

static mlx_fast_metal_kernel g_llm_hc_k[3]; static int g_llm_hc_ready; static pthread_mutex_t g_llm_hc_mu = PTHREAD_MUTEX_INITIALIZER;
static char g_llm_hc_hdr[16384];

static inline void llm_hc_kernels_init(void) {
    if (g_llm_hc_ready) return;
    snprintf(g_llm_hc_hdr, sizeof g_llm_hc_hdr, "%s%s", llm_q4_msl_helpers, llm_hc_msl_helpers);
    static const char* const i0[] = { "BASE", "SUB", "INJP", "HW", "HP", "EPS", "DW","DS","DB","PD", "IW","IS","IB","PI" };
    static const char* const o0[] = { "DSILU", "INJ", "RSO" };
    static const char* const i1[] = { "BASE", "SUB", "INJP", "HW", "RSI", "DSILU", "UW","US","UB","PU", "HP" }; static const char* const o1[] = { "MIXED" };
    static const char* const i2[] = { "SUB", "RES", "INJ", "HP" };           static const char* const o2[] = { "OUT" };
    const char* const* ins[3] = { i0, i1, i2 }; const int nin[3] = { 14, 11, 4 };
    const char* const* outs[3] = { o0, o1, o2 }; const int nout[3] = { 3, 1, 1 };
    const char* srcs[3] = { llm_hc_src_down, llm_hc_src_upmix, llm_hc_src_comb };
    const char* names[3] = { "llm_hc_down2", "llm_hc_upmix2", "llm_hc_combine" };
    for (int k = 0; k < 3; k++) {
        mlx_vector_string vi = mlx_vector_string_new(), vo = mlx_vector_string_new();
        for (int i = 0; i < nin[k]; i++) mlx_vector_string_append_value(vi, ins[k][i]);
        for (int i = 0; i < nout[k]; i++) mlx_vector_string_append_value(vo, outs[k][i]);
        g_llm_hc_k[k] = mlx_fast_metal_kernel_new(names[k], vi, vo, srcs[k], g_llm_hc_hdr, true, false);
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

/* Lato "mix". base: [nt, hcn*h] (flusso residuo, oppure il residuo della combine precedente se pend); sub [nt,h] e injp [nt,4] sono gli
 * ingressi di quella combine (usati solo se pend=1: x = base + sub*injp). hw32 = pesi hc_norm in f32 (kernel usa 1+w). 0 = ok. */
static inline int llm_hc_mix_fused(mlx_array* mixed, mlx_array* inj_out, mlx_array base, mlx_array sub, mlx_array injp, int pend,
                                   int nt, int hcn, int h, mlx_array hw32, float eps,
                                   const LlmMlxPackedLazy* dz, const LlmMlxPackedW* wd,
                                   const LlmMlxPackedLazy* up, const LlmMlxPackedW* wu,
                                   const LlmMlxPackedLazy* ij, const LlmMlxPackedW* wi, mlx_stream s) {
    if (hcn != 4 || (int)wd->cols != hcn * h || (int)wu->rows != hcn * h || (int)wu->cols != (int)wd->rows || (int)wi->cols != hcn * h ||
        (int)wi->rows != 4 || (int)wd->rows % 32 != 0 || h > 65535) return -1;
    pthread_mutex_lock(&g_llm_hc_mu); llm_hc_kernels_init(); pthread_mutex_unlock(&g_llm_hc_mu);
    int32_t hp[4] = { hcn, h, pend ? 1 : 0, 0 }; int hps[1] = { 4 }; float ep[1] = { eps }; int eps_[1] = { 1 };
    mlx_array HP = mlx_array_new_data(hp, hps, 1, MLX_INT32), EP = mlx_array_new_data(ep, eps_, 1, MLX_FLOAT32);
    mlx_array dsilu = mlx_array_new(), inj = mlx_array_new(), rso = mlx_array_new(), mx = mlx_array_new(), tmp[3];
    tmp[0] = mlx_array_new(); tmp[1] = mlx_array_new(); tmp[2] = mlx_array_new();
    int rc = -1; const int MD = (int)wd->rows;
    int sh[3][2]; int nd[3]; mlx_dtype dt[3];
    { mlx_array in[14] = { base, sub, injp, hw32, HP, EP, dz->w, dz->s, dz->b, dz->p, ij->w, ij->s, ij->b, ij->p };
      sh[0][0] = nt; sh[0][1] = MD; nd[0] = 2; dt[0] = MLX_FLOAT32; sh[1][0] = nt; sh[1][1] = 4; nd[1] = 2; dt[1] = MLX_FLOAT32;
      sh[2][0] = nt; sh[2][1] = 4; nd[2] = 2; dt[2] = MLX_FLOAT32;
      if (llm_hc_apply(0, in, 14, sh, nd, dt, 3, (((MD + 31) / 32) + 1) * 256, nt, 256, tmp, s) != 0) goto done;
      mlx_array_set(&dsilu, tmp[0]); mlx_array_set(&inj, tmp[1]); mlx_array_set(&rso, tmp[2]); }
    { mlx_array in[11] = { base, sub, injp, hw32, rso, dsilu, up->w, up->s, up->b, up->p, HP };
      sh[0][0] = nt; sh[0][1] = h; nd[0] = 2; dt[0] = MLX_FLOAT32;
      if (llm_hc_apply(1, in, 11, sh, nd, dt, 1, ((h + 7) / 8) * 256, nt, 256, &mx, s) != 0) goto done; }
    mlx_array_set(mixed, mx); mlx_array_set(inj_out, inj);
    rc = 0;
done:
    mlx_array_free(HP); mlx_array_free(EP); mlx_array_free(dsilu); mlx_array_free(inj); mlx_array_free(rso); mlx_array_free(mx);
    mlx_array_free(tmp[0]); mlx_array_free(tmp[1]); mlx_array_free(tmp[2]);
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
    int rc = llm_hc_apply(2, in, 4, sh, nd, dt, 1, ((W + 255) / 256) * 256, nt, 256, &o, s);
    if (rc == 0) rc = mlx_array_set(out, o);
    mlx_array_free(HP); mlx_array_free(o);
    return rc;
}

#endif
