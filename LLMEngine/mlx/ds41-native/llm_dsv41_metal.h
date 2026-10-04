/* llm_dsv41_metal.h — DeepSeek V4.1: kernel Metal custom (via MLX `fast.metal_kernel`),
 * header-only. Semantica portata da oMLX `hyper_connection.py`/`kernels.py`.
 *
 * REGOLE (§0bis): ogni kernel e' una leva **opt-in gated** (kill-switch `LLMMETAL_DSV41_METAL=1`,
 * default OFF) con parita' vs la versione semantica; nessun kernel inline nei backend.
 * Questo modulo e' dedicato e viene chiamato dall'engine. */
#ifndef LLM_DSV41_METAL_H
#define LLM_DSV41_METAL_H

#include <stdlib.h>
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <stdbool.h>

#include <mlx/c/mlx.h>

/* kill-switch globale (default OFF). */
static inline int llm_dsv41_metal_enabled(void) {
    static int e = -1;
    if (e < 0) {
        const char* v = llm_ai_mlx_opt_env("LLMMETAL_DSV41_METAL");
        e = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return e;
}

/* ── Sinkhorn fused (hyper_connection.py _sinkhorn_kernel): comb [rows,16] f32. ── */
static const char* const LLM_DSV41_SINKHORN_SRC =
    "    const uint row = thread_position_in_grid.x;\n"
    "    if (row >= ROWS) return;\n"
    "    float values[16];\n"
    "    for (int i = 0; i < 16; ++i) values[i] = x[row * 16 + i];\n"
    "    for (int iteration = 0; iteration < ITERS; ++iteration) {\n"
    "        if (iteration > 0) {\n"
    "            for (int r = 0; r < 4; ++r) {\n"
    "                float total = 0.0f;\n"
    "                for (int c = 0; c < 4; ++c) total = values[r * 4 + c] + total;\n"
    "                total = total + eps[0];\n"
    "                for (int c = 0; c < 4; ++c) values[r * 4 + c] /= total;\n"
    "            }\n"
    "        }\n"
    "        for (int c = 0; c < 4; ++c) {\n"
    "            float total = 0.0f;\n"
    "            for (int r = 0; r < 4; ++r) total = values[r * 4 + c] + total;\n"
    "            total = total + eps[0];\n"
    "            for (int r = 0; r < 4; ++r) values[r * 4 + c] /= total;\n"
    "        }\n"
    "    }\n"
    "    for (int i = 0; i < 16; ++i) y[row * 16 + i] = values[i];\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_sinkhorn_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        mlx_vector_string_append_value(in, "x");
        mlx_vector_string_append_value(in, "eps");
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "y");
        k = mlx_fast_metal_kernel_new(
            "deepseek_v41_sinkhorn", in, outn, LLM_DSV41_SINKHORN_SRC,
            "#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n",
            true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return k;
}

/* comb [...,4,4] f32 -> sinkhorn fuso. Ritorna false se non applicabile (il chiamante fa fallback). */
static inline bool llm_dsv41_metal_sinkhorn(mlx_array* out, mlx_array comb, float eps, int iters,
                                            mlx_stream s) {
    int nd = (int)mlx_array_ndim(comb);
    if (nd < 2) return false;
    const int* sh = mlx_array_shape(comb);
    if (sh[nd - 2] != 4 || sh[nd - 1] != 4) return false;
    int rows = 1;
    for (int i = 0; i < nd - 2; i++) rows *= sh[i];
    if (rows <= 0) return false;
    mlx_fast_metal_kernel k = llm_dsv41_metal_sinkhorn_kernel();
    if (!k.ctx) return false;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, sh, (size_t)nd, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, rows, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", iters < 1 ? 1 : iters);
    mlx_array epsa = mlx_array_new_data(&eps, (const int[]){ 1 }, 1, MLX_FLOAT32);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, comb);
    mlx_vector_array_append_value(ins, epsa);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false;
        else mlx_array_set(out, o0);
    }
    mlx_vector_array_free(ins);
    mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_array_free(epsa);
    return ok;
}

/* ── hc_post fused (hyper_connection.py _post_kernel). ── */
static const char* const LLM_DSV41_HC_POST_SRC =
    "    const uint z = thread_position_in_grid.x;\n"
    "    if (z >= ROWS * D) return;\n"
    "    const uint row = z / D, d = z % D;\n"
    "    float values[4];\n"
    "    for (uint i = 0; i < 4; ++i) values[i] = float(residual[(row * 4 + i) * D + d]);\n"
    "    const float value = float(x[z]);\n"
    "    for (uint j = 0; j < 4; ++j) {\n"
    "        float sum = 0.0f;\n"
    "        for (uint i = 0; i < 4; ++i) sum = fma(comb[row * 16 + i * 4 + j], values[i], sum);\n"
    "        y[(row * 4 + j) * D + d] = T(post[row * 4 + j] * value + sum);\n"
    "    }\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_hc_post_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "x", "residual", "post", "comb" };
        for (int i = 0; i < 4; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "y");
        k = mlx_fast_metal_kernel_new("deepseek_v41_hc_post", in, outn,
                                      LLM_DSV41_HC_POST_SRC,
                                      "#pragma clang fp contract(off)\n", true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return k;
}

/* x [..,4,D] residuo, residual [..,4,D], post [..,4], comb [..,4,4] -> y = residual.shape. */
static inline bool llm_dsv41_metal_hc_post(mlx_array* out, mlx_array x, mlx_array residual,
                                           mlx_array post, mlx_array comb, mlx_stream s) {
    int ndr = (int)mlx_array_ndim(residual);
    if (ndr < 3) return false;
    const int* rsh = mlx_array_shape(residual);
    if (rsh[ndr - 2] != 4) return false;
    int D = rsh[ndr - 1];
    int rows = 1;
    for (int i = 0; i < ndr - 2; i++) rows *= rsh[i];
    if ((int)mlx_array_size(x) != rows * D) return false;
    mlx_fast_metal_kernel k = llm_dsv41_metal_hc_post_kernel();
    if (!k.ctx) return false;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, rsh, (size_t)ndr, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, rows * D, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, x);
    mlx_vector_array_append_value(ins, residual);
    mlx_vector_array_append_value(ins, post);
    mlx_vector_array_append_value(ins, comb);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false;
        else mlx_array_set(out, o0);
    }
    mlx_vector_array_free(ins);
    mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    return ok;
}

/* ── hc_projection fused (hyper_connection.py _projection_kernel): mixes = x·fnᵀ·rsqrt(...). ── */
static const char* const LLM_DSV41_HC_PROJ_WARP_SRC =
    "    const uint row = threadgroup_position_in_grid.x;\n"
    "    const uint block = threadgroup_position_in_grid.y;\n"
    "    const uint lane = thread_index_in_simdgroup;\n"
    "    float sums[12];\n"
    "    for (uint j = 0; j < 12; ++j) sums[j] = 0.0f;\n"
    "    float square = 0.0f;\n"
    "    for (uint k = lane; k < D; k += 32) {\n"
    "        const float value = float(x[row * D + k]);\n"
    "        square = fma(value, value, square);\n"
    "        for (uint j = 0; j < 12; ++j) sums[j] = fma(value, fn[(block * 12 + j) * D + k], sums[j]);\n"
    "    }\n"
    "    const float inv = rsqrt(simd_sum(square) / float(D) + eps[0]);\n"
    "    for (uint j = 0; j < 12; ++j) {\n"
    "        const float value = simd_sum(sums[j]);\n"
    "        if (lane == 0) y[row * 24 + block * 12 + j] = value * inv;\n"
    "    }\n";

static const char* const LLM_DSV41_HC_PROJ_SRC =
    "    const uint row = threadgroup_position_in_grid.x;\n"
    "    const uint block = threadgroup_position_in_grid.y;\n"
    "    const uint tid = thread_position_in_threadgroup.x;\n"
    "    const uint lane = thread_index_in_simdgroup;\n"
    "    const uint sg = simdgroup_index_in_threadgroup;\n"
    "    threadgroup float scratch[13 * 8];\n"
    "    float sums[12];\n"
    "    for (uint j = 0; j < 12; ++j) sums[j] = 0.0f;\n"
    "    float square = 0.0f;\n"
    "    for (uint k = tid; k < D; k += 256) {\n"
    "        const float value = float(x[row * D + k]);\n"
    "        square = fma(value, value, square);\n"
    "        for (uint j = 0; j < 12; ++j) sums[j] = fma(value, fn[(block * 12 + j) * D + k], sums[j]);\n"
    "    }\n"
    "    for (uint j = 0; j < 12; ++j) {\n"
    "        const float value = simd_sum(sums[j]);\n"
    "        if (lane == 0) scratch[j * 8 + sg] = value;\n"
    "    }\n"
    "    const float sq = simd_sum(square);\n"
    "    if (lane == 0) scratch[96 + sg] = sq;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (sg == 0) {\n"
    "        float q = lane < 8 ? scratch[96 + lane] : 0.0f;\n"
    "        q = simd_sum(q);\n"
    "        const float inv = rsqrt(q / float(D) + eps[0]);\n"
    "        for (uint j = 0; j < 12; ++j) {\n"
    "            float value = lane < 8 ? scratch[j * 8 + lane] : 0.0f;\n"
    "            value = simd_sum(value);\n"
    "            if (lane == 0) y[row * 24 + block * 12 + j] = value * inv;\n"
    "        }\n"
    "    }\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_proj_kernel(bool warp) {
    static mlx_fast_metal_kernel kw, kn;
    mlx_fast_metal_kernel* p = warp ? &kw : &kn;
    if (!p->ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "x", "fn", "eps" };
        for (int i = 0; i < 3; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "y");
        *p = mlx_fast_metal_kernel_new(
            warp ? "deepseek_v41_hc_projection_warp" : "deepseek_v41_hc_projection",
            in, outn, warp ? LLM_DSV41_HC_PROJ_WARP_SRC : LLM_DSV41_HC_PROJ_SRC, "", true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return *p;
}

/* x [1,S,D] (D=hc*dim), fn [24,D] -> mixes [1,S,24]. */
static inline bool llm_dsv41_metal_hc_projection(mlx_array* out, mlx_array x, mlx_array fn,
                                                 float eps, mlx_stream s) {
    int nd = (int)mlx_array_ndim(x);
    if (nd < 2 || mlx_array_ndim(fn) != 2) return false;
    const int* xsh = mlx_array_shape(x);
    const int* fsh = mlx_array_shape(fn);
    int D = xsh[nd - 1];
    if (fsh[0] != 24 || fsh[1] != D) return false;
    int rows = (int)mlx_array_size(x) / D;
    if (rows <= 0) return false;
    int threads = rows >= 1024 ? 32 : 256;
    mlx_fast_metal_kernel k = llm_dsv41_metal_proj_kernel(threads == 32);
    if (!k.ctx) return false;
    int osh[8];
    for (int i = 0; i < nd - 1; i++) osh[i] = xsh[i];
    osh[nd - 1] = 24;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, (size_t)nd, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, rows * threads, 2, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_array epsa = mlx_array_new_data(&eps, (const int[]){ 1 }, 1, MLX_FLOAT32);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, x);
    mlx_vector_array_append_value(ins, fn);
    mlx_vector_array_append_value(ins, epsa);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false;
        else mlx_array_set(out, o0);
    }
    mlx_vector_array_free(ins);
    mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_array_free(epsa);
    return ok;
}

/* ── hc_pre_norm fused (hyper_connection.py _pre_norm_kernel): hc_pre + rmsnorm. ── */
static const char* const LLM_DSV41_PRE_NORM_SRC =
    "    const uint row = threadgroup_position_in_grid.x;\n"
    "    const uint tid = thread_position_in_threadgroup.x;\n"
    "    const uint lane = thread_index_in_simdgroup;\n"
    "    const uint simd = simdgroup_index_in_threadgroup;\n"
    "    threadgroup float sums[8];\n"
    "    float values[(D + 255) / 256];\n"
    "    float total = 0.0f;\n"
    "    for (uint t = 0; t < (D + 255) / 256; ++t) {\n"
    "        const uint d = tid + 256 * t;\n"
    "        float value = 0.0f;\n"
    "        if (d < D) {\n"
    "            for (uint i = 0; i < 4; ++i)\n"
    "                value = float(x[(row * 4 + i) * D + d]) * pre[row * 4 + i] + value;\n"
    "            value = float(T(value));\n"
    "        }\n"
    "        values[t] = value;\n"
    "        total = total + value * value;\n"
    "    }\n"
    "    total = simd_sum(total);\n"
    "    if (lane == 0) sums[simd] = total;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (simd == 0) {\n"
    "        float sum = lane < 8 ? sums[lane] : 0.0f;\n"
    "        sum = simd_sum(sum);\n"
    "        if (lane == 0) sums[0] = rsqrt(sum / float(D) + eps[0]);\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint t = 0; t < (D + 255) / 256; ++t) {\n"
    "        const uint d = tid + 256 * t;\n"
    "        if (d < D) y[row * D + d] = T((values[t] * sums[0]) * float(weight[d]));\n"
    "    }\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_pre_norm_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "x", "pre", "weight", "eps" };
        for (int i = 0; i < 4; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "y");
        k = mlx_fast_metal_kernel_new("deepseek_v41_hc_pre_norm", in, outn,
                                      LLM_DSV41_PRE_NORM_SRC,
                                      "#pragma clang fp contract(off)\n", true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return k;
}

/* x [..,4,D], pre [..,4], weight [D] -> y [..,D] = rmsnorm(hc_pre(x,pre), weight, eps). */
static inline bool llm_dsv41_metal_hc_pre_norm(mlx_array* out, mlx_array x, mlx_array pre,
                                               mlx_array weight, float eps, mlx_stream s) {
    int nd = (int)mlx_array_ndim(x);
    if (nd < 3) return false;
    const int* xsh = mlx_array_shape(x);
    if (xsh[nd - 2] != 4) return false;
    int D = xsh[nd - 1];
    if ((int)mlx_array_size(weight) != D) return false;
    int rows = (int)mlx_array_size(x) / (4 * D);
    if (rows <= 0) return false;
    mlx_fast_metal_kernel k = llm_dsv41_metal_pre_norm_kernel();
    if (!k.ctx) return false;
    int osh[8];
    for (int i = 0; i < nd - 2; i++) osh[i] = xsh[i];
    osh[nd - 2] = D;
    int ndo = nd - 1;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, (size_t)ndo, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, rows * 256, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_array epsa = mlx_array_new_data(&eps, (const int[]){ 1 }, 1, MLX_FLOAT32);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, x);
    mlx_vector_array_append_value(ins, pre);
    mlx_vector_array_append_value(ins, weight);
    mlx_vector_array_append_value(ins, epsa);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false;
        else mlx_array_set(out, o0);
    }
    mlx_vector_array_free(ins);
    mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_array_free(epsa);
    return ok;
}

/* ── FP8 activation round-trip fused (activation.py quantize_fp8_activation, group 32). ── */
static const char* const LLM_DSV41_FP8_SRC =
    "    const uint i = thread_position_in_grid.x;\n"
    "    const uint n = N;\n"
    "    const float v = i < n ? float(x[i]) : 0.0f;\n"
    "    const float amax = max(simd_max(abs(v)), 0x1.cp-118f);\n"
    "    const int scale_exponent = max(int(ceil(log2(amax / 448.0f))), -126);\n"
    "    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);\n"
    "    const float scaled = clamp(v / scale, -448.0f, 448.0f);\n"
    "    const float a = abs(scaled);\n"
    "    const int step_exponent = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);\n"
    "    const float step = as_type<float>(uint(step_exponent + 127) << 23);\n"
    "    const float q = sign(scaled) * min(rint(a / step) * step, 448.0f);\n"
    "    if (i < n) y[i] = T(q * scale);\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_fp8_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        mlx_vector_string_append_value(in, "x");
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "y");
        k = mlx_fast_metal_kernel_new("deepseek_v41_fp8_activation", in, outn, LLM_DSV41_FP8_SRC,
                                      "", true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return k;
}

/* x f32 (qualsiasi shape, ultima dim multipla di 32) -> round-trip fp8 e4m3 group 32. */
static inline bool llm_dsv41_metal_fp8_roundtrip(mlx_array* out, mlx_array x, mlx_stream s) {
    int nd = (int)mlx_array_ndim(x);
    if (nd < 1) return false;
    const int* sh = mlx_array_shape(x);
    if (sh[nd - 1] % 32 != 0) return false;
    int n = (int)mlx_array_size(x);
    if (n <= 0) return false;
    mlx_fast_metal_kernel k = llm_dsv41_metal_fp8_kernel();
    if (!k.ctx) return false;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, sh, (size_t)nd, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, n, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, x);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false;
        else mlx_array_set(out, o0);
    }
    mlx_vector_array_free(ins);
    mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    return ok;
}

/* ── packed_index_scores (kernels.py _INDEX, non-MMA): chiavi fp4-packed, row D/2+D/32. ── */
static const char* const LLM_DSV41_INDEX_HDR =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "inline float v41_fp4(uchar code) {\n"
    "    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};\n"
    "    return (code & 8 ? -1.0f : 1.0f) * table[code & 7];\n"
    "}\n";

static const char* const LLM_DSV41_INDEX_SRC =
    "    const uint lane = thread_index_in_threadgroup % 32;\n"
    "    const uint position = threadgroup_position_in_grid.x * 4 + thread_index_in_threadgroup / 32;\n"
    "    const uint query = threadgroup_position_in_grid.y;\n"
    "    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;\n"
    "    if (position >= M) return;\n"
    "    const int row = CANDIDATES ? candidates[query * M + position] : int(position);\n"
    "    if (row < 0 || row >= N || row + meta[5] >= (start + int(query) + 1) / ratio) {\n"
    "        if (lane == 0) scores[query * M + position] = -INFINITY;\n"
    "        return;\n"
    "    }\n"
    "    const size_t base = size_t(row) * (D / 2 + D / 32);\n"
    "    float key[D / 32];\n"
    "    for (int v = 0; v < D / 32; ++v) {\n"
    "        const uint d = lane + v * 32;\n"
    "        const uchar code = (keys[base + d / 2] >> ((d % 2) * 4)) & 15;\n"
    "        key[v] = ldexp(v41_fp4(code), int(keys[base + D / 2 + d / 32]) - 127);\n"
    "    }\n"
    "    float result = 0.0f;\n"
    "    for (int h = 0; h < H; ++h) {\n"
    "        float dot = 0.0f;\n"
    "        for (int v = 0; v < D / 32; ++v)\n"
    "            dot += float(q[(query * H + h) * D + lane + v * 32]) * key[v];\n"
    "        result += max(simd_sum(dot), 0.0f) * weights[query * H + h];\n"
    "    }\n"
    "    if (lane == 0) scores[query * M + position] = result;\n";

static const char* const LLM_DSV41_INDEX_MMA_SRC =
    "    const uint tid = thread_index_in_threadgroup;\n"
    "    const uint lane = tid % 32, sg = tid / 32;\n"
    "    const uint query = threadgroup_position_in_grid.y;\n"
    "    constexpr uint COLS = HEAD_SPLIT ? 16 : 32;\n"
    "    const uint key_group = HEAD_SPLIT ? sg % 2 : sg;\n"
    "    const uint block = threadgroup_position_in_grid.x * COLS;\n"
    "    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;\n"
    "    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);\n"
    "    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);\n"
    "    threadgroup float tile[D * COLS];\n"
    "    for (uint i = tid; i < D * COLS; i += 128) {\n"
    "        const uint d = i / COLS, col = i % COLS, pos = block + col;\n"
    "        const int row = pos < M ? (CANDIDATES ? candidates[query * M + pos] : int(pos)) : -1;\n"
    "        float value = 0.0f;\n"
    "        if (row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio) {\n"
    "            const size_t base = size_t(row) * (D / 2 + D / 32);\n"
    "            const uchar code = (keys[base + d / 2] >> ((d % 2) * 4)) & 15;\n"
    "            value = ldexp(v41_fp4(code), int(keys[base + D / 2 + d / 32]) - 127);\n"
    "        }\n"
    "        tile[i] = value;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float total0 = 0.0f, total1 = 0.0f;\n"
    "    for (int h = HEAD_SPLIT ? int(sg / 2) * 8 : 0; h < H; h += HEAD_SPLIT ? 16 : 8) {\n"
    "        simdgroup_matrix<float, 8, 8> acc, a, b;\n"
    "        acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;\n"
    "        for (int d = 0; d < D; d += 8) {\n"
    "            a.thread_elements()[0] = h + fm < H ? float(q[(query * H + h + fm) * D + d + fn]) : 0.0f;\n"
    "            a.thread_elements()[1] = h + fm < H ? float(q[(query * H + h + fm) * D + d + fn + 1]) : 0.0f;\n"
    "            b.thread_elements()[0] = tile[(d + fm) * COLS + key_group * 8 + fn];\n"
    "            b.thread_elements()[1] = tile[(d + fm) * COLS + key_group * 8 + fn + 1];\n"
    "            simdgroup_multiply_accumulate(acc, a, b, acc);\n"
    "        }\n"
    "        const float weight = h + fm < H ? weights[query * H + h + fm] : 0.0f;\n"
    "        float v0 = max(acc.thread_elements()[0], 0.0f) * weight;\n"
    "        float v1 = max(acc.thread_elements()[1], 0.0f) * weight;\n"
    "        for (uint i = 0; i < 3; ++i) {\n"
    "            const uint mask = i == 0 ? 2 : (i == 1 ? 4 : 16);\n"
    "            v0 += simd_shuffle_xor(v0, mask); v1 += simd_shuffle_xor(v1, mask);\n"
    "        }\n"
    "        total0 += v0; total1 += v1;\n"
    "    }\n"
    "    threadgroup float head_totals[32];\n"
    "    if (HEAD_SPLIT) {\n"
    "        if (fm == 0) { head_totals[sg * 8 + fn] = total0; head_totals[sg * 8 + fn + 1] = total1; }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (tid < COLS && block + tid < M) {\n"
    "            const uint pos = block + tid;\n"
    "            const int row = CANDIDATES ? candidates[query * M + pos] : int(pos);\n"
    "            const bool valid = row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio;\n"
    "            scores[query * M + pos] = valid ? head_totals[tid] + head_totals[tid + COLS] : -INFINITY;\n"
    "        }\n"
    "    } else if (fm == 0) {\n"
    "        for (uint j = 0; j < 2; ++j) {\n"
    "            const uint pos = block + sg * 8 + fn + j;\n"
    "            if (pos >= M) continue;\n"
    "            const int row = CANDIDATES ? candidates[query * M + pos] : int(pos);\n"
    "            const bool valid = row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio;\n"
    "            scores[query * M + pos] = valid ? (j == 0 ? total0 : total1) : -INFINITY;\n"
    "        }\n"
    "    }\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_index_kernel(bool mma) {
    static mlx_fast_metal_kernel kn, km;
    mlx_fast_metal_kernel* p = mma ? &km : &kn;
    if (!p->ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "q", "keys", "weights", "candidates", "meta" };
        for (int i = 0; i < 5; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "scores");
        *p = mlx_fast_metal_kernel_new(mma ? "deepseek_v41_packed_index_mma" : "deepseek_v41_packed_index",
                                       in, outn, mma ? LLM_DSV41_INDEX_MMA_SRC : LLM_DSV41_INDEX_SRC,
                                       LLM_DSV41_INDEX_HDR, true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return *p;
}

/* q [1,L,H,D] f32 · keys uint8 [1,N,D/2+D/32] · weights [1,L,H] · start/ratio/key_start.
 * scores [1,L,M] (M=N senza candidati). candidates opzionale int32 [1,L,M]. */
static inline bool llm_dsv41_metal_index_scores(mlx_array* out, mlx_array q, mlx_array keys,
                                                mlx_array weights, int start, int ratio,
                                                int key_start, mlx_array candidates, int has_cand,
                                                mlx_stream s) {
    if (mlx_array_ndim(q) != 4 || mlx_array_ndim(keys) != 3) return false;
    const int* qsh = mlx_array_shape(q);
    const int* ksh = mlx_array_shape(keys);
    const int H = qsh[2], D = qsh[3], L = qsh[1], N = ksh[1];
    if (qsh[0] != 1 || D % 32 || ksh[0] != 1 || ksh[2] != D / 2 + D / 32 || ratio <= 0) return false;
    int use_cand = has_cand ? 1 : 0;
    int M = use_cand ? mlx_array_shape(candidates)[2] : N;
    if (M <= 0) return false;
    int mma = (D <= 128 && H >= 8) ? 1 : 0;
    int head_split = (mma && H >= 32) ? 1 : 0;
    int cols = head_split ? 16 : 32;
    mlx_fast_metal_kernel k = llm_dsv41_metal_index_kernel(mma);
    if (!k.ctx) return false;
    int meta[6] = { H, N, M, start, ratio, key_start };
    mlx_array meta_a = mlx_array_new_data(meta, (const int[]){ 6 }, 1, MLX_INT32);
    mlx_array cand_a = use_cand ? candidates
                               : mlx_array_new_data((const int[]){ 0 }, (const int[]){ 1 }, 1, MLX_INT32);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (const int[]){ 1, L, M }, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (mma ? ((M + cols - 1) / cols) : ((M + 3) / 4)) * 128, L, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "CANDIDATES", use_cand);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "HEAD_SPLIT", head_split);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HEADS", H);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RATIO", ratio);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, q);
    mlx_vector_array_append_value(ins, keys);
    mlx_vector_array_append_value(ins, weights);
    mlx_vector_array_append_value(ins, cand_a);
    mlx_vector_array_append_value(ins, meta_a);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) {
        mlx_array o0 = mlx_array_new();
        if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false;
        else mlx_array_set(out, o0);
    }
    mlx_vector_array_free(ins);
    mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_array_free(meta_a);
    if (!use_cand) mlx_array_free(cand_a);
    return ok;
}

/* ── packed_sparse_attention (kernels.py _ATTENTION non-MMA + _MERGE). ── */
static const char* const LLM_DSV41_ATT_HDR =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "inline float v41_fp8(uchar code) {\n"
    "    const uint a = code & 127, exponent = a >> 3, mantissa = a & 7;\n"
    "    const float value = exponent == 0 ? ldexp(float(mantissa), -9)\n"
    "        : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);\n"
    "    return code & 128 ? -value : value;\n"
    "}\n"
    "inline float v41_fp4(uchar code) {\n"
    "    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};\n"
    "    return (code & 8 ? -1.0f : 1.0f) * table[code & 7];\n"
    "}\n";

static const char* const LLM_DSV41_ATT_SRC =
    "    const uint lane = thread_index_in_threadgroup % 32;\n"
    "    const uint head = threadgroup_position_in_grid.x * 4 + thread_index_in_threadgroup / 32;\n"
    "    const uint query = threadgroup_position_in_grid.y;\n"
    "    const uint split = threadgroup_position_in_grid.z;\n"
    "    const int H = meta[0], W = meta[1], C = meta[2];\n"
    "    const int NW = meta[3], NC = meta[4], NS = meta[5];\n"
    "    if (head >= H) return;\n"
    "    constexpr bool REUSE_SCALE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;\n"
    "    float query_values[D / 32], acc[D / 32];\n"
    "    for (int v = 0; v < D / 32; ++v) {\n"
    "        query_values[v] = float(q[(query * H + head) * D + (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32)]);\n"
    "        acc[v] = 0.0f;\n"
    "    }\n"
    "    float maximum = -INFINITY, denominator = 0.0f;\n"
    "    const int end = min(int((split + 1) * CHUNK), W + C);\n"
    "    for (int j = split * CHUNK; j < end; ++j) {\n"
    "        const bool compressed = j >= W;\n"
    "        const int row = compressed ? ci[query * C + j - W] : wi[query * W + j];\n"
    "        if (row < 0 || row >= (compressed ? NC : NW)) continue;\n"
    "        const size_t base = size_t(row) * (compressed ? (D / 2 + D / 16) : (D + D / 32));\n"
    "        const uint first = lane * (D / 32);\n"
    "        const float pooled_scale = REUSE_SCALE && compressed\n"
    "            ? v41_fp8(pooled[base + D / 2 + first / 16]) : 0.0f;\n"
    "        const int window_exp = REUSE_SCALE && !compressed\n"
    "            ? int(window[base + D + first / 32]) - 127 : 0;\n"
    "        float values[D / 32], dot = 0.0f;\n"
    "        for (int v = 0; v < D / 32; ++v) {\n"
    "            const uint d = (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32);\n"
    "            float value;\n"
    "            if (compressed) {\n"
    "                const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;\n"
    "                value = v41_fp4(code) * (REUSE_SCALE ? pooled_scale : v41_fp8(pooled[base + D / 2 + d / 16]));\n"
    "            } else {\n"
    "                value = ldexp(v41_fp8(window[base + d]), REUSE_SCALE ? window_exp : int(window[base + D + d / 32]) - 127);\n"
    "            }\n"
    "            values[v] = value;\n"
    "            dot += query_values[v] * value;\n"
    "        }\n"
    "        const float score = simd_sum(dot) * scalep[0];\n"
    "        const float next_maximum = max(maximum, score);\n"
    "        const float correction = exp(maximum - next_maximum);\n"
    "        const float probability = exp(score - next_maximum);\n"
    "        denominator = denominator * correction + probability;\n"
    "        for (int v = 0; v < D / 32; ++v) acc[v] = acc[v] * correction + probability * values[v];\n"
    "        maximum = next_maximum;\n"
    "    }\n"
    "    const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);\n"
    "    for (int v = 0; v < D / 32; ++v) partial[base + (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32)] = acc[v];\n"
    "    if (lane == 0) { partial[base + D] = maximum; partial[base + D + 1] = denominator; }\n";

static const char* const LLM_DSV41_MERGE_SRC =
    "    const uint lane = thread_index_in_threadgroup % 32;\n"
    "    const uint head = threadgroup_position_in_grid.x * 4 + thread_index_in_threadgroup / 32;\n"
    "    const uint query = threadgroup_position_in_grid.y;\n"
    "    const int H = meta[0], NS = meta[1];\n"
    "    if (head >= H) return;\n"
    "    const size_t base = (size_t(query) * H + head) * NS * (D + 2);\n"
    "    float maximum = float(sink[head]);\n"
    "    for (int s = 0; s < NS; ++s) maximum = max(maximum, partial[base + s * (D + 2) + D]);\n"
    "    float denominator = exp(float(sink[head]) - maximum);\n"
    "    float acc[D / 32];\n"
    "    for (int v = 0; v < D / 32; ++v) acc[v] = 0.0f;\n"
    "    for (int s = 0; s < NS; ++s) {\n"
    "        const size_t pos = base + s * (D + 2);\n"
    "        const float correction = exp(partial[pos + D] - maximum);\n"
    "        denominator += partial[pos + D + 1] * correction;\n"
    "        for (int v = 0; v < D / 32; ++v) acc[v] += partial[pos + lane + v * 32] * correction;\n"
    "    }\n"
    "    for (int v = 0; v < D / 32; ++v) out[(query * H + head) * D + lane + v * 32] = T(acc[v] / denominator);\n";

static const char* const LLM_DSV41_ATT_MMA_SRC =
    "\n"
    "    const uint tid = thread_index_in_threadgroup, lane = tid % 32, sg = tid / 32;\n"
    "    const uint query = threadgroup_position_in_grid.y, split = threadgroup_position_in_grid.z;\n"
    "    const uint first_head = threadgroup_position_in_grid.x * 8;\n"
    "    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NS = meta[5];\n"
    "    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);\n"
    "    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);\n"
    "    threadgroup float tile[8 * D], dots[4 * 64], probability[64];\n"
    "    threadgroup float maxima[8], denominator[8], correction[8];\n"
    "    threadgroup bool valid[8];\n"
    "    simdgroup_matrix<float, 8, 8> output[D / 32];\n"
    "    for (int v = 0; v < D / 32; ++v) {\n"
    "        output[v].thread_elements()[0] = 0.0f;\n"
    "        output[v].thread_elements()[1] = 0.0f;\n"
    "    }\n"
    "    if (tid < 8) { maxima[tid] = -INFINITY; denominator[tid] = 0.0f; }\n"
    "    const int end = min(int((split + 1) * CHUNK), W + C);\n"
    "    for (int begin = split * CHUNK; begin < end; begin += 8) {\n"
    "        for (uint i = tid; i < 8 * D; i += 128) {\n"
    "            const uint r = i / D, d = i % D, j = begin + r;\n"
    "            const bool compressed = j >= W;\n"
    "            int row = -1;\n"
    "            if (j < end) row = compressed ? ci[query * C + j - W] : wi[query * W + j];\n"
    "            const bool ok = row >= 0 && row < (compressed ? NC : NW);\n"
    "            float value = 0.0f;\n"
    "            if (ok) {\n"
    "                if (compressed) {\n"
    "                    const size_t base = size_t(row) * (D / 2 + D / 16);\n"
    "                    const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;\n"
    "                    value = v41_fp4(code) * v41_fp8(pooled[base + D / 2 + d / 16]);\n"
    "                } else {\n"
    "                    const size_t base = size_t(row) * (D + D / 32);\n"
    "                    value = ldexp(v41_fp8(window[base + d]), int(window[base + D + d / 32]) - 127);\n"
    "                }\n"
    "            }\n"
    "            tile[i] = value;\n"
    "            if (d == 0) valid[r] = ok;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        // Four simdgroups split the QK dot product's channel dimension.\n"
    "        simdgroup_matrix<float, 8, 8> score, a, b;\n"
    "        score.thread_elements()[0] = 0.0f;\n"
    "        score.thread_elements()[1] = 0.0f;\n"
    "        for (int k = sg * (D / 4); k < (sg + 1) * (D / 4); k += 8) {\n"
    "            const uint head = first_head + fm;\n"
    "            a.thread_elements()[0] = head < H ? float(q[(query * H + head) * D + k + fn]) : 0.0f;\n"
    "            a.thread_elements()[1] = head < H ? float(q[(query * H + head) * D + k + fn + 1]) : 0.0f;\n"
    "            b.thread_elements()[0] = tile[fn * D + k + fm];\n"
    "            b.thread_elements()[1] = tile[(fn + 1) * D + k + fm];\n"
    "            simdgroup_multiply_accumulate(score, a, b, score);\n"
    "        }\n"
    "        dots[sg * 64 + fm * 8 + fn] = score.thread_elements()[0];\n"
    "        dots[sg * 64 + fm * 8 + fn + 1] = score.thread_elements()[1];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (tid < 8) {\n"
    "            float values[8], next_max = maxima[tid];\n"
    "            for (uint r = 0; r < 8; ++r) {\n"
    "                float sum = dots[tid * 8 + r] + dots[64 + tid * 8 + r]\n"
    "                    + dots[128 + tid * 8 + r] + dots[192 + tid * 8 + r];\n"
    "                values[r] = valid[r] ? sum * scalep[0] : -INFINITY;\n"
    "                next_max = max(next_max, values[r]);\n"
    "            }\n"
    "            correction[tid] = maxima[tid] == -INFINITY ? 0.0f : exp(maxima[tid] - next_max);\n"
    "            float total = denominator[tid] * correction[tid];\n"
    "            for (uint r = 0; r < 8; ++r) {\n"
    "                const float prob = valid[r] ? exp(values[r] - next_max) : 0.0f;\n"
    "                probability[tid * 8 + r] = prob;\n"
    "                total += prob;\n"
    "            }\n"
    "            maxima[tid] = next_max;\n"
    "            denominator[tid] = total;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        // PV reuses the decoded tile. Float matrices retain FP32 probabilities.\n"
    "        a.thread_elements()[0] = probability[fm * 8 + fn];\n"
    "        a.thread_elements()[1] = probability[fm * 8 + fn + 1];\n"
    "        for (int v = 0; v < D / 32; ++v) {\n"
    "            const uint d = sg * (D / 4) + v * 8 + fn;\n"
    "            b.thread_elements()[0] = tile[fm * D + d];\n"
    "            b.thread_elements()[1] = tile[fm * D + d + 1];\n"
    "            output[v].thread_elements()[0] *= correction[fm];\n"
    "            output[v].thread_elements()[1] *= correction[fm];\n"
    "            simdgroup_multiply_accumulate(output[v], a, b, output[v]);\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    const uint head = first_head + fm;\n"
    "    if (head < H) {\n"
    "        const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);\n"
    "        for (int v = 0; v < D / 32; ++v) {\n"
    "            const uint d = sg * (D / 4) + v * 8 + fn;\n"
    "            partial[base + d] = output[v].thread_elements()[0];\n"
    "            partial[base + d + 1] = output[v].thread_elements()[1];\n"
    "        }\n"
    "    }\n"
    "    if (tid < 8 && first_head + tid < H) {\n"
    "        const size_t base = ((size_t(query) * H + first_head + tid) * NS + split) * (D + 2);\n"
    "        partial[base + D] = maxima[tid];\n"
    "        partial[base + D + 1] = denominator[tid];\n"
    "    }\n"
    "\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_att_kernel(bool mma) {
    static mlx_fast_metal_kernel kn, km;
    mlx_fast_metal_kernel* p = mma ? &km : &kn;
    if (!p->ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "q", "window", "pooled", "wi", "ci", "meta", "scalep" };
        for (int i = 0; i < 7; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "partial");
        *p = mlx_fast_metal_kernel_new(mma ? "deepseek_v41_packed_attention_mma" : "deepseek_v41_packed_attention",
                                       in, outn, mma ? LLM_DSV41_ATT_MMA_SRC : LLM_DSV41_ATT_SRC,
                                       LLM_DSV41_ATT_HDR, true, false);
        mlx_vector_string_free(in); mlx_vector_string_free(outn);
    }
    return *p;
}
static inline mlx_fast_metal_kernel llm_dsv41_metal_merge_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "partial", "sink", "meta" };
        for (int i = 0; i < 3; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "out");
        k = mlx_fast_metal_kernel_new("deepseek_v41_packed_attention_merge", in, outn,
                                      LLM_DSV41_MERGE_SRC, LLM_DSV41_ATT_HDR, true, false);
        mlx_vector_string_free(in); mlx_vector_string_free(outn);
    }
    return k;
}

/* q [1,L,H,D] f32 · window uint8 [1,NW,D+D/32] · pooled uint8 [1,NC,D/2+D/16] ·
 * wi int32 [1,L,W] · ci int32 [1,L,C] · sink [H] · scale. out [1,L,H,D]. */
static inline bool llm_dsv41_metal_packed_attention(mlx_array* out, mlx_array q, mlx_array window,
                                                    mlx_array pooled, mlx_array wi, mlx_array ci,
                                                    mlx_array sink, float scale, mlx_stream s) {
    if (mlx_array_ndim(q) != 4) return false;
    const int* qsh = mlx_array_shape(q);
    const int L = qsh[1], H = qsh[2], D = qsh[3];
    if (qsh[0] != 1 || D % 32) return false;
    const int NW = (mlx_array_ndim(window) == 3) ? mlx_array_shape(window)[1] : 0;
    const int NC = (mlx_array_ndim(pooled) == 3) ? mlx_array_shape(pooled)[1] : 0;
    const int W = mlx_array_shape(wi)[2];
    const int C = (mlx_array_ndim(ci) == 3) ? mlx_array_shape(ci)[2] : 0;
    int chunk = (L <= 8) ? 32 : 128;
    int splits = (W + C + chunk - 1) / chunk; if (splits < 1) splits = 1;
    int meta[6] = { H, W, C, NW, NC, splits };
    mlx_array meta_a = mlx_array_new_data(meta, (const int[]){ 6 }, 1, MLX_INT32);
    mlx_array scale_a = mlx_array_new_data(&scale, (const int[]){ 1 }, 1, MLX_FLOAT32);
    mlx_array win_a = (NW > 0) ? window : mlx_array_new_data((const char*)"", (const int[]){ 1 }, 1, MLX_UINT8);
    mlx_array pool_a = (NC > 0) ? pooled : mlx_array_new_data((const char*)"", (const int[]){ 1 }, 1, MLX_UINT8);
    mlx_array ci_a = (C > 0) ? ci : mlx_array_new_data((const int[]){ 0 }, (const int[]){ 1 }, 1, MLX_INT32);

    int mma = (L > 8 && D <= 512) ? 1 : 0;
    int hg = mma ? 8 : 4;
    mlx_fast_metal_kernel k = llm_dsv41_metal_att_kernel(mma);
    if (!k.ctx) return false;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (const int[]){ 1, L, H, splits, D + 2 }, 5, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, ((H + hg - 1) / hg) * 128, L, splits);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CHUNK", chunk);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, q);
    mlx_vector_array_append_value(ins, win_a);
    mlx_vector_array_append_value(ins, pool_a);
    mlx_vector_array_append_value(ins, wi);
    mlx_vector_array_append_value(ins, ci_a);
    mlx_vector_array_append_value(ins, meta_a);
    mlx_vector_array_append_value(ins, scale_a);
    mlx_vector_array mids = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&mids, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(mids) >= 1);
    mlx_array partial = mlx_array_new();
    if (ok) { if (mlx_vector_array_get(&partial, mids, 0) != 0) ok = false; }
    if (ok) {
        mlx_fast_metal_kernel mk = llm_dsv41_metal_merge_kernel();
        int mmeta[2] = { H, splits };
        mlx_array mmeta_a = mlx_array_new_data(mmeta, (const int[]){ 2 }, 1, MLX_INT32);
        mlx_fast_metal_kernel_config mcfg = mlx_fast_metal_kernel_config_new();
        mlx_fast_metal_kernel_config_add_output_arg(mcfg, qsh, 4, MLX_FLOAT32);
        mlx_fast_metal_kernel_config_set_grid(mcfg, ((H + 3) / 4) * 128, L, 1);
        mlx_fast_metal_kernel_config_set_thread_group(mcfg, 128, 1, 1);
        mlx_fast_metal_kernel_config_add_template_arg_int(mcfg, "D", D);
        mlx_fast_metal_kernel_config_add_template_arg_dtype(mcfg, "T", MLX_FLOAT32);
        mlx_vector_array mins = mlx_vector_array_new();
        mlx_vector_array_append_value(mins, partial);
        mlx_vector_array_append_value(mins, sink);
        mlx_vector_array_append_value(mins, mmeta_a);
        mlx_vector_array mouts = mlx_vector_array_new();
        int mrc = mlx_fast_metal_kernel_apply(&mouts, mk, mins, mcfg, s);
        ok = (mrc == 0 && mlx_vector_array_size(mouts) >= 1);
        if (ok) { mlx_array o0 = mlx_array_new(); if (mlx_vector_array_get(&o0, mouts, 0) != 0) ok = false; else mlx_array_set(out, o0); mlx_array_free(o0); }
        mlx_vector_array_free(mins); mlx_vector_array_free(mouts);
        mlx_fast_metal_kernel_config_free(mcfg); mlx_array_free(mmeta_a);
    }
    mlx_vector_array_free(ins); mlx_vector_array_free(mids);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_array_free(meta_a); mlx_array_free(scale_a);
    if (NW <= 0) mlx_array_free(win_a);
    if (NC <= 0) mlx_array_free(pool_a);
    if (C <= 0) mlx_array_free(ci_a);
    return ok;
}

static const char* const LLM_DSV41_TOPK_SRC =
    "\n"
    "    const uint tid = thread_index_in_threadgroup;\n"
    "    const uint query = threadgroup_position_in_grid.y;\n"
    "    const uint tile = threadgroup_position_in_grid.x;\n"
    "    const int width = meta[0], base = meta[1];\n"
    "    threadgroup float exchange_values[TILE];\n"
    "    threadgroup int exchange_ids[TILE];\n"
    "    float values[TILE / THREADS], next_values[TILE / THREADS];\n"
    "    int indices[TILE / THREADS], next_ids[TILE / THREADS];\n"
    "    for (uint j = 0; j < TILE / THREADS; ++j) {\n"
    "        const uint i = tid + j * THREADS, pos = tile * TILE + i;\n"
    "        float value = -INFINITY;\n"
    "        if (pos < width) {\n"
    "            for (uint k = 0; k < BLOCK; ++k) {\n"
    "                const uint src = pos * BLOCK + k;\n"
    "                if (src < uint(meta[2])) value = max(value, scores[query * meta[2] + src]);\n"
    "            }\n"
    "            const int length = (meta[3] + int(query) + 1) / RATIO;\n"
    "            if (FORCE && length > 0 && int(pos) + base == (length - 1) / BLOCK)\n"
    "                value = INFINITY;\n"
    "        }\n"
    "        values[j] = value;\n"
    "        indices[j] = pos < width ? (EXPLICIT ? ids[query * width + pos] : base + int(pos)) : INT_MAX;\n"
    "    }\n"
    "    // Only cross-SIMD exchanges need threadgroup storage and barriers.\n"
    "    for (uint size = 2; size <= TILE; size <<= 1) {\n"
    "        for (uint stride = size >> 1; stride; stride >>= 1) {\n"
    "            const bool cross_simd = stride >= 32 && stride < THREADS;\n"
    "            if (cross_simd) {\n"
    "                for (uint j = 0; j < TILE / THREADS; ++j) {\n"
    "                    exchange_values[tid + j * THREADS] = values[j];\n"
    "                    exchange_ids[tid + j * THREADS] = indices[j];\n"
    "                }\n"
    "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "            }\n"
    "            for (uint j = 0; j < TILE / THREADS; ++j) {\n"
    "                const uint i = tid + j * THREADS;\n"
    "                float other;\n"
    "                int other_id;\n"
    "                if (stride < 32) {\n"
    "                    other = simd_shuffle_xor(values[j], stride);\n"
    "                    other_id = simd_shuffle_xor(indices[j], stride);\n"
    "                } else if (cross_simd) {\n"
    "                    other = exchange_values[i ^ stride];\n"
    "                    other_id = exchange_ids[i ^ stride];\n"
    "                } else {\n"
    "                    other = values[j ^ (stride / THREADS)];\n"
    "                    other_id = indices[j ^ (stride / THREADS)];\n"
    "                }\n"
    "                const bool better = other > values[j] || (other == values[j] && other_id < indices[j]);\n"
    "                const bool worse = other < values[j] || (other == values[j] && other_id > indices[j]);\n"
    "                const bool descending = ((i & size) == 0) == ((i & stride) == 0);\n"
    "                const bool take = descending ? better : worse;\n"
    "                next_values[j] = take ? other : values[j];\n"
    "                next_ids[j] = take ? other_id : indices[j];\n"
    "            }\n"
    "            if (cross_simd) threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "            for (uint j = 0; j < TILE / THREADS; ++j) {\n"
    "                values[j] = next_values[j]; indices[j] = next_ids[j];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    for (uint j = 0; j < TILE / THREADS; ++j) {\n"
    "        const uint i = tid + j * THREADS;\n"
    "        if (i < K) {\n"
    "            const size_t out = (size_t(query) * NT + tile) * K + i;\n"
    "            selected_scores[out] = values[j];\n"
    "            selected_ids[out] = indices[j];\n"
    "        }\n"
    "    }\n"
    "\n"
    "\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_topk_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        const char* nm[] = { "scores", "ids", "meta" };
        for (int i = 0; i < 3; i++) mlx_vector_string_append_value(in, nm[i]);
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "selected_scores");
        mlx_vector_string_append_value(outn, "selected_ids");
        k = mlx_fast_metal_kernel_new("deepseek_v41_tile_topk", in, outn, LLM_DSV41_TOPK_SRC,
                                      LLM_DSV41_INDEX_HDR, true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outn);
    }
    return k;
}

/* Top-k su un singolo tile (width<=32, block=1, no force). out_ids int32 [1,L,count]
 * (desc per score), out_scores f32 [1,L,count]. Ritorna false se non applicabile. */
static inline bool llm_dsv41_metal_tile_topk(mlx_array* out_scores, mlx_array* out_ids,
                                             mlx_array scores, int count, mlx_stream s) {
    if (mlx_array_ndim(scores) != 3) return false;
    const int* sh = mlx_array_shape(scores);
    int L = sh[1], M = sh[2];
    if (M <= 0 || M > 32 || count <= 0 || count > M) return false;
    int tile = 32, threads = 32, tiles = 1;
    int meta[5] = { M, 0, M, 0, 1 };
    mlx_array meta_a = mlx_array_new_data(meta, (const int[]){ 5 }, 1, MLX_INT32);
    mlx_array ids_a = mlx_array_new_data((const int[]){ 0 }, (const int[]){ 1 }, 1, MLX_INT32);
    mlx_fast_metal_kernel k = llm_dsv41_metal_topk_kernel();
    if (!k.ctx) { mlx_array_free(meta_a); mlx_array_free(ids_a); return false; }
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (const int[]){ 1, L, tiles * count }, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (const int[]){ 1, L, tiles * count }, 3, MLX_INT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, tiles * threads, L, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TILE", tile);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", count);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NT", tiles);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "EXPLICIT", false);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BLOCK", 1);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "FORCE", false);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "THREADS", threads);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RATIO", 1);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, scores);
    mlx_vector_array_append_value(ins, ids_a);
    mlx_vector_array_append_value(ins, meta_a);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 2);
    if (ok) {
        mlx_array s0 = mlx_array_new(), i0 = mlx_array_new();
        if (mlx_vector_array_get(&s0, outs, 0) != 0 || mlx_vector_array_get(&i0, outs, 1) != 0) ok = false;
        else { mlx_array_set(out_scores, s0); mlx_array_set(out_ids, i0); }
    }
    mlx_vector_array_free(ins); mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg); mlx_array_free(meta_a); mlx_array_free(ids_a);
    return ok;
}

/* ── PACKING di attivazioni (f32 -> uint8 packed) per alimentare i kernel packed ──
 * Emettono ESATTAMENTE gli stessi byte di `llmcodec_pack_fp8` (fp8 e4m3 grp32) e
 * `llmcodec_pack_activation` (fp4, group_size e scala e4m3 o potenza-di-2).
 * Layout riga: fp8 grp32 -> [D byte valore][D/32 byte scala]; fp4 -> [D*bits/8 valore][D/gs scala].
 * Così il forward puo' produrre i buffer packed lato device (nessun round-trip host). */

static const char* const LLM_DSV41_PACK_HDR =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "inline float v41_round_fp8(float x) {\n"
    "    float a = fabs(x); if (a > 448.0f) a = 448.0f;\n"
    "    const float am = a < 1.953125e-3f ? 1.953125e-3f : a;\n"
    "    const int e = int(floor(log2(am)));\n"
    "    int ste = e - 3; if (ste < -9) ste = -9;\n"
    "    const float step = ldexp(1.0f, ste);\n"
    "    float q = rint(a / step) * step; if (q > 448.0f) q = 448.0f;\n"
    "    return x < 0.0f ? -q : q;\n"
    "}\n"
    "inline uchar v41_to_fp8(float v) {\n"
    "    const int sign = v < 0.0f ? 1 : 0;\n"
    "    const float a = fabs(v);\n"
    "    if (a == 0.0f) return uchar(sign << 7);\n"
    "    int e; const float f = frexp(a, e);\n"
    "    const int E = e - 1; int eb, mant;\n"
    "    if (E < -6) { mant = int(round(a / ldexp(1.0f, -9))); eb = 0; if (mant > 7) { eb = 1; mant = 0; } }\n"
    "    else { mant = int(round((2.0f * f - 1.0f) * 8.0f)); eb = E + 7; if (mant == 8) { mant = 0; eb++; } if (eb > 15) { eb = 15; mant = 6; } }\n"
    "    return uchar((sign << 7) | ((eb & 15) << 3) | (mant & 7));\n"
    "}\n"
    "inline int v41_fp4_code(float a) {\n"
    "    if (a > 5.0f) return 7; if (a >= 3.5f) return 6; if (a > 2.5f) return 5;\n"
    "    if (a >= 1.75f) return 4; if (a > 1.25f) return 3; if (a >= 0.75f) return 2;\n"
    "    if (a > 0.25f) return 1; return 0;\n"
    "}\n";

/* fp8 e4m3 grp32: x f32 [ROWS*D] -> out uint8 [ROWS*(D+D/32)] (1 thread per gruppo da 32). */
static const char* const LLM_DSV41_PACK8_SRC =
    "    const uint t = thread_position_in_grid.x;\n"
    "    const uint ng = D / 32;\n"
    "    const uint g = t % ng, row = t / ng;\n"
    "    if (row >= ROWS) return;\n"
    "    const uint base = row * D + g * 32;\n"
    "    float amax = 0.0f;\n"
    "    for (uint i = 0; i < 32; ++i) amax = max(amax, fabs(x[base + i]));\n"
    "    const float minv = 448.0f * ldexp(1.0f, -126);\n"
    "    if (amax < minv) amax = minv;\n"
    "    float ex = ceil(log2(amax / 448.0f)); if (ex < -126.0f) ex = -126.0f;\n"
    "    const int ei = int(ex); const float scale = ldexp(1.0f, ei);\n"
    "    const uint orow = row * (D + D / 32);\n"
    "    out[orow + D + g] = uchar(ei + 127);\n"
    "    for (uint i = 0; i < 32; ++i) {\n"
    "        float s = clamp(x[base + i] / scale, -448.0f, 448.0f);\n"
    "        out[orow + g * 32 + i] = v41_to_fp8(v41_round_fp8(s));\n"
    "    }\n";

/* fp4: x f32 [ROWS*D] -> out uint8 [ROWS*(D/2+D/GS)] (1 thread per gruppo da GS). */
static const char* const LLM_DSV41_PACK4_SRC =
    "    const uint t = thread_position_in_grid.x;\n"
    "    const uint ng = D / GS;\n"
    "    const uint g = t % ng, row = t / ng;\n"
    "    if (row >= ROWS) return;\n"
    "    const uint base = row * D + g * GS;\n"
    "    float amax = 0.0f;\n"
    "    for (uint i = 0; i < GS; ++i) amax = max(amax, fabs(x[base + i]));\n"
    "    const float limit = 6.0f;\n"
    "    const float minimum = limit * (E4M3 ? ldexp(1.0f, -9) : ldexp(1.0f, -126));\n"
    "    if (amax < minimum) amax = minimum;\n"
    "    const uint orow = row * (D / 2 + D / GS);\n"
    "    float scale;\n"
    "    if (E4M3) { scale = v41_round_fp8(amax / limit); out[orow + D / 2 + g] = v41_to_fp8(scale); }\n"
    "    else { float ex = ceil(log2(amax / limit)); if (ex < -126.0f) ex = -126.0f; const int ei = int(ex);\n"
    "           scale = ldexp(1.0f, ei); out[orow + D / 2 + g] = uchar(ei + 127); }\n"
    "    const uint bbase = orow + g * (GS / 2);\n"
    "    for (uint i = 0; i < GS; ++i) {\n"
    "        float s = clamp(x[base + i] / scale, -limit, limit);\n"
    "        int code = v41_fp4_code(fabs(s)); if (s < 0.0f) code |= 8;\n"
    "        device uchar* byte = &out[bbase + i / 2];\n"
    "        if ((i & 1) == 0) *byte = uchar((*byte & 0xF0) | (code & 15));\n"
    "        else *byte = uchar((*byte & 0x0F) | ((code & 15) << 4));\n"
    "    }\n";

static inline mlx_fast_metal_kernel llm_dsv41_metal_pack8_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        mlx_vector_string_append_value(in, "x");
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "out");
        k = mlx_fast_metal_kernel_new("deepseek_v41_pack8", in, outn, LLM_DSV41_PACK8_SRC,
                                      LLM_DSV41_PACK_HDR, true, false);
        mlx_vector_string_free(in); mlx_vector_string_free(outn);
    }
    return k;
}

static inline mlx_fast_metal_kernel llm_dsv41_metal_pack4_kernel(void) {
    static mlx_fast_metal_kernel k;
    if (!k.ctx) {
        mlx_vector_string in = mlx_vector_string_new();
        mlx_vector_string_append_value(in, "x");
        mlx_vector_string outn = mlx_vector_string_new();
        mlx_vector_string_append_value(outn, "out");
        k = mlx_fast_metal_kernel_new("deepseek_v41_pack4", in, outn, LLM_DSV41_PACK4_SRC,
                                      LLM_DSV41_PACK_HDR, true, false);
        mlx_vector_string_free(in); mlx_vector_string_free(outn);
    }
    return k;
}

/* x f32 [rows, D] -> uint8 [rows, D + D/32] (fp8 e4m3 grp32). D%32==0. */
static inline bool llm_dsv41_metal_pack8(mlx_array* out, mlx_array x, int rows, int D, mlx_stream s) {
    if (rows <= 0 || D <= 0 || D % 32 != 0) return false;
    mlx_fast_metal_kernel k = llm_dsv41_metal_pack8_kernel();
    if (!k.ctx) return false;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (const int[]){ rows, D + D / 32 }, 2, MLX_UINT8);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)rows * (D / 32) * 32, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, x);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) { mlx_array o0 = mlx_array_new(); if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false; else mlx_array_set(out, o0); mlx_array_free(o0); }
    mlx_vector_array_free(ins); mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    return ok;
}

/* x f32 [rows, D] -> uint8 [rows, D/2 + D/GS] (fp4; e4m3=scala e4m3, altrimenti potenza-di-2). */
static inline bool llm_dsv41_metal_pack4(mlx_array* out, mlx_array x, int rows, int D,
                                         int group_size, int e4m3, mlx_stream s) {
    if (rows <= 0 || D <= 0 || group_size <= 0 || D % group_size != 0 || D % 32 != 0) return false;
    mlx_fast_metal_kernel k = llm_dsv41_metal_pack4_kernel();
    if (!k.ctx) return false;
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (const int[]){ rows, D / 2 + D / group_size }, 2, MLX_UINT8);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)rows * (D / group_size) * 32, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", group_size);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "E4M3", e4m3 != 0);
    mlx_vector_array ins = mlx_vector_array_new();
    mlx_vector_array_append_value(ins, x);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, k, ins, cfg, s);
    bool ok = (rc == 0 && mlx_vector_array_size(outs) >= 1);
    if (ok) { mlx_array o0 = mlx_array_new(); if (mlx_vector_array_get(&o0, outs, 0) != 0) ok = false; else mlx_array_set(out, o0); mlx_array_free(o0); }
    mlx_vector_array_free(ins); mlx_vector_array_free(outs);
    mlx_fast_metal_kernel_config_free(cfg);
    return ok;
}

#endif /* LLM_DSV41_METAL_H */
