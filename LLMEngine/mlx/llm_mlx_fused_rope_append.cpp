/* Fused RoPE + Q/K/V relayout for the one-token decode path.
 *
 * Adapted from mlxcel/src/lib/mlx-cpp/turbo/fused_rope_append.cpp
 * Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
 * Licensed under the Apache License, Version 2.0.
 *
 * mlxcel consumes a single fused-QKV projection. LLMEnv deliberately keeps
 * the three quantized projections independent, so this adaptation accepts
 * three row-contiguous arrays and selects the corresponding input in-kernel.
 * It retains MLX's RoPE arithmetic and produces dense cache layout
 * [B,H,L,D], avoiding reshape + transpose + two generic RoPE dispatches.
 */
#include "llm_mlx_fused_rope_append.h"

#include <cmath>
#include <cstdio>
#include <mutex>

namespace {

constexpr const char* kHeader = R"(
#include <metal_stdlib>
using namespace metal;
)";

constexpr const char* kSource = R"(
    uint p = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint z = thread_position_in_grid.z;

    const uint dim = (uint)HeadDim;
    const uint half_dim = dim / 2u;
    const uint rdims = (uint)RopeDims;
    const uint rhalf = rdims / 2u;
    const uint hq = (uint)NHeadsQ;
    const uint hkv = (uint)NHeadsKV;
    const uint batch = (uint)q_shape[0];
    const uint seq = (uint)q_shape[1];
    const uint slots = hq + 2u * hkv;

    if (p >= half_dim || t >= seq || z >= batch * slots) return;

    uint b = z / slots;
    uint slot = z - b * slots;
    uint kind;
    uint h;
    const device T* src;
    uint heads;
    if (slot < hq) {
        kind = 0u; h = slot; src = q; heads = hq;
    } else if (slot < hq + hkv) {
        kind = 1u; h = slot - hq; src = k; heads = hkv;
    } else {
        kind = 2u; h = slot - hq - hkv; src = v; heads = hkv;
    }

    ulong in_base = ((ulong)b * seq + t) * (ulong)(heads * dim)
        + (ulong)h * dim;
    ulong out_base = (((ulong)b * heads + h) * seq + t) * dim;

    if (kind == 2u) {
        uint j = 2u * p;
        v_out[out_base + j] = src[in_base + j];
        v_out[out_base + j + 1u] = src[in_base + j + 1u];
        return;
    }

    bool rotate = p < rhalf;
    uint i1;
    uint i2;
    if (rotate) {
        if (Traditional) { i1 = 2u * p; i2 = i1 + 1u; }
        else { i1 = p; i2 = p + rhalf; }
    } else {
        i1 = rdims + 2u * (p - rhalf);
        i2 = i1 + 1u;
    }

    float x1 = (float)src[in_base + i1];
    float x2 = (float)src[in_base + i2];
    float r1 = x1;
    float r2 = x2;
    if (rotate) {
        float inv_freq;
        if (UseFreqs) {
            /* 1:1 con rope_single_freqs in mlx rope.metal: inv_freq = 1/freqs[p] */
            inv_freq = 1.0f / rope_freqs[p];
        } else {
            /* 1:1 con rope_single: d = p/grid.x, inv_freq = exp2(-d*log2(base)) */
            float d = (float)p / (float)rhalf;
            inv_freq = metal::exp2(-d * rope_params[0]);
        }
        float pos = (float)((int)t + positions_base[0]);
        float theta = rope_params[1] * pos * inv_freq;
        float c = metal::fast::cos(theta);
        float s = metal::fast::sin(theta);
        r1 = x1 * c - x2 * s;
        r2 = x1 * s + x2 * c;
    }

    if (kind == 0u) {
        q_out[out_base + i1] = (T)r1;
        q_out[out_base + i2] = (T)r2;
    } else {
        k_out[out_base + i1] = (T)r1;
        k_out[out_base + i2] = (T)r2;
    }
)";

std::mutex g_kernel_mutex;
mlx_fast_metal_kernel g_kernel = {nullptr};

bool ensure_kernel() {
    std::lock_guard<std::mutex> guard(g_kernel_mutex);
    if (g_kernel.ctx != nullptr) return true;
    const char* inputs[] = {"q", "k", "v", "rope_params", "positions_base", "rope_freqs"};
    const char* outputs[] = {"q_out", "k_out", "v_out"};
    mlx_vector_string iv = mlx_vector_string_new();
    mlx_vector_string ov = mlx_vector_string_new();
    for (const char* name : inputs) mlx_vector_string_append_value(iv, name);
    for (const char* name : outputs) mlx_vector_string_append_value(ov, name);
    g_kernel = mlx_fast_metal_kernel_new(
        "llm_fused_rope_qkv_layout", iv, ov, kSource, kHeader,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    return g_kernel.ctx != nullptr;
}

/* Il rifiuto deve restare visibile: il chiamante ripiega in silenzio sul path
 * originale, quindi senza questo log una geometria non supportata e' invisibile.
 * Una sola riga per processo (verrebbe ripetuta per ogni layer di ogni token). */
std::mutex g_reason_mutex;
const char* g_reject_reason = nullptr;
void reject_once(const char* why) {
    std::lock_guard<std::mutex> guard(g_reason_mutex);
    if (g_reject_reason != nullptr) return;
    g_reject_reason = why;
    fprintf(stderr, "[llm_fused_rope] rejected: %s\n", why);
}

}  // namespace

#define LLM_REJ(why) do { reject_once(why); return false; } while (0)

extern "C" bool llm_mlx_fused_rope_append(
    mlx_array* q_out, mlx_array* k_out, mlx_array* v_out,
    mlx_array q, mlx_array k, mlx_array v,
    int num_heads, int num_kv_heads, int head_dim, int rope_dims,
    float rope_base, float rope_scale, bool traditional,
    mlx_array freqs,
    int positions_base, mlx_stream stream) {
    const bool use_freqs = (freqs.ctx != nullptr);
    if (q_out == nullptr || k_out == nullptr || v_out == nullptr ||
        q.ctx == nullptr || k.ctx == nullptr || v.ctx == nullptr)
        LLM_REJ("null handle");
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || (head_dim & 1) != 0 ||
        rope_dims <= 0 || rope_dims > head_dim || (rope_dims & 1) != 0)
        LLM_REJ("unsupported head/rope dims");
    if (!use_freqs && rope_base <= 0.0f) LLM_REJ("no freqs and rope_base<=0");
    if (mlx_array_ndim(q) != 3 || mlx_array_ndim(k) != 3 || mlx_array_ndim(v) != 3)
        LLM_REJ("q/k/v not rank-3");
    if (mlx_array_dtype(q) != mlx_array_dtype(k) || mlx_array_dtype(q) != mlx_array_dtype(v))
        LLM_REJ("mixed q/k/v dtypes");

    /* Tabella frequenze custom (llama3/yarn/proportional): stessa convenzione di
     * mlx_fast_rope -> inv_freq[i] = 1/freqs[i], float32 [>= rope_dims/2]. */
    if (use_freqs) {
        if (mlx_array_ndim(freqs) != 1 || mlx_array_dtype(freqs) != MLX_FLOAT32)
            LLM_REJ("freqs not rank-1 float32");
        const int* fsh = mlx_array_shape(freqs);
        if (fsh == nullptr || fsh[0] < rope_dims / 2) LLM_REJ("freqs table too short");
    }

    const int* qs = mlx_array_shape(q);
    const int* ks = mlx_array_shape(k);
    const int* vs = mlx_array_shape(v);
    if (qs == nullptr || ks == nullptr || vs == nullptr) LLM_REJ("null shape");
    if (qs[0] != ks[0] || qs[0] != vs[0] || qs[1] != ks[1] || qs[1] != vs[1])
        LLM_REJ("q/k/v batch/seq mismatch");
    if (qs[2] != num_heads * head_dim || ks[2] != num_kv_heads * head_dim ||
        vs[2] != num_kv_heads * head_dim)
        LLM_REJ("projection width != heads*head_dim");
    if (!ensure_kernel()) LLM_REJ("metal kernel registration failed");


    /* Con tabella freqs attiva rope_base puo' essere 0 (base non usata): log2(0)
     * -inf non deve finire nel buffer, anche se il ramo UseFreqs=false e' morto. */
    const float params[2] = {rope_base > 0.0f ? std::log2(rope_base) : 0.0f, rope_scale};
    const int param_shape[1] = {2};
    const int pos_shape[1] = {1};
    static const float kDummyFreq = 1.0f; /* 1.0 -> inv_freq 1.0, mai letto */
    mlx_array rope_params = mlx_array_new_data(params, param_shape, 1, MLX_FLOAT32);
    mlx_array pos = mlx_array_new_data(&positions_base, pos_shape, 1, MLX_INT32);
    mlx_array freqs_in = use_freqs ? freqs
                                   : mlx_array_new_data(&kDummyFreq, pos_shape, 1, MLX_FLOAT32);
    bool own_dummy_freqs = !use_freqs;
    mlx_array ins_data[6] = {q, k, v, rope_params, pos, freqs_in};
    mlx_vector_array inputs = mlx_vector_array_new_data(ins_data, 6);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_dtype dtype = mlx_array_dtype(q);
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HeadDim", head_dim);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RopeDims", rope_dims);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHeadsQ", num_heads);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHeadsKV", num_kv_heads);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "Traditional", traditional);
    mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "UseFreqs", use_freqs);
    const int qshape[4] = {qs[0], num_heads, qs[1], head_dim};
    const int kvshape[4] = {qs[0], num_kv_heads, qs[1], head_dim};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, qshape, 4, dtype);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, kvshape, 4, dtype);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, kvshape, 4, dtype);
    mlx_fast_metal_kernel_config_set_grid(
        cfg, head_dim / 2, qs[1], qs[0] * (num_heads + 2 * num_kv_heads));
    mlx_fast_metal_kernel_config_set_thread_group(
        cfg, head_dim / 2 < 64 ? head_dim / 2 : 64, 1, 1);

    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, g_kernel, inputs, cfg, stream);
    bool ok = rc == 0 && mlx_vector_array_size(outputs) == 3;
    mlx_array qo = mlx_array_new(), ko = mlx_array_new(), vo = mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&qo, outputs, 0) == 0 &&
                 mlx_vector_array_get(&ko, outputs, 1) == 0 &&
                 mlx_vector_array_get(&vo, outputs, 2) == 0;
    if (ok) ok = mlx_array_set(q_out, qo) == 0 &&
                 mlx_array_set(k_out, ko) == 0 && mlx_array_set(v_out, vo) == 0;

    mlx_array_free(qo); mlx_array_free(ko); mlx_array_free(vo);
    mlx_vector_array_free(outputs);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_array_free(rope_params); mlx_array_free(pos);
    if (own_dummy_freqs) mlx_array_free(freqs_in);
    return ok;
}
