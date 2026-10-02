/* llm_mlx_moe_fused_kernel.h — kernel Metal fuso per il decode MoE (n_tokens==1, K esperti).
 *
 * §2026-09-08 notte fonda: il decode MoE oggi passa da 2-3 mlx_gather_qmm separate
 * (gate/up/down) + sigmoid/multiply elementwise per SwiGLU -- ognuna un dispatch MLX
 * generico, pensato per batch, non specializzato per il caso "un token, K esperti".
 * Questo modulo adatta (con attribuzione, licenza Apache-2.0 compatibile) i due kernel
 * Metal di mlxcel (github.com/lablup/mlxcel, src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp,
 * commento originale: "beats gather_qmm by ~3.5% on qwen3-30b-a3b") via mlx_fast_metal_kernel
 * -- stesso meccanismo gia' in produzione per GDN/GeGLU in questo file.
 *
 * Un SIMD-group (32 lane) per riga di output; grid.y = riga (0..Dff o 0..Din), grid.z =
 * slot esperto (0..K-1). Ogni lane legge un intero "pack" di pesi quantizzati alla volta
 * (vpw = 32/bits pesi per parola da 32 bit), decodifica scala/bias per gruppo, accumula
 * via simd_sum. Kernel A fonde gate+up+attivazione (SwiGLU/GeGLU) in un solo dispatch;
 * Kernel B fa la down-projection pesata per lo score dell'esperto. La somma pesata sui K
 * esperti resta un mlx_sum normale in f32 (non kernelizzata) -- nota di precisione
 * dell'originale: sommare i K contributi in f32 e arrotondare UNA sola volta al dtype
 * finale (non un arrotondamento per esperto) riduce l'errore di quantizzazione accumulato.
 *
 * Contratto: SOLO power-of-2 bits (4/8) -- 6-bit ha un layout di packing diverso (3 byte
 * per 4 pesi), non portato qui, il chiamante deve verificare bits in {4,8} ed escludere 6.
 * Precisione: RMS<5e-3 attesa (l'ordine di riduzione SIMD differisce da gather_qmm), NON
 * bit-identica -- verificare con soglia, non con confronto esatto.
 */
#ifndef LLM_MLX_MOE_FUSED_KERNEL_H
#define LLM_MLX_MOE_FUSED_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_kernel_fallback.h"

static const char llm_moe_fused_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
;

/* Kernel A: gate+up fuse + attivazione (SwiGLU act=0, GeGLU act=1) -> act_g[K,Dff] f32.
 * Input: x[Din], indices[K] (uint32), gate_w/up_w (packed uint32, [E,Dff,Din/vpw]),
 * gate_s/gate_b/up_s/up_b (T, [E,Dff,Din/group_size]). Output: act_g[K,Dff] f32. */
static const char llm_moe_gateup_kernel_src[] =
    "\n"
    "    uint lane  = thread_position_in_threadgroup.x;\n"
    "    uint f     = thread_position_in_grid.y;\n"
    "    uint eslot = thread_position_in_grid.z;\n"
    "    if (f >= (uint)Dff) return;\n"
    "    uint e = indices[eslot];\n"
    "\n"
    "    constexpr uint vpw   = 32u / bits;\n"
    "    constexpr uint wmask = (1u << bits) - 1u;\n"
    "    constexpr uint Din_p = Din / vpw;\n"
    "    constexpr uint G     = Din / group_size;\n"
    "\n"
    "    uint row = e * (uint)Dff + f;\n"
    "    const device uint* gwr = gate_w + row * Din_p;\n"
    "    const device T*    gsr = gate_s + row * G;\n"
    "    const device T*    gbr = gate_b + row * G;\n"
    "    const device uint* uwr = up_w   + row * Din_p;\n"
    "    const device T*    usr = up_s   + row * G;\n"
    "    const device T*    ubr = up_b   + row * G;\n"
    "    float g = 0.0f, u = 0.0f;\n"
    "    for (uint p = lane; p < Din_p; p += 32u) {\n"
    "        uint base = p * vpw;\n"
    "        uint grp  = base / group_size;\n"
    "        float gs = (float)gsr[grp], gb = (float)gbr[grp];\n"
    "        float us = (float)usr[grp], ub = (float)ubr[grp];\n"
    "        uint gpk = gwr[p];\n"
    "        uint upk = uwr[p];\n"
    "        for (uint j = 0; j < vpw; ++j) {\n"
    "            float xv = (float)x[base + j];\n"
    "            g += xv * ((float)((gpk >> (j * bits)) & wmask) * gs + gb);\n"
    "            u += xv * ((float)((upk >> (j * bits)) & wmask) * us + ub);\n"
    "        }\n"
    "    }\n"
    "    g = simd_sum(g);\n"
    "    u = simd_sum(u);\n"
    "    if (lane == 0u) {\n"
    "        if (act == 1) {\n"
    "            float g3 = g * g * g;\n"
    "            float inner = 0.7978845608028654f * (g + 0.044715f * g3);\n"
    "            float gelu = 0.5f * g * (1.0f + precise::tanh(clamp(inner, -20.0f, 20.0f)));\n"
    "            act_g[eslot * (uint)Dff + f] = gelu * u;\n"
    "        } else {\n"
    "            act_g[eslot * (uint)Dff + f] = (g / (1.0f + fast::exp(-g))) * u;\n"
    "        }\n"
    "    }\n"
;

/* Kernel B: down-projection pesata per score -> out[K,Din] f32 (il chiamante somma su K
 * e arrotonda al dtype finale UNA sola volta, fuori da questo kernel). Solo bits in {4,8}
 * (il caso 6-bit di mlxcel, packing a byte, non e' portato -- il chiamante deve escluderlo). */
static const char llm_moe_down_kernel_src[] =
    "\n"
    "    uint lane  = thread_position_in_threadgroup.x;\n"
    "    uint h     = thread_position_in_grid.y;\n"
    "    uint eslot = thread_position_in_grid.z;\n"
    "    if (h >= (uint)Din) return;\n"
    "    uint e = indices[eslot];\n"
    "\n"
    "    constexpr uint Gd = Dff / group_size;\n"
    "    uint row = e * (uint)Din + h;\n"
    "    const device T*     dsr = down_s + row * Gd;\n"
    "    const device T*     dbr = down_b + row * Gd;\n"
    "    const device float* a   = act_g  + eslot * (uint)Dff;\n"
    "    float d = 0.0f;\n"
    "\n"
    "    constexpr uint vpw   = 32u / bits;\n"
    "    constexpr uint wmask = (1u << bits) - 1u;\n"
    "    constexpr uint Dff_p = Dff / vpw;\n"
    "    const device uint* dwr = down_w + row * Dff_p;\n"
    "    for (uint p = lane; p < Dff_p; p += 32u) {\n"
    "        uint base = p * vpw;\n"
    "        uint grp  = base / group_size;\n"
    "        float ds = (float)dsr[grp], db = (float)dbr[grp];\n"
    "        uint dpk = dwr[p];\n"
    "        for (uint j = 0; j < vpw; ++j) {\n"
    "            d += a[base + j] * ((float)((dpk >> (j * bits)) & wmask) * ds + db);\n"
    "        }\n"
    "    }\n"
    "    d = simd_sum(d);\n"
    "    if (lane == 0u) {\n"
    "        out[eslot * (uint)Din + h] = (float)scores[eslot] * d;\n"
    "    }\n"
;

typedef struct {
    mlx_fast_metal_kernel k;
    bool built;
} llm_moe_kernel_cache;

static llm_moe_kernel_cache llm_moe_gateup_kernel_singleton = { {0}, false };
static llm_moe_kernel_cache llm_moe_down_kernel_singleton = { {0}, false };

static int llm_mlx_moe_gateup_kernel_build(void) {
    if (llm_moe_gateup_kernel_singleton.built) return 0;
    const char* in_names[] = {"x", "indices", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"};
    const char* out_names[] = {"act_g"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 8; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_moe_gateup_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llmengine_moe_gateup_fused", iv, ov, llm_moe_gateup_kernel_src, llm_moe_fused_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_moe_gateup_kernel_singleton.built = true;
    return 0;
}

static int llm_mlx_moe_down_kernel_build(void) {
    if (llm_moe_down_kernel_singleton.built) return 0;
    const char* in_names[] = {"indices", "down_w", "down_s", "down_b", "act_g", "scores"};
    const char* out_names[] = {"out"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 6; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_moe_down_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llmengine_moe_down_fused", iv, ov, llm_moe_down_kernel_src, llm_moe_fused_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_moe_down_kernel_singleton.built = true;
    return 0;
}

/* Decode MoE fuso: x [Din] (dtype T), indices [K] uint32, gate/up/down_w packed uint32
 * [E, Dff-o-Din, Din-o-Dff / vpw], gate/up/down_s/b [E, Dff-o-Din, Din-o-Dff / group_size]
 * dtype T, scores [K] f32 (gia' rinormalizzato se norm_topk_prob). out_summed: [Din] f32
 * (il chiamante fa astype al dtype finale + reshape). gate e up DEVONO condividere lo
 * stesso bits/group_size (kernel A li processa insieme in un dispatch); down puo' avere
 * bits/group_size diversi (kernel B separato) -- stesso schema di mlxcel (dots.llm1:
 * gate/up 4-bit, down 6-bit -- qui pero' SOLO power-of-2 in {4,8}, il 6-bit non e' portato).
 * Ritorna false su qualunque condizione non soddisfatta o errore -- il chiamante deve fare
 * fallback al path gather_qmm esistente, mai un errore bloccante. */
static bool llm_mlx_moe_fused_decode(
    mlx_array* out_summed,
    mlx_array x, mlx_array indices,
    mlx_array gate_w, mlx_array gate_s, mlx_array gate_b,
    mlx_array up_w, mlx_array up_s, mlx_array up_b,
    mlx_array down_w, mlx_array down_s, mlx_array down_b,
    mlx_array scores,
    int din, int dff, int k, int gu_bits, int gu_group_size,
    int d_bits, int d_group_size, int act,
    mlx_stream stream)
{
    if ((gu_bits != 4 && gu_bits != 8) || (d_bits != 4 && d_bits != 8)) return false;
    if (din <= 0 || dff <= 0 || k <= 0 || gu_group_size <= 0 || d_group_size <= 0) return false;
    if (din % (32 / gu_bits) != 0 || dff % (32 / d_bits) != 0) return false;
    if (din % gu_group_size != 0 || dff % d_group_size != 0) return false;
    /* Il kernel dichiara gate_s/gate_b/up_s/up_b (e down_s/down_b) come `const device T*`, con T
     * preso dal solo x: se il dtype delle scale del checkpoint DIVERSO da quello di x, metal non
     * compila la libreria e l'eccezione C++ esce dalla API C, uccidendo il processo (misurato su
     * Qwen3-Coder-30B-A3B-8bit con bf16=off: x fp32, scale bf16 -> exit 255, nessuna fallback).
     * Il contratto di "fallback silenzioso" di questo file va dunque onorato QUI, prima di
     * qualunque build/apply. */
    const mlx_dtype T = mlx_array_dtype(x);
    if (mlx_array_dtype(gate_s) != T || mlx_array_dtype(gate_b) != T
        || mlx_array_dtype(up_s) != T || mlx_array_dtype(up_b) != T
        || mlx_array_dtype(down_s) != T || mlx_array_dtype(down_b) != T) {
        LLM_KFALLBACK("moe_fused", "dtype scale/bias != attivazione (es. con LLM_F32_SCALES le scale sono f32)");
        return false;
    }
    if (llm_mlx_moe_gateup_kernel_build() != 0) return false;
    if (llm_mlx_moe_down_kernel_build() != 0) return false;

    /* simdgroups/threadgroup -- 8 era il valore mlxcel per qwen3-30b-a3b su M1 Ultra, non
     * detto sia ottimale su altro hardware. Override runtime per sweep senza ricompilare. */
    static int s_sgy = 0;
    if (s_sgy == 0) {
        const char* e = llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_SGY");
        int v = e ? atoi(e) : 8;
        s_sgy = (v >= 1 && v <= 32) ? v : 8;
    }
    const int sgy = s_sgy;

    /* Kernel A: gate+up+attivazione -> act_g [K, Dff] f32 */
    mlx_array act_g = mlx_array_new();
    {
        mlx_vector_array inputs = mlx_vector_array_new();
        mlx_vector_array_append_value(inputs, x);
        mlx_vector_array_append_value(inputs, indices);
        mlx_vector_array_append_value(inputs, gate_w);
        mlx_vector_array_append_value(inputs, gate_s);
        mlx_vector_array_append_value(inputs, gate_b);
        mlx_vector_array_append_value(inputs, up_w);
        mlx_vector_array_append_value(inputs, up_s);
        mlx_vector_array_append_value(inputs, up_b);

        mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
        mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", T);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Din", din);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dff", dff);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "bits", gu_bits);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "group_size", gu_group_size);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "act", act);
        int actg_shape[2] = {k, dff};
        mlx_fast_metal_kernel_config_add_output_arg(cfg, actg_shape, 2, MLX_FLOAT32);
        int dff_pad = ((dff + sgy - 1) / sgy) * sgy;
        mlx_fast_metal_kernel_config_set_grid(cfg, 32, dff_pad, k);
        mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgy, 1);
        if (getenv("LLMENGINE_MOE_FUSE_MOE_VERBOSE")) mlx_fast_metal_kernel_config_set_verbose(cfg, true);

        mlx_vector_array outputs = mlx_vector_array_new();
        int rc = mlx_fast_metal_kernel_apply(&outputs, llm_moe_gateup_kernel_singleton.k, inputs, cfg, stream);
        bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 1;
        if (ok) ok = mlx_vector_array_get(&act_g, outputs, 0) == 0;
        mlx_fast_metal_kernel_config_free(cfg);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(outputs);
        if (!ok) { mlx_array_free(act_g); return false; }
    }

    /* Kernel B: down-projection pesata per score -> partial [K, Din] f32 */
    mlx_array partial = mlx_array_new();
    {
        mlx_vector_array inputs = mlx_vector_array_new();
        mlx_vector_array_append_value(inputs, indices);
        mlx_vector_array_append_value(inputs, down_w);
        mlx_vector_array_append_value(inputs, down_s);
        mlx_vector_array_append_value(inputs, down_b);
        mlx_vector_array_append_value(inputs, act_g);
        mlx_vector_array_append_value(inputs, scores);

        mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
        mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", T);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Din", din);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dff", dff);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "bits", d_bits);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "group_size", d_group_size);
        int out_shape[2] = {k, din};
        mlx_fast_metal_kernel_config_add_output_arg(cfg, out_shape, 2, MLX_FLOAT32);
        int din_pad = ((din + sgy - 1) / sgy) * sgy;
        mlx_fast_metal_kernel_config_set_grid(cfg, 32, din_pad, k);
        mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgy, 1);

        mlx_vector_array outputs = mlx_vector_array_new();
        int rc = mlx_fast_metal_kernel_apply(&outputs, llm_moe_down_kernel_singleton.k, inputs, cfg, stream);
        bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 1;
        if (ok) ok = mlx_vector_array_get(&partial, outputs, 0) == 0;
        mlx_fast_metal_kernel_config_free(cfg);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(outputs);
        if (!ok) { mlx_array_free(act_g); mlx_array_free(partial); return false; }
    }

    /* Somma sui K esperti in f32 (un solo arrotondamento al dtype finale, fatto dal
     * chiamante) -- non kernelizzata, un mlx_sum normale basta qui. */
    bool ok = mlx_sum_axis(out_summed, partial, /*axis=*/0, /*keepdims=*/false, stream) == 0;
    mlx_array_free(act_g);
    mlx_array_free(partial);
    return ok;
}

/* ============================================================================
 * Variante MXFP4 (gpt-oss) -- §2026-09-09 "procedi": il fused kernel affine
 * sopra NON puo' servire gpt-oss per tre motivi strutturali: (1) le scale MXFP4
 * sono uint8 e8m0, non dtype T -> il guard "dtype(scale)==T" del path affine
 * darebbe sempre fallback; (2) MXFP4 non ha bias di quantizzazione (dequant =
 * e8m0*scale * e2m1(nibble), nessun addendo), mentre l'affine ha scale+bias;
 * (3) gpt-oss aggiunge un BIAS LINEARE per-esperto (gate/up/down .bias, dtype T)
 * DOPO ogni matmul e usa l'attivazione "oml2" CLAMPATA (alpha 1.702, clip +/-7),
 * non SwiGLU/GeGLU. Quindi: kernel dedicati, con gli helper e8m0/e2m1 riprodotti
 * bit-esatti dai sorgenti MLX (fp8.h operator float, fp4.h half-trick) gia'
 * validati in native/llmmetal_gemv/llmmetal_gemv_mxfp4.h. Packing dei pesi
 * identico all'affine: uint32, 8 nibble/parola, nibble basso = colonna pari
 * (MLX quantized.cpp packs_per_int = 8/bits). group_size fisso 32 (formato MX).
 * ========================================================================== */

static const char llm_moe_mxfp4_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    /* E8M0 (byte) -> float: identico a mlx fp8.h operator float, caso byte==0 incluso. */
    "static inline float e8m0_f(uint8_t b) {\n"
    "    uint s = (b == 0u) ? 0x400000u : (uint(b) << 23);\n"
    "    return as_type<float>(s);\n"
    "}\n"
    /* E2M1 (nibble basso a 3 bit + segno dal bit 3) -> float: identico a mlx fp4.h. */
    "static inline float e2m1_f(uint8_t nib) {\n"
    "    half v = as_type<half>(ushort((nib & 7u) << 9)) * 16384.0h;\n"
    "    return (nib & 8u) ? -float(v) : float(v);\n"
    "}\n"
;

/* Kernel A (mxfp4): gate+up fuse + attivazione clamped gpt-oss -> act_g[K,Dff] dtype T.
 * Fedelta' bit vs eager: arrotonda a T dopo l'accumulo (uscita gather), dopo il bias e a
 * ogni passo dell'attivazione (come llm_mlx_swiglu_gptoss_kernel.h). Template T = dtype
 * dell'attivazione (= dtype di x e dei bias). Input: x[Din] (T), indices[K] (uint32),
 * gate_w/up_w (uint32 packed [E,Dff,Din/8]), gate_s/up_s (uint8 e8m0 [E,Dff,Din/32]),
 * gate_lb/up_lb (T [E,Dff] bias lineare). */
static const char llm_moe_gateup_mxfp4_kernel_src[] =
    "\n"
    "    uint lane  = thread_position_in_threadgroup.x;\n"
    "    uint f     = thread_position_in_grid.y;\n"
    "    uint eslot = thread_position_in_grid.z;\n"
    "    if (f >= (uint)Dff) return;\n"
    "    uint e = indices[eslot];\n"
    "\n"
    "    constexpr uint vpw   = 8u;        // 32/bits, bits=4 (mxfp4)\n"
    "    constexpr uint wmask = 15u;\n"
    "    constexpr uint Din_p = Din / vpw;\n"
    "    constexpr uint G     = Din / 32u; // group_size fisso 32\n"
    "\n"
    "    uint row = e * (uint)Dff + f;\n"
    "    const device uint*    gwr = gate_w + (uint)row * Din_p;\n"
    "    const device uint*    uwr = up_w   + (uint)row * Din_p;\n"
    "    const device uint8_t* gsr = gate_s + (uint)row * G;\n"
    "    const device uint8_t* usr = up_s   + (uint)row * G;\n"
    "    float g = 0.0f, u = 0.0f;\n"
    "    for (uint p = lane; p < Din_p; p += 32u) {\n"
    "        uint base = p * vpw;\n"
    "        uint grp  = base / 32u;\n"
    "        float gs = e8m0_f(gsr[grp]);\n"
    "        float us = e8m0_f(usr[grp]);\n"
    "        uint gpk = gwr[p];\n"
    "        uint upk = uwr[p];\n"
    /* Fedelta' bit: replica qdot() di fp_quantized.h (due gruppi di 4 termini ciascuno,
     * somma left-assoc dentro il gruppo, scalati una volta per blocco di 8). */
    "        float ga = 0.0f, ua = 0.0f;\n"
    "        for (uint i = 0u; i < 2u; ++i) {\n"
    "            uint gw = (gpk >> (i * 16u)) & 0xFFFFu;\n"
    "            uint uw = (upk >> (i * 16u)) & 0xFFFFu;\n"
    "            float x0 = (float)x[base + i * 4u];\n"
    "            float x1 = (float)x[base + i * 4u + 1u];\n"
    "            float x2 = (float)x[base + i * 4u + 2u];\n"
    "            float x3 = (float)x[base + i * 4u + 3u];\n"
    "            ga += x0 * e2m1_f((uint8_t)(gw & 15u)) + x1 * e2m1_f((uint8_t)((gw >> 4u) & 15u))\n"
    "                + x2 * e2m1_f((uint8_t)((gw >> 8u) & 15u)) + x3 * e2m1_f((uint8_t)((gw >> 12u) & 15u));\n"
    "            ua += x0 * e2m1_f((uint8_t)(uw & 15u)) + x1 * e2m1_f((uint8_t)((uw >> 4u) & 15u))\n"
    "                + x2 * e2m1_f((uint8_t)((uw >> 8u) & 15u)) + x3 * e2m1_f((uint8_t)((uw >> 12u) & 15u));\n"
    "        }\n"
    "        g += gs * ga;\n"
    "        u += us * ua;\n"
    "    }\n"
    "    g = simd_sum(g);\n"
    "    u = simd_sum(u);\n"
    "    if (lane == 0u) {\n"
    /* Fedelta' bit: l'eager gather_qmm arrotonda l'uscita a T, poi l'add del bias e
     * OGNI passo dell'attivazione arrotondano a T (stessa sequenza di
     * llm_mlx_swiglu_gptoss_kernel.h). Senza questi round il kernel accumulava f32
     * fino alla fine -> rel_rms ~0,005 e IDS divergenti (LLMENGINE_MOE_FUSE_MOE_MXFP4 bloccata). */
    "        T gg = (T)g;\n"
    "        T uu = (T)u;\n"
    "        T gb = (T)((float)gg + (float)gate_lb[row]);\n"
    "        T ub = (T)((float)uu + (float)up_lb[row]);\n"
    "        T cg = (T)min((float)gb, 7.0f);\n"
    "        T cu = (T)max(min((float)ub, 7.0f), -7.0f);\n"
    "        T alpha = (T)1.702f;\n"
    "        T ag = (T)((float)cg * (float)alpha);\n"
    "        T sy = 1 / (1 + metal::exp(metal::abs(ag)));\n"   /* Sigmoid di MLX in T (parita' mlx-lm) */
    "        T s = (ag < 0) ? sy : 1 - sy;\n"
    "        T sg = (T)((float)s * (float)cg);\n"
    "        T u1 = (T)((float)cu + 1.0f);\n"
    "        act_g[eslot * (uint)Dff + f] = (T)((float)sg * (float)u1);\n"
    "    }\n"
;

/* Kernel B (mxfp4): down-projection pesata per score + bias lineare -> out[K,Din] dtype T.
 * Fedelta' bit: arrotonda a T dopo l'accumulo (uscita gather) e dopo il bias; moltiplica
 * per lo score in T. Input: indices[K] uint32, down_w uint32 [E,Din,Dff/8], down_s uint8
 * e8m0 [E,Din,Dff/32], down_lb T [E,Din], act_g T [K,Dff], scores T [K]. */
static const char llm_moe_down_mxfp4_kernel_src[] =
    "\n"
    "    uint lane  = thread_position_in_threadgroup.x;\n"
    "    uint h     = thread_position_in_grid.y;\n"
    "    uint eslot = thread_position_in_grid.z;\n"
    "    if (h >= (uint)Din) return;\n"
    "    uint e = indices[eslot];\n"
    "\n"
    "    constexpr uint vpw   = 8u;\n"
    "    constexpr uint wmask = 15u;\n"
    "    constexpr uint Dff_p = Dff / vpw;\n"
    "    constexpr uint Gd    = Dff / 32u;\n"
    "    uint row = e * (uint)Din + h;\n"
    "    const device uint*    dwr = down_w + (uint)row * Dff_p;\n"
    "    const device uint8_t* dsr = down_s + (uint)row * Gd;\n"
    "    const device T*       a   = act_g  + eslot * (uint)Dff;\n"
    "    float d = 0.0f;\n"
    "    for (uint p = lane; p < Dff_p; p += 32u) {\n"
    "        uint base = p * vpw;\n"
    "        uint grp  = base / 32u;\n"
    "        float ds = e8m0_f(dsr[grp]);\n"
    "        uint dpk = dwr[p];\n"
    /* Fedelta' bit: replica qdot() di fp_quantized.h (gruppi di 4). */
    "        float da = 0.0f;\n"
    "        for (uint i = 0u; i < 2u; ++i) {\n"
    "            uint dw = (dpk >> (i * 16u)) & 0xFFFFu;\n"
    "            float a0 = (float)a[base + i * 4u];\n"
    "            float a1 = (float)a[base + i * 4u + 1u];\n"
    "            float a2 = (float)a[base + i * 4u + 2u];\n"
    "            float a3 = (float)a[base + i * 4u + 3u];\n"
    "            da += a0 * e2m1_f((uint8_t)(dw & 15u)) + a1 * e2m1_f((uint8_t)((dw >> 4u) & 15u))\n"
    "                + a2 * e2m1_f((uint8_t)((dw >> 8u) & 15u)) + a3 * e2m1_f((uint8_t)((dw >> 12u) & 15u));\n"
    "        }\n"
    "        d += ds * da;\n"
    "    }\n"
    "    d = simd_sum(d);\n"
    "    if (lane == 0u) {\n"
    /* Fedelta' bit: gather_qmm arrotonda a T, poi add bias (T) e multiply score (T). */
    "        T dd = (T)d;\n"
    "        T db = (T)((float)dd + (float)down_lb[row]);\n"
    "        out[eslot * (uint)Din + h] = (T)((float)db * (float)scores[eslot]);\n"
    "    }\n"
;

static llm_moe_kernel_cache llm_moe_gateup_mxfp4_kernel_singleton = { {0}, false };
static llm_moe_kernel_cache llm_moe_down_mxfp4_kernel_singleton = { {0}, false };

static int llm_mlx_moe_gateup_mxfp4_kernel_build(void) {
    if (llm_moe_gateup_mxfp4_kernel_singleton.built) return 0;
    const char* in_names[] = {"x", "indices", "gate_w", "gate_s", "up_w", "up_s", "gate_lb", "up_lb"};
    const char* out_names[] = {"act_g"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 8; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_moe_gateup_mxfp4_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llmengine_moe_gateup_fused_mxfp4", iv, ov, llm_moe_gateup_mxfp4_kernel_src, llm_moe_mxfp4_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_moe_gateup_mxfp4_kernel_singleton.built = true;
    return 0;
}

static int llm_mlx_moe_down_mxfp4_kernel_build(void) {
    if (llm_moe_down_mxfp4_kernel_singleton.built) return 0;
    const char* in_names[] = {"indices", "down_w", "down_s", "down_lb", "act_g", "scores"};
    const char* out_names[] = {"out"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 6; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    llm_moe_down_mxfp4_kernel_singleton.k = mlx_fast_metal_kernel_new(
        "llmengine_moe_down_fused_mxfp4", iv, ov, llm_moe_down_mxfp4_kernel_src, llm_moe_mxfp4_kernel_header,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    llm_moe_down_mxfp4_kernel_singleton.built = true;
    return 0;
}

/* Decode MoE fuso MXFP4 (gpt-oss): x [Din] dtype T, indices [K] uint32 (gia' appiattiti,
 * ordine partition), gate/up/down_w uint32 packed [E, (Dff|Din), (Din|Dff)/8], gate/up/down_s
 * uint8 e8m0 [E, (Dff|Din), (Din|Dff)/32], gate_lb [E,Dff] / up_lb [E,Dff] / down_lb [E,Din]
 * dtype T (bias lineare per-esperto), scores [K] f32 (softmax top-k, gia' normalizzata).
 * group_size fissato a 32 (formato MX), 4-bit (vpw=8). out_summed: [Din] f32 (chiamante fa
 * astype + reshape). Ritorna false su qualunque condizione non soddisfatta o errore -- il
 * chiamante DEVE fare fallback silenzioso al path gather_qmm, mai un errore bloccante. */
static bool llm_mlx_moe_fused_decode_mxfp4(
    mlx_array* out_summed,
    mlx_array x, mlx_array indices,
    mlx_array gate_w, mlx_array gate_s, mlx_array gate_lb,
    mlx_array up_w,   mlx_array up_s,   mlx_array up_lb,
    mlx_array down_w, mlx_array down_s, mlx_array down_lb,
    mlx_array scores,
    int din, int dff, int k,
    mlx_stream stream)
{
    if (din <= 0 || dff <= 0 || k <= 0) return false;
    if (din % 8 != 0 || dff % 8 != 0) return false;      /* vpw=8 (4-bit) */
    if (din % 32 != 0 || dff % 32 != 0) return false;    /* group_size=32 */
    const mlx_dtype T = mlx_array_dtype(x);
    /* Contratti di dtype: se NON soddisfatti, metal NON compila (param tipo concreto dal
     * codegen) e l'eccezione C++ esce dalla API C. Guard QUI, prima di ogni build/apply. */
    if (mlx_array_dtype(gate_w) != MLX_UINT32 || mlx_array_dtype(up_w) != MLX_UINT32
        || mlx_array_dtype(down_w) != MLX_UINT32) return false;
    if (mlx_array_dtype(gate_s) != MLX_UINT8 || mlx_array_dtype(up_s) != MLX_UINT8
        || mlx_array_dtype(down_s) != MLX_UINT8) return false;
    if (mlx_array_dtype(gate_lb) != T || mlx_array_dtype(up_lb) != T || mlx_array_dtype(down_lb) != T) return false;
    if (mlx_array_dtype(indices) != MLX_UINT32) return false;
    if (mlx_array_dtype(scores) != T) return false;   /* score nello stesso dtype dell'attivazione (fedelta' bit) */
    if (llm_mlx_moe_gateup_mxfp4_kernel_build() != 0) return false;
    if (llm_mlx_moe_down_mxfp4_kernel_build() != 0) return false;

    static int s_sgy = 0;
    if (s_sgy == 0) {
        const char* e = llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_SGY");
        int v = e ? atoi(e) : 8;
        s_sgy = (v >= 1 && v <= 32) ? v : 8;
    }
    const int sgy = s_sgy;

    /* Kernel A: gate+up+attivazione clamped -> act_g [K, Dff] T */
    mlx_array act_g = mlx_array_new();
    {
        mlx_vector_array inputs = mlx_vector_array_new();
        mlx_vector_array_append_value(inputs, x);
        mlx_vector_array_append_value(inputs, indices);
        mlx_vector_array_append_value(inputs, gate_w);
        mlx_vector_array_append_value(inputs, gate_s);
        mlx_vector_array_append_value(inputs, up_w);
        mlx_vector_array_append_value(inputs, up_s);
        mlx_vector_array_append_value(inputs, gate_lb);
        mlx_vector_array_append_value(inputs, up_lb);

        mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
        /* Template "T" = dtype dell'attivazione, usato dai round di fedelta' bit nel corpo.
         * Gl'interi Din/Dff sono gli unici altri parametri. */
        mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", T);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Din", din);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dff", dff);
        int actg_shape[2] = {k, dff};
        mlx_fast_metal_kernel_config_add_output_arg(cfg, actg_shape, 2, T);
        int dff_pad = ((dff + sgy - 1) / sgy) * sgy;
        mlx_fast_metal_kernel_config_set_grid(cfg, 32, dff_pad, k);
        mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgy, 1);
        if (getenv("LLMENGINE_MOE_FUSE_MOE_VERBOSE")) mlx_fast_metal_kernel_config_set_verbose(cfg, true);

        mlx_vector_array outputs = mlx_vector_array_new();
        int rc = mlx_fast_metal_kernel_apply(&outputs, llm_moe_gateup_mxfp4_kernel_singleton.k, inputs, cfg, stream);
        bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 1;
        if (ok) ok = mlx_vector_array_get(&act_g, outputs, 0) == 0;
        mlx_fast_metal_kernel_config_free(cfg);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(outputs);
        if (!ok) { mlx_array_free(act_g); return false; }
    }

    /* Kernel B: down+score+bias -> partial [K, Din] T */
    mlx_array partial = mlx_array_new();
    {
        mlx_vector_array inputs = mlx_vector_array_new();
        mlx_vector_array_append_value(inputs, indices);
        mlx_vector_array_append_value(inputs, down_w);
        mlx_vector_array_append_value(inputs, down_s);
        mlx_vector_array_append_value(inputs, down_lb);
        mlx_vector_array_append_value(inputs, act_g);
        mlx_vector_array_append_value(inputs, scores);

        mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
        mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", T);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Din", din);
        mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dff", dff);
        int out_shape[2] = {k, din};
        mlx_fast_metal_kernel_config_add_output_arg(cfg, out_shape, 2, T);
        int din_pad = ((din + sgy - 1) / sgy) * sgy;
        mlx_fast_metal_kernel_config_set_grid(cfg, 32, din_pad, k);
        mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgy, 1);

        mlx_vector_array outputs = mlx_vector_array_new();
        int rc = mlx_fast_metal_kernel_apply(&outputs, llm_moe_down_mxfp4_kernel_singleton.k, inputs, cfg, stream);
        bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 1;
        if (ok) ok = mlx_vector_array_get(&partial, outputs, 0) == 0;
        mlx_fast_metal_kernel_config_free(cfg);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(outputs);
        if (!ok) { mlx_array_free(act_g); mlx_array_free(partial); return false; }
    }

    bool ok = mlx_sum_axis(out_summed, partial, /*axis=*/0, /*keepdims=*/false, stream) == 0;
    mlx_array_free(act_g);
    mlx_array_free(partial);
    return ok;
}

#endif /* LLM_MLX_MOE_FUSED_KERNEL_H */
