/* llm_mlx_kv_quant_sdpa_kernel.h — kernel Metal "dequant-on-read" per l'attenzione KV-quant
 * (decode, Lq=1). 2026-09-13/14.
 *
 * Legge la KV cold QUANTIZZATA e la dequantizza IN-KERNEL con online-softmax (flash-decode),
 * fusa col recente FP16: il cold non viene mai materializzato.
 *
 * AFFERMAZIONE PARAMETRICA: K e V possono avere **bit diversi** (HETERO-KV, es. K8V4).
 * I bit sono template arg (BK/BV) — costanti per modello, non per step. L'unpacking segue il
 * layout affine di MLX = flusso di bit little-endian continuo: valore d ai bit [d*b, d*b+b).
 * (Verificato in mlx/backend/metal/kernels/quantized.h `dequantize`, bits 2/3/4/5/6/8.)
 *
 * Due dispatch (split-K + combine). NB: `mlx_fast_metal_kernel_config_set_grid` vuole il n. di
 * **THREAD** (non threadgroup): `nh*32`. NC/CAP sono input runtime (crescono ogni step; da
 * template arg farebbero ricompilare il kernel ad ogni token).
 *
 * GQA via kh = qh*nkv/nh. Contratto: hd%32==0, group_size=64, affine. Prefill non coperto. */
#ifndef LLM_MLX_KV_QUANT_SDPA_KERNEL_H
#define LLM_MLX_KV_QUANT_SDPA_KERNEL_H

#include "mlx/c/mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <stdio.h>
#include <stdlib.h>

static const char llm_kvq_sdpa_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "// dequant: scale*q + bias (q con segno quando Sym = simmetrico Splash, bias 0)\n"
    "static inline float llm_kv_dq(int q, float s, float b) { return s * (float)q + b; }\n"
    "// estrae il valore d dal flusso di bit (bits=b) stile MLX (little-endian continuo).\n"
    "// b==4 usa il percorso veloce a word (nibble). Sym: int8 SIM metrico (segno) invece\n"
    "// dell'offset affine [0,2^b-1] (Splash: q in [-127,127] con bias 0).\n"
    "template <uint B, bool Sym>\n"
    "static inline int llm_kv_val(const device uint* w, const device uint8_t* wb, uint d) {\n"
    "    if (B == 4u) { uint word = w[d >> 3]; return (int)((word >> ((d & 7u) * 4u)) & 15u); }\n"
    "    if (B == 8u) { return Sym ? (int)(int8_t)wb[d] : (int)wb[d]; }\n"
    "    uint bit = d * B, b0 = bit >> 3, sh = bit & 7u;\n"
    "    uint v = (uint)wb[b0] >> sh;\n"
    "    if (sh + B > 8u) v |= (uint)wb[b0 + 1u] << (8u - sh);\n"
    "    return (int)(v & ((1u << B) - 1u));\n"
    "}\n"
;

/* Un threadgroup (32 lane) per (query head, split): parziali stabilizzati. */
static const char llm_kvq_sdpa_kernel_src[] =
    "\n"
    "    uint lane  = thread_position_in_threadgroup.x;\n"
    "    uint qg    = threadgroup_position_in_grid.x;\n"
    "    uint split = threadgroup_position_in_grid.y;\n"
    "    constexpr uint DPL = Dhead / 32u;\n"
    "    constexpr uint G = GK;\n"
    "    constexpr uint KWB = (Dhead * BK) / 8u;\n"
    "    constexpr uint VWB = (Dhead * BV) / 8u;\n"
    "    // W3 TQ-tile: il threadgroup gestisce TQ query dello STESSO qh_in CONDIVIDENDO il load K/V\n"
    "    // (prima: 1 threadgroup per (q-head, split) -> K/V letti+dequant nq volte di troppo). qg su nh*(NQX/TQ).\n"
    "    constexpr uint NT = NQX / TQ;   // n. query-tile per q-head\n"
    "    uint qh_in = qg / NT;\n"
    "    uint qi0   = (qg % NT) * TQ;\n"
    "    uint kh    = qh_in * NKV / NH;\n"
    "    float qreg[TQ][DPL]; float m[TQ]; float l[TQ]; float acc[TQ][DPL];\n"
    "    for (uint tq = 0; tq < TQ; ++tq) {\n"
    "        m[tq] = -INFINITY; l[tq] = 0.0f;\n"
    "        uint qoff = (qh_in * NQX + qi0 + tq) * Dhead;\n"
    "        for (uint i = 0; i < DPL; ++i) { qreg[tq][i] = (float)q[qoff + lane + 32u*i]; acc[tq][i] = 0.0f; }\n"
    "    }\n"
    "    uint c0 = split * nci / SPLITS;\n"
    "    uint c1 = (split + 1) * nci / SPLITS;\n"
    "    for (uint t = c0; t < c1; ++t) {\n"
    "        const uint pt = (uint)ptab[t / PT] * PT + (t % PT);\n"
    "        const device uint*   kw  = (const device uint*)((const device uint8_t*)Kc_w + (size_t)(kh * capi + pt) * KWB);\n"
    "        const device uint8_t* kwb = (const device uint8_t*)kw;\n"
    "        const device ST*     ks  = Kc_s + (kh * capi + pt) * (Dhead / G);\n"
    "        const device ST*     kb  = Kc_b + (kh * capi + pt) * (Dhead / G);\n"
    "        const device uint*   vw  = (const device uint*)((const device uint8_t*)Vc_w + (size_t)(kh * capi + pt) * VWB);\n"
    "        const device uint8_t* vwb = (const device uint8_t*)vw;\n"
    "        const device ST*     vs  = Vc_s + (kh * capi + pt) * (Dhead / G);\n"
    "        const device ST*     vb  = Vc_b + (kh * capi + pt) * (Dhead / G);\n"
    "        float kk[DPL], vv[DPL];\n"
    "        for (uint i = 0; i < DPL; ++i) {\n"
    "            uint d = lane + 32u * i;\n"
    "            kk[i] = llm_kv_dq(llm_kv_val<BK, SYM>(kw, kwb, d), (float)ks[d / G], (float)kb[d / G]);\n"
    "            vv[i] = llm_kv_dq(llm_kv_val<BV, SYM>(vw, vwb, d), (float)vs[d / G], (float)vb[d / G]);\n"
    "        }\n"
    "        for (uint tq = 0; tq < TQ; ++tq) {\n"
    "            if ((int)t > (int)nci + NR - NQX + (int)(qi0 + tq)) continue;  // causale\n"
    "            float p = 0.0f; for (uint i = 0; i < DPL; ++i) p += qreg[tq][i] * kk[i];\n"
    "            float sc = simd_sum(p) * scale;\n"
    "            float mn = max(m[tq], sc); float f = exp(m[tq] - mn); float e = exp(sc - mn);\n"
    "            l[tq] = l[tq] * f + e;\n"
    "            for (uint i = 0; i < DPL; ++i) acc[tq][i] = acc[tq][i] * f + e * vv[i];\n"
    "            m[tq] = mn;\n"
    "        }\n"
    "    }\n"
    "    if (split == 0u) {\n"
    "        for (uint t = 0; t < NR; ++t) {\n"
    "            const device T* kr = Kr + (kh * NR + t) * Dhead;\n"
    "            const device T* vr = Vr + (kh * NR + t) * Dhead;\n"
    "            float krr[DPL], vrr[DPL];\n"
    "            for (uint i = 0; i < DPL; ++i) { krr[i] = (float)kr[lane+32u*i]; vrr[i] = (float)vr[lane+32u*i]; }\n"
    "            for (uint tq = 0; tq < TQ; ++tq) {\n"
    "                if ((int)nci + (int)t > (int)nci + NR - NQX + (int)(qi0 + tq)) continue;\n"
    "                float p = 0.0f; for (uint i = 0; i < DPL; ++i) p += qreg[tq][i] * krr[i];\n"
    "                float sc = simd_sum(p) * scale;\n"
    "                float mn = max(m[tq], sc); float f = exp(m[tq] - mn); float e = exp(sc - mn);\n"
    "                l[tq] = l[tq] * f + e;\n"
    "                for (uint i = 0; i < DPL; ++i) acc[tq][i] = acc[tq][i] * f + e * vrr[i];\n"
    "                m[tq] = mn;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    for (uint tq = 0; tq < TQ; ++tq) {\n"
    "        uint qh = qh_in * NQX + qi0 + tq;\n"
    "        if (lane == 0u) { pm[qh * SPLITS + split] = m[tq]; pl[qh * SPLITS + split] = l[tq]; }\n"
    "        for (uint i = 0; i < DPL; ++i) pacc[(qh * SPLITS + split) * Dhead + lane + 32u * i] = acc[tq][i];\n"
    "    }\n"
;

/* ---- v2 (2026-09-19): GQA-SHARED. Un threadgroup per kv-head (grid.x = NKV), ogni lane
 * tiene NQ query-head dello stesso kv-head e RIUSA l'unpack di K/V per token (prima: un
 * threadgroup per q-head -> K/V letti+unpackati NKV/NH volte di troppo; misurato 2,1-2,3x
 * piu' lento della f16). La matematica per q-head e' identica: cambia solo il mapping.
 * Fallback A/B: LLM_KVQ_NOGQA=1 usa la src v1. */
static const char llm_kvq_sdpa_kernel_src_gqa[] =
    "\n"
    "    uint lane  = thread_position_in_threadgroup.x;\n"
    "    uint kh    = threadgroup_position_in_grid.x;\n"
    "    uint split = threadgroup_position_in_grid.y;\n"
    "    constexpr uint DPL = Dhead / 32u;\n"
    "    constexpr uint G = GK;\n"
    "    constexpr uint KWB = (Dhead * BK) / 8u;\n"
    "    constexpr uint VWB = (Dhead * BV) / 8u;\n"
    "\n"
    "    float qreg[NQ][DPL];\n"
    "    float m[NQ], l[NQ], acc[NQ][DPL];\n"
    "    for (uint h = 0; h < NQ; ++h) {\n"
    "        for (uint i = 0; i < DPL; ++i)\n"
    "            qreg[h][i] = (float)q[(kh * NQ + h) * Dhead + lane + 32u * i];\n"
    "        m[h] = -INFINITY; l[h] = 0.0f;\n"
    "        for (uint i = 0; i < DPL; ++i) acc[h][i] = 0.0f;\n"
    "    }\n"
    "\n"
    "    uint c0 = split * nci / SPLITS;\n"
    "    uint c1 = (split + 1) * nci / SPLITS;\n"
    "    for (uint t = c0; t < c1; ++t) {\n"
    "        const uint pt = (uint)ptab[t / PT] * PT + (t % PT);  /* logical->physical */\n"
    "        const device uint*   kw  = (const device uint*)((const device uint8_t*)Kc_w + (size_t)(kh * capi + pt) * KWB);\n"
    "        const device uint8_t* kwb = (const device uint8_t*)kw;\n"
    "        const device ST*     ks  = Kc_s + (kh * capi + pt) * (Dhead / G);\n"
    "        const device ST*     kb  = Kc_b + (kh * capi + pt) * (Dhead / G);\n"
    "        float kd[DPL];\n"
    "        for (uint i = 0; i < DPL; ++i) {\n"
    "            uint d = lane + 32u * i;\n"
    "            kd[i] = llm_kv_dq(llm_kv_val<BK, SYM>(kw, kwb, d), (float)ks[d / G], (float)kb[d / G]);\n"
    "        }\n"
    "        const device uint*   vw  = (const device uint*)((const device uint8_t*)Vc_w + (size_t)(kh * capi + pt) * VWB);\n"
    "        const device uint8_t* vwb = (const device uint8_t*)vw;\n"
    "        const device ST*     vs  = Vc_s + (kh * capi + pt) * (Dhead / G);\n"
    "        const device ST*     vb  = Vc_b + (kh * capi + pt) * (Dhead / G);\n"
    "        float vd[DPL];\n"
    "        for (uint i = 0; i < DPL; ++i) {\n"
    "            uint d = lane + 32u * i;\n"
    "            vd[i] = llm_kv_dq(llm_kv_val<BV, SYM>(vw, vwb, d), (float)vs[d / G], (float)vb[d / G]);\n"
    "        }\n"
    "        for (uint h = 0; h < NQ; ++h) {\n"
    "            float p = 0.0f;\n"
    "            for (uint i = 0; i < DPL; ++i) p += qreg[h][i] * kd[i];\n"
    "            float sg = simd_sum(p) * scale;\n"
    "            float mn = max(m[h], sg);\n"
    "            float f = exp(m[h] - mn);\n"
    "            float e = exp(sg - mn);\n"
    "            l[h] = l[h] * f + e;\n"
    "            for (uint i = 0; i < DPL; ++i) acc[h][i] = acc[h][i] * f + e * vd[i];\n"
    "            m[h] = mn;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    if (split == 0u) {\n"
    "        for (uint t = 0; t < NR; ++t) {\n"
    "            const device T* kr = Kr + (kh * NR + t) * Dhead;\n"
    "            float kd[DPL];\n"
    "            for (uint i = 0; i < DPL; ++i) kd[i] = (float)kr[lane + 32u * i];\n"
    "            const device T* vr = Vr + (kh * NR + t) * Dhead;\n"
    "            float vd[DPL];\n"
    "            for (uint i = 0; i < DPL; ++i) vd[i] = (float)vr[lane + 32u * i];\n"
    "            for (uint h = 0; h < NQ; ++h) {\n"
    "                float p = 0.0f;\n"
    "                for (uint i = 0; i < DPL; ++i) p += qreg[h][i] * kd[i];\n"
    "                float sg = simd_sum(p) * scale;\n"
    "                float mn = max(m[h], sg);\n"
    "                float f = exp(m[h] - mn);\n"
    "                float e = exp(sg - mn);\n"
    "                l[h] = l[h] * f + e;\n"
    "                for (uint i = 0; i < DPL; ++i) acc[h][i] = acc[h][i] * f + e * vd[i];\n"
    "                m[h] = mn;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    for (uint h = 0; h < NQ; ++h) {\n"
    "        uint qh = kh * NQ + h;\n"
    "        if (lane == 0u) { pm[qh * SPLITS + split] = m[h]; pl[qh * SPLITS + split] = l[h]; }\n"
    "        for (uint i = 0; i < DPL; ++i)\n"
    "            pacc[(qh * SPLITS + split) * Dhead + lane + 32u * i] = acc[h][i];\n"
    "    }\n"
;

/* Combine: un threadgroup (32 lane) per query head. */
static const char llm_kvq_combine_src[] =
    "\n"
    "    uint lane = thread_position_in_threadgroup.x;\n"
    "    uint qh   = threadgroup_position_in_grid.x;\n"
    "    constexpr uint DPL = Dhead / 32u;\n"
    "    float m = -INFINITY;\n"
    "    for (uint s = 0; s < SPLITS; ++s) { float v = pm[qh * SPLITS + s]; if (v > m) m = v; }\n"
    "    float l = 0.0f;\n"
    "    for (uint s = 0; s < SPLITS; ++s) l += pl[qh * SPLITS + s] * exp(pm[qh * SPLITS + s] - m);\n"
    "    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;\n"
    "    for (uint i = 0; i < DPL; ++i) {\n"
    "        uint d = lane + 32u * i;\n"
    "        float a = 0.0f;\n"
    "        for (uint s = 0; s < SPLITS; ++s)\n"
    "            a += pacc[(qh * SPLITS + s) * Dhead + d] * exp(pm[qh * SPLITS + s] - m);\n"
    "        out[qh * Dhead + d] = a * inv;\n"
    "    }\n"
;

/* FA2 (2026-10-03): kernel di PREFILL con tensor unit (simdgroup MMA). Un threadgroup per
 * (q-head, tile di KTQ=8 posizioni). Loop sulle key-tile [0, nci+NR): cold int8 (dequant-on-read
 * via ptab) e recent fp16. QK^T = Q @ Kt (K trasposto in threadgroup), online softmax causale,
 * poi PV = P @ V. Scrive l'output direttamente (niente split-K/combine). Algoritmo validato in
 * oracoli/fa2_probe.c. Template: come il kernel v1 + KTQ (query-tile) e KT (key-tile). */
static const char llm_kvq_sdpa_kernel_src_fa2[] =
    "\n"
    "    uint tid = thread_position_in_threadgroup.x;\n"
    "    uint gid = thread_position_in_grid.x;\n"
    "    constexpr uint G = GK;\n"
    "    constexpr uint KWB = (Dhead * BK) / 8u;\n"
    "    constexpr uint VWB = (Dhead * BV) / 8u;\n"
    "    constexpr uint KTM = 8u;\n"
    "    constexpr uint NT = (NQX + KTQ - 1u) / KTQ;\n"
    "    uint qh = gid / NT;\n"
    "    uint p0 = (gid % NT) * KTQ;\n"
    "    uint NVQ = (NQX - p0 < KTM) ? (NQX - p0) : KTM;\n"
    "    uint kh = qh * NKV / NH;\n"
    "    uint nkey = nci + NR;\n"
    "    threadgroup half  qt[KTM*Dhead];\n"
    "    threadgroup half  kt[Dhead*KT];\n"
    "    threadgroup half  vt[KT*Dhead];\n"
    "    threadgroup half  zt[KTM*KT];\n"
    "    threadgroup half  pt[KTM*KT];\n"
    "    threadgroup half  ot[KTM*Dhead];\n"
    "    threadgroup float oacc[KTM*Dhead];\n"
    "    threadgroup float mrow[KTM], lrow[KTM], frow[KTM];\n"
    "    threadgroup half  zbuf[64];\n"
    "    for (uint i = tid; i < KTM*Dhead; i += 32u) oacc[i] = 0.0f;\n"
    "    for (uint i = tid; i < 64u; i += 32u) zbuf[i] = (half)0;\n"
    "    for (uint i = tid; i < KTM*KT; i += 32u) pt[i] = (half)0;\n"
    "    for (uint i = tid; i < KTM; i += 32u) { mrow[i] = -INFINITY; lrow[i] = 0.0f; frow[i] = 0.0f; }\n"
    "    for (uint i = tid; i < KTM*Dhead; i += 32u) {\n"
    "        uint r = i / Dhead;\n"
    "        qt[i] = (r < NVQ) ? (half)q[(qh*NQX + p0 + r)*Dhead + (i%Dhead)] : (half)0;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    simdgroup_half8x8 Aq[Dhead/8];\n"
    "    for (uint k = 0; k < Dhead/8; ++k) simdgroup_load(Aq[k], qt + k*8u, Dhead);\n"
    "    uint ntiles = (nkey + KT - 1u) / KT;\n"
    "    for (uint tl = 0; tl < ntiles; ++tl) {\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint i = tid; i < KT*Dhead; i += 32u) {\n"
    "            uint j = i / Dhead, d = i % Dhead;\n"
    "            uint t = tl*KT + j;\n"
    "            float kd = 0.0f, vd = 0.0f;\n"
    "            if (t < nkey && t < nci) {\n"
    "                const uint pg = (uint)ptab[t / PT] * PT + (t % PT);\n"
    "                const device uint*    kw  = (const device uint*)((const device uint8_t*)Kc_w + (size_t)(kh * capi + pg) * KWB);\n"
    "                const device uint8_t* kwb = (const device uint8_t*)kw;\n"
    "                const device ST*      ks  = Kc_s + (kh * capi + pg) * (Dhead / G);\n"
    "                const device ST*      kb  = Kc_b + (kh * capi + pg) * (Dhead / G);\n"
    "                kd = llm_kv_dq(llm_kv_val<BK, SYM>(kw, kwb, d), (float)ks[d / G], (float)kb[d / G]);\n"
    "                const device uint*    vw  = (const device uint*)((const device uint8_t*)Vc_w + (size_t)(kh * capi + pg) * VWB);\n"
    "                const device uint8_t* vwb = (const device uint8_t*)vw;\n"
    "                const device ST*      vs  = Vc_s + (kh * capi + pg) * (Dhead / G);\n"
    "                const device ST*      vb  = Vc_b + (kh * capi + pg) * (Dhead / G);\n"
    "                vd = llm_kv_dq(llm_kv_val<BV, SYM>(vw, vwb, d), (float)vs[d / G], (float)vb[d / G]);\n"
    "            } else if (t < nkey) {\n"
    "                kd = (float)Kr[(kh * NR + (t - nci)) * Dhead + d];\n"
    "                vd = (float)Vr[(kh * NR + (t - nci)) * Dhead + d];\n"
    "            }\n"
    "            kt[d*KT + j] = (half)kd;\n"
    "            vt[j*Dhead + d] = (half)vd;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint nb = 0; nb < KT/8; ++nb) {\n"
    "            simdgroup_half8x8 acc; simdgroup_load(acc, zbuf, 8u);\n"
    "            for (uint k = 0; k < Dhead/8; ++k) {\n"
    "                simdgroup_half8x8 Bk; simdgroup_load(Bk, kt + k*8u*KT + nb*8u, KT);\n"
    "                simdgroup_multiply_accumulate(acc, Aq[k], Bk, acc);\n"
    "            }\n"
    "            simdgroup_store(acc, zt + nb*8u, KT);\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (tid < NVQ) {\n"
    "            uint m = tid; int qpos = (int)(nci + NR - NQX + p0 + m);\n"
    "            float mx = -INFINITY;\n"
    "            for (uint j = 0; j < KT; ++j) { int kn = (int)(tl*KT + j);\n"
    "                if (kn >= (int)nkey || kn > qpos) continue;\n"
    "                mx = max(mx, (float)zt[m*KT+j] * scale); }\n"
    "            float mnew = max(mrow[m], mx);\n"
    "            float f = (mrow[m] > -INFINITY) ? exp(mrow[m] - mnew) : 0.0f;\n"
    "            float s = 0.0f;\n"
    "            for (uint j = 0; j < KT; ++j) { int kn = (int)(tl*KT + j); float e = 0.0f;\n"
    "                if (kn < (int)nkey && kn <= qpos) e = exp((float)zt[m*KT+j] * scale - mnew);\n"
    "                pt[m*KT+j] = (half)e; s += e; }\n"
    "            frow[m] = f; lrow[m] = lrow[m]*f + s; mrow[m] = mnew;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        simdgroup_half8x8 Pa[KT/8];\n"
    "        for (uint k = 0; k < KT/8; ++k) simdgroup_load(Pa[k], pt + k*8u, KT);\n"
    "        for (uint nb = 0; nb < Dhead/8; ++nb) {\n"
    "            simdgroup_half8x8 acc; simdgroup_load(acc, zbuf, 8u);\n"
    "            for (uint k = 0; k < KT/8; ++k) {\n"
    "                simdgroup_half8x8 Bk; simdgroup_load(Bk, vt + (k*8u)*Dhead + nb*8u, Dhead);\n"
    "                simdgroup_multiply_accumulate(acc, Pa[k], Bk, acc);\n"
    "            }\n"
    "            simdgroup_store(acc, ot + nb*8u, Dhead);\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint i = tid; i < KTM*Dhead; i += 32u)\n"
    "            oacc[i] = oacc[i]*frow[i/Dhead] + (float)ot[i];\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint i = tid; i < NVQ*Dhead; i += 32u) {\n"
    "        uint m = i / Dhead;\n"
    "        out[(qh*NQX + p0 + m)*Dhead + (i%Dhead)] = (lrow[m] > 0.0f) ? oacc[i]/lrow[m] : 0.0f;\n"
    "    }\n"
;

typedef struct { mlx_fast_metal_kernel part, combine, fa2; bool built; bool built_fa2; } llm_kvq_sdpa_cache;
static llm_kvq_sdpa_cache llm_kvq_sdpa_singleton = { {0}, {0}, {0}, false, false };

#ifndef LLM_KVQ_SPLITS
#define LLM_KVQ_SPLITS 8
#endif
/* Token cold sequenziali per thread "target": lo split-K viene fatto crescere con nc in modo
 * che il lavoro sequenziale per thread resti circa COSTANTE invece di crescere linearmente col
 * contesto (bug misurato 2026-09-17, AiWiki "MLX-KVQuant-Contesto-Lungo-5x-Peggiore": a SPLITS
 * fisso=8, a 32k/64k ogni thread scansiona ~4000/8000 token in sequenza — la pendenza ms/token
 * cresce piu' ripida di quella fp16, che ha molto piu' parallelismo).
 *
 * NOTA (2026-09-17, sweep misurato su Qwen3-8B-4bit @32k, LLM_KV_DEQUANT_READ=1): l'oracolo
 * fp16 (kernel Apple sdpa_vector_2pass, scaled_dot_product_attention.cpp:445-476) usa un
 * `blocks` che a N=32768-65536 arriva a 512-1024 — ma COPIARE quei valori qui e' stato
 * misurato PEGGIORARE le cose (splits~480: 5,2 tok/s, PEGGIO del vecchio SPLITS=8 fisso:
 * 24,9-48,9 tok/s a seconda del batching del flush, vedi sotto). Il combine kernel di MLX
 * riduce la fase finale su BN=32 lane in parallelo (`sums[simd_lid+BN*b]` + simd_sum), il
 * NOSTRO combine (llm_kvq_combine_src) fa un loop seriale O(SPLITS) IDENTICO su ognuna delle
 * 32 lane (ridondante, mai riscritto per usare lo stesso schema) — a poche decine di split va
 * bene, ma la crescita lineare della memoria intermedia pm/pl/pacc (O(nh*splits*hd)) scritta E
 * riletta ad ogni step inizia a dominare molto prima dei bucket 512/1024 di MLX. Sweep
 * misurato (nc~30720, decode 24 token): split_tokens=4096 (splits=8) 53,3 tok/s;
 * 2048 (splits=15) 56,1; 1024 (splits=30) 56,3 — plateau gia' raggiunto a ~15-30 split, NON
 * a centinaia. Override: opzione di registro `kvq_split_tokens`.
 *
 * AGGIORNAMENTO (2026-09-30, M5 Max, Llama-3.2-1B, A/B interleaved, mediane): il default 1024
 * lasciava solo splits=8 a 8k (nc~7.7k) e 16 a 16k — SOTTO il plateau. Con split_tokens=256
 * (splits=30 a 8k, 62 a 16k): decode 8k 88 -> 101 tok/s (+15%), 16k 92 -> 102 (+11%). Il kernel
 * e' quindi parallelismo-bound sotto il plateau. Default abbassato a 256. */
#ifndef LLM_KVQ_SPLIT_TOKENS
#define LLM_KVQ_SPLIT_TOKENS 256
#endif
/* Tetto di split: tenuto DELIBERATAMENTE lontano dalla zona misurata come dannosa (~480,
 * 5,2 tok/s vs il plateau ~56 tok/s misurato a splits=15-30) finche' llm_kvq_combine_src non
 * viene riscritto per ridurre in parallelo (schema BN di MLX) invece che serialmente per
 * lane — vedi nota sopra. Override: opzione di registro `kvq_splits_max`. */
#ifndef LLM_KVQ_SPLITS_MAX
#define LLM_KVQ_SPLITS_MAX 64
#endif
static int llm_mlx_kvq_splits_max(void) {
    static int v = -1;
    if (v < 0) { int o = 0; v = (llm_ai_mlx_opt_get_int("kvq_splits_max", &o) == 0 && o > 0) ? o : LLM_KVQ_SPLITS_MAX; }
    return v;
}
static int llm_mlx_kvq_split_tokens(void) {
    static int v = -1;
    if (v < 0) { int o = 0; v = (llm_ai_mlx_opt_get_int("kvq_split_tokens", &o) == 0 && o > 0) ? o : LLM_KVQ_SPLIT_TOKENS; }
    return v;
}

/* v2 GQA-shared: OPT-IN. Misurato 2026-09-19 (8B, ctxdecode 8k, finestra ~quieta):
 * v2 23,8/27,2 tok/s vs v1 29,4/32,6 -> ~15% PIU' LENTO. Il riuso di K/V tra le query head
 * NON e' il collo di bottiglia (la riduzione di grid da nh=32 a nkv=8 costa piu' di quanto
 * il riuso renda). Tenuto come esperimento riproducibile dietro l'opzione `kvq_gqa=1`. */
static bool llm_kvq_gqa(void) {
    int b = 0;
    return llm_ai_mlx_opt_get_bool("kvq_gqa", &b) == 0 && b != 0;
}

/* FA2 (2026-10-03): kernel prefill con tensor-unit (simdgroup MMA) invece dei dot scalari.
 * OPT-IN (default OFF: il default su main resta il W3 TQ-tile). Vedi
 * oracoli/fa2_probe.c per l'algoritmo validato (online softmax + causale). */
static bool llm_kvq_fa2(void) {
    int b = 0;
    return llm_ai_mlx_opt_get_bool("kvq_fa2", &b) == 0 && b != 0;
}

static int llm_mlx_kvq_sdpa_kernel_build(void) {
    if (llm_kvq_sdpa_singleton.built) return 0;
    {
        const char* in_names[] = {"q", "scale", "nci", "capi", "Kc_w", "Kc_s", "Kc_b", "Vc_w", "Vc_s", "Vc_b", "Kr", "Vr", "ptab"};
        const char* out_names[] = {"pm", "pl", "pacc"};
        mlx_vector_string iv = mlx_vector_string_new();
        for (int i = 0; i < 13; i++) mlx_vector_string_append_value(iv, in_names[i]);
        mlx_vector_string ov = mlx_vector_string_new();
        for (int i = 0; i < 3; i++) mlx_vector_string_append_value(ov, out_names[i]);
        llm_kvq_sdpa_singleton.part = mlx_fast_metal_kernel_new(
            "llm_kvq_sdpa_decode", iv, ov,
            llm_kvq_gqa() ? llm_kvq_sdpa_kernel_src_gqa : llm_kvq_sdpa_kernel_src, llm_kvq_sdpa_kernel_header,
            /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
        mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    }
    {
        const char* in_names[] = {"pm", "pl", "pacc"};
        const char* out_names[] = {"out"};
        mlx_vector_string iv = mlx_vector_string_new();
        for (int i = 0; i < 3; i++) mlx_vector_string_append_value(iv, in_names[i]);
        mlx_vector_string ov = mlx_vector_string_new();
        mlx_vector_string_append_value(ov, out_names[0]);
        llm_kvq_sdpa_singleton.combine = mlx_fast_metal_kernel_new(
            "llm_kvq_combine", iv, ov, llm_kvq_combine_src, llm_kvq_sdpa_kernel_header,
            /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
        mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    }
    {
        const char* in_names[] = {"q", "scale", "nci", "capi", "Kc_w", "Kc_s", "Kc_b", "Vc_w", "Vc_s", "Vc_b", "Kr", "Vr", "ptab"};
        const char* out_names[] = {"out"};
        mlx_vector_string iv = mlx_vector_string_new();
        for (int i = 0; i < 13; i++) mlx_vector_string_append_value(iv, in_names[i]);
        mlx_vector_string ov = mlx_vector_string_new();
        mlx_vector_string_append_value(ov, out_names[0]);
        llm_kvq_sdpa_singleton.fa2 = mlx_fast_metal_kernel_new(
            "llm_kvq_sdpa_fa2", iv, ov,
            llm_kvq_sdpa_kernel_src_fa2, llm_kvq_sdpa_kernel_header,
            /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
        llm_kvq_sdpa_singleton.built_fa2 = (llm_kvq_sdpa_singleton.fa2.ctx != NULL);
        mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    }
    llm_kvq_sdpa_singleton.built = true;
    return 0;
}

/* Decode dequant-on-read, split-K. `nc` = token da attendere (0 = tutta la capacita').
 * `kbits`/`vbits` = bit affine di K/V (2..8). false = fallback (chiamante usa il path normale). */
static bool llm_mlx_kvq_sdpa_decode(mlx_array* out, mlx_array q,
                                    mlx_array Kc_w, mlx_array Kc_s, mlx_array Kc_b,
                                    mlx_array Vc_w, mlx_array Vc_s, mlx_array Vc_b,
                                    mlx_array Kr, mlx_array Vr, int nc, int kbits, int vbits,
                                    float scale, int gval, mlx_array ptab, mlx_stream stream) {
    if (llm_mlx_kvq_sdpa_kernel_build() != 0) return false;
    if (kbits < 2 || kbits > 8 || vbits < 2 || vbits > 8) return false;
    /* W3 (2026-10-03): q = [nh,hd] (decode, nq=1) oppure [nq,nh,hd] (PREFILL, nq>1). Il kernel
     * indicizza la riga q con qh = qi*NH+qh_in e usa kh=(qh%NH)*NKV/NH (KV condiviso fra query). */
    int nqy = 1, nh, hd;
    if (mlx_array_ndim(q) == 2) { nh = (int)mlx_array_shape(q)[0]; hd = (int)mlx_array_shape(q)[1]; }
    else if (mlx_array_ndim(q) == 3) { nh = (int)mlx_array_shape(q)[0]; nqy = (int)mlx_array_shape(q)[1]; hd = (int)mlx_array_shape(q)[2]; }
    else return false;
    mlx_array q_flat = mlx_array_new();
    if (nqy > 1) { int fsh[2] = {nqy * nh, hd}; if (mlx_reshape(&q_flat, q, fsh, 2, stream) != 0) { mlx_array_free(q_flat); return false; } q = q_flat; }
    else mlx_array_set(&q_flat, q);
    if (hd % 32 != 0) { mlx_array_free(q_flat); return false; }
    if (mlx_array_ndim(Kr) != 3 || mlx_array_ndim(Kc_w) != 3) return false;
    int nkv = (int)mlx_array_shape(Kr)[0];
    int nr = (int)mlx_array_shape(Kr)[1];
    int cap = (int)mlx_array_shape(Kc_w)[1];
    if (nc <= 0 || nc > cap) nc = cap;
    if (nkv <= 0 || nh <= 0 || nh % nkv != 0) return false;
    int nq = nh / nkv;   /* q-head per kv-head (GQA): il kernel v2 le condivide */
    const int kw_last = (gval > 0) ? hd : (hd * kbits) / 32;   /* Splash: int8 byte, ultima dim = hd */
    const int vw_last = (gval > 0) ? hd : (hd * vbits) / 32;
    if ((int)mlx_array_shape(Kr)[2] != hd || (int)mlx_array_shape(Kc_w)[2] != kw_last) return false;
    if ((int)mlx_array_shape(Vc_w)[1] != cap || (int)mlx_array_shape(Vc_w)[2] != vw_last) return false;
    if (gval <= 0 && (mlx_array_dtype(Kc_w) != MLX_UINT32 || mlx_array_dtype(Vc_w) != MLX_UINT32)) return false;
    mlx_dtype st = mlx_array_dtype(Kc_s);
    if (mlx_array_dtype(Kc_b) != st || mlx_array_dtype(Vc_s) != st || mlx_array_dtype(Vc_b) != st) return false;
    mlx_dtype T = mlx_array_dtype(q);
    if (mlx_array_dtype(Kr) != T || mlx_array_dtype(Vr) != T) return false;

    /* Split-K dinamico: almeno LLM_KVQ_SPLITS (piccoli contesti, dove piu' split
     * aggiungerebbero solo overhead di combine), altrimenti nc/kvq_split_tokens
     * arrotondato per eccesso, cosi' il numero di token scansionati IN SEQUENZA da ogni
     * thread resta vicino a kvq_split_tokens indipendentemente da quanto e' lungo il
     * contesto (prima: SPLITS fisso a 8, lavoro sequenziale per thread = nc/8, cresce senza
     * limite col contesto — la causa misurata della pendenza 5-7x peggiore della fp16). */
    int split_tokens = llm_mlx_kvq_split_tokens();
    int splits = LLM_KVQ_SPLITS;
    if (nc > 0 && split_tokens > 0) {
        int wanted = (nc + split_tokens - 1) / split_tokens;
        if (wanted > splits) splits = wanted;
    }
    if (nc > 0 && splits > nc) splits = nc;
    if (splits < 1) splits = 1;
    int splits_max = llm_mlx_kvq_splits_max();
    if (splits > splits_max) splits = splits_max;
    /* W3 (prefill): le query (nh*nq threadgroup) parallellizzano gia' -> lo split-K non serve e le
     * sue partial pacc=[nh*nq,splits,hd] esplodono con nq. Clampa a 1 split per nq>1. */
    if (nqy > 1) { int ps = 8; llm_ai_mlx_opt_get_int("kvq_prefill_splits", &ps); if (ps > 0) splits = ps; }

    /* page table: se il chiamante non ne fornisce una, usa l'IDENTITA' (pool contiguo). */
    mlx_array ptab_own = mlx_array_new();
    if (ptab.ctx == NULL) {
        int np = cap / 32; if (np < 1) np = 1;
        int32_t* ib = (int32_t*)malloc((size_t)np * sizeof(int32_t));
        if (!ib) return false;
        for (int i = 0; i < np; i++) ib[i] = i;
        int psh[1] = {np};
        ptab_own = mlx_array_new_data(ib, psh, 1, MLX_INT32); free(ib);
        if (ptab_own.ctx == NULL) return false;
        ptab = ptab_own;
    }
    mlx_array fa2_saved = mlx_array_new();
    bool fa2_verify = (getenv("LLM_FA2_VERIFY") != NULL);
    /* FA2 opt-in (kvq_fa2=1): prefill con tensor unit (simdgroup MMA). Default OFF -> path W3
     * intatto. Richiede nqy multiplo di KTQ=8 e hd multiplo di 8; altrimenti fallback W3. */
    if (getenv("LLM_FA2_DBG")) fprintf(stderr, "[KVQ] nqy=%d nh=%d hd=%d nkv=%d nc=%d cap=%d fa2opt=%d built=%d nq=%d\n", nqy, nh, hd, nkv, nc, cap, (int)llm_kvq_fa2(), (int)llm_kvq_sdpa_singleton.built_fa2, nq);
    if (nqy > 1 && llm_kvq_fa2() && llm_kvq_sdpa_singleton.built_fa2 && (hd % 8) == 0) {
        const int KTQ = 8, KT = 32;
        if (getenv("LLM_FA2_DBG")) fprintf(stderr, "[FA2] nqy=%d hd=%d nh=%d nkv=%d nc=%d nr=%d cap=%d\n", nqy, hd, nh, nkv, nc, nr, cap);
        mlx_vector_array fin = mlx_vector_array_new();
        mlx_array fscale = mlx_array_new_float(scale);
        mlx_array fnci = mlx_array_new_int(nc);
        mlx_array fcapi = mlx_array_new_int(cap);
        mlx_vector_array_append_value(fin, q);
        mlx_vector_array_append_value(fin, fscale);
        mlx_vector_array_append_value(fin, fnci);
        mlx_vector_array_append_value(fin, fcapi);
        mlx_vector_array_append_value(fin, Kc_w);
        mlx_vector_array_append_value(fin, Kc_s);
        mlx_vector_array_append_value(fin, Kc_b);
        mlx_vector_array_append_value(fin, Vc_w);
        mlx_vector_array_append_value(fin, Vc_s);
        mlx_vector_array_append_value(fin, Vc_b);
        mlx_vector_array_append_value(fin, Kr);
        mlx_vector_array_append_value(fin, Vr);
        mlx_vector_array_append_value(fin, ptab);
        mlx_fast_metal_kernel_config fcfg = mlx_fast_metal_kernel_config_new();
        mlx_fast_metal_kernel_config_add_template_arg_dtype(fcfg, "T", T);
        mlx_fast_metal_kernel_config_add_template_arg_dtype(fcfg, "ST", st);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "Dhead", hd);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "NH", nh);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "NQX", nqy);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "NKV", nkv);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "GK", gval > 0 ? gval : 64);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "SYM", gval > 0 ? 1 : 0);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "BK", kbits);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "BV", vbits);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "NR", nr);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "PT", 32);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "KTQ", KTQ);
        mlx_fast_metal_kernel_config_add_template_arg_int(fcfg, "KT", KT);
        int fosh[2] = {nh*nqy, hd};
        mlx_fast_metal_kernel_config_add_output_arg(fcfg, fosh, 2, MLX_FLOAT32);
        mlx_fast_metal_kernel_config_set_grid(fcfg, (nh * ((nqy + KTQ - 1) / KTQ)) * 32, 1, 1);
        mlx_fast_metal_kernel_config_set_thread_group(fcfg, 32, 1, 1);
        mlx_vector_array fouts = mlx_vector_array_new();
        int frc = mlx_fast_metal_kernel_apply(&fouts, llm_kvq_sdpa_singleton.fa2, fin, fcfg, stream);
        bool fok = (frc == 0) && mlx_vector_array_size(fouts) == 1;
        if (fok) { mlx_array a0 = mlx_array_new(); fok = mlx_vector_array_get(&a0, fouts, 0) == 0; if (fok) mlx_array_set(&fa2_saved, a0); mlx_array_free(a0); }
        mlx_fast_metal_kernel_config_free(fcfg);
        mlx_vector_array_free(fin); mlx_vector_array_free(fouts);
        mlx_array_free(fscale); mlx_array_free(fnci); mlx_array_free(fcapi);
        if (fok && !fa2_verify) { mlx_array_set(out, fa2_saved); mlx_array_free(fa2_saved); mlx_array_free(ptab_own); mlx_array_free(q_flat); return true; }
    }
    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_array scale_arr = mlx_array_new_float(scale);
    mlx_array nci_arr = mlx_array_new_int(nc);
    mlx_array capi_arr = mlx_array_new_int(cap);
    mlx_vector_array_append_value(inputs, q);
    mlx_vector_array_append_value(inputs, scale_arr);
    mlx_vector_array_append_value(inputs, nci_arr);
    mlx_vector_array_append_value(inputs, capi_arr);
    mlx_vector_array_append_value(inputs, Kc_w);
    mlx_vector_array_append_value(inputs, Kc_s);
    mlx_vector_array_append_value(inputs, Kc_b);
    mlx_vector_array_append_value(inputs, Vc_w);
    mlx_vector_array_append_value(inputs, Vc_s);
    mlx_vector_array_append_value(inputs, Vc_b);
    mlx_vector_array_append_value(inputs, Kr);
    mlx_vector_array_append_value(inputs, Vr);
    mlx_vector_array_append_value(inputs, ptab);

    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", T);
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "ST", st);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dhead", hd);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NH", nh);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NQX", nqy);
    int tq = (nqy % 8 == 0) ? 8 : 1; mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TQ", tq);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NKV", nkv);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GK", gval > 0 ? gval : 64);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SYM", gval > 0 ? 1 : 0);
    if (llm_kvq_gqa()) mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NQ", nq);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BK", kbits);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BV", vbits);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NR", nr);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SPLITS", splits);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PT", 32);
    int pmsh[2] = {nh*nqy, splits}, plsh[2] = {nh*nqy, splits}, pash[3] = {nh*nqy, splits, hd};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, pmsh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, plsh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, pash, 3, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (llm_kvq_gqa() ? nkv : nh*(nqy/tq)) * 32, splits, 1);   /* THREAD totali */
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1);

    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_kvq_sdpa_singleton.part, inputs, cfg, stream);
    bool ok = (rc == 0) && mlx_vector_array_size(outputs) == 3;
    mlx_array pm = mlx_array_new(), pl = mlx_array_new(), pacc = mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&pm, outputs, 0) == 0 &&
                 mlx_vector_array_get(&pl, outputs, 1) == 0 &&
                 mlx_vector_array_get(&pacc, outputs, 2) == 0;
    if (ok) {
        mlx_vector_array cin = mlx_vector_array_new();
        mlx_vector_array_append_value(cin, pm);
        mlx_vector_array_append_value(cin, pl);
        mlx_vector_array_append_value(cin, pacc);
        mlx_fast_metal_kernel_config ccfg = mlx_fast_metal_kernel_config_new();
        mlx_fast_metal_kernel_config_add_template_arg_int(ccfg, "Dhead", hd);
        mlx_fast_metal_kernel_config_add_template_arg_int(ccfg, "SPLITS", splits);
        int osh[2] = {nh*nqy, hd};
        mlx_fast_metal_kernel_config_add_output_arg(ccfg, osh, 2, MLX_FLOAT32);
        mlx_fast_metal_kernel_config_set_grid(ccfg, nh*nqy * 32, 1, 1);   /* THREAD totali */
        mlx_fast_metal_kernel_config_set_thread_group(ccfg, 32, 1, 1);
        mlx_vector_array couts = mlx_vector_array_new();
        int rc2 = mlx_fast_metal_kernel_apply(&couts, llm_kvq_sdpa_singleton.combine, cin, ccfg, stream);
        ok = (rc2 == 0) && mlx_vector_array_size(couts) == 1;
        if (ok) {
            mlx_array a0 = mlx_array_new();
            ok = mlx_vector_array_get(&a0, couts, 0) == 0;
            if (ok) mlx_array_set(out, a0);
            mlx_array_free(a0);
        }
        mlx_fast_metal_kernel_config_free(ccfg);
        mlx_vector_array_free(cin);
        mlx_vector_array_free(couts);
    }
    if (fa2_verify && fa2_saved.ctx != NULL && out->ctx != NULL && (int)mlx_array_size(fa2_saved) == (int)mlx_array_size(*out)
        && mlx_array_dtype(fa2_saved) == MLX_FLOAT32 && mlx_array_dtype(*out) == MLX_FLOAT32) {
        mlx_array_eval(fa2_saved); mlx_array_eval(*out);
        const float* A = mlx_array_data_float32(fa2_saved);
        const float* B = mlx_array_data_float32(*out);
        float gmax = -1.0f; int rowmax = -1;
        for (int r = 0; r < nh*nqy; r++) { float d = 0.0f;
            for (int c = 0; c < hd; c++) { float a = A[r*hd+c], b = B[r*hd+c]; float dd = a>b?(a-b):(b-a); if (dd > d) d = dd; }
            if (d > gmax) { gmax = d; rowmax = r; } }
        float ma=-1e30f, mb=-1e30f; int na=0, nb=0;
        for (int i = 0; i < nh*nqy*hd; i++) { float a=A[i], b=B[i];
            if (!(a==a) || a>1e30f || a<-1e30f) na++; if (!(b==b) || b>1e30f || b<-1e30f) nb++;
            if (a>ma) ma=a; if (b>mb) mb=b; }
        fprintf(stderr, "[FA2VERIFY] nqy=%d nh=%d nc=%d nr=%d maxdiff=%.5g row=%d head=%d pos=%d | maxA=%.5g maxB=%.5g badA=%d badB=%d\n",
                nqy, nh, nc, nr, gmax, rowmax, rowmax/nqy, rowmax%nqy, ma, mb, na, nb);
    }
    mlx_array_free(fa2_saved);
    mlx_array_free(pm); mlx_array_free(pl); mlx_array_free(pacc);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(scale_arr); mlx_array_free(nci_arr); mlx_array_free(capi_arr);
    mlx_array_free(ptab_own);
    mlx_array_free(q_flat);
    return ok;
}

#endif /* LLM_MLX_KV_QUANT_SDPA_KERNEL_H */
