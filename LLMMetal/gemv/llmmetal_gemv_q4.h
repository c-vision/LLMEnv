/* llmmetal_gemv_q4.h — GEMV decode Q4 affine, kernel Metal custom.
 *
 * Attacca il "dato decisivo" del piano
 * (~/Documents/AiWiki/01-Projects/LLMEnv/Piani/Proposta-614GBs-Banda-Metal-DeepDive.md, §2.1/§4
 * L3): la GEMV quantizzata Q4 affine di MLX misura ~26 GB/s (~4% del picco 614 GB/s) su shape
 * Llama-1B (x[1,2048] @ W[8192,2048], g64 affine). Questo modulo implementa il design del piano
 * (staging cooperativo di x in threadgroup memory, un simdgroup per riga, riduzione simd_sum)
 * per lo stesso caso.
 *
 * Formato Q4 affine (autoconsistente per questo spike — NON ancora verificato bit-esatto contro
 * il layout interno di MLX; l'integrazione con mx.quantize reale è lavoro successivo, qui si
 * valida la TECNICA del kernel e la banda raggiungibile, non la compatibilità di formato):
 *   - pesi: [M, H/2] uint8, 2 nibble/byte (nibble basso = colonna pari, alto = colonna dispari)
 *   - scale/bias: [M, H/group_size] half, affine: w = nibble*scale + bias
 *
 * Vincoli v1: H multiplo di 64 (32 lane * 2 nibble/byte, e compatibile con group_size=64);
 * group_size multiplo di 1 (nessun vincolo di allineamento ulteriore in questa versione).
 *
 * Design: un simdgroup (32 lane) per riga di W; ROWS_PER_TG simdgroup per threadgroup
 * condividono lo staging di x (caricato una volta in threadgroup memory, non riletto per ogni
 * riga) — stesso principio del kernel GDN di produzione (llmmetal_mlx_gdn_kernel.h).
 */
#ifndef LLMMETAL_METAL_GEMV_Q4_H
#define LLMMETAL_METAL_GEMV_Q4_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LLMMETAL_GEMV_ROWS_PER_TG 16

static const char llmmetal_gemv_q4_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Q4Params { uint H; uint M; uint group_size; uint rows_per_tg; uint lead_w; uint lead_s; uint lead_b; uint scale_bf16; };\n"
"inline float q4_ld_sb(const device uchar* p, uint idx, uint bf) {\n"
"    const ushort u = (ushort)p[2u * idx] | ((ushort)p[2u * idx + 1u] << 8);   // lettura a byte: i tensori safetensors non sono allineati a 2\n"
"    return bf ? as_type<float>(((uint)u) << 16) : (float)as_type<half>(u);\n"
"}\n"
"kernel void gemv_q4_affine(\n"
"    const device half*  x       [[buffer(0)]],\n"
"    const device uchar* Wq      [[buffer(1)]],\n"
"    const device uchar* scales  [[buffer(2)]],\n"
"    const device uchar* biases  [[buffer(3)]],\n"
"    device half*        out     [[buffer(4)]],\n"
"    constant Q4Params&  p       [[buffer(5)]],\n"
"    threadgroup half*   x_shared [[threadgroup(0)]],\n"
"    uint3 tgid   [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3  [[thread_position_in_threadgroup]],\n"
"    uint  lane   [[thread_index_in_simdgroup]],\n"
"    uint  simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    Wq = (const device uchar*)Wq + p.lead_w;\n"
"    scales = scales + p.lead_s;\n"
"    biases = biases + p.lead_b;\n"
"    const uint ltid = ltid3.x;\n"
"    const uint tg_threads = p.rows_per_tg * 32;\n"
"    for (uint i = ltid; i < p.H; i += tg_threads) x_shared[i] = x[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    const uint row = tgid.x * p.rows_per_tg + simd_id;\n"
"    if (row >= p.M) return;\n"
"\n"
"    const uint per_lane = p.H / 32;\n"
"    const device uchar* row_packed = Wq + (size_t)row * (p.H / 2);\n"
"    const uint n_groups = p.H / p.group_size;\n"
"    const device uchar* row_scales = scales + (size_t)row * n_groups * 2;\n"
"    const device uchar* row_biases = biases + (size_t)row * n_groups * 2;\n"
"\n"
"    // Bug di prestazioni RISOLTO (2026-09-06): con `h = lane*per_lane + i`, a un dato `i` i 32\n"
"    // lane leggono `row_packed` a byte_idx SPACED `per_lane/2` byte l'uno dall'altro (32\n"
"    // locazioni sparse invece di una lettura coalescente) — lo spacing CRESCE con `per_lane`\n"
"    // (=H/32), spiegando perche' la banda misurata CROLLAVA al crescere di H (26 GB/s a\n"
"    // H=2048, ~9 GB/s a H=8192) mentre un kernel fp16 equivalente senza dequant SALIVA con H\n"
"    // (22->64 GB/s): la firma classica di un accesso non coalescente che peggiora quanto piu'\n"
"    // le letture dei singoli lane si allontanano fra loro. Fix: `h = i*32 + lane` (round-robin\n"
"    // sui lane) — a un dato `i` i 32 lane leggono 32 `h` CONSECUTIVI, quindi byte_idx copre 16\n"
"    // byte consecutivi (2 nibble/byte): una singola lettura coalescente per iterazione invece\n"
"    // di 32 sparse. Risultato: 27->28 GB/s (H=2048, invariato, gia' vicino al limite fisso di\n"
"    // dispatch a quella dimensione) ma 9->73 GB/s a H=8192 (8x, ora SUPERA sia MLX ~26 GB/s sia\n"
"    // il kernel fp16 di controllo alla stessa shape, 63.7 GB/s) — la banda ora SALE con H come\n"
"    // atteso.\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < per_lane; i++) {\n"
"        uint h = i * 32 + lane;\n"
"        uint byte_idx = h >> 1;\n"
"        uchar b = row_packed[byte_idx];\n"
"        uint nib = (h & 1u) ? (b >> 4) : (b & 0xF);\n"
"        uint g = h / p.group_size;\n"
"        float w = (float)nib * q4_ld_sb(row_scales, g, p.scale_bf16) + q4_ld_sb(row_biases, g, p.scale_bf16);\n"
"        acc += w * (float)x_shared[h];\n"
"    }\n"
"    float sum = simd_sum(acc);\n"
"    if (lane == 0) out[row] = (half)sum;\n"
"}\n"
"\n"
"struct Q4MParams { uint H; uint M; uint group_size; uint rows_per_tg; uint scale_bf16; uint w_stride; uint s_stride; uint x_stride; };\n"
"kernel void gemv_q4_affine_multi(\n"
"    const device half*  x       [[buffer(0)]],\n"
"    const device uchar* Wq      [[buffer(1)]],\n"
"    const device uchar* scales  [[buffer(2)]],\n"
"    const device uchar* biases  [[buffer(3)]],\n"
"    device half*        out     [[buffer(4)]],\n"
"    const device int*   sel     [[buffer(5)]],\n"
"    constant Q4MParams& p       [[buffer(6)]],\n"
"    threadgroup half*   x_shared [[threadgroup(0)]],\n"
"    uint3 tgid   [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3  [[thread_position_in_threadgroup]],\n"
"    uint  lane   [[thread_index_in_simdgroup]],\n"
"    uint  simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    const int e = sel[tgid.y];\n"
"    x = x + (size_t)tgid.y * p.x_stride;\n"
"    Wq     = (const device uchar*)Wq + (size_t)e * p.w_stride;\n"
"    scales = scales + (size_t)e * p.s_stride;\n"
"    biases = biases + (size_t)e * p.s_stride;\n"
"    out    = out + (size_t)tgid.y * p.M;\n"
"    const uint ltid = ltid3.x;\n"
"    const uint tg_threads = p.rows_per_tg * 32;\n"
"    for (uint i = ltid; i < p.H; i += tg_threads) x_shared[i] = x[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const uint row = tgid.x * p.rows_per_tg + simd_id;\n"
"    if (row >= p.M) return;\n"
"    const uint per_lane = p.H / 32;\n"
"    const device uchar* row_packed = Wq + (size_t)row * (p.H / 2);\n"
"    const uint n_groups = p.H / p.group_size;\n"
"    const device uchar* row_scales = scales + (size_t)row * n_groups * 2;\n"
"    const device uchar* row_biases = biases + (size_t)row * n_groups * 2;\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < per_lane; i++) {\n"
"        uint h = i * 32 + lane;\n"
"        uint byte_idx = h >> 1;\n"
"        uchar b = row_packed[byte_idx];\n"
"        uint nib = (h & 1u) ? (b >> 4) : (b & 0xF);\n"
"        uint g = h / p.group_size;\n"
"        float w = (float)nib * q4_ld_sb(row_scales, g, p.scale_bf16) + q4_ld_sb(row_biases, g, p.scale_bf16);\n"
"        acc += w * (float)x_shared[h];\n"
"    }\n"
"    float sum = simd_sum(acc);\n"
"    if (lane == 0) out[row] = (half)sum;\n"
"}\n"
"struct Q4MoeParams { uint H; uint M; uint group_size; uint rows_per_tg; uint scale_bf16; uint w_stride; uint s_stride; uint x_stride; uint n_shared; uint n_routed; };\n"
"inline float q4_rowdot(const device uchar* Wq, const device uchar* scales, const device uchar* biases,\n"
"                       const threadgroup float* xs, uint H, uint group_size, uint bf, uint row, uint lane) {\n"
"    const uint per_lane = H / 32;\n"
"    const device uchar* rp = Wq + (size_t)row * (H / 2);\n"
"    const uint ng = H / group_size;\n"
"    const device uchar* rs = scales + (size_t)row * ng * 2;\n"
"    const device uchar* rb = biases + (size_t)row * ng * 2;\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < per_lane; i++) {\n"
"        uint h = i * 32 + lane; uint bi = h >> 1; uchar b = rp[bi];\n"
"        uint nib = (h & 1u) ? (b >> 4) : (b & 0xF); uint g = h / group_size;\n"
"        float w = (float)nib * q4_ld_sb(rs, g, bf) + q4_ld_sb(rb, g, bf);\n"
"        acc += w * xs[h];\n"
"    }\n"
"    return simd_sum(acc);\n"
"}\n"
"kernel void moe_mid(\n"
"    const device float* x, const device uchar* Wg, const device uchar* Sg, const device uchar* Bg,\n"
"    const device uchar* Wu, const device uchar* Su, const device uchar* Bu,\n"
"    const device uchar* Wsg, const device uchar* Ssg, const device uchar* Bsg,\n"
"    const device uchar* Wsu, const device uchar* Ssu, const device uchar* Bsu,\n"
"    device float* act, const device int* sel, constant Q4MoeParams& p,\n"
"    threadgroup float* xs [[threadgroup(0)]], uint3 tgid [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3 [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],\n"
"    uint simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    const bool sh = (p.n_shared > 0u) && (tgid.y >= p.n_routed);   /* slot >= K = shared expert */\n"
"    const int e = sh ? 0 : sel[tgid.y];\n"
"    x = x + (size_t)tgid.y * p.x_stride;\n"
"    if (sh) { Wg = Wsg; Sg = Ssg; Bg = Bsg; Wu = Wsu; Su = Ssu; Bu = Bsu; }\n"
"    else { Wg += (size_t)e * p.w_stride; Sg += (size_t)e * p.s_stride; Bg += (size_t)e * p.s_stride;\n"
"           Wu += (size_t)e * p.w_stride; Su += (size_t)e * p.s_stride; Bu += (size_t)e * p.s_stride; }\n"
"    const uint ltid = ltid3.x, tt = p.rows_per_tg * 32;\n"
"    for (uint i = ltid; i < p.H; i += tt) xs[i] = x[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const uint base = (tgid.x * p.rows_per_tg + simd_id) * 2u;   /* NR=2: row0 come ds4 */\n"
"    for (uint i = 0; i < 2u; i++) {\n"
"        const uint row = base + i;\n"
"        if (row >= p.M) break;\n"
"        float g = q4_rowdot(Wg,Sg,Bg,xs,p.H,p.group_size,p.scale_bf16,row,lane);\n"
"        float u = q4_rowdot(Wu,Su,Bu,xs,p.H,p.group_size,p.scale_bf16,row,lane);\n"
"        if (lane == 0) { float sg = 1.f/(1.f+exp(-g)); act[(size_t)tgid.y * p.M + row] = g*sg*u; }\n"
"    }\n"
"}\n"
"kernel void moe_down(\n"
"    const device float* act, const device uchar* Wd, const device uchar* Sd, const device uchar* Bd,\n"
"    const device uchar* Wsd, const device uchar* Ssd, const device uchar* Bsd,\n"
"    device float* out, const device int* sel, constant Q4MoeParams& p,\n"
"    threadgroup float* xs [[threadgroup(0)]], uint3 tgid [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3 [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],\n"
"    uint simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    const bool sh = (p.n_shared > 0u) && (tgid.y >= p.n_routed);   /* slot >= K = shared expert */\n"
"    const int e = sh ? 0 : sel[tgid.y];\n"
"    act = act + (size_t)tgid.y * p.x_stride;\n"
"    if (sh) { Wd = Wsd; Sd = Ssd; Bd = Bsd; }\n"
"    else { Wd += (size_t)e * p.w_stride; Sd += (size_t)e * p.s_stride; Bd += (size_t)e * p.s_stride; }\n"
"    const uint ltid = ltid3.x, tt = p.rows_per_tg * 32;\n"
"    for (uint i = ltid; i < p.H; i += tt) xs[i] = act[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const uint base2 = (tgid.x * p.rows_per_tg + simd_id) * 2u;\n"
"    for (uint i = 0; i < 2u; i++) {\n"
"        const uint row = base2 + i;\n"
"        if (row >= p.M) break;\n"
"        float d = q4_rowdot(Wd,Sd,Bd,xs,p.H,p.group_size,p.scale_bf16,row,lane);\n"
"        if (lane == 0) out[(size_t)tgid.y * p.M + row] = d;\n"
"    }\n"
"}\n"
"struct Q4RtParams { uint H; uint NE; uint K; uint rows_per_tg; };\n"
"kernel void moe_gates(const device float* x [[buffer(0)]], const device float* W [[buffer(1)]], device float* gates [[buffer(2)]],\n"
"    constant Q4RtParams& p [[buffer(3)]], uint3 tgid [[threadgroup_position_in_grid]],\n"
"    uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    const uint row = tgid.x * p.rows_per_tg + simd_id;\n"
"    if (row > p.NE) return;   /* NE+1 righe: l'ultima e' il gate dello shared expert */\n"
"    const device float* w = W + (ulong)row * p.H;\n"
"    float acc = 0.0f;\n"
"    for (uint i = lane; i < p.H; i += 32) acc += w[i] * x[i];\n"
"    const float s = simd_sum(acc);\n"
"    if (lane == 0) gates[row] = s;\n"
"}\n"
"kernel void moe_topk(const device float* gates [[buffer(0)]], device int* sel [[buffer(1)]], device float* wts [[buffer(2)]],\n"
"    constant Q4RtParams& p [[buffer(3)]], threadgroup float* sh [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]])\n"
"{\n"
"    /* 512 thread. softmax sui NE logit, top-K (a pari valore vince l'indice minore, come la versione CPU),\n"
"     * pesi normalizzati sui K scelti; wts[K] = sigmoid(logit dello shared expert). */\n"
"    threadgroup float* sv = sh;\n"
"    threadgroup int* si = (threadgroup int*)(sh + 512);\n"
"    threadgroup int* chosen = (threadgroup int*)(sh + 1024);\n"
"    threadgroup float* chosenv = sh + 1040;\n"
"    float mx = -INFINITY;\n"
"    for (uint i = tid; i < p.NE; i += 512) mx = max(mx, gates[i]);\n"
"    sv[tid] = mx; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = 256; s > 0; s >>= 1) { if (tid < s) sv[tid] = max(sv[tid], sv[tid + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    mx = sv[0]; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint k = 0; k < p.K; k++) {\n"
"        float best = -INFINITY; int bi = 0x7fffffff;\n"
"        for (uint i = tid; i < p.NE; i += 512) {\n"
"            bool used = false;\n"
"            for (uint j = 0; j < k; j++) used = used || (chosen[j] == (int)i);\n"
"            if (used) continue;\n"
"            const float v = gates[i];\n"
"            if (v > best) { best = v; bi = (int)i; }\n"
"        }\n"
"        sv[tid] = best; si[tid] = bi; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint s = 256; s > 0; s >>= 1) {\n"
"            if (tid < s) {\n"
"                const float ov = sv[tid + s]; const int oi = si[tid + s];\n"
"                if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; }\n"
"            }\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        }\n"
"        if (tid == 0) { chosen[k] = si[0]; chosenv[k] = sv[0]; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) {\n"
"        float ws = 0.0f;\n"
"        for (uint k = 0; k < p.K; k++) ws += exp(chosenv[k] - mx);\n"
"        for (uint k = 0; k < p.K; k++) { sel[k] = chosen[k]; wts[k] = exp(chosenv[k] - mx) / ws; }\n"
"        sel[p.K] = 0;\n"
"        wts[p.K] = 1.0f / (1.0f + exp(-gates[p.NE]));\n"
"    }\n"
"}\n"
"kernel void moe_combine(const device float* down [[buffer(0)]], const device float* wts [[buffer(1)]], device float* out [[buffer(2)]],\n"
"    constant Q4RtParams& p [[buffer(3)]], uint d [[thread_position_in_grid]])\n"
"{\n"
"    if (d >= p.H) return;\n"
"    float a = 0.0f;\n"
"    for (uint k = 0; k <= p.K; k++) a += wts[k] * down[(ulong)k * p.H + d];\n"
"    out[d] = a;\n"
"}\n"
;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso;
    /* pesi persistenti — caricati UNA VOLTA da llmmetal_gemv_q4_load_weights, riusati per ogni
     * token di decode. La versione precedente riallocava/copiava Wq (megabyte) ad ogni chiamata:
     * misurava soprattutto newBufferWithBytes+memcpy, non il kernel — non rappresentativo del
     * decode reale, dove i pesi sono fissi e solo x cambia ad ogni step. */
    id<MTLBuffer> w_buf, scale_buf, bias_buf, out_buf, x_buf;
    id<MTLComputePipelineState> pso_multi;   /* esperto-major: K esperti in UN dispatch */
    id<MTLComputePipelineState> pso_mid, pso_down;  /* fusi come ds4: mid(silu) e down(accumulo) */
    id<MTLComputePipelineState> pso_gates, pso_topk, pso_comb;   /* router + top-k + combine su GPU */
    id<MTLBuffer> mk_rw, mk_gates, mk_wts, mk_comb; uint32_t rt_ne;
    id<MTLBuffer> mk_w, mk_s, mk_b, mk_x, mk_out, mk_sel;   /* riusati a ogni token: nessuna alloc per dispatch */
    size_t mk_lw, mk_ls, mk_lb; uint32_t mk_cap_x, mk_cap_out;
    size_t lead_w, lead_s, lead_b;   /* offset del tensore dentro la view page-aligned (nocopy) */
    uint32_t H, M, group_size;
    uint32_t scale_bf16;   /* 1 = scales/biases BF16 (default 0 = half) */
    size_t w_stride, s_stride;   /* vista 3D (esperti): byte per esperto */
    uint32_t n_experts;
    const uint8_t *p_w, *p_s, *p_b;   /* vista 3D: puntatori base ai tensori interi (mmap) */
} LlmMetalGemvQ4Ctx;

static id<MTLBuffer> llmmetal_gq4_view(id<MTLDevice> dev, const uint8_t* p, size_t len, size_t* lead);

/* Zero-copy: le view Metal puntano DIRETTAMENTE al mmap delle safetensors (page-aligned);
 * il lead del tensore viene applicato come offset al momento del dispatch. */
static int llmmetal_gemv_q4_load_weights_nocopy(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                                const uint16_t* scales, const uint16_t* biases,
                                                uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx || !Wq || !scales || H % 64 != 0) return -1;
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    const size_t page = (size_t)getpagesize();
    const size_t wbytes = (size_t)M * (H / 2);
    const size_t ngrp = H / group_size;
    const size_t sbytes = (size_t)M * ngrp * sizeof(uint16_t);
    const uint8_t* wp = Wq; const uint8_t* sp = (const uint8_t*)scales; const uint8_t* bp = (const uint8_t*)biases;
    uintptr_t wb = (uintptr_t)wp & ~(uintptr_t)(page - 1); ctx->lead_w = (size_t)((uintptr_t)wp - wb);
    uintptr_t sb = (uintptr_t)sp & ~(uintptr_t)(page - 1); ctx->lead_s = (size_t)((uintptr_t)sp - sb);
    uintptr_t bb = bp ? ((uintptr_t)bp & ~(uintptr_t)(page - 1)) : 0; ctx->lead_b = bp ? (size_t)((uintptr_t)bp - bb) : 0;
    size_t wl = ((ctx->lead_w + wbytes + page - 1) / page) * page;
    size_t sl = ((ctx->lead_s + sbytes + page - 1) / page) * page;
    size_t bl = bp ? ((ctx->lead_b + sbytes + page - 1) / page) * page : 0;
    ctx->w_buf = [ctx->device newBufferWithBytesNoCopy:(void*)wb length:wl options:MTLResourceStorageModeShared deallocator:nil];
    ctx->scale_buf = [ctx->device newBufferWithBytesNoCopy:(void*)sb length:sl options:MTLResourceStorageModeShared deallocator:nil];
    ctx->bias_buf = bp ? [ctx->device newBufferWithBytesNoCopy:(void*)bb length:bl options:MTLResourceStorageModeShared deallocator:nil] : nil;
    ctx->out_buf = [ctx->device newBufferWithLength:(NSUInteger)M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(NSUInteger)H * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    if (ctx->w_buf && ctx->scale_buf) {
        const uint8_t* wc = (const uint8_t*)[ctx->w_buf contents] + ctx->lead_w;
        const uint8_t* sc = (const uint8_t*)[ctx->scale_buf contents] + ctx->lead_s;
        int badw = memcmp(wc, Wq, 8) != 0, bads = memcmp(sc, scales, 8) != 0;
        if (badw || bads) fprintf(stderr, "[nocopy] mismatch w=%d s=%d lead_w=%zu lead_s=%zu\n", badw, bads, ctx->lead_w, ctx->lead_s);
    }
    if (!ctx->w_buf || !ctx->scale_buf || !ctx->out_buf || !ctx->x_buf) {
        fprintf(stderr, "llmmetal_gemv_q4_load_weights_nocopy: view fallita (lead w=%zu s=%zu)\n", ctx->lead_w, ctx->lead_s);
        return -1;
    }
    return 0;
}

/* Zero-copy SOLO sui pesi (i grandi); scales/biases sono piccoli e vengono copiati: evita i
 * problemi di allineamento dei tipi half su view file-backed. */
static int llmmetal_gemv_q4_load_weights_wonly_nocopy(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                                      const uint16_t* scales, const uint16_t* biases,
                                                      uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx || !Wq || !scales || H % 64 != 0) return -1;
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    const size_t page = (size_t)getpagesize();
    const size_t wbytes = (size_t)M * (H / 2);
    const size_t ngrp = H / group_size;
    const size_t sbytes = (size_t)M * ngrp * sizeof(uint16_t);
    uintptr_t wb = (uintptr_t)Wq & ~(uintptr_t)(page - 1); ctx->lead_w = (size_t)((uintptr_t)Wq - wb);
    size_t wl = ((ctx->lead_w + wbytes + page - 1) / page) * page;
    ctx->w_buf = [ctx->device newBufferWithBytesNoCopy:(void*)wb length:wl options:MTLResourceStorageModeShared deallocator:nil];
    ctx->scale_buf = [ctx->device newBufferWithBytes:scales length:sbytes options:MTLResourceStorageModeShared];
    ctx->bias_buf = biases ? [ctx->device newBufferWithBytes:biases length:sbytes options:MTLResourceStorageModeShared] : nil;
    ctx->lead_s = 0; ctx->lead_b = 0;
    ctx->out_buf = [ctx->device newBufferWithLength:(NSUInteger)M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(NSUInteger)H * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    return (ctx->w_buf && ctx->scale_buf && ctx->out_buf && ctx->x_buf) ? 0 : -1;
}

/* device/queue/libreria/pipeline sono condivisi da tutti i contesti: compilare il sorgente Metal a ogni create
 * costava ~ms per gemv (migliaia di create per token nel percorso packed MoE). */
static id<MTLDevice> g_gq4_device = nil; static id<MTLCommandQueue> g_gq4_queue = nil;
static id<MTLLibrary> g_gq4_library = nil; static id<MTLComputePipelineState> g_gq4_pso = nil;
static id<MTLComputePipelineState> g_gq4_pso_multi = nil;
static id<MTLComputePipelineState> g_gq4_pso_mid = nil, g_gq4_pso_down = nil;
static id<MTLComputePipelineState> g_gq4_pso_gates = nil, g_gq4_pso_topk = nil, g_gq4_pso_comb = nil;
static id<MTLResidencySet> g_gq4_rset = nil;   /* le viste 3D (centinaia di MB) restano residenti: evita la validazione a ogni commit */
static void llmmetal_gq4_make_resident(id<MTLDevice> dev, id<MTLCommandQueue> q, NSArray* bufs) {
    if (@available(macOS 15.0, *)) {
        @synchronized([NSProcessInfo processInfo]) {
            if (!g_gq4_rset) {
                MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
                d.initialCapacity = 1024;
                NSError* err = nil;
                g_gq4_rset = [dev newResidencySetWithDescriptor:d error:&err];
                if (!g_gq4_rset) { fprintf(stderr, "gemv_q4: residency set non disponibile: %s\n", err.localizedDescription.UTF8String); return; }
                [q addResidencySet:g_gq4_rset];
            }
            for (id<MTLBuffer> b in bufs) [g_gq4_rset addAllocation:b];
            [g_gq4_rset commit];
        }
    }
}

/* Vista 3D [E][M][H]: zero-copy su pesi, scales e biases dell'INTERO tensore; l'esperto si sceglie a run-time
 * con gli offset lead_* (nessuna copia e nessun nuovo contesto per esperto). */
static int llmmetal_gemv_q4_load_weights_3d_nocopy(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                                   const uint16_t* scales, const uint16_t* biases,
                                                   uint32_t H, uint32_t M, uint32_t group_size,
                                                   uint32_t E, size_t w_stride, size_t s_stride) {
    if (!ctx || !Wq || !scales || !biases || H % 64 != 0 || E == 0) return -1;
    ctx->H = H; ctx->M = M; ctx->group_size = group_size; ctx->n_experts = E;
    ctx->w_stride = w_stride; ctx->s_stride = s_stride;
    ctx->p_w = Wq; ctx->p_s = (const uint8_t*)scales; ctx->p_b = (const uint8_t*)biases;
    ctx->out_buf = [ctx->device newBufferWithLength:(NSUInteger)M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(NSUInteger)H * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    /* Buffer del percorso esperto-major: allocati UNA volta (prima erano per-dispatch -> leak/thrash). */
    const size_t wbytes = (size_t)E * w_stride, sbytes = (size_t)E * s_stride;
    ctx->mk_w = llmmetal_gq4_view(ctx->device, ctx->p_w, wbytes, &ctx->mk_lw);
    ctx->mk_s = llmmetal_gq4_view(ctx->device, ctx->p_s, sbytes, &ctx->mk_ls);
    ctx->mk_b = llmmetal_gq4_view(ctx->device, ctx->p_b, sbytes, &ctx->mk_lb);
    ctx->mk_cap_x = H + 16u * (H > M ? H : M);
    ctx->mk_cap_out = 16u * M;
    ctx->mk_x = [ctx->device newBufferWithLength:(NSUInteger)ctx->mk_cap_x * 4 options:MTLResourceStorageModeShared];
    ctx->mk_out = [ctx->device newBufferWithLength:(NSUInteger)ctx->mk_cap_out * 4 options:MTLResourceStorageModeShared];
    ctx->mk_sel = [ctx->device newBufferWithLength:64 * sizeof(int32_t) options:MTLResourceStorageModeShared];
    if (ctx->mk_w && ctx->mk_s && ctx->mk_b) llmmetal_gq4_make_resident(ctx->device, ctx->queue, @[ctx->mk_w, ctx->mk_s, ctx->mk_b]);
    return (ctx->out_buf && ctx->x_buf && ctx->mk_w && ctx->mk_s && ctx->mk_b && ctx->mk_x && ctx->mk_out && ctx->mk_sel) ? 0 : -1;
}

static int llmmetal_gemv_q4_init(LlmMetalGemvQ4Ctx* ctx) {
    ctx->device = nil; ctx->queue = nil; ctx->library = nil; ctx->pso = nil;
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->bias_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    @synchronized([NSProcessInfo processInfo]) {
        if (g_gq4_pso) { ctx->device = g_gq4_device; ctx->queue = g_gq4_queue; ctx->library = g_gq4_library; ctx->pso = g_gq4_pso; ctx->pso_multi = g_gq4_pso_multi; ctx->pso_mid = g_gq4_pso_mid; ctx->pso_down = g_gq4_pso_down; ctx->pso_gates = g_gq4_pso_gates; ctx->pso_topk = g_gq4_pso_topk; ctx->pso_comb = g_gq4_pso_comb; return 0; }
    }
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { fprintf(stderr, "llmmetal_gemv_q4_init: nessun device Metal\n"); return -1; }
    ctx->queue = [ctx->device newCommandQueue];
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:llmmetal_gemv_q4_src];
    ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&err];
    if (!ctx->library) { fprintf(stderr, "llmmetal_gemv_q4_init: compile err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> f = [ctx->library newFunctionWithName:@"gemv_q4_affine"];
    ctx->pso = [ctx->device newComputePipelineStateWithFunction:f error:&err];
    if (!ctx->pso) { fprintf(stderr, "llmmetal_gemv_q4_init: pso err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> fm = [ctx->library newFunctionWithName:@"gemv_q4_affine_multi"];
    ctx->pso_multi = fm ? [ctx->device newComputePipelineStateWithFunction:fm error:&err] : nil;
    if (!ctx->pso_multi) fprintf(stderr, "llmmetal_gemv_q4_init: pso_multi non disponibile\n");
    id<MTLFunction> fmid = [ctx->library newFunctionWithName:@"moe_mid"];
    id<MTLFunction> fdown = [ctx->library newFunctionWithName:@"moe_down"];
    ctx->pso_mid = fmid ? [ctx->device newComputePipelineStateWithFunction:fmid error:&err] : nil;
    ctx->pso_down = fdown ? [ctx->device newComputePipelineStateWithFunction:fdown error:&err] : nil;
    if (!ctx->pso_mid || !ctx->pso_down) fprintf(stderr, "llmmetal_gemv_q4_init: pso moe non disponibile\n");
    { id<MTLFunction> f1 = [ctx->library newFunctionWithName:@"moe_gates"], f2 = [ctx->library newFunctionWithName:@"moe_topk"], f3 = [ctx->library newFunctionWithName:@"moe_combine"];
      ctx->pso_gates = f1 ? [ctx->device newComputePipelineStateWithFunction:f1 error:&err] : nil;
      ctx->pso_topk = f2 ? [ctx->device newComputePipelineStateWithFunction:f2 error:&err] : nil;
      ctx->pso_comb = f3 ? [ctx->device newComputePipelineStateWithFunction:f3 error:&err] : nil;
      if (!ctx->pso_gates || !ctx->pso_topk || !ctx->pso_comb) fprintf(stderr, "llmmetal_gemv_q4_init: pso router non disponibile (%s)\n", err.localizedDescription.UTF8String); }
    @synchronized([NSProcessInfo processInfo]) { g_gq4_device = ctx->device; g_gq4_queue = ctx->queue; g_gq4_library = ctx->library; g_gq4_pso = ctx->pso; g_gq4_pso_multi = ctx->pso_multi; g_gq4_pso_mid = ctx->pso_mid; g_gq4_pso_down = ctx->pso_down; g_gq4_pso_gates = ctx->pso_gates; g_gq4_pso_topk = ctx->pso_topk; g_gq4_pso_comb = ctx->pso_comb; }
    return 0;
}

static void llmmetal_gemv_q4_free(LlmMetalGemvQ4Ctx* ctx) {
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->bias_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    ctx->pso = nil; ctx->library = nil; ctx->queue = nil; ctx->device = nil;
}

/* Carica i pesi quantizzati UNA VOLTA (equivalente al load del modello): Wq [M,H/2] uint8,
 * scales/biases [M,H/group_size] half. Alloca anche i buffer persistenti per x/out. */
static int llmmetal_gemv_q4_load_weights(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                           const uint16_t* scales, const uint16_t* biases,
                                           uint32_t H, uint32_t M, uint32_t group_size) {
    if (H % 64 != 0) { fprintf(stderr, "llmmetal_gemv_q4_load_weights: H %u non multiplo di 64\n", H); return -1; }
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    uint32_t n_groups = H / group_size;
    ctx->w_buf = [ctx->device newBufferWithBytes:Wq length:(size_t)M*(H/2) options:MTLResourceStorageModeShared];
    ctx->scale_buf = [ctx->device newBufferWithBytes:scales length:(size_t)M*n_groups*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->bias_buf = [ctx->device newBufferWithBytes:biases length:(size_t)M*n_groups*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->out_buf = [ctx->device newBufferWithLength:(size_t)M*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(size_t)H*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    if (!ctx->w_buf || !ctx->scale_buf || !ctx->bias_buf || !ctx->out_buf || !ctx->x_buf) {
        fprintf(stderr, "llmmetal_gemv_q4_load_weights: allocazione fallita\n"); return -1;
    }
    return 0;
}

/* Un passo di decode: scrive x (memcpy in memoria unificata, come llmmetal_kv_write), esegue
 * il kernel sui pesi gia' residenti, legge out. Nessuna riallocazione/copia dei pesi. */
static int llmmetal_gemv_q4_run_bufs(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out,
                                     id<MTLBuffer> wb, id<MTLBuffer> sb, id<MTLBuffer> bb,
                                     size_t lead_w, size_t lead_s, size_t lead_b) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg, lead_w, lead_s, lead_b, scale_bf16; } params = {
        ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMV_ROWS_PER_TG,
        (uint32_t)lead_w, (uint32_t)lead_s, (uint32_t)lead_b, ctx->scale_bf16 };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:sb offset:0 atIndex:2];
    [enc setBuffer:bb offset:0 atIndex:3];
    [enc setBuffer:ctx->out_buf offset:0 atIndex:4];
    [enc setBytes:&params length:sizeof(params) atIndex:5];
    [enc setThreadgroupMemoryLength:(ctx->H*sizeof(uint16_t)) atIndex:0];
    uint32_t n_tg = (ctx->M + LLMMETAL_GEMV_ROWS_PER_TG - 1) / LLMMETAL_GEMV_ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->M*sizeof(uint16_t));
    return 0;
}

static int llmmetal_gemv_q4_run(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    return llmmetal_gemv_q4_run_bufs(ctx, x, out, ctx->w_buf, ctx->scale_buf, ctx->bias_buf, ctx->lead_w, ctx->lead_s, ctx->lead_b);
}

/* view zero-copy (page-aligned) su [p, p+len): ritorna il buffer e il lead */
static id<MTLBuffer> llmmetal_gq4_view(id<MTLDevice> dev, const uint8_t* p, size_t len, size_t* lead) {
    const size_t page = (size_t)getpagesize();
    uintptr_t b = (uintptr_t)p & ~(uintptr_t)(page - 1);
    *lead = (size_t)((uintptr_t)p - b);
    const size_t l = ((*lead + len + page - 1) / page) * page;
    return [dev newBufferWithBytesNoCopy:(void*)b length:l options:MTLResourceStorageModeShared deallocator:nil];
}

/* esperto `e` di una vista 3D (llmmetal_gemv_q4_load_weights_3d_nocopy) */
static int llmmetal_gemv_q4_run_expert(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out, uint32_t e) {
    if (e >= ctx->n_experts) return -1;
    const size_t wbytes = (size_t)ctx->M * (ctx->H / 2), sbytes = (size_t)ctx->M * (ctx->H / ctx->group_size) * 2;
    size_t lw = 0, ls = 0, lb = 0;
    id<MTLBuffer> wb = llmmetal_gq4_view(ctx->device, ctx->p_w + (size_t)e * ctx->w_stride, wbytes, &lw);
    id<MTLBuffer> sb = llmmetal_gq4_view(ctx->device, ctx->p_s + (size_t)e * ctx->s_stride, sbytes, &ls);
    id<MTLBuffer> bb = llmmetal_gq4_view(ctx->device, ctx->p_b + (size_t)e * ctx->s_stride, sbytes, &lb);
    if (!wb || !sb || !bb) return -1;
    return llmmetal_gemv_q4_run_bufs(ctx, x, out, wb, sb, bb, lw, ls, lb);
}

/* Esperto-major come ds4: UN solo dispatch per tutti i K esperti selezionati.
 * out_all ha K*M half (esperto k -> out_all[k*M]). experts[] sono gli indici nel tensore 3D. */
static int llmmetal_gemv_q4_run_experts(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out_all,
                                        const int32_t* experts, int K, uint32_t x_stride) {
    if (!ctx || !ctx->pso_multi || K <= 0 || K > 16 || !ctx->mk_w) return -1;
    const size_t xneed = (size_t)ctx->H + (size_t)x_stride * (size_t)(K - 1);
    const size_t oneed = (size_t)ctx->M * (size_t)K;
    if (xneed > ctx->mk_cap_x || oneed > ctx->mk_cap_out) return -1;
    memcpy(ctx->mk_x.contents, x, xneed * 2);
    memcpy(ctx->mk_sel.contents, experts, (size_t)K * sizeof(int32_t));
    id<MTLBuffer> xb = ctx->mk_x, ob = ctx->mk_out, sbuf = ctx->mk_sel;
    id<MTLBuffer> wb = ctx->mk_w, sb = ctx->mk_s, bb = ctx->mk_b;
    const size_t lw = ctx->mk_lw, ls = ctx->mk_ls, lb = ctx->mk_lb;
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride; uint32_t x_stride; } params = {
        ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMV_ROWS_PER_TG, ctx->scale_bf16,
        (uint32_t)ctx->w_stride, (uint32_t)ctx->s_stride, x_stride };
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso_multi];
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:wb offset:lw atIndex:1];
    [enc setBuffer:sb offset:ls atIndex:2];
    [enc setBuffer:bb offset:lb atIndex:3];
    [enc setBuffer:ob offset:0 atIndex:4];
    [enc setBuffer:sbuf offset:0 atIndex:5];
    [enc setBytes:&params length:sizeof(params) atIndex:6];
    [enc setThreadgroupMemoryLength:(ctx->H * sizeof(uint16_t)) atIndex:0];
    const uint32_t n_tg = (ctx->M + LLMMETAL_GEMV_ROWS_PER_TG - 1) / LLMMETAL_GEMV_ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, (NSUInteger)K, 1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out_all, ob.contents, (size_t)ctx->M * 2 * (size_t)K);
    return 0;
}

/* ds4: mid e down nello STESSO command buffer (un solo commit per layer); i dispatch
 * dentro un encoder sono eseguiti in ordine, quindi down vede il mid gia' scritto. */
static int llmmetal_gemv_q4_run_moe(LlmMetalGemvQ4Ctx* cg, LlmMetalGemvQ4Ctx* cu, LlmMetalGemvQ4Ctx* sg, LlmMetalGemvQ4Ctx* su,
                                    LlmMetalGemvQ4Ctx* cd, LlmMetalGemvQ4Ctx* sd,
                                    const uint16_t* x, float* act, float* out_all,
                                    const int32_t* experts, int K, uint32_t x_stride) {
    if (!cg || !cu || !cd || !cg->pso_mid || !cd->pso_down || K <= 0 || K > 15) return -1;
    const int NS = (sg && su && sd) ? 1 : 0;
    { float* xd=(float*)cg->mk_x.contents; for (uint32_t i=0;i<cg->H;i++){ _Float16 h; memcpy(&h,&x[i],2); xd[i]=(float)h; } }
    memcpy(cg->mk_sel.contents, experts, (size_t)K * sizeof(int32_t));
    memcpy(cd->mk_sel.contents, experts, (size_t)K * sizeof(int32_t));   /* il down legge il PROPRIO buffer sel */
    id<MTLCommandBuffer> cb = [cg->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride, x_stride; uint32_t n_shared, n_routed; } pm = {
        cg->H, cg->M, cg->group_size, LLMMETAL_GEMV_ROWS_PER_TG, cg->scale_bf16,
        (uint32_t)cg->w_stride, (uint32_t)cg->s_stride, 0, (uint32_t)NS, (uint32_t)K };
    [enc setComputePipelineState:cg->pso_mid];
    [enc setBuffer:cg->mk_x offset:0 atIndex:0];
    [enc setBuffer:cg->mk_w offset:cg->mk_lw atIndex:1];
    [enc setBuffer:cg->mk_s offset:cg->mk_ls atIndex:2];
    [enc setBuffer:cg->mk_b offset:cg->mk_lb atIndex:3];
    [enc setBuffer:cu->mk_w offset:cu->mk_lw atIndex:4];
    [enc setBuffer:cu->mk_s offset:cu->mk_ls atIndex:5];
    [enc setBuffer:cu->mk_b offset:cu->mk_lb atIndex:6];
    [enc setBuffer:(sg ? sg->w_buf : cg->mk_w) offset:(sg ? sg->lead_w : cg->mk_lw) atIndex:7];
    [enc setBuffer:(sg ? sg->scale_buf : cg->mk_s) offset:(sg ? sg->lead_s : cg->mk_ls) atIndex:8];
    [enc setBuffer:(sg ? sg->bias_buf : cg->mk_b) offset:(sg ? sg->lead_b : cg->mk_lb) atIndex:9];
    [enc setBuffer:(su ? su->w_buf : cg->mk_w) offset:(su ? su->lead_w : cg->mk_lw) atIndex:10];
    [enc setBuffer:(su ? su->scale_buf : cg->mk_s) offset:(su ? su->lead_s : cg->mk_ls) atIndex:11];
    [enc setBuffer:(su ? su->bias_buf : cg->mk_b) offset:(su ? su->lead_b : cg->mk_lb) atIndex:12];
    [enc setBuffer:cg->mk_out offset:0 atIndex:13];
    [enc setBuffer:cg->mk_sel offset:0 atIndex:14];
    [enc setBytes:&pm length:sizeof(pm) atIndex:15];
    [enc setThreadgroupMemoryLength:(cg->H * sizeof(float)) atIndex:0];
    { const uint32_t n_tg = (cg->M + 2*LLMMETAL_GEMV_ROWS_PER_TG - 1) / (2*LLMMETAL_GEMV_ROWS_PER_TG);
      [enc dispatchThreadgroups:MTLSizeMake(n_tg, (NSUInteger)(K + NS), 1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)]; }
    /* down: legge cg->mk_out; serve una BARRIERA di memoria fra i due dispatch (Metal non
     * garantisce l'ordine fra dispatch consecutivi dello stesso encoder). */
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride, x_stride; uint32_t n_shared, n_routed; } pd = {
        cd->H, cd->M, cd->group_size, LLMMETAL_GEMV_ROWS_PER_TG, cd->scale_bf16,
        (uint32_t)cd->w_stride, (uint32_t)cd->s_stride, x_stride, (uint32_t)NS, (uint32_t)K };
    [enc setComputePipelineState:cd->pso_down];
    [enc setBuffer:cg->mk_out offset:0 atIndex:0];
    [enc setBuffer:cd->mk_w offset:cd->mk_lw atIndex:1];
    [enc setBuffer:cd->mk_s offset:cd->mk_ls atIndex:2];
    [enc setBuffer:cd->mk_b offset:cd->mk_lb atIndex:3];
    [enc setBuffer:(sd ? sd->w_buf : cd->mk_w) offset:(sd ? sd->lead_w : cd->mk_lw) atIndex:4];
    [enc setBuffer:(sd ? sd->scale_buf : cd->mk_s) offset:(sd ? sd->lead_s : cd->mk_ls) atIndex:5];
    [enc setBuffer:(sd ? sd->bias_buf : cd->mk_b) offset:(sd ? sd->lead_b : cd->mk_lb) atIndex:6];
    [enc setBuffer:cd->mk_out offset:0 atIndex:7];
    [enc setBuffer:cd->mk_sel offset:0 atIndex:8];
    [enc setBytes:&pd length:sizeof(pd) atIndex:9];
    [enc setThreadgroupMemoryLength:(cd->H * sizeof(float)) atIndex:0];
    { const uint32_t n_tg = (cd->M + 2*LLMMETAL_GEMV_ROWS_PER_TG - 1) / (2*LLMMETAL_GEMV_ROWS_PER_TG);
      [enc dispatchThreadgroups:MTLSizeMake(n_tg, (NSUInteger)(K + NS), 1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)]; }
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "[moe] cb error: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
    memcpy(act, cg->mk_out.contents, sizeof(float) * (size_t)cg->M * (size_t)(K + NS));
    memcpy(out_all, cd->mk_out.contents, sizeof(float) * (size_t)cd->M * (size_t)(K + NS));
    return 0;
}

/* FUSIONE stile ds4: mid (gate+up+silu) e down (accumulo pesato) in DUE dispatch per layer.
 * act_out ha K*F half; out_all ha K*M half. wts sono i pesi del router (f32). */
static int llmmetal_gemv_q4_run_moe_mid(LlmMetalGemvQ4Ctx* cg, LlmMetalGemvQ4Ctx* cu,
                                        LlmMetalGemvQ4Ctx* sg, LlmMetalGemvQ4Ctx* su,
                                        const uint16_t* x, float* act_out,
                                        const int32_t* experts, int K) {
    if (!cg || !cu || !cg->pso_mid || K <= 0 || K > 15) return -1;
    const int NS = sg && su ? 1 : 0;
    { float* xd=(float*)cg->mk_x.contents; for (uint32_t i=0;i<cg->H;i++){ _Float16 h; memcpy(&h,&x[i],2); xd[i]=(float)h; } }
    memcpy(cg->mk_sel.contents, experts, (size_t)K * sizeof(int32_t));
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride, x_stride; uint32_t n_shared, n_routed; } p = {
        cg->H, cg->M, cg->group_size, LLMMETAL_GEMV_ROWS_PER_TG, cg->scale_bf16,
        (uint32_t)cg->w_stride, (uint32_t)cg->s_stride, 0, (uint32_t)NS, (uint32_t)K };
    id<MTLCommandBuffer> cb = [cg->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:cg->pso_mid];
    [enc setBuffer:cg->mk_x offset:0 atIndex:0];
    [enc setBuffer:cg->mk_w offset:cg->mk_lw atIndex:1];
    [enc setBuffer:cg->mk_s offset:cg->mk_ls atIndex:2];
    [enc setBuffer:cg->mk_b offset:cg->mk_lb atIndex:3];
    [enc setBuffer:cu->mk_w offset:cu->mk_lw atIndex:4];
    [enc setBuffer:cu->mk_s offset:cu->mk_ls atIndex:5];
    [enc setBuffer:cu->mk_b offset:cu->mk_lb atIndex:6];
    [enc setBuffer:(sg ? sg->w_buf : cg->mk_w) offset:(sg ? sg->lead_w : cg->mk_lw) atIndex:7];
    [enc setBuffer:(sg ? sg->scale_buf : cg->mk_s) offset:(sg ? sg->lead_s : cg->mk_ls) atIndex:8];
    [enc setBuffer:(sg ? sg->bias_buf : cg->mk_b) offset:(sg ? sg->lead_b : cg->mk_lb) atIndex:9];
    [enc setBuffer:(su ? su->w_buf : cg->mk_w) offset:(su ? su->lead_w : cg->mk_lw) atIndex:10];
    [enc setBuffer:(su ? su->scale_buf : cg->mk_s) offset:(su ? su->lead_s : cg->mk_ls) atIndex:11];
    [enc setBuffer:(su ? su->bias_buf : cg->mk_b) offset:(su ? su->lead_b : cg->mk_lb) atIndex:12];
    [enc setBuffer:cg->mk_out offset:0 atIndex:13];
    [enc setBuffer:cg->mk_sel offset:0 atIndex:14];
    [enc setBytes:&p length:sizeof(p) atIndex:15];
    [enc setThreadgroupMemoryLength:(cg->H * sizeof(float)) atIndex:0];
    const uint32_t n_tg = (cg->M + 2*LLMMETAL_GEMV_ROWS_PER_TG - 1) / (2*LLMMETAL_GEMV_ROWS_PER_TG);
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, (NSUInteger)(K + NS), 1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "[moe-mid] cb error: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
    memcpy(act_out, cg->mk_out.contents, sizeof(float) * (size_t)cg->M * (size_t)(K + NS));
    return 0;
}

static int llmmetal_gemv_q4_run_moe_down(LlmMetalGemvQ4Ctx* cd, LlmMetalGemvQ4Ctx* sd,
                                         const float* act,
                                         float* out_all, const int32_t* experts, int K, uint32_t x_stride) {
    if (!cd || !cd->pso_down || K <= 0 || K > 15) return -1;
    const int NS = sd ? 1 : 0;
    const size_t xneed = (size_t)cd->H + (size_t)x_stride * (size_t)(K + NS - 1);   /* K routed + eventuale slot shared */
    if (xneed > cd->mk_cap_x) return -1;
    memcpy(cd->mk_x.contents, act, xneed * sizeof(float));
    memcpy(cd->mk_sel.contents, experts, (size_t)K * sizeof(int32_t));
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride, x_stride; uint32_t n_shared, n_routed; } p = {
        cd->H, cd->M, cd->group_size, LLMMETAL_GEMV_ROWS_PER_TG, cd->scale_bf16,
        (uint32_t)cd->w_stride, (uint32_t)cd->s_stride, x_stride, (uint32_t)NS, (uint32_t)K };
    id<MTLCommandBuffer> cb = [cd->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:cd->pso_down];
    [enc setBuffer:cd->mk_x offset:0 atIndex:0];
    [enc setBuffer:cd->mk_w offset:cd->mk_lw atIndex:1];
    [enc setBuffer:cd->mk_s offset:cd->mk_ls atIndex:2];
    [enc setBuffer:cd->mk_b offset:cd->mk_lb atIndex:3];
    [enc setBuffer:(sd ? sd->w_buf : cd->mk_w) offset:(sd ? sd->lead_w : cd->mk_lw) atIndex:4];
    [enc setBuffer:(sd ? sd->scale_buf : cd->mk_s) offset:(sd ? sd->lead_s : cd->mk_ls) atIndex:5];
    [enc setBuffer:(sd ? sd->bias_buf : cd->mk_b) offset:(sd ? sd->lead_b : cd->mk_lb) atIndex:6];
    [enc setBuffer:cd->mk_out offset:0 atIndex:7];
    [enc setBuffer:cd->mk_sel offset:0 atIndex:8];
    [enc setBytes:&p length:sizeof(p) atIndex:9];
    [enc setThreadgroupMemoryLength:(cd->H * sizeof(float)) atIndex:0];
    const uint32_t n_tg = (cd->M + 2*LLMMETAL_GEMV_ROWS_PER_TG - 1) / (2*LLMMETAL_GEMV_ROWS_PER_TG);
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, (NSUInteger)(K + NS), 1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(out_all, cd->mk_out.contents, sizeof(float) * (size_t)cd->M * (size_t)(K + NS));
    return 0;
}


/* Router + top-k + mid + down + combine in UN SOLO command buffer (come ds4: nessun passaggio dalla CPU fra i layer-stage).
 * Il tensore dei pesi del router ([NE+1][H] f32, l'ultima riga = gate dello shared expert) si carica una volta su cg. */
static int llmmetal_gemv_q4_router_upload(LlmMetalGemvQ4Ctx* cg, const float* W, int rows, int H) {
    if (!cg || !W || rows < 2 || H <= 0 || (uint32_t)H != cg->H) return -1;
    cg->rt_ne = (uint32_t)(rows - 1);
    cg->mk_rw = [cg->device newBufferWithBytes:W length:(size_t)rows * (size_t)H * 4 options:MTLResourceStorageModeShared];
    cg->mk_gates = [cg->device newBufferWithLength:(size_t)rows * 4 options:MTLResourceStorageModeShared];
    cg->mk_wts = [cg->device newBufferWithLength:17 * 4 options:MTLResourceStorageModeShared];
    cg->mk_comb = [cg->device newBufferWithLength:(size_t)H * 4 options:MTLResourceStorageModeShared];
    if (!cg->mk_rw || !cg->mk_gates || !cg->mk_wts || !cg->mk_comb) return -1;
    llmmetal_gq4_make_resident(cg->device, cg->queue, @[cg->mk_rw]);
    return 0;
}

static int llmmetal_gemv_q4_run_moe_gpu(LlmMetalGemvQ4Ctx* cg, LlmMetalGemvQ4Ctx* cu, LlmMetalGemvQ4Ctx* sg, LlmMetalGemvQ4Ctx* su,
                                        LlmMetalGemvQ4Ctx* cd, LlmMetalGemvQ4Ctx* sd, const float* x, float* out, int K) {
    if (!cg || !cu || !cd || !sg || !su || !sd || !cg->pso_mid || !cd->pso_down || !cg->pso_gates || !cg->pso_topk || !cg->pso_comb ||
        !cg->mk_rw || K <= 0 || K > 15 || cd->M != cg->H || cg->rt_ne > 512) return -1;
    memcpy(cg->mk_x.contents, x, (size_t)cg->H * sizeof(float));
    id<MTLCommandBuffer> cb = [cg->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    struct { uint32_t H, NE, K, rows_per_tg; } rp = { cg->H, cg->rt_ne, (uint32_t)K, LLMMETAL_GEMV_ROWS_PER_TG };
    /* 1) gates = X . W_router^T (+ riga dello shared expert) */
    [enc setComputePipelineState:cg->pso_gates];
    [enc setBuffer:cg->mk_x offset:0 atIndex:0]; [enc setBuffer:cg->mk_rw offset:0 atIndex:1]; [enc setBuffer:cg->mk_gates offset:0 atIndex:2];
    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((cg->rt_ne + 1 + LLMMETAL_GEMV_ROWS_PER_TG - 1) / LLMMETAL_GEMV_ROWS_PER_TG, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG * 32, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    /* 2) softmax + top-K + pesi normalizzati -> sel[], wts[] */
    [enc setComputePipelineState:cg->pso_topk];
    [enc setBuffer:cg->mk_gates offset:0 atIndex:0]; [enc setBuffer:cg->mk_sel offset:0 atIndex:1]; [enc setBuffer:cg->mk_wts offset:0 atIndex:2];
    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
    [enc setThreadgroupMemoryLength:1056 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    /* 3) mid */
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride, x_stride; uint32_t n_shared, n_routed; } pm = {
        cg->H, cg->M, cg->group_size, LLMMETAL_GEMV_ROWS_PER_TG, cg->scale_bf16,
        (uint32_t)cg->w_stride, (uint32_t)cg->s_stride, 0, 1u, (uint32_t)K };
    [enc setComputePipelineState:cg->pso_mid];
    [enc setBuffer:cg->mk_x offset:0 atIndex:0];
    [enc setBuffer:cg->mk_w offset:cg->mk_lw atIndex:1]; [enc setBuffer:cg->mk_s offset:cg->mk_ls atIndex:2]; [enc setBuffer:cg->mk_b offset:cg->mk_lb atIndex:3];
    [enc setBuffer:cu->mk_w offset:cu->mk_lw atIndex:4]; [enc setBuffer:cu->mk_s offset:cu->mk_ls atIndex:5]; [enc setBuffer:cu->mk_b offset:cu->mk_lb atIndex:6];
    [enc setBuffer:sg->w_buf offset:sg->lead_w atIndex:7]; [enc setBuffer:sg->scale_buf offset:sg->lead_s atIndex:8]; [enc setBuffer:sg->bias_buf offset:sg->lead_b atIndex:9];
    [enc setBuffer:su->w_buf offset:su->lead_w atIndex:10]; [enc setBuffer:su->scale_buf offset:su->lead_s atIndex:11]; [enc setBuffer:su->bias_buf offset:su->lead_b atIndex:12];
    [enc setBuffer:cg->mk_out offset:0 atIndex:13]; [enc setBuffer:cg->mk_sel offset:0 atIndex:14];
    [enc setBytes:&pm length:sizeof(pm) atIndex:15];
    [enc setThreadgroupMemoryLength:(cg->H * sizeof(float)) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((cg->M + 2*LLMMETAL_GEMV_ROWS_PER_TG - 1) / (2*LLMMETAL_GEMV_ROWS_PER_TG), (NSUInteger)(K + 1), 1)
        threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    /* 4) down (legge act da cg->mk_out, sel dallo STESSO buffer scritto dal top-k) */
    struct { uint32_t H, M, group_size, rows_per_tg, scale_bf16; uint32_t w_stride, s_stride, x_stride; uint32_t n_shared, n_routed; } pd = {
        cd->H, cd->M, cd->group_size, LLMMETAL_GEMV_ROWS_PER_TG, cd->scale_bf16,
        (uint32_t)cd->w_stride, (uint32_t)cd->s_stride, cd->H, 1u, (uint32_t)K };
    [enc setComputePipelineState:cd->pso_down];
    [enc setBuffer:cg->mk_out offset:0 atIndex:0];
    [enc setBuffer:cd->mk_w offset:cd->mk_lw atIndex:1]; [enc setBuffer:cd->mk_s offset:cd->mk_ls atIndex:2]; [enc setBuffer:cd->mk_b offset:cd->mk_lb atIndex:3];
    [enc setBuffer:sd->w_buf offset:sd->lead_w atIndex:4]; [enc setBuffer:sd->scale_buf offset:sd->lead_s atIndex:5]; [enc setBuffer:sd->bias_buf offset:sd->lead_b atIndex:6];
    [enc setBuffer:cd->mk_out offset:0 atIndex:7]; [enc setBuffer:cg->mk_sel offset:0 atIndex:8];
    [enc setBytes:&pd length:sizeof(pd) atIndex:9];
    [enc setThreadgroupMemoryLength:(cd->H * sizeof(float)) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((cd->M + 2*LLMMETAL_GEMV_ROWS_PER_TG - 1) / (2*LLMMETAL_GEMV_ROWS_PER_TG), (NSUInteger)(K + 1), 1)
        threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    /* 5) combine pesata: out = sum_{k<=K} wts[k] * down[k] */
    struct { uint32_t H, NE, K, rows_per_tg; } cp = { cd->M, cg->rt_ne, (uint32_t)K, 0 };
    [enc setComputePipelineState:cg->pso_comb];
    [enc setBuffer:cd->mk_out offset:0 atIndex:0]; [enc setBuffer:cg->mk_wts offset:0 atIndex:1]; [enc setBuffer:cg->mk_comb offset:0 atIndex:2];
    [enc setBytes:&cp length:sizeof(cp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((cd->M + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "[moe-gpu] cb error: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
    memcpy(out, cg->mk_comb.contents, (size_t)cd->M * sizeof(float));
    return 0;
}
/* Batch: N gemv (anche su contesti/tensori diversi) in UN solo command buffer. expert < 0 = lineare 2D (viste del contesto).
 * x/out sono buffer transitori per job, quindi lo stesso contesto puo' comparire piu' volte (esperti diversi). */
typedef struct { LlmMetalGemvQ4Ctx* ctx; int64_t expert; const uint16_t* x; uint16_t* out; } LlmMetalGemvQ4Job;

static int llmmetal_gemv_q4_run_batch(const LlmMetalGemvQ4Job* jobs, int n) {
    if (n <= 0) return 0;
    LlmMetalGemvQ4Ctx* c0 = jobs[0].ctx;
    id<MTLCommandBuffer> cb = [c0->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c0->pso];
    NSMutableArray* keep = [NSMutableArray arrayWithCapacity:(NSUInteger)n * 5];
    NSMutableArray* outs = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    for (int i = 0; i < n; i++) {
        LlmMetalGemvQ4Ctx* c = jobs[i].ctx;
        id<MTLBuffer> wb, sb, bb; size_t lw, ls, lb;
        if (jobs[i].expert >= 0) {
            if ((uint64_t)jobs[i].expert >= c->n_experts) { [enc endEncoding]; return -1; }
            const size_t e = (size_t)jobs[i].expert;
            const size_t wbytes = (size_t)c->M * (c->H / 2), sbytes = (size_t)c->M * (c->H / c->group_size) * 2;
            wb = llmmetal_gq4_view(c->device, c->p_w + e * c->w_stride, wbytes, &lw);
            sb = llmmetal_gq4_view(c->device, c->p_s + e * c->s_stride, sbytes, &ls);
            bb = llmmetal_gq4_view(c->device, c->p_b + e * c->s_stride, sbytes, &lb);
            if (!wb || !sb || !bb) { [enc endEncoding]; return -1; }
        } else { wb = c->w_buf; sb = c->scale_buf; bb = c->bias_buf; lw = c->lead_w; ls = c->lead_s; lb = c->lead_b; }
        id<MTLBuffer> xb = [c->device newBufferWithBytes:jobs[i].x length:(size_t)c->H * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [c->device newBufferWithLength:(size_t)c->M * 2 options:MTLResourceStorageModeShared];
        if (!xb || !ob) { [enc endEncoding]; return -1; }
        [keep addObject:wb]; [keep addObject:sb]; [keep addObject:bb]; [keep addObject:xb]; [outs addObject:ob];
        struct { uint32_t H, M, group_size, rows_per_tg, lead_w, lead_s, lead_b, scale_bf16; } params = {
            c->H, c->M, c->group_size, LLMMETAL_GEMV_ROWS_PER_TG, (uint32_t)lw, (uint32_t)ls, (uint32_t)lb, c->scale_bf16 };
        [enc setBuffer:xb offset:0 atIndex:0]; [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:sb offset:0 atIndex:2]; [enc setBuffer:bb offset:0 atIndex:3];
        [enc setBuffer:ob offset:0 atIndex:4];
        [enc setBytes:&params length:sizeof(params) atIndex:5];
        [enc setThreadgroupMemoryLength:(c->H * sizeof(uint16_t)) atIndex:0];
        const uint32_t n_tg = (c->M + LLMMETAL_GEMV_ROWS_PER_TG - 1) / LLMMETAL_GEMV_ROWS_PER_TG;
        [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    for (int i = 0; i < n; i++) memcpy(jobs[i].out, ((id<MTLBuffer>)outs[(NSUInteger)i]).contents, (size_t)jobs[i].ctx->M * 2);
    return 0;
}

#endif /* LLMMETAL_METAL_GEMV_Q4_H */
