/* llm_dsv4_metal.mm — backend Metal DS4 (IQ2_XXS/Q2_K), ObjC++/Metal, ABI C.
 * Kernel GEMV single-token con attivazioni f32 (parità con llm_dsv4_dot_tensor).
 * Zero-copy sul mmap del GGUF via "view" page-aligned (pattern ds4_metal.m).
 * Le tabelle IQ2 sono byte-exact dal CODEC (llm_dsv4_metal_tables.h, generato). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "llm_dsv4_metal.h"
#include "llm_engine_opts.h"
#include "llm_dsv4_metal_tables.h"
#include "llm_dsv4_metal_mmid.h"   /* MoE batched mul_mm_id (estratto da ds4) */
#include "llm_dsv4_metal_mmid_mpp.h"   /* stessa cosa su tensor ops MPP (Metal 4), libreria separata */
#include "llm_dsv4_actquant.h"   /* riuso: llm_dsv4_q8_K_quantize_row (metodo ds4) */

/* TEMP (dsv4_profile): tempo GPU REALE (timestamp MSL `GPUStartTime/GPUEndTime` del command
 * buffer), accumulato per classe di op: [0]=altro, [1]=attn, [2]=MoE. Etichettato dal chiamante
 * (llm_dsv4_gpu_ctx.mm) prima della call. Immune al carico CPU. */
static double g_dsv4_mtl_gpu_ms[4] = {0, 0, 0, 0};
static int    g_dsv4_mtl_op = 0;
/* TEMP (dsv4_profile): conteggio dispatch Metal (una chiamata setComputePipelineState ~ un dispatch). */
long g_dsv4_dispatchs = 0;
long llm_dsv4_metal_dispatches(void) { return g_dsv4_dispatchs; }
void llm_dsv4_metal_set_op(int op) { g_dsv4_mtl_op = (op >= 0 && op < 4) ? op : 0; }
void llm_dsv4_metal_gpu_ms(double* attn, double* moe) {
    if (attn) *attn = g_dsv4_mtl_gpu_ms[1];
    if (moe)  *moe  = g_dsv4_mtl_gpu_ms[2];
}
static void dsv4_gpu_acct(id<MTLCommandBuffer> cb) {
    double s = (double)cb.GPUStartTime, e = (double)cb.GPUEndTime;
    if (e > s) g_dsv4_mtl_gpu_ms[g_dsv4_mtl_op] += (e - s) * 1e3;
}

/* ── sorgente kernel (le tabelle IQ2 vengono anteposte a runtime) ── */
static const char* const LLM_DSV4_METAL_KERNELS =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"static inline float dsv4_iq2_xxs_value(device const uchar* blk, uint idx) {\n"
"    ushort dh = (ushort)blk[0] | ((ushort)blk[1] << 8);\n"
"    float d = (float)as_type<half>(dh);\n"
"    uint ib = idx / 32u, r = idx % 32u, li = r / 8u, j = r % 8u;\n"
"    device const uchar* aux = blk + 2u + 8u * ib;\n"
"    uint aux1 = (uint)aux[4] | ((uint)aux[5] << 8) | ((uint)aux[6] << 16) | ((uint)aux[7] << 24);\n"
"    uint ls = 2u * (aux1 >> 28) + 1u;\n"
"    uint sidx = (aux1 >> (7u * li)) & 127u;\n"
"    ulong gv = iq2xxs_grid[aux[li]];\n"
"    float v = (float)(uchar)((gv >> (8u * j)) & 0xffu);\n"
"    if ((ksigns_iq2xs[sidx] & kmask_iq2xs[j]) != 0u) v = -v;\n"
"    return 0.125f * d * (float)ls * v;\n"
"}\n"
"static inline float dsv4_q2_K_value(device const uchar* blk, uint idx) {\n"
"    uint group = idx / 16u, l = idx % 16u;\n"
"    uint q_base = 32u * (group / 8u) + 16u * (group & 1u);\n"
"    uint shift = ((group / 2u) & 3u) * 2u;\n"
"    uint q = ((uint)blk[16u + q_base + l] >> shift) & 3u;\n"
"    uint sc = blk[group];\n"
"    ushort dh = (ushort)blk[80] | ((ushort)blk[81] << 8);\n"
"    ushort mh = (ushort)blk[82] | ((ushort)blk[83] << 8);\n"
"    float d = (float)as_type<half>(dh), mn = (float)as_type<half>(mh);\n"
"    return d * (float)(sc & 0x0Fu) * (float)q - mn * (float)(sc >> 4u);\n"
"}\n"
"static inline float dsv4_q8_0_value(device const uchar* blk, uint i) {\n"
"    ushort dh = (ushort)blk[0] | ((ushort)blk[1] << 8);\n"
"    float d = (float)as_type<half>(dh);\n"
"    return d * (float)(char)blk[2u + i];\n"
"}\n"
"kernel void dsv4_gemv(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* O [[buffer(2)]], constant uint* P [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint voff = P[0], row_bytes = P[1], nblk = P[2], out_dim = P[3], type = P[4], blk_bytes = P[5], blk_elems = P[6];\n"
"    if (tg >= out_dim) return;\n"
"    device const uchar* row = W + voff + (ulong)tg * (ulong)row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < nblk; ++b) {\n"
"        device const uchar* blk = row + (ulong)b * (ulong)blk_bytes;\n"
"        uint base = b * blk_elems;\n"
"        for (uint i = tid; i < blk_elems; i += 32u) {\n"
"            float wv = (type == 16u) ? dsv4_iq2_xxs_value(blk, i)\n"
"                     : (type == 10u) ? dsv4_q2_K_value(blk, i)\n"
"                     : dsv4_q8_0_value(blk, i);\n"
"            acc += wv * X[base + i];\n"
"        }\n"
"    }\n"
"    float s = simd_sum(acc); if (tid == 0u) O[tg] = s;\n"
"}\n"
/* GEMV Q8_0 veloce: un intero blocco (32 elementi) per iterazione, scale caricato 1x/blocco.
 * P = {voff, row_bytes, nblk, out_dim}. Stessa semantica di dsv4_gemv(type=8). */
"kernel void dsv4_gemv_q8_0(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* O [[buffer(2)]], constant uint* P [[buffer(3)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint voff = P[0], row_bytes = P[1], nblk = P[2], out_dim = P[3];\n"
"    if (tg >= out_dim) return;\n"
"    device const uchar* row = W + voff + (ulong)tg * (ulong)row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint b = tid; b < nblk; b += 32u) {\n"
"        device const uchar* blk = row + (ulong)b * 34ul;\n"
"        ushort dh = (ushort)blk[0] | ((ushort)blk[1] << 8);\n"
"        float d = (float)as_type<half>(dh);\n"
"        const uint base = b * 32u; float s = 0.0f;\n"
"        for (uint i = 0; i < 32u; i += 4) {\n"
"            s += (float)(char)blk[2u+i]     * X[base+i]\n"
"               + (float)(char)blk[2u+i+1u]  * X[base+i+1u]\n"
"               + (float)(char)blk[2u+i+2u]  * X[base+i+2u]\n"
"               + (float)(char)blk[2u+i+3u]  * X[base+i+3u];\n"
"        }\n"
"        acc += d * s;\n"
"    }\n"
"    float r = simd_sum(acc); if (tid == 0u) O[tg] = r;\n"
"}\n"
/* FUSO: gate+up+swiglu in UN kernel (era 2 gemv + 1 swiglu = 3 dispatch). Stessa aritmetica del
 * dsv4_gemv_q8_0 (Q8_0, simd_sum) + dsv4_swiglu (clamp). P={vg,rbg,vu,rbu,nblk,ff}, CF={clamp}. */
"kernel void dsv4_pair_swiglu_q8_0(device const uchar* WG [[buffer(0)]], device const uchar* WU [[buffer(1)]],\n"
"    device const float* X [[buffer(2)]], device float* MID [[buffer(3)]], constant uint* P [[buffer(4)]],\n"
"    constant float* CF [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint vg = P[0], rbg = P[1], vu = P[2], rbu = P[3], nblk = P[4], out_dim = P[5];\n"
"    if (tg >= out_dim) return;\n"
"    device const uchar* rg = WG + vg + (ulong)tg * (ulong)rbg;\n"
"    device const uchar* ru = WU + vu + (ulong)tg * (ulong)rbu;\n"
"    float accg = 0.0f, accu = 0.0f;\n"
"    for (uint b = tid; b < nblk; b += 32u) {\n"
"        device const uchar* bg = rg + (ulong)b * 34ul; device const uchar* bu = ru + (ulong)b * 34ul;\n"
"        ushort dgh = (ushort)bg[0] | ((ushort)bg[1] << 8);\n"
"        ushort duh = (ushort)bu[0] | ((ushort)bu[1] << 8);\n"
"        float dg = (float)as_type<half>(dgh);\n"
"        float du = (float)as_type<half>(duh);\n"
"        const uint base = b * 32u; float sg = 0.0f, su = 0.0f;\n"
"        for (uint i = 0; i < 32u; i += 4) {\n"
"            float x0=X[base+i], x1=X[base+i+1u], x2=X[base+i+2u], x3=X[base+i+3u];\n"
"            sg += (float)(char)bg[2u+i]*x0 + (float)(char)bg[2u+i+1u]*x1 + (float)(char)bg[2u+i+2u]*x2 + (float)(char)bg[2u+i+3u]*x3;\n"
"            su += (float)(char)bu[2u+i]*x0 + (float)(char)bu[2u+i+1u]*x1 + (float)(char)bu[2u+i+2u]*x2 + (float)(char)bu[2u+i+3u]*x3;\n"
"        }\n"
"        accg += dg * sg; accu += du * su;\n"
"    }\n"
"    float g = simd_sum(accg), u = simd_sum(accu);\n"
"    if (tid == 0u) { const float c = CF[0]; if (c > 1.0e-6f) { g = min(g, c); u = clamp(u, -c, c); } MID[tg] = (g / (1.f + exp(-g))) * u; }\n"
"}\n"
/* FUSO: down (Q8_0) + OUT += (era 2 dispatch). P={voff,row_bytes,nblk,out_dim,out_off}. */
"kernel void dsv4_gemv_q8_0_add(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* OUT [[buffer(2)]], constant uint* P [[buffer(3)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint voff = P[0], row_bytes = P[1], nblk = P[2], out_dim = P[3], out_off = P[4];\n"
"    if (tg >= out_dim) return;\n"
"    device const uchar* row = W + voff + (ulong)tg * (ulong)row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint b = tid; b < nblk; b += 32u) {\n"
"        device const uchar* blk = row + (ulong)b * 34ul;\n"
"        ushort dh = (ushort)blk[0] | ((ushort)blk[1] << 8); float d = (float)as_type<half>(dh);\n"
"        const uint base = b * 32u; float s = 0.0f;\n"
"        for (uint i = 0; i < 32u; i += 4) { s += (float)(char)blk[2u+i]*X[base+i] + (float)(char)blk[2u+i+1u]*X[base+i+1u] + (float)(char)blk[2u+i+2u]*X[base+i+2u] + (float)(char)blk[2u+i+3u]*X[base+i+3u]; }\n"
"        acc += d * s;\n"
"    }\n"
"    float r = simd_sum(acc); if (tid == 0u) OUT[(ulong)out_off + tg] += r;\n"
"}\n"
/* ── Attn decode packed8 (PORT del vendored ds4_flash_attn_vec_packed8_reduce_f16_512):
 * 1 threadgroup/head, 256 thread = 8 simdgroup, time-slice dei 32 split virtuari di 32 chiavi,
 * online-softmax con simd_max/simd_sum, reduce finale IN-THREADGROUP (nessun buffer parziale su
 * device, nessuna merge separata). Legge i nostri buffer misti f32 (RAWK finestra + COMPK
 * compresse) con maschera `allowed`, K=V. hd=512. P={n_raw,n_comp,hd,has_allowed,raw_off}. ── */
"kernel void dsv4_attn_decode_p8(device const float* Q [[buffer(0)]], device const float* RAWK [[buffer(1)]],\n"
"    device const float* COMPK [[buffer(2)]], device const int* ALLOWED [[buffer(3)]],\n"
"    device const float* SINKS [[buffer(4)]], device float* OUT [[buffer(5)]], constant uint* P [[buffer(6)]],\n"
"    threadgroup char* shmem [[threadgroup(0)]], uint h [[threadgroup_position_in_grid]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint n_raw = P[0], n_comp = P[1], hd = P[2], has_allowed = P[3], raw_off = P[4];\n"
"    const int n_keys = (int)(n_raw + n_comp);\n"
"    const float scale = 1.f / sqrt((float)hd);\n"
"    const short NW = 32, C = 32, NSG = 8, NWG = 32, DK4 = 128, DV4 = 128, SH = 128;\n"
"    threadgroup float4* q_shared = (threadgroup float4*)shmem;\n"
"    threadgroup half* score_banks = (threadgroup half*)(q_shared + DK4);\n"
"    threadgroup volatile float* weights = (threadgroup volatile float*)(score_banks + NSG*SH);\n"
"    threadgroup volatile float* stats = weights + NWG*C;\n"
"    threadgroup volatile float* sink_scale = stats + 2*NWG;\n"
"    threadgroup volatile float4* plane = (threadgroup volatile float4*)(sink_scale + NWG);\n"
"    const short lane = (short)tiisg;\n"
"    threadgroup half* bank = score_banks + (short)sgitg * SH;\n"
"    threadgroup float* ss = (threadgroup float*)bank;\n"
"    threadgroup half* sm = bank + 2*C;\n"
"    device const float4* q4 = (device const float4*)(Q + (ulong)h * (ulong)hd);\n"
"    if (sgitg == 0) { for (short i = lane; i < DK4; i += NW) q_shared[i] = q4[i]; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {\n"
"        float S = 0.f; float M = -FLT_MAX/2; float oscale = 1.f;\n"
"        const int ic0 = (int)iwg * C;\n"
"        weights[(uint)iwg*C + (uint)lane] = 0.f; ss[lane] = 0.f; sm[lane] = (half)0.0h;\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (ic0 < n_keys) {\n"
"            { const int kk = ic0 + (int)lane; int ok = (kk < n_keys);\n"
"              if (ok && kk >= (int)n_raw && has_allowed && !ALLOWED[kk - (int)n_raw]) ok = 0;\n"
"              sm[lane] = ok ? (half)0.0h : (half)(-INFINITY); }\n"
"            if (simd_max(sm[lane]) > (half)(-65504.0f)) {\n"
"                float lane_mqk = 0.f;\n"
"                for (short cc = 0; cc < C; ++cc) {\n"
"                    const int kk = ic0 + cc; if (kk >= n_keys) continue;\n"
"                    device const float* krow = (kk < (int)n_raw) ? (RAWK + raw_off + (ulong)kk*(ulong)hd) : (COMPK + (ulong)(kk-(int)n_raw)*(ulong)hd);\n"
"                    device const float4* pk4 = (device const float4*)krow;\n"
"                    float mqk = 0.f;\n"
"                    for (short ii = 0; ii < DK4/NW; ++ii) mqk += dot(pk4[ii*NW + lane], q_shared[ii*NW + lane]);\n"
"                    mqk = simd_sum(mqk);\n"
"                    if (lane == cc) lane_mqk = mqk;\n"
"                }\n"
"                ss[lane] = fma(lane_mqk, scale, (float)sm[lane]);\n"
"                simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"                const float old_m = M; const float score = ss[lane];\n"
"                M = simd_max(max(M, score));\n"
"                const float ms = exp(old_m - M); const float vs = exp(score - M);\n"
"                S = S * ms + simd_sum(vs); ss[lane] = vs;\n"
"                simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"                weights[(uint)iwg*C + (uint)lane] = ss[lane];\n"
"            }\n"
"            if (iwg == 0) {\n"
"                const float old_m = M; const float sink = lane == 0 ? SINKS[h] : -FLT_MAX/2;\n"
"                M = simd_max(max(M, sink)); const float ms = exp(old_m - M); const float vs = exp(sink - M);\n"
"                S = S*ms + simd_sum(vs); oscale = ms;\n"
"            }\n"
"        } else if (iwg == 0) {\n"
"            const float old_m = M; const float sink = lane == 0 ? SINKS[h] : -FLT_MAX/2;\n"
"            M = simd_max(max(M, sink)); const float ms = exp(old_m - M); const float vs = exp(sink - M);\n"
"            S = S*ms + simd_sum(vs); oscale = ms;\n"
"        }\n"
"        if (lane == 0) { stats[2*(uint)iwg] = S; stats[2*(uint)iwg+1] = M; sink_scale[(uint)iwg] = oscale; }\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const short split = lane;\n"
"    float rS = stats[2*(uint)split]; float rM = stats[2*(uint)split+1];\n"
"    const float rmax = simd_max(rM); const float rms = exp(rM - rmax);\n"
"    rS = simd_sum(rS * rms);\n"
"    const float rinv = rS == 0.f ? 0.f : 1.f/rS;\n"
"    device float4* dst4 = (device float4*)(OUT + (ulong)h*(ulong)hd);\n"
"    for (short quad = 0; quad < 4; ++quad) {\n"
"        for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {\n"
"            float4 lo = float4(0.f); const int ic0 = (int)iwg*C;\n"
"            if (ic0 < n_keys) {\n"
"                threadgroup volatile float* sw = weights + (uint)iwg*C;\n"
"                const short oc = quad*NW + lane;\n"
"                for (short cc = 0; cc < C; ++cc) {\n"
"                    const int kk = ic0 + cc; if (kk >= n_keys) continue;\n"
"                    device const float* vrow = (kk < (int)n_raw) ? (RAWK + raw_off + (ulong)kk*(ulong)hd) : (COMPK + (ulong)(kk-(int)n_raw)*(ulong)hd);\n"
"                    lo += ((device const float4*)vrow)[oc] * float4(sw[cc]);\n"
"                }\n"
"                if (iwg == 0) lo *= sink_scale[0];\n"
"            }\n"
"            plane[(uint)iwg*33u + (uint)lane] = lo;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (short ol = (short)sgitg; ol < NW; ol += NSG) {\n"
"            const float4 m = (float4)plane[(uint)lane*33u + (uint)ol];\n"
"            const float4 red = simd_sum(m * rms);\n"
"            if (lane == 0) dst4[quad*NW + ol] = red * rinv;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"}\n";

/* ── Metodo ds4: attivazione Q8_K + integer-dot a blocchi (8 elementi per lookup di griglia).
 * Bit-exact col riferimento CPU `llm_dsv4_dot_w_q8_K` (llm_dsv4_actquant.h). ── */
static const char* const LLM_DSV4_METAL_Q8K =
"struct q8k { float d; char qs[256]; short bsums[16]; };\n"
"static inline int iq2_entry_dot(device const uchar* blk, uint ib, uint l, device const char* q8) {\n"
"    device const uchar* aux = blk + 2u + 8u * ib;\n"
"    uint aux1 = (uint)aux[4] | ((uint)aux[5] << 8) | ((uint)aux[6] << 16) | ((uint)aux[7] << 24);\n"
"    uint sidx = (aux1 >> (7u * l)) & 127u;\n"
"    ulong gv = iq2xxs_grid[aux[l]];\n"
"    int s = 0;\n"
"    for (uint j = 0u; j < 8u; j++) { int v = (int)((gv >> (8u * j)) & 0xffu);\n"
"        if ((ksigns_iq2xs[sidx] & kmask_iq2xs[j]) != 0u) v = -v;\n"
"        s += v * (int)q8[j]; }\n"
"    return s;\n"
"}\n"
"kernel void dsv4_gemv_q8k_iq2(device const uchar* W [[buffer(0)]], device const uchar* Y [[buffer(1)]], device float* O [[buffer(2)]], constant uint* P [[buffer(3)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint lead = P[0], row_bytes = P[1], nblk = P[2], out_dim = P[3];\n"
"    uint row = gid / nblk, b = gid % nblk;\n"
"    if (row >= out_dim) return;\n"
"    device const uchar* blk = W + lead + (ulong)row * (ulong)row_bytes + (ulong)b * 66u;\n"
"    device const struct q8k* y = (device const struct q8k*)(Y + (ulong)b * 292u);\n"
"    threadgroup int sh[32];\n"
"    threadgroup int out8[8];\n"
"    uint ib = tid / 4u, l = tid % 4u;\n"
"    sh[tid] = iq2_entry_dot(blk, ib, l, y->qs + (ulong)ib * 32u + (ulong)l * 8u);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 8u) {\n"
"        device const uchar* aux = blk + 2u + 8u * tid;\n"
"        uint aux1 = (uint)aux[4] | ((uint)aux[5] << 8) | ((uint)aux[6] << 16) | ((uint)aux[7] << 24);\n"
"        uint ls = 2u * (aux1 >> 28) + 1u;\n"
"        int s = sh[tid*4] + sh[tid*4+1] + sh[tid*4+2] + sh[tid*4+3];\n"
"        out8[tid] = s * (int)ls;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) {\n"
"        int bsum = 0; for (uint z = 0u; z < 8u; z++) bsum += out8[z];\n"
"        ushort dh = (ushort)blk[0] | ((ushort)blk[1] << 8);\n"
"        float d = (float)as_type<half>(dh);\n"
"        O[(ulong)row * nblk + b] = 0.125f * d * y->d * (float)bsum;\n"
"    }\n"
"}\n"
"static inline int q2_16(device const uchar* q2, device const char* q8, uint shift) {\n"
"    int s = 0; for (uint j = 0u; j < 16u; j++) s += (int)(((uint)q2[j] >> shift) & 3u) * (int)q8[j]; return s;\n"
"}\n"
"kernel void dsv4_gemv_q8k_q2k(device const uchar* W [[buffer(0)]], device const uchar* Y [[buffer(1)]], device float* O [[buffer(2)]], constant uint* P [[buffer(3)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint lead = P[0], row_bytes = P[1], nblk = P[2], out_dim = P[3];\n"
"    uint row = gid / nblk, b = gid % nblk;\n"
"    if (row >= out_dim) return;\n"
"    device const uchar* blk = W + lead + (ulong)row * (ulong)row_bytes + (ulong)b * 84u;\n"
"    device const struct q8k* y = (device const struct q8k*)(Y + (ulong)b * 292u);\n"
"    device const uchar* sc = blk; device const uchar* qs = blk + 16; device const char* q8 = y->qs;\n"
"    threadgroup int sh[16];\n"
"    if (tid < 16u) {\n"
"        uint c = tid / 2u, hf = tid % 2u;\n"
"        device const uchar* q2 = qs + (ulong)(c / 4u) * 32u + (ulong)hf * 16u;\n"
"        device const char* q8p = q8 + (ulong)c * 32u + (ulong)hf * 16u;\n"
"        sh[tid] = (int)(sc[2u*c + hf] & 0x0fu) * q2_16(q2, q8p, (c % 4u) * 2u);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) {\n"
"        int isum = 0; for (uint t = 0u; t < 16u; t++) isum += sh[t];\n"
"        int summs = 0; for (uint j = 0u; j < 16u; j++) summs += (int)y->bsums[j] * (int)(sc[j] >> 4);\n"
"        ushort dh = (ushort)blk[80] | ((ushort)blk[81] << 8);\n"
"        ushort mh = (ushort)blk[82] | ((ushort)blk[83] << 8);\n"
"        float dall = y->d * (float)as_type<half>(dh), dmin = y->d * (float)as_type<half>(mh);\n"
"        O[(ulong)row * nblk + b] = dall * (float)isum - dmin * (float)summs;\n"
"    }\n"
"}\n"
"static inline void q4k_scale_min(uint j, device const uchar* q, thread uchar* sd, thread uchar* sm) {\n"
"    if (j < 4u) { *sd = q[j] & 63u; *sm = q[j+4u] & 63u; }\n"
"    else { *sd = (q[j+4u] & 0x0Fu) | ((q[j-4u] >> 6) << 4); *sm = (q[j+4u] >> 4) | ((q[j] >> 6) << 4); }\n"
"}\n"
"kernel void dsv4_gemv_q8k_q4k(device const uchar* W [[buffer(0)]], device const uchar* Y [[buffer(1)]], device float* O [[buffer(2)]], constant uint* P [[buffer(3)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    uint lead = P[0], row_bytes = P[1], nblk = P[2], out_dim = P[3];\n"
"    uint row = gid / nblk, b = gid % nblk;\n"
"    if (row >= out_dim) return;\n"
"    device const uchar* blk = W + lead + (ulong)row * (ulong)row_bytes + (ulong)b * 144u;\n"
"    device const struct q8k* y = (device const struct q8k*)(Y + (ulong)b * 292u);\n"
"    device const uchar* sc = blk + 4; device const uchar* qs = blk + 16; device const char* q8 = y->qs;\n"
"    threadgroup int sh[8]; threadgroup int sh2[8];\n"
"    if (tid < 8u) {\n"
"        uchar scv, mv; q4k_scale_min(tid, sc, &scv, &mv);\n"
"        uint bo = (tid >> 1) * 32u, shf = (tid & 1u) * 4u;\n"
"        int s = 0;\n"
"        for (uint l = 0u; l < 32u; l++) s += (int)(((uint)qs[bo+l] >> shf) & 0x0Fu) * (int)q8[tid*32u + l];\n"
"        sh[tid]  = (int)scv * s;\n"
"        sh2[tid] = (int)mv * ((int)y->bsums[2u*tid] + (int)y->bsums[2u*tid+1]);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) {\n"
"        int isum = 0, summs = 0; for (uint z = 0u; z < 8u; z++) { isum += sh[z]; summs += sh2[z]; }\n"
"        ushort dh = (ushort)blk[0] | ((ushort)blk[1] << 8);\n"
"        ushort mh = (ushort)blk[2] | ((ushort)blk[3] << 8);\n"
"        float dall = y->d * (float)as_type<half>(dh), dmin = -(y->d * (float)as_type<half>(mh));\n"
"        O[(ulong)row * nblk + b] = dall * (float)isum + dmin * (float)summs;\n"
"    }\n"
"}\n"
/* Somma ORDINATA dei parziali per riga (deterministica: niente atomic float). P={nblk, out_dim}. */
"kernel void dsv4_red_parts(device const float* PT [[buffer(0)]], device float* O [[buffer(1)]], constant uint* P [[buffer(2)]], uint row [[thread_position_in_grid]]) {\n"
"    const uint nblk = P[0]; if (row >= P[1]) return;\n"
"    float s = 0.f; for (uint b = 0; b < nblk; b++) s += PT[(ulong)row * nblk + b];\n"
"    O[row] = s;\n"
"}\n";

/* Port FEDELE di ds4 `kernel_mul_mv_iq2_xxs_pair_f32_impl` (metal/moe.metal:3137): simdgroup,
 * nr0=4 righe/simdgroup, QK_K=256, gate+up in un pass, attivazioni f32; swiglu×peso in coda. */
static const char* const LLM_DSV4_METAL_MOE =
"static inline void dsv4_pair_core(device const char* SG, device const char* SU, device const float* Y,\n"
"    device float* MID, float rw, int mid_off, float clampv, int ne00, int nb01, int ne0,\n"
"    uint leadg, uint leadu, uint tgx, ushort tiisg, ushort sgitg, threadgroup char* shmem) {\n"
"    const int nr0 = 4; const short NSG = 4;\n"
"    const int nb = ne00 / 256; const int first_row = (tgx * NSG + sgitg) * nr0;\n"
"    device const char* xg = SG + (ulong)leadg + (ulong)first_row * (ulong)nb01;\n"
"    device const char* xu = SU + (ulong)leadu + (ulong)first_row * (ulong)nb01;\n"
"    float yl[32]; float sumg[4] = {0.f,0.f,0.f,0.f}; float sumu[4] = {0.f,0.f,0.f,0.f};\n"
"    const int nb32 = nb * 8;\n"
"    threadgroup ulong* sv = (threadgroup ulong*)shmem;\n"
"    threadgroup uchar* ss = (threadgroup uchar*)(sv + 256);\n"
"    { int nval = 4, pos = (32*sgitg + tiisg)*nval; for (int i=0;i<nval;i++) sv[pos+i] = iq2xxs_grid[pos+i];\n"
"      nval = 2; pos = (32*sgitg + tiisg)*nval; for (int i=0;i<nval;i++) ss[pos+i] = ksigns_iq2xs[pos+i];\n"
"      threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    device const float* y4 = Y + 32*tiisg;\n"
"    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {\n"
"        for (short i = 0; i < 32; ++i) yl[i] = y4[i];\n"
"        const int ibl = ib32 / 8, ib = ib32 % 8;\n"
"        device const char* bg = xg + (ulong)ibl * 66;\n"
"        device const char* bu = xu + (ulong)ibl * 66;\n"
"        device const ushort* qg = (device const ushort*)(bg + 2) + 4*ib;\n"
"        device const ushort* qu = (device const ushort*)(bu + 2) + 4*ib;\n"
"        device const half* dhg = (device const half*)bg;\n"
"        device const half* dhu = (device const half*)bu;\n"
"        for (short row = 0; row < nr0; ++row) {\n"
"            device const uchar* a8g = (device const uchar*)qg;\n"
"            device const uchar* a8u = (device const uchar*)qu;\n"
"            const uint a32g = qg[2] | (qg[3] << 16);\n"
"            const uint a32u = qu[2] | (qu[3] << 16);\n"
"            const float dg = (float)dhg[0] * (0.5f + (a32g >> 28));\n"
"            const float du = (float)dhu[0] * (0.5f + (a32u >> 28));\n"
"            float sg = 0.f, su = 0.f;\n"
"            for (short l = 0; l < 4; ++l) {\n"
"                const threadgroup uchar* gg = (const threadgroup uchar*)(sv + a8g[l]);\n"
"                const threadgroup uchar* gu = (const threadgroup uchar*)(sv + a8u[l]);\n"
"                const uchar signg = ss[(a32g >> 7*l) & 127u];\n"
"                const uchar signu = ss[(a32u >> 7*l) & 127u];\n"
"                for (short j = 0; j < 8; ++j) {\n"
"                    const float v = yl[8*l + j];\n"
"                    sg += v * (float)gg[j] * ((signg & kmask_iq2xs[j]) ? -1.f : 1.f);\n"
"                    su += v * (float)gu[j] * ((signu & kmask_iq2xs[j]) ? -1.f : 1.f);\n"
"                }\n"
"            }\n"
"            sumg[row] += dg * sg; sumu[row] += du * su;\n"
"            dhg = (device const half*)((device const char*)dhg + nb01);\n"
"            dhu = (device const half*)((device const char*)dhu + nb01);\n"
"            qg  = (device const ushort*)((device const char*)qg + nb01);\n"
"            qu  = (device const ushort*)((device const char*)qu + nb01);\n"
"        }\n"
"        y4 += 32*32;\n"
"    }\n"
"    for (int row = 0; row < nr0 && first_row + row < ne0; ++row) {\n"
"        const float g = simd_sum(sumg[row]) * 0.25f;\n"
"        const float u = simd_sum(sumu[row]) * 0.25f;\n"
"        if (tiisg == 0) {\n"
"            float gg = g, uu = u;\n"
"            if (clampv > 1.0e-6f) { gg = min(gg, clampv); uu = clamp(uu, -clampv, clampv); }\n"
"            const float silu = gg / (1.f + exp(-gg));\n"
"            MID[mid_off + first_row + row] = silu * uu * rw;\n"
"        }\n"
"    }\n"
"}\n"
"kernel void dsv4_pair_swiglu_iq2(device const char* SG [[buffer(0)]], device const char* SU [[buffer(1)]],\n"
"    device const float* Y [[buffer(2)]], device float* MID [[buffer(3)]], device const float* RW [[buffer(4)]],\n"
"    constant float* CF [[buffer(5)]], constant uint* P [[buffer(6)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]], threadgroup char* shmem [[threadgroup(0)]]) {\n"
"    dsv4_pair_core(SG, SU, Y, MID, RW[0], (int)P[5], CF[0], (int)P[0], (int)P[1], (int)P[2], P[3], P[4], tgpig.x, tiisg, sgitg, shmem);\n"
"}\n"
"kernel void dsv4_slots6_pair_swiglu_iq2(\n"
"    device const char* G0 [[buffer(0)]], device const char* G1 [[buffer(1)]], device const char* G2 [[buffer(2)]],\n"
"    device const char* G3 [[buffer(3)]], device const char* G4 [[buffer(4)]], device const char* G5 [[buffer(5)]],\n"
"    device const char* U0 [[buffer(6)]], device const char* U1 [[buffer(7)]], device const char* U2 [[buffer(8)]],\n"
"    device const char* U3 [[buffer(9)]], device const char* U4 [[buffer(10)]], device const char* U5 [[buffer(11)]],\n"
"    device const float* Y [[buffer(12)]], device float* MID [[buffer(13)]], constant float* RW [[buffer(14)]],\n"
"    constant float* CF [[buffer(15)]], constant uint* P [[buffer(16)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]], threadgroup char* shmem [[threadgroup(0)]]) {\n"
"    const int idx = (int)tgpig.z;\n"
"    device const char* SG = G0; device const char* SU = U0;\n"
"    switch (idx) {\n"
"    case 1: SG = G1; SU = U1; break; case 2: SG = G2; SU = U2; break;\n"
"    case 3: SG = G3; SU = U3; break; case 4: SG = G4; SU = U4; break;\n"
"    case 5: SG = G5; SU = U5; break; default: break; }\n"
"    const int ne00 = (int)P[0], nb01 = (int)P[1], ne0 = (int)P[2];\n"
"    dsv4_pair_core(SG, SU, Y, MID, RW[idx], idx * ne0, CF[0], ne00, nb01, ne0,\n"
"                   P[3 + idx], P[9 + idx], tgpig.x, tiisg, sgitg, shmem);\n"
"}\n"
"kernel void dsv4_sum6_q2k(\n"
"    device const char* S0 [[buffer(0)]], device const char* S1 [[buffer(1)]],\n"
"    device const char* S2 [[buffer(2)]], device const char* S3 [[buffer(3)]],\n"
"    device const char* S4 [[buffer(4)]], device const char* S5 [[buffer(5)]],\n"
"    device const float* MID [[buffer(6)]], device float* OUT [[buffer(7)]],\n"
"    constant uint* P [[buffer(8)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const short NSG = 4; const short nr0 = 4;\n"
"    const int nb = (int)P[0]; const uint nb01 = P[1];\n"
"    const int ne0 = (int)P[2]; const int ne00 = (int)P[3];\n"
"    const int first_row = (tgpig.x * NSG + sgitg) * nr0;\n"
"    float sumf[4] = {0.f,0.f,0.f,0.f};\n"
"    const short ix = tiisg / 8; const short it = tiisg % 8;\n"
"    const short iq = it / 4; const short ir = it % 4; const short is = (8 * ir) / 16;\n"
"    for (int slot = 0; slot < 6; slot++) {\n"
"        device const char* s0 = (slot==0)?S0:(slot==1)?S1:(slot==2)?S2:(slot==3)?S3:(slot==4)?S4:S5;\n"
"        device const float* y = MID + (ulong)slot * (ulong)ne00;\n"
"        device const float* y4 = y + ix * 256 + 128 * iq + 8 * ir;\n"
"        for (int ib = ix; ib < nb; ib += 4) {\n"
"            float yl[32]; float4 sumy = {0.f,0.f,0.f,0.f};\n"
"            for (short i = 0; i < 8; ++i) {\n"
"                yl[i+0]=y4[i+0]; sumy[0]+=yl[i+0]; yl[i+8]=y4[i+32]; sumy[1]+=yl[i+8];\n"
"                yl[i+16]=y4[i+64]; sumy[2]+=yl[i+16]; yl[i+24]=y4[i+96]; sumy[3]+=yl[i+24]; }\n"
"            for (short row = 0; row < nr0; row++) {\n"
"                if (first_row + row < ne0) {\n"
"                    device const uchar* blk = (device const uchar*)s0 + (ulong)(first_row+row) * (ulong)nb01 + (ulong)ib * 84u;\n"
"                    device const uint8_t* scr = blk + 8*iq + is;\n"
"                    device const uint16_t* qsr = (device const uint16_t*)(blk + 16u) + 16*iq + 4*ir;\n"
"                    ushort dh0 = (ushort)blk[80] | ((ushort)blk[81] << 8);\n"
"                    ushort dh1 = (ushort)blk[82] | ((ushort)blk[83] << 8);\n"
"                    const float dall = (float)as_type<half>(dh0);\n"
"                    const float dmin = (float)as_type<half>(dh1) * (1.f/16.f);\n"
"                    float4 acc1 = {0.f,0.f,0.f,0.f}; float4 acc2 = {0.f,0.f,0.f,0.f};\n"
"                    for (int i = 0; i < 8; i += 2) {\n"
"                        acc1[0] += yl[i+0]*(qsr[i/2]&0x0003); acc2[0] += yl[i+1]*(qsr[i/2]&0x0300);\n"
"                        acc1[1] += yl[i+8]*(qsr[i/2]&0x000c); acc2[1] += yl[i+9]*(qsr[i/2]&0x0c00);\n"
"                        acc1[2] += yl[i+16]*(qsr[i/2]&0x0030); acc2[2] += yl[i+17]*(qsr[i/2]&0x3000);\n"
"                        acc1[3] += yl[i+24]*(qsr[i/2]&0x00c0); acc2[3] += yl[i+25]*(qsr[i/2]&0xc000);\n"
"                    }\n"
"                    sumf[row] += dall * ((acc1[0]+1.f/256.f*acc2[0])*(scr[0]&0xF)\n"
"                                       + (acc1[1]+1.f/256.f*acc2[1])*(scr[2]&0xF)*0.25f\n"
"                                       + (acc1[2]+1.f/256.f*acc2[2])*(scr[4]&0xF)*0.0625f\n"
"                                       + (acc1[3]+1.f/256.f*acc2[3])*(scr[6]&0xF)*0.015625f)\n"
"                               - dmin * (sumy[0]*(scr[0]&0xF0) + sumy[1]*(scr[2]&0xF0)\n"
"                                       + sumy[2]*(scr[4]&0xF0) + sumy[3]*(scr[6]&0xF0));\n"
"                }\n"
"            }\n"
"            y4 += 4 * 256;\n"
"        }\n"
"    }\n"
"    for (int row = 0; row < nr0 && first_row + row < ne0; row++) {\n"
"        const float s = simd_sum(sumf[row]);\n"
"        if (tiisg == 0) OUT[first_row + row] = s;\n"
"    }\n"
"}\n"
"kernel void dsv4_rms_norm_w(device const float* X [[buffer(0)]], device const float* W [[buffer(1)]],\n"
"    device float* O [[buffer(2)]], constant uint* P [[buffer(3)]], constant float* CF [[buffer(4)]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint n = P[0]; const float eps = CF[0];\n"
"    float s = 0.f;\n"
"    for (uint i = tiisg; i < n; i += 32u) { float v = X[i]; s += v*v; }\n"
"    s = simd_sum(s);\n"
"    const float scale = 1.f / sqrt(s / (float)n + eps);\n"
"    for (uint i = tiisg; i < n; i += 32u) O[i] = X[i] * scale * W[i];\n"
"}\n"
"kernel void dsv4_head_rms(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], constant float* CF [[buffer(2)]],\n"
"    uint h [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint hd = P[0]; const float eps = CF[0];\n"
"    device float* x = X + (ulong)h * (ulong)hd;\n"
"    float s = 0.f;\n"
"    for (uint i = tiisg; i < hd; i += 32u) { float v = x[i]; s += v*v; }\n"
"    s = simd_sum(s);\n"
"    const float scale = 1.f / sqrt(s / (float)hd + eps);\n"
"    for (uint i = tiisg; i < hd; i += 32u) x[i] *= scale;\n"
"}\n"
"kernel void dsv4_rope_tail_ext(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], constant float* CF [[buffer(2)]],\n"
"    uint h [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const int n_rot = (int)P[0], head_dim = (int)P[1], pos = (int)P[2], inverse = (int)P[3];\n"
"    const float freq_base = CF[0], freq_scale = CF[1], ext_factor = CF[2], attn_factor = CF[3];\n"
"    const float beta_fast = CF[4], beta_slow = CF[5]; const float n_ctx_orig = CF[6];\n"
"    const int n_nope = head_dim - n_rot;\n"
"    const float theta_scale = pow(freq_base, -2.f / (float)n_rot);\n"
"    const float sin_sign = inverse ? -1.f : 1.f;\n"
"    float corr0 = 0.f, corr1 = 0.f;\n"
"    const float TWO_PI = 6.28318530717958647692f;\n"
"    if (ext_factor != 0.f) {\n"
"        float start = floor((float)n_rot * log(n_ctx_orig / (beta_fast * TWO_PI)) / (2.f * log(freq_base)));\n"
"        float end = ceil((float)n_rot * log(n_ctx_orig / (beta_slow * TWO_PI)) / (2.f * log(freq_base)));\n"
"        corr0 = max(0.f, start); corr1 = min((float)(n_rot - 1), end);\n"
"    }\n"
"    device float* tail = X + (ulong)h * (ulong)head_dim + n_nope;\n"
"    for (int i = (int)tiisg * 2; i < n_rot; i += 64) {\n"
"        float theta_extrap = (float)pos;\n"
"        for (int t = 0; t < i; t += 2) theta_extrap *= theta_scale;\n"
"        float theta_interp = freq_scale * theta_extrap;\n"
"        float theta = theta_interp, mscale = attn_factor;\n"
"        if (ext_factor != 0.f) {\n"
"            float y = ((float)(i / 2) - corr0) / max(0.001f, corr1 - corr0);\n"
"            float ramp = 1.f - min(1.f, max(0.f, y));\n"
"            float ramp_mix = ramp * ext_factor;\n"
"            theta = theta_interp * (1.f - ramp_mix) + theta_extrap * ramp_mix;\n"
"            mscale *= 1.f + 0.1f * log(1.f / freq_scale);\n"
"        }\n"
"        float c = cos(theta) * mscale, s = sin_sign * sin(theta) * mscale;\n"
"        float x0 = tail[i], x1 = tail[i + 1];\n"
"        tail[i] = x0 * c - x1 * s;\n"
"        tail[i + 1] = x0 * s + x1 * c;\n"
"    }\n"
"}\n"
/* Varianti con dimensione TOKEN (grid y = token) per il prefill a chunk: stessa matematica dei kernel per-token. */
"kernel void dsv4_rms_w_rows(device const float* X [[buffer(0)]], device const float* W [[buffer(1)]],\n"
"    device float* O [[buffer(2)]], constant uint* P [[buffer(3)]], constant float* CF [[buffer(4)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint n = P[0]; const float eps = CF[0]; const ulong o = (ulong)tg.y * n;\n"
"    float s = 0.f;\n"
"    for (uint i = tiisg; i < n; i += 32u) { float v = X[o + i]; s += v*v; }\n"
"    s = simd_sum(s);\n"
"    const float scale = 1.f / sqrt(s / (float)n + eps);\n"
"    for (uint i = tiisg; i < n; i += 32u) O[o + i] = X[o + i] * scale * W[i];\n"
"}\n"
"kernel void dsv4_head_rms_rows(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], constant float* CF [[buffer(2)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint hd = P[0], nh = P[1]; const float eps = CF[0];\n"
"    device float* x = X + ((ulong)tg.y * nh + tg.x) * (ulong)hd;\n"
"    float s = 0.f;\n"
"    for (uint i = tiisg; i < hd; i += 32u) { float v = x[i]; s += v*v; }\n"
"    s = simd_sum(s);\n"
"    const float scale = 1.f / sqrt(s / (float)hd + eps);\n"
"    for (uint i = tiisg; i < hd; i += 32u) x[i] *= scale;\n"
"}\n"
"kernel void dsv4_rope_rows(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], constant float* CF [[buffer(2)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const int n_rot = (int)P[0], head_dim = (int)P[1], pos = (int)(P[2] + tg.y); const uint nh = P[3];\n"
"    const float freq_base = CF[0], freq_scale = CF[1], ext_factor = CF[2], attn_factor = CF[3];\n"
"    const float beta_fast = CF[4], beta_slow = CF[5]; const float n_ctx_orig = CF[6];\n"
"    const int n_nope = head_dim - n_rot;\n"
"    const float theta_scale = pow(freq_base, -2.f / (float)n_rot);\n"
"    float corr0 = 0.f, corr1 = 0.f;\n"
"    const float TWO_PI = 6.28318530717958647692f;\n"
"    if (ext_factor != 0.f) {\n"
"        float start = floor((float)n_rot * log(n_ctx_orig / (beta_fast * TWO_PI)) / (2.f * log(freq_base)));\n"
"        float end = ceil((float)n_rot * log(n_ctx_orig / (beta_slow * TWO_PI)) / (2.f * log(freq_base)));\n"
"        corr0 = max(0.f, start); corr1 = min((float)(n_rot - 1), end);\n"
"    }\n"
"    device float* tail = X + ((ulong)tg.y * nh + tg.x) * (ulong)head_dim + n_nope;\n"
"    for (int i = (int)tiisg * 2; i < n_rot; i += 64) {\n"
"        float theta_extrap = (float)pos;\n"
"        for (int t = 0; t < i; t += 2) theta_extrap *= theta_scale;\n"
"        float theta_interp = freq_scale * theta_extrap;\n"
"        float theta = theta_interp, mscale = attn_factor;\n"
"        if (ext_factor != 0.f) {\n"
"            float y = ((float)(i / 2) - corr0) / max(0.001f, corr1 - corr0);\n"
"            float ramp = 1.f - min(1.f, max(0.f, y));\n"
"            float ramp_mix = ramp * ext_factor;\n"
"            theta = theta_interp * (1.f - ramp_mix) + theta_extrap * ramp_mix;\n"
"            mscale *= 1.f + 0.1f * log(1.f / freq_scale);\n"
"        }\n"
"        float c = cos(theta) * mscale, s = sin(theta) * mscale;\n"
"        float x0 = tail[i], x1 = tail[i + 1];\n"
"        tail[i] = x0 * c - x1 * s;\n"
"        tail[i + 1] = x0 * s + x1 * c;\n"
"    }\n"
"}\n"
"/* RoPE tail INVERSA (per gli heads dopo l'attenzione): identica a dsv4_rope_rows ma con sin negato\n"
" * (inverse = sin_sign -1, come llm_dsv4_rope_tail_ext_inplace). Stesso grid (nh, nt). */\n"
"kernel void dsv4_rope_rows_inv(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], constant float* CF [[buffer(2)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const int n_rot = (int)P[0], head_dim = (int)P[1], pos = (int)(P[2] + tg.y); const uint nh = P[3];\n"
"    const float freq_base = CF[0], freq_scale = CF[1], ext_factor = CF[2], attn_factor = CF[3];\n"
"    const float beta_fast = CF[4], beta_slow = CF[5]; const float n_ctx_orig = CF[6];\n"
"    const int n_nope = head_dim - n_rot;\n"
"    const float theta_scale = pow(freq_base, -2.f / (float)n_rot);\n"
"    float corr0 = 0.f, corr1 = 0.f;\n"
"    const float TWO_PI = 6.28318530717958647692f;\n"
"    if (ext_factor != 0.f) {\n"
"        float start = floor((float)n_rot * log(n_ctx_orig / (beta_fast * TWO_PI)) / (2.f * log(freq_base)));\n"
"        float end = ceil((float)n_rot * log(n_ctx_orig / (beta_slow * TWO_PI)) / (2.f * log(freq_base)));\n"
"        corr0 = max(0.f, start); corr1 = min((float)(n_rot - 1), end);\n"
"    }\n"
"    device float* tail = X + ((ulong)tg.y * nh + tg.x) * (ulong)head_dim + n_nope;\n"
"    for (int i = (int)tiisg * 2; i < n_rot; i += 64) {\n"
"        float theta_extrap = (float)pos;\n"
"        for (int t = 0; t < i; t += 2) theta_extrap *= theta_scale;\n"
"        float theta_interp = freq_scale * theta_extrap;\n"
"        float theta = theta_interp, mscale = attn_factor;\n"
"        if (ext_factor != 0.f) {\n"
"            float y = ((float)(i / 2) - corr0) / max(0.001f, corr1 - corr0);\n"
"            float ramp = 1.f - min(1.f, max(0.f, y));\n"
"            float ramp_mix = ramp * ext_factor;\n"
"            theta = theta_interp * (1.f - ramp_mix) + theta_extrap * ramp_mix;\n"
"            mscale *= 1.f + 0.1f * log(1.f / freq_scale);\n"
"        }\n"
"        float c = cos(theta) * mscale, s = -sin(theta) * mscale;\n"
"        float x0 = tail[i], x1 = tail[i + 1];\n"
"        tail[i] = x0 * c - x1 * s;\n"
"        tail[i + 1] = x0 * s + x1 * c;\n"
"    }\n"
"}\n"
/* ── mHC BATCHED (token = grid y): stessa matematica di llm_dsv4_block_pre/post, per il prefill a chunk ── */
/* scale[t] = 1/sqrt(mean(x^2)+eps) sull'intero stato [HC*D] del token. P={n}. */
"kernel void dsv4_hcb_scale(device const float* X [[buffer(0)]], device float* SC [[buffer(1)]], constant uint* P [[buffer(2)]], constant float* CF [[buffer(3)]],\n"
"    uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {\n"
"    const uint n = P[0]; device const float* x = X + (ulong)t * n; float s = 0.f;\n"
"    for (uint i = tid; i < n; i += 256u) { float v = x[i]; s += v * v; }\n"
"    s = simd_sum(s); threadgroup float sh[8]; if (lane == 0u) sh[sg] = s;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) { float a = 0.f; for (uint k = 0; k < 8u; k++) a += sh[k]; SC[t] = 1.f / sqrt(a / (float)n + CF[0]); }\n"
"}\n"
/* mix[t][r] = scale[t] * dot(fn[r], x[t]); grid (SIZE, N), 128 thread. P={HCD, SIZE}. */
"kernel void dsv4_hcb_mixes(device const float* X [[buffer(0)]], device const float* FN [[buffer(1)]], device const float* SC [[buffer(2)]],\n"
"    device float* MIX [[buffer(3)]], constant uint* P [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],\n"
"    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {\n"
"    const uint n = P[0], size = P[1], r = tg.x, t = tg.y;\n"
"    device const float4* w = (device const float4*)(FN + (ulong)r * n); device const float4* x = (device const float4*)(X + (ulong)t * n);\n"
"    float s = 0.f; for (uint i = tid; i < n / 4u; i += 128u) s += dot(w[i], x[i]);\n"
"    s = simd_sum(s); threadgroup float sh[4]; if (lane == 0u) sh[sg] = s;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) MIX[(ulong)t * size + r] = (sh[0] + sh[1] + sh[2] + sh[3]) * SC[t];\n"
"}\n"
/* sinkhorn per token (1 thread/token): copia di dsv4_hc_sinkhorn con offset di token. P={n_hc, iters, size}. */
"kernel void dsv4_hcb_sinkhorn(device const float* MIX [[buffer(0)]], device const float* SCALE [[buffer(1)]],\n"
"    device const float* BASE [[buffer(2)]], device float* OUT [[buffer(3)]],\n"
"    constant uint* P [[buffer(4)]], constant float* CF [[buffer(5)]], uint t [[thread_position_in_grid]]) {\n"
"    const uint n_hc = P[0], iters = P[1], size = P[2]; const float eps = CF[0];\n"
"    device const float* mix = MIX + (ulong)t * size; device float* out = OUT + (ulong)t * size;\n"
"    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];\n"
"    for (uint i = 0; i < n_hc; i++) { float z = mix[i]*pre_scale + BASE[i]; out[i] = 1.f/(1.f+exp(-z)) + eps; }\n"
"    for (uint i = 0; i < n_hc; i++) { uint off = n_hc+i; float z = mix[off]*post_scale + BASE[off]; out[off] = 2.f/(1.f+exp(-z)); }\n"
"    thread float c[256];\n"
"    for (uint dst = 0; dst < n_hc; dst++) {\n"
"        float rm = -INFINITY;\n"
"        for (uint src = 0; src < n_hc; src++) { uint idx=src+dst*n_hc; uint off=2*n_hc+idx; float v=mix[off]*comb_scale+BASE[off]; c[idx]=v; if(v>rm)rm=v; }\n"
"        float rs=0.f; for(uint src=0;src<n_hc;src++){uint idx=src+dst*n_hc;float v=exp(c[idx]-rm);c[idx]=v;rs+=v;}\n"
"        float iv=1.f/rs; for(uint src=0;src<n_hc;src++){uint idx=src+dst*n_hc;c[idx]=c[idx]*iv+eps;}\n"
"    }\n"
"    for (uint src=0;src<n_hc;src++){float sum=0.f;for(uint dst=0;dst<n_hc;dst++)sum+=c[src+dst*n_hc];float iv=1.f/(sum+eps);for(uint dst=0;dst<n_hc;dst++)c[src+dst*n_hc]*=iv;}\n"
"    for (uint it=1;it<iters;it++){\n"
"        for(uint dst=0;dst<n_hc;dst++){float sum=0.f;for(uint src=0;src<n_hc;src++)sum+=c[src+dst*n_hc];float iv=1.f/(sum+eps);for(uint src=0;src<n_hc;src++)c[src+dst*n_hc]*=iv;}\n"
"        for(uint src=0;src<n_hc;src++){float sum=0.f;for(uint dst=0;dst<n_hc;dst++)sum+=c[src+dst*n_hc];float iv=1.f/(sum+eps);for(uint dst=0;dst<n_hc;dst++)c[src+dst*n_hc]*=iv;}\n"
"    }\n"
"    for (uint i=0;i<n_hc*n_hc;i++) out[2*n_hc+i]=c[i];\n"
"}\n"
/* normed[t][d] = rms_weight( sum_h res[t][h][d]*pre[t][h] ) ; TG per token, 256 thread. P={D, HC, size}. */
"kernel void dsv4_hcb_norm(device const float* RES [[buffer(0)]], device const float* SPL [[buffer(1)]], device const float* W [[buffer(2)]],\n"
"    device float* NORMED [[buffer(3)]], constant uint* P [[buffer(4)]], constant float* CF [[buffer(5)]],\n"
"    uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {\n"
"    const uint D = P[0], HC = P[1], size = P[2];\n"
"    threadgroup float hin[4096]; threadgroup float sh[8];\n"
"    device const float* pre = SPL + (ulong)t * size; device const float* res = RES + (ulong)t * HC * D;\n"
"    float s = 0.f;\n"
"    for (uint d = tid; d < D; d += 256u) { float a = 0.f; for (uint h = 0; h < HC; h++) a += res[(ulong)h * D + d] * pre[h]; hin[d] = a; s += a * a; }\n"
"    s = simd_sum(s); if (lane == 0u) sh[sg] = s;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float tot = 0.f; for (uint k = 0; k < 8u; k++) tot += sh[k];\n"
"    const float scale = 1.f / sqrt(tot / (float)D + CF[0]);\n"
"    for (uint d = tid; d < D; d += 256u) NORMED[(ulong)t * D + d] = hin[d] * scale * W[d];\n"
"}\n"
/* out[t][dst][d] = bo[t][d]*post[t][dst] + sum_src comb[t][dst+src*HC]*res[t][src][d]; grid (ceil(HC*D/256), N). P={D, HC, size}. */
"kernel void dsv4_hcb_post(device const float* BO [[buffer(0)]], device const float* RES [[buffer(1)]], device const float* SPL [[buffer(2)]],\n"
"    device float* OUT [[buffer(3)]], constant uint* P [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {\n"
"    const uint D = P[0], HC = P[1], size = P[2]; const uint t = g.y, i = g.x; if (i >= HC * D) return;\n"
"    const uint dst = i / D, d = i % D; device const float* spl = SPL + (ulong)t * size; device const float* res = RES + (ulong)t * HC * D;\n"
"    float acc = BO[(ulong)t * D + d] * spl[HC + dst];\n"
"    for (uint src = 0; src < HC; src++) acc += spl[2*HC + dst + src*HC] * res[(ulong)src * D + d];\n"
"    OUT[(ulong)t * HC * D + i] = acc;\n"
"}\n"
/* GEMV f32 MULTI-TOKEN: O[t][row] = dot(W[row], X[t]) per t in [0,nt); W letto una volta per tile di 16 token. Stesso ordine di accumulo
 * per token di dsv4_gemv_f32 (lane i, i+32, ... poi simd_sum) -> risultato identico bit a bit. P={n, nt, out_dim}. Grid (out_dim, ceil(nt/16)). */
"kernel void dsv4_gemv_f32_nt(device const float* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* O [[buffer(2)]],\n"
"    constant uint* P [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {\n"
"    const uint n = P[0], nt = P[1], od = P[2]; const uint row0 = tg.x * 4u, t0 = tg.y * 16u; if (row0 >= od) return;\n"
"    const uint nr = min(4u, od - row0);\n"
"    device const float* wr0 = W + (ulong)row0 * n;\n"
"    float acc[4][16]; for (uint r = 0; r < 4u; r++) for (uint k = 0; k < 16u; k++) acc[r][k] = 0.f;\n"
"    for (uint i = tid; i < n; i += 32u) {\n"
"        float xv[16]; for (uint k = 0; k < 16u; k++) xv[k] = (t0 + k < nt) ? X[(ulong)(t0 + k) * n + i] : 0.f;\n"
"        for (uint r = 0; r < 4u; r++) { if (r < nr) { const float w = wr0[(ulong)r * n + i]; for (uint k = 0; k < 16u; k++) acc[r][k] += w * xv[k]; } }\n"
"    }\n"
"    for (uint r = 0; r < 4u; r++) { if (r < nr) { for (uint k = 0; k < 16u; k++) { if (t0 + k < nt) { const float sm = simd_sum(acc[r][k]); if (tid == 0u) O[(ulong)(t0 + k) * od + row0 + r] = sm; } } } }\n"
"}\n"
/* Riga di stato del compressor: STKV[i] = PKV[i]; STSC[i] = PSC[i] + APE[i] (gli offset di buffer scelgono token/riga). P={width}. */
"kernel void dsv4_state_store(device const float* PKV [[buffer(0)]], device const float* PSC [[buffer(1)]], device const float* APE [[buffer(2)]],\n"
"    device float* STKV [[buffer(3)]], device float* STSC [[buffer(4)]], constant uint* P [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= P[0]) return; STKV[gid] = PKV[gid]; STSC[gid] = PSC[gid] + APE[gid];\n"
"}\n"
/* Routing BATCHED: probs = sqrt(softplus(logit)); selezione top-k (stesso ordine/spareggi di llm_dsv4_topk_desc: a pari score vince l'indice minore)
 * su probs+bias, oppure esperti da tabella hash; pesi = probs/sum*scale. Un thread per token. P={n_expert, used, has_bias, is_hash, nt}. */
"kernel void dsv4_route_batch(device const float* LG [[buffer(0)]], device const float* BIAS [[buffer(1)]], device const int* HSEL [[buffer(2)]],\n"
"    device int* SEL [[buffer(3)]], device float* WT [[buffer(4)]], constant uint* P [[buffer(5)]], constant float* CF [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
"    const uint n = P[0], used = P[1], has_bias = P[2], is_hash = P[3]; if (t >= P[4]) return;\n"
"    float probs[256]; int idx[16];\n"
"    for (uint i = 0; i < n; i++) { const float x = LG[(ulong)t * n + i]; const float sp = (x > 20.f) ? x : ((x < -20.f) ? exp(x) : log(1.f + exp(x))); probs[i] = sqrt(sp); }\n"
"    for (uint j = 0; j < used; j++) idx[j] = -1;\n"
"    if (is_hash) { for (uint j = 0; j < used; j++) idx[j] = HSEL[(ulong)t * used + j]; }\n"
"    else { for (uint i = 0; i < n; i++) { const float si = probs[i] + (has_bias ? BIAS[i] : 0.f);\n"
"        for (uint j = 0; j < used; j++) { const bool take = idx[j] < 0 || si > probs[idx[j]] + (has_bias ? BIAS[idx[j]] : 0.f);\n"
"            if (take) { for (int m = (int)used - 1; m > (int)j; m--) idx[m] = idx[m - 1]; idx[j] = (int)i; break; } } } }\n"
"    float sum = 0.f; for (uint j = 0; j < used; j++) sum += probs[idx[j]];\n"
"    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;\n"
"    for (uint j = 0; j < used; j++) { SEL[(ulong)t * used + j] = idx[j]; WT[(ulong)t * used + j] = probs[idx[j]] / sum * CF[0]; }\n"
"}\n"
"kernel void dsv4_swiglu(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],\n"
"    device float* MID [[buffer(2)]], constant uint* P [[buffer(3)]], constant float* CF [[buffer(4)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0]; if (gid >= n) return;\n"
"    const float c = CF[0]; float g = G[gid], u = U[gid];\n"
"    if (c > 1.0e-6f) { g = min(g, c); u = clamp(u, -c, c); }\n"
"    const float silu = g / (1.f + exp(-g)); MID[gid] = silu * u;\n"
"}\n"
/* MoE batched: MID[r][j] = silu(min(G,c)) * clamp(U,-c,c) * RW[r] (r = token*6+slot). P={ff, rows}. */
"kernel void dsv4_swiglu_w(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],\n"
"    device const float* RW [[buffer(2)]], device float* MID [[buffer(3)]], constant uint* P [[buffer(4)]],\n"
"    constant float* CF [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    const uint ff = P[0], rows = P[1]; if (gid.x >= ff || gid.y >= rows) return;\n"
"    const ulong k = (ulong)gid.y * ff + gid.x; const float c = CF[0]; float g = G[k], u = U[k];\n"
"    if (c > 1.0e-6f) { g = min(g, c); u = clamp(u, -c, c); }\n"
"    MID[k] = g / (1.f + exp(-g)) * u * RW[gid.y];\n"
"}\n"
/* out[t][i] = somma sui 6 slot di D[t*6+s][i]. P={embd, nt}. */
"kernel void dsv4_sum_slots6(device const float* D [[buffer(0)]], device float* OUT [[buffer(1)]],\n"
"    constant uint* P [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    const uint embd = P[0], nt = P[1]; if (gid.x >= embd || gid.y >= nt) return;\n"
"    float s = 0.f; for (uint k = 0; k < 6u; k++) s += D[((ulong)gid.y * 6u + k) * embd + gid.x];\n"
"    OUT[(ulong)gid.y * embd + gid.x] = s;\n"
"}\n"
"kernel void dsv4_add_inplace(device float* A [[buffer(0)]], device const float* B [[buffer(1)]],\n"
"    constant uint* P [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0]; if (gid >= n) return; A[gid] += B[gid];\n"
"}\n"
"kernel void dsv4_rms_no_w(device const float* X [[buffer(0)]], device float* O [[buffer(1)]],\n"
"    constant uint* P [[buffer(2)]], constant float* CF [[buffer(3)]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint n = P[0]; const float eps = CF[0];\n"
"    float s = 0.f; for (uint i = tiisg; i < n; i += 32u) { float v = X[i]; s += v*v; }\n"
"    s = simd_sum(s); const float sc = 1.f / sqrt(s / (float)n + eps);\n"
"    for (uint i = tiisg; i < n; i += 32u) O[i] = X[i] * sc;\n"
"}\n"
"kernel void dsv4_gemv_f32(device const float* W [[buffer(0)]], device const float* X [[buffer(1)]],\n"
"    device float* O [[buffer(2)]], constant uint* P [[buffer(3)]],\n"
"    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {\n"
"    const uint n = P[0]; const uint lead = P[1]; device const float* wr = W + lead + (ulong)row * n;\n"
"    float acc = 0.f; for (uint i = tid; i < n; i += 32u) acc += wr[i] * X[i];\n"
"    acc = simd_sum(acc); if (tid == 0u) O[row] = acc;\n"
"}\n"
"kernel void dsv4_hc_sinkhorn(device const float* MIX [[buffer(0)]], device const float* SCALE [[buffer(1)]],\n"
"    device const float* BASE [[buffer(2)]], device float* OUT [[buffer(3)]],\n"
"    constant uint* P [[buffer(4)]], constant float* CF [[buffer(5)]]) {\n"
"    const uint n_hc = P[0], iters = P[1]; const float eps = CF[0];\n"
"    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];\n"
"    for (uint i = 0; i < n_hc; i++) { float z = MIX[i]*pre_scale + BASE[i]; OUT[i] = 1.f/(1.f+exp(-z)) + eps; }\n"
"    for (uint i = 0; i < n_hc; i++) { uint off = n_hc+i; float z = MIX[off]*post_scale + BASE[off]; OUT[off] = 2.f/(1.f+exp(-z)); }\n"
"    thread float c[256];\n"
"    for (uint dst = 0; dst < n_hc; dst++) {\n"
"        float rm = -INFINITY;\n"
"        for (uint src = 0; src < n_hc; src++) { uint idx=src+dst*n_hc; uint off=2*n_hc+idx; float v=MIX[off]*comb_scale+BASE[off]; c[idx]=v; if(v>rm)rm=v; }\n"
"        float rs=0.f; for(uint src=0;src<n_hc;src++){uint idx=src+dst*n_hc;float v=exp(c[idx]-rm);c[idx]=v;rs+=v;}\n"
"        float iv=1.f/rs; for(uint src=0;src<n_hc;src++){uint idx=src+dst*n_hc;c[idx]=c[idx]*iv+eps;}\n"
"    }\n"
"    for (uint src=0;src<n_hc;src++){float sum=0.f;for(uint dst=0;dst<n_hc;dst++)sum+=c[src+dst*n_hc];float iv=1.f/(sum+eps);for(uint dst=0;dst<n_hc;dst++)c[src+dst*n_hc]*=iv;}\n"
"    for (uint it=1;it<iters;it++){\n"
"        for(uint dst=0;dst<n_hc;dst++){float sum=0.f;for(uint src=0;src<n_hc;src++)sum+=c[src+dst*n_hc];float iv=1.f/(sum+eps);for(uint src=0;src<n_hc;src++)c[src+dst*n_hc]*=iv;}\n"
"        for(uint src=0;src<n_hc;src++){float sum=0.f;for(uint dst=0;dst<n_hc;dst++)sum+=c[src+dst*n_hc];float iv=1.f/(sum+eps);for(uint dst=0;dst<n_hc;dst++)c[src+dst*n_hc]*=iv;}\n"
"    }\n"
"    for (uint i=0;i<n_hc*n_hc;i++) OUT[2*n_hc+i]=c[i];\n"
"}\n"
"kernel void dsv4_hc_sum(device const float* RES [[buffer(0)]], device const float* PRE [[buffer(1)]],\n"
"    device float* HIN [[buffer(2)]], constant uint* P [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint D = P[0], HC = P[1]; if (gid >= D) return;\n"
"    float a = 0.f; for (uint h = 0; h < HC; h++) a += RES[(ulong)h*D + gid] * PRE[h];\n"
"    HIN[gid] = a;\n"
"}\n"
"kernel void dsv4_hc_post(device const float* BO [[buffer(0)]], device const float* RES [[buffer(1)]],\n"
"    device const float* POST [[buffer(2)]], device const float* COMB [[buffer(3)]], device float* OUT [[buffer(4)]],\n"
"    constant uint* P [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint D = P[0], HC = P[1]; if (gid >= HC*D) return;\n"
"    const uint dst = gid / D, d = gid % D;\n"
"    float acc = BO[d] * POST[dst];\n"
"    for (uint src = 0; src < HC; src++) acc += COMB[dst + src*HC] * RES[(ulong)src*D + d];\n"
"    OUT[gid] = acc;\n"
"}\n"
"kernel void dsv4_route(device const float* LOGITS [[buffer(0)]], device const float* BIAS [[buffer(1)]],\n"
"    device int* SELECTED [[buffer(2)]], device float* WEIGHT [[buffer(3)]],\n"
"    constant uint* P [[buffer(4)]], constant float* CF [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n_expert=P[0], used=P[1], has_bias=P[2];\n"
"    if (gid != 0u) return;\n"
"    const float scale = CF[0];\n"
"    float probs[256]; float sel[256];\n"
"    for (uint i=0;i<n_expert;i++){\n"
"        float x=LOGITS[i];\n"
"        float sp=(x>20.f)?x:((x<-20.f)?exp(x):log(1.f+exp(x)));\n"
"        float p=sqrt(sp); probs[i]=p; sel[i]=p+(has_bias?BIAS[i]:0.f);\n"
"    }\n"
"    for (uint k=0;k<used;k++){\n"
"        float bs=-1.0e30f; int besti=0; int any=0;\n"
"        for (uint i=0;i<n_expert;i++){\n"
"            int already=0; for(uint j=0;j<k;j++) if (SELECTED[j]==(int)i){already=1;break;}\n"
"            if (!already && (!any || sel[i]>bs)){ bs=sel[i]; besti=(int)i; any=1; }\n"
"        }\n"
"        SELECTED[k]=besti;\n"
"    }\n"
"    float sum=0.f; for(uint i=0;i<used;i++) sum+=probs[SELECTED[i]];\n"
"    if (sum<6.103515625e-5f) sum=6.103515625e-5f;\n"
"    for(uint i=0;i<used;i++) WEIGHT[i]=probs[SELECTED[i]]/sum*scale;\n"
"}\n"
"kernel void dsv4_slots6_addr_pair_swiglu_iq2(\n"
"    device const int* SEL [[buffer(0)]], device const ulong* GADDR [[buffer(1)]], device const ulong* UADDR [[buffer(2)]],\n"
"    device const float* Y [[buffer(3)]], device float* MID [[buffer(4)]], device const float* RW [[buffer(5)]],\n"
"    constant float* CF [[buffer(6)]], constant uint* P [[buffer(7)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]], threadgroup char* shmem [[threadgroup(0)]]) {\n"
"    const int idx = (int)tgpig.z; const int e = SEL[idx];\n"
"    device const char* SG = (device const char*)(ulong)GADDR[e];\n"
"    device const char* SU = (device const char*)(ulong)UADDR[e];\n"
"    const int ne00=(int)P[0], nb01=(int)P[1], ne0=(int)P[2];\n"
"    dsv4_pair_core(SG, SU, Y, MID, RW[idx], idx*ne0, CF[0], ne00, nb01, ne0, 0u, 0u, tgpig.x, tiisg, sgitg, shmem);\n"
"}\n"
"kernel void dsv4_sum6_addr_q2k(device const int* SEL [[buffer(0)]], device const ulong* DADDR [[buffer(1)]],\n"
"    device const float* MID [[buffer(2)]], device float* OUT [[buffer(3)]], constant uint* P [[buffer(4)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const short NSG = 4; const short nr0 = 4;\n"
"    const int nb = (int)P[0]; const uint nb01 = P[1];\n"
"    const int ne0 = (int)P[2]; const int ne00 = (int)P[3];\n"
"    const int first_row = (tgpig.x * NSG + sgitg) * nr0;\n"
"    float sumf[4] = {0.f,0.f,0.f,0.f};\n"
"    const short ix = tiisg / 8; const short it = tiisg % 8;\n"
"    const short iq = it / 4; const short ir = it % 4; const short is = (8 * ir) / 16;\n"
"    for (int slot = 0; slot < 6; slot++) {\n"
"        device const char* s0 = (device const char*)(ulong)DADDR[SEL[slot]];\n"
"        device const float* y = MID + (ulong)slot * (ulong)ne00;\n"
"        device const float* y4 = y + ix * 256 + 128 * iq + 8 * ir;\n"
"        for (int ib = ix; ib < nb; ib += 4) {\n"
"            float yl[32]; float4 sumy = {0.f,0.f,0.f,0.f};\n"
"            for (short i = 0; i < 8; ++i) {\n"
"                yl[i+0]=y4[i+0]; sumy[0]+=yl[i+0]; yl[i+8]=y4[i+32]; sumy[1]+=yl[i+8];\n"
"                yl[i+16]=y4[i+64]; sumy[2]+=yl[i+16]; yl[i+24]=y4[i+96]; sumy[3]+=yl[i+24]; }\n"
"            for (short row = 0; row < nr0; row++) {\n"
"                if (first_row + row < ne0) {\n"
"                    device const uchar* blk = (device const uchar*)s0 + (ulong)(first_row+row) * (ulong)nb01 + (ulong)ib * 84u;\n"
"                    device const uint8_t* scr = blk + 8*iq + is;\n"
"                    device const uint16_t* qsr = (device const uint16_t*)(blk + 16u) + 16*iq + 4*ir;\n"
"                    ushort dh0 = (ushort)blk[80] | ((ushort)blk[81] << 8);\n"
"                    ushort dh1 = (ushort)blk[82] | ((ushort)blk[83] << 8);\n"
"                    const float dall = (float)as_type<half>(dh0);\n"
"                    const float dmin = (float)as_type<half>(dh1) * (1.f/16.f);\n"
"                    float4 acc1 = {0.f,0.f,0.f,0.f}; float4 acc2 = {0.f,0.f,0.f,0.f};\n"
"                    for (int i = 0; i < 8; i += 2) {\n"
"                        acc1[0] += yl[i+0]*(qsr[i/2]&0x0003); acc2[0] += yl[i+1]*(qsr[i/2]&0x0300);\n"
"                        acc1[1] += yl[i+8]*(qsr[i/2]&0x000c); acc2[1] += yl[i+9]*(qsr[i/2]&0x0c00);\n"
"                        acc1[2] += yl[i+16]*(qsr[i/2]&0x0030); acc2[2] += yl[i+17]*(qsr[i/2]&0x3000);\n"
"                        acc1[3] += yl[i+24]*(qsr[i/2]&0x00c0); acc2[3] += yl[i+25]*(qsr[i/2]&0xc000);\n"
"                    }\n"
"                    sumf[row] += dall * ((acc1[0]+1.f/256.f*acc2[0])*(scr[0]&0xF)\n"
"                                       + (acc1[1]+1.f/256.f*acc2[1])*(scr[2]&0xF)*0.25f\n"
"                                       + (acc1[2]+1.f/256.f*acc2[2])*(scr[4]&0xF)*0.0625f\n"
"                                       + (acc1[3]+1.f/256.f*acc2[3])*(scr[6]&0xF)*0.015625f)\n"
"                               - dmin * (sumy[0]*(scr[0]&0xF0) + sumy[1]*(scr[2]&0xF0)\n"
"                                       + sumy[2]*(scr[4]&0xF0) + sumy[3]*(scr[6]&0xF0));\n"
"                }\n"
"            }\n"
"            y4 += 4 * 256;\n"
"        }\n"
"    }\n"
"    for (int row = 0; row < nr0 && first_row + row < ne0; row++) {\n"
"        const float s = simd_sum(sumf[row]);\n"
"        if (tiisg == 0) OUT[first_row + row] = s;\n"
"    }\n"
"}\n"
"typedef struct { device const char* g[256]; device const char* u[256]; } D4PT;\n"
"kernel void dsv4_slots6_tbl_pair(\n"
"    device const int* SEL [[buffer(0)]], device const D4PT& T [[buffer(1)]],\n"
"    device const float* Y [[buffer(2)]], device float* MID [[buffer(3)]], device const float* RW [[buffer(4)]],\n"
"    constant float* CF [[buffer(5)]], constant uint* P [[buffer(6)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]], threadgroup char* shmem [[threadgroup(0)]]) {\n"
"    const int idx = (int)tgpig.z; const int e = SEL[idx];\n"
"    device const char* SG = T.g[e]; device const char* SU = T.u[e];\n"
"    const int ne00=(int)P[0], nb01=(int)P[1], ne0=(int)P[2];\n"
"    dsv4_pair_core(SG, SU, Y, MID, RW[idx], idx*ne0, CF[0], ne00, nb01, ne0, 0u, 0u, tgpig.x, tiisg, sgitg, shmem);\n"
"}\n"
"typedef struct { device const char* g[256]; } D4DT;\n"
"kernel void dsv4_sum6_tbl(device const int* SEL [[buffer(0)]], device const D4DT& T [[buffer(1)]],\n"
"    device const float* MID [[buffer(2)]], device float* OUT [[buffer(3)]], constant uint* P [[buffer(4)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const short NSG = 4; const short nr0 = 4;\n"
"    const int nb = (int)P[0]; const uint nb01 = P[1];\n"
"    const int ne0 = (int)P[2]; const int ne00 = (int)P[3];\n"
"    const int first_row = (tgpig.x * NSG + sgitg) * nr0;\n"
"    float sumf[4] = {0.f,0.f,0.f,0.f};\n"
"    const short ix = tiisg / 8; const short it = tiisg % 8;\n"
"    const short iq = it / 4; const short ir = it % 4; const short is = (8 * ir) / 16;\n"
"    for (int slot = 0; slot < 6; slot++) {\n"
"        device const char* s0 = T.g[SEL[slot]];\n"
"        device const float* y = MID + (ulong)slot * (ulong)ne00;\n"
"        device const float* y4 = y + ix * 256 + 128 * iq + 8 * ir;\n"
"        for (int ib = ix; ib < nb; ib += 4) {\n"
"            float yl[32]; float4 sumy = {0.f,0.f,0.f,0.f};\n"
"            for (short i = 0; i < 8; ++i) {\n"
"                yl[i+0]=y4[i+0]; sumy[0]+=yl[i+0]; yl[i+8]=y4[i+32]; sumy[1]+=yl[i+8];\n"
"                yl[i+16]=y4[i+64]; sumy[2]+=yl[i+16]; yl[i+24]=y4[i+96]; sumy[3]+=yl[i+24]; }\n"
"            for (short row = 0; row < nr0; row++) {\n"
"                if (first_row + row < ne0) {\n"
"                    device const uchar* blk = (device const uchar*)s0 + (ulong)(first_row+row) * (ulong)nb01 + (ulong)ib * 84u;\n"
"                    device const uint8_t* scr = blk + 8*iq + is;\n"
"                    device const uint16_t* qsr = (device const uint16_t*)(blk + 16u) + 16*iq + 4*ir;\n"
"                    ushort dh0 = (ushort)blk[80] | ((ushort)blk[81] << 8);\n"
"                    ushort dh1 = (ushort)blk[82] | ((ushort)blk[83] << 8);\n"
"                    const float dall = (float)as_type<half>(dh0);\n"
"                    const float dmin = (float)as_type<half>(dh1) * (1.f/16.f);\n"
"                    float4 acc1 = {0.f,0.f,0.f,0.f}; float4 acc2 = {0.f,0.f,0.f,0.f};\n"
"                    for (int i = 0; i < 8; i += 2) {\n"
"                        acc1[0] += yl[i+0]*(qsr[i/2]&0x0003); acc2[0] += yl[i+1]*(qsr[i/2]&0x0300);\n"
"                        acc1[1] += yl[i+8]*(qsr[i/2]&0x000c); acc2[1] += yl[i+9]*(qsr[i/2]&0x0c00);\n"
"                        acc1[2] += yl[i+16]*(qsr[i/2]&0x0030); acc2[2] += yl[i+17]*(qsr[i/2]&0x3000);\n"
"                        acc1[3] += yl[i+24]*(qsr[i/2]&0x00c0); acc2[3] += yl[i+25]*(qsr[i/2]&0xc000);\n"
"                    }\n"
"                    sumf[row] += dall * ((acc1[0]+1.f/256.f*acc2[0])*(scr[0]&0xF)\n"
"                                       + (acc1[1]+1.f/256.f*acc2[1])*(scr[2]&0xF)*0.25f\n"
"                                       + (acc1[2]+1.f/256.f*acc2[2])*(scr[4]&0xF)*0.0625f\n"
"                                       + (acc1[3]+1.f/256.f*acc2[3])*(scr[6]&0xF)*0.015625f)\n"
"                               - dmin * (sumy[0]*(scr[0]&0xF0) + sumy[1]*(scr[2]&0xF0)\n"
"                                       + sumy[2]*(scr[4]&0xF0) + sumy[3]*(scr[6]&0xF0));\n"
"                }\n"
"            }\n"
"            y4 += 4 * 256;\n"
"        }\n"
"    }\n"
"    for (int row = 0; row < nr0 && first_row + row < ne0; row++) {\n"
"        const float s = simd_sum(sumf[row]);\n"
"        if (tiisg == 0) OUT[first_row + row] = s;\n"
"    }\n"
"}\n"
/* Attn mista su KV device: fedele a llm_dsv4_attention_mixed (raw poi comp, sink, allowed).
 * Un threadgroup per testa; dot cooperativo (simd_sum), due pass (max, poi somma pesata).
 * P = {n_raw, n_comp, hd, has_allowed}. */
"kernel void dsv4_attn_mixed(device const float* Q [[buffer(0)]], device const float* RAWK [[buffer(1)]],\n"
"    device const float* COMPK [[buffer(2)]], device const int* ALLOWED [[buffer(3)]],\n"
"    device const float* SINKS [[buffer(4)]], device float* OUT [[buffer(5)]],\n"
"    constant uint* P [[buffer(6)]],\n"
"    uint h [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint n_raw = P[0], n_comp = P[1], hd = P[2]; const uint has_allowed = P[3];\n"
"    const float kq = 1.f / sqrt((float)hd);\n"
"    device const float* qh = Q + (ulong)h * (ulong)hd;\n"
"    device float* oh = OUT + (ulong)h * (ulong)hd;\n"
"    float mx = SINKS[h];\n"
"    for (uint r = 0; r < n_raw; r++) {\n"
"        device const float* kv = RAWK + (ulong)r * (ulong)hd;\n"
"        float acc = 0.f; for (uint i = tiisg; i < hd; i += 32u) acc += qh[i] * kv[i];\n"
"        acc = simd_sum(acc); float s = acc * kq; if (s > mx) mx = s;\n"
"    }\n"
"    for (uint r = 0; r < n_comp; r++) {\n"
"        if (has_allowed && !ALLOWED[r]) continue;\n"
"        device const float* kv = COMPK + (ulong)r * (ulong)hd;\n"
"        float acc = 0.f; for (uint i = tiisg; i < hd; i += 32u) acc += qh[i] * kv[i];\n"
"        acc = simd_sum(acc); float s = acc * kq; if (s > mx) mx = s;\n"
"    }\n"
"    float den = exp(SINKS[h] - mx);\n"
"    float accE[32]; for (uint k = 0; k < 32u; k++) accE[k] = 0.f;\n"
"    for (uint r = 0; r < n_raw; r++) {\n"
"        device const float* kv = RAWK + (ulong)r * (ulong)hd;\n"
"        float acc = 0.f; for (uint i = tiisg; i < hd; i += 32u) acc += qh[i] * kv[i];\n"
"        acc = simd_sum(acc); float w = exp(acc * kq - mx); den += w;\n"
"        uint k = 0; for (uint i = tiisg; i < hd; i += 32u) { accE[k] += w * kv[i]; k++; }\n"
"    }\n"
"    for (uint r = 0; r < n_comp; r++) {\n"
"        if (has_allowed && !ALLOWED[r]) continue;\n"
"        device const float* kv = COMPK + (ulong)r * (ulong)hd;\n"
"        float acc = 0.f; for (uint i = tiisg; i < hd; i += 32u) acc += qh[i] * kv[i];\n"
"        acc = simd_sum(acc); float w = exp(acc * kq - mx); den += w;\n"
"        uint k = 0; for (uint i = tiisg; i < hd; i += 32u) { accE[k] += w * kv[i]; k++; }\n"
"    }\n"
"    float inv = 1.f / den;\n"
"    { uint k = 0; for (uint i = tiisg; i < hd; i += 32u) { oh[i] = accE[k] * inv; k++; } }\n"
"}\n"
/* Attn mista SPLIT (flash-decoding): grid (n_head, ns), 4 simdgroup per TG. Le posizioni t in
 * [0, n_raw+n_comp) sono ripartite fra gli `ns` split e fra i 4 simdgroup; softmax online (m, l, acc);
 * ogni TG scrive il parziale {acc[hd], m, l} in PART. hd multiplo di 128, <= 512.
 * P = {n_raw, n_comp, hd, has_allowed, ns}. */
"kernel void dsv4_attn_part(device const float* Q [[buffer(0)]], device const float* RAWK [[buffer(1)]],\n"
"    device const float* COMPK [[buffer(2)]], device const int* ALLOWED [[buffer(3)]],\n"
"    device float* PART [[buffer(4)]], constant uint* P [[buffer(5)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint n_raw = P[0], n_comp = P[1], hd = P[2], has_allowed = P[3], ns = P[4];\n"
"    const uint h = tg.x, s = tg.y, nch = hd / 128u;\n"
"    const uint total = n_raw + n_comp, per = (total + ns - 1u) / ns;\n"
"    const uint t0 = s * per, t1 = min(total, t0 + per);\n"
"    const float kq = 1.f / sqrt((float)hd);\n"
"    device const float4* q4 = (device const float4*)(Q + (ulong)h * (ulong)hd);\n"
"    float4 qv[4], acc[4];\n"
"    for (uint j = 0; j < 4u; j++) { qv[j] = j < nch ? q4[j * 32u + tiisg] : float4(0.f); acc[j] = float4(0.f); }\n"
"    float m = -INFINITY, l = 0.f;\n"
"    for (uint t = t0 + sg; t < t1; t += 4u) {\n"
"        device const float* kvp;\n"
"        if (t < n_raw) kvp = RAWK + (ulong)t * (ulong)hd;\n"
"        else { const uint r = t - n_raw; if (has_allowed && !ALLOWED[r]) continue; kvp = COMPK + (ulong)r * (ulong)hd; }\n"
"        device const float4* k4 = (device const float4*)kvp;\n"
"        float4 kv[4]; float d = 0.f;\n"
"        for (uint j = 0; j < 4u; j++) { if (j < nch) { kv[j] = k4[j * 32u + tiisg]; d += dot(qv[j], kv[j]); } }\n"
"        d = simd_sum(d) * kq;\n"
"        const float mn = max(m, d), corr = exp(m - mn), w = exp(d - mn);\n"
"        l = l * corr + w;\n"
"        for (uint j = 0; j < 4u; j++) if (j < nch) acc[j] = acc[j] * corr + w * kv[j];\n"
"        m = mn;\n"
"    }\n"
"    threadgroup float sm[4], sl[4]; threadgroup float sacc[4 * 512];\n"
"    if (tiisg == 0u) { sm[sg] = m; sl[sg] = l; }\n"
"    for (uint j = 0; j < 4u; j++) if (j < nch) *(threadgroup float4*)(sacc + sg * 512u + j * 128u + tiisg * 4u) = acc[j];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const float M = max(max(sm[0], sm[1]), max(sm[2], sm[3]));\n"
"    device float* out = PART + ((ulong)h * (ulong)ns + (ulong)s) * (ulong)(hd + 2u);\n"
"    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;\n"
"    if (M > -INFINITY) { e0 = exp(sm[0] - M); e1 = exp(sm[1] - M); e2 = exp(sm[2] - M); e3 = exp(sm[3] - M); }\n"
"    for (uint i = tid; i < hd; i += 128u)\n"
"        out[i] = e0 * sacc[i] + e1 * sacc[512u + i] + e2 * sacc[1024u + i] + e3 * sacc[1536u + i];\n"
"    if (tid == 0u) { out[hd] = M; out[hd + 1u] = e0 * sl[0] + e1 * sl[1] + e2 * sl[2] + e3 * sl[3]; }\n"
"}\n"
/* Merge degli split + sink: out = sum_s e^{m_s-M} acc_s / (e^{sink-M} + sum_s e^{m_s-M} l_s). P = {hd, ns}. */
"kernel void dsv4_attn_merge(device const float* PART [[buffer(0)]], device const float* SINKS [[buffer(1)]],\n"
"    device float* OUT [[buffer(2)]], constant uint* P [[buffer(3)]],\n"
"    uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {\n"
"    const uint hd = P[0], ns = P[1]; const uint stride = hd + 2u;\n"
"    device const float* base = PART + (ulong)h * (ulong)ns * (ulong)stride;\n"
"    const float sink = SINKS[h];\n"
"    float M = sink;\n"
"    for (uint s = 0; s < ns; s++) M = max(M, base[(ulong)s * stride + hd]);\n"
"    float den = exp(sink - M);\n"
"    for (uint s = 0; s < ns; s++) { const float ms = base[(ulong)s * stride + hd]; if (ms > -INFINITY) den += exp(ms - M) * base[(ulong)s * stride + hd + 1u]; }\n"
"    const float inv = 1.f / den;\n"
"    for (uint i = tid; i < hd; i += 128u) {\n"
"        float a = 0.f;\n"
"        for (uint s = 0; s < ns; s++) { const float ms = base[(ulong)s * stride + hd]; if (ms > -INFINITY) a += exp(ms - M) * base[(ulong)s * stride + i]; }\n"
"        OUT[(ulong)h * (ulong)hd + i] = a * inv;\n"
"    }\n"
"}\n"
/* Compressor pool (port di llm_dsv4_compressor_pool_state). P = {hd, ratio, co}. */
"kernel void dsv4_comp_pool(device const float* SKV [[buffer(0)]], device const float* SSC [[buffer(1)]],\n"
"    device float* OUT [[buffer(2)]], constant uint* P [[buffer(3)]], uint tid [[thread_position_in_grid]]) {\n"
"    const uint hd = P[0], ratio = P[1], co = P[2]; const uint width = co * hd;\n"
"    if (tid >= hd) return;\n"
"    float mx = -1.0e30f;\n"
"    if (co == 2u) { for (uint r = 0; r < ratio; r++) { float sp = SSC[r*width+tid]; float sc = SSC[(ratio+r)*width+hd+tid]; if (sp > mx) mx = sp; if (sc > mx) mx = sc; } }\n"
"    else { for (uint r = 0; r < ratio; r++) { float s = SSC[r*width+tid]; if (s > mx) mx = s; } }\n"
"    if (mx <= -1.0e30f * 0.5f) { OUT[tid] = 0.f; return; }\n"
"    float den = 0.f, sum = 0.f;\n"
"    if (co == 2u) { for (uint r = 0; r < ratio; r++) { float wp = exp(SSC[r*width+tid]-mx), wc = exp(SSC[(ratio+r)*width+hd+tid]-mx); den += wp + wc; sum += wp*SKV[r*width+tid]; sum += wc*SKV[(ratio+r)*width+hd+tid]; } }\n"
"    else { for (uint r = 0; r < ratio; r++) { float w = exp(SSC[r*width+tid]-mx); den += w; sum += w*SKV[r*width+tid]; } }\n"
"    OUT[tid] = sum / den;\n"
"}\n"
/* Quantizzazione fp8 e4m3 per blocco di 64 (port di llm_dsv4_fp8_kv_quantize_row). P = {n_nope}. */
"kernel void dsv4_comp_fp8(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n_nope = P[0]; const uint off = gid * 64u; if (off + 64u > n_nope) return;\n"
"    float amax = 0.f;\n"
"    for (uint i = 0; i < 64u; i++) { float av = fabs(X[off+i]); if (av > amax) amax = av; }\n"
"    if (amax < 1.0e-4f) amax = 1.0e-4f;\n"
"    float scale = exp2(ceil(log2(amax / 448.f)));\n"
"    for (uint i = 0; i < 64u; i++) {\n"
"        float v = X[off+i] / scale; if (v > 448.f) v = 448.f; if (v < -448.f) v = -448.f;\n"
"        float a = fabs(v); if (a > 448.f) a = 448.f;\n"
"        float am = a < 1.953125e-3f ? 1.953125e-3f : a;\n"
"        int e = (int)floor(log2(am)); int ste = e - 3; if (ste < -9) ste = -9;\n"
"        float stp = exp2((float)ste); float q = rint(a / stp) * stp; if (q > 448.f) q = 448.f;\n"
"        X[off+i] = (v < 0.f ? -q : q) * scale;\n"
"    }\n"
"}\n"
"kernel void dsv4_comp_fp8_rows(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    const uint n_nope = P[0], hd = P[1]; const uint off = gid.x * 64u; if (off + 64u > n_nope) return;\n"
"    device float* x = X + (ulong)gid.y * hd;\n"
"    float amax = 0.f;\n"
"    for (uint i = 0; i < 64u; i++) { float av = fabs(x[off+i]); if (av > amax) amax = av; }\n"
"    if (amax < 1.0e-4f) amax = 1.0e-4f;\n"
"    float scale = exp2(ceil(log2(amax / 448.f)));\n"
"    for (uint i = 0; i < 64u; i++) {\n"
"        float v = x[off+i] / scale; if (v > 448.f) v = 448.f; if (v < -448.f) v = -448.f;\n"
"        float a = fabs(v); if (a > 448.f) a = 448.f;\n"
"        float am = a < 1.953125e-3f ? 1.953125e-3f : a;\n"
"        int e = (int)floor(log2(am)); int ste = e - 3; if (ste < -9) ste = -9;\n"
"        float stp = exp2((float)ste); float q = rint(a / stp) * stp; if (q > 448.f) q = 448.f;\n"
"        x[off+i] = (v < 0.f ? -q : q) * scale;\n"
"    }\n"
"}\n"
/* ── Mid BATCHED (token in griglia): compressor/indexer a eventi, attenzione con finestra raw lineare ── */
"kernel void dsv4_comp_pool_ev(device const float* SKV0 [[buffer(0)]], device const float* SSC0 [[buffer(1)]],\n"
"    device float* OUT0 [[buffer(2)]], constant uint* P [[buffer(3)]], uint2 gg [[thread_position_in_grid]]) {\n"
"    const uint hd = P[0], ratio = P[1], co = P[2], row0 = P[3], nev = P[4]; const uint width = co * hd;\n"
"    const uint tid = gg.x, ev = gg.y; if (tid >= hd || ev >= nev) return;\n"
"    device const float* SKV = SKV0 + (ulong)(row0 + ev * ratio) * width; device const float* SSC = SSC0 + (ulong)(row0 + ev * ratio) * width; device float* OUT = OUT0 + (ulong)ev * hd;\n"
"    float mx = -1.0e30f;\n"
"    if (co == 2u) { for (uint r = 0; r < ratio; r++) { float sp = SSC[r*width+tid]; float sc = SSC[(ratio+r)*width+hd+tid]; if (sp > mx) mx = sp; if (sc > mx) mx = sc; } }\n"
"    else { for (uint r = 0; r < ratio; r++) { float s = SSC[r*width+tid]; if (s > mx) mx = s; } }\n"
"    if (mx <= -1.0e30f * 0.5f) { OUT[tid] = 0.f; return; }\n"
"    float den = 0.f, sum = 0.f;\n"
"    if (co == 2u) { for (uint r = 0; r < ratio; r++) { float wp = exp(SSC[r*width+tid]-mx), wc = exp(SSC[(ratio+r)*width+hd+tid]-mx); den += wp + wc; sum += wp*SKV[r*width+tid]; sum += wc*SKV[(ratio+r)*width+hd+tid]; } }\n"
"    else { for (uint r = 0; r < ratio; r++) { float w = exp(SSC[r*width+tid]-mx); den += w; sum += w*SKV[r*width+tid]; } }\n"
"    OUT[tid] = sum / den;\n"
"}\n"
"/* Catena compressore FUSA (item 2.1 Fase B, esempio qwen4_native): pool_ev -> rms_w -> rope_ev ->\n"
" * fp8 round-trip -> f16, in UN kernel (una TG per evento, 32 thread). Elimina 4 dispatch/evento/pass\n"
" * e i round-trip su device (la riga compressa resta in threadgroup). Stessa matematica esatta. */\n"
"kernel void dsv4_comp_chain(device const float* SKV0 [[buffer(0)]], device const float* SSC0 [[buffer(1)]],\n"
"    device half* OUT0 [[buffer(2)]], device const float* WN [[buffer(3)]], constant uint* P [[buffer(4)]], constant float* CF [[buffer(5)]],\n"
"    uint ev [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint hd=P[0], ratio=P[1], co=P[2], row0=P[3]; const int n_nope=(int)P[4], n_rot=(int)P[5], pos=(int)(P[6]+ev*(int)ratio);\n"
"    const float eps=CF[0], freq_base=CF[1], freq_scale=CF[2], ext_factor=CF[3], attn_factor=CF[4], beta_fast=CF[5], beta_slow=CF[6], n_ctx_orig=CF[7];\n"
"    const uint width=co*hd;\n"
"    device const float* SKV=SKV0+(ulong)(row0+ev*ratio)*width; device const float* SSC=SSC0+(ulong)(row0+ev*ratio)*width;\n"
"    device half* OUT=OUT0+(ulong)ev*hd; threadgroup float buf[512];\n"
"    for (uint d=tiisg; d<hd; d+=32u) {\n"
"        float mx=-1.0e30f;\n"
"        if (co==2u) { for (uint r=0;r<ratio;r++){ float sp=SSC[r*width+d]; float sc=SSC[(ratio+r)*width+hd+d]; if(sp>mx)mx=sp; if(sc>mx)mx=sc; } }\n"
"        else { for (uint r=0;r<ratio;r++){ float s=SSC[r*width+d]; if(s>mx)mx=s; } }\n"
"        if (mx<=-1.0e30f*0.5f) { buf[d]=0.f; continue; }\n"
"        float den=0.f, sum=0.f;\n"
"        if (co==2u) { for (uint r=0;r<ratio;r++){ float wp=exp(SSC[r*width+d]-mx), wc=exp(SSC[(ratio+r)*width+hd+d]-mx); den+=wp+wc; sum+=wp*SKV[r*width+d]; sum+=wc*SKV[(ratio+r)*width+hd+d]; } }\n"
"        else { for (uint r=0;r<ratio;r++){ float w=exp(SSC[r*width+d]-mx); den+=w; sum+=w*SKV[r*width+d]; } }\n"
"        buf[d]=sum/den;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    { float ss=0.f; for (uint d=tiisg; d<hd; d+=32u) ss+=buf[d]*buf[d]; ss=simd_sum(ss);\n"
"      const float rscale=1.f/sqrt(ss/(float)hd+eps); for (uint d=tiisg; d<hd; d+=32u) buf[d]=buf[d]*rscale*WN[d]; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    { const float theta_scale=pow(freq_base,-2.f/(float)n_rot); float corr0=0.f,corr1=0.f; const float TWO_PI=6.28318530717958647692f;\n"
"      if (ext_factor!=0.f){ float st=floor((float)n_rot*log(n_ctx_orig/(beta_fast*TWO_PI))/(2.f*log(freq_base))); float en=ceil((float)n_rot*log(n_ctx_orig/(beta_slow*TWO_PI))/(2.f*log(freq_base))); corr0=max(0.f,st); corr1=min((float)(n_rot-1),en); }\n"
"      for (int i=(int)tiisg*2; i<n_rot; i+=64) {\n"
"        float theta_extrap=(float)pos; for (int t=0;t<i;t+=2) theta_extrap*=theta_scale;\n"
"        float theta_interp=freq_scale*theta_extrap; float theta=theta_interp, mscale=attn_factor;\n"
"        if (ext_factor!=0.f){ float y=((float)(i/2)-corr0)/max(0.001f,corr1-corr0); float ramp=1.f-min(1.f,max(0.f,y)); float ramp_mix=ramp*ext_factor; theta=theta_interp*(1.f-ramp_mix)+theta_extrap*ramp_mix; mscale*=1.f+0.1f*log(1.f/freq_scale); }\n"
"        float c=cos(theta)*mscale, s=sin(theta)*mscale; float x0=buf[n_nope+i], x1=buf[n_nope+i+1]; buf[n_nope+i]=x0*c-x1*s; buf[n_nope+i+1]=x0*s+x1*c;\n"
"      } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (n_nope>0) { const uint nb=(uint)n_nope;\n"
"      for (uint off=(uint)tiisg*64u; off+64u<=nb; off+=64u*32u) {\n"
"        float amax=0.f; for (uint i=0;i<64u;i++){ float av=fabs(buf[off+i]); if(av>amax)amax=av; }\n"
"        if (amax<1.0e-4f) amax=1.0e-4f; float scale=exp2(ceil(log2(amax/448.f)));\n"
"        for (uint i=0;i<64u;i++){ float v=buf[off+i]/scale; if(v>448.f)v=448.f; if(v<-448.f)v=-448.f; float a=fabs(v); if(a>448.f)a=448.f; float am=a<1.953125e-3f?1.953125e-3f:a; int e=(int)floor(log2(am)); int ste=e-3; if(ste<-9)ste=-9; float stp=exp2((float)ste); float q=rint(a/stp)*stp; if(q>448.f)q=448.f; buf[off+i]=(v<0.f?-q:q)*scale; }\n"
"      } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint d=tiisg; d<hd; d+=32u) OUT[d]=(half)buf[d];\n"
"}\n"
"kernel void dsv4_rope_ev(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], constant float* CF [[buffer(2)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const int n_rot = (int)P[0], head_dim = (int)P[1], pos = (int)(P[2] + tg.y * P[4]); const uint nh = P[3];\n"
"    const float freq_base = CF[0], freq_scale = CF[1], ext_factor = CF[2], attn_factor = CF[3];\n"
"    const float beta_fast = CF[4], beta_slow = CF[5]; const float n_ctx_orig = CF[6];\n"
"    const int n_nope = head_dim - n_rot;\n"
"    const float theta_scale = pow(freq_base, -2.f / (float)n_rot);\n"
"    float corr0 = 0.f, corr1 = 0.f;\n"
"    const float TWO_PI = 6.28318530717958647692f;\n"
"    if (ext_factor != 0.f) {\n"
"        float start = floor((float)n_rot * log(n_ctx_orig / (beta_fast * TWO_PI)) / (2.f * log(freq_base)));\n"
"        float end = ceil((float)n_rot * log(n_ctx_orig / (beta_slow * TWO_PI)) / (2.f * log(freq_base)));\n"
"        corr0 = max(0.f, start); corr1 = min((float)(n_rot - 1), end);\n"
"    }\n"
"    device float* tail = X + ((ulong)tg.y * nh + tg.x) * (ulong)head_dim + n_nope;\n"
"    for (int i = (int)tiisg * 2; i < n_rot; i += 64) {\n"
"        float theta_extrap = (float)pos;\n"
"        for (int t = 0; t < i; t += 2) theta_extrap *= theta_scale;\n"
"        float theta_interp = freq_scale * theta_extrap;\n"
"        float theta = theta_interp, mscale = attn_factor;\n"
"        if (ext_factor != 0.f) {\n"
"            float y = ((float)(i / 2) - corr0) / max(0.001f, corr1 - corr0);\n"
"            float ramp = 1.f - min(1.f, max(0.f, y));\n"
"            float ramp_mix = ramp * ext_factor;\n"
"            theta = theta_interp * (1.f - ramp_mix) + theta_extrap * ramp_mix;\n"
"            mscale *= 1.f + 0.1f * log(1.f / freq_scale);\n"
"        }\n"
"        float c = cos(theta) * mscale, s = sin(theta) * mscale;\n"
"        float x0 = tail[i], x1 = tail[i + 1];\n"
"        tail[i] = x0 * c - x1 * s;\n"
"        tail[i + 1] = x0 * s + x1 * c;\n"
"    }\n"
"}\n"
"kernel void dsv4_indexer_scores_t(device const float* Q [[buffer(0)]], device const float* KVC [[buffer(1)]],\n"
"    device const float* W [[buffer(2)]], device float* SCORES [[buffer(3)]], constant uint* P [[buffer(4)]], device const uint* NC [[buffer(5)]],\n"
"    uint2 cg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint nh = P[0], hd = P[1], maxc = P[2]; const uint c = cg.x, t = cg.y; const uint n_comp = NC[t]; if (c >= n_comp) return;\n"
"    device const float* kv = KVC + (ulong)c * (ulong)hd; float s = 0.f;\n"
"    for (uint h = sg; h < nh; h += 4u) {\n"
"        device const float* qh = Q + ((ulong)t * nh + h) * (ulong)hd;\n"
"        float acc = 0.f; for (uint i = tiisg; i < hd; i += 32u) acc += qh[i] * kv[i];\n"
"        acc = simd_sum(acc); if (acc < 0.f) acc = 0.f; s += acc * W[(ulong)t * nh + h];\n"
"    }\n"
"    threadgroup float ss[4]; if (tiisg == 0u) ss[sg] = s;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0u && tiisg == 0u) SCORES[(ulong)t * maxc + c] = ss[0] + ss[1] + ss[2] + ss[3];\n"
"}\n"
"/* Indexer scores VARIANTE MMA (rif. dsv41 LLM_DSV41_INDEX_MMA_SRC): una threadgroup per (blocco di\n"
" * 32 chiavi, token); QK^T con simdgroup_matrix<float,8,8>, relu*pesi, riduzione teste. Fonte KV f32\n"
" * (icomp). Riduce la rilettura di q di ~32x e usa i tensor core. Stessa semantica di dsv4_indexer_scores_t. */\n"
"kernel void dsv4_indexer_scores_t_mma(device const float* Q0 [[buffer(0)]], device const float* KVC0 [[buffer(1)]],\n"
"    device const float* W0 [[buffer(2)]], device float* SCORES0 [[buffer(3)]], constant uint* P [[buffer(4)]],\n"
"    device const uint* NC [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {\n"
"    const uint H = P[0], D = P[1], Mmax = P[2]; const uint t = g.y, n_comp = NC[t]; const uint block = g.x * 32u;\n"
"    const uint sg = tid / 32u, lane = tid % 32u;\n"
"    const uint fm = ((lane >> 2) & 4u) + ((lane >> 1) & 3u);\n"
"    const uint fn = (((lane >> 2) & 2u) << 1u) + ((lane & 1u) << 1u);\n"
"    device const float* q = Q0 + (ulong)t * H * D; device const float* w = W0 + (ulong)t * H;\n"
"    threadgroup float tile[128u * 32u];\n"
"    for (uint i = tid; i < D * 32u; i += 128u) {\n"
"        const uint d = i / 32u, col = i % 32u, pos = block + col;\n"
"        tile[i] = (pos < n_comp) ? KVC0[(ulong)pos * D + d] : 0.f;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total0 = 0.f, total1 = 0.f;\n"
"    for (uint h = 0u; h < H; h += 8u) {\n"
"        simdgroup_matrix<float,8,8> acc, a, b;\n"
"        acc.thread_elements()[0] = 0.f; acc.thread_elements()[1] = 0.f;\n"
"        for (uint d = 0u; d < D; d += 8u) {\n"
"            a.thread_elements()[0] = (h + fm < H) ? q[(ulong)(h + fm) * D + d + fn] : 0.f;\n"
"            a.thread_elements()[1] = (h + fm < H) ? q[(ulong)(h + fm) * D + d + fn + 1u] : 0.f;\n"
"            b.thread_elements()[0] = tile[(d + fm) * 32u + sg * 8u + fn];\n"
"            b.thread_elements()[1] = tile[(d + fm) * 32u + sg * 8u + fn + 1u];\n"
"            simdgroup_multiply_accumulate(acc, a, b, acc);\n"
"        }\n"
"        const float weight = (h + fm < H) ? w[h + fm] : 0.f;\n"
"        float v0 = max(acc.thread_elements()[0], 0.f) * weight;\n"
"        float v1 = max(acc.thread_elements()[1], 0.f) * weight;\n"
"        for (uint i = 0u; i < 3u; ++i) { const uint mask = i == 0u ? 2u : (i == 1u ? 4u : 16u); v0 += simd_shuffle_xor(v0, (ushort)mask); v1 += simd_shuffle_xor(v1, (ushort)mask); }\n"
"        total0 += v0; total1 += v1;\n"
"    }\n"
"    device float* sc = SCORES0 + (ulong)t * Mmax;\n"
"    if (fm == 0u) {\n"
"        for (uint j = 0u; j < 2u; ++j) { const uint pos = block + sg * 8u + fn + j; if (pos < Mmax) sc[pos] = (pos < n_comp) ? (j == 0u ? total0 : total1) : 0.f; }\n"
"    }\n"
"}\n"
"/* Chiave order-preserving float->uint (monotona anche per negativi; -0 canonico a +0).\n"
" * Gli score dell'indexer possono essere negativi (somma di dot*pesi), quindi NON si puo'\n"
" * usare il trucco ds4 max(v,0): la radix deve ordinare anche i negativi. */\n"
"static inline uint dsv4_idx_key(float v) {\n"
"    if (v == 0.0f) v = 0.0f;\n"
"    const uint bb = as_type<uint>(v);\n"
"    return (bb & 0x80000000u) ? ~bb : (bb | 0x80000000u);\n"
"}\n"
"/* Top-k indexer BATCHED (una threadgroup per token): radix-select a 4 passi sugli 8 bit piu'\n"
" * significativi della chiave -> O(n_comp) invece di O(n_comp^2) per token (contesti lunghi).\n"
" * Esatto e deterministico: rank(chiave) < top_k, con tie-break su indice MINORE (come la\n"
" * reference). Scrive ALLOWED[t*maxc + b] = 0/1 per b < n_comp. */\n"
"kernel void dsv4_indexer_topk_t(device const float* SCORES0 [[buffer(0)]], device int* ALLOWED0 [[buffer(1)]],\n"
"    constant uint* P [[buffer(2)]], device const uint* NC [[buffer(3)]], device int* SELT0 [[buffer(4)]],\n"
"    uint tgid [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]], ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint maxc = P[0], top_k = P[1], nth = P[2]; const uint t = tgid, n = NC[t];\n"
"    device const float* SCORES = SCORES0 + (ulong)t * maxc; device int* ALLOWED = ALLOWED0 + (ulong)t * maxc;\n"
"    device int* SELT = SELT0 + (ulong)t * top_k;\n"
"    if (n == 0u) return;\n"
"    if (n <= top_k) { for (uint b = tid; b < n; b += nth) { ALLOWED[b] = 1; SELT[b] = (int)b; } return; }\n"
"    threadgroup atomic_uint hist[256]; threadgroup uint scan[32]; threadgroup uint found[2];\n"
"    uint prefix = 0u, need = top_k;\n"
"    for (uint pass = 0u; pass < 4u; pass++) {\n"
"        const uint shift = 24u - 8u * pass;\n"
"        const uint mask_hi = pass == 0u ? 0u : (0xFFFFFFFFu << (shift + 8u));\n"
"        for (uint i = tid; i < 256u; i += nth) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint b = tid; b < n; b += nth) {\n"
"            const uint key = dsv4_idx_key(SCORES[b]);\n"
"            if ((key & mask_hi) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint c = tid < 256u ? atomic_load_explicit(&hist[255u - tid], memory_order_relaxed) : 0u;\n"
"        const uint p = simd_prefix_inclusive_sum(c);\n"
"        if (tiisg == 31u && sgitg < 8u) scan[sgitg] = p;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (tid < 256u) {\n"
"            uint above = p - c;\n"
"            for (uint g = 0u; g < sgitg; g++) above += scan[g];\n"
"            if (above < need && above + c >= need) { found[0] = prefix | ((255u - tid) << shift); found[1] = need - above; }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        prefix = found[0]; need = found[1];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const uint chunk = (n + nth - 1u) / nth;\n"
"    const uint b0 = min(tid * chunk, n), b1 = min(b0 + chunk, n);\n"
"    uint n_eq = 0u;\n"
"    for (uint b = b0; b < b1; b++) n_eq += (dsv4_idx_key(SCORES[b]) == prefix) ? 1u : 0u;\n"
"    const uint pe = simd_prefix_exclusive_sum(n_eq);\n"
"    if (tiisg == 31u) scan[sgitg] = pe + n_eq;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sgitg == 0u) {\n"
"        const uint nsg = (nth + 31u) / 32u;\n"
"        const uint sv = tiisg < nsg ? scan[tiisg] : 0u;\n"
"        const uint sp = simd_prefix_exclusive_sum(sv);\n"
"        if (tiisg < nsg) scan[tiisg] = sp;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const uint r_eq0 = pe + scan[sgitg];\n"
"    uint nsel = 0u; { uint rq = r_eq0; for (uint b = b0; b < b1; b++) { const uint key = dsv4_idx_key(SCORES[b]); if (key > prefix) nsel++; else if (key == prefix) { if (rq < need) nsel++; rq++; } } }\n"
"    const uint psel = simd_prefix_exclusive_sum(nsel);\n"
"    if (tiisg == 31u) scan[sgitg] = psel + nsel;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sgitg == 0u) {\n"
"        const uint nsg = (nth + 31u) / 32u;\n"
"        const uint sv = tiisg < nsg ? scan[tiisg] : 0u;\n"
"        const uint sp = simd_prefix_exclusive_sum(sv);\n"
"        if (tiisg < nsg) scan[tiisg] = sp;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    uint wsel = psel + scan[sgitg], r_eq = r_eq0;\n"
"    for (uint b = b0; b < b1; b++) {\n"
"        const uint key = dsv4_idx_key(SCORES[b]);\n"
"        int a = 0;\n"
"        if (key > prefix) { a = 1; SELT[wsel++] = (int)b; }\n"
"        else if (key == prefix) { if (r_eq < need) { a = 1; SELT[wsel++] = (int)b; } r_eq++; }\n"
"        ALLOWED[b] = a;\n"
"    }\n"
"}\n"
"kernel void dsv4_attn_part_t(device const float* Q0 [[buffer(0)]], device const float* RAWK0 [[buffer(1)]],\n"
"    device const half* COMPK [[buffer(2)]], device const int* SELT0 [[buffer(3)]],\n"
"    device float* PART [[buffer(4)]], constant uint* P [[buffer(5)]], device const uint4* TP [[buffer(6)]],\n"
"    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint hd = P[0], top_k = P[1], nsm = P[2], g0 = P[3], maxc = P[4], nh = P[5]; const uint tok = g0 + tg.z;\n"
"    const uint4 tp = TP[tok]; const uint n_raw = tp.x, n_sel = tp.y, ns = tp.w;\n"
"    const uint h = tg.x, s = tg.y, nch = hd / 128u; if (s >= ns) return;\n"
"    device const float* RAWK = RAWK0 + (ulong)tp.z * (ulong)hd; device const int* SELT = SELT0 + (ulong)tok * top_k; device const float* Q = Q0 + (ulong)tok * nh * hd;\n"
"    const uint total = n_raw + n_sel, per = (total + ns - 1u) / ns;\n"
"    const uint t0 = s * per, t1 = min(total, t0 + per);\n"
"    const float kq = 1.f / sqrt((float)hd);\n"
"    device const float4* q4 = (device const float4*)(Q + (ulong)h * (ulong)hd);\n"
"    float4 qv[4], acc[4];\n"
"    for (uint j = 0; j < 4u; j++) { qv[j] = j < nch ? q4[j * 32u + tiisg] : float4(0.f); acc[j] = float4(0.f); }\n"
"    float m = -INFINITY, l = 0.f;\n"
"    for (uint t = t0 + sg; t < t1; t += 4u) {\n"
"        float4 kv[4]; float d = 0.f;\n"
"        if (t < n_raw) { device const float4* k4 = (device const float4*)(RAWK + (ulong)t * (ulong)hd);\n"
"            for (uint j = 0; j < 4u; j++) if (j < nch) { kv[j] = k4[j * 32u + tiisg]; d += dot(qv[j], kv[j]); } }\n"
"        else { const uint rr = t - n_raw; const uint r = (top_k != 0u) ? (uint)SELT[rr] : rr; device const half* kh = COMPK + (ulong)r * (ulong)hd;\n"
"            for (uint j = 0; j < 4u; j++) if (j < nch) { device const half* hh = kh + j * 128u + tiisg * 4u; kv[j] = float4((float)hh[0], (float)hh[1], (float)hh[2], (float)hh[3]); d += dot(qv[j], kv[j]); } }\n"
"        d = simd_sum(d) * kq;\n"
"        const float mn = max(m, d), corr = exp(m - mn), w = exp(d - mn);\n"
"        l = l * corr + w;\n"
"        for (uint j = 0; j < 4u; j++) if (j < nch) acc[j] = acc[j] * corr + w * kv[j];\n"
"        m = mn;\n"
"    }\n"
"    threadgroup float sm[4], sl[4]; threadgroup float sacc[4 * 512];\n"
"    if (tiisg == 0u) { sm[sg] = m; sl[sg] = l; }\n"
"    for (uint j = 0; j < 4u; j++) if (j < nch) *(threadgroup float4*)(sacc + sg * 512u + j * 128u + tiisg * 4u) = acc[j];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const float M = max(max(sm[0], sm[1]), max(sm[2], sm[3]));\n"
"    device float* out = PART + (((ulong)tg.z * nh + h) * (ulong)nsm + (ulong)s) * (ulong)(hd + 2u);\n"
"    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;\n"
"    if (M > -INFINITY) { e0 = exp(sm[0] - M); e1 = exp(sm[1] - M); e2 = exp(sm[2] - M); e3 = exp(sm[3] - M); }\n"
"    for (uint i = tid; i < hd; i += 128u)\n"
"        out[i] = e0 * sacc[i] + e1 * sacc[512u + i] + e2 * sacc[1024u + i] + e3 * sacc[1536u + i];\n"
"    if (tid == 0u) { out[hd] = M; out[hd + 1u] = e0 * sl[0] + e1 * sl[1] + e2 * sl[2] + e3 * sl[3]; }\n"
"}\n"
"/* Attenzione di mezzo VARIANTE MMA (rif. dsv41 packed_sparse_attention): 8 TESTE per threadgroup e\n"
" * tile KV CONDIVISO -> la KV (RAWK/COMPK, f32) e' letta una volta per 8 teste (nel kernel scalare\n"
" * sopra era letta una volta PER TESTA). QK^T e PV con simdgroup_matrix<float,8,8>. Stesso layout di\n"
" * PART del kernel scalare (merge invariato). 4 simdgroup dividono la dimensione canale D/4. */\n"
"kernel void dsv4_attn_part_mma(device const float* Q0 [[buffer(0)]], device const float* RAWK0 [[buffer(1)]],\n"
"    device const float* COMPK [[buffer(2)]], device const int* ALLOWED0 [[buffer(3)]],\n"
"    device float* PART [[buffer(4)]], constant uint* P [[buffer(5)]], device const uint4* TP [[buffer(6)]],\n"
"    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint hd = P[0], has_allowed = P[1], nsm = P[2], g0 = P[3], maxc = P[4], nh = P[5]; const uint tok = g0 + tg.z;\n"
"    const uint4 tp = TP[tok]; const uint n_raw = tp.x, n_comp = tp.y, ns = tp.w;\n"
"    const uint s = tg.y; if (s >= ns) return;\n"
"    const uint first_head = tg.x * 8u; if (first_head >= nh) return;\n"
"    device const float* RAWK = RAWK0 + (ulong)tp.z * (ulong)hd;\n"
"    device const int* ALLOWED = ALLOWED0 + (ulong)tok * maxc;\n"
"    device const float* Q = Q0 + (ulong)tok * nh * hd;\n"
"    const uint total = n_raw + n_comp, per = (total + ns - 1u) / ns;\n"
"    const uint t0 = s * per, t1 = min(total, t0 + per);\n"
"    const float kq = 1.f / sqrt((float)hd);\n"
"    const uint lane = tiisg;\n"
"    const uint fm = ((lane >> 2) & 4u) + ((lane >> 1) & 3u);\n"
"    const uint fn = (((lane >> 2) & 2u) << 1u) + ((lane & 1u) << 1u);\n"
"    threadgroup float tile[8 * 128]; threadgroup float dots[4 * 64], probability[64];\n"
"    threadgroup float maxima[8], denom[8], corr[8]; threadgroup int valid[8];\n"
"    if (tid < 8u) { maxima[tid] = -INFINITY; denom[tid] = 0.f; }\n"
"    const uint nv = hd / 32u, D4 = hd / 4u;\n"
"    simdgroup_matrix<float,8,8> out_m[16];\n"
"    for (uint v = 0; v < nv; ++v) { out_m[v].thread_elements()[0] = 0.f; out_m[v].thread_elements()[1] = 0.f; }\n"
"    for (uint begin = t0; begin < t1; begin += 8u) {\n"
"        for (uint i = tid; i < 8u*hd; i += 128u) {\n"
"            const uint r = i / hd, d = i % hd, j = begin + r;\n"
"            int okk = 0, israw = 0; uint row = 0u;\n"
"            if (j < t1) {\n"
"                if (j < n_raw) { okk = 1; israw = 1; row = j; }\n"
"                else { const uint rr = j - n_raw; if (!has_allowed || ALLOWED[rr]) { okk = 1; row = rr; } }\n"
"            }\n"
"            tile[i] = okk ? (israw ? RAWK[(ulong)row*hd + d] : COMPK[(ulong)row*hd + d]) : 0.f;\n"
"            if (d == 0u) valid[r] = okk;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        simdgroup_matrix<float,8,8> score, a, b;\n"
"        score.thread_elements()[0] = 0.f; score.thread_elements()[1] = 0.f;\n"
"        for (uint k = sg*D4; k < (sg+1u)*D4; k += 8u) {\n"
"            const uint head = first_head + fm;\n"
"            a.thread_elements()[0] = head < nh ? Q[(ulong)head*hd + k + fn] : 0.f;\n"
"            a.thread_elements()[1] = head < nh ? Q[(ulong)head*hd + k + fn + 1u] : 0.f;\n"
"            b.thread_elements()[0] = tile[fn*hd + k + fm];\n"
"            b.thread_elements()[1] = tile[(fn+1u)*hd + k + fm];\n"
"            simdgroup_multiply_accumulate(score, a, b, score);\n"
"        }\n"
"        dots[sg*64u + fm*8u + fn] = score.thread_elements()[0];\n"
"        dots[sg*64u + fm*8u + fn + 1u] = score.thread_elements()[1];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (tid < 8u) {\n"
"            float nm = maxima[tid];\n"
"            for (uint r = 0u; r < 8u; ++r) { const float sc = (dots[tid*8u+r]+dots[64u+tid*8u+r]+dots[128u+tid*8u+r]+dots[192u+tid*8u+r])*kq; if (valid[r] && sc > nm) nm = sc; }\n"
"            corr[tid] = (maxima[tid] == -INFINITY) ? 0.f : exp(maxima[tid] - nm);\n"
"            float tot = denom[tid]*corr[tid];\n"
"            for (uint r = 0u; r < 8u; ++r) { const float sc = (dots[tid*8u+r]+dots[64u+tid*8u+r]+dots[128u+tid*8u+r]+dots[192u+tid*8u+r])*kq; const float pr = valid[r] ? exp(sc - nm) : 0.f; probability[tid*8u+r] = pr; tot += pr; }\n"
"            maxima[tid] = nm; denom[tid] = tot;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        a.thread_elements()[0] = probability[fm*8u + fn];\n"
"        a.thread_elements()[1] = probability[fm*8u + fn + 1u];\n"
"        for (uint v = 0; v < nv; ++v) {\n"
"            const uint d = sg*D4 + v*8u + fn;\n"
"            b.thread_elements()[0] = tile[fm*hd + d];\n"
"            b.thread_elements()[1] = tile[fm*hd + d + 1u];\n"
"            out_m[v].thread_elements()[0] *= corr[fm]; out_m[v].thread_elements()[1] *= corr[fm];\n"
"            simdgroup_multiply_accumulate(out_m[v], a, b, out_m[v]);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const uint head = first_head + fm;\n"
"    if (head < nh) {\n"
"        device float* out = PART + (((ulong)tg.z*nh + head)*(ulong)nsm + (ulong)s)*(ulong)(hd + 2u);\n"
"        for (uint v = 0; v < nv; ++v) { const uint d = sg*D4 + v*8u + fn; out[d] = out_m[v].thread_elements()[0]; out[d+1u] = out_m[v].thread_elements()[1]; }\n"
"    }\n"
"    if (tid < 8u && first_head + tid < nh) {\n"
"        device float* out = PART + (((ulong)tg.z*nh + first_head + tid)*(ulong)nsm + (ulong)s)*(ulong)(hd + 2u);\n"
"        out[hd] = maxima[tid]; out[hd + 1u] = denom[tid];\n"
"    }\n"
"}\n"
"kernel void dsv4_attn_merge_t(device const float* PART [[buffer(0)]], device const float* SINKS [[buffer(1)]],\n"
"    device float* OUT [[buffer(2)]], constant uint* P [[buffer(3)]], device const uint4* TP [[buffer(4)]],\n"
"    uint2 hz [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {\n"
"    const uint hd = P[0], nsm = P[1], g0 = P[2], nh = P[3]; const uint stride = hd + 2u; const uint h = hz.x, tok = g0 + hz.y; const uint ns = TP[tok].w;\n"
"    device const float* base = PART + (((ulong)hz.y * nh + h) * (ulong)nsm) * (ulong)stride;\n"
"    const float sink = SINKS[h];\n"
"    float M = sink;\n"
"    for (uint s = 0; s < ns; s++) M = max(M, base[(ulong)s * stride + hd]);\n"
"    float den = exp(sink - M);\n"
"    for (uint s = 0; s < ns; s++) { const float ms = base[(ulong)s * stride + hd]; if (ms > -INFINITY) den += exp(ms - M) * base[(ulong)s * stride + hd + 1u]; }\n"
"    const float inv = 1.f / den;\n"
"    for (uint i = tid; i < hd; i += 128u) {\n"
"        float a = 0.f;\n"
"        for (uint s = 0; s < ns; s++) { const float ms = base[(ulong)s * stride + hd]; if (ms > -INFINITY) a += exp(ms - M) * base[(ulong)s * stride + i]; }\n"
"        OUT[((ulong)tok * nh + h) * (ulong)hd + i] = a * inv;\n"
"    }\n"
"}\n"
"kernel void dsv4_ape_rows(device const float* PSC [[buffer(0)]], device const float* APE [[buffer(1)]], device float* XSC [[buffer(2)]], constant uint* P [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {\n"
"    const uint width = P[0], ratio = P[1], pm0 = P[2], nt = P[3]; if (g.x >= width || g.y >= nt) return;\n"
"    XSC[(ulong)g.y * width + g.x] = PSC[(ulong)g.y * width + g.x] + APE[(ulong)((pm0 + g.y) % ratio) * width + g.x];\n"
"}\n"
/* Indexer QAT: Hadamard128 + FP4 e2m1 per blocco-32, in-place per riga (head_dim=128). P={n_rows,hd}. */
"kernel void dsv4_indexer_qat(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n_rows = P[0], hd = P[1]; if (gid >= n_rows || hd != 128u) return;\n"
"    thread float x[128]; device float* row = X + (ulong)gid * 128ul;\n"
"    for (uint i = 0; i < 128u; i++) x[i] = row[i];\n"
"    for (uint stride = 1u; stride < 128u; stride <<= 1)\n"
"        for (uint base = 0; base < 128u; base += 2u * stride)\n"
"            for (uint i = 0; i < stride; i++) { float a = x[base+i], b = x[base+stride+i]; x[base+i] = a+b; x[base+stride+i] = a-b; }\n"
"    const float hs = 0.08838834764831845f;\n"
"    for (uint i = 0; i < 128u; i++) x[i] *= hs;\n"
"    { const float lv[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };\n"
"      for (uint off = 0; off < 128u; off += 32u) {\n"
"        float amax = 0.f; for (uint i = 0; i < 32u; i++) { float av = fabs(x[off+i]); if (av > amax) amax = av; }\n"
"        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;\n"
"        float scale = exp2(ceil(log2(amax / 6.f)));\n"
"        for (uint i = 0; i < 32u; i++) {\n"
"            float v = x[off+i] / scale; if (v > 6.f) v = 6.f; if (v < -6.f) v = -6.f;\n"
"            float sg = v < 0.f ? -1.f : 1.f; float ax = fmin(fabs(v), 6.f);\n"
"            int best = 0; float bd = fabs(ax - lv[0]);\n"
"            for (int k = 1; k < 8; k++) { float d = fabs(ax - lv[k]); if (d < bd || (d == bd && (k & 1) == 0 && (best & 1) != 0)) { best = k; bd = d; } }\n"
"            x[off+i] = sg * lv[best] * scale;\n"
"        }\n"
"      } }\n"
"    for (uint i = 0; i < 128u; i++) row[i] = x[i];\n"
"}\n"
/* Indexer score: s[c] = sum_h relu(dot(q_h, kvc_c)) * w_h. Un threadgroup per riga c. */
"kernel void dsv4_indexer_scores(device const float* Q [[buffer(0)]], device const float* KVC [[buffer(1)]],\n"
"    device const float* W [[buffer(2)]], device float* SCORES [[buffer(3)]], constant uint* P [[buffer(4)]],\n"
"    uint c [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint nh = P[0], hd = P[1], n_comp = P[2]; if (c >= n_comp) return;\n"
"    device const float* kv = KVC + (ulong)c * (ulong)hd; float s = 0.f;\n"
"    for (uint h = sg; h < nh; h += 4u) {\n"
"        device const float* qh = Q + (ulong)h * (ulong)hd;\n"
"        float acc = 0.f; for (uint i = tiisg; i < hd; i += 32u) acc += qh[i] * kv[i];\n"
"        acc = simd_sum(acc); if (acc < 0.f) acc = 0.f; s += acc * W[h];\n"
"    }\n"
"    threadgroup float ss[4]; if (tiisg == 0u) ss[sg] = s;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0u && tiisg == 0u) SCORES[c] = ss[0] + ss[1] + ss[2] + ss[3];\n"
"}\n"
/* Indexer top-k deterministico (stesso insieme della selezione seriale «primo massimo, indice
 * crescente»): un thread per riga c, rank = #righe che la battono (score maggiore, o uguale con indice
 * minore); allowed[c] = rank < top_k. Parallelo, O(n_comp) per thread. P={n_comp, top_k}. */
"kernel void dsv4_indexer_topk(device const float* SCORES [[buffer(0)]], device int* ALLOWED [[buffer(1)]],\n"
"    constant uint* P [[buffer(2)]], uint c [[thread_position_in_grid]]) {\n"
"    const uint n_comp = P[0], top_k = P[1]; if (c >= n_comp) return;\n"
"    const float s = SCORES[c]; uint rank = 0;\n"
"    for (uint j = 0; j < n_comp; j++) { const float sj = SCORES[j]; if (sj > s || (sj == s && j < c)) rank++; }\n"
"    ALLOWED[c] = rank < top_k ? 1 : 0;\n"
"}\n"
/* Shift del buffer di stato: copia [src, src+n) in [0, n) (ds4: righe ratio..2ratio -> 0..ratio). P={n,src}. */
"kernel void dsv4_comp_shift(device float* BUF [[buffer(0)]], constant uint* P [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0], src = P[1]; if (gid >= n) return; BUF[gid] = BUF[src + gid];\n"
"}\n"
/* X *= CF[0] su n elementi (scala dei pesi indexer). P={n}. */
"kernel void dsv4_scale(device float* X [[buffer(0)]], constant float* CF [[buffer(1)]], constant uint* P [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0]; if (gid >= n) return; X[gid] *= CF[0];\n"
"}\n"
/* DST[gid] = SRC[gid]. P={n}. */
"kernel void dsv4_copy(device const float* SRC [[buffer(0)]], device float* DST [[buffer(1)]], constant uint* P [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0]; if (gid >= n) return; DST[gid] = SRC[gid];\n"
"}\n"
/* Arrotondamento a f16 in-place (ingresso cache KV, come llm_ggml_f16_round). P={n}. */
"kernel void dsv4_f16(device float* X [[buffer(0)]], constant uint* P [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0]; if (gid >= n) return; X[gid] = (float)((half)X[gid]);\n"
"}\n";

struct LlmDsv4Metal {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
    id<MTLComputePipelineState> pso_q8k_iq2;
    id<MTLComputePipelineState> pso_q8k_q2k;
    id<MTLComputePipelineState> pso_q8k_q4k;
    id<MTLComputePipelineState> pso_red_parts;   /* somma ordinata dei parziali Q8K (determinismo) */
    id<MTLBuffer> spt[64]; size_t spt_cap[64];   /* parziali per job di gemv_batch */
    id<MTLComputePipelineState> pso_q8_0;  /* GEMV Q8_0 veloce (scale 1x/blocco) */
    id<MTLComputePipelineState> pso_pair_swiglu_q8_0;   /* FUSO: gate+up+swiglu Q8_0 (3->1 dispatch) */
    id<MTLComputePipelineState> pso_gemv_q8_0_add;       /* FUSO: down Q8_0 + OUT += (2->1 dispatch) */
    id<MTLComputePipelineState> pso_pair;   /* MoE fuso gate+up+swiglu (ds4 pair) */
    id<MTLComputePipelineState> pso_sum6_q2k; /* down 6 esperti Q2_K + somma (ds4 sum6) */
    id<MTLComputePipelineState> pso_slots6;  /* pair_swiglu 6 esperti in 1 dispatch (ds4 slots6) */
    id<MTLComputePipelineState> pso_mmp_q8, pso_mmp_iq2, pso_mmp_q2k, pso_mmp_q4k, pso_mmp_iq2a, pso_mmp_q2ka, pso_mmp_q4ka;   /* MPP (nil se non disponibile) */
    id<MTLComputePipelineState> pso_route_b, pso_gemv_nt, pso_state_store, pso_hcb_scale, pso_hcb_mixes, pso_hcb_sink, pso_hcb_norm, pso_hcb_post, pso_mm_iq2a, pso_mm_q2ka, pso_mm_q4ka, pso_q4_pair, pso_q4_sum6, pso_rms_rows, pso_head_rms_rows, pso_rope_rows, pso_rope_rows_inv, pso_mm_map1, pso_mm_map8, pso_fp8_rows, pso_pool_ev, pso_comp_chain, pso_rope_ev, pso_iscore_t, pso_iscore_t_mma, pso_itopk_t, pso_apart_t, pso_apart_mma, pso_amerge_t, pso_ape_rows, pso_mm_q8, pso_mm_map0, pso_mm_iq2, pso_mm_q2k, pso_mm_q4k, pso_swiglu_w, pso_sum_slots6;  /* MoE batched mul_mm_id */
    id<MTLComputePipelineState> pso_rms_w, pso_head_rms, pso_rope; /* RMSNorm/RoPE su GPU (ds4) */
    id<MTLComputePipelineState> pso_swiglu, pso_add; /* shared expert su GPU (swiglu + accumulate) */
    id<MTLComputePipelineState> pso_rms_no_w, pso_gemv_f32, pso_sinkhorn, pso_hc_sum, pso_hc_post; /* mHC */
    id<MTLComputePipelineState> pso_route;   /* router su GPU (softplus/top-k) */
    id<MTLComputePipelineState> pso_addr_pair, pso_addr_sum6; /* MoE via tabella indirizzi (ds4 addr) */
    id<MTLComputePipelineState> pso_tbl_pair, pso_tbl_sum6;  /* MoE via argument buffer (ds4 table) */
    id<MTLComputePipelineState> pso_attn_mixed;  /* attn mista raw+comp su device (ds4) */
    id<MTLComputePipelineState> pso_attn_part, pso_attn_merge;  /* attn split (flash-decoding) + merge */
    id<MTLComputePipelineState> pso_attn_p8;  /* attn decode packed8 (port vendored) */
    id<MTLComputePipelineState> pso_comp_pool, pso_comp_fp8;  /* compressor su device (ds4) */
    id<MTLComputePipelineState> pso_idx_qat, pso_idx_scores, pso_idx_topk;  /* indexer su device (ds4) */
    id<MTLComputePipelineState> pso_comp_shift;  /* shift dello stato compressor (ds4) */
    id<MTLComputePipelineState> pso_scale, pso_copy;  /* ausiliari middle su GPU (ds4) */
    id<MTLComputePipelineState> pso_f16;  /* f16-round ingresso cache KV (ds4) */
    id<MTLArgumentEncoder> enc_pair, enc_down;
    id<MTLResidencySet> residency;  /* CONDIVISO da tutti gli argbuf (limite 32 set/coda) */
    int res_dirty;                  /* allocazioni aggiunte al set e non ancora committate */
    /* Command buffer CONDIVISO (fusione per-layer): piu' funzioni attn encodano qui e si
     * committa/aspetta UNA volta sola (riduce i round-trip CPU<->GPU per layer). */
    id<MTLCommandBuffer> scb;
    id<MTLComputeCommandEncoder> se;
    int sactive;
    /* Buffer scratch PERSISTENTI riusati (slot per job): niente alloc/copia per-token. */
    id<MTLBuffer> sx[96], so[64], spb[64];
    /* Cache PERSISTENTE delle view no-copy (base pagina -> MTLBuffer): creare una view costa decine
     * di µs e nel MoE ne servono ~20 per layer; l'esperto in store/mmap ha un indirizzo stabile. */
    struct { uintptr_t base; size_t len; id<MTLBuffer> buf; } *nc;
    uint32_t nc_cap, nc_n;
    size_t sx_cap[96], so_cap[64], spb_cap[64];
};

/* Restituisce uno slot scratch persistente di almeno `need` byte (rialloca solo se cresce). */
static id<MTLBuffer> dsv4_scratch(LlmDsv4Metal* b, id<MTLBuffer>* slot, size_t* cap, size_t need) {
    if (*slot == nil || *cap < need) {
        if (*slot) { [*slot release]; *slot = nil; *cap = 0; }
        *slot = [b->dev newBufferWithLength:need options:MTLResourceStorageModeShared];
        if (*slot) *cap = need;
    }
    return *slot;
}

static void dsv4_res_commit(LlmDsv4Metal* b);   /* definita piu' sotto */
/* Apre/chiude il command buffer CONDIVISO: le funzioni attn che trovano `sactive` encodano nel suo
 * encoder e NON committano/attendono; il chiamante fa begin/end attorno alla catena. */
void llm_dsv4_metal_shared_begin(LlmDsv4Metal* b) {
    if (!b || b->sactive) return;
    b->scb = [b->queue commandBuffer];
    b->se = [b->scb computeCommandEncoder];
    b->sactive = 1;
}
void llm_dsv4_metal_shared_end(LlmDsv4Metal* b) {
    if (!b || !b->sactive) return;
    [b->se endEncoding];
    dsv4_res_commit(b); [b->scb commit];
    [b->scb waitUntilCompleted]; dsv4_gpu_acct(b->scb);
    b->scb = nil; b->se = nil; b->sactive = 0;
}

/* Aggiunge un'allocazione al residency set CONDIVISO (creato al bisogno): le view no-copy dei pesi restano
 * residenti fra una command buffer e l'altra, senza rivalidarle a ogni submit. Il commit e' rimandato. */
static void dsv4_res_add(LlmDsv4Metal* b, id<MTLBuffer> buf) {
    if (!b->residency) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.initialCapacity = 1024;
        b->residency = [b->dev newResidencySetWithDescriptor:d error:nil];
        [d release];
        if (!b->residency) return;
        [b->queue addResidencySet:b->residency];
    }
    [b->residency addAllocation:buf];
    b->res_dirty = 1;
}
static void dsv4_res_commit(LlmDsv4Metal* b) {
    if (b->residency && b->res_dirty) { [b->residency commit]; [b->residency requestResidency]; b->res_dirty = 0; }
}

/* View no-copy della memoria [ptr, ptr+bytes): base allineata a pagina, `*lead` = scarto di `ptr`
 * dalla base. Cached e persistente (rilasciata in destroy): NON va rilasciata dal chiamante.
 * Tabella piena/allocazione fallita -> nil. */
static id<MTLBuffer> dsv4_nocopy_view(LlmDsv4Metal* b, const void* ptr, size_t bytes, size_t* lead) {
    const size_t page = (size_t)getpagesize();
    uintptr_t base = (uintptr_t)ptr & ~(uintptr_t)(page - 1);
    *lead = (size_t)((uintptr_t)ptr - base);
    size_t len = ((*lead + bytes + page - 1) / page) * page;
    if (!b->nc) {
        b->nc_cap = 1u << 17;
        b->nc = (__typeof__(b->nc))calloc(b->nc_cap, sizeof *b->nc);
        if (!b->nc) { b->nc_cap = 0; return nil; }
    }
    uint32_t mask = b->nc_cap - 1, h = (uint32_t)((base >> 14) * 2654435761u) & mask;
    for (uint32_t p = 0; p < b->nc_cap; p++) {
        uint32_t i = (h + p) & mask;
        if (b->nc[i].buf == nil) {
            if (b->nc_n >= b->nc_cap / 2) return nil;
            id<MTLBuffer> v = [b->dev newBufferWithBytesNoCopy:(void*)base length:len
                                                       options:MTLResourceStorageModeShared deallocator:nil];
            if (!v) return nil;
            dsv4_res_add(b, v);
            b->nc[i].base = base; b->nc[i].len = len; b->nc[i].buf = v; b->nc_n++;
            return v;
        }
        if (b->nc[i].base == base) {
            if (b->nc[i].len >= len) return b->nc[i].buf;
            id<MTLBuffer> v = [b->dev newBufferWithBytesNoCopy:(void*)base length:len
                                                       options:MTLResourceStorageModeShared deallocator:nil];
            if (!v) return nil;
            [b->nc[i].buf release]; b->nc[i].buf = v; b->nc[i].len = len;
            return v;
        }
    }
    return nil;
}

int llm_dsv4_metal_available(void) {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (d) { [d release]; return 1; }
    return 0;
}

LlmDsv4Metal* llm_dsv4_metal_create(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return NULL;
    NSError* err = nil;
    NSString* src = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s", LLM_DSV4_METAL_TABLES, LLM_DSV4_METAL_KERNELS, LLM_DSV4_METAL_Q8K, LLM_DSV4_METAL_MOE, LLM_DSV4_METAL_MMID];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "dsv4_metal: compile fallita: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        [dev release]; return NULL;
    }
    id<MTLFunction> f1 = [lib newFunctionWithName:@"dsv4_gemv"];
    id<MTLFunction> f2 = [lib newFunctionWithName:@"dsv4_gemv_q8k_iq2"];
    id<MTLFunction> f3 = [lib newFunctionWithName:@"dsv4_gemv_q8k_q2k"];
    id<MTLFunction> f4 = [lib newFunctionWithName:@"dsv4_gemv_q8k_q4k"];
    if (!f1 || !f2 || !f3 || !f4) { fprintf(stderr, "dsv4_metal: funzioni mancanti\n"); [lib release]; [dev release]; return NULL; }
    id<MTLFunction> f5 = [lib newFunctionWithName:@"dsv4_pair_swiglu_iq2"];
    if (!f5) fprintf(stderr, "dsv4_metal: kernel dsv4_pair_swiglu_iq2 NON compilato\n");
    id<MTLFunction> f6 = [lib newFunctionWithName:@"dsv4_sum6_q2k"];
    if (!f6) fprintf(stderr, "dsv4_metal: kernel dsv4_sum6_q2k NON compilato\n");
    id<MTLFunction> f7 = [lib newFunctionWithName:@"dsv4_rms_norm_w"];
    id<MTLFunction> f8 = [lib newFunctionWithName:@"dsv4_head_rms"];
    id<MTLFunction> f9 = [lib newFunctionWithName:@"dsv4_rope_tail_ext"];
    id<MTLFunction> f10 = [lib newFunctionWithName:@"dsv4_slots6_pair_swiglu_iq2"];
    id<MTLFunction> f12 = [lib newFunctionWithName:@"dsv4_swiglu"];
    id<MTLFunction> f13 = [lib newFunctionWithName:@"dsv4_add_inplace"];
    id<MTLFunction> f15 = [lib newFunctionWithName:@"dsv4_rms_no_w"];
    id<MTLFunction> f16 = [lib newFunctionWithName:@"dsv4_gemv_f32"];
    id<MTLFunction> f17 = [lib newFunctionWithName:@"dsv4_hc_sinkhorn"];
    id<MTLFunction> f18 = [lib newFunctionWithName:@"dsv4_hc_sum"];
    id<MTLFunction> f19 = [lib newFunctionWithName:@"dsv4_hc_post"];
    id<MTLFunction> f20 = [lib newFunctionWithName:@"dsv4_route"];
    id<MTLFunction> f21 = [lib newFunctionWithName:@"dsv4_slots6_addr_pair_swiglu_iq2"];
    id<MTLFunction> f22 = [lib newFunctionWithName:@"dsv4_sum6_addr_q2k"];
    if (!f7 || !f8 || !f9) fprintf(stderr, "dsv4_metal: kernel rms/rope NON compilato\n");
    if (!f10) fprintf(stderr, "dsv4_metal: kernel dsv4_slots6_pair_swiglu_iq2 NON compilato\n");
    LlmDsv4Metal* b = (LlmDsv4Metal*)calloc(1, sizeof *b);
    b->dev = [dev retain];
    b->queue = [dev newCommandQueue];
    b->pso = [dev newComputePipelineStateWithFunction:f1 error:&err];
    b->pso_q8k_iq2 = [dev newComputePipelineStateWithFunction:f2 error:&err];
    b->pso_q8k_q2k = [dev newComputePipelineStateWithFunction:f3 error:&err];
    b->pso_q8k_q4k = [dev newComputePipelineStateWithFunction:f4 error:&err];
    { id<MTLFunction> fr = [lib newFunctionWithName:@"dsv4_red_parts"]; b->pso_red_parts = fr ? [dev newComputePipelineStateWithFunction:fr error:&err] : nil; if (fr) [fr release]; }
    { id<MTLFunction> fq0 = [lib newFunctionWithName:@"dsv4_gemv_q8_0"];
      b->pso_q8_0 = fq0 ? [dev newComputePipelineStateWithFunction:fq0 error:&err] : nil;
      if (!fq0) fprintf(stderr, "dsv4_metal: kernel dsv4_gemv_q8_0 NON compilato\n"); }
    { id<MTLFunction> fps = [lib newFunctionWithName:@"dsv4_pair_swiglu_q8_0"];
      b->pso_pair_swiglu_q8_0 = fps ? [dev newComputePipelineStateWithFunction:fps error:&err] : nil;
      if (!fps) fprintf(stderr, "dsv4_metal: kernel dsv4_pair_swiglu_q8_0 NON compilato\n"); }
    { id<MTLFunction> fda = [lib newFunctionWithName:@"dsv4_gemv_q8_0_add"];
      b->pso_gemv_q8_0_add = fda ? [dev newComputePipelineStateWithFunction:fda error:&err] : nil;
      if (!fda) fprintf(stderr, "dsv4_metal: kernel dsv4_gemv_q8_0_add NON compilato\n"); }
    b->pso_pair    = f5 ? [dev newComputePipelineStateWithFunction:f5 error:&err] : nil;
    b->pso_sum6_q2k = f6 ? [dev newComputePipelineStateWithFunction:f6 error:&err] : nil;
    if (@available(macOS 26.0, *)) {   /* kernel MPP (tensor ops Metal 4): libreria separata, opzionale */
        NSString* msrc = [NSString stringWithFormat:@"#include <metal_stdlib>\n#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp::tensor_ops;\n%s\n%s\n%s",
                          LLM_DSV4_METAL_TABLES, LLM_DSV4_METAL_MMID, LLM_DSV4_METAL_MMID_MPP];
        MTLCompileOptions* mco = [MTLCompileOptions new]; mco.languageVersion = MTLLanguageVersion4_0;
        NSError* merr = nil;
        id<MTLLibrary> mlib = [dev newLibraryWithSource:msrc options:mco error:&merr];
        [mco release];
        if (!mlib) fprintf(stderr, "dsv4_metal: kernel MPP non disponibili (%s) -> simdgroup\n", merr ? [[merr localizedDescription] UTF8String] : "?");
        else {
            struct { const char* n; id<MTLComputePipelineState> __strong* dst; } mk[] = {
                { "kernel_mul_mm_id_q8_0_f32_mpp", &b->pso_mmp_q8 }, { "kernel_mul_mm_id_iq2_xxs_f32_mpp", &b->pso_mmp_iq2 }, { "kernel_mul_mm_id_q2_K_f32_mpp", &b->pso_mmp_q2k }, { "kernel_mul_mm_id_q4_K_f32_mpp", &b->pso_mmp_q4k },
                { "kernel_mul_mm_id_iq2_xxs_addr_f32_mpp", &b->pso_mmp_iq2a }, { "kernel_mul_mm_id_q2_K_addr_f32_mpp", &b->pso_mmp_q2ka }, { "kernel_mul_mm_id_q4_K_addr_f32_mpp", &b->pso_mmp_q4ka } };
            for (size_t i = 0; i < sizeof mk / sizeof mk[0]; i++) {
                id<MTLFunction> mf = [mlib newFunctionWithName:[NSString stringWithUTF8String:mk[i].n]];
                *mk[i].dst = mf ? [dev newComputePipelineStateWithFunction:mf error:&merr] : nil;
                if (mf) [mf release];
            }
            [mlib release];
        }
    }
    {   /* MoE batched (mul_mm_id): opzionale, se manca il path per-token resta valido */
        struct { const char* n; id<MTLComputePipelineState> __strong* dst; } mmk[] = {
            { "kernel_mul_mm_id_map0_ne20_6", &b->pso_mm_map0 }, { "kernel_mul_mm_id_map0_ne20_1", &b->pso_mm_map1 }, { "dsv4_gemv_f32_nt", &b->pso_gemv_nt }, { "dsv4_route_batch", &b->pso_route_b }, { "dsv4_state_store", &b->pso_state_store }, { "dsv4_hcb_scale", &b->pso_hcb_scale }, { "dsv4_hcb_mixes", &b->pso_hcb_mixes },
            { "dsv4_hcb_sinkhorn", &b->pso_hcb_sink }, { "dsv4_hcb_norm", &b->pso_hcb_norm }, { "dsv4_hcb_post", &b->pso_hcb_post }, { "kernel_mul_mm_id_iq2_xxs_addr_f32", &b->pso_mm_iq2a },
            { "kernel_mul_mm_id_q2_K_addr_f32", &b->pso_mm_q2ka }, { "kernel_mul_mm_id_q4_K_addr_f32", &b->pso_mm_q4ka }, { "kernel_mul_mv_slots6_q4_K_pair_swiglu_f32", &b->pso_q4_pair },
            { "kernel_mul_mv_slots6_q4_K_sum6_f32", &b->pso_q4_sum6 }, { "dsv4_rms_w_rows", &b->pso_rms_rows },
            { "dsv4_head_rms_rows", &b->pso_head_rms_rows }, { "dsv4_comp_fp8_rows", &b->pso_fp8_rows }, { "dsv4_comp_pool_ev", &b->pso_pool_ev }, { "dsv4_comp_chain", &b->pso_comp_chain }, { "dsv4_rope_ev", &b->pso_rope_ev }, { "dsv4_indexer_scores_t", &b->pso_iscore_t }, { "dsv4_indexer_scores_t_mma", &b->pso_iscore_t_mma }, { "dsv4_indexer_topk_t", &b->pso_itopk_t }, { "dsv4_attn_part_t", &b->pso_apart_t }, { "dsv4_attn_part_mma", &b->pso_apart_mma }, { "dsv4_attn_merge_t", &b->pso_amerge_t }, { "dsv4_ape_rows", &b->pso_ape_rows }, { "dsv4_rope_rows", &b->pso_rope_rows }, { "dsv4_rope_rows_inv", &b->pso_rope_rows_inv },
            { "kernel_mul_mm_id_map0_ne20_8", &b->pso_mm_map8 }, { "kernel_mul_mm_id_q8_0_f32", &b->pso_mm_q8 }, { "kernel_mul_mm_id_iq2_xxs_f32", &b->pso_mm_iq2 },
            { "kernel_mul_mm_id_q2_K_f32", &b->pso_mm_q2k }, { "kernel_mul_mm_id_q4_K_f32", &b->pso_mm_q4k },
            { "dsv4_swiglu_w", &b->pso_swiglu_w }, { "dsv4_sum_slots6", &b->pso_sum_slots6 } };
        for (size_t i = 0; i < sizeof mmk / sizeof mmk[0]; i++) {
            id<MTLFunction> mf = [lib newFunctionWithName:[NSString stringWithUTF8String:mmk[i].n]];
            *mmk[i].dst = mf ? [dev newComputePipelineStateWithFunction:mf error:&err] : nil;
            if (!*mmk[i].dst) fprintf(stderr, "dsv4_metal: kernel %s NON compilato\n", mmk[i].n);
            if (mf) [mf release];
        }
    }
    b->pso_rms_w    = f7 ? [dev newComputePipelineStateWithFunction:f7 error:&err] : nil;
    b->pso_head_rms = f8 ? [dev newComputePipelineStateWithFunction:f8 error:&err] : nil;
    b->pso_rope     = f9 ? [dev newComputePipelineStateWithFunction:f9 error:&err] : nil;
    b->pso_slots6   = f10 ? [dev newComputePipelineStateWithFunction:f10 error:&err] : nil;
    b->pso_swiglu   = f12 ? [dev newComputePipelineStateWithFunction:f12 error:&err] : nil;
    b->pso_add      = f13 ? [dev newComputePipelineStateWithFunction:f13 error:&err] : nil;
    b->pso_rms_no_w = f15 ? [dev newComputePipelineStateWithFunction:f15 error:&err] : nil;
    b->pso_gemv_f32 = f16 ? [dev newComputePipelineStateWithFunction:f16 error:&err] : nil;
    b->pso_sinkhorn = f17 ? [dev newComputePipelineStateWithFunction:f17 error:&err] : nil;
    b->pso_hc_sum   = f18 ? [dev newComputePipelineStateWithFunction:f18 error:&err] : nil;
    b->pso_hc_post  = f19 ? [dev newComputePipelineStateWithFunction:f19 error:&err] : nil;
    b->pso_route    = f20 ? [dev newComputePipelineStateWithFunction:f20 error:&err] : nil;
    b->pso_addr_pair = f21 ? [dev newComputePipelineStateWithFunction:f21 error:&err] : nil;
    b->pso_addr_sum6 = f22 ? [dev newComputePipelineStateWithFunction:f22 error:&err] : nil;
    id<MTLFunction> f23 = [lib newFunctionWithName:@"dsv4_slots6_tbl_pair"];
    id<MTLFunction> f24 = [lib newFunctionWithName:@"dsv4_sum6_tbl"];
    b->pso_tbl_pair = f23 ? [dev newComputePipelineStateWithFunction:f23 error:&err] : nil;
    b->pso_tbl_sum6 = f24 ? [dev newComputePipelineStateWithFunction:f24 error:&err] : nil;
    b->enc_pair = f23 ? [f23 newArgumentEncoderWithBufferIndex:1] : nil;
    b->enc_down = f24 ? [f24 newArgumentEncoderWithBufferIndex:1] : nil;
    if (!f23) fprintf(stderr, "dsv4_metal: kernel dsv4_slots6_tbl_pair NON compilato\n");
    if (!f24) fprintf(stderr, "dsv4_metal: kernel dsv4_sum6_tbl NON compilato\n");
    id<MTLFunction> f25 = [lib newFunctionWithName:@"dsv4_attn_mixed"];
    b->pso_attn_mixed = f25 ? [dev newComputePipelineStateWithFunction:f25 error:&err] : nil;
    if (!f25) fprintf(stderr, "dsv4_metal: kernel dsv4_attn_mixed NON compilato\n");
    id<MTLFunction> f25p = [lib newFunctionWithName:@"dsv4_attn_part"];
    id<MTLFunction> f25m = [lib newFunctionWithName:@"dsv4_attn_merge"];
    b->pso_attn_part  = f25p ? [dev newComputePipelineStateWithFunction:f25p error:&err] : nil;
    b->pso_attn_merge = f25m ? [dev newComputePipelineStateWithFunction:f25m error:&err] : nil;
    { id<MTLFunction> f25x = [lib newFunctionWithName:@"dsv4_attn_decode_p8"];
      b->pso_attn_p8 = f25x ? [dev newComputePipelineStateWithFunction:f25x error:&err] : nil;
      if (!f25x) fprintf(stderr, "dsv4_metal: kernel dsv4_attn_decode_p8 NON compilato\n");
      if (f25x) [f25x release]; }
    if (!f25p || !f25m) fprintf(stderr, "dsv4_metal: kernel attn split NON compilato\n");
    id<MTLFunction> f26 = [lib newFunctionWithName:@"dsv4_comp_pool"];
    id<MTLFunction> f27 = [lib newFunctionWithName:@"dsv4_comp_fp8"];
    b->pso_comp_pool = f26 ? [dev newComputePipelineStateWithFunction:f26 error:&err] : nil;
    b->pso_comp_fp8  = f27 ? [dev newComputePipelineStateWithFunction:f27 error:&err] : nil;
    if (!f26 || !f27) fprintf(stderr, "dsv4_metal: kernel compressor NON compilato\n");
    id<MTLFunction> f28 = [lib newFunctionWithName:@"dsv4_indexer_qat"];
    id<MTLFunction> f29 = [lib newFunctionWithName:@"dsv4_indexer_scores"];
    id<MTLFunction> f30 = [lib newFunctionWithName:@"dsv4_indexer_topk"];
    b->pso_idx_qat    = f28 ? [dev newComputePipelineStateWithFunction:f28 error:&err] : nil;
    b->pso_idx_scores = f29 ? [dev newComputePipelineStateWithFunction:f29 error:&err] : nil;
    b->pso_idx_topk   = f30 ? [dev newComputePipelineStateWithFunction:f30 error:&err] : nil;
    if (!f28 || !f29 || !f30) fprintf(stderr, "dsv4_metal: kernel indexer NON compilato\n");
    id<MTLFunction> f31 = [lib newFunctionWithName:@"dsv4_comp_shift"];
    b->pso_comp_shift = f31 ? [dev newComputePipelineStateWithFunction:f31 error:&err] : nil;
    if (!f31) fprintf(stderr, "dsv4_metal: kernel dsv4_comp_shift NON compilato\n");
    id<MTLFunction> f32 = [lib newFunctionWithName:@"dsv4_scale"];
    id<MTLFunction> f33 = [lib newFunctionWithName:@"dsv4_copy"];
    b->pso_scale = f32 ? [dev newComputePipelineStateWithFunction:f32 error:&err] : nil;
    b->pso_copy  = f33 ? [dev newComputePipelineStateWithFunction:f33 error:&err] : nil;
    if (!f32 || !f33) fprintf(stderr, "dsv4_metal: kernel scale/copy NON compilato\n");
    id<MTLFunction> f34 = [lib newFunctionWithName:@"dsv4_f16"];
    b->pso_f16 = f34 ? [dev newComputePipelineStateWithFunction:f34 error:&err] : nil;
    if (!f34) fprintf(stderr, "dsv4_metal: kernel dsv4_f16 NON compilato\n");
    [lib release];
    if (!b->pso || !b->pso_q8k_iq2 || !b->pso_q8k_q2k || !b->pso_q8k_q4k || !b->pso_red_parts || !b->pso_pair || !b->pso_sum6_q2k || !b->pso_rms_w || !b->pso_head_rms || !b->pso_rope || !b->pso_slots6 || !b->pso_swiglu || !b->pso_add || !b->pso_rms_no_w || !b->pso_gemv_f32 || !b->pso_sinkhorn || !b->pso_hc_sum || !b->pso_hc_post || !b->pso_route || !b->pso_addr_pair || !b->pso_addr_sum6 || !b->queue) {
        fprintf(stderr, "dsv4_metal: pipeline fallita: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        llm_dsv4_metal_destroy(b); return NULL;
    }
    return b;
}

static void dsv4_view_cache_clear(void);
void llm_dsv4_metal_destroy(LlmDsv4Metal* b) {
    if (!b) return;
    if (b->pso)          [b->pso release];
    if (b->pso_q8k_iq2)  [b->pso_q8k_iq2 release];
    if (b->pso_q8k_q2k)  [b->pso_q8k_q2k release];
    if (b->pso_q8k_q4k)  [b->pso_q8k_q4k release];
    if (b->pso_red_parts) [b->pso_red_parts release];
    for (int i = 0; i < 64; i++) if (b->spt[i]) [b->spt[i] release];
    if (b->pso_q8_0)     [b->pso_q8_0 release];
    if (b->pso_pair)     [b->pso_pair release];
    if (b->pso_sum6_q2k) [b->pso_sum6_q2k release];
    if (b->pso_attn_part) [b->pso_attn_part release];
    if (b->pso_mmp_q8) [b->pso_mmp_q8 release]; if (b->pso_mmp_iq2) [b->pso_mmp_iq2 release]; if (b->pso_mmp_q2k) [b->pso_mmp_q2k release]; if (b->pso_mmp_q4k) [b->pso_mmp_q4k release];
    if (b->pso_mmp_iq2a) [b->pso_mmp_iq2a release]; if (b->pso_mmp_q2ka) [b->pso_mmp_q2ka release]; if (b->pso_mmp_q4ka) [b->pso_mmp_q4ka release];
    if (b->pso_route_b) [b->pso_route_b release]; if (b->pso_gemv_nt) [b->pso_gemv_nt release]; if (b->pso_state_store) [b->pso_state_store release];
    if (b->pso_hcb_scale) [b->pso_hcb_scale release]; if (b->pso_hcb_mixes) [b->pso_hcb_mixes release]; if (b->pso_hcb_sink) [b->pso_hcb_sink release];
    if (b->pso_hcb_norm) [b->pso_hcb_norm release]; if (b->pso_hcb_post) [b->pso_hcb_post release];
    if (b->pso_mm_iq2a) [b->pso_mm_iq2a release]; if (b->pso_mm_q2ka) [b->pso_mm_q2ka release]; if (b->pso_mm_q4ka) [b->pso_mm_q4ka release];
    if (b->pso_q4_pair) [b->pso_q4_pair release]; if (b->pso_q4_sum6) [b->pso_q4_sum6 release];
    if (b->pso_rms_rows) [b->pso_rms_rows release]; if (b->pso_head_rms_rows) [b->pso_head_rms_rows release]; if (b->pso_rope_rows) [b->pso_rope_rows release]; if (b->pso_fp8_rows) [b->pso_fp8_rows release];
    if (b->pso_mm_map1) [b->pso_mm_map1 release]; if (b->pso_mm_map8) [b->pso_mm_map8 release]; if (b->pso_mm_q8) [b->pso_mm_q8 release];
    if (b->pso_mm_map0) [b->pso_mm_map0 release]; if (b->pso_mm_iq2) [b->pso_mm_iq2 release];
    if (b->pso_mm_q2k) [b->pso_mm_q2k release]; if (b->pso_mm_q4k) [b->pso_mm_q4k release];
    if (b->pso_swiglu_w) [b->pso_swiglu_w release]; if (b->pso_sum_slots6) [b->pso_sum_slots6 release];
    if (b->pso_attn_merge) [b->pso_attn_merge release];
    if (b->pso_slots6)   [b->pso_slots6 release];
    if (b->pso_swiglu)   [b->pso_swiglu release];
    if (b->pso_add)      [b->pso_add release];
    if (b->pso_rms_no_w) [b->pso_rms_no_w release];
    if (b->pso_gemv_f32) [b->pso_gemv_f32 release];
    if (b->pso_sinkhorn) [b->pso_sinkhorn release];
    if (b->pso_hc_sum)   [b->pso_hc_sum release];
    if (b->pso_hc_post)  [b->pso_hc_post release];
    if (b->pso_route)    [b->pso_route release];
    if (b->pso_addr_pair) [b->pso_addr_pair release];
    if (b->pso_addr_sum6) [b->pso_addr_sum6 release];
    if (b->pso_tbl_pair) [b->pso_tbl_pair release];
    if (b->pso_tbl_sum6) [b->pso_tbl_sum6 release];
    if (b->enc_pair)     [b->enc_pair release];
    if (b->enc_down)     [b->enc_down release];
    dsv4_view_cache_clear();
    if (b->pso_rms_w)    [b->pso_rms_w release];
    if (b->pso_head_rms) [b->pso_head_rms release];
    if (b->pso_rope)     [b->pso_rope release];
    for (int i = 64; i < 96; i++) if (b->sx[i]) [b->sx[i] release];
    for (int i = 0; i < 64; i++) {
        if (b->sx[i])  [b->sx[i] release];
        if (b->so[i])  [b->so[i] release];
        if (b->spb[i]) [b->spb[i] release];
    }
    if (b->nc) { for (uint32_t i = 0; i < b->nc_cap; i++) if (b->nc[i].buf) [b->nc[i].buf release]; free(b->nc); }
    if (b->residency) [b->residency release];
    if (b->queue)   [b->queue release];
    if (b->dev)     [b->dev release];
    free(b);
}

void* llm_dsv4_metal_nocopy(LlmDsv4Metal* b, const void* mmap_base, size_t file_off, size_t bytes,
                            size_t* out_view_off) {
    if (!b || !mmap_base) return NULL;
    const size_t page = (size_t)getpagesize();
    uintptr_t addr = (uintptr_t)mmap_base;
    if (addr & (page - 1)) return NULL;
    size_t aligned = file_off & ~(page - 1);
    size_t lead = file_off - aligned;
    size_t view_len = ((lead + bytes + page - 1) / page) * page;
    id<MTLBuffer> buf = [b->dev newBufferWithBytesNoCopy:(void*)(addr + aligned)
                          length:view_len options:MTLResourceStorageModeShared deallocator:nil];
    if (!buf) return NULL;
    if (out_view_off) *out_view_off = lead;
    return (void*)buf;
}

void* llm_dsv4_metal_nocopy_ptr(LlmDsv4Metal* b, const void* ptr, size_t bytes, size_t* out_view_off) {
    if (!b || !ptr) return NULL;
    const size_t page = (size_t)getpagesize();
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t aligned = p & ~(uintptr_t)(page - 1);
    size_t lead = (size_t)(p - aligned);
    size_t view_len = ((lead + bytes + page - 1) / page) * page;
    id<MTLBuffer> buf = [b->dev newBufferWithBytesNoCopy:(void*)aligned
                          length:view_len options:MTLResourceStorageModeShared deallocator:nil];
    if (!buf) {
        fprintf(stderr, "dsv4_metal: nocopy_ptr(%p, lead=%zu, len=%zu) NIL (maxBufferLength=%llu)\n",
                (void*)aligned, lead, view_len, (unsigned long long)[b->dev maxBufferLength]);
        return NULL;
    }
    if (out_view_off) *out_view_off = lead;
    return (void*)buf;
}

void llm_dsv4_metal_release(LlmDsv4Metal* b, void* handle) {
    (void)b;
    if (handle) [(id<MTLBuffer>)handle release];
}

int llm_dsv4_metal_gemv(LlmDsv4Metal* b, void* handle, size_t view_off, int ggml_type,
                        size_t row_bytes, uint32_t n, uint32_t out_dim,
                        const float* x, float* out) {
    if (!b || !handle || !x || !out) return -1;
    if (ggml_type != 16 && ggml_type != 10 && ggml_type != 8) return -1;
    const uint32_t blk_elems = (ggml_type == 8) ? 32u : 256u;
    const uint32_t blk_bytes = (ggml_type == 16) ? 66u : (ggml_type == 10) ? 84u : 34u;
    if ((n % blk_elems) != 0) return -1;
    id<MTLBuffer> W  = (id<MTLBuffer>)handle;
    id<MTLBuffer> X  = [b->dev newBufferWithBytes:x length:(NSUInteger)n * sizeof(float)
                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> O  = [b->dev newBufferWithBytes:out length:(NSUInteger)out_dim * sizeof(float)
                        options:MTLResourceStorageModeShared];
    uint32_t P[7] = { (uint32_t)view_off, (uint32_t)row_bytes, n / blk_elems, out_dim, (uint32_t)ggml_type, blk_bytes, blk_elems };
    id<MTLBuffer> PB = [b->dev newBufferWithBytes:P length:sizeof P options:MTLResourceStorageModeShared];
    if (!X || !O || !PB) { if (X) [X release]; if (O) [O release]; if (PB) [PB release]; return -2; }
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
    [e setBuffer:W  offset:0 atIndex:0];
    [e setBuffer:X  offset:0 atIndex:1];
    [e setBuffer:O  offset:0 atIndex:2];
    [e setBuffer:PB offset:0 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    memcpy(out, [O contents], (size_t)out_dim * sizeof(float));
    [X release]; [O release]; [PB release];
    return 0;
}

/* Batch: una command buffer + un wait. View zero-copy TRANSITORIE.
 * IQ2_XXS/Q2_K -> kernel Q8_K integer-dot (metodo ds4); Q8_0 -> kernel f32. */
int llm_dsv4_metal_gemv_batch(LlmDsv4Metal* b, const LlmDsv4GpuJob* jobs, int n_jobs) {
    if (!b || !jobs || n_jobs <= 0 || n_jobs > 64) return -1;
    const size_t page = (size_t)getpagesize();
    id<MTLBuffer> ws[64], xs[64], os[64], pbs[64];
    for (int i = 0; i < 64; i++) { ws[i] = xs[i] = os[i] = pbs[i] = nil; }
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    int nq = 0, rc = 0;
    for (int i = 0; i < n_jobs; i++) {
        const LlmDsv4GpuJob* j = &jobs[i];
        int t = j->ggml_type;
        uint32_t blk_elems = (t == 8) ? 32u : 256u;
        if ((t != 16 && t != 10 && t != 12 && t != 8) || (j->n % blk_elems) != 0 || j->out_dim == 0 || j->n == 0) { rc = -1; break; }
        const uint8_t* p = (const uint8_t*)j->w + j->byte_off;
        size_t lead;
        ws[i] = dsv4_nocopy_view(b, p, (size_t)j->out_dim * j->row_bytes, &lead);
        if (t == 8) {
            xs[i] = dsv4_scratch(b, &b->sx[i], &b->sx_cap[i], (size_t)j->n * sizeof(float));
            if (xs[i]) memcpy([xs[i] contents], j->x, (size_t)j->n * sizeof(float));
            os[i] = dsv4_scratch(b, &b->so[i], &b->so_cap[i], (size_t)j->out_dim * sizeof(float));
            uint32_t P[7] = { (uint32_t)lead, (uint32_t)j->row_bytes, j->n / 32u, j->out_dim, 8u, 34u, 32u };
            pbs[i] = dsv4_scratch(b, &b->spb[i], &b->spb_cap[i], sizeof P);
            if (pbs[i]) memcpy([pbs[i] contents], P, sizeof P);
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
            if (!ws[i] || !xs[i] || !os[i] || !pbs[i]) { rc = -2; break; }
            [e setBuffer:ws[i] offset:0 atIndex:0];
            [e setBuffer:xs[i] offset:0 atIndex:1];
            [e setBuffer:os[i] offset:0 atIndex:2];
            [e setBuffer:pbs[i] offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(j->out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            uint32_t nblk = j->n / 256u;
            LlmDsv4Q8K* y = (LlmDsv4Q8K*)malloc((size_t)nblk * sizeof(LlmDsv4Q8K));
            llm_dsv4_q8_K_quantize_row(j->x, (int)j->n, y);
            xs[i] = dsv4_scratch(b, &b->sx[i], &b->sx_cap[i], (size_t)nblk * sizeof(LlmDsv4Q8K));
            if (xs[i]) memcpy([xs[i] contents], y, (size_t)nblk * sizeof(LlmDsv4Q8K));
            free(y);
            os[i] = dsv4_scratch(b, &b->so[i], &b->so_cap[i], (size_t)j->out_dim * sizeof(float));
            id<MTLBuffer> PT = dsv4_scratch(b, &b->spt[i], &b->spt_cap[i], (size_t)j->out_dim * nblk * sizeof(float));
            uint32_t P[4] = { (uint32_t)lead, (uint32_t)j->row_bytes, nblk, j->out_dim };
            pbs[i] = dsv4_scratch(b, &b->spb[i], &b->spb_cap[i], sizeof P);
            if (pbs[i]) memcpy([pbs[i] contents], P, sizeof P);
            g_dsv4_dispatchs++; [e setComputePipelineState:(t == 16) ? b->pso_q8k_iq2 : (t == 10) ? b->pso_q8k_q2k : b->pso_q8k_q4k];
            if (!ws[i] || !xs[i] || !os[i] || !pbs[i] || !PT) { rc = -2; break; }
            [e setBuffer:ws[i] offset:0 atIndex:0];
            [e setBuffer:xs[i] offset:0 atIndex:1];
            [e setBuffer:PT offset:0 atIndex:2];
            [e setBuffer:pbs[i] offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)j->out_dim * nblk, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_red_parts];
            [e setBuffer:PT offset:0 atIndex:0]; [e setBuffer:os[i] offset:0 atIndex:1];
            { uint32_t RP[2] = { nblk, j->out_dim }; [e setBytes:RP length:sizeof RP atIndex:2]; }
            [e dispatchThreads:MTLSizeMake(j->out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        nq++;
    }
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    if (cb.error) {
        static int once = 0;
        if (!once) { once = 1; fprintf(stderr, "dsv4_metal: command buffer ERROR: %s\n",
                                       [[cb.error localizedDescription] UTF8String]); }
        rc = -3;
    }
    if (rc == 0)
        for (int i = 0; i < nq; i++)
            memcpy((void*)jobs[i].out, [os[i] contents], (size_t)jobs[i].out_dim * sizeof(float));
    return rc;
}

int llm_dsv4_metal_pair_swiglu(LlmDsv4Metal* b, const void* gate, const void* up,
                               const float* x, float* mid, float route_w, float clamp,
                               uint32_t ne00, uint32_t nb01, uint32_t ne0) {
    if (!b || !b->pso_pair || !gate || !up || !x || !mid) return -1;
    const size_t page = (size_t)getpagesize();
    const size_t gbytes = (size_t)ne0 * (size_t)nb01;
    uintptr_t apg = (uintptr_t)gate & ~(uintptr_t)(page - 1);
    size_t leadg = (size_t)((uintptr_t)gate - apg);
    size_t vlg = ((leadg + gbytes + page - 1) / page) * page;
    uintptr_t apu = (uintptr_t)up & ~(uintptr_t)(page - 1);
    size_t leadu = (size_t)((uintptr_t)up - apu);
    size_t vlu = ((leadu + gbytes + page - 1) / page) * page;
    id<MTLBuffer> BG = [b->dev newBufferWithBytesNoCopy:(void*)apg length:vlg options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> BU = [b->dev newBufferWithBytesNoCopy:(void*)apu length:vlu options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> Y  = [b->dev newBufferWithBytes:x length:(NSUInteger)ne00 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> MID = [b->dev newBufferWithLength:(NSUInteger)ne0 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> RW = [b->dev newBufferWithBytes:&route_w length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> CF = [b->dev newBufferWithBytes:&clamp length:sizeof(float) options:MTLResourceStorageModeShared];
    uint32_t P[6] = { ne00, nb01, ne0, (uint32_t)leadg, (uint32_t)leadu, 0u };
    id<MTLBuffer> PB = [b->dev newBufferWithBytes:P length:sizeof P options:MTLResourceStorageModeShared];
    if (!BG || !BU || !Y || !MID || !RW || !CF || !PB) {        if (BG) [BG release]; if (BU) [BU release]; if (Y) [Y release];
        if (MID) [MID release]; if (RW) [RW release]; if (CF) [CF release]; if (PB) [PB release];
        return -2;
    }
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_pair];
    [e setBuffer:BG offset:0 atIndex:0];
    [e setBuffer:BU offset:0 atIndex:1];
    [e setBuffer:Y  offset:0 atIndex:2];
    [e setBuffer:MID offset:0 atIndex:3];
    [e setBuffer:RW offset:0 atIndex:4];
    [e setBuffer:CF offset:0 atIndex:5];
    [e setBuffer:PB offset:0 atIndex:6];
    [e setThreadgroupMemoryLength:256 * 8 + 128 atIndex:0];
    const uint32_t NSG = 4, nr0 = 4;
    [e dispatchThreadgroups:MTLSizeMake((ne0 + NSG * nr0 - 1) / (NSG * nr0), 1, 1)
     threadsPerThreadgroup:MTLSizeMake(32 * NSG, 1, 1)];
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    int rc = cb.error ? -3 : 0;
    if (rc == 0) memcpy(mid, [MID contents], (size_t)ne0 * sizeof(float));
    [BG release]; [BU release]; [Y release]; [MID release]; [RW release]; [CF release]; [PB release];
    return rc;
}

/* Batch dei pair: `n` esperti (gate/up diversi) in UNA command buffer + UN wait (readback differito).
 * Riusa il kernel verificato `dsv4_pair_swiglu_iq2`. Sostituisce `n` commit+wait con 1. */
int llm_dsv4_metal_pair_swiglu_batch(LlmDsv4Metal* b, const void* const* gates, const void* const* ups,
                                     const float* x, float* const* mids, const float* rws, int n,
                                     float clamp, uint32_t ne00, uint32_t nb01, uint32_t ne0) {
    if (!b || !b->pso_pair || n <= 0 || n > 16 || !gates || !ups || !mids || !rws) return -1;
    const size_t page = (size_t)getpagesize();
    const size_t gbytes = (size_t)ne0 * (size_t)nb01;
    id<MTLBuffer> BG[16], BU[16], MID[16], RW[16], CF[16], PB[16];
    for (int i = 0; i < 16; i++) { BG[i]=BU[i]=MID[i]=RW[i]=CF[i]=PB[i]=nil; }
    id<MTLBuffer> Y = [b->dev newBufferWithBytes:x length:(NSUInteger)ne00*sizeof(float) options:MTLResourceStorageModeShared];
    int rc = 0;
    for (int i = 0; i < n && rc == 0; i++) {
        uintptr_t apg = (uintptr_t)gates[i] & ~(uintptr_t)(page-1);
        size_t leadg = (size_t)((uintptr_t)gates[i] - apg);
        uintptr_t apu = (uintptr_t)ups[i] & ~(uintptr_t)(page-1);
        size_t leadu = (size_t)((uintptr_t)ups[i] - apu);
        size_t vlg = ((leadg+gbytes+page-1)/page)*page;
        size_t vlu = ((leadu+gbytes+page-1)/page)*page;
        BG[i] = [b->dev newBufferWithBytesNoCopy:(void*)apg length:vlg options:MTLResourceStorageModeShared deallocator:nil];
        BU[i] = [b->dev newBufferWithBytesNoCopy:(void*)apu length:vlu options:MTLResourceStorageModeShared deallocator:nil];
        MID[i] = [b->dev newBufferWithLength:(NSUInteger)ne0*sizeof(float) options:MTLResourceStorageModeShared];
        RW[i] = [b->dev newBufferWithBytes:&rws[i] length:sizeof(float) options:MTLResourceStorageModeShared];
        CF[i] = [b->dev newBufferWithBytes:&clamp length:sizeof(float) options:MTLResourceStorageModeShared];
        uint32_t P[6] = { ne00, nb01, ne0, (uint32_t)leadg, (uint32_t)leadu, 0u };
        PB[i] = [b->dev newBufferWithBytes:P length:sizeof P options:MTLResourceStorageModeShared];
        if (!BG[i]||!BU[i]||!MID[i]||!RW[i]||!CF[i]||!PB[i]||!Y) rc = -2;
    }
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const uint32_t NSG = 4, nr0 = 4;
        for (int i = 0; i < n; i++) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_pair];
            [e setBuffer:BG[i] offset:0 atIndex:0];
            [e setBuffer:BU[i] offset:0 atIndex:1];
            [e setBuffer:Y offset:0 atIndex:2];
            [e setBuffer:MID[i] offset:0 atIndex:3];
            [e setBuffer:RW[i] offset:0 atIndex:4];
            [e setBuffer:CF[i] offset:0 atIndex:5];
            [e setBuffer:PB[i] offset:0 atIndex:6];
            [e setThreadgroupMemoryLength:256 * 8 + 128 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((ne0 + NSG*nr0 - 1)/(NSG*nr0), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
        }
        [e endEncoding];
        dsv4_res_commit(b); [cb commit];
        [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        if (cb.error) rc = -3;
        else for (int i = 0; i < n; i++) memcpy(mids[i], [MID[i] contents], (size_t)ne0*sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        if (BG[i]) [BG[i] release]; if (BU[i]) [BU[i] release]; if (MID[i]) [MID[i] release];
        if (RW[i]) [RW[i] release]; if (CF[i]) [CF[i] release]; if (PB[i]) [PB[i] release];
    }
    if (Y) [Y release];
    return rc;
}

/* MoE Q4_K per `nt` token in UNA command buffer (kernel ds4 slots6 pair_swiglu + slots6 sum6, mul_mv Q4_K a simdgroup, nsg=2):
 * per token 6 esperti (gate/up/down Q4_K, buffer per-esperto dello store), attivazioni f32, deterministico (niente atomic).
 * Stessa interfaccia di moe_iq2_q2k_batch. Gli scarti dalla pagina dei pesi devono essere multipli di 4 (sono setBuffer offset). */
typedef struct { int32_t nei0, nei1; uint64_t nbi1; int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02; int32_t ne10, ne11, ne12, ne13;
                 uint64_t nb10, nb11, nb12; int32_t ne0, ne1; uint64_t nb1; int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base; } Dsv4MvIdArgs;
typedef struct { uint32_t width, rows; uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride; uint32_t write_clamped; float clamp_value; } Dsv4SwiWArgs;
int llm_dsv4_metal_moe_q4k_batch(LlmDsv4Metal* b, int nt, const void* const* gates, const void* const* ups,
                               const void* const* downs, const float* x, const float* rws, int n,
                               float clamp, uint32_t embd, uint32_t gu_nb01, uint32_t ff,
                               uint32_t down_nb01, const void* sg, const void* su, const void* sd,
                               uint32_t s_nb_g, uint32_t s_nb_d, float* out) {
    if (!b || !b->pso_q4_pair || !b->pso_q4_sum6 || n != 6 || nt <= 0 || !gates || !ups || !downs || !x || !rws || !out) return -1;
    if ((ff % 256u) != 0u || (embd % 256u) != 0u || (ff % 4u) || (embd % 4u)) return -1;
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[0], &b->sx_cap[0], (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> MID = dsv4_scratch(b, &b->sx[1], &b->sx_cap[1], (size_t)n * (size_t)ff * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[2], &b->sx_cap[2], (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> GT  = dsv4_scratch(b, &b->sx[24], &b->sx_cap[24], (size_t)n * ff * sizeof(float));
    id<MTLBuffer> UP  = dsv4_scratch(b, &b->sx[25], &b->sx_cap[25], (size_t)n * ff * sizeof(float));
    id<MTLBuffer> RW  = dsv4_scratch(b, &b->sx[22], &b->sx_cap[22], (size_t)nt * n * sizeof(float));
    if (!X || !MID || !OUT || !GT || !UP || !RW) return -2;
    memcpy([X contents], x, (size_t)nt * embd * sizeof(float));
    memcpy([RW contents], rws, (size_t)nt * n * sizeof(float));
    const size_t gbytes = (size_t)ff * gu_nb01, dbytes = (size_t)embd * down_nb01;
    id<MTLBuffer> WG = nil, WU = nil, WD = nil, G = nil, U = nil, MID2 = nil, D = nil;
    uint32_t lg_g = 0, lu_g = 0, ld_g = 0;
    if (sg && su && sd) {
        size_t sbg = (size_t)ff * s_nb_g, sbd = (size_t)embd * s_nb_d, l0, l1, l2;
        WG = dsv4_nocopy_view(b, sg, sbg, &l0);
        WU = dsv4_nocopy_view(b, su, sbg, &l1);
        WD = dsv4_nocopy_view(b, sd, sbd, &l2);
        lg_g = (uint32_t)l0; lu_g = (uint32_t)l1; ld_g = (uint32_t)l2;
        G    = dsv4_scratch(b, &b->sx[15], &b->sx_cap[15], (size_t)ff * sizeof(float));
        U    = dsv4_scratch(b, &b->sx[16], &b->sx_cap[16], (size_t)ff * sizeof(float));
        MID2 = dsv4_scratch(b, &b->sx[17], &b->sx_cap[17], (size_t)ff * sizeof(float));
        D    = dsv4_scratch(b, &b->sx[18], &b->sx_cap[18], (size_t)embd * sizeof(float));
        if (!WG || !WU || !WD || !G || !U || !MID2 || !D) return -2;
    }
    Dsv4MvIdArgs pa; memset(&pa, 0, sizeof pa);   /* pair: gate/up [ff righe x embd] */
    pa.nei0 = 6; pa.nei1 = 1; pa.ne00 = (int32_t)embd; pa.ne01 = (int32_t)ff; pa.ne02 = 1; pa.nb00 = 144; pa.nb01 = gu_nb01; pa.nb02 = (uint64_t)ff * gu_nb01;
    pa.ne10 = (int32_t)embd; pa.ne11 = 1; pa.ne12 = 1; pa.ne13 = 1; pa.nb10 = 4; pa.nb11 = (uint64_t)embd * 4; pa.nb12 = pa.nb11;
    pa.ne0 = (int32_t)ff; pa.ne1 = 6; pa.nb1 = (uint64_t)ff * 4; pa.nr0 = 2;
    Dsv4SwiWArgs act = { ff, 6, (uint64_t)ff * 4, (uint64_t)ff * 4, (uint64_t)ff * 4, 4, 0, clamp };
    Dsv4MvIdArgs sa; memset(&sa, 0, sizeof sa);   /* down + sum6: [embd righe x ff] */
    sa.nei0 = 6; sa.nei1 = 1; sa.ne00 = (int32_t)ff; sa.ne01 = (int32_t)embd; sa.ne02 = 1; sa.nb00 = 144; sa.nb01 = down_nb01; sa.nb02 = (uint64_t)embd * down_nb01;
    sa.ne10 = (int32_t)ff; sa.ne11 = 6; sa.ne12 = 1; sa.ne13 = 1; sa.nb10 = 4; sa.nb11 = (uint64_t)ff * 4; sa.nb12 = 6 * sa.nb11;
    sa.ne0 = (int32_t)embd; sa.ne1 = 1; sa.nb1 = (uint64_t)embd * 4; sa.nr0 = 2;
    int rc = 0;
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const uint32_t NSG = 2, nr0 = 2;
    for (int t = 0; t < nt && rc == 0; t++) {
        id<MTLBuffer> BG[6], BU[6], BD[6]; size_t LG[6], LU[6], LD[6];
        for (int i = 0; i < n; i++) {
            const int k = t * n + i;
            BG[i] = dsv4_nocopy_view(b, gates[k], gbytes, &LG[i]);
            BU[i] = dsv4_nocopy_view(b, ups[k],   gbytes, &LU[i]);
            BD[i] = dsv4_nocopy_view(b, downs[k], dbytes, &LD[i]);
            if (!BG[i] || !BU[i] || !BD[i] || (LG[i] & 3u) || (LU[i] & 3u) || (LD[i] & 3u)) { rc = -2; break; }
        }
        if (rc != 0) break;
        const NSUInteger xoff = (NSUInteger)((size_t)t * embd * sizeof(float));
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q4_pair];
        [e setBytes:&pa length:sizeof pa atIndex:0];
        [e setBytes:&act length:sizeof act atIndex:1];
        for (int i = 0; i < 6; i++) { [e setBuffer:BG[i] offset:(NSUInteger)LG[i] atIndex:2 + i]; [e setBuffer:BU[i] offset:(NSUInteger)LU[i] atIndex:8 + i]; }
        [e setBuffer:X offset:xoff atIndex:14];
        [e setBuffer:GT offset:0 atIndex:15]; [e setBuffer:UP offset:0 atIndex:16]; [e setBuffer:MID offset:0 atIndex:17];
        [e setBuffer:RW offset:(NSUInteger)((size_t)t * n * sizeof(float)) atIndex:18];
        [e dispatchThreadgroups:MTLSizeMake((ff + NSG * nr0 - 1) / (NSG * nr0), 1, 6) threadsPerThreadgroup:MTLSizeMake(32, NSG, 1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q4_sum6];
        [e setBytes:&sa length:sizeof sa atIndex:0];
        for (int i = 0; i < 6; i++) [e setBuffer:BD[i] offset:(NSUInteger)LD[i] atIndex:1 + i];
        [e setBuffer:MID offset:0 atIndex:7];
        [e setBuffer:OUT offset:xoff atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake((embd + NSG * nr0 - 1) / (NSG * nr0), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, NSG, 1)];
        int fu8 = 1; { int v = 1; if (llm_ai_mlx_opt_get_bool("dsv4_fused_q8", &v) == 0) fu8 = v; }
        if (WG) {
            /* shared expert su GPU: gate/up (Q8_0) + swiglu + down + OUT[t] += D */
            if (fu8 && b->pso_pair_swiglu_q8_0) {
                /* FUSO: gate+up+swiglu in UN dispatch (era 3). Stessa aritmetica. */
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_pair_swiglu_q8_0];
                [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:WU offset:0 atIndex:1];
                [e setBuffer:X offset:xoff atIndex:2]; [e setBuffer:MID2 offset:0 atIndex:3];
                { uint32_t P[6]={lg_g,s_nb_g,lu_g,s_nb_g,embd/32u,ff}; [e setBytes:P length:sizeof(P) atIndex:4];
                  float cf[1]={clamp}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
                [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            } else {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:X offset:xoff atIndex:1]; [e setBuffer:G offset:0 atIndex:2];
            { uint32_t P[7]={lg_g,s_nb_g,embd/32u,ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WU offset:0 atIndex:0]; [e setBuffer:X offset:xoff atIndex:1]; [e setBuffer:U offset:0 atIndex:2];
            { uint32_t P[7]={lu_g,s_nb_g,embd/32u,ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu];
            [e setBuffer:G offset:0 atIndex:0]; [e setBuffer:U offset:0 atIndex:1]; [e setBuffer:MID2 offset:0 atIndex:2];
            { uint32_t P[1]={ff}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={clamp}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake((ff+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
            if (fu8 && b->pso_gemv_q8_0_add) {
                /* FUSO: down (Q8_0) + OUT += in UN dispatch (era 2). Stessa aritmetica. */
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_gemv_q8_0_add];
                [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:OUT offset:0 atIndex:2];
                { uint32_t P[5]={ld_g,s_nb_d,ff/32u,embd,(uint32_t)(xoff/4)}; [e setBytes:P length:sizeof(P) atIndex:3]; }
                [e dispatchThreadgroups:MTLSizeMake(embd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            } else {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:D offset:0 atIndex:2];
            { uint32_t P[7]={ld_g,s_nb_d,ff/32u,embd,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(embd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_add];
            [e setBuffer:OUT offset:xoff atIndex:0]; [e setBuffer:D offset:0 atIndex:1];
            { uint32_t P[1]={embd}; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreadgroups:MTLSizeMake((embd+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
        }
    }
    [e endEncoding];
    if (rc != 0) return rc;
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    rc = cb.error ? -3 : 0;
    if (rc == 0) memcpy(out, [OUT contents], (size_t)nt * embd * sizeof(float));
    return rc;
}

/* MoE ds4 FUSO per `nt` token in UNA command buffer (gate/up IQ2_XXS + down Q2_K, `n`=6 esperti/token).
 * Per ogni token: pair_swiglu (gate+up+swiglu*peso -> mid f32 su device) + sum6 (down dei 6 + somma) +
 * shared expert Q8_0. I dispatch sono in ordine (encoder seriale): MID e i buffer del shared expert sono
 * condivisi fra i token. Puntatori/pesi: gates/ups/downs [nt*6], x [nt*embd], rws [nt*6], out [nt*embd].
 * Un solo commit+wait per l'intero chunk, nessun round-trip CPU. 0 = ok. */
int llm_dsv4_metal_moe_iq2_q2k_batch(LlmDsv4Metal* b, int nt, const void* const* gates, const void* const* ups,
                               const void* const* downs, const float* x, const float* rws, int n,
                               float clamp, uint32_t embd, uint32_t gu_nb01, uint32_t ff,
                               uint32_t down_nb01, const void* sg, const void* su, const void* sd,
                               uint32_t s_nb_g, uint32_t s_nb_d, float* out) {
    if (!b || !b->pso_slots6 || !b->pso_sum6_q2k || n != 6 || nt <= 0 || !gates || !ups || !downs || !x || !rws || !out) return -1;
    if ((ff % 256u) != 0u) return -1;
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[0], &b->sx_cap[0], (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> MID = dsv4_scratch(b, &b->sx[1], &b->sx_cap[1], (size_t)n * (size_t)ff * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[2], &b->sx_cap[2], (size_t)nt * embd * sizeof(float));
    if (!X || !MID || !OUT) return -2;
    memcpy([X contents], x, (size_t)nt * embd * sizeof(float));
    const size_t gbytes = (size_t)ff * gu_nb01, dbytes = (size_t)embd * down_nb01;
    id<MTLBuffer> WG = nil, WU = nil, WD = nil, G = nil, U = nil, MID2 = nil, D = nil;
    uint32_t lg_g = 0, lu_g = 0, ld_g = 0;
    if (sg && su && sd) {
        size_t sbg = (size_t)ff * s_nb_g, sbd = (size_t)embd * s_nb_d, l0, l1, l2;
        WG = dsv4_nocopy_view(b, sg, sbg, &l0);
        WU = dsv4_nocopy_view(b, su, sbg, &l1);
        WD = dsv4_nocopy_view(b, sd, sbd, &l2);
        lg_g = (uint32_t)l0; lu_g = (uint32_t)l1; ld_g = (uint32_t)l2;
        G    = dsv4_scratch(b, &b->sx[15], &b->sx_cap[15], (size_t)ff * sizeof(float));
        U    = dsv4_scratch(b, &b->sx[16], &b->sx_cap[16], (size_t)ff * sizeof(float));
        MID2 = dsv4_scratch(b, &b->sx[17], &b->sx_cap[17], (size_t)ff * sizeof(float));
        D    = dsv4_scratch(b, &b->sx[18], &b->sx_cap[18], (size_t)embd * sizeof(float));
        if (!WG || !WU || !WD || !G || !U || !MID2 || !D) return -2;
    }
    const uint32_t PS[4] = { ff / 256u, down_nb01, embd, ff };
    int rc = 0;
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const uint32_t NSG = 4, nr0 = 4;
    for (int t = 0; t < nt && rc == 0; t++) {
        id<MTLBuffer> BG[6], BU[6], BD[6]; size_t LD[6];
        uint32_t PV[15]; PV[0] = embd; PV[1] = gu_nb01; PV[2] = ff;
        for (int i = 0; i < n; i++) {
            size_t lg, lu, ld; const int k = t * n + i;
            BG[i] = dsv4_nocopy_view(b, gates[k], gbytes, &lg);
            BU[i] = dsv4_nocopy_view(b, ups[k],   gbytes, &lu);
            BD[i] = dsv4_nocopy_view(b, downs[k], dbytes, &ld);
            if (!BG[i] || !BU[i] || !BD[i]) { rc = -2; break; }
            PV[3+i] = (uint32_t)lg; PV[9+i] = (uint32_t)lu; LD[i] = ld;
        }
        if (rc != 0) break;
        const NSUInteger xoff = (NSUInteger)((size_t)t * embd * sizeof(float));
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_slots6];
        for (int i = 0; i < 6; i++) [e setBuffer:BG[i] offset:0 atIndex:i];
        for (int i = 0; i < 6; i++) [e setBuffer:BU[i] offset:0 atIndex:6+i];
        [e setBuffer:X   offset:xoff atIndex:12];
        [e setBuffer:MID offset:0 atIndex:13];
        [e setBytes:rws + (size_t)t * n length:(NSUInteger)n*sizeof(float) atIndex:14];
        [e setBytes:&clamp length:sizeof(float) atIndex:15];
        [e setBytes:PV length:sizeof(PV) atIndex:16];
        [e setThreadgroupMemoryLength:256 * 8 + 128 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((ff + NSG*nr0 - 1) / (NSG*nr0), 1, 6)
         threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_sum6_q2k];
        for (int i = 0; i < 6; i++) [e setBuffer:BD[i] offset:(NSUInteger)LD[i] atIndex:i];   /* scarto dalla pagina */
        [e setBuffer:MID offset:0 atIndex:6];
        [e setBuffer:OUT offset:xoff atIndex:7];
        [e setBytes:PS length:sizeof(PS) atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake((embd + NSG*nr0 - 1) / (NSG*nr0), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
        int fu8 = 1; { int v = 1; if (llm_ai_mlx_opt_get_bool("dsv4_fused_q8", &v) == 0) fu8 = v; }
        if (WG) {
            /* shared expert su GPU: gate/up (Q8_0) + swiglu + down + OUT[t] += D */
            if (fu8 && b->pso_pair_swiglu_q8_0) {
                /* FUSO: gate+up+swiglu in UN dispatch (era 3). Stessa aritmetica. */
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_pair_swiglu_q8_0];
                [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:WU offset:0 atIndex:1];
                [e setBuffer:X offset:xoff atIndex:2]; [e setBuffer:MID2 offset:0 atIndex:3];
                { uint32_t P[6]={lg_g,s_nb_g,lu_g,s_nb_g,embd/32u,ff}; [e setBytes:P length:sizeof(P) atIndex:4];
                  float cf[1]={clamp}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
                [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            } else {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:X offset:xoff atIndex:1]; [e setBuffer:G offset:0 atIndex:2];
            { uint32_t P[7]={lg_g,s_nb_g,embd/32u,ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WU offset:0 atIndex:0]; [e setBuffer:X offset:xoff atIndex:1]; [e setBuffer:U offset:0 atIndex:2];
            { uint32_t P[7]={lu_g,s_nb_g,embd/32u,ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu];
            [e setBuffer:G offset:0 atIndex:0]; [e setBuffer:U offset:0 atIndex:1]; [e setBuffer:MID2 offset:0 atIndex:2];
            { uint32_t P[1]={ff}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={clamp}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake((ff+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
            if (fu8 && b->pso_gemv_q8_0_add) {
                /* FUSO: down (Q8_0) + OUT += in UN dispatch (era 2). Stessa aritmetica. */
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_gemv_q8_0_add];
                [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:OUT offset:0 atIndex:2];
                { uint32_t P[5]={ld_g,s_nb_d,ff/32u,embd,(uint32_t)(xoff/4)}; [e setBytes:P length:sizeof(P) atIndex:3]; }
                [e dispatchThreadgroups:MTLSizeMake(embd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            } else {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:D offset:0 atIndex:2];
            { uint32_t P[7]={ld_g,s_nb_d,ff/32u,embd,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(embd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_add];
            [e setBuffer:OUT offset:xoff atIndex:0]; [e setBuffer:D offset:0 atIndex:1];
            { uint32_t P[1]={embd}; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreadgroups:MTLSizeMake((embd+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
        }
    }
    [e endEncoding];
    if (rc != 0) return rc;   /* encoder chiuso, cb mai committata (rilasciata da ARC/pool) */
    dsv4_res_commit(b);
    [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    rc = cb.error ? -3 : 0;
    if (rc == 0) memcpy(out, [OUT contents], (size_t)nt * embd * sizeof(float));
    return rc;
}

/* MoE ds4 FUSO per un token (wrapper del batch). */
int llm_dsv4_metal_moe_iq2_q2k(LlmDsv4Metal* b, const void* const* gates, const void* const* ups,
                               const void* const* downs, const float* x, const float* rws, int n,
                               float clamp, uint32_t embd, uint32_t gu_nb01, uint32_t ff,
                               uint32_t down_nb01, const void* sg, const void* su, const void* sd,
                               uint32_t s_nb_g, uint32_t s_nb_d, float* out) {
    return llm_dsv4_metal_moe_iq2_q2k_batch(b, 1, gates, ups, downs, x, rws, n, clamp, embd, gu_nb01, ff,
                                            down_nb01, sg, su, sd, s_nb_g, s_nb_d, out);
}

/* ── MoE BATCHED via mul_mm_id (kernel estratti da ds4): map0 + gate + up + swiglu*peso + down + somma sui 6 slot,
 * tutto in UNA command buffer per `nt` token. I pesi sono i tensori del layer con gli esperti impilati
 * (stride costante): gate/up [n_expert][ff][embd] e down [n_expert][embd][ff]. Solo routed (niente shared). ── */
typedef struct { int32_t ne02, ne10, ne11; uint64_t nb11, nb12; int32_t ne21, ne20; uint64_t nb21; } Dsv4MmMapArgs;
typedef struct { int32_t ne00, ne02; uint64_t nb01, nb02, nb03; int32_t ne11; uint64_t nb10, nb11, nb12, nb13;
                 int32_t ne20, ne21, ne0, ne1; int16_t r2, r3; int32_t tp_rank, tp_world, tp_expert_base; } Dsv4MmArgs;

static int dsv4_use_mpp(void);
static id<MTLComputePipelineState> dsv4_pso_q8(LlmDsv4Metal* b);
static int dsv4_use_mpp(void) { int v = 1; if (llm_ai_mlx_opt_get_bool("dsv4_mpp", &v) != 0) v = 1; return v; }
static id<MTLComputePipelineState> dsv4_pso_q8(LlmDsv4Metal* b) { return (dsv4_use_mpp() && b->pso_mmp_q8) ? b->pso_mmp_q8 : b->pso_mm_q8; }
static id<MTLComputePipelineState> dsv4_mm_pso_addr(LlmDsv4Metal* b, int ggml_type) {
    if (dsv4_use_mpp()) { id<MTLComputePipelineState> m = ggml_type == 16 ? b->pso_mmp_iq2a : ggml_type == 10 ? b->pso_mmp_q2ka : ggml_type == 12 ? b->pso_mmp_q4ka : nil; if (m) return m; }
    return ggml_type == 16 ? b->pso_mm_iq2a : ggml_type == 10 ? b->pso_mm_q2ka : ggml_type == 12 ? b->pso_mm_q4ka : nil;
}
static id<MTLComputePipelineState> dsv4_mm_pso(LlmDsv4Metal* b, int ggml_type) {
    if (dsv4_use_mpp()) { id<MTLComputePipelineState> m = ggml_type == 16 ? b->pso_mmp_iq2 : ggml_type == 10 ? b->pso_mmp_q2k : ggml_type == 12 ? b->pso_mmp_q4k : nil; if (m) return m; }
    return ggml_type == 16 ? b->pso_mm_iq2 : ggml_type == 10 ? b->pso_mm_q2k : ggml_type == 12 ? b->pso_mm_q4k : nil;
}

static void dsv4_encode_mm_id(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pso, const Dsv4MmArgs* a,
        id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst,
        id<MTLBuffer> map, NSUInteger tpe_bytes, NSUInteger work_off, NSUInteger work_cap) {
    g_dsv4_dispatchs++; [e setComputePipelineState:pso];
    [e setBytes:a length:sizeof(*a) atIndex:0];
    [e setBuffer:src0 offset:src0_off atIndex:1];
    [e setBuffer:src1 offset:src1_off atIndex:2];
    [e setBuffer:map offset:0 atIndex:3];
    [e setBuffer:map offset:tpe_bytes atIndex:4];
    [e setBuffer:dst offset:0 atIndex:5];
    [e setBuffer:map offset:work_off atIndex:6];
    [e setThreadgroupMemoryLength:12288 atIndex:0];   /* 8192 bastano al kernel simdgroup; il kernel MPP (ds4: 12288) puo' staged di piu' */
    [e dispatchThreadgroups:MTLSizeMake(work_cap, ((NSUInteger)a->ne0 + 63u) / 64u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

/* map0 + mul_mm_id in un colpo: `ne02` esperti, `ne20` selezioni/token, `nt` token. ids [nt*ne20] su IDS. `MAP` scratch. */
static void dsv4_encode_mm_id_full(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pmap, id<MTLComputePipelineState> pmm,
        const Dsv4MmMapArgs* ma, const Dsv4MmArgs* a, id<MTLBuffer> IDS, id<MTLBuffer> MAP,
        id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst) {
    const NSUInteger n_expert = (NSUInteger)a->ne02, pairs = (NSUInteger)a->ne20 * (NSUInteger)a->ne21;
    const NSUInteger tpe_bytes = n_expert * 4u, hids_bytes = n_expert * (NSUInteger)a->ne21 * 4u;
    const NSUInteger work_off = (tpe_bytes + hids_bytes + 7u) & ~(NSUInteger)7u;
    const NSUInteger work_cap = (pairs + 31u * n_expert + 31u) / 32u;
    g_dsv4_dispatchs++; [e setComputePipelineState:pmap];
    [e setBytes:ma length:sizeof(*ma) atIndex:0];
    [e setBuffer:IDS offset:0 atIndex:1]; [e setBuffer:MAP offset:0 atIndex:2];
    [e setBuffer:MAP offset:tpe_bytes atIndex:3]; [e setBuffer:MAP offset:work_off atIndex:4];
    [e setThreadgroupMemoryLength:MAX(n_expert * (NSUInteger)a->ne20 * 2u, (NSUInteger)32) atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(n_expert, 1, 1)];
    dsv4_encode_mm_id(e, pmm, a, src0, src0_off, src1, src1_off, dst, MAP, tpe_bytes, work_off, work_cap);
}
static size_t dsv4_mm_map_bytes(uint32_t n_expert, int nt, uint32_t ne20) {
    const size_t tpe = (size_t)n_expert * 4, hids = (size_t)n_expert * nt * 4, wo = (tpe + hids + 7) & ~(size_t)7;
    return wo + 8 + (((size_t)nt * ne20 + 31u * n_expert + 31u) / 32u) * 8;
}

int llm_dsv4_metal_moe_mm_batch(LlmDsv4Metal* b, int nt, const void* gate_w, const void* up_w, const void* down_w,
        int gtype, int dtype, uint32_t n_expert, uint32_t embd, uint32_t ff, uint32_t gu_row_bytes, uint32_t down_row_bytes,
        const int* ids, const float* rws, const float* x, float clamp,
        const void* sg, const void* su, const void* sd, uint32_t s_nb_g, uint32_t s_nb_d, float* out) {
    if (!b || nt <= 0 || nt > 4096 || !gate_w || !up_w || !down_w || !ids || !rws || !x || !out) return -1;
    id<MTLComputePipelineState> pg = dsv4_mm_pso(b, gtype), pd = dsv4_mm_pso(b, dtype);
    if (!b->pso_mm_map0 || !pg || !pd || !b->pso_swiglu_w || !b->pso_sum_slots6 || n_expert > 1024u) return -1;
    if ((ff % 64u) != 0u || (embd % 64u) != 0u) return -1;
    const uint32_t ne20 = 6;
    const NSUInteger pairs = (NSUInteger)nt * ne20;
    const NSUInteger tpe_bytes = (NSUInteger)n_expert * sizeof(int32_t), hids_bytes = (NSUInteger)n_expert * nt * sizeof(int32_t);
    const NSUInteger work_off = (tpe_bytes + hids_bytes + 7u) & ~(NSUInteger)7u;
    const NSUInteger work_cap = (pairs + 31u * n_expert + 31u) / 32u;
    id<MTLBuffer> IDS = dsv4_scratch(b, &b->sx[20], &b->sx_cap[20], pairs * sizeof(int32_t));
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[21], &b->sx_cap[21], (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> RW  = dsv4_scratch(b, &b->sx[22], &b->sx_cap[22], pairs * sizeof(float));
    id<MTLBuffer> MAP = dsv4_scratch(b, &b->sx[23], &b->sx_cap[23], work_off + 8u + work_cap * 8u);
    id<MTLBuffer> GT  = dsv4_scratch(b, &b->sx[24], &b->sx_cap[24], pairs * ff * sizeof(float));
    id<MTLBuffer> UP  = dsv4_scratch(b, &b->sx[25], &b->sx_cap[25], pairs * ff * sizeof(float));
    id<MTLBuffer> MID = dsv4_scratch(b, &b->sx[26], &b->sx_cap[26], pairs * ff * sizeof(float));
    id<MTLBuffer> DN  = dsv4_scratch(b, &b->sx[27], &b->sx_cap[27], pairs * embd * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[28], &b->sx_cap[28], (size_t)nt * embd * sizeof(float));
    size_t lg, lu, ld;
    id<MTLBuffer> WG = dsv4_nocopy_view(b, gate_w, (size_t)n_expert * ff * gu_row_bytes, &lg);
    id<MTLBuffer> WU = dsv4_nocopy_view(b, up_w,   (size_t)n_expert * ff * gu_row_bytes, &lu);
    id<MTLBuffer> WD = dsv4_nocopy_view(b, down_w, (size_t)n_expert * embd * down_row_bytes, &ld);
    if (!IDS || !X || !RW || !MAP || !GT || !UP || !MID || !DN || !OUT || !WG || !WU || !WD) return -2;
    memcpy([IDS contents], ids, pairs * sizeof(int32_t));
    memcpy([X contents], x, (size_t)nt * embd * sizeof(float));
    memcpy([RW contents], rws, pairs * sizeof(float));
    const int hasShared = (sg && su && sd);
    id<MTLBuffer> IDB = nil, MAPS = nil, SWG = nil, SWU = nil, SWD = nil, SG_ = nil, SU_ = nil, SMID = nil, SD_ = nil, ONES = nil;
    size_t slg = 0, slu = 0, sld = 0;
    Dsv4MmMapArgs mh; Dsv4MmArgs sga, sda, mhd_dummy; memset(&mh, 0, sizeof mh); memset(&sga, 0, sizeof sga); memset(&sda, 0, sizeof sda); memset(&mhd_dummy, 0, sizeof mhd_dummy);
    Dsv4MmMapArgs mhd; memset(&mhd, 0, sizeof mhd);
    if (hasShared) {
        IDB  = dsv4_scratch(b, &b->sx[35], &b->sx_cap[35], (size_t)nt * 4);
        MAPS = dsv4_scratch(b, &b->sx[36], &b->sx_cap[36], dsv4_mm_map_bytes(1, nt, 1));
        SG_  = dsv4_scratch(b, &b->sx[37], &b->sx_cap[37], (size_t)nt * ff * sizeof(float));
        SU_  = dsv4_scratch(b, &b->sx[38], &b->sx_cap[38], (size_t)nt * ff * sizeof(float));
        SMID = dsv4_scratch(b, &b->sx[39], &b->sx_cap[39], (size_t)nt * ff * sizeof(float));
        SD_  = dsv4_scratch(b, &b->sx[49], &b->sx_cap[49], (size_t)nt * embd * sizeof(float));
        ONES = dsv4_scratch(b, &b->sx[50], &b->sx_cap[50], (size_t)nt * sizeof(float));
        SWG = dsv4_nocopy_view(b, sg, (size_t)ff * s_nb_g, &slg);
        SWU = dsv4_nocopy_view(b, su, (size_t)ff * s_nb_g, &slu);
        SWD = dsv4_nocopy_view(b, sd, (size_t)embd * s_nb_d, &sld);
        if (!IDB || !MAPS || !SG_ || !SU_ || !SMID || !SD_ || !ONES || !SWG || !SWU || !SWD) return -2;
        memset([IDB contents], 0, (size_t)nt * 4); { float* o = (float*)[ONES contents]; for (int t = 0; t < nt; t++) o[t] = 1.0f; }
        mh = (Dsv4MmMapArgs){ 1, (int32_t)embd, 1, (uint64_t)embd * 4, (uint64_t)embd * 4, nt, 1, 4 };
        mhd = (Dsv4MmMapArgs){ 1, (int32_t)ff, 1, (uint64_t)ff * 4, (uint64_t)ff * 4, nt, 1, 4 };
        sga.ne00 = (int32_t)embd; sga.ne02 = 1; sga.nb01 = s_nb_g; sga.nb02 = (uint64_t)ff * s_nb_g; sga.nb03 = sga.nb02; sga.ne11 = 1; sga.nb10 = 4;
        sga.nb11 = (uint64_t)embd * 4; sga.nb12 = sga.nb11; sga.nb13 = (uint64_t)nt * sga.nb12; sga.ne20 = 1; sga.ne21 = nt; sga.ne0 = (int32_t)ff; sga.ne1 = 1; sga.r2 = 1; sga.r3 = 1;
        sda.ne00 = (int32_t)ff; sda.ne02 = 1; sda.nb01 = s_nb_d; sda.nb02 = (uint64_t)embd * s_nb_d; sda.nb03 = sda.nb02; sda.ne11 = 1; sda.nb10 = 4;
        sda.nb11 = (uint64_t)ff * 4; sda.nb12 = sda.nb11; sda.nb13 = (uint64_t)nt * sda.nb12; sda.ne20 = 1; sda.ne21 = nt; sda.ne0 = (int32_t)embd; sda.ne1 = 1; sda.r2 = 1; sda.r3 = 1;
    }

    Dsv4MmMapArgs ma = { (int32_t)n_expert, (int32_t)embd, 1, (uint64_t)embd * 4, (uint64_t)embd * 4, nt, (int32_t)ne20, (uint64_t)ne20 * 4 };
    Dsv4MmArgs ga; memset(&ga, 0, sizeof ga);
    ga.ne00 = (int32_t)embd; ga.ne02 = (int32_t)n_expert; ga.nb01 = gu_row_bytes; ga.nb02 = (uint64_t)ff * gu_row_bytes;
    ga.nb03 = (uint64_t)n_expert * ga.nb02; ga.ne11 = 1; ga.nb10 = 4; ga.nb11 = (uint64_t)embd * 4; ga.nb12 = ga.nb11;
    ga.nb13 = (uint64_t)nt * ga.nb12; ga.ne20 = (int32_t)ne20; ga.ne21 = nt; ga.ne0 = (int32_t)ff; ga.ne1 = (int32_t)ne20; ga.r2 = 1; ga.r3 = 1;
    Dsv4MmArgs da; memset(&da, 0, sizeof da);
    da.ne00 = (int32_t)ff; da.ne02 = (int32_t)n_expert; da.nb01 = down_row_bytes; da.nb02 = (uint64_t)embd * down_row_bytes;
    da.nb03 = (uint64_t)n_expert * da.nb02; da.ne11 = (int32_t)ne20; da.nb10 = 4; da.nb11 = (uint64_t)ff * 4; da.nb12 = ne20 * da.nb11;
    da.nb13 = (uint64_t)nt * da.nb12; da.ne20 = (int32_t)ne20; da.ne21 = nt; da.ne0 = (int32_t)embd; da.ne1 = (int32_t)ne20; da.r2 = 1; da.r3 = 1;

    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_mm_map0];
    [e setBytes:&ma length:sizeof ma atIndex:0];
    [e setBuffer:IDS offset:0 atIndex:1]; [e setBuffer:MAP offset:0 atIndex:2];
    [e setBuffer:MAP offset:tpe_bytes atIndex:3]; [e setBuffer:MAP offset:work_off atIndex:4];
    [e setThreadgroupMemoryLength:(NSUInteger)n_expert * ne20 * 2u atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(n_expert, 1, 1)];
    dsv4_encode_mm_id(e, pg, &ga, WG, lg, X, 0, GT, MAP, tpe_bytes, work_off, work_cap);
    dsv4_encode_mm_id(e, pg, &ga, WU, lu, X, 0, UP, MAP, tpe_bytes, work_off, work_cap);
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu_w];
    [e setBuffer:GT offset:0 atIndex:0]; [e setBuffer:UP offset:0 atIndex:1]; [e setBuffer:RW offset:0 atIndex:2]; [e setBuffer:MID offset:0 atIndex:3];
    { uint32_t P[2] = { ff, (uint32_t)pairs }; [e setBytes:P length:sizeof P atIndex:4]; [e setBytes:&clamp length:sizeof(float) atIndex:5]; }
    [e dispatchThreads:MTLSizeMake(ff, pairs, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    dsv4_encode_mm_id(e, pd, &da, WD, ld, MID, 0, DN, MAP, tpe_bytes, work_off, work_cap);
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_sum_slots6];
    [e setBuffer:DN offset:0 atIndex:0]; [e setBuffer:OUT offset:0 atIndex:1];
    { uint32_t P[2] = { embd, (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:2]; }
    [e dispatchThreads:MTLSizeMake(embd, nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (hasShared) {   /* shared expert Q8_0 batched: gate/up mul_mm_id (1 esperto), swiglu (peso 1), down, somma in OUT */
        dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mh, &sga, IDB, MAPS, SWG, slg, X, 0, SG_);
        dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mh, &sga, IDB, MAPS, SWU, slu, X, 0, SU_);
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu_w];
        [e setBuffer:SG_ offset:0 atIndex:0]; [e setBuffer:SU_ offset:0 atIndex:1]; [e setBuffer:ONES offset:0 atIndex:2]; [e setBuffer:SMID offset:0 atIndex:3];
        { uint32_t P[2] = { ff, (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:4]; [e setBytes:&clamp length:sizeof(float) atIndex:5]; }
        [e dispatchThreads:MTLSizeMake(ff, nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mhd, &sda, IDB, MAPS, SWD, sld, SMID, 0, SD_);
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_add];
        [e setBuffer:OUT offset:0 atIndex:0]; [e setBuffer:SD_ offset:0 atIndex:1];
        { uint32_t P[1] = { embd * (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:2]; }
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)embd * nt + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    if (cb.error) return -3;
    memcpy(out, [OUT contents], (size_t)nt * embd * sizeof(float));
    return 0;
}

int llm_dsv4_metal_moe_mm_addr_batch(LlmDsv4Metal* b, int nt, const void* const* gate_p, const void* const* up_p, const void* const* down_p,
        int gtype, int dtype, uint32_t n_expert, uint32_t embd, uint32_t ff, uint32_t gu_row_bytes, uint32_t down_row_bytes,
        const int* ids, const float* rws, const float* x, float clamp,
        const void* sg, const void* su, const void* sd, uint32_t s_nb_g, uint32_t s_nb_d, float* out) {
    if (!b || nt <= 0 || nt > 4096 || !gate_p || !up_p || !down_p || !ids || !rws || !x || !out) return -1;
    id<MTLComputePipelineState> pg = dsv4_mm_pso_addr(b, gtype), pd = dsv4_mm_pso_addr(b, dtype);
    if (!b->pso_mm_map0 || !pg || !pd || !b->pso_swiglu_w || !b->pso_sum_slots6 || n_expert > 1024u) return -1;
    if ((ff % 64u) != 0u || (embd % 64u) != 0u) return -1;
    const uint32_t ne20 = 6;
    const NSUInteger pairs = (NSUInteger)nt * ne20;
    const NSUInteger tpe_bytes = (NSUInteger)n_expert * sizeof(int32_t), hids_bytes = (NSUInteger)n_expert * nt * sizeof(int32_t);
    const NSUInteger work_off = (tpe_bytes + hids_bytes + 7u) & ~(NSUInteger)7u;
    const NSUInteger work_cap = (pairs + 31u * n_expert + 31u) / 32u;
    id<MTLBuffer> IDS = dsv4_scratch(b, &b->sx[20], &b->sx_cap[20], pairs * sizeof(int32_t));
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[21], &b->sx_cap[21], (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> RW  = dsv4_scratch(b, &b->sx[22], &b->sx_cap[22], pairs * sizeof(float));
    id<MTLBuffer> MAP = dsv4_scratch(b, &b->sx[23], &b->sx_cap[23], work_off + 8u + work_cap * 8u);
    id<MTLBuffer> GT  = dsv4_scratch(b, &b->sx[24], &b->sx_cap[24], pairs * ff * sizeof(float));
    id<MTLBuffer> UP  = dsv4_scratch(b, &b->sx[25], &b->sx_cap[25], pairs * ff * sizeof(float));
    id<MTLBuffer> MID = dsv4_scratch(b, &b->sx[26], &b->sx_cap[26], pairs * ff * sizeof(float));
    id<MTLBuffer> DN  = dsv4_scratch(b, &b->sx[27], &b->sx_cap[27], pairs * embd * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[28], &b->sx_cap[28], (size_t)nt * embd * sizeof(float));
    /* Tabelle di indirizzi GPU per esperto (gate/up/down): gli esperti stanno in buffer separati dello store, tutti nel
     * residency set; useResource li rende accessibili anche dalla cb (come fa ds4). */
    id<MTLBuffer> WG = dsv4_scratch(b, &b->sx[51], &b->sx_cap[51], (size_t)n_expert * 8);
    id<MTLBuffer> WU = dsv4_scratch(b, &b->sx[52], &b->sx_cap[52], (size_t)n_expert * 8);
    id<MTLBuffer> WD = dsv4_scratch(b, &b->sx[53], &b->sx_cap[53], (size_t)n_expert * 8);
    size_t lg = 0, lu = 0, ld = 0;
    NSMutableArray<id<MTLBuffer>>* used_views = [NSMutableArray arrayWithCapacity:(NSUInteger)n_expert * 3];
    if (WG && WU && WD) {
        uint64_t* tg = (uint64_t*)[WG contents]; uint64_t* tu = (uint64_t*)[WU contents]; uint64_t* td = (uint64_t*)[WD contents];
        for (uint32_t ex = 0; ex < n_expert; ex++) {
            tg[ex] = tu[ex] = td[ex] = 0;
            if (!gate_p[ex]) continue;
            size_t l0, l1, l2;
            id<MTLBuffer> v0 = dsv4_nocopy_view(b, gate_p[ex], (size_t)ff * gu_row_bytes, &l0);
            id<MTLBuffer> v1 = dsv4_nocopy_view(b, up_p[ex],   (size_t)ff * gu_row_bytes, &l1);
            id<MTLBuffer> v2 = dsv4_nocopy_view(b, down_p[ex], (size_t)embd * down_row_bytes, &l2);
            if (!v0 || !v1 || !v2) return -2;
            tg[ex] = [v0 gpuAddress] + l0; tu[ex] = [v1 gpuAddress] + l1; td[ex] = [v2 gpuAddress] + l2;
            [used_views addObject:v0]; [used_views addObject:v1]; [used_views addObject:v2];
        }
    }
    if (!IDS || !X || !RW || !MAP || !GT || !UP || !MID || !DN || !OUT || !WG || !WU || !WD) return -2;
    memcpy([IDS contents], ids, pairs * sizeof(int32_t));
    memcpy([X contents], x, (size_t)nt * embd * sizeof(float));
    memcpy([RW contents], rws, pairs * sizeof(float));
    const int hasShared = (sg && su && sd);
    id<MTLBuffer> IDB = nil, MAPS = nil, SWG = nil, SWU = nil, SWD = nil, SG_ = nil, SU_ = nil, SMID = nil, SD_ = nil, ONES = nil;
    size_t slg = 0, slu = 0, sld = 0;
    Dsv4MmMapArgs mh; Dsv4MmArgs sga, sda, mhd_dummy; memset(&mh, 0, sizeof mh); memset(&sga, 0, sizeof sga); memset(&sda, 0, sizeof sda); memset(&mhd_dummy, 0, sizeof mhd_dummy);
    Dsv4MmMapArgs mhd; memset(&mhd, 0, sizeof mhd);
    if (hasShared) {
        IDB  = dsv4_scratch(b, &b->sx[35], &b->sx_cap[35], (size_t)nt * 4);
        MAPS = dsv4_scratch(b, &b->sx[36], &b->sx_cap[36], dsv4_mm_map_bytes(1, nt, 1));
        SG_  = dsv4_scratch(b, &b->sx[37], &b->sx_cap[37], (size_t)nt * ff * sizeof(float));
        SU_  = dsv4_scratch(b, &b->sx[38], &b->sx_cap[38], (size_t)nt * ff * sizeof(float));
        SMID = dsv4_scratch(b, &b->sx[39], &b->sx_cap[39], (size_t)nt * ff * sizeof(float));
        SD_  = dsv4_scratch(b, &b->sx[49], &b->sx_cap[49], (size_t)nt * embd * sizeof(float));
        ONES = dsv4_scratch(b, &b->sx[50], &b->sx_cap[50], (size_t)nt * sizeof(float));
        SWG = dsv4_nocopy_view(b, sg, (size_t)ff * s_nb_g, &slg);
        SWU = dsv4_nocopy_view(b, su, (size_t)ff * s_nb_g, &slu);
        SWD = dsv4_nocopy_view(b, sd, (size_t)embd * s_nb_d, &sld);
        if (!IDB || !MAPS || !SG_ || !SU_ || !SMID || !SD_ || !ONES || !SWG || !SWU || !SWD) return -2;
        memset([IDB contents], 0, (size_t)nt * 4); { float* o = (float*)[ONES contents]; for (int t = 0; t < nt; t++) o[t] = 1.0f; }
        mh = (Dsv4MmMapArgs){ 1, (int32_t)embd, 1, (uint64_t)embd * 4, (uint64_t)embd * 4, nt, 1, 4 };
        mhd = (Dsv4MmMapArgs){ 1, (int32_t)ff, 1, (uint64_t)ff * 4, (uint64_t)ff * 4, nt, 1, 4 };
        sga.ne00 = (int32_t)embd; sga.ne02 = 1; sga.nb01 = s_nb_g; sga.nb02 = (uint64_t)ff * s_nb_g; sga.nb03 = sga.nb02; sga.ne11 = 1; sga.nb10 = 4;
        sga.nb11 = (uint64_t)embd * 4; sga.nb12 = sga.nb11; sga.nb13 = (uint64_t)nt * sga.nb12; sga.ne20 = 1; sga.ne21 = nt; sga.ne0 = (int32_t)ff; sga.ne1 = 1; sga.r2 = 1; sga.r3 = 1;
        sda.ne00 = (int32_t)ff; sda.ne02 = 1; sda.nb01 = s_nb_d; sda.nb02 = (uint64_t)embd * s_nb_d; sda.nb03 = sda.nb02; sda.ne11 = 1; sda.nb10 = 4;
        sda.nb11 = (uint64_t)ff * 4; sda.nb12 = sda.nb11; sda.nb13 = (uint64_t)nt * sda.nb12; sda.ne20 = 1; sda.ne21 = nt; sda.ne0 = (int32_t)embd; sda.ne1 = 1; sda.r2 = 1; sda.r3 = 1;
    }

    Dsv4MmMapArgs ma = { (int32_t)n_expert, (int32_t)embd, 1, (uint64_t)embd * 4, (uint64_t)embd * 4, nt, (int32_t)ne20, (uint64_t)ne20 * 4 };
    Dsv4MmArgs ga; memset(&ga, 0, sizeof ga);
    ga.ne00 = (int32_t)embd; ga.ne02 = (int32_t)n_expert; ga.nb01 = gu_row_bytes; ga.nb02 = (uint64_t)ff * gu_row_bytes;
    ga.nb03 = (uint64_t)n_expert * ga.nb02; ga.ne11 = 1; ga.nb10 = 4; ga.nb11 = (uint64_t)embd * 4; ga.nb12 = ga.nb11;
    ga.nb13 = (uint64_t)nt * ga.nb12; ga.ne20 = (int32_t)ne20; ga.ne21 = nt; ga.ne0 = (int32_t)ff; ga.ne1 = (int32_t)ne20; ga.r2 = 1; ga.r3 = 1;
    Dsv4MmArgs da; memset(&da, 0, sizeof da);
    da.ne00 = (int32_t)ff; da.ne02 = (int32_t)n_expert; da.nb01 = down_row_bytes; da.nb02 = (uint64_t)embd * down_row_bytes;
    da.nb03 = (uint64_t)n_expert * da.nb02; da.ne11 = (int32_t)ne20; da.nb10 = 4; da.nb11 = (uint64_t)ff * 4; da.nb12 = ne20 * da.nb11;
    da.nb13 = (uint64_t)nt * da.nb12; da.ne20 = (int32_t)ne20; da.ne21 = nt; da.ne0 = (int32_t)embd; da.ne1 = (int32_t)ne20; da.r2 = 1; da.r3 = 1;

    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (id<MTLBuffer> v in used_views) [e useResource:v usage:MTLResourceUsageRead];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_mm_map0];
    [e setBytes:&ma length:sizeof ma atIndex:0];
    [e setBuffer:IDS offset:0 atIndex:1]; [e setBuffer:MAP offset:0 atIndex:2];
    [e setBuffer:MAP offset:tpe_bytes atIndex:3]; [e setBuffer:MAP offset:work_off atIndex:4];
    [e setThreadgroupMemoryLength:(NSUInteger)n_expert * ne20 * 2u atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(n_expert, 1, 1)];
    dsv4_encode_mm_id(e, pg, &ga, WG, lg, X, 0, GT, MAP, tpe_bytes, work_off, work_cap);
    dsv4_encode_mm_id(e, pg, &ga, WU, lu, X, 0, UP, MAP, tpe_bytes, work_off, work_cap);
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu_w];
    [e setBuffer:GT offset:0 atIndex:0]; [e setBuffer:UP offset:0 atIndex:1]; [e setBuffer:RW offset:0 atIndex:2]; [e setBuffer:MID offset:0 atIndex:3];
    { uint32_t P[2] = { ff, (uint32_t)pairs }; [e setBytes:P length:sizeof P atIndex:4]; [e setBytes:&clamp length:sizeof(float) atIndex:5]; }
    [e dispatchThreads:MTLSizeMake(ff, pairs, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    dsv4_encode_mm_id(e, pd, &da, WD, ld, MID, 0, DN, MAP, tpe_bytes, work_off, work_cap);
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_sum_slots6];
    [e setBuffer:DN offset:0 atIndex:0]; [e setBuffer:OUT offset:0 atIndex:1];
    { uint32_t P[2] = { embd, (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:2]; }
    [e dispatchThreads:MTLSizeMake(embd, nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (hasShared) {   /* shared expert Q8_0 batched: gate/up mul_mm_id (1 esperto), swiglu (peso 1), down, somma in OUT */
        dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mh, &sga, IDB, MAPS, SWG, slg, X, 0, SG_);
        dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mh, &sga, IDB, MAPS, SWU, slu, X, 0, SU_);
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu_w];
        [e setBuffer:SG_ offset:0 atIndex:0]; [e setBuffer:SU_ offset:0 atIndex:1]; [e setBuffer:ONES offset:0 atIndex:2]; [e setBuffer:SMID offset:0 atIndex:3];
        { uint32_t P[2] = { ff, (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:4]; [e setBytes:&clamp length:sizeof(float) atIndex:5]; }
        [e dispatchThreads:MTLSizeMake(ff, nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mhd, &sda, IDB, MAPS, SWD, sld, SMID, 0, SD_);
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_add];
        [e setBuffer:OUT offset:0 atIndex:0]; [e setBuffer:SD_ offset:0 atIndex:1];
        { uint32_t P[1] = { embd * (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:2]; }
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)embd * nt + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    if (cb.error) return -3;
    memcpy(out, [OUT contents], (size_t)nt * embd * sizeof(float));
    return 0;
}

/* Output grouped-LoRA per `nt` token via mul_mm_id Q8_0 (come ds4): out_a con i gruppi come «esperti», out_b a 1 esperto.
 * Pesi Q8_0, heads [nt*heads_total] f32 -> out [nt*embd] f32. Attivazioni a half nel kernel (staging come ds4). 0 = ok. */
int llm_dsv4_metal_attn_out_mm_batch(LlmDsv4Metal* b, int nt, const void* outa_w, size_t outa_row_bytes, int ng, uint32_t gd,
        uint32_t rank, const void* outb_w, size_t outb_row_bytes, uint32_t ol, uint32_t embd, uint32_t heads_total,
        const float* heads, float* out, int n_rot, int hd_rope, int pos0, int n_head, const float* rope_cfg, const void* heads_dev) {
    if (!b || nt <= 0 || nt > 4096 || ng <= 0 || ng != 8 || !b->pso_mm_q8 || !b->pso_mm_map8 || !b->pso_mm_map1) return -1;
    if ((rank % 64u) || (embd % 64u) || (gd % 32u) || (ol % 32u)) return -1;
    id<MTLBuffer> IDS = dsv4_scratch(b, &b->sx[29], &b->sx_cap[29], (size_t)nt * ng * 4);
    id<MTLBuffer> IDB = dsv4_scratch(b, &b->sx[30], &b->sx_cap[30], (size_t)nt * 4);
    id<MTLBuffer> H   = heads_dev ? (id<MTLBuffer>)heads_dev : dsv4_scratch(b, &b->sx[3], &b->sx_cap[3], (size_t)nt * heads_total * sizeof(float));
    id<MTLBuffer> LOW = dsv4_scratch(b, &b->sx[4], &b->sx_cap[4], (size_t)nt * ol * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[5], &b->sx_cap[5], (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> MA  = dsv4_scratch(b, &b->sx[31], &b->sx_cap[31], dsv4_mm_map_bytes((uint32_t)ng, nt, (uint32_t)ng));
    id<MTLBuffer> MB  = dsv4_scratch(b, &b->sx[32], &b->sx_cap[32], dsv4_mm_map_bytes(1, nt, 1));
    size_t la, lb;
    id<MTLBuffer> WA = dsv4_nocopy_view(b, outa_w, (size_t)ng * rank * outa_row_bytes, &la);
    id<MTLBuffer> WB = dsv4_nocopy_view(b, outb_w, (size_t)embd * outb_row_bytes, &lb);
    if (!IDS || !IDB || !H || !LOW || !OUT || !MA || !MB || !WA || !WB) return -2;
    { int32_t* ip = (int32_t*)[IDS contents]; for (int t = 0; t < nt; t++) for (int g = 0; g < ng; g++) ip[t * ng + g] = g;
      int32_t* ib = (int32_t*)[IDB contents]; for (int t = 0; t < nt; t++) ib[t] = 0; }
    if (!heads_dev) memcpy([H contents], heads, (size_t)nt * heads_total * sizeof(float));
    Dsv4MmMapArgs ma = { ng, (int32_t)gd, ng, (uint64_t)gd * 4, (uint64_t)ng * gd * 4, nt, ng, (uint64_t)ng * 4 };
    Dsv4MmArgs aa; memset(&aa, 0, sizeof aa);
    aa.ne00 = (int32_t)gd; aa.ne02 = ng; aa.nb01 = outa_row_bytes; aa.nb02 = (uint64_t)rank * outa_row_bytes; aa.nb03 = (uint64_t)ng * aa.nb02;
    aa.ne11 = ng; aa.nb10 = 4; aa.nb11 = (uint64_t)gd * 4; aa.nb12 = (uint64_t)ng * aa.nb11; aa.nb13 = (uint64_t)nt * aa.nb12;
    aa.ne20 = ng; aa.ne21 = nt; aa.ne0 = (int32_t)rank; aa.ne1 = ng; aa.r2 = 1; aa.r3 = 1;
    Dsv4MmMapArgs mb = { 1, (int32_t)ol, 1, (uint64_t)ol * 4, (uint64_t)ol * 4, nt, 1, 4 };
    Dsv4MmArgs ab; memset(&ab, 0, sizeof ab);
    ab.ne00 = (int32_t)ol; ab.ne02 = 1; ab.nb01 = outb_row_bytes; ab.nb02 = (uint64_t)embd * outb_row_bytes; ab.nb03 = ab.nb02;
    ab.ne11 = 1; ab.nb10 = 4; ab.nb11 = (uint64_t)ol * 4; ab.nb12 = ab.nb11; ab.nb13 = (uint64_t)nt * ab.nb12;
    ab.ne20 = 1; ab.ne21 = nt; ab.ne0 = (int32_t)embd; ab.ne1 = 1; ab.r2 = 1; ab.r3 = 1;
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    /* RoPE INVERSA su GPU sugli heads (device): elimina il sincosf CPU del path a chunk (come ds4). */
    if (n_rot > 0 && b->pso_rope_rows_inv && rope_cfg) {
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope_rows_inv];
        [e setBuffer:H offset:0 atIndex:0];
        { uint32_t PR[4] = { (uint32_t)n_rot, (uint32_t)hd_rope, (uint32_t)pos0, (uint32_t)n_head }; [e setBytes:PR length:sizeof(PR) atIndex:1]; }
        [e setBytes:rope_cfg length:7 * sizeof(float) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    dsv4_encode_mm_id_full(e, b->pso_mm_map8, dsv4_pso_q8(b), &ma, &aa, IDS, MA, WA, la, H, 0, LOW);
    dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &mb, &ab, IDB, MB, WB, lb, LOW, 0, OUT);
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    if (cb.error) return -3;
    memcpy(out, [OUT contents], (size_t)nt * embd * sizeof(float));
    return 0;
}

/* mHC pre per `nt` token in UNA command buffer: res [nt*HC*D] -> normed [nt*D], sp [nt*size] (= pre|post|comb, size=(2+HC)*HC).
 * Stessa matematica di llm_dsv4_block_pre (accumuli float su GPU, ordine diverso). 0 = ok. */
int llm_dsv4_metal_hc_pre_batch(LlmDsv4Metal* b, int nt, int HC, int D, int iters, float hc_eps, float rms_eps, const float* res,
        const float* fn, const float* scale, const float* base, const float* norm_w, float* normed, float* sp) {
    if (!b || nt <= 0 || !b->pso_hcb_scale || !b->pso_hcb_mixes || !b->pso_hcb_sink || !b->pso_hcb_norm || D > 4096 || (D % 4) || HC > 16) return -1;
    const uint32_t size = (uint32_t)((2 + HC) * HC), hcd = (uint32_t)(HC * D);
    if ((hcd % 4u)) return -1;
    id<MTLBuffer> R  = dsv4_scratch(b, &b->sx[54], &b->sx_cap[54], (size_t)nt * hcd * 4);
    id<MTLBuffer> SC = dsv4_scratch(b, &b->sx[55], &b->sx_cap[55], (size_t)nt * 4);
    id<MTLBuffer> MX = dsv4_scratch(b, &b->sx[56], &b->sx_cap[56], (size_t)nt * size * 4);
    id<MTLBuffer> SP = dsv4_scratch(b, &b->sx[57], &b->sx_cap[57], (size_t)nt * size * 4);
    id<MTLBuffer> NM = dsv4_scratch(b, &b->sx[58], &b->sx_cap[58], (size_t)nt * D * 4);
    id<MTLBuffer> FNb = [b->dev newBufferWithBytes:fn length:(NSUInteger)size * hcd * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> SCb = [b->dev newBufferWithBytes:scale length:12 options:MTLResourceStorageModeShared];
    id<MTLBuffer> BSb = [b->dev newBufferWithBytes:base length:(NSUInteger)size * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> NWb = [b->dev newBufferWithBytes:norm_w length:(NSUInteger)D * 4 options:MTLResourceStorageModeShared];
    if (!R || !SC || !MX || !SP || !NM || !FNb || !SCb || !BSb || !NWb) { if (FNb) [FNb release]; if (SCb) [SCb release]; if (BSb) [BSb release]; if (NWb) [NWb release]; return -2; }
    memcpy([R contents], res, (size_t)nt * hcd * 4);
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hcb_scale];
    [e setBuffer:R offset:0 atIndex:0]; [e setBuffer:SC offset:0 atIndex:1];
    { uint32_t P[1] = { hcd }; [e setBytes:P length:sizeof P atIndex:2]; [e setBytes:&rms_eps length:4 atIndex:3]; }
    [e dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hcb_mixes];
    [e setBuffer:R offset:0 atIndex:0]; [e setBuffer:FNb offset:0 atIndex:1]; [e setBuffer:SC offset:0 atIndex:2]; [e setBuffer:MX offset:0 atIndex:3];
    { uint32_t P[2] = { hcd, size }; [e setBytes:P length:sizeof P atIndex:4]; }
    [e dispatchThreadgroups:MTLSizeMake(size, nt, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hcb_sink];
    [e setBuffer:MX offset:0 atIndex:0]; [e setBuffer:SCb offset:0 atIndex:1]; [e setBuffer:BSb offset:0 atIndex:2]; [e setBuffer:SP offset:0 atIndex:3];
    { uint32_t P[3] = { (uint32_t)HC, (uint32_t)iters, size }; [e setBytes:P length:sizeof P atIndex:4]; [e setBytes:&hc_eps length:4 atIndex:5]; }
    [e dispatchThreads:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hcb_norm];
    [e setBuffer:R offset:0 atIndex:0]; [e setBuffer:SP offset:0 atIndex:1]; [e setBuffer:NWb offset:0 atIndex:2]; [e setBuffer:NM offset:0 atIndex:3];
    { uint32_t P[3] = { (uint32_t)D, (uint32_t)HC, size }; [e setBytes:P length:sizeof P atIndex:4]; [e setBytes:&rms_eps length:4 atIndex:5]; }
    [e dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    int rc = cb.error ? -3 : 0;
    if (rc == 0) { memcpy(normed, [NM contents], (size_t)nt * D * 4); memcpy(sp, [SP contents], (size_t)nt * size * 4); }
    [FNb release]; [SCb release]; [BSb release]; [NWb release];
    return rc;
}

/* mHC post per `nt` token: bo [nt*D], res [nt*HC*D], sp [nt*size] (pre|post|comb) -> out [nt*HC*D]. 0 = ok. */
int llm_dsv4_metal_hc_post_batch(LlmDsv4Metal* b, int nt, int HC, int D, const float* bo, const float* res, const float* sp, float* out) {
    if (!b || nt <= 0 || !b->pso_hcb_post || HC > 16) return -1;
    const uint32_t size = (uint32_t)((2 + HC) * HC), hcd = (uint32_t)(HC * D);
    id<MTLBuffer> BO = dsv4_scratch(b, &b->sx[59], &b->sx_cap[59], (size_t)nt * D * 4);
    id<MTLBuffer> R  = dsv4_scratch(b, &b->sx[60], &b->sx_cap[60], (size_t)nt * hcd * 4);
    id<MTLBuffer> SP = dsv4_scratch(b, &b->sx[61], &b->sx_cap[61], (size_t)nt * size * 4);
    id<MTLBuffer> O  = dsv4_scratch(b, &b->sx[62], &b->sx_cap[62], (size_t)nt * hcd * 4);
    if (!BO || !R || !SP || !O) return -2;
    memcpy([BO contents], bo, (size_t)nt * D * 4); memcpy([R contents], res, (size_t)nt * hcd * 4); memcpy([SP contents], sp, (size_t)nt * size * 4);
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hcb_post];
    [e setBuffer:BO offset:0 atIndex:0]; [e setBuffer:R offset:0 atIndex:1]; [e setBuffer:SP offset:0 atIndex:2]; [e setBuffer:O offset:0 atIndex:3];
    { uint32_t P[3] = { (uint32_t)D, (uint32_t)HC, size }; [e setBytes:P length:sizeof P atIndex:4]; }
    [e dispatchThreads:MTLSizeMake(hcd, nt, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    if (cb.error) return -3;
    memcpy(out, [O contents], (size_t)nt * hcd * 4);
    return 0;
}

/* Routing di `nt` token su GPU: logits = gate_inp[n_expert x embd] · x (gemv multi-token), poi top-k/pesi. hsel [nt*used] (esperti da tabella hash) o NULL.
 * bias [n_expert] o NULL. sel [nt*used], wt [nt*used] in uscita. 0 = ok. */
int llm_dsv4_metal_route_batch(LlmDsv4Metal* b, int nt, int n_expert, int embd, int used, const float* gate_inp, const float* bias,
        const int* hsel, float scale, const float* x, int* sel, float* wt) {
    if (!b || nt <= 0 || !b->pso_route_b || !b->pso_gemv_nt || n_expert > 256 || used > 16 || (embd % 4)) return -1;
    id<MTLBuffer> X  = dsv4_scratch(b, &b->sx[68], &b->sx_cap[68], (size_t)nt * embd * 4);
    id<MTLBuffer> LG = dsv4_scratch(b, &b->sx[69], &b->sx_cap[69], (size_t)nt * n_expert * 4);
    id<MTLBuffer> SL = dsv4_scratch(b, &b->sx[70], &b->sx_cap[70], (size_t)nt * used * 4);
    id<MTLBuffer> WT = dsv4_scratch(b, &b->sx[71], &b->sx_cap[71], (size_t)nt * used * 4);
    id<MTLBuffer> HS = dsv4_scratch(b, &b->sx[72], &b->sx_cap[72], (size_t)nt * used * 4);
    id<MTLBuffer> BI = dsv4_scratch(b, &b->sx[73], &b->sx_cap[73], (size_t)n_expert * 4);
    size_t lead; id<MTLBuffer> WG = dsv4_nocopy_view(b, gate_inp, (size_t)n_expert * embd * 4, &lead);
    if (!X || !LG || !SL || !WT || !HS || !BI || !WG || (lead & 3u)) return -2;
    memcpy([X contents], x, (size_t)nt * embd * 4);
    if (bias) memcpy([BI contents], bias, (size_t)n_expert * 4); else memset([BI contents], 0, (size_t)n_expert * 4);
    if (hsel) memcpy([HS contents], hsel, (size_t)nt * used * 4); else memset([HS contents], 0, (size_t)nt * used * 4);
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_gemv_nt];
    [e setBuffer:WG offset:(NSUInteger)lead atIndex:0]; [e setBuffer:X offset:0 atIndex:1]; [e setBuffer:LG offset:0 atIndex:2];
    { uint32_t P[3] = { (uint32_t)embd, (uint32_t)nt, (uint32_t)n_expert }; [e setBytes:P length:sizeof P atIndex:3]; }
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_expert + 3) / 4, ((NSUInteger)nt + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_route_b];
    [e setBuffer:LG offset:0 atIndex:0]; [e setBuffer:BI offset:0 atIndex:1]; [e setBuffer:HS offset:0 atIndex:2]; [e setBuffer:SL offset:0 atIndex:3]; [e setBuffer:WT offset:0 atIndex:4];
    { uint32_t P[5] = { (uint32_t)n_expert, (uint32_t)used, bias ? 1u : 0u, hsel ? 1u : 0u, (uint32_t)nt }; [e setBytes:P length:sizeof P atIndex:5]; [e setBytes:&scale length:4 atIndex:6]; }
    [e dispatchThreads:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    if (cb.error) return -3;
    memcpy(sel, [SL contents], (size_t)nt * used * 4); memcpy(wt, [WT contents], (size_t)nt * used * 4);
    return 0;
}

/* Prewarm: crea (e mette nel residency set) le view no-copy dei pesi statici a load, cosi' il commit e' UNO solo invece di uno per cb nel primo chunk. */
void llm_dsv4_metal_prewarm_view(LlmDsv4Metal* b, const void* ptr, size_t bytes) { size_t lead; if (b && ptr && bytes) (void)dsv4_nocopy_view(b, ptr, bytes, &lead); }
void llm_dsv4_metal_prewarm_commit(LlmDsv4Metal* b) { if (b) dsv4_res_commit(b); }

/* Proiezioni Q-LoRA + KV del chunk via mul_mm_id Q8_0 (1 esperto) + kernel rms/head_rms/rope con dimensione token.
 * Stesse uscite di attn_proj_fused_batch (q_out [nt*qd], kv_out [nt*hd], qrn_out [nt*lq]); attivazioni half nel kernel. 0 = ok. */
int llm_dsv4_metal_attn_proj_mm_batch(LlmDsv4Metal* b, int nt,
        const void* qa_w, size_t qa_row_bytes, const void* kv_w, size_t kv_row_bytes, const void* qb_w, size_t qb_row_bytes,
        const float* xn, uint32_t embd, const float* qan_w, const float* kvan_w, uint32_t lq, uint32_t hd, uint32_t qd,
        int n_head, int n_rot, const float* rope_cfg, int pos0, float eps, float* q_out, float* kv_out, float* qrn_out) {
    if (!b || nt <= 0 || nt > 4096 || !b->pso_mm_q8 || !b->pso_mm_map1 || !b->pso_rms_rows || !b->pso_head_rms_rows || !b->pso_rope_rows) return -1;
    if ((embd % 32u) || (lq % 32u) || n_head <= 0) return -1;
    id<MTLBuffer> IDS = dsv4_scratch(b, &b->sx[33], &b->sx_cap[33], (size_t)nt * 4);
    id<MTLBuffer> MP  = dsv4_scratch(b, &b->sx[34], &b->sx_cap[34], dsv4_mm_map_bytes(1, nt, 1));
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[6],  &b->sx_cap[6],  (size_t)nt*embd*sizeof(float));
    id<MTLBuffer> QR  = dsv4_scratch(b, &b->sx[7],  &b->sx_cap[7],  (size_t)nt*lq*sizeof(float));
    id<MTLBuffer> KVR = dsv4_scratch(b, &b->sx[8],  &b->sx_cap[8],  (size_t)nt*hd*sizeof(float));
    id<MTLBuffer> QB  = dsv4_scratch(b, &b->sx[9],  &b->sx_cap[9],  (size_t)nt*qd*sizeof(float));
    id<MTLBuffer> QRN = dsv4_scratch(b, &b->sx[10], &b->sx_cap[10], (size_t)nt*lq*sizeof(float));
    size_t la, lk, lb;
    id<MTLBuffer> WA = dsv4_nocopy_view(b, qa_w, (size_t)lq * qa_row_bytes, &la);
    id<MTLBuffer> WK = dsv4_nocopy_view(b, kv_w, (size_t)hd * kv_row_bytes, &lk);
    id<MTLBuffer> WQ = dsv4_nocopy_view(b, qb_w, (size_t)qd * qb_row_bytes, &lb);
    id<MTLBuffer> WQAN = [b->dev newBufferWithBytes:qan_w length:(NSUInteger)lq*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> WKAN = [b->dev newBufferWithBytes:kvan_w length:(NSUInteger)hd*sizeof(float) options:MTLResourceStorageModeShared];
    if (!IDS || !MP || !X || !QR || !KVR || !QB || !QRN || !WA || !WK || !WQ || !WQAN || !WKAN) { if (WQAN) [WQAN release]; if (WKAN) [WKAN release]; return -2; }
    memset([IDS contents], 0, (size_t)nt * 4);
    memcpy([X contents], xn, (size_t)nt * embd * sizeof(float));
    const uint32_t hdim = qd / (uint32_t)n_head;
    Dsv4MmMapArgs m1 = { 1, (int32_t)embd, 1, (uint64_t)embd * 4, (uint64_t)embd * 4, nt, 1, 4 };
    Dsv4MmMapArgs m3 = { 1, (int32_t)lq, 1, (uint64_t)lq * 4, (uint64_t)lq * 4, nt, 1, 4 };
    #define MMARGS(NAME, K, ROWS, RB) Dsv4MmArgs NAME; memset(&NAME, 0, sizeof NAME); NAME.ne00 = (int32_t)(K); NAME.ne02 = 1; NAME.nb01 = (RB); \
        NAME.nb02 = (uint64_t)(ROWS) * (RB); NAME.nb03 = NAME.nb02; NAME.ne11 = 1; NAME.nb10 = 4; NAME.nb11 = (uint64_t)(K) * 4; NAME.nb12 = NAME.nb11; \
        NAME.nb13 = (uint64_t)nt * NAME.nb12; NAME.ne20 = 1; NAME.ne21 = nt; NAME.ne0 = (int32_t)(ROWS); NAME.ne1 = 1; NAME.r2 = 1; NAME.r3 = 1;
    MMARGS(aqa, embd, lq, qa_row_bytes) MMARGS(akv, embd, hd, kv_row_bytes) MMARGS(aqb, lq, qd, qb_row_bytes)
    #undef MMARGS
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &m1, &aqa, IDS, MP, WA, la, X, 0, QR);
    dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &m1, &akv, IDS, MP, WK, lk, X, 0, KVR);
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_rows];   /* QRN = rms(QR) * q_a_norm */
    [e setBuffer:QR offset:0 atIndex:0]; [e setBuffer:WQAN offset:0 atIndex:1]; [e setBuffer:QRN offset:0 atIndex:2];
    { uint32_t P[1] = { lq }; [e setBytes:P length:sizeof P atIndex:3]; float cf[1] = { eps }; [e setBytes:cf length:sizeof cf atIndex:4]; }
    [e dispatchThreadgroups:MTLSizeMake(1, nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    dsv4_encode_mm_id_full(e, b->pso_mm_map1, dsv4_pso_q8(b), &m3, &aqb, IDS, MP, WQ, lb, QRN, 0, QB);
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_head_rms_rows];
    [e setBuffer:QB offset:0 atIndex:0];
    { uint32_t P[2] = { hdim, (uint32_t)n_head }; [e setBytes:P length:sizeof P atIndex:1]; float cf[1] = { eps }; [e setBytes:cf length:sizeof cf atIndex:2]; }
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope_rows];
    [e setBuffer:QB offset:0 atIndex:0];
    { uint32_t P[4] = { (uint32_t)n_rot, hdim, (uint32_t)pos0, (uint32_t)n_head }; [e setBytes:P length:sizeof P atIndex:1]; }
    [e setBytes:rope_cfg length:7 * sizeof(float) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_rows];   /* KV: rms in-place con kv_a_norm, poi rope (1 testa) */
    [e setBuffer:KVR offset:0 atIndex:0]; [e setBuffer:WKAN offset:0 atIndex:1]; [e setBuffer:KVR offset:0 atIndex:2];
    { uint32_t P[1] = { hd }; [e setBytes:P length:sizeof P atIndex:3]; float cf[1] = { eps }; [e setBytes:cf length:sizeof cf atIndex:4]; }
    [e dispatchThreadgroups:MTLSizeMake(1, nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope_rows];
    [e setBuffer:KVR offset:0 atIndex:0];
    { uint32_t P[4] = { (uint32_t)n_rot, hd, (uint32_t)pos0, 1u }; [e setBytes:P length:sizeof P atIndex:1]; }
    [e setBytes:rope_cfg length:7 * sizeof(float) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    dsv4_res_commit(b); [cb commit];
    [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    int rc = cb.error ? -3 : 0;
    if (rc == 0) {
        if (q_out)   memcpy(q_out, [QB contents], (size_t)nt*qd*sizeof(float));   /* NULL = resta device (item b) */
        if (kv_out)  memcpy(kv_out, [KVR contents], (size_t)nt*hd*sizeof(float));
        if (qrn_out) memcpy(qrn_out, [QRN contents], (size_t)nt*lq*sizeof(float));
    }
    [WQAN release]; [WKAN release];
    return rc;
}

/* Output grouped-LoRA FUSO (ds4) per `nt` token in UNA command buffer, con `low` intermedio su DEVICE.
 * Pesi Q8_0. heads [nt*heads_total] -> out [nt*embd]. 0 = ok. */
int llm_dsv4_metal_attn_out_fused_batch(LlmDsv4Metal* b, int nt,
        const void* outa_w, size_t outa_row_bytes, size_t outa_lead0,
        int ng, uint32_t gd, uint32_t rank,
        const void* outb_w, size_t outb_row_bytes,
        uint32_t ol, uint32_t embd, uint32_t heads_total,
        const float* heads, float* out, int n_rot, int hd_rope, int pos0, int n_head, const float* rope_cfg, const void* heads_dev) {
    if (!b || !b->pso || ng <= 0 || ng > 64 || nt <= 0 || !outa_w || !outb_w || !out || (!heads && !heads_dev)) return -1;
    size_t abytes = (size_t)ng * rank * outa_row_bytes;
    size_t bbytes = (size_t)embd * outb_row_bytes;
    size_t la, lb;
    id<MTLBuffer> WA  = dsv4_nocopy_view(b, outa_w, abytes, &la);
    id<MTLBuffer> WB  = dsv4_nocopy_view(b, outb_w, bbytes, &lb);
    id<MTLBuffer> H   = heads_dev ? (id<MTLBuffer>)heads_dev : dsv4_scratch(b, &b->sx[3], &b->sx_cap[3], (size_t)nt * heads_total * sizeof(float));
    id<MTLBuffer> LOW = dsv4_scratch(b, &b->sx[4], &b->sx_cap[4], (size_t)ol * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[5], &b->sx_cap[5], (size_t)nt * embd * sizeof(float));
    int rc = (WA && WB && H && LOW && OUT) ? 0 : -2;
    if (rc == 0 && !heads_dev) memcpy([H contents], heads, (size_t)nt * heads_total * sizeof(float));
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        if (n_rot > 0 && b->pso_rope_rows_inv && rope_cfg) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope_rows_inv];
            [e setBuffer:H offset:0 atIndex:0];
            { uint32_t PR[4] = { (uint32_t)n_rot, (uint32_t)hd_rope, (uint32_t)pos0, (uint32_t)n_head }; [e setBytes:PR length:sizeof(PR) atIndex:1]; }
            [e setBytes:rope_cfg length:7 * sizeof(float) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        for (int t = 0; t < nt; t++) {
            const size_t hbase = (size_t)t * heads_total * sizeof(float);
            for (int g = 0; g < ng; g++) {
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
                [e setBuffer:WA offset:0 atIndex:0];
                [e setBuffer:H  offset:(NSUInteger)(hbase + (size_t)g * gd * sizeof(float)) atIndex:1];
                [e setBuffer:LOW offset:(NSUInteger)((size_t)g * rank * sizeof(float)) atIndex:2];
                uint32_t P[7] = { (uint32_t)(la + outa_lead0 + (size_t)g * rank * outa_row_bytes), (uint32_t)outa_row_bytes,
                                  (uint32_t)(gd / 32u), rank, 8u, 34u, 32u };
                [e setBytes:P length:sizeof(P) atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_q8_0];
            [e setBuffer:WB  offset:0 atIndex:0];
            [e setBuffer:LOW offset:0 atIndex:1];
            [e setBuffer:OUT offset:(NSUInteger)((size_t)t * embd * sizeof(float)) atIndex:2];
            uint32_t P2[7] = { (uint32_t)lb, (uint32_t)outb_row_bytes, (uint32_t)(ol / 32u), embd, 8u, 34u, 32u };
            [e setBytes:P2 length:sizeof(P2) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(embd, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e endEncoding];
        dsv4_res_commit(b); [cb commit];
        [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(out, [OUT contents], (size_t)nt * embd * sizeof(float));
    }
    return rc;
}

int llm_dsv4_metal_attn_out_fused(LlmDsv4Metal* b,
        const void* outa_w, size_t outa_row_bytes, size_t outa_lead0,
        int ng, uint32_t gd, uint32_t rank,
        const void* outb_w, size_t outb_row_bytes,
        uint32_t ol, uint32_t embd, uint32_t heads_total,
        const float* heads, float* out) {
    return llm_dsv4_metal_attn_out_fused_batch(b, 1, outa_w, outa_row_bytes, outa_lead0, ng, gd, rank,
                                               outb_w, outb_row_bytes, ol, embd, heads_total, heads, out,
                                               0, 0, 0, 0, NULL, NULL);   /* single-token: RoPE gia' fatta su CPU dal chiamante */
}

/* Codifica l'attenzione mista: split (flash-decoding) + merge se possibile (hd multiplo di 128, <=512),
 * altrimenti il kernel seriale. Le posizioni sono ripartite in `ns` split da >= ~64 posizioni. */
static void dsv4_encode_attn_mixed(LlmDsv4Metal* b, id<MTLComputeCommandEncoder> e,
        id<MTLBuffer> Q, id<MTLBuffer> RAWK, id<MTLBuffer> COMPK, id<MTLBuffer> ALLOWED,
        id<MTLBuffer> SINKS, id<MTLBuffer> OUT, NSUInteger qoff, NSUInteger ooff,
        int n_head, int hd, int n_raw, int n_comp, int has_allowed, NSUInteger raw_boff = 0) {
    /* Percorso packed8 (port vendored): 1 TG/head, 8 simdgroup, reduce in-TG. Solo DS4 (hd=512, 64 head). */
    { int use_p8 = 1; { int v; if (llm_ai_mlx_opt_get_bool("dsv4_attn_p8", &v) == 0) use_p8 = v; }
      if (use_p8 && b->pso_attn_p8 && hd == 512 && n_head == 64) {
          g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_attn_p8];
          [e setBuffer:Q offset:qoff atIndex:0]; [e setBuffer:RAWK offset:raw_boff atIndex:1]; [e setBuffer:COMPK offset:0 atIndex:2];
          [e setBuffer:ALLOWED offset:0 atIndex:3]; [e setBuffer:SINKS offset:0 atIndex:4]; [e setBuffer:OUT offset:ooff atIndex:5];
          { uint32_t P[5] = { (uint32_t)n_raw, (uint32_t)n_comp, (uint32_t)hd, (uint32_t)(has_allowed ? 1u : 0u), (uint32_t)(raw_boff / 4u) }; [e setBytes:P length:sizeof(P) atIndex:6]; }
          [e setThreadgroupMemoryLength:25472u atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          return;
      }
    }
    const uint32_t total = (uint32_t)(n_raw + n_comp);
    id<MTLBuffer> PART = nil;
    uint32_t ns = (total + 63u) / 64u; if (ns < 1u) ns = 1u; if (ns > 16u) ns = 16u;
    if (b->pso_attn_part && b->pso_attn_merge && (hd % 128) == 0 && hd <= 512)
        PART = dsv4_scratch(b, &b->sx[48], &b->sx_cap[48], (size_t)n_head * ns * (size_t)(hd + 2) * sizeof(float));
    if (PART) {
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_attn_part];
        [e setBuffer:Q offset:qoff atIndex:0]; [e setBuffer:RAWK offset:raw_boff atIndex:1]; [e setBuffer:COMPK offset:0 atIndex:2];
        [e setBuffer:ALLOWED offset:0 atIndex:3]; [e setBuffer:PART offset:0 atIndex:4];
        { uint32_t P[5] = { (uint32_t)n_raw, (uint32_t)n_comp, (uint32_t)hd, (uint32_t)(has_allowed ? 1u : 0u), ns }; [e setBytes:P length:sizeof(P) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, ns, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_attn_merge];
        [e setBuffer:PART offset:0 atIndex:0]; [e setBuffer:SINKS offset:0 atIndex:1]; [e setBuffer:OUT offset:ooff atIndex:2];
        { uint32_t P[2] = { (uint32_t)hd, ns }; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_attn_mixed];
    [e setBuffer:Q offset:qoff atIndex:0]; [e setBuffer:RAWK offset:raw_boff atIndex:1]; [e setBuffer:COMPK offset:0 atIndex:2];
    [e setBuffer:ALLOWED offset:0 atIndex:3]; [e setBuffer:SINKS offset:0 atIndex:4]; [e setBuffer:OUT offset:ooff atIndex:5];
    { uint32_t P[4] = { (uint32_t)n_raw, (uint32_t)n_comp, (uint32_t)hd, (uint32_t)(has_allowed ? 1u : 0u) }; [e setBytes:P length:sizeof(P) atIndex:6]; }
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

/* Attn mista (raw+comp) su KV device: replica llm_dsv4_attention_mixed. `allowed` puo' essere
 * NULL (tutte le compresse ammesse). Attenzioni: upload dei KV host; con KV persistente su device
 * il chiamante usera' la variante a buffer condivisi. 0 = ok. */
int llm_dsv4_metal_attention_mixed(LlmDsv4Metal* b, const float* q, int n_head, int hd,
        const float* raw_kv, int n_raw, const float* comp_kv, int n_comp,
        const int* allowed, const float* sinks, float* out) {
    if (!b || !b->pso_attn_mixed || !q || !sinks || !out) return -1;
    if (n_head <= 0 || hd <= 0 || hd > 1024 || n_raw < 0 || n_comp < 0) return -1;
    id<MTLBuffer> BQ = [b->dev newBufferWithBytes:q length:(NSUInteger)((size_t)n_head*(size_t)hd*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BS = [b->dev newBufferWithBytes:sinks length:(NSUInteger)((size_t)n_head*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BR = (n_raw > 0) ? [b->dev newBufferWithBytes:raw_kv length:(NSUInteger)((size_t)n_raw*(size_t)hd*sizeof(float)) options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> BC = (n_comp > 0) ? [b->dev newBufferWithBytes:comp_kv length:(NSUInteger)((size_t)n_comp*(size_t)hd*sizeof(float)) options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> BA = (allowed && n_comp > 0) ? [b->dev newBufferWithBytes:allowed length:(NSUInteger)((size_t)n_comp*sizeof(int)) options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> BO = [b->dev newBufferWithLength:(NSUInteger)((size_t)n_head*(size_t)hd*sizeof(float)) options:MTLResourceStorageModeShared];
    int rc = (BQ && BS && BO && (n_raw == 0 || BR) && (n_comp == 0 || BC)) ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        dsv4_encode_attn_mixed(b, e, BQ, BR, BC, BA, BS, BO, 0, 0, n_head, hd, n_raw, n_comp, BA ? 1 : 0);
        [e endEncoding];
        dsv4_res_commit(b); [cb commit];
        [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(out, [BO contents], (size_t)n_head * (size_t)hd * sizeof(float));
    }
    if (BQ) [BQ release];
    if (BS) [BS release];
    if (BR) [BR release];
    if (BC) [BC release];
    if (BA) [BA release];
    if (BO) [BO release];
    return rc;
}

/* Compressor: pool dello stato -> pooled[hd]. 0 = ok. */
int llm_dsv4_metal_comp_pool(LlmDsv4Metal* b, const float* state_kv, const float* state_score,
        float* pooled, int hd, int ratio, int co) {
    if (!b || !b->pso_comp_pool || !state_kv || !state_score || !pooled) return -1;
    if (hd <= 0 || ratio <= 0 || (co != 1 && co != 2)) return -1;
    int width = co * hd, rows = co * ratio * width;
    id<MTLBuffer> SKV = [b->dev newBufferWithBytes:state_kv length:(NSUInteger)((size_t)rows*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> SSC = [b->dev newBufferWithBytes:state_score length:(NSUInteger)((size_t)rows*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> OUT = [b->dev newBufferWithLength:(NSUInteger)((size_t)hd*sizeof(float)) options:MTLResourceStorageModeShared];
    int rc = (SKV && SSC && OUT) ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_pool];
        [e setBuffer:SKV offset:0 atIndex:0]; [e setBuffer:SSC offset:0 atIndex:1]; [e setBuffer:OUT offset:0 atIndex:2];
        uint32_t P[3] = { (uint32_t)hd, (uint32_t)ratio, (uint32_t)co }; [e setBytes:P length:sizeof(P) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)hd + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(pooled, [OUT contents], (size_t)hd * sizeof(float));
    }
    if (SKV) [SKV release]; if (SSC) [SSC release]; if (OUT) [OUT release];
    return rc;
}

/* Compressor: quantizzazione fp8 e4m3 in-place di una riga (n_nope multiplo di 64). 0 = ok. */
int llm_dsv4_metal_comp_fp8(LlmDsv4Metal* b, float* x, int hd, int n_rot) {
    if (!b || !b->pso_comp_fp8 || !x) return -1;
    int n_nope = hd - n_rot;
    if (n_nope <= 0 || (n_nope % 64) != 0) return 0;   /* come la reference: no-op */
    id<MTLBuffer> X = [b->dev newBufferWithBytes:x length:(NSUInteger)((size_t)hd*sizeof(float)) options:MTLResourceStorageModeShared];
    int rc = X ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_fp8];
        [e setBuffer:X offset:0 atIndex:0];
        uint32_t P[1] = { (uint32_t)n_nope }; [e setBytes:P length:sizeof(P) atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_nope / 64), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(x, [X contents], (size_t)hd * sizeof(float));
    }
    if (X) [X release];
    return rc;
}

/* Indexer QAT in-place (Hadamard128 + fp4) su n_rows righe di head_dim 128. 0 = ok. */
int llm_dsv4_metal_indexer_qat(LlmDsv4Metal* b, float* x, int n_rows, int hd) {
    if (!b || !b->pso_idx_qat || !x) return -1;
    if (n_rows <= 0 || hd != 128) return 0;   /* come la reference: no-op se head_dim != 128 */
    id<MTLBuffer> X = [b->dev newBufferWithBytes:x length:(NSUInteger)((size_t)n_rows*128u*sizeof(float)) options:MTLResourceStorageModeShared];
    int rc = X ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_qat];
        [e setBuffer:X offset:0 atIndex:0];
        uint32_t P[2] = { (uint32_t)n_rows, 128u }; [e setBytes:P length:sizeof(P) atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_rows + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(x, [X contents], (size_t)n_rows * 128u * sizeof(float));
    }
    if (X) [X release];
    return rc;
}

/* Indexer score per le righe compresse: scores[c] = sum_h relu(dot(q_h,kvc_c)) * w[h]. 0 = ok. */
int llm_dsv4_metal_indexer_scores(LlmDsv4Metal* b, const float* q, int nh, int hd,
        const float* kvc, int n_comp, const float* w, float* scores) {
    if (!b || !b->pso_idx_scores || !q || !kvc || !w || !scores) return -1;
    if (nh <= 0 || hd <= 0 || n_comp <= 0) return -1;
    id<MTLBuffer> BQ = [b->dev newBufferWithBytes:q length:(NSUInteger)((size_t)nh*hd*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BK = [b->dev newBufferWithBytes:kvc length:(NSUInteger)((size_t)n_comp*hd*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BW = [b->dev newBufferWithBytes:w length:(NSUInteger)((size_t)nh*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BS = [b->dev newBufferWithLength:(NSUInteger)((size_t)n_comp*sizeof(float)) options:MTLResourceStorageModeShared];
    int rc = (BQ && BK && BW && BS) ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_scores];
        [e setBuffer:BQ offset:0 atIndex:0]; [e setBuffer:BK offset:0 atIndex:1]; [e setBuffer:BW offset:0 atIndex:2]; [e setBuffer:BS offset:0 atIndex:3];
        uint32_t P[3] = { (uint32_t)nh, (uint32_t)hd, (uint32_t)n_comp }; [e setBytes:P length:sizeof(P) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_comp, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(scores, [BS contents], (size_t)n_comp*sizeof(float));
    }
    if (BQ)[BQ release]; if (BK)[BK release]; if (BW)[BW release]; if (BS)[BS release];
    return rc;
}

/* Score indexer BATCHED (una threadgroup per (riga c, token t)) -> scores[nt*max_comp].
 * scores[t][c] = sum_h relu(dot(q[t][h], kv[c][h])) * w[t][h], per c < n_comp[t]. 0 = ok.
 * (Stessa struttura del kernel usato nel mid prefill; qui esposta per il test unit.) */
int llm_dsv4_metal_indexer_scores_t(LlmDsv4Metal* b, const float* q, int nh, int hd,
        const float* kvc, int n_tokens, int max_comp, const int* n_comp, const float* w, float* scores) {
    if (!b || !b->pso_iscore_t || !q || !kvc || !n_comp || !w || !scores) return -1;
    if (nh <= 0 || hd <= 0 || n_tokens <= 0 || max_comp <= 0) return -1;
    uint32_t* nc = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)n_tokens);
    if (!nc) return -2;
    for (int t = 0; t < n_tokens; t++) nc[t] = (uint32_t)(n_comp[t] > 0 ? n_comp[t] : 0);
    id<MTLBuffer> BQ = [b->dev newBufferWithBytes:q length:(NSUInteger)((size_t)n_tokens*nh*hd*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BK = [b->dev newBufferWithBytes:kvc length:(NSUInteger)((size_t)max_comp*hd*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BW = [b->dev newBufferWithBytes:w length:(NSUInteger)((size_t)n_tokens*nh*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BS = [b->dev newBufferWithLength:(NSUInteger)((size_t)n_tokens*max_comp*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BN = [b->dev newBufferWithBytes:nc length:(NSUInteger)((size_t)n_tokens*sizeof(uint32_t)) options:MTLResourceStorageModeShared];
    int rc = (BQ && BK && BW && BS && BN) ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_iscore_t];
        [e setBuffer:BQ offset:0 atIndex:0]; [e setBuffer:BK offset:0 atIndex:1]; [e setBuffer:BW offset:0 atIndex:2]; [e setBuffer:BS offset:0 atIndex:3];
        uint32_t P[3] = { (uint32_t)nh, (uint32_t)hd, (uint32_t)max_comp }; [e setBytes:P length:sizeof(P) atIndex:4];
        [e setBuffer:BN offset:0 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)max_comp, (NSUInteger)n_tokens, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(scores, [BS contents], (size_t)n_tokens*max_comp*sizeof(float));
    }
    if (BQ)[BQ release]; if (BK)[BK release]; if (BW)[BW release]; if (BS)[BS release]; if (BN)[BN release];
    free(nc);
    return rc;
}

/* Indexer top-k deterministico -> allowed[n_comp]. 0 = ok. */
int llm_dsv4_metal_indexer_topk(LlmDsv4Metal* b, const float* scores, int n_comp, int top_k, int* allowed) {
    if (!b || !b->pso_idx_topk || !scores || !allowed) return -1;
    if (n_comp <= 0) return -1;
    if (top_k > n_comp) top_k = n_comp;
    id<MTLBuffer> BS = [b->dev newBufferWithBytes:scores length:(NSUInteger)((size_t)n_comp*sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BA = [b->dev newBufferWithLength:(NSUInteger)((size_t)n_comp*sizeof(int)) options:MTLResourceStorageModeShared];
    int rc = (BS && BA) ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_topk];
        [e setBuffer:BS offset:0 atIndex:0]; [e setBuffer:BA offset:0 atIndex:1];
        uint32_t P[2] = { (uint32_t)n_comp, (uint32_t)top_k }; [e setBytes:P length:sizeof(P) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(allowed, [BA contents], (size_t)n_comp*sizeof(int));
    }
    if (BS)[BS release]; if (BA)[BA release];
    return rc;
}

/* Indexer top-k BATCHED (una threadgroup per token, radix-select) -> allowed[nt*max_comp]. 0 = ok.
 * Stessa selezione della reference (rank<top_k, tie-break indice minore), O(n_comp) per token. */
int llm_dsv4_metal_indexer_topk_t(LlmDsv4Metal* b, const float* scores, int n_tokens, int max_comp,
        const int* n_comp, int top_k, int* allowed) {
    if (!b || !b->pso_itopk_t || !scores || !n_comp || !allowed) return -1;
    if (n_tokens <= 0 || max_comp <= 0) return -1;
    if (top_k < 1) top_k = 1; if (top_k > max_comp) top_k = max_comp;
    uint32_t* nc = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)n_tokens);
    if (!nc) return -2;
    for (int t = 0; t < n_tokens; t++) nc[t] = (uint32_t)(n_comp[t] > 0 ? n_comp[t] : 0);
    id<MTLBuffer> BS = [b->dev newBufferWithBytes:scores length:(NSUInteger)((size_t)n_tokens * max_comp * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BA = [b->dev newBufferWithLength:(NSUInteger)((size_t)n_tokens * max_comp * sizeof(int)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> BN = [b->dev newBufferWithBytes:nc length:(NSUInteger)((size_t)n_tokens * sizeof(uint32_t)) options:MTLResourceStorageModeShared];
    int rc = (BS && BA && BN) ? 0 : -2;
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_itopk_t];
        [e setBuffer:BS offset:0 atIndex:0]; [e setBuffer:BA offset:0 atIndex:1];
        uint32_t P[3] = { (uint32_t)max_comp, (uint32_t)top_k, 256u }; [e setBytes:P length:sizeof(P) atIndex:2];
        [e setBuffer:BN offset:0 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(allowed, [BA contents], (size_t)n_tokens * max_comp * sizeof(int));
    }
    if (BS)[BS release]; if (BA)[BA release]; if (BN)[BN release];
    free(nc);
    return rc;
}

/* ── Buffer device opachi (per il middle persistente su GPU: pesi f32 + stato KV) ── */
void* llm_dsv4_metal_buf_alloc(LlmDsv4Metal* b, size_t bytes) {
    if (!b || bytes == 0) return NULL;
    return (void*)[b->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}
void llm_dsv4_metal_buf_write(void* h, size_t off, const void* src, size_t bytes) {
    id<MTLBuffer> buf = (id<MTLBuffer>)h; if (!buf || !src) return;
    memcpy((uint8_t*)[buf contents] + off, src, bytes);
}
void llm_dsv4_metal_buf_read(void* h, size_t off, void* dst, size_t bytes) {
    id<MTLBuffer> buf = (id<MTLBuffer>)h; if (!buf || !dst) return;
    memcpy(dst, (const uint8_t*)[buf contents] + off, bytes);
}
void llm_dsv4_metal_buf_free(void* h) { id<MTLBuffer> buf = (id<MTLBuffer>)h; if (buf) [buf release]; }

/* Shift [src, src+n) -> [0,n) di un buffer device (stato compressor). 0 = ok. */
int llm_dsv4_metal_comp_shift(LlmDsv4Metal* b, void* h, int n, int src) {
    id<MTLBuffer> buf = (id<MTLBuffer>)h;
    if (!b || !b->pso_comp_shift || !buf || n <= 0 || src < n) return -1;
    id<MTLCommandBuffer> cb = [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_shift];
    [e setBuffer:buf offset:0 atIndex:0];
    uint32_t P[2] = { (uint32_t)n, (uint32_t)src }; [e setBytes:P length:sizeof(P) atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding]; dsv4_res_commit(b); [cb commit]; [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
    return cb.error ? -3 : 0;
}

/* Middle attenzione (compressor+indexer+attn) in UNA command buffer. Replica
 * llm_dsv4_attn_cached_mid a valle della proiezione. KV/stato su device. 0 = ok. */
int llm_dsv4_metal_attn_mid_batch(LlmDsv4Metal* b, const LlmDsv4MidParams* ps, int nt, void** heads_dev_out) {
    if (!b || !ps || nt <= 0 || !ps[0].heads) return -1;
    if (!b->pso_gemv_f32 || !b->pso_rms_w || !b->pso_rope || !b->pso_add || !b->pso_comp_pool || !b->pso_attn_mixed) return -1;
    const LlmDsv4MidParams* p0 = &ps[0];
    const int embd = p0->embd, hd = p0->hd, ratio = p0->ratio, co = p0->co;
    if (ratio < 0 || (co != 1 && co != 2)) return -1;   /* ratio 0 = solo finestra raw */
    int max_icomp = 1; for (int t = 0; t < nt; t++) if (ps[t].n_icomp > max_icomp) max_icomp = ps[t].n_icomp;
    /* Il riferimento C (llm_dsv4_indexer_allowed) fa early-return "tutte ammesse" quando
     * top_k >= n_comp, e in quel caso NON calcola query indexer ne' scores/topk. Replichiamo:
     * a contesto corto/medio (n_icomp <= idx_top_k) saltiamo QI/WI/scores/topk (bit-exact:
     * con has_allowed=0 l'attenzione scandisce tutte le compresse come il C). Lo stato
     * dell'indexer (st_ikv/st_isc) continua ad avanzare come prima. */
    int idx_scores_need = 1;   /* default 1 = nessuno skip (prefill/batch INVARIATI) */
    if (nt == 1)
        idx_scores_need = (ps[0].has_idx && (int)ps[0].n_icomp > (int)p0->idx_top_k) ? 1 : 0;
    /* X/KVC/Q/QRN sono gli STESSI slot (sx[6/8/9/10]) scritti dal project (device-resident, item b):
     * cosi' il project non scarica su host e il mid non ri-carica. Con dev_in=0 il mid li riempie da host. */
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[6],  &b->sx_cap[6],  (size_t)nt * embd * sizeof(float));
    id<MTLBuffer> KVC = dsv4_scratch(b, &b->sx[8],  &b->sx_cap[8],  (size_t)nt * hd * sizeof(float));
    id<MTLBuffer> Q   = dsv4_scratch(b, &b->sx[9],  &b->sx_cap[9],  (size_t)nt * p0->n_head * hd * sizeof(float));
    id<MTLBuffer> QRN = dsv4_scratch(b, &b->sx[10], &b->sx_cap[10], (size_t)nt * p0->lq * sizeof(float));
    const int W_ = co * hd, IW_ = co * p0->ihd;
    id<MTLBuffer> QI  = dsv4_scratch(b, &b->sx[44], &b->sx_cap[44], (size_t)nt * (p0->has_idx ? p0->qdim : 1) * sizeof(float));
    id<MTLBuffer> WI  = dsv4_scratch(b, &b->sx[45], &b->sx_cap[45], (size_t)nt * (p0->has_idx ? p0->idx_nh : 1) * sizeof(float));
    id<MTLBuffer> PKV = dsv4_scratch(b, &b->sx[64], &b->sx_cap[64], (size_t)nt * W_ * 4);
    id<MTLBuffer> PSC = dsv4_scratch(b, &b->sx[65], &b->sx_cap[65], (size_t)nt * W_ * 4);
    id<MTLBuffer> PIKV = dsv4_scratch(b, &b->sx[66], &b->sx_cap[66], (size_t)nt * (p0->has_idx ? IW_ : 1) * 4);
    id<MTLBuffer> PISC = dsv4_scratch(b, &b->sx[67], &b->sx_cap[67], (size_t)nt * (p0->has_idx ? IW_ : 1) * 4);
    id<MTLBuffer> SCO = dsv4_scratch(b, &b->sx[46], &b->sx_cap[46], (size_t)max_icomp * sizeof(float));
    id<MTLBuffer> HD  = dsv4_scratch(b, &b->sx[47], &b->sx_cap[47], (size_t)nt * p0->n_head * hd * sizeof(float));
    if (!X || !KVC || !Q || !QRN || !QI || !WI || !SCO || !HD || !PKV || !PSC || !PIKV || !PISC || !b->pso_gemv_nt || !b->pso_state_store) return -2;
    if (!p0->dev_in)   /* device-resident: i dati sono gia' negli slot sx[6/8/9/10] scritti dal project */
    for (int t = 0; t < nt; t++) {
        const LlmDsv4MidParams* p = &ps[t];
        memcpy((char*)[X contents] + (size_t)t * embd * sizeof(float), p->xn, (size_t)embd * sizeof(float));
        memcpy((char*)[Q contents] + (size_t)t * p->n_head * hd * sizeof(float), p->q, (size_t)p->n_head * hd * sizeof(float));
        memcpy((char*)[KVC contents] + (size_t)t * hd * sizeof(float), p->kv_cur, (size_t)hd * sizeof(float));
        if (p->has_idx) memcpy((char*)[QRN contents] + (size_t)t * p->lq * sizeof(float), p->qrn, (size_t)p->lq * sizeof(float));
    }
    id<MTLCommandBuffer> cb = b->sactive ? b->scb : [b->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = b->sactive ? b->se : [cb computeCommandEncoder];
    /* FASE A — proiezioni f32 indipendenti per TUTTI i token (pesi letti una volta per tile di 16 token invece che per token) */
    if (ratio > 0) {
        #define GEMV_NT(W_BUF, XBUF, OBUF, NIN, NOUT) do { g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_gemv_nt]; \
            [e setBuffer:(id<MTLBuffer>)(W_BUF) offset:0 atIndex:0]; [e setBuffer:(XBUF) offset:0 atIndex:1]; [e setBuffer:(OBUF) offset:0 atIndex:2]; \
            { uint32_t P_[3] = { (uint32_t)(NIN), (uint32_t)nt, (uint32_t)(NOUT) }; [e setBytes:P_ length:sizeof(P_) atIndex:3]; } \
            [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)(NOUT) + 3) / 4, ((NSUInteger)nt + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; } while (0)
        GEMV_NT(p0->w_ckv, X, PKV, embd, W_);
        GEMV_NT(p0->w_cgate, X, PSC, embd, W_);
        if (p0->has_idx) {
            GEMV_NT(p0->w_ickv, X, PIKV, embd, IW_);
            GEMV_NT(p0->w_icgate, X, PISC, embd, IW_);
            if (idx_scores_need) {
                GEMV_NT(p0->w_iqb, QRN, QI, p0->lq, p0->qdim);
                GEMV_NT(p0->w_iproj, X, WI, embd, p0->idx_nh);
            }
        }
        #undef GEMV_NT
    }
    /* RAW lineare per il chunk: [righe di stato gia' presenti | KV correnti di ogni token]: la finestra del token t e' un intervallo
     * contiguo (stesso ordine della finestra scorrevole), quindi niente shift+copia per token. A fine chunk lo stato = ultime righe. */
    const int cap_raw = p0->raw_cap;
    const int n_prev = p0->raw_full ? cap_raw : p0->n_raw - 1;
    id<MTLBuffer> RAWX = dsv4_scratch(b, &b->sx[74], &b->sx_cap[74], (size_t)(n_prev + nt) * hd * sizeof(float));
    if (!RAWX) return -2;
    /* Prepara in blocco (indipendente dal resto della catena): fp8+f16 della riga KV di ogni token; q indexer (rope, qat, scala pesi). */
    {
        const int nnope0 = hd - p0->n_rot;
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_fp8_rows];
        [e setBuffer:KVC offset:0 atIndex:0];
        { uint32_t P[2] = { (uint32_t)nnope0, (uint32_t)hd }; [e setBytes:P length:sizeof(P) atIndex:1]; }
        [e dispatchThreads:MTLSizeMake((NSUInteger)(nnope0 / 64), (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_f16];
        [e setBuffer:KVC offset:0 atIndex:0];
        { uint32_t P[1] = { (uint32_t)((size_t)nt * hd) }; [e setBytes:P length:sizeof(P) atIndex:1]; }
        [e dispatchThreads:MTLSizeMake((NSUInteger)nt * hd, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (n_prev > 0) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_copy];
            [e setBuffer:(id<MTLBuffer>)p0->raw offset:0 atIndex:0]; [e setBuffer:RAWX offset:0 atIndex:1];
            { uint32_t P[1] = { (uint32_t)((size_t)n_prev * hd) }; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreads:MTLSizeMake((NSUInteger)n_prev * hd, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_copy];
        [e setBuffer:KVC offset:0 atIndex:0]; [e setBuffer:RAWX offset:(NSUInteger)((size_t)n_prev * hd * 4) atIndex:1];
        { uint32_t P[1] = { (uint32_t)((size_t)nt * hd) }; [e setBytes:P length:sizeof(P) atIndex:2]; }
        [e dispatchThreads:MTLSizeMake((NSUInteger)nt * hd, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (idx_scores_need) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope_rows];
            [e setBuffer:QI offset:0 atIndex:0];
            { uint32_t P[4] = { (uint32_t)p0->n_rot, (uint32_t)p0->idx_hd, (uint32_t)p0->pos, (uint32_t)p0->idx_nh }; [e setBytes:P length:sizeof(P) atIndex:1]; }
            [e setBytes:p0->rope_cfg length:7 * sizeof(float) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)p0->idx_nh, (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_qat];
            [e setBuffer:QI offset:0 atIndex:0];
            { uint32_t P[2] = { (uint32_t)(p0->idx_nh * nt), (uint32_t)p0->idx_hd }; [e setBytes:P length:sizeof(P) atIndex:1]; }
            [e dispatchThreads:MTLSizeMake((NSUInteger)p0->idx_nh * nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_scale];
            [e setBuffer:WI offset:0 atIndex:0];
            { float cf[1] = { 1.0f / sqrtf((float)(p0->idx_hd * p0->idx_nh)) }; [e setBytes:cf length:sizeof(cf) atIndex:1]; }
            { uint32_t P[1] = { (uint32_t)((size_t)p0->idx_nh * nt) }; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreads:MTLSizeMake((NSUInteger)p0->idx_nh * nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
    }
    /* ── PATH A TOKEN IN GRIGLIA (come ds4: niente catena per token): compressor/indexer a EVENTI su righe lineari [stato | chunk],
     * indexer score/top-k con token in griglia, attenzione con finestra raw lineare a gruppi di 32 token. Stessa matematica del path per-token. ── */
    int use_tok = 0;
    if (nt >= 8 && b->pso_apart_t && b->pso_amerge_t && b->pso_pool_ev && b->pso_rope_ev && b->pso_iscore_t && b->pso_itopk_t && b->pso_ape_rows &&
        b->pso_rms_rows && b->pso_fp8_rows && (hd % 128) == 0 && hd <= 512) use_tok = 1;
    int nev = 0, ev_first = -1, nevi = 0, ev_first_i = -1, ndue = 0;
    if (use_tok && ratio > 0) {
        for (int t = 0; t < nt; t++) {
            if (ps[t].due) ndue++;
            if (ps[t].push_acomp) { if (nev == 0) ev_first = t; else if (t != ev_first + nev * ratio) use_tok = 0; nev++; }
            if (ps[t].push_icomp) { if (nevi == 0) ev_first_i = t; else if (t != ev_first_i + nevi * ratio) use_tok = 0; nevi++; }
        }
        if (ndue != nev) use_tok = 0;
        if (p0->has_idx && nevi != nev) use_tok = 0;
    }
    if (use_tok) {
        const int nh_ = p0->n_head;
        const int pm0 = ratio > 0 ? p0->pos % ratio : 0;
        const int S0 = ratio > 0 ? (co == 2 ? ratio : 0) + pm0 : 0;
        const int pm_end = ratio > 0 ? (p0->pos + nt) % ratio : 0;
        const int keep = ratio > 0 ? (co == 2 ? ratio : 0) + pm_end : 0;
        const int nnope_ = hd - p0->n_rot;
        int max_ic = 1; for (int t = 0; t < nt; t++) if (ps[t].n_icomp > max_ic) max_ic = ps[t].n_icomp;
        if (!p0->has_idx) max_ic = 1;
        id<MTLBuffer> TPB = dsv4_scratch(b, &b->sx[75], &b->sx_cap[75], (size_t)nt * 16);
        id<MTLBuffer> NCB = dsv4_scratch(b, &b->sx[76], &b->sx_cap[76], (size_t)nt * 4);
        id<MTLBuffer> SCB = dsv4_scratch(b, &b->sx[81], &b->sx_cap[81], (size_t)nt * max_ic * 4);
        id<MTLBuffer> ALB = dsv4_scratch(b, &b->sx[82], &b->sx_cap[82], (size_t)nt * max_ic * 4);
        id<MTLBuffer> SELB = dsv4_scratch(b, &b->sx[84], &b->sx_cap[84], (size_t)nt * (size_t)(p0->has_idx ? p0->idx_top_k : 1) * 4);
        if (!TPB || !NCB || !SCB || !ALB) return -2;
        int nsm = 1;
        { uint32_t* tp = (uint32_t*)[TPB contents]; uint32_t* nc = (uint32_t*)[NCB contents];
          for (int t = 0; t < nt; t++) {
            const int nr_ = ps[t].n_raw, na_ = ps[t].n_acomp; int nsel_ = ps[t].has_idx ? (int)p0->idx_top_k : na_; if (nsel_ > na_) nsel_ = na_; int total = nr_ + nsel_; int ns_ = (total + 63) / 64; if (ns_ < 1) ns_ = 1; if (ns_ > 16) ns_ = 16;
            tp[4 * t] = (uint32_t)nr_; tp[4 * t + 1] = (uint32_t)nsel_; tp[4 * t + 2] = (uint32_t)(n_prev + t + 1 - nr_); tp[4 * t + 3] = (uint32_t)ns_;
            nc[t] = (uint32_t)(ps[t].has_idx ? ps[t].n_icomp : 0);
            if (ns_ > nsm) nsm = ns_;
          } }
        #define COPYF(SRC, SOFF, DST, DOFF, NELEM) do { g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_copy]; [e setBuffer:(SRC) offset:(SOFF) atIndex:0]; [e setBuffer:(DST) offset:(DOFF) atIndex:1]; \
            { uint32_t P_[1] = { (uint32_t)(NELEM) }; [e setBytes:P_ length:sizeof(P_) atIndex:2]; } \
            [e dispatchThreads:MTLSizeMake((NSUInteger)(NELEM), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; } while (0)
        /* compressor (attn) e, se presente, indexer: stessi passi con larghezze/pesi diversi */
        for (int pass = 0; pass < 2 && ratio > 0; pass++) {
            if (pass == 1 && !p0->has_idx) break;
            const int w_ = pass == 0 ? co * hd : co * p0->ihd, hdd = pass == 0 ? hd : p0->ihd, evn = pass == 0 ? nev : nevi, evf = pass == 0 ? ev_first : ev_first_i;
            id<MTLBuffer> PK = pass == 0 ? PKV : PIKV, PS_ = pass == 0 ? PSC : PISC;
            id<MTLBuffer> XK = dsv4_scratch(b, &b->sx[pass == 0 ? 77 : 79], &b->sx_cap[pass == 0 ? 77 : 79], (size_t)(S0 + nt) * w_ * 4);
            id<MTLBuffer> XS = dsv4_scratch(b, &b->sx[pass == 0 ? 78 : 80], &b->sx_cap[pass == 0 ? 78 : 80], (size_t)(S0 + nt) * w_ * 4);
            if (!XK || !XS) return -2;
            id<MTLBuffer> stk = (id<MTLBuffer>)(pass == 0 ? p0->st_akv : p0->st_ikv), sts = (id<MTLBuffer>)(pass == 0 ? p0->st_asc : p0->st_isc);
            id<MTLBuffer> apeB = (id<MTLBuffer>)(pass == 0 ? p0->w_cape : p0->w_icape);
            if (S0 > 0) { COPYF(stk, 0, XK, 0, (size_t)S0 * w_); COPYF(sts, 0, XS, 0, (size_t)S0 * w_); }
            COPYF(PK, 0, XK, (NSUInteger)((size_t)S0 * w_ * 4), (size_t)nt * w_);
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_ape_rows];
            [e setBuffer:PS_ offset:0 atIndex:0]; [e setBuffer:apeB offset:0 atIndex:1]; [e setBuffer:XS offset:(NSUInteger)((size_t)S0 * w_ * 4) atIndex:2];
            { uint32_t P_[4] = { (uint32_t)w_, (uint32_t)ratio, (uint32_t)pm0, (uint32_t)nt }; [e setBytes:P_ length:sizeof(P_) atIndex:3]; }
            [e dispatchThreads:MTLSizeMake((NSUInteger)w_, (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            if (evn > 0) {
                /* pass 0 (KV compresso per l'attenzione): PACKED half in acomp_h -> l'attenzione legge
                 * half (banda KV /2). pass 1 (indexer): f32 in icomp. */
                id<MTLBuffer> cB = (id<MTLBuffer>)(pass == 0 ? p0->acomp_h : p0->icomp);
                const NSUInteger coff = (NSUInteger)(pass == 0 ? (size_t)ps[evf].acomp_off * 2 : (size_t)ps[evf].icomp_off * 4);
                const int idx_first = S0 + (ps[evf].pos - p0->pos);
                const int row0 = idx_first - (co * ratio - 1);
                int fuse_ok = (pass == 0) && (b->pso_comp_chain != nil) && ((hdd - p0->n_rot) % 64 == 0);
                { int v; if (llm_ai_mlx_opt_get_bool("dsv4_comp_fuse", &v) == 0 && !v) fuse_ok = 0; }
                if (fuse_ok) {
                    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_chain];
                    [e setBuffer:XK offset:0 atIndex:0]; [e setBuffer:XS offset:0 atIndex:1]; [e setBuffer:cB offset:coff atIndex:2];
                    [e setBuffer:(id<MTLBuffer>)p0->w_cnorm offset:0 atIndex:3];
                    { uint32_t P_[7] = { (uint32_t)hdd, (uint32_t)ratio, (uint32_t)co, (uint32_t)row0, (uint32_t)(hdd - p0->n_rot), (uint32_t)p0->n_rot, (uint32_t)(ps[evf].pos + 1 - ratio) }; [e setBytes:P_ length:sizeof(P_) atIndex:4]; }
                    { float cf[8] = { p0->rms_eps, p0->rope_cfg[0], p0->rope_cfg[1], p0->rope_cfg[2], p0->rope_cfg[3], p0->rope_cfg[4], p0->rope_cfg[5], p0->rope_cfg[6] }; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
                    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)evn, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else {
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_pool_ev];
                [e setBuffer:XK offset:0 atIndex:0]; [e setBuffer:XS offset:0 atIndex:1]; [e setBuffer:cB offset:coff atIndex:2];
                { uint32_t P_[5] = { (uint32_t)hdd, (uint32_t)ratio, (uint32_t)co, (uint32_t)row0, (uint32_t)evn }; [e setBytes:P_ length:sizeof(P_) atIndex:3]; }
                [e dispatchThreads:MTLSizeMake((NSUInteger)hdd, (NSUInteger)evn, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_rows];
                [e setBuffer:cB offset:coff atIndex:0]; [e setBuffer:(id<MTLBuffer>)(pass == 0 ? p0->w_cnorm : p0->w_icnorm) offset:0 atIndex:1]; [e setBuffer:cB offset:coff atIndex:2];
                { uint32_t P_[1] = { (uint32_t)hdd }; [e setBytes:P_ length:sizeof(P_) atIndex:3]; float cf[1] = { p0->rms_eps }; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
                [e dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)evn, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope_ev];
                [e setBuffer:cB offset:coff atIndex:0];
                { uint32_t P_[5] = { (uint32_t)p0->n_rot, (uint32_t)hdd, (uint32_t)(ps[evf].pos + 1 - ratio), 1u, (uint32_t)ratio }; [e setBytes:P_ length:sizeof(P_) atIndex:1]; }
                [e setBytes:p0->rope_cfg length:7 * sizeof(float) atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)evn, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                if (pass == 0) {
                    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_fp8_rows];
                    [e setBuffer:cB offset:coff atIndex:0];
                    { uint32_t P_[2] = { (uint32_t)nnope_, (uint32_t)hdd }; [e setBytes:P_ length:sizeof(P_) atIndex:1]; }
                    [e dispatchThreads:MTLSizeMake((NSUInteger)(nnope_ / 64), (NSUInteger)evn, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } else {
                    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_qat];
                    [e setBuffer:cB offset:coff atIndex:0];
                    { uint32_t P_[2] = { (uint32_t)evn, (uint32_t)hdd }; [e setBytes:P_ length:sizeof(P_) atIndex:1]; }
                    [e dispatchThreads:MTLSizeMake((NSUInteger)evn, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_f16];
                [e setBuffer:cB offset:coff atIndex:0];
                { uint32_t P_[1] = { (uint32_t)((size_t)evn * hdd) }; [e setBytes:P_ length:sizeof(P_) atIndex:1]; }
                [e dispatchThreads:MTLSizeMake((NSUInteger)evn * hdd, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }
            }
            if (keep > 0) {
                COPYF(XK, (NSUInteger)((size_t)(S0 + nt - keep) * w_ * 4), stk, 0, (size_t)keep * w_);
                COPYF(XS, (NSUInteger)((size_t)(S0 + nt - keep) * w_ * 4), sts, 0, (size_t)keep * w_);
            }
        }
        if (p0->has_idx) {
            /* Indexer scores MMA (una TG per blocco di 32 chiavi): riduce la rilettura di q (~32x) e usa i
             * tensor core. Opt-in dsv4_indexer_mma=1 (default OFF: l'ordine di accumulo differisce dallo scalare). */
            /* Default ADATTIVO: a contesto lungo (max_ic>=2048, ~ctx>=8k) l'MMA e' piu' veloce (+14% a 16k);
             * sotto e' piu' lento -> scalare. Override: dsv4_indexer_mma=0|1. */
            int imma = (max_ic >= 2048u) ? 1 : 0; { int v; if (llm_ai_mlx_opt_get_bool("dsv4_indexer_mma", &v) == 0) imma = v; }
            if (imma) imma = (b->pso_iscore_t_mma != nil) && (p0->idx_hd % 32u == 0u) && p0->idx_hd <= 128 && p0->idx_nh >= 8;
            g_dsv4_dispatchs++; [e setComputePipelineState:imma ? b->pso_iscore_t_mma : b->pso_iscore_t];
            [e setBuffer:QI offset:0 atIndex:0]; [e setBuffer:(id<MTLBuffer>)p0->icomp offset:0 atIndex:1]; [e setBuffer:WI offset:0 atIndex:2]; [e setBuffer:SCB offset:0 atIndex:3];
            { uint32_t P_[3] = { (uint32_t)p0->idx_nh, (uint32_t)p0->idx_hd, (uint32_t)max_ic }; [e setBytes:P_ length:sizeof(P_) atIndex:4]; }
            [e setBuffer:NCB offset:0 atIndex:5];
            if (imma) [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((max_ic + 31) / 32), (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            else      [e dispatchThreadgroups:MTLSizeMake((NSUInteger)max_ic, (NSUInteger)nt, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_itopk_t];
            [e setBuffer:SCB offset:0 atIndex:0]; [e setBuffer:ALB offset:0 atIndex:1];
            { uint32_t P_[3] = { (uint32_t)max_ic, (uint32_t)p0->idx_top_k, 256u }; [e setBytes:P_ length:sizeof(P_) atIndex:2]; }
            [e setBuffer:NCB offset:0 atIndex:3]; [e setBuffer:SELB offset:0 atIndex:4];
            /* radix-select: una threadgroup (256) per token, O(n_comp) invece di O(n_comp^2) */
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        {   /* attenzione a gruppi di 32 token */
            const int G_ = 32;
            id<MTLBuffer> PT = dsv4_scratch(b, &b->sx[83], &b->sx_cap[83], (size_t)G_ * nh_ * nsm * (size_t)(hd + 2) * 4);
            if (!PT) return -2;
            for (int g0 = 0; g0 < nt; g0 += G_) {
                const int gt = nt - g0 < G_ ? nt - g0 : G_;
                /* attenzione: GATHER delle top_k (SELT) invece di scandire n_comp (O(top_k) vs O(n_comp)).
                 * P[1] = top_k per i layer CSA (has_idx), 0 per HCA (scansiona tutte le compresse, SELT=identita'). */
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_apart_t];
                [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:RAWX offset:0 atIndex:1]; [e setBuffer:(id<MTLBuffer>)p0->acomp_h offset:0 atIndex:2];
                [e setBuffer:SELB offset:0 atIndex:3]; [e setBuffer:PT offset:0 atIndex:4];
                { uint32_t P_[6] = { (uint32_t)hd, (uint32_t)(p0->has_idx ? (uint32_t)p0->idx_top_k : 0u), (uint32_t)nsm, (uint32_t)g0, (uint32_t)max_ic, (uint32_t)nh_ }; [e setBytes:P_ length:sizeof(P_) atIndex:5]; }
                [e setBuffer:TPB offset:0 atIndex:6];
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh_, (NSUInteger)nsm, (NSUInteger)gt) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_amerge_t];
                [e setBuffer:PT offset:0 atIndex:0]; [e setBuffer:(id<MTLBuffer>)p0->sinks offset:0 atIndex:1]; [e setBuffer:HD offset:0 atIndex:2];
                { uint32_t P_[4] = { (uint32_t)hd, (uint32_t)nsm, (uint32_t)g0, (uint32_t)nh_ }; [e setBytes:P_ length:sizeof(P_) atIndex:3]; }
                [e setBuffer:TPB offset:0 atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh_, (NSUInteger)gt, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }
        #undef COPYF
    }
    if (!use_tok)
    for (int t = 0; t < nt; t++) {
        const LlmDsv4MidParams* p = &ps[t];
        const NSUInteger qio = (NSUInteger)((size_t)t * (p->has_idx ? p->qdim : 0) * 4), wio = (NSUInteger)((size_t)t * (p->has_idx ? p->idx_nh : 0) * 4);
        const int ihd = p->ihd, width = co * hd, iwidth = co * ihd, nnope = hd - p->n_rot;
        const NSUInteger xo = (NSUInteger)((size_t)t * embd * sizeof(float)), ko = (NSUInteger)((size_t)t * hd * sizeof(float));
        const NSUInteger qo = (NSUInteger)((size_t)t * p->n_head * hd * sizeof(float)), no = (NSUInteger)((size_t)t * p->lq * sizeof(float));
        const NSUInteger ho = qo;
        id<MTLBuffer> rawB = (id<MTLBuffer>)p->raw, aB = (id<MTLBuffer>)p->acomp, icB = (id<MTLBuffer>)p->icomp;
        id<MTLBuffer> sakv = (id<MTLBuffer>)p->st_akv, sasc = (id<MTLBuffer>)p->st_asc;
        id<MTLBuffer> sikv = (id<MTLBuffer>)p->st_ikv, sisc = (id<MTLBuffer>)p->st_isc;
        id<MTLBuffer> allowedB = (id<MTLBuffer>)p->allowed;
        const NSUInteger off_a = (NSUInteger)((size_t)p->state_row * width * 4);
        const NSUInteger off_i = (NSUInteger)((size_t)p->state_row * iwidth * 4);
        if (ratio > 0) {   /* ratio 0: nessun compressor */
        /* 1) riga di stato del compressor: kv, gate+ape (proiezioni gia' calcolate in fase A) */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_state_store];
        [e setBuffer:PKV offset:(NSUInteger)((size_t)t * width * 4) atIndex:0]; [e setBuffer:PSC offset:(NSUInteger)((size_t)t * width * 4) atIndex:1];
        [e setBuffer:(id<MTLBuffer>)p->w_cape offset:(NSUInteger)((size_t)p->ape_off * width * 4) atIndex:2];
        [e setBuffer:sakv offset:off_a atIndex:3]; [e setBuffer:sasc offset:off_a atIndex:4];
        { uint32_t P[1] = { (uint32_t)width }; [e setBytes:P length:sizeof(P) atIndex:5]; }
        [e dispatchThreads:MTLSizeMake((NSUInteger)width, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        /* 3) attn compressa (se dovuta): pool -> rms -> rope -> fp8, poi shift dello stato */
        if (p->due) {
          if (p->push_acomp) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_pool];
            [e setBuffer:sakv offset:0 atIndex:0]; [e setBuffer:sasc offset:0 atIndex:1];
            [e setBuffer:aB offset:(NSUInteger)((size_t)p->acomp_off * 4) atIndex:2];
            { uint32_t P[3] = { (uint32_t)hd, (uint32_t)ratio, (uint32_t)co }; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)hd + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_w];
            [e setBuffer:aB offset:(NSUInteger)((size_t)p->acomp_off * 4) atIndex:0]; [e setBuffer:(id<MTLBuffer>)p->w_cnorm offset:0 atIndex:1];
            [e setBuffer:aB offset:(NSUInteger)((size_t)p->acomp_off * 4) atIndex:2];
            { uint32_t P[1] = { (uint32_t)hd }; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1] = { p->rms_eps }; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope];
            [e setBuffer:aB offset:(NSUInteger)((size_t)p->acomp_off * 4) atIndex:0];
            { uint32_t P[4] = { (uint32_t)p->n_rot, (uint32_t)hd, (uint32_t)(p->pos + 1 - ratio), 0u }; [e setBytes:P length:sizeof(P) atIndex:1]; }
            [e setBytes:p->rope_cfg length:7 * sizeof(float) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_fp8];
            [e setBuffer:aB offset:(NSUInteger)((size_t)p->acomp_off * 4) atIndex:0];
            { uint32_t P[1] = { (uint32_t)nnope }; [e setBytes:P length:sizeof(P) atIndex:1]; }
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(nnope / 64), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_f16];
            [e setBuffer:aB offset:(NSUInteger)((size_t)p->acomp_off * 4) atIndex:0];
            { uint32_t P[1] = { (uint32_t)hd }; [e setBytes:P length:sizeof(P) atIndex:1]; }
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((hd + 31) / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
          }
            if (ratio == 4) {
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_shift];
                [e setBuffer:sakv offset:0 atIndex:0]; { uint32_t P[2] = { (uint32_t)(ratio * width), (uint32_t)(ratio * width) }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((ratio * width + 31) / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_shift];
                [e setBuffer:sasc offset:0 atIndex:0]; { uint32_t P[2] = { (uint32_t)(ratio * width), (uint32_t)(ratio * width) }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((ratio * width + 31) / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        }

        /* 4) indexer: compressor + allowed */
        if (p->has_idx) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_state_store];
            [e setBuffer:PIKV offset:(NSUInteger)((size_t)t * iwidth * 4) atIndex:0]; [e setBuffer:PISC offset:(NSUInteger)((size_t)t * iwidth * 4) atIndex:1];
            [e setBuffer:(id<MTLBuffer>)p->w_icape offset:(NSUInteger)((size_t)p->ape_off * iwidth * 4) atIndex:2];
            [e setBuffer:sikv offset:off_i atIndex:3]; [e setBuffer:sisc offset:off_i atIndex:4];
            { uint32_t P[1] = { (uint32_t)iwidth }; [e setBytes:P length:sizeof(P) atIndex:5]; }
            [e dispatchThreads:MTLSizeMake((NSUInteger)iwidth, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            if (p->due) {
              if (p->push_icomp) {
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_pool];
                [e setBuffer:sikv offset:0 atIndex:0]; [e setBuffer:sisc offset:0 atIndex:1];
                [e setBuffer:icB offset:(NSUInteger)((size_t)p->icomp_off * 4) atIndex:2];
                { uint32_t P[3] = { (uint32_t)ihd, (uint32_t)ratio, (uint32_t)co }; [e setBytes:P length:sizeof(P) atIndex:3]; }
                [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)ihd + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_w];
                [e setBuffer:icB offset:(NSUInteger)((size_t)p->icomp_off * 4) atIndex:0]; [e setBuffer:(id<MTLBuffer>)p->w_icnorm offset:0 atIndex:1];
                [e setBuffer:icB offset:(NSUInteger)((size_t)p->icomp_off * 4) atIndex:2];
                { uint32_t P[1] = { (uint32_t)ihd }; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1] = { p->rms_eps }; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope];
                [e setBuffer:icB offset:(NSUInteger)((size_t)p->icomp_off * 4) atIndex:0];
                { uint32_t P[4] = { (uint32_t)p->n_rot, (uint32_t)ihd, (uint32_t)(p->pos + 1 - ratio), 0u }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                [e setBytes:p->rope_cfg length:7 * sizeof(float) atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_qat];
                [e setBuffer:icB offset:(NSUInteger)((size_t)p->icomp_off * 4) atIndex:0];
                { uint32_t P[2] = { 1u, (uint32_t)ihd }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_f16];
                [e setBuffer:icB offset:(NSUInteger)((size_t)p->icomp_off * 4) atIndex:0];
                { uint32_t P[1] = { (uint32_t)ihd }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((ihd + 31) / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
              }
                if (ratio == 4) {
                    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_shift];
                    [e setBuffer:sikv offset:0 atIndex:0]; { uint32_t P[2] = { (uint32_t)(ratio * iwidth), (uint32_t)(ratio * iwidth) }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((ratio * iwidth + 31) / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_comp_shift];
                    [e setBuffer:sisc offset:0 atIndex:0]; { uint32_t P[2] = { (uint32_t)(ratio * iwidth), (uint32_t)(ratio * iwidth) }; [e setBytes:P length:sizeof(P) atIndex:1]; }
                    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((ratio * iwidth + 31) / 32), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
            }
            /* scores + top-k (solo se serve: con top_k >= n_comp tutte le compresse sono ammesse) */
            if (p->has_idx && (int)p->n_icomp > (int)p->idx_top_k) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_scores];
            [e setBuffer:QI offset:qio atIndex:0]; [e setBuffer:icB offset:0 atIndex:1]; [e setBuffer:WI offset:wio atIndex:2]; [e setBuffer:SCO offset:0 atIndex:3];
            { uint32_t P[3] = { (uint32_t)p->idx_nh, (uint32_t)p->idx_hd, (uint32_t)p->n_icomp }; [e setBytes:P length:sizeof(P) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)p->n_icomp, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_idx_topk];
            [e setBuffer:SCO offset:0 atIndex:0]; [e setBuffer:allowedB offset:0 atIndex:1];
            { uint32_t P[2] = { (uint32_t)p->n_icomp, (uint32_t)p->idx_top_k }; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)p->n_icomp + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
        }

        /* 5) attenzione mista (raw + compresse) */
        dsv4_encode_attn_mixed(b, e, Q, RAWX, aB, allowedB, (id<MTLBuffer>)p->sinks, HD, qo, ho, p->n_head, hd, p->n_raw, p->n_acomp,
                               (p->has_idx && (int)p->n_icomp > (int)p->idx_top_k) ? 1 : 0,
                               (NSUInteger)((size_t)(n_prev + t + 1 - p->n_raw) * hd * 4));

        /* PIPELINE CPU/GPU: ogni 16 token chiude e SOTTOMETTE la cb (senza attendere) e ne apre un'altra: la GPU esegue i primi token
         * mentre la CPU codifica i successivi (coda seriale: l'ordine e' garantito). L'attesa e' solo sull'ultima. */
        if (((t + 1) % 16) == 0 && (t + 1) < nt) {
            [e endEncoding]; dsv4_res_commit(b); [cb commit];
            cb = [b->queue commandBuffer]; e = [cb computeCommandEncoder];
        }
    }
    {   /* stato raw = ultime min(cap, n_prev+nt) righe del buffer lineare */
        const int keep = (n_prev + nt < cap_raw) ? n_prev + nt : cap_raw;
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_copy];
        [e setBuffer:RAWX offset:(NSUInteger)((size_t)(n_prev + nt - keep) * hd * 4) atIndex:0]; [e setBuffer:(id<MTLBuffer>)p0->raw offset:0 atIndex:1];
        { uint32_t P[1] = { (uint32_t)((size_t)keep * hd) }; [e setBytes:P length:sizeof(P) atIndex:2]; }
        [e dispatchThreads:MTLSizeMake((NSUInteger)keep * hd, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    if (!b->sactive) {
        [e endEncoding];
        dsv4_res_commit(b); [cb commit];
        [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        if (cb.error) return -3;
    }
    if (heads_dev_out) *heads_dev_out = NULL;
    if (p0->dev_heads) { if (heads_dev_out) *heads_dev_out = (void*)HD; return 0; }   /* heads restano device (item b) */
    if (!b->sactive)
    for (int t = 0; t < nt; t++)
        memcpy(ps[t].heads, (const char*)[HD contents] + (size_t)t * p0->n_head * hd * sizeof(float), (size_t)p0->n_head * hd * sizeof(float));
    return 0;
}

int llm_dsv4_metal_attn_mid(LlmDsv4Metal* b, const LlmDsv4MidParams* p) { return llm_dsv4_metal_attn_mid_batch(b, p, 1, NULL); }

/* Proiezioni Q-LoRA + KV FUSE (ds4) per `nt` token consecutivi (pos0..pos0+nt-1) in UNA command buffer:
 * q_a, kv, RMS, q_b, head_rms, RoPE. Pesi Q8_0. xn [nt*embd] -> q_out [nt*qd], kv_out [nt*hd], qrn_out [nt*lq]. 0 = ok. */
int llm_dsv4_metal_attn_proj_fused_batch(LlmDsv4Metal* b, int nt,
        const void* qa_w, size_t qa_row_bytes,
        const void* kv_w, size_t kv_row_bytes,
        const void* qb_w, size_t qb_row_bytes,
        const float* xn, uint32_t embd,
        const float* qan_w, const float* kvan_w, uint32_t lq, uint32_t hd, uint32_t qd,
        int n_head, int n_rot, const float* rope_cfg, int pos0, float eps,
        float* q_out, float* kv_out, float* qrn_out) {
    if (!b || !b->pso || !b->pso_rms_w || !b->pso_head_rms || !b->pso_rope || nt <= 0) return -1;
    if (!qa_w || !kv_w || !qb_w || !xn || !qan_w || !kvan_w || !q_out || !kv_out || !qrn_out) return -1;
    if ((lq % 32u) || (qd % 32u) || (embd % 32u) || n_head <= 0) return -1;
    size_t qasz = (size_t)lq * qa_row_bytes, kvsz = (size_t)hd * kv_row_bytes, qbsz = (size_t)qd * qb_row_bytes;
    size_t l_qa, l_kv, l_qb;
    id<MTLBuffer> WA  = dsv4_nocopy_view(b, qa_w, qasz, &l_qa);
    id<MTLBuffer> WK  = dsv4_nocopy_view(b, kv_w, kvsz, &l_kv);
    id<MTLBuffer> WQ  = dsv4_nocopy_view(b, qb_w, qbsz, &l_qb);
    id<MTLBuffer> WQAN = [b->dev newBufferWithBytes:qan_w length:(NSUInteger)lq*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> WKAN = [b->dev newBufferWithBytes:kvan_w length:(NSUInteger)hd*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[6],  &b->sx_cap[6],  (size_t)nt*embd*sizeof(float));
    id<MTLBuffer> QR  = dsv4_scratch(b, &b->sx[7],  &b->sx_cap[7],  (size_t)lq*sizeof(float));
    id<MTLBuffer> KVR = dsv4_scratch(b, &b->sx[8],  &b->sx_cap[8],  (size_t)nt*hd*sizeof(float));
    id<MTLBuffer> QB  = dsv4_scratch(b, &b->sx[9],  &b->sx_cap[9],  (size_t)nt*qd*sizeof(float));
    id<MTLBuffer> QRN = dsv4_scratch(b, &b->sx[10], &b->sx_cap[10], (size_t)nt*lq*sizeof(float));
    int rc = (WA && WK && WQ && WQAN && WKAN && X && QR && KVR && QB && QRN) ? 0 : -2;
    if (rc == 0) memcpy([X contents], xn, (size_t)nt*embd*sizeof(float));
    if (rc == 0) {
        const uint32_t hdim = (uint32_t)(qd / (uint32_t)n_head);
        id<MTLCommandBuffer> cb = b->sactive ? b->scb : [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> en = b->sactive ? b->se : [cb computeCommandEncoder];
        for (int t = 0; t < nt; t++) {
            const NSUInteger xo = (NSUInteger)((size_t)t*embd*sizeof(float)), ko = (NSUInteger)((size_t)t*hd*sizeof(float));
            const NSUInteger qo = (NSUInteger)((size_t)t*qd*sizeof(float)),   no = (NSUInteger)((size_t)t*lq*sizeof(float));
            /* gemv q_a: X -> QR */
            [en setComputePipelineState:b->pso_q8_0];
            [en setBuffer:WA offset:0 atIndex:0]; [en setBuffer:X offset:xo atIndex:1]; [en setBuffer:QR offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)l_qa,(uint32_t)qa_row_bytes,embd/32u,lq,8u,34u,32u}; [en setBytes:P length:sizeof(P) atIndex:3]; }
            [en dispatchThreadgroups:MTLSizeMake(lq,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* gemv kv: X -> KVR */
            [en setComputePipelineState:b->pso_q8_0];
            [en setBuffer:WK offset:0 atIndex:0]; [en setBuffer:X offset:xo atIndex:1]; [en setBuffer:KVR offset:ko atIndex:2];
            { uint32_t P[7]={(uint32_t)l_kv,(uint32_t)kv_row_bytes,embd/32u,hd,8u,34u,32u}; [en setBytes:P length:sizeof(P) atIndex:3]; }
            [en dispatchThreadgroups:MTLSizeMake(hd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* rms q_a_norm: QRN = rms(QR)*qan */
            [en setComputePipelineState:b->pso_rms_w];
            [en setBuffer:QR offset:0 atIndex:0]; [en setBuffer:WQAN offset:0 atIndex:1]; [en setBuffer:QRN offset:no atIndex:2];
            { uint32_t P[1]={lq}; [en setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={eps}; [en setBytes:cf length:sizeof(cf) atIndex:4]; }
            [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* gemv q_b: QRN -> QB */
            [en setComputePipelineState:b->pso_q8_0];
            [en setBuffer:WQ offset:0 atIndex:0]; [en setBuffer:QRN offset:no atIndex:1]; [en setBuffer:QB offset:qo atIndex:2];
            { uint32_t P[7]={(uint32_t)l_qb,(uint32_t)qb_row_bytes,lq/32u,qd,8u,34u,32u}; [en setBytes:P length:sizeof(P) atIndex:3]; }
            [en dispatchThreadgroups:MTLSizeMake(qd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* head_rms Q */
            [en setComputePipelineState:b->pso_head_rms];
            [en setBuffer:QB offset:qo atIndex:0];
            { uint32_t P[1]={hdim}; [en setBytes:P length:sizeof(P) atIndex:1]; float cf[1]={eps}; [en setBytes:cf length:sizeof(cf) atIndex:2]; }
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* rope Q */
            [en setComputePipelineState:b->pso_rope];
            [en setBuffer:QB offset:qo atIndex:0];
            { uint32_t P[4]={(uint32_t)n_rot,hdim,(uint32_t)(pos0 + t),0u}; [en setBytes:P length:sizeof(P) atIndex:1]; }
            [en setBytes:rope_cfg length:7*sizeof(float) atIndex:2];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* rms kv_a_norm in-place */
            [en setComputePipelineState:b->pso_rms_w];
            [en setBuffer:KVR offset:ko atIndex:0]; [en setBuffer:WKAN offset:0 atIndex:1]; [en setBuffer:KVR offset:ko atIndex:2];
            { uint32_t P[1]={hd}; [en setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={eps}; [en setBytes:cf length:sizeof(cf) atIndex:4]; }
            [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            /* rope KV (1 head) */
            [en setComputePipelineState:b->pso_rope];
            [en setBuffer:KVR offset:ko atIndex:0];
            { uint32_t P[4]={(uint32_t)n_rot,hd,(uint32_t)(pos0 + t),0u}; [en setBytes:P length:sizeof(P) atIndex:1]; }
            [en setBytes:rope_cfg length:7*sizeof(float) atIndex:2];
            [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        if (!b->sactive) {
            [en endEncoding];
            dsv4_res_commit(b); [cb commit];
            [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
            rc = cb.error ? -3 : 0;
            if (rc == 0) {
                if (q_out)   memcpy(q_out,   [QB contents],  (size_t)nt*qd*sizeof(float));   /* NULL = resta device (item b) */
                if (kv_out)  memcpy(kv_out,  [KVR contents], (size_t)nt*hd*sizeof(float));
                if (qrn_out) memcpy(qrn_out, [QRN contents], (size_t)nt*lq*sizeof(float));
            }
        }
    }
    if (WQAN) [WQAN release]; if (WKAN) [WKAN release];
    return rc;
}

int llm_dsv4_metal_attn_proj_fused(LlmDsv4Metal* b,
        const void* qa_w, size_t qa_row_bytes, const void* kv_w, size_t kv_row_bytes,
        const void* qb_w, size_t qb_row_bytes, const float* xn, uint32_t embd,
        const float* qan_w, const float* kvan_w, uint32_t lq, uint32_t hd, uint32_t qd,
        int n_head, int n_rot, const float* rope_cfg, int pos, float eps,
        float* q_out, float* kv_out, float* qrn_out) {
    return llm_dsv4_metal_attn_proj_fused_batch(b, 1, qa_w, qa_row_bytes, kv_w, kv_row_bytes, qb_w, qb_row_bytes,
            xn, embd, qan_w, kvan_w, lq, hd, qd, n_head, n_rot, rope_cfg, pos, eps, q_out, kv_out, qrn_out);
}

/* Upload di un blocco host in un buffer device (ritorna MTLBuffer "retained", da rilasciare). */
void* llm_dsv4_metal_upload(LlmDsv4Metal* b, const void* data, size_t bytes) {
    if (!b || !data || !bytes) return NULL;
    return (void*)[b->dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}
void llm_dsv4_metal_buffer_release(void* h) { if (h) [(id<MTLBuffer>)h release]; }

/* Registra una regione host come risorsa Metal (no-copy, retained): rende valido l'accesso
 * via puntatore GREZZO dai kernel addr (altrimenti Metal non garantisce l'accesso). */
void* llm_dsv4_metal_register(LlmDsv4Metal* b, const void* ptr, size_t bytes) {
    if (!b || !ptr || !bytes) return NULL;
    const size_t page = (size_t)getpagesize();
    uintptr_t ap = (uintptr_t)ptr & ~(uintptr_t)(page-1);
    size_t lead = (size_t)((uintptr_t)ptr - ap);
    size_t len = ((lead + bytes + page - 1) / page) * page;
    return (void*)[b->dev newBufferWithBytesNoCopy:(void*)ap length:len options:MTLResourceStorageModeShared deallocator:nil];
}
void llm_dsv4_metal_register_release(void* h) { if (h) [(id<MTLBuffer>)h release]; }

/* Hook di test (nessuna env): se impostato, il tail scrive gli stadi in <dir>/d4_*.bin. */
static const char* g_dsv4_tail_dump = NULL;
void llm_dsv4_metal_tail_set_dump(const char* dir) { g_dsv4_tail_dump = dir; }
static void dsv4_dump_buf(const char* nm, id<MTLBuffer> buf, size_t nb) {
    if (!g_dsv4_tail_dump || !buf) return;
    char p[512]; snprintf(p, sizeof p, "%s/d4_%s.bin", g_dsv4_tail_dump, nm);
    FILE* f = fopen(p, "wb"); if (f) { fwrite([buf contents], 1, nb, f); fclose(f); }
}

/* Vista zero-copy pagina-allineata su un blocco pesi (f32 o packed). Ritorna MTLBuffer retained. */
static id<MTLBuffer> dsv4_view(LlmDsv4Metal* b, const void* p, size_t bytes, size_t* out_lead) {
    const size_t page = (size_t)getpagesize();
    uintptr_t ap = (uintptr_t)p & ~(uintptr_t)(page-1);
    size_t lead = (size_t)((uintptr_t)p - ap);
    size_t vl = ((lead + bytes + page - 1) / page) * page;
    if (out_lead) *out_lead = lead;
    return [b->dev newBufferWithBytesNoCopy:(void*)ap length:vl options:MTLResourceStorageModeShared deallocator:nil];
}

/* Cache PERSISTENTE (hash) di viste no-copy: i tensori/esperti sono fissi (mmap) -> la vista
 * si crea UNA volta. Regge decine di migliaia di buffer per-esperto (piccoli). */
#define DSV4_TVC_N 131072u
static struct { const void* ptr; size_t bytes; id<MTLBuffer> buf; size_t lead; } g_tvc[DSV4_TVC_N];
static void dsv4_view_cache_clear(void) {
    for (size_t i = 0; i < DSV4_TVC_N; i++) {
        if (g_tvc[i].buf) [g_tvc[i].buf release];
        g_tvc[i].ptr = NULL; g_tvc[i].bytes = 0;
        g_tvc[i].buf = nil; g_tvc[i].lead = 0;
    }
}
static id<MTLBuffer> dsv4_view_cached(LlmDsv4Metal* b, const void* p, size_t bytes, size_t* out_lead) {
    size_t h = (((uintptr_t)p >> 4) ^ (bytes >> 4)) & (DSV4_TVC_N - 1);
    for (size_t i = 0; i < 128; i++) {
        size_t j = (h + i) & (DSV4_TVC_N - 1);
        if (g_tvc[j].buf == NULL) {
            size_t lead = 0; id<MTLBuffer> buf = dsv4_view(b, p, bytes, &lead);
            if (buf) { g_tvc[j].ptr = p; g_tvc[j].bytes = bytes; g_tvc[j].buf = buf; g_tvc[j].lead = lead; }
            if (out_lead) *out_lead = lead;
            return buf;
        }
        if (g_tvc[j].ptr == p && g_tvc[j].bytes == bytes) { if (out_lead) *out_lead = g_tvc[j].lead; return g_tvc[j].buf; }
    }
    return dsv4_view(b, p, bytes, out_lead);
}

/* gpuAddress (buffer registrato persistente + lead) dell'esperto: indirizzo valido per la GPU. */
uint64_t llm_dsv4_metal_expert_gaddr(LlmDsv4Metal* b, const void* ptr, size_t bytes) {
    size_t lead = 0; id<MTLBuffer> buf = dsv4_view_cached(b, ptr, bytes, &lead);
    return buf ? (uint64_t)[buf gpuAddress] + (uint64_t)lead : 0;
}

/* Argument buffer ds4 "table": struct { device const char* g[N]; ... } con le view per-esperto. */
typedef struct {
    id<MTLBuffer> ab;
    id<MTLBuffer>* views;
    int nv;
    id<MTLResidencySet> residency;
    id<MTLCommandQueue> queue; /* posseduta da LlmDsv4Metal */
} Dsv4ArgBuf;
static int dsv4_argbuf_make_resident(LlmDsv4Metal* b, Dsv4ArgBuf* h) {
    if (!b->residency) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.initialCapacity = 1024;
        b->residency = [b->dev newResidencySetWithDescriptor:d error:nil];
        [d release];
        if (!b->residency) return 0;
        [b->queue addResidencySet:b->residency];
    }
    [b->residency addAllocation:h->ab];
    for (int i = 0; i < h->nv; i++) [b->residency addAllocation:h->views[i]];
    [b->residency commit];
    [b->residency requestResidency];
    h->residency = b->residency;   /* condiviso: NON posseduto dall'argbuf */
    h->queue = b->queue;
    return 1;
}
void* llm_dsv4_metal_argbuf_pair(LlmDsv4Metal* b, const void* const* gs, const void* const* us, int n,
                                 size_t gbytes, size_t ubytes) {
    if (!b->enc_pair || n <= 0 || n > 256) return NULL;
    Dsv4ArgBuf* h = (Dsv4ArgBuf*)calloc(1, sizeof *h); if (!h) return NULL;
    h->ab = [b->dev newBufferWithLength:[b->enc_pair encodedLength] options:MTLResourceStorageModeShared];
    h->views = (id<MTLBuffer>*)calloc((size_t)n * 2, sizeof(id<MTLBuffer>));
    if (!h->ab || !h->views) { llm_dsv4_metal_argbuf_free(h); return NULL; }
    [b->enc_pair setArgumentBuffer:h->ab offset:0];
    for (int e = 0; e < n; e++) {
        size_t glead = 0, ulead = 0;
        id<MTLBuffer> gv = dsv4_view_cached(b, gs[e], gbytes, &glead);
        id<MTLBuffer> uv = dsv4_view_cached(b, us[e], ubytes, &ulead);
        if (!gv || !uv) { llm_dsv4_metal_argbuf_free(h); return NULL; }
        [b->enc_pair setBuffer:gv offset:(NSUInteger)glead atIndex:(NSUInteger)e];
        [b->enc_pair setBuffer:uv offset:(NSUInteger)ulead atIndex:(NSUInteger)(n + e)];
        h->views[h->nv++] = gv; h->views[h->nv++] = uv;
    }
    if (!dsv4_argbuf_make_resident(b, h)) { llm_dsv4_metal_argbuf_free(h); return NULL; }
    return (void*)h;
}
void* llm_dsv4_metal_argbuf_down(LlmDsv4Metal* b, const void* const* ds, int n, size_t dbytes) {
    if (!b->enc_down || n <= 0 || n > 256) return NULL;
    Dsv4ArgBuf* h = (Dsv4ArgBuf*)calloc(1, sizeof *h); if (!h) return NULL;
    h->ab = [b->dev newBufferWithLength:[b->enc_down encodedLength] options:MTLResourceStorageModeShared];
    h->views = (id<MTLBuffer>*)calloc((size_t)n, sizeof(id<MTLBuffer>));
    if (!h->ab || !h->views) { llm_dsv4_metal_argbuf_free(h); return NULL; }
    [b->enc_down setArgumentBuffer:h->ab offset:0];
    for (int e = 0; e < n; e++) {
        size_t dlead = 0;
        id<MTLBuffer> dv = dsv4_view_cached(b, ds[e], dbytes, &dlead);
        if (!dv) { llm_dsv4_metal_argbuf_free(h); return NULL; }
        [b->enc_down setBuffer:dv offset:(NSUInteger)dlead atIndex:(NSUInteger)e];
        h->views[h->nv++] = dv;
    }
    if (!dsv4_argbuf_make_resident(b, h)) { llm_dsv4_metal_argbuf_free(h); return NULL; }
    return (void*)h;
}
void llm_dsv4_metal_argbuf_bind(void* enc, void* handle, int index) {
    Dsv4ArgBuf* h = (Dsv4ArgBuf*)handle; id<MTLComputeCommandEncoder> e = (id<MTLComputeCommandEncoder>)enc;
    if (!h) return;
    [e setBuffer:h->ab offset:0 atIndex:(NSUInteger)index];
}
void llm_dsv4_metal_argbuf_free(void* handle) {
    Dsv4ArgBuf* h = (Dsv4ArgBuf*)handle; if (!h) return;
    if (h->queue && h->residency) {   /* residenza: rimuove le allocazioni dal set CONDIVISO */
        for (int i = 0; i < h->nv; i++) [h->residency removeAllocation:h->views[i]];
        if (h->ab) [h->residency removeAllocation:h->ab];
        [h->residency commit];
    }
    if (h->ab) [h->ab release];
    free(h->views); free(h);
}

/* TAIL FUSO del layer (ds4): attn-out + mhc_post + mhc_pre + route + moe + mhc_post in UNA cb.
 * 0 = ok, <0 = fallback CPU. */
int llm_dsv4_metal_layer_tail(LlmDsv4Metal* b, const LlmDsv4TailParams* q) {
    if (!b || !q) return -1;
    if (!b->pso || !b->pso_rope || !b->pso_hc_post || !b->pso_rms_no_w || !b->pso_gemv_f32 ||
        !b->pso_sinkhorn || !b->pso_hc_sum || !b->pso_rms_w || !b->pso_route ||
        !b->pso_addr_pair || !b->pso_addr_sum6 || !b->pso_swiglu || !b->pso_add) return -1;
    const uint32_t D = (uint32_t)q->D, HC = (uint32_t)q->HC, SIZE = (uint32_t)q->SIZE, HCD = (uint32_t)q->HCD;
    if (!q->heads || !q->out_hc || !q->res_attn || !q->post_attn || !q->comb_attn || !q->fn ||
        !q->scale || !q->base || !q->norm_w || !q->router_w || !q->arg_pair || !q->arg_down) return -1;
    if (!b->pso_tbl_pair || !b->pso_tbl_sum6) return -1;
    if ((q->hd % 1) || (D % 32u)) return -1;

    id<MTLBuffer> HEADS = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->heads, (size_t)q->n_head*q->hd*sizeof(float));
    id<MTLBuffer> RES   = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->res_attn, (size_t)HC*D*sizeof(float));
    id<MTLBuffer> POST  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->post_attn, (size_t)HC*sizeof(float));
    id<MTLBuffer> COMB  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->comb_attn, (size_t)HC*HC*sizeof(float));
    id<MTLBuffer> FN    = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->fn, (size_t)SIZE*HCD*sizeof(float));
    id<MTLBuffer> SCALE = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->scale, 3*sizeof(float));
    id<MTLBuffer> BASE  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->base, (size_t)SIZE*sizeof(float));
    id<MTLBuffer> NORMW = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->norm_w, (size_t)D*sizeof(float));
    id<MTLBuffer> EXPB  = q->exp_probs_b ? (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->exp_probs_b, (size_t)q->n_expert*sizeof(float)) : nil;
    size_t rtw_lead = 0;
    id<MTLBuffer> GADR = nil; id<MTLBuffer> UADR = nil; id<MTLBuffer> DADR = nil;
    id<MTLBuffer> RTW = dsv4_view(b, q->router_w, (size_t)q->n_expert*D*sizeof(float), &rtw_lead);
    size_t oa_lead = 0, ob_lead = 0;
    id<MTLBuffer> OUTA = dsv4_view(b, q->outa_w, (size_t)q->ng*q->rank*q->outa_row_bytes, &oa_lead);
    id<MTLBuffer> OUTB = dsv4_view(b, q->outb_w, (size_t)D*q->outb_row_bytes, &ob_lead);
    size_t sg_lead = 0, su_lead = 0, sd_lead = 0;
    id<MTLBuffer> WG = q->sg ? dsv4_view(b, q->sg, (size_t)q->ff*q->s_nb_g, &sg_lead) : nil;
    id<MTLBuffer> WU = q->su ? dsv4_view(b, q->su, (size_t)q->ff*q->s_nb_g, &su_lead) : nil;
    id<MTLBuffer> WD = q->sd ? dsv4_view(b, q->sd, (size_t)D*q->s_nb_d, &sd_lead) : nil;

    id<MTLBuffer> ATTN = dsv4_scratch(b, &b->sx[21], &b->sx_cap[21], (size_t)D*sizeof(float));
    id<MTLBuffer> NXT  = dsv4_scratch(b, &b->sx[22], &b->sx_cap[22], (size_t)HC*D*sizeof(float));
    id<MTLBuffer> FLAT = dsv4_scratch(b, &b->sx[23], &b->sx_cap[23], (size_t)HCD*sizeof(float));
    id<MTLBuffer> MIXES= dsv4_scratch(b, &b->sx[24], &b->sx_cap[24], (size_t)SIZE*sizeof(float));
    id<MTLBuffer> SPLIT= dsv4_scratch(b, &b->sx[25], &b->sx_cap[25], (size_t)SIZE*sizeof(float));
    id<MTLBuffer> HIN  = dsv4_scratch(b, &b->sx[26], &b->sx_cap[26], (size_t)D*sizeof(float));
    id<MTLBuffer> NRM  = dsv4_scratch(b, &b->sx[27], &b->sx_cap[27], (size_t)D*sizeof(float));
    id<MTLBuffer> LOGI = dsv4_scratch(b, &b->sx[28], &b->sx_cap[28], (size_t)q->n_expert*sizeof(float));
    id<MTLBuffer> SEL  = dsv4_scratch(b, &b->sx[29], &b->sx_cap[29], (size_t)q->used*sizeof(int));
    id<MTLBuffer> RW   = dsv4_scratch(b, &b->sx[30], &b->sx_cap[30], (size_t)q->used*sizeof(float));
    id<MTLBuffer> MID  = dsv4_scratch(b, &b->sx[31], &b->sx_cap[31], (size_t)q->used*q->ff*sizeof(float));
    id<MTLBuffer> ROUT = dsv4_scratch(b, &b->sx[32], &b->sx_cap[32], (size_t)D*sizeof(float));
    id<MTLBuffer> OUTC = dsv4_scratch(b, &b->sx[33], &b->sx_cap[33], (size_t)HC*D*sizeof(float));
    id<MTLBuffer> HLOW = dsv4_scratch(b, &b->sx[34], &b->sx_cap[34], (size_t)q->ol*sizeof(float));
    id<MTLBuffer> G    = dsv4_scratch(b, &b->sx[35], &b->sx_cap[35], (size_t)q->ff*sizeof(float));
    id<MTLBuffer> U    = dsv4_scratch(b, &b->sx[36], &b->sx_cap[36], (size_t)q->ff*sizeof(float));
    id<MTLBuffer> MID2 = dsv4_scratch(b, &b->sx[37], &b->sx_cap[37], (size_t)q->ff*sizeof(float));
    id<MTLBuffer> D2   = dsv4_scratch(b, &b->sx[38], &b->sx_cap[38], (size_t)D*sizeof(float));

    int rc = (HEADS&&RES&&POST&&COMB&&FN&&SCALE&&BASE&&NORMW&&RTW&&OUTA&&OUTB&&
              ATTN&&NXT&&FLAT&&MIXES&&SPLIT&&HIN&&NRM&&LOGI&&SEL&&RW&&MID&&ROUT&&OUTC&&HLOW&&
              (q->sg?(WG&&WU&&WD&&G&&U&&MID2&&D2):1)) ? 0 : -2;

    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const uint32_t NSG = 4, nr0 = 4;
        /* 1) RoPE inversa su heads (n_head) */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rope];
        [e setBuffer:HEADS offset:0 atIndex:0];
        { uint32_t P[4]={(uint32_t)q->n_rot,(uint32_t)q->hd,(uint32_t)q->pos,1u}; [e setBytes:P length:sizeof(P) atIndex:1]; }
        [e setBytes:q->rope_cfg length:7*sizeof(float) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)q->n_head,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* 2) out_a (ng) + out_b -> ATTN */
        for (int g = 0; g < q->ng; g++) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
            [e setBuffer:OUTA offset:0 atIndex:0];
            [e setBuffer:HEADS offset:(NSUInteger)((size_t)g*q->gd*sizeof(float)) atIndex:1];
            [e setBuffer:HLOW offset:(NSUInteger)((size_t)g*q->rank*sizeof(float)) atIndex:2];
            uint32_t P[7]={(uint32_t)(oa_lead+(size_t)g*q->rank*q->outa_row_bytes),(uint32_t)q->outa_row_bytes,(uint32_t)(q->gd/32u),(uint32_t)q->rank,8u,34u,32u};
            [e setBytes:P length:sizeof(P) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(q->rank,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
        [e setBuffer:OUTB offset:0 atIndex:0]; [e setBuffer:HLOW offset:0 atIndex:1]; [e setBuffer:ATTN offset:0 atIndex:2];
        { uint32_t P[7]={(uint32_t)ob_lead,(uint32_t)q->outb_row_bytes,(uint32_t)(q->ol/32u),D,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* 3) hc_post_attn: NXT = post*ATTN + comb*RES */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hc_post];
        [e setBuffer:ATTN offset:0 atIndex:0]; [e setBuffer:RES offset:0 atIndex:1]; [e setBuffer:POST offset:0 atIndex:2];
        [e setBuffer:COMB offset:0 atIndex:3]; [e setBuffer:NXT offset:0 atIndex:4];
        { uint32_t P[2]={D,HC}; [e setBytes:P length:sizeof(P) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake((HC*D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        /* 4) mhc_pre_ffn: rms_no_w(NXT)->FLAT; gemv_f32(fn)->MIXES; sinkhorn->SPLIT; hc_sum->HIN; rms_w->NRM */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_no_w];
        [e setBuffer:NXT offset:0 atIndex:0]; [e setBuffer:FLAT offset:0 atIndex:1];
        { uint32_t P[1]={HCD}; [e setBytes:P length:sizeof(P) atIndex:2]; float cf[1]={q->rms_eps}; [e setBytes:cf length:sizeof(cf) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_gemv_f32];
        [e setBuffer:FN offset:0 atIndex:0]; [e setBuffer:FLAT offset:0 atIndex:1]; [e setBuffer:MIXES offset:0 atIndex:2];
        { uint32_t P[2]={HCD,0u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake(SIZE,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_sinkhorn];
        [e setBuffer:MIXES offset:0 atIndex:0]; [e setBuffer:SCALE offset:0 atIndex:1]; [e setBuffer:BASE offset:0 atIndex:2]; [e setBuffer:SPLIT offset:0 atIndex:3];
        { uint32_t P[2]={HC,(uint32_t)q->sinkhorn_iters}; [e setBytes:P length:sizeof(P) atIndex:4]; float cf[1]={q->hc_eps}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hc_sum];
        [e setBuffer:NXT offset:0 atIndex:0]; [e setBuffer:SPLIT offset:0 atIndex:1]; [e setBuffer:HIN offset:0 atIndex:2];
        { uint32_t P[2]={D,HC}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake((D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_rms_w];
        [e setBuffer:HIN offset:0 atIndex:0]; [e setBuffer:NORMW offset:0 atIndex:1]; [e setBuffer:NRM offset:0 atIndex:2];
        { uint32_t P[1]={D}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={q->rms_eps}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* 5) route: logits=gemv_f32(router,NRM); SEL/RW */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_gemv_f32];
        [e setBuffer:RTW offset:0 atIndex:0]; [e setBuffer:NRM offset:0 atIndex:1]; [e setBuffer:LOGI offset:0 atIndex:2];
        { uint32_t P[2]={D,(uint32_t)(rtw_lead/sizeof(float))}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)q->n_expert,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_route];
        [e setBuffer:LOGI offset:0 atIndex:0]; [e setBuffer:(EXPB?EXPB:SCALE) offset:0 atIndex:1]; [e setBuffer:SEL offset:0 atIndex:2]; [e setBuffer:RW offset:0 atIndex:3];
        { uint32_t P[3]={(uint32_t)q->n_expert,(uint32_t)q->used,(EXPB?1u:0u)}; [e setBytes:P length:sizeof(P) atIndex:4]; float cf[1]={q->expert_scale}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        /* 6) MoE addr: pair slots6 (grid z=used) + sum6 -> ROUT */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_tbl_pair];
        [e setBuffer:SEL offset:0 atIndex:0];
        llm_dsv4_metal_argbuf_bind((void*)e, (void*)q->arg_pair, 1);
        [e setBuffer:NRM offset:0 atIndex:2]; [e setBuffer:MID offset:0 atIndex:3]; [e setBuffer:RW offset:0 atIndex:4];
        { float cf[1]={q->clamp}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
        { uint32_t P[3]={D,q->gu_nb01,q->ff}; [e setBytes:P length:sizeof(P) atIndex:6]; }
        [e setThreadgroupMemoryLength:256*8+128 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((q->ff+NSG*nr0-1)/(NSG*nr0),1,(NSUInteger)q->used) threadsPerThreadgroup:MTLSizeMake(32*NSG,1,1)];
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_tbl_sum6];
        [e setBuffer:SEL offset:0 atIndex:0];
        llm_dsv4_metal_argbuf_bind((void*)e, (void*)q->arg_down, 1);
        [e setBuffer:MID offset:0 atIndex:2]; [e setBuffer:ROUT offset:0 atIndex:3];
        { uint32_t P[4]={q->ff/256u,q->down_nb01,D,q->ff}; [e setBytes:P length:sizeof(P) atIndex:4]; }
        [e dispatchThreadgroups:MTLSizeMake((D+NSG*nr0-1)/(NSG*nr0),1,1) threadsPerThreadgroup:MTLSizeMake(32*NSG,1,1)];
        /* shared expert -> ROUT += D2 */
        if (q->sg) {
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
            [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:NRM offset:0 atIndex:1]; [e setBuffer:G offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)sg_lead,q->s_nb_g,D/32u,q->ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(q->ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
            [e setBuffer:WU offset:0 atIndex:0]; [e setBuffer:NRM offset:0 atIndex:1]; [e setBuffer:U offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)su_lead,q->s_nb_g,D/32u,q->ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(q->ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_swiglu];
            [e setBuffer:G offset:0 atIndex:0]; [e setBuffer:U offset:0 atIndex:1]; [e setBuffer:MID2 offset:0 atIndex:2];
            { uint32_t P[1]={q->ff}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={q->clamp}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake((q->ff+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso];
            [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:D2 offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)sd_lead,q->s_nb_d,q->ff/32u,D,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_add];
            [e setBuffer:ROUT offset:0 atIndex:0]; [e setBuffer:D2 offset:0 atIndex:1];
            { uint32_t P[1]={D}; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreadgroups:MTLSizeMake((D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        }
        /* 7) hc_post_ffn: OUTC = routed-post + comb*NXT  (post/comb = SPLIT[HC..2HC), SPLIT[2HC..)) */
        g_dsv4_dispatchs++; [e setComputePipelineState:b->pso_hc_post];
        [e setBuffer:ROUT offset:0 atIndex:0]; [e setBuffer:NXT offset:0 atIndex:1];
        [e setBuffer:SPLIT offset:(NSUInteger)((size_t)HC*sizeof(float)) atIndex:2];
        [e setBuffer:SPLIT offset:(NSUInteger)((size_t)2*HC*sizeof(float)) atIndex:3];
        [e setBuffer:OUTC offset:0 atIndex:4];
        { uint32_t P[2]={D,HC}; [e setBytes:P length:sizeof(P) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake((HC*D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding];
        dsv4_res_commit(b); [cb commit];
        [cb waitUntilCompleted]; dsv4_gpu_acct(cb);
        rc = cb.error ? -3 : 0;
        if (rc == 0) {
            memcpy(q->out_hc, [OUTC contents], (size_t)HC*D*sizeof(float));
            dsv4_dump_buf("heads", HEADS, (size_t)q->n_head*q->hd*sizeof(float));
            dsv4_dump_buf("attn",  ATTN,  (size_t)D*sizeof(float));
            dsv4_dump_buf("nxt",   NXT,   (size_t)HC*D*sizeof(float));
            dsv4_dump_buf("nrm",   NRM,   (size_t)D*sizeof(float));
            dsv4_dump_buf("sel",   SEL,   (size_t)q->used*sizeof(int));
            dsv4_dump_buf("rw",    RW,    (size_t)q->used*sizeof(float));
            dsv4_dump_buf("mid",   MID,   (size_t)q->used*q->ff*sizeof(float));
            dsv4_dump_buf("rout",  ROUT,  (size_t)D*sizeof(float));
            dsv4_dump_buf("outc",  OUTC,  (size_t)HC*D*sizeof(float));
        }
    }

    if (HEADS) [HEADS release]; if (RES) [RES release]; if (POST) [POST release]; if (COMB) [COMB release];
    if (FN) [FN release]; if (SCALE) [SCALE release]; if (BASE) [BASE release]; if (NORMW) [NORMW release];
    if (EXPB) [EXPB release];
    if (GADR) [GADR release]; if (UADR) [UADR release]; if (DADR) [DADR release];
    if (RTW) [RTW release]; if (OUTA) [OUTA release]; if (OUTB) [OUTB release];
    if (WG) [WG release]; if (WU) [WU release]; if (WD) [WD release];
    return rc;
}
