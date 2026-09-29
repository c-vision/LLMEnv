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
#include "llm_dsv4_metal_tables.h"
#include "llm_dsv4_actquant.h"   /* riuso: llm_dsv4_q8_K_quantize_row (metodo ds4) */

/* ── sorgente kernel (le tabelle IQ2 vengono anteposte a runtime) ── */
static const char* const LLM_DSV4_METAL_KERNELS =
"#include <metal_stdlib>\n"
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
"        atomic_fetch_add_explicit((device atomic_float*)&O[row], 0.125f * d * y->d * (float)bsum, memory_order_relaxed);\n"
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
"        atomic_fetch_add_explicit((device atomic_float*)&O[row], dall * (float)isum - dmin * (float)summs, memory_order_relaxed);\n"
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
"        atomic_fetch_add_explicit((device atomic_float*)&O[row], dall * (float)isum + dmin * (float)summs, memory_order_relaxed);\n"
"    }\n"
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
"    device const float* Y [[buffer(12)]], device float* MID [[buffer(13)]], device const float* RW [[buffer(14)]],\n"
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
"kernel void dsv4_swiglu(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],\n"
"    device float* MID [[buffer(2)]], constant uint* P [[buffer(3)]], constant float* CF [[buffer(4)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint n = P[0]; if (gid >= n) return;\n"
"    const float c = CF[0]; float g = G[gid], u = U[gid];\n"
"    if (c > 1.0e-6f) { g = min(g, c); u = clamp(u, -c, c); }\n"
"    const float silu = g / (1.f + exp(-g)); MID[gid] = silu * u;\n"
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
"    const uint n = P[0]; device const float* wr = W + (ulong)row * n;\n"
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
"}\n";

struct LlmDsv4Metal {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
    id<MTLComputePipelineState> pso_q8k_iq2;
    id<MTLComputePipelineState> pso_q8k_q2k;
    id<MTLComputePipelineState> pso_q8k_q4k;
    id<MTLComputePipelineState> pso_pair;   /* MoE fuso gate+up+swiglu (ds4 pair) */
    id<MTLComputePipelineState> pso_sum6_q2k; /* down 6 esperti Q2_K + somma (ds4 sum6) */
    id<MTLComputePipelineState> pso_slots6;  /* pair_swiglu 6 esperti in 1 dispatch (ds4 slots6) */
    id<MTLComputePipelineState> pso_rms_w, pso_head_rms, pso_rope; /* RMSNorm/RoPE su GPU (ds4) */
    id<MTLComputePipelineState> pso_swiglu, pso_add; /* shared expert su GPU (swiglu + accumulate) */
    id<MTLComputePipelineState> pso_rms_no_w, pso_gemv_f32, pso_sinkhorn, pso_hc_sum, pso_hc_post; /* mHC */
    id<MTLComputePipelineState> pso_route;   /* router su GPU (softplus/top-k) */
    id<MTLComputePipelineState> pso_addr_pair, pso_addr_sum6; /* MoE via tabella indirizzi (ds4 addr) */
    /* Buffer scratch PERSISTENTI riusati (slot per job): niente alloc/copia per-token. */
    id<MTLBuffer> sx[64], so[64], spb[64];
    size_t sx_cap[64], so_cap[64], spb_cap[64];
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

int llm_dsv4_metal_available(void) {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (d) { [d release]; return 1; }
    return 0;
}

LlmDsv4Metal* llm_dsv4_metal_create(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return NULL;
    NSError* err = nil;
    NSString* src = [NSString stringWithFormat:@"%s\n%s\n%s\n%s", LLM_DSV4_METAL_TABLES, LLM_DSV4_METAL_KERNELS, LLM_DSV4_METAL_Q8K, LLM_DSV4_METAL_MOE];
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
    b->pso_pair    = f5 ? [dev newComputePipelineStateWithFunction:f5 error:&err] : nil;
    b->pso_sum6_q2k = f6 ? [dev newComputePipelineStateWithFunction:f6 error:&err] : nil;
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
    [lib release];
    if (!b->pso || !b->pso_q8k_iq2 || !b->pso_q8k_q2k || !b->pso_q8k_q4k || !b->pso_pair || !b->pso_sum6_q2k || !b->pso_rms_w || !b->pso_head_rms || !b->pso_rope || !b->pso_slots6 || !b->pso_swiglu || !b->pso_add || !b->pso_rms_no_w || !b->pso_gemv_f32 || !b->pso_sinkhorn || !b->pso_hc_sum || !b->pso_hc_post || !b->pso_route || !b->pso_addr_pair || !b->pso_addr_sum6 || !b->queue) {
        fprintf(stderr, "dsv4_metal: pipeline fallita: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        llm_dsv4_metal_destroy(b); return NULL;
    }
    return b;
}

void llm_dsv4_metal_destroy(LlmDsv4Metal* b) {
    if (!b) return;
    if (b->pso)          [b->pso release];
    if (b->pso_q8k_iq2)  [b->pso_q8k_iq2 release];
    if (b->pso_q8k_q2k)  [b->pso_q8k_q2k release];
    if (b->pso_q8k_q4k)  [b->pso_q8k_q4k release];
    if (b->pso_pair)     [b->pso_pair release];
    if (b->pso_sum6_q2k) [b->pso_sum6_q2k release];
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
    if (b->pso_rms_w)    [b->pso_rms_w release];
    if (b->pso_head_rms) [b->pso_head_rms release];
    if (b->pso_rope)     [b->pso_rope release];
    for (int i = 0; i < 64; i++) {
        if (b->sx[i])  [b->sx[i] release];
        if (b->so[i])  [b->so[i] release];
        if (b->spb[i]) [b->spb[i] release];
    }
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
    [e setComputePipelineState:b->pso];
    [e setBuffer:W  offset:0 atIndex:0];
    [e setBuffer:X  offset:0 atIndex:1];
    [e setBuffer:O  offset:0 atIndex:2];
    [e setBuffer:PB offset:0 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
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
        uintptr_t ap = (uintptr_t)p & ~(uintptr_t)(page - 1);
        size_t lead = (size_t)((uintptr_t)p - ap);
        size_t slice = (size_t)j->out_dim * j->row_bytes;
        size_t view_len = ((lead + slice + page - 1) / page) * page;
        ws[i] = [b->dev newBufferWithBytesNoCopy:(void*)ap length:view_len
                    options:MTLResourceStorageModeShared deallocator:nil];
        if (t == 8) {
            xs[i] = dsv4_scratch(b, &b->sx[i], &b->sx_cap[i], (size_t)j->n * sizeof(float));
            if (xs[i]) memcpy([xs[i] contents], j->x, (size_t)j->n * sizeof(float));
            os[i] = dsv4_scratch(b, &b->so[i], &b->so_cap[i], (size_t)j->out_dim * sizeof(float));
            uint32_t P[7] = { (uint32_t)lead, (uint32_t)j->row_bytes, j->n / 32u, j->out_dim, 8u, 34u, 32u };
            pbs[i] = dsv4_scratch(b, &b->spb[i], &b->spb_cap[i], sizeof P);
            if (pbs[i]) memcpy([pbs[i] contents], P, sizeof P);
            [e setComputePipelineState:b->pso];
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
            if (os[i]) memset([os[i] contents], 0, (size_t)j->out_dim * sizeof(float));
            uint32_t P[4] = { (uint32_t)lead, (uint32_t)j->row_bytes, nblk, j->out_dim };
            pbs[i] = dsv4_scratch(b, &b->spb[i], &b->spb_cap[i], sizeof P);
            if (pbs[i]) memcpy([pbs[i] contents], P, sizeof P);
            [e setComputePipelineState:(t == 16) ? b->pso_q8k_iq2 : (t == 10) ? b->pso_q8k_q2k : b->pso_q8k_q4k];
            if (!ws[i] || !xs[i] || !os[i] || !pbs[i]) { rc = -2; break; }
            [e setBuffer:ws[i] offset:0 atIndex:0];
            [e setBuffer:xs[i] offset:0 atIndex:1];
            [e setBuffer:os[i] offset:0 atIndex:2];
            [e setBuffer:pbs[i] offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)j->out_dim * nblk, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        nq++;
    }
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        static int once = 0;
        if (!once) { once = 1; fprintf(stderr, "dsv4_metal: command buffer ERROR: %s\n",
                                       [[cb.error localizedDescription] UTF8String]); }
        rc = -3;
    }
    if (rc == 0)
        for (int i = 0; i < nq; i++)
            memcpy((void*)jobs[i].out, [os[i] contents], (size_t)jobs[i].out_dim * sizeof(float));
    for (int i = 0; i < 64; i++)
        if (ws[i]) [ws[i] release];   /* xs/os/pbs: buffer PERSISTENTI, non rilasciati */
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
    [e setComputePipelineState:b->pso_pair];
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
    [cb commit];
    [cb waitUntilCompleted];
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
            [e setComputePipelineState:b->pso_pair];
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
        [cb commit];
        [cb waitUntilCompleted];
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

/* MoE ds4 FUSO (gate/up IQ2_XXS + down Q2_K): `n`=6 esperti.
 * UNA command buffer: pair_swiglu (gate+up+swiglu*peso -> mid f32 su device) per i 6,
 * poi sum6 (down dei 6 + somma -> out). Un solo commit+wait, nessun round-trip CPU. 0 = ok. */
int llm_dsv4_metal_moe_iq2_q2k(LlmDsv4Metal* b, const void* const* gates, const void* const* ups,
                               const void* const* downs, const float* x, const float* rws, int n,
                               float clamp, uint32_t embd, uint32_t gu_nb01, uint32_t ff,
                               uint32_t down_nb01, const void* sg, const void* su, const void* sd,
                               uint32_t s_nb_g, uint32_t s_nb_d, float* out) {
    if (!b || !b->pso_slots6 || !b->pso_sum6_q2k || n != 6 || !gates || !ups || !downs || !x || !rws || !out) return -1;
    if ((ff % 256u) != 0u) return -1;
    const size_t page = (size_t)getpagesize();
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[0], &b->sx_cap[0], (size_t)embd * sizeof(float));
    id<MTLBuffer> MID = dsv4_scratch(b, &b->sx[1], &b->sx_cap[1], (size_t)n * (size_t)ff * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[2], &b->sx_cap[2], (size_t)embd * sizeof(float));
    if (!X || !MID || !OUT) return -2;
    memcpy([X contents], x, (size_t)embd * sizeof(float));
    id<MTLBuffer> BG[6], BU[6], BD[6];
    for (int i = 0; i < 6; i++) { BG[i] = BU[i] = BD[i] = nil; }
    uint32_t PV[15]; PV[0] = embd; PV[1] = gu_nb01; PV[2] = ff;
    int rc = 0;
    for (int i = 0; i < n; i++) {
        size_t gbytes = (size_t)ff * gu_nb01, dbytes = (size_t)embd * down_nb01;
        uintptr_t apg = (uintptr_t)gates[i] & ~(uintptr_t)(page-1); size_t lg = (size_t)((uintptr_t)gates[i]-apg);
        uintptr_t apu = (uintptr_t)ups[i]   & ~(uintptr_t)(page-1); size_t lu = (size_t)((uintptr_t)ups[i]-apu);
        uintptr_t apd = (uintptr_t)downs[i] & ~(uintptr_t)(page-1); size_t ld = (size_t)((uintptr_t)downs[i]-apd);
        size_t vlg = ((lg+gbytes+page-1)/page)*page, vlu = ((lu+gbytes+page-1)/page)*page, vld = ((ld+dbytes+page-1)/page)*page;
        PV[3+i] = (uint32_t)lg; PV[9+i] = (uint32_t)lu;
        BG[i] = [b->dev newBufferWithBytesNoCopy:(void*)apg length:vlg options:MTLResourceStorageModeShared deallocator:nil];
        BU[i] = [b->dev newBufferWithBytesNoCopy:(void*)apu length:vlu options:MTLResourceStorageModeShared deallocator:nil];
        BD[i] = [b->dev newBufferWithBytesNoCopy:(void*)apd length:vld options:MTLResourceStorageModeShared deallocator:nil];
        if (!BG[i] || !BU[i] || !BD[i]) { rc = -2; break; }
    }
    id<MTLBuffer> RWV  = [b->dev newBufferWithBytes:rws length:(NSUInteger)n*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> PBV  = [b->dev newBufferWithBytes:PV length:sizeof(PV) options:MTLResourceStorageModeShared];
    id<MTLBuffer> CF   = [b->dev newBufferWithBytes:&clamp length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> PSUM = nil;
    if (rc == 0 && (!RWV || !PBV || !CF)) rc = -2;
    { uint32_t P[4] = { ff / 256u, down_nb01, embd, ff };
      PSUM = [b->dev newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
      if (!PSUM) rc = -2; }
    id<MTLBuffer> WG = nil, WU = nil, WD = nil, G = nil, U = nil, MID2 = nil, D = nil;
    uint32_t lg_g = 0, lu_g = 0, ld_g = 0;
    if (sg && su && sd) {
        size_t sbg = (size_t)ff * s_nb_g, sbd = (size_t)embd * s_nb_d;
        uintptr_t apg = (uintptr_t)sg & ~(uintptr_t)(page-1); lg_g = (uint32_t)((uintptr_t)sg-apg);
        uintptr_t apu = (uintptr_t)su & ~(uintptr_t)(page-1); lu_g = (uint32_t)((uintptr_t)su-apu);
        uintptr_t apd = (uintptr_t)sd & ~(uintptr_t)(page-1); ld_g = (uint32_t)((uintptr_t)sd-apd);
        size_t vlg = ((lg_g+sbg+page-1)/page)*page, vlu = ((lu_g+sbg+page-1)/page)*page, vld = ((ld_g+sbd+page-1)/page)*page;
        WG = [b->dev newBufferWithBytesNoCopy:(void*)apg length:vlg options:MTLResourceStorageModeShared deallocator:nil];
        WU = [b->dev newBufferWithBytesNoCopy:(void*)apu length:vlu options:MTLResourceStorageModeShared deallocator:nil];
        WD = [b->dev newBufferWithBytesNoCopy:(void*)apd length:vld options:MTLResourceStorageModeShared deallocator:nil];
        G    = dsv4_scratch(b, &b->sx[15], &b->sx_cap[15], (size_t)ff * sizeof(float));
        U    = dsv4_scratch(b, &b->sx[16], &b->sx_cap[16], (size_t)ff * sizeof(float));
        MID2 = dsv4_scratch(b, &b->sx[17], &b->sx_cap[17], (size_t)ff * sizeof(float));
        D    = dsv4_scratch(b, &b->sx[18], &b->sx_cap[18], (size_t)embd * sizeof(float));
        if (!WG || !WU || !WD || !G || !U || !MID2 || !D) rc = -2;
      }
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const uint32_t NSG = 4, nr0 = 4;
        [e setComputePipelineState:b->pso_slots6];
        for (int i = 0; i < 6; i++) [e setBuffer:BG[i] offset:0 atIndex:i];
        for (int i = 0; i < 6; i++) [e setBuffer:BU[i] offset:0 atIndex:6+i];
        [e setBuffer:X   offset:0 atIndex:12];
        [e setBuffer:MID offset:0 atIndex:13];
        [e setBuffer:RWV offset:0 atIndex:14];
        [e setBuffer:CF  offset:0 atIndex:15];
        [e setBuffer:PBV offset:0 atIndex:16];
        [e setThreadgroupMemoryLength:256 * 8 + 128 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((ff + NSG*nr0 - 1) / (NSG*nr0), 1, 6)
         threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
        [e setComputePipelineState:b->pso_sum6_q2k];
        [e setBuffer:BD[0] offset:0 atIndex:0];
        [e setBuffer:BD[1] offset:0 atIndex:1];
        [e setBuffer:BD[2] offset:0 atIndex:2];
        [e setBuffer:BD[3] offset:0 atIndex:3];
        [e setBuffer:BD[4] offset:0 atIndex:4];
        [e setBuffer:BD[5] offset:0 atIndex:5];
        [e setBuffer:MID offset:0 atIndex:6];
        [e setBuffer:OUT offset:0 atIndex:7];
        [e setBuffer:PSUM offset:0 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake((embd + NSG*nr0 - 1) / (NSG*nr0), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
        if (WG) {
            /* shared expert su GPU: gate/up (Q8_0) + swiglu + down + OUT += D */
            [e setComputePipelineState:b->pso];
            [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:X offset:0 atIndex:1]; [e setBuffer:G offset:0 atIndex:2];
            { uint32_t P[7]={lg_g,s_nb_g,embd/32u,ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e setComputePipelineState:b->pso];
            [e setBuffer:WU offset:0 atIndex:0]; [e setBuffer:X offset:0 atIndex:1]; [e setBuffer:U offset:0 atIndex:2];
            { uint32_t P[7]={lu_g,s_nb_g,embd/32u,ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e setComputePipelineState:b->pso_swiglu];
            [e setBuffer:G offset:0 atIndex:0]; [e setBuffer:U offset:0 atIndex:1]; [e setBuffer:MID2 offset:0 atIndex:2];
            { uint32_t P[1]={ff}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={clamp}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake((ff+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [e setComputePipelineState:b->pso];
            [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:D offset:0 atIndex:2];
            { uint32_t P[7]={ld_g,s_nb_d,ff/32u,embd,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(embd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e setComputePipelineState:b->pso_add];
            [e setBuffer:OUT offset:0 atIndex:0]; [e setBuffer:D offset:0 atIndex:1];
            { uint32_t P[1]={embd}; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreadgroups:MTLSizeMake((embd+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(out, [OUT contents], (size_t)embd * sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        if (BG[i]) [BG[i] release]; if (BU[i]) [BU[i] release]; if (BD[i]) [BD[i] release];
    }
    if (RWV) [RWV release];
    if (PBV) [PBV release];
    if (CF) [CF release];
    if (PSUM) [PSUM release];
    if (WG) [WG release]; if (WU) [WU release]; if (WD) [WD release];
    return rc;
}

/* Output grouped-LoRA FUSO (ds4): out_a (ng gruppi) + out_b in UNA command buffer, con `low`
 * intermedio su DEVICE (nessun readback/upload). Pesi Q8_0. `out[embd]` scritto per intero. 0 = ok. */
int llm_dsv4_metal_attn_out_fused(LlmDsv4Metal* b,
        const void* outa_w, size_t outa_row_bytes, size_t outa_lead0,
        int ng, uint32_t gd, uint32_t rank,
        const void* outb_w, size_t outb_row_bytes,
        uint32_t ol, uint32_t embd, uint32_t heads_total,
        const float* heads, float* out) {
    if (!b || !b->pso || ng <= 0 || ng > 64 || !outa_w || !outb_w || !heads || !out) return -1;
    const size_t page = (size_t)getpagesize();
    size_t abytes = (size_t)ng * rank * outa_row_bytes;
    size_t bbytes = (size_t)embd * outb_row_bytes;
    uintptr_t apa = (uintptr_t)outa_w & ~(uintptr_t)(page-1); size_t la = (size_t)((uintptr_t)outa_w-apa);
    uintptr_t apb = (uintptr_t)outb_w & ~(uintptr_t)(page-1); size_t lb = (size_t)((uintptr_t)outb_w-apb);
    size_t vla = ((la+abytes+page-1)/page)*page;
    size_t vlb = ((lb+bbytes+page-1)/page)*page;
    id<MTLBuffer> WA  = [b->dev newBufferWithBytesNoCopy:(void*)apa length:vla options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> WB  = [b->dev newBufferWithBytesNoCopy:(void*)apb length:vlb options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> H   = dsv4_scratch(b, &b->sx[3], &b->sx_cap[3], (size_t)heads_total * sizeof(float));
    id<MTLBuffer> LOW = dsv4_scratch(b, &b->sx[4], &b->sx_cap[4], (size_t)ol * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[5], &b->sx_cap[5], (size_t)embd * sizeof(float));
    int rc = (WA && WB && H && LOW && OUT) ? 0 : -2;
    if (rc == 0) memcpy([H contents], heads, (size_t)heads_total * sizeof(float));
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int g = 0; g < ng; g++) {
            [e setComputePipelineState:b->pso];
            [e setBuffer:WA offset:0 atIndex:0];
            [e setBuffer:H  offset:(NSUInteger)((size_t)g * gd * sizeof(float)) atIndex:1];
            [e setBuffer:LOW offset:(NSUInteger)((size_t)g * rank * sizeof(float)) atIndex:2];
            uint32_t P[7] = { (uint32_t)(la + outa_lead0 + (size_t)g * rank * outa_row_bytes), (uint32_t)outa_row_bytes,
                              (uint32_t)(gd / 32u), rank, 8u, 34u, 32u };
            [e setBytes:P length:sizeof(P) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [e setComputePipelineState:b->pso];
        [e setBuffer:WB  offset:0 atIndex:0];
        [e setBuffer:LOW offset:0 atIndex:1];
        [e setBuffer:OUT offset:0 atIndex:2];
        uint32_t P2[7] = { (uint32_t)lb, (uint32_t)outb_row_bytes, (uint32_t)(ol / 32u), embd, 8u, 34u, 32u };
        [e setBytes:P2 length:sizeof(P2) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(embd, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(out, [OUT contents], (size_t)embd * sizeof(float));
    }
    if (WA) [WA release];
    if (WB) [WB release];
    return rc;
}

/* Proiezioni Q-LoRA + KV FUSE (ds4): q_a, kv, RMS, q_b, head_rms, RoPE in UNA command buffer.
 * Pesi Q8_0. `q_out[qd]`, `kv_out[hd]`, `qrn_out[lq]` (per l'indexer). 0 = ok. */
int llm_dsv4_metal_attn_proj_fused(LlmDsv4Metal* b,
        const void* qa_w, size_t qa_row_bytes,
        const void* kv_w, size_t kv_row_bytes,
        const void* qb_w, size_t qb_row_bytes,
        const float* xn, uint32_t embd,
        const float* qan_w, const float* kvan_w, uint32_t lq, uint32_t hd, uint32_t qd,
        int n_head, int n_rot, const float* rope_cfg, int pos, float eps,
        float* q_out, float* kv_out, float* qrn_out) {
    if (!b || !b->pso || !b->pso_rms_w || !b->pso_head_rms || !b->pso_rope) return -1;
    if (!qa_w || !kv_w || !qb_w || !xn || !qan_w || !kvan_w || !q_out || !kv_out || !qrn_out) return -1;
    if ((lq % 32u) || (qd % 32u) || (embd % 32u) || n_head <= 0) return -1;
    const size_t page = (size_t)getpagesize();
    size_t qasz = (size_t)lq * qa_row_bytes, kvsz = (size_t)hd * kv_row_bytes, qbsz = (size_t)qd * qb_row_bytes;
    uintptr_t ap_qa = (uintptr_t)qa_w & ~(uintptr_t)(page-1); size_t l_qa = (size_t)((uintptr_t)qa_w-ap_qa);
    uintptr_t ap_kv = (uintptr_t)kv_w & ~(uintptr_t)(page-1); size_t l_kv = (size_t)((uintptr_t)kv_w-ap_kv);
    uintptr_t ap_qb = (uintptr_t)qb_w & ~(uintptr_t)(page-1); size_t l_qb = (size_t)((uintptr_t)qb_w-ap_qb);
    size_t vl_qa = ((l_qa+qasz+page-1)/page)*page, vl_kv = ((l_kv+kvsz+page-1)/page)*page, vl_qb = ((l_qb+qbsz+page-1)/page)*page;
    id<MTLBuffer> WA  = [b->dev newBufferWithBytesNoCopy:(void*)ap_qa length:vl_qa options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> WK  = [b->dev newBufferWithBytesNoCopy:(void*)ap_kv length:vl_kv options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> WQ  = [b->dev newBufferWithBytesNoCopy:(void*)ap_qb length:vl_qb options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> WQAN = [b->dev newBufferWithBytes:qan_w length:(NSUInteger)lq*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> WKAN = [b->dev newBufferWithBytes:kvan_w length:(NSUInteger)hd*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[6],  &b->sx_cap[6],  (size_t)embd*sizeof(float));
    id<MTLBuffer> QR  = dsv4_scratch(b, &b->sx[7],  &b->sx_cap[7],  (size_t)lq*sizeof(float));
    id<MTLBuffer> KVR = dsv4_scratch(b, &b->sx[8],  &b->sx_cap[8],  (size_t)hd*sizeof(float));
    id<MTLBuffer> QB  = dsv4_scratch(b, &b->sx[9],  &b->sx_cap[9],  (size_t)qd*sizeof(float));
    id<MTLBuffer> QRN = dsv4_scratch(b, &b->sx[10], &b->sx_cap[10], (size_t)lq*sizeof(float));
    int rc = (WA && WK && WQ && WQAN && WKAN && X && QR && KVR && QB && QRN) ? 0 : -2;
    if (rc == 0) memcpy([X contents], xn, (size_t)embd*sizeof(float));
    if (rc == 0) {
        const uint32_t hdim = (uint32_t)(qd / (uint32_t)n_head);
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        /* gemv q_a: X -> QR */
        [en setComputePipelineState:b->pso];
        [en setBuffer:WA offset:0 atIndex:0]; [en setBuffer:X offset:0 atIndex:1]; [en setBuffer:QR offset:0 atIndex:2];
        { uint32_t P[7]={(uint32_t)l_qa,(uint32_t)qa_row_bytes,embd/32u,lq,8u,34u,32u}; [en setBytes:P length:sizeof(P) atIndex:3]; }
        [en dispatchThreadgroups:MTLSizeMake(lq,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* gemv kv: X -> KVR */
        [en setComputePipelineState:b->pso];
        [en setBuffer:WK offset:0 atIndex:0]; [en setBuffer:X offset:0 atIndex:1]; [en setBuffer:KVR offset:0 atIndex:2];
        { uint32_t P[7]={(uint32_t)l_kv,(uint32_t)kv_row_bytes,embd/32u,hd,8u,34u,32u}; [en setBytes:P length:sizeof(P) atIndex:3]; }
        [en dispatchThreadgroups:MTLSizeMake(hd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* rms q_a_norm: QRN = rms(QR)*qan */
        [en setComputePipelineState:b->pso_rms_w];
        [en setBuffer:QR offset:0 atIndex:0]; [en setBuffer:WQAN offset:0 atIndex:1]; [en setBuffer:QRN offset:0 atIndex:2];
        { uint32_t P[1]={lq}; [en setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={eps}; [en setBytes:cf length:sizeof(cf) atIndex:4]; }
        [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* gemv q_b: QRN -> QB */
        [en setComputePipelineState:b->pso];
        [en setBuffer:WQ offset:0 atIndex:0]; [en setBuffer:QRN offset:0 atIndex:1]; [en setBuffer:QB offset:0 atIndex:2];
        { uint32_t P[7]={(uint32_t)l_qb,(uint32_t)qb_row_bytes,lq/32u,qd,8u,34u,32u}; [en setBytes:P length:sizeof(P) atIndex:3]; }
        [en dispatchThreadgroups:MTLSizeMake(qd,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* head_rms Q */
        [en setComputePipelineState:b->pso_head_rms];
        [en setBuffer:QB offset:0 atIndex:0];
        { uint32_t P[1]={hdim}; [en setBytes:P length:sizeof(P) atIndex:1]; float cf[1]={eps}; [en setBytes:cf length:sizeof(cf) atIndex:2]; }
        [en dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* rope Q */
        [en setComputePipelineState:b->pso_rope];
        [en setBuffer:QB offset:0 atIndex:0];
        { uint32_t P[4]={(uint32_t)n_rot,hdim,(uint32_t)pos,0u}; [en setBytes:P length:sizeof(P) atIndex:1]; }
        [en setBytes:rope_cfg length:7*sizeof(float) atIndex:2];
        [en dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* rms kv_a_norm in-place */
        [en setComputePipelineState:b->pso_rms_w];
        [en setBuffer:KVR offset:0 atIndex:0]; [en setBuffer:WKAN offset:0 atIndex:1]; [en setBuffer:KVR offset:0 atIndex:2];
        { uint32_t P[1]={hd}; [en setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={eps}; [en setBytes:cf length:sizeof(cf) atIndex:4]; }
        [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* rope KV (1 head) */
        [en setComputePipelineState:b->pso_rope];
        [en setBuffer:KVR offset:0 atIndex:0];
        { uint32_t P[4]={(uint32_t)n_rot,hd,(uint32_t)pos,0u}; [en setBytes:P length:sizeof(P) atIndex:1]; }
        [en setBytes:rope_cfg length:7*sizeof(float) atIndex:2];
        [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        rc = cb.error ? -3 : 0;
        if (rc == 0) {
            memcpy(q_out,   [QB contents],  (size_t)qd*sizeof(float));
            memcpy(kv_out,  [KVR contents], (size_t)hd*sizeof(float));
            memcpy(qrn_out, [QRN contents], (size_t)lq*sizeof(float));
        }
    }
    if (WA) [WA release]; if (WK) [WK release]; if (WQ) [WQ release];
    if (WQAN) [WQAN release]; if (WKAN) [WKAN release];
    return rc;
}

/* Upload di un blocco host in un buffer device (ritorna MTLBuffer "retained", da rilasciare). */
void* llm_dsv4_metal_upload(LlmDsv4Metal* b, const void* data, size_t bytes) {
    if (!b || !data || !bytes) return NULL;
    return (void*)[b->dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}
void llm_dsv4_metal_buffer_release(void* h) { if (h) [(id<MTLBuffer>)h release]; }

/* Vista zero-copy pagina-allineata su un blocco pesi (f32 o packed). Ritorna MTLBuffer retained. */
static id<MTLBuffer> dsv4_view(LlmDsv4Metal* b, const void* p, size_t bytes, size_t* out_lead) {
    const size_t page = (size_t)getpagesize();
    uintptr_t ap = (uintptr_t)p & ~(uintptr_t)(page-1);
    size_t lead = (size_t)((uintptr_t)p - ap);
    size_t vl = ((lead + bytes + page - 1) / page) * page;
    if (out_lead) *out_lead = lead;
    return [b->dev newBufferWithBytesNoCopy:(void*)ap length:vl options:MTLResourceStorageModeShared deallocator:nil];
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
        !q->scale || !q->base || !q->norm_w || !q->router_w || !q->gaddr || !q->uaddr || !q->daddr) return -1;
    if ((q->hd % 1) || (D % 32u)) return -1;

    id<MTLBuffer> HEADS = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->heads, (size_t)q->n_head*q->hd*sizeof(float));
    id<MTLBuffer> RES   = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->res_attn, (size_t)HC*D*sizeof(float));
    id<MTLBuffer> POST  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->post_attn, (size_t)HC*sizeof(float));
    id<MTLBuffer> COMB  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->comb_attn, (size_t)HC*HC*sizeof(float));
    id<MTLBuffer> FN    = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->fn, (size_t)SIZE*HCD*sizeof(float));
    id<MTLBuffer> SCALE = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->scale, 3*sizeof(float));
    id<MTLBuffer> BASE  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->base, (size_t)SIZE*sizeof(float));
    id<MTLBuffer> NORMW = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->norm_w, (size_t)D*sizeof(float));
    id<MTLBuffer> GADR  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->gaddr, (size_t)q->n_expert*sizeof(uint64_t));
    id<MTLBuffer> UADR  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->uaddr, (size_t)q->n_expert*sizeof(uint64_t));
    id<MTLBuffer> DADR  = (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->daddr, (size_t)q->n_expert*sizeof(uint64_t));
    id<MTLBuffer> EXPB  = q->exp_probs_b ? (id<MTLBuffer>)llm_dsv4_metal_upload(b, q->exp_probs_b, (size_t)q->n_expert*sizeof(float)) : nil;
    size_t rtw_lead = 0;
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

    int rc = (HEADS&&RES&&POST&&COMB&&FN&&SCALE&&BASE&&NORMW&&GADR&&UADR&&DADR&&RTW&&OUTA&&OUTB&&
              ATTN&&NXT&&FLAT&&MIXES&&SPLIT&&HIN&&NRM&&LOGI&&SEL&&RW&&MID&&ROUT&&OUTC&&HLOW&&
              (q->sg?(WG&&WU&&WD&&G&&U&&MID2&&D2):1)) ? 0 : -2;

    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const uint32_t NSG = 4, nr0 = 4;
        /* 1) RoPE inversa su heads (n_head) */
        [e setComputePipelineState:b->pso_rope];
        [e setBuffer:HEADS offset:0 atIndex:0];
        { uint32_t P[4]={(uint32_t)q->n_rot,(uint32_t)q->hd,(uint32_t)q->pos,1u}; [e setBytes:P length:sizeof(P) atIndex:1]; }
        [e setBytes:q->rope_cfg length:7*sizeof(float) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)q->n_head,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* 2) out_a (ng) + out_b -> ATTN */
        for (int g = 0; g < q->ng; g++) {
            [e setComputePipelineState:b->pso];
            [e setBuffer:OUTA offset:0 atIndex:0];
            [e setBuffer:HEADS offset:(NSUInteger)((size_t)g*q->gd*sizeof(float)) atIndex:1];
            [e setBuffer:HLOW offset:(NSUInteger)((size_t)g*q->rank*sizeof(float)) atIndex:2];
            uint32_t P[7]={(uint32_t)(oa_lead+(size_t)g*q->rank*q->outa_row_bytes),(uint32_t)q->outa_row_bytes,(uint32_t)(q->gd/32u),(uint32_t)q->rank,8u,34u,32u};
            [e setBytes:P length:sizeof(P) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(q->rank,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        [e setComputePipelineState:b->pso];
        [e setBuffer:OUTB offset:0 atIndex:0]; [e setBuffer:HLOW offset:0 atIndex:1]; [e setBuffer:ATTN offset:0 atIndex:2];
        { uint32_t P[7]={(uint32_t)ob_lead,(uint32_t)q->outb_row_bytes,(uint32_t)(q->ol/32u),D,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* 3) hc_post_attn: NXT = post*ATTN + comb*RES */
        [e setComputePipelineState:b->pso_hc_post];
        [e setBuffer:ATTN offset:0 atIndex:0]; [e setBuffer:RES offset:0 atIndex:1]; [e setBuffer:POST offset:0 atIndex:2];
        [e setBuffer:COMB offset:0 atIndex:3]; [e setBuffer:NXT offset:0 atIndex:4];
        { uint32_t P[2]={D,HC}; [e setBytes:P length:sizeof(P) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake((HC*D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        /* 4) mhc_pre_ffn: rms_no_w(NXT)->FLAT; gemv_f32(fn)->MIXES; sinkhorn->SPLIT; hc_sum->HIN; rms_w->NRM */
        [e setComputePipelineState:b->pso_rms_no_w];
        [e setBuffer:NXT offset:0 atIndex:0]; [e setBuffer:FLAT offset:0 atIndex:1];
        { uint32_t P[1]={HCD}; [e setBytes:P length:sizeof(P) atIndex:2]; float cf[1]={q->rms_eps}; [e setBytes:cf length:sizeof(cf) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e setComputePipelineState:b->pso_gemv_f32];
        [e setBuffer:FN offset:0 atIndex:0]; [e setBuffer:FLAT offset:0 atIndex:1]; [e setBuffer:MIXES offset:0 atIndex:2];
        { uint32_t P[1]={HCD}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake(SIZE,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e setComputePipelineState:b->pso_sinkhorn];
        [e setBuffer:MIXES offset:0 atIndex:0]; [e setBuffer:SCALE offset:0 atIndex:1]; [e setBuffer:BASE offset:0 atIndex:2]; [e setBuffer:SPLIT offset:0 atIndex:3];
        { uint32_t P[2]={HC,(uint32_t)q->sinkhorn_iters}; [e setBytes:P length:sizeof(P) atIndex:4]; float cf[1]={q->hc_eps}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e setComputePipelineState:b->pso_hc_sum];
        [e setBuffer:NXT offset:0 atIndex:0]; [e setBuffer:SPLIT offset:0 atIndex:1]; [e setBuffer:HIN offset:0 atIndex:2];
        { uint32_t P[2]={D,HC}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake((D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e setComputePipelineState:b->pso_rms_w];
        [e setBuffer:HIN offset:0 atIndex:0]; [e setBuffer:NORMW offset:0 atIndex:1]; [e setBuffer:NRM offset:0 atIndex:2];
        { uint32_t P[1]={D}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={q->rms_eps}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        /* 5) route: logits=gemv_f32(router,NRM); SEL/RW */
        [e setComputePipelineState:b->pso_gemv_f32];
        [e setBuffer:RTW offset:0 atIndex:0]; [e setBuffer:NRM offset:0 atIndex:1]; [e setBuffer:LOGI offset:0 atIndex:2];
        { uint32_t P[1]={D}; [e setBytes:P length:sizeof(P) atIndex:3]; }
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)q->n_expert,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e setComputePipelineState:b->pso_route];
        [e setBuffer:LOGI offset:0 atIndex:0]; [e setBuffer:(EXPB?EXPB:SCALE) offset:0 atIndex:1]; [e setBuffer:SEL offset:0 atIndex:2]; [e setBuffer:RW offset:0 atIndex:3];
        { uint32_t P[3]={(uint32_t)q->n_expert,(uint32_t)q->used,(EXPB?1u:0u)}; [e setBytes:P length:sizeof(P) atIndex:4]; float cf[1]={q->expert_scale}; [e setBytes:cf length:sizeof(cf) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        /* 6) MoE addr: pair slots6 (grid z=used) + sum6 -> ROUT */
        [e setComputePipelineState:b->pso_addr_pair];
        [e setBuffer:SEL offset:0 atIndex:0]; [e setBuffer:GADR offset:0 atIndex:1]; [e setBuffer:UADR offset:0 atIndex:2];
        [e setBuffer:NRM offset:0 atIndex:3]; [e setBuffer:MID offset:0 atIndex:4]; [e setBuffer:RW offset:0 atIndex:5];
        { float cf[1]={q->clamp}; [e setBytes:cf length:sizeof(cf) atIndex:6]; }
        { uint32_t P[3]={D,q->gu_nb01,q->ff}; [e setBytes:P length:sizeof(P) atIndex:7]; }
        [e setThreadgroupMemoryLength:256*8+128 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((q->ff+NSG*nr0-1)/(NSG*nr0),1,(NSUInteger)q->used) threadsPerThreadgroup:MTLSizeMake(32*NSG,1,1)];
        [e setComputePipelineState:b->pso_addr_sum6];
        [e setBuffer:SEL offset:0 atIndex:0]; [e setBuffer:DADR offset:0 atIndex:1]; [e setBuffer:MID offset:0 atIndex:2]; [e setBuffer:ROUT offset:0 atIndex:3];
        { uint32_t P[4]={q->ff/256u,q->down_nb01,D,q->ff}; [e setBytes:P length:sizeof(P) atIndex:4]; }
        [e dispatchThreadgroups:MTLSizeMake((D+NSG*nr0-1)/(NSG*nr0),1,1) threadsPerThreadgroup:MTLSizeMake(32*NSG,1,1)];
        /* shared expert -> ROUT += D2 */
        if (q->sg) {
            [e setComputePipelineState:b->pso];
            [e setBuffer:WG offset:0 atIndex:0]; [e setBuffer:NRM offset:0 atIndex:1]; [e setBuffer:G offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)sg_lead,q->s_nb_g,D/32u,q->ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(q->ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e setComputePipelineState:b->pso];
            [e setBuffer:WU offset:0 atIndex:0]; [e setBuffer:NRM offset:0 atIndex:1]; [e setBuffer:U offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)su_lead,q->s_nb_g,D/32u,q->ff,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(q->ff,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e setComputePipelineState:b->pso_swiglu];
            [e setBuffer:G offset:0 atIndex:0]; [e setBuffer:U offset:0 atIndex:1]; [e setBuffer:MID2 offset:0 atIndex:2];
            { uint32_t P[1]={q->ff}; [e setBytes:P length:sizeof(P) atIndex:3]; float cf[1]={q->clamp}; [e setBytes:cf length:sizeof(cf) atIndex:4]; }
            [e dispatchThreadgroups:MTLSizeMake((q->ff+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [e setComputePipelineState:b->pso];
            [e setBuffer:WD offset:0 atIndex:0]; [e setBuffer:MID2 offset:0 atIndex:1]; [e setBuffer:D2 offset:0 atIndex:2];
            { uint32_t P[7]={(uint32_t)sd_lead,q->s_nb_d,q->ff/32u,D,8u,34u,32u}; [e setBytes:P length:sizeof(P) atIndex:3]; }
            [e dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e setComputePipelineState:b->pso_add];
            [e setBuffer:ROUT offset:0 atIndex:0]; [e setBuffer:D2 offset:0 atIndex:1];
            { uint32_t P[1]={D}; [e setBytes:P length:sizeof(P) atIndex:2]; }
            [e dispatchThreadgroups:MTLSizeMake((D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        }
        /* 7) hc_post_ffn: OUTC = routed-post + comb*NXT  (post/comb = SPLIT[HC..2HC), SPLIT[2HC..)) */
        [e setComputePipelineState:b->pso_hc_post];
        [e setBuffer:ROUT offset:0 atIndex:0]; [e setBuffer:NXT offset:0 atIndex:1];
        [e setBuffer:SPLIT offset:(NSUInteger)((size_t)HC*sizeof(float)) atIndex:2];
        [e setBuffer:SPLIT offset:(NSUInteger)((size_t)2*HC*sizeof(float)) atIndex:3];
        [e setBuffer:OUTC offset:0 atIndex:4];
        { uint32_t P[2]={D,HC}; [e setBytes:P length:sizeof(P) atIndex:5]; }
        [e dispatchThreadgroups:MTLSizeMake((HC*D+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(q->out_hc, [OUTC contents], (size_t)HC*D*sizeof(float));
    }

    if (HEADS) [HEADS release]; if (RES) [RES release]; if (POST) [POST release]; if (COMB) [COMB release];
    if (FN) [FN release]; if (SCALE) [SCALE release]; if (BASE) [BASE release]; if (NORMW) [NORMW release];
    if (GADR) [GADR release]; if (UADR) [UADR release]; if (DADR) [DADR release]; if (EXPB) [EXPB release];
    if (RTW) [RTW release]; if (OUTA) [OUTA release]; if (OUTB) [OUTB release];
    if (WG) [WG release]; if (WU) [WU release]; if (WD) [WD release];
    return rc;
}
