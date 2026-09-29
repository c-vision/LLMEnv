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
"kernel void dsv4_pair_swiglu_iq2(\n"
"    device const char* SG [[buffer(0)]], device const char* SU [[buffer(1)]],\n"
"    device const float* Y [[buffer(2)]], device float* MID [[buffer(3)]],\n"
"    device const float* RW [[buffer(4)]], constant float* CF [[buffer(5)]],\n"
"    constant uint* P [[buffer(6)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]],\n"
"    threadgroup char* shmem [[threadgroup(0)]]) {\n"
"    const int nr0 = 4; const short NSG = 4;\n"
"    const int ne00 = (int)P[0], nb01 = (int)P[1], ne0 = (int)P[2];\n"
"    const uint leadg = P[3], leadu = P[4]; const int mid_off = (int)P[5];\n"
"    const int nb = ne00 / 256; const int first_row = (tgpig.x * NSG + sgitg) * nr0;\n"
"    const float clampv = CF[0];\n"
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
"            MID[mid_off + first_row + row] = silu * uu * RW[0];\n"
"        }\n"
"    }\n"
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
    LlmDsv4Metal* b = (LlmDsv4Metal*)calloc(1, sizeof *b);
    b->dev = [dev retain];
    b->queue = [dev newCommandQueue];
    b->pso = [dev newComputePipelineStateWithFunction:f1 error:&err];
    b->pso_q8k_iq2 = [dev newComputePipelineStateWithFunction:f2 error:&err];
    b->pso_q8k_q2k = [dev newComputePipelineStateWithFunction:f3 error:&err];
    b->pso_q8k_q4k = [dev newComputePipelineStateWithFunction:f4 error:&err];
    b->pso_pair    = f5 ? [dev newComputePipelineStateWithFunction:f5 error:&err] : nil;
    b->pso_sum6_q2k = f6 ? [dev newComputePipelineStateWithFunction:f6 error:&err] : nil;
    [lib release];
    if (!b->pso || !b->pso_q8k_iq2 || !b->pso_q8k_q2k || !b->pso_q8k_q4k || !b->pso_pair || !b->pso_sum6_q2k || !b->queue) {
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
                               uint32_t down_nb01, float* out) {
    if (!b || !b->pso_pair || !b->pso_sum6_q2k || n != 6 || !gates || !ups || !downs || !x || !rws || !out) return -1;
    if ((ff % 256u) != 0u) return -1;
    const size_t page = (size_t)getpagesize();
    id<MTLBuffer> X   = dsv4_scratch(b, &b->sx[0], &b->sx_cap[0], (size_t)embd * sizeof(float));
    id<MTLBuffer> MID = dsv4_scratch(b, &b->sx[1], &b->sx_cap[1], (size_t)n * (size_t)ff * sizeof(float));
    id<MTLBuffer> OUT = dsv4_scratch(b, &b->sx[2], &b->sx_cap[2], (size_t)embd * sizeof(float));
    if (!X || !MID || !OUT) return -2;
    memcpy([X contents], x, (size_t)embd * sizeof(float));
    id<MTLBuffer> BG[6], BU[6], BD[6], RW[6], PB[6];
    for (int i = 0; i < 6; i++) { BG[i] = BU[i] = BD[i] = RW[i] = PB[i] = nil; }
    id<MTLBuffer> CF = [b->dev newBufferWithBytes:&clamp length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> PSUM = nil;
    int rc = (CF == nil) ? -2 : 0;
    for (int i = 0; i < n; i++) {
        size_t gbytes = (size_t)ff * gu_nb01, dbytes = (size_t)embd * down_nb01;
        uintptr_t apg = (uintptr_t)gates[i] & ~(uintptr_t)(page-1); size_t lg = (size_t)((uintptr_t)gates[i]-apg);
        uintptr_t apu = (uintptr_t)ups[i]   & ~(uintptr_t)(page-1); size_t lu = (size_t)((uintptr_t)ups[i]-apu);
        uintptr_t apd = (uintptr_t)downs[i] & ~(uintptr_t)(page-1); size_t ld = (size_t)((uintptr_t)downs[i]-apd);
        size_t vlg = ((lg+gbytes+page-1)/page)*page, vlu = ((lu+gbytes+page-1)/page)*page, vld = ((ld+dbytes+page-1)/page)*page;
        BG[i] = [b->dev newBufferWithBytesNoCopy:(void*)apg length:vlg options:MTLResourceStorageModeShared deallocator:nil];
        BU[i] = [b->dev newBufferWithBytesNoCopy:(void*)apu length:vlu options:MTLResourceStorageModeShared deallocator:nil];
        BD[i] = [b->dev newBufferWithBytesNoCopy:(void*)apd length:vld options:MTLResourceStorageModeShared deallocator:nil];
        float rw = rws[i];
        RW[i] = [b->dev newBufferWithBytes:&rw length:sizeof(float) options:MTLResourceStorageModeShared];
        uint32_t P[6] = { embd, gu_nb01, ff, (uint32_t)lg, (uint32_t)lu, (uint32_t)((size_t)i * ff) };
        PB[i] = [b->dev newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
        if (!BG[i] || !BU[i] || !BD[i] || !RW[i] || !PB[i]) { rc = -2; break; }
    }
    { uint32_t P[4] = { ff / 256u, down_nb01, embd, ff };
      PSUM = [b->dev newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
      if (!PSUM) rc = -2; }
    if (rc == 0) {
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const uint32_t NSG = 4, nr0 = 4;
        for (int i = 0; i < n; i++) {
            [e setComputePipelineState:b->pso_pair];
            [e setBuffer:BG[i] offset:0 atIndex:0];
            [e setBuffer:BU[i] offset:0 atIndex:1];
            [e setBuffer:X offset:0 atIndex:2];
            [e setBuffer:MID offset:0 atIndex:3];
            [e setBuffer:RW[i] offset:0 atIndex:4];
            [e setBuffer:CF offset:0 atIndex:5];
            [e setBuffer:PB[i] offset:0 atIndex:6];
            [e setThreadgroupMemoryLength:256 * 8 + 128 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((ff + NSG*nr0 - 1) / (NSG*nr0), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
        }
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
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        rc = cb.error ? -3 : 0;
        if (rc == 0) memcpy(out, [OUT contents], (size_t)embd * sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        if (BG[i]) [BG[i] release]; if (BU[i]) [BU[i] release]; if (BD[i]) [BD[i] release];
        if (RW[i]) [RW[i] release]; if (PB[i]) [PB[i] release];
    }
    if (CF) [CF release];
    if (PSUM) [PSUM release];
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
