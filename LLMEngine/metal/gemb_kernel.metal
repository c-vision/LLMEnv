#include <metal_stdlib>
using namespace metal;

// =====================================================================================
// GEMV quantizzata 4-bit affine (group 64) — decode m=1.
//   w      : pesi packed, 8 nibble per uint32, shape [N, K/8]   (uint)
//   scales : per-gruppo (K/64), shape [N, K/64]                  (float)
//   biases : per-gruppo (K/64), shape [N, K/64]                  (float)
//   x      : attivazione [K]                                     (float)
//   out    : risultato [N]                                       (float)
// Dequant affine: w = nib*s + b, poi dot con x.
// =====================================================================================

#define TGTHREADS 256

// ---- qgemv4s : VINCITORE. 1 simdgroup (32 lane) per riga, 8 righe per threadgroup.
// Carichi scalari per-uint ma riduzione con simd_sum (niente barrier per-riga).
kernel void qgemv4s(device const uint*  w      [[buffer(0)]],
                    device const float* scales [[buffer(1)]],
                    device const float* biases [[buffer(2)]],
                    device const float* x      [[buffer(3)]],
                    device float*       out    [[buffer(4)]],
                    constant int& N [[buffer(5)]],
                    constant int& K [[buffer(6)]],
                    uint  tgid [[threadgroup_position_in_grid]],
                    uint  tid  [[thread_position_in_threadgroup]],
                    uint  lane [[thread_index_in_simdgroup]],
                    uint  sid  [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgid*8 + (int)sid; if (n >= N) return;
    int Kw = K >> 3;
    const device uint*  wr = w      + (size_t)n*Kw;
    const device float* sc = scales + (size_t)n*(K >> 6);
    const device float* bs = biases + (size_t)n*(K >> 6);
    float acc = 0.0f;
    for (int u = (int)lane; u < Kw; u += 32) {
        uint pk = wr[u];
        int base = u << 3, g = base >> 6;
        float s = sc[g], b = bs[g];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float((pk >> (j << 2)) & 0xF) * s + b) * x[base + j];
    }
    float tot = simd_sum(acc);
    if (lane == 0) out[n] = tot;
}

// ---- qgemv4 : 1 threadgroup intero per riga, riduzione threadgroup (reference).
kernel void qgemv4(device const uint*  w      [[buffer(0)]],
                   device const float* scales [[buffer(1)]],
                   device const float* biases [[buffer(2)]],
                   device const float* x      [[buffer(3)]],
                   device float*       out    [[buffer(4)]],
                   constant int& N [[buffer(5)]],
                   constant int& K [[buffer(6)]],
                   uint  tgid [[threadgroup_position_in_grid]],
                   uint  tid  [[thread_position_in_threadgroup]],
                   uint  lane [[thread_index_in_simdgroup]],
                   uint  sid  [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgid; if (n >= N) return;
    int Kw = K >> 3;
    const device uint*  wr = w      + (size_t)n*Kw;
    const device float* sc = scales + (size_t)n*(K >> 6);
    const device float* bs = biases + (size_t)n*(K >> 6);
    float acc = 0.0f;
    for (int u = (int)tid; u < Kw; u += TGTHREADS) {
        uint pk = wr[u];
        int base = u << 3, g = base >> 6;
        float s = sc[g], b = bs[g];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float((pk >> (j << 2)) & 0xF) * s + b) * x[base + j];
    }
    for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    threadgroup float sm[TGTHREADS/32];
    if (lane == 0) sm[sid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float s = 0; for (uint i = 0; i < TGTHREADS/32; i++) s += sm[i]; out[n] = s; }
}

// ---- qgemv4v : carichi vettoriali (uint4/float4), un gruppo (64 pesi) per iterazione.
kernel void qgemv4v(device const uint*  w      [[buffer(0)]],
                    device const float* scales [[buffer(1)]],
                    device const float* biases [[buffer(2)]],
                    device const float* x      [[buffer(3)]],
                    device float*       out    [[buffer(4)]],
                    constant int& N [[buffer(5)]],
                    constant int& K [[buffer(6)]],
                    uint  tgid [[threadgroup_position_in_grid]],
                    uint  tid  [[thread_position_in_threadgroup]],
                    uint  lane [[thread_index_in_simdgroup]],
                    uint  sid  [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgid; if (n >= N) return;
    int G = K >> 6;
    const device uint*  wr = w      + (size_t)n*(K >> 3);
    const device float* sc = scales + (size_t)n*G;
    const device float* bs = biases + (size_t)n*G;
    float acc = 0.0f;
    for (int g = (int)tid; g < G; g += TGTHREADS) {
        float s = sc[g], b = bs[g];
        const device uint4*  w4 = (const device uint4*)(wr + (g << 3));
        const device float4* x4 = (const device float4*)(x + (g << 6));
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            uint4 pk = w4[q];
            uint pp[4] = {pk.x, pk.y, pk.z, pk.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint p = pp[j];
                float4 xa = x4[j*2], xb = x4[j*2+1];
                acc += (float((p     )&0xF)*s+b)*xa.x;
                acc += (float((p>>4  )&0xF)*s+b)*xa.y;
                acc += (float((p>>8  )&0xF)*s+b)*xa.z;
                acc += (float((p>>12 )&0xF)*s+b)*xa.w;
                acc += (float((p>>16 )&0xF)*s+b)*xb.x;
                acc += (float((p>>20 )&0xF)*s+b)*xb.y;
                acc += (float((p>>24 )&0xF)*s+b)*xb.z;
                acc += (float((p>>28 )&0xF)*s+b)*xb.w;
            }
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    threadgroup float sm[TGTHREADS/32];
    if (lane == 0) sm[sid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float s = 0; for (uint i = 0; i < TGTHREADS/32; i++) s += sm[i]; out[n] = s; }
}

// ---- bcopy : sanity di banda (copia vettoriale pura).
kernel void bcopy(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                  constant int& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = src[gid];
}
