#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
// GEMV con dequant FUSO per GGUF q4_0 (blocco = half d + 16 byte qs = 32 pesi, 18 byte/blocco).
// w[n] ha NB=K/32 blocchi contigui. 1 simdgroup per riga (8 righe/TG), lane sui blocchi.
kernel void gemv_q4_0(device const uchar* w   [[buffer(0)]],
                      device const float* x   [[buffer(1)]],
                      device float*       out  [[buffer(2)]],
                      constant int& N [[buffer(3)]],
                      constant int& K [[buffer(4)]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgpig.x*8 + (int)sg; if (n >= N) return;
    const int NB = K >> 5;                     // blocchi da 32 pesi
    const device uchar* wr = w + (size_t)n * (size_t)NB * 18;
    float acc = 0.0f;
    for (int b = (int)lane; b < NB; b += 32) {
        half d = *((const device half*)(wr + (size_t)b*18));
        const device uchar* qs = wr + (size_t)b*18 + 2;
        const device float* xb = x + (b << 5);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uchar q = qs[i];
            int lo = q & 0xF, hi = (q >> 4) & 0xF;
            acc += (float)d * ((float)(lo - 8) * xb[i] + (float)(hi - 8) * xb[i + 16]);
        }
    }
    float tot = simd_sum(acc);
    if (lane == 0) out[n] = tot;
}
// F16 GEMV di riferimento (stessa shape): 1 simdgroup per riga.
kernel void gemv_f16(device const half* w  [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant int& N [[buffer(3)]],
                     constant int& K [[buffer(4)]],
                     uint3 tgpig [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgpig.x*8 + (int)sg; if (n >= N) return;
    const device half* wr = w + (size_t)n*K;
    float acc = 0.0f;
    for (int k = (int)lane; k < K; k += 32) acc += (float)wr[k] * x[k];
    float tot = simd_sum(acc);
    if (lane == 0) out[n] = tot;
}
