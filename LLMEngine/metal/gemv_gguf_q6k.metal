#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
// GEMV con dequant FUSO per GGUF q6_K (super-blocco = 256 pesi, 210 byte:
//   uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d
// 6 bit per peso: 4 bit bassi da ql, 2 bit alti da qh, scala int8 per 16 valori.
// y = d * sc * (q - 32). Blocchi GREZZI, nessuna espansione a F16.

kernel void gemv_q6_k(device const uchar* w   [[buffer(0)]],
                      device const float* x   [[buffer(1)]],
                      device float*       out [[buffer(2)]],
                      constant int& N [[buffer(3)]],
                      constant int& K [[buffer(4)]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgpig.x * 8 + (int)sg; if (n >= N) return;
    const int NB = K >> 8;                          // super-blocchi da 256 pesi
    const device uchar* wr = w + (size_t)n * (size_t)NB * 210;
    float acc = 0.0f;
    for (int b = (int)lane; b < NB; b += 32) {
        const device uchar* p  = wr + (size_t)b * 210;
        const device uchar* ql = p;
        const device uchar* qh = p + 128;
        const device char*  sc = (const device char*)(p + 192);
        const float d = (float)(*((const device half*)(p + 208)));
        const float* xb = x + (size_t)b * 256;
        #pragma unroll
        for (int hf = 0; hf < 2; hf++) {      // due metà da 128 valori
            const device uchar* ql2 = ql + hf * 64;
            const device uchar* qh2 = qh + hf * 32;
            const device char*  sc2 = sc + hf * 8;
            const float* xh = xb + hf * 128;
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int is = l >> 4;              // 0 o 1 dentro la metà
                const int q1 = (int)((ql2[l]      & 0xF) | (((qh2[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql2[l + 32] & 0xF) | (((qh2[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql2[l]      >> 4)  | (((qh2[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql2[l + 32] >> 4)  | (((qh2[l] >> 6) & 3) << 4)) - 32;
                const float s1 = d * (float)sc2[is + 0];
                const float s2 = d * (float)sc2[is + 2];
                const float s3 = d * (float)sc2[is + 4];
                const float s4 = d * (float)sc2[is + 6];
                acc += s1 * (float)q1 * xh[l];
                acc += s2 * (float)q2 * xh[l + 32];
                acc += s3 * (float)q3 * xh[l + 64];
                acc += s4 * (float)q4 * xh[l + 96];
            }
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
