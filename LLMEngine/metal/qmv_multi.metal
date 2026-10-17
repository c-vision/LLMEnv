#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// QMV multi-riga (tall-skinny), variante simdgroup-per-riga: 1 simdgroup (32 lane) per riga n,
// lane su K (stride 32), M accumulatori per lane, riduzione simd_sum. Pesi 8-bit affine g64.
// w[N, K/4] u32 | sc,bs[N, K/64] half | x[M, K] half | y[M, N] half.  8 righe (n) per threadgroup.
#define MMAX 8

kernel void qmv8_multi(device const uint*   w   [[buffer(0)]],
                       device const half*   sc  [[buffer(1)]],
                       device const half*   bs  [[buffer(2)]],
                       device const half*   x   [[buffer(3)]],
                       device half*         y   [[buffer(4)]],
                       constant int& N [[buffer(5)]],
                       constant int& K [[buffer(6)]],
                       constant int& M [[buffer(7)]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgpig.x * 8 + (int)sg;
    float acc[MMAX];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;
    if (n < N) {
        int Kw = K >> 2, G = K >> 6;
        device const uint*  wr  = w  + (size_t)n * Kw;
        device const half*  scr = sc + (size_t)n * G;
        device const half*  bsr = bs + (size_t)n * G;
        for (int u = (int)lane; u < Kw; u += 32) {
            uint pk = wr[u];
            int k0 = u << 2, g = k0 >> 6;
            float s = (float)scr[g], b = (float)bsr[g];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float wv = float((pk >> (8 * j)) & 0xFF) * s + b;
                int k = k0 + j;
                for (int m = 0; m < M; m++) acc[m] += wv * (float)x[(size_t)m * K + k];
            }
        }
    }
    for (int m = 0; m < M; m++) {
        float tot = simd_sum(acc[m]);
        if (lane == 0 && n < N) y[(size_t)m * N + n] = (half)tot;
    }
}
