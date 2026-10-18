// mpp_gemm_f16.metal — GEMM f16 con MPP/NAX (MetalPerformancePrimitives) su cooperative tensor.
// Tile 32x32 (MxN), K-step 16, 1 SIMD-Group (32 thread). Double-buffer in threadgroup memory
// con carichi coalescenti uint4: si carica il tile k+1 mentre si calcola il tile k.
// I/O su buffer plain (macOS 26: i tensor kernel-arg non hanno storage CPU).
// Mapping cooperative: indice = (minor, major) -> A[k,m], B[n,k], C[n,m].
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define TM 32
#define TN 32
#define TK 16
#define NTH 32
#define TILE (TM * TK + TK * TN)   // half per set di buffer

kernel void mpp_gemm_f16(device const half* A [[buffer(0)]],
                         device const half* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant int* dims [[buffer(3)]],
                         threadgroup half* sh [[threadgroup(0)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TM, TN, TK);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>,
        metal::remove_addrspace_t<decltype(cb)>, float>();

    float acc[64];
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;
    const uint16_t cap  = (uint16_t)cc.get_capacity();
    const uint16_t capA = (uint16_t)ca.get_capacity();
    const uint16_t capB = (uint16_t)cb.get_capacity();
    short offA[64], offB[64];
    for (uint16_t i = 0; i < capA; ++i) {
        offA[i] = -1;
        if (ca.is_valid_element(i)) { auto ix = ca.get_multidimensional_index(i); offA[i] = (short)((int)ix[1] * TK + (int)ix[0]); }
    }
    for (uint16_t i = 0; i < capB; ++i) {
        offB[i] = -1;
        if (cb.is_valid_element(i)) { auto ix = cb.get_multidimensional_index(i); offB[i] = (short)((int)ix[1] * TN + (int)ix[0]); }
    }

    const int N = dims[1], K = dims[2];
    const int row0 = (int)tgid.y * TM;
    const int col0 = (int)tgid.x * TN;
    const int iters = K / TK;

    // prologo: tile k=0 nel buffer 0
    {
        threadgroup half* aS = sh;
        threadgroup half* bS = sh + TM * TK;
        for (int q = 0; q < 2; q++) {
            int e = tid * 2 + q;
            int r = e >> 1, c = (e & 1) * 8;
            *(threadgroup uint4*)(aS + r * TK + c) = *(const device uint4*)(A + (size_t)(row0 + r) * K + c);
        }
        for (int q = 0; q < 2; q++) {
            int e = tid * 2 + q;
            int r = e >> 2, c = (e & 3) * 8;
            *(threadgroup uint4*)(bS + r * TN + c) = *(const device uint4*)(B + (size_t)r * N + col0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int it = 0; it < iters; ++it) {
        const int cur = it & 1;
        const int nxt = cur ^ 1;

        if (it + 1 < iters) {
            const int kn = (it + 1) * TK;
            threadgroup half* aN = sh + nxt * TILE;
            threadgroup half* bN = aN + TM * TK;
            for (int q = 0; q < 2; q++) {
                int e = tid * 2 + q;
                int r = e >> 1, c = (e & 1) * 8;
                *(threadgroup uint4*)(aN + r * TK + c) = *(const device uint4*)(A + (size_t)(row0 + r) * K + kn + c);
            }
            for (int q = 0; q < 2; q++) {
                int e = tid * 2 + q;
                int r = e >> 2, c = (e & 3) * 8;
                *(threadgroup uint4*)(bN + r * TN + c) = *(const device uint4*)(B + (size_t)(kn + r) * N + col0 + c);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* aC = sh + cur * TILE;
        threadgroup half* bC = aC + TM * TK;
        for (uint16_t i = 0; i < capA; ++i)
            if (offA[i] >= 0) ca[i] = aC[offA[i]];
        for (uint16_t i = 0; i < capB; ++i)
            if (offB[i] >= 0) cb[i] = bC[offB[i]];

        op.run(ca, cb, cc);
        for (uint16_t i = 0; i < cap; ++i)
            if (cc.is_valid_element(i)) acc[i] += cc[i];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < cap; ++i) {
        if (!cc.is_valid_element(i)) continue;
        auto ix = cc.get_multidimensional_index(i);          // (n, m)
        C[(size_t)(row0 + (int)ix[1]) * N + col0 + (int)ix[0]] = acc[i];
    }
}
