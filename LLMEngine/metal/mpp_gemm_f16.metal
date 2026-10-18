// mpp_gemm_f16.metal — GEMM f16 con MPP/NAX (MetalPerformancePrimitives) su cooperative tensor.
// Tile 64x64 (MxN), K-step 16, 4 SIMD-Group (128 thread), staging in threadgroup memory con
// carichi coalescenti uint4 (16B). I/O su buffer plain (macOS 26: i tensor kernel-arg non
// hanno storage CPU). Mapping cooperative: indice = (minor, major) -> A[k,m], B[n,k], C[n,m].
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define TM 32
#define TN 32
#define TK 16
#define NTH 32

kernel void mpp_gemm_f16(device const half* A [[buffer(0)]],
                         device const half* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant int* dims [[buffer(3)]],
                         threadgroup half* sh [[threadgroup(0)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint  tid   [[thread_index_in_threadgroup]]) {
    threadgroup half* shA = sh;            // [TM][TK]
    threadgroup half* shB = sh + TM * TK;  // [TK][TN]

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TM, TN, TK);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>,
        metal::remove_addrspace_t<decltype(cb)>, float>();

    float acc[64];
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;
    const uint16_t cap = (uint16_t)cc.get_capacity();

    const int M = dims[0], N = dims[1], K = dims[2];
    const int row0 = (int)tgid.y * TM;
    const int col0 = (int)tgid.x * TN;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // A tile: 32x16, 2 uint4/row -> 64 carichi; 2 per thread
        for (int q = 0; q < 2; q++) {
            int e = tid * 2 + q;
            int r = e >> 1, c = (e & 1) * 8;
            *(threadgroup uint4*)(shA + r * TK + c) =
                *(const device uint4*)(A + (size_t)(row0 + r) * K + k0 + c);
        }
        // B tile: 16x32, 4 uint4/row -> 64 carichi; 2 per thread
        for (int q = 0; q < 2; q++) {
            int e = tid * 2 + q;
            int r = e >> 2, c = (e & 3) * 8;
            *(threadgroup uint4*)(shB + r * TN + c) =
                *(const device uint4*)(B + (size_t)(k0 + r) * N + col0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint16_t i = 0; i < ca.get_capacity(); ++i) {
            if (!ca.is_valid_element(i)) continue;
            auto ix = ca.get_multidimensional_index(i);      // (k, m)
            ca[i] = shA[(int)ix[1] * TK + (int)ix[0]];
        }
        for (uint16_t i = 0; i < cb.get_capacity(); ++i) {
            if (!cb.is_valid_element(i)) continue;
            auto ix = cb.get_multidimensional_index(i);      // (n, k)
            cb[i] = shB[(int)ix[1] * TN + (int)ix[0]];
        }

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
