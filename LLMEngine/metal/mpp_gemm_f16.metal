// mpp_gemm_f16.metal — GEMM f16 con MPP (NAX) su cooperative tensor.
// Tile 16x32 (MxN), K-step 16, 1 simdgroup. I/O da buffer plain (macOS 26).
// Mapping cooperative: indice = (minor, major) -> A[k,m], B[n,k], C[n,m].
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define TM 16
#define TN 32
#define TK 16

kernel void mpp_gemm_f16(device const half* A [[buffer(0)]],
                         device const half* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant int* dims [[buffer(3)]],
                         uint2 tgid [[threadgroup_position_in_grid]]) {
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
        for (uint16_t i = 0; i < ca.get_capacity(); ++i) {
            if (!ca.is_valid_element(i)) continue;
            auto ix = ca.get_multidimensional_index(i);      // (k, m)
            ca[i] = A[(size_t)(row0 + (int)ix[1]) * K + k0 + (int)ix[0]];
        }
        for (uint16_t i = 0; i < cb.get_capacity(); ++i) {
            if (!cb.is_valid_element(i)) continue;
            auto ix = cb.get_multidimensional_index(i);      // (n, k)
            cb[i] = B[(size_t)(k0 + (int)ix[1]) * N + col0 + (int)ix[0]];
        }
        for (uint16_t i = 0; i < cap; ++i)
            if (cc.is_valid_element(i)) cc[i] = acc[i];
        op.run(ca, cb, cc);
        for (uint16_t i = 0; i < cap; ++i)
            if (cc.is_valid_element(i)) acc[i] += cc[i];
    }

    for (uint16_t i = 0; i < cc.get_capacity(); ++i) {
        if (!cc.is_valid_element(i)) continue;
        auto ix = cc.get_multidimensional_index(i);          // (n, m)
        C[(size_t)(row0 + (int)ix[1]) * N + col0 + (int)ix[0]] = acc[i];
    }
}
