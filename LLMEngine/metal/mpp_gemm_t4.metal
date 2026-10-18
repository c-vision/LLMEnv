// mpp_gemm_t4.metal — GEMM f16 MPP/NAX con INPUT `tensor` e 4 SIMD-Group (struttura MLX).
// Layout dei tensor (per riusare i nostri buffer row-major):
//   A: extents (K, M), strides (1, K)   -> A[k][m] = m*K + k
//   B: extents (N, K), strides (1, N)   -> B[n][k] = k*N + n
//   C: extents (N, M), strides (1, N)   -> C[n][m] = m*N + n  (float, azzerato dall'host)
// Descrittore con K dinamico: MPP loops over K internamente; accumula in C (memoria).
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void mpp_gemm_t4(tensor<device half,  dextents<int32_t, 2>> A,
                        tensor<device half,  dextents<int32_t, 2>> B,
                        tensor<device float, dextents<int32_t, 2>> C,
                        uint2 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 32, static_cast<int>(dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
}
