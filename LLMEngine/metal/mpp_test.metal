// mpp_test.metal — MPP (NAX) f16: 16x32x16, 1 simdgroup.
// Extent statiche (nessun metadato dinamico): i tensor si bindano come gpuAddress di buffer.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void mpp_f16_16x32x16(tensor<device half, extents<int, 16, 16>> A,
                             tensor<device half, extents<int, 16, 32>> B,
                             tensor<device float, extents<int, 16, 32>> C) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto cT = op.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(A)>,
        metal::remove_addrspace_t<decltype(B)>, float>();

    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);

    op.run(A, B, cT);
    cT.store(C);
}
