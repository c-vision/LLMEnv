// mpp_test.metal — test MetalPerformancePrimitives (NAX) f16: 16x32x16, 1 simdgroup.
// Pattern ufficiale: parametri tensor<device half, dextents<int32_t,2>> + op.run(mA, mB, cT).
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void mpp_f16_16x32x16(tensor<device half, dextents<int32_t, 2>> A,
                             tensor<device half, dextents<int32_t, 2>> B,
                             tensor<device float, dextents<int32_t, 2>> C) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto mA = A.slice(0, 0);
    auto mB = B.slice(0, 0);
    auto mC = C.slice(0, 0);

    auto cT = op.get_destination_cooperative_tensor<
        __remove_addrspace_t<decltype(mA)>,
        __remove_addrspace_t<decltype(mB)>, float>();

    #pragma unroll full
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.get_mask(i)) cT[i] = 0.0f;

    op.run(mA, mB, cT);

    #pragma unroll full
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.get_mask(i)) mC[i] = cT[i];
}
