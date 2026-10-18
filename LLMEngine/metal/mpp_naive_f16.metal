// mpp_naive_f16.metal — MPP (NAX) f16 16x32x16, cooperative tensor (registri) + I/O da buffer plain.
// NB: su macOS 26 il tensor kernel-binding non ha storage CPU; si usano A/B/C device pointers.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define MM 16
#define NN 32
#define KK 16

kernel void mpp_naive_f16(device const half* A [[buffer(0)]],
                          device const half* B [[buffer(1)]],
                          device float* C [[buffer(2)]]) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MM, NN, KK);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>,
        metal::remove_addrspace_t<decltype(cb)>, float>();

    for (uint16_t i = 0; i < ca.get_capacity(); ++i) {
        if (!ca.is_valid_element(i)) continue;
        auto ix = ca.get_multidimensional_index(i);
        ca[i] = A[(int)ix[1] * KK + (int)ix[0]];
    }
    for (uint16_t i = 0; i < cb.get_capacity(); ++i) {
        if (!cb.is_valid_element(i)) continue;
        auto ix = cb.get_multidimensional_index(i);
        cb[i] = B[(int)ix[1] * NN + (int)ix[0]];
    }
    for (uint16_t i = 0; i < cc.get_capacity(); ++i)
        if (cc.is_valid_element(i)) cc[i] = 0.0f;

    op.run(ca, cb, cc);

    for (uint16_t i = 0; i < cc.get_capacity(); ++i) {
        if (!cc.is_valid_element(i)) continue;
        auto ix = cc.get_multidimensional_index(i);
        C[(int)ix[1] * NN + (int)ix[0]] = cc[i];
    }
}
