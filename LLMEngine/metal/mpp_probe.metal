#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void mpp_probe(tensor<device half, extents<int, 16, 16>> A,
                      tensor<device half, extents<int, 16, 32>> B,
                      tensor<device float, extents<int, 16, 32>> C) {
    C[0, 0] = (float)A[0, 0] + 1000.0f;
    C[0, 1] = (float)B[0, 0];
}
