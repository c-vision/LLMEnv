#include <metal_stdlib>
using namespace metal;
kernel void mpp_probe2(device const half* A, device const half* B, device float* C) {
    C[0] = (float)A[0] + 1000.0f;
    C[1] = (float)B[0];
}
