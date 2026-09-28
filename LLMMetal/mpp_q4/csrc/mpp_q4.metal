// mpp_q4.metal — GEMM q4 affine (4-bit, group GS) su MPP (tensor unit Apple).
// Template: M (righe della spec), GS (group size: 32 o 64). K, N runtime (constant buffer).
// Layout pesi: w U32 [N, K/8] (n-major, k contiguo, nibble low-first), scales/biases bf16 [N, K/GS].
// OBBLIGO build: -std=metal4.0 -mmacosx-version-min=26.2 (floor NAX/MPP).
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <uint M, uint GS>
inline void mpp_q4_body(device bfloat* x, device const uint32_t* w, device const bfloat* sc,
                        device const bfloat* bi, device bfloat* out, threadgroup float* sums_g,
                        uint K, uint N, uint tile, uint tid) {
  constexpr uint TileN = 128;
  const uint G = K / GS;
  auto a = tensor((device bfloat*)x, dextents<int,2>{int(K), int(M)}, array<int,2>{1, int(K)});
  constexpr auto desc = matmul2d_descriptor(M, TileN, GS, false, true, false);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto a0 = a.slice<GS, M>(0, 0);
  tensor<device uint4b_format, dextents<int,2>, tensor_inline> b0(
      (device uchar*)w + ulong(tile)*TileN*K/2, dextents<int,2>{GS, TileN}, array<int,2>{1, int(K)});
  auto b00 = b0.slice<GS, TileN>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b00), float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  auto part = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b00), float>();
  for (uint g = 0; g < G; ++g) {
    if (tid < M) { float ss = 0.0f; for (uint k = 0; k < GS; ++k) ss += float(x[tid*K + g*GS + k]); sums_g[tid] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto as = a.slice<GS, M>(g*GS, 0);
    tensor<device uint4b_format, dextents<int,2>, tensor_inline> bg(
        (device uchar*)w + ulong(tile)*TileN*K/2 + ulong(g)*GS/2, dextents<int,2>{GS, TileN}, array<int,2>{1, int(K)});
    auto bs = bg.slice<GS, TileN>(0, 0);
    for (uint i = 0; i < part.get_capacity(); ++i) if (part.is_valid_element(i)) part[i] = 0.0f;
    op.run(as, bs, part);
    for (uint i = 0; i < part.get_capacity(); ++i) if (part.is_valid_element(i)) {
      auto idx = part.get_multidimensional_index(i);
      uint nl = idx[0], mm = idx[1]; uint n = tile*TileN + nl;
      acc[i] += part[i]*float(sc[n*G + g]) + sums_g[mm]*float(bi[n*G + g]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
    auto idx = acc.get_multidimensional_index(i);
    out[idx[1]*int(N) + int(tile)*TileN + idx[0]] = bfloat(acc[i]);
  }
}

#define MPP_Q4_ENTRY(MV, GSV) \
kernel void mpp_q4_m##MV##_gs##GSV(device bfloat* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]], \
                device const bfloat* sc [[buffer(2)]], device const bfloat* bi [[buffer(3)]], \
                device bfloat* out [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]], \
                uint3 tgid [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) { \
  threadgroup float sums_g[MV]; \
  mpp_q4_body<MV, GSV>(x, w, sc, bi, out, sums_g, K, N, tgid.x, ti); \
}
MPP_Q4_ENTRY(8,32)  MPP_Q4_ENTRY(8,64)
MPP_Q4_ENTRY(16,32) MPP_Q4_ENTRY(16,64)
MPP_Q4_ENTRY(32,32) MPP_Q4_ENTRY(32,64)
MPP_Q4_ENTRY(64,32) MPP_Q4_ENTRY(64,64)
