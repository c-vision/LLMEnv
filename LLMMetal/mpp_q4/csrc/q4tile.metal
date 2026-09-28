// q4tile.metal — GEMM q4 affine su MPP col **schema tile di Splash** (q4_mpp_tiles.h),
// generalizzato a GS=32 E GS=64, e con layout pesi parametrico (repack oppure stride diretto).
//
// Differenze dal kernel "ingenuo" (per-gruppo con somma ricalcolata), che PERDEVA contro qmm:
//  1) SOMME D'INGRESSO CONDIVISE: sum_i x_i*bias[n,g] = bias[n,g]*sum_i x_i -> le somme per
//     riga (NS = 256/GS per blocco di 256 elementi K) si calcolano UNA volta e si riusano per
//     TUTTE le colonne N, con doppio buffer (regione 0/1).
//  2) Accumulo fp32 e epilogo scale/bias applicati in ordine di gruppo (stesso ordine della
//     forma sequenziale), matmul2d per gruppo.
//  3) Opzionale: pesi ri-impacchettati per (tile, gruppo) -> load MPP coalescente (wgroup_adv
//     = TileN*GS/2) oppure stride diretto dal layout MLX [N, K/8] (wgroup_adv = GS/2,
//     wstride = K).
//
// Parametri (un `constant Q4TParams&`):
//   input_size/output_size : K e N
//   wstride     : stride in nibble fra colonne adiacenti (GS per il repack, K per il layout MLX)
//   wgroup_adv  : byte fra blocchi di gruppi consecutivi (TileN*GS/2 oppure GS/2)
//   wtile_base  : byte per l'avanzamento di tile (qgroups*TileN*GS/2 oppure TileN*K/8)
//   s_a,s_b,s_c : indice scale/bias = tile*s_a + g*s_b + nl*s_c
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Q4TParams {
  uint input_size;
  uint output_size;
  uint wstride;
  uint wgroup_adv;
  uint wtile_base;
  uint s_a;
  uint s_b;
  uint s_c;
};

template <uint Rows, uint GS, uint Simdgroups>
inline void q4t_sums(device const bfloat *input, uint input_size, uint origin,
                     threadgroup float *sums, uint sum_origin, uint lane,
                     uint sg) {
  constexpr uint NS = 256 / GS;
  for (uint row = sg; row < Rows; row += Simdgroups) {
    uint base = row * input_size + origin;
#pragma unroll
    for (uint s = 0; s < NS; ++s) {
      float acc = 0.0f;
#pragma unroll
      for (uint q = 0; q < GS / 32; ++q)
        acc += float(input[base + s * GS + q * 32 + lane]);
      float v = simd_sum(acc);
      if (lane == 0) sums[sum_origin + s * Rows + row] = v;
    }
  }
}

template <uint Rows, uint TileN, uint GS, uint Simdgroups = 8>
inline void q4t_body(device bfloat *input, device uchar *weights,
                     device bfloat *scales, device bfloat *biases,
                     device bfloat *output, threadgroup float *input_sums,
                     uint tile, uint lane, uint sg, constant Q4TParams &p) {
  constexpr uint NS = 256 / GS;
  const uint input_size = p.input_size;
  const uint output_size = p.output_size;
  uint qgroups = input_size / GS;
  device uchar *tw = weights + ulong(tile) * p.wtile_base;

  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  constexpr auto desc = matmul2d_descriptor(Rows, TileN, GS, false, true, false);
  matmul2d<desc, execution_simdgroups<Simdgroups>> op;

  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
      tw, dextents<int, 2>{GS, TileN}, array<int, 2>{1, int(p.wstride)});
  auto a0 = a.slice<GS, Rows>(0, 0);
  auto b00 = b0.slice<GS, TileN>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0),
                                                           decltype(b00), float>();
  const uint cap = acc.get_capacity();
  for (uint i = 0; i < cap; ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  q4t_sums<Rows, GS, Simdgroups>(input, input_size, 0, input_sums, 0, lane, sg);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint g = 0; g < qgroups; ++g) {
    auto as = a.slice<GS, Rows>(g * GS, 0);
    device uchar *gw = tw + ulong(g) * p.wgroup_adv;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bg(
        gw, dextents<int, 2>{GS, TileN}, array<int, 2>{1, int(p.wstride)});
    auto bs = bg.slice<GS, TileN>(0, 0);
    auto part = op.template get_destination_cooperative_tensor<
        decltype(as), decltype(bs), float>();
    for (uint i = 0; i < cap; ++i)
      if (part.is_valid_element(i)) part[i] = 0.0f;
    op.run(as, bs, part);
    uint sum_off = ((g / NS) & 1) * (NS * Rows) + (g % NS) * Rows;
    for (uint i = 0; i < cap; ++i)
      if (part.is_valid_element(i)) {
        auto idx = acc.get_multidimensional_index(i);
        uint nl = idx[0], row = idx[1];
        ulong param = ulong(tile) * p.s_a + ulong(g) * p.s_b + ulong(nl) * p.s_c;
        acc[i] += part[i] * float(scales[param]) +
                  input_sums[sum_off + row] * float(biases[param]);
      }
    if ((g & (NS - 1)) == (NS - 1) && g + 1 < qgroups) {
      uint next = (g + 1) / NS;
      q4t_sums<Rows, GS, Simdgroups>(input, input_size, (g + 1) * GS, input_sums,
                                     (next & 1) * (NS * Rows), lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = 0; i < cap; ++i)
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);
      uint nl = idx[0], row = idx[1];
      output[row * output_size + tile * TileN + nl] = bfloat(acc[i]);
    }
}

#define Q4T_ENTRY(Name, Rows, TileN, GS)                                       \
  kernel void Name(                                                            \
      device bfloat *input [[buffer(0)]],                                      \
      device uchar *weights [[buffer(1)]],                                     \
      device bfloat *scales [[buffer(2)]],                                     \
      device bfloat *biases [[buffer(3)]],                                     \
      device bfloat *output [[buffer(4)]],                                     \
      constant Q4TParams &params [[buffer(5)]],                                \
      uint tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]]) {                            \
    threadgroup float input_sums[2 * (256 / GS) * (Rows)];                     \
    q4t_body<Rows, TileN, GS>(input, weights, scales, biases, output,          \
                              input_sums, tgid, lane, sg, params);             \
  }

// GS=32 (trunk del nostro modello) e GS=64 (sidecar MTP / Splash).
Q4T_ENTRY(q4t_m16_n128_g32, 16, 128, 32)
Q4T_ENTRY(q4t_m24_n128_g32, 24, 128, 32)
Q4T_ENTRY(q4t_m32_n128_g32, 32, 128, 32)
Q4T_ENTRY(q4t_m16_n128_g64, 16, 128, 64)
Q4T_ENTRY(q4t_m24_n128_g64, 24, 128, 64)
Q4T_ENTRY(q4t_m32_n128_g64, 32, 128, 64)
