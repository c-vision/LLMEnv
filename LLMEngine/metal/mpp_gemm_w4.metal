// mpp_gemm_w4.metal — GEMM f16 x w4 (dequant FUSO nel load) con MPP/NAX.
// Formato pesi (q4_0-like): per ogni blocco di 32 valori lungo N: 16 byte di nibble
// (valore = nibble - 8) + 1 half di scala. Attivazioni f16, output f32.
// Tile 32x32, K-step 16, 1 simdgroup, double-buffer threadgroup.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define TM 32
#define TN 32
#define TK 16
#define TILE (TM * TK + TK * TN)

kernel void mpp_gemm_w4(device const half*  A  [[buffer(0)]],
                        device const uchar* Bn [[buffer(1)]],   // 16B per blocco (32 nibble)
                        device const half*  Bs [[buffer(2)]],   // 1 scala per blocco
                        device float*       C  [[buffer(3)]],
                        constant int* dims [[buffer(4)]],
                        threadgroup half* sh [[threadgroup(0)]],
                        uint2 tgid [[threadgroup_position_in_grid]],
                        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TM, TN, TK);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>,
        metal::remove_addrspace_t<decltype(cb)>, float>();

    float acc[64];
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;
    const uint16_t cap  = (uint16_t)cc.get_capacity();
    const uint16_t capA = (uint16_t)ca.get_capacity();
    const uint16_t capB = (uint16_t)cb.get_capacity();
    short offA[64], offB[64];
    for (uint16_t i = 0; i < capA; ++i) {
        offA[i] = -1;
        if (ca.is_valid_element(i)) { auto ix = ca.get_multidimensional_index(i); offA[i] = (short)((int)ix[1] * TK + (int)ix[0]); }
    }
    for (uint16_t i = 0; i < capB; ++i) {
        offB[i] = -1;
        if (cb.is_valid_element(i)) { auto ix = cb.get_multidimensional_index(i); offB[i] = (short)((int)ix[1] * TN + (int)ix[0]); }
    }

    const int N = dims[1], K = dims[2];
    const int row0 = (int)tgid.y * TM;
    const int col0 = (int)tgid.x * TN;
    const int nb  = N / 32;                    // blocchi per riga
    const int iters = K / TK;

    // carica il tile k nel set `dst`: A coalescente, B dequantizzato dai nibble
    #define LOAD_TILE(dst, k0)                                                                    \
        {                                                                                         \
            threadgroup half* aS = sh + (dst) * TILE;                                             \
            threadgroup half* bS = aS + TM * TK;                                                  \
            for (int q = 0; q < 2; q++) {                                                         \
                int e = tid * 2 + q;                                                              \
                int r = e >> 1, c = (e & 1) * 8;                                                  \
                *(threadgroup uint4*)(aS + r * TK + c) =                                          \
                    *(const device uint4*)(A + (size_t)(row0 + r) * K + (k0) + c);                \
            }                                                                                     \
            if (tid < 16) {                                                                       \
                int r = tid;                                    /* riga del tile (k) */          \
                int blk = (col0 / 32);                          /* blocco di 32 colonne */       \
                const device uchar* p = Bn + ((size_t)((k0) + r) * nb + blk) * 16;                \
                half s = Bs[(size_t)((k0) + r) * nb + blk];                                       \
                uint4 q = *(const device uint4*)p;                                                \
                uint wv[4] = { q.x, q.y, q.z, q.w };                                              \
                for (int w = 0; w < 4; w++) {                                                     \
                    for (int j = 0; j < 8; j++) {                                                 \
                        int nib = (int)((wv[w] >> (4 * j)) & 0xFu) - 8;                           \
                        bS[r * TN + w * 8 + j] = (half)((float)nib * (float)s);                   \
                    }                                                                             \
                }                                                                                 \
            }                                                                                     \
        }

    LOAD_TILE(0, 0)
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int it = 0; it < iters; ++it) {
        const int cur = it & 1, nxt = cur ^ 1;
        if (it + 1 < iters) { LOAD_TILE(nxt, (it + 1) * TK) }
        threadgroup half* aC = sh + cur * TILE;
        threadgroup half* bC = aC + TM * TK;

        for (uint16_t i = 0; i < capA; ++i)
            if (offA[i] >= 0) ca[i] = aC[offA[i]];
        for (uint16_t i = 0; i < capB; ++i)
            if (offB[i] >= 0) cb[i] = bC[offB[i]];

        op.run(ca, cb, cc);
        for (uint16_t i = 0; i < cap; ++i)
            if (cc.is_valid_element(i)) acc[i] += cc[i];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #undef LOAD_TILE

    for (uint16_t i = 0; i < cap; ++i) {
        if (!cc.is_valid_element(i)) continue;
        auto ix = cc.get_multidimensional_index(i);
        C[(size_t)(row0 + (int)ix[1]) * N + col0 + (int)ix[0]] = acc[i];
    }
}
