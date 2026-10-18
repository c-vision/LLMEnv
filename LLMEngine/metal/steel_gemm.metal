// steel_gemm.metal — GEMM f16 "steel" (standalone, no engine deps)
// Tile 64x64x32, 4 simdgroup (2x2 warp-tile), register blocking 4x4 fragment
// simdgroup_matrix, carichi uint4 16B, double-buffer As/Bs.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BM 64
#define BN 64
#define BK 32

kernel void steel_f16_gemm(device const half* A [[buffer(0)]],
                           device const half* B [[buffer(1)]],
                           device float* C [[buffer(2)]],
                           constant int& M [[buffer(3)]],
                           constant int& N [[buffer(4)]],
                           constant int& K [[buffer(5)]],
                           threadgroup half* shmem [[threadgroup(0)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint  tiitg [[thread_index_in_threadgroup]],
                           uint  sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half* As[2] = { shmem, shmem + BM * BK + BK * BN };
    threadgroup half* Bs[2] = { shmem + BM * BK, shmem + BM * BK + BK * BN + BM * BK };
    const int row0 = tgpig.y * BM;
    const int col0 = tgpig.x * BN;
    const int t = tiitg;

    simdgroup_matrix<half, 8, 8> ma[4], mb[4];
    simdgroup_matrix<float, 8, 8> mc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mc[i][j] = simdgroup_matrix<float, 8, 8>(0.f);

    const int sr = (sgitg / 2) * 32;
    const int sc = (sgitg % 2) * 32;
    const int iters = K / BK;

    for (int it = 0; it < iters; it++) {
        const int cur = it & 1;
        const int k0 = it * BK;
        for (int q = 0; q < 2; q++) {
            int idx = t * 2 + q;
            int r = idx >> 2, c = (idx & 3) * 8;
            *(threadgroup uint4*)(As[cur] + r * BK + c) =
                *(const device uint4*)(A + (size_t)(row0 + r) * K + k0 + c);
        }
        for (int q = 0; q < 2; q++) {
            int idx = t * 2 + q;
            int r = idx >> 3, c = (idx & 7) * 8;
            *(threadgroup uint4*)(Bs[cur] + r * BN + c) =
                *(const device uint4*)(B + (size_t)(k0 + r) * N + col0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; kk++) {
            for (int i = 0; i < 4; i++)
                simdgroup_load(ma[i], As[cur] + (sr + 8 * i) * BK + 8 * kk, BK);
            for (int j = 0; j < 4; j++)
                simdgroup_load(mb[j], Bs[cur] + (8 * kk) * BN + sc + 8 * j, BN);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(mc[i][j], ma[i], mb[j], mc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(mc[i][j],
                C + (size_t)(row0 + sr + 8 * i) * N + (col0 + sc + 8 * j), N);
}
