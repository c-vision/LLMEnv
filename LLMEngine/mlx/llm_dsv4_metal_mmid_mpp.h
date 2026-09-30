/* GENERATO da tools/gen_dsv4_mmid_metal.py — NON modificare a mano. Kernel mul_mm_id su tensor ops MPP (Metal 4), estratti da ds4. */
#ifndef LLM_DSV4_METAL_MMID_MPP_H
#define LLM_DSV4_METAL_MMID_MPP_H

static const char* const LLM_DSV4_METAL_MMID_MPP = R"MSL(
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4, bool EXPERT_ADDRESSES = false>
kernel void kernel_mul_mm_id_mpp(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const char * work,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    device const uint32_t *work_count = (device const uint32_t *)work;
    const uint32_t work_index = tgpig.x;
    if (work_index >= work_count[0]) {
        return;
    }
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[work_index];
    const int im = (int)item.x;
    const int r0 = tgpig.y*NR0;
    const int r1 = (int)item.y;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    if (!ds4_tp_owns_expert(im, args.ne02, args.tp_rank, args.tp_world)) {
        for (short j = sgitg; j < nr1; j += 4) {
            const int idj = ids_i32[im*args.ne21 + r1 + j];
            const short ide = idj % args.ne20;
            const short idt = idj / args.ne20;
            device float *D = (device float *)dst + r0 + ide*args.ne0 +
                              idt*args.ne1*args.ne0;
            for (int i = tiisg; i < nr0; i += 32) D[i] = 0.0f;
        }
        return;
    }

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = EXPERT_ADDRESSES ? 0 :
        (uint64_t)(im - args.tp_expert_base)*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    if (EXPERT_ADDRESSES) {
        src0 = reinterpret_cast<device const char *>(
            ((device const uint64_t *)src0)[im]);
    }
    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NR1, NK));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) =
                    loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;
                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) =
                    loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) =
                (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = tiitg/32; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

typedef decltype(kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id_mpp_t;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_addr_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4, true>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_addr_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4, true>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_addr_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4, true>;
template [[host_name("kernel_mul_mm_id_q8_0_f32_mpp")]] kernel mul_mm_id_mpp_t kernel_mul_mm_id_mpp<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
)MSL";

#endif
