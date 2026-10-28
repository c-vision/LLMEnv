/* mma_bench_probe.c — ORACOLO: microbenchmark MMA vs scalare su QK^T.
 * Stesso lavoro (NQ*NK*DK MAC/iter): S[NQ,NK] = Q[NQ,DK] @ K[NK,DK]^T.
 * MMA: 128 simdgroup MMA (8x8x8) per iter. Scalare: 512 dot da DK, 32 lane.
 * Output: tempo/iter e speedup. Decisivo per "se veramente effettivo". */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static const char* HDR = "using namespace metal;\n";

#define NQ 8
#define DK 128
#define NK 64
#define ITERS 20000

static char SRC_MMA[4096], SRC_SCA[4096];

static void fmt(char* buf){
  snprintf(buf,4096,
    "    constexpr uint NQ=%d, DK=%d, NK=%d, IT=%d;\n"
    "    threadgroup half kt[DK*NK];\n"
    "    for (uint i = thread_position_in_threadgroup.x; i < DK*NK; i += 32u)\n"
    "        kt[(i%%DK)*NK + (i/DK)] = kk[i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    threadgroup half zbuf[64];\n"
    "    for (uint i = thread_position_in_threadgroup.x; i < 64u; i += 32u) zbuf[i] = (half)0;\n"
    "    simdgroup_half8x8 A[DK/8];\n"
    "    for (uint k = 0; k < DK/8; ++k) simdgroup_load(A[k], q + k*8u, DK);\n"
    "    for (uint it = 0; it < IT; ++it) {\n"
    "        for (uint nb = 0; nb < NK/8; ++nb) {\n"
    "            simdgroup_half8x8 acc;\n"
    "            simdgroup_load(acc, zbuf, 8u);\n"
    "            for (uint k = 0; k < DK/8; ++k) {\n"
    "                simdgroup_half8x8 Bk;\n"
    "                simdgroup_load(Bk, kt + k*8u*NK + nb*8u, NK);\n"
    "                simdgroup_multiply_accumulate(acc, A[k], Bk, acc);\n"
    "            }\n"
    "            simdgroup_store(acc, S + nb*8u, NK);\n"
    "        }\n"
    "    }\n", NQ,DK,NK,ITERS);
}

static const char* SRC_SCA_SIMD =
  "    constexpr uint NQ=8, DK=128, NK=64, IT=20000;\n"
  "    uint lane = thread_position_in_threadgroup.x;\n"
  "    float loc = 0.0f;\n"
  "    for (uint it = 0; it < IT; ++it) {\n"
  "        for (uint p = 0; p < NQ*NK; ++p) {\n"
  "            uint qi = p/NK, ki = p%NK; float a = 0.0f;\n"
  "            for (uint d = lane; d < DK; d += 32u) a += float(q[qi*DK+d]) * float(kk[ki*DK+d]);\n"
  "            loc += simd_sum(a);\n"
  "        }\n"
  "    }\n"
  "    if (lane == 0u) S[0] = half(loc);\n";

static const char* SRC_SCALAR_T =
  "    constexpr uint NQ=8, DK=128, NK=64, IT=20000;\n"
  "    threadgroup float tsum[32];\n"
  "    for (uint i = thread_position_in_threadgroup.x; i < 32u; i += 32u) tsum[i] = 0.0f;\n"
  "    float loc = 0.0f;\n"
  "    for (uint it = 0; it < IT; ++it) {\n"
  "        for (uint j = thread_position_in_threadgroup.x; j < NQ*NK; j += 32u) {\n"
  "            uint qi = j/NK, ki = j%NK; float acc = 0.0f;\n"
  "            for (uint d = 0; d < DK; ++d) acc += float(q[qi*DK+d]) * float(kk[ki*DK+d]);\n"
  "            loc += acc;\n"
  "        }\n"
  "    }\n"
  "    tsum[thread_position_in_threadgroup.x] = loc;\n"
  "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "    if (thread_position_in_threadgroup.x == 0u) { float s=0; for(uint i=0;i<32u;++i)s+=tsum[i]; S[0]=half(s); }\n";

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

static double run(const char* name, const char* src, mlx_array q, mlx_array k, mlx_stream s, int is_mma){
  mlx_vector_string in=mlx_vector_string_new(); mlx_vector_string_append_value(in,"q"); mlx_vector_string_append_value(in,"kk");
  mlx_vector_string on=mlx_vector_string_new(); mlx_vector_string_append_value(on,"S");
  mlx_fast_metal_kernel kern=mlx_fast_metal_kernel_new(name, in, on, src, HDR, true, false);
  if(kern.ctx==NULL){ printf("%s: build FAIL\n",name); return -1; }
  mlx_fast_metal_kernel_config cfg=mlx_fast_metal_kernel_config_new();
  if(is_mma) mlx_fast_metal_kernel_config_add_output_arg(cfg,(int[]){NQ,NK},2,MLX_FLOAT16);
  else       mlx_fast_metal_kernel_config_add_output_arg(cfg,(int[]){1,1},2,MLX_FLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg,32,1,1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg,32,1,1);
  mlx_vector_array ins=mlx_vector_array_new(); mlx_vector_array_append_value(ins,q); mlx_vector_array_append_value(ins,k);
  /* warmup */
  mlx_vector_array o0=mlx_vector_array_new(); mlx_fast_metal_kernel_apply(&o0,kern,ins,cfg,s);
  double t0=now_s();
  for(int r=0;r<3;r++){ mlx_vector_array oo=mlx_vector_array_new(); mlx_fast_metal_kernel_apply(&oo,kern,ins,cfg,s); mlx_array e=mlx_array_new(); mlx_vector_array_get(&e,oo,0); mlx_array_eval(e); }
  double t1=now_s();
  return (t1-t0)/3.0;
}

int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream s=mlx_default_gpu_stream_new();
  float* qf=(float*)malloc(NQ*DK*4); float* kf=(float*)malloc(NK*DK*4);
  for(int i=0;i<NQ*DK;i++) qf[i]=sinf(i*0.013f);
  for(int i=0;i<NK*DK;i++) kf[i]=cosf(i*0.017f);
  mlx_array q32=mlx_array_new_data(qf,(int[]){NQ,DK},2,MLX_FLOAT32);
  mlx_array k32=mlx_array_new_data(kf,(int[]){NK,DK},2,MLX_FLOAT32);
  mlx_array q=mlx_array_new(), k=mlx_array_new();
  mlx_astype(&q,q32,MLX_FLOAT16,s); mlx_astype(&k,k32,MLX_FLOAT16,s); mlx_array_eval(q); mlx_array_eval(k);

  fmt(SRC_MMA);
  double tm=run("qkt_mma", SRC_MMA, q, k, s, 1);
  double ts=run("qkt_sca", SRC_SCALAR_T, q, k, s, 0);
  double tu=run("qkt_simd", SRC_SCA_SIMD, q, k, s, 0);
  double macs=(double)NQ*NK*DK*ITERS;
  printf("MAC totali/run = %.3g\n", macs);
  printf("MMA        : %.4f ms/run -> %.2f GMAC/s\n", tm*1e3, macs/tm/1e9);
  printf("Scala naive: %.4f ms/run -> %.2f GMAC/s\n", ts*1e3, macs/ts/1e9);
  printf("Scala simd : %.4f ms/run -> %.2f GMAC/s\n", tu*1e3, macs/tu/1e9);
  printf("SPEEDUP MMA/naive = %.2fx | MMA/simd = %.2fx\n", ts/tm, tu/tm);
  return 0;
}
