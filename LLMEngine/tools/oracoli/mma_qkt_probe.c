/* mma_qkt_probe.c — ORACOLO: verifica S = Q(8x64) @ K^T(64x32) via MMA (simdgroup_half8x8)
 * contro il riferimento scalare. Bounded: solo il QK^T del futuro kernel FA2. */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char* HDR = "using namespace metal;\n";
static const char* SRC =
  "    constexpr uint DH = 64u, N = 32u;\n"
  "    threadgroup half kt[64u*32u];\n"
  "    for (uint i = thread_position_in_threadgroup.x; i < 2048u; i += 32u)\n"
  "        kt[(i%64u)*32u + (i/64u)] = kk[i];\n"
  "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "    threadgroup half zbuf[64];\n"
  "    for (uint i = thread_position_in_threadgroup.x; i < 64u; i += 32u) zbuf[i] = (half)0;\n"
  "    simdgroup_half8x8 A[8];\n"
  "    for (uint k = 0; k < 8u; ++k) simdgroup_load(A[k], q + k*8u, DH);\n"
  "    for (uint nb = 0; nb < 4u; ++nb) {\n"
  "        simdgroup_half8x8 acc;\n"
  "        simdgroup_load(acc, zbuf, 8u);\n"
  "        for (uint k = 0; k < 8u; ++k) {\n"
  "            simdgroup_half8x8 Bk;\n"
  "            simdgroup_load(Bk, kt + (k*8u)*32u + nb*8u, 32u);\n"
  "            simdgroup_multiply_accumulate(acc, A[k], Bk, acc);\n"
  "        }\n"
  "        simdgroup_store(acc, S + nb*8u, N);\n"
  "    }\n";

int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream s=mlx_default_gpu_stream_new();
  const int M=8, N=32, DH=64;
  uint16_t* qh=(uint16_t*)malloc(M*DH*2); uint16_t* kh=(uint16_t*)malloc(N*DH*2);
  float* qf=(float*)malloc(M*DH*4); float* kf=(float*)malloc(N*DH*4);
  for(int i=0;i<M*DH;i++){ qf[i]=sinf(i*0.13f); qh[i]=0; }
  for(int i=0;i<N*DH;i++){ kf[i]=cosf(i*0.07f); kh[i]=0; }
  /* f32 -> f16 via mlx per robustezza: costruiamo gli array f16 dal f32 */
  mlx_array q32=mlx_array_new_data(qf,(int[]){M,DH},2,MLX_FLOAT32);
  mlx_array k32=mlx_array_new_data(kf,(int[]){N,DH},2,MLX_FLOAT32);
  mlx_array q=mlx_array_new(), k=mlx_array_new();
  mlx_astype(&q,q32,MLX_FLOAT16,s); mlx_astype(&k,k32,MLX_FLOAT16,s); mlx_array_eval(q); mlx_array_eval(k);

  mlx_vector_string in=mlx_vector_string_new(); mlx_vector_string_append_value(in,"q"); mlx_vector_string_append_value(in,"kk");
  mlx_vector_string on=mlx_vector_string_new(); mlx_vector_string_append_value(on,"S");
  mlx_fast_metal_kernel kern=mlx_fast_metal_kernel_new("qkt", in, on, SRC, HDR, true, false);
  if(kern.ctx==NULL){ printf("kernel build FAIL\n"); return 1; }
  mlx_fast_metal_kernel_config cfg=mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg,(int[]){M,N},2,MLX_FLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg,32,1,1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg,32,1,1);
  mlx_vector_array ins=mlx_vector_array_new(); mlx_vector_array_append_value(ins,q); mlx_vector_array_append_value(ins,k);
  mlx_vector_array outs=mlx_vector_array_new();
  int rc=mlx_fast_metal_kernel_apply(&outs,kern,ins,cfg,s);
  if(rc!=0 || mlx_vector_array_size(outs)<1){ printf("apply FAIL rc=%d\n",rc); return 1; }
  mlx_array S=mlx_array_new(); mlx_vector_array_get(&S,outs,0); mlx_array_eval(S);
  const uint16_t* Sp=mlx_array_data_uint16(S);
  /* riferimento scalare f32 dalle copie f32 */
  double maxd=0; int bad=0;
  for(int m=0;m<M;m++) for(int n=0;n<N;n++){
    float acc=0; for(int d=0;d<DH;d++) acc += qf[m*DH+d]*kf[n*DH+d];
    float got=0; { /* f16->f32 */ uint16_t h=Sp[m*N+n]; /* decode half */ unsigned e=(h>>10)&0x1f, mnt=h&0x3ff; int sg=h>>15; if(e==0) got=0; else { float v=1.0f+mnt/1024.0f; int ex=e-15; v=ldexpf(v,ex); got=sg?-v:v; } }
    double df=fabs((double)got-(double)acc); if(df>maxd)maxd=df; if(df>0.10) bad++;
  }
  printf("MMA QK^T: bad=%d/%d maxdiff=%.4g -> %s\n", bad, M*N, maxd, bad?"FAIL":"OK");
  return 0;
}
