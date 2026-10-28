/* mma_one_probe.c — ORACOLO: UN solo MMA C[8,8] = Q[8,8] @ K^T[8,8].
 * Testa (a) load transposto di K, (b) simdgroup_multiply_accumulate. */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char* HDR = "using namespace metal;\n";
static const char* SRC =
  "    threadgroup half tk[64];\n"
  "    for (uint i = thread_position_in_threadgroup.x; i < 64u; i += 32u) tk[(i%8u)*8u + (i/8u)] = kk[i];\n"
  "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "    simdgroup_half8x8 B;\n"
  "    simdgroup_load(B, tk, 8u);\n"
  "    simdgroup_store(B, S, 8u);\n";

static float h2f(uint16_t h){
  unsigned e=(h>>10)&0x1f, mnt=h&0x3ff; int sg=h>>15;
  if(e==0) return 0.0f; if(e==0x1f) return sg?-INFINITY:INFINITY;
  float v=1.0f+mnt/1024.0f; return sg?-ldexpf(v,e-15):ldexpf(v,e-15);
}

int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream s=mlx_default_gpu_stream_new();
  const int D=8;
  float* qf=(float*)malloc(64*4); float* kf=(float*)malloc(64*4);
  for(int i=0;i<64;i++){ qf[i]=sinf(i*0.31f); kf[i]=cosf(i*0.17f); }
  mlx_array q32=mlx_array_new_data(qf,(int[]){D,D},2,MLX_FLOAT32);
  mlx_array k32=mlx_array_new_data(kf,(int[]){D,D},2,MLX_FLOAT32);
  mlx_array q=mlx_array_new(), k=mlx_array_new();
  mlx_astype(&q,q32,MLX_FLOAT16,s); mlx_astype(&k,k32,MLX_FLOAT16,s); mlx_array_eval(q); mlx_array_eval(k);

  mlx_vector_string in=mlx_vector_string_new(); mlx_vector_string_append_value(in,"q"); mlx_vector_string_append_value(in,"kk");
  mlx_vector_string on=mlx_vector_string_new(); mlx_vector_string_append_value(on,"S");
  mlx_fast_metal_kernel kern=mlx_fast_metal_kernel_new("one", in, on, SRC, HDR, true, false);
  if(kern.ctx==NULL){ printf("kernel build FAIL\n"); return 1; }
  mlx_fast_metal_kernel_config cfg=mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg,(int[]){8,8},2,MLX_FLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg,32,1,1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg,32,1,1);
  mlx_vector_array ins=mlx_vector_array_new(); mlx_vector_array_append_value(ins,q); mlx_vector_array_append_value(ins,k);
  mlx_vector_array outs=mlx_vector_array_new();
  int rc=mlx_fast_metal_kernel_apply(&outs,kern,ins,cfg,s);
  if(rc!=0){ printf("apply FAIL rc=%d\n",rc); return 1; }
  mlx_array S=mlx_array_new(); mlx_vector_array_get(&S,outs,0); mlx_array_eval(S);
  const uint16_t* Sp=mlx_array_data_uint16(S);
  double dK=0,dKt=0; int bK=0,bKt=0;
  for(int m=0;m<8;m++) for(int n=0;n<8;n++){
    float got=h2f(Sp[m*8+n]);
    double a=fabs((double)got-(double)kf[m*D+n]); if(a>dK)dK=a; if(a>0.05)bK++;
    double b=fabs((double)got-(double)kf[n*D+m]); if(b>dKt)dKt=b; if(b>0.05)bKt++;
  }
  printf("TRANS-load: vs K bad=%d/64 max=%.4g | vs K^T bad=%d/64 max=%.4g\n", bK,dK,bKt,dKt);
  return 0;
}
