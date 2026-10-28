/* mma_frag_io_probe.c — ORACOLO: round-trip di UN frammento simdgroup_half8x8.
 * S[0:8,0:8] = load(q[0:8,0:8]) + store. Riferimento = qf[0:8,0:8].
 * Serve a capire se MLX fast_metal_kernel supporta i frammenti (load/store). */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char* HDR = "using namespace metal;\n";
static const char* SRC =
  "    constexpr uint DH = 64u, N = 8u;\n"
  "    simdgroup_half8x8 A0;\n"
  "    simdgroup_load(A0, q, DH);\n"
  "    simdgroup_store(A0, S, N);\n";

static float h2f(uint16_t h){
  unsigned e=(h>>10)&0x1f, mnt=h&0x3ff; int sg=h>>15;
  if(e==0) return sg?-0.0f:0.0f;
  if(e==0x1f) return sg?-INFINITY:INFINITY;
  float v=1.0f+mnt/1024.0f; return sg?-ldexpf(v,e-15):ldexpf(v,e-15);
}

int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream s=mlx_default_gpu_stream_new();
  const int M=8, DH=64;
  float* qf=(float*)malloc(M*DH*4);
  for(int i=0;i<M*DH;i++) qf[i]=sinf(i*0.13f);
  mlx_array q32=mlx_array_new_data(qf,(int[]){M,DH},2,MLX_FLOAT32);
  mlx_array q=mlx_array_new(); mlx_astype(&q,q32,MLX_FLOAT16,s); mlx_array_eval(q);

  mlx_vector_string in=mlx_vector_string_new(); mlx_vector_string_append_value(in,"q");
  mlx_vector_string on=mlx_vector_string_new(); mlx_vector_string_append_value(on,"S");
  mlx_fast_metal_kernel kern=mlx_fast_metal_kernel_new("fio", in, on, SRC, HDR, true, false);
  if(kern.ctx==NULL){ printf("kernel build FAIL\n"); return 1; }
  mlx_fast_metal_kernel_config cfg=mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg,(int[]){8,8},2,MLX_FLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg,32,1,1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg,32,1,1);
  mlx_vector_array ins=mlx_vector_array_new(); mlx_vector_array_append_value(ins,q);
  mlx_vector_array outs=mlx_vector_array_new();
  int rc=mlx_fast_metal_kernel_apply(&outs,kern,ins,cfg,s);
  if(rc!=0){ printf("apply FAIL rc=%d\n",rc); return 1; }
  mlx_array S=mlx_array_new(); mlx_vector_array_get(&S,outs,0); mlx_array_eval(S);
  const uint16_t* Sp=mlx_array_data_uint16(S);
  double maxd=0; int bad=0;
  for(int m=0;m<8;m++) for(int n=0;n<8;n++){
    float want=qf[m*DH+n]; float got=h2f(Sp[m*8+n]);
    double df=fabs((double)got-(double)want); if(df>maxd)maxd=df; if(df>0.01) bad++;
  }
  printf("FRAG IO round-trip: bad=%d/64 maxdiff=%.4g -> %s\n", bad, maxd, bad?"FAIL":"OK");
  return 0;
}
