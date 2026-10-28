#include "qwen4_native_dispatch.h"
#include <stdio.h>
#include <math.h>
int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream st=mlx_default_gpu_stream_new(); float eps=1e-6f;
  int B=1,L=2,H=3,D=4; float x[B*L*H*D],w[D],ref[B*L*H*D];
  for(int i=0;i<B*L*H*D;i++) x[i]=(i%7)*0.1f-0.3f;
  for(int i=0;i<D;i++) w[i]=(i%3)*0.05f;
  for(int r=0;r<B*L*H;r++){ float ss=0; for(int d=0;d<D;d++){float v=x[r*D+d];ss+=v*v;} float inv=1.f/sqrtf(ss/(float)D+eps);
    for(int d=0;d<D;d++) ref[r*D+d]=x[r*D+d]*inv*(1.f+w[d]); }
  int xsh[4]={B,L,H,D}, wsh[1]={D};
  mlx_array X=mlx_array_new_data(x,xsh,4,MLX_FLOAT32),W=mlx_array_new_data(w,wsh,1,MLX_FLOAT32);
  mlx_array o=mlx_array_new(); int rc=qwen4_native_qk_norm(&o,X,W,eps,st); int ok=(rc==0); float mx=0;
  if(ok){mlx_array_eval(o);const float*p=mlx_array_data_float32(o);for(int i=0;i<B*L*H*D;i++){float d=fabsf(p[i]-ref[i]);if(d>mx)mx=d;} ok=mx<1e-5f;}
  printf("qwen4_native P7 qk_norm parity: rc=%d max|diff|=%.3g %s\n",rc,mx,ok?"PASS":"FAIL");
  return ok?0:1;
}
