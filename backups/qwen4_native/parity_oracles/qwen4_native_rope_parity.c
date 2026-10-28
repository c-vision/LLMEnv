#include "qwen4_native_dispatch.h"
#include <stdio.h>
#include <math.h>
int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream st=mlx_default_gpu_stream_new();
  int B=1,T=2,H=2,dim=8,rd=4,half=rd/2; float base=10000.f;
  float x[B*T*H*dim],pos[B*T],ref[B*T*H*dim];
  for(int i=0;i<B*T*H*dim;i++) x[i]=(i%7)*0.1f-0.3f;
  pos[0]=0; pos[1]=5;
  for(int b=0;b<B;b++)for(int t=0;t<T;t++)for(int h=0;h<H;h++)for(int i=0;i<dim;i++){
    long idx=((long)(b*T+t)*H+h)*dim+i;
    if(i>=rd){ref[idx]=x[idx];continue;}
    int n=i%half; float ang=pos[b*T+t]*powf(base,-2.f*n/(float)rd);
    float c=cosf(ang),sn=sinf(ang); float xr=x[idx];
    float other=(i<half)?-x[idx+half]:x[idx-half];
    ref[idx]=xr*c+other*sn;
  }
  int xsh[4]={B,T,H,dim}, psh[2]={B,T};
  mlx_array X=mlx_array_new_data(x,xsh,4,MLX_FLOAT32),P=mlx_array_new_data(pos,psh,2,MLX_FLOAT32);
  mlx_array o=mlx_array_new(); int rc=qwen4_native_rope_partial(&o,X,P,rd,base,st); int ok=(rc==0); float mx=0;
  if(ok){mlx_array_eval(o);const float*p=mlx_array_data_float32(o);for(int i=0;i<B*T*H*dim;i++){float d=fabsf(p[i]-ref[i]);if(d>mx)mx=d;} ok=mx<1e-5f;}
  printf("qwen4_native P7 rope_partial parity: rc=%d max|diff|=%.3g %s\n",rc,mx,ok?"PASS":"FAIL");
  return ok?0:1;
}
