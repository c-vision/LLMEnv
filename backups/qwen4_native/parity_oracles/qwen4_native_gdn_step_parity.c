#include "qwen4_native_dispatch.h"
#include <stdio.h>
#include <math.h>
int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream st=mlx_default_gpu_stream_new();
  int Hv=2,Dv=3,Dk=4;
  float q[1*Hv*Dk],k[1*Hv*Dk],v[1*Hv*Dv],g[Hv],beta[Hv],S[1*Hv*Dv*Dk];
  for(int i=0;i<Hv*Dk;i++){q[i]=(i%5)*0.1f-0.2f;k[i]=(i%7)*0.05f-0.15f;}
  for(int i=0;i<Hv*Dv;i++)v[i]=(i%4)*0.07f-0.1f;
  for(int i=0;i<Hv;i++){g[i]=0.8f;beta[i]=0.5f;}
  for(int i=0;i<Hv*Dv*Dk;i++)S[i]=(i%6)*0.01f;
  float Sref[1*Hv*Dv*Dk],yref[1*Hv*Dv];
  for(int i=0;i<Hv*Dv*Dk;i++)Sref[i]=S[i];
  for(int h=0;h<Hv;h++)for(int dv=0;dv<Dv;dv++){
    float*SS=&Sref[(h*Dv+dv)*Dk]; float acc=0;
    for(int dk=0;dk<Dk;dk++){SS[dk]*=g[h]; acc+=SS[dk]*k[h*Dk+dk];}
    float delta=(v[h*Dv+dv]-acc)*beta[h]; float yv=0;
    for(int dk=0;dk<Dk;dk++){SS[dk]+=k[h*Dk+dk]*delta; yv+=SS[dk]*q[h*Dk+dk];}
    yref[h*Dv+dv]=yv;
  }
  int q3[3]={1,Hv,Dk},v3[3]={1,Hv,Dv},g2[2]={1,Hv},s4[4]={1,Hv,Dv,Dk};
  mlx_array Q=mlx_array_new_data(q,q3,3,MLX_FLOAT32),K=mlx_array_new_data(k,q3,3,MLX_FLOAT32),V=mlx_array_new_data(v,v3,3,MLX_FLOAT32);
  mlx_array G=mlx_array_new_data(g,g2,2,MLX_FLOAT32),B=mlx_array_new_data(beta,g2,2,MLX_FLOAT32),ST=mlx_array_new_data(S,s4,4,MLX_FLOAT32);
  mlx_array y=mlx_array_new(),so=mlx_array_new(); mlx_array_set(&so,ST);
  int rc=qwen4_native_gdn_step(&y,&so,Q,K,V,G,B,st); int ok=(rc==0); float mx=0;
  if(ok){mlx_array_eval(y);mlx_array_eval(so);
    const float*py=mlx_array_data_float32(y),*ps=mlx_array_data_float32(so);
    for(int i=0;i<Hv*Dv;i++){float d=fabsf(py[i]-yref[i]); if(d>mx)mx=d;}
    for(int i=0;i<Hv*Dv*Dk;i++){float d=fabsf(ps[i]-Sref[i]); if(d>mx)mx=d;} ok=mx<1e-4f;}
  printf("qwen4_native P7 gdn_step parity: rc=%d max|diff|=%.3g %s\n",rc,mx,ok?"PASS":"FAIL");
  return ok?0:1;
}
