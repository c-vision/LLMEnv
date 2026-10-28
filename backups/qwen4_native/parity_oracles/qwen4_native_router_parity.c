#include "qwen4_native_dispatch.h"
#include <stdio.h>
#include <math.h>
int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream st=mlx_default_gpu_stream_new();
  int T=2,NE=6,K=3; float lg[12]; int ri[6]; float rp[6];
  for(int i=0;i<T*NE;i++) lg[i]=(float)((i*17%23)-11)*0.3f;
  for(int t=0;t<T;t++){ float mx=-1e30f; for(int e=0;e<NE;e++) mx=fmaxf(mx,lg[t*NE+e]);
    float s=0; for(int e=0;e<NE;e++) s+=expf(lg[t*NE+e]-mx); int used[16]={0};
    for(int r=0;r<K;r++){ float bv=-1; int bi=-1; for(int e=0;e<NE;e++){ if(used[e])continue; float p=expf(lg[t*NE+e]-mx)/s; if(p>bv){bv=p;bi=e;} }
      used[bi]=1; ri[t*K+r]=bi; rp[t*K+r]=bv; }
    float ps=0; for(int r=0;r<K;r++) ps+=rp[t*K+r]; for(int r=0;r<K;r++) rp[t*K+r]/=ps; }
  int l2[2]={T,NE}; mlx_array La=mlx_array_new_data(lg,l2,2,MLX_FLOAT32);
  mlx_array ti=mlx_array_new(), pr=mlx_array_new();
  int rc=qwen4_native_router(&ti,&pr,La,K,1,st); int ok=(rc==0); float mx=0;
  if(ok){ mlx_array_eval(ti); mlx_array_eval(pr); const int*pi=mlx_array_data_int32(ti); const float*pp=mlx_array_data_float32(pr);
    for(int t=0;t<T;t++){ int got[16]; for(int r=0;r<K;r++) got[r]=pi[t*K+r];
      int seen[16]={0}; for(int r=0;r<K;r++) seen[got[r]]=1; for(int r=0;r<K;r++) if(!seen[ri[t*K+r]]) ok=0;
      for(int r=0;r<K;r++){ float d=fabsf(pp[t*K+r]-rp[t*K+r]); if(d>mx)mx=d; } } ok=ok&&(mx<1e-5f); }
  printf("qwen4_native P7 router parity: rc=%d max|diff(probs)|=%.3g %s\n",rc,mx,ok?"PASS":"FAIL");
  return ok?0:1;
}
