/* kvq_pool_scale_probe.c — ORACOLO: il pool int8 conserva i dati a N>8192?
 * Store di N token in chunk (come il flush), poi dequant e confronto col riferimento. */
#include "llm_mlx_kvq_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(int argc, char** argv){
  int N = argc>1 ? atoi(argv[1]) : 9000;
  const int nkv=2, hd=64;
  mlx_stream s = mlx_default_gpu_stream_new();
  llm_kvq_pool p; memset(&p,0,sizeof p);
  if(!llm_kvq_pool_init(&p, 1, nkv, hd, LLM_KVQ_FMT_INT8, 2048, s)){ printf("init FAIL\n"); return 1; }
  int ksh[3]={nkv,N,hd};
  size_t tot=(size_t)nkv*N*hd;
  float* kb=(float*)malloc(tot*sizeof(float));
  for(size_t i=0;i<tot;i++) kb[i]=sinf((float)i*0.7f)*3.0f;
  mlx_array K=mlx_array_new_data(kb,ksh,3,MLX_FLOAT32);
  int chunk=2048, ok=1;
  for(int pos=0; pos<N && ok; pos+=chunk){ int n=(pos+chunk<=N)?chunk:(N-pos);
     mlx_array kc=mlx_array_new();
     int a0[3]={0,pos,0}, a1[3]={nkv,pos+n,hd}, st[3]={1,1,1};
     mlx_slice(&kc,K,a0,3,a1,3,st,3,s);
     ok = llm_kvq_pool_store(&p,0,kc,kc,pos,n,s);
     mlx_array_free(kc);
     if(!ok) printf("  store FAIL at pos=%d n=%d\n",pos,n);
  }
  mlx_array_eval(p.keyData[0]); mlx_array_eval(p.keyScales[0]);
  const int8_t* KQ=mlx_array_data_int8(p.keyData[0]);
  const float* SC=mlx_array_data_float32(p.keyScales[0]);
  int cap = mlx_array_shape(p.keyData[0])[1];
  double maxdiff=0; int bad=0; int first_bad=-1;
  for(int t=0;t<N;t++){
     int lp=t/32, r=t%32; int phys=p.pt.table[lp];
     if(phys<0){ if(first_bad<0) first_bad=t; continue; }
     int prowo=phys*32+r;
     for(int h=0;h<nkv;h++){ float sc=SC[(size_t)h*cap+prowo];
        for(int d=0;d<hd;d++){ float deq=(float)KQ[((size_t)h*cap+prowo)*hd+d]*sc; float ref=kb[((size_t)h*N+t)*hd+d];
           double diff=fabs(deq-ref); if(diff>0.05){ bad++; if(first_bad<0) first_bad=t; } if(diff>maxdiff)maxdiff=diff; } } }
  printf("N=%d cap=%d committed=%d first_bad_token=%d maxdiff=%.4g\n",
         N, cap, p.committed_l[0], first_bad, maxdiff);
  return 0;
}
