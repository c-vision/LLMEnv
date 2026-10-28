/* kvq_kernel_bench.c — ORACOLO model-free: quanto costa il kernel int8 di decode?
 * Separa (A) la CHIAMATA C `llm_mlx_kvq_sdpa_decode` (config-build + 2 launch, CPU) dalla
 * (B) `eval` (lavoro GPU). Cosi' si sa se il collo e' CPU-side (config/launch) o GPU (kernel).
 * Uso: kvq_kernel_bench <N> <reps>
 */
#include "llm_mlx_splash_store.h"
#include "llm_mlx_kv_quant_sdpa_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static double med(double*a,int n){for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)if(a[j]<a[i]){double t=a[i];a[i]=a[j];a[j]=t;}return a[n/2];}
int main(int argc,char**argv){
    int N = argc>1?atoi(argv[1]):16384; int reps = argc>2?atoi(argv[2]):40;
    const int nh=32, nkv=8, hd=64;
    const float attn = 1.0f/sqrtf((float)hd);
    size_t nk=(size_t)nkv*N*hd;
    float* kb=(float*)malloc(nk*sizeof(float)); float* vb=(float*)malloc(nk*sizeof(float));
    float* qb=(float*)malloc((size_t)nh*hd*sizeof(float));
    for(size_t i=0;i<nk;i++){kb[i]=sinf((float)i*0.7f)*3.0f; vb[i]=cosf((float)i*0.31f)*2.0f;}
    for(int i=0;i<nh*hd;i++) qb[i]=sinf((float)i*0.13f);
    int ksh[3]={nkv,N,hd};
    mlx_array ka=mlx_array_new_data(kb,ksh,3,MLX_FLOAT32), va=mlx_array_new_data(vb,ksh,3,MLX_FLOAT32);
    mlx_array qa=mlx_array_new_data(qb,(int[]){nh,hd},2,MLX_FLOAT32);
    mlx_stream s=mlx_default_gpu_stream_new();
    mlx_array kq=mlx_array_new(),vq=mlx_array_new(),ks=mlx_array_new(),vs=mlx_array_new();
    if(llm_splash_store_run(&kq,&vq,&ks,&vs,ka,va,s)!=0){printf("store FAIL\n");return 1;}
    mlx_array kb0=mlx_array_new(),vb0=mlx_array_new();
    mlx_zeros(&kb0,(int[]){nkv,N,1},3,MLX_FLOAT32,s); mlx_zeros(&vb0,(int[]){nkv,N,1},3,MLX_FLOAT32,s);
    mlx_array kr=mlx_array_new(),vr=mlx_array_new();
    mlx_zeros(&kr,(int[]){nkv,1,hd},3,MLX_FLOAT32,s); mlx_zeros(&vr,(int[]){nkv,1,hd},3,MLX_FLOAT32,s);
    int npg=N/32; int32_t* pid=(int32_t*)malloc(npg*4); for(int i=0;i<npg;i++)pid[i]=i;
    mlx_array ptab=mlx_array_new_data(pid,(int[]){npg},1,MLX_INT32);
    {   /* W3: coerenza nq>1 — nq copie della STESSA query -> nq output identici == o1 */
        int nq = 4; size_t nkb = (size_t)nq*nh*hd;
        float* qbn = (float*)malloc(nkb*sizeof(float));
        for(int q=0;q<nq;q++) memcpy(qbn+(size_t)q*nh*hd, qb, (size_t)nh*hd*sizeof(float));
        mlx_array qa_nq = mlx_array_new_data(qbn,(int[]){nq,nh,hd},3,MLX_FLOAT32);
        mlx_array o1=mlx_array_new(), onq=mlx_array_new();
        llm_mlx_kvq_sdpa_decode(&o1,qa,kq,ks,kb0,vq,vs,vb0,kr,vr,N,8,8,attn,hd,ptab,s);
        llm_mlx_kvq_sdpa_decode(&onq,qa_nq,kq,ks,kb0,vq,vs,vb0,kr,vr,N,8,8,attn,hd,ptab,s);
        mlx_array_eval(o1); mlx_array_eval(onq);
        const float* p1=mlx_array_data_float32(o1); const float* pn=mlx_array_data_float32(onq);
        double md=0; for(int q=0;q<nq;q++) for(int i=0;i<nh*hd;i++){ double d=fabs((double)pn[(size_t)q*nh*hd+i]-p1[i]); if(d>md)md=d; }
        printf("W3 nq=%d: max|onq[q]-o1|=%.3e -> %s\n", nq, md, md<1e-4?"OK":"FAIL");
    }
    /* warmup */
    for(int w=0;w<5;w++){ mlx_array o=mlx_array_new(); llm_mlx_kvq_sdpa_decode(&o,qa,kq,ks,kb0,vq,vs,vb0,kr,vr,N,8,8,attn,hd,ptab,s); mlx_array_eval(o); mlx_array_free(o); }
    double A[64],B[64];
    for(int r=0;r<reps;r++){
        mlx_array o=mlx_array_new();
        double t0=now_ms(); llm_mlx_kvq_sdpa_decode(&o,qa,kq,ks,kb0,vq,vs,vb0,kr,vr,N,8,8,attn,hd,ptab,s); A[r]=now_ms()-t0;
        t0=now_ms(); mlx_array_eval(o); B[r]=now_ms()-t0; mlx_array_free(o);
    }
    printf("N=%-6d CPU(config+launch)=%.3f ms  GPU(eval)=%.3f ms  tot=%.3f\n", N, med(A,reps), med(B,reps), med(A,reps)+med(B,reps));
    return 0;
}
