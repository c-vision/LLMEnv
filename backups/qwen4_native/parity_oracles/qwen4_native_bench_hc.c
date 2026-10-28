/* qwen4_native_bench_hc.c — A/B: hyper-connection nativa (fusa) vs sequenza op MLX del motore.
 * Shape reali qwen4_exp decode: hc=4, E=2560 (hcE=10240), rank=320, T=1. */
#include "qwen4_native_forward.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0+ts.tv_nsec/1e6; }

int main(void){
    const int hc=4, E=2560, hcE=hc*E, rank=320, T=1; float eps=1e-6f;
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    mlx_stream st=mlx_default_gpu_stream_new();
    float *x=malloc(sizeof(float)*T*hcE), *gam=malloc(sizeof(float)*hcE);
    float *dw=malloc(sizeof(float)*rank*hcE), *uw=malloc(sizeof(float)*hcE*rank), *iw=malloc(sizeof(float)*hc*hcE);
    float *sub=malloc(sizeof(float)*T*E);
    for(int i=0;i<T*hcE;i++) x[i]=(float)((i*17%100)-50)*0.001f;
    for(int i=0;i<hcE;i++) gam[i]=(float)((i*7%50)-25)*0.0002f;
    for(int i=0;i<rank*hcE;i++) dw[i]=(float)((i*13%100)-50)*0.0001f;
    for(int i=0;i<hcE*rank;i++) uw[i]=(float)((i*11%100)-50)*0.0001f;
    for(int i=0;i<hc*hcE;i++) iw[i]=(float)((i*5%100)-50)*0.0001f;
    for(int i=0;i<T*E;i++) sub[i]=(float)((i*19%100)-50)*0.001f;
    int x2[2]={T,hcE}, g1[1]={hcE}, dw2[2]={rank,hcE}, uw2[2]={hcE,rank}, iw2[2]={hc,hcE}, s2[2]={T,E};
    mlx_array X=mlx_array_new_data(x,x2,2,MLX_FLOAT32), G=mlx_array_new_data(gam,g1,1,MLX_FLOAT32);
    mlx_array DW=mlx_array_new_data(dw,dw2,2,MLX_FLOAT32), UW=mlx_array_new_data(uw,uw2,2,MLX_FLOAT32);
    mlx_array IW=mlx_array_new_data(iw,iw2,2,MLX_FLOAT32), SUB=mlx_array_new_data(sub,s2,2,MLX_FLOAT32);

    /* --- MLX path: ~25 op (rmsnorm zero-centered + down + silu + up + sigmoid + mean + inject + combine) --- */
    #define MLX_RUN() do { \
        mlx_array sq=mlx_array_new(); mlx_square(&sq,X,st); \
        mlx_array mn=mlx_array_new(); mlx_mean_axis(&mn,sq,-1,true,st); \
        mlx_array ep=mlx_array_new_float32(eps); mlx_array mne=mlx_array_new(); mlx_add(&mne,mn,ep,st); \
        mlx_array rs=mlx_array_new(); mlx_rsqrt(&rs,mne,st); \
        mlx_array nrm=mlx_array_new(); mlx_multiply(&nrm,X,rs,st); \
        mlx_array one=mlx_array_new_float32(1.f); mlx_array w1=mlx_array_new(); mlx_add(&w1,G,one,st); \
        mlx_array nrm2=mlx_array_new(); mlx_multiply(&nrm2,nrm,w1,st); \
        mlx_array dWT=mlx_array_new(); mlx_transpose(&dWT,DW,st); \
        mlx_array lo=mlx_array_new(); mlx_matmul(&lo,nrm2,dWT,st); \
        mlx_array hcf=mlx_array_new_float32((float)hc); mlx_array lo2=mlx_array_new(); mlx_divide(&lo2,lo,hcf,st); \
        mlx_array lsq=mlx_array_new(); mlx_sigmoid(&lsq,lo2,st); mlx_array lsilu=mlx_array_new(); mlx_multiply(&lsilu,lo2,lsq,st); \
        mlx_array uWT=mlx_array_new(); mlx_transpose(&uWT,UW,st); \
        mlx_array up=mlx_array_new(); mlx_matmul(&up,lsilu,uWT,st); \
        mlx_array mix=mlx_array_new(); mlx_sigmoid(&mix,up,st); \
        int m3[3]={T,hc,E}; mlx_array mix3=mlx_array_new(), nrm3=mlx_array_new(); \
        mlx_reshape(&mix3,mix,m3,3,st); mlx_reshape(&nrm3,nrm2,m3,3,st); \
        mlx_array pr=mlx_array_new(); mlx_multiply(&pr,mix3,nrm3,st); \
        mlx_array mixed=mlx_array_new(); mlx_mean_axis(&mixed,pr,-2,false,st); \
        mlx_array iWT=mlx_array_new(); mlx_transpose(&iWT,IW,st); \
        mlx_array inj=mlx_array_new(); mlx_matmul(&inj,nrm2,iWT,st); \
        mlx_array inj2=mlx_array_new(); mlx_divide(&inj2,inj,hcf,st); \
        mlx_array isg=mlx_array_new(); mlx_sigmoid(&isg,inj2,st); \
        mlx_array two=mlx_array_new_float32(2.f); mlx_array is2=mlx_array_new(); mlx_multiply(&is2,isg,two,st); \
        int s3[3]={T,E,1}; mlx_array sub3=mlx_array_new(); mlx_reshape(&sub3,SUB,s3,3,st); \
        int i3[3]={T,1,hc}; mlx_array is3=mlx_array_new(); mlx_reshape(&is3,is2,i3,3,st); \
        mlx_array bl=mlx_array_new(); mlx_multiply(&bl,sub3,is3,st); \
        mlx_array blf=mlx_array_new(); mlx_reshape(&blf,bl,x2,2,st); \
        mlx_array out=mlx_array_new(); mlx_add(&out,X,blf,st); mlx_array_eval(out); \
        mlx_array_free(sq);mlx_array_free(mn);mlx_array_free(ep);mlx_array_free(mne);mlx_array_free(rs);mlx_array_free(nrm); \
        mlx_array_free(one);mlx_array_free(w1);mlx_array_free(nrm2);mlx_array_free(dWT);mlx_array_free(lo);mlx_array_free(hcf); \
        mlx_array_free(lo2);mlx_array_free(lsq);mlx_array_free(lsilu);mlx_array_free(uWT);mlx_array_free(up);mlx_array_free(mix); \
        mlx_array_free(mix3);mlx_array_free(nrm3);mlx_array_free(pr);mlx_array_free(mixed);mlx_array_free(iWT);mlx_array_free(inj); \
        mlx_array_free(inj2);mlx_array_free(isg);mlx_array_free(two);mlx_array_free(is2);mlx_array_free(sub3);mlx_array_free(is3); \
        mlx_array_free(bl);mlx_array_free(blf);mlx_array_free(out); \
    } while(0)

    /* --- Native path: hc_norm(1) + matmul lo(1) + hc_gate_mix(1) + matmul inj(1) + hc_combine(1) = 5 nodi --- */
    #define NAT_RUN() do { \
        mlx_array xn=mlx_array_new(); llm_q4n_hc_norm(&xn,X,G,hc,E,eps,st); \
        mlx_array lo=mlx_array_new(); mlx_matmul(&lo,xn,dWT_,st); \
        mlx_array mixed=mlx_array_new(); llm_q4n_hc_gate_mix(&mixed,xn,lo,UW,hc,E,st); \
        mlx_array inj=mlx_array_new(); mlx_matmul(&inj,xn,iWT_,st); \
        mlx_array out=mlx_array_new(); llm_q4n_hc_combine(&out,X,mixed,inj,st); mlx_array_eval(out); \
        mlx_array_free(xn);mlx_array_free(lo);mlx_array_free(mixed);mlx_array_free(inj);mlx_array_free(out); \
    } while(0)
    mlx_array dWT_=mlx_array_new(); mlx_transpose(&dWT_,DW,st);
    mlx_array iWT_=mlx_array_new(); mlx_transpose(&iWT_,IW,st);

    for(int w=0;w<5;w++){ MLX_RUN(); NAT_RUN(); }
    int reps=300;
    double t0=now_ms(); for(int i=0;i<reps;i++){ MLX_RUN(); } double tm=now_ms()-t0;
    t0=now_ms(); for(int i=0;i<reps;i++){ NAT_RUN(); } double tn=now_ms()-t0;
    printf("BENCH hyper-connection  hc=%d E=%d hcE=%d rank=%d T=%d reps=%d\n",hc,E,hcE,rank,T,reps);
    printf("  MLX (~25 op) : %.4f ms/op\n", tm/reps);
    printf("  nativo (3 kern+2 mm): %.4f ms/op   -> nativo/mlx = %.3fx  (>1 = nativo piu' veloce)\n", tn/reps, tm/tn);
    mlx_stream_free(st); mlx_device_free(dev); return 0;
}
