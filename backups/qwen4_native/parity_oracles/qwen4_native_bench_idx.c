/* qwen4_native_bench_idx.c — A/B: idx_score nativo vs op MLX equivalenti (matmul+relu+sum). */
#include "qwen4_native_qsa_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0+ts.tv_nsec/1e6; }
int main(int argc,char**argv){
    int T=(argc>1)?atoi(argv[1]):1;
    int Hi=4, Di=128, NB=512, reps=200;
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    mlx_stream st=mlx_default_gpu_stream_new();
    float *iq=malloc(sizeof(float)*T*Hi*Di); _Float16 *bk=malloc(sizeof(_Float16)*NB*Di);
    for(int i=0;i<T*Hi*Di;i++) iq[i]=(float)((i*17%100)-50)*0.01f;
    for(int i=0;i<NB*Di;i++) bk[i]=(_Float16)((float)((i*13%100)-50)*0.01f);
    int i2[2]={T,Hi*Di}, k2[2]={NB,Di};
    mlx_array iqa=mlx_array_new_data(iq,i2,2,MLX_FLOAT32), bka=mlx_array_new_data(bk,k2,2,MLX_FLOAT16);
    /* MLX baseline: reshape q [T,Hi,Di]; keyT [Di,NB]; matmul -> [T,Hi,NB]; relu; sum -2 -> [T,NB] */
    float *bkt=malloc(sizeof(float)*Di*NB);
    for(int b=0;b<NB;b++) for(int d=0;d<Di;d++) bkt[d*NB+b]=(float)bk[b*Di+d];
    int kt2[2]={Di,NB}; mlx_array bktA=mlx_array_new_data(bkt,kt2,2,MLX_FLOAT32);
    for(int w=0;w<3;w++){
        mlx_array o=mlx_array_new(); llm_q4n_idx_score(&o,iqa,bka,Hi,Di,0,1,st); mlx_array_eval(o); mlx_array_free(o);
        mlx_array q3=mlx_array_new(); int s3[3]={T,Hi,Di}; mlx_reshape(&q3,iqa,s3,3,st);
        mlx_array mm=mlx_array_new(); mlx_matmul(&mm,q3,bktA,st);
        mlx_array z=mlx_array_new_float32(0.f); mlx_array r=mlx_array_new(); mlx_maximum(&r,mm,z,st);
        mlx_array o2=mlx_array_new(); mlx_sum_axis(&o2,r,-2,false,st); mlx_array_eval(o2);
        mlx_array_free(q3);mlx_array_free(mm);mlx_array_free(z);mlx_array_free(r);mlx_array_free(o2);
    }
    double t0=now_ms();
    for(int i=0;i<reps;i++){ mlx_array o=mlx_array_new(); llm_q4n_idx_score(&o,iqa,bka,Hi,Di,0,1,st); mlx_array_eval(o); mlx_array_free(o); }
    double tn=now_ms()-t0;
    t0=now_ms();
    for(int i=0;i<reps;i++){
        mlx_array q3=mlx_array_new(); int s3[3]={T,Hi,Di}; mlx_reshape(&q3,iqa,s3,3,st);
        mlx_array mm=mlx_array_new(); mlx_matmul(&mm,q3,bktA,st);
        mlx_array z=mlx_array_new_float32(0.f); mlx_array r=mlx_array_new(); mlx_maximum(&r,mm,z,st);
        mlx_array o2=mlx_array_new(); mlx_sum_axis(&o2,r,-2,false,st); mlx_array_eval(o2);
        mlx_array_free(q3);mlx_array_free(mm);mlx_array_free(z);mlx_array_free(r);mlx_array_free(o2);
    }
    double tm=now_ms()-t0;
    printf("BENCH idx_score T=%d Hi=%d Di=%d NB=%d reps=%d\n",T,Hi,Di,NB,reps);
    printf("  nativo    : %.4f ms/op\n", tn/reps);
    printf("  MLX ops   : %.4f ms/op   -> nativo/mlx = %.3fx (>1 = nativo piu' veloce)\n", tm/reps, tm/tn);
    mlx_stream_free(st); mlx_device_free(dev); return 0;
}
