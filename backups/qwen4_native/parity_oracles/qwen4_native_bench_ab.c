/* qwen4_native_bench_ab.c — MISURA (1, autorizzata): dense_mm nativo vs mlx_matmul su shape reali.
 * decode GEMV: x[1][K] . W[N][K]^T -> [1][N]. Timing medio su ripetizioni. */
#include "qwen4_native_dense_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

int main(int argc, char** argv){
    int T = (argc>1)?atoi(argv[1]):1;
    int K = 4096, N = 4096;
    int reps = 200;
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    mlx_stream st=mlx_default_gpu_stream_new();
    float *x=malloc(sizeof(float)*T*K), *w=malloc(sizeof(float)*N*K);
    for(int i=0;i<T*K;i++) x[i]=(float)((i*17%100)-50)*0.01f;
    for(int i=0;i<N*K;i++) w[i]=(float)((i*13%100)-50)*0.01f;
    int xsh[2]={T,K}, wsh[2]={N,K}, wtsh[2]={K,N};
    mlx_array xa=mlx_array_new_data(x,xsh,2,MLX_FLOAT32);
    mlx_array wa=mlx_array_new_data(w,wsh,2,MLX_FLOAT32);          /* [N][K] per nativo */
    float *wt=malloc(sizeof(float)*K*N);                            /* [K][N] per mlx_matmul */
    for(int n=0;n<N;n++) for(int k=0;k<K;k++) wt[k*N+n]=w[n*K+k];
    mlx_array wta=mlx_array_new_data(wt,wtsh,2,MLX_FLOAT32);

    /* warmup */
    for(int i=0;i<5;i++){
        mlx_array o=mlx_array_new(); llm_q4n_dense_mm(&o,xa,wa,st); mlx_array_eval(o); mlx_array_free(o);
        mlx_array mo=mlx_array_new(); mlx_matmul(&mo,xa,wta,st); mlx_array_eval(mo); mlx_array_free(mo);
    }
    /* native */
    double t0=now_ms();
    for(int i=0;i<reps;i++){ mlx_array o=mlx_array_new(); llm_q4n_dense_mm(&o,xa,wa,st); mlx_array_eval(o); mlx_array_free(o); }
    double tn=now_ms()-t0;
    /* mlx matmul */
    t0=now_ms();
    for(int i=0;i<reps;i++){ mlx_array mo=mlx_array_new(); mlx_matmul(&mo,xa,wta,st); mlx_array_eval(mo); mlx_array_free(mo); }
    double tm=now_ms()-t0;
    printf("BENCH dense_mm  T=%d K=%d N=%d reps=%d\n", T,K,N,reps);
    printf("  nativo : %.3f ms/op  (%.1f GFLOP/s)\n", tn/reps, (2.0*T*K*N/1e9)/(tn/reps/1000.0));
    printf("  mlx_matmul: %.3f ms/op  (%.1f GFLOP/s)  -> speedup nativo/mlx = %.2fx\n", tm/reps, (2.0*T*K*N/1e9)/(tm/reps/1000.0), tm/tn);
    mlx_stream_free(st); mlx_device_free(dev); return 0;
}
