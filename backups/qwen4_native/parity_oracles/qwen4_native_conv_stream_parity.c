/* qwen4_native_conv_stream_parity.c — P2 gate B: conv_stream nativo vs riferimento CPU (ds4 math). */
#include "qwen4_native_gdn_kernel.h"
#include <stdio.h>
#include <math.h>

static float sg(float x){ return x>=0.f ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x)); }

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T=3, C=4, K=3, H=K-1;
    float *x=malloc(sizeof(float)*T*C), *s0=malloc(sizeof(float)*H*C), *w=malloc(sizeof(float)*C*K);
    for(int i=0;i<T*C;i++) x[i]=(float)((i*19%140)-70)*0.03f;
    for(int i=0;i<H*C;i++) s0[i]=(float)((i*11%100)-50)*0.04f;
    for(int i=0;i<C*K;i++) w[i]=(float)((i*7%90)-45)*0.05f;
    float *rx=malloc(sizeof(float)*T*C), *rs=malloc(sizeof(float)*H*C);
    for(int i=0;i<T*C;i++) rx[i]=x[i];
    for(int i=0;i<H*C;i++) rs[i]=s0[i];
    for(int c=0;c<C;c++){
        float win[3]; for(int t=0;t+1<K;t++) win[t]=rs[t*C+c];
        float taps[4]; for(int t=0;t<K;t++) taps[t]=w[c*K+t];
        for(int tok=0;tok<T;tok++){
            float raw=rx[tok*C+c]; float acc=taps[K-1]*raw;
            for(int t=0;t+1<K;t++) acc+=taps[t]*win[t];
            for(int t=0;t+2<K;t++) win[t]=win[t+1];
            win[K-2]=raw;
            float sgv=sg(acc); rx[tok*C+c]=acc*sgv;
        }
        for(int t=0;t+1<K;t++) rs[t*C+c]=win[t];
    }
    int xsh[2]={T,C}, ssh[2]={H,C}, wsh[2]={C,K};
    mlx_array xa=mlx_array_new_data(x,xsh,2,MLX_FLOAT32);
    mlx_array sa=mlx_array_new_data(s0,ssh,2,MLX_FLOAT32);
    mlx_array wa=mlx_array_new_data(w,wsh,2,MLX_FLOAT32);
    mlx_array xo=mlx_array_new(), so=mlx_array_new();
    int rc=llm_q4n_conv_stream(&xo,&so,xa,sa,wa,K,1,st);
    int ok=(rc==0); float mx=0.f;
    if(ok){ mlx_array_eval(xo); mlx_array_eval(so);
        const float*px=mlx_array_data_float32(xo),*ps=mlx_array_data_float32(so);
        for(int i=0;i<T*C&&px;i++){float d=fabsf(px[i]-rx[i]); if(d>mx)mx=d;}
        for(int i=0;i<H*C&&ps;i++){float d=fabsf(ps[i]-rs[i]); if(d>mx)mx=d;}
        ok=(mx<1e-5f);
    }
    printf("qwen4_native P2 conv_stream parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
