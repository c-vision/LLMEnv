/* qwen4_native_gdn_scan_parity.c — P2 gate B: gdn_scan nativo vs riferimento CPU (ds4 math). */
#include "qwen4_native_gdn_kernel.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T=2, Hk=1, Hv=2, D=3; const int cd = 2*Hk*D + Hv*D;
    float *qkv=malloc(sizeof(float)*T*cd);
    float *ga=malloc(sizeof(float)*T*Hv), *gb=malloc(sizeof(float)*T*Hv);
    float *s0=malloc(sizeof(float)*Hv*D*D);
    for(int i=0;i<T*cd;i++) qkv[i]=(float)((i*23%160)-80)*0.02f;
    for(int i=0;i<T*Hv;i++){ ga[i]=0.7f+0.05f*i; gb[i]=0.3f+0.03f*i; }
    for(int i=0;i<Hv*D*D;i++) s0[i]=(float)((i*13%120)-60)*0.01f;
    float *rs=malloc(sizeof(float)*Hv*D*D), *ro=malloc(sizeof(float)*T*Hv*D);
    for(int i=0;i<Hv*D*D;i++) rs[i]=s0[i];
    for(int tok=0;tok<T;tok++) for(int h=0;h<Hv;h++){
        int kh=h%Hk; float g=ga[tok*Hv+h], beta=gb[tok*Hv+h];
        for(int dv=0;dv<D;dv++){
            float*S=&rs[((long)h*D+dv)*D]; float u=0;
            for(int dk=0;dk<D;dk++){ S[dk]*=g; u+=S[dk]*qkv[(long)tok*cd+(Hk+kh)*D+dk]; }
            float v=qkv[(long)tok*cd+2*Hk*D+h*D+dv]; float delta=(v-u)*beta; float o=0;
            for(int dk=0;dk<D;dk++){ S[dk]+=qkv[(long)tok*cd+(Hk+kh)*D+dk]*delta; o+=S[dk]*qkv[(long)tok*cd+kh*D+dk]; }
            ro[((long)tok*Hv+h)*D+dv]=o;
        }
    }
    int qsh[2]={T,cd}, gsh[2]={T,Hv}, ssh[3]={Hv,D,D};
    mlx_array qa=mlx_array_new_data(qkv,qsh,2,MLX_FLOAT32);
    mlx_array a=mlx_array_new_data(ga,gsh,2,MLX_FLOAT32);
    mlx_array b=mlx_array_new_data(gb,gsh,2,MLX_FLOAT32);
    mlx_array sa=mlx_array_new_data(s0,ssh,3,MLX_FLOAT32);
    mlx_array so=mlx_array_new(), oo=mlx_array_new();
    int rc=llm_q4n_gdn_scan(&so,&oo,qa,a,b,sa,Hk,Hv,D,st);
    int ok=(rc==0); float mx=0.f;
    if(ok){ mlx_array_eval(so); mlx_array_eval(oo);
        const float*ps=mlx_array_data_float32(so),*po=mlx_array_data_float32(oo);
        for(int i=0;i<Hv*D*D&&ps;i++){float d=fabsf(ps[i]-rs[i]); if(d>mx)mx=d;}
        for(int i=0;i<T*Hv*D&&po;i++){float d=fabsf(po[i]-ro[i]); if(d>mx)mx=d;}
        ok=(mx<1e-5f);
    }
    printf("qwen4_native P2 gdn_scan parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
