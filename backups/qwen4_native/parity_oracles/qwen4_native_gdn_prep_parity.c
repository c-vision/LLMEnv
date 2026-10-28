/* qwen4_native_gdn_prep_parity.c — P2 gate B: gdn_prep nativo vs riferimento CPU (ds4 math). */
#include "qwen4_native_gdn_kernel.h"
#include <stdio.h>
#include <math.h>

static float sp(float x){ if(x>20.f) return x; if(x<-20.f) return expf(x); return logf(1.f+expf(x)); }
static float sg(float x){ return x>=0.f ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x)); }

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T=2, Hk=2, Hv=2, D=4; const int cd = 2*Hk*D + Hv*D;
    float *qkv=malloc(sizeof(float)*T*cd), *a=malloc(sizeof(float)*T*Hv), *b=malloc(sizeof(float)*T*Hv);
    float *ssm=malloc(sizeof(float)*Hv), *dtb=malloc(sizeof(float)*Hv);
    for (int i=0;i<T*cd;i++) qkv[i]=(float)((i*29%180)-90)*0.02f;
    for (int i=0;i<T*Hv;i++){ a[i]=(float)((i*13%100)-50)*0.05f; b[i]=(float)((i*7%80)-40)*0.06f; }
    for (int i=0;i<Hv;i++){ ssm[i]=-0.5f-0.1f*i; dtb[i]=0.1f*i; }
    float *rq=malloc(sizeof(float)*T*cd), *ra=malloc(sizeof(float)*T*Hv), *rb=malloc(sizeof(float)*T*Hv);
    for (int i=0;i<T*cd;i++) rq[i]=qkv[i];
    for (int t=0;t<T;t++) for (int h=0;h<Hk;h++){
        long q0=(long)t*cd+(long)h*D, k0=q0+(long)Hk*D; float sq=0,sk=0;
        for(int d=0;d<D;d++){ sq+=rq[q0+d]*rq[q0+d]; sk+=rq[k0+d]*rq[k0+d]; }
        float qs=1.f/sqrtf(sq+1e-6f)*(1.f/sqrtf((float)D)), ks=1.f/sqrtf(sk+1e-6f);
        for(int d=0;d<D;d++){ rq[q0+d]*=qs; rq[k0+d]*=ks; }
    }
    for (int t=0;t<T;t++) for (int j=0;j<Hv;j++){
        ra[t*Hv+j]=expf(ssm[j]*sp(a[t*Hv+j]+dtb[j]));
        rb[t*Hv+j]=sg(b[t*Hv+j]);
    }
    int qsh[2]={T,cd}, ash[2]={T,Hv};
    mlx_array qa=mlx_array_new_data(qkv,qsh,2,MLX_FLOAT32);
    mlx_array aa=mlx_array_new_data(a,ash,2,MLX_FLOAT32);
    mlx_array ba=mlx_array_new_data(b,ash,2,MLX_FLOAT32);
    int hs[1]={Hv};
    mlx_array sa=mlx_array_new_data(ssm,hs,1,MLX_FLOAT32);
    mlx_array da=mlx_array_new_data(dtb,hs,1,MLX_FLOAT32);
    mlx_array qo=mlx_array_new(), ao=mlx_array_new(), bo=mlx_array_new();
    int rc=llm_q4n_gdn_prep(&qo,&ao,&bo,qa,aa,ba,sa,da,Hk,Hv,D,st);
    int ok=(rc==0); float mx=0.f;
    if(ok){ mlx_array_eval(qo); mlx_array_eval(ao); mlx_array_eval(bo);
        const float*pq=mlx_array_data_float32(qo),*pa=mlx_array_data_float32(ao),*pb=mlx_array_data_float32(bo);
        for(int i=0;i<T*cd&&pq;i++){float d=fabsf(pq[i]-rq[i]); if(d>mx)mx=d;}
        for(int i=0;i<T*Hv&&pa;i++){float d=fabsf(pa[i]-ra[i]); if(d>mx)mx=d;}
        for(int i=0;i<T*Hv&&pb;i++){float d=fabsf(pb[i]-rb[i]); if(d>mx)mx=d;}
        ok=(mx<1e-5f);
    }
    printf("qwen4_native P2 gdn_prep parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
