/* qwen4_native_gdn_out_parity.c — P2 gate B: gdn_out nativo vs riferimento CPU (ds4 math). */
#include "qwen4_native_gdn_kernel.h"
#include <stdio.h>
#include <math.h>

static float sg(float x){ return x>=0.f ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x)); }

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T=2, H=2, D=4; const int hd=H*D; float eps=1e-6f;
    float *o=malloc(sizeof(float)*T*hd), *z=malloc(sizeof(float)*T*hd), *w=malloc(sizeof(float)*D);
    for(int i=0;i<T*hd;i++){ o[i]=(float)((i*17%140)-70)*0.02f; z[i]=(float)((i*31%160)-80)*0.03f; }
    for(int i=0;i<D;i++) w[i]=(float)((i*9%80)-40)*0.04f;
    float *ro=malloc(sizeof(float)*T*hd);
    for(int tok=0;tok<T;tok++) for(int h=0;h<H;h++){
        long base=((long)tok*H+h)*D; float ss=0;
        for(int d=0;d<D;d++) ss+=o[base+d]*o[base+d];
        float r=1.f/sqrtf(ss/(float)D+eps);
        for(int d=0;d<D;d++) ro[base+d]=o[base+d]*r*w[d]*sg(z[base+d]);
    }
    int osh[2]={T,hd}, wsh[1]={D};
    mlx_array oa=mlx_array_new_data(o,osh,2,MLX_FLOAT32);
    mlx_array za=mlx_array_new_data(z,osh,2,MLX_FLOAT32);
    mlx_array wa=mlx_array_new_data(w,wsh,1,MLX_FLOAT32);
    mlx_array oo=mlx_array_new();
    int rc=llm_q4n_gdn_out(&oo,oa,za,wa,H,D,eps,st);
    int ok=(rc==0); float mx=0.f;
    if(ok){ mlx_array_eval(oo); const float*p=mlx_array_data_float32(oo);
        for(int i=0;i<T*hd&&p;i++){float d=fabsf(p[i]-ro[i]); if(d>mx)mx=d;}
        ok=(mx<1e-5f);
    }
    printf("qwen4_native P2 gdn_out parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
