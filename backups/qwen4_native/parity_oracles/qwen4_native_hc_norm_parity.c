/* qwen4_native_hc_norm_parity.c — P1 gate B: kernel nativo vs riferimento CPU (zero-centered RMSNorm f32). */
#include "qwen4_native_hc_kernel.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T = 2, E = 8, hc = 4, hcE = hc*E;
    float eps = 1e-6f;
    float* R   = malloc(sizeof(float)*T*hcE);
    float* gam = malloc(sizeof(float)*hcE);
    float* ref = malloc(sizeof(float)*T*hcE);
    for (int i=0;i<T*hcE;i++) R[i] = (float)((i*41%220)-110)*0.01f;
    for (int i=0;i<hcE;i++)   gam[i] = (float)((i*17%100)-50)*0.02f;
    for (int t=0;t<T;t++) for (int s=0;s<hc;s++) {
        int base=(t*hc+s)*E; float ss=0.f;
        for (int d=0;d<E;d++) ss += R[base+d]*R[base+d];
        float inv = 1.f/sqrtf(ss/(float)E + eps);
        for (int d=0;d<E;d++) ref[base+d] = R[base+d]*inv*(1.f+gam[s*E+d]);
    }
    int rsh[2]={T,hcE}, gsh[1]={hcE};
    mlx_array Ra=mlx_array_new_data(R,rsh,2,MLX_FLOAT32);
    mlx_array ga=mlx_array_new_data(gam,gsh,1,MLX_FLOAT32);
    mlx_array Xn=mlx_array_new();
    int rc = llm_q4n_hc_norm(&Xn, Ra, ga, hc, E, eps, st);
    int ok = (rc==0); float mx=0.f;
    if (ok) { mlx_array_eval(Xn); const float* p=mlx_array_data_float32(Xn);
        for (int i=0;i<T*hcE && p;i++){ float d=fabsf(p[i]-ref[i]); if(d>mx) mx=d; }
        ok = (mx < 1e-5f);
    }
    printf("qwen4_native P1 hc_norm parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
