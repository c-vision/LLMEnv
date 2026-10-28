/* qwen4_native_hc_combine_parity.c — P1 gate B: kernel nativo vs riferimento CPU. Model-free. */
#include "qwen4_native_hc_kernel.h"
#include <stdio.h>
#include <math.h>

static float sig(float x){ return x>=0.f ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x)); }

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T = 2, E = 8, hc = 4, hcE = hc*E;
    float* R   = malloc(sizeof(float)*T*hcE);
    float* out = malloc(sizeof(float)*T*E);
    float* inj = malloc(sizeof(float)*T*hc);
    float* ref = malloc(sizeof(float)*T*hcE);
    for (int i=0;i<T*hcE;i++) R[i] = (float)((i*37%200)-100)*0.01f;
    for (int i=0;i<T*E;i++)   out[i] = (float)((i*53%160)-80)*0.013f;
    for (int i=0;i<T*hc;i++)  inj[i] = (float)((i*29%120)-60)*0.02f;
    for (int t=0;t<T;t++) for (int s=0;s<hc;s++) for (int d=0;d<E;d++)
        ref[(t*hc+s)*E+d] = R[(t*hc+s)*E+d] + 2.f*sig(inj[t*hc+s]/(float)hc)*out[t*E+d];

    int rsh[3]={T,hcE}, osh[2]={T,E}, ish[2]={T,hc};
    mlx_array Ra=mlx_array_new_data(R,rsh,2,MLX_FLOAT32);
    mlx_array oa=mlx_array_new_data(out,osh,2,MLX_FLOAT32);
    mlx_array ia=mlx_array_new_data(inj,ish,2,MLX_FLOAT32);
    mlx_array Rn=mlx_array_new();
    int rc = llm_q4n_hc_combine(&Rn, Ra, oa, ia, st);
    int ok = (rc==0);
    float mx = 0.f;
    if (ok) { mlx_array_eval(Rn); const float* p = mlx_array_data_float32(Rn);
        for (int i=0;i<T*hcE && p;i++){ float d=fabsf(p[i]-ref[i]); if(d>mx) mx=d; }
        ok = (mx < 1e-5f);
    }
    printf("qwen4_native P1 hc_combine parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
