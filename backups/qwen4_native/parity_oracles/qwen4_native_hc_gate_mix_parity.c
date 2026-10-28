/* qwen4_native_hc_gate_mix_parity.c — P1 gate B: gate_mix nativo vs riferimento CPU (f32). */
#include "qwen4_native_hc_kernel.h"
#include <stdio.h>
#include <math.h>

static float silu(float x){ return x/(1.f+expf(-x)); }
static float sig(float x){ return 1.f/(1.f+expf(-x)); }

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0); mlx_set_default_device(dev);
    mlx_stream st = mlx_default_gpu_stream_new();
    const int T = 2, E = 8, hc = 4, R = 3, hcE = hc*E;
    float* xn  = malloc(sizeof(float)*T*hcE);
    float* lo  = malloc(sizeof(float)*T*R);
    float* wup = malloc(sizeof(float)*hcE*R);
    float* ref = malloc(sizeof(float)*T*E);
    for (int i=0;i<T*hcE;i++) xn[i]  = (float)((i*31%180)-90)*0.01f;
    for (int i=0;i<T*R;i++)   lo[i]  = (float)((i*23%140)-70)*0.02f;
    for (int i=0;i<hcE*R;i++) wup[i] = (float)((i*13%160)-80)*0.015f;
    for (int t=0;t<T;t++) for (int d=0;d<E;d++) {
        float g=0.f;
        for (int s=0;s<hc;s++){
            float acc=0.f;
            for (int r=0;r<R;r++) acc += wup[(s*E+d)*R+r]*silu(lo[t*R+r]/(float)hc);
            g += sig(acc)*xn[t*hc*E+s*E+d];
        }
        ref[t*E+d]=g/(float)hc;
    }
    int xsh[2]={T,hcE}, lsh[2]={T,R}, wsh[2]={hcE,R};
    mlx_array xa=mlx_array_new_data(xn,xsh,2,MLX_FLOAT32);
    mlx_array la=mlx_array_new_data(lo,lsh,2,MLX_FLOAT32);
    mlx_array wa=mlx_array_new_data(wup,wsh,2,MLX_FLOAT32);
    mlx_array m=mlx_array_new();
    int rc = llm_q4n_hc_gate_mix(&m, xa, la, wa, hc, E, st);
    int ok=(rc==0); float mx=0.f;
    if (ok) { mlx_array_eval(m); const float* p=mlx_array_data_float32(m);
        for (int i=0;i<T*E && p;i++){ float d=fabsf(p[i]-ref[i]); if(d>mx) mx=d; }
        ok=(mx<1e-5f);
    }
    printf("qwen4_native P1 hc_gate_mix parity: rc=%d max|diff|=%.3g %s\n", rc, mx, ok?"PASS":"FAIL");
    mlx_stream_free(st); mlx_device_free(dev);
    return ok?0:1;
}
