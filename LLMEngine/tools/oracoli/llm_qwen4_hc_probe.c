/* llm_qwen4_hc_probe.c — ORACOLO model-free per la hyper-connection di qwen4_exp (P6).
 * Riproduce la formula di qwen4_exp_gated_residual_q + llm_mlx_hc_combine alle SHAPE REALI
 * (hc=4, hidden=2560, hc_width=10240, lowrank=320) e misura: numero op MLX, build+eval ms.
 * È il BEFORE che una fusione (kernel dedicato) deve battere. Nessun modello.
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static mlx_stream st; static int NOPS=0;
#define OP(x) do { (x); NOPS++; } while(0)
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e6;}
static mlx_array full(int* sh,int nd,float v){mlx_array a=mlx_array_new(),x=mlx_array_new_float32(v);mlx_full(&a,sh,nd,x,MLX_FLOAT32,st);mlx_array_free(x);return a;}
static mlx_array mm(mlx_array x, mlx_array w){ mlx_array wt=mlx_array_new(),o=mlx_array_new(); OP(mlx_transpose(&wt,w,st)); OP(mlx_matmul(&o,x,wt,st)); mlx_array_free(wt); return o; }

int main(void){
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    st=mlx_default_gpu_stream_new();
    const int hc=4, hidden=2560, hcw=hc*hidden, low=320;
    int xsh[3]={1,1,hcw}; mlx_array x=full(xsh,3,0.3f);
    int nw[1]={hcw}; mlx_array hcw_w=full(nw,1,0.1f);
    int dsh[2]={low,hcw}; mlx_array down_w=full(dsh,2,0.01f);
    int ush[2]={hcw,low}; mlx_array up_w=full(ush,2,0.01f);
    int ish[2]={hc,hcw}; mlx_array inj_w=full(ish,2,0.01f);
    int osh[3]={1,1,hidden}; mlx_array output=full(osh,3,0.5f);
    mlx_array_eval(x);mlx_array_eval(hcw_w);mlx_array_eval(down_w);mlx_array_eval(up_w);mlx_array_eval(inj_w);mlx_array_eval(output);
    mlx_array one=mlx_array_new_float32(1.0f), two=mlx_array_new_float32(2.0f), hcf=mlx_array_new_float32((float)hc);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    NOPS=0;
    /* zcrms group=hidden: reshape (1,1,hc,hidden); rms; *(1+w) */
    int g4[4]={1,1,hc,hidden}; mlx_array xr=mlx_array_new(); OP(mlx_reshape(&xr,x,g4,4,st));
    mlx_array sq=mlx_array_new(); OP(mlx_square(&sq,xr,st));
    mlx_array ms=mlx_array_new(); OP(mlx_mean_axis(&ms,sq,-1,true,st));
    mlx_array rs=mlx_array_new(); OP(mlx_rsqrt(&rs,ms,st));
    mlx_array nr4=mlx_array_new(); OP(mlx_multiply(&nr4,xr,rs,st));
    int w4[2]={hc,hidden}; mlx_array wr=mlx_array_new(); OP(mlx_reshape(&wr,hcw_w,w4,2,st));
    mlx_array one_plus_w=mlx_array_new(); OP(mlx_add(&one_plus_w,one,wr,st));
    mlx_array nrm4=mlx_array_new(); OP(mlx_multiply(&nrm4,nr4,one_plus_w,st));
    mlx_array normed=mlx_array_new(); OP(mlx_reshape(&normed,nrm4,xsh,3,st));
    /* down -> /hc -> silu -> up -> sigmoid */
    mlx_array dn=mm(normed,down_w);
    mlx_array dn2=mlx_array_new(); OP(mlx_divide(&dn2,dn,hcf,st));
    mlx_array sg=mlx_array_new(); OP(mlx_sigmoid(&sg,dn2,st));
    mlx_array si=mlx_array_new(); OP(mlx_multiply(&si,dn2,sg,st));
    mlx_array up=mm(si,up_w);
    mlx_array mix=mlx_array_new(); OP(mlx_sigmoid(&mix,up,st));
    /* mix_r * normed_r -> mean(-2) */
    mlx_array mixr=mlx_array_new(); OP(mlx_reshape(&mixr,mix,g4,4,st));
    mlx_array prod=mlx_array_new(); OP(mlx_multiply(&prod,mixr,nrm4,st));
    mlx_array mixed=mlx_array_new(); OP(mlx_mean_axis(&mixed,prod,-2,false,st));
    /* inject -> /hc -> 2*sigmoid */
    mlx_array inj=mm(normed,inj_w);
    mlx_array inj2=mlx_array_new(); OP(mlx_divide(&inj2,inj,hcf,st));
    mlx_array injs=mlx_array_new(); OP(mlx_sigmoid(&injs,inj2,st));
    mlx_array injection=mlx_array_new(); OP(mlx_multiply(&injection,injs,two,st));
    /* hc_combine: output[...,1,:]*injection[...,:,1] -> flatten -> + residual(normed) */
    int oe[4]={1,1,1,hidden}; mlx_array oute=mlx_array_new(); OP(mlx_reshape(&oute,output,oe,4,st));
    int ie[4]={1,1,hc,1}; mlx_array inje=mlx_array_new(); OP(mlx_reshape(&inje,injection,ie,4,st));
    mlx_array injprod=mlx_array_new(); OP(mlx_multiply(&injprod,oute,inje,st));
    mlx_array injflat=mlx_array_new(); OP(mlx_reshape(&injflat,injprod,xsh,3,st));
    mlx_array combined=mlx_array_new(); OP(mlx_add(&combined,normed,injflat,st));
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double build=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
    mlx_array_eval(mixed); mlx_array_eval(combined);   /* warmup: compila gli shader Metal */
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for (int r=0;r<20;r++){ mlx_array_eval(mixed); mlx_array_eval(combined); }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ev=((t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6)/20.0;
    printf("qwen4_exp hyper-connection (hc=4,hidden=2560) — UN layer attn/mlp sub-path:\n");
    printf("  op MLX: %d   build=%.3f ms   eval=%.3f ms   TOT=%.3f ms\n", NOPS, build, ev, build+ev);
    printf("  (x48 layer x2 (attn+mlp) => ~%.1f ms/token se fosse questo il collo)\n", (build+ev)*48*2);
    mlx_stream_free(st); mlx_device_free(dev); return 0;
}
