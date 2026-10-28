/* llm_qwen4_shape_bench.c — ORACOLO per-funzione alle SHAPE REALI qwen4_exp, model-free.
 * Misura per ogni matmul che legge pesi: ms/call (build+eval) e GB/s effettivi = bytes_peso/ms.
 * Obiettivo: attribuire dove si perde banda (il decode reale fa ~6,4 GB/s su ~400 disponibili).
 * Nessun modello: pesi sintetici quantizzati alle shape reali.
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static mlx_stream st;
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static mlx_array rand_f32(int* sh,int nd){mlx_array a=mlx_array_new();mlx_array v=mlx_array_new_float32(0.01f);mlx_full(&a,sh,nd,v,MLX_FLOAT32,st);mlx_array_free(v);return a;}

typedef struct { const char* name; int out, in, gs, bits, rows; } Case;
static const Case CASES[] = {
    {"qkv+gate (12288x2560) g32b4",   12288,  2560, 32, 4, 1},
    {"gdn_out  (10240x2560) g32b4",   10240,  2560, 32, 4, 1},
    {"moe_exp  (640x2560)   g32b4 x10", 640,  2560, 32, 4, 10},
    {"lm_head  (248320x2560) g64b4", 248320,  2560, 64, 4, 1},
};

int main(void){
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    st=mlx_default_gpu_stream_new();
    printf("%-34s %10s %10s %10s %10s\n","funzione","ms/call","GB/s","bytes(MB)","flop/us");
    for (size_t c=0;c<sizeof(CASES)/sizeof(CASES[0]);c++){
        const Case* C=&CASES[c];
        int wsh[2]={C->out,C->in};
        mlx_array wf=rand_f32(wsh,2);
        mlx_optional_int g={C->gs,true}, b={C->bits,true};
        mlx_vector_array q=mlx_vector_array_new();
        mlx_array wq=mlx_array_new(),ws=mlx_array_new(),wb=mlx_array_new();
        if (mlx_quantize(&q,wf,g,b,"affine",mlx_array_new(),st)!=0 || mlx_vector_array_size(q)<2) { printf("%-34s QUANTIZE FAIL\n",C->name); continue; }
        mlx_vector_array_get(&wq,q,0); mlx_vector_array_get(&ws,q,1); if(mlx_vector_array_size(q)>=3) mlx_vector_array_get(&wb,q,2);
        mlx_array_eval(wq); mlx_array_eval(ws);
        size_t wbytes = mlx_array_nbytes(wq)+mlx_array_nbytes(ws)+(wb.ctx?mlx_array_nbytes(wb):0);
        int xsh[3]={C->rows,1,C->in};
        mlx_array x=rand_f32(xsh,3); mlx_array_eval(x);
        /* warmup */
        for(int i=0;i<3;i++){ mlx_array o=mlx_array_new(); if(mlx_quantized_matmul(&o,x,wq,ws,wb,true,g,b,"affine",st)==0) mlx_array_eval(o); mlx_array_free(o); }
        int reps=20; double t0=now_ms();
        for(int i=0;i<reps;i++){ mlx_array o=mlx_array_new(); if(mlx_quantized_matmul(&o,x,wq,ws,wb,true,g,b,"affine",st)==0) mlx_array_eval(o); mlx_array_free(o); }
        double ms=(now_ms()-t0)/reps;
        double gbs=(double)wbytes/1e9/(ms/1e3);
        double flops=2.0*(double)C->out*C->in*C->rows;
        printf("%-34s %10.3f %10.1f %10.1f %10.1f\n",C->name,ms,gbs,(double)wbytes/1e6, flops/(ms*1e3));
        mlx_array_free(wf);mlx_array_free(wq);mlx_array_free(ws);mlx_array_free(wb);mlx_vector_array_free(q);mlx_array_free(x);
    }
    /* --- MoE REALE: gather_qmm (il motore USA questo, non 10 matvec seriali) --- */
    {
        const int E=512, dff=640, h=2560, K=10;
        int wsh[3]={E,dff,h}; mlx_array wf=rand_f32(wsh,3);
        mlx_optional_int g={32,true}, b={4,true};
        mlx_vector_array q=mlx_vector_array_new();
        mlx_array wq=mlx_array_new(),ws=mlx_array_new(),wb=mlx_array_new();
        if (mlx_quantize(&q,wf,g,b,"affine",mlx_array_new(),st)==0 && mlx_vector_array_size(q)>=3) {
            mlx_vector_array_get(&wq,q,0); mlx_vector_array_get(&ws,q,1); mlx_vector_array_get(&wb,q,2);
            mlx_array_eval(wq); mlx_array_eval(ws);
            int xsh[3]={K,1,h}; mlx_array x=rand_f32(xsh,3); mlx_array_eval(x);
            int ish[1]={K}; mlx_array idx=mlx_array_new();
            int32_t ids[K]; for(int i=0;i<K;i++) ids[i]=i; mlx_array_set_data(&idx,ids,ish,1,MLX_INT32);
            mlx_array nulls=mlx_array_new();
            size_t wbytes=mlx_array_nbytes(wq)+mlx_array_nbytes(ws)+mlx_array_nbytes(wb);
            for(int i=0;i<3;i++){ mlx_array o=mlx_array_new(); if(mlx_gather_qmm(&o,x,wq,ws,wb,nulls,idx,true,g,b,"affine",true,st)==0) mlx_array_eval(o); mlx_array_free(o); }
            int reps=20; double t0=now_ms();
            for(int i=0;i<reps;i++){ mlx_array o=mlx_array_new(); if(mlx_gather_qmm(&o,x,wq,ws,wb,nulls,idx,true,g,b,"affine",true,st)==0) mlx_array_eval(o); mlx_array_free(o); }
            double ms=(now_ms()-t0)/reps;
            printf("%-34s %10.3f %10.1f %10.1f %10s\n","gather_qmm K10 dff640 E512",ms,(double)wbytes/1e9/(ms/1e3),(double)wbytes/1e6,"-");
            mlx_array_free(x);mlx_array_free(idx);mlx_array_free(nulls);
        }
        mlx_array_free(wf);mlx_array_free(wq);mlx_array_free(ws);mlx_array_free(wb);mlx_vector_array_free(q);
    }
    printf("nota: decode reale qwen4_exp = ~6,4 GB/s su ~400 disponibili → se qui le singole matmul\n"
           "danno GB/s alto, la perdita e' nel NUMERO/serializzazione delle op, non nel singolo kernel.\n");
    mlx_stream_free(st); mlx_device_free(dev); return 0;
}
