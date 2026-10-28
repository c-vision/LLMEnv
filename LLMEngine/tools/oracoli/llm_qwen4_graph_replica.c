/* llm_qwen4_graph_replica.c — replica del grafo-token qwen4_exp (shape/op mix reali), model-free.
 * Obiettivo: capire se il costo del decode (~700-1300 ms/token) si riproduce con l'INSIEME delle op
 * (senza il modello), o se e' un effetto di sistema (allocatore/scheduler a 72 GB).
 * 48 layer: per layer ~2 qmm pesanti (12288/2560) + ~200 op elementwise/reshape chained;
 * + 1 gather_qmm (MoE K10) + 1 lm_head (248320). Una sola eval (come il path normale).
 * Pesi QUANTIZZATI sintetici, condivisi tra i layer (resident basso: non e' la RAM il punto).
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static mlx_stream st;
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e6;}
static mlx_array full_f32(int* sh,int nd,float v){mlx_array a=mlx_array_new(),x=mlx_array_new_float32(v);mlx_full(&a,sh,nd,x,MLX_FLOAT32,st);mlx_array_free(x);return a;}
static void qz(mlx_array* wq,mlx_array* ws,mlx_array* wb,mlx_array wf,int gs,int bits){
    mlx_vector_array q=mlx_vector_array_new(); mlx_optional_int g={gs,true},b={bits,true};
    *wq=mlx_array_new();*ws=mlx_array_new();*wb=mlx_array_new();
    if(mlx_quantize(&q,wf,g,b,"affine",mlx_array_new(),st)==0&&mlx_vector_array_size(q)>=3){
        mlx_vector_array_get(wq,q,0);mlx_vector_array_get(ws,q,1);mlx_vector_array_get(wb,q,2);
    }
    mlx_vector_array_free(q); mlx_array_eval(*wq);mlx_array_eval(*ws);
}
static int SMALL_PER_LAYER=200, NLAYER=48;
int main(int argc,char**argv){
    if(argc>1) NLAYER=atoi(argv[1]);
    if(argc>2) SMALL_PER_LAYER=atoi(argv[2]);
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    st=mlx_default_gpu_stream_new();
    int wsh1[2]={12288,2560}, wsh2[2]={2560,12288}, wl[2]={248320,2560}, wm[3]={512,640,2560};
    mlx_array W1=full_f32(wsh1,2,0.01f), W2=full_f32(wsh2,2,0.01f), WL=full_f32(wl,2,0.01f), WM=full_f32(wm,3,0.01f);
    mlx_array q1,s1,b1,q2,s2,b2,ql,sl,bl,qm,sm,bm;
    qz(&q1,&s1,&b1,W1,32,4); qz(&q2,&s2,&b2,W2,32,4); qz(&ql,&sl,&bl,WL,64,4); qz(&qm,&sm,&bm,WM,32,4);
    mlx_optional_int g={32,true},b={4,true},gl={64,true},bl_={4,true};
    int xsh[3]={1,1,2560}; mlx_array x=full_f32(xsh,3,0.01f); mlx_array_eval(x);
    mlx_array two=mlx_array_new_float32(2.0f), half=mlx_array_new_float32(0.5f);
    int x10sh[3]={10,1,2560}; mlx_array x10=full_f32(x10sh,3,0.01f); mlx_array_eval(x10);
    int ish[1]={10}; int32_t ids[10]; for(int i=0;i<10;i++) ids[i]=i;
    mlx_array idx=mlx_array_new_data(ids,ish,1,MLX_INT32); mlx_array nulls=mlx_array_new();
    printf("replica: %d layer x (~2 qmm + %d small) + gather_qmm + lm_head\n",NLAYER,SMALL_PER_LAYER);
    clock_gettime(CLOCK_MONOTONIC,(struct timespec[]){0}); struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    int nops=0;
    mlx_array last=mlx_array_new(); mlx_array_set(&last,x);
    for(int l=0;l<NLAYER;l++){
        mlx_array a=mlx_array_new();
        if(mlx_quantized_matmul(&a,last,q1,s1,b1,true,g,b,"affine",st)!=0) break; nops++;
        /* ~small ops chained su (1,1,12288) */
        for(int i=0;i<SMALL_PER_LAYER;i++){
            mlx_array r=mlx_array_new(); int rc;
            switch(i%5){
              case 0: rc=mlx_multiply(&r,a,two,st); break;
              case 1: rc=mlx_add(&r,a,two,st); break;
              case 2: rc=mlx_sigmoid(&r,a,st); break;
              case 3: rc=mlx_multiply(&r,a,half,st); break;
              default: rc=mlx_tanh(&r,a,st); break;
            }
            if(rc!=0){mlx_array_free(r);break;}
            mlx_array_free(a); a=r; nops++;
        }
        mlx_array nx=mlx_array_new();
        if(mlx_quantized_matmul(&nx,a,q2,s2,b2,true,g,b,"affine",st)!=0){mlx_array_free(a);break;} nops++;
        mlx_array_free(a); mlx_array_free(last); last=nx;
    }
    mlx_array mo=mlx_array_new();
    if(mlx_gather_qmm(&mo,x10,qm,sm,bm,nulls,idx,true,g,b,"affine",true,st)==0) nops++;
    mlx_array lm=mlx_array_new();
    if(mlx_quantized_matmul(&lm,last,ql,sl,bl,true,gl,bl_,"affine",st)==0) nops++;
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double build=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e-6/1e9;
    build=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    mlx_array_eval(last); if(lm.ctx) mlx_array_eval(lm); if(mo.ctx) mlx_array_eval(mo);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ev=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
    printf("nops_costruite=%d  build=%.1f ms  eval=%.1f ms  TOT=%.1f ms\n",nops,build,ev,build+ev);
    printf("(reale qwen4_exp: build~632 eval~81 TOT~713 ms @10629 nodi)\n");
    mlx_array_free(W1);mlx_array_free(W2);mlx_array_free(WL);mlx_array_free(WM);
    mlx_array_free(last);mlx_stream_free(st);mlx_device_free(dev); return 0;
}
