/* kv_append_probe.c — ORACOLO model-free: l'append KV via `mlx_slice_update` è O(1) (donation)
 * o O(cap) (copia)? Confronta due pattern di gestione riferimenti al crescere di C.
 *
 * Uso: kv_append_probe <C> <reps> <mode>
 *   mode 0 = NAIVE   : A = slice_update(A, seg) con eval PRIMA di ri-settare A (vecchio A ancora
 *            referenziato dalla variabile -> use_count>=2 all'eval -> NON donatable -> copia).
 *   mode 1 = CLEARED : il buffer viene "svuotato" (la variabile cede il riferimento) PRIMA
 *            dell'eval, cosi' all'eval il buffer e' referenziato SOLO dall'op -> DONATABLE.
 *   mode 2 = CONCAT  : baseline `mlx_concatenate` (O(C) per costruzione).
 *
 * Se mode1 è ~piatto al crescere di C ⇒ basterebbe la donation (nessun kernel). Se cresce O(C)
 * come concat ⇒ serve la primitiva in-place custom (W5).
 *
 * Build: bash oracoli/build_probes.sh   (aggiunto alla lista)
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}

int main(int argc, char** argv){
    if(argc<4){printf("uso: %s <C> <reps> <mode 0=naive 1=cleared 2=concat>\n",argv[0]);return 2;}
    int C=atoi(argv[1]), reps=atoi(argv[2]), mode=atoi(argv[3]);
    mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
    mlx_stream s=mlx_default_gpu_stream_new();
    const int H=8, D=64;
    int sh[4]={1,H,C,D};
    mlx_array one=mlx_array_new_float32(1.0f), zero=mlx_array_new_float32(0.0f);
    mlx_array A=mlx_array_new(); mlx_full(&A,sh,4,zero,MLX_FLOAT16,s);
    int sg[4]={1,H,1,D}; mlx_array seg=mlx_array_new(); mlx_full(&seg,sg,4,one,MLX_FLOAT16,s);
    mlx_array_eval(A); mlx_array_eval(seg);
    if(mode==3){ /* correttezza: la GPU vede il memcpy CPU nel buffer Shared? */
        uint16_t* base=(uint16_t*)mlx_array_data_uint16(A);
        base[0]=0x3C00;   /* fp16 1.0 */
        mlx_array sm=mlx_array_new(); mlx_sum(&sm,A,false,s);
        mlx_array smf=mlx_array_new(); mlx_astype(&smf,sm,MLX_FLOAT32,s); mlx_array_eval(smf);
        const float* sd=mlx_array_data_float32(smf); float v=sd?sd[0]:-1.0f;
        printf("[check] sum dopo memcpy elem0=1.0 -> %g (atteso 1.0) => GPU %s\n",(double)v, (v>0.5?"VISTO":"NON visto"));
        mlx_array_free(sm);
    }
    /* warmup */
    for(int w=0;w<3;w++){ int st[4]={0,0,0,0},sp[4]={1,H,1,D},sx[4]={1,1,1,1};
        mlx_array r=mlx_array_new(); mlx_slice_update(&r,A,seg,st,4,sp,4,sx,4,s); mlx_array_eval(r); mlx_array_set(&A,r); mlx_array_free(r); }
    double t0=now_ms();
    for(int i=0;i<reps;i++){
        int p=i%C;
        if(mode==3){ /* memcpy in-place nel buffer persistente (StorageModeShared) + read GPU */
            uint16_t* base=(uint16_t*)mlx_array_data_uint16(A);
            for(int h=0;h<H;h++){ uint16_t* dst=base + (size_t)h*C*D + (size_t)p*D;
                for(int d=0;d<D;d++) dst[d]=0x3C00; }   /* fp16 1.0 */
            mlx_array sm=mlx_array_new(); mlx_sum(&sm,A,false,s); mlx_array_eval(sm); mlx_array_free(sm);
        } else if(mode==1){ /* rebind PRIMA dell'eval (pattern Python `a = f(a)`) */
            int st[4]={0,0,p,0},sp[4]={1,H,p+1,D},sx[4]={1,1,1,1};
            mlx_array r=mlx_array_new(); mlx_slice_update(&r,A,seg,st,4,sp,4,sx,4,s);
            mlx_array_set(&A,r);   /* il vecchio buffer resta referenziato SOLO dall'op */
            mlx_array_eval(A);     /* all'eval: use_count==1 -> DONATABLE */
            mlx_array_free(r);
        } else { /* naive */
            int st[4]={0,0,p,0},sp[4]={1,H,p+1,D},sx[4]={1,1,1,1};
            mlx_array r=mlx_array_new(); mlx_slice_update(&r,A,seg,st,4,sp,4,sx,4,s);
            mlx_array_eval(r); mlx_array_set(&A,r); mlx_array_free(r);
        }
    }
    { mlx_array af=mlx_array_new(); mlx_astype(&af,A,MLX_FLOAT32,s);
      mlx_array sm=mlx_array_new(); mlx_sum(&sm,af,false,s); mlx_array_eval(sm);
      const float* d=mlx_array_data_float32(sm);
      printf("  checksum sum(A)=%g\n", d?(double)d[0]:-1.0); mlx_array_free(sm); mlx_array_free(af); }
    double dt=(now_ms()-t0)/reps;
    printf("C=%-6d mode=%d reps=%d -> %.4f ms/append\n",C,mode,reps,dt);
    return 0;
}
