/* Self-test: routing compaction (sort/group) — correttezza + stabilita'. */
#include <stdio.h>
#include <string.h>
extern int llmengine_moe_diverge_plan(const int*, int, int, int*, int*);
int main2(void);
int main(void){
    const int a[10] = {2,0,2,1,0,2,1,1,0,2}; int es[6]; int sorted[10];
    if (llmengine_moe_diverge_plan(a,10,5,es,sorted)!=0){ printf("FAIL rc\n"); return 1; }
    int exp[6]; memset(exp,0,sizeof exp); for(int i=0;i<10;i++) exp[a[i]+1]++; for(int e=0;e<5;e++) exp[e+1]+=exp[e];
    for(int e=0;e<=5;e++) if(es[e]!=exp[e]){ printf("FAIL experts e=%d got=%d exp=%d\n",e,es[e],exp[e]); return 1; }
    /* sorted: tutti gli appartenenti all'esperto e in ordine stabile */
    int occ[10]; memset(occ,0,sizeof occ);
    for(int e=0;e<5;e++) for(int k=es[e];k<es[e+1];k++){ int idx=sorted[k]; if(a[idx]!=e){ printf("FAIL group e=%d\n",e); return 1; } occ[idx]++; }
    for(int i=0;i<10;i++) if(occ[i]!=1){ printf("FAIL permutation i=%d\n",i); return 1; }
    /* stabilita': per ogni esperto, gli indici crescenti */
    for(int e=0;e<5;e++) for(int k=es[e]+1;k<es[e+1];k++) if(sorted[k]<sorted[k-1]){ printf("FAIL stable e=%d\n",e); return 1; }
    if(llmengine_moe_diverge_plan(a,10,5,es,NULL)!=0){ printf("FAIL rc2\n"); return 1; }
    if(llmengine_moe_diverge_plan((int[]){3},1,2,es,NULL)!=-1){ printf("FAIL range\n"); return 1; }
    printf("PASS llmengine_moe_diverge: compaction ok (stabile, range-check)\n"); return main2();
}

/* forward divergence-aware vs NAIVE (parity numerica). */
#include <stdlib.h>
#include <math.h>
extern int llmengine_moe_diverge_forward(const float*,int,int,int,const float*,int,const int*,const float*,int,float*);
static int test_fwd(void){
    int T=7,H=5,O=4,E=3,K=2;
    float* x=malloc(sizeof(float)*T*H); for(int i=0;i<T*H;i++) x[i]=sinf(i*0.3f);
    float* W=malloc(sizeof(float)*(size_t)E*H*O); for(int i=0;i<E*H*O;i++) W[i]=cosf(i*0.17f);
    int aex[T*K]; float aw[T*K];
    int seed=12345; for(int a=0;a<T*K;a++){ seed=(seed*1103515245+12345)&0x7fffffff; aex[a]=seed%E; aw[a]=0.25f+ (float)((seed>>8)%4)/4.0f; }
    float* out=malloc(sizeof(float)*T*O);
    if(llmengine_moe_diverge_forward(x,T,H,O,W,E,aex,aw,K,out)!=0){printf("FAIL fwd rc\n");return 1;}
    /* naive: per assegnazione accumula w * x_t @ W[e] */
    float* ref=calloc((size_t)T*O,sizeof(float));
    for(int a=0;a<T*K;a++){ int t=a/K,e=aex[a]; float w=aw[a];
        for(int o=0;o<O;o++){ float acc=0; for(int h=0;h<H;h++) acc+=x[t*H+h]*W[((size_t)e*H+h)*O+o]; ref[t*O+o]+=w*acc; } }
    float md=0; for(int i=0;i<T*O;i++){ float d=fabsf(out[i]-ref[i]); if(d>md)md=d; }
    if(md>1e-4f){ printf("FAIL fwd parity max|diff|=%.6f\n",md); return 1; }
    printf("PASS llmengine_moe_diverge_forward: divergence-aware == naive (max|diff|=%.2e)\n", md);
    return 0;
}
int main2(void){ return test_fwd(); }
