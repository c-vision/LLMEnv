#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern int llmengine_moe_diverge_plan(const int*, int, int, int*, int*);
extern int llmengine_moe_diverge_forward(const float*,int,int,int,const float*,int,const int*,const float*,int,float*);
extern int llmengine_moe_diverge_forward_metal(const float*,int,int,int,const float*,int,const int*,const float*,int,const int*,const int*,float*);
int main(void){
    int T=7,H=5,O=4,E=3,K=2; int n=T*K;
    float* x=malloc(sizeof(float)*T*H); for(int i=0;i<T*H;i++) x[i]=sinf(i*0.3f);
    float* W=malloc(sizeof(float)*(size_t)E*H*O); for(int i=0;i<E*H*O;i++) W[i]=cosf(i*0.17f);
    int* aex=malloc(sizeof(int)*n); float* aw=malloc(sizeof(float)*n); int s=12345;
    for(int a=0;a<n;a++){ s=(s*1103515245+12345)&0x7fffffff; aex[a]=s%E; aw[a]=0.25f+(float)((s>>8)%4)/4.0f; }
    int* start=malloc(sizeof(int)*(E+1)); int* sorted=malloc(sizeof(int)*n);
    if(llmengine_moe_diverge_plan(aex,n,E,start,sorted)!=0){printf("FAIL plan\n");return 1;}
    float* ref=calloc((size_t)T*O,sizeof(float)); float* out=calloc((size_t)T*O,sizeof(float));
    llmengine_moe_diverge_forward(x,T,H,O,W,E,aex,aw,K,ref);
    int rc=llmengine_moe_diverge_forward_metal(x,T,H,O,W,E,aex,aw,K,start,sorted,out);
    if(rc<0){ printf("SKIP metal (rc=%d, no device?)\n",rc); return 0; }
    float md=0; for(int i=0;i<T*O;i++){ float df=fabsf(out[i]-ref[i]); if(df>md)md=df; }
    if(md>1e-4f){ printf("FAIL metal vs cpu max|diff|=%.6f\n",md); return 1; }
    printf("PASS llmengine_moe_diverge_forward_metal: GPU == CPU (max|diff|=%.2e)\n", md);
    return 0;
}
