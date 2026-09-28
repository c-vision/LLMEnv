/* oq_a8_bridge_selfcheck.m — verifica il BRIDGE C (llmmetal_oq_a8_run) vs riferimento CPU.
 * Build: clang -fobjc-arc -O2 -I native/llmmetal_oq_a8 native/llmmetal_oq_a8/llmmetal_oq_a8_bridge.m <questo> \
 *        -framework Metal -framework Foundation -o /tmp/oq_bridge
 * Uso: /tmp/oq_bridge <qmm.metallib> <quant.metallib> [M] [N] [K] */
#import <Foundation/Foundation.h>
#include "../llmmetal_oq_a8_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static uint16_t f2h(float x){uint32_t f;memcpy(&f,&x,4);uint32_t s=(f>>16)&0x8000;int32_t e=((f>>23)&0xFF)-127+15;uint32_t m=f&0x7FFFFF;if(e<=0)return(uint16_t)s;if(e>=0x1F)return(uint16_t)(s|0x7C00);return(uint16_t)(s|(e<<10)|(m>>13));}
static float h2f(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,u;if(e==0){if(m==0)u=s;else{int sh=0;while(!(m&0x400)){m<<=1;sh++;}m&=0x3FF;u=s|((uint32_t)(127-15-sh+1)<<23)|(m<<13);}}else if(e==31)u=s|0x7F800000|(m<<13);else u=s|((e-15+127)<<23)|(m<<13);float f;memcpy(&f,&u,4);return f;}
int main(int argc,char**argv){@autoreleasepool{
  if(argc<3){fprintf(stderr,"uso: %s <qmm.metallib> <quant.metallib> [M N K]\n",argv[0]);return 2;}
  int M=argc>3?atoi(argv[3]):64,N=argc>4?atoi(argv[4]):128,K=argc>5?atoi(argv[5]):128;
  const int G=K/64, WO=K/8;
  LlmMetalOqA8Ctx* ctx=llmmetal_oq_a8_create(argv[1],argv[2]);
  if(!ctx){printf("bridge create FALLITA (metallib/NAX?)\n");return 1;}
  uint16_t* x=malloc((size_t)M*K*2); srand(3); for(size_t i=0;i<(size_t)M*K;i++) x[i]=f2h((rand()%2001-1000)/500.0f);
  uint32_t* w=malloc((size_t)N*WO*4); memset(w,0,(size_t)N*WO*4);
  uint16_t* sc=malloc((size_t)G*N*2),*bi=malloc((size_t)G*N*2);
  int8_t* code=malloc((size_t)N*K);
  for(int n=0;n<N;n++)for(int k=0;k<K;k++){int c=rand()%16;code[(size_t)n*K+k]=c;w[(size_t)n*WO+k/8]|=(uint32_t)c<<(4*(k%8));}
  for(size_t i=0;i<(size_t)G*N;i++){sc[i]=f2h((rand()%1000-500)/2000.0f);bi[i]=f2h((rand()%1000-500)/2000.0f);}
  uint16_t* out=malloc((size_t)M*N*2);
  int8_t* qa=malloc((size_t)M*K); float* sa=malloc((size_t)M*4); int16_t* ra=malloc((size_t)M*G*2);
  if(llmmetal_oq_a8_stage_a(ctx,x,qa,sa,ra,M,K)!=0){printf("stage_a rc\n");return 1;}
  int8_t* qa8=malloc((size_t)M*K); for(int m=0;m<M;m++)for(int g=0;g<G;g++)for(int c=0;c<4;c++)for(int t=0;t<4;t++)for(int j=0;j<4;j++){
    int slot=16*c+4*t+j, k=16*c+8*(t>>1)+2*j+(t&1); qa8[(size_t)m*K+g*64+slot]=qa[(size_t)m*K+g*64+k]; }
  int16_t* raT=malloc((size_t)G*M*2); for(int m=0;m<M;m++)for(int g=0;g<G;g++) raT[(size_t)g*M+m]=ra[(size_t)m*G+g];
  int rc=llmmetal_oq_a8_gemm(ctx,qa8,sa,raT,w,sc,bi,out,M,N,K,4);
  if(rc!=0){printf("gemm rc=%d\n",rc);return 1;}
  /* CPU: Stage-A mode0 + ref */
  long* raL=malloc((size_t)M*G*sizeof(long));
  for(int m=0;m<M;m++){ float amax=0; for(int k=0;k<K;k++){float v=fabsf(h2f(x[(size_t)m*K+k])); if(v>amax)amax=v;}
    float s=amax>0?amax/127.0f:0.0f; sa[m]=s; float inv=amax>0?127.0f/amax:0.0f;
    for(int g=0;g<G;g++){ long sum=0; for(int kk=0;kk<64;kk++){int k=g*64+kk; int q=(int)lrintf(h2f(x[(size_t)m*K+k])*inv); if(q>127)q=127; if(q<-127)q=-127; qa[(size_t)m*K+k]=(int8_t)q; sum+=q;} raL[(size_t)m*G+g]=sum; } }
  long bad=0; float maxd=0;
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double acc=0;
    for(int g=0;g<G;g++){ long dot=0; for(int kk=0;kk<64;kk++){int k=g*64+kk; dot+=(long)qa[(size_t)m*K+k]*(long)code[(size_t)n*K+k];}
      acc += (double)h2f(sc[(size_t)g*N+n])*(double)dot + (double)h2f(bi[(size_t)g*N+n])*(double)raL[(size_t)m*G+g]; }
    float ref=(float)((double)sa[m]*acc); float gv=h2f(out[(size_t)m*N+n]); float d=fabsf(gv-ref);
    if(d>maxd)maxd=d; if(d>0.05f+0.01f*fabsf(ref))bad++; }
  printf("BRIDGE oQ A8 (M=%d N=%d K=%d): mismatch=%ld maxdiff=%.4f -> %s\n",M,N,K,bad,maxd,bad==0?"OK":"FALLITO");
  llmmetal_oq_a8_destroy(ctx); return bad==0?0:1;
}}
