/* oq_a8_omlx_selfcheck.m — self-check dei kernel di oMLX (qwen35_oq_a8) portati verbatim:
 * Stage-A (oq_a8_quantize, ACT_MODE 0) + GEMM int8-act x Q4 (oq_a8_qmm_t_nax_v8).
 * CPU reference della LORO formula:
 *   y[m,n] = sa[m] * sum_g ( wscale[g,n] * sum_{k in g} qa[m,k]*code4[n,k] + wbias[g,n]*ra[m,g] )
 * Build: clang -fobjc-arc -O2 <questo> -framework Metal -framework Foundation
 * Metallib: xcrun metal -I <mlx-src> -I csrc -std=metal4.0 -c csrc/oq_a8_qmm_nax.metal -o x.air && metallib
 *           idem csrc/oq_a8_quantize.metal */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float h2f(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,u;
  if(e==0){if(m==0)u=s;else{int sh=0;while(!(m&0x400)){m<<=1;sh++;}m&=0x3FF;u=s|((uint32_t)(127-15-sh+1)<<23)|(m<<13);}}
  else if(e==31)u=s|0x7F800000|(m<<13);else u=s|((e-15+127)<<23)|(m<<13);float f;memcpy(&f,&u,4);return f;}
static uint16_t f2h(float x){uint32_t f;memcpy(&f,&x,4);uint32_t s=(f>>16)&0x8000;int32_t e=((f>>23)&0xFF)-127+15;uint32_t m=f&0x7FFFFF;
  if(e<=0)return(uint16_t)s;if(e>=0x1F)return(uint16_t)(s|0x7C00);return(uint16_t)(s|(e<<10)|(m>>13));}

int main(int argc, char** argv) {
  @autoreleasepool {
    const char* qmm_lib = argc>1?argv[1]:"/tmp/oq_a8_qmm_nax.metallib";
    const char* q_lib   = argc>2?argv[2]:"/tmp/oq_a8_quantize.metallib";
    int APPLY_SA=argc>6?atoi(argv[6]):1; int RA_T=argc>7?atoi(argv[7]):1;
    int ONE_HOT=argc>8?atoi(argv[8]):0;
    uint M=argc>3?(uint)atoi(argv[3]):64, N=argc>4?(uint)atoi(argv[4]):128, K=argc>5?(uint)atoi(argv[5]):128;
    const uint G=K/64, WO=K/8;                    /* groups, words per row (Q4: 64/8) */
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* err=nil;
    id<MTLLibrary> lq=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:q_lib]] error:&err];
    id<MTLLibrary> lgemm=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:qmm_lib]] error:&err];
    if(!lq||!lgemm){printf("ERR lib: %s\n",err.localizedDescription.UTF8String);return 1;}
    id<MTLFunction> fq=[lq newFunctionWithName:@"oq_a8_quantize_float16_t_am0"];
    id<MTLFunction> fg=[lgemm newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_float16_t_wm_2_wn_2"];
    id<MTLComputePipelineState> pq=[dev newComputePipelineStateWithFunction:fq error:&err];
    id<MTLComputePipelineState> pg=[dev newComputePipelineStateWithFunction:fg error:&err];
    if(!pq||!pg){printf("ERR pso: %s\n",err.localizedDescription.UTF8String);return 1;}
    id<MTLCommandQueue> q=[dev newCommandQueue];
    /* dati */
    uint16_t* x=malloc((size_t)M*K*2); srand(3); for(size_t i=0;i<(size_t)M*K;i++) x[i]=ONE_HOT?f2h(0.0f):f2h((rand()%2001-1000)/500.0f);
    uint32_t* w4=malloc((size_t)N*WO*4); for(size_t i=0;i<(size_t)N*WO;i++) w4[i]=0;
    uint16_t* ws=malloc((size_t)G*N*2), *wb=malloc((size_t)G*N*2);
    uint8_t* code4=malloc((size_t)N*K);
    const uint one_m=3, one_n=7, one_k=19;
    if(ONE_HOT==1) x[(size_t)one_m*K+one_k]=f2h(1.0f);
    if(ONE_HOT==2) for(uint m=0;m<M;m++) x[(size_t)m*K+one_k]=f2h(1.0f);
    for(uint n=0;n<N;n++) for(uint k=0;k<K;k++){ uint8_t c=ONE_HOT?(uint8_t)(k==one_k&&(ONE_HOT==2||n==one_n)):(uint8_t)(rand()%16); code4[(size_t)n*K+k]=c;
      uint32_t word=(size_t)n*WO + k/8; w4[word] |= (uint32_t)c << (4*(k%8)); }
    for(size_t i=0;i<(size_t)G*N;i++){ ws[i]=ONE_HOT?f2h(1.0f):f2h((rand()%1000-500)/2000.0f); wb[i]=ONE_HOT?f2h(0.0f):f2h((rand()%1000-500)/2000.0f); }
    int8_t* qa=malloc((size_t)M*K); float* sa=malloc((size_t)M*4); int16_t* ra=malloc((size_t)M*G*2);
    uint16_t* out=malloc((size_t)M*N*2);
    id<MTLBuffer> bx=[dev newBufferWithBytes:x length:(size_t)M*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bqa=[dev newBufferWithBytes:qa length:(size_t)M*K options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsa=[dev newBufferWithBytes:sa length:(size_t)M*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bra=[dev newBufferWithBytes:ra length:(size_t)M*G*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bra_transposed=nil; char bra_index_dummy[1]; id<MTLBuffer> qa_v8=nil;
    id<MTLBuffer> bw=[dev newBufferWithBytes:w4 length:(size_t)N*WO*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bws=[dev newBufferWithBytes:ws length:(size_t)G*N*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bwb=[dev newBufferWithBytes:wb length:(size_t)G*N*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout=[dev newBufferWithBytes:out length:(size_t)M*N*2 options:MTLResourceStorageModeShared];
    int argsK=K,argsN=N,argsM=M; id<MTLBuffer> bpa=[dev newBufferWithBytes:&argsK length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bpn=[dev newBufferWithBytes:&argsN length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bpm=[dev newBufferWithBytes:&argsM length:4 options:MTLResourceStorageModeShared];
    /* Stage A: 1 threadgroup/row, 32*8 threads */
    {id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
     [e setComputePipelineState:pq]; [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bqa offset:0 atIndex:1];
     [e setBuffer:bsa offset:0 atIndex:2]; [e setBuffer:bra offset:0 atIndex:3]; [e setBuffer:bpa offset:0 atIndex:4];
     [e dispatchThreadgroups:MTLSizeMake(M,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];}
    /* Stage-A v8: permuta qa dentro ogni GS64: slot 16c+4t+j <- k=16c+8*(t>>1)+2j+(t&1) */
    { const int8_t* qp=(const int8_t*)bqa.contents; int8_t* qv=malloc((size_t)M*K);
      for(uint m=0;m<M;m++) for(uint g=0;g<G;g++) for(int c=0;c<4;c++) for(int t=0;t<4;t++) for(int j=0;j<4;j++){
        int slot=16*c+4*t+j; int k=16*c+8*(t>>1)+2*j+(t&1);
        qv[(size_t)m*K+g*64+slot]=qp[(size_t)m*K+g*64+k]; }
      id<MTLBuffer> bqv=[dev newBufferWithBytes:qv length:(size_t)M*K options:MTLResourceStorageModeShared];
      qa_v8 = bqv; }
    /* Ra: Stage-A produce [M, groups]; il GEMM vuole [groups, M] (group-major) -> traspongo */
    { const int16_t* raA=(const int16_t*)bra.contents; int16_t* raT=malloc((size_t)G*M*2);
      for(uint m=0;m<M;m++) for(uint g=0;g<G;g++) raT[(size_t)g*M+m]=raA[(size_t)m*G+g];
      id<MTLBuffer> brt=[dev newBufferWithBytes:raT length:(size_t)G*M*2 options:MTLResourceStorageModeShared];
      memcpy((void*)bra_index_dummy,0,0); bra_transposed = brt; } if(!RA_T) bra_transposed = bra;
    /* GEMM variant 800: BM=2*16*2=64, BN=32*2=64; threads = WM*WN*32=128 */
    {id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
     [e setComputePipelineState:pg]; [e setBuffer:qa_v8 offset:0 atIndex:0]; [e setBuffer:bsa offset:0 atIndex:1];
     [e setBuffer:bra_transposed offset:0 atIndex:2]; [e setBuffer:bw offset:0 atIndex:3]; [e setBuffer:bws offset:0 atIndex:4];
     [e setBuffer:bwb offset:0 atIndex:5]; [e setBuffer:bout offset:0 atIndex:6];
     [e setBuffer:bpa offset:0 atIndex:7]; [e setBuffer:bpn offset:0 atIndex:8]; [e setBuffer:bpm offset:0 atIndex:9];
     [e dispatchThreadgroups:MTLSizeMake(N/64,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(32,2,2)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];}
    /* validazione packing: decodifica i pesi col LORO kernel di riferimento */
    { id<MTLFunction> fd=[lq newFunctionWithName:@"oq_a8_decode_weights_q4"];
      id<MTLComputePipelineState> pd=[dev newComputePipelineStateWithFunction:fd error:&err];
      id<MTLBuffer> bdec=[dev newBufferWithLength:(size_t)N*K options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:pd]; [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bdec offset:0 atIndex:1]; [e setBuffer:bpa offset:0 atIndex:2];
      [e dispatchThreads:MTLSizeMake(G, N, 1) threadsPerThreadgroup:MTLSizeMake(64,4,1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      const int8_t* dec=(const int8_t*)bdec.contents; long badW=0;
      for(size_t i=0;i<(size_t)N*K;i++) if(dec[i]!=code4[i]) badW++;
      printf("  packing check (kernel vs CPU): mismatch=%ld\n", badW); }
    /* CPU ref */
    const int8_t* gqa=(const int8_t*)bqa.contents; const float* gsa=(const float*)bsa.contents; const int16_t* gra=(const int16_t*)bra.contents;
    /* verifica Stage-A */
    long badA=0; for(size_t i=0;i<(size_t)M*K;i++){ float xv=[[NSNumber numberWithInt:0] intValue]; (void)xv;
      float xf=h2f(x[i]); float s=gsa[i/K]; int8_t q=(int8_t)lrintf(xf*(s>0? (127.0f/(s*127.0f)) : 0.0f)); if(s<=0)q=0;
      if(gqa[i]!=q) badA++; }
    const uint16_t* got=(const uint16_t*)bout.contents; long bad=0; float maxd=0;
    if(ONE_HOT==1){
      printf("  one-hot source=(m=%u,n=%u,k=%u): qa=%d sa=%.8f ra=%d\n",
             one_m,one_n,one_k,(int)gqa[(size_t)one_m*K+one_k],gsa[one_m],(int)gra[(size_t)one_m*G+one_k/64]);
      for(uint m=0;m<M;m++) for(uint n=0;n<N;n++){
        float gv=h2f(got[(size_t)m*N+n]);
        if(fabsf(gv)>0.0001f) printf("  one-hot out[m=%u,n=%u]=%.8f\n",m,n,gv);
      }
    }
    if(ONE_HOT==2){
      for(uint m=0;m<M;m++){
        uint nonzero=0; int first=-1, last=-1;
        for(uint n=0;n<N;n++) if(fabsf(h2f(got[(size_t)m*N+n]))>0.0001f){ nonzero++; if(first<0)first=(int)n; last=(int)n; }
        printf("  broadcast row=%u nonzero=%u first=%d last=%d\n",m,nonzero,first,last);
      }
    }
    { double acc0=0; for(uint g=0;g<G;g++){ long dot=0; for(uint kk=0;kk<64;kk++){uint k=g*64+kk; dot+=(long)gqa[k]*(long)code4[k];}
        acc0 += (double)h2f(ws[(size_t)g*N+0])*(double)dot + (double)h2f(wb[(size_t)g*N+0])*(double)gra[(size_t)0*G+g]; }
      printf("  [m=0,n=0] got=%.5f | sa[0]=%.6f | raw=%.5f | sa*raw=%.5f\n", h2f(got[0]), gsa[0], acc0, gsa[0]*acc0); }
    for(uint m=0;m<M;m++) for(uint n=0;n<N;n++){ double acc=0;
      for(uint g=0;g<G;g++){ long dot=0; for(uint kk=0;kk<64;kk++){ uint k=g*64+kk; dot += (long)gqa[(size_t)m*K+k]*(long)code4[(size_t)n*K+k]; }
        acc += (double)h2f(ws[(size_t)g*N+n])*(double)dot + (double)h2f(wb[(size_t)g*N+n])*(double)gra[(size_t)m*G+g]; }
      float ref=(float)((double)gsa[m]*acc); float gv=h2f(got[(size_t)m*N+n]); if(!APPLY_SA) gv=gsa[m]*gv; float d=fabsf(gv-ref);
      if(d>maxd)maxd=d; if(d>0.05f+0.01f*fabsf(ref)) bad++; }
    printf("oMLX qmm_t_nax_v8 (M=%u N=%u K=%u): Stage-A mismatch=%ld | GEMM mismatch=%ld maxdiff=%.4f -> %s\n",
           M,N,K,badA,bad,maxd, bad==0&&badA==0?"OK":"FALLITO");
    return (bad==0&&badA==0)?0:1;
  }
}
