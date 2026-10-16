#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>
static NSString* exe_dir(void){ char b[4096]; uint32_t s=sizeof b; if(_NSGetExecutablePath(b,&s)!=0)return @"."; return [[NSString stringWithUTF8String:b] stringByDeletingLastPathComponent]; }
int main(int argc,char**argv){ @autoreleasepool {
    int N=argc>1?atoi(argv[1]):5120, K=argc>2?atoi(argv[2]):5120, reps=argc>3?atoi(argv[3]):50, BATCH=argc>4?atoi(argv[4]):20;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    NSString* src=[NSString stringWithContentsOfFile:[exe_dir() stringByAppendingPathComponent:@"gemv_gguf.metal"] encoding:NSUTF8StringEncoding error:nil];
    NSError* err=nil; id<MTLLibrary> lib=[dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if(!lib){fprintf(stderr,"compile: %s\n",err.localizedDescription.UTF8String);return 1;}
    id<MTLComputePipelineState> p0=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_q4_0"] error:&err];
    id<MTLComputePipelineState> p16=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_f16"] error:&err];
    const int NB=K>>5; size_t rowb=(size_t)NB*18;
    unsigned char* w0=(unsigned char*)malloc((size_t)N*rowb);
    _Float16* w16=(_Float16*)malloc((size_t)N*K*2);
    float* x=(float*)malloc((size_t)K*4); float* o=(float*)malloc((size_t)N*4);
    for(size_t i=0;i<(size_t)N*NB;i++){ /* blocco: d + 16 byte qs */
        _Float16 d=(_Float16)(0.01f+(rand()%50)*1e-4f);
        memcpy(w0+i*18,&d,2);
        for(int j=0;j<16;j++) w0[i*18+2+j]=(unsigned char)(rand()&0xFF);
    }
    /* ricostruisci w16 dai blocchi q4_0 per confronto identico */
    for(int n=0;n<N;n++) for(int b=0;b<NB;b++){ const unsigned char* p=w0+(size_t)n*rowb+(size_t)b*18;
        _Float16 d; memcpy(&d,p,2); const unsigned char* qs=p+2;
        for(int i=0;i<16;i++){ w16[(size_t)n*K+b*32+i]   = (_Float16)((float)d*((qs[i]&0xF)-8));
                               w16[(size_t)n*K+b*32+i+16]= (_Float16)((float)d*(((qs[i]>>4)&0xF)-8)); } }
    for(int i=0;i<K;i++) x[i]=((rand()%2000)-1000)*0.001f;
    id<MTLBuffer> bw0=[dev newBufferWithBytes:w0 length:(size_t)N*rowb options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw16=[dev newBufferWithBytes:w16 length:(size_t)N*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bx=[dev newBufferWithBytes:x length:(size_t)K*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo=[dev newBufferWithLength:(size_t)N*4 options:MTLResourceStorageModeShared];
    int pr[2]={N,K}; id<MTLBuffer> bp=[dev newBufferWithBytes:pr length:8 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q=[dev newCommandQueue];
    __block double t_q4=0,t_f16=0; __block float y0=0,y16=0;
    void(^run)(id<MTLComputePipelineState>,id<MTLBuffer>) = ^(id<MTLComputePipelineState>pso,id<MTLBuffer>bw){
        for(int r=0;r<3;r++){id<MTLCommandBuffer> cb=[q commandBuffer];{id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];[e setComputePipelineState:pso];[e setBuffer:bw offset:0 atIndex:0];[e setBuffer:bx offset:0 atIndex:1];[e setBuffer:bo offset:0 atIndex:2];[e setBuffer:bp offset:0 atIndex:3];[e setBuffer:bp offset:4 atIndex:4];[e dispatchThreadgroups:MTLSizeMake((N+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];}[cb commit];[cb waitUntilCompleted];}
        uint64_t t0=mach_absolute_time();
        for(int r=0;r<reps;r++){id<MTLCommandBuffer> cb=[q commandBuffer];for(int i=0;i<BATCH;i++){id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];[e setComputePipelineState:pso];[e setBuffer:bw offset:0 atIndex:0];[e setBuffer:bx offset:0 atIndex:1];[e setBuffer:bo offset:0 atIndex:2];[e setBuffer:bp offset:0 atIndex:3];[e setBuffer:bp offset:4 atIndex:4];[e dispatchThreadgroups:MTLSizeMake((N+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];}[cb commit];[cb waitUntilCompleted];}
        uint64_t t1=mach_absolute_time(); mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        double ms=(double)(t1-t0)*tb.numer/tb.denom/1e6/(reps*BATCH);
        if(pso==p0){t_q4=ms; y0=((float*)bo.contents)[0];} else {t_f16=ms; y16=((float*)bo.contents)[0];}
    };
    run(p0,bw0); run(p16,bw16);
    printf("N=%d K=%d | q4_0(fuso): %.4f ms (%.1f GB/s blocchi) | f16: %.4f ms (%.1f GB/s) | speedup %.2fx\n",
        N,K,t_q4,(double)N*rowb/t_q4/1e6,t_f16,(double)N*K*2/t_f16/1e6,t_f16/t_q4);
    printf("   y0(q4_0)=%.5f  y16(f16)=%.5f  diff=%.6f\n", y0, y16, fabs(y0-y16));
    } return 0; }
