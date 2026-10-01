/* llmengine_moe_diverge_metal.mm — kernel Metal divergence-aware MoE (§4.7 SIMT).
 * Pipeline: compact (host) -> GEMM per-esperto (kernel1, per threadgroup=esperto, no atomici:
 * scrive in inter[E,T,O] col peso gate) -> reduce (kernel2: out[t,o]=sum_e inter). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include "llmengine_moe_diverge_metal.h"

static const char* KS =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
// x[T,H], W[E,H,O], aex/sorted/start, aw -> inter[E,T,O] (0 se token non assegnato all'esperto e)
"kernel void moe_div_gemm(device const float* x [[buffer(0)]], device const float* W [[buffer(1)]],\n"
"    device const int* sorted [[buffer(2)]], device const float* aw [[buffer(3)]],\n"
"    device const int* start [[buffer(4)]], device float* inter [[buffer(5)]],\n"
"    constant int* P [[buffer(6)]], uint gid [[thread_position_in_grid]]) {\n"
"    int T=P[0], H=P[1], O=P[2], K=P[3], E=P[4];\n"
"    int e=gid/O, o=gid%O; if(o>=O || e>=E) return;\n"
"    int s0=start[e], s1=start[e+1];\n"
"    for(int t=0;t<T;++t) inter[((long)e*T + t)*O + o] = 0.0f;\n"
"    for(int k=s0;k<s1;++k){ int a=sorted[k]; int ti=a/K; float w=aw[a];\n"
"        float acc=0.0f; for(int h=0;h<H;++h) acc += x[(long)ti*H+h]*W[(((long)e*H)+h)*O+o];\n"
"        inter[((long)e*T + ti)*O + o] += w*acc; }\n"
"}\n"
"kernel void moe_div_reduce(device const float* inter [[buffer(0)]], device float* out [[buffer(1)]], constant int* P [[buffer(2)]], uint tid [[thread_position_in_grid]]) {\n"
"    int T=P[0], O=P[2], E=P[4]; long n=(long)T*O; if((long)tid>=n) return; int t=tid/O, o=tid%O; float s=0.0f;\n"
"    for(int e=0;e<E;++e) s += inter[((long)e*T + t)*O + o];\n"
"    out[(long)t*O+o]=s;\n"
"}\n";

static id<MTLDevice> dev(void){ static id<MTLDevice> d=nil; if(!d) d=MTLCreateSystemDefaultDevice(); return d; }
static id<MTLComputePipelineState> pipe(id<MTLDevice> d, id<MTLLibrary> lib, const char* n){ NSError* e=nil; id<MTLFunction> f=[lib newFunctionWithName:[NSString stringWithUTF8String:n]]; if(!f) return nil; return [d newComputePipelineStateWithFunction:f error:&e]; }

int llmengine_moe_diverge_forward_metal(const float* x,int T,int H,int O,const float* W,int E,
    const int* aex,const float* aw,int K,const int* start,const int* sorted,float* out){
    @autoreleasepool {
        id<MTLDevice> d=dev(); if(!d) return -2;
        NSError* err=nil; id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:KS] options:nil error:&err];
        if(!lib) return -3;
        id<MTLComputePipelineState> pg=pipe(d,lib,"moe_div_gemm"), pr=pipe(d,lib,"moe_div_reduce");
        if(!pg||!pr) return -3;
        long sx=(long)T*H, sw=(long)E*H*O, si=(long)E*T*O, so=(long)T*O;
        id<MTLBuffer> bx=[d newBufferWithBytes:x length:sx*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bW=[d newBufferWithBytes:W length:sw*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bs=[d newBufferWithBytes:sorted length:(long)T*K*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bA=[d newBufferWithBytes:aw length:(long)T*K*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> be=[d newBufferWithBytes:start length:(long)(E+1)*4 options:MTLResourceStorageModeShared];
        int P[5]={T,H,O,K,E};
        id<MTLBuffer> bP=[d newBufferWithBytes:P length:sizeof P options:MTLResourceStorageModeShared];
        id<MTLBuffer> bI=[d newBufferWithLength:si*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO=[d newBufferWithLength:so*4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q=[d newCommandQueue];
        id<MTLCommandBuffer> cb=[q commandBuffer];
        { id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:pg];
          [en setBuffer:bx offset:0 atIndex:0]; [en setBuffer:bW offset:0 atIndex:1]; [en setBuffer:bs offset:0 atIndex:2];
          [en setBuffer:bA offset:0 atIndex:3]; [en setBuffer:be offset:0 atIndex:4]; [en setBuffer:bI offset:0 atIndex:5]; [en setBuffer:bP offset:0 atIndex:6];
          [en dispatchThreads:MTLSizeMake((NSUInteger)E*O,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)((E*O)<64?(E*O):64),1,1)]; [en endEncoding]; }
        { id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:pr];
          [en setBuffer:bI offset:0 atIndex:0]; [en setBuffer:bO offset:0 atIndex:1]; [en setBuffer:bP offset:0 atIndex:2];
          [en dispatchThreads:MTLSizeMake((NSUInteger)so,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)(so<64?so:64),1,1)]; [en endEncoding]; }
        [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bO.contents, (size_t)so*4);
    }
    return 0;
}
