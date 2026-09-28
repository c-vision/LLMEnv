/* oq_a8_mlxhelpers_selfcheck.m — verifica il GEMM int8xint8 (tensor unit M5) compilato in metallib
 * con gli helper NAX di MLX (steel/gemm/nax.h). Confronto vs riferimento CPU (A @ W^T).
 * Build metallib: xcrun metal -I <mlx-src> -std=metal4.0 -c <metal> -o x.air && xcrun metallib x.air -o x.metallib
 * Build test: clang -fobjc-arc -O2 <questo> -framework Metal -framework Foundation */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* nota: numeri errati finche' int8 non usa il frammento 16x32/K=32 (vedi .metal) */

int main(int argc, char** argv) {
  @autoreleasepool {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/oq_a8.metallib";
    const uint M = argc > 2 ? (uint)atoi(argv[2]) : 64;
    const uint N = argc > 3 ? (uint)atoi(argv[3]) : 64;
    const uint K = argc > 4 ? (uint)atoi(argv[4]) : 64;
    if (M % 16 || N % 32 || K % 16) { printf("shape non multipla di 16x32x16\n"); return 2; }
    int8_t *A = malloc((size_t)M*K), *W = malloc((size_t)N*K);
    int *C = calloc((size_t)M*N, sizeof(int));
    srand(11); for (size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)(rand()%9-4);
    for (size_t i=0;i<(size_t)N*K;i++) W[i]=(int8_t)(rand()%9-4);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err=nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:libpath] encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary> lib = src ? nil : [dev newLibraryWithFile:[NSString stringWithUTF8String:libpath] error:&err];
    id<MTLFunction> fn=[lib newFunctionWithName:@"oq_a8_gemm"];
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { printf("ERRORE pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLBuffer> bA=[dev newBufferWithBytes:A length:(size_t)M*K options:MTLResourceStorageModeShared];
    id<MTLBuffer> bW=[dev newBufferWithBytes:W length:(size_t)N*K options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC=[dev newBufferWithBytes:C length:(size_t)M*N*4 options:MTLResourceStorageModeShared];
    uint args[3]={M,N,K}; id<MTLBuffer> bP=[dev newBufferWithBytes:args length:sizeof(args) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:bA offset:0 atIndex:0]; [e setBuffer:bW offset:0 atIndex:1]; [e setBuffer:bC offset:0 atIndex:2];
    [e setBuffer:bP offset:0 atIndex:3]; [e setBuffer:bP offset:4 atIndex:4]; [e setBuffer:bP offset:8 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(N/32, M/16, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    int* got=(int*)bC.contents; long bad=0; int maxd=0;
    for (uint m=0;m<M;m++) for (uint n=0;n<N;n++){ int ref=0; for(uint k=0;k<K;k++) ref+=A[(size_t)m*K+k]*W[(size_t)n*K+k];
      int d=abs(got[(size_t)m*N+n]-ref); if(d>maxd)maxd=d; if(d)bad++; }
    printf("  got[0..3]=%d,%d,%d,%d | ref[0..3]=%d,%d,%d,%d\n", got[0],got[1],got[2],got[3], (int)({int r=0;for(uint k=0;k<K;k++)r+=A[k]*W[k];r;}), (int)({int r=0;for(uint k=0;k<K;k++)r+=A[k]*W[K+k];r;}), (int)({int r=0;for(uint k=0;k<K;k++)r+=A[k]*W[2*K+k];r;}), (int)({int r=0;for(uint k=0;k<K;k++)r+=A[k]*W[3*K+k];r;}));
    printf("GEMM int8xint8 %ux%ux%u (helper MLX/nax.h): mismatch=%ld maxdiff=%d -> %s\n", M,N,K,bad,maxd, bad==0?"OK":"FALLITO");
    return bad==0?0:1;
  }
}
