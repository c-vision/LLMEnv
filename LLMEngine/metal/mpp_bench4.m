// mpp_bench4.m — harness Metal 4 (MTL4) per mpp_test.metal (MPP matmul2d f16).
// I parametri `tensor<device ..., extents<...>>` (statici) si bindano come gpuAddress di MTLBuffer
// via MTL4ArgumentTable; encoder MTL4 + commit:count:. Verifica C=A*B vs CPU.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

typedef _Float16 h16;

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / (double)tb.denom / 1e6;
}
static char* read_file(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f); return b;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* mpath = argc > 1 ? argv[1] : "mpp_test.metal";
        char* src = read_file(mpath);
        if (!src) { fprintf(stderr, "cannot read %s\n", mpath); return 1; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;

        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:opts error:&err];
        free(src);
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 2; }

        MTL4LibraryFunctionDescriptor* fdesc = [MTL4LibraryFunctionDescriptor new];
        fdesc.library = lib;
        fdesc.name = argc > 2 ? [NSString stringWithUTF8String:argv[2]] : @"mpp_f16_16x32x16";
        MTL4ComputePipelineDescriptor* pdesc = [MTL4ComputePipelineDescriptor new];
        pdesc.computeFunctionDescriptor = fdesc;
        pdesc.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        if (!compiler) { fprintf(stderr, "compiler: %s\n", [[err description] UTF8String]); return 3; }
        id<MTLComputePipelineState> pipe = [compiler newComputePipelineStateWithDescriptor:pdesc compilerTaskOptions:nil error:&err];
        if (!pipe) { fprintf(stderr, "pipe: %s\n", [[err description] UTF8String]); return 4; }

        int M = 16, N = 32, K = 16;
        h16* A = (h16*)malloc(M * K * sizeof(h16));
        h16* B = (h16*)malloc(K * N * sizeof(h16));
        float* C = (float*)calloc(M * N, sizeof(float));
        srand(7);
        for (int i = 0; i < M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
        for (int i = 0; i < K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
        id<MTLBuffer> bA = [dev newBufferWithBytes:A length:M * K * sizeof(h16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:B length:K * N * sizeof(h16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [dev newBufferWithBytes:C length:M * N * sizeof(float) options:MTLResourceStorageModeShared];

        MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
        atd.maxBufferBindCount = 3;
        id<MTL4ArgumentTable> at = [dev newArgumentTableWithDescriptor:atd error:&err];
        if (!at) { fprintf(stderr, "argtable: %s\n", [[err description] UTF8String]); return 5; }
        [at setAddress:[bA gpuAddress] atIndex:0];
        [at setAddress:[bB gpuAddress] atIndex:1];
        [at setAddress:[bC gpuAddress] atIndex:2];

        id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
        id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
        [cb beginCommandBufferWithAllocator:alloc];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setArgumentTable:at];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb endCommandBuffer];

        id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        [queue commit:&cb count:1];
        [queue signalEvent:ev value:1];
        BOOL okw = [ev waitUntilSignaledValue:1 timeoutMS:5000];
        fprintf(stderr, "LOG wait=%d\n", (int)okw); fflush(stderr);

        float* g = (float*)bC.contents;
        double md = 0;
        for (int i = 0; i < M * N; i++) {
            double r = 0; int rr = i / N, cc = i % N;
            for (int k = 0; k < K; k++) r += (double)A[rr * K + k] * (double)B[k * N + cc];
            double d = fabs(r - g[i]); if (d > md) md = d;
        }
        double r0=0; for(int k=0;k<K;k++) r0 += (double)A[0*K+k]*(double)B[k*N+0];
        double r1=0; for(int k=0;k<K;k++) r1 += (double)A[0*K+k]*(double)B[k*N+1];
        printf("MPP-MTL4 verify: maxabs=%.5f | g[0]=%.4f r0=%.4f | g[1]=%.4f r1=%.4f\n", md, g[0], r0, g[1], r1);
    }
    return 0;
}
