// mpp_bench_gemm.m — harness MTL4 per MPP GEMM f16.
// Uso: ./mpp_bench_gemm <metal> <fn>
// Verifica 64x64x64 vs CPU; bench M=128/1024, N=K=4096, BATCH=20 (throughput).
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
        const char* mpath = argc > 1 ? argv[1] : "mpp_gemm_f16.metal";
        const char* fname = argc > 2 ? argv[2] : "mpp_gemm_f16";
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
        fdesc.name = [NSString stringWithUTF8String:fname];
        MTL4ComputePipelineDescriptor* pdesc = [MTL4ComputePipelineDescriptor new];
        pdesc.computeFunctionDescriptor = fdesc;
        pdesc.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        if (!compiler) { fprintf(stderr, "compiler: %s\n", [[err description] UTF8String]); return 3; }
        id<MTLComputePipelineState> pipe = [compiler newComputePipelineStateWithDescriptor:pdesc compilerTaskOptions:nil error:&err];
        if (!pipe) { fprintf(stderr, "pipe: %s\n", [[err description] UTF8String]); return 4; }


        // ---------- verify 64x64x64 ----------
        {
            int M = 64, N = 64, K = 64;
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            h16* B = (h16*)malloc((size_t)K * N * sizeof(h16));
            float* C = (float*)calloc((size_t)M * N, sizeof(float));
            srand(11);
            for (int i = 0; i < M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            for (int i = 0; i < K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            int dims[3] = {M, N, K};
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB = [dev newBufferWithBytes:B length:(size_t)K * N * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC = [dev newBufferWithBytes:C length:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bD = [dev newBufferWithBytes:dims length:3 * sizeof(int) options:MTLResourceStorageModeShared];

            MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
            atd.maxBufferBindCount = 4;
            id<MTL4ArgumentTable> at = [dev newArgumentTableWithDescriptor:atd error:&err];
            if (!at) { fprintf(stderr, "argtable: %s\n", [[err description] UTF8String]); return 5; }
            [at setAddress:[bA gpuAddress] atIndex:0];
            [at setAddress:[bB gpuAddress] atIndex:1];
            [at setAddress:[bC gpuAddress] atIndex:2];
            [at setAddress:[bD gpuAddress] atIndex:3];

            id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
                        id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
            [cb beginCommandBufferWithAllocator:alloc];
            id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setArgumentTable:at];
                [enc setThreadgroupMemoryLength:2048 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(N / 32, M / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cb endCommandBuffer];
            id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
            MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
            rsd.initialCapacity = 4;
            id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
            if (rs) { [rs addAllocation:bA]; [rs addAllocation:bB]; [rs addAllocation:bC]; [rs addAllocation:bD]; [rs commit]; [rs requestResidency]; [queue addResidencySet:rs]; }
            id<MTLSharedEvent> ev = [dev newSharedEvent];
            [queue commit:&cb count:1];
            [queue signalEvent:ev value:1];
            [ev waitUntilSignaledValue:1 timeoutMS:10000];

            float* g = (float*)bC.contents;
            double md = 0;
            for (int i = 0; i < M * N; i++) {
                double r = 0; int rr = i / N, cc = i % N;
                for (int k = 0; k < K; k++) r += (double)A[rr * K + k] * (double)B[k * N + cc];
                double d = fabs(r - g[i]); if (d > md) md = d;
            }
            fprintf(stderr, "verify 64x64x64: maxabs=%.5f (g[0]=%.4f)\n", md, g[0]);
            free(A); free(B); free(C);
        }

        // ---------- bench M=128/1024, N=K=4096 ----------
        int Ms[2] = {128, 1024};
        const int N = 4096, K = 4096, BATCH = 20;
        h16* B = (h16*)malloc((size_t)K * N * sizeof(h16));
        srand(3);
        for (long i = 0; i < (long)K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
        id<MTLBuffer> bB = [dev newBufferWithBytes:B length:(size_t)K * N * sizeof(h16) options:MTLResourceStorageModeShared];
        for (int mi = 0; mi < 2; mi++) {
            int M = Ms[mi];
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            srand(5);
            for (long i = 0; i < (long)M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC = [dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            int dims[3] = {M, N, K};
            id<MTLBuffer> bD = [dev newBufferWithBytes:dims length:3 * sizeof(int) options:MTLResourceStorageModeShared];
            MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
            atd.maxBufferBindCount = 4;
            id<MTL4ArgumentTable> at = [dev newArgumentTableWithDescriptor:atd error:&err];
            [at setAddress:[bA gpuAddress] atIndex:0];
            [at setAddress:[bB gpuAddress] atIndex:1];
            [at setAddress:[bC gpuAddress] atIndex:2];
            [at setAddress:[bD gpuAddress] atIndex:3];

            id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
            id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
            MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
            rsd.initialCapacity = 4;
            id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
            if (rs) { [rs addAllocation:bA]; [rs addAllocation:bB]; [rs addAllocation:bC]; [rs addAllocation:bD]; [rs commit]; [rs requestResidency]; [queue addResidencySet:rs]; }
            id<MTLSharedEvent> ev = [dev newSharedEvent];
            for (int rep = 0; rep < 3; rep++) {   // warmup
                id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
                [cb beginCommandBufferWithAllocator:alloc];
                id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipe]; [enc setArgumentTable:at];
                [enc setThreadgroupMemoryLength:2048 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(N / 32, M / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding]; [cb endCommandBuffer];
                [queue commit:&cb count:1];
            }
            id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
            [cb beginCommandBufferWithAllocator:alloc];
            for (int b = 0; b < BATCH; b++) {
                id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipe]; [enc setArgumentTable:at];
                [enc setThreadgroupMemoryLength:2048 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(N / 32, M / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
            }
            [cb endCommandBuffer];
            double t0 = now_ms();
            [queue commit:&cb count:1];
            [queue signalEvent:ev value:1];
            [ev waitUntilSignaledValue:1 timeoutMS:120000];
            double dt = (now_ms() - t0) / BATCH;
            double tf = 2.0 * M * N * (double)K / dt / 1e9;
            printf("MPP M=%d N=%d K=%d: %.4f ms/call  %.1f TFLOP/s\n", M, N, K, dt, tf);
            free(A);
        }
        free(B);
    }
    return 0;
}
