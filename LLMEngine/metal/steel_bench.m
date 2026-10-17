// steel_bench.m — harness standalone per steel_gemm.metal (f16)
// Uso: ./steel_bench steel_gemm.metal
// Verifica 64x64x32 vs CPU f32; bench M=128/1024, N=K=4096, BATCH=50 (throughput, non per-call).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
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
        const char* mpath = argc > 1 ? argv[1] : "steel_gemm.metal";
        char* src = read_file(mpath);
        if (!src) { fprintf(stderr, "cannot read %s\n", mpath); return 1; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:opts error:&err];
        free(src);
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 2; }
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"steel_f16_gemm"] error:&err];
        if (!pipe) { fprintf(stderr, "pipe: %s\n", [[err description] UTF8String]); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const NSUInteger shmem = (64 * 32 + 32 * 64) * 2 * sizeof(h16);

        // --- verify 64x64x32 ---
        {
            int M = 64, N = 64, K = 32;
            h16* A = (h16*)malloc(M * K * sizeof(h16));
            h16* B = (h16*)malloc(K * N * sizeof(h16));
            float* C = (float*)malloc(M * N * sizeof(float));
            srand(1);
            for (int i = 0; i < M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            for (int i = 0; i < K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB = [dev newBufferWithBytes:B length:K * N * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC = [dev newBufferWithLength:M * N * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pipe];
            [e setBuffer:bA offset:0 atIndex:0]; [e setBuffer:bB offset:0 atIndex:1]; [e setBuffer:bC offset:0 atIndex:2];
            int dM = M, dN = N, dK = K;
            [e setBytes:&dM length:4 atIndex:3]; [e setBytes:&dN length:4 atIndex:4]; [e setBytes:&dK length:4 atIndex:5];
            [e setThreadgroupMemoryLength:shmem atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(N / 64, M / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            float* g = (float*)bC.contents;
            double md = 0, mref = 0;
            for (int i = 0; i < M * N; i++) {
                double r = 0; int rr = i / N, cc = i % N;
                for (int k = 0; k < K; k++) r += (double)A[rr * K + k] * (double)B[k * N + cc];
                if (i < 4) printf("  i=%d g=%.4f r=%.4f\n", i, g[i], r);
                double d = fabs(r - g[i]); if (d > md) md = d;
                if (fabs(r) > mref) mref = fabs(r);
            }
            printf("verify 64x64x32: maxabs=%.5f (maxref=%.3f)\n", md, mref);
            free(A); free(B); free(C);
        }

        // --- bench M=128/1024, N=K=4096, BATCH=50 ---
        int Ms[2] = {128, 1024};
        const int N = 4096, K = 4096, BATCH = 50;
        h16* B = (h16*)malloc((size_t)K * N * sizeof(h16));
        srand(2); for (long i = 0; i < (long)K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
        id<MTLBuffer> bB = [dev newBufferWithBytes:B length:(size_t)K * N * sizeof(h16) options:MTLResourceStorageModeShared];
        free(B);
        for (int mi = 0; mi < 2; mi++) {
            int M = Ms[mi];
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            srand(3); for (long i = 0; i < (long)M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            free(A);
            id<MTLBuffer> bC = [dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            for (int w = 0; w < 3; w++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pipe];
                [e setBuffer:bA offset:0 atIndex:0]; [e setBuffer:bB offset:0 atIndex:1]; [e setBuffer:bC offset:0 atIndex:2];
                int dM = M, dN = N, dK = K;
                [e setBytes:&dM length:4 atIndex:3]; [e setBytes:&dN length:4 atIndex:4]; [e setBytes:&dK length:4 atIndex:5];
                [e setThreadgroupMemoryLength:shmem atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(N / 64, M / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int b = 0; b < BATCH; b++) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pipe];
                [e setBuffer:bA offset:0 atIndex:0]; [e setBuffer:bB offset:0 atIndex:1]; [e setBuffer:bC offset:0 atIndex:2];
                int dM = M, dN = N, dK = K;
                [e setBytes:&dM length:4 atIndex:3]; [e setBytes:&dN length:4 atIndex:4]; [e setBytes:&dK length:4 atIndex:5];
                [e setThreadgroupMemoryLength:shmem atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(N / 64, M / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e endEncoding];
            }
            double t0 = now_ms();
            [cb commit]; [cb waitUntilCompleted];
            double dt = (now_ms() - t0) / BATCH;
            double tf = 2.0 * M * N * (double)K / dt / 1e9;
            printf("M=%d N=%d K=%d: %.4f ms/call  %.1f TFLOP/s\n", M, N, K, dt, tf);
        }
    }
    return 0;
}
