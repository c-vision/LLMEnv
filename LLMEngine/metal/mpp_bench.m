// mpp_bench.m — harness per mpp_test.metal (MPP matmul2d f16, 16x32x16)
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
        const char* mpath = argc > 1 ? argv[1] : "mpp_test.metal";
        char* src = read_file(mpath);
        if (!src) { fprintf(stderr, "cannot read %s\n", mpath); return 1; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:opts error:&err];
        free(src);
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 2; }
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mpp_f16_16x32x16"] error:&err];
        if (!pipe) { fprintf(stderr, "pipe: %s\n", [[err description] UTF8String]); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

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
        const NSUInteger tga = 16 * 16 * sizeof(h16), tgb = 16 * 32 * sizeof(h16);

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pipe];
        [e setBuffer:bA offset:0 atIndex:0]; [e setBuffer:bB offset:0 atIndex:1]; [e setBuffer:bC offset:0 atIndex:2];
        [e setThreadgroupMemoryLength:tga atIndex:0];
        [e setThreadgroupMemoryLength:tgb atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        float* g = (float*)bC.contents;
        double md = 0;
        for (int i = 0; i < M * N; i++) {
            double r = 0; int rr = i / N, cc = i % N;
            for (int k = 0; k < K; k++) r += (double)A[rr * K + k] * (double)B[k * N + cc];
            double d = fabs(r - g[i]); if (d > md) md = d;
        }
        printf("MPP verify 16x32x16: maxabs=%.5f (g[0]=%.4f)\n", md, g[0]);

        const int BATCH = 1000;
        double t0 = now_ms();
        id<MTLCommandBuffer> cb2 = [q commandBuffer];
        for (int b = 0; b < BATCH; b++) {
            id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
            [e2 setComputePipelineState:pipe];
            [e2 setBuffer:bA offset:0 atIndex:0]; [e2 setBuffer:bB offset:0 atIndex:1]; [e2 setBuffer:bC offset:0 atIndex:2];
            [e2 setThreadgroupMemoryLength:tga atIndex:0];
            [e2 setThreadgroupMemoryLength:tgb atIndex:1];
            [e2 dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e2 endEncoding];
        }
        [cb2 commit]; [cb2 waitUntilCompleted];
        double dt = (now_ms() - t0) / BATCH;
        printf("MPP 16x32x16: %.6f ms/call (%.1f GFLOPS)\n", dt, 2.0 * M * N * K / dt / 1e6);
    }
    return 0;
}
