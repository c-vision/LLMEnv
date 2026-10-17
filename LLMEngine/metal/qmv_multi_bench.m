// qmv_multi_bench — bench isolato del kernel qmv8_multi (QMV multi-riga, 8-bit affine g64).
// Scopo: misurare la banda a M piccolo (verify DFlash2 M=8) vs M=1, e la correttezza.
//   build: clang -O2 -fobjc-arc -framework Foundation -framework Metal -o qmv_multi_bench qmv_multi_bench.m
//   run  : qmv_multi_bench [N] [K] [M] [reps] [batch]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>

static NSString* exe_dir(void) {
    char buf[4096]; uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) != 0) return @".";
    return [[NSString stringWithUTF8String:buf] stringByDeletingLastPathComponent];
}

int main(int argc, char** argv) { @autoreleasepool {
    int N = argc > 1 ? atoi(argv[1]) : 5120;
    int K = argc > 2 ? atoi(argv[2]) : 5120;
    int M = argc > 3 ? atoi(argv[3]) : 8;
    int reps = argc > 4 ? atoi(argv[4]) : 20;
    int BATCH = argc > 5 ? atoi(argv[5]) : 20;
    if (M < 1 || M > 8) { fprintf(stderr, "M in [1,8]\n"); return 2; }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSString* metalPath = [exe_dir() stringByAppendingPathComponent:@"qmv_multi.metal"];
    NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "cannot read %s\n", metalPath.UTF8String); return 1; }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"qmv8_multi"] error:&err];
    if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }

    size_t Kw = (size_t)K / 4, G = (size_t)K / 64;
    uint32_t* w = malloc((size_t)N * Kw * 4);
    _Float16* sc = malloc((size_t)N * G * 2), *bs = malloc((size_t)N * G * 2);
    _Float16* x = malloc((size_t)M * K * 2), *y = malloc((size_t)M * N * 2);
    for (size_t i = 0; i < (size_t)N * Kw; i++) w[i] = (uint32_t)rand();
    for (size_t i = 0; i < (size_t)N * G; i++) { sc[i] = (_Float16)0.01f; bs[i] = (_Float16)0.001f; }
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (_Float16)(((rand() % 2000) - 1000) * 0.001f);

    id<MTLBuffer> bw = [dev newBufferWithBytes:w length:(size_t)N * Kw * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsc = [dev newBufferWithBytes:sc length:(size_t)N * G * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bbs = [dev newBufferWithBytes:bs length:(size_t)N * G * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bx = [dev newBufferWithBytes:x length:(size_t)M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [dev newBufferWithLength:(size_t)M * N * 2 options:MTLResourceStorageModeShared];
    int pr[3] = {N, K, M};
    id<MTLBuffer> bp = [dev newBufferWithBytes:pr length:12 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    void(^one)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1]; [e setBuffer:bbs offset:0 atIndex:2];
        [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:by offset:0 atIndex:4];
        [e setBuffer:bp offset:0 atIndex:5]; [e setBuffer:bp offset:4 atIndex:6]; [e setBuffer:bp offset:8 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((N + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
    };
    for (int r = 0; r < 3; r++) { id<MTLCommandBuffer> cb = [q commandBuffer]; for (int i = 0; i < BATCH; i++) one(cb); [cb commit]; [cb waitUntilCompleted]; }
    uint64_t t0 = mach_absolute_time();
    for (int r = 0; r < reps; r++) { id<MTLCommandBuffer> cb = [q commandBuffer]; for (int i = 0; i < BATCH; i++) one(cb); [cb commit]; [cb waitUntilCompleted]; }
    uint64_t t1 = mach_absolute_time();
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    double ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6 / (reps * BATCH);
    double wbytes = (double)N * K;   // 8-bit: 1 byte/peso
    _Float16* yp = by.contents;
    printf("qmv8_multi N=%-6d K=%-6d M=%-2d : %.4f ms  %.1f GB/s(W)  y[0][0]=%.5f y[%d][0]=%.5f\n",
           N, K, M, ms, wbytes / ms / 1e6, yp[0], M - 1, yp[(size_t)(M - 1) * N]);
    // check CPU per y[0][0] e y[M-1][0] (colonna n=0)
    for (int mm = 0; mm < M; mm += (M - 1 > 0 ? M - 1 : 1)) {
        double ref = 0;
        for (int k = 0; k < K; k++) {
            uint pk = w[(k >> 2)];
            double wv = (double)((pk >> (8 * (k & 3))) & 0xFF) * 0.01 + 0.001;
            ref += wv * (double)x[(size_t)mm * K + k];
        }
        printf("   chk y[%d][0]=%.6f ref=%.6f diff=%.6f\n", mm, (double)yp[(size_t)mm * N], ref, fabs((double)yp[(size_t)mm * N] - ref));
    }
    } return 0; }
