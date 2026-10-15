// gemb_bench — harness per kernel Metal GEMV quantizzata propri (m=1, decode).
// Confronto equo col vecchio path MLX: misurare SEMPRE batched/throughput (il decode reale
// accoda i dispatch e NON sincronizza per-op; il commit+waitUntilCompleted per-op ~0.3 ms
// oscura il compute).
//
//   build: bash LLMEngine/metal/build_bench.sh
//   run  : build/gemb_bench [kernel] [N] [K] [reps] [BATCH]
//          kernel: qgemv4s (default) | qgemv4 | qgemv4v | bcopy
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>

static NSString* exe_dir(void) {
    char buf[4096]; uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) != 0) return @".";
    return [[NSString stringWithUTF8String:buf] stringByDeletingLastPathComponent];
}

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* kname = argc > 1 ? argv[1] : "qgemv4s";
        int N = argc > 2 ? atoi(argv[2]) : 4096;
        int K = argc > 3 ? atoi(argv[3]) : 4096;
        int reps = argc > 4 ? atoi(argv[4]) : 30;
        int BATCH = argc > 5 ? atoi(argv[5]) : 50;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSString* metalPath = [exe_dir() stringByAppendingPathComponent:@"gemb_kernel.metal"];
        NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read %s\n", metalPath.UTF8String); return 1; }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:kname]] error:&err];
        if (!pso) { fprintf(stderr, "pso(%s): %s\n", kname, err.localizedDescription.UTF8String); return 1; }

        size_t Kw = (size_t)K / 8, G = (size_t)K / 64;
        uint32_t* w  = malloc((size_t)N*Kw*4);
        float*    sc = malloc((size_t)N*G*4);
        float*    bs = malloc((size_t)N*G*4);
        float*    x  = malloc((size_t)K*4);
        float*    o  = malloc((size_t)N*4);
        for (size_t i = 0; i < (size_t)N*Kw; i++) { uint32_t a = (uint32_t)rand(), b = (uint32_t)rand(); w[i] = (a & 0x0F0F0F0Fu) | ((b & 0x0F0F0F0Fu) << 4); }
        for (size_t i = 0; i < (size_t)N*G; i++) { sc[i] = 0.01f; bs[i] = 0.001f; }
        for (int i = 0; i < K; i++) x[i] = ((rand()%2000) - 1000) * 0.001f;

        int use_bcopy = (strcmp(kname, "bcopy") == 0);
        size_t BF4 = (size_t)N*K/4;                        // float4 di una copia da N*K*4 byte
        void* sb = use_bcopy ? malloc(BF4*16) : NULL;
        void* db = use_bcopy ? malloc(BF4*16) : NULL;
        id<MTLBuffer> bcs = use_bcopy ? [dev newBufferWithBytes:sb length:BF4*16 options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> bcd = use_bcopy ? [dev newBufferWithLength:BF4*16 options:MTLResourceStorageModeShared] : nil;

        id<MTLBuffer> bw  = [dev newBufferWithBytes:w  length:(size_t)N*Kw*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsc = [dev newBufferWithBytes:sc length:(size_t)N*G*4  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bbs = [dev newBufferWithBytes:bs length:(size_t)N*G*4  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx  = [dev newBufferWithBytes:x  length:(size_t)K*4    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo  = [dev newBufferWithLength:(size_t)N*4 options:MTLResourceStorageModeShared];
        int pr[2] = {N, use_bcopy ? (int)BF4 : K};
        id<MTLBuffer> bp = [dev newBufferWithBytes:pr length:8 options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> q = [dev newCommandQueue];
        /* grid: bcopy = over float4; qgemv4s = 8 righe/TG; qgemv4/qgemv4v = 1 riga/TG (grid N!). */
        MTLSize grid;
        if (use_bcopy) grid = MTLSizeMake((BF4 + 255)/256, 1, 1);
        else if (strcmp(kname, "qgemv4s") == 0) grid = MTLSizeMake((N + 7)/8, 1, 1);
        else grid = MTLSizeMake(N, 1, 1);
        MTLSize threads = MTLSizeMake(256, 1, 1);

        void(^one)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            if (use_bcopy) { [e setBuffer:bcs offset:0 atIndex:0]; [e setBuffer:bcd offset:0 atIndex:1]; [e setBuffer:bp offset:4 atIndex:2]; }
            else { [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1]; [e setBuffer:bbs offset:0 atIndex:2];
                   [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bo offset:0 atIndex:4];
                   [e setBuffer:bp offset:0 atIndex:5]; [e setBuffer:bp offset:4 atIndex:6]; }
            [e dispatchThreadgroups:grid threadsPerThreadgroup:threads];
            [e endEncoding];
        };

        for (int r = 0; r < 3; r++) { id<MTLCommandBuffer> cb = [q commandBuffer]; for (int i = 0; i < BATCH; i++) one(cb); [cb commit]; [cb waitUntilCompleted]; }
        uint64_t t0 = mach_absolute_time();
        for (int r = 0; r < reps; r++) { id<MTLCommandBuffer> cb = [q commandBuffer]; for (int i = 0; i < BATCH; i++) one(cb); [cb commit]; [cb waitUntilCompleted]; }
        uint64_t t1 = mach_absolute_time();
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        double ms = (double)(t1-t0)*tb.numer/tb.denom/1e6/(reps*BATCH);

        double bytes = use_bcopy ? (double)N*K*8 : (double)N*K*0.5;   // pesi (4-bit) letti
        float* p = bo.contents;
        printf("NEW %-8s N=%-5d K=%-5d BATCH=%-3d : %.4f ms/call  ~%.0f GB/s  out[0..2]= %.4f %.4f %.4f\n",
               kname, N, K, BATCH, ms, bytes/ms/1e6, p[0], p[1], p[2]);
    }
    return 0;
}
