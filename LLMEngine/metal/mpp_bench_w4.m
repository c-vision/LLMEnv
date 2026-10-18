// mpp_bench_w4.m — harness per mpp_gemm_w4.metal (attivazioni f16 x pesi 4-bit).
// Uso: ./mpp_bench_w4 <metal> <fn>
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

// dequant CPU coerente col kernel: valore = (nibble - 8) * scala
static float dq(const unsigned char* Bn, const h16* Bs, int K, int N, int k, int n) {
    int nb = N / 32, blk = n / 32, inb = n % 32;
    const unsigned char* p = Bn + ((size_t)k * nb + blk) * 16;
    int byte = p[inb / 2];
    int nib = (inb & 1) ? (byte >> 4) : (byte & 0x0F);
    return (float)(nib - 8) * (float)Bs[(size_t)k * nb + blk];
}

static int run(id<MTLDevice> dev, id<MTL4CommandQueue> queue, id<MTL4CommandAllocator> alloc,
               id<MTLComputePipelineState> pipe,
               id<MTLBuffer> bA, id<MTLBuffer> bBn, id<MTLBuffer> bBs, id<MTLBuffer> bC, id<MTLBuffer> bD,
               int M, int N, int K, int batch) {
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 5;
    NSError* err = nil;
    id<MTL4ArgumentTable> at = [dev newArgumentTableWithDescriptor:atd error:&err];
    if (!at) { fprintf(stderr, "argtable: %s\n", [[err description] UTF8String]); return 1; }
    [at setAddress:[bA gpuAddress] atIndex:0];
    [at setAddress:[bBn gpuAddress] atIndex:1];
    [at setAddress:[bBs gpuAddress] atIndex:2];
    [at setAddress:[bC gpuAddress] atIndex:3];
    [at setAddress:[bD gpuAddress] atIndex:4];

    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    rsd.initialCapacity = 5;
    id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
    if (rs) { [rs addAllocation:bA]; [rs addAllocation:bBn]; [rs addAllocation:bBs]; [rs addAllocation:bC]; [rs addAllocation:bD]; [rs commit]; [rs requestResidency]; [queue addResidencySet:rs]; }

    id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:alloc];
    for (int b = 0; b < batch; b++) {
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setArgumentTable:at];
        [enc setThreadgroupMemoryLength:(32 * 16 + 16 * 32) * 2 * sizeof(h16) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(N / 32, M / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
    }
    [cb endCommandBuffer];
    id<MTLSharedEvent> ev = [dev newSharedEvent];
    [queue commit:&cb count:1];
    [queue signalEvent:ev value:1];
    [ev waitUntilSignaledValue:1 timeoutMS:120000];
    return 0;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* mpath = argc > 1 ? argv[1] : "mpp_gemm_w4.metal";
        const char* fname = argc > 2 ? argv[2] : "mpp_gemm_w4";
        char* src = read_file(mpath);
        if (!src) { fprintf(stderr, "cannot read %s\n", mpath); return 1; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:[MTLCompileOptions new] error:&err];
        free(src);
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 2; }

        MTL4LibraryFunctionDescriptor* fdesc = [MTL4LibraryFunctionDescriptor new];
        fdesc.library = lib; fdesc.name = [NSString stringWithUTF8String:fname];
        MTL4ComputePipelineDescriptor* pdesc = [MTL4ComputePipelineDescriptor new];
        pdesc.computeFunctionDescriptor = fdesc;
        pdesc.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        id<MTLComputePipelineState> pipe = [compiler newComputePipelineStateWithDescriptor:pdesc compilerTaskOptions:nil error:&err];
        if (!pipe) { fprintf(stderr, "pipe: %s\n", [[err description] UTF8String]); return 3; }
        id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];

        // ---- verify 64x64x64 ----
        {
            int M = 64, N = 64, K = 64, nb = N / 32;
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            unsigned char* Bn = (unsigned char*)malloc((size_t)K * nb * 16);
            h16* Bs = (h16*)malloc((size_t)K * nb * sizeof(h16));
            float* C = (float*)calloc((size_t)M * N, sizeof(float));
            srand(21);
            for (int i = 0; i < M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            for (size_t i = 0; i < (size_t)K * nb * 16; i++) Bn[i] = (unsigned char)(rand() & 0xFF);
            for (int i = 0; i < K * nb; i++) Bs[i] = (h16)((rand() % 500 + 100) / 1000.0f);
            int dims[4] = {M, N, K, 0};

            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bBn = [dev newBufferWithBytes:Bn length:(size_t)K * nb * 16 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bBs = [dev newBufferWithBytes:Bs length:(size_t)K * nb * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC = [dev newBufferWithBytes:C length:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bD = [dev newBufferWithBytes:dims length:16 options:MTLResourceStorageModeShared];

            if (run(dev, queue, alloc, pipe, bA, bBn, bBs, bC, bD, M, N, K, 1)) return 4;

            float* g = (float*)bC.contents;
            double md = 0;
            for (int i = 0; i < M * N; i++) {
                int rr = i / N, cc = i % N;
                double r = 0;
                for (int k = 0; k < K; k++) r += (double)A[rr * K + k] * (double)dq(Bn, Bs, K, N, k, cc);
                double d = fabs(r - g[i]); if (d > md) md = d;
            }
            fprintf(stderr, "verify w4 64x64x64: maxabs=%.5f (g[0]=%.4f)\n", md, g[0]);
            free(A); free(Bn); free(Bs); free(C);
        }

        // ---- bench ----
        int Ms[2] = {128, 1024};
        const int N = 4096, K = 4096, BATCH = 20, nb = N / 32;
        unsigned char* Bn = (unsigned char*)malloc((size_t)K * nb * 16);
        h16* Bs = (h16*)malloc((size_t)K * nb * sizeof(h16));
        srand(3);
        for (size_t i = 0; i < (size_t)K * nb * 16; i++) Bn[i] = (unsigned char)(rand() & 0xFF);
        for (int i = 0; i < K * nb; i++) Bs[i] = (h16)((rand() % 500 + 100) / 1000.0f);
        id<MTLBuffer> bBn = [dev newBufferWithBytes:Bn length:(size_t)K * nb * 16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBs = [dev newBufferWithBytes:Bs length:(size_t)K * nb * sizeof(h16) options:MTLResourceStorageModeShared];
        for (int mi = 0; mi < 2; mi++) {
            int M = Ms[mi];
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            srand(5);
            for (long i = 0; i < (long)M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC = [dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            int dims[4] = {M, N, K, 0};
            id<MTLBuffer> bD = [dev newBufferWithBytes:dims length:16 options:MTLResourceStorageModeShared];

            for (int w = 0; w < 3; w++) run(dev, queue, alloc, pipe, bA, bBn, bBs, bC, bD, M, N, K, 1);
            double t0 = now_ms();
            run(dev, queue, alloc, pipe, bA, bBn, bBs, bC, bD, M, N, K, BATCH);
            double dt = (now_ms() - t0) / BATCH;
            printf("W4 M=%d N=%d K=%d: %.4f ms/call  %.1f TFLOP/s\n", M, N, K, dt, 2.0 * M * N * (double)K / dt / 1e9);
            free(A);
        }
        free(Bn); free(Bs);
    }
    return 0;
}
