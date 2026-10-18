// mpp_bench_t4.m — harness per mpp_gemm_t4.metal (MPP con input tensor, 4 simdgroup).
// Crea MTLTensor con attachments sui nostri buffer (macOS 27), li binda con
// setResource:atBufferIndex:, dispatcha 128 thread. Verifica + bench.
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
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f); return b;
}

static id<MTLTensor> mk_tensor(id<MTLDevice> dev, id<MTLBuffer> buf, MTLTensorDataType dt,
                               int d0, int d1, int s0, int s1, NSError** err) {
    MTLTensorDescriptor* d = [MTLTensorDescriptor new];
    d.dataType = dt;
    d.usage = MTLTensorUsageMachineLearning;
    d.storageMode = MTLStorageModeShared;
    NSInteger dims[2] = {d0, d1}, str[2] = {s0, s1};
    d.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
    d.strides = [[MTLTensorExtents alloc] initWithRank:2 values:str];
    MTLTensorBufferAttachments* at = [MTLTensorBufferAttachments new];
    [at setBuffer:buf offset:0 forPlane:MTLTensorPlaneTypeData];
    return [dev newTensorWithDescriptor:d attachments:at error:err];
}

static int launch(id<MTLDevice> dev, id<MTL4CommandQueue> queue, id<MTL4CommandAllocator> alloc,
                  id<MTLComputePipelineState> pipe, id<MTLTensor> tA, id<MTLTensor> tB, id<MTLTensor> tC,
                  int M, int N, int batch) {
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 3;
    NSError* err = nil;
    id<MTL4ArgumentTable> tab = [dev newArgumentTableWithDescriptor:atd error:&err];
    if (!tab) { fprintf(stderr, "argtable: %s\n", [[err description] UTF8String]); return 1; }
    [tab setResource:[tA gpuResourceID] atBufferIndex:0];
    [tab setResource:[tB gpuResourceID] atBufferIndex:1];
    [tab setResource:[tC gpuResourceID] atBufferIndex:2];

    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    rsd.initialCapacity = 3;
    id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
    if (rs) { [rs addAllocation:[tA buffer]]; [rs addAllocation:[tB buffer]]; [rs addAllocation:[tC buffer]]; [rs commit]; [rs requestResidency]; [queue addResidencySet:rs]; }

    id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:alloc];
    for (int b = 0; b < batch; b++) {
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setArgumentTable:tab];
        [enc dispatchThreadgroups:MTLSizeMake(N / 32, M / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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
        const char* mpath = argc > 1 ? argv[1] : "mpp_gemm_t4.metal";
        const char* fname = argc > 2 ? argv[2] : "mpp_gemm_t4";
        char* src = read_file(mpath);
        if (!src) { fprintf(stderr, "cannot read %s\n", mpath); return 1; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:[MTLCompileOptions new] error:&err];
        free(src);
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 2; }

        MTL4LibraryFunctionDescriptor* fd = [MTL4LibraryFunctionDescriptor new];
        fd.library = lib; fd.name = [NSString stringWithUTF8String:fname];
        MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
        pd.computeFunctionDescriptor = fd;
        pd.requiredThreadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        id<MTL4Compiler> comp = [dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        id<MTLComputePipelineState> pipe = [comp newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&err];
        if (!pipe) { fprintf(stderr, "pipe: %s\n", [[err description] UTF8String]); return 3; }
        id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];

        // ---- verify 64x64x64 ----
        {
            int M = 64, N = 64, K = 64;
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            h16* B = (h16*)malloc((size_t)K * N * sizeof(h16));
            float* C = (float*)calloc((size_t)M * N, sizeof(float));
            srand(31);
            for (int i = 0; i < M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            for (int i = 0; i < K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB = [dev newBufferWithBytes:B length:(size_t)K * N * sizeof(h16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC = [dev newBufferWithBytes:C length:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLTensor> tA = mk_tensor(dev, bA, MTLTensorDataTypeFloat16, K, M, 1, K, &err);
            id<MTLTensor> tB = mk_tensor(dev, bB, MTLTensorDataTypeFloat16, N, K, 1, N, &err);
            id<MTLTensor> tC = mk_tensor(dev, bC, MTLTensorDataTypeFloat32, N, M, 1, N, &err);
            if (!tA || !tB || !tC) { fprintf(stderr, "tensor: %s\n", err ? [[err description] UTF8String] : "nil"); return 4; }
            if (launch(dev, queue, alloc, pipe, tA, tB, tC, M, N, 1)) return 5;
            float* g = (float*)bC.contents;
            double md = 0;
            for (int i = 0; i < M * N; i++) {
                int rr = i / N, cc = i % N; double r = 0;
                for (int k = 0; k < K; k++) r += (double)A[rr * K + k] * (double)B[k * N + cc];
                double d = fabs(r - g[i]); if (d > md) md = d;
            }
            fprintf(stderr, "verify t4 64x64x64: maxabs=%.5f (g[0]=%.4f)\n", md, g[0]);
            free(A); free(B); free(C);
        }

        // ---- bench ----
        int Ms[2] = {128, 1024};
        const int N = 4096, K = 4096, BATCH = 20;
        h16* B = (h16*)malloc((size_t)K * N * sizeof(h16));
        srand(3);
        for (long i = 0; i < (long)K * N; i++) B[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
        id<MTLBuffer> bB = [dev newBufferWithBytes:B length:(size_t)K * N * sizeof(h16) options:MTLResourceStorageModeShared];
        free(B);
        for (int mi = 0; mi < 2; mi++) {
            int M = Ms[mi];
            h16* A = (h16*)malloc((size_t)M * K * sizeof(h16));
            srand(5);
            for (long i = 0; i < (long)M * K; i++) A[i] = (h16)((rand() % 2001 - 1000) / 1000.0f);
            id<MTLBuffer> bA = [dev newBufferWithBytes:A length:(size_t)M * K * sizeof(h16) options:MTLResourceStorageModeShared];
            float* Cz = (float*)calloc((size_t)M * N, sizeof(float));
            id<MTLBuffer> bC = [dev newBufferWithBytes:Cz length:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
            free(A); free(Cz);
            id<MTLTensor> tA = mk_tensor(dev, bA, MTLTensorDataTypeFloat16, K, M, 1, K, &err);
            id<MTLTensor> tB = mk_tensor(dev, bB, MTLTensorDataTypeFloat16, N, K, 1, N, &err);
            id<MTLTensor> tC = mk_tensor(dev, bC, MTLTensorDataTypeFloat32, N, M, 1, N, &err);
            if (!tA || !tB || !tC) { fprintf(stderr, "tensor bench: %s\n", err ? [[err description] UTF8String] : "nil"); return 6; }
            for (int w = 0; w < 3; w++) launch(dev, queue, alloc, pipe, tA, tB, tC, M, N, 1);
            double t0 = now_ms();
            launch(dev, queue, alloc, pipe, tA, tB, tC, M, N, BATCH);
            double dt = (now_ms() - t0) / BATCH;
            printf("T4 M=%d N=%d K=%d: %.4f ms/call  %.1f TFLOP/s\n", M, N, K, dt, 2.0 * M * N * (double)K / dt / 1e9);
        }
    }
    return 0;
}
