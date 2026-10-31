// q4k_bench.mm — oracolo del GEMV q4_K a dequant FUSO (I2-full, mattone: gemv_gguf_q4k.metal).
//
// Legge un tensore q4_K VERO (blocchi grezzi estratti da un GGUF con `dd`) e confronta:
//   y_ref[n] = <dequant_row_q4_K(riga n) , x>      (llmcodec_ggml_dequant_row, path del motore)
//   y_k[n]   = gemv_q4_k(...)                       (kernel Metal, legge i blocchi grezzi)
// e misura: GB/s dei blocchi vs GEMV F16 (pesi F16 costruiti dal dequant), speedup, max|diff|.
//
// Build:
//   clang++ -std=c++20 -O2 -fobjc-arc -x objective-c++ LLMEngine/metal/q4k_bench.mm \
//     -I LLMCodec/include build/LLMCodec/libllmcodec.a \
//     -framework Metal -framework Foundation -o /tmp/q4k_bench
// Uso:
//   /tmp/q4k_bench <tensor_q4k.raw> <N> <K>
// (K multiplo di 256; il file deve contenere N*K/256*144 byte)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>
#include "llmcodec_ggml_quants.h"

static double now_ms(void) { return (double)mach_absolute_time() * 1e-6; } /* units: ns*1e-6 ~ ms */

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 4) { fprintf(stderr, "uso: %s <tensor_q4k.raw> <N> <K>\n", argv[0]); return 2; }
        const int N = atoi(argv[2]), K = atoi(argv[3]);
        if (K % 256) { fprintf(stderr, "K deve essere multiplo di 256\n"); return 2; }
        const size_t NB = (size_t)K / 256, rowb = NB * 144;
        FILE* f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "open fallito\n"); return 3; }
        unsigned char* w = (unsigned char*)malloc(rowb * (size_t)N);
        if (!w || fread(w, rowb * (size_t)N, 1, f) != 1) { fprintf(stderr, "lettura corta\n"); return 3; }
        fclose(f);

        // x sintetico + tensore F16 di riferimento (dal dequant del codec = path del motore)
        float* x = (float*)malloc((size_t)K * sizeof(float));
        for (int i = 0; i < K; i++) x[i] = (float)(((i * 37) % 2000) - 1000) * 0.001f;
        _Float16* w16 = (_Float16*)malloc((size_t)N * (size_t)K * sizeof(_Float16));
        float* rowf = (float*)malloc((size_t)K * sizeof(float));
        double* y_ref = (double*)malloc((size_t)N * sizeof(double));
        for (int n = 0; n < N; n++) {
            if (llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_K, w + (size_t)n * rowb, rowf, K) != 0) {
                fprintf(stderr, "dequant_row fallito (riga %d)\n", n); return 4;
            }
            double acc = 0.0;
            for (int k = 0; k < K; k++) { w16[(size_t)n * K + k] = (_Float16)rowf[k]; acc += (double)rowf[k] * (double)x[k]; }
            y_ref[n] = acc;
        }
        free(rowf);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        /* il .metal sta accanto all'eseguibile (o accanto al sorgente, se si compila in-tree) */
        char exe[4096]; uint32_t exe_sz = sizeof exe;
        NSString* dir = @".";
        if (_NSGetExecutablePath(exe, &exe_sz) == 0)
            dir = [[NSString stringWithUTF8String:exe] stringByDeletingLastPathComponent];
        NSString* path = [dir stringByAppendingPathComponent:@"gemv_gguf_q4k.metal"];
        NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
        if (!src) {
            NSString* p2 = [[[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent]
                            stringByAppendingPathComponent:@"gemv_gguf_q4k.metal"];
            src = [NSString stringWithContentsOfFile:p2 encoding:NSUTF8StringEncoding error:nil];
        }
        if (!src) { fprintf(stderr, "kernel .metal non trovato accanto al binario\n"); return 5; }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 5; }
        id<MTLComputePipelineState> p0  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_q4_k"] error:&err];
        id<MTLComputePipelineState> p16 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_f16"]  error:&err];
        if (!p0 || !p16) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 5; }

        id<MTLBuffer> bw  = [dev newBufferWithBytes:w   length:rowb * (size_t)N options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw16= [dev newBufferWithBytes:w16 length:(size_t)N * (size_t)K * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx  = [dev newBufferWithBytes:x   length:(size_t)K * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo  = [dev newBufferWithLength:(size_t)N * 4 options:MTLResourceStorageModeShared];
        /* N e K in buffer DISTINTI: passarli dallo stesso buffer fa leggere K=N (bug latente
         * dei bench: innocuo solo per le forme quadrate). */
        int pN = N, pK = K;
        id<MTLBuffer> bpN = [dev newBufferWithBytes:&pN length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bpK = [dev newBufferWithBytes:&pK length:4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        const int reps = 20, BATCH = 10;
        __block double t_q4 = 0, t_f16 = 0;
        void (^run)(id<MTLComputePipelineState>, id<MTLBuffer>, double*) = ^(id<MTLComputePipelineState> pso, id<MTLBuffer> bwp, double* out_ms) {
            for (int r = 0; r < 3; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pso];
                [e setBuffer:bwp offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
                [e setBuffer:bo offset:0 atIndex:2];  [e setBuffer:bpN offset:0 atIndex:3]; [e setBuffer:bpK offset:0 atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake((N + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            uint64_t t0 = mach_absolute_time();
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int i = 0; i < BATCH; i++) {
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:pso];
                    [e setBuffer:bwp offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
                    [e setBuffer:bo offset:0 atIndex:2];  [e setBuffer:bpN offset:0 atIndex:3]; [e setBuffer:bpK offset:0 atIndex:4];
                    [e dispatchThreadgroups:MTLSizeMake((N + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [e endEncoding];
                }
                [cb commit]; [cb waitUntilCompleted];
            }
            uint64_t t1 = mach_absolute_time();
            mach_timebase_info_data_t tb; mach_timebase_info(&tb);
            *out_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6 / (double)(reps * BATCH);
        };
        run(p0, bw, &t_q4);
        /* COPIA subito: la run f16 riusa lo stesso buffer di output */
        float* yk = (float*)malloc((size_t)N * sizeof(float));
        memcpy(yk, bo.contents, (size_t)N * sizeof(float));
        run(p16, bw16, &t_f16);

        double mx = 0; int worst = -1, nan = 0;
        for (int n = 0; n < N; n++) {
            if (!isfinite(yk[n]) || !isfinite(y_ref[n])) { nan++; continue; }
            double d = fabs((double)yk[n] - y_ref[n]); if (d > mx) { mx = d; worst = n; }
        }
        double scale = 1.0; for (int n = 0; n < N; n++) if (isfinite(y_ref[n]) && fabs(y_ref[n]) > scale) scale = fabs(y_ref[n]);
        if (nan) printf("  ATTENZIONE: %d righe non finite (dati estremi/overflow) -> escluse dalla parita'\n", nan);
        printf("q4_K VERO (GGUF): N=%d K=%d  riga=%zu B  (%.1f B/peso)\n", N, K, rowb, (double)rowb / K);
        printf("  fused : %8.4f ms  %7.1f GB/s(blocchi)\n", t_q4,  (double)rowb * N / t_q4 / 1e6);
        printf("  f16   : %8.4f ms  %7.1f GB/s\n",          t_f16, (double)N * K * 2 / t_f16 / 1e6);
        printf("  speedup fused vs f16: %.2fx\n", t_f16 / t_q4);
        printf("  parita' vs dequant_row del codec: max|diff|=%.3e (rel %.2e) riga %d\n", mx, mx / scale, worst);
        return (mx / scale < 1e-5) ? 0 : 1;
    }
}
