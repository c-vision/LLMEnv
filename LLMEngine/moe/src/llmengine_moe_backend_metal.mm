/* llmengine_moe_backend_metal.mm — backend Metal (ObjC++, M8). Nessun C#; solo API Metal di sistema. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "llmengine_moe_backend_metal.h"

#include <string.h>

static NSString* const kLlmEngineMoeKernelSrc =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void moe_expert(device const float* blob [[buffer(0)]],\n"
     "                       device const float* x [[buffer(1)]],\n"
     "                       device float* y [[buffer(2)]],\n"
     "                       constant uint& K [[buffer(3)]],\n"
     "                       uint j [[thread_position_in_grid]]) {\n"
     "    float s = 0.0f;\n"
     "    for (uint i = 0; i < K; ++i) s += blob[j * K + i] * x[i];\n"
     "    y[j] = s;\n"
     "}\n";

struct LlmEngineMoeMetalBackend {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
};

int llmengine_moe_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        return d != nil ? 1 : 0;
    }
}

LlmEngineMoeStatus llmengine_moe_metal_create(LlmEngineMoeMetalBackend** out, const char* kernel_src) {
    if (out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    *out = NULL;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED; /* nessuna GPU Metal */
        NSError* err = nil;
        NSString* src = (kernel_src != NULL && kernel_src[0] != '\0')
                            ? [NSString stringWithUTF8String:kernel_src]
                            : kLlmEngineMoeKernelSrc;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (lib == nil) return LLMENGINE_MOE_ERR_ADAPTER;
        id<MTLFunction> fn = [lib newFunctionWithName:@"moe_expert"];
        if (fn == nil) { [lib release]; return LLMENGINE_MOE_ERR_ADAPTER; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        [lib release];
        if (pso == nil) return LLMENGINE_MOE_ERR_ADAPTER;
        LlmEngineMoeMetalBackend* b = (LlmEngineMoeMetalBackend*)calloc(1, sizeof(*b));
        if (b == NULL) { [pso release]; return LLMENGINE_MOE_ERR_NO_MEMORY; }
        b->dev = [dev retain];
        b->queue = [[dev newCommandQueue] retain];
        b->pso = pso; /* gia' +1 da new */
        *out = b;
        return LLMENGINE_MOE_OK;
    }
}

void llmengine_moe_metal_destroy(LlmEngineMoeMetalBackend* b) {
    if (b == NULL) return;
    @autoreleasepool {
        if (b->pso) [b->pso release];
        if (b->queue) [b->queue release];
        if (b->dev) [b->dev release];
        free(b);
    }
}

LlmEngineMoeStatus llmengine_moe_metal_upload(LlmEngineMoeMetalBackend* b, const void* blob, size_t size, void** addr_out) {
    if (b == NULL || blob == NULL || addr_out == NULL || size == 0) return LLMENGINE_MOE_ERR_INVALID;
    *addr_out = NULL;
    @autoreleasepool {
        id<MTLBuffer> buf = [b->dev newBufferWithBytes:blob length:size options:MTLResourceStorageModeShared];
        if (buf == nil) return LLMENGINE_MOE_ERR_NO_MEMORY;
        *addr_out = (void*)buf; /* +1 (new); released da llmengine_moe_metal_release */
        return LLMENGINE_MOE_OK;
    }
}

void llmengine_moe_metal_release(LlmEngineMoeMetalBackend* b, void* addr) {
    (void)b;
    if (addr == NULL) return;
    @autoreleasepool { [(id<MTLBuffer>)addr release]; }
}

LlmEngineMoeStatus llmengine_moe_metal_expert_compute(LlmEngineMoeMetalBackend* b, void* addr, size_t blob_bytes,
                                          const float* in, uint32_t in_size,
                                          float* out, uint32_t out_size) {
    if (b == NULL || addr == NULL || in == NULL || out == NULL || in_size == 0 || out_size == 0)
        return LLMENGINE_MOE_ERR_INVALID;
    if (blob_bytes < (size_t)in_size * out_size * sizeof(float)) return LLMENGINE_MOE_ERR_INVALID;
    @autoreleasepool {
        id<MTLBuffer> blob = (id<MTLBuffer>)addr;
        id<MTLBuffer> bx = [b->dev newBufferWithBytes:in length:(NSUInteger)in_size * sizeof(float)
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [b->dev newBufferWithLength:(NSUInteger)out_size * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        if (bx == nil || by == nil) { if (bx) [bx release]; if (by) [by release]; return LLMENGINE_MOE_ERR_NO_MEMORY; }
        uint32_t K = in_size;
        id<MTLCommandBuffer> cb = [b->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:b->pso];
        [enc setBuffer:blob offset:0 atIndex:0];
        [enc setBuffer:bx offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&K length:sizeof(K) atIndex:3];
        MTLSize grid = MTLSizeMake(out_size, 1, 1);
        NSUInteger tg = b->pso.maxTotalThreadsPerThreadgroup;
        if (tg > out_size) tg = out_size;
        if (tg == 0) tg = 1;
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted]; /* sync: il risultato e' pronto (M8: sync esplicito) */
        memcpy(out, [by contents], (size_t)out_size * sizeof(float));
        [bx release]; [by release];
        return LLMENGINE_MOE_OK;
    }
}

LlmEngineMoeStatus llmengine_moe_metal_sync(LlmEngineMoeMetalBackend* b) {
    (void)b; /* expert_compute e' sincrono; nulla in coda */
    return LLMENGINE_MOE_OK;
}
