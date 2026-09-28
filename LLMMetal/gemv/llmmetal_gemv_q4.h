/* llmmetal_gemv_q4.h — GEMV decode Q4 affine, kernel Metal custom.
 *
 * Attacca il "dato decisivo" del piano
 * (~/Documents/AiWiki/01-Projects/LLMEnv/Piani/Proposta-614GBs-Banda-Metal-DeepDive.md, §2.1/§4
 * L3): la GEMV quantizzata Q4 affine di MLX misura ~26 GB/s (~4% del picco 614 GB/s) su shape
 * Llama-1B (x[1,2048] @ W[8192,2048], g64 affine). Questo modulo implementa il design del piano
 * (staging cooperativo di x in threadgroup memory, un simdgroup per riga, riduzione simd_sum)
 * per lo stesso caso.
 *
 * Formato Q4 affine (autoconsistente per questo spike — NON ancora verificato bit-esatto contro
 * il layout interno di MLX; l'integrazione con mx.quantize reale è lavoro successivo, qui si
 * valida la TECNICA del kernel e la banda raggiungibile, non la compatibilità di formato):
 *   - pesi: [M, H/2] uint8, 2 nibble/byte (nibble basso = colonna pari, alto = colonna dispari)
 *   - scale/bias: [M, H/group_size] half, affine: w = nibble*scale + bias
 *
 * Vincoli v1: H multiplo di 64 (32 lane * 2 nibble/byte, e compatibile con group_size=64);
 * group_size multiplo di 1 (nessun vincolo di allineamento ulteriore in questa versione).
 *
 * Design: un simdgroup (32 lane) per riga di W; ROWS_PER_TG simdgroup per threadgroup
 * condividono lo staging di x (caricato una volta in threadgroup memory, non riletto per ogni
 * riga) — stesso principio del kernel GDN di produzione (llmmetal_mlx_gdn_kernel.h).
 */
#ifndef LLMMETAL_METAL_GEMV_Q4_H
#define LLMMETAL_METAL_GEMV_Q4_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LLMMETAL_GEMV_ROWS_PER_TG 16

static const char llmmetal_gemv_q4_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Q4Params { uint H; uint M; uint group_size; uint rows_per_tg; };\n"
"\n"
"kernel void gemv_q4_affine(\n"
"    const device half*  x       [[buffer(0)]],\n"
"    const device uchar* Wq      [[buffer(1)]],\n"
"    const device half*  scales  [[buffer(2)]],\n"
"    const device half*  biases  [[buffer(3)]],\n"
"    device half*        out     [[buffer(4)]],\n"
"    constant Q4Params&  p       [[buffer(5)]],\n"
"    threadgroup half*   x_shared [[threadgroup(0)]],\n"
"    uint3 tgid   [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3  [[thread_position_in_threadgroup]],\n"
"    uint  lane   [[thread_index_in_simdgroup]],\n"
"    uint  simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    const uint ltid = ltid3.x;\n"
"    const uint tg_threads = p.rows_per_tg * 32;\n"
"    for (uint i = ltid; i < p.H; i += tg_threads) x_shared[i] = x[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    const uint row = tgid.x * p.rows_per_tg + simd_id;\n"
"    if (row >= p.M) return;\n"
"\n"
"    const uint per_lane = p.H / 32;\n"
"    const device uchar* row_packed = Wq + (size_t)row * (p.H / 2);\n"
"    const uint n_groups = p.H / p.group_size;\n"
"    const device half* row_scales = scales + (size_t)row * n_groups;\n"
"    const device half* row_biases = biases + (size_t)row * n_groups;\n"
"\n"
"    // Bug di prestazioni RISOLTO (2026-09-06): con `h = lane*per_lane + i`, a un dato `i` i 32\n"
"    // lane leggono `row_packed` a byte_idx SPACED `per_lane/2` byte l'uno dall'altro (32\n"
"    // locazioni sparse invece di una lettura coalescente) — lo spacing CRESCE con `per_lane`\n"
"    // (=H/32), spiegando perche' la banda misurata CROLLAVA al crescere di H (26 GB/s a\n"
"    // H=2048, ~9 GB/s a H=8192) mentre un kernel fp16 equivalente senza dequant SALIVA con H\n"
"    // (22->64 GB/s): la firma classica di un accesso non coalescente che peggiora quanto piu'\n"
"    // le letture dei singoli lane si allontanano fra loro. Fix: `h = i*32 + lane` (round-robin\n"
"    // sui lane) — a un dato `i` i 32 lane leggono 32 `h` CONSECUTIVI, quindi byte_idx copre 16\n"
"    // byte consecutivi (2 nibble/byte): una singola lettura coalescente per iterazione invece\n"
"    // di 32 sparse. Risultato: 27->28 GB/s (H=2048, invariato, gia' vicino al limite fisso di\n"
"    // dispatch a quella dimensione) ma 9->73 GB/s a H=8192 (8x, ora SUPERA sia MLX ~26 GB/s sia\n"
"    // il kernel fp16 di controllo alla stessa shape, 63.7 GB/s) — la banda ora SALE con H come\n"
"    // atteso.\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < per_lane; i++) {\n"
"        uint h = i * 32 + lane;\n"
"        uint byte_idx = h >> 1;\n"
"        uchar b = row_packed[byte_idx];\n"
"        uint nib = (h & 1u) ? (b >> 4) : (b & 0xF);\n"
"        uint g = h / p.group_size;\n"
"        float w = (float)nib * (float)row_scales[g] + (float)row_biases[g];\n"
"        acc += w * (float)x_shared[h];\n"
"    }\n"
"    float sum = simd_sum(acc);\n"
"    if (lane == 0) out[row] = (half)sum;\n"
"}\n"
;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso;
    /* pesi persistenti — caricati UNA VOLTA da llmmetal_gemv_q4_load_weights, riusati per ogni
     * token di decode. La versione precedente riallocava/copiava Wq (megabyte) ad ogni chiamata:
     * misurava soprattutto newBufferWithBytes+memcpy, non il kernel — non rappresentativo del
     * decode reale, dove i pesi sono fissi e solo x cambia ad ogni step. */
    id<MTLBuffer> w_buf, scale_buf, bias_buf, out_buf, x_buf;
    uint32_t H, M, group_size;
} LlmMetalGemvQ4Ctx;

static int llmmetal_gemv_q4_init(LlmMetalGemvQ4Ctx* ctx) {
    ctx->device = nil; ctx->queue = nil; ctx->library = nil; ctx->pso = nil;
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->bias_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { fprintf(stderr, "llmmetal_gemv_q4_init: nessun device Metal\n"); return -1; }
    ctx->queue = [ctx->device newCommandQueue];
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:llmmetal_gemv_q4_src];
    ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&err];
    if (!ctx->library) { fprintf(stderr, "llmmetal_gemv_q4_init: compile err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> f = [ctx->library newFunctionWithName:@"gemv_q4_affine"];
    ctx->pso = [ctx->device newComputePipelineStateWithFunction:f error:&err];
    if (!ctx->pso) { fprintf(stderr, "llmmetal_gemv_q4_init: pso err: %s\n", err.localizedDescription.UTF8String); return -1; }
    return 0;
}

static void llmmetal_gemv_q4_free(LlmMetalGemvQ4Ctx* ctx) {
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->bias_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    ctx->pso = nil; ctx->library = nil; ctx->queue = nil; ctx->device = nil;
}

/* Carica i pesi quantizzati UNA VOLTA (equivalente al load del modello): Wq [M,H/2] uint8,
 * scales/biases [M,H/group_size] half. Alloca anche i buffer persistenti per x/out. */
static int llmmetal_gemv_q4_load_weights(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                           const uint16_t* scales, const uint16_t* biases,
                                           uint32_t H, uint32_t M, uint32_t group_size) {
    if (H % 64 != 0) { fprintf(stderr, "llmmetal_gemv_q4_load_weights: H %u non multiplo di 64\n", H); return -1; }
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    uint32_t n_groups = H / group_size;
    ctx->w_buf = [ctx->device newBufferWithBytes:Wq length:(size_t)M*(H/2) options:MTLResourceStorageModeShared];
    ctx->scale_buf = [ctx->device newBufferWithBytes:scales length:(size_t)M*n_groups*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->bias_buf = [ctx->device newBufferWithBytes:biases length:(size_t)M*n_groups*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->out_buf = [ctx->device newBufferWithLength:(size_t)M*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(size_t)H*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    if (!ctx->w_buf || !ctx->scale_buf || !ctx->bias_buf || !ctx->out_buf || !ctx->x_buf) {
        fprintf(stderr, "llmmetal_gemv_q4_load_weights: allocazione fallita\n"); return -1;
    }
    return 0;
}

/* Un passo di decode: scrive x (memcpy in memoria unificata, come llmmetal_kv_write), esegue
 * il kernel sui pesi gia' residenti, legge out. Nessuna riallocazione/copia dei pesi. */
static int llmmetal_gemv_q4_run(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg; } params = { ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMV_ROWS_PER_TG };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->w_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->scale_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->bias_buf offset:0 atIndex:3];
    [enc setBuffer:ctx->out_buf offset:0 atIndex:4];
    [enc setBytes:&params length:sizeof(params) atIndex:5];
    [enc setThreadgroupMemoryLength:(ctx->H*sizeof(uint16_t)) atIndex:0];
    uint32_t n_tg = (ctx->M + LLMMETAL_GEMV_ROWS_PER_TG - 1) / LLMMETAL_GEMV_ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->M*sizeof(uint16_t));
    return 0;
}

#endif /* LLMMETAL_METAL_GEMV_Q4_H */
