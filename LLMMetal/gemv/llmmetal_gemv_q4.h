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
"struct Q4Params { uint H; uint M; uint group_size; uint rows_per_tg; uint lead_w; uint lead_s; uint lead_b; uint scale_bf16; };\n"
"inline float q4_ld_sb(const device uchar* p, uint idx, uint bf) {\n"
"    const ushort u = (ushort)p[2u * idx] | ((ushort)p[2u * idx + 1u] << 8);   // lettura a byte: i tensori safetensors non sono allineati a 2\n"
"    return bf ? as_type<float>(((uint)u) << 16) : (float)as_type<half>(u);\n"
"}\n"
"kernel void gemv_q4_affine(\n"
"    const device half*  x       [[buffer(0)]],\n"
"    const device uchar* Wq      [[buffer(1)]],\n"
"    const device uchar* scales  [[buffer(2)]],\n"
"    const device uchar* biases  [[buffer(3)]],\n"
"    device half*        out     [[buffer(4)]],\n"
"    constant Q4Params&  p       [[buffer(5)]],\n"
"    threadgroup half*   x_shared [[threadgroup(0)]],\n"
"    uint3 tgid   [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3  [[thread_position_in_threadgroup]],\n"
"    uint  lane   [[thread_index_in_simdgroup]],\n"
"    uint  simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    Wq = (const device uchar*)Wq + p.lead_w;\n"
"    scales = scales + p.lead_s;\n"
"    biases = biases + p.lead_b;\n"
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
"    const device uchar* row_scales = scales + (size_t)row * n_groups * 2;\n"
"    const device uchar* row_biases = biases + (size_t)row * n_groups * 2;\n"
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
"        float w = (float)nib * q4_ld_sb(row_scales, g, p.scale_bf16) + q4_ld_sb(row_biases, g, p.scale_bf16);\n"
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
    size_t lead_w, lead_s, lead_b;   /* offset del tensore dentro la view page-aligned (nocopy) */
    uint32_t H, M, group_size;
    uint32_t scale_bf16;   /* 1 = scales/biases BF16 (default 0 = half) */
    size_t w_stride, s_stride;   /* vista 3D (esperti): byte per esperto */
    uint32_t n_experts;
    const uint8_t *p_w, *p_s, *p_b;   /* vista 3D: puntatori base ai tensori interi (mmap) */
} LlmMetalGemvQ4Ctx;

/* Zero-copy: le view Metal puntano DIRETTAMENTE al mmap delle safetensors (page-aligned);
 * il lead del tensore viene applicato come offset al momento del dispatch. */
static int llmmetal_gemv_q4_load_weights_nocopy(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                                const uint16_t* scales, const uint16_t* biases,
                                                uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx || !Wq || !scales || H % 64 != 0) return -1;
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    const size_t page = (size_t)getpagesize();
    const size_t wbytes = (size_t)M * (H / 2);
    const size_t ngrp = H / group_size;
    const size_t sbytes = (size_t)M * ngrp * sizeof(uint16_t);
    const uint8_t* wp = Wq; const uint8_t* sp = (const uint8_t*)scales; const uint8_t* bp = (const uint8_t*)biases;
    uintptr_t wb = (uintptr_t)wp & ~(uintptr_t)(page - 1); ctx->lead_w = (size_t)((uintptr_t)wp - wb);
    uintptr_t sb = (uintptr_t)sp & ~(uintptr_t)(page - 1); ctx->lead_s = (size_t)((uintptr_t)sp - sb);
    uintptr_t bb = bp ? ((uintptr_t)bp & ~(uintptr_t)(page - 1)) : 0; ctx->lead_b = bp ? (size_t)((uintptr_t)bp - bb) : 0;
    size_t wl = ((ctx->lead_w + wbytes + page - 1) / page) * page;
    size_t sl = ((ctx->lead_s + sbytes + page - 1) / page) * page;
    size_t bl = bp ? ((ctx->lead_b + sbytes + page - 1) / page) * page : 0;
    ctx->w_buf = [ctx->device newBufferWithBytesNoCopy:(void*)wb length:wl options:MTLResourceStorageModeShared deallocator:nil];
    ctx->scale_buf = [ctx->device newBufferWithBytesNoCopy:(void*)sb length:sl options:MTLResourceStorageModeShared deallocator:nil];
    ctx->bias_buf = bp ? [ctx->device newBufferWithBytesNoCopy:(void*)bb length:bl options:MTLResourceStorageModeShared deallocator:nil] : nil;
    ctx->out_buf = [ctx->device newBufferWithLength:(NSUInteger)M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(NSUInteger)H * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    if (ctx->w_buf && ctx->scale_buf) {
        const uint8_t* wc = (const uint8_t*)[ctx->w_buf contents] + ctx->lead_w;
        const uint8_t* sc = (const uint8_t*)[ctx->scale_buf contents] + ctx->lead_s;
        int badw = memcmp(wc, Wq, 8) != 0, bads = memcmp(sc, scales, 8) != 0;
        if (badw || bads) fprintf(stderr, "[nocopy] mismatch w=%d s=%d lead_w=%zu lead_s=%zu\n", badw, bads, ctx->lead_w, ctx->lead_s);
    }
    if (!ctx->w_buf || !ctx->scale_buf || !ctx->out_buf || !ctx->x_buf) {
        fprintf(stderr, "llmmetal_gemv_q4_load_weights_nocopy: view fallita (lead w=%zu s=%zu)\n", ctx->lead_w, ctx->lead_s);
        return -1;
    }
    return 0;
}

/* Zero-copy SOLO sui pesi (i grandi); scales/biases sono piccoli e vengono copiati: evita i
 * problemi di allineamento dei tipi half su view file-backed. */
static int llmmetal_gemv_q4_load_weights_wonly_nocopy(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                                      const uint16_t* scales, const uint16_t* biases,
                                                      uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx || !Wq || !scales || H % 64 != 0) return -1;
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    const size_t page = (size_t)getpagesize();
    const size_t wbytes = (size_t)M * (H / 2);
    const size_t ngrp = H / group_size;
    const size_t sbytes = (size_t)M * ngrp * sizeof(uint16_t);
    uintptr_t wb = (uintptr_t)Wq & ~(uintptr_t)(page - 1); ctx->lead_w = (size_t)((uintptr_t)Wq - wb);
    size_t wl = ((ctx->lead_w + wbytes + page - 1) / page) * page;
    ctx->w_buf = [ctx->device newBufferWithBytesNoCopy:(void*)wb length:wl options:MTLResourceStorageModeShared deallocator:nil];
    ctx->scale_buf = [ctx->device newBufferWithBytes:scales length:sbytes options:MTLResourceStorageModeShared];
    ctx->bias_buf = biases ? [ctx->device newBufferWithBytes:biases length:sbytes options:MTLResourceStorageModeShared] : nil;
    ctx->lead_s = 0; ctx->lead_b = 0;
    ctx->out_buf = [ctx->device newBufferWithLength:(NSUInteger)M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(NSUInteger)H * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    return (ctx->w_buf && ctx->scale_buf && ctx->out_buf && ctx->x_buf) ? 0 : -1;
}

/* device/queue/libreria/pipeline sono condivisi da tutti i contesti: compilare il sorgente Metal a ogni create
 * costava ~ms per gemv (migliaia di create per token nel percorso packed MoE). */
static id<MTLDevice> g_gq4_device = nil; static id<MTLCommandQueue> g_gq4_queue = nil;
static id<MTLLibrary> g_gq4_library = nil; static id<MTLComputePipelineState> g_gq4_pso = nil;

/* Vista 3D [E][M][H]: zero-copy su pesi, scales e biases dell'INTERO tensore; l'esperto si sceglie a run-time
 * con gli offset lead_* (nessuna copia e nessun nuovo contesto per esperto). */
static int llmmetal_gemv_q4_load_weights_3d_nocopy(LlmMetalGemvQ4Ctx* ctx, const uint8_t* Wq,
                                                   const uint16_t* scales, const uint16_t* biases,
                                                   uint32_t H, uint32_t M, uint32_t group_size,
                                                   uint32_t E, size_t w_stride, size_t s_stride) {
    if (!ctx || !Wq || !scales || !biases || H % 64 != 0 || E == 0) return -1;
    ctx->H = H; ctx->M = M; ctx->group_size = group_size; ctx->n_experts = E;
    ctx->w_stride = w_stride; ctx->s_stride = s_stride;
    ctx->p_w = Wq; ctx->p_s = (const uint8_t*)scales; ctx->p_b = (const uint8_t*)biases;
    ctx->out_buf = [ctx->device newBufferWithLength:(NSUInteger)M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(NSUInteger)H * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    return (ctx->out_buf && ctx->x_buf) ? 0 : -1;
}

static int llmmetal_gemv_q4_init(LlmMetalGemvQ4Ctx* ctx) {
    ctx->device = nil; ctx->queue = nil; ctx->library = nil; ctx->pso = nil;
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->bias_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    @synchronized([NSProcessInfo processInfo]) {
        if (g_gq4_pso) { ctx->device = g_gq4_device; ctx->queue = g_gq4_queue; ctx->library = g_gq4_library; ctx->pso = g_gq4_pso; return 0; }
    }
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
    @synchronized([NSProcessInfo processInfo]) { g_gq4_device = ctx->device; g_gq4_queue = ctx->queue; g_gq4_library = ctx->library; g_gq4_pso = ctx->pso; }
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
static int llmmetal_gemv_q4_run_bufs(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out,
                                     id<MTLBuffer> wb, id<MTLBuffer> sb, id<MTLBuffer> bb,
                                     size_t lead_w, size_t lead_s, size_t lead_b) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg, lead_w, lead_s, lead_b, scale_bf16; } params = {
        ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMV_ROWS_PER_TG,
        (uint32_t)lead_w, (uint32_t)lead_s, (uint32_t)lead_b, ctx->scale_bf16 };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:sb offset:0 atIndex:2];
    [enc setBuffer:bb offset:0 atIndex:3];
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

static int llmmetal_gemv_q4_run(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    return llmmetal_gemv_q4_run_bufs(ctx, x, out, ctx->w_buf, ctx->scale_buf, ctx->bias_buf, ctx->lead_w, ctx->lead_s, ctx->lead_b);
}

/* view zero-copy (page-aligned) su [p, p+len): ritorna il buffer e il lead */
static id<MTLBuffer> llmmetal_gq4_view(id<MTLDevice> dev, const uint8_t* p, size_t len, size_t* lead) {
    const size_t page = (size_t)getpagesize();
    uintptr_t b = (uintptr_t)p & ~(uintptr_t)(page - 1);
    *lead = (size_t)((uintptr_t)p - b);
    const size_t l = ((*lead + len + page - 1) / page) * page;
    return [dev newBufferWithBytesNoCopy:(void*)b length:l options:MTLResourceStorageModeShared deallocator:nil];
}

/* esperto `e` di una vista 3D (llmmetal_gemv_q4_load_weights_3d_nocopy) */
static int llmmetal_gemv_q4_run_expert(LlmMetalGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out, uint32_t e) {
    if (e >= ctx->n_experts) return -1;
    const size_t wbytes = (size_t)ctx->M * (ctx->H / 2), sbytes = (size_t)ctx->M * (ctx->H / ctx->group_size) * 2;
    size_t lw = 0, ls = 0, lb = 0;
    id<MTLBuffer> wb = llmmetal_gq4_view(ctx->device, ctx->p_w + (size_t)e * ctx->w_stride, wbytes, &lw);
    id<MTLBuffer> sb = llmmetal_gq4_view(ctx->device, ctx->p_s + (size_t)e * ctx->s_stride, sbytes, &ls);
    id<MTLBuffer> bb = llmmetal_gq4_view(ctx->device, ctx->p_b + (size_t)e * ctx->s_stride, sbytes, &lb);
    if (!wb || !sb || !bb) return -1;
    return llmmetal_gemv_q4_run_bufs(ctx, x, out, wb, sb, bb, lw, ls, lb);
}

/* Batch: N gemv (anche su contesti/tensori diversi) in UN solo command buffer. expert < 0 = lineare 2D (viste del contesto).
 * x/out sono buffer transitori per job, quindi lo stesso contesto puo' comparire piu' volte (esperti diversi). */
typedef struct { LlmMetalGemvQ4Ctx* ctx; int64_t expert; const uint16_t* x; uint16_t* out; } LlmMetalGemvQ4Job;

static int llmmetal_gemv_q4_run_batch(const LlmMetalGemvQ4Job* jobs, int n) {
    if (n <= 0) return 0;
    LlmMetalGemvQ4Ctx* c0 = jobs[0].ctx;
    id<MTLCommandBuffer> cb = [c0->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c0->pso];
    NSMutableArray* keep = [NSMutableArray arrayWithCapacity:(NSUInteger)n * 5];
    NSMutableArray* outs = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    for (int i = 0; i < n; i++) {
        LlmMetalGemvQ4Ctx* c = jobs[i].ctx;
        id<MTLBuffer> wb, sb, bb; size_t lw, ls, lb;
        if (jobs[i].expert >= 0) {
            if ((uint64_t)jobs[i].expert >= c->n_experts) { [enc endEncoding]; return -1; }
            const size_t e = (size_t)jobs[i].expert;
            const size_t wbytes = (size_t)c->M * (c->H / 2), sbytes = (size_t)c->M * (c->H / c->group_size) * 2;
            wb = llmmetal_gq4_view(c->device, c->p_w + e * c->w_stride, wbytes, &lw);
            sb = llmmetal_gq4_view(c->device, c->p_s + e * c->s_stride, sbytes, &ls);
            bb = llmmetal_gq4_view(c->device, c->p_b + e * c->s_stride, sbytes, &lb);
            if (!wb || !sb || !bb) { [enc endEncoding]; return -1; }
        } else { wb = c->w_buf; sb = c->scale_buf; bb = c->bias_buf; lw = c->lead_w; ls = c->lead_s; lb = c->lead_b; }
        id<MTLBuffer> xb = [c->device newBufferWithBytes:jobs[i].x length:(size_t)c->H * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [c->device newBufferWithLength:(size_t)c->M * 2 options:MTLResourceStorageModeShared];
        if (!xb || !ob) { [enc endEncoding]; return -1; }
        [keep addObject:wb]; [keep addObject:sb]; [keep addObject:bb]; [keep addObject:xb]; [outs addObject:ob];
        struct { uint32_t H, M, group_size, rows_per_tg, lead_w, lead_s, lead_b, scale_bf16; } params = {
            c->H, c->M, c->group_size, LLMMETAL_GEMV_ROWS_PER_TG, (uint32_t)lw, (uint32_t)ls, (uint32_t)lb, c->scale_bf16 };
        [enc setBuffer:xb offset:0 atIndex:0]; [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:sb offset:0 atIndex:2]; [enc setBuffer:bb offset:0 atIndex:3];
        [enc setBuffer:ob offset:0 atIndex:4];
        [enc setBytes:&params length:sizeof(params) atIndex:5];
        [enc setThreadgroupMemoryLength:(c->H * sizeof(uint16_t)) atIndex:0];
        const uint32_t n_tg = (c->M + LLMMETAL_GEMV_ROWS_PER_TG - 1) / LLMMETAL_GEMV_ROWS_PER_TG;
        [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV_ROWS_PER_TG*32,1,1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    for (int i = 0; i < n; i++) memcpy(jobs[i].out, ((id<MTLBuffer>)outs[(NSUInteger)i]).contents, (size_t)jobs[i].ctx->M * 2);
    return 0;
}

#endif /* LLMMETAL_METAL_GEMV_Q4_H */
