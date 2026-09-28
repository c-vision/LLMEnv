/* llmmetal_kv.h — KV cache Metal reale: pagine fisiche + scrittura in-place + decode
 * attention "paged" con online softmax split-K (stile FlashAttention-decode / Flash-Decoding).
 *
 * Implementa la parte "Via B" del piano
 * (~/Documents/AiWiki/01-Projects/LLMEnv/Piani/Proposta-614GBs-Banda-Metal-DeepDive.md, §4 L4):
 * la KV cache vive in un MTLBuffer nostro (MTLResourceStorageModeShared, memoria unificata,
 * letto/scritto da CPU e GPU senza copia), FUORI dal grafo MLX. Solo il vettore di OUTPUT
 * dell'attenzione (piccolo: [Hq,D] per token, non l'intera cache) deve poi rientrare nel grafo
 * MLX come mlx_array normale (via mlx_array_new_data, che copia — accettabile perché e' un
 * vettore di poche centinaia di float, non l'intera KV).
 *
 * Risposta alla domanda aperta del piano ("MLX supporta il wrapping zero-copy di un MTLBuffer
 * esterno?"): verificato leggendo mlx-c (mlx/c/array.h) e mlx C++ (mlx/backend/metal/allocator.h)
 * — mlx-c espone SOLO costruttori che copiano (mlx_array_new_data/_managed/_managed_payload,
 * documentati "will be copied"); l'allocatore Metal interno di MLX HA un metodo
 * MetalAllocator::make_buffer(ptr,size) che chiama device_->newBuffer(ptr,size,opts,nullptr)
 * (= newBufferWithBytesNoCopy, zero-copy reale), ma NON e' esposto dalla ABI C che LLMEnv
 * collega. Decisione: non patchare mlx-c per esporlo (accoppierebbe ogni bump di versione a
 * una patch aggiuntiva) — la KV vive interamente fuori da MLX in un MTLDevice/MTLBuffer
 * gestito qui; solo il risultato finale (piccolo) dell'attention rientra nel grafo MLX con una
 * copia, che e' O(1)/token, non O(capacita') come il buffer statico attuale (zeri+concat+where).
 *
 * Layout fisico: K e V separati, ciascuno [n_pages, page_tokens, n_kv_heads, head_dim] fp16,
 * row-major, pagina = blocco contiguo (nessuna view strided). GQA: n_q_heads deve essere
 * multiplo di n_kv_heads.
 *
 * Scrittura (kv_write): memcpy CPU diretto in MTLResourceStorageModeShared, NIENTE dispatch
 * GPU. Prima versione usava un kernel Metal per-token dispatchato via command buffer sincrono:
 * ~6700 token/s, dominato dal costo fisso di submit+wait di un command buffer (микроsecondi)
 * molto piu' grande dei ~4KB reali da scrivere. Su memoria UNIFICATA un memcpy CPU nel buffer
 * condiviso e' visibile alla GPU immediatamente (nessun DMA/staging separato) — e' la scelta
 * corretta per una scrittura di questa dimensione (O(1) elementi/token), non un compromesso.
 *
 * Decode attention: split-K ("Flash-Decoding") in due kernel. decode_attention_partial
 * partiziona le pagine di un head fra N_SPLITS threadgroup indipendenti (ognuno online-softmax
 * sul proprio pezzo), decode_attention_merge combina gli N_SPLITS stati parziali (m,l,acc) nel
 * risultato finale con la regola di merge dell'online softmax. Necessario perche' un solo
 * threadgroup per head (prima versione) usa solo Hq*32 thread totali — enormemente sotto-
 * occupato su un M5 Max; con split-K la GPU ha Hq*N_SPLITS*32 thread indipendenti.
 */
#ifndef LLMMETAL_METAL_KV_H
#define LLMMETAL_METAL_KV_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LLMMETAL_KV_MAX_SPLITS 256

static const char llmmetal_kv_metal_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct AttnParams {\n"
"    uint n_q_heads;\n"
"    uint n_kv_heads;\n"
"    uint head_dim;       // D, multiplo di 32\n"
"    uint page_tokens;\n"
"    uint n_pages_used;\n"
"    uint last_page_len;  // token validi nell'ultima pagina della page_table\n"
"    uint n_splits;\n"
"    uint pages_per_split;\n"
"    float scale;\n"
"};\n"
"\n"
"// Passata 1 (split-K): un threadgroup per (head, split). Ogni threadgroup fa online softmax\n"
"// SOLO sulle pagine del proprio split, scrive lo stato parziale (m,l,acc[D]) — non il\n"
"// risultato finale. Grid: (n_q_heads, n_splits), 32 thread (1 simdgroup) per threadgroup.\n"
"kernel void decode_attention_partial(\n"
"    const device half* q [[buffer(0)]],           // [Hq, D]\n"
"    const device half* k_pool [[buffer(1)]],       // [n_pages, page_tokens, Hkv, D]\n"
"    const device half* v_pool [[buffer(2)]],       // [n_pages, page_tokens, Hkv, D]\n"
"    const device int*  page_table [[buffer(3)]],   // [n_pages_used] id pagina fisica\n"
"    constant AttnParams& p [[buffer(4)]],\n"
"    device float* part_m [[buffer(5)]],             // [Hq, n_splits]\n"
"    device float* part_l [[buffer(6)]],             // [Hq, n_splits]\n"
"    device float* part_acc [[buffer(7)]],           // [Hq, n_splits, D]\n"
"    uint3 tgid [[threadgroup_position_in_grid]],\n"
"    uint  lane [[thread_index_in_simdgroup]])\n"
"{\n"
"    const uint h = tgid.x;\n"
"    const uint split = tgid.y;\n"
"    const uint group = p.n_q_heads / p.n_kv_heads;\n"
"    const uint hk = h / group;\n"
"    const uint D = p.head_dim;\n"
"    const uint per_lane = D / 32;\n"
"\n"
"    const uint page_lo = split * p.pages_per_split;\n"
"    const uint page_hi = min(page_lo + p.pages_per_split, p.n_pages_used);\n"
"    if (page_lo >= page_hi) {\n"
"        // split senza pagine assegnate (n_pages_used non multiplo esatto): stato neutro\n"
"        if (lane == 0) { part_m[h*p.n_splits+split] = -INFINITY; part_l[h*p.n_splits+split] = 0.0f; }\n"
"        for (uint i = 0; i < per_lane; i++) part_acc[(h*p.n_splits+split)*D + lane*per_lane + i] = 0.0f;\n"
"        return;\n"
"    }\n"
"\n"
"    float qf[8];\n"
"    for (uint i = 0; i < per_lane; i++) qf[i] = (float)q[(size_t)h*D + lane*per_lane + i];\n"
"\n"
"    float m = -INFINITY, l = 0.0f;\n"
"    float acc[8];\n"
"    for (uint i = 0; i < per_lane; i++) acc[i] = 0.0f;\n"
"\n"
"    const size_t page_stride = (size_t)p.page_tokens * p.n_kv_heads * D;\n"
"    const size_t head_off = (size_t)hk * D;\n"
"    const size_t tok_stride = (size_t)p.n_kv_heads * D;\n"
"\n"
"    for (uint pi = page_lo; pi < page_hi; pi++) {\n"
"        const int page = page_table[pi];\n"
"        const uint n_tok = (pi + 1 == p.n_pages_used) ? p.last_page_len : p.page_tokens;\n"
"        const device half* k_page = k_pool + (size_t)page * page_stride + head_off;\n"
"        const device half* v_page = v_pool + (size_t)page * page_stride + head_off;\n"
"        for (uint t = 0; t < n_tok; t++) {\n"
"            const device half* k_vec = k_page + (size_t)t * tok_stride;\n"
"            const device half* v_vec = v_page + (size_t)t * tok_stride;\n"
"            float partial = 0.0f;\n"
"            for (uint i = 0; i < per_lane; i++) partial += qf[i] * (float)k_vec[lane*per_lane + i];\n"
"            float score = simd_sum(partial) * p.scale;\n"
"            float new_m = max(m, score);\n"
"            float correction = exp(m - new_m);\n"
"            float pw = exp(score - new_m);\n"
"            l = l * correction + pw;\n"
"            for (uint i = 0; i < per_lane; i++)\n"
"                acc[i] = acc[i] * correction + pw * (float)v_vec[lane*per_lane + i];\n"
"            m = new_m;\n"
"        }\n"
"    }\n"
"    if (lane == 0) { part_m[h*p.n_splits+split] = m; part_l[h*p.n_splits+split] = l; }\n"
"    for (uint i = 0; i < per_lane; i++) part_acc[(h*p.n_splits+split)*D + lane*per_lane + i] = acc[i];\n"
"}\n"
"\n"
"// Passata 2: combina gli n_splits stati parziali di ogni head nel risultato finale, con la\n"
"// regola di merge dell'online softmax (stessa identità matematica della passata 1, applicata\n"
"// fra split invece che fra token). Grid: (n_q_heads), 32 thread per threadgroup.\n"
"kernel void decode_attention_merge(\n"
"    const device float* part_m [[buffer(0)]],\n"
"    const device float* part_l [[buffer(1)]],\n"
"    const device float* part_acc [[buffer(2)]],\n"
"    constant AttnParams& p [[buffer(3)]],\n"
"    device half* out [[buffer(4)]],                 // [Hq, D]\n"
"    uint3 tgid [[threadgroup_position_in_grid]],\n"
"    uint  lane [[thread_index_in_simdgroup]])\n"
"{\n"
"    const uint h = tgid.x;\n"
"    const uint D = p.head_dim;\n"
"    const uint per_lane = D / 32;\n"
"\n"
"    // max globale fra gli split (ogni lane lo calcola in modo ridondante: n_splits e' piccolo)\n"
"    float gm = -INFINITY;\n"
"    for (uint s = 0; s < p.n_splits; s++) gm = max(gm, part_m[h*p.n_splits+s]);\n"
"\n"
"    float l = 0.0f;\n"
"    float acc[8];\n"
"    for (uint i = 0; i < per_lane; i++) acc[i] = 0.0f;\n"
"    for (uint s = 0; s < p.n_splits; s++) {\n"
"        float ms = part_m[h*p.n_splits+s];\n"
"        float ls = part_l[h*p.n_splits+s];\n"
"        float corr = (ms == -INFINITY) ? 0.0f : exp(ms - gm);\n"
"        l += ls * corr;\n"
"        for (uint i = 0; i < per_lane; i++)\n"
"            acc[i] += part_acc[(h*p.n_splits+s)*D + lane*per_lane + i] * corr;\n"
"    }\n"
"    for (uint i = 0; i < per_lane; i++)\n"
"        out[(size_t)h*D + lane*per_lane + i] = (half)(l > 0.0f ? acc[i] / l : 0.0f);\n"
"}\n"
;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> partial_pso;
    id<MTLComputePipelineState> merge_pso;
    id<MTLBuffer> k_pool;
    id<MTLBuffer> v_pool;
    /* buffer di lavoro per il merge, pre-allocati alla dimensione massima (evita malloc nel
     * percorso caldo — coerente col principio "zero alloc nel percorso token" del piano). */
    id<MTLBuffer> part_m, part_l, part_acc, q_buf, pt_buf, out_buf;
    int n_pages, page_tokens, n_kv_heads, n_q_heads, head_dim;
} LlmMetalKvCtx;

static int llmmetal_kv_init(LlmMetalKvCtx* ctx, int n_pages, int page_tokens,
                              int n_q_heads, int n_kv_heads, int head_dim) {
    ctx->device = nil; ctx->queue = nil; ctx->library = nil;
    ctx->partial_pso = nil; ctx->merge_pso = nil;
    ctx->k_pool = nil; ctx->v_pool = nil;
    ctx->part_m = nil; ctx->part_l = nil; ctx->part_acc = nil;
    ctx->q_buf = nil; ctx->pt_buf = nil; ctx->out_buf = nil;
    if (head_dim % 32 != 0) {
        fprintf(stderr, "llmmetal_kv_init: head_dim %d non multiplo di 32 (limite v1)\n", head_dim);
        return -1;
    }
    if (n_q_heads % n_kv_heads != 0) {
        fprintf(stderr, "llmmetal_kv_init: n_q_heads %d non multiplo di n_kv_heads %d\n", n_q_heads, n_kv_heads);
        return -1;
    }
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { fprintf(stderr, "llmmetal_kv_init: nessun device Metal\n"); return -1; }
    ctx->queue = [ctx->device newCommandQueue];
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:llmmetal_kv_metal_src];
    ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&err];
    if (!ctx->library) {
        fprintf(stderr, "llmmetal_kv_init: compile err: %s\n", err.localizedDescription.UTF8String);
        return -1;
    }
    id<MTLFunction> pf = [ctx->library newFunctionWithName:@"decode_attention_partial"];
    id<MTLFunction> mf = [ctx->library newFunctionWithName:@"decode_attention_merge"];
    ctx->partial_pso = [ctx->device newComputePipelineStateWithFunction:pf error:&err];
    if (!ctx->partial_pso) { fprintf(stderr, "llmmetal_kv_init: partial pso err: %s\n", err.localizedDescription.UTF8String); return -1; }
    ctx->merge_pso = [ctx->device newComputePipelineStateWithFunction:mf error:&err];
    if (!ctx->merge_pso) { fprintf(stderr, "llmmetal_kv_init: merge pso err: %s\n", err.localizedDescription.UTF8String); return -1; }

    ctx->n_pages = n_pages; ctx->page_tokens = page_tokens;
    ctx->n_q_heads = n_q_heads; ctx->n_kv_heads = n_kv_heads; ctx->head_dim = head_dim;
    size_t pool_bytes = (size_t)n_pages * page_tokens * n_kv_heads * head_dim * sizeof(uint16_t);
    ctx->k_pool = [ctx->device newBufferWithLength:pool_bytes options:MTLResourceStorageModeShared];
    ctx->v_pool = [ctx->device newBufferWithLength:pool_bytes options:MTLResourceStorageModeShared];
    if (!ctx->k_pool || !ctx->v_pool) { fprintf(stderr, "llmmetal_kv_init: alloc pool fallita (%zu byte ciascuno)\n", pool_bytes); return -1; }

    size_t ms = (size_t)n_q_heads * LLMMETAL_KV_MAX_SPLITS;
    ctx->part_m = [ctx->device newBufferWithLength:ms*sizeof(float) options:MTLResourceStorageModeShared];
    ctx->part_l = [ctx->device newBufferWithLength:ms*sizeof(float) options:MTLResourceStorageModeShared];
    ctx->part_acc = [ctx->device newBufferWithLength:ms*head_dim*sizeof(float) options:MTLResourceStorageModeShared];
    ctx->q_buf = [ctx->device newBufferWithLength:(size_t)n_q_heads*head_dim*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->pt_buf = [ctx->device newBufferWithLength:(size_t)n_pages*sizeof(int) options:MTLResourceStorageModeShared];
    ctx->out_buf = [ctx->device newBufferWithLength:(size_t)n_q_heads*head_dim*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    return 0;
}

static void llmmetal_kv_free(LlmMetalKvCtx* ctx) {
    ctx->k_pool = nil; ctx->v_pool = nil;
    ctx->part_m = nil; ctx->part_l = nil; ctx->part_acc = nil;
    ctx->q_buf = nil; ctx->pt_buf = nil; ctx->out_buf = nil;
    ctx->partial_pso = nil; ctx->merge_pso = nil;
    ctx->library = nil; ctx->queue = nil; ctx->device = nil;
}

/* Scrive UN token (k_tok/v_tok: n_kv_heads*head_dim half ciascuno) nella pagina fisica page_id,
 * slot offset. memcpy diretto in memoria unificata condivisa — O(n_kv_heads*head_dim), NON
 * O(capacita'): e' la correzione diretta al bug "zeri+concat+where sull'intero buffer" del
 * motore attuale (vedi piano, §2/§4 L4). Nessun dispatch GPU, nessuna allocazione. */
static int llmmetal_kv_write(LlmMetalKvCtx* ctx, int page_id, int offset,
                               const uint16_t* k_tok, const uint16_t* v_tok) {
    if (page_id < 0 || page_id >= ctx->n_pages || offset < 0 || offset >= ctx->page_tokens) {
        fprintf(stderr, "llmmetal_kv_write: page/offset fuori range\n"); return -1;
    }
    size_t elem_bytes = (size_t)ctx->n_kv_heads * ctx->head_dim * sizeof(uint16_t);
    size_t page_stride = (size_t)ctx->page_tokens * ctx->n_kv_heads * ctx->head_dim;
    size_t base_elems = (size_t)page_id * page_stride + (size_t)offset * ctx->n_kv_heads * ctx->head_dim;
    uint16_t* kp = (uint16_t*)ctx->k_pool.contents;
    uint16_t* vp = (uint16_t*)ctx->v_pool.contents;
    memcpy(kp + base_elems, k_tok, elem_bytes);
    memcpy(vp + base_elems, v_tok, elem_bytes);
    return 0;
}

/* Decode attention paged, split-K: q [n_q_heads*head_dim] half; page_table [n_pages_used] id
 * pagine fisiche in ordine logico; last_page_len = token validi nell'ultima pagina. out
 * [n_q_heads*head_dim] half (buffer host fornito dal chiamante). Ogni K/V letto UNA sola volta
 * in totale (ripartito fra gli split), nessuna matrice score materializzata. */
static int llmmetal_kv_decode_attention(LlmMetalKvCtx* ctx, const uint16_t* q,
                                          const int* page_table, int n_pages_used, int last_page_len,
                                          float scale, uint16_t* out) {
    int n_splits = n_pages_used < LLMMETAL_KV_MAX_SPLITS ? (n_pages_used > 0 ? n_pages_used : 1) : LLMMETAL_KV_MAX_SPLITS;
    int pages_per_split = (n_pages_used + n_splits - 1) / n_splits;

    memcpy(ctx->q_buf.contents, q, (size_t)ctx->n_q_heads * ctx->head_dim * sizeof(uint16_t));
    memcpy(ctx->pt_buf.contents, page_table, (size_t)n_pages_used * sizeof(int));

    struct { uint32_t n_q_heads, n_kv_heads, head_dim, page_tokens, n_pages_used, last_page_len, n_splits, pages_per_split; float scale; } params = {
        (uint32_t)ctx->n_q_heads, (uint32_t)ctx->n_kv_heads, (uint32_t)ctx->head_dim,
        (uint32_t)ctx->page_tokens, (uint32_t)n_pages_used, (uint32_t)last_page_len,
        (uint32_t)n_splits, (uint32_t)pages_per_split, scale };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
    [e1 setComputePipelineState:ctx->partial_pso];
    [e1 setBuffer:ctx->q_buf offset:0 atIndex:0];
    [e1 setBuffer:ctx->k_pool offset:0 atIndex:1];
    [e1 setBuffer:ctx->v_pool offset:0 atIndex:2];
    [e1 setBuffer:ctx->pt_buf offset:0 atIndex:3];
    [e1 setBytes:&params length:sizeof(params) atIndex:4];
    [e1 setBuffer:ctx->part_m offset:0 atIndex:5];
    [e1 setBuffer:ctx->part_l offset:0 atIndex:6];
    [e1 setBuffer:ctx->part_acc offset:0 atIndex:7];
    [e1 dispatchThreadgroups:MTLSizeMake(ctx->n_q_heads, n_splits, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e1 endEncoding];

    id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
    [e2 setComputePipelineState:ctx->merge_pso];
    [e2 setBuffer:ctx->part_m offset:0 atIndex:0];
    [e2 setBuffer:ctx->part_l offset:0 atIndex:1];
    [e2 setBuffer:ctx->part_acc offset:0 atIndex:2];
    [e2 setBytes:&params length:sizeof(params) atIndex:3];
    [e2 setBuffer:ctx->out_buf offset:0 atIndex:4];
    [e2 dispatchThreadgroups:MTLSizeMake(ctx->n_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e2 endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->n_q_heads * ctx->head_dim * sizeof(uint16_t));
    return 0;
}

#endif /* LLMMETAL_METAL_KV_H */
