/* llmmetal_gemv_mxfp4.h — GEMV decode MXFP4-native, kernel Metal custom (kernel #36).

 * Perché: il decode gpt-oss (esperti in MXFP4 32/4) è il più grande gap di velocità rimasto
 * nella matrice §27 (~4,1x vs rapidMLX sul 120B). Il path attuale passa da
 * mlx_quantized_matmul(mode="mxfp4") che per una GEMV a singolo token è banda-limited; questo
 * modulo implementa una GEMV Metal dedicata MXFP4 (dequant E2M1 inline + scala E8M0), stessa
 * famiglia di design di llmmetal_gemv_q4.h (affine), ma per il formato a blocchi MX.

 * Formato MXFP4 (VERIFICATO bit-esatto contro i sorgenti MLX, NON autoconsistente):
 *   - pesi:  [M, H/2] uint8, 2 nibble/byte — nibble BASSO = colonna pari, ALTO = dispari
 *            (identico packing dell'affine; vedi mlx fp_quantized.h qdot: ws[i],ws[i]>>4,...)
 *   - valore = e8m0(scale_byte) * e2m1(nibble)   → NESSUN bias (a differenza dell'affine)
 *   - scale: [M, H/group_size] uint8, group_size=32, codifica FP8-E8M0 (solo esponente)
 *   - e2m1 (nibble&7) -> {0,.5,1,1.5,2,3,4,6}, segno dal bit 3. Riprodotto col medesimo
 *     half-trick di MLX (mlx fp4.h): as_type<half>((nib&7)<<9) * 16384, poi negato se nib&8.
 *   - e8m0 (byte) -> MLX fp8.h operator float: as_type<float>(byte==0 ? 0x400000 : byte<<23).
 *     Copiata ESATTAMENTE la stessa espressione (incl. il caso byte==0) per parità bit.

 * Vincoli v1: H multiplo di 64 (coalescenza coppie di nibble + group_size=32), group_size=32,
 * decode a 1 token (x = [H], non GEMM multi-token). Fallback sempre sicuro lato chiamante.

 * Design: un simdgroup (32 lane) per riga di W; ROWS_PER_TG simdgroup per threadgroup
 * condividono lo staging di x in threadgroup memory. Con h = i*32 + lane e group_size=32, a una
 * data iterazione i tutti i 32 lane cadono nello STESSO blocco di scala (g = i), quindi la
 * lettura della scala è un broadcast — niente 32 accessi sparsi (lezione del fix di banda
 * §"Bug di prestazioni" del kernel affine).
 */
#ifndef LLMMETAL_METAL_GEMV_MXFP4_H
#define LLMMETAL_METAL_GEMV_MXFP4_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LLMMETAL_GEMVMX4_ROWS_PER_TG 16
/* #47 v4 column-parallel: warp (simdgroup) che cooperano su UNA riga. Il kernel ha un
 * `constexpr uint WPB` che DEVE combaciare con questo valore host. */
#define LLMMETAL_GEMV4_WARPS_PER_ROW 8

static const char llmmetal_gemv_mxfp4_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Mx4Params { uint H; uint M; uint group_size; uint rows_per_tg; };\n"
"\n"
// E8M0 -> float: identico a mlx fp8.h operator float (caso byte==0 incluso).
"static inline float e8m0_f(uint8_t b) {\n"
"    uint s = (b == 0u) ? 0x400000u : (uint(b) << 23);\n"
"    return as_type<float>(s);\n"
"}\n"
// E2M1 (nibble basso a 3 bit) -> float: identico a mlx fp4.h (half-trick + segno dal bit 3).
"static inline float e2m1_f(uint8_t nib) {\n"
"    half v = as_type<half>(ushort((nib & 7u) << 9)) * 16384.0h;\n"
"    return (nib & 8u) ? -float(v) : float(v);\n"
"}\n"
"\n"
"kernel void gemv_mxfp4(\n"
"    const device half*   x       [[buffer(0)]],\n"
"    const device uchar*  Wq      [[buffer(1)]],\n"
"    const device uchar*  scales  [[buffer(2)]],\n"
"    device half*         out     [[buffer(3)]],\n"
"    constant Mx4Params&  p       [[buffer(4)]],\n"
"    threadgroup half*    x_shared [[threadgroup(0)]],\n"
"    uint3 tgid    [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3   [[thread_position_in_threadgroup]],\n"
"    uint  lane    [[thread_index_in_simdgroup]],\n"
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
"    const device uchar* row_scales = scales + (size_t)row * n_groups;\n"
"\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < per_lane; i++) {\n"
"        uint h = i * 32 + lane;\n"
"        uchar b = row_packed[h >> 1];\n"
"        uint nib = (h & 1u) ? (b >> 4) : (b & 0xFu);\n"
"        uint g = h / p.group_size;            // group_size=32 => g==i, scala in broadcast\n"
"        float w = e8m0_f(row_scales[g]) * e2m1_f((uchar)nib);\n"
"        acc += w * float(x_shared[h]);\n"
"    }\n"
"    float sum = simd_sum(acc);\n"
"    if (lane == 0) out[row] = (half)sum;\n"
"}\n"
;

/* ── v2: GEMV MXFP4 bandwidth-ottimizzato (task #47). Stessa struttura di v1 (un simdgroup =
 * una riga di W, ROWS_PER_TG simdgroup/TG condividono lo staging di x), MA i caricamenti dei
 * pesi passano da SCALARI `uchar` (v1: 32 lane x 1 byte = 32 byte/transaction/warp -> 1/4
 * dell'occupazione) a `uint` a 32 BIT (32 lane x 4 byte = 128 byte/warp = il minimo per usare
 * tutta la banda). Un uint = 4 byte = 8 nibble = 8 colonne; group_size=32 copre ESATTAMENTE 4
 * uint consecutivi, quindi la scala della parola w e' row_scales[w>>2]. Dequant con le STESSE
 * espressioni e8m0_f/e2m1_f (parita' bit con MLX), estrazione nibble via shift (niente cast di
 * puntatori device). Riduzione identica (simd_sum).
 *
 * MISURA (soffitto asintotico, M=92160 => 141 MB pesi, window libera ~85 GB): v1 78-84 GB/s ->
 * v2 242-290 GB/s = ~3,3x; correttezza byte-identica (max_abs_diff 0,06184 = stesso v1, 0 righe
 * fuori tol). Sui load piccoli (2880x2880, 4,4 MB) domina il pavimento di launch/sync (~18-24
 * GB/s) e il guadagno si vede solo sul passo grande (streaming). v2 = DESIGN ADOTTATO. Cablaggio
 * nel motore (intercept esperti gpt-oss al posto di gather_qmm) = passo separato, default OFF. ── */
static const char llmmetal_gemv_mxfp4_v2_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Mx4Params { uint H; uint M; uint group_size; uint rows_per_tg; };\n"
"\n"
"static inline float e8m0_f(uint8_t b) {\n"
"    uint s = (b == 0u) ? 0x400000u : (uint(b) << 23);\n"
"    return as_type<float>(s);\n"
"}\n"
"static inline float e2m1_f(uint8_t nib) {\n"
"    half v = as_type<half>(ushort((nib & 7u) << 9)) * 16384.0h;\n"
"    return (nib & 8u) ? -float(v) : float(v);\n"
"}\n"
"\n"
"kernel void gemv_mxfp4_v2(\n"
"    const device half*   x        [[buffer(0)]],\n"
"    const device uchar*  Wq       [[buffer(1)]],\n"
"    const device uchar*  scales   [[buffer(2)]],\n"
"    device half*         out      [[buffer(3)]],\n"
"    constant Mx4Params&  p        [[buffer(4)]],\n"
"    threadgroup half*    x_shared [[threadgroup(0)]],\n"
"    uint3 tgid    [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3   [[thread_position_in_threadgroup]],\n"
"    uint  lane    [[thread_index_in_simdgroup]],\n"
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
"    // parole da 4 byte (8 colonne) in una riga: (H/2 bytes)/4 = H/8. 4 parole == un gruppo 32.\n"
"    const uint words = p.H >> 3;\n"
"    const device uint* row_w = (const device uint*)(Wq + (size_t)row * (p.H / 2));\n"
"    const device uchar* row_scales = scales + (size_t)row * (p.H / p.group_size);\n"
"\n"
"    float acc = 0.0f;\n"
"    for (uint w = lane; w < words; w += 32u) {\n"
"        uint word = row_w[w];\n"
"        float sc = e8m0_f(row_scales[w >> 2]);   // group_size=32 => 4 parole/gruppo\n"
"        const uint cb = w << 3;                   // colonna base (8w)\n"
"        #pragma unroll\n"
"        for (uint j = 0u; j < 4u; j++) {\n"
"            uint b = (word >> (8u * j)) & 0xFFu;  // byte j = colonne 8w+2j (basso), 8w+2j+1 (alto)\n"
"            float lo = e2m1_f((uint8_t)(b & 0xFu));\n"
"            float hi = e2m1_f((uint8_t)(b >> 4));\n"
"            acc += sc * (lo * float(x_shared[cb + 2u*j]) + hi * float(x_shared[cb + 2u*j + 1u]));\n"
"        }\n"
"    }\n"
"    float sum = simd_sum(acc);\n"
"    if (lane == 0) out[row] = (half)sum;\n"
"}\n"
;

/* ── v3: come v2 ma con caricamenti `uint4` (16 byte/lane = 32 nibble = ESATTAMENTE un gruppo
 * group_size=32). Un warp muove 32x16 = 512 byte per transazione -> piu' MLP di v2 (128 B).
 * Unpack SENZA cast di puntatori device: il uint4 ha 4 componenti uint, ciascuna 4 byte = 8
 * colonne, e si riusa la stessa estrazione a shift di v2. Scala row_scales[g] in broadcast.
 *
 * MISURA = DEAD END: non batte mai v2 (asintotico: rep1 0,70x, rep2 0,98x vs v2). I 128 byte/warp
 * di v2 saturano gia' la banda raggiungibile; allargare a 512 B/warp aggiunge solo pressione su
 * registro/issue senza guadagno di streaming. Mantenuta SOLO come comparatore A/B nel selfcheck
 * (documenta il risultato negativo). v2 resta il design adottato. ── */
static const char llmmetal_gemv_mxfp4_v3_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Mx4Params { uint H; uint M; uint group_size; uint rows_per_tg; };\n"
"\n"
"static inline float e8m0_f(uint8_t b) {\n"
"    uint s = (b == 0u) ? 0x400000u : (uint(b) << 23);\n"
"    return as_type<float>(s);\n"
"}\n"
"static inline float e2m1_f(uint8_t nib) {\n"
"    half v = as_type<half>(ushort((nib & 7u) << 9)) * 16384.0h;\n"
"    return (nib & 8u) ? -float(v) : float(v);\n"
"}\n"
"\n"
"static inline float dot_group(uint word, uint colbase, const threadgroup half* x_shared) {\n"
"    float s = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint j = 0u; j < 4u; j++) {\n"
"        uint b = (word >> (8u * j)) & 0xFFu;\n"
"        float lo = e2m1_f((uint8_t)(b & 0xFu));\n"
"        float hi = e2m1_f((uint8_t)(b >> 4));\n"
"        s += lo * float(x_shared[colbase + 2u*j]) + hi * float(x_shared[colbase + 2u*j + 1u]);\n"
"    }\n"
"    return s;\n"
"}\n"
"\n"
"kernel void gemv_mxfp4_v3(\n"
"    const device half*   x        [[buffer(0)]],\n"
"    const device uchar*  Wq       [[buffer(1)]],\n"
"    const device uchar*  scales   [[buffer(2)]],\n"
"    device half*         out      [[buffer(3)]],\n"
"    constant Mx4Params&  p        [[buffer(4)]],\n"
"    threadgroup half*    x_shared [[threadgroup(0)]],\n"
"    uint3 tgid    [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3   [[thread_position_in_threadgroup]],\n"
"    uint  lane    [[thread_index_in_simdgroup]],\n"
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
"    const uint n_groups = p.H / p.group_size;   // un uint4 == un gruppo (16 B = 32 nibble)\n"
"    const device uint4* w16 = (const device uint4*)(Wq + (size_t)row * (p.H / 2));\n"
"    const device uchar* row_scales = scales + (size_t)row * n_groups;\n"
"\n"
"    float acc = 0.0f;\n"
"    for (uint g = lane; g < n_groups; g += 32u) {\n"
"        uint4 q = w16[g];\n"
"        float sc = e8m0_f(row_scales[g]);\n"
"        const uint cb = g * p.group_size;   // colonna base del gruppo (32g)\n"
"        float d = dot_group(q.x, cb, x_shared) + dot_group(q.y, cb + 8u, x_shared)\n"
"                + dot_group(q.z, cb + 16u, x_shared) + dot_group(q.w, cb + 24u, x_shared);\n"
"        acc += sc * d;\n"
"    }\n"
"    float sum = simd_sum(acc);\n"
"    if (lane == 0) out[row] = (half)sum;\n"
"}\n"
;

/* ── v4: GEMV MXFP4 COLUMN-PARALLEL (task #47, la leva dopo la parità di v2). Diagnosi §J: a T=1
 * la GEMV simdgroup-per-riga (v1/v2/v3/fuso #46) è latency/occupancy-bound, non load-bound — una
 * riga è letta da UN solo warp (32 lane), poche richieste in volo, e la banda NON arriva al picco.
 * v4 dà WPB warp (256 thread) a OGNI riga: un threadgroup == una riga di output, le colonne sono
 * spaccate tra i 256 thread (stride NT), 8x piu' richieste di load in volo per riga -> nasconde la
 * latenza di streaming. Dequant identico a v2 (load uint, scala row_scales[w>>2], e8m0*e2m1).
 * Riduzione: simd_sum per warp -> red[WPB] in threadgroup -> warp 0 somma le WPB parziali.
 *
 * MISURA = DEAD END (ipotesi FALSIFICATA): v4 e' PIU' LENTO di v2 a OGNI forma decode (K=1 0,88x;
 * K=16 0,44x = 91 vs 210 GB/s pulito; asintoto 0,37x). Dare 8 warp a una riga da 360 parole lascia
 * ~1,4 iter/thread e i due threadgroup_barrier costano piu' di quanto nascondano la latenza: la
 * GEMV mxfp4 a T=1 NON e' occupancy-starved nel senso "piu' warp per riga". Con v3 (uint4) questo
 * chiude lo spazio di ristrutturazione della GEMV simdgroup: v2 e' il soffitto raggiungibile (=
 * parita' gather_qmm, vedi §J). Mantenuto come comparatore/documentazione del negativo. ── */
static const char llmmetal_gemv_mxfp4_v4_src[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Mx4Params { uint H; uint M; uint group_size; uint rows_per_tg; };\n"
"\n"
"static inline float e8m0_f(uint8_t b) {\n"
"    uint s = (b == 0u) ? 0x400000u : (uint(b) << 23);\n"
"    return as_type<float>(s);\n"
"}\n"
"static inline float e2m1_f(uint8_t nib) {\n"
"    half v = as_type<half>(ushort((nib & 7u) << 9)) * 16384.0h;\n"
"    return (nib & 8u) ? -float(v) : float(v);\n"
"}\n"
"\n"
"kernel void gemv_mxfp4_v4(\n"
"    const device half*   x        [[buffer(0)]],\n"
"    const device uchar*  Wq       [[buffer(1)]],\n"
"    const device uchar*  scales   [[buffer(2)]],\n"
"    device half*         out      [[buffer(3)]],\n"
"    constant Mx4Params&  p        [[buffer(4)]],\n"
"    threadgroup half*    x_shared [[threadgroup(0)]],\n"
"    uint3 tgid    [[threadgroup_position_in_grid]],\n"
"    uint3 ltid3   [[thread_position_in_threadgroup]],\n"
"    uint  lane    [[thread_index_in_simdgroup]],\n"
"    uint  simd_id [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    constexpr uint WPB = 8u;              // warp per riga (deve = LLMMETAL_GEMV4_WARPS_PER_ROW)\n"
"    constexpr uint NT  = WPB * 32u;       // 256 thread = un threadgroup = una riga\n"
"    const uint tid = ltid3.x;\n"
"    const uint row = tgid.x;\n"
"    if (row >= p.M) return;\n"
"\n"
"    for (uint i = tid; i < p.H; i += NT) x_shared[i] = x[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    const uint words = p.H >> 3;\n"
"    const device uint* row_w = (const device uint*)(Wq + (size_t)row * (p.H / 2));\n"
"    const device uchar* row_scales = scales + (size_t)row * (p.H / p.group_size);\n"
"\n"
"    float acc = 0.0f;\n"
"    for (uint w = tid; w < words; w += NT) {\n"
"        uint word = row_w[w];\n"
"        float sc = e8m0_f(row_scales[w >> 2]);\n"
"        const uint cb = w << 3;\n"
"        #pragma unroll\n"
"        for (uint j = 0u; j < 4u; j++) {\n"
"            uint b = (word >> (8u * j)) & 0xFFu;\n"
"            float lo = e2m1_f((uint8_t)(b & 0xFu));\n"
"            float hi = e2m1_f((uint8_t)(b >> 4));\n"
"            acc += sc * (lo * float(x_shared[cb + 2u*j]) + hi * float(x_shared[cb + 2u*j + 1u]));\n"
"        }\n"
"    }\n"
"    float part = simd_sum(acc);\n"
"    threadgroup float red[WPB];\n"
"    if (lane == 0u) red[simd_id] = part;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (simd_id == 0u) {\n"
"        float v = (lane < WPB) ? red[lane] : 0.0f;\n"
"        v = simd_sum(v);\n"
"        if (lane == 0u) out[row] = (half)v;\n"
"    }\n"
"}\n"
;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso;
    id<MTLComputePipelineState> pso_v2;   /* #47: GEMV bandwidth-ottimizzato (uint load) */
    id<MTLComputePipelineState> pso_v3;   /* #47: come v2 ma uint4 load (512 B/warp) */
    id<MTLComputePipelineState> pso_v4;   /* #47: column-parallel multi-warp-per-row (occupancy a T=1) */
    /* pesi persistenti — caricati UNA VOLTA da llmmetal_gemv_mxfp4_load_weights, riusati per
     * ogni token di decode (come il kernel affine). */
    id<MTLBuffer> w_buf, scale_buf, out_buf, x_buf;
    uint32_t H, M, group_size;
} LlmMetalGemvMxfp4Ctx;

static int llmmetal_gemv_mxfp4_init(LlmMetalGemvMxfp4Ctx* ctx) {
    ctx->device = nil; ctx->queue = nil; ctx->library = nil; ctx->pso = nil; ctx->pso_v2 = nil; ctx->pso_v3 = nil; ctx->pso_v4 = nil;
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: nessun device Metal\n"); return -1; }
    ctx->queue = [ctx->device newCommandQueue];
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:llmmetal_gemv_mxfp4_src];
    ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&err];
    if (!ctx->library) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: compile err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> f = [ctx->library newFunctionWithName:@"gemv_mxfp4"];
    ctx->pso = [ctx->device newComputePipelineStateWithFunction:f error:&err];
    if (!ctx->pso) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: pso err: %s\n", err.localizedDescription.UTF8String); return -1; }
    /* #47: kernel v2 (bandwidth) in una libreria separata, funzione gemv_mxfp4_v2. */
    NSString* src2 = [NSString stringWithUTF8String:llmmetal_gemv_mxfp4_v2_src];
    id<MTLLibrary> lib2 = [ctx->device newLibraryWithSource:src2 options:nil error:&err];
    if (!lib2) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: compile v2 err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> f2 = [lib2 newFunctionWithName:@"gemv_mxfp4_v2"];
    ctx->pso_v2 = [ctx->device newComputePipelineStateWithFunction:f2 error:&err];
    if (!ctx->pso_v2) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: pso_v2 err: %s\n", err.localizedDescription.UTF8String); return -1; }
    /* #47: kernel v3 (uint4 load) in una terza libreria. */
    NSString* src3 = [NSString stringWithUTF8String:llmmetal_gemv_mxfp4_v3_src];
    id<MTLLibrary> lib3 = [ctx->device newLibraryWithSource:src3 options:nil error:&err];
    if (!lib3) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: compile v3 err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> f3 = [lib3 newFunctionWithName:@"gemv_mxfp4_v3"];
    ctx->pso_v3 = [ctx->device newComputePipelineStateWithFunction:f3 error:&err];
    if (!ctx->pso_v3) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: pso_v3 err: %s\n", err.localizedDescription.UTF8String); return -1; }
    /* #47: kernel v4 (column-parallel multi-warp-per-row) in una quarta libreria. */
    NSString* src4 = [NSString stringWithUTF8String:llmmetal_gemv_mxfp4_v4_src];
    id<MTLLibrary> lib4 = [ctx->device newLibraryWithSource:src4 options:nil error:&err];
    if (!lib4) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: compile v4 err: %s\n", err.localizedDescription.UTF8String); return -1; }
    id<MTLFunction> f4 = [lib4 newFunctionWithName:@"gemv_mxfp4_v4"];
    ctx->pso_v4 = [ctx->device newComputePipelineStateWithFunction:f4 error:&err];
    if (!ctx->pso_v4) { fprintf(stderr, "llmmetal_gemv_mxfp4_init: pso_v4 err: %s\n", err.localizedDescription.UTF8String); return -1; }
    return 0;
}

static void llmmetal_gemv_mxfp4_free(LlmMetalGemvMxfp4Ctx* ctx) {
    ctx->w_buf = nil; ctx->scale_buf = nil; ctx->out_buf = nil; ctx->x_buf = nil;
    ctx->pso = nil; ctx->pso_v2 = nil; ctx->pso_v3 = nil; ctx->pso_v4 = nil; ctx->library = nil; ctx->queue = nil; ctx->device = nil;
}

/* Carica i pesi MXFP4 UNA VOLTA: Wq [M,H/2] uint8 (nibble-packed), scales [M,H/group_size]
 * uint8 (e8m0). group_size deve essere 32 (MXFP4). Alloca anche i buffer persistenti x/out. */
static int llmmetal_gemv_mxfp4_load_weights(LlmMetalGemvMxfp4Ctx* ctx, const uint8_t* Wq,
                                              const uint8_t* scales,
                                              uint32_t H, uint32_t M, uint32_t group_size) {
    if (H % 64 != 0) { fprintf(stderr, "llmmetal_gemv_mxfp4_load_weights: H %u non multiplo di 64\n", H); return -1; }
    if (group_size == 0 || H % group_size != 0) { fprintf(stderr, "llmmetal_gemv_mxfp4_load_weights: group_size %u non compatibile con H %u\n", group_size, H); return -1; }
    ctx->H = H; ctx->M = M; ctx->group_size = group_size;
    uint32_t n_groups = H / group_size;
    ctx->w_buf = [ctx->device newBufferWithBytes:Wq length:(size_t)M*(H/2) options:MTLResourceStorageModeShared];
    ctx->scale_buf = [ctx->device newBufferWithBytes:scales length:(size_t)M*n_groups*sizeof(uint8_t) options:MTLResourceStorageModeShared];
    ctx->out_buf = [ctx->device newBufferWithLength:(size_t)M*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    ctx->x_buf = [ctx->device newBufferWithLength:(size_t)H*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    if (!ctx->w_buf || !ctx->scale_buf || !ctx->out_buf || !ctx->x_buf) {
        fprintf(stderr, "llmmetal_gemv_mxfp4_load_weights: allocazione fallita\n"); return -1;
    }
    return 0;
}

/* Un passo di decode: x [H] half, out [M] half. Pesi già residenti, nessuna copia dei pesi. */
static int llmmetal_gemv_mxfp4_run(LlmMetalGemvMxfp4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg; } params = { ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMVMX4_ROWS_PER_TG };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->w_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->scale_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    [enc setThreadgroupMemoryLength:(ctx->H*sizeof(uint16_t)) atIndex:0];
    uint32_t n_tg = (ctx->M + LLMMETAL_GEMVMX4_ROWS_PER_TG - 1) / LLMMETAL_GEMVMX4_ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMVMX4_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->M*sizeof(uint16_t));
    return 0;
}

/* #47: variante v2 (uint load) dello stesso passo di decode. Buffer e dispatch identici, solo
 * il pipeline e' pso_v2. Usata per A/B nel selfcheck; cablaggio nel motore = passo separato. */
static int llmmetal_gemv_mxfp4_run_v2(LlmMetalGemvMxfp4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg; } params = { ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMVMX4_ROWS_PER_TG };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso_v2];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->w_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->scale_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    [enc setThreadgroupMemoryLength:(ctx->H*sizeof(uint16_t)) atIndex:0];
    uint32_t n_tg = (ctx->M + LLMMETAL_GEMVMX4_ROWS_PER_TG - 1) / LLMMETAL_GEMVMX4_ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMVMX4_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->M*sizeof(uint16_t));
    return 0;
}
/* #47: variante v3 (uint4 load, 512 B/warp) dello stesso passo. Solo il pipeline e' pso_v3. */
static int llmmetal_gemv_mxfp4_run_v3(LlmMetalGemvMxfp4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg; } params = { ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMVMX4_ROWS_PER_TG };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso_v3];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->w_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->scale_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    [enc setThreadgroupMemoryLength:(ctx->H*sizeof(uint16_t)) atIndex:0];
    uint32_t n_tg = (ctx->M + LLMMETAL_GEMVMX4_ROWS_PER_TG - 1) / LLMMETAL_GEMVMX4_ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMVMX4_ROWS_PER_TG*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->M*sizeof(uint16_t));
    return 0;
}
/* #47: variante v4 column-parallel. Dispatch DIVERSO: un threadgroup per riga (n_tg=M), e
 * threadsPerThreadgroup = WPB*32 (256), non ROWS_PER_TG*32. Stessi buffer/pesi/parametri. */
static int llmmetal_gemv_mxfp4_run_v4(LlmMetalGemvMxfp4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    memcpy(ctx->x_buf.contents, x, (size_t)ctx->H*sizeof(uint16_t));
    struct { uint32_t H, M, group_size, rows_per_tg; } params = { ctx->H, ctx->M, ctx->group_size, LLMMETAL_GEMV4_WARPS_PER_ROW };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->pso_v4];
    [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->w_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->scale_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    [enc setThreadgroupMemoryLength:(ctx->H*sizeof(uint16_t)) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(ctx->M,1,1) threadsPerThreadgroup:MTLSizeMake(LLMMETAL_GEMV4_WARPS_PER_ROW*32,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, ctx->out_buf.contents, (size_t)ctx->M*sizeof(uint16_t));
    return 0;
}

#endif /* LLMMETAL_METAL_GEMV_MXFP4_H */
