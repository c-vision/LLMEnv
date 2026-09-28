/* llmmetal_oq_a8_bridge.mm — bridge oQ A8 (int8-activation × Q4/Q5, tensor unit M5) ZERO-COPY.
 *
 * Usa l'API **C++ di MLX** per ottenere i **buffer Metal** degli array (allocator::Buffer::ptr()
 * e' il MTL::Buffer*): i kernel scrivono/leggono DIRETTAMENTE nei buffer MLX, nessuna copia
 * host<->device e nessun puntatore host. Il layout (permutazione v8 GS64, trasposizione Ra e
 * scales/biases) e' fatto con op MLX (GPU). E' l'approccio di oMLX.
 * Vedi llmmetal_oq_a8_bridge.h e Knowledge/oQ-A8-INT8-Activation-GEMM-M5-NAX.md. */
#include "llmmetal_oq_a8_bridge.h"

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/backend/metal/device.h"

#include <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace mlx::core;

namespace {
struct Ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pq;   /* Stage-A f16: oq_a8_quantize_float16_t_am0 */
    id<MTLComputePipelineState> pg;   /* GEMM f16 */
    id<MTLComputePipelineState> pq_bf; /* Stage-A bf16 (dtype del modello, come oMLX) */
    id<MTLComputePipelineState> pg_bf; /* GEMM bf16 (dtype del modello, come oMLX) */
    id<MTLComputePipelineState> pg_tm1, pg_bf_tm1; /* GEMM TM=1 (BM=16): M<=16, meta' spreco di tile */
    id<MTLComputePipelineState> pg_q8, pg_bf_q8, pg_q8_tm1, pg_bf_q8_tm1; /* Q8 (W8A8, target 8-bit) */
    /* GEMV multi-output (M<=8): [bits==8][M 1..8][0=f16 1=bf16], dal metallib support. */
    id<MTLComputePipelineState> qmv[2][9][2];
    /* ACT_MODE 1 (scale per-gruppo): Stage-A e GEMM (tm2/tm1, q4/q8, f16/bf16). */
    id<MTLComputePipelineState> pq_am1, pq_bf_am1;
    id<MTLComputePipelineState> pg_am1, pg_bf_am1, pg_tm1_am1, pg_bf_tm1_am1;
    id<MTLComputePipelineState> pg_q8_am1, pg_bf_q8_am1, pg_q8_tm1_am1, pg_bf_q8_tm1_am1;
    id<MTLComputePipelineState> pg_n1, pg_bf_n1;   /* GEMM WN=1 (BN=32): 2x blocchi a M piccolo */
    id<MTLComputePipelineState> repack, tr16, tri16, trbf;  /* support: repack v8 + trasposizioni (f16/i16/bf16) */
    bool ok = false;
};
Ctx* g_ctx = nullptr;

/* ACT_MODE 1 (scale per-GRUPPO) e' il DEFAULT (2026-09-19): stesso costo di am0 (misurato:
 * chunk 8 375 vs 371 tok/s; specbench M=7 19,8 vs 19,7 ms/round) ma outlier-robusto
 * (relL2 con outlier 0,0072 vs 0,019) e 6/7 generazioni identiche a MLX su prompt reali
 * (am0: 1/3). Opt-out: LLMMETAL_OQ_A8_AM0=1. */
static bool oq_am1_enabled(void) {
    static int s = -1;
    if (s < 0) s = getenv("LLMMETAL_OQ_A8_AM0") ? 0 : 1;
    return s == 1;
}

/* MTL::Buffer dell'array MLX (zero-copy) */
inline id<MTLBuffer> mtlb(const array& a) {
    return (__bridge id<MTLBuffer>)(const_cast<void*>(a.buffer().ptr()));
}
inline NSUInteger moff(const array& a) { return (NSUInteger)a.offset(); }

int encode_stage_a(Ctx* c, const array& x, array& qa, array& sa, array& ra, int M, int K) {
    id<MTLCommandBuffer> cb = [c->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:c->pq];
    [e setBuffer:mtlb(x)  offset:moff(x)  atIndex:0];
    [e setBuffer:mtlb(qa) offset:moff(qa) atIndex:1];
    [e setBuffer:mtlb(sa) offset:moff(sa) atIndex:2];
    [e setBuffer:mtlb(ra) offset:moff(ra) atIndex:3];
    int kk = K;
    id<MTLBuffer> bk = [c->dev newBufferWithBytes:&kk length:4 options:MTLResourceStorageModeShared];
    [e setBuffer:bk offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return 0;
}

int encode_gemm(Ctx* c, const array& qa8, const array& sa, const array& ra, const array& w,
                const array& scT, const array& biT, array& out, int M, int N, int K, int bits) {
    int argsK = K, argsN = N, argsM = M;
    id<MTLBuffer> bK = [c->dev newBufferWithBytes:&argsK length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bN = [c->dev newBufferWithBytes:&argsN length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bM = [c->dev newBufferWithBytes:&argsM length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [c->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const bool q8 = (bits == 8);
    const bool wn1 = (M <= 16) && getenv("LLMMETAL_OQ_A8_WN1") && c->pg_n1;
    const bool tm1 = (M <= 16) && !getenv("LLMMETAL_OQ_A8_NOTM1") && (q8 ? c->pg_q8_tm1 : c->pg_tm1);
    [e setComputePipelineState:q8 ? (tm1 ? c->pg_q8_tm1 : c->pg_q8)
                                  : (wn1 ? c->pg_n1 : (tm1 ? c->pg_tm1 : c->pg))];
    [e setBuffer:mtlb(qa8) offset:moff(qa8) atIndex:0];
    [e setBuffer:mtlb(sa)  offset:moff(sa)  atIndex:1];
    [e setBuffer:mtlb(ra) offset:moff(ra) atIndex:2];
    [e setBuffer:mtlb(w)   offset:moff(w)   atIndex:3];
    [e setBuffer:mtlb(scT) offset:moff(scT) atIndex:4];
    [e setBuffer:mtlb(biT) offset:moff(biT) atIndex:5];
    [e setBuffer:mtlb(out) offset:moff(out) atIndex:6];
    [e setBuffer:bK offset:0 atIndex:7]; [e setBuffer:bN offset:0 atIndex:8]; [e setBuffer:bM offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(N / (wn1 ? 32 : 64)), (NSUInteger)((M + ((wn1 || tm1) ? 15 : 31)) / ((wn1 || tm1) ? 16 : 32)), 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 2)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    (void)bits;
    return 0;
}
} // namespace

extern "C" LlmMetalOqA8Ctx* llmmetal_oq_a8_create(const char* qmm_lib, const char* quant_lib) {
    if (!qmm_lib || !quant_lib) return nullptr;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return nullptr;
    NSError* err = nil;
    id<MTLLibrary> lq = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:quant_lib]] error:&err];
    id<MTLLibrary> lg = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:qmm_lib]] error:&err];
    if (!lq || !lg) return nullptr;
    id<MTLFunction> fq = [lq newFunctionWithName:@"oq_a8_quantize_float16_t_am0"];
    id<MTLFunction> fg = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_float16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fqb = [lq newFunctionWithName:@"oq_a8_quantize_bfloat16_t_am0"];
    id<MTLFunction> fgb = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_bfloat16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fg1 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_float16_t_tm_1_wm_1_wn_2"];
    id<MTLFunction> fgb1 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_bfloat16_t_tm_1_wm_1_wn_2"];
    id<MTLFunction> fn1 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_float16_t_tm_1_wm_1_wn_1"];
    id<MTLFunction> fnb1 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am0_bfloat16_t_tm_1_wm_1_wn_1"];
    id<MTLFunction> fq8  = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am0_float16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fqb8 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am0_bfloat16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fq81 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am0_float16_t_tm_1_wm_1_wn_2"];
    id<MTLFunction> fqb81 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am0_bfloat16_t_tm_1_wm_1_wn_2"];
    if (!fq || !fg) return nullptr;
    Ctx* c = new (std::nothrow) Ctx();
    if (!c) return nullptr;
    c->dev = dev;
    c->queue = [dev newCommandQueue];
    c->pq = [dev newComputePipelineStateWithFunction:fq error:&err];
    c->pg = [dev newComputePipelineStateWithFunction:fg error:&err];
    if (fqb) c->pq_bf = [dev newComputePipelineStateWithFunction:fqb error:&err];
    if (fgb) c->pg_bf = [dev newComputePipelineStateWithFunction:fgb error:&err];
    if (fg1)  c->pg_tm1 = [dev newComputePipelineStateWithFunction:fg1 error:&err];
    if (fgb1) c->pg_bf_tm1 = [dev newComputePipelineStateWithFunction:fgb1 error:&err];
    id<MTLFunction> fqa  = [lq newFunctionWithName:@"oq_a8_quantize_float16_t_am1"];
    id<MTLFunction> fqba = [lq newFunctionWithName:@"oq_a8_quantize_bfloat16_t_am1"];
    id<MTLFunction> fga  = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am1_float16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fgba = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am1_bfloat16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fga1 = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am1_float16_t_tm_1_wm_1_wn_2"];
    id<MTLFunction> fgba1= [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q4_am1_bfloat16_t_tm_1_wm_1_wn_2"];
    id<MTLFunction> fq8a = [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am1_float16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fqb8a= [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am1_bfloat16_t_tm_2_wm_1_wn_2"];
    id<MTLFunction> fq81a= [lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am1_float16_t_tm_1_wm_1_wn_2"];
    id<MTLFunction> fqb81a=[lg newFunctionWithName:@"oq_a8_qmm_t_nax_v8_q8_am1_bfloat16_t_tm_1_wm_1_wn_2"];
    if (fqa)   c->pq_am1 = [dev newComputePipelineStateWithFunction:fqa error:&err];
    if (fqba)  c->pq_bf_am1 = [dev newComputePipelineStateWithFunction:fqba error:&err];
    if (fga)   c->pg_am1 = [dev newComputePipelineStateWithFunction:fga error:&err];
    if (fgba)  c->pg_bf_am1 = [dev newComputePipelineStateWithFunction:fgba error:&err];
    if (fga1)  c->pg_tm1_am1 = [dev newComputePipelineStateWithFunction:fga1 error:&err];
    if (fgba1) c->pg_bf_tm1_am1 = [dev newComputePipelineStateWithFunction:fgba1 error:&err];
    if (fq8a)  c->pg_q8_am1 = [dev newComputePipelineStateWithFunction:fq8a error:&err];
    if (fqb8a) c->pg_bf_q8_am1 = [dev newComputePipelineStateWithFunction:fqb8a error:&err];
    if (fq81a) c->pg_q8_tm1_am1 = [dev newComputePipelineStateWithFunction:fq81a error:&err];
    if (fqb81a)c->pg_bf_q8_tm1_am1 = [dev newComputePipelineStateWithFunction:fqb81a error:&err];
    if (fn1)  c->pg_n1 = [dev newComputePipelineStateWithFunction:fn1 error:&err];
    if (fnb1) c->pg_bf_n1 = [dev newComputePipelineStateWithFunction:fnb1 error:&err];
    if (fq8)   c->pg_q8 = [dev newComputePipelineStateWithFunction:fq8 error:&err];
    if (fqb8)  c->pg_bf_q8 = [dev newComputePipelineStateWithFunction:fqb8 error:&err];
    if (fq81)  c->pg_q8_tm1 = [dev newComputePipelineStateWithFunction:fq81 error:&err];
    if (fqb81) c->pg_bf_q8_tm1 = [dev newComputePipelineStateWithFunction:fqb81 error:&err];
    /* support metallib: stessa directory di qmm_lib, nome oq_a8_support.metallib */
    std::string sup = qmm_lib;
    size_t slash = sup.find_last_of('/');
    sup = (slash == std::string::npos) ? std::string("oq_a8_support.metallib")
                                       : sup.substr(0, slash + 1) + "oq_a8_support.metallib";
    id<MTLLibrary> ls = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:sup.c_str()]] error:&err];
    if (ls) {
        id<MTLFunction> fr = [ls newFunctionWithName:@"oq_repack_v8"];
        id<MTLFunction> f16 = [ls newFunctionWithName:@"oq_transpose2d_f16"];
        id<MTLFunction> fi16 = [ls newFunctionWithName:@"oq_transpose2d_i16"];
        id<MTLFunction> fbf = [ls newFunctionWithName:@"oq_transpose2d_bf16"];
        if (fr) c->repack = [dev newComputePipelineStateWithFunction:fr error:&err];
        if (f16) c->tr16 = [dev newComputePipelineStateWithFunction:f16 error:&err];
        if (fi16) c->tri16 = [dev newComputePipelineStateWithFunction:fi16 error:&err];
        if (fbf) c->trbf = [dev newComputePipelineStateWithFunction:fbf error:&err];
    }
    /* GEMV multi-output dal support metallib: qmv[bits8][M][bf16]. */
    for (int b = 0; b < 2; ++b)
        for (int m = 1; m <= 8; ++m)
            for (int d = 0; d < 2; ++d) c->qmv[b][m][d] = nil;
    if (ls) {
        for (int b = 0; b < 2; ++b) {
            const int bits = b ? 8 : 4;
            for (int m = 1; m <= 8; ++m) {
                char nm[96];
                snprintf(nm, sizeof nm, "oq_a8_qmv_q%d_m%d_float16_t", bits, m);
                id<MTLFunction> f = [ls newFunctionWithName:[NSString stringWithUTF8String:nm]];
                if (f) c->qmv[b][m][0] = [dev newComputePipelineStateWithFunction:f error:&err];
                snprintf(nm, sizeof nm, "oq_a8_qmv_q%d_m%d_bfloat16_t", bits, m);
                f = [ls newFunctionWithName:[NSString stringWithUTF8String:nm]];
                if (f) c->qmv[b][m][1] = [dev newComputePipelineStateWithFunction:f error:&err];
            }
        }
    }
    if (!c->queue || !c->pq || !c->pg || !c->tr16 || !c->tri16) { /* repack non piu' usato (permutazione fusa nel quantize) */
        if (getenv("LLMMETAL_OQ_A8_DBG")) fprintf(stderr, "[oq-create] FAIL queue=%d pq=%d pg=%d tr16=%d tri16=%d sup=%s\n", !!c->queue, !!c->pq, !!c->pg, !!c->tr16, !!c->tri16, sup.c_str());
        delete c; return nullptr; }
    c->ok = true;
    g_ctx = c;
    return (LlmMetalOqA8Ctx*)c;
}

extern "C" void llmmetal_oq_a8_destroy(LlmMetalOqA8Ctx* ctx) { if (ctx) { if (g_ctx == (Ctx*)ctx) g_ctx = nullptr; delete (Ctx*)ctx; } }

extern "C" int llmmetal_oq_a8_forward(mlx_array xh, mlx_array wh, mlx_array sch, mlx_array bih, mlx_array oh,
                                 mlx_stream sh, int bits) {
    Ctx* c = g_ctx;
    if (!c || !c->ok) return -2;
    if (!xh.ctx || !wh.ctx || !sch.ctx || !bih.ctx || !oh.ctx) return -1;
    array& x = *(array*)xh.ctx;
    array& w = *(array*)wh.ctx;
    array& sc = *(array*)sch.ctx;
    array& bi = *(array*)bih.ctx;
    array& out = *(array*)oh.ctx;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    if (x.ndim() < 2 || w.ndim() < 2) return -1;
    int K = x.shape(-1);
    int N = w.shape(0);
    int M = (int)(x.size() / (size_t)K);
    if (K <= 0 || N <= 0 || M <= 0 || (K % 64) != 0 || (N % 64) != 0) return -1;
    if (bits != 4 && bits != 5 && bits != 8) return -1;
    if (oq_am1_enabled()) return -1;   /* ACT_MODE 1 non supportato nel path eager (sa [M,G]) */
    int G = K / 64;

    /* Stage-A in tre array MLX (buffer Metal) */
    array qa = zeros({M, K}, int8, s);
    array sa = zeros({M}, float32, s);
    array ra = zeros({M, G}, int16, s);
    eval({qa, sa, ra, x});   /* UNA sincronizzazione */
    encode_stage_a(c, x, qa, sa, ra, M, K);

    /* Layout su GPU (op MLX): scales/biases -> f16 [G,N]. Ra resta ROW-major [M,G] (la
     * trasposizione e' stata eliminata: 1 dispatch in meno per QMM, il GEMM la legge diretta).
     * La permutazione v8 di qa NON si fa qui: e' fusa nel kernel di quantize
     * (oq_a8_quantize.metal) -> niente passaggio O(M*K). */
    array& qa8 = qa;
    array scT = contiguous(transpose(astype(sc, float16, s), s), false, s);
    array biT = contiguous(transpose(astype(bi, float16, s), s), false, s);
    eval({qa8, ra, scT, biT, w, out});   /* UNA sincronizzazione */

    if ((int)qa8.size() != M * K || (int)ra.size() != G * M ||
        (int)scT.size() != (size_t)N * G || (int)biT.size() != (size_t)N * G ||
        (int)out.size() != (size_t)M * N) return -3;
    encode_gemm(c, qa8, sa, ra, w, scT, biT, out, M, N, K, bits);
    return 0;
}

/* ── MLX primitive: oQ A8 nel GRAFO di MLX (una sola submit, zero sync per matmul) ── */
namespace {
class OqA8Prim : public mlx::core::Primitive {
 public:
    OqA8Prim(int bits, int use_bf16, int am1, Stream s)
        : Primitive(s), bits_(bits), use_bf16_(use_bf16), am1_(am1) {}
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("[oq_a8] nessuna implementazione CPU");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
    const char* name() const override { return "OqA8"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& o = static_cast<const OqA8Prim&>(other);
        return bits_ == o.bits_ && use_bf16_ == o.use_bf16_ && am1_ == o.am1_;
    }
    int bits_;
    int use_bf16_;
    int am1_;
};
} // namespace

void OqA8Prim::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    if (!g_ctx || !g_ctx->ok) throw std::runtime_error("[oq_a8] contesto non inizializzato");
    const array& x = inputs[0];
    const array& w = inputs[1];
    const array& sc = inputs[2];
    const array& bi = inputs[3];
    array& out = outputs[0];
    int K = x.shape(-1);
    int N = w.shape(0);
    int M = (int)(x.size() / (size_t)K);
    int G = K / 64;
    auto& s = stream();
    auto& enc = metal::get_command_encoder(s);
    array qa(Shape{M, K}, int8, nullptr, {}), sa(Shape{M}, float32, nullptr, {}), ra(Shape{M, G}, int16, nullptr, {});
    /* Prep scale/bias CACHATA (come _prepared_weights di oMLX): trasposta [N,G]->[G,N] UNA
     * volta per peso, non ad ogni chiamata. Le due tabelle sono ~4 ordini di grandezza piu'
     * piccole dei pesi. */
    struct Prep { const void* k1; const void* k2; int N; int G; array scT; array biT; };
    static std::vector<Prep> g_prep;
    Prep* pr = nullptr;
    for (auto& e : g_prep) if (e.k1 == sc.buffer().ptr() && e.k2 == bi.buffer().ptr() && e.N == N && e.G == G) { pr = &e; break; }
    bool prep_new = false;
    if (pr == nullptr) {
        const auto sd = use_bf16_ ? bfloat16 : float16;
        array sT(Shape{G, N}, sd, nullptr, {}), bT(Shape{G, N}, sd, nullptr, {});
        sT.set_data(allocator::malloc(sT.nbytes())); bT.set_data(allocator::malloc(bT.nbytes()));
        array sT2 = std::move(sT), bT2 = std::move(bT);
        Prep e{sc.buffer().ptr(), bi.buffer().ptr(), N, G, std::move(sT2), std::move(bT2)};
        g_prep.push_back(std::move(e)); pr = &g_prep.back(); prep_new = true;
    }
    array& scT = pr->scT; array& biT = pr->biT;
    /* MLX NON prealloca output/scratch di un primitive: va fatto QUI (cfr CustomKernel).
     * scT/biT NON si riallocano qui: sono CACHED per peso e il loro buffer e' allocato una
     * sola volta alla creazione della Prep (sopra) e RIEMPITO solo quando prep_new (traspose
     * sotto). Riallocarli ad ogni call lasciava la call successiva (es. 2o chunk del prefill
     * multi-chunk) a leggere un buffer mai scritto = output degenere (bug
     * MLX-A8-Prefill-MultiChunk-Degenera). */
    out.set_data(allocator::malloc(out.nbytes()));
    qa.set_data(allocator::malloc(qa.nbytes()));
    sa.set_data(allocator::malloc(sa.nbytes()));
    ra.set_data(allocator::malloc(ra.nbytes()));
    enc.add_temporary(qa); enc.add_temporary(sa); enc.add_temporary(ra);
    enc.add_temporary(scT); enc.add_temporary(biT);
    /* I parametri scalari K/N/M/G vanno passati con `set_bytes` (API di MLX per gli scalari,
     * es. mlx/backend/metal/binary.cpp: `set_bytes<int>(ndim, idx)`). Con `array kK(K)` +
     * `set_input_array` si legava il buffer di un array NON materializzato (e locale, quindi
     * distrutto a fine eval_gpu) -> il kernel leggeva valori invalidi: e' la causa dei
     * 'command-buffer failure' (K/N/M garbage -> indici fuori range) e del comportamento
     * dipendente dalla sync (con sync l'array sopravviveva alla lettura). Vedi §6ter. */
    /* Stage-A (x,qa,sa,ra,K) */
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)(use_bf16_ ? g_ctx->pq_bf : g_ctx->pq));
    enc.set_input_array(x, 0); enc.set_output_array(qa, 1); enc.set_output_array(sa, 2); enc.set_output_array(ra, 3);
    enc.set_bytes<int>(K, 4);
    enc.dispatch_threadgroups(MTL::Size(M, 1, 1), MTL::Size(256, 1, 1));
    /* NB: NIENTE repack qa->qa8 -- la permutazione v8 e' fusa nel kernel di quantize
     * (oq_a8_quantize.metal), eliminando un passaggio O(M*K) che rendeva l'A8 piu' lento. */
    /* Ra ROW-major [M,G]: nessuna trasposizione (dispatch eliminato). */
    /* scales/biases f16 [N,G] -> [G,N] */
    if (prep_new) {   /* scale/bias [N,G] -> [G,N] — solo la prima volta per peso (cache) */
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)(use_bf16_ ? g_ctx->trbf : g_ctx->tr16));
    enc.set_input_array(sc, 0); enc.set_output_array(scT, 1); enc.set_bytes<int>(N, 2); enc.set_bytes<int>(G, 3);
    enc.dispatch_threads(MTL::Size(G, N, 1), MTL::Size(16, 16, 1));
    enc.set_input_array(bi, 0); enc.set_output_array(biT, 1); enc.set_bytes<int>(N, 2); enc.set_bytes<int>(G, 3);
    enc.dispatch_threads(MTL::Size(G, N, 1), MTL::Size(16, 16, 1));
    }
    /* DIAGNOSTICA, GIA' TESTATA — CRASHA, NON RIPROVARE COSI' (Knowledge/oQ-A8-...md §6ter):
     * LLMMETAL_OQ_A8_SYNC_INTRA doveva forzare un vero waitUntilCompleted (enc.synchronize():
     * end_encoding+commit+waitUntilCompleted, poi un command buffer fresco) fra le fasi di
     * layout (Stage-A/repack/transpose) e la GEMM, invece di affidarsi solo al memoryBarrier
     * automatico — per isolare se memoryBarrier(BarrierScopeBuffers) e' insufficiente per i
     * kernel NAX/MPP sullo stesso encoder DispatchTypeConcurrent (ipotesi, vedi §6ter). RISULTATO
     * (testato 2026-09-15): SIGSEGV immediato al primo forward pass, con o senza LLMMETAL_DUMP_LAYERS.
     * Chiamare una synchronize() BLOCCANTE da DENTRO eval_gpu (quindi rientrante nello
     * scheduler/eval() di MLX che sta chiamando QUESTA stessa funzione) non e' un'operazione
     * sicura — probabile rientranza in un lock/stato dello scheduler. Non rifare un test
     * equivalente senza prima capire come MLX gestisce la rientranza in device.cpp/eval.cpp.
     * Un fix reale andrebbe cercato in una sincronizzazione che non richieda un round-trip
     * CPU bloccante dentro eval_gpu (fence dedicato? risolvere fuori da eval_gpu?). Default
     * OFF, nessun impatto quando non impostata — MA se impostata, crasha: e' rimasta solo come
     * marcatore per non riprovare la stessa cosa. */
    if (getenv("LLMMETAL_OQ_A8_DBG2")) fprintf(stderr, "[prim-eval] M=%d N=%d K=%d bits=%d bf16=%d out_bytes=%zu\n", M, N, K, bits_, (int)use_bf16_, (size_t)out.nbytes());
    if (getenv("LLMMETAL_OQ_A8_SYNC_INTRA")) enc.synchronize();
    /* GEMM (qa8,sa,ra,w,scT,biT,out,K,N,M) */
    const bool q8 = (bits_ == 8);
    const bool wn1 = (M <= 16) && getenv("LLMMETAL_OQ_A8_WN1") && (use_bf16_ ? g_ctx->pg_bf_n1 : g_ctx->pg_n1);
    const bool tm1 = (M <= 16) && !getenv("LLMMETAL_OQ_A8_NOTM1") &&
        (q8 ? (use_bf16_ ? g_ctx->pg_bf_q8_tm1 : g_ctx->pg_q8_tm1)
            : (use_bf16_ ? g_ctx->pg_bf_tm1 : g_ctx->pg_tm1));
    id<MTLComputePipelineState> pipe = nil;
    if (am1_) {
        /* ACT_MODE 1 (scale per-gruppo): stesse varianti, tabelle am1. */
        pipe = q8 ? (use_bf16_ ? (tm1 ? g_ctx->pg_bf_q8_tm1_am1 : g_ctx->pg_bf_q8_am1)
                               : (tm1 ? g_ctx->pg_q8_tm1_am1 : g_ctx->pg_q8_am1))
                  : (use_bf16_ ? (tm1 ? g_ctx->pg_bf_tm1_am1 : g_ctx->pg_bf_am1)
                               : (tm1 ? g_ctx->pg_tm1_am1 : g_ctx->pg_am1));
    }
    if (pipe == nil) {
        pipe = q8 ? (use_bf16_ ? (tm1 ? g_ctx->pg_bf_q8_tm1 : g_ctx->pg_bf_q8)
                               : (tm1 ? g_ctx->pg_q8_tm1 : g_ctx->pg_q8))
                  : (use_bf16_ ? (wn1 ? g_ctx->pg_bf_n1 : (tm1 ? g_ctx->pg_bf_tm1 : g_ctx->pg_bf))
                               : (wn1 ? g_ctx->pg_n1 : (tm1 ? g_ctx->pg_tm1 : g_ctx->pg)));
    }
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)pipe);
    enc.set_input_array(qa, 0); enc.set_input_array(sa, 1); enc.set_input_array(ra, 2);
    enc.set_input_array(w, 3); enc.set_input_array(scT, 4); enc.set_input_array(biT, 5);
    enc.set_output_array(out, 6);
    enc.set_bytes<int>(K, 7); enc.set_bytes<int>(N, 8); enc.set_bytes<int>(M, 9);
    enc.dispatch_threadgroups(MTL::Size(N / (wn1 ? 32 : 64), (M + ((wn1 || tm1) ? 15 : 31)) / ((wn1 || tm1) ? 16 : 32), 1), MTL::Size(32, 1, 2));  /* variante 806 (oMLX Q4) */
}

/* Helper definito DOPO le classi dei primitive (Stage-A + QMM in catena). */
static mlx_array oq_a8_prim_chain(const array& x, const array& w, const array& sc, const array& bi,
                                  int bits, bool use_bf16, Stream s);

/* C ABI: costruisce l'array di output col primitive (entra nel grafo; nessuna sync qui). */
extern "C" mlx_array llmmetal_oq_a8_prim(mlx_array xh, mlx_array wh, mlx_array sch, mlx_array bih, mlx_stream sh, int bits) {
    mlx_array r; r.ctx = nullptr;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    if (getenv("LLMMETAL_OQ_A8_DBG")) fprintf(stderr, "[oq-prim] g_ctx=%p bits=%d x=%p w=%p\n", (void*)g_ctx, bits, xh.ctx, wh.ctx);
    if (!g_ctx || (bits != 4 && bits != 8)) return r;
    if (bits == 8 && !g_ctx->pg_q8) return r;   /* metallib senza Q8 -> fail-open (MLX) */
    if (!xh.ctx || !wh.ctx || !sch.ctx || !bih.ctx) return r;
    array x = *(array*)xh.ctx, w = *(array*)wh.ctx, sc = *(array*)sch.ctx, bi = *(array*)bih.ctx;
    if (x.ndim() < 2 || w.ndim() < 2) return r;
    int K = x.shape(-1), N = w.shape(0);
    if (getenv("LLMMETAL_OQ_A8_DBG")) fprintf(stderr, "[oq-prim] K=%d N=%d M=%d\n", K, N, (int)(x.size()/(size_t)K));
    if (K <= 0 || N <= 0 || (K % 64) != 0 || (N % 64) != 0) return r;
    if (w.dtype() != mlx::core::uint32) return r;
    /* dtype del modello (come oMLX): bf16 se il modello e' bf16 -> pipeline bf16, NESSUN cast */
    const bool use_bf16 = (x.dtype() == bfloat16) && g_ctx && g_ctx->pq_bf && g_ctx->pg_bf && g_ctx->trbf;
    const int M0 = (int)(x.size() / (size_t)K);
    /* Split-K a M piccolo: il fused (Stage-A + GEMM in un primitive) non ha il percorso
     * condiviso; con LLMMETAL_OQ_A8_SPLITK>1 si delega a stage_a + qmm (che ha lo split). */
    int split0 = 1;
    if (const char* se = getenv("LLMMETAL_OQ_A8_SPLITK")) { split0 = atoi(se); if (split0 < 1) split0 = 1; }
    if (split0 > 1 && (M0 > 16 || ((K / 64) % split0) != 0)) split0 = 1;
    if (split0 > 1) {
        mlx_array qa = mlx_array_new(), sa = mlx_array_new(), ra = mlx_array_new();
        int src = llmmetal_oq_a8_stage_a(xh, &qa, &sa, &ra, sh);
        if (src == 0) {
            mlx_array r2 = llmmetal_oq_a8_qmm(qa, sa, ra, wh, sch, bih, sh, bits, use_bf16 ? 1 : 0);
            mlx_array_free(qa); mlx_array_free(sa); mlx_array_free(ra);
            if (r2.ctx != nullptr) return r2;
        } else {
            mlx_array_free(qa); mlx_array_free(sa); mlx_array_free(ra);
        }
    }
    if (M0 < 16) return r;
    if (!use_bf16 && x.dtype() != float16) x = astype(x, float16, s);
    if (!x.flags().row_contiguous) x = contiguous(x, false, s);
    if (!w.flags().row_contiguous) w = contiguous(w, false, s);
    if (!use_bf16) { if (sc.dtype() != float16) sc = astype(sc, float16, s); if (bi.dtype() != float16) bi = astype(bi, float16, s); }
    if (!sc.flags().row_contiguous) sc = contiguous(sc, false, s);
    if (!bi.flags().row_contiguous) bi = contiguous(bi, false, s);
    r = oq_a8_prim_chain(x, w, sc, bi, bits, use_bf16, s);
    return r;
}

/* ══════════════════════════════════════════════════════════════════════════════════════
 * FEDELTA' oMLX (qwen35_oq_a8.py): Stage A e' un'op SEPARATA e CONDIVISA.
 * oMLX: `stage_a(x)` una volta per attivazione -> la stessa (qa,sa,ra) alimenta ogni
 * proiezione che consuma quell'attivazione (MLP gate+up; qkv/z/a/b). Il fused OqA8Prim qui
 * sopra ricalcola Stage A a OGNI proiezione: corretto ma ridondante.
 * Qui: due primitive separate, con la STESSA struttura del fused (stessi kernel, encoder del
 * grafo MLX, cache prep) -> il motore puo' quantizzare una volta e riusare.
 * ══════════════════════════════════════════════════════════════════════════════════════ */
namespace {
class OqA8QuantPrim : public mlx::core::Primitive {
 public:
    OqA8QuantPrim(int use_bf16, int am1, Stream s) : Primitive(s), use_bf16_(use_bf16), am1_(am1) {}
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("[oq_a8] nessuna implementazione CPU");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
    /* OBBLIGATORIA per un prim multi-uscita (il default lancia "cannot infer output shapes",
     * chiamato da mx.compile in modalita' shapeless). Copiata da oMLX. */
    std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
        auto qa_shape = inputs[0].shape();
        ShapeElem K = qa_shape.back();
        auto M = static_cast<ShapeElem>(inputs[0].size() / K);
        /* Ra ROW-major [M, G] (trasposizione eliminata: il GEMM la legge diretta).
         * Sa: [M] in ACT_MODE 0, [M, G] in ACT_MODE 1 (scale per-gruppo). */
        return {qa_shape, am1_ ? Shape{M, K / 64} : Shape{M}, Shape{M, K / 64}};
    }
    const char* name() const override { return "OqA8Quant"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& o = static_cast<const OqA8QuantPrim&>(other);
        return use_bf16_ == o.use_bf16_ && am1_ == o.am1_;
    }
    int use_bf16_, am1_;
};

class OqA8QmmPrim : public mlx::core::Primitive {
 public:
    OqA8QmmPrim(int bits, int use_bf16, int g0, int g1, int am1, Stream s)
        : Primitive(s), bits_(bits), use_bf16_(use_bf16), g0_(g0), g1_(g1), am1_(am1) {}
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("[oq_a8] nessuna implementazione CPU");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
    std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
        auto s = inputs[0].shape();
        s.back() = inputs[3].shape(0);
        return {std::move(s)};
    }
    const char* name() const override { return "OqA8Qmm"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& o = static_cast<const OqA8QmmPrim&>(other);
        return bits_ == o.bits_ && use_bf16_ == o.use_bf16_ && g0_ == o.g0_ && g1_ == o.g1_ &&
               am1_ == o.am1_;
    }
    int bits_, use_bf16_, g0_, g1_, am1_;
};
} // namespace

/* Catena dei DUE primitive gia' validati (Stage-A + QMM) invece del fused OqA8Prim: stesso
 * lavoro (stesse dispatch), una primitive in piu' nel grafo. Il fused restava a output zero
 * in isolamento (harness unitario) pur funzionando nel motore: non isolato, e i due primitive
 * separati sono la strada su cui girano anche lo stage condiviso e il Q8 -> una sola
 * implementazione validata. */
static mlx_array oq_a8_prim_chain(const array& x, const array& w, const array& sc, const array& bi,
                                  int bits, bool use_bf16, Stream s) {
    mlx_array r; r.ctx = nullptr;
    const int K = x.shape(-1);
    const int N = w.shape(0);
    const int M = (int)(x.size() / (size_t)K);
    const int G = K / 64;
    Shape oshape = x.shape();
    oshape.back() = N;
    const int am1 = oq_am1_enabled() ? 1 : 0;
    auto qprim = std::make_shared<OqA8QuantPrim>(use_bf16 ? 1 : 0, am1, s);
    /* sa: [M] in ACT_MODE 0, [M,G] in ACT_MODE 1. */
    auto outs = array::make_arrays(
        {x.shape(), am1 ? Shape{M, G} : Shape{M}, Shape{M, G}}, {int8, float32, int16}, qprim, {x});
    array out(oshape, use_bf16 ? bfloat16 : float16,
              std::make_shared<OqA8QmmPrim>(bits, use_bf16 ? 1 : 0, 0, G, am1, s),
              {outs[0], outs[1], outs[2], w, sc, bi});
    r.ctx = new array(std::move(out));
    return r;
}

/* Stage A: x [.., K] -> qa [.., K] int8 (permutazione v8 fusa nel kernel) + sa [M] f32
 * + ra [M, G] int16 ROW-major (la trasposizione [M,G]->[G,M] e' stata eliminata). */
void OqA8QuantPrim::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    if (!g_ctx || !g_ctx->ok) throw std::runtime_error("[oq_a8] contesto non inizializzato");
    const array& x = inputs[0];
    array& qa = outputs[0];
    array& sa = outputs[1];
    array& ra = outputs[2];   /* Ra ROW-major [M, G]: nessuna trasposizione (dispatch eliminato) */
    int K = x.shape(-1);
    int M = (int)(x.size() / (size_t)K);
    auto& s = stream();
    auto& enc = metal::get_command_encoder(s);
    qa.set_data(allocator::malloc(qa.nbytes()));
    sa.set_data(allocator::malloc(sa.nbytes()));
    ra.set_data(allocator::malloc(ra.nbytes()));
    id<MTLComputePipelineState> qp = nil;
    if (am1_) qp = use_bf16_ ? g_ctx->pq_bf_am1 : g_ctx->pq_am1;
    if (qp == nil) qp = use_bf16_ ? g_ctx->pq_bf : g_ctx->pq;
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)qp);
    enc.set_input_array(x, 0); enc.set_output_array(qa, 1); enc.set_output_array(sa, 2); enc.set_output_array(ra, 3);
    enc.set_bytes<int>(K, 4);
    enc.dispatch_threadgroups(MTL::Size(M, 1, 1), MTL::Size(256, 1, 1));
}

/* GEMM su Stage A gia' calcolata: (qa, sa, ra, w, scales, biases) -> out [.., N].
 * La prep scale/bias [N,G]->[G,N] e' cachata per peso (come _prepared_weights di oMLX). */
void OqA8QmmPrim::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    if (!g_ctx || !g_ctx->ok) throw std::runtime_error("[oq_a8] contesto non inizializzato");
    const array& qa = inputs[0];
    const array& sa = inputs[1];
    const array& ra = inputs[2];
    const array& w = inputs[3];
    const array& sc = inputs[4];
    const array& bi = inputs[5];
    array& out = outputs[0];
    int K = qa.shape(-1);
    int N = w.shape(0);
    int M = (int)(qa.size() / (size_t)K);
    int G = K / 64;
    auto& s = stream();
    auto& enc = metal::get_command_encoder(s);
    struct Prep { const void* k1; const void* k2; int N; int G; array scT; array biT; };
    static std::vector<Prep> g_prep2;
    Prep* pr = nullptr;
    for (auto& e : g_prep2) if (e.k1 == sc.buffer().ptr() && e.k2 == bi.buffer().ptr() && e.N == N && e.G == G) { pr = &e; break; }
    bool prep_new = false;
    if (pr == nullptr) {
        const auto sd = use_bf16_ ? bfloat16 : float16;
        array sT(Shape{G, N}, sd, nullptr, {}), bT(Shape{G, N}, sd, nullptr, {});
        sT.set_data(allocator::malloc(sT.nbytes())); bT.set_data(allocator::malloc(bT.nbytes()));
        Prep e{sc.buffer().ptr(), bi.buffer().ptr(), N, G, std::move(sT), std::move(bT)};
        g_prep2.push_back(std::move(e)); pr = &g_prep2.back(); prep_new = true;
    }
    array& scT = pr->scT; array& biT = pr->biT;
    /* scT/biT: allocati alla creazione della Prep e riempiti solo se prep_new (come OqA8Prim:
     * NIENTE riallocazione per-call, altrimenti la call successiva legge un buffer non scritto). */
    out.set_data(allocator::malloc(out.nbytes()));
    enc.add_temporary(scT); enc.add_temporary(biT);
    if (prep_new) {
        enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)(use_bf16_ ? g_ctx->trbf : g_ctx->tr16));
        enc.set_input_array(sc, 0); enc.set_output_array(scT, 1); enc.set_bytes<int>(N, 2); enc.set_bytes<int>(G, 3);
        enc.dispatch_threads(MTL::Size(G, N, 1), MTL::Size(16, 16, 1));
        enc.set_input_array(bi, 0); enc.set_output_array(biT, 1); enc.set_bytes<int>(N, 2); enc.set_bytes<int>(G, 3);
        enc.dispatch_threads(MTL::Size(G, N, 1), MTL::Size(16, 16, 1));
    }
    /* GEMV multi-output: a M<=8 la griglia per colonna satura meglio del GEMM a tile
     * (misurato: il GEMM a M=8 sta a ~100-230 GB/s, il GEMV MLX a M=1 a ~465). Opt-in. */
    if (getenv("LLMMETAL_OQ_A8_QMV") && M >= 1 && M <= 8 && (bits_ == 4 || bits_ == 8)) {
        id<MTLComputePipelineState> qmv_pipe = g_ctx->qmv[(bits_ == 8) ? 1 : 0][M][use_bf16_ ? 1 : 0];
        if (qmv_pipe) {
            const int cols_per_tg = (bits_ == 8) ? 2 : 4;
            enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)qmv_pipe);
            enc.set_input_array(qa, 0); enc.set_input_array(sa, 1); enc.set_input_array(ra, 2);
            enc.set_input_array(w, 3); enc.set_input_array(scT, 4); enc.set_input_array(biT, 5);
            enc.set_output_array(out, 6);
            enc.set_bytes<int>(K, 7); enc.set_bytes<int>(N, 8); enc.set_bytes<int>(M, 9);
            enc.set_bytes<int>(g0_, 10); enc.set_bytes<int>(g1_, 11);
            enc.dispatch_threadgroups(MTL::Size((N + cols_per_tg - 1) / cols_per_tg, 1, 1), MTL::Size(32, 1, 1));
            return;
        }
    }
    const bool q8 = (bits_ == 8);
    const bool wn1 = (M <= 16) && getenv("LLMMETAL_OQ_A8_WN1") && (use_bf16_ ? g_ctx->pg_bf_n1 : g_ctx->pg_n1);
    const bool tm1 = (M <= 16) && !getenv("LLMMETAL_OQ_A8_NOTM1") &&
        (q8 ? (use_bf16_ ? g_ctx->pg_bf_q8_tm1 : g_ctx->pg_q8_tm1)
            : (use_bf16_ ? g_ctx->pg_bf_tm1 : g_ctx->pg_tm1));
    id<MTLComputePipelineState> pipe = nil;
    if (am1_) {
        /* ACT_MODE 1 (scale per-gruppo): stesse varianti, tabelle am1. */
        pipe = q8 ? (use_bf16_ ? (tm1 ? g_ctx->pg_bf_q8_tm1_am1 : g_ctx->pg_bf_q8_am1)
                               : (tm1 ? g_ctx->pg_q8_tm1_am1 : g_ctx->pg_q8_am1))
                  : (use_bf16_ ? (tm1 ? g_ctx->pg_bf_tm1_am1 : g_ctx->pg_bf_am1)
                               : (tm1 ? g_ctx->pg_tm1_am1 : g_ctx->pg_am1));
    }
    if (pipe == nil) {
        pipe = q8 ? (use_bf16_ ? (tm1 ? g_ctx->pg_bf_q8_tm1 : g_ctx->pg_bf_q8)
                               : (tm1 ? g_ctx->pg_q8_tm1 : g_ctx->pg_q8))
                  : (use_bf16_ ? (wn1 ? g_ctx->pg_bf_n1 : (tm1 ? g_ctx->pg_bf_tm1 : g_ctx->pg_bf))
                               : (wn1 ? g_ctx->pg_n1 : (tm1 ? g_ctx->pg_tm1 : g_ctx->pg)));
    }
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)pipe);
    enc.set_input_array(qa, 0); enc.set_input_array(sa, 1); enc.set_input_array(ra, 2);
    enc.set_input_array(w, 3); enc.set_input_array(scT, 4); enc.set_input_array(biT, 5);
    enc.set_output_array(out, 6);
    enc.set_bytes<int>(K, 7); enc.set_bytes<int>(N, 8); enc.set_bytes<int>(M, 9);
    enc.set_bytes<int>(g0_, 10); enc.set_bytes<int>(g1_, 11);
    enc.dispatch_threadgroups(MTL::Size(N / (wn1 ? 32 : 64), (M + ((wn1 || tm1) ? 15 : 31)) / ((wn1 || tm1) ? 16 : 32), 1), MTL::Size(32, 1, 2));
}

/* C ABI Stage A. Convenzione di uscita del bridge: il chiamante passa strutture mlx_array NON
 * inizializzate (ctx NULL) e libera con mlx_array_free. Ritorna 0 su successo. */
extern "C" int llmmetal_oq_a8_stage_a(mlx_array xh, mlx_array* qah, mlx_array* sah, mlx_array* rah, mlx_stream sh) {
    if (!qah || !sah || !rah) return -1;
    qah->ctx = nullptr; sah->ctx = nullptr; rah->ctx = nullptr;
    if (!g_ctx || !g_ctx->ok) return -2;
    if (!xh.ctx) return -1;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    array x = *(array*)xh.ctx;
    if (x.ndim() < 2) return -1;
    int K = x.shape(-1);
    int M = (int)(x.size() / (size_t)K);
    if (K <= 0 || M <= 0 || (K % 64) != 0) return -1;
    const bool use_bf16 = (x.dtype() == bfloat16) && g_ctx->pq_bf;
    if (!use_bf16 && x.dtype() != float16) x = astype(x, float16, s);
    if (!use_bf16 && !x.flags().row_contiguous) x = contiguous(x, false, s);
    const int am1 = oq_am1_enabled() ? 1 : 0;
    auto prim = std::make_shared<OqA8QuantPrim>(use_bf16 ? 1 : 0, am1, s);
    /* array::make_arrays e' OBBLIGATORIO: imposta i siblings/l'output index. Con 3 array
     * costruiti a mano (senza siblings) il compilatore MLX, sotto mx.compile, traccia N prim
     * a una uscita -> eval_gpu scrive outputs[0..2] con una sola uscita allocata = corruzione.
     * Sa: [M] in ACT_MODE 0, [M, G] in ACT_MODE 1 (deve combaciare con output_shapes). */
    auto outs = array::make_arrays(
        {x.shape(), am1 ? Shape{M, K / 64} : Shape{M}, Shape{M, K / 64}}, {int8, float32, int16},
        prim, {x});
    qah->ctx = new array(std::move(outs[0]));
    sah->ctx = new array(std::move(outs[1]));
    rah->ctx = new array(std::move(outs[2]));
    return 0;
}

/* C ABI GEMM su Stage A. qa/sa/ra da llmmetal_oq_a8_stage_a; w/sc/bi come il fused. */
extern "C" mlx_array llmmetal_oq_a8_qmm(mlx_array qah, mlx_array sah, mlx_array rah, mlx_array wh,
                                   mlx_array sch, mlx_array bih, mlx_stream sh, int bits, int want_bf16) {
    mlx_array r; r.ctx = nullptr;
    if (!g_ctx || (bits != 4 && bits != 8)) return r;
    if (bits == 8 && !g_ctx->pg_q8) return r;   /* metallib senza Q8 -> fail-open (MLX) */
    if (!qah.ctx || !sah.ctx || !rah.ctx || !wh.ctx || !sch.ctx || !bih.ctx) return r;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    array qa = *(array*)qah.ctx, sa = *(array*)sah.ctx, ra = *(array*)rah.ctx;
    array w = *(array*)wh.ctx, sc = *(array*)sch.ctx, bi = *(array*)bih.ctx;
    if (qa.ndim() < 2 || w.ndim() < 2) return r;
    int K = qa.shape(-1), N = w.shape(0);
    if (K <= 0 || N <= 0 || (K % 64) != 0 || (N % 64) != 0) return r;
    if (w.dtype() != mlx::core::uint32) return r;
    const bool use_bf16 = (want_bf16 != 0) && g_ctx->pg_bf && g_ctx->trbf && (sc.dtype() == bfloat16);
    if (!use_bf16) {
        if (sc.dtype() != float16) sc = astype(sc, float16, s);
        if (bi.dtype() != float16) bi = astype(bi, float16, s);
    }
    if (!w.flags().row_contiguous) w = contiguous(w, false, s);
    if (!sc.flags().row_contiguous) sc = contiguous(sc, false, s);
    if (!bi.flags().row_contiguous) bi = contiguous(bi, false, s);
    Shape qs = qa.shape();
    Shape oshape = qa.shape();
    oshape.back() = N;
    const auto dt = use_bf16 ? bfloat16 : float16;
    const int M = (int)(qa.size() / (size_t)K);
    /* Split-K (P5): a M piccolo il kernel e' occupancy-bound (~100 GB/s su 546 di picco);
     * spezzare K in `split` slice moltiplica i blocchi e il read rate sale (~3x a 4x blocchi,
     * misurato: N 12288 -> 49152 = 75 -> 215 GB/s). Ogni slice scrive il parziale corretto e
     * la somma e' esatta (correzione affine additiva per gruppo). Env LLMMETAL_OQ_A8_SPLITK. */
    int split = 1;
    if (const char* se = getenv("LLMMETAL_OQ_A8_SPLITK")) { split = atoi(se); if (split < 1) split = 1; }
    const int G = K / 64;
    if (split > 1 && (M > 16 || (G % split) != 0)) split = 1;
    if (split == 1) {
        array out(oshape, dt, std::make_shared<OqA8QmmPrim>(bits, use_bf16 ? 1 : 0, 0, G,
                                                             oq_am1_enabled() ? 1 : 0, s),
                  {qa, sa, ra, w, sc, bi});
        r.ctx = new array(std::move(out));
        return r;
    }
    array acc(oshape, dt, nullptr, {});
    for (int sl = 0; sl < split; ++sl) {
        const int g0 = sl * (G / split), g1 = (sl + 1) * (G / split);
        array part(oshape, dt, std::make_shared<OqA8QmmPrim>(bits, use_bf16 ? 1 : 0, g0, g1,
                                                             oq_am1_enabled() ? 1 : 0, s),
                   {qa, sa, ra, w, sc, bi});
        acc = (sl == 0) ? std::move(part) : mlx::core::add(acc, part, s);
    }
    r.ctx = new array(std::move(acc));
    return r;
}

/* N-fusione gate+up (vedi header): concat cachato + QMM singolo con N doppio.
 * FALSIFICATA nel motore (2026-09-19): -5,5% a chunk 8 vs due QMM separate — il guadagno
 * apparente della curva N era l'overhead per-call. Tenuta per A/B e documentazione. */
extern "C" int llmmetal_oq_a8_qmm_fused2(mlx_array* out, mlx_array qah, mlx_array sah, mlx_array rah,
                                    mlx_array w1h, mlx_array s1h, mlx_array b1h,
                                    mlx_array w2h, mlx_array s2h, mlx_array b2h,
                                    mlx_stream sh, int bits, int want_bf16) {
    if (out == nullptr) return -1;
    out->ctx = nullptr;
    if (!g_ctx || (bits != 4 && bits != 8)) return -1;
    if (!qah.ctx || !sah.ctx || !rah.ctx || !w1h.ctx || !s1h.ctx || !b1h.ctx ||
        !w2h.ctx || !s2h.ctx || !b2h.ctx) return -1;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    array w1 = *(array*)w1h.ctx, w2 = *(array*)w2h.ctx;
    array s1 = *(array*)s1h.ctx, s2 = *(array*)s2h.ctx;
    array b1 = *(array*)b1h.ctx, b2 = *(array*)b2h.ctx;
    /* materializza (i pesi del modello lo sono gia': eval e' un no-op) */
    eval({w1, w2, s1, s2, b1, b2});
    struct Fused { const void* k1; const void* k2; array w; array sc; array bi; };
    static std::vector<Fused> g_fused2;
    Fused* pr = nullptr;
    for (auto& e : g_fused2)
        if (e.k1 == w1.buffer().ptr() && e.k2 == w2.buffer().ptr()) { pr = &e; break; }
    if (pr == nullptr) {
        array w = concatenate({w1, w2}, 0, s);
        array sc = concatenate({s1, s2}, 0, s);
        array bi = concatenate({b1, b2}, 0, s);
        eval({w, sc, bi});   /* copia una-tantum, cachata */
        Fused e{w1.buffer().ptr(), w2.buffer().ptr(), std::move(w), std::move(sc), std::move(bi)};
        g_fused2.push_back(std::move(e));
        pr = &g_fused2.back();
    }
    mlx_array wh{}; mlx_array sch{}; mlx_array bih{};
    wh.ctx = new array(pr->w); sch.ctx = new array(pr->sc); bih.ctx = new array(pr->bi);
    mlx_array r = llmmetal_oq_a8_qmm(qah, sah, rah, wh, sch, bih, sh, bits, want_bf16);
    mlx_array_free(wh); mlx_array_free(sch); mlx_array_free(bih);
    if (r.ctx == nullptr) return -1;
    *out = r;
    return 0;
}
