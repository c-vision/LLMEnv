#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "llmmetal_mpp_q4_bridge.h"
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/backend/metal/device.h"
#include <string>
#include <stdexcept>
using namespace mlx::core;

struct LlmMetalMppQ4Ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    bool ok = false;
    id<MTLComputePipelineState> ps[4][2];  /* [M idx: 8,16,32,64][GS idx: 32,64] */
    /* Variante "tile" (schema Splash): [M idx: 16,24,32][GS idx: 32,64] */
    id<MTLComputePipelineState> tile_ps[3][2];
    bool tile_ok = false;
};

/* Deve combaciare con Q4TParams in csrc/q4tile.metal. */
struct Q4TParams {
    uint32_t input_size, output_size, wstride, wgroup_adv, wtile_base, s_a, s_b, s_c;
};

static LlmMetalMppQ4Ctx* g_mpp_q4_ctx = NULL;   /* singleton (un modello attivo) */
static inline id<MTLBuffer> mtlb(const array& a) {
    return (__bridge id<MTLBuffer>)(const_cast<void*>(a.buffer().ptr()));
}
static inline NSUInteger moff(const array& a) { return (NSUInteger)a.offset(); }
static int midx(int M){ return (M<=8)?0:(M<=16)?1:(M<=32)?2:3; }
static int gidx(int GS){ return (GS==32)?0:(GS==64)?1:-1; }

extern "C" LlmMetalMppQ4Ctx* llmmetal_mpp_q4_create(const char* metallib_path) {
    LlmMetalMppQ4Ctx* c = new LlmMetalMppQ4Ctx();
    c->dev = MTLCreateSystemDefaultDevice();
    if (!c->dev) { delete c; return NULL; }
    NSError* err = nil;
    id<MTLLibrary> lib = [c->dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]] error:&err];
    if (!lib) { fprintf(stderr, "mpp_q4: library load failed: %s\n", err.localizedDescription.UTF8String); delete c; return NULL; }
    int Ms[4] = {8,16,32,64}; int GSs[2] = {32,64};
    for (int i=0;i<4;i++) for (int j=0;j<2;j++) {
        char nm[64]; snprintf(nm,sizeof(nm),"mpp_q4_m%d_gs%d",Ms[i],GSs[j]);
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:nm]];
        if (!f) { fprintf(stderr, "mpp_q4: function %s missing\n", nm); delete c; return NULL; }
        c->ps[i][j] = [c->dev newComputePipelineStateWithFunction:f error:&err];
        if (!c->ps[i][j]) { fprintf(stderr, "mpp_q4: pipeline %s failed\n", nm); delete c; return NULL; }
    }
    /* Variante tile: opzionale (una metallib piu' vecchia puo' non averla). */
    int TMs[3] = {16, 24, 32};
    c->tile_ok = true;
    for (int i = 0; i < 3 && c->tile_ok; i++)
        for (int j = 0; j < 2; j++) {
            char nm[64]; snprintf(nm, sizeof(nm), "q4t_m%d_n128_g%d", TMs[i], GSs[j]);
            id<MTLFunction> tf = [lib newFunctionWithName:[NSString stringWithUTF8String:nm]];
            if (!tf) { c->tile_ok = false; break; }
            c->tile_ps[i][j] = [c->dev newComputePipelineStateWithFunction:tf error:&err];
            if (!c->tile_ps[i][j]) { c->tile_ok = false; break; }
        }
    c->queue = [c->dev newCommandQueue];
    c->ok = true;
    g_mpp_q4_ctx = c;
    fprintf(stderr, "[mppbr] create OK dev=%p queue=%p\n", (__bridge void*)c->dev, (__bridge void*)c->queue);
    return c;
}
extern "C" void llmmetal_mpp_q4_destroy(LlmMetalMppQ4Ctx* c) { if (c) delete c; }

extern "C" int llmmetal_mpp_q4_forward(mlx_array xh, mlx_array wh, mlx_array sch, mlx_array bih, mlx_array oh,
                                  int M, int GS) {
    LlmMetalMppQ4Ctx* c = g_mpp_q4_ctx;
    if (!c || !c->ok) return -2;
    if (!xh.ctx || !wh.ctx || !sch.ctx || !bih.ctx || !oh.ctx) return -1;
    if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr,"[mppbr] fwd M=%d GS=%d x.ctx=%p w.ctx=%p\n", M, GS, (void*)xh.ctx, (void*)wh.ctx);
    int gi = gidx(GS); if (gi < 0) return -1;
    int mi = midx(M);
    if (M != (mi==0?8:mi==1?16:mi==2?32:64)) return -1;
    if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr,"[mppbr] cast arrays...\n");
    array& x = *(array*)xh.ctx; array& w = *(array*)wh.ctx;
    array& sc = *(array*)sch.ctx; array& bi = *(array*)bih.ctx; array& out = *(array*)oh.ctx;
    int K = x.shape(-1), N = w.shape(0), xr = x.shape(0);
    if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr,"[mppbr] K=%d N=%d xr=%d ndim=%d\n", K, N, xr, x.ndim());
    if (K%GS || N%128 || xr!=M) return -1;
    /* materializza i buffer GPU (le array possono essere lazy) */
    mlx_array_eval(xh); mlx_array_eval(wh); mlx_array_eval(sch); mlx_array_eval(bih); mlx_array_eval(oh);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:c->ps[mi][gi]];
        [e setBuffer:mtlb(x)  offset:moff(x)  atIndex:0];
        [e setBuffer:mtlb(w)  offset:moff(w)  atIndex:1];
        [e setBuffer:mtlb(sc) offset:moff(sc) atIndex:2];
        [e setBuffer:mtlb(bi) offset:moff(bi) atIndex:3];
        [e setBuffer:mtlb(out) offset:moff(out) atIndex:4];
        uint32_t kk=(uint32_t)K, nn=(uint32_t)N;
        id<MTLBuffer> bK=[c->dev newBufferWithBytes:&kk length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bN=[c->dev newBufferWithBytes:&nn length:4 options:MTLResourceStorageModeShared];
        [e setBuffer:bK offset:0 atIndex:5]; [e setBuffer:bN offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(N/128),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (getenv("LLMMETAL_MPP_Q4_DUMP")) fprintf(stderr,"[mppbr] encode ok grid=%d\n", N/128);
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr,"mpp_q4: %s\n", cb.error.localizedDescription.UTF8String); return -3; }
    }
    return 0;
}

/* ---------------------------------------------------------------------------------------
 * Primitive MLX in-graph: il kernel MPP viene codificato NELL'encoder Metal di MLX
 * (metal::get_command_encoder), quindi nessuna sync per chiamata e nessun command buffer
 * privato: e' la via veloce (lo stesso schema di OqA8QmmPrim). L'array di output entra nel
 * grafo e viene valutato da MLX insieme al resto del forward.
 * ------------------------------------------------------------------------------------- */
class MppQ4Prim : public mlx::core::Primitive {
  public:
    MppQ4Prim(int mi, int gi, id<MTLComputePipelineState> ps, Stream s)
        : Primitive(s), mi_(mi), gi_(gi), ps_(ps) {}
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("[mpp_q4] nessuna implementazione CPU");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
    std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
        Shape s = inputs[0].shape();
        s.back() = inputs[1].shape(0);
        return {std::move(s)};
    }
    const char* name() const override { return "MppQ4"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& o = static_cast<const MppQ4Prim&>(other);
        return mi_ == o.mi_ && gi_ == o.gi_;
    }
    int mi_, gi_;
    id<MTLComputePipelineState> ps_;
};

void MppQ4Prim::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    if (!g_mpp_q4_ctx || !g_mpp_q4_ctx->ok) throw std::runtime_error("[mpp_q4] contesto non inizializzato");
    const array& x = inputs[0];
    const array& w = inputs[1];
    const array& sc = inputs[2];
    const array& bi = inputs[3];
    array& out = outputs[0];
    const int K = x.shape(-1), N = w.shape(0);
    auto& s = stream();
    auto& enc = metal::get_command_encoder(s);
    out.set_data(allocator::malloc(out.nbytes()));
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)ps_);
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_input_array(sc, 2);
    enc.set_input_array(bi, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes<int>(K, 5);
    enc.set_bytes<int>(N, 6);
    enc.dispatch_threadgroups(MTL::Size((size_t)(N / 128), 1, 1), MTL::Size(256, 1, 1));
}

/* C ABI: costruisce il nodo del grafo (x, w, sc, bi) -> out [.., N]. Nessuna sync qui. */
extern "C" mlx_array llmmetal_mpp_q4_prim(mlx_array xh, mlx_array wh, mlx_array sch, mlx_array bih,
                                     mlx_stream sh, int M, int GS) {
    mlx_array r; r.ctx = nullptr;
    LlmMetalMppQ4Ctx* c = g_mpp_q4_ctx;
    if (!c || !c->ok) return r;
    if (!xh.ctx || !wh.ctx || !sch.ctx || !bih.ctx) return r;
    const int gi = gidx(GS);
    const int mi = midx(M);
    if (gi < 0 || M != (mi == 0 ? 8 : mi == 1 ? 16 : mi == 2 ? 32 : 64)) return r;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    array x = *(array*)xh.ctx, w = *(array*)wh.ctx;
    array sc = *(array*)sch.ctx, bi = *(array*)bih.ctx;
    if (x.ndim() != 2 || x.dtype() != bfloat16) return r;
    const int K = x.shape(-1), N = w.shape(0);
    if (K % GS || N % 128 || x.shape(0) != M) return r;
    Shape os = x.shape();
    os.back() = N;
    array out(os, bfloat16, std::make_shared<MppQ4Prim>(mi, gi, c->ps[mi][gi], s), {x, w, sc, bi});
    r.ctx = new array(std::move(out));
    return r;
}

/* ---------------------------------------------------------------------------------------
 * Variante "tile" (schema Splash, q4_mpp_tiles.h): il kernel legge i pesi DIRETTAMENTE dal
 * layout MLX [N, K/8] (stride K in nibble fra colonne, nessun repack) e usa le somme
 * d'ingresso condivise per il termine bias. E' la variante misurata 1.2-2.7x sopra qmm MLX.
 * ------------------------------------------------------------------------------------- */
class MppQ4TilePrim : public mlx::core::Primitive {
  public:
    MppQ4TilePrim(int mi, int gi, int rows, Q4TParams p,
                  id<MTLComputePipelineState> ps, Stream s)
        : Primitive(s), mi_(mi), gi_(gi), rows_(rows), p_(p), ps_(ps) {}
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("[mpp_q4t] nessuna implementazione CPU");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
    std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
        Shape s = inputs[0].shape();
        s.back() = inputs[1].shape(0);
        return {std::move(s)};
    }
    const char* name() const override { return "MppQ4Tile"; }
    bool is_equivalent(const Primitive& other) const override {
        const auto& o = static_cast<const MppQ4TilePrim&>(other);
        return mi_ == o.mi_ && gi_ == o.gi_ && rows_ == o.rows_;
    }
    int mi_, gi_, rows_;
    Q4TParams p_;
    id<MTLComputePipelineState> ps_;
};

void MppQ4TilePrim::eval_gpu(const std::vector<array>& inputs,
                             std::vector<array>& outputs) {
    if (!g_mpp_q4_ctx || !g_mpp_q4_ctx->ok) throw std::runtime_error("[mpp_q4t] contesto non inizializzato");
    const array& x = inputs[0];
    const array& w = inputs[1];
    const array& sc = inputs[2];
    const array& bi = inputs[3];
    array& out = outputs[0];
    auto& s = stream();
    auto& enc = metal::get_command_encoder(s);
    out.set_data(allocator::malloc(out.nbytes()));
    enc.set_compute_pipeline_state((__bridge MTL::ComputePipelineState*)ps_);
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_input_array(sc, 2);
    enc.set_input_array(bi, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes<Q4TParams>(p_, 5);
    enc.dispatch_threadgroups(MTL::Size((size_t)(p_.output_size / 128), 1, 1),
                              MTL::Size(256, 1, 1));
}

extern "C" mlx_array llmmetal_mpp_q4_tile_prim(mlx_array xh, mlx_array wh, mlx_array sch,
                                          mlx_array bih, mlx_stream sh, int M, int GS) {
    mlx_array r; r.ctx = nullptr;
    LlmMetalMppQ4Ctx* c = g_mpp_q4_ctx;
    if (!c || !c->ok || !c->tile_ok) return r;
    const int gi = gidx(GS);
    const int mi = (M == 16) ? 0 : (M == 24) ? 1 : (M == 32) ? 2 : -1;
    if (gi < 0 || mi < 0) return r;
    if (!xh.ctx || !wh.ctx || !sch.ctx || !bih.ctx) return r;
    Stream s = (sh.ctx != nullptr) ? *(Stream*)sh.ctx : default_stream(default_device());
    array x = *(array*)xh.ctx, w = *(array*)wh.ctx;
    array sc = *(array*)sch.ctx, bi = *(array*)bih.ctx;
    if (x.ndim() != 2 || x.dtype() != bfloat16 || x.shape(0) != M) return r;
    if (w.dtype() != mlx::core::uint32 || sc.dtype() != bfloat16 || bi.dtype() != bfloat16) return r;
    const int K = x.shape(-1), N = w.shape(0), G = K / GS;
    if (K % GS || N % 128 || w.ndim() != 2) return r;
    if ((int)w.shape(1) != K / 8 || (int)sc.shape(0) != N || (int)sc.shape(1) != G) return r;
    Q4TParams p{};
    p.input_size = (uint32_t)K; p.output_size = (uint32_t)N;
    p.wstride = (uint32_t)K;                 /* nibble fra colonne = K (layout MLX) */
    p.wgroup_adv = (uint32_t)(GS / 2);       /* byte fra blocchi di gruppo */
    p.wtile_base = (uint32_t)((size_t)128 * K / 2); /* byte per tile di 128 colonne */
    p.s_a = (uint32_t)(128 * G); p.s_b = 1; p.s_c = (uint32_t)G;
    Shape os = x.shape();
    os.back() = N;
    array out(os, bfloat16, std::make_shared<MppQ4TilePrim>(mi, gi, M, p, c->tile_ps[mi][gi], s),
              {x, w, sc, bi});
    r.ctx = new array(std::move(out));
    return r;
}
