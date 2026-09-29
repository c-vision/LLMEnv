/* llm_mlx_splash_attn_mpp.h — read paged int8 via MPP matmul2d (port fedele di
 * kernels/common/paged_attention_tile.h per il nostro host mlx_fast_metal_kernel).
 *
 * QK: scores[kSplashRowsQ,kSplashToksP] = Q[kSplashRowsQ,D] @ K[kSplashToksP,D]^T  (matmul2d)
 *     score *= key_scale  (dequant-folded, come Splash)
 * PV: out[kSplashRowsQ,D] = softmax(scores)[kSplashRowsQ,kSplashToksP] @ V[kSplashToksP,D]  (matmul2d, V scalato per value_scale)
 *
 * kSplashRowsQ = righe query (tile), kSplashToksP = 32 token/pagina, D = head_dim.
 * Pattern MPP validato dal probe: tensor<threadgroup half,...> + .template.
 */
#ifndef LLM_MLX_SPLASH_ATTN_MPP_H
#define LLM_MLX_SPLASH_ATTN_MPP_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdint.h>

/* q [kSplashRowsQ, D] f32 ; kpool int8 [nkv, cap, D] ; vpool int8 [nkv, hd, cap] ;
 * ks [nkv, cap] f32 (key/value scale, condivisa come Splash) ; nc, capi ; attn. */
static const char llm_splash_attn_mpp_src[] =
    "  const uint hd = q_shape[q_ndim - 1];\n"
    "  const uint t0 = (nc > kSplashToksP) ? (nc - kSplashToksP) : 0u;   /* pagina visibile corrente */\n"
    "  const uint kh = 0u;\n"
    "  threadgroup float qstage[kSplashRowsQ * Dhead];\n"
    "  threadgroup half kstage[kSplashToksP * Dhead];\n"
    "  threadgroup half vstage[kSplashToksP * Dhead];\n"
    "  threadgroup float sstage[kSplashRowsQ * kSplashToksP];\n"
    "  threadgroup half pstage[kSplashRowsQ * kSplashToksP];\n"
    "  for (uint i = thread_index_in_threadgroup; i < kSplashToksP * hd; i += threads_per_threadgroup.x) {\n"
    "    uint t = t0 + i / hd, d = i % hd;\n"
    "    kstage[i] = (half)((float)kpool[(kh*capi + t)*hd + d] * ks[kh*capi + t]);\n"
    "    vstage[i] = (half)((float)vpool[(kh*hd + d)*capi + t] * ks[kh*capi + t]);\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* QK: a = Q [kSplashRowsQ,hd] device ; b = K [kSplashToksP,hd] threadgroup */\n"
    "  auto a = tensor(qstage, dextents<int,2>{Dhead,kSplashRowsQ}, array<int,2>{1,(int)hd});\n"
    "  auto a0 = a.slice<Dhead,kSplashRowsQ>(0,0);\n"
    "  tensor<threadgroup half, dextents<int,2>, tensor_inline> bkt(kstage, dextents<int,2>{Dhead,kSplashToksP}, array<int,2>{1,(int)hd});\n"
    "  auto b0 = bkt.slice<Dhead,kSplashToksP>(0,0);\n"
    "  constexpr auto qk_desc = matmul2d_descriptor(kSplashRowsQ,kSplashToksP,Dhead,false,true,false, matmul2d_descriptor::mode::multiply);\n"
    "  matmul2d<qk_desc, execution_simdgroups<1>> qkop;\n"
    "  auto sc = qkop.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n"
    "  qkop.run(a0, b0, sc);\n"
    "  /* scores -> threadgroup float [kSplashRowsQ,kSplashToksP] (con attn_scale) */\n"
    "  {\n"
    "    auto st = tensor(sstage, dextents<int,2>{kSplashToksP,kSplashRowsQ}, array<int,2>{1,(int)kSplashToksP});\n"
    "    sc.store(st.slice<kSplashToksP,kSplashRowsQ>(0,0));\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* softmax per riga (attn_scale) -> pstage half */\n"
    "  for (uint row = 0; row < kSplashRowsQ; ++row) {\n"
    "    if (thread_index_in_threadgroup == 0) {\n"
    "      float mx = -INFINITY;\n"
    "      for (uint t = 0; t < kSplashToksP; ++t) { float s = sstage[row*kSplashToksP+t] * attn_scale; sstage[row*kSplashToksP+t] = s; if (s > mx) mx = s; }\n"
    "      float sum = 0.0f;\n"
    "      for (uint t = 0; t < kSplashToksP; ++t) { float e = exp(sstage[row*kSplashToksP+t] - mx); sstage[row*kSplashToksP+t] = e; sum += e; }\n"
    "      float inv = sum > 0.0f ? 1.0f / sum : 0.0f;\n"
    "      for (uint t = 0; t < kSplashToksP; ++t) pstage[row*kSplashToksP+t] = (half)(sstage[row*kSplashToksP+t] * inv);\n"
    "    }\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* PV: a = P [kSplashRowsQ,kSplashToksP] threadgroup half ; b = V [kSplashToksP,hd] threadgroup half */\n"
    "  tensor<threadgroup half, dextents<int,2>, tensor_inline> pht(pstage, dextents<int,2>{kSplashToksP,kSplashRowsQ}, array<int,2>{1,(int)kSplashToksP});\n"
    "  auto pa0 = pht.slice<kSplashToksP,kSplashRowsQ>(0,0);\n"
    "  tensor<threadgroup half, dextents<int,2>, tensor_inline> bvt(vstage, dextents<int,2>{Dhead,kSplashToksP}, array<int,2>{1,(int)hd});\n"
    "  auto bv0 = bvt.slice<Dhead,kSplashToksP>(0,0);\n"
    "  constexpr auto pv_desc = matmul2d_descriptor(kSplashRowsQ,Dhead,kSplashToksP,false,true,false, matmul2d_descriptor::mode::multiply);\n"
    "  matmul2d<pv_desc, execution_simdgroups<1>> pvop;\n"
    "  auto o = pvop.template get_destination_cooperative_tensor<decltype(pa0), decltype(bv0), float>();\n"
    "  pvop.run(pa0, bv0, o);\n"
    "  auto ot = tensor(out, dextents<int,2>{Dhead,kSplashRowsQ}, array<int,2>{1,(int)hd});\n"
    "  o.store(ot.slice<Dhead,kSplashRowsQ>(0,0));\n";

static mlx_fast_metal_kernel g_splash_attn_mpp; static int g_splash_attn_mpp_built = 0;
static inline int llm_splash_attn_mpp_build(void) {
    if (g_splash_attn_mpp_built) return g_splash_attn_mpp.ctx ? 0 : -1;
    g_splash_attn_mpp_built = 1;
    const char* hdr = "#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                      "#define kSplashRowsQ 8\n#define kSplashToksP 32\n"
                      "using namespace metal;\nusing namespace mpp::tensor_ops;\n";
    mlx_vector_string iv = mlx_vector_string_new();
    const char* ins[] = {"q","kpool","vpool","ks","nc","capi","attn_scale"};
    for (int i = 0; i < 7; i++) mlx_vector_string_append_value(iv, ins[i]);
    mlx_vector_string ov = mlx_vector_string_new(); mlx_vector_string_append_value(ov, "out");
    g_splash_attn_mpp = mlx_fast_metal_kernel_new("splash_attn_mpp", iv, ov, llm_splash_attn_mpp_src, hdr, true, false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    return g_splash_attn_mpp.ctx ? 0 : -1;
}

/* q [kSplashRowsQ,hd] f32; kpool int8 [nkv,cap,hd]; vpool int8 [nkv,hd,cap]; ks f32 [nkv,cap]; out [kSplashRowsQ,hd]. */
static inline int llm_splash_attn_mpp_run(mlx_array* out, mlx_array q, mlx_array kpool, mlx_array vpool,
                                          mlx_array ks, int nc, int cap, float attn_scale, mlx_stream s) {
    if (llm_splash_attn_mpp_build() != 0) return -1;
    int nrows = (int)mlx_array_shape(q)[0], hdv = (int)mlx_array_shape(q)[1];
    mlx_array ncA = mlx_array_new_int(nc), capA = mlx_array_new_int(cap), asA = mlx_array_new_float(attn_scale);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, q); mlx_vector_array_append_value(in, kpool);
    mlx_vector_array_append_value(in, vpool); mlx_vector_array_append_value(in, ks);
    mlx_vector_array_append_value(in, ncA); mlx_vector_array_append_value(in, capA);
    mlx_vector_array_append_value(in, asA);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dhead", hdv);
    int osh[2] = {nrows, hdv}; mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    int threads = hdv > 64 ? hdv : 64;
    mlx_fast_metal_kernel_config_set_grid(cfg, threads, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_splash_attn_mpp, in, cfg, s);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok) { mlx_array a0 = mlx_array_new(); ok = mlx_vector_array_get(&a0, outs, 0) == 0; if (ok) mlx_array_set(out, a0); mlx_array_free(a0); }
    mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(in); mlx_vector_array_free(outs);
    mlx_array_free(ncA); mlx_array_free(capA); mlx_array_free(asA);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_SPLASH_ATTN_MPP_H */
