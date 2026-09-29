/* llm_mlx_splash_store.h — store/quant in scrittura di Splash, RESO ESEGUIBILE.
 *
 * Trascrizione fedele di kernels/common/paged_store_row.h (`splash_store_kv_row`,
 * ramo INT8): massimo |value| ridotto sul head_dim, scala simmetrica = max/127,
 * quant = clamp(rint(value*127/max), -127, 127). Qui e' montato come
 * mlx_fast_metal_kernel (il nostro host), senza cambiare la matematica.
 *
 * Nota: Splash scrive direttamente nella pagina del pool tramite page_table;
 * questa versione produce la riga quantizzata (data + scala) — la scrittura nel
 * pool e' il passo successivo (page_table), NON sostituito.
 */
#ifndef LLM_MLX_SPLASH_STORE_H
#define LLM_MLX_SPLASH_STORE_H

#include "mlx/c/mlx.h"
#include <stdbool.h>
#include <stdint.h>

/* Un threadgroup per (kv-head, token): i thread sono le dimensioni (head_dim). */
static const char llm_splash_store_src[] =
    "    threadgroup float sh[1024];\n"
    "    const uint tg = threadgroup_position_in_grid.x;\n"
    "    const uint lane = thread_position_in_threadgroup.x;\n"
    "    const uint tgsz = threads_per_threadgroup.x;\n"
    "    uint hd = k_shape[k_ndim - 1];\n"
    "  float km = lane < hd ? fabs((float)k[tg*hd+lane]) : 0.0f;\n"
    "  float vm = lane < hd ? fabs((float)v[tg*hd+lane]) : 0.0f;\n"
    "  sh[lane] = fmax(km, vm);\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  for (uint s = tgsz/2; s > 0; s >>= 1) { if (lane < s) sh[lane] = fmax(sh[lane], sh[lane+s]);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "  float mx = sh[0];\n"
    "  if (lane < hd) {\n"
    "    kq[tg*hd+lane] = (char)(mx == 0.0f ? 0 : clamp((int)rint((float)k[tg*hd+lane]*127.0f/mx), -127, 127));\n"
    "    vq[tg*hd+lane] = (char)(mx == 0.0f ? 0 : clamp((int)rint((float)v[tg*hd+lane]*127.0f/mx), -127, 127));\n"
    "  }\n"
    "  if (lane == 0) { ks[tg] = mx == 0.0f ? 0.0f : mx/127.0f; vs[tg] = ks[tg]; }\n";

static mlx_fast_metal_kernel g_splash_store; static int g_splash_store_built = 0;
static inline int llm_splash_store_build(void) {
    if (g_splash_store_built) return g_splash_store.ctx ? 0 : -1;
    g_splash_store_built = 1;
    const char* hdr = "#include <metal_stdlib>\nusing namespace metal;\n";
    mlx_vector_string iv = mlx_vector_string_new();
    mlx_vector_string_append_value(iv, "k"); mlx_vector_string_append_value(iv, "v");
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, "kq"); mlx_vector_string_append_value(ov, "vq");
    mlx_vector_string_append_value(ov, "ks"); mlx_vector_string_append_value(ov, "vs");
    g_splash_store = mlx_fast_metal_kernel_new("splash_store", iv, ov, llm_splash_store_src, hdr,
                                               /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    return g_splash_store.ctx ? 0 : -1;
}

/* k,v: [nkv, n, hd]. Ritorna kq,vq int8 [nkv,n,hd] + ks,vs f32 [nkv,n]. */
static inline int llm_splash_store_run(mlx_array* kq, mlx_array* vq, mlx_array* ks, mlx_array* vs,
                                       mlx_array k, mlx_array v, mlx_stream s) {
    if (llm_splash_store_build() != 0) return -1;
    int nkv = (int)mlx_array_shape(k)[0], n = (int)mlx_array_shape(k)[1], hd = (int)mlx_array_shape(k)[2];
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, k); mlx_vector_array_append_value(in, v);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx_array_dtype(k));
    int qsh[3] = {nkv, n, hd}, ssh[2] = {nkv, n};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, qsh, 3, MLX_INT8);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, qsh, 3, MLX_INT8);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ssh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, ssh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (nkv * n) * hd, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, hd, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_splash_store, in, cfg, s);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 4;
    if (ok) {
        mlx_array a0 = mlx_array_new(), a1 = mlx_array_new(), a2 = mlx_array_new(), a3 = mlx_array_new();
        ok = mlx_vector_array_get(&a0, outs, 0) == 0 && mlx_vector_array_get(&a1, outs, 1) == 0 &&
             mlx_vector_array_get(&a2, outs, 2) == 0 && mlx_vector_array_get(&a3, outs, 3) == 0;
        if (ok) { mlx_array_set(kq, a0); mlx_array_set(vq, a1); mlx_array_set(ks, a2); mlx_array_set(vs, a3); }
        mlx_array_free(a0); mlx_array_free(a1); mlx_array_free(a2); mlx_array_free(a3);
    }
    mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(in); mlx_vector_array_free(outs);
    return ok ? 0 : -1;
}

/* ── Scrittura nel pool paginato (indici Splash fedeli) ─────────────────────
 * Le righe int8 prodotte dallo store vanno nel pool: K token-major
 * [nkv, cap, hd], V dimension-major [nkv, hd, cap], scale [nkv, cap].
 * La page_table (logical->physical) e' il passo successivo: qui il pool e'
 * contiguo (page_table identita'), che per gli indici di Splash equivale a
 * [head][token_globale]. */
static inline int llm_splash_pool_scatter(mlx_array kpool, mlx_array vpool, mlx_array spool,
                                          mlx_array kq, mlx_array vq, mlx_array sc,
                                          int pos, int n, int nkv, int hd, mlx_stream s) {
    /* K: [nkv, pos:pos+n, hd] */
    int k0[3] = {0,pos,0}, k1[3] = {nkv,pos+n,hd}, st[3] = {1,1,1};
    mlx_array nk = mlx_array_new();
    bool ok = mlx_slice_update(&nk, kpool, kq, k0,3,k1,3,st,3, s) == 0;
    /* V dimension-major: vq [nkv,n,hd] -> [nkv,hd,n] */
    mlx_array vqt = mlx_array_new();
    if (ok) { int perm[3] = {0,2,1}; ok = mlx_transpose_axes(&vqt, vq, perm, 3, s) == 0; }
    int v0[3] = {0,0,pos}, v1[3] = {nkv,hd,pos+n};
    mlx_array nv = mlx_array_new();
    if (ok) ok = mlx_slice_update(&nv, vpool, vqt, v0,3,v1,3,st,3, s) == 0;
    /* scale [nkv, pos:pos+n] */
    mlx_array s3 = mlx_array_new(), ns = mlx_array_new();
    if (ok) { int sh[3] = {nkv,n,1}; ok = mlx_reshape(&s3, sc, sh, 3, s) == 0; }
    int p0[3] = {0,pos,0}, p1[3] = {nkv,pos+n,1};
    if (ok) ok = mlx_slice_update(&ns, spool, s3, p0,3,p1,3,st,3, s) == 0;
    if (ok) { mlx_array_set(&kpool, nk); mlx_array_set(&vpool, nv); mlx_array_set(&spool, ns); }
    mlx_array_free(nk); mlx_array_free(nv); mlx_array_free(vqt); mlx_array_free(s3); mlx_array_free(ns);
    return ok ? 0 : -1;
}

#endif /* LLM_MLX_SPLASH_STORE_H */