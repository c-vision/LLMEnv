/* qwen4_native_indexer_kernel.h — P5b: indexer QSA (select a blocchi)
 * (port di ds4 `qwen4_ref_select` / `kernel_qwen4_idx_block_key` + `_idx_score`).
 *
 * Fasi (testo, pos3 = pos):
 *  - idx_q:   qi [T][Hi*Di] -> RMS(gq) + RoPE per head (un thread per (tok,h)).
 *  - idx_scr: per (tok, blocco) pool ratio righe di idx_k, RMS(gk), RoPE alla posizione
 *             del blocco, score = somma su Hi di max(dot,0). Score [T][nblocks].
 *  - idx_sel: se nblocks <= kblocks -> tutti i token visibili; altrimenti top-kblocks
 *             blocchi (indice piu' basso a parita'), poi tutti i token dei blocchi
 *             selezionati in ordine crescente + coda [nblocks*ratio, nvis).
 *             sel [T][nvis], nsel [T].
 */
#ifndef LLM_QWEN4_NATIVE_INDEXER_KERNEL_H
#define LLM_QWEN4_NATIVE_INDEXER_KERNEL_H

#include "mlx/c/mlx.h"

static const char llm_q4n_idx_header[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n";

/* ---- idx_q ---- */
static const char llm_q4n_idx_q_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint Hi = uint(nhead), Di = uint(hdim), T = uint(ntok), R = uint(nrot);\n"
    "if (i >= T*Hi || R > Di || (R & 1u)) return;\n"
    "uint tok = i / Hi, h = i % Hi;\n"
    "ulong b = ((ulong)tok*Hi + h)*Di;\n"
    "float ss = 0.f;\n"
    "for (uint d = 0; d < Di; d++) ss += qi[b+d]*qi[b+d];\n"
    "float r = metal::rsqrt(ss/(float)Di + eps);\n"
    "float tmp[256];\n"
    "for (uint d = 0; d < Di; d++) tmp[d] = qi[b+d]*r*gq[d];\n"
    "uint nrh = R/2u;\n"
    "for (uint j = 0; j < nrh; j++) {\n"
    "  float th = (float)pos * freq[j];\n"
    "  float c = metal::cos(th)*mscale, sn = metal::sin(th)*mscale;\n"
    "  float x0 = tmp[j], x1 = tmp[j+nrh];\n"
    "  tmp[j] = x0*c - x1*sn; tmp[j+nrh] = x0*sn + x1*c;\n"
    "}\n"
    "for (uint d = 0; d < Di; d++) qi_o[b+d] = tmp[d];\n";

static mlx_fast_metal_kernel g_q4n_idx_q;
static int g_q4n_idx_q_built = 0;

static int llm_q4n_idx_q_build(void) {
    if (g_q4n_idx_q_built) return 0;
    const char* in_names[]  = {"qi","gq","freq","ntok","nhead","hdim","nrot","pos","mscale","eps"};
    const char* out_names[] = {"qi_o"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 10; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_idx_q = mlx_fast_metal_kernel_new(
        "qwen4_idx_q_native", iv, ov, llm_q4n_idx_q_src, llm_q4n_idx_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_idx_q_built = (g_q4n_idx_q.ctx != NULL) ? 1 : 0;
    return g_q4n_idx_q_built ? 0 : -1;
}

static int llm_q4n_idx_q(mlx_array* qi_o, mlx_array qi, mlx_array gq, mlx_array freq,
                         int n_rot, int pos, float mscale, float eps, mlx_stream s) {
    if (!qi_o || mlx_array_ndim(qi) != 2 || mlx_array_ndim(gq) != 1 || mlx_array_ndim(freq) != 1) return -1;
    const int* qs = mlx_array_shape(qi);
    const int* gs = mlx_array_shape(gq);
    const int* fs = mlx_array_shape(freq);
    const int T = qs[0], hd = qs[1], Di = gs[0];
    if (Di <= 0 || hd % Di != 0) return -1;
    const int Hi = hd / Di;
    if (n_rot <= 0 || n_rot > Di || (n_rot & 1) || fs[0] != n_rot / 2) return -1;
    if (mlx_array_dtype(qi) != MLX_FLOAT32 || mlx_array_dtype(gq) != MLX_FLOAT32 ||
        mlx_array_dtype(freq) != MLX_FLOAT32) return -1;
    if (llm_q4n_idx_q_build() != 0) return -1;
    int ntok = T, nhead = Hi, hdim = Di, nrot = n_rot, ipos = pos;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhead, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_array ar = mlx_array_new_data(&nrot, NULL, 0, MLX_INT32);
    mlx_array ap = mlx_array_new_data(&ipos, NULL, 0, MLX_INT32);
    mlx_array am = mlx_array_new_float32(mscale);
    mlx_array ae = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, qi);
    mlx_vector_array_append_value(in, gq);
    mlx_vector_array_append_value(in, freq);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ad);
    mlx_vector_array_append_value(in, ar);
    mlx_vector_array_append_value(in, ap);
    mlx_vector_array_append_value(in, am);
    mlx_vector_array_append_value(in, ae);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {T, Hi * Di};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)Hi, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_idx_q, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(qi_o, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(ad); mlx_array_free(ar);
    mlx_array_free(ap); mlx_array_free(am); mlx_array_free(ae);
    return rc == 0 ? 0 : -1;
}

/* ---- idx_scr ---- */
static const char llm_q4n_idx_scr_src[] =
    "\n"
    "uint i = uint(thread_position_in_grid.x);\n"
    "uint Di = uint(hdim), Hi = uint(nhead), T = uint(ntok), NB = uint(nblock), Rt = uint(ratio), R = uint(nrot);\n"
    "if (i >= T*NB || Rt == 0u) return;\n"
    "uint tok = i / NB, blk = i % NB;\n"
    "float pooled[256], key[256], ss = 0.f;\n"
    "for (uint d = 0; d < Di; d++) {\n"
    "  float acc = 0.f;\n"
    "  for (uint t = 0; t < Rt; t++) acc += idx_k[(ulong)(blk*Rt + t)*Di + d];\n"
    "  pooled[d] = acc/(float)Rt;\n"
    "}\n"
    "for (uint d = 0; d < Di; d++) ss += pooled[d]*pooled[d];\n"
    "float r = metal::rsqrt(ss/(float)Di + eps);\n"
    "for (uint d = 0; d < Di; d++) key[d] = pooled[d]*r*gk[d];\n"
    "uint nrh = R/2u;\n"
    "uint ipos = blk*Rt;\n"
    "for (uint j = 0; j < nrh; j++) {\n"
    "  float th = (float)ipos * freq[j];\n"
    "  float c = metal::cos(th)*mscale, sn = metal::sin(th)*mscale;\n"
    "  float x0 = key[j], x1 = key[j+nrh];\n"
    "  key[j] = x0*c - x1*sn; key[j+nrh] = x0*sn + x1*c;\n"
    "}\n"
    "float sum = 0.f;\n"
    "for (uint h = 0; h < Hi; h++) {\n"
    "  ulong qb = ((ulong)tok*Hi + h)*Di;\n"
    "  float dot = 0.f;\n"
    "  for (uint d = 0; d < Di; d++) dot += qi[qb+d]*key[d];\n"
    "  if (dot > 0.f) sum += dot;\n"
    "}\n"
    "score[(ulong)tok*NB + blk] = sum;\n";

static mlx_fast_metal_kernel g_q4n_idx_scr;
static int g_q4n_idx_scr_built = 0;

static int llm_q4n_idx_scr_build(void) {
    if (g_q4n_idx_scr_built) return 0;
    const char* in_names[]  = {"qi","idx_k","gk","freq","ntok","nhead","hdim","nblock","ratio","nrot","mscale","eps"};
    const char* out_names[] = {"score"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 12; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_q4n_idx_scr = mlx_fast_metal_kernel_new(
        "qwen4_idx_scr_native", iv, ov, llm_q4n_idx_scr_src, llm_q4n_idx_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_idx_scr_built = (g_q4n_idx_scr.ctx != NULL) ? 1 : 0;
    return g_q4n_idx_scr_built ? 0 : -1;
}

/* qi:[T][Hi*Di], idx_k:[nvis][Di], gk:[Di] -> score [T][nvis/ratio]. */
static int llm_q4n_idx_scr(mlx_array* score, mlx_array qi, mlx_array idx_k, mlx_array gk, mlx_array freq,
                           int ratio, int n_rot, float mscale, float eps, mlx_stream s) {
    if (!score || mlx_array_ndim(qi) != 2 || mlx_array_ndim(idx_k) != 2 ||
        mlx_array_ndim(gk) != 1 || mlx_array_ndim(freq) != 1) return -1;
    const int* qs = mlx_array_shape(qi);
    const int* ks = mlx_array_shape(idx_k);
    const int Di = mlx_array_shape(gk)[0];
    if (Di <= 0 || qs[1] % Di != 0 || ks[1] != Di || ratio <= 0) return -1;
    const int T = qs[0], nvis = ks[0], Hi = qs[1] / Di, nb = nvis / ratio;
    if (nb <= 0 || nb > 2048) return -1;
    if (n_rot <= 0 || n_rot > Di || (n_rot & 1) || mlx_array_shape(freq)[0] != n_rot / 2) return -1;
    if (llm_q4n_idx_scr_build() != 0) return -1;
    int ntok = T, nhead = Hi, hdim = Di, nblock = nb, rt = ratio, nrot = n_rot;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ah = mlx_array_new_data(&nhead, NULL, 0, MLX_INT32);
    mlx_array ad = mlx_array_new_data(&hdim, NULL, 0, MLX_INT32);
    mlx_array ab = mlx_array_new_data(&nblock, NULL, 0, MLX_INT32);
    mlx_array ar = mlx_array_new_data(&rt, NULL, 0, MLX_INT32);
    mlx_array anr = mlx_array_new_data(&nrot, NULL, 0, MLX_INT32);
    mlx_array am = mlx_array_new_float32(mscale);
    mlx_array ae = mlx_array_new_float32(eps);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, qi);
    mlx_vector_array_append_value(in, idx_k);
    mlx_vector_array_append_value(in, gk);
    mlx_vector_array_append_value(in, freq);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ah);
    mlx_vector_array_append_value(in, ad);
    mlx_vector_array_append_value(in, ab);
    mlx_vector_array_append_value(in, ar);
    mlx_vector_array_append_value(in, anr);
    mlx_vector_array_append_value(in, am);
    mlx_vector_array_append_value(in, ae);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int osh[2] = {T, nb};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 2, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T * (size_t)nb, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_idx_scr, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 1) {
        mlx_array r0 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0) mlx_array_set(score, r0); else rc = -1;
        mlx_array_free(r0);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ah); mlx_array_free(ad); mlx_array_free(ab);
    mlx_array_free(ar); mlx_array_free(anr); mlx_array_free(am); mlx_array_free(ae);
    return rc == 0 ? 0 : -1;
}

/* ---- idx_sel ---- */
static const char llm_q4n_idx_sel_src[] =
    "\n"
    "uint tok = uint(thread_position_in_grid.x);\n"
    "uint T = uint(ntok), NB = uint(nblock), KB = uint(kblock), Rt = uint(ratio), NV = uint(nvis);\n"
    "if (tok >= T || NB == 0u || NB > 2048u) return;\n"
    "ulong sb = (ulong)tok*NB;\n"
    "uint n = 0u;\n"
    "if (NB <= KB) {\n"
    "  for (uint t = 0; t < NV; t++) sel[(ulong)tok*NV + n++] = (int)t;\n"
    "  nsel[tok] = (int)n;\n"
    "  return;\n"
    "}\n"
    "uchar taken[2048];\n"
    "for (uint b = 0; b < NB; b++) taken[b] = 0u;\n"
    "for (uint i = 0; i < KB; i++) {\n"
    "  int best = -1; float bv = -3.0e38f;\n"
    "  for (uint b = 0; b < NB; b++) {\n"
    "    if (taken[b]) continue;\n"
    "    if (best < 0 || score[sb+b] > bv) { bv = score[sb+b]; best = (int)b; }\n"
    "  }\n"
    "  if (best < 0) break;\n"
    "  taken[best] = 1u;\n"
    "}\n"
    "for (uint b = 0; b < NB; b++) {\n"
    "  if (!taken[b]) continue;\n"
    "  for (uint t = 0; t < Rt; t++) sel[(ulong)tok*NV + n++] = (int)(b*Rt + t);\n"
    "}\n"
    "for (uint t = NB*Rt; t < NV; t++) sel[(ulong)tok*NV + n++] = (int)t;\n"
    "nsel[tok] = (int)n;\n";

static mlx_fast_metal_kernel g_q4n_idx_sel;
static int g_q4n_idx_sel_built = 0;

static int llm_q4n_idx_sel_build(void) {
    if (g_q4n_idx_sel_built) return 0;
    const char* in_names[]  = {"score","ntok","nblock","kblock","ratio","nvis"};
    const char* out_names[] = {"sel","nsel"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 6; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i = 0; i < 2; i++) mlx_vector_string_append_value(ov, out_names[i]);
    g_q4n_idx_sel = mlx_fast_metal_kernel_new(
        "qwen4_idx_sel_native", iv, ov, llm_q4n_idx_sel_src, llm_q4n_idx_header, true, false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    g_q4n_idx_sel_built = (g_q4n_idx_sel.ctx != NULL) ? 1 : 0;
    return g_q4n_idx_sel_built ? 0 : -1;
}

/* score:[T][nblocks] -> sel [T][nvis] i32, nsel [T] i32. */
static int llm_q4n_idx_sel(mlx_array* sel, mlx_array* nsel, mlx_array score,
                           int n_vis, int ratio, int k_blocks, mlx_stream s) {
    if (!sel || !nsel || mlx_array_ndim(score) != 2 || ratio <= 0 || k_blocks <= 0) return -1;
    const int* ss = mlx_array_shape(score);
    const int T = ss[0], nb = ss[1];
    if (nb <= 0 || nb > 2048 || nb != n_vis / ratio) return -1;
    if (mlx_array_dtype(score) != MLX_FLOAT32) return -1;
    if (llm_q4n_idx_sel_build() != 0) return -1;
    int ntok = T, nblock = nb, kb = k_blocks, rt = ratio, nv = n_vis;
    mlx_array at = mlx_array_new_data(&ntok, NULL, 0, MLX_INT32);
    mlx_array ab = mlx_array_new_data(&nblock, NULL, 0, MLX_INT32);
    mlx_array ak = mlx_array_new_data(&kb, NULL, 0, MLX_INT32);
    mlx_array ar = mlx_array_new_data(&rt, NULL, 0, MLX_INT32);
    mlx_array an = mlx_array_new_data(&nv, NULL, 0, MLX_INT32);
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, score);
    mlx_vector_array_append_value(in, at);
    mlx_vector_array_append_value(in, ab);
    mlx_vector_array_append_value(in, ak);
    mlx_vector_array_append_value(in, ar);
    mlx_vector_array_append_value(in, an);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int selsh[2] = {T, n_vis}, nsh[1] = {T};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, selsh, 2, MLX_INT32);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, nsh, 1, MLX_INT32);
    mlx_fast_metal_kernel_config_set_grid(cfg, (size_t)T, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 1, 1, 1);
    mlx_vector_array outs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outs, g_q4n_idx_sel, in, cfg, s);
    if (rc == 0 && mlx_vector_array_size(outs) == 2) {
        mlx_array r0 = mlx_array_new(), r1 = mlx_array_new();
        if (mlx_vector_array_get(&r0, outs, 0) == 0 && mlx_vector_array_get(&r1, outs, 1) == 0) {
            mlx_array_set(sel, r0); mlx_array_set(nsel, r1);
        } else rc = -1;
        mlx_array_free(r0); mlx_array_free(r1);
    } else rc = -1;
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(outs);
    mlx_array_free(at); mlx_array_free(ab); mlx_array_free(ak); mlx_array_free(ar); mlx_array_free(an);
    return rc == 0 ? 0 : -1;
}

#endif /* LLM_QWEN4_NATIVE_INDEXER_KERNEL_H */
