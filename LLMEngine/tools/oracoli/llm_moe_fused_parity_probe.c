/* llm_moe_fused_parity_probe.c — ORACOLO model-free: il kernel MoE fuso di decode
 * (llm_mlx_moe_fused_decode, fast::exp) quanto devia dal path ESATTO (gather_qmm + silu eager)
 * sulle forme di un MoE affine g64 b4? Serve a decidere il Residuo 1 del todo parità:
 * accettare il trade-off (opt-out) documentando la deviazione misurata.
 * Uso: llm_moe_fused_parity_probe [E] [din] [dff] [k]   (default 128 4096 1408 8)
 */
#include "llm_mlx_moe_fused_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GS 64
#define BITS 4

static mlx_stream g_st;
static uint32_t g_rng = 0x6d2b79f5u;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }

typedef struct { int N, K; mlx_array w, sc, bi; uint32_t* wpk; uint16_t* sch, *bih; } Wq;
static void wq_build(Wq* q, int E, int N, int K) {
    q->N = N; q->K = K;
    const long pr = (long)K * BITS / 32, ng = K / GS;
    q->wpk = malloc((size_t)E * N * pr * 4);
    q->sch = malloc((size_t)E * N * ng * 2);
    q->bih = malloc((size_t)E * N * ng * 2);
    for (long i = 0; i < (long)E * N * pr; i++) q->wpk[i] = xr();
    for (long i = 0; i < (long)E * N * ng; i++) {
        q->sch[i] = (uint16_t)(0x3C00 | (xr() & 0x00FF));
        q->bih[i] = (uint16_t)((xr() & 1 ? 0x8000 : 0) | 0x3F00 | (xr() & 0x007F));
    }
    const int ws[3] = {E, N, (int)pr}, ss[3] = {E, N, (int)ng};
    q->w = mlx_array_new_data(q->wpk, ws, 3, MLX_UINT32);
    q->sc = mlx_array_new_data(q->sch, ss, 3, MLX_BFLOAT16);
    q->bi = mlx_array_new_data(q->bih, ss, 3, MLX_BFLOAT16);
}
static void wq_free(Wq* q) { mlx_array_free(q->w); mlx_array_free(q->sc); mlx_array_free(q->bi); free(q->wpk); free(q->sch); free(q->bih); }

static double maxdiff(mlx_array a, mlx_array b, double* refmax) {
    mlx_array a32 = mlx_array_new(), b32 = mlx_array_new(), d = mlx_array_new(), ad = mlx_array_new(), mx = mlx_array_new();
    mlx_astype(&a32, a, MLX_FLOAT32, g_st); mlx_astype(&b32, b, MLX_FLOAT32, g_st);
    mlx_subtract(&d, a32, b32, g_st); mlx_abs(&ad, d, g_st); mlx_max(&mx, ad, false, g_st);
    float v = 0; mlx_array_item_float32(&v, mx);
    if (refmax) { mlx_array r = mlx_array_new(), rm = mlx_array_new();
        mlx_abs(&r, b32, g_st); mlx_max(&rm, r, false, g_st);
        float rv = 0; mlx_array_item_float32(&rv, rm); *refmax = rv; mlx_array_free(r); mlx_array_free(rm); }
    mlx_array_free(a32); mlx_array_free(b32); mlx_array_free(d); mlx_array_free(ad); mlx_array_free(mx);
    return v;
}

int main(int argc, char** argv) {
    const int E = argc > 1 ? atoi(argv[1]) : 128;
    const int din = argc > 2 ? atoi(argv[2]) : 4096;
    const int dff = argc > 3 ? atoi(argv[3]) : 1408;
    const int k = argc > 4 ? atoi(argv[4]) : 8;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    Wq gw, uw, dw;
    wq_build(&gw, E, dff, din);
    wq_build(&uw, E, dff, din);
    wq_build(&dw, E, din, dff);

    uint16_t* xh = malloc((size_t)din * 2);
    for (int i = 0; i < din; i++) xh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF));
    uint32_t* idx = malloc((size_t)k * 4);
    float* sch = malloc((size_t)k * 4);
    for (int i = 0; i < k; i++) { idx[i] = xr() % E; sch[i] = 0.1f + (float)(xr() % 1000) / 2000.0f; }
    const int xs[1] = {din}, is[1] = {k};
    mlx_array x = mlx_array_new_data(xh, xs, 1, MLX_BFLOAT16);
    mlx_array indices = mlx_array_new_data(idx, is, 1, MLX_UINT32);
    mlx_array scores = mlx_array_new_data(sch, is, 1, MLX_FLOAT32);

    /* Riferimento ESATTO = path eager del motore: gather_qmm unsorted + silu + down + somma pesata */
    mlx_array no_lhs = mlx_array_new(), x1 = mlx_array_new(), xb = mlx_array_new();
    mlx_reshape(&x1, x, (const int[]){1, 1, din}, 3, g_st);
    mlx_broadcast_to(&xb, x1, (const int[]){k, 1, din}, 3, g_st);
    mlx_array gate = mlx_array_new(), up = mlx_array_new(), act = mlx_array_new(), dn = mlx_array_new();
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    mlx_gather_qmm(&gate, xb, gw.w, gw.sc, gw.bi, no_lhs, indices, true, gsz, bts, "affine", false, g_st);
    mlx_gather_qmm(&up, xb, uw.w, uw.sc, uw.bi, no_lhs, indices, true, gsz, bts, "affine", false, g_st);
    mlx_array sg = mlx_array_new();
    mlx_sigmoid(&sg, gate, g_st);
    mlx_multiply(&act, gate, sg, g_st);
    mlx_multiply(&act, act, up, g_st);
    mlx_gather_qmm(&dn, act, dw.w, dw.sc, dw.bi, no_lhs, indices, true, gsz, bts, "affine", false, g_st);
    mlx_array sc4 = mlx_array_new(), m4 = mlx_array_new(), s4 = mlx_array_new(), acc = mlx_array_new();
    mlx_reshape(&sc4, scores, (const int[]){k, 1, 1}, 3, g_st);
    mlx_multiply(&m4, dn, sc4, g_st);
    mlx_sum_axis(&s4, m4, 0, true, g_st);
    mlx_reshape(&acc, s4, xs, 1, g_st);
    mlx_array_free(sc4); mlx_array_free(m4); mlx_array_free(s4);
    mlx_array_free(sg); mlx_array_free(dn); mlx_array_free(act); mlx_array_free(up); mlx_array_free(gate);
    mlx_array_free(xb); mlx_array_free(x1); mlx_array_free(no_lhs);
    mlx_array_eval(acc);

    /* kernel FUSO reale (stesso codice dell'engine) */
    mlx_array fused = mlx_array_new();
    bool ok = llm_mlx_moe_fused_decode(&fused, x, indices,
        gw.w, gw.sc, gw.bi, uw.w, uw.sc, uw.bi, dw.w, dw.sc, dw.bi, scores,
        din, dff, k, BITS, GS, BITS, GS, /*act=SwiGLU*/0, g_st);
    printf("llm_moe_fused_parity_probe — E=%d din=%d dff=%d k=%d g%d b%d (x bf16)\n", E, din, dff, k, GS, BITS);
    if (!ok) { printf("kernel fuso NON applicato (fallback) -> nessuna deviazione\n"); return 0; }
    mlx_array_eval(fused);
    double rm = 0, md = maxdiff(fused, acc, &rm);
    printf("max|ref|=%.4g  max|diff|=%.4g  rel=%.3g%%\n", rm, md, rm > 0 ? 100.0 * md / rm : 0.0);
    printf("ESITO: deviazione %s\n", md <= 1e-2 + 5e-2 * rm ? "entro tolleranza bf16 (accettabile, opt-out)" : "FUORI tolleranza");

    mlx_array_free(fused); mlx_array_free(acc);
    mlx_array_free(scores); mlx_array_free(indices); mlx_array_free(x);
    free(sch); free(idx); free(xh);
    wq_free(&dw); wq_free(&uw); wq_free(&gw);
    mlx_stream_free(g_st); mlx_device_free(dev);
    return 0;
}
