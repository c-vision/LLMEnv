/* llm_glm_qkv_probe.c — ORACOLO model-free: le proiezioni q/k/v di GLM conviene fonderle in
 * UNA quantized_matmul ([12288+1024+1024=14336, 4096]) invece di 3 separate a M=1? (k/v da
 * 2,1 MB sono latency-bound). Passo-token 46 layer + UNA eval, come l'engine.
 *   A sep     : qmm(q) + qmm(k) + qmm(v) + qmm(o)
 *   B fused   : qmm(qkv [14336,4096]) + split + contiguous + qmm(o)
 *   C fused_nc: come B senza contiguous
 * Parità vs A + mediana interleaved + stima ms/token.
 * Uso: llm_glm_qkv_probe [reps]   (default 15)
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H 4096
#define NQ 12288
#define NKV 1024
#define GS 64
#define BITS 4
#define MODE "affine"
#define NL 46

static mlx_stream g_st;
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static uint32_t g_rng = 0x2f9a4c11u;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static double median(double* v, int n) {
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

typedef struct { int N, K; mlx_array w, sc, bi; uint32_t* wpk; uint16_t* sch, *bih; } Wq;
static void wq_build(Wq* q, int N, int K) {
    q->N = N; q->K = K;
    const long pr = (long)K * BITS / 32, ng = K / GS;
    q->wpk = malloc((size_t)N * pr * 4);
    q->sch = malloc((size_t)N * ng * 2);
    q->bih = malloc((size_t)N * ng * 2);
    for (long i = 0; i < (long)N * pr; i++) q->wpk[i] = xr();
    for (long i = 0; i < (long)N * ng; i++) {
        q->sch[i] = (uint16_t)(0x3C00 | (xr() & 0x00FF));
        q->bih[i] = (uint16_t)((xr() & 1 ? 0x8000 : 0) | 0x3F00 | (xr() & 0x007F));
    }
    const int ws[2] = {N, (int)pr}, ss[2] = {N, (int)ng};
    q->w = mlx_array_new_data(q->wpk, ws, 2, MLX_UINT32);
    q->sc = mlx_array_new_data(q->sch, ss, 2, MLX_BFLOAT16);
    q->bi = mlx_array_new_data(q->bih, ss, 2, MLX_BFLOAT16);
}
static void wq_free(Wq* q) { mlx_array_free(q->w); mlx_array_free(q->sc); mlx_array_free(q->bi); free(q->wpk); free(q->sch); free(q->bih); }
static int qlin(mlx_array* out, mlx_array x, const Wq* q) {
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    return mlx_quantized_matmul(out, x, q->w, q->sc, q->bi, true, gsz, bts, MODE, g_st);
}
static double maxdiff(mlx_array a, mlx_array b) {
    mlx_array a32 = mlx_array_new(), b32 = mlx_array_new(), d = mlx_array_new(), ad = mlx_array_new(), mx = mlx_array_new();
    mlx_astype(&a32, a, MLX_FLOAT32, g_st); mlx_astype(&b32, b, MLX_FLOAT32, g_st);
    mlx_subtract(&d, a32, b32, g_st); mlx_abs(&ad, d, g_st); mlx_max(&mx, ad, false, g_st);
    float v = 0; mlx_array_item_float32(&v, mx);
    mlx_array_free(a32); mlx_array_free(b32); mlx_array_free(d); mlx_array_free(ad); mlx_array_free(mx);
    return v;
}

/* concatena i tre pesi (stesso K) in un Wq [NQ+2*NKV, K] */
static Wq wq_qkv(const Wq* q, const Wq* k, const Wq* v) {
    Wq o; o.N = q->N + k->N + v->N; o.K = q->K;
    const long pr = (long)o.K * BITS / 32, ng = o.K / GS;
    o.wpk = malloc((size_t)o.N * pr * 4);
    o.sch = malloc((size_t)o.N * ng * 2);
    o.bih = malloc((size_t)o.N * ng * 2);
    const Wq* src[3] = {q, k, v}; long off = 0;
    for (int i = 0; i < 3; i++) {
        memcpy(&o.wpk[off * pr], src[i]->wpk, (size_t)src[i]->N * pr * 4);
        memcpy(&o.sch[off * ng], src[i]->sch, (size_t)src[i]->N * ng * 2);
        memcpy(&o.bih[off * ng], src[i]->bih, (size_t)src[i]->N * ng * 2);
        off += src[i]->N;
    }
    const int ws[2] = {o.N, (int)pr}, ss[2] = {o.N, (int)ng};
    o.w = mlx_array_new_data(o.wpk, ws, 2, MLX_UINT32);
    o.sc = mlx_array_new_data(o.sch, ss, 2, MLX_BFLOAT16);
    o.bi = mlx_array_new_data(o.bih, ss, 2, MLX_BFLOAT16);
    return o;
}

/* una layer: esegue la forma e APPENDE le uscite (q,k,v,o) nel vettore del chiamante. */
static int layer_run(int form, int li, mlx_vector_array* outs, mlx_array x, const Wq* wq_,
                     const Wq* wk, const Wq* wv, const Wq* wqkv, const Wq* wo) {
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    int rc = 0;
    if (form == 0) {
        rc = qlin(&q, x, &wq_[li]);
        if (rc == 0) rc = qlin(&k, x, &wk[li]);
        if (rc == 0) rc = qlin(&v, x, &wv[li]);
    } else {
        mlx_array qkv = mlx_array_new();
        rc = qlin(&qkv, x, &wqkv[li]);
        if (rc == 0) {
            /* split NON uguale -> 3 slice: q [0,12288), k [12288,13312), v [13312,14336) */
            const int s0[3] = {0, 0, 0}, s1[3] = {0, 0, NQ}, s2[3] = {0, 0, NQ + NKV};
            const int e0[3] = {1, 1, NQ}, e1[3] = {1, 1, NQ + NKV}, e2[3] = {1, 1, NQ + 2 * NKV};
            const int sv[3] = {1, 1, 1};
            mlx_array vq = mlx_array_new(), vk = mlx_array_new(), vv = mlx_array_new();
            rc = mlx_slice(&vq, qkv, s0, 3, e0, 3, sv, 3, g_st);
            if (rc == 0) rc = mlx_slice(&vk, qkv, s1, 3, e1, 3, sv, 3, g_st);
            if (rc == 0) rc = mlx_slice(&vv, qkv, s2, 3, e2, 3, sv, 3, g_st);
            if (rc == 0) {
                mlx_array_set(&q, vq); mlx_array_set(&k, vk); mlx_array_set(&v, vv);
                if (form == 1) {
                    mlx_array qc = mlx_array_new(), kc = mlx_array_new(), vc = mlx_array_new();
                    if (mlx_contiguous(&qc, q, false, g_st) == 0 && mlx_contiguous(&kc, k, false, g_st) == 0 &&
                        mlx_contiguous(&vc, v, false, g_st) == 0) {
                        mlx_array_set(&q, qc); mlx_array_set(&k, kc); mlx_array_set(&v, vc);
                    }
                    mlx_array_free(qc); mlx_array_free(kc); mlx_array_free(vc);
                }
            }
            mlx_array_free(vq); mlx_array_free(vk); mlx_array_free(vv);
        }
        mlx_array_free(qkv);
    }
    /* o_proj su una "vista attenzione" di H elementi (stand-in fedele per i byte del peso) */
    mlx_array o = mlx_array_new();
    if (rc == 0) {
        const int st[3] = {0, 0, 0}, sp[3] = {1, 1, H}, sv[3] = {1, 1, 1};
        mlx_array qs = mlx_array_new();
        rc = mlx_slice(&qs, q, st, 3, sp, 3, sv, 3, g_st);
        if (rc == 0) rc = qlin(&o, qs, &wo[li]);
        mlx_array_free(qs);
    }
    if (rc == 0) {
        mlx_vector_array_append_value(*outs, q);
        mlx_vector_array_append_value(*outs, k);
        mlx_vector_array_append_value(*outs, v);
        mlx_vector_array_append_value(*outs, o);
    }
    mlx_array_free(o); mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
    return rc;
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 15;
    if (reps > 100) reps = 100;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    /* 46 set di pesi DISTINTI (come il modello reale: niente cache-hit tra layer) */
    static Wq wq_[NL], wk[NL], wv[NL], wo[NL], wqkv[NL];
    for (int l = 0; l < NL; l++) {
        wq_build(&wq_[l], NQ, H);
        wq_build(&wk[l], NKV, H);
        wq_build(&wv[l], NKV, H);
        wq_build(&wo[l], H, H);
        wqkv[l] = wq_qkv(&wq_[l], &wk[l], &wv[l]);
        mlx_array_eval(wqkv[l].w);
        mlx_array_eval(wo[l].w);
    }

    uint16_t* xh = malloc((size_t)H * 2);
    for (int i = 0; i < H; i++) xh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF));
    const int xs[3] = {1, 1, H};
    mlx_array x = mlx_array_new_data(xh, xs, 3, MLX_BFLOAT16);

    mlx_vector_array ref = mlx_vector_array_new();
    if (layer_run(0, 0, &ref, x, wq_, wk, wv, wqkv, wo) != 0) { printf("A FAIL\n"); return 1; }
    mlx_eval(ref);
    for (int f = 1; f <= 2; f++) {
        mlx_vector_array o = mlx_vector_array_new();
        if (layer_run(f, 0, &o, x, wq_, wk, wv, wqkv, wo) != 0) { printf("form %d FAIL\n", f); return 1; }
        mlx_eval(o);
        double md = 0;
        for (int i = 0; i < 4; i++) {
            mlx_array a = mlx_array_new(), b = mlx_array_new();
            mlx_vector_array_get(&a, ref, i); mlx_vector_array_get(&b, o, i);
            double d = maxdiff(a, b); if (d > md) md = d;
            mlx_array_free(a); mlx_array_free(b);
        }
        printf("parita' forma %s: max|d|=%.3g\n", f == 1 ? "B fused" : "C fused_nc", md);
        mlx_vector_array_free(o);
    }

    const char* names[3] = {"A sep_qkv", "B fused_qkv", "C fused_nc"};
    double ts[3][100];
    for (int r = 0; r < reps + 2; r++) {
        for (int f = 0; f < 3; f++) {
            mlx_vector_array outs = mlx_vector_array_new();
            double t0 = now_ms();
            for (int l = 0; l < NL; l++) {
                if (layer_run(f, l, &outs, x, wq_, wk, wv, wqkv, wo) != 0) { printf("FAIL\n"); return 1; }
            }
            mlx_eval(outs);
            double t1 = now_ms();
            mlx_vector_array_free(outs);
            if (r >= 2) ts[f][r - 2] = t1 - t0;
        }
    }
    double base = median(ts[0], reps);
    printf("llm_glm_qkv_probe — q[%d,%d] k/v[%d,%d] o[%d,%d] 4bit g64, %d layer, M=1, %d reps\n",
           NQ, H, NKV, H, H, H, NL, reps);
    for (int f = 0; f < 3; f++)
        printf("%-12s ms/passo-token=%.4f  vs A=%.2f\n", names[f], median(ts[f], reps), median(ts[f], reps) / base);
    printf("(confronto col costo token GLM: 22 ms — risparmio B=%.3f, C=%.3f ms/token)\n",
           base - median(ts[1], reps), base - median(ts[2], reps));

    mlx_vector_array_free(ref); mlx_array_free(x); free(xh);
    for (int l = 0; l < NL; l++) { wq_free(&wqkv[l]); wq_free(&wo[l]); wq_free(&wv[l]); wq_free(&wk[l]); wq_free(&wq_[l]); }
    mlx_stream_free(g_st); mlx_device_free(dev);
    return 0;
}
