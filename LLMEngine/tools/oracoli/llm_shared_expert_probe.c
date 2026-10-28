/* llm_shared_expert_probe.c — ORACOLO model-free §3 GLM: il blocco shared_experts (SwiGLU
 * quantizzato affine g64 b4, gate/up [1408,4096] + down [4096,1408], M=1 decode) conviene
 * fuso (gate_up in una matmul) o con la silu compilata? Confronta passo-token (46 layer
 * in catena + UNA eval, come l'engine) di:
 *   A eager      : qmm(gate) + qmm(up) + sigmoid+mul+mul + qmm(down)   [default decode]
 *   B fused_gu   : qmm(gate_up [2816,4096]) + split + contiguous + silu + qmm(down)
 *   C swiglu_comp: A con la silu in closure compilata (come llm_mlx_swiglu_compiled)
 * Parità vs A + mediana interleaved + stima ms/token.
 * Uso: llm_shared_expert_probe [reps]   (default 15)
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H 4096
#define SH 1408
#define GS 64
#define BITS 4
#define MODE "affine"
#define NL 46

static mlx_stream g_st;
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static uint32_t g_rng = 0x1f123bb5u;
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
static Wq wq_fuse(const Wq* g, const Wq* u) {
    Wq q; q.N = g->N + u->N; q.K = g->K;
    const long pr = (long)q.K * BITS / 32, ng = q.K / GS;
    q.wpk = malloc((size_t)q.N * pr * 4);
    q.sch = malloc((size_t)q.N * ng * 2);
    q.bih = malloc((size_t)q.N * ng * 2);
    memcpy(q.wpk, g->wpk, (size_t)g->N * pr * 4);
    memcpy(&q.wpk[(size_t)g->N * pr], u->wpk, (size_t)u->N * pr * 4);
    memcpy(q.sch, g->sch, (size_t)g->N * ng * 2);
    memcpy(&q.sch[(size_t)g->N * ng], u->sch, (size_t)u->N * ng * 2);
    memcpy(q.bih, g->bih, (size_t)g->N * ng * 2);
    memcpy(&q.bih[(size_t)g->N * ng], u->bih, (size_t)u->N * ng * 2);
    const int ws[2] = {q.N, (int)pr}, ss[2] = {q.N, (int)ng};
    q.w = mlx_array_new_data(q.wpk, ws, 2, MLX_UINT32);
    q.sc = mlx_array_new_data(q.sch, ss, 2, MLX_BFLOAT16);
    q.bi = mlx_array_new_data(q.bih, ss, 2, MLX_BFLOAT16);
    return q;
}
static int qlin(mlx_array* out, mlx_array x, const Wq* q) {
    mlx_optional_int gsz = {GS, true}, bts = {BITS, true};
    return mlx_quantized_matmul(out, x, q->w, q->sc, q->bi, true, gsz, bts, MODE, g_st);
}
static int silu_mul(mlx_array* act, mlx_array g, mlx_array u) {
    mlx_array sg = mlx_array_new(), s = mlx_array_new();
    int rc = mlx_sigmoid(&sg, g, g_st);
    if (rc == 0) rc = mlx_multiply(&s, g, sg, g_st);
    if (rc == 0) rc = mlx_multiply(act, s, u, g_st);
    mlx_array_free(s); mlx_array_free(sg);
    return rc;
}

/* silu compilata (come llm_mlx_swiglu_compiled): closure (gate,up) -> silu(gate)*up */
static int swiglu_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    (void)payload;
    mlx_array g = mlx_array_new(), u = mlx_array_new(), act = mlx_array_new();
    if (mlx_vector_array_get(&g, in, 0) != 0 || mlx_vector_array_get(&u, in, 1) != 0) return 1;
    int rc = silu_mul(&act, g, u);
    if (rc == 0) { mlx_array o[1] = {act}; mlx_vector_array b = mlx_vector_array_new_data(o, 1); mlx_vector_array_set(out, b); mlx_vector_array_free(b); }
    mlx_array_free(g); mlx_array_free(u); mlx_array_free(act);
    return rc;
}

/* una "layer": forma A (sep), B (fused), C (sep+silu compilata). Ritorna l'uscita. */
static int layer_run(int form, int li, mlx_array* out, mlx_array x, const Wq* g, const Wq* u, const Wq* gu, const Wq* d, mlx_closure* swiglu) {
    mlx_array go = mlx_array_new(), uo = mlx_array_new(), act = mlx_array_new();
    int rc = 0;
    if (form == 1) {
        mlx_array gau = mlx_array_new();
        rc = qlin(&gau, x, &gu[li]);
        if (rc == 0) {
            mlx_vector_array parts = mlx_vector_array_new();
            rc = mlx_split(&parts, gau, 2, -1, g_st);
            if (rc == 0) {
                mlx_array gv = mlx_array_new(), uv = mlx_array_new();
                mlx_vector_array_get(&gv, parts, 0);
                mlx_vector_array_get(&uv, parts, 1);
                mlx_array gc = mlx_array_new(), uc = mlx_array_new();
                if (mlx_contiguous(&gc, gv, false, g_st) == 0 && mlx_contiguous(&uc, uv, false, g_st) == 0) {
                    mlx_array_set(&gv, gc); mlx_array_set(&uv, uc);
                }
                mlx_array_free(gc); mlx_array_free(uc);
                rc = silu_mul(&act, gv, uv);
                mlx_array_free(gv); mlx_array_free(uv);
            }
            mlx_vector_array_free(parts);
        }
        mlx_array_free(gau);
    } else {
        rc = qlin(&go, x, &g[li]);
        if (rc == 0) rc = qlin(&uo, x, &u[li]);
        if (rc == 0 && form == 2) {
            mlx_array ins[2] = {go, uo};
            mlx_vector_array in = mlx_vector_array_new_data(ins, 2);
            mlx_vector_array res = mlx_vector_array_new();
            rc = mlx_closure_apply(&res, *swiglu, in);
            if (rc == 0) mlx_vector_array_get(&act, res, 0);
            mlx_vector_array_free(res); mlx_vector_array_free(in);
        } else if (rc == 0) {
            rc = silu_mul(&act, go, uo);
        }
    }
    if (rc == 0) rc = qlin(out, act, &d[li]);
    mlx_array_free(go); mlx_array_free(uo); mlx_array_free(act);
    return rc;
}

static double maxdiff(mlx_array a, mlx_array b) {
    mlx_array a32 = mlx_array_new(), b32 = mlx_array_new(), d = mlx_array_new(), ad = mlx_array_new(), mx = mlx_array_new();
    mlx_astype(&a32, a, MLX_FLOAT32, g_st); mlx_astype(&b32, b, MLX_FLOAT32, g_st);
    mlx_subtract(&d, a32, b32, g_st); mlx_abs(&ad, d, g_st); mlx_max(&mx, ad, false, g_st);
    float v = 0; mlx_array_item_float32(&v, mx);
    mlx_array_free(a32); mlx_array_free(b32); mlx_array_free(d); mlx_array_free(ad); mlx_array_free(mx);
    return v;
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 15;
    if (reps > 100) reps = 100;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    /* 46 set DISTINTI (niente cache-hit tra layer) */
    static Wq wg[NL], wu[NL], wd[NL], wgu[NL];
    for (int l = 0; l < NL; l++) {
        wq_build(&wg[l], SH, H);
        wq_build(&wu[l], SH, H);
        wq_build(&wd[l], H, SH);
        wgu[l] = wq_fuse(&wg[l], &wu[l]);
        mlx_array_eval(wgu[l].w); mlx_array_eval(wd[l].w);
    }
    uint16_t* xh = malloc((size_t)H * 2);
    for (int i = 0; i < H; i++) xh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF));
    const int xs[3] = {1, 1, H};
    mlx_array x = mlx_array_new_data(xh, xs, 3, MLX_BFLOAT16);
    

    mlx_closure raw = mlx_closure_new_func_payload(swiglu_fn, NULL, NULL);
    mlx_closure swiglu = mlx_closure_new();
    mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
    int crc = mlx_compile(&swiglu, raw, false);
    mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
    mlx_closure_free(raw);
    if (crc != 0) { printf("mlx_compile silu FALLITO\n"); return 1; }

    /* riferimento A + parità B/C */
    mlx_array ref = mlx_array_new();
    if (layer_run(0, 0, &ref, x, wg, wu, wgu, wd, &swiglu) != 0) { printf("A FAIL\n"); return 1; }
    mlx_array_eval(ref);
    for (int f = 1; f <= 2; f++) {
        mlx_array o = mlx_array_new();
        if (layer_run(f, 0, &o, x, wg, wu, wgu, wd, &swiglu) != 0) { printf("form %d FAIL\n", f); return 1; }
        mlx_array_eval(o);
        printf("parita' forma %s: max|d|=%.3g\n", f == 1 ? "B fused_gu" : "C swiglu_comp", maxdiff(o, ref));
        mlx_array_free(o);
    }

    const char* names[3] = {"A eager", "B fused_gu", "C swiglu_comp"};
    double ts[3][100];
    for (int r = 0; r < reps + 2; r++) {
        for (int f = 0; f < 3; f++) {
            mlx_vector_array outs = mlx_vector_array_new();
            double t0 = now_ms();
            for (int l = 0; l < NL; l++) {
                mlx_array o = mlx_array_new();
                if (layer_run(f, 0, &o, x, wg, wu, wgu, wd, &swiglu) != 0) { printf("FAIL\n"); return 1; }
                mlx_vector_array_append_value(outs, o);
                mlx_array_free(o);
            }
            mlx_eval(outs);
            double t1 = now_ms();
            mlx_vector_array_free(outs);
            if (r >= 2) ts[f][r - 2] = t1 - t0;
        }
    }
    double base = median(ts[0], reps);
    printf("llm_shared_expert_probe — shared_experts quantizzato g64 b4, %d layer, M=1, %d reps\n", NL, reps);
    for (int f = 0; f < 3; f++)
        printf("%-14s ms/passo-token=%.4f  vs A=%.2f\n", names[f], median(ts[f], reps), median(ts[f], reps) / base);
    printf("(confronto col costo token GLM: 22 ms — risparmio B=%.3f, C=%.3f ms/token)\n",
           base - median(ts[1], reps), base - median(ts[2], reps));

    mlx_array_free(ref); mlx_closure_free(swiglu);
    mlx_array_free(x); free(xh);
    for (int l = 0; l < NL; l++) { wq_free(&wgu[l]); wq_free(&wd[l]); wq_free(&wu[l]); wq_free(&wg[l]); }
    mlx_stream_free(g_st); mlx_device_free(dev);
    return 0;
}
