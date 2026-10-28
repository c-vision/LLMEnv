/* llm_glm_router_compile_probe.c — ORACOLO model-free: la closure COMPILATA del router GLM
 * conviene a n_tokens=1 (decode)? Replica la catena eager del motore
 * (llm_mlx_glm_ds_ffn.c, n_group=1 -> niente group-pruning) sulle shape reali:
 *   gates [1,n,128] f32 -> sigmoid -> +e_score_correction_bias -> argpartition(-8) ->
 *   slice coda 8 -> contiguous -> take_along_axis(scores) -> sum -> divide
 * e la confronta con la stessa catena dentro mlx_compile (closure).
 *
 * Metodo oracoli: A/B INTERLEAVED, mediana, rapporto (mai valore assoluto isolato).
 * Uso: llm_glm_router_compile_probe [n_tokens] [reps]   (default 1 60)
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define E 128
#define TOPK 8

static mlx_stream g_st;
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static uint32_t g_rng = 0x51ed2701u;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static float fr(void) { return ((float)(xr() % 20000) / 10000.0f) - 1.0f; }
static double median(double* v, int n) {
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

/* Catena router eager: out_top (int32 [1,n,TOPK]), out_w (f32 [1,n,TOPK]). */
static int router_eager(mlx_array* out_top, mlx_array* out_w, mlx_array gates, mlx_array escb, int n) {
    mlx_array scores = mlx_array_new(), biased = mlx_array_new(), part = mlx_array_new();
    mlx_array view = mlx_array_new(), top = mlx_array_new(), tv = mlx_array_new();
    mlx_array den = mlx_array_new(), w = mlx_array_new();
    int rc = mlx_sigmoid(&scores, gates, g_st);
    if (rc == 0) rc = mlx_add(&biased, scores, escb, g_st);
    if (rc == 0) rc = mlx_argpartition_axis(&part, biased, -TOPK, -1, g_st);
    if (rc == 0) {
        const int start[3] = {0, 0, E - TOPK}, stop[3] = {1, n, E}, strides[3] = {1, 1, 1};
        rc = mlx_slice(&view, part, start, 3, stop, 3, strides, 3, g_st);
    }
    if (rc == 0) rc = mlx_contiguous(&top, view, false, g_st);
    if (rc == 0) rc = mlx_take_along_axis(&tv, scores, top, -1, g_st);
    if (rc == 0) rc = mlx_sum_axis(&den, tv, -1, true, g_st);
    if (rc == 0) rc = mlx_divide(&w, tv, den, g_st);
    if (rc == 0) { mlx_array_set(out_top, top); mlx_array_set(out_w, w); }
    mlx_array_free(scores); mlx_array_free(biased); mlx_array_free(part);
    mlx_array_free(view); mlx_array_free(top); mlx_array_free(tv);
    mlx_array_free(den); mlx_array_free(w);
    return rc;
}

typedef struct { int n; } RCtx;

/* Closure: in[0]=gates [1,n,128] f32, in[1]=escb [128] f32 -> out[0]=top int32, out[1]=w f32 */
static int router_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    RCtx* c = (RCtx*)payload;
    mlx_array gates = mlx_array_new(), escb = mlx_array_new();
    if (mlx_vector_array_get(&gates, in, 0) != 0 || mlx_vector_array_get(&escb, in, 1) != 0) return 1;
    mlx_array top = mlx_array_new(), w = mlx_array_new();
    int rc = router_eager(&top, &w, gates, escb, c->n);
    if (rc == 0) { mlx_array outs[2] = {top, w}; mlx_vector_array b = mlx_vector_array_new_data(outs, 2); mlx_vector_array_set(out, b); mlx_vector_array_free(b); }
    mlx_array_free(gates); mlx_array_free(escb); mlx_array_free(top); mlx_array_free(w);
    return rc;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 1;
    int reps = argc > 2 ? atoi(argv[2]) : 60;
    if (reps > 200) reps = 200;
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    float* gh = malloc((size_t)n * E * 4);
    float* bh = malloc((size_t)E * 4);
    for (int i = 0; i < n * E; i++) gh[i] = fr();
    for (int i = 0; i < E; i++) bh[i] = fr() * 0.5f;
    const int gs[3] = {1, n, E}, bs[1] = {E};
    mlx_array gates = mlx_array_new_data(gh, gs, 3, MLX_FLOAT32);
    mlx_array escb = mlx_array_new_data(bh, bs, 1, MLX_FLOAT32);
    mlx_array_eval(gates); mlx_array_eval(escb);

    RCtx ctx = { n };
    mlx_closure raw = mlx_closure_new_func_payload(router_fn, &ctx, NULL);
    mlx_closure cl = mlx_closure_new();
    mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
    int crc = mlx_compile(&cl, raw, /*shapeless=*/false);
    mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
    mlx_closure_free(raw);
    if (crc != 0) { printf("mlx_compile FALLITO\n"); return 1; }

    /* parità eager vs compiled (top identici, pesi uguali) */
    mlx_array t0 = mlx_array_new(), w0 = mlx_array_new(), t1 = mlx_array_new(), w1 = mlx_array_new();
    if (router_eager(&t0, &w0, gates, escb, n) != 0) { printf("eager FAIL\n"); return 1; }
    mlx_array ins[2] = {gates, escb};
    mlx_vector_array in = mlx_vector_array_new_data(ins, 2);
    mlx_vector_array res = mlx_vector_array_new();
    if (mlx_closure_apply(&res, cl, in) != 0) { printf("closure FAIL\n"); return 1; }
    mlx_vector_array_get(&t1, res, 0);
    mlx_vector_array_get(&w1, res, 1);
    mlx_array_eval(t0); mlx_array_eval(w0); mlx_array_eval(t1); mlx_array_eval(w1);
    const int32_t* p0 = mlx_array_data_int32(t0); const int32_t* p1 = mlx_array_data_int32(t1);
    int top_par = 1; for (int i = 0; i < n * TOPK; i++) if (p0[i] != p1[i]) { top_par = 0; break; }
    mlx_array d = mlx_array_new(), mx = mlx_array_new();
    mlx_subtract(&d, w0, w1, g_st);
    mlx_abs(&d, d, g_st);
    mlx_max(&mx, d, false, g_st);
    float wdiff = 0; mlx_array_item_float32(&wdiff, mx);
    printf("llm_glm_router_compile_probe — n=%d E=%d topk=%d | %d reps\n", n, E, TOPK, reps);
    printf("parita': top_inds %s, max|dw|=%.3g\n", top_par ? "IDENTICI" : "DIVERSI", (double)wdiff);

    /* A/B interleaved a livello PASSO-TOKEN: NL router in catena + UNA eval (come l'engine). */
    enum { NL = 46 };
    double te[200], tc[200];
    for (int r = 0; r < reps + 2; r++) {
        mlx_vector_array outs_e = mlx_vector_array_new();
        double s0 = now_ms();
        for (int l = 0; l < NL; l++) {
            mlx_array a = mlx_array_new(), b = mlx_array_new();
            if (router_eager(&a, &b, gates, escb, n) != 0) { printf("eager FAIL\n"); return 1; }
            mlx_vector_array_append_value(outs_e, a);
            mlx_vector_array_append_value(outs_e, b);
            mlx_array_free(a); mlx_array_free(b);
        }
        mlx_eval(outs_e);
        double s1 = now_ms();
        mlx_vector_array_free(outs_e);
        mlx_vector_array outs_c = mlx_vector_array_new();
        double s2 = now_ms();
        for (int l = 0; l < NL; l++) {
            mlx_vector_array r2 = mlx_vector_array_new();
            if (mlx_closure_apply(&r2, cl, in) != 0) { printf("closure FAIL\n"); return 1; }
            mlx_array x = mlx_array_new(), y = mlx_array_new();
            mlx_vector_array_get(&x, r2, 0); mlx_vector_array_get(&y, r2, 1);
            mlx_vector_array_append_value(outs_c, x);
            mlx_vector_array_append_value(outs_c, y);
            mlx_array_free(x); mlx_array_free(y);
            mlx_vector_array_free(r2);
        }
        mlx_eval(outs_c);
        double s3 = now_ms();
        mlx_vector_array_free(outs_c);
        if (r >= 2) { te[r - 2] = s1 - s0; tc[r - 2] = s3 - s2; }
    }
    double me = median(te, reps), mc = median(tc, reps);
    printf("passo-token (%d router, 1 eval): eager=%.4f ms  compiled=%.4f ms  ratio=%.2f  %s\n",
           NL, me, mc, mc / me, mc < me ? "COMPILED PAGA" : "compiled NON paga");
    printf("stima GLM (46 layer, decode): risparmio=%.3f ms/token (%.1f%% di 22 ms)\n",
           me - mc, 100.0 * (me - mc) / 22.0);

    mlx_array_free(d); mlx_array_free(mx); mlx_array_free(t0); mlx_array_free(w0);
    mlx_array_free(t1); mlx_array_free(w1);
    mlx_vector_array_free(res); mlx_vector_array_free(in);
    mlx_closure_free(cl);
    mlx_array_free(gates); mlx_array_free(escb);
    free(gh); free(bh);
    mlx_stream_free(g_st); mlx_device_free(dev);
    return 0;
}
