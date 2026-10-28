/* llm_q4_gate_probe.c — ORACOLO model-free: q4_sigmoid_rmsnorm_gated (engine, dylib) deve
 * ONORARE use_silu come i gemelli (cb.c / qwen4exp.h):
 *   use_silu=true  -> silu(gate) * rms_norm = gate*sigmoid(gate) * rms_norm  (in fp32)
 *   use_silu=false -> sigmoid(gate) * rms_norm
 * Confronta l'uscita della funzione ESPORTATA con la formula replicata qui (stesso ordine di
 * op) su dati sintetici; atteso max|diff| = 0 su entrambe.
 * Uso: llm_q4_gate_probe
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern bool q4_sigmoid_rmsnorm_gated(mlx_array* res, mlx_array hidden, mlx_array gate,
                                     mlx_array w, float eps, bool use_silu, mlx_stream s);

static mlx_stream g_st;
static uint32_t g_rng = 0x7a1c0f33u;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }

/* formula di riferimento, stesso ordine: rms_norm -> f32 -> act(gate) -> mul -> cast */
static bool ref_gate(mlx_array* out, mlx_array hidden, mlx_array gate, mlx_array w, float eps, bool use_silu) {
    mlx_array n = mlx_array_new(), nf = mlx_array_new(), gf = mlx_array_new();
    mlx_array act = mlx_array_new(), m = mlx_array_new(), od = mlx_array_new();
    bool ok = mlx_fast_rms_norm(&n, hidden, w, eps, g_st) == 0;
    if (ok) ok = mlx_astype(&nf, n, MLX_FLOAT32, g_st) == 0;
    if (ok) ok = mlx_astype(&gf, gate, MLX_FLOAT32, g_st) == 0;
    if (ok && use_silu) {
        mlx_array sg = mlx_array_new();
        ok = mlx_sigmoid(&sg, gf, g_st) == 0;
        if (ok) ok = mlx_multiply(&act, gf, sg, g_st) == 0;
        mlx_array_free(sg);
    } else if (ok) ok = mlx_sigmoid(&act, gf, g_st) == 0;
    if (ok) ok = mlx_multiply(&m, nf, act, g_st) == 0;
    if (ok) ok = mlx_astype(&od, m, mlx_array_dtype(hidden), g_st) == 0;
    if (ok) ok = mlx_array_set(out, od) == 0;
    mlx_array_free(n); mlx_array_free(nf); mlx_array_free(gf);
    mlx_array_free(act); mlx_array_free(m); mlx_array_free(od);
    return ok;
}

static double maxdiff(mlx_array a, mlx_array b) {
    mlx_array a32 = mlx_array_new(), b32 = mlx_array_new(), d = mlx_array_new(), ad = mlx_array_new(), mx = mlx_array_new();
    mlx_astype(&a32, a, MLX_FLOAT32, g_st); mlx_astype(&b32, b, MLX_FLOAT32, g_st);
    mlx_subtract(&d, a32, b32, g_st); mlx_abs(&ad, d, g_st); mlx_max(&mx, ad, false, g_st);
    float v = 0; mlx_array_item_float32(&v, mx);
    mlx_array_free(a32); mlx_array_free(b32); mlx_array_free(d); mlx_array_free(ad); mlx_array_free(mx);
    return v;
}

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    const int S = 2, T = 4, D = 128;
    uint16_t* hh = malloc((size_t)S * T * D * 2);
    uint16_t* gh = malloc((size_t)S * T * D * 2);
    uint16_t* wh = malloc((size_t)D * 2);
    for (long i = 0; i < (long)S * T * D; i++) { hh[i] = (uint16_t)(0x3E00 | (xr() & 0x01FF)); gh[i] = (uint16_t)(0xBE00 | (xr() & 0x01FF)); }
    for (int i = 0; i < D; i++) wh[i] = (uint16_t)(0x3F00 | (xr() & 0x007F));
    const int hs[3] = {S, T, D}, ws[1] = {D};
    mlx_array hidden = mlx_array_new_data(hh, hs, 3, MLX_BFLOAT16);
    mlx_array gate = mlx_array_new_data(gh, hs, 3, MLX_BFLOAT16);
    mlx_array w = mlx_array_new_data(wh, ws, 1, MLX_BFLOAT16);

    printf("llm_q4_gate_probe — q4_sigmoid_rmsnorm_gated vs formula (shape %dx%dx%d, bf16)\n", S, T, D);
    int bad = 0;
    for (int silu = 1; silu >= 0; silu--) {
        mlx_array got = mlx_array_new(), exp = mlx_array_new();
        if (!q4_sigmoid_rmsnorm_gated(&got, hidden, gate, w, 1e-6f, silu ? true : false, g_st)) { printf("  use_silu=%d: CALL FAIL\n", silu); return 1; }
        if (!ref_gate(&exp, hidden, gate, w, 1e-6f, silu ? true : false)) { printf("  use_silu=%d: REF FAIL\n", silu); return 1; }
        mlx_array_eval(got); mlx_array_eval(exp);
        double md = maxdiff(got, exp);
        printf("  use_silu=%d -> max|diff|=%.3g  %s\n", silu, md, md == 0 ? "BIT-EXACT" : "DIVERGE");
        if (md != 0) bad = 1;
        mlx_array_free(got); mlx_array_free(exp);
    }
    /* regressione: con use_silu=false l'uscita NON deve essere cambiata dal fix */
    printf("ESITO: %s\n", bad ? "FALLITO" : "OK (silu onorata, sigmoid invariata)");
    mlx_array_free(hidden); mlx_array_free(gate); mlx_array_free(w);
    free(hh); free(gh); free(wh);
    mlx_stream_free(g_st); mlx_device_free(dev);
    return bad;
}
