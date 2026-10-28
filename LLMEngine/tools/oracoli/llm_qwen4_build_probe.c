/* llm_qwen4_build_probe.c — ORACOLO load-free: costo di COSTRUZIONE (build, NO eval)
 * delle op dominanti del decode qwen4_exp, alle forme reali. Confronta la somma dei
 * costi stimati (con i conteggi nodi/token di LLM_DISPATCH) con i ~632 ms di build misurati.
 *
 * Se la somma e' ~2 µs/op (come il compile_cost_probe) -> i 632 ms NON sono op-count.
 * Se certe op costano ~1 ms -> quelle sono il collo (es. quantized_matmul).
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static mlx_stream st;
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

static mlx_array full_f32(int* sh, int nd, float v) {
    mlx_array a = mlx_array_new(), vv = mlx_array_new_float32(v);
    mlx_full(&a, sh, nd, vv, MLX_FLOAT32, st);
    mlx_array_free(vv);
    return a;
}
static mlx_array full_bf16(int* sh, int nd, float v) {
    mlx_array a = mlx_array_new(), vv = mlx_array_new_float32(v);
    mlx_full(&a, sh, nd, vv, MLX_BFLOAT16, st);
    mlx_array_free(vv);
    return a;
}

typedef enum { K_MUL, K_BCAST, K_RESHAPE, K_ASTYPE, K_TRANSPOSE, K_SIGMOID, K_ADD, K_SUM, K_QMM, K_NKIND } kind_t;
static const char* kname[K_NKIND] = {"multiply","broadcast(mul)","reshape","astype","transpose","sigmoid","add","sum_axis","quantized_mm"};
/* conteggi per token da LLM_DISPATCH (decode) */
static const int kcount[K_NKIND] = {1405, 1940, 1580, 494, 240, 518, 524, 388, 614};

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    st = mlx_default_gpu_stream_new();

    int shA[4] = {1,1,1,2560};   mlx_array A = full_f32(shA,4,1.0f);
    int shB[4] = {1,1,4,2560};   mlx_array B = full_f32(shB,4,1.0f);
    int shC[3] = {1,1,10240};    mlx_array C = full_f32(shC,3,1.0f);
    int shD[2] = {4,10240};      mlx_array D = full_bf16(shD,2,1.0f);
    int shM[3] = {1,1,2560};     mlx_array Sm = full_f32(shM,3,1.0f);
    mlx_array_eval(A); mlx_array_eval(B); mlx_array_eval(C); mlx_array_eval(D); mlx_array_eval(Sm);

    /* peso quantizzato reale (hidden x hidden, affine g64 4-bit) per misurare il costo di
     * COSTRUZIONE di quantized_matmul. */
    int shW[2] = {2560,2560};    mlx_array W = full_f32(shW,2,0.01f); mlx_array_eval(W);
    mlx_optional_int gg = {64, true}, bb = {4, true};
    mlx_vector_array qr = mlx_vector_array_new();
    mlx_array gscale = mlx_array_new(); /* null */
    int qrc = mlx_quantize(&qr, W, gg, bb, "affine", gscale, st);
    mlx_array wq = mlx_array_new(), ws = mlx_array_new(), wb = mlx_array_new();
    if (qrc == 0 && mlx_vector_array_size(qr) >= 2) { mlx_vector_array_get(&wq, qr, 0); mlx_vector_array_get(&ws, qr, 1); if (mlx_vector_array_size(qr) >= 3) mlx_vector_array_get(&wb, qr, 2); }
    mlx_array_eval(wq); mlx_array_eval(ws);

    const int N = 2000;             /* op costruite per misura */
    const int REPS = 3;
    double us[K_NKIND] = {0};

    for (int kind = 0; kind < K_NKIND; kind++) {
        double best = 1e30;
        for (int r = 0; r < REPS; r++) {
            mlx_array* o = (mlx_array*)malloc(sizeof(mlx_array)*N);
            double t0 = now_ms();
            for (int i = 0; i < N; i++) {
                o[i] = mlx_array_new();
                switch (kind) {
                    case K_MUL:       mlx_multiply(&o[i], B, B, st); break;
                    case K_BCAST:     mlx_multiply(&o[i], A, B, st); break;
                    case K_RESHAPE:   { int sk[4]={1,1,4,2560}; mlx_reshape(&o[i], C, sk, 4, st); } break;
                    case K_ASTYPE:    mlx_astype(&o[i], D, MLX_FLOAT32, st); break;
                    case K_TRANSPOSE: mlx_transpose(&o[i], D, st); break;
                    case K_SIGMOID:   mlx_sigmoid(&o[i], C, st); break;
                    case K_ADD:       mlx_add(&o[i], C, C, st); break;
                    case K_SUM:       mlx_sum_axis(&o[i], Sm, -1, false, st); break;
                    case K_QMM:       mlx_quantized_matmul(&o[i], Sm, wq, ws, wb, true, gg, bb, "affine", st); break;
                    default: break;
                }
            }
            double ms = now_ms() - t0;
            for (int i = 0; i < N; i++) mlx_array_free(o[i]);
            free(o);
            if (ms < best) best = ms;
        }
        us[kind] = best * 1000.0 / N;   /* µs per op */
    }

    printf("llm_qwen4_build_probe — costo COSTRUZIONE (no eval), forme reali, N=%d\n", N);
    printf("%-16s %10s %10s %14s\n", "op", "us/op", "count/tok", "ms/tok stimati");
    double tot = 0;
    for (int k = 0; k < K_NKIND; k++) {
        double ms_est = us[k] * kcount[k] / 1000.0;
        tot += ms_est;
        printf("%-16s %10.3f %10d %14.3f\n", kname[k], us[k], kcount[k], ms_est);
    }
    printf("%-16s %10s %10s %14.3f  (build misurato ~632 ms)\n", "TOTALE", "", "", tot);
    printf("nota: mancano QuantizedMatmul(614) e Gather/Sort MoE -> se il totale e' <<632 ms,\n"
           "il costo e' in quelle op (quantized matmul/gather), non nelle elementwise.\n");

    mlx_array_free(A); mlx_array_free(B); mlx_array_free(C); mlx_array_free(D); mlx_array_free(Sm);
    mlx_array_free(W); mlx_array_free(wq); mlx_array_free(ws); mlx_array_free(wb); mlx_vector_array_free(qr);
    mlx_stream_free(st); mlx_device_free(dev);
    return 0;
}
