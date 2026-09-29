/* llm_mlx_oq_a8_q8_check.c — test UNITARIO del kernel OQ-A8 a bits=8 (W8A8 su pesi MLX 8-bit).
 * Confronta il nostro QMM (llmmetal_oq_a8_stage_a + llmmetal_oq_a8_qmm) con mlx_quantized_matmul sugli
 * STESSI pesi quantizzati da MLX (mlx_quantize, affine, group 64): l'unica differenza e' la
 * quantizzazione delle ATTIVAZIONI (int8 per riga) -> errore atteso piccolo (W8A8 approx), non
 * garbage. Il controllo a bits=4 fa da riferimento noto-buono.
 *
 * Build (stessa ricetta del probe smallm + bridge.mm):
 *   cd native/llmengine_mlx && /usr/bin/c++ -O3 -DNDEBUG -std=gnu++20 -arch arm64 -fPIC \
 *     -fobjc-arc -Ibuild/_deps/metal_cpp-src -I../../libs/mlx-deps/mlx-src -Ibuild/_deps/cjson-src \
 *     -I../llmmetal_oq_a8 -I. -I../../libs/mlx-deps/mlx-c-src \
 *     llm_mlx_oq_a8_q8_check.c ../llmmetal_oq_a8/llmmetal_oq_a8_bridge.mm \
 *     build/_deps/mlx_c-build/libmlxc.a build/_deps/mlx-build/libmlx.a build/_deps/cjson-build/libcjson.a \
 *     build/_deps/mlx-build/mlx/io/libgguflib.a build/jaccl/libjaccl.a \
 *     -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
 *     -o /tmp/llmmetal_oq_a8_q8_check
 * Uso: /tmp/llmmetal_oq_a8_q8_check [K] [N] [M] [bits] [qmm.metallib] [quantize.metallib]
 */
#include "mlx/c/mlx.h"
#include "llm_engine_opts.h"
#include "llmmetal_oq_a8_bridge.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 512;
    const int N = argc > 2 ? atoi(argv[2]) : 256;
    const int M = argc > 3 ? atoi(argv[3]) : 8;
    const int bits = argc > 4 ? atoi(argv[4]) : 8;
    const char* qmm = argc > 5 ? argv[5] : "oq_a8_qmm_nax.metallib";
    const char* qz = argc > 6 ? argv[6] : "oq_a8_quantize.metallib";
    LlmMetalOqA8Ctx* ctx = llmmetal_oq_a8_create(qmm, qz);
    if (!ctx) { printf("FAIL: ctx\n"); return 1; }
    mlx_stream st = mlx_default_gpu_stream_new();

    /* pesi f32 casuali -> mlx_quantize (affine, group 64, bits) = FORMATO MLX REALE */
    float* wf = (float*)malloc((size_t)N * K * sizeof(float));
    uint32_t seed = 12345u;
    for (size_t i = 0; i < (size_t)N * K; i++) { seed = seed * 1664525u + 1013904223u; wf[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
    int wsh[2] = {N, K};
    mlx_array Wf = mlx_array_new_data(wf, wsh, 2, MLX_FLOAT32);
    free(wf);
    mlx_optional_int gs = {64, true}, bt = {bits, true};
    mlx_vector_array qv = mlx_vector_array_new();
    mlx_array Wq = mlx_array_new(), Sc = mlx_array_new(), Bi = mlx_array_new();
    if (mlx_quantize(&qv, Wf, gs, bt, "affine", mlx_array_new(), st) != 0 ||
        mlx_vector_array_size(qv) < 3 ||
        mlx_vector_array_get(&Wq, qv, 0) != 0 || mlx_vector_array_get(&Sc, qv, 1) != 0 ||
        mlx_vector_array_get(&Bi, qv, 2) != 0) { printf("FAIL: mlx_quantize\n"); return 1; }

    /* x f16 [M,K] */
    float* xf = (float*)malloc((size_t)M * K * sizeof(float));
    for (size_t i = 0; i < (size_t)M * K; i++) { seed = seed * 1664525u + 1013904223u; xf[i] = 0.5f * (((seed >> 8) & 1023) / 512.0f - 1.0f); }
    /* Outlier reali: le attivazioni LLM hanno valori molto grandi e rari (canali outlier).
     * Con scala per-RIGA (absmax) un outlier per riga distrugge la precisione degli altri. */
    if (llm_ai_mlx_opt_env("LLM_OQ_CHECK_OUTLIER")) {
        for (int m = 0; m < M; ++m) for (int k = 0; k < K; k += 64) xf[(size_t)m * K + k] *= 100.0f;
    }
    int xsh[2] = {M, K};
    mlx_array X32 = mlx_array_new_data(xf, xsh, 2, MLX_FLOAT32);
    free(xf);
    const int use_bf16 = argc > 7 && atoi(argv[7]) != 0;
    mlx_array X = mlx_array_new();
    mlx_astype(&X, X32, use_bf16 ? MLX_BFLOAT16 : MLX_FLOAT16, st);
    /* Come l'engine (modello bf16): anche scales/biases in bf16 -> pipeline bf16 del kernel. */
    mlx_array Sc2 = mlx_array_new(), Bi2 = mlx_array_new();
    if (use_bf16) { mlx_astype(&Sc2, Sc, MLX_BFLOAT16, st); mlx_astype(&Bi2, Bi, MLX_BFLOAT16, st); }
    else { mlx_array_set(&Sc2, Sc); mlx_array_set(&Bi2, Bi); }

    /* Materializza gli input (le Prep cache del bridge sono chiavate sul BUFFER: array lazy
     * hanno ptr null -> chiave sbagliata e transpose su buffer non scritto). */
    mlx_array_eval(X); mlx_array_eval(Wq); mlx_array_eval(Sc2); mlx_array_eval(Bi2);
    /* Diagnostica layout qa: stampa i primi 32 byte di qa[0] e i valori attesi per k=0..31
     * (ACT_MODE 0: inv_scale = 127/amax della riga). */
    if (llm_ai_mlx_opt_env("LLM_OQ_CHECK_LAYOUT")) {
        mlx_array qa2 = mlx_array_new(), sa2 = mlx_array_new(), ra2 = mlx_array_new();
        if (llmmetal_oq_a8_stage_a(X, &qa2, &sa2, &ra2, st) == 0) {
            mlx_array_eval(qa2); mlx_array_eval(sa2);
            const int8_t* qd = (const int8_t*)mlx_array_data_int8(qa2);
            const float* sd = (const float*)mlx_array_data_float32(sa2);
            const float inv = 1.0f / sd[0];   /* sa = amax/127 -> inv = 127/amax */
            const float* xrow = (const float*)mlx_array_data_float32(X32);  /* xf e' liberato */
            fprintf(stderr, "[layout] sa[0]=%.6f inv=%.4f\n", sd[0], inv);
            for (int k = 0; k < 32; ++k) {
                const int q = (int)lrintf(xrow[k] * inv);
                fprintf(stderr, "[layout] k=%2d qa_mem=%4d atteso=%4d\n", k, (int)qd[k],
                        q < -127 ? -127 : (q > 127 ? 127 : q));
            }
            mlx_array_free(qa2); mlx_array_free(sa2); mlx_array_free(ra2);
        }
    }
    /* riferimento MLX */
    mlx_array Yref = mlx_array_new();
    int rc = mlx_quantized_matmul(&Yref, X, Wq, Sc2, Bi2, /*transpose=*/true, gs, bt, "affine", st);
    if (rc != 0 || mlx_array_eval(Yref) != 0) { printf("FAIL: quantized_matmul rc=%d\n", rc); return 1; }

    /* nostro (W8A8): stage_a + qmm */
    mlx_array qa = mlx_array_new(), sa = mlx_array_new(), ra = mlx_array_new();
    mlx_array Y = mlx_array_new();
    if (!llm_ai_mlx_opt_env("LLM_OQ_CHECK_ONLY_PRIM")) {
        if (llmmetal_oq_a8_stage_a(X, &qa, &sa, &ra, st) != 0) { printf("FAIL: stage_a\n"); return 1; }
        Y = llmmetal_oq_a8_qmm(qa, sa, ra, Wq, Sc2, Bi2, st, bits, use_bf16 ? 1 : 0);
        if (!Y.ctx) { printf("FAIL: qmm (bits=%d)\n", bits); return 1; }
        if (mlx_array_eval(Y) != 0) { printf("FAIL: eval\n"); return 1; }
    }

    mlx_array Yr32 = mlx_array_new();
    mlx_astype(&Yr32, Yref, MLX_FLOAT32, st);
    mlx_array_eval(Yr32);
    mlx_array Yo32 = mlx_array_new();
    if (Y.ctx) { mlx_astype(&Yo32, Y, MLX_FLOAT32, st); mlx_array_eval(Yo32); }
    const float* yr0 = (const float*)mlx_array_data_float32(Yr32);
    const size_t n0 = mlx_array_size(Yr32);
    double ref2_0 = 0.0; for (size_t i = 0; i < n0; i++) ref2_0 += (double)yr0[i] * yr0[i];
    const double ref2 = ref2_0;
    /* Anche il path FUSED (llmmetal_oq_a8_prim, usato dall'engine a M>=16). */
    mlx_array Yp = llmmetal_oq_a8_prim(X, Wq, Sc2, Bi2, st, bits);
    double rel_l2p = -1.0;
    if (Yp.ctx) {
        fprintf(stderr, "[dbg] Yp dtype=%d shape=[%d,%d]\n", (int)mlx_array_dtype(Yp),
                mlx_array_ndim(Yp) > 0 ? mlx_array_shape(Yp)[0] : -1,
                mlx_array_ndim(Yp) > 1 ? mlx_array_shape(Yp)[1] : -1);
        mlx_array_eval(Yp);
        if (mlx_array_dtype(Yp) == MLX_BFLOAT16) {
            const uint16_t* b16 = (const uint16_t*)mlx_array_data_bfloat16(Yp);
            fprintf(stderr, "[dbg] bf16 diretti: %04x %04x %04x (ptr=%p)\n",
                    b16 ? b16[0] : 0, b16 ? b16[1] : 0, b16 ? b16[2] : 0, (const void*)b16);
        }
        mlx_array Yp32 = mlx_array_new();
        int arc = mlx_astype(&Yp32, Yp, MLX_FLOAT32, st);
        int erc = mlx_array_eval(Yp32);
        const float* yp = (const float*)mlx_array_data_float32(Yp32);
        fprintf(stderr, "[dbg] astype=%d eval=%d yp=%p yp[0..2]=%.4f %.4f %.4f size=%zu\n",
                arc, erc, (const void*)yp, yp ? yp[0] : 0.f, yp ? yp[1] : 0.f, yp ? yp[2] : 0.f,
                (size_t)mlx_array_size(Yp32));
        double s2 = 0.0;
        for (size_t i = 0; i < n0; i++) { const double d = (double)yr0[i] - yp[i]; s2 += d * d; }
        rel_l2p = sqrt(s2 / (ref2 > 0 ? ref2 : 1.0));
        mlx_array_free(Yp32);
    }
    const float* yr = (const float*)mlx_array_data_float32(Yr32);
    const float* yo = Y.ctx ? (const float*)mlx_array_data_float32(Yo32) : NULL;
    size_t n = mlx_array_size(Yr32);
    double maxabs = 0.0, maxrel = 0.0, sum2 = 0.0;
    for (size_t i = 0; yo != NULL && i < n; i++) {
        const double a = yr[i], b = yo[i];
        const double d = fabs(a - b);
        if (d > maxabs) maxabs = d;
        if (fabs(a) > 1e-3) { const double r = d / fabs(a); if (r > maxrel) maxrel = r; }
        sum2 += d * d;
    }
    const double rel_l2 = (yo != NULL) ? sqrt(sum2 / (ref2 > 0 ? ref2 : 1.0)) : -1.0;
    if (argc > 8 && atoi(argv[8]) != 0 && yo != NULL) {
        fprintf(stderr, "[dump] ref: %.4f %.4f %.4f %.4f\n", yr[0], yr[1], yr[2], yr[3]);
        fprintf(stderr, "[dump] our: %.4f %.4f %.4f %.4f\n", yo[0], yo[1], yo[2], yo[3]);
    }
    printf("bits=%d K=%d N=%d M=%d dt=%s | maxabs=%.6f maxrel=%.4f relL2=%.5f primL2=%.5f (%s)\n",
           bits, K, N, M, use_bf16 ? "bf16" : "f16", maxabs, maxrel, rel_l2, rel_l2p,
           rel_l2 < 0.05 ? "OK (errore di quantizzazione attivazioni)" : "SOSPETTO: decode sbagliato");
    mlx_array_free(Yo32); mlx_array_free(Yr32); mlx_array_free(Sc2); mlx_array_free(Bi2);
    mlx_array_free(Y); mlx_array_free(qa); mlx_array_free(sa); mlx_array_free(ra);
    mlx_array_free(Yref); mlx_array_free(X); mlx_array_free(X32);
    mlx_vector_array_free(qv); mlx_array_free(Wf);
    mlx_array_free(Wq); mlx_array_free(Sc); mlx_array_free(Bi);
    llmmetal_oq_a8_destroy(ctx);
    return rel_l2 < 0.05 ? 0 : 2;
}
