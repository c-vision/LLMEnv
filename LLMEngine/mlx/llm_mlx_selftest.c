/* ============================================================================
 * llm_mlx_selftest.c — dominio self-test / benchmark / checkpoint (model-free + checkpoint
 * diagnostici). Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* Self-test di DRIFT (2026-09-16): costruisce UN grafo (n_ops add + una slice di cache KV
 * [1,8,cap,64]) e lo valuta `iters` volte, stampando il tempo ogni 64 iterazioni. Isola lo stato
 * interno di MLX (allocator/pool/cache delle primitive) dal loop del motore: se il tempo CRESCE
 * con le iterazioni, il degrado del decode (misurato: 7,7 -> 22 ms/token nelle prime finestre,
 * INDIPENDENTE dal contesto) e' dentro MLX, non nel nostro loop. */
int llm_ai_mlx_eval_drift_selftest(int n_ops, int iters, char* out, int cap) {
    if (out == NULL || cap < 64 || n_ops < 1 || iters < 128) return -1;
    mlx_device chosen = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }
    const int D = 512;
    float* d = (float*)malloc(sizeof(float) * (size_t)D);
    if (d == NULL) { mlx_device_free(chosen); return -1; }
    for (int i = 0; i < D; i++) d[i] = (float)(i % 17) * 0.0625f;
    const int ash[3] = {1, 1, D};
    mlx_array a = mlx_array_new_data(d, ash, 3, MLX_FLOAT32);
    double first = 0, last = 0, worst = 0;
    int printed = 0;
    for (int it = 0; it < iters; it++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mlx_array y = mlx_array_new();
        mlx_array_set(&y, a);
        mlx_array t = mlx_array_new();
        for (int i = 0; i < n_ops; i++) { mlx_array_set(&t, y); if (mlx_add(&y, t, a, st) != 0) break; }
        mlx_vector_array vec = mlx_vector_array_new();
        mlx_vector_array_append_value(vec, y);
        /* come il motore: ASYNC_EVAL (non eval) -> se il drift e' qui, e' in MLX async */
        mlx_async_eval(vec);
        mlx_vector_array_free(vec);
        mlx_synchronize(st);
        mlx_array_free(t); mlx_array_free(y);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
        if (it == 0) first = ms;
        if (it == iters - 1) last = ms;
        if (it >= 64 && ms > worst) worst = ms;
        printed++;
    }
    snprintf(out, (size_t)cap, "eval_drift: n_ops=%d iters=%d | primo=%.3f ms ultimo=%.3f ms worst(>=64)=%.3f ms -> %s",
             n_ops, printed, first, last, worst, (last > first * 1.5) ? "*** DRIFT: cresce con le iterazioni ***" : "stabile");
    mlx_array_free(a);
    free(d);
    mlx_device_free(chosen);
    return 0;
}

int llm_ai_mlx_encode_cost_selftest(int n_ops, int reps, char* out, int cap) {
    if (out == NULL || cap < 64 || n_ops <= 1 || reps <= 0) return -1;
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool gpu_ok = false;
    mlx_device_is_available(&gpu_ok, gpu);
    mlx_device chosen = gpu_ok ? gpu : mlx_device_new_type(MLX_CPU, 0);
    if (!gpu_ok) mlx_device_free(gpu);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }

    const int D = 1024;
    float* d = (float*)malloc(sizeof(float) * (size_t)D);
    int rc = -1;
    if (d != NULL) {
        for (int i = 0; i < D; i++) d[i] = (float)((i % 17) - 8) * 0.0625f;
        int sh[3] = {1, 1, D};
        mlx_array a = mlx_array_new_data(d, sh, 3, MLX_FLOAT32);
        double b_min = 1e30, s_min = 1e30, w_min = 1e30, e_min = 1e30;
        double rd_b_min = 1e30, rd_s_min = 1e30;
        double rc_b_min = 1e30, rc_s_min = 1e30;
        struct timespec t0, t1, t2, t3;
#define ECM_MS(x, y) ((double)((y).tv_sec - (x).tv_sec) * 1e3 + (double)((y).tv_nsec - (x).tv_nsec) * 1e-6)
        for (int rep = 0; rep < reps; rep++) {
            /* (a)+(b) BUILD poi ASYNC SUBMIT */
            clock_gettime(CLOCK_MONOTONIC, &t0);
            mlx_array y = mlx_array_new();
            mlx_array_set(&y, a);
            mlx_array tmp = mlx_array_new();
            for (int i = 0; i < n_ops; i++) {
                mlx_array_set(&tmp, y);
                if (mlx_add(&y, tmp, a, st) != 0) { mlx_array_free(tmp); mlx_array_free(y); mlx_device_free(chosen); free(d); return -1; }
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            mlx_vector_array vec = mlx_vector_array_new();
            mlx_vector_array_append_value(vec, y);
            if (mlx_async_eval(vec) != 0) { mlx_vector_array_free(vec); mlx_array_free(tmp); mlx_array_free(y); mlx_device_free(chosen); free(d); return -1; }
            clock_gettime(CLOCK_MONOTONIC, &t2);
            mlx_synchronize(st);
            clock_gettime(CLOCK_MONOTONIC, &t3);
            double b = ECM_MS(t0, t1), s = ECM_MS(t1, t2), w = ECM_MS(t2, t3);
            if (b < b_min) b_min = b;
            if (s < s_min) s_min = s;
            if (w < w_min) w_min = w;
            mlx_vector_array_free(vec);
            mlx_array_free(tmp);
            mlx_array_free(y);

            /* (d) EAGER: build + mlx_eval bloccante (submit+wait insieme) */
            clock_gettime(CLOCK_MONOTONIC, &t0);
            mlx_array y2 = mlx_array_new();
            mlx_array_set(&y2, a);
            mlx_array tmp2 = mlx_array_new();
            for (int i = 0; i < n_ops; i++) {
                mlx_array_set(&tmp2, y2);
                if (mlx_add(&y2, tmp2, a, st) != 0) break;
            }
            mlx_vector_array vec2 = mlx_vector_array_new();
            mlx_vector_array_append_value(vec2, y2);
            if (mlx_eval(vec2) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double e = ECM_MS(t0, t1);
                if (e < e_min) e_min = e;
            }
            mlx_vector_array_free(vec2);
            mlx_array_free(tmp2);
            mlx_array_free(y2);

            /* N chiamate C, ma un solo nodo reshape vivo: ogni risultato precedente viene
             * rilasciato e il successivo riparte dalla costante `a`. Se il costo di submit
             * seguisse le chiamate anziche' i nodi raggiungibili, resterebbe O(n_ops). */
            clock_gettime(CLOCK_MONOTONIC, &t0);
            mlx_array rd = mlx_array_new();
            const int rd_shape[2] = {1, D};
            for (int i = 0; i < n_ops; i++) {
                mlx_array_free(rd);
                rd = mlx_array_new();
                if (mlx_reshape(&rd, a, rd_shape, 2, st) != 0) break;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            mlx_vector_array rdv = mlx_vector_array_new();
            mlx_vector_array_append_value(rdv, rd);
            if (mlx_async_eval(rdv) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t2);
                mlx_synchronize(st);
                double b = ECM_MS(t0, t1), s = ECM_MS(t1, t2);
                if (b < rd_b_min) rd_b_min = b;
                if (s < rd_s_min) rd_s_min = s;
            }
            mlx_vector_array_free(rdv);
            mlx_array_free(rd);

            /* N chiamate reshape e N nodi vivi in catena, alternando due shape equivalenti. */
            clock_gettime(CLOCK_MONOTONIC, &t0);
            mlx_array rc = mlx_array_new();
            mlx_array_set(&rc, a);
            for (int i = 0; i < n_ops; i++) {
                mlx_array prev = mlx_array_new();
                mlx_array_set(&prev, rc);
                const int r2[2] = {1, D};
                const int r3[3] = {1, 1, D};
                int erc = (i & 1) ? mlx_reshape(&rc, prev, r3, 3, st)
                                    : mlx_reshape(&rc, prev, r2, 2, st);
                mlx_array_free(prev);
                if (erc != 0) break;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            mlx_vector_array rcv = mlx_vector_array_new();
            mlx_vector_array_append_value(rcv, rc);
            if (mlx_async_eval(rcv) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t2);
                mlx_synchronize(st);
                double b = ECM_MS(t0, t1), s = ECM_MS(t1, t2);
                if (b < rc_b_min) rc_b_min = b;
                if (s < rc_s_min) rc_s_min = s;
            }
            mlx_vector_array_free(rcv);
            mlx_array_free(rc);
        }
#undef ECM_MS
        snprintf(out, (size_t)cap,
                 "encode_cost: n_ops=%d reps=%d | add_chain build=%.3f us/call submit=%.3f us/node "
                 "wait=%.3f ms eager=%.3f us/node | reshape_dead build=%.3f us/call submit=%.3f ms/1-live-node "
                 "| reshape_chain build=%.3f us/call submit=%.3f us/node",
                 n_ops, reps, b_min * 1000.0 / n_ops, s_min * 1000.0 / n_ops, w_min,
                 e_min * 1000.0 / n_ops, rd_b_min * 1000.0 / n_ops, rd_s_min,
                 rc_b_min * 1000.0 / n_ops, rc_s_min * 1000.0 / n_ops);
        mlx_array_free(a);
        rc = 0;
    }
    free(d);
    mlx_device_free(chosen);
    return rc;
}

/* Self-test BANDA SOSTENIBILE (2026-09-16, Fase 1 del documento "M5 Max GPU engine" di altra AI):
 * verifica la premessa "siamo bandwidth-bound" misurando il massimo che QUESTO stack sa erogare su
 * un kernel puro memory-bound, senza modello e senza MLX-model: mode 0 = triad C=A+B (3*S byte),
 * 1 = read-only sum(A) (S byte), 2 = write-only zeros (S byte). f32, array di S MB. Riporta il MAX
 * GB/s (min tempo) su `reps`. Il tetto dichiarato del chip va confrontato con QUESTO numero, non
 * assunto. */
int llm_ai_mlx_bw_selftest(int size_mb, int mode, int reps, char* out, int cap) {
    if (out == NULL || cap < 64 || size_mb < 1 || size_mb > 8192 || reps <= 0 || mode < 0 || mode > 2)
        return -1;
    size_t S = (size_t)size_mb << 20;
    size_t N = S / 4;
    /* Init come gemv_bench: device_new_type(GPU) -> set_default_device -> get_default_stream.
     * NON usare mlx_device_is_available/free prima della stream: su M5 crasha in
     * mlx_get_default_stream (trappola MLX documentata). */
    mlx_device chosen = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }
    float* da = (float*)malloc(N * sizeof(float));
    float* db = (mode == 0) ? (float*)malloc(N * sizeof(float)) : NULL;
    int rc = -1;
    if (da != NULL && (mode != 0 || db != NULL)) {
        for (size_t i = 0; i < N; i++) { da[i] = (float)(i % 251) * 0.00390625f; if (db) db[i] = (float)(i % 127) * 0.0078125f; }
        size_t sh[1] = {N};
        mlx_array a = mlx_array_new_data(da, (const int*)sh, 1, MLX_FLOAT32);
        mlx_array b = (mode == 0) ? mlx_array_new_data(db, (const int*)sh, 1, MLX_FLOAT32) : mlx_array_new();
        size_t bytes = (mode == 0) ? 3 * S : S;
        double best_gbs = 0, best_ms = 1e30;
        for (int rep = 0; rep < reps; rep++) {
            struct timespec t0, t1;
            mlx_array res = mlx_array_new();
            clock_gettime(CLOCK_MONOTONIC, &t0);
            int erc = 0;
            if (mode == 0) {
                erc = mlx_add(&res, a, b, st);
            } else if (mode == 1) {
                erc = mlx_sum(&res, a, /*keepdims=*/false, st);
            } else {
                erc = mlx_zeros(&res, (const int*)sh, 1, MLX_FLOAT32, st);
            }
            if (erc == 0) {
                mlx_vector_array vec = mlx_vector_array_new();
                mlx_vector_array_append_value(vec, res);
                erc = mlx_eval(vec);
                mlx_vector_array_free(vec);
            }
            mlx_synchronize(st);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
            if (erc == 0 && ms < best_ms) {
                best_ms = ms;
                best_gbs = (double)bytes / (ms * 1e6);
            }
            mlx_array_free(res);
        }
        snprintf(out, (size_t)cap,
                 "bw: size=%d MB mode=%d (%s) | %.1f GB/s sostenuti (min %.3f ms, %llu MB trasferiti)",
                 size_mb, mode, mode == 0 ? "triad C=A+B" : (mode == 1 ? "read-only sum" : "write-only zeros"),
                 best_gbs, best_ms, (unsigned long long)(bytes >> 20));
        mlx_array_free(a); mlx_array_free(b);
        rc = 0;
    }
    free(da); free(db);
    mlx_device_free(chosen);
    return rc;
}

/* llm_ai_mlx_gemv_bench (2026-09-16) — microbench SYNTHETIC della GEMV di DECODE (M piccolo)
 * al chokepoint quantized_linear_cfg, senza modello. Serve a P0-ter: il collo del decode e' la
 * GEMV M=1 e i suoi tempi dipendono dalla COPPIA (dtype attivazione, dtype scale) -- la leva
 * che ha chiuso il gap di Qwen3-8B (fp32 -> bf16 = +11%). Qui si misura ogni coppia sulle forme
 * reali dei modelli e si individua chi e' su una combinazione subottimale SENZA caricare modelli.
 * adt/sdt: 0=f32 1=f16 2=bf16.  Scrive in out ms/chiamata. Ritorna 0 ok. */
int llm_ai_mlx_gemv_bench(int m, int n, int k, int bits, int group, int adt, int sdt, int reps, int pipe, char* out, int cap) {
    if (out == NULL || cap < 48 || m < 1 || n < 64 || k < 64 || (bits != 4 && bits != 8) ||
        (group != 32 && group != 64 && group != 128) || reps < 1) return -1;
    mlx_device chosen = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_dtype dts[3] = {MLX_FLOAT32, MLX_FLOAT16, MLX_BFLOAT16};
    if (adt < 0 || adt > 2 || sdt < 0 || sdt > 2) { mlx_device_free(chosen); return -1; }
    const mlx_dtype dt_a = dts[adt], dt_s = dts[sdt];

    const int wcols = (int)(((long)k * bits) / 32);
    const int gcols = k / group;
    uint16_t* xb = (uint16_t*)calloc((size_t)m * k, sizeof(uint16_t));
    uint32_t* wb = (uint32_t*)calloc((size_t)n * wcols, sizeof(uint32_t));
    uint16_t* sb = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)n * gcols);
    uint16_t* bb = (uint16_t*)calloc((size_t)n * gcols, sizeof(uint16_t));
    int rc = -1;
    if (xb && wb && sb && bb) {
        /* scale = 1.0 nel dtype scelto: f16 0x3C00, bf16 0x3F80, f32 0x3F800000 (qui 16-bit buffer) */
        const uint16_t one16 = (dt_s == MLX_BFLOAT16) ? 0x3F80 : 0x3C00;
        for (size_t i = 0; i < (size_t)n * gcols; i++) sb[i] = one16;
        int xsh[2] = {m, k}, wsh[2] = {n, wcols}, gsh[2] = {n, gcols};
        mlx_array x = mlx_array_new_data(xb, xsh, 2, dt_a);
        mlx_array w = mlx_array_new_data(wb, wsh, 2, MLX_UINT32);
        mlx_array sc = mlx_array_new_data(sb, gsh, 2, dt_s);
        mlx_array bi = mlx_array_new_data(bb, gsh, 2, dt_s);
        struct timespec t0, t1;
        double tot = 0; int ok_n = 0;
        if (pipe == 1) {
            /* PIPELINED: costruisce `reps` call lazy e fa UNA eval -> misura il THROUGHPUT
             * (il decode reale non sincronizza a ogni GEMV: le accoda). */
            for (int r = 0; r < 2; r++) { mlx_array res = mlx_array_new();
                bool ok = quantized_linear_cfg(&res, x, w, sc, bi, group, bits, "affine", st);
                if (ok) ok = mlx_array_eval(res) == 0; mlx_array_free(res); if (!ok) break; }
            mlx_array* rs = (mlx_array*)calloc((size_t)reps, sizeof(mlx_array));
            bool ok = (rs != NULL);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int r = 0; ok && r < reps; r++) {
                rs[r] = mlx_array_new();
                ok = quantized_linear_cfg(&rs[r], x, w, sc, bi, group, bits, "affine", st);
            }
            for (int r = 0; ok && r < reps; r++) ok = mlx_array_eval(rs[r]) == 0;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            for (int r = 0; r < reps; r++) mlx_array_free(rs[r]);
            free(rs);
            if (ok) { tot = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6; ok_n = reps; }
        } else {
            for (int r = 0; r < reps + 2; r++) {   /* 2 di warm-up, scartate */
                mlx_array res = mlx_array_new();
                clock_gettime(CLOCK_MONOTONIC, &t0);
                bool ok = quantized_linear_cfg(&res, x, w, sc, bi, group, bits, "affine", st);
                if (ok) ok = mlx_array_eval(res) == 0;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                mlx_array_free(res);
                if (!ok) break;
                if (r >= 2) { tot += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6; ok_n++; }
            }
        }
        if (ok_n > 0) {
            const char* an[3] = {"f32", "f16", "bf16"};
            snprintf(out, (size_t)cap, "M=%d N=%d K=%d %dbit g%d act=%s scale=%s %s -> %.4f ms/call",
                     m, n, k, bits, group, an[adt], an[sdt], pipe ? "PIPELINED" : "isolata", tot / ok_n);
            rc = 0;
        }
        mlx_array_free(x); mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi);
    }
    free(xb); free(wb); free(sb); free(bb);
    mlx_device_free(chosen);
    return rc;
}

int llm_ai_mlx_smoketest(void) {
    /* Prefer GPU (Metal); fall back to CPU when Metal isn't available in
     * this build. No code change needed once Metal is available again:
     * this picks it up automatically. */
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool gpu_available = false;
    mlx_device_is_available(&gpu_available, gpu);

    mlx_device chosen = gpu_available ? gpu : mlx_device_new_type(MLX_CPU, 0);
    if (!gpu_available) {
        printf("llm_ai_mlx_smoketest: MLX GPU (Metal) not available in this build, using CPU\n");
        mlx_device_free(gpu);
    }

    if (mlx_set_default_device(chosen) != 0) {
        fprintf(stderr, "llm_ai_mlx_smoketest: mlx_set_default_device failed\n");
        mlx_device_free(chosen);
        return -1;
    }

    mlx_string dev_desc = mlx_string_new();
    if (mlx_device_tostring(&dev_desc, chosen) == 0) {
        printf("llm_ai_mlx_smoketest: default device = %s\n", mlx_string_data(dev_desc));
    }
    mlx_string_free(dev_desc);

    /* Out-params in mlx-c follow the same "pre-existing, reassigned into"
     * convention as mlx_device_set: the destination must already be a
     * valid (if empty) object -- passing raw uninitialized stack memory
     * makes the internal release-old-then-retain-new logic free garbage,
     * which crashed here (SIGABRT, "pointer being freed was not allocated")
     * until this was changed to mlx_stream_new() first. */
    mlx_stream stream = mlx_stream_new();
    if (mlx_get_default_stream(&stream, chosen) != 0) {
        fprintf(stderr, "llm_ai_mlx_smoketest: mlx_get_default_stream failed\n");
        mlx_device_free(chosen);
        return -1;
    }
    mlx_device_free(chosen);

    /* A = [[1,2],[3,4]], B = [[5,6],[7,8]] -- hand-computed expected result
     * A*B = [[19,22],[43,50]]. */
    const float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const int shape[2] = {2, 2};

    mlx_array a = mlx_array_new_data(a_data, shape, 2, MLX_FLOAT32);
    mlx_array b = mlx_array_new_data(b_data, shape, 2, MLX_FLOAT32);
    mlx_array c = mlx_array_new();

    int rc = mlx_matmul(&c, a, b, stream);
    if (rc != 0) {
        fprintf(stderr, "llm_ai_mlx_smoketest: mlx_matmul failed rc=%d\n", rc);
        mlx_array_free(a); mlx_array_free(b); mlx_array_free(c); mlx_stream_free(stream);
        return -1;
    }

    if (mlx_array_eval(c) != 0) {
        fprintf(stderr, "llm_ai_mlx_smoketest: mlx_array_eval failed\n");
        mlx_array_free(a); mlx_array_free(b); mlx_array_free(c); mlx_stream_free(stream);
        return -1;
    }

    const float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    const float* result = mlx_array_data_float32(c);
    bool ok = (result != NULL);
    if (ok) {
        for (int i = 0; i < 4; i++) {
            if (fabsf(result[i] - expected[i]) > 1e-4f) { ok = false; break; }
        }
        printf("llm_ai_mlx_smoketest: result = [%.2f, %.2f, %.2f, %.2f] (expected [19.00, 22.00, 43.00, 50.00])\n",
               result[0], result[1], result[2], result[3]);
    } else {
        fprintf(stderr, "llm_ai_mlx_smoketest: mlx_array_data_float32 returned null\n");
    }

    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(c);
    mlx_stream_free(stream);

    return ok ? 0 : -1;
}

int llm_ai_mlx_embedding_checkpoint(const char* safetensors_path, int token_id) {
    if (safetensors_path == NULL) return -1;

    mlx_stream stream = mlx_default_cpu_stream_new();

    mlx_map_string_to_array weights = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
    if (mlx_load_safetensors(&weights, &metadata, safetensors_path, stream) != 0) {
        fprintf(stderr, "llm_ai_mlx_embedding_checkpoint: mlx_load_safetensors failed for %s\n", safetensors_path);
        mlx_map_string_to_array_free(weights);
        mlx_map_string_to_string_free(metadata);
        mlx_stream_free(stream);
        return -1;
    }

    mlx_array w = mlx_array_new(), scales = mlx_array_new(), biases = mlx_array_new();
    bool have_all = (mlx_map_string_to_array_get(&w, weights, "model.embed_tokens.weight") == 0) &&
                    (mlx_map_string_to_array_get(&scales, weights, "model.embed_tokens.scales") == 0) &&
                    (mlx_map_string_to_array_get(&biases, weights, "model.embed_tokens.biases") == 0);
    if (!have_all) {
        fprintf(stderr, "llm_ai_mlx_embedding_checkpoint: model.embed_tokens.{weight,scales,biases} not found in %s\n", safetensors_path);
        mlx_array_free(w); mlx_array_free(scales); mlx_array_free(biases);
        mlx_map_string_to_array_free(weights);
        mlx_map_string_to_string_free(metadata);
        mlx_stream_free(stream);
        return -1;
    }

    /* Extract just the row for token_id (axis 0) from weight/scales/biases
     * BEFORE dequantizing -- dequantizing the full [151936, ...] embedding
     * matrix just to read one row would be wasteful and is not what a real
     * embedding lookup does anyway. */
    const int idx_shape[1] = {1};
    const int32_t idx_data[1] = {token_id};
    mlx_array indices = mlx_array_new_data(idx_data, idx_shape, 1, MLX_INT32);

    mlx_array w_row = mlx_array_new(), scales_row = mlx_array_new(), biases_row = mlx_array_new();
    bool gathered = (mlx_take_axis(&w_row, w, indices, 0, stream) == 0) &&
                    (mlx_take_axis(&scales_row, scales, indices, 0, stream) == 0) &&
                    (mlx_take_axis(&biases_row, biases, indices, 0, stream) == 0);

    mlx_array deq = mlx_array_new();
    bool ok = false;
    if (gathered) {
        mlx_optional_int group_size = {64, true};
        mlx_optional_int bits = {8, true};
        mlx_optional_dtype dtype = {MLX_FLOAT32, true};
        if (mlx_dequantize(&deq, w_row, scales_row, biases_row, group_size, bits, "affine",
                            /*global_scale=*/mlx_array_new(), dtype, stream) == 0 &&
            mlx_array_eval(deq) == 0) {
            const float* data = mlx_array_data_float32(deq);
            size_t n = mlx_array_size(deq);
            if (data != NULL && n > 0) {
                ok = true;
                printf("llm_ai_mlx_embedding_checkpoint: token_id=%d, embedding dim=%zu, first values = [", token_id, n);
                for (size_t i = 0; i < n && i < 8; i++) {
                    if (isnan(data[i])) ok = false;
                    printf("%s%.4f", i == 0 ? "" : ", ", data[i]);
                }
                printf("%s]\n", n > 8 ? ", ..." : "");
            }
        } else {
            fprintf(stderr, "llm_ai_mlx_embedding_checkpoint: mlx_dequantize/eval failed\n");
        }
    } else {
        fprintf(stderr, "llm_ai_mlx_embedding_checkpoint: mlx_take_axis failed\n");
    }

    mlx_array_free(indices);
    mlx_array_free(w_row);
    mlx_array_free(scales_row);
    mlx_array_free(biases_row);
    mlx_array_free(deq);
    mlx_array_free(w);
    mlx_array_free(scales);
    mlx_array_free(biases);
    mlx_map_string_to_array_free(weights);
    mlx_map_string_to_string_free(metadata);
    mlx_stream_free(stream);

    return ok ? 0 : -1;
}

int llm_ai_mlx_attention_checkpoint(const char* safetensors_path, int token_id, float rope_theta) {
    if (safetensors_path == NULL) return -1;

    /* Model dims (layer 0, this specific model) -- verified against
     * config.json in Fase1-MLX-Qwen3MoE-Plan.md, not hardcoded guesses. */
    const int hidden_size = 2048;
    const int n_heads = 32;
    const int n_kv_heads = 4;
    const int head_dim = 128;
    const float rms_norm_eps = 1e-6f;

    mlx_stream stream = mlx_default_cpu_stream_new();

    mlx_map_string_to_array weights = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
    if (mlx_load_safetensors(&weights, &metadata, safetensors_path, stream) != 0) {
        fprintf(stderr, "llm_ai_mlx_attention_checkpoint: mlx_load_safetensors failed\n");
        mlx_map_string_to_array_free(weights);
        mlx_map_string_to_string_free(metadata);
        mlx_stream_free(stream);
        return -1;
    }

    bool ok = true;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "llm_ai_mlx_attention_checkpoint: %s failed\n", (what)); ok = false; } } while (0)

    mlx_array embed_w = mlx_array_new(), embed_scales = mlx_array_new(), embed_biases = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.embed_tokens", &embed_w, &embed_scales, &embed_biases);
    CHK(ok, "embed_tokens lookup");

    const int idx_shape[1] = {1};
    const int32_t idx_data[1] = {token_id};
    mlx_array indices = mlx_array_new_data(idx_data, idx_shape, 1, MLX_INT32);

    mlx_array embed_w_row = mlx_array_new(), embed_scales_row = mlx_array_new(), embed_biases_row = mlx_array_new();
    if (ok) {
        ok = mlx_take_axis(&embed_w_row, embed_w, indices, 0, stream) == 0 &&
             mlx_take_axis(&embed_scales_row, embed_scales, indices, 0, stream) == 0 &&
             mlx_take_axis(&embed_biases_row, embed_biases, indices, 0, stream) == 0;
        CHK(ok, "embedding row gather");
    }

    mlx_array x_flat = mlx_array_new();
    if (ok) {
        mlx_optional_int gsz = {64, true};
        mlx_optional_int bts = {8, true};
        mlx_optional_dtype dt = {MLX_FLOAT32, true};
        ok = mlx_dequantize(&x_flat, embed_w_row, embed_scales_row, embed_biases_row, gsz, bts, "affine",
                             mlx_array_new(), dt, stream) == 0;
        CHK(ok, "embedding dequantize");
    }
    mlx_array x = mlx_array_new();
    if (ok) {
        const int x_shape[3] = {1, 1, hidden_size};
        ok = mlx_reshape(&x, x_flat, x_shape, 3, stream) == 0;
        CHK(ok, "x reshape");
    }

    mlx_array ln_w = mlx_array_new();
    ok = ok && mlx_map_string_to_array_get(&ln_w, weights, "model.layers.0.input_layernorm.weight") == 0;
    CHK(ok, "input_layernorm.weight lookup");
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, ln_w, rms_norm_eps, stream); CHK(ok, "input_layernorm"); }

    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.q_proj", &qw, &qs, &qb);
    CHK(ok, "q_proj lookup");
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.k_proj", &kw, &ks, &kb);
    CHK(ok, "k_proj lookup");
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.v_proj", &vw, &vs, &vb);
    CHK(ok, "v_proj lookup");

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) { ok = quantized_linear(&q, normed, qw, qs, qb, stream); CHK(ok, "q_proj matmul"); }
    if (ok) { ok = quantized_linear(&k, normed, kw, ks, kb, stream); CHK(ok, "k_proj matmul"); }
    if (ok) { ok = quantized_linear(&v, normed, vw, vs, vb, stream); CHK(ok, "v_proj matmul"); }

    /* reshape to per-head BEFORE qk-norm (matches mlx_lm: norm is applied on
     * the reshaped-but-not-yet-transposed tensor) */
    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, 1, n_heads, head_dim};
        const int kv_shape[4] = {1, 1, n_kv_heads, head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, stream) == 0;
        CHK(ok, "head reshape");
    }

    /* QK-norm (RMSNorm per-head, last axis = head_dim) */
    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    ok = ok && mlx_map_string_to_array_get(&qn_w, weights, "model.layers.0.self_attn.q_norm.weight") == 0;
    CHK(ok, "q_norm.weight lookup");
    ok = ok && mlx_map_string_to_array_get(&kn_w, weights, "model.layers.0.self_attn.k_norm.weight") == 0;
    CHK(ok, "k_norm.weight lookup");
    mlx_array q_normed = mlx_array_new(), k_normed = mlx_array_new();
    if (ok) {
        ok = rms_norm_cfg(&q_normed, q_heads, qn_w, rms_norm_eps, stream) &&
             rms_norm_cfg(&k_normed, k_heads, kn_w, rms_norm_eps, stream);
        CHK(ok, "qk-norm");
    }

    /* transpose to [B, n_heads, L, head_dim] for attention */
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_normed, perm, 4, stream) == 0 &&
             mlx_transpose_axes(&k_t, k_normed, perm, 4, stream) == 0 &&
             mlx_transpose_axes(&v_t, v_heads, perm, 4, stream) == 0;
        CHK(ok, "qkv transpose");
    }

    /* RoPE (traditional=false, base=rope_theta from config, offset=0 --
     * single token, no KV-cache yet) */
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) {
        mlx_optional_float base = {rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                            /*offset=*/0, no_freqs, stream) == 0 &&
             mlx_fast_rope(&k_rope, k_t, head_dim, false, base, 1.0f, 0, no_freqs, stream) == 0;
        CHK(ok, "rope");
        mlx_array_free(no_freqs);
    }

    /* GQA scaled-dot-product-attention (native GQA broadcast, verified
     * against mlx_lm/models/base.py -- no manual repeat/tile needed) */
    mlx_array attn = mlx_array_new();
    if (ok) {
        float scale = 1.0f / sqrtf((float)head_dim);
        mlx_array no_mask = mlx_array_new();
        mlx_array no_sinks = mlx_array_new();
        ok = mlx_fast_scaled_dot_product_attention(&attn, q_rope, k_rope, v_t, scale, "", no_mask, no_sinks, stream) == 0;
        CHK(ok, "scaled_dot_product_attention");
    }

    /* transpose back [B,L,n_heads,head_dim], flatten, output projection */
    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, stream) == 0;
        CHK(ok, "attn transpose back");
        if (ok) {
            const int flat_shape[3] = {1, 1, n_heads * head_dim};
            ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, stream) == 0;
            CHK(ok, "attn flatten");
        }
    }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.o_proj", &ow, &os, &ob);
    CHK(ok, "o_proj lookup");
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear(&attn_out, attn_flat, ow, os, ob, stream); CHK(ok, "o_proj matmul"); }

    /* residual: h = x + attn_out (x is the ORIGINAL pre-norm embedding) */
    mlx_array h = mlx_array_new();
    if (ok) { ok = mlx_add(&h, x, attn_out, stream) == 0; CHK(ok, "residual add"); }

    if (ok) { ok = mlx_array_eval(attn_out) == 0 && mlx_array_eval(h) == 0; CHK(ok, "eval"); }
    if (ok) {
        const float* attn_data = mlx_array_data_float32(attn_out);
        const float* h_data = mlx_array_data_float32(h);
        ok = (attn_data != NULL) && (h_data != NULL);
        if (ok) {
            printf("llm_ai_mlx_attention_checkpoint: attn_out first8 = [");
            for (int i = 0; i < 8; i++) {
                if (isnan(attn_data[i])) ok = false;
                printf("%s%.6f", i == 0 ? "" : ", ", attn_data[i]);
            }
            printf("]\n");
            printf("llm_ai_mlx_attention_checkpoint: post_residual first8 = [");
            for (int i = 0; i < 8; i++) {
                if (isnan(h_data[i])) ok = false;
                printf("%s%.6f", i == 0 ? "" : ", ", h_data[i]);
            }
            printf("]\n");
        } else {
            fprintf(stderr, "llm_ai_mlx_attention_checkpoint: null result data\n");
        }
    }

    /* Cleanup -- checkpoint code, not a hot path; free everything created
     * regardless of which branch was taken (mlx_array_free on an
     * already-empty mlx_array_new() is a documented no-op). */
    mlx_array all[] = {
        embed_w, embed_scales, embed_biases, indices, embed_w_row, embed_scales_row, embed_biases_row,
        x_flat, x, ln_w, normed,
        qw, qs, qb, kw, ks, kb, vw, vs, vb, q, k, v,
        q_heads, k_heads, v_heads, qn_w, kn_w, q_normed, k_normed,
        q_t, k_t, v_t, q_rope, k_rope, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out, h,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

    mlx_map_string_to_array_free(weights);
    mlx_map_string_to_string_free(metadata);
    mlx_stream_free(stream);

#undef CHK
    return ok ? 0 : -1;
}

int llm_ai_mlx_moe_checkpoint(const char* safetensors_path, int token_id, float rope_theta) {
    if (safetensors_path == NULL) return -1;

    /* Model dims (layer 0, this specific model) -- verified against
     * config.json in Fase1-MLX-Qwen3MoE-Plan.md, not hardcoded guesses. */
    const int hidden_size = 2048;
    const int n_heads = 32;
    const int n_kv_heads = 4;
    const int head_dim = 128;
    const float rms_norm_eps = 1e-6f;
    const int top_k = 8; /* num_experts_per_tok */
    const bool norm_topk_prob = true;

    mlx_stream stream = mlx_default_cpu_stream_new();

    mlx_map_string_to_array weights = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
    if (mlx_load_safetensors(&weights, &metadata, safetensors_path, stream) != 0) {
        fprintf(stderr, "llm_ai_mlx_moe_checkpoint: mlx_load_safetensors failed\n");
        mlx_map_string_to_array_free(weights);
        mlx_map_string_to_string_free(metadata);
        mlx_stream_free(stream);
        return -1;
    }

    bool ok = true;
#define CHK(cond, what) do { if (!(cond)) { fprintf(stderr, "llm_ai_mlx_moe_checkpoint: %s failed\n", (what)); ok = false; } } while (0)

    /* attention block, identical to llm_ai_mlx_attention_checkpoint (not yet
     * factored into a shared helper -- each checkpoint function stays
     * self-contained by design, see Fase1-MLX-Qwen3MoE-Plan.md) */
    mlx_array embed_w = mlx_array_new(), embed_scales = mlx_array_new(), embed_biases = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.embed_tokens", &embed_w, &embed_scales, &embed_biases);
    CHK(ok, "embed_tokens lookup");

    const int idx_shape[1] = {1};
    const int32_t idx_data[1] = {token_id};
    mlx_array indices = mlx_array_new_data(idx_data, idx_shape, 1, MLX_INT32);

    mlx_array embed_w_row = mlx_array_new(), embed_scales_row = mlx_array_new(), embed_biases_row = mlx_array_new();
    if (ok) {
        ok = mlx_take_axis(&embed_w_row, embed_w, indices, 0, stream) == 0 &&
             mlx_take_axis(&embed_scales_row, embed_scales, indices, 0, stream) == 0 &&
             mlx_take_axis(&embed_biases_row, embed_biases, indices, 0, stream) == 0;
        CHK(ok, "embedding row gather");
    }

    mlx_array x_flat = mlx_array_new();
    if (ok) {
        mlx_optional_int gsz = {64, true};
        mlx_optional_int bts = {8, true};
        mlx_optional_dtype dt = {MLX_FLOAT32, true};
        ok = mlx_dequantize(&x_flat, embed_w_row, embed_scales_row, embed_biases_row, gsz, bts, "affine",
                             mlx_array_new(), dt, stream) == 0;
        CHK(ok, "embedding dequantize");
    }
    mlx_array x = mlx_array_new();
    if (ok) {
        const int x_shape[3] = {1, 1, hidden_size};
        ok = mlx_reshape(&x, x_flat, x_shape, 3, stream) == 0;
        CHK(ok, "x reshape");
    }

    mlx_array ln_w = mlx_array_new();
    ok = ok && mlx_map_string_to_array_get(&ln_w, weights, "model.layers.0.input_layernorm.weight") == 0;
    CHK(ok, "input_layernorm.weight lookup");
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, ln_w, rms_norm_eps, stream); CHK(ok, "input_layernorm"); }

    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.q_proj", &qw, &qs, &qb);
    CHK(ok, "q_proj lookup");
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.k_proj", &kw, &ks, &kb);
    CHK(ok, "k_proj lookup");
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.v_proj", &vw, &vs, &vb);
    CHK(ok, "v_proj lookup");

    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) { ok = quantized_linear(&q, normed, qw, qs, qb, stream); CHK(ok, "q_proj matmul"); }
    if (ok) { ok = quantized_linear(&k, normed, kw, ks, kb, stream); CHK(ok, "k_proj matmul"); }
    if (ok) { ok = quantized_linear(&v, normed, vw, vs, vb, stream); CHK(ok, "v_proj matmul"); }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, 1, n_heads, head_dim};
        const int kv_shape[4] = {1, 1, n_kv_heads, head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, stream) == 0;
        CHK(ok, "head reshape");
    }

    mlx_array qn_w = mlx_array_new(), kn_w = mlx_array_new();
    ok = ok && mlx_map_string_to_array_get(&qn_w, weights, "model.layers.0.self_attn.q_norm.weight") == 0;
    CHK(ok, "q_norm.weight lookup");
    ok = ok && mlx_map_string_to_array_get(&kn_w, weights, "model.layers.0.self_attn.k_norm.weight") == 0;
    CHK(ok, "k_norm.weight lookup");
    mlx_array q_normed = mlx_array_new(), k_normed = mlx_array_new();
    if (ok) {
        ok = rms_norm_cfg(&q_normed, q_heads, qn_w, rms_norm_eps, stream) &&
             rms_norm_cfg(&k_normed, k_heads, kn_w, rms_norm_eps, stream);
        CHK(ok, "qk-norm");
    }

    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_normed, perm, 4, stream) == 0 &&
             mlx_transpose_axes(&k_t, k_normed, perm, 4, stream) == 0 &&
             mlx_transpose_axes(&v_t, v_heads, perm, 4, stream) == 0;
        CHK(ok, "qkv transpose");
    }

    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) {
        mlx_optional_float base = {rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, head_dim, /*traditional=*/false, base, /*scale=*/1.0f,
                            /*offset=*/0, no_freqs, stream) == 0 &&
             mlx_fast_rope(&k_rope, k_t, head_dim, false, base, 1.0f, 0, no_freqs, stream) == 0;
        CHK(ok, "rope");
        mlx_array_free(no_freqs);
    }

    mlx_array attn = mlx_array_new();
    if (ok) {
        float scale = 1.0f / sqrtf((float)head_dim);
        mlx_array no_mask = mlx_array_new();
        mlx_array no_sinks = mlx_array_new();
        ok = mlx_fast_scaled_dot_product_attention(&attn, q_rope, k_rope, v_t, scale, "", no_mask, no_sinks, stream) == 0;
        CHK(ok, "scaled_dot_product_attention");
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, stream) == 0;
        CHK(ok, "attn transpose back");
        if (ok) {
            const int flat_shape[3] = {1, 1, n_heads * head_dim};
            ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, stream) == 0;
            CHK(ok, "attn flatten");
        }
    }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.layers.0.self_attn.o_proj", &ow, &os, &ob);
    CHK(ok, "o_proj lookup");
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear(&attn_out, attn_flat, ow, os, ob, stream); CHK(ok, "o_proj matmul"); }

    mlx_array h = mlx_array_new();
    if (ok) { ok = mlx_add(&h, x, attn_out, stream) == 0; CHK(ok, "residual add (attn)"); }

    /* post_attention_layernorm */
    mlx_array ln2_w = mlx_array_new();
    ok = ok && mlx_map_string_to_array_get(&ln2_w, weights, "model.layers.0.post_attention_layernorm.weight") == 0;
    CHK(ok, "post_attention_layernorm.weight lookup");
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = mlx_fast_rms_norm(&normed2, h, ln2_w, rms_norm_eps, stream) == 0; CHK(ok, "post_attention_layernorm"); }

    /* router: quantized matmul -> softmax(precise) -> top-k via
     * argpartition -> renormalize (norm_topk_prob=true for this model) */
    mlx_array gate_w = mlx_array_new(), gate_s = mlx_array_new(), gate_b = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.layers.0.mlp.gate", &gate_w, &gate_s, &gate_b);
    CHK(ok, "mlp.gate lookup");
    mlx_array gates = mlx_array_new();
    if (ok) { ok = quantized_linear(&gates, normed2, gate_w, gate_s, gate_b, stream); CHK(ok, "router matmul"); }

    mlx_array gates_sm = mlx_array_new();
    if (ok) { ok = mlx_softmax_axis(&gates_sm, gates, -1, /*precise=*/true, stream) == 0; CHK(ok, "router softmax"); }

    /* argpartition puts the top-k (largest) values in the LAST k slots
     * along the axis (kth=-top_k mirrors mx.argpartition(gates, kth=-k,
     * axis=-1)[..., -k:] in mlx_lm/models/qwen3_moe.py) -- slice them out. */
    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, gates_sm, /*kth=*/-top_k, /*axis=*/-1, stream) == 0;
        CHK(ok, "router argpartition");
        if (ok) {
            const int num_experts = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, num_experts - top_k};
            const int stop[3] = {1, 1, num_experts};
            const int strides[3] = {1, 1, 1};
            ok = mlx_slice(&top_inds, part, start, 3, stop, 3, strides, 3, stream) == 0;
            CHK(ok, "top-k slice");
        }
    }

    /* Evaluate router outputs and read them host-side -- with B=1, L=1
     * this is 128 + 8 scalars, cheaper and just as correct as staying in
     * the mlx graph for the trivial per-expert score lookup/renormalize
     * that follows. */
    float gates_sm_host[128];
    uint32_t top_inds_host[8];
    if (ok) { ok = mlx_array_eval(gates_sm) == 0 && mlx_array_eval(top_inds) == 0; CHK(ok, "router eval"); }
    if (ok) {
        const float* g = mlx_array_data_float32(gates_sm);
        const uint32_t* ti = mlx_array_data_uint32(top_inds);
        ok = g != NULL && ti != NULL && mlx_array_size(gates_sm) == 128 && mlx_array_size(top_inds) == (size_t)top_k;
        CHK(ok, "router data readback");
        if (ok) {
            memcpy(gates_sm_host, g, sizeof(gates_sm_host));
            memcpy(top_inds_host, ti, sizeof(top_inds_host));
        }
    }

    float scores_host[8] = {0};
    if (ok) {
        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            scores_host[i] = gates_sm_host[top_inds_host[i]];
            sum += scores_host[i];
        }
        if (norm_topk_prob && sum > 0.0f) {
            for (int i = 0; i < top_k; i++) scores_host[i] /= sum;
        }
        printf("llm_ai_mlx_moe_checkpoint: selected experts = [");
        for (int i = 0; i < top_k; i++) printf("%s%u", i == 0 ? "" : ", ", top_inds_host[i]);
        printf("]\n");
        printf("llm_ai_mlx_moe_checkpoint: scores = [");
        for (int i = 0; i < top_k; i++) printf("%s%.6f", i == 0 ? "" : ", ", scores_host[i]);
        printf("]\n");
    }

    /* per-expert SwiGLU FFN, weighted-summed by the router scores. One
     * quantized matmul per selected expert against a weight slice taken
     * from the stacked switch_mlp tensors -- mathematically identical to
     * mlx_gather_qmm's unsorted dispatch for a single token (the
     * sort-by-expert path in mlx_lm/models/switch_layers.py only triggers
     * for indices.size >= 64; here indices.size == top_k == 8), and easier
     * to verify step by step against the mlx-lm oracle. */
    mlx_array gate_proj_w = mlx_array_new(), gate_proj_s = mlx_array_new(), gate_proj_b = mlx_array_new();
    mlx_array up_proj_w = mlx_array_new(), up_proj_s = mlx_array_new(), up_proj_b = mlx_array_new();
    mlx_array down_proj_w = mlx_array_new(), down_proj_s = mlx_array_new(), down_proj_b = mlx_array_new();
    ok = ok && get_quantized_triple(weights, "model.layers.0.mlp.switch_mlp.gate_proj", &gate_proj_w, &gate_proj_s, &gate_proj_b);
    CHK(ok, "switch_mlp.gate_proj lookup");
    ok = ok && get_quantized_triple(weights, "model.layers.0.mlp.switch_mlp.up_proj", &up_proj_w, &up_proj_s, &up_proj_b);
    CHK(ok, "switch_mlp.up_proj lookup");
    ok = ok && get_quantized_triple(weights, "model.layers.0.mlp.switch_mlp.down_proj", &down_proj_w, &down_proj_s, &down_proj_b);
    CHK(ok, "switch_mlp.down_proj lookup");

    mlx_array moe_out = mlx_array_new();
    /* Everything allocated inside the per-expert loop, tracked here so it
     * can be freed after the loop regardless of how many iterations ran.
     * 18 mlx_array pushes per iteration (eidx=1, 3 quantized triples=9,
     * gate_e/up_e=2, sig/silu/act=3, down_e=1, weight_scalar/scaled_e=2) --
     * must match exactly, a stack buffer overflow here silently corrupted
     * adjacent locals (scores_host) the first time this was sized at 16. */
    mlx_array expert_tmp[8 * 18];
    int expert_tmp_n = 0;
    if (ok) {
        for (int i = 0; i < top_k && ok; i++) {
            const int32_t eid = (int32_t)top_inds_host[i];
            const int eidx_shape[1] = {1};
            mlx_array eidx = mlx_array_new_data(&eid, eidx_shape, 1, MLX_INT32);
            expert_tmp[expert_tmp_n++] = eidx;

            mlx_array gw_e = mlx_array_new(), gs_e = mlx_array_new(), gb_e = mlx_array_new();
            mlx_array uw_e = mlx_array_new(), us_e = mlx_array_new(), ub_e = mlx_array_new();
            mlx_array dw_e = mlx_array_new(), ds_e = mlx_array_new(), db_e = mlx_array_new();
            ok = mlx_take_axis(&gw_e, gate_proj_w, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&gs_e, gate_proj_s, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&gb_e, gate_proj_b, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&uw_e, up_proj_w, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&us_e, up_proj_s, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&ub_e, up_proj_b, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&dw_e, down_proj_w, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&ds_e, down_proj_s, eidx, 0, stream) == 0 &&
                 mlx_take_axis(&db_e, down_proj_b, eidx, 0, stream) == 0;
            CHK(ok, "expert weight slice");
            expert_tmp[expert_tmp_n++] = gw_e; expert_tmp[expert_tmp_n++] = gs_e; expert_tmp[expert_tmp_n++] = gb_e;
            expert_tmp[expert_tmp_n++] = uw_e; expert_tmp[expert_tmp_n++] = us_e; expert_tmp[expert_tmp_n++] = ub_e;
            expert_tmp[expert_tmp_n++] = dw_e; expert_tmp[expert_tmp_n++] = ds_e; expert_tmp[expert_tmp_n++] = db_e;

            mlx_array gate_e = mlx_array_new(), up_e = mlx_array_new();
            if (ok) { ok = quantized_linear(&gate_e, normed2, gw_e, gs_e, gb_e, stream); CHK(ok, "expert gate_proj matmul"); }
            if (ok) { ok = quantized_linear(&up_e, normed2, uw_e, us_e, ub_e, stream); CHK(ok, "expert up_proj matmul"); }
            expert_tmp[expert_tmp_n++] = gate_e; expert_tmp[expert_tmp_n++] = up_e;

            /* swiglu(gate, x) = silu(gate) * x = (gate * sigmoid(gate)) * up */
            mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
            if (ok) { ok = mlx_sigmoid(&sig, gate_e, stream) == 0; CHK(ok, "expert sigmoid"); }
            if (ok) { ok = mlx_multiply(&silu, gate_e, sig, stream) == 0; CHK(ok, "expert silu"); }
            if (ok) { ok = mlx_multiply(&act, silu, up_e, stream) == 0; CHK(ok, "expert swiglu"); }
            expert_tmp[expert_tmp_n++] = sig; expert_tmp[expert_tmp_n++] = silu; expert_tmp[expert_tmp_n++] = act;

            mlx_array down_e = mlx_array_new();
            if (ok) { ok = quantized_linear(&down_e, act, dw_e, ds_e, db_e, stream); CHK(ok, "expert down_proj matmul"); }
            expert_tmp[expert_tmp_n++] = down_e;

            mlx_array weight_scalar = mlx_array_new_float32(scores_host[i]);
            mlx_array scaled_e = mlx_array_new();
            if (ok) { ok = mlx_multiply(&scaled_e, down_e, weight_scalar, stream) == 0; CHK(ok, "expert weight scale"); }
            expert_tmp[expert_tmp_n++] = weight_scalar; expert_tmp[expert_tmp_n++] = scaled_e;

            if (ok) {
                if (i == 0) {
                    ok = mlx_add(&moe_out, scaled_e, mlx_array_new_float32(0.0f), stream) == 0;
                    CHK(ok, "moe_out init");
                } else {
                    mlx_array next = mlx_array_new();
                    ok = mlx_add(&next, moe_out, scaled_e, stream) == 0;
                    CHK(ok, "moe_out accumulate");
                    if (ok) { mlx_array_free(moe_out); moe_out = next; }
                    else mlx_array_free(next);
                }
            }
        }
    }

    mlx_array layer_out = mlx_array_new();
    if (ok) { ok = mlx_add(&layer_out, h, moe_out, stream) == 0; CHK(ok, "final residual add"); }

    if (ok) { ok = mlx_array_eval(moe_out) == 0 && mlx_array_eval(layer_out) == 0; CHK(ok, "final eval"); }
    if (ok) {
        const float* moe_data = mlx_array_data_float32(moe_out);
        const float* out_data = mlx_array_data_float32(layer_out);
        ok = (moe_data != NULL) && (out_data != NULL);
        if (ok) {
            printf("llm_ai_mlx_moe_checkpoint: moe_out first8 = [");
            for (int i = 0; i < 8; i++) {
                if (isnan(moe_data[i])) ok = false;
                printf("%s%.6f", i == 0 ? "" : ", ", moe_data[i]);
            }
            printf("]\n");
            printf("llm_ai_mlx_moe_checkpoint: layer0_output first8 = [");
            for (int i = 0; i < 8; i++) {
                if (isnan(out_data[i])) ok = false;
                printf("%s%.6f", i == 0 ? "" : ", ", out_data[i]);
            }
            printf("]\n");
        } else {
            fprintf(stderr, "llm_ai_mlx_moe_checkpoint: null result data\n");
        }
    }

    /* Cleanup. */
    mlx_array all2[] = {
        embed_w, embed_scales, embed_biases, indices, embed_w_row, embed_scales_row, embed_biases_row,
        x_flat, x, ln_w, normed,
        qw, qs, qb, kw, ks, kb, vw, vs, vb, q, k, v,
        q_heads, k_heads, v_heads, qn_w, kn_w, q_normed, k_normed,
        q_t, k_t, v_t, q_rope, k_rope, attn,
        attn_bt, attn_flat, ow, os, ob, attn_out, h,
        ln2_w, normed2,
        gate_w, gate_s, gate_b, gates, gates_sm, part, top_inds,
        gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
        down_proj_w, down_proj_s, down_proj_b,
        moe_out, layer_out,
    };
    for (size_t i = 0; i < sizeof(all2) / sizeof(all2[0]); i++) mlx_array_free(all2[i]);
    for (int i = 0; i < expert_tmp_n; i++) mlx_array_free(expert_tmp[i]);

    mlx_map_string_to_array_free(weights);
    mlx_map_string_to_string_free(metadata);
    mlx_stream_free(stream);

#undef CHK
    return ok ? 0 : -1;
}
