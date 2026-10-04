/* llm_dsv41_mlx_cand_test.c — verifica dei CANDIDATE BLOCKS nel forward MLX vs riferimento tiny.
 * Build: tools/dsv41_cand_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_tensors_cand_ref.h"
#include "llm_dsv41_mlx_cache.h"

#define TOL 1e-3f
static int nclose_tol(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > tol * m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}

int main(void) {
    printf("llm_dsv41_mlx_cand — candidate blocks vs riferimento\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_cand(&w)) { printf("FAIL load\n"); return 1; }

    mlx_array lg = mlx_array_new();
    if (!llm_dsv41_mlx_forward(&w, REF_CAND_IDS, REF_CAND_S, &lg, s)) {
        printf("FAIL forward\n"); llm_dsv41_weights_free(&w); return 1;
    }
    mlx_array_eval(lg);
    const float* got = mlx_array_data_float32(lg);
    int n = REF_CAND_S * REF_CAND_VOCAB, fail = 0;
    if (nclose_tol(got, REF_CAND_LOGITS, n, TOL)) printf("PASS %-46s\n", "candidate blocks -> logits");
    else { printf("FAIL %-46s\n", "candidate blocks -> logits"); fail = 1; }
    if (nclose_tol(got, REF_CAND_LOGITS, REF_CAND_VOCAB, 1e-5f))
        printf("PASS %-46s\n", "candidate token0");
    else { printf("FAIL %-46s\n", "candidate token0"); fail = 1; }
    printf("  logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           got[0], got[1], got[2], got[3],
           REF_CAND_LOGITS[0], REF_CAND_LOGITS[1], REF_CAND_LOGITS[2], REF_CAND_LOGITS[3]);

    mlx_array_free(lg);
    /* path cache (prefill start=0): deve combaciare col forward non-cached. */
    {
        LlmDsv41Cache ck; llm_dsv41_cache_init(&ck);
        mlx_array lgc = mlx_array_new();
        if (!llm_dsv41_mlx_forward_cached(&w, &ck, REF_CAND_IDS, REF_CAND_S, &lgc, NULL, 0, NULL, s)) {
            printf("FAIL %-46s\n", "candidate -> cache forward"); fail = 1;
        } else {
            mlx_array_eval(lgc);
            if (nclose_tol(mlx_array_data_float32(lgc), REF_CAND_LOGITS, n, TOL))
                printf("PASS %-46s\n", "candidate -> cache logits");
            else { printf("FAIL %-46s\n", "candidate -> cache logits"); fail = 1; }
        }
        mlx_array_free(lgc);
        LlmDsv41Cache ck2; llm_dsv41_cache_init(&ck2);
        mlx_array lgp = mlx_array_new(), lgd = mlx_array_new();
        if (llm_dsv41_mlx_forward_cached(&w, &ck2, REF_CAND_IDS, REF_CAND_S - 1, &lgp, NULL, 0, NULL, s) &&
            llm_dsv41_mlx_forward_cached(&w, &ck2, &REF_CAND_IDS[REF_CAND_S - 1], 1, &lgd, NULL, 0, NULL, s)) {
            mlx_array_eval(lgd);
            const float* row = mlx_array_data_float32(lgd);
            const float* exp = REF_CAND_LOGITS + (size_t)(REF_CAND_S - 1) * REF_CAND_VOCAB;
            if (nclose_tol(row, exp, REF_CAND_VOCAB, TOL))
                printf("PASS %-46s\n", "candidate -> cache decode");
            else { printf("FAIL %-46s\n", "candidate -> cache decode"); fail = 1; }
        } else { printf("FAIL %-46s\n", "candidate -> cache decode fwd"); fail = 1; }
        mlx_array_free(lgp); mlx_array_free(lgd); llm_dsv41_cache_free(&ck2);
    }
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
