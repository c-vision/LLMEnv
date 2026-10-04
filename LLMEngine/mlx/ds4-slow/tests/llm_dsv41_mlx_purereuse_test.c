/* llm_dsv41_mlx_purereuse_test.c — verifica condivisione cross-layer (index_source-only) vs
 * riferimento tiny (`llm_dsv41_purereuse_ref.h`). Build: tools/dsv41_share_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_tensors_purereuse_ref.h"
#include "llm_dsv41_mlx_cache.h"

#define TOL 5e-4f
static int nclose_tol(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > tol * m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}

int main(void) {
    printf("llm_dsv41_mlx_purereuse — condivisione cross-layer (index_source-only)\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_purereuse(&w)) { printf("FAIL load\n"); return 1; }
    mlx_array lg = mlx_array_new();
    if (!llm_dsv41_mlx_forward(&w, REF_PR_IDS, REF_PR_S, &lg, s)) {
        printf("FAIL forward\n"); llm_dsv41_weights_free(&w); return 1;
    }
    mlx_array_eval(lg);
    const float* got = mlx_array_data_float32(lg);
    int fail = 0;
    if (nclose_tol(got, REF_PR_LOGITS, REF_PR_S * REF_PR_VOCAB, TOL))
        printf("PASS %-46s\n", "pure-reuse -> logits");
    else { printf("FAIL %-46s\n", "pure-reuse -> logits"); fail = 1; }
    if (nclose_tol(got, REF_PR_LOGITS, REF_PR_VOCAB, 1e-5f))
        printf("PASS %-46s\n", "pure-reuse token0");
    else { printf("FAIL %-46s\n", "pure-reuse token0"); fail = 1; }
    mlx_array_free(lg);
    /* path cache (prefill start=0): deve combaciare con il forward non-cached. */
    {
        LlmDsv41Cache ck; llm_dsv41_cache_init(&ck);
        mlx_array lgc = mlx_array_new();
        if (!llm_dsv41_mlx_forward_cached(&w, &ck, REF_PR_IDS, REF_PR_S, &lgc, NULL, 0, NULL, s)) {
            printf("FAIL %-46s\n", "share -> cache forward"); fail = 1;
        } else {
            mlx_array_eval(lgc);
            if (nclose_tol(mlx_array_data_float32(lgc), REF_PR_LOGITS, REF_PR_S * REF_PR_VOCAB, TOL))
                printf("PASS %-46s\n", "pure-reuse -> cache logits");
            else { printf("FAIL %-46s\n", "pure-reuse -> cache logits"); fail = 1; }
        }
        mlx_array_free(lgc);
        /* decode: prefill S-1 poi 1 token; l'ultima riga deve combaciare col riferimento. */
        LlmDsv41Cache ck2; llm_dsv41_cache_init(&ck2);
        mlx_array lgp = mlx_array_new(), lgd = mlx_array_new();
        if (llm_dsv41_mlx_forward_cached(&w, &ck2, REF_PR_IDS, REF_PR_S - 1, &lgp, NULL, 0, NULL, s) &&
            llm_dsv41_mlx_forward_cached(&w, &ck2, &REF_PR_IDS[REF_PR_S - 1], 1, &lgd, NULL, 0, NULL, s)) {
            mlx_array_eval(lgd);
            const float* row = mlx_array_data_float32(lgd);
            const float* exp = REF_PR_LOGITS + (size_t)(REF_PR_S - 1) * REF_PR_VOCAB;
            if (nclose_tol(row, exp, REF_PR_VOCAB, TOL))
                printf("PASS %-46s\n", "pure-reuse -> cache decode");
            else { printf("FAIL %-46s\n", "pure-reuse -> cache decode"); fail = 1; }
        } else { printf("FAIL %-46s\n", "pure-reuse -> cache decode fwd"); fail = 1; }
        mlx_array_free(lgp); mlx_array_free(lgd); llm_dsv41_cache_free(&ck2);
    }
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
