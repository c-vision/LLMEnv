/* llm_dsv41_mlx_engram_decode_test.c — verifica Engram nel path DECODE (history slot 6) vs
 * riferimento tiny prefill+decode (`llm_dsv41_engram_decode_ref.h`). Build: tools/dsv41_engram_decode_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_tensors_engram_ref.h"
#include "llm_dsv41_mlx_cache.h"
#include "llm_dsv41_engram_decode_ref.h"

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
    printf("llm_dsv41_mlx_engram_decode — prefill+decode con Engram\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_engram(&w)) { printf("FAIL load\n"); return 1; }
    LlmDsv41Cache ck; llm_dsv41_cache_init(&ck);
    mlx_array lg = mlx_array_new();
    int fail = 0;

    if (!llm_dsv41_mlx_forward_cached(&w, &ck, REF_ENGD_IDS_PREFILL, REF_ENGD_SPREFILL, &lg, NULL, 0, NULL, s)) {
        printf("FAIL prefill\n"); llm_dsv41_weights_free(&w); return 1;
    }
    mlx_array_eval(lg);
    if (nclose_tol(mlx_array_data_float32(lg), REF_ENGD_LOGITS_PREFILL,
                   REF_ENGD_SPREFILL * REF_ENGD_VOCAB, TOL))
        printf("PASS %-46s\n", "prefill+Engram -> logits");
    else { printf("FAIL %-46s\n", "prefill+Engram -> logits"); fail = 1; }

    for (int t = 0; t < REF_ENGD_SDECODE; t++) {
        if (!llm_dsv41_mlx_forward_cached(&w, &ck, &REF_ENGD_IDS_DECODE[t], 1, &lg, NULL, 0, NULL, s)) {
            printf("FAIL decode %d\n", t); fail = 1; break;
        }
        mlx_array_eval(lg);
        char nm[64]; snprintf(nm, sizeof nm, "decode step %d + Engram -> logits", t);
        if (nclose_tol(mlx_array_data_float32(lg), REF_ENGD_LOGITS_DECODE + (size_t)t * REF_ENGD_VOCAB,
                       REF_ENGD_VOCAB, TOL))
            printf("PASS %-46s\n", nm);
        else { printf("FAIL %-46s\n", nm); fail = 1; }
    }

    mlx_array_free(lg);
    llm_dsv41_cache_free(&ck);
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
