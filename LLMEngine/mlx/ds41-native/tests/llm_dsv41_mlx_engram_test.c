/* llm_dsv41_mlx_engram_test.c — verifica dell'integrazione Engram nel forward MLX vs
 * riferimento tiny (`llm_dsv41_engram_fwd_ref.h`). Nessun download.
 * Build: tools/dsv41_engram_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_tensors_engram_ref.h"
#include "llm_dsv41_mlx.h"

#define TOL 1e-3f
static int nclose_tol(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > tol * m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}
static int nclose(const float* a, const float* b, int n) { return nclose_tol(a, b, n, TOL); }

int main(void) {
    printf("llm_dsv41_mlx_engram — forward con Engram vs riferimento\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_engram(&w)) { printf("FAIL load\n"); return 1; }
    if (!w.layers[1].engram.present) { printf("FAIL engram layer\n"); return 1; }

    mlx_array lg = mlx_array_new();
    if (!llm_dsv41_mlx_forward(&w, REF_ENGF_IDS, REF_ENGF_S, &lg, s)) {
        printf("FAIL forward\n"); llm_dsv41_weights_free(&w); return 1;
    }
    mlx_array_eval(lg);
    const float* got = mlx_array_data_float32(lg);
    int fail = 0;
    if (nclose(got, REF_ENGF_LOGITS, REF_ENGF_S * REF_ENGF_VOCAB))
        printf("PASS %-46s\n", "forward con Engram -> logits");
    else { printf("FAIL %-46s\n", "forward con Engram -> logits"); fail = 1; }
    if (nclose_tol(got, REF_ENGF_LOGITS, REF_ENGF_VOCAB, 1e-5f))
        printf("PASS %-46s\n", "Engram token0 (hash+gate)");
    else { printf("FAIL %-46s\n", "Engram token0 (hash+gate)"); fail = 1; }
    printf("  logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           got[0], got[1], got[2], got[3],
           REF_ENGF_LOGITS[0], REF_ENGF_LOGITS[1], REF_ENGF_LOGITS[2], REF_ENGF_LOGITS[3]);

    mlx_array_free(lg);
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
