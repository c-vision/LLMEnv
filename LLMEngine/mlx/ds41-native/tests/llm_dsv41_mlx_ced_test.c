/* llm_dsv41_mlx_ced_test.c — verifica CED prefill-skip vs riferimento tiny
 * (`llm_dsv41_ced_ref.h`). Build: tools/dsv41_ced_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_tensors_ced_ref.h"
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

int main(void) {
    printf("llm_dsv41_mlx_ced — CED prefill-skip vs riferimento\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_ced(&w)) { printf("FAIL load\n"); return 1; }
    mlx_array lg = mlx_array_new();
    if (!llm_dsv41_mlx_forward_ced(&w, REF_CED_IDS, REF_CED_S, &lg, s)) {
        printf("FAIL forward_ced\n"); llm_dsv41_weights_free(&w); return 1;
    }
    mlx_array_eval(lg);
    int n = (int)mlx_array_shape(lg)[1];
    const float* got = mlx_array_data_float32(lg);
    int fail = 0;
    printf("  logits shape [1,%d,64] (atteso tail=%d)\n", n, REF_CED_TAIL);
    if (n != REF_CED_TAIL) { printf("FAIL %-46s\n", "CED shape tail"); fail = 1; }
    if (nclose_tol(got, REF_CED_LOGITS, REF_CED_TAIL * REF_CED_VOCAB, TOL))
        printf("PASS %-46s\n", "CED prefill-skip -> logits");
    else { printf("FAIL %-46s\n", "CED prefill-skip -> logits"); fail = 1; }
    printf("  logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           got[0], got[1], got[2], got[3],
           REF_CED_LOGITS[0], REF_CED_LOGITS[1], REF_CED_LOGITS[2], REF_CED_LOGITS[3]);
    mlx_array_free(lg);
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
