/* llm_dsv41_mlx_test.c — verifica del forward MLX V4.1 vs il riferimento tiny (REF_FW_LOGITS).
 *
 * Carica i pesi del pak tiny (llm_dsv41_fwd_ref.h) in `LlmDsv41Weights` con array MLX, esegue
 * `llm_dsv41_mlx_forward` sugli stessi input del riferimento (ids {3,7}) e confronta i logits.
 * Nessun download, nessun modello reale.
 *
 * Build (linka libmlxc.a/libmlx.a di build/_deps):
 *   see tools/dsv41_mlx_test.sh
 */
#include <stdio.h>
#include <math.h>

#include <mlx/c/mlx.h>

#include "llm_dsv41_tensors_ref.h"
#include "llm_dsv41_mlx_cache.h"

#define TOL 5e-4f
static int nclose(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        float m = b[i] < 0 ? -b[i] : b[i]; if (m < 1.0f) m = 1.0f;
        if (d > TOL * m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}

int main(void) {
    printf("llm_dsv41_mlx — forward MLX vs riferimento tiny (REF_FW_LOGITS)\n");
    mlx_stream s = mlx_default_gpu_stream_new();

    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_tiny(&w)) {
        printf("FAIL load_ref_tiny\n");
        return 1;
    }

    mlx_array logits = mlx_array_new();
    if (!llm_dsv41_mlx_forward(&w, REF_FW_IDS, REF_FW_L, &logits, s)) {
        printf("FAIL forward\n");
        llm_dsv41_weights_free(&w);
        return 1;
    }
    mlx_array_eval(logits);
    const float* got = mlx_array_data_float32(logits);
    const int n = REF_FW_L * REF_FW_VOCAB;
    printf("logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           got[0], got[1], got[2], got[3],
           REF_FW_LOGITS[0], REF_FW_LOGITS[1], REF_FW_LOGITS[2], REF_FW_LOGITS[3]);

    int fail = 0;
    if (nclose(got, REF_FW_LOGITS, n)) printf("PASS %-50s\n", "forward MLX -> logits");
    else { printf("FAIL %-50s\n", "forward MLX -> logits"); fail = 1; }

    mlx_array_free(logits);
    /* path cache (prefill start=0): deve combaciare col forward non-cached (ratio 0). */
    {
        LlmDsv41Cache ck; llm_dsv41_cache_init(&ck);
        mlx_array lgc = mlx_array_new();
        if (!llm_dsv41_mlx_forward_cached(&w, &ck, REF_FW_IDS, REF_FW_L, &lgc, NULL, 0, NULL, s)) {
            printf("FAIL %-50s\n", "forward cached -> logits"); fail = 1;
        } else {
            mlx_array_eval(lgc);
            if (nclose(mlx_array_data_float32(lgc), REF_FW_LOGITS, n))
                printf("PASS %-50s\n", "forward cached -> logits");
            else { printf("FAIL %-50s\n", "forward cached -> logits"); fail = 1; }
        }
        mlx_array_free(lgc); llm_dsv41_cache_free(&ck);
    }
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
