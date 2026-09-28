/* llm_dsv41_mlx_csa_test.c — verifica del forward MLX CSA2 (ratio>0) vs riferimento tiny.
 * Carica il pak `llm_dsv41_fwd_csa_ref.h` (layer1 ratio=2, kv+index source) ed esegue il
 * forward MLX con compressor + KV compresso fp4. Nessun download.
 * Build: tools/dsv41_mlx_csa_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_tensors_csa_ref.h"
#include "llm_dsv41_mlx.h"

/* Tolleranza full: il riferimento usa i kernel Metal (riduzione on-line) -> rumore ~2.5e-4. */
#define TOL 5e-4f
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
    printf("llm_dsv41_mlx_csa — forward MLX CSA2 ratio>0 vs riferimento tiny\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Weights w;
    if (!llm_dsv41_mlx_weights_load_ref_csa(&w)) { printf("FAIL load_ref_csa\n"); return 1; }

    mlx_array logits = mlx_array_new();
    if (!llm_dsv41_mlx_forward(&w, REF_CSA_IDS, REF_CSA_S, &logits, s)) {
        printf("FAIL forward\n");
        llm_dsv41_weights_free(&w);
        return 1;
    }
    mlx_array_eval(logits);
    const float* got = mlx_array_data_float32(logits);
    const int n = REF_CSA_S * REF_CSA_VOCAB;
    printf("logits[0..3]=%.6g %.6g %.6g %.6g | ref=%.6g %.6g %.6g %.6g\n",
           got[0], got[1], got[2], got[3],
           REF_CSA_LOGITS[0], REF_CSA_LOGITS[1], REF_CSA_LOGITS[2], REF_CSA_LOGITS[3]);

    int fail = 0;
    if (nclose(got, REF_CSA_LOGITS, n)) printf("PASS %-50s\n", "forward MLX CSA2 -> logits");
    else { printf("FAIL %-50s\n", "forward MLX CSA2 -> logits"); fail = 1; }
    /* token 0: nessuna chiave compressa visibile -> deve combaciare quasi esattamente
     * (cattura una regressione della causalita' per-query delle chiavi compresse). */
    if (nclose_tol(got, REF_CSA_LOGITS, REF_CSA_VOCAB, 1e-5f))
        printf("PASS %-50s\n", "CSA2 token0 (causalita' compresse)");
    else { printf("FAIL %-50s\n", "CSA2 token0 (causalita' compresse)"); fail = 1; }

    mlx_array_free(logits);
    llm_dsv41_weights_free(&w);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
