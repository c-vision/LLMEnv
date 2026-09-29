/* llm_dsv41_real_dequant_test.c — verifica del loader C su TUTTI i tipi di tensore del layer 0
 * reale (fp8 32x32, fp4 e2m1, bf16/f32 plain). Richiede shard reali + index.
 * Build: tools/dsv41_real_dequant_test.sh */
#include <stdio.h>
#include "llm_engine_opts.h"
#include <math.h>
#include <stdlib.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_loader.h"
#include "llm_dsv41_real_ref.h"
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llm_dsv41_real_dequant — dequant su pesi reali (layer 0 + vision)\n");
    const char* dir = llm_ai_mlx_opt_env("DSV41_MODEL_DIR");
    if (!dir) dir = "/Volumes/Armor5TB/LocalLLM/ai/models/DeepSeek-V4.1-Flash";
    mlx_stream s = mlx_default_cpu_stream_new();
    LlmDsv41Ckpt ck;
    if (!llm_dsv41_ckpt_open(&ck, dir)) { printf("FAIL ckpt_open %s\n", dir); return 2; }
    int fail = 0;
    for (int t = 0; t < REF_RT_COUNT; t++) {
        bool ok = true;
        mlx_array a = (REF_RT_MODE[t] == 0)
            ? llm_dsv41_load_plain(&ck, REF_RT_NAME[t], s, &ok)
            : llm_dsv41_load_weight(&ck, REF_RT_NAME[t], REF_RT_OUT[t], REF_RT_IN[t],
                                    REF_RT_MODE[t] == 2, s, &ok);
        if (!ok) { printf("FAIL %-42s load\n", REF_RT_NAME[t]); fail++; continue; }
        mlx_array_eval(a);
        const float* p = mlx_array_data_float32(a);
        if (!p) { printf("FAIL %-42s data\n", REF_RT_NAME[t]); fail++; continue; }
        int bad = 0;
        for (int i = 0; i < REF_RT_NS[t]; i++) {
            float got = p[REF_RT_IDX[t][i]], exp = REF_RT_VAL[t][i];
            float d = fabsf(got - exp), m = fabsf(exp) > 1e-3f ? fabsf(exp) : 1e-3f;
            if (d > 1e-3f * m) { if (bad++ < 3) printf("  [%ld] got=%.6g exp=%.6g\n", REF_RT_IDX[t][i], got, exp); }
        }
        long n = (REF_RT_MODE[t] == 0) ? REF_RT_OUT[t]
                                       : (long)REF_RT_OUT[t] * REF_RT_IN[t];
        double sum = 0.0;
        for (long i = 0; i < n; i++) sum += (double)p[i];
        double rel = fabs(sum - REF_RT_SUM[t]) / (fabs(REF_RT_SUM[t]) > 1 ? fabs(REF_RT_SUM[t]) : 1);
        int good = (bad == 0 && rel < 1e-4);
        printf("%s %-42s samples=%s sum_rel=%.2e\n", good ? "PASS" : "FAIL",
               REF_RT_NAME[t], bad ? "BAD" : "ok", rel);
        if (!good) fail++;
    }
    llm_dsv41_ckpt_close(&ck);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
