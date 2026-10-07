/* llm_dsv4_comp_test.c — test unitario dei kernel Metal del compressor (`dsv4_comp_pool`,
 * `dsv4_comp_fp8`) contro le reference CPU `llm_dsv4_compressor_pool_state` e
 * `llm_dsv4_fp8_kv_quantize_row`. Deterministico, niente modello. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_metal.h"
#include "llm_dsv4_cache.h"

static unsigned s_state = 987654321u;
static float fr(void) { s_state = s_state * 1103515245u + 12345u; return (float)((s_state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; }

static int test_pool(LlmDsv4Metal* b, int hd, int ratio, int co, int with_sentinel) {
    int width = co * hd, rows = co * ratio * width;
    float* skv = (float*)malloc(sizeof(float) * (size_t)rows);
    float* ssc = (float*)malloc(sizeof(float) * (size_t)rows);
    for (int i = 0; i < rows; i++) { skv[i] = fr() * 3.0f; ssc[i] = fr(); }
    if (with_sentinel) for (int j = 0; j < hd; j += 7) for (int r = 0; r < rows / width; r++) ssc[(size_t)r * width + j] = -1.0e30f;
    float* ref = (float*)malloc(sizeof(float) * (size_t)hd);
    float* gpu = (float*)malloc(sizeof(float) * (size_t)hd);
    llm_dsv4_compressor_pool_state(ref, skv, ssc, hd, ratio);
    int rc = llm_dsv4_metal_comp_pool(b, skv, ssc, gpu, hd, ratio, co);
    double mx = 0.0, ss = 0.0, sr = 0.0;
    for (int j = 0; j < hd; j++) { double d = fabs(ref[j] - gpu[j]); if (d > mx) mx = d; ss += d * d; sr += (double)ref[j] * ref[j]; }
    double rel = sr > 0.0 ? sqrt(ss / sr) : sqrt(ss);
    int ok = (rc == 0 && mx < 1e-4);
    printf("[POOL] hd=%d ratio=%d co=%d sent=%d rc=%d max|diff|=%.3e rel=%.3e %s\n",
           hd, ratio, co, with_sentinel, rc, mx, rel, ok ? "ok" : "BAD");
    free(skv); free(ssc); free(ref); free(gpu);
    return ok;
}

static int test_fp8(LlmDsv4Metal* b, int hd, int n_rot, float scale) {
    float* ref = (float*)malloc(sizeof(float) * (size_t)hd);
    float* gpu = (float*)malloc(sizeof(float) * (size_t)hd);
    for (int i = 0; i < hd; i++) { float v = fr() * scale; ref[i] = v; gpu[i] = v; }
    llm_dsv4_fp8_kv_quantize_row(ref, hd, n_rot);
    int rc = llm_dsv4_metal_comp_fp8(b, gpu, hd, n_rot);
    double mx = 0.0;
    for (int i = 0; i < hd; i++) { double d = fabs(ref[i] - gpu[i]); if (d > mx) mx = d; }
    int ok = (rc == 0 && mx == 0.0);
    printf("[FP8] hd=%d n_rot=%d scale=%.1f rc=%d max|diff|=%.3e %s\n", hd, n_rot, scale, rc, mx, ok ? "ok" : "BAD");
    free(ref); free(gpu);
    return ok;
}

static int test_shift(LlmDsv4Metal* b) {
    int rows = 8, width = 16, n = 4 * width, src = 4 * width, total = rows * width;
    float* ref = (float*)malloc(sizeof(float) * (size_t)total);
    float* gpu = (float*)malloc(sizeof(float) * (size_t)total);
    for (int i = 0; i < total; i++) { ref[i] = fr(); gpu[i] = ref[i]; }
    for (int i = 0; i < n; i++) ref[i] = ref[src + i];
    void* h = llm_dsv4_metal_buf_alloc(b, (size_t)total * sizeof(float));
    llm_dsv4_metal_buf_write(h, 0, gpu, (size_t)total * sizeof(float));
    int rc = llm_dsv4_metal_comp_shift(b, h, n, src);
    llm_dsv4_metal_buf_read(h, 0, gpu, (size_t)total * sizeof(float));
    double mx = 0.0;
    for (int i = 0; i < total; i++) { double d = fabs(ref[i] - gpu[i]); if (d > mx) mx = d; }
    int ok = (rc == 0 && mx == 0.0);
    printf("[SHIFT] rows=%d width=%d rc=%d max|diff|=%.3e %s\n", rows, width, rc, mx, ok ? "ok" : "BAD");
    llm_dsv4_metal_buf_free(h);
    free(ref); free(gpu);
    return ok;
}

int main(void) {
    if (!llm_dsv4_metal_available()) { printf("SKIP: nessun device Metal\n"); return 0; }
    LlmDsv4Metal* b = llm_dsv4_metal_create();
    if (!b) { printf("FAIL: llm_dsv4_metal_create\n"); return 1; }
    int ok = 1;
    ok &= test_pool(b, 128, 4, 2, 0);    /* compressor attn (ratio 4, co 2) */
    ok &= test_pool(b, 512, 4, 2, 0);    /* compressor attn reale (hd=512) */
    ok &= test_pool(b, 512, 4, 2, 1);
    ok &= test_pool(b, 128, 4, 2, 1);    /* con sentinelle -1e30 -> out 0 */
    ok &= test_pool(b, 128, 128, 1, 0);  /* compressor indexer (ratio 128, co 1) */
    ok &= test_fp8(b, 128, 64, 3.0f);    /* n_nope=64 */
    ok &= test_fp8(b, 256, 128, 0.5f);   /* due blocchi, valori piccoli */
    ok &= test_fp8(b, 256, 128, 900.0f); /* saturazione >448 */
    ok &= test_shift(b);
    printf(ok ? "PASS compressor: pool/fp8 bit-exact/compatibili con la reference CPU\n"
              : "FAIL compressor\n");
    llm_dsv4_metal_destroy(b);
    return ok ? 0 : 1;
}
