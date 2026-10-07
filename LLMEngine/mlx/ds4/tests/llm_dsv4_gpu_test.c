/* llm_dsv4_gpu_test.c — parità GPU (MLX) vs C del blocco mHC DS4 (F1).
 * Compila con: mlx include + link mlxc (vedi come gli altri test mlx del repo).
 * Gate: se LLM_DSV4_GPU e' off o la GPU fallisce, esce SKIP (nessun falso PASS). */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_block.h"   /* path C di riferimento */
#include "llm_dsv4_gpu.h"     /* path GPU (riuso llm_dsv41_mlx_*) */

static int cmp(const char* what, const float* a, const float* b, int n, float tol) {
    int bad = 0; float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        float rel = d / (fabsf(b[i]) + 1e-4f);
        if (rel > worst) worst = rel;
        if (rel > tol) bad++;
    }
    printf("%s %-28s worst_rel=%.3e\n", bad ? "FAIL" : "PASS", what, (double)worst);
    return bad;
}

int main(void) {
    LlmDsv4Config c = llm_dsv4_config_flash();
    c.n_embd = 8; c.n_hc = 4;   /* tiny, come llm_dsv4_block_test */
    const int HC = c.n_hc, D = c.n_embd, SIZE = (2 + HC) * HC;
    float fn[SIZE * HC * D], scale[3], base[SIZE], norm[D], res[HC * D];
    for (int i = 0; i < SIZE * HC * D; i++) fn[i] = 0.05f * (float)((i * 7) % 13) - 0.3f;
    scale[0] = 0.3f; scale[1] = 0.4f; scale[2] = 0.5f;
    for (int i = 0; i < SIZE; i++) base[i] = 0.1f * (float)((i * 3) % 5) - 0.2f;
    for (int i = 0; i < D; i++) norm[i] = 0.5f + 0.05f * (float)(i % 7);
    for (int i = 0; i < HC * D; i++) res[i] = 0.1f * (float)((i * 5) % 9) - 0.4f;

    /* riferimento C */
    float rx[D], rpost[HC], rcomb[HC * HC];
    llm_dsv4_block_pre(&c, fn, scale, base, norm, res, rx, rpost, rcomb);

    mlx_stream s = mlx_default_gpu_stream_new();

    float gx[D], gpost[HC], gcomb[HC * HC];
    if (!llm_dsv4_gpu_block_pre(&c, fn, scale, base, norm, res, s, gx, gpost, gcomb)) {
        printf("FAIL llm_dsv4_gpu_block_pre (fallback richiesto)\n"); return 1;
    }
    int bad = 0;
    bad += cmp("gpu block_pre x", gx, rx, D, 2e-3f);
    bad += cmp("gpu block_pre post", gpost, rpost, HC, 2e-3f);
    bad += cmp("gpu block_pre comb", gcomb, rcomb, HC * HC, 2e-3f);

    /* block_post */
    float xd[D], rout[HC * D], gout[HC * D];
    for (int i = 0; i < D; i++) xd[i] = 0.2f * (float)((i * 3) % 7) - 0.5f;
    llm_dsv4_block_post(&c, xd, res, rpost, rcomb, rout);
    if (!llm_dsv4_gpu_block_post(&c, xd, res, gpost, gcomb, s, gout)) {
        printf("FAIL llm_dsv4_gpu_block_post (fallback richiesto)\n"); return 1;
    }
    bad += cmp("gpu block_post out", gout, rout, HC * D, 2e-3f);

    printf("%s\n", bad ? "ALL FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
