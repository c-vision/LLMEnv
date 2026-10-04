/* llm_dsv41_vision_test.c — verifica ViT + Aligner vs riferimento tiny. Build: tools/dsv41_vision_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_vision.h"
#include "llm_dsv41_vision_ref.h"

static int nclose(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float d = a[i]-b[i]; if (d<0) d=-d; float m = b[i]<0?-b[i]:b[i]; if (m<1.0f) m=1.0f;
        if (d > tol*m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; }
    }
    return 1;
}
#define ARR(p, nd, ...) llm_dsv41_array_f32(p, nd, (const int[]){ __VA_ARGS__ })

int main(void) {
    printf("llm_dsv41_vision — ViT + Aligner vs riferimento\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    const int D = REF_VIS_DIM, ID = REF_VIS_INTER, P = REF_VIS_PATCH, V = 3*P*P;
    LlmDsv41VisionWeights w; memset(&w, 0, sizeof w);
    w.patch_w = ARR(REF_VIS_PATCH_W, 2, D, V);
    w.patch_b = ARR(REF_VIS_PATCH_B, 1, D);
    w.n_blocks = REF_VIS_LAYERS;
    for (int b = 0; b < REF_VIS_LAYERS; b++) {
        LlmDsv41VisionBlock* B = &w.blk[b];
        const float *n1,*wqkv,*wqb,*wo,*wob,*n2,*w1,*w2;
        if (b==0){n1=REF_VIS_N10;wqkv=REF_VIS_WQKV0;wqb=REF_VIS_WQKVB0;wo=REF_VIS_WO0;wob=REF_VIS_WOB0;n2=REF_VIS_N20;w1=REF_VIS_W10;w2=REF_VIS_W20;}
        else     {n1=REF_VIS_N11;wqkv=REF_VIS_WQKV1;wqb=REF_VIS_WQKVB1;wo=REF_VIS_WO1;wob=REF_VIS_WOB1;n2=REF_VIS_N21;w1=REF_VIS_W11;w2=REF_VIS_W21;}
        B->n1 = ARR(n1,1,D); B->wqkv = ARR(wqkv,2,3*D,D); B->wqkvb = ARR(wqb,1,3*D);
        B->wo = ARR(wo,2,D,D); B->wob = ARR(wob,1,D); B->n2 = ARR(n2,1,D);
        B->w1 = ARR(w1,2,2*ID,D); B->w2 = ARR(w2,2,D,ID);
    }
    w.final_norm = ARR(REF_VIS_NORM,1,D);
    LlmDsv41AlignerWeights a; a.ratio = REF_VIS_RATIO;
    a.w1 = ARR(REF_VIS_AW1,2,REF_VIS_ALIGN_OUT,D*REF_VIS_RATIO*REF_VIS_RATIO); a.w1b = ARR(REF_VIS_AW1B,1,REF_VIS_ALIGN_OUT);
    a.w2 = ARR(REF_VIS_AW2,2,REF_VIS_ALIGN_OUT,REF_VIS_ALIGN_OUT); a.w2b = ARR(REF_VIS_AW2B,1,REF_VIS_ALIGN_OUT);

    mlx_array patches = ARR(REF_VIS_PATCHES, 2, REF_VIS_N, V);
    bool ok = true;
    mlx_array vit = llm_dsv41_vision_vit(&w, patches, REF_VIS_NH, REF_VIS_NW, D, REF_VIS_HEADS, 10000.0f, s, &ok);
    mlx_array_eval(vit);
    int fail = 0;
    if (ok && nclose(mlx_array_data_float32(vit), REF_VIS_VIT_OUT, REF_VIS_N*REF_VIS_OUT, 2e-3f))
        printf("PASS %-46s\n", "ViT -> output");
    else { printf("FAIL %-46s\n", "ViT -> output"); fail = 1; }
    if (ok) {
        mlx_array al = llm_dsv41_vision_aligner(&a, vit, REF_VIS_NH, REF_VIS_NW, D, REF_VIS_ALIGN_OUT, s, &ok);
        mlx_array_eval(al);
        int oh = (REF_VIS_NH + REF_VIS_RATIO - 1)/REF_VIS_RATIO, ow = (REF_VIS_NW + REF_VIS_RATIO - 1)/REF_VIS_RATIO;
        if (ok && nclose(mlx_array_data_float32(al), REF_VIS_ALIGNER_OUT, oh*ow*REF_VIS_ALIGN_OUT, 2e-3f))
            printf("PASS %-46s\n", "Aligner -> output");
        else { printf("FAIL %-46s\n", "Aligner -> output"); fail = 1; }
        mlx_array_free(al);
    } else { printf("FAIL alle %-46s\n", "vision forward"); fail = 1; }
    mlx_array_free(vit);
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
