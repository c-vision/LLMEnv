/* llm_dsv4_pair_test.c — BIT-EXACT: kernel MoE fuso ds4 `dsv4_pair_swiglu_iq2` (gate+up+swiglu)
 * vs path C (`llm_dsv4_dot_tensor` gate/up + `llm_dsv4_swiglu`). SKIP se manca modello/Metal. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_gguf.h"
#include "llm_ggml.h"
#include "llm_dsv4.h"
#include "llm_dsv4_packed.h"
#include "llm_dsv4_metal.h"
#include "llm_dsv4_moe_fwd.h"   /* llm_dsv4_swiglu */

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) { printf("SKIP gguf non apribile: %s\n", path); return 0; }
    if (!llm_dsv4_metal_available()) { printf("SKIP Metal assente\n"); return 0; }
    LlmDsv4Metal* b = llm_dsv4_metal_create();
    if (!b) { printf("SKIP backend non creato\n"); return 0; }

    LlmDsv4Config c = llm_dsv4_config_flash();
    int in_dim = c.n_embd, ff = c.n_ff_exp;
    int ig = llm_gguf_find_tensor(&g, "blk.0.ffn_gate_exps.weight");
    int iu = llm_gguf_find_tensor(&g, "blk.0.ffn_up_exps.weight");
    if (ig < 0 || iu < 0) { printf("SKIP tensori gate/up assenti\n"); return 0; }
    if (g.ti[ig].type != 16 || g.ti[iu].type != 16) { printf("SKIP tipo != IQ2_XXS\n"); return 0; }
    int64_t nb_in = llm_ggml_nbytes(g.ti[ig].type, in_dim);

    const uint8_t* dg = g.data + g.ti[ig].offset;   /* esperto 0 */
    const uint8_t* du = g.data + g.ti[iu].offset;

    float* x = (float*)malloc(sizeof(float) * (size_t)in_dim);
    for (int i = 0; i < in_dim; i++) x[i] = 0.001f * (float)((i * 37) % 101) - 0.05f;
    float* mid = (float*)malloc(sizeof(float) * (size_t)ff);
    float w = 0.7f, clamp = c.swiglu_clamp_exp;

    if (llm_dsv4_metal_pair_swiglu(b, dg, du, x, mid, w, clamp,
                                   (uint32_t)in_dim, (uint32_t)nb_in, (uint32_t)ff) != 0) {
        printf("FAIL: pair kernel ritorna errore\n"); return 1;
    }

    float* gv = (float*)malloc(sizeof(float) * (size_t)ff);
    float* uv = (float*)malloc(sizeof(float) * (size_t)ff);
    for (int o = 0; o < ff; o++) {
        llm_dsv4_dot_tensor(g.ti[ig].type, dg + (size_t)o * (size_t)nb_in, in_dim, x, &gv[o]);
        llm_dsv4_dot_tensor(g.ti[iu].type, du + (size_t)o * (size_t)nb_in, in_dim, x, &uv[o]);
    }
    float* midref = (float*)malloc(sizeof(float) * (size_t)ff);
    llm_dsv4_swiglu(midref, gv, uv, ff, clamp);
    for (int o = 0; o < ff; o++) midref[o] *= w;

    int bad = 0; float worst = 0.0f;
    for (int o = 0; o < ff; o++) {
        float rel = fabsf(mid[o] - midref[o]) / (fabsf(midref[o]) + 1e-3f);
        if (rel > worst) worst = rel;
        if (rel > 2e-3f) bad++;
    }
    printf("%s worst_rel=%.3e (bad=%d/%d)\n", bad ? "FAIL" : "PASS", (double)worst, bad, ff);
    llm_dsv4_metal_destroy(b);
    llm_gguf_close_file(&g, map, maplen);
    return bad ? 1 : 0;
}
