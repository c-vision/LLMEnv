/* llm_dsv4_metal_test.c — parità kernel Metal DS4 (IQ2_XXS=16 / Q2_K=10) vs
 * llm_dsv4_dot_tensor (CPU) su blocchi REALI del GGUF. SKIP se manca modello/Metal. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llm_gguf.h"
#include "llm_ggml.h"
#include "llm_dsv4.h"
#include "llm_dsv4_packed.h"
#include "llm_dsv4_metal.h"

static int test_tensor(LlmDsv4Metal* b, const LlmEngineMoeGguf* g, void* map,
                       const char* name, int exp_type, int in_dim, int out_dim) {
    int idx = llm_gguf_find_tensor(g, name);
    if (idx < 0) { printf("SKIP %-34s assente\n", name); return 0; }
    int t = g->ti[idx].type;
    if (t != exp_type) { printf("SKIP %-34s tipo %d != %d\n", name, t, exp_type); return 0; }
    int64_t nb_in = llm_ggml_nbytes(t, in_dim);
    size_t file_off = (size_t)(g->data - g->buf) + (size_t)g->ti[idx].offset;
    size_t slice = (size_t)out_dim * (size_t)nb_in;   /* primo esperto */
    size_t view_off = 0;
    void* h = llm_dsv4_metal_nocopy(b, map, file_off, slice, &view_off);
    if (!h) { printf("FAIL %-34s nocopy\n", name); return 1; }

    float* x = (float*)malloc(sizeof(float) * (size_t)in_dim);
    float* out = (float*)malloc(sizeof(float) * (size_t)out_dim);
    for (int i = 0; i < in_dim; i++) x[i] = 0.001f * (float)((i * 37) % 101) - 0.05f;

    if (llm_dsv4_metal_gemv(b, h, view_off, t, (size_t)nb_in,
                            (uint32_t)in_dim, (uint32_t)out_dim, x, out) != 0) {
        printf("FAIL %-34s gemv\n", name); llm_dsv4_metal_release(b, h); return 1;
    }

    const uint8_t* data = g->data + g->ti[idx].offset;
    int bad = 0; float worst = 0.0f;
    for (int o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        if (!llm_dsv4_dot_tensor(t, data + (size_t)o * (size_t)nb_in, in_dim, x, &ref)) {
            printf("FAIL %-34s dot_tensor\n", name); llm_dsv4_metal_release(b, h); return 1;
        }
        float rel = fabsf(out[o] - ref) / (fabsf(ref) + 1e-3f);
        if (rel > worst) worst = rel;
        if (rel > 2e-3f) bad++;
    }
    printf("%s %-34s worst_rel=%.3e\n", bad ? "FAIL" : "PASS", name, (double)worst);
    free(x); free(out); llm_dsv4_metal_release(b, h);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) { printf("SKIP gguf non apribile: %s\n", path); return 0; }
    if (!llm_dsv4_metal_available()) { printf("SKIP Metal assente\n"); return 0; }
    LlmDsv4Metal* b = llm_dsv4_metal_create();
    if (!b) { printf("SKIP backend non creato\n"); return 0; }

    LlmDsv4Config c = llm_dsv4_config_flash();
    int bad = 0;
    bad += test_tensor(b, &g, map, "blk.0.ffn_gate_exps.weight", 16, c.n_embd, c.n_ff_exp);
    bad += test_tensor(b, &g, map, "blk.0.ffn_up_exps.weight",   16, c.n_embd, c.n_ff_exp);
    bad += test_tensor(b, &g, map, "blk.0.ffn_down_exps.weight", 10, c.n_ff_exp, c.n_embd);

    printf("%s\n", bad ? "ALL FAIL" : "ALL PASS");
    llm_dsv4_metal_destroy(b);
    llm_gguf_close_file(&g, map, maplen);
    return bad ? 1 : 0;
}
