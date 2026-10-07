/* llm_dsv4_gpu_moe_test.c — parità MoE GPU (kernel Metal) vs riferimento CPU f32,
 * su pesi REALI (layer 0) del GGUF. SKIP se manca modello/Metal. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llm_gguf.h"
#include "llm_ggml.h"
#include "llm_dsv4.h"
#include "llm_dsv4_weights.h"
#include "llm_dsv4_packed.h"
#include "llm_dsv4_gpu_moe.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) { printf("SKIP gguf: %s\n", path); return 0; }
    if (!llm_dsv4_metal_available()) { printf("SKIP Metal assente\n"); return 0; }

    LlmDsv4Config c = llm_dsv4_config_flash();
    const int ff = c.n_ff_exp, embd = c.n_embd, il = 0, token = 5;
    bool ok = true;
    LlmDsv4LayerWeights L; memset(&L, 0, sizeof L);
    L.ffn_gate_inp = llm_dsv4_lw(&g, il, "ffn_gate_inp", true, &ok);
    L.exp_probs_b  = llm_gguf_load_f32(&g, "blk.0.exp_probs_b.bias", NULL);
    L.ffn_gate_tid2eid = llm_dsv4_lw(&g, il, "ffn_gate_tid2eid", true, &ok);  /* il<3: hash routing */
    L.ffn_gate_shexp = llm_dsv4_lw(&g, il, "ffn_gate_shexp", true, &ok);
    L.ffn_up_shexp   = llm_dsv4_lw(&g, il, "ffn_up_shexp",   true, &ok);
    L.ffn_down_shexp = llm_dsv4_lw(&g, il, "ffn_down_shexp", true, &ok);
    if (!ok || !L.ffn_gate_inp) { printf("SKIP pesi layer 0 mancanti\n"); return 0; }

    LlmDsv4PackedExperts pe; memset(&pe, 0, sizeof pe);
    pe.gate = llm_dsv4_bind_packed(&g, il, "ffn_gate_exps", &pe.gate_type, &ok);
    pe.up   = llm_dsv4_bind_packed(&g, il, "ffn_up_exps",   &pe.up_type,   &ok);
    pe.down = llm_dsv4_bind_packed(&g, il, "ffn_down_exps", &pe.down_type, &ok);
    pe.in_dim = embd; pe.ff_dim = ff;
    if (!ok) { printf("SKIP esperti packed mancanti\n"); return 0; }

    /* array completo dei layer (come w->exps): view per ogni tensore */
    LlmDsv4PackedExperts* exps = (LlmDsv4PackedExperts*)calloc((size_t)c.n_layer, sizeof *exps);
    for (int l = 0; l < c.n_layer; l++) {
        bool lok = true;
        exps[l].gate = llm_dsv4_bind_packed(&g, l, "ffn_gate_exps", &exps[l].gate_type, &lok);
        exps[l].up   = llm_dsv4_bind_packed(&g, l, "ffn_up_exps",   &exps[l].up_type,   &lok);
        exps[l].down = llm_dsv4_bind_packed(&g, l, "ffn_down_exps", &exps[l].down_type, &lok);
        exps[l].in_dim = embd; exps[l].ff_dim = ff;
        if (!lok) { printf("SKIP layer %d esperti mancanti\n", l); return 0; }
    }

    float* x = (float*)malloc(sizeof(float) * embd);
    for (int i = 0; i < embd; i++) x[i] = 0.01f * (float)((i * 29) % 97) - 0.3f;

    /* riferimento CPU di PRODUZIONE: moe_forward_packed (Q8_K integer, come ds4) */
    pe.shexp_gate = llm_dsv4_bind_packed(&g, il, "ffn_gate_shexp", &pe.shexp_gate_type, &ok);
    pe.shexp_up   = llm_dsv4_bind_packed(&g, il, "ffn_up_shexp",   &pe.shexp_up_type,   &ok);
    pe.shexp_down = llm_dsv4_bind_packed(&g, il, "ffn_down_shexp", &pe.shexp_down_type, &ok);
    float* ref = (float*)calloc(embd, sizeof(float));
    llm_dsv4_moe_forward_packed(&c, &L, &pe, x, il, token, ref);

    /* GPU */
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    if (!be) { printf("SKIP backend\n"); return 0; }
    LlmDsv4MoeGpu moe; memset(&moe, 0, sizeof moe);
    if (!llm_dsv4_gpu_moe_init(&moe, be, &c, exps)) { printf("SKIP init views\n"); return 0; }
    float* out = calloc(embd, sizeof(float));
    if (!llm_dsv4_gpu_moe(&moe, &c, &L, il, x, token, out)) { printf("FAIL gpu_moe\n"); return 1; }

    int bad = 0; float worst = 0.0f;
    for (int o = 0; o < embd; o++) {
        float d = fabsf(out[o] - ref[o]);
        if (d > worst) worst = d;
        if (d > 3e-3f) bad++;
    }
    printf("%s MoE layer0 (6 esperti + shared)  worst_rel=%.3e\n", bad ? "FAIL" : "PASS", (double)worst);

    llm_dsv4_gpu_moe_destroy(&moe);
    llm_dsv4_metal_destroy(be);
    llm_gguf_close_file(&g, map, maplen);
    free(x); free(ref); free(out);
    printf("%s\n", bad ? "ALL FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
