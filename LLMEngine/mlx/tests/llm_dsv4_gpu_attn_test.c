/* llm_dsv4_gpu_attn_test.c — parità attenzione GPU (proiezioni Q8_0) vs CPU f32, pesi REALI. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llm_gguf.h"
#include "llm_ggml.h"
#include "llm_dsv4.h"
#include "llm_dsv4_weights.h"
#include "llm_dsv4_attn_fwd.h"
#include "llm_dsv4_gpu_attn.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) { printf("SKIP gguf: %s\n", path); return 0; }
    if (!llm_dsv4_metal_available()) { printf("SKIP Metal assente\n"); return 0; }

    LlmDsv4Config c = llm_dsv4_config_flash();
    const int il = 0, pos = 7, embd = c.n_embd, qd = c.n_head * c.n_head_dim, hd = c.n_head_dim, lq = c.n_lora_q;
    bool ok = true;
    LlmDsv4LayerWeights L; memset(&L, 0, sizeof L);
    L.attn_q_a        = llm_dsv4_lw(&g, il, "attn_q_a", true, &ok);
    L.attn_q_a_norm   = llm_dsv4_lw(&g, il, "attn_q_a_norm", true, &ok);
    L.attn_q_b        = llm_dsv4_lw(&g, il, "attn_q_b", true, &ok);
    L.attn_kv         = llm_dsv4_lw(&g, il, "attn_kv", true, &ok);
    L.attn_kv_a_norm  = llm_dsv4_lw(&g, il, "attn_kv_a_norm", true, &ok);
    L.attn_output_a   = llm_dsv4_lw(&g, il, "attn_output_a", true, &ok);
    L.attn_output_b   = llm_dsv4_lw(&g, il, "attn_output_b", true, &ok);
    if (!ok) { printf("SKIP pesi attn layer 0 mancanti\n"); return 0; }

    LlmDsv4PackedExperts* exps = (LlmDsv4PackedExperts*)calloc((size_t)c.n_layer, sizeof *exps);
    for (int l = 0; l < c.n_layer; l++) {
        bool lok = true;
        exps[l].q_a   = llm_dsv4_bind_packed(&g, l, "attn_q_a",       &exps[l].q_a_type,   &lok);
        exps[l].q_b   = llm_dsv4_bind_packed(&g, l, "attn_q_b",       &exps[l].q_b_type,   &lok);
        exps[l].kv    = llm_dsv4_bind_packed(&g, l, "attn_kv",        &exps[l].kv_type,    &lok);
        exps[l].out_a = llm_dsv4_bind_packed(&g, l, "attn_output_a",  &exps[l].out_a_type, &lok);
        exps[l].out_b = llm_dsv4_bind_packed(&g, l, "attn_output_b",  &exps[l].out_b_type, &lok);
        if (!lok) { printf("SKIP layer %d proiezioni mancanti\n", l); return 0; }
    }

    float* xn = (float*)malloc(sizeof(float) * embd);
    for (int i = 0; i < embd; i++) xn[i] = 0.01f * (float)((i * 31) % 101) - 0.4f;

    /* CPU f32 (pe=NULL -> usa i pesi f32 = dequant dei Q8_0) */
    float *qr_ = malloc(4*qd), *kvr = malloc(4*hd), *qnr = malloc(4*lq);

    /* GPU */
    LlmDsv4Metal* be = llm_dsv4_metal_create();
    if (!be) { printf("SKIP backend\n"); return 0; }
    LlmDsv4AttnGpu attn; memset(&attn, 0, sizeof attn);
    if (!llm_dsv4_gpu_attn_init(&attn, be, &c, exps)) { printf("SKIP init views attn\n"); return 0; }
    float *qg = malloc(4*qd), *kvg = malloc(4*hd), *qng = malloc(4*lq);

    int bad = 0;
    llm_dsv4_attn_project(&c, &L, NULL, xn, pos, il, qr_, kvr, qnr);
    if (!llm_dsv4_gpu_attn_project(&attn, &c, &L, il, xn, pos, qg, kvg, qng)) { printf("FAIL gpu project\n"); return 1; }
    { float w = 0.0f; for (int i = 0; i < qd; i++) { float r = fabsf(qg[i]-qr_[i])/(fabsf(qr_[i])+1e-3f); if(r>w)w=r; if(r>2e-3f)bad++; }
      printf("%s gpu attn q    worst_rel=%.3e\n", bad?"FAIL":"PASS", (double)w); }
    { int b2 = 0; float w = 0.0f; for (int i = 0; i < hd; i++) { float r = fabsf(kvg[i]-kvr[i])/(fabsf(kvr[i])+1e-3f); if(r>w)w=r; if(r>2e-3f)b2++; }
      printf("%s gpu attn kv   worst_rel=%.3e\n", b2?"FAIL":"PASS", (double)w); bad += b2; }

    /* finish (output grouped-LoRA) */
    float *hr = malloc(4*qd), *hg = malloc(4*qd), *or = malloc(4*embd), *og = malloc(4*embd);
    for (int i = 0; i < qd; i++) { float v = 0.02f*(float)((i*17)%71) - 0.5f; hr[i]=v; hg[i]=v; }
    llm_dsv4_attn_finish(&c, &L, NULL, hr, pos, il, or);
    if (!llm_dsv4_gpu_attn_finish(&attn, &c, &L, il, hg, pos, og)) { printf("FAIL gpu finish\n"); return 1; }
    { int b3 = 0; float w = 0.0f; for (int i = 0; i < embd; i++) { float r = fabsf(og[i]-or[i])/(fabsf(or[i])+1e-3f); if(r>w)w=r; if(r>5e-3f)b3++; }
      printf("%s gpu attn out  worst_rel=%.3e\n", b3?"FAIL":"PASS", (double)w); bad += b3; }

    printf("%s\n", bad ? "ALL FAIL" : "ALL PASS");
    llm_dsv4_gpu_attn_destroy(&attn);
    llm_dsv4_metal_destroy(be);
    llm_gguf_close_file(&g, map, maplen);
    return bad ? 1 : 0;
}
