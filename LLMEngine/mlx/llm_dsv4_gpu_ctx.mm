/* llm_dsv4_gpu_ctx.mm — contesto GPU DS4 (F4), singleton, ABI C.
 * Crea il backend Metal e le view (moe/attn) una sola volta; espone gli hook usati dal layer,
 * dall'attenzione e dall'engine. GPU attiva di default (fallback al path C se assente). */
#include <stdio.h>
#include <stdlib.h>
#include "llm_dsv4_gpu_ctx.h"
#include "llm_dsv4_gpu.h"
#include "llm_dsv4_gpu_moe.h"
#include "llm_dsv4_gpu_attn.h"

static LlmDsv4Metal*  g_be = NULL;
static LlmDsv4MoeGpu  g_moe;
static LlmDsv4AttnGpu g_attn;
static int g_on = 0;
static int g_ref = 0;

int llm_dsv4_gpu_ctx_active(void) { return g_on; }

int llm_dsv4_gpu_ctx_init(const LlmDsv4Config* c, const LlmDsv4PackedExperts* exps) {
    if (g_on) { g_ref++; return 1; }
    if (!c || !exps) {
        fprintf(stderr, "DSV4 GPU: OFF (config o esperti packed assenti)\n");
        return 0;
    }
    if (!llm_dsv4_metal_available()) {
        fprintf(stderr, "DSV4 GPU: OFF (nessun device Metal)\n");
        return 0;
    }
    g_be = llm_dsv4_metal_create();
    if (!g_be) { fprintf(stderr, "DSV4 GPU: OFF (backend Metal non creato)\n"); return 0; }
    if (!llm_dsv4_gpu_moe_init(&g_moe, g_be, c, exps)) {
        fprintf(stderr, "DSV4 GPU: OFF (moe init fallita: esperti non residenti? store/swap)\n");
        llm_dsv4_metal_destroy(g_be); g_be = NULL; return 0;
    }
    if (!llm_dsv4_gpu_attn_init(&g_attn, g_be, c, exps)) {
        fprintf(stderr, "DSV4 GPU: OFF (attn init fallita)\n");
        llm_dsv4_gpu_moe_destroy(&g_moe);
        llm_dsv4_metal_destroy(g_be); g_be = NULL; return 0;
    }
    g_on = 1; g_ref = 1;
    fprintf(stderr, "DSV4 GPU: contesto attivo (Metal; esperti IQ2_XXS/Q2_K + attn Q8_0)\n");
    return 1;
}

/* Re-init con lo STORE (cache esperti) attivo: al load l'init avviene prima dello swap, quindi
 * gli esperti non-residenti farebbero fallire la GPU. Chiamata dopo `enable_swap`. */
void llm_dsv4_gpu_ctx_reinit(const LlmDsv4Config* c, const LlmDsv4PackedExperts* exps) {
    if (!g_on) { llm_dsv4_gpu_ctx_init(c, exps); return; }
    llm_dsv4_gpu_moe_destroy(&g_moe);
    llm_dsv4_gpu_attn_destroy(&g_attn);
    if (!llm_dsv4_gpu_moe_init(&g_moe, g_be, c, exps)) {
        fprintf(stderr, "DSV4 GPU: reinit MoE fallita -> path C\n"); g_on = 0; return;
    }
    if (!llm_dsv4_gpu_attn_init(&g_attn, g_be, c, exps)) {
        fprintf(stderr, "DSV4 GPU: reinit attn fallita -> path C\n");
        llm_dsv4_gpu_moe_destroy(&g_moe); g_on = 0; return;
    }
    fprintf(stderr, "DSV4 GPU: ricaricato con cache esperti (streaming/GPU)\n");
}

void llm_dsv4_gpu_ctx_close(void) {
    if (!g_on) return;
    if (--g_ref > 0) return;
    llm_dsv4_gpu_moe_destroy(&g_moe);
    llm_dsv4_gpu_attn_destroy(&g_attn);
    llm_dsv4_metal_destroy(g_be);
    g_be = NULL; g_on = 0; g_ref = 0;
}

int llm_dsv4_gpu_moe_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                         const float* x, int token, float* out) {
    static int logged = 0;
    int r;
    if (!g_on) return 0;
    r = llm_dsv4_gpu_moe(&g_moe, c, L, il, x, token, out) ? 1 : 0;
    if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: MoE %s\n",
                                       r ? "eseguito su GPU" : "FALLITO -> fallback CPU"); }
    return r;
}

int llm_dsv4_gpu_attn_project_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                  const float* xn, int pos, float* q, float* kv, float* qrn) {
    static int logged = 0;
    int r;
    if (!g_on) return 0;
    r = llm_dsv4_gpu_attn_project(&g_attn, c, L, il, xn, pos, q, kv, qrn) ? 1 : 0;
    if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: attn-proj %s\n",
                                       r ? "eseguita su GPU" : "FALLITA -> fallback CPU"); }
    return r;
}

int llm_dsv4_gpu_attn_finish_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                 float* heads, int pos, float* out) {
    if (!g_on) return 0;
    return llm_dsv4_gpu_attn_finish(&g_attn, c, L, il, heads, pos, out) ? 1 : 0;
}
