/* llm_dsv4_gpu_ctx.mm — contesto GPU DS4 (F4), singleton, ABI C.
 * Crea il backend Metal e le view (moe/attn) una sola volta; espone gli hook usati dal layer,
 * dall'attenzione e dall'engine. GPU attiva di default (fallback al path C se assente). */
#include <stdio.h>
#include <stdlib.h>
#include "llm_dsv4_gpu_ctx.h"
#include "llm_dsv4_gpu.h"
#include "llm_dsv4_gpu_moe.h"
#include "llm_dsv4_gpu_attn.h"
#include "llm_dsv4_store.h"

static LlmDsv4Metal*  g_be = NULL;
static LlmDsv4MoeGpu  g_moe;
static LlmDsv4AttnGpu g_attn;
static int g_on = 0;
static int g_ref = 0;

/* Tabella indirizzi esperti per layer (gate|up|down), riusata finche' lo store non sfratta. */
static uint64_t* g_addr_tbl[LLM_DSV4_MAX_LAYER];
static uint64_t  g_addr_ep[LLM_DSV4_MAX_LAYER];
static int       g_addr_ne = -1;

static const uint64_t* dsv4_addr_table(const LlmDsv4PackedExperts* pe, int il, int n_expert, int embd) {
    LlmDsv4Store* st = (LlmDsv4Store*)pe->store;
    uint64_t ep = st ? st->evictions : 0;
    if (il < 0 || il >= LLM_DSV4_MAX_LAYER) return NULL;
    if (g_addr_tbl[il] && g_addr_ep[il] == ep && g_addr_ne == n_expert) return g_addr_tbl[il];
    free(g_addr_tbl[il]);
    uint64_t* t = (uint64_t*)malloc(3 * (size_t)n_expert * sizeof(uint64_t));
    if (!t) return NULL;
    size_t nbg = (size_t)llm_ggml_nbytes(pe->gate_type, pe->in_dim);
    size_t nbu = (size_t)llm_ggml_nbytes(pe->up_type,   pe->in_dim);
    size_t nbd = (size_t)llm_ggml_nbytes(pe->down_type, pe->ff_dim);
    for (int e = 0; e < n_expert; e++) {
        /* Puntatori del MMAP: stabili e sempre validi (mai sfrattati/riusati come i buffer
         * dello store). La GPU li indirizza direttamente (memoria unificata). */
        const uint8_t* g = pe->gate + (size_t)e * (size_t)pe->ff_dim * nbg;
        const uint8_t* u = pe->up   + (size_t)e * (size_t)pe->ff_dim * nbu;
        const uint8_t* d = pe->down + (size_t)e * (size_t)embd * nbd;
        t[e] = (uint64_t)(uintptr_t)g;
        t[n_expert + e] = (uint64_t)(uintptr_t)u;
        t[2*n_expert + e] = (uint64_t)(uintptr_t)d;
    }
    g_addr_tbl[il] = t; g_addr_ep[il] = ep; g_addr_ne = n_expert;
    return t;
}

int llm_dsv4_gpu_attn_tail_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, int il, const float* heads, const float* saved_attn,
        const float* post_attn, const float* comb_attn, int pos, float* out_hc) {
    if (!g_on || !g_be || !g_attn.inited || !c || !L || !pe) return 0;
    if (il < c->n_hash_layer) return 0;                          /* hash routing: path C */
    if (g_attn.outa[il].type != 8 || g_attn.outb[il].type != 8) return 0;
    if (g_moe.gate[il].type != 16 || g_moe.up[il].type != 16 || g_moe.down[il].type != 10) return 0; /* pair IQ2 + down Q2_K */
    int n_expert = c->n_expert;
    const uint64_t* tbl = dsv4_addr_table(pe, il, n_expert, c->n_embd);
    if (!tbl) return 0;
    const int ng = c->n_out_group, rank = c->n_lora_o, embd = c->n_embd, HC = c->n_hc;
    const int gd = c->n_head_dim * (c->n_head / ng), ol = ng * rank;
    int compressed = llm_dsv4_layer_compress_ratio(c, il) != 0;
    float freq_base = (compressed && c->compress_rope_freq_base > 0.0f) ? c->compress_rope_freq_base : c->rope_freq_base;
    float freq_scale = (!compressed || c->rope_scale_factor <= 0.0f) ? 1.0f : 1.0f / c->rope_scale_factor;
    float ext_factor = (compressed && c->rope_scale_factor > 1.0f) ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    LlmDsv4TailParams q; memset(&q, 0, sizeof q);
    q.outa_w = g_attn.outa[il].w; q.outa_row_bytes = g_attn.outa[il].row_bytes; q.ng = ng; q.gd = (uint32_t)gd; q.rank = (uint32_t)rank;
    q.outb_w = g_attn.outb[il].w; q.outb_row_bytes = g_attn.outb[il].row_bytes; q.ol = (uint32_t)ol;
    q.n_head = c->n_head; q.hd = c->n_head_dim; q.n_rot = c->n_rot; q.pos = pos;
    q.rope_cfg[0]=freq_base; q.rope_cfg[1]=freq_scale; q.rope_cfg[2]=ext_factor; q.rope_cfg[3]=attn_factor;
    q.rope_cfg[4]=c->rope_yarn_beta_fast; q.rope_cfg[5]=c->rope_yarn_beta_slow;
    q.rope_cfg[6]=(float)(compressed ? c->rope_orig_ctx : 0);
    q.res_attn = saved_attn; q.post_attn = post_attn; q.comb_attn = comb_attn;
    q.fn = L->hc_ffn_fn; q.scale = L->hc_ffn_scale; q.base = L->hc_ffn_base; q.norm_w = L->ffn_norm;
    q.router_w = L->ffn_gate_inp; q.exp_probs_b = L->exp_probs_b;
    q.n_expert = n_expert; q.used = c->n_expert_used; q.expert_scale = c->expert_weight_scale;
    q.gaddr = tbl; q.uaddr = tbl + n_expert; q.daddr = tbl + 2*n_expert;
    q.clamp = c->swiglu_clamp_exp;
    q.gu_nb01 = (uint32_t)llm_ggml_nbytes(pe->gate_type, pe->in_dim);
    q.ff = (uint32_t)c->n_ff_exp;
    q.down_nb01 = (uint32_t)llm_ggml_nbytes(pe->down_type, c->n_ff_exp);
    q.sg = pe->shexp_gate; q.su = pe->shexp_up; q.sd = pe->shexp_down;
    q.s_nb_g = (uint32_t)llm_ggml_nbytes(pe->shexp_gate_type, embd);
    q.s_nb_d = (uint32_t)llm_ggml_nbytes(pe->shexp_down_type, c->n_ff_exp);
    q.HC = HC; q.D = embd; q.SIZE = (2 + HC) * HC; q.HCD = HC * embd;
    q.sinkhorn_iters = c->n_hc_sinkhorn_iter; q.hc_eps = c->hc_eps; q.rms_eps = c->rms_eps;
    q.heads = heads; q.out_hc = out_hc;
    return llm_dsv4_metal_layer_tail(g_be, &q) == 0;
}

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
