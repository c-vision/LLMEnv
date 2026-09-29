/* llm_dsv4_gpu_ctx.mm — contesto GPU DS4 (F4), singleton, ABI C.
 * Crea il backend Metal e le view (moe/attn) una sola volta; espone gli hook usati dal layer,
 * dall'attenzione e dall'engine. GPU attiva di default (fallback al path C se assente). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_gpu_ctx.h"
#include "llm_dsv4_gpu.h"
#include "llm_dsv4_gpu_moe.h"
#include "llm_dsv4_gpu_attn.h"
#include "llm_dsv4_store.h"

#include "llm_dsv4_store.h"

static LlmDsv4Metal*  g_be = NULL;
static LlmDsv4MoeGpu  g_moe;
static LlmDsv4AttnGpu g_attn;
static int g_on = 0;
static int g_ref = 0;

/* Argument buffer per layer sugli ESPERTI dello STORE (residenti, veloci); ricostruito solo
 * quando lo store sfratta (epoch). Il mmap sarebbe stabile ma NON residente sotto streaming. */
static void* g_argp[LLM_DSV4_MAX_LAYER]; static void* g_argd[LLM_DSV4_MAX_LAYER];
static int g_arg_ready[LLM_DSV4_MAX_LAYER]; static uint64_t g_arg_ep[LLM_DSV4_MAX_LAYER];
static int dsv4_build_argbufs(const LlmDsv4PackedExperts* pe, int il, int n_expert) {
    if (il < 0 || il >= LLM_DSV4_MAX_LAYER) return 0;
    LlmDsv4Store* st = (LlmDsv4Store*)pe->store;
    uint64_t ep = st ? st->evictions : 0;
    if (g_arg_ready[il] && g_arg_ep[il] == ep) return 1;
    size_t nbg = (size_t)llm_ggml_nbytes(pe->gate_type, pe->in_dim);
    size_t nbd = (size_t)llm_ggml_nbytes(pe->down_type, pe->ff_dim);
    size_t gbytes = (size_t)pe->ff_dim * nbg, dbytes = (size_t)pe->in_dim * nbd;
    const void** gp = (const void**)malloc((size_t)n_expert*sizeof(void*));
    const void** up = (const void**)malloc((size_t)n_expert*sizeof(void*));
    const void** dp = (const void**)malloc((size_t)n_expert*sizeof(void*));
    if (!gp || !up || !dp) { free(gp); free(up); free(dp); return 0; }
    for (int e = 0; e < n_expert; e++) {
        const uint8_t *g = NULL, *u = NULL, *d = NULL;
        if (pe->store && pe->store_get) pe->store_get(pe->store, il, e, &g, &u, &d);
        else { g = pe->gate + (size_t)e*gbytes; u = pe->up + (size_t)e*gbytes; d = pe->down + (size_t)e*dbytes; }
        gp[e] = g; up[e] = u; dp[e] = d;
    }
    if (g_argp[il]) { llm_dsv4_metal_argbuf_free(g_argp[il]); g_argp[il] = NULL; }
    if (g_argd[il]) { llm_dsv4_metal_argbuf_free(g_argd[il]); g_argd[il] = NULL; }
    g_argp[il] = llm_dsv4_metal_argbuf_pair(g_be, gp, up, n_expert, gbytes, gbytes);
    g_argd[il] = llm_dsv4_metal_argbuf_down(g_be, dp, n_expert, dbytes);
    free(gp); free(up); free(dp);
    if (!g_argp[il] || !g_argd[il]) return 0;
    g_arg_ready[il] = 1; g_arg_ep[il] = ep;
    return 1;
}

int llm_dsv4_gpu_attn_tail_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, int il, const float* heads, const float* saved_attn,
        const float* post_attn, const float* comb_attn, int pos, float* out_hc) {
    if (!g_on || !g_be || !g_attn.inited || !c || !L || !pe) return 0;
    if (il < c->n_hash_layer) return 0;                          /* hash routing: path C */
    if (g_attn.outa[il].type != 8 || g_attn.outb[il].type != 8) return 0;
    if (g_moe.gate[il].type != 16 || g_moe.up[il].type != 16 || g_moe.down[il].type != 10) return 0; /* pair IQ2 + down Q2_K */
    int n_expert = c->n_expert;
    if (!dsv4_build_argbufs(pe, il, n_expert)) return 0;
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
    q.arg_pair = g_argp[il]; q.arg_down = g_argd[il];
    int rc = llm_dsv4_metal_layer_tail(g_be, &q) == 0;
    if (rc) {
        static int logged = 0;
        if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: tail table eseguito su GPU\n"); }
    }
    return rc;
}

int llm_dsv4_gpu_ctx_active(void) { return g_on; }

/* ── Middle attenzione su GPU: pesi compressor/indexer (f32) + KV/stato su device ── */
typedef struct {
    int inited, ratio, co, hd, ihd, width, iwidth, raw_cap, comp_cap;
    int n_raw, n_acomp, n_icomp;
    void *w_ckv, *w_cgate, *w_cape, *w_cnorm, *w_ickv, *w_icgate, *w_icape, *w_icnorm, *w_iqb, *w_iproj, *sinks;
    void *st_akv, *st_asc, *st_ikv, *st_isc, *raw, *acomp, *icomp, *allowed;
} Dsv4Mid;
static Dsv4Mid g_mid[LLM_DSV4_MAX_LAYER];

static void dsv4_mid_state_zero(void* kv, void* sc, int n) {
    if (!kv || !sc || n <= 0) return;
    float* z = (float*)calloc((size_t)n, sizeof(float));
    if (z) { llm_dsv4_metal_buf_write(kv, 0, z, (size_t)n * 4); free(z); }
    float* s = (float*)malloc(sizeof(float) * (size_t)n);
    if (s) { for (int i = 0; i < n; i++) s[i] = -1.0e30f; llm_dsv4_metal_buf_write(sc, 0, s, (size_t)n * 4); free(s); }
}
static void dsv4_mid_free(Dsv4Mid* m) {
    llm_dsv4_metal_buf_free(m->w_ckv); llm_dsv4_metal_buf_free(m->w_cgate); llm_dsv4_metal_buf_free(m->w_cape); llm_dsv4_metal_buf_free(m->w_cnorm);
    llm_dsv4_metal_buf_free(m->w_ickv); llm_dsv4_metal_buf_free(m->w_icgate); llm_dsv4_metal_buf_free(m->w_icape); llm_dsv4_metal_buf_free(m->w_icnorm);
    llm_dsv4_metal_buf_free(m->w_iqb); llm_dsv4_metal_buf_free(m->w_iproj); llm_dsv4_metal_buf_free(m->sinks);
    llm_dsv4_metal_buf_free(m->st_akv); llm_dsv4_metal_buf_free(m->st_asc); llm_dsv4_metal_buf_free(m->st_ikv); llm_dsv4_metal_buf_free(m->st_isc);
    llm_dsv4_metal_buf_free(m->raw); llm_dsv4_metal_buf_free(m->acomp); llm_dsv4_metal_buf_free(m->icomp); llm_dsv4_metal_buf_free(m->allowed);
    memset(m, 0, sizeof *m);
}
static int dsv4_mid_init(Dsv4Mid* m, const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int ratio, int raw_cap, int comp_cap) {
    if (!g_be || (ratio != 4 && ratio != 128) || !L->comp_kv || !L->comp_gate || !L->comp_ape || !L->comp_norm || !L->attn_sinks) return 0;
    memset(m, 0, sizeof *m);
    int co = (ratio == 4) ? 2 : 1, hd = c->n_head_dim, ihd = c->n_indexer_head_dim, embd = c->n_embd;
    int width = co * hd, iwidth = co * ihd;
    m->ratio = ratio; m->co = co; m->hd = hd; m->ihd = ihd; m->width = width; m->iwidth = iwidth;
    m->raw_cap = raw_cap; m->comp_cap = comp_cap;
    m->w_ckv  = llm_dsv4_metal_buf_alloc(g_be, (size_t)width * embd * 4);  llm_dsv4_metal_buf_write(m->w_ckv, 0, L->comp_kv,   (size_t)width * embd * 4);
    m->w_cgate= llm_dsv4_metal_buf_alloc(g_be, (size_t)width * embd * 4);  llm_dsv4_metal_buf_write(m->w_cgate, 0, L->comp_gate, (size_t)width * embd * 4);
    m->w_cape = llm_dsv4_metal_buf_alloc(g_be, (size_t)ratio * width * 4); llm_dsv4_metal_buf_write(m->w_cape, 0, L->comp_ape,  (size_t)ratio * width * 4);
    m->w_cnorm= llm_dsv4_metal_buf_alloc(g_be, (size_t)hd * 4);            llm_dsv4_metal_buf_write(m->w_cnorm, 0, L->comp_norm, (size_t)hd * 4);
    m->sinks  = llm_dsv4_metal_buf_alloc(g_be, (size_t)c->n_head * 4);     llm_dsv4_metal_buf_write(m->sinks, 0, L->attn_sinks,  (size_t)c->n_head * 4);
    if (ratio == 4) {
        if (!L->idx_c_kv || !L->idx_c_gate || !L->idx_c_ape || !L->idx_c_norm || !L->idx_q_b || !L->idx_proj) { dsv4_mid_free(m); return 0; }
        int qdim = c->n_indexer_head * c->n_indexer_head_dim;
        m->w_ickv  = llm_dsv4_metal_buf_alloc(g_be, (size_t)iwidth * embd * 4);  llm_dsv4_metal_buf_write(m->w_ickv, 0, L->idx_c_kv,   (size_t)iwidth * embd * 4);
        m->w_icgate= llm_dsv4_metal_buf_alloc(g_be, (size_t)iwidth * embd * 4);  llm_dsv4_metal_buf_write(m->w_icgate, 0, L->idx_c_gate, (size_t)iwidth * embd * 4);
        m->w_icape = llm_dsv4_metal_buf_alloc(g_be, (size_t)ratio * iwidth * 4); llm_dsv4_metal_buf_write(m->w_icape, 0, L->idx_c_ape,  (size_t)ratio * iwidth * 4);
        m->w_icnorm= llm_dsv4_metal_buf_alloc(g_be, (size_t)ihd * 4);            llm_dsv4_metal_buf_write(m->w_icnorm, 0, L->idx_c_norm, (size_t)ihd * 4);
        m->w_iqb   = llm_dsv4_metal_buf_alloc(g_be, (size_t)qdim * c->n_lora_q * 4); llm_dsv4_metal_buf_write(m->w_iqb, 0, L->idx_q_b, (size_t)qdim * c->n_lora_q * 4);
        m->w_iproj = llm_dsv4_metal_buf_alloc(g_be, (size_t)c->n_indexer_head * embd * 4); llm_dsv4_metal_buf_write(m->w_iproj, 0, L->idx_proj, (size_t)c->n_indexer_head * embd * 4);
    }
    m->st_akv = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * width * 4);
    m->st_asc = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * width * 4);
    if (ratio == 4) { m->st_ikv = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * iwidth * 4); m->st_isc = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * iwidth * 4); }
    m->raw   = llm_dsv4_metal_buf_alloc(g_be, (size_t)raw_cap * hd * 4);
    m->acomp = llm_dsv4_metal_buf_alloc(g_be, (size_t)comp_cap * hd * 4);
    m->icomp = llm_dsv4_metal_buf_alloc(g_be, (size_t)comp_cap * ihd * 4);
    m->allowed = llm_dsv4_metal_buf_alloc(g_be, (size_t)comp_cap * sizeof(int));
    if (!m->w_ckv || !m->w_cgate || !m->w_cape || !m->w_cnorm || !m->sinks || !m->st_akv || !m->st_asc || !m->raw || !m->acomp || !m->icomp || !m->allowed) { dsv4_mid_free(m); return 0; }
    dsv4_mid_state_zero(m->st_akv, m->st_asc, co * ratio * width);
    if (ratio == 4) dsv4_mid_state_zero(m->st_ikv, m->st_isc, co * ratio * iwidth);
    m->inited = 1;
    return 1;
}

int llm_dsv4_gpu_attn_mid_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap,
        const float* q, const float* kv, const float* qrn, const float* xn, int pos, float* heads) {
    if (!g_on || !g_be || !c || !L || il < 0 || il >= LLM_DSV4_MAX_LAYER) return 0;
    if (ratio != 4 && ratio != 128) return 0;
    if (il < c->n_hash_layer) return 0;
    Dsv4Mid* m = &g_mid[il];
    if (!m->inited && !dsv4_mid_init(m, c, L, ratio, raw_cap, comp_cap)) return 0;
    const int hd = c->n_head_dim, ihd = c->n_indexer_head_dim;
    int due = ((pos + 1) % ratio) == 0, pos_mod = pos % ratio;
    int state_row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;
    int raw_full = (m->n_raw >= raw_cap);
    int raw_off = raw_full ? (raw_cap - 1) * hd : m->n_raw * hd;
    int na = (due && m->n_acomp < comp_cap) ? m->n_acomp + 1 : m->n_acomp;
    int ni = (due && m->n_icomp < comp_cap) ? m->n_icomp + 1 : m->n_icomp;
    int nr = raw_full ? raw_cap : m->n_raw + 1;
    int compressed = ratio != 0;
    float freq_base = (compressed && c->compress_rope_freq_base > 0.0f) ? c->compress_rope_freq_base : c->rope_freq_base;
    float freq_scale = (!compressed || c->rope_scale_factor <= 0.0f) ? 1.0f : 1.0f / c->rope_scale_factor;
    float ext_factor = (compressed && c->rope_scale_factor > 1.0f) ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    LlmDsv4MidParams p; memset(&p, 0, sizeof p);
    p.embd = c->n_embd; p.hd = hd; p.ihd = ihd; p.ratio = ratio; p.co = m->co;
    p.n_head = c->n_head; p.n_rot = c->n_rot; p.rms_eps = c->rms_eps;
    p.idx_nh = c->n_indexer_head; p.idx_hd = ihd; p.idx_top_k = c->n_indexer_top_k; p.lq = c->n_lora_q;
    p.qdim = c->n_indexer_head * c->n_indexer_head_dim;
    p.pos = pos; p.due = due; p.state_row = state_row; p.ape_off = pos_mod; p.has_idx = (ratio == 4);
    p.push_acomp = (due && m->n_acomp < comp_cap); p.push_icomp = (due && m->n_icomp < comp_cap);
    p.rope_cfg[0] = freq_base; p.rope_cfg[1] = freq_scale; p.rope_cfg[2] = ext_factor; p.rope_cfg[3] = attn_factor;
    p.rope_cfg[4] = c->rope_yarn_beta_fast; p.rope_cfg[5] = c->rope_yarn_beta_slow; p.rope_cfg[6] = (float)(compressed ? c->rope_orig_ctx : 0);
    p.w_ckv = m->w_ckv; p.w_cgate = m->w_cgate; p.w_cape = m->w_cape; p.w_cnorm = m->w_cnorm;
    p.w_ickv = m->w_ickv; p.w_icgate = m->w_icgate; p.w_icape = m->w_icape; p.w_icnorm = m->w_icnorm;
    p.w_iqb = m->w_iqb; p.w_iproj = m->w_iproj; p.sinks = m->sinks;
    p.st_akv = m->st_akv; p.st_asc = m->st_asc; p.st_ikv = m->st_ikv; p.st_isc = m->st_isc;
    p.raw = m->raw; p.raw_cap = raw_cap; p.raw_off = raw_off; p.raw_full = raw_full;
    p.acomp = m->acomp; p.acomp_off = m->n_acomp * hd; p.icomp = m->icomp; p.icomp_off = m->n_icomp * ihd;
    p.allowed = m->allowed;
    p.xn = xn; p.qrn = qrn; p.q = q; p.kv_cur = kv;
    p.n_raw = nr; p.n_acomp = na; p.n_icomp = ni;
    p.heads = heads;
    if (llm_dsv4_metal_attn_mid(g_be, &p) != 0) return 0;
    m->n_raw = nr; m->n_acomp = na; m->n_icomp = ni;
    { static int logged = 0; if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: middle attenzione eseguito su GPU\n"); } }
    return 1;
}

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
    for (int il = 0; il < LLM_DSV4_MAX_LAYER; il++) {
        llm_dsv4_metal_argbuf_free(g_argp[il]); g_argp[il] = NULL;
        llm_dsv4_metal_argbuf_free(g_argd[il]); g_argd[il] = NULL;
        g_arg_ready[il] = 0;
        dsv4_mid_free(&g_mid[il]);
    }
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
