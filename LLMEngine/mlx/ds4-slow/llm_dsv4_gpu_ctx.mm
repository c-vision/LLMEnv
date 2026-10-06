/* llm_dsv4_gpu_ctx.mm — contesto GPU DS4 (F4), singleton, ABI C.
 * Crea il backend Metal e le view (moe/attn) una sola volta; espone gli hook usati dal layer,
 * dall'attenzione e dall'engine. GPU attiva di default (fallback al path C se assente). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_gpu_ctx.h"
#include "llm_dsv4_hcroute.h"
#include "llm_engine_opts.h"
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
void llm_dsv4_gpu_set_route(const int* sel, const float* rw, int used) { dsv4_pre_route_set(sel, rw, used); }

/* ── Middle attenzione su GPU: pesi compressor/indexer (f32) + KV/stato su device ── */
typedef struct {
    int inited, ratio, co, hd, ihd, width, iwidth, raw_cap, comp_cap;
    int n_raw, n_acomp, n_icomp;
    void *w_ckv, *w_cgate, *w_cape, *w_cnorm, *w_ickv, *w_icgate, *w_icape, *w_icnorm, *w_iqb, *w_iproj, *sinks;
    void *st_akv, *st_asc, *st_ikv, *st_isc, *raw, *acomp, *icomp, *allowed;
    void *acomp_h;   /* KV compresso PACKED half (solo path chunk): l'attenzione legge half -> banda/2 */
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
    llm_dsv4_metal_buf_free(m->raw); llm_dsv4_metal_buf_free(m->acomp); llm_dsv4_metal_buf_free(m->acomp_h); llm_dsv4_metal_buf_free(m->icomp); llm_dsv4_metal_buf_free(m->allowed);
    memset(m, 0, sizeof *m);
}
/* Attenzione solo-raw (layer 0-1, ratio 0) su GPU: opzione di registro dsv4_gpu_raw_attn (default 1; 0 = percorso CPU). */
static int dsv4_raw_attn_gpu(void) { int v = 1; if (llm_ai_mlx_opt_get_bool("dsv4_gpu_raw_attn", &v) != 0) v = 1; return v; }
static int dsv4_mid_init(Dsv4Mid* m, const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int ratio, int raw_cap, int comp_cap) {
    if (!g_be || (ratio != 0 && ratio != 4 && ratio != 128) || !L->attn_sinks) return 0;
    if (ratio != 0 && (!L->comp_kv || !L->comp_gate || !L->comp_ape || !L->comp_norm)) return 0;
    memset(m, 0, sizeof *m);
    int co = (ratio == 4) ? 2 : 1, hd = c->n_head_dim, ihd = c->n_indexer_head_dim, embd = c->n_embd;
    int width = co * hd, iwidth = co * ihd;
    m->ratio = ratio; m->co = co; m->hd = hd; m->ihd = ihd; m->width = width; m->iwidth = iwidth;
    m->raw_cap = raw_cap; m->comp_cap = comp_cap;
    if (ratio > 0) {
    m->w_ckv  = llm_dsv4_metal_buf_alloc(g_be, (size_t)width * embd * 4);  llm_dsv4_metal_buf_write(m->w_ckv, 0, L->comp_kv,   (size_t)width * embd * 4);
    m->w_cgate= llm_dsv4_metal_buf_alloc(g_be, (size_t)width * embd * 4);  llm_dsv4_metal_buf_write(m->w_cgate, 0, L->comp_gate, (size_t)width * embd * 4);
    m->w_cape = llm_dsv4_metal_buf_alloc(g_be, (size_t)ratio * width * 4); llm_dsv4_metal_buf_write(m->w_cape, 0, L->comp_ape,  (size_t)ratio * width * 4);
    m->w_cnorm= llm_dsv4_metal_buf_alloc(g_be, (size_t)hd * 4);            llm_dsv4_metal_buf_write(m->w_cnorm, 0, L->comp_norm, (size_t)hd * 4);
    }
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
    if (ratio > 0) {
    m->st_akv = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * width * 4);
    m->st_asc = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * width * 4);
    }
    if (ratio == 4) { m->st_ikv = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * iwidth * 4); m->st_isc = llm_dsv4_metal_buf_alloc(g_be, (size_t)co * ratio * iwidth * 4); }
    m->raw   = llm_dsv4_metal_buf_alloc(g_be, (size_t)raw_cap * hd * 4);
    { const size_t cc = comp_cap > 0 ? (size_t)comp_cap : 4;   /* ratio 0: buffer minimi (mai letti) */
    m->acomp = llm_dsv4_metal_buf_alloc(g_be, cc * hd * 4);
    m->acomp_h = llm_dsv4_metal_buf_alloc(g_be, cc * hd * 2);
    m->icomp = llm_dsv4_metal_buf_alloc(g_be, cc * ihd * 4);
    m->allowed = llm_dsv4_metal_buf_alloc(g_be, cc * sizeof(int)); }
    if (!m->sinks || !m->raw || !m->acomp || !m->acomp_h || !m->icomp || !m->allowed || (ratio > 0 && (!m->w_ckv || !m->w_cgate || !m->w_cape || !m->w_cnorm || !m->st_akv || !m->st_asc))) { dsv4_mid_free(m); return 0; }
    if (ratio > 0) dsv4_mid_state_zero(m->st_akv, m->st_asc, co * ratio * width);
    if (ratio == 4) dsv4_mid_state_zero(m->st_ikv, m->st_isc, co * ratio * iwidth);
    m->inited = 1;
    return 1;
}

/* Costruisce i parametri del middle per il token a `pos` dato lo stato corrente (n_raw/n_acomp/n_icomp) e
 * restituisce il nuovo stato in *nr/*na/*ni (NON aggiorna m). */
static void dsv4_mid_fill(const Dsv4Mid* m, const LlmDsv4Config* c, int ratio, int raw_cap, int comp_cap,
        int n_raw0, int n_acomp0, int n_icomp0, const float* q, const float* kv, const float* qrn, const float* xn,
        int pos, float* heads, LlmDsv4MidParams* p, int* nr_out, int* na_out, int* ni_out) {
    const int hd = c->n_head_dim, ihd = c->n_indexer_head_dim;
    int due = ratio > 0 && ((pos + 1) % ratio) == 0, pos_mod = ratio > 0 ? pos % ratio : 0;
    int state_row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;
    int raw_full = (n_raw0 >= raw_cap);
    int raw_off = raw_full ? (raw_cap - 1) * hd : n_raw0 * hd;
    int na = (due && n_acomp0 < comp_cap) ? n_acomp0 + 1 : n_acomp0;
    int ni = (due && n_icomp0 < comp_cap) ? n_icomp0 + 1 : n_icomp0;
    int nr = raw_full ? raw_cap : n_raw0 + 1;
    int compressed = ratio != 0;
    float freq_base = (compressed && c->compress_rope_freq_base > 0.0f) ? c->compress_rope_freq_base : c->rope_freq_base;
    float freq_scale = (!compressed || c->rope_scale_factor <= 0.0f) ? 1.0f : 1.0f / c->rope_scale_factor;
    float ext_factor = (compressed && c->rope_scale_factor > 1.0f) ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    LlmDsv4MidParams* pp = p; memset(pp, 0, sizeof *pp);
    pp->embd = c->n_embd; pp->hd = hd; pp->ihd = ihd; pp->ratio = ratio; pp->co = m->co;
    pp->n_head = c->n_head; pp->n_rot = c->n_rot; pp->rms_eps = c->rms_eps;
    pp->idx_nh = c->n_indexer_head; pp->idx_hd = ihd; pp->idx_top_k = c->n_indexer_top_k; pp->lq = c->n_lora_q;
    pp->qdim = c->n_indexer_head * c->n_indexer_head_dim;
    pp->pos = pos; pp->due = due; pp->state_row = state_row; pp->ape_off = pos_mod; pp->has_idx = (ratio == 4);
    pp->push_acomp = (due && n_acomp0 < comp_cap); pp->push_icomp = (due && n_icomp0 < comp_cap);
    pp->rope_cfg[0] = freq_base; pp->rope_cfg[1] = freq_scale; pp->rope_cfg[2] = ext_factor; pp->rope_cfg[3] = attn_factor;
    pp->rope_cfg[4] = c->rope_yarn_beta_fast; pp->rope_cfg[5] = c->rope_yarn_beta_slow; pp->rope_cfg[6] = (float)(compressed ? c->rope_orig_ctx : 0);
    pp->w_ckv = m->w_ckv; pp->w_cgate = m->w_cgate; pp->w_cape = m->w_cape; pp->w_cnorm = m->w_cnorm;
    pp->w_ickv = m->w_ickv; pp->w_icgate = m->w_icgate; pp->w_icape = m->w_icape; pp->w_icnorm = m->w_icnorm;
    pp->w_iqb = m->w_iqb; pp->w_iproj = m->w_iproj; pp->sinks = m->sinks;
    pp->st_akv = m->st_akv; pp->st_asc = m->st_asc; pp->st_ikv = m->st_ikv; pp->st_isc = m->st_isc;
    pp->raw = m->raw; pp->raw_cap = raw_cap; pp->raw_off = raw_off; pp->raw_full = raw_full;
    pp->acomp = m->acomp; pp->acomp_off = n_acomp0 * hd; pp->acomp_h = m->acomp_h; pp->icomp = m->icomp; pp->icomp_off = n_icomp0 * ihd;
    pp->allowed = m->allowed;
    pp->xn = xn; pp->qrn = qrn; pp->q = q; pp->kv_cur = kv;
    pp->n_raw = nr; pp->n_acomp = na; pp->n_icomp = ni;
    pp->heads = heads;
    *nr_out = nr; *na_out = na; *ni_out = ni;
}

int llm_dsv4_gpu_attn_mid_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap,
        const float* q, const float* kv, const float* qrn, const float* xn, int pos, float* heads) {
    if (!g_on || !g_be || !c || !L || il < 0 || il >= LLM_DSV4_MAX_LAYER) return 0;
    if (ratio != 0 && ratio != 4 && ratio != 128) return 0;   /* ratio 0 (layer 0-1): solo finestra raw; i layer hash (routing) hanno attenzione normale */
    if (ratio == 0 && !dsv4_raw_attn_gpu()) return 0;
    Dsv4Mid* m = &g_mid[il];
    if (!m->inited && !dsv4_mid_init(m, c, L, ratio, raw_cap, comp_cap)) return 0;
    LlmDsv4MidParams p; int nr, na, ni;
    dsv4_mid_fill(m, c, ratio, raw_cap, comp_cap, m->n_raw, m->n_acomp, m->n_icomp, q, kv, qrn, xn, pos, heads, &p, &nr, &na, &ni);
    llm_dsv4_metal_set_op(4);
    { int mr = llm_dsv4_metal_attn_mid(g_be, &p); llm_dsv4_metal_set_op(0); if (mr != 0) return 0; }
    m->n_raw = nr; m->n_acomp = na; m->n_icomp = ni;
    { static int logged = 0; if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: middle attenzione eseguito su GPU\n"); } }
    return 1;
}

/* Attenzione di un CHUNK di `nt` token consecutivi (pos0..) del layer `il`: proiezioni, middle e output
 * grouped-LoRA, ciascuna fase in UNA command buffer per tutto il chunk. xn [nt*embd] -> out [nt*embd].
 * Tutto-o-niente: 1 = eseguito su GPU; 0 = nulla e' stato modificato (usare il path per-token). */
/* Decode device-resident. Ritorna flag: bit0 = attn eseguita; bit1 = anche mHC post/pre + route eseguiti (se `hr`). */
static int dsv4_attn_dev_run(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap, const float* xn, int pos0, int nt, float* out, const LlmDsv4HcRouteParams* hr) {
    if (!g_on || !g_be || !c || !L || il < 0 || il >= LLM_DSV4_MAX_LAYER || nt <= 0) return 0;
    if (nt != 1) return 0;   /* device-resident solo per il decode (nt=1); il prefill batch resta per-token (come prima: il project con out=NULL falliva sempre) */
    if (ratio != 0 && ratio != 4 && ratio != 128) return 0;
    if (ratio == 0 && !dsv4_raw_attn_gpu()) return 0;
    if (!llm_dsv4_gpu_attn_batch_ready(&g_attn, il)) return 0;
    Dsv4Mid* m = &g_mid[il];
    if (!m->inited && !dsv4_mid_init(m, c, L, ratio, raw_cap, comp_cap)) return 0;
    const int hd = c->n_head_dim, qd = c->n_head * hd;
    /* ITEM B device-resident (branch non-bit-exact): il project scrive q/kv/qrn negli slot device sx[6/8/9/10]
     * (out=NULL -> no download); il mid li legge (dev_in=1) e lascia heads in device (dev_heads=1); l'out-batch
     * legge heads da device e applica la RoPE inversa su GPU (dsv4_rope_rows_inv). */
    /* Scrath PERSISTENTE (era malloc/free per-chiamata: 128 KB × layer/token = overhead puro §0septies). */
    static float* hds = NULL; static size_t hds_cap = 0;
    static LlmDsv4MidParams* ps = NULL; static size_t ps_cap = 0;
    if ((size_t)nt * (size_t)qd > hds_cap) { free(hds); hds = (float*)malloc(sizeof(float) * (size_t)nt * (size_t)qd); hds_cap = (size_t)nt * (size_t)qd; }
    if ((size_t)nt > ps_cap) { free(ps); ps = (LlmDsv4MidParams*)malloc(sizeof(LlmDsv4MidParams) * (size_t)nt); ps_cap = (size_t)nt; }
    void* heads_dev = NULL;
    int ok = hds && ps;
    const int use_shared = (nt == 1);   /* decode: project+mid in UNA cb (un solo wait) */
    if (use_shared && ok) llm_dsv4_metal_shared_begin(g_be);
    if (ok) ok = llm_dsv4_gpu_attn_project_batch(&g_attn, c, L, il, xn, pos0, nt, NULL, NULL, NULL);
    int nr = m->n_raw, na = m->n_acomp, ni = m->n_icomp;
    if (ok) {
        for (int t = 0; t < nt; t++) {
            int nr2, na2, ni2;
            dsv4_mid_fill(m, c, ratio, raw_cap, comp_cap, nr, na, ni, NULL, NULL, NULL, NULL, pos0 + t,
                          hds + (size_t)t * qd, &ps[t], &nr2, &na2, &ni2);
            ps[t].dev_in = 1;
            ps[t].dev_heads = 1;
            nr = nr2; na = na2; ni = ni2;
        }
        ok = llm_dsv4_metal_attn_mid_batch(g_be, ps, nt, &heads_dev) == 0;
        if (!ok) heads_dev = NULL;
        if (ok) { m->n_raw = nr; m->n_acomp = na; m->n_icomp = ni; }
    }
    /* nt==1: anche il finish (RoPE inversa + out_a/out_b) nella STESSA cb condivisa -> UN solo wait per l'attn del layer
     * (project+mid+finish); il download di `out` avviene in shared_end. */
    if (ok) ok = llm_dsv4_gpu_attn_finish_batch(&g_attn, c, L, il, hds, pos0, nt, out, heads_dev);
    int flags = ok ? 1 : 0;
    /* stessa cb: mHC post(attn) + mHC pre(ffn) + route MoE -> dopo il wait unico l'host salta CPU post/pre/route */
    if (ok && hr && use_shared) { if (llm_dsv4_metal_attn_hc_route(g_be, hr) == 0) flags |= 2; }
    if (use_shared) llm_dsv4_metal_shared_end(g_be);
    return flags;
}

extern "C" int llm_dsv4_gpu_attn_hcroute_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap, const float* xn, int pos, float* out, const LlmDsv4HcRouteParams* hr);
int llm_dsv4_gpu_attn_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap, const float* xn, int pos0, int nt, float* out) {
    return dsv4_attn_dev_run(c, L, il, ratio, raw_cap, comp_cap, xn, pos0, nt, out, NULL) & 1;
}
int llm_dsv4_gpu_attn_hcroute_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap, const float* xn, int pos, float* out, const LlmDsv4HcRouteParams* hr) {
    return dsv4_attn_dev_run(c, L, il, ratio, raw_cap, comp_cap, xn, pos, 1, out, hr);
}

/* Prewarm dei pesi statici (attn Q8_0 + shared expert): una sola commit del residency set invece di una per cb nel primo chunk. */
static void dsv4_prewarm_static(const LlmDsv4Config* c, const LlmDsv4PackedExperts* exps) {
    if (!g_be || !exps) return;
    const int lq = c->n_lora_q, embd = c->n_embd, qd = c->n_head * c->n_head_dim, hd = c->n_head_dim, ng = c->n_out_group, rank = c->n_lora_o, ff = c->n_ff_exp;
    for (int il = 0; il < c->n_layer && il < LLM_DSV4_MAX_LAYER; il++) {
        const LlmDsv4PackedExperts* pe = &exps[il];
        if (g_attn.inited) {
            llm_dsv4_metal_prewarm_view(g_be, g_attn.qa[il].w, (size_t)lq * g_attn.qa[il].row_bytes);
            llm_dsv4_metal_prewarm_view(g_be, g_attn.kv[il].w, (size_t)hd * g_attn.kv[il].row_bytes);
            llm_dsv4_metal_prewarm_view(g_be, g_attn.qb[il].w, (size_t)qd * g_attn.qb[il].row_bytes);
            llm_dsv4_metal_prewarm_view(g_be, g_attn.outa[il].w, (size_t)ng * rank * g_attn.outa[il].row_bytes);
            llm_dsv4_metal_prewarm_view(g_be, g_attn.outb[il].w, (size_t)embd * g_attn.outb[il].row_bytes);
        }
        if (pe->shexp_gate && pe->shexp_up && pe->shexp_down) {
            size_t sg = (size_t)llm_ggml_nbytes(pe->shexp_gate_type, embd), sd = (size_t)llm_ggml_nbytes(pe->shexp_down_type, ff);
            llm_dsv4_metal_prewarm_view(g_be, pe->shexp_gate, (size_t)ff * sg);
            llm_dsv4_metal_prewarm_view(g_be, pe->shexp_up, (size_t)ff * sg);
            llm_dsv4_metal_prewarm_view(g_be, pe->shexp_down, (size_t)embd * sd);
        }
    }
    llm_dsv4_metal_prewarm_commit(g_be);
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
    dsv4_prewarm_static(c, exps);
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
    dsv4_prewarm_static(c, exps);
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

int g_dsv4_prof_global = -1;   /* TEMP: flag di profilazione condiviso (v. gpu_ctx.h) */

int llm_dsv4_gpu_moe_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                         const float* x, int token, float* out) {
    static int logged = 0;
    int r;
    if (!g_on) return 0;
    llm_dsv4_metal_set_op(2);
    r = llm_dsv4_gpu_moe(&g_moe, c, L, il, x, token, out) ? 1 : 0;
    llm_dsv4_metal_set_op(0);
    if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: MoE %s\n",
                                       r ? "eseguito su GPU" : "FALLITO -> fallback CPU"); }
    return r;
}

int llm_dsv4_gpu_route_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il, const float* x, const int* tokens, int nt, int* sel, float* wt) {
    if (!g_on || !g_be || !L->ffn_gate_inp) return 0;
    const int used = c->n_expert_used; int* hs = NULL;
    if (il < c->n_hash_layer && L->ffn_gate_tid2eid) {
        hs = (int*)malloc(sizeof(int) * (size_t)nt * used); if (!hs) return 0;
        for (int t = 0; t < nt; t++) for (int j = 0; j < used; j++) hs[(size_t)t * used + j] = (int)L->ffn_gate_tid2eid[(long)tokens[t] * used + j];
    }
    int r = llm_dsv4_metal_route_batch(g_be, nt, c->n_expert, c->n_embd, used, L->ffn_gate_inp, L->exp_probs_b, hs, c->expert_weight_scale, x, sel, wt) == 0;
    free(hs);
    return r;
}
int llm_dsv4_gpu_hc_pre_batch_try(const LlmDsv4Config* c, const float* fn, const float* scale, const float* base, const float* norm_w,
                                  const float* res, int nt, float* normed, float* sp) {
    if (!g_on || !g_be) return 0;
    return llm_dsv4_metal_hc_pre_batch(g_be, nt, c->n_hc, c->n_embd, c->n_hc_sinkhorn_iter, c->hc_eps, c->rms_eps, res, fn, scale, base, norm_w, normed, sp) == 0;
}
int llm_dsv4_gpu_hc_post_batch_try(const LlmDsv4Config* c, const float* bo, const float* res, const float* sp, int nt, float* out) {
    if (!g_on || !g_be) return 0;
    return llm_dsv4_metal_hc_post_batch(g_be, nt, c->n_hc, c->n_embd, bo, res, sp, out) == 0;
}
int llm_dsv4_gpu_moe_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                               const float* x, const int* tokens, int nt, float* out) {
    if (!g_on) return 0;
    llm_dsv4_metal_set_op(2);
    { int r = llm_dsv4_gpu_moe_batch(&g_moe, c, L, il, x, tokens, nt, out) ? 1 : 0; llm_dsv4_metal_set_op(0); return r; }
}

/* TEMP (gated ds4_profile): espone gli accumulatori I/O vs GEMM del MoE batched (definiti in
 * llm_dsv4_gpu_moe.h, TU di questo .mm). Chiamata dal dump del profiler in llm_dsv4_layer.h. */
void llm_dsv4_gpu_moe_prof(double* io_ms, double* mm_ms) {
    if (io_ms) *io_ms = g_dsv4_moe_io_ms;
    if (mm_ms) *mm_ms = g_dsv4_moe_mm_ms;
}
void llm_dsv4_gpu_moe_pt_prof(double* route_ms, double* store_ms, double* metal_ms) {
    if (route_ms) *route_ms = g_dsv4_pt_route_ms;
    if (store_ms) *store_ms = g_dsv4_pt_store_ms;
    if (metal_ms) *metal_ms = g_dsv4_pt_metal_ms;
}

int llm_dsv4_gpu_attn_project_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                  const float* xn, int pos, float* q, float* kv, float* qrn) {
    static int logged = 0;
    int r;
    if (!g_on) return 0;
    llm_dsv4_metal_set_op(1);
    r = llm_dsv4_gpu_attn_project(&g_attn, c, L, il, xn, pos, q, kv, qrn) ? 1 : 0;
    llm_dsv4_metal_set_op(0);
    if (!logged) { logged = 1; fprintf(stderr, "DSV4 GPU: attn-proj %s\n",
                                       r ? "eseguita su GPU" : "FALLITA -> fallback CPU"); }
    return r;
}

int llm_dsv4_gpu_attn_finish_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
                                 float* heads, int pos, float* out) {
    if (!g_on) return 0;
    llm_dsv4_metal_set_op(3);   /* op 3 = attn FINISH/out (separato dai project+mid, op 1) */
    { int r = llm_dsv4_gpu_attn_finish(&g_attn, c, L, il, heads, pos, out) ? 1 : 0; llm_dsv4_metal_set_op(0); return r; }
}
