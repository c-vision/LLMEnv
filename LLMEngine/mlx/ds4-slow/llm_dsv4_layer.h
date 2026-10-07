/* llm_dsv4_layer.h — layer loop DS4: mHC pre -> attenzione -> mHC post -> mHC pre ->
 * MoE -> mHC post, ripetuto sugli n_layer. Port fedele del riferimento CPU ds4.c.
 * Header-only. Test: llm_dsv4_layer_test.c. */
#ifndef LLM_DSV4_LAYER_H
#define LLM_DSV4_LAYER_H

#include <stdlib.h>
#include <stdio.h>
#include "llm_dsv4_block.h"
#include "llm_dsv4_moe_fwd.h"
#include "llm_dsv4_cache.h"
#include "llm_engine_opts.h"   /* opzione dsv4_gpu_hc (registro, no env) */
#include <time.h>

/* ── PROFILER per-fase (TEMPORANEO; gated da opzione registro `dsv4_profile`, default 0). Serve
 *    a diagnosticare il prefill: DOVE va il tempo (fase1=mHC+attn, MoE, fase3=mHC) e QUALE ramo
 *    gira (GPU/batched vs CPU). I contatori sono deterministici (validi anche a macchina carica);
 *    i tempi sono indicativi e vanno letti solo a macchina quieta. ── */
static long g_pf_hc_gpu = 0, g_pf_hc_cpu = 0, g_pf_attn_gpu = 0, g_pf_attn_cpu = 0;
static long g_pf_moe_batch = 0, g_pf_moe_cpu = 0, g_pf_layers = 0;
static double g_pf_t1 = 0, g_pf_tmoe = 0, g_pf_t3 = 0;
static double g_pf_d_attn = 0, g_pf_d_moe = 0;   /* DECODE: attn vs MoE per-token */
static int g_pf_on = -1;
static inline int llm_dsv4_pf_on(void) {
    if (g_pf_on < 0) { int v = 0; g_pf_on = (llm_ai_mlx_opt_get_bool("dsv4_profile", &v) == 0) ? v : 0; g_dsv4_prof_global = g_pf_on; }
    return g_pf_on;
}
static inline double llm_dsv4_pf_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
static inline void llm_dsv4_pf_dump(const char* tag) {
    if (!llm_dsv4_pf_on() || g_pf_layers == 0) return;
    double _io = 0, _mm = 0; llm_dsv4_gpu_moe_prof(&_io, &_mm);
    fprintf(stderr, "[dsv4-prof %s] layers=%ld | fase1(mHC+attn)=%.1fms MoE=%.1fms (io=%.1f gemm=%.1f) "
            "fase3(mHC)=%.1fms | hc gpu/cpu=%ld/%ld attn gpu/cpu=%ld/%ld moe batch/pertok=%ld/%ld\n",
            tag, g_pf_layers, g_pf_t1, g_pf_tmoe, _io, _mm, g_pf_t3,
            g_pf_hc_gpu, g_pf_hc_cpu, g_pf_attn_gpu, g_pf_attn_cpu, g_pf_moe_batch, g_pf_moe_cpu);
    /* TEMP: split GPU dei 5 sub-kernel mHC (solo a profilo ON, `hc_pre_batch` emette 4 cb). */
    fprintf(stderr, "[dsv4-prof mhc] scale=%.1f mixes=%.1f sink=%.1f norm=%.1f post=%.1f ms (cb 5..9=%ld/%ld/%ld/%ld/%ld)\n",
            llm_dsv4_metal_gpu_ms_op(5), llm_dsv4_metal_gpu_ms_op(6), llm_dsv4_metal_gpu_ms_op(7),
            llm_dsv4_metal_gpu_ms_op(8), llm_dsv4_metal_gpu_ms_op(9),
            llm_dsv4_metal_command_buffers_op(5), llm_dsv4_metal_command_buffers_op(6),
            llm_dsv4_metal_command_buffers_op(7), llm_dsv4_metal_command_buffers_op(8),
            llm_dsv4_metal_command_buffers_op(9));
}
typedef struct {
    double gpu_total, gpu_attn, gpu_moe, route, store, metal;
    long command_buffers, dispatch, rescommit, resadd;
    long cb_attn, cb_moe, cb_other;
    double gpu_finish; long cb_finish;
    double gpu_mid; long cb_mid;
    long waits;
} LlmDsv4DecodeProfileSnapshot;

static inline LlmDsv4DecodeProfileSnapshot llm_dsv4_pf_decode_snapshot(void) {
    LlmDsv4DecodeProfileSnapshot s = {0};
    s.gpu_total = llm_dsv4_metal_gpu_total_ms();
    llm_dsv4_metal_gpu_ms(&s.gpu_attn, &s.gpu_moe);
    llm_dsv4_gpu_moe_pt_prof(&s.route, &s.store, &s.metal);
    s.command_buffers = llm_dsv4_metal_command_buffers();
    s.cb_attn = llm_dsv4_metal_command_buffers_op(1);
    s.cb_moe = llm_dsv4_metal_command_buffers_op(2);
    s.cb_other = llm_dsv4_metal_command_buffers_op(0);
    s.gpu_finish = llm_dsv4_metal_gpu_ms_op(3);
    s.cb_finish = llm_dsv4_metal_command_buffers_op(3);
    s.gpu_mid = llm_dsv4_metal_gpu_ms_op(4);       /* op4 = attn MID */
    s.cb_mid = llm_dsv4_metal_command_buffers_op(4);
    s.dispatch = llm_dsv4_metal_dispatches();
    s.waits = llm_dsv4_metal_waits();
    s.rescommit = llm_dsv4_metal_res_commits();
    s.resadd = llm_dsv4_metal_res_adds();
    return s;
}

static inline void llm_dsv4_pf_decode_dump(int ntok, LlmDsv4DecodeProfileSnapshot before, double wall_ms) {
    if (!llm_dsv4_pf_on() || ntok <= 0) return;
    LlmDsv4DecodeProfileSnapshot after = llm_dsv4_pf_decode_snapshot();
    /* wall-GPU e' solo un residuo: include wait/idle e non equivale a tempo CPU puro. */
    fprintf(stderr, "[dsv4-decode] wall total=%.1f attn=%.1f moe=%.1f | GPU counted total=%.1f attn=%.1f moe=%.1f | wall-GPU=%.1f | MoEhost route=%.1f store=%.1f metal=%.1f | cb=%ld dispatch=%ld rescommit=%ld resadd=%ld (n=%d tok, %.1f ms/tok)\n",
            wall_ms,
            g_pf_d_attn, g_pf_d_moe,
            after.gpu_total - before.gpu_total,
            after.gpu_attn - before.gpu_attn, after.gpu_moe - before.gpu_moe,
            wall_ms - (after.gpu_total - before.gpu_total),
            after.route - before.route, after.store - before.store, after.metal - before.metal,
            after.command_buffers - before.command_buffers,
            after.dispatch - before.dispatch, after.rescommit - before.rescommit,
            after.resadd - before.resadd, ntok,
            (g_pf_d_attn + g_pf_d_moe) / (double)ntok);
    fprintf(stderr, "[dsv4-decode-wait] waitUntilCompleted/token=%.1f (tot=%ld cb=%ld n=%d tok)\n",
            ntok ? (double)(after.waits - before.waits) / ntok : 0, after.waits - before.waits, after.command_buffers - before.command_buffers, ntok);
    fprintf(stderr, "[dsv4-decode-cb] GPU cl (delta): project(op1)=%.1f | mid(op4)=%.1f | finish(op3)=%.1f | moe(op2)=%.1f  (n=%d tok)\n",
            (after.gpu_attn - before.gpu_attn), (after.gpu_mid - before.gpu_mid), (after.gpu_finish - before.gpu_finish), (after.gpu_moe - before.gpu_moe), ntok);
    fprintf(stderr, "[dsv4-decode-cb] cb (delta): project=%ld mid=%ld finish=%ld moe=%ld altro=%ld (tot=%.1f/tok)\n",
            after.cb_attn - before.cb_attn, after.cb_mid - before.cb_mid, after.cb_finish - before.cb_finish, after.cb_moe - before.cb_moe, after.cb_other - before.cb_other,
            ntok ? (double)(after.command_buffers - before.command_buffers) / ntok : 0);
    g_pf_d_attn = 0; g_pf_d_moe = 0;   /* per-token: azzera per il prossimo dump */
}

/* Un layer su un token. hc [n_hc*embd] in ingresso, out_hc [n_hc*embd] in uscita.
 * kv = righe KV (raw+compresse) se presenti, altrimenti NULL (solo kv corrente). */
static inline void llm_dsv4_layer_forward(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* hc, int token, int pos, int il, const float* kv, int n_kv, float* out_hc) {
    int HC = c->n_hc, D = c->n_embd;
    llm_dsv4_arena_reset();
    float* attn_x = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* attn_out = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* after = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC * D);
    float* ffn_x = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* ffn_out = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* post = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC);
    float* comb = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC * HC);

    /* attenzione */
    llm_dsv4_block_pre(c, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, L->attn_norm,
                       hc, attn_x, post, comb);
    llm_dsv4_attn_from_normed(c, L, attn_x, pos, il, kv, n_kv, attn_out);
    llm_dsv4_block_post(c, attn_out, hc, post, comb, after);

    /* MoE */
    llm_dsv4_block_pre(c, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, L->ffn_norm,
                       after, ffn_x, post, comb);
    llm_dsv4_moe_forward(c, L, ffn_x, il, token, ffn_out);
    llm_dsv4_block_post(c, ffn_out, after, post, comb, out_hc);

    /* arena: nessun free */
}

/* ── COMPOSIZIONE (livello 3): il layer DS4 come PIPELINE di passi dichiarati (`.llmarch`,
 * structure `layer`). Ogni `op` -> la primitiva DS4 corrispondente. Nessuna biforcazione:
 * `llm_dsv4_layer_forward_cached_pe` esegue QUESTA pipeline (ops==NULL -> sequenza canonica).
 * Lo stato (globals + setter) vive nel motore (llm_mlx_dsv4_layer_ops.h): il wiring del
 * `.llmarch` non dipende da questa cartella. ── */
#include "llm_mlx_dsv4_layer_ops.h"
#include "llm_dsv4_hcroute.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Decode device-resident: attn (project+mid+finish) + mHC post/pre + route in UNA cb. Flag: bit0 = attn eseguita, bit1 = mHC+route eseguiti. */
int llm_dsv4_gpu_attn_hcroute_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il,
        int ratio, int raw_cap, int comp_cap, const float* xn, int pos, float* out, const LlmDsv4HcRouteParams* hr);
#ifdef __cplusplus
}
#endif

/* Un layer su un token eseguendo la PIPELINE `ops` (default: canonica mhc/attn/mhc/mhc/moe/mhc). */
static inline void llm_dsv4_layer_forward_cached_pe_pipe(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* cache, const float* hc, int token, int pos, int il,
        float* out_hc, const char* const* ops, int n_ops) {
    int HC = c->n_hc, D = c->n_embd;
    int pf = llm_dsv4_pf_on(); double _t = 0;   /* TEMP: profilo decode (attn vs MoE) */
    static const char* def_ops[] = { "mhc_pre_attn", "attention", "mhc_post_attn",
                                     "mhc_pre_ffn", "moe", "mhc_post_ffn" };
    if (ops == NULL || n_ops <= 0) { ops = def_ops; n_ops = 6; }
    llm_dsv4_arena_reset();   /* arena azzerata a ogni layer: nessun malloc/free intra-layer */
    float* cur    = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC * D); memcpy(cur, hc, sizeof(float) * (size_t)HC * D);
    float* normed = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* proj   = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* saved  = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC * D);
    float* nxt    = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC * D);
    float* post   = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC);
    float* comb   = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)HC * HC);
    int tail_done = 0;
    /* decode device-resident (dsv4_attn_dev): mHC post(attn)+pre(ffn)+route fatti su GPU nella cb dell'attn */
    int route_fused = 0;
    float* ffn_normed = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D);
    float* ffn_split  = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)(2 + HC) * HC);
    int*   hr_sel     = (int*)llm_dsv4_arena_alloc(sizeof(int) * (size_t)c->n_expert_used);
    float* hr_rw      = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)c->n_expert_used);
    const int canon6 = n_ops == 6 && strcmp(ops[0], "mhc_pre_attn") == 0 && strcmp(ops[1], "attention") == 0 &&
                       strcmp(ops[2], "mhc_post_attn") == 0 && strcmp(ops[3], "mhc_pre_ffn") == 0 &&
                       strcmp(ops[4], "moe") == 0 && strcmp(ops[5], "mhc_post_ffn") == 0;
    for (int k = 0; k < n_ops; k++) {
        const char* op = ops[k];
        if (strcmp(op, "mhc_pre_attn") == 0) {
            memcpy(saved, cur, sizeof(float) * (size_t)HC * D);
            llm_dsv4_block_pre(c, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, L->attn_norm, cur, normed, post, comb);
        } else if (strcmp(op, "mhc_pre_ffn") == 0) {
            memcpy(saved, cur, sizeof(float) * (size_t)HC * D);
            if (route_fused) {   /* gia' calcolato su GPU: split = pre[HC] | post[HC] | comb[HC*HC] */
                memcpy(normed, ffn_normed, sizeof(float) * (size_t)D);
                memcpy(post, ffn_split + HC, sizeof(float) * (size_t)HC);
                memcpy(comb, ffn_split + 2 * HC, sizeof(float) * (size_t)HC * HC);
            } else
            llm_dsv4_block_pre(c, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, L->ffn_norm, cur, normed, post, comb);
        } else if (strcmp(op, "attention") == 0) {
            if (pf) _t = llm_dsv4_pf_now();
            /* La tabella completa registra 256 esperti per layer: sullo store
             * streaming il path selected-slots rimane il ramo efficiente (verificato:
             * il tail con store e' ~9.5ms/layer per l'I/O di 256 esperti). */
            int canonical_tail = out_hc && pe && !pe->store && n_ops == 6 && k == 1 &&
                strcmp(ops[0], "mhc_pre_attn") == 0 &&
                strcmp(ops[2], "mhc_post_attn") == 0 &&
                strcmp(ops[3], "mhc_pre_ffn") == 0 &&
                strcmp(ops[4], "moe") == 0 &&
                strcmp(ops[5], "mhc_post_ffn") == 0;
            if (canonical_tail) {
                /* Decode device-resident: project+mid+out in UNA catena (un solo wait) via
                 * llm_dsv4_gpu_attn_batch_try(nt=1); poi mHC post. Gated `dsv4_attn_dev` (default ON). */
                int dev = 0; { int v = 0; if (llm_ai_mlx_opt_get_bool("dsv4_attn_dev", &v) == 0) dev = v; }   /* default OFF (bit-exact) */
                if (dev && llm_dsv4_gpu_attn_batch_try(c, L, il, cache->compress_ratio, cache->raw_cap, cache->comp_cap,
                                                       normed, pos, 1, proj)) {
                    llm_dsv4_block_post(c, proj, saved, post, comb, out_hc);
                    tail_done = 1;
                } else {
                float* heads = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)c->n_head * c->n_head_dim);
                if (heads) {
                    if (llm_dsv4_attn_cached_mid(c, L, pe, cache, normed, pos, il, heads)) {
                        tail_done = llm_dsv4_gpu_attn_tail_try(c, L, pe, il, heads, saved, post, comb, pos, out_hc);
                        if (!tail_done && !llm_dsv4_gpu_attn_finish_try(c, L, il, heads, pos, proj))
                            llm_dsv4_attn_finish(c, L, pe, heads, pos, il, proj);
                    }
                } else {
                    llm_dsv4_attn_cached(c, L, pe, cache, normed, pos, il, proj);
                }
                }
            } else {
                /* DEVICE-RESIDENT per-token (gated `dsv4_attn_dev`): riusa il percorso del batch con
                 * nt=1 (project->slot device, mid device, out da device) -> 1 cb invece di 3.
                 * Verificare con l'oracolo di parita' (ramo dichiarato non-bit-exact). */
                /* DEFAULT ON (streaming): project+mid+finish in UNA cb (86 wait/token invece di 172). Verifica: id identici al
                 * percorso host su essay 24 tok e prompt corto 48 tok; sul prompt lungo (99 tok + 30 gen, attraversa pos 128) il
                 * device-resident coincide col vendored 30/30 mentre il percorso host diverge al token 14. Opt-out: dsv4_attn_dev=0. */
                int dev = 1; { int v = 1; if (llm_ai_mlx_opt_get_bool("dsv4_attn_dev", &v) == 0) dev = v; }
                int attn_done = 0;
                /* mHC post/pre + route su GPU nella cb dell'attn: opt-in `dsv4_hcroute` (default OFF). MISURATO NEGATIVO
                 * (id identici ma +27 ms/token di GPU: i kernel hc/sinkhorn/rms a 1 threadgroup sono piu' lenti del CPU e i wait
                 * restano 86): servirebbe prima ottimizzare quei kernel. */
                int hcr_on = 0; { int v = 0; if (llm_ai_mlx_opt_get_bool("dsv4_hcroute", &v) == 0) hcr_on = v; }
                if (dev && hcr_on && canon6 && il >= c->n_hash_layer && L->ffn_gate_inp) {
                    LlmDsv4HcRouteParams hr; memset(&hr, 0, sizeof hr);
                    hr.D = D; hr.HC = HC; hr.SIZE = (2 + HC) * HC; hr.HCD = HC * D;
                    hr.sinkhorn_iters = c->n_hc_sinkhorn_iter; hr.hc_eps = c->hc_eps; hr.rms_eps = c->rms_eps;
                    hr.n_expert = c->n_expert; hr.used = c->n_expert_used; hr.expert_scale = c->expert_weight_scale;
                    hr.res = saved; hr.post_attn = post; hr.comb_attn = comb;
                    hr.fn = L->hc_ffn_fn; hr.scale = L->hc_ffn_scale; hr.base = L->hc_ffn_base; hr.norm_w = L->ffn_norm;
                    hr.router_w = L->ffn_gate_inp; hr.exp_probs_b = L->exp_probs_b;
                    hr.after = nxt; hr.normed = ffn_normed; hr.split = ffn_split; hr.sel = hr_sel; hr.rw = hr_rw;
                    int fl = llm_dsv4_gpu_attn_hcroute_try(c, L, il, cache->compress_ratio, cache->raw_cap, cache->comp_cap,
                                                           normed, pos, proj, &hr);
                    attn_done = (fl & 1) != 0;
                    if (attn_done && (fl & 2)) { route_fused = 1; memcpy(cur, nxt, sizeof(float) * (size_t)HC * D); }
                }
                if (!attn_done && !(dev && llm_dsv4_gpu_attn_batch_try(c, L, il, cache->compress_ratio, cache->raw_cap,
                                                         cache->comp_cap, normed, pos, 1, proj)))
                    llm_dsv4_attn_cached(c, L, pe, cache, normed, pos, il, proj);
            }
            if (pf) g_pf_d_attn += llm_dsv4_pf_now() - _t;
            if (tail_done) break;
        } else if (strcmp(op, "moe") == 0) {
            if (pf) _t = llm_dsv4_pf_now();
            if (route_fused) llm_dsv4_gpu_set_route(hr_sel, hr_rw, c->n_expert_used);
            int moe_gpu_ok = llm_dsv4_gpu_moe_try(c, L, il, normed, token, proj);
            if (route_fused) llm_dsv4_gpu_set_route(NULL, NULL, 0);   /* mai lasciare una route stantia */
            if (!moe_gpu_ok) {
                if (pe && pe->gate) llm_dsv4_moe_forward_packed(c, L, pe, normed, il, token, proj);
                else                llm_dsv4_moe_forward(c, L, normed, il, token, proj);
            }
            if (pf) g_pf_d_moe += llm_dsv4_pf_now() - _t;
        } else if (strcmp(op, "mhc_post_attn") == 0 && route_fused) {
            /* mHC post(attn) gia' eseguito su GPU nella cb dell'attn: cur = after */
        } else if (strcmp(op, "mhc_post_attn") == 0 || strcmp(op, "mhc_post_ffn") == 0) {
            llm_dsv4_block_post(c, proj, saved, post, comb, nxt);
            memcpy(cur, nxt, sizeof(float) * (size_t)HC * D);
        }
        /* op sconosciuto: ignorato (il gap e' visibile nel log della pipeline) */
    }
    if (out_hc && !tail_done) memcpy(out_hc, cur, sizeof(float) * (size_t)HC * D);
    /* scratch persistente: NESSUN free (§0septies: niente malloc/churn nel hot path) */
}

/* Un layer su un token con cache KV. `pe` = esperti packed (NULL -> usa gli esperti f32).
 * Esegue la pipeline dichiarata (g_dsv4_layer_ops) o quella canonica se non impostata. */
static inline void llm_dsv4_layer_forward_cached_pe(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* cache, const float* hc, int token, int pos, int il,
        float* out_hc) {
    llm_dsv4_layer_forward_cached_pe_pipe(c, L, pe, cache, hc, token, pos, il, out_hc,
                                          g_dsv4_layer_nops > 0 ? g_dsv4_layer_ops : NULL,
                                          g_dsv4_layer_nops);
}
static inline void llm_dsv4_layer_forward_cached(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        LlmDsv4LayerCache* cache, const float* hc, int token, int pos, int il, float* out_hc) {
    llm_dsv4_layer_forward_cached_pe(c, L, NULL, cache, hc, token, pos, il, out_hc);
}

/* Un layer su un CHUNK di N token consecutivi (pos0..pos0+N-1), pipeline canonica. Stessa matematica di N
 * chiamate a llm_dsv4_layer_forward_cached_pe_pipe (le cache del layer avanzano token per token, in ordine),
 * ma in 3 fasi per layer: (1) mHC+attenzione per ogni token, (2) MoE per tutti i token (punto di batching:
 * esperti dell'unione letti una volta, una sola command buffer), (3) mHC post. `hc_in/hc_out` [N][HC*D]. */
static inline void llm_dsv4_layer_forward_batch(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* cache, const float* hc_in, const int* tokens,
        int pos0, int N, int il, float* hc_out) {
    const int HC = c->n_hc, D = c->n_embd; const size_t hcn = (size_t)HC * D; const int SZ = (2 + HC) * HC;
    llm_dsv4_arena_reset();   /* arena azzerata a ogni layer batch */
    int pf = llm_dsv4_pf_on(); double _t0 = 0; if (pf) { _t0 = llm_dsv4_pf_now(); g_pf_layers++; }
    /* mHC su GPU (opzione di registro dsv4_gpu_hc, default 1; 0 = percorso CPU bit-exact col token-per-token). Layout `sp` per token:
     * pre[HC] | post[HC] | comb[HC*HC]; il percorso CPU lo riempie nello stesso layout (post/comb) cosi' i due si alternano. */
    int gpu_hc = 1; { int v; if (llm_ai_mlx_opt_get_bool("dsv4_gpu_hc", &v) == 0) gpu_hc = v; }   /* default ON (verificato); 0 = CPU bit-exact */
    float* after  = (float*)llm_dsv4_arena_alloc(sizeof(float) * hcn * (size_t)N);
    float* ffn_x  = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D * N);
    float* ffn_o  = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D * N);
    float* sp_f   = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)SZ * N);
    float* sp_a   = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)SZ * N);
    float* normed_all = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D * N);
    float* attn_all   = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)D * N);
    #define SP_POST(sp, t) ((sp) + (size_t)SZ * (t) + HC)
    #define SP_COMB(sp, t) ((sp) + (size_t)SZ * (t) + 2 * HC)
    /* fase 1: mHC pre attn (tutti i token) -> attenzione (chunk su GPU, o per-token) -> mHC post attn + mHC pre ffn */
    int hc_a = gpu_hc && llm_dsv4_gpu_hc_pre_batch_try(c, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, L->attn_norm, hc_in, N, normed_all, sp_a);
    for (int t = 0; !hc_a && t < N; t++)
        llm_dsv4_block_pre(c, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, L->attn_norm, hc_in + hcn * t,
                           normed_all + (size_t)D * t, SP_POST(sp_a, t), SP_COMB(sp_a, t));
    int attn_gpu = llm_dsv4_gpu_attn_batch_try(c, L, il, cache->compress_ratio, cache->raw_cap, cache->comp_cap,
                                               normed_all, pos0, N, attn_all);
    if (!attn_gpu) {
        for (int t = 0; t < N; t++)
            llm_dsv4_attn_cached(c, L, pe, cache, normed_all + (size_t)D * t, pos0 + t, il, attn_all + (size_t)D * t);
    }
    int post_a_gpu = gpu_hc && llm_dsv4_gpu_hc_post_batch_try(c, attn_all, hc_in, sp_a, N, after);
    for (int t = 0; !post_a_gpu && t < N; t++)
        llm_dsv4_block_post(c, attn_all + (size_t)D * t, hc_in + hcn * t, SP_POST(sp_a, t), SP_COMB(sp_a, t), after + hcn * t);
    int hc_f = gpu_hc && llm_dsv4_gpu_hc_pre_batch_try(c, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, L->ffn_norm, after, N, ffn_x, sp_f);
    for (int t = 0; !hc_f && t < N; t++)
        llm_dsv4_block_pre(c, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, L->ffn_norm, after + hcn * t,
                           ffn_x + (size_t)D * t, SP_POST(sp_f, t), SP_COMB(sp_f, t));
    /* sp_a/attn_all/normed_all: arena (nessun free) */
    if (pf) { double _n = llm_dsv4_pf_now(); g_pf_t1 += _n - _t0; _t0 = _n;
              g_pf_hc_gpu += (hc_a?1:0)+(post_a_gpu?1:0)+(hc_f?1:0);
              g_pf_hc_cpu += (!hc_a)+(!post_a_gpu)+(!hc_f);
              g_pf_attn_gpu += attn_gpu?1:0; g_pf_attn_cpu += attn_gpu?0:1; }
    /* fase 2: MoE su tutti i token */
    const int moe_batched = llm_dsv4_gpu_moe_batch_try(c, L, il, ffn_x, tokens, N, ffn_o);
    for (int t = 0; !moe_batched && t < N; t++) {
        const float* x = ffn_x + (size_t)D * t; float* o = ffn_o + (size_t)D * t;
        if (!llm_dsv4_gpu_moe_try(c, L, il, x, tokens[t], o)) {
            if (pe && pe->gate) llm_dsv4_moe_forward_packed(c, L, pe, x, il, tokens[t], o);
            else                llm_dsv4_moe_forward(c, L, x, il, tokens[t], o);
        }
    }
    if (pf) { double _n = llm_dsv4_pf_now(); g_pf_tmoe += _n - _t0; _t0 = _n;
              g_pf_moe_batch += moe_batched?1:0; g_pf_moe_cpu += moe_batched?0:1; }
    /* fase 3: mHC post ffn */
    int post_f_gpu = gpu_hc && llm_dsv4_gpu_hc_post_batch_try(c, ffn_o, after, sp_f, N, hc_out);
    for (int t = 0; !post_f_gpu && t < N; t++)
        llm_dsv4_block_post(c, ffn_o + (size_t)D * t, after + hcn * t, SP_POST(sp_f, t), SP_COMB(sp_f, t), hc_out + hcn * t);
    if (pf) { double _n = llm_dsv4_pf_now(); g_pf_t3 += _n - _t0;
              g_pf_hc_gpu += post_f_gpu?1:0; g_pf_hc_cpu += post_f_gpu?0:1; }
    #undef SP_POST
    #undef SP_COMB
    /* sp_f/ffn_o/ffn_x/after: arena (nessun free) */
}

/* Tutti i layer su un token (nessuna cache: attenzione solo sul token corrente). */
static inline void llm_dsv4_forward_token(const LlmDsv4Config* c, const LlmDsv4Weights* w,
        const float* token_emb, int token, int pos, float* out_hc) {
    int HC = c->n_hc, D = c->n_embd;
    float* cur = llm_dsv4_slot(6, (size_t)HC * D);
    float* nxt = llm_dsv4_slot(7, (size_t)HC * D);
    llm_dsv4_hc_from_embedding(c, token_emb, cur);
    for (int il = 0; il < c->n_layer; il++) {
        llm_dsv4_layer_forward(c, &w->layers[il], cur, token, pos, il, NULL, 0, nxt);
        float* t = cur; cur = nxt; nxt = t;
    }
    if (out_hc) memcpy(out_hc, cur, sizeof(float) * (size_t)HC * D);
    /* slot: persistente, nessun free */
}

#endif /* LLM_DSV4_LAYER_H */
