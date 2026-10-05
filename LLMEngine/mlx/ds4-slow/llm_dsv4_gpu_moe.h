/* llm_dsv4_gpu_moe.h — MoE DS4 su GPU (F3): esperti instradati via kernel Metal IQ2_XXS/Q2_K,
 * routing e shared-expert riusano il path C (§0bis, nessuna duplicazione). Attivazioni f32
 * (parità con llm_dsv4_dot_tensor). GPU attiva di default. Fallback: false -> path C.
 *
 * Il contesto crea una view zero-copy per tensore (gate/up/down) di ogni layer sul mmap:
 * niente copia dei 97 GB (store/swap non supportato qui -> false). */
#ifndef LLM_DSV4_GPU_MOE_H
#define LLM_DSV4_GPU_MOE_H

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "llm_ggml.h"
#include "llm_dsv4.h"
#include "llm_dsv4_gpu.h"       /* backend Metal + llm_dsv4_gpu_enabled */
#include "llm_dsv4_metal.h"     /* backend Metal (C API) */
#include "llm_dsv4_moe_fwd.h"   /* router/swiglu/linear riusati */
#include "llm_dsv4_store.h"     /* capacità dello store (guardia puntatori stabili: cap >= used*nt) */
#include "llm_engine_opts.h"       /* registro opzioni (dsv4_prefill_f16) */
#ifdef __cplusplus
extern "C" {
#endif
int llm_dsv4_gpu_route_batch_try(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L, int il, const float* x, const int* tokens, int nt, int* sel, float* wt);
#ifdef __cplusplus
}
#endif
#define LLM_DSV4_MM_MIN_TOKENS 192   /* soglia chunk per mul_mm_id (sotto: kernel per-token) */

/* TEMP (gated ds4_profile): split I/O (store) vs GEMM del MoE batched. Le variabili vivono nel
 * TU del .mm (dove gira il batch); esposte via llm_dsv4_gpu_moe_prof() definita in gpu_ctx.mm. */
#include <time.h>
static double g_dsv4_moe_io_ms = 0, g_dsv4_moe_mm_ms = 0;
/* TEMP (dsv4_profile): sotto-timer della glue HOST del MoE per-token: route (CPU) / store_get /
 * call Metal (setup+dispatch+wait). */
static double g_dsv4_pt_route_ms = 0, g_dsv4_pt_store_ms = 0, g_dsv4_pt_metal_ms = 0;
extern int g_dsv4_prof_global;   /* flag condiviso (il registro opzioni e' per-TU) */
static inline double llm_dsv4_gm_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
    LlmDsv4Metal* be;
    const LlmDsv4PackedExperts* exps;   /* per store_get (cache esperti condivisa) */
    LlmDsv4GpuView gate[LLM_DSV4_MAX_LAYER], up[LLM_DSV4_MAX_LAYER], down[LLM_DSV4_MAX_LAYER];
    int inited;
} LlmDsv4MoeGpu;

static inline void llm_dsv4_gpu_moe_destroy(LlmDsv4MoeGpu* m) {
    if (!m) return;
    memset(m, 0, sizeof *m);   /* nessuna view persistente da rilasciare (transitorie) */
}

static inline bool llm_dsv4_gpu_moe_init(LlmDsv4MoeGpu* m, LlmDsv4Metal* be, const LlmDsv4Config* c,
                                         const LlmDsv4PackedExperts* exps) {
    memset(m, 0, sizeof *m);
    m->be = be;
    m->exps = exps;
    for (int il = 0; il < c->n_layer; il++) {
        const LlmDsv4PackedExperts* pe = &exps[il];
        size_t gb = (size_t)llm_ggml_nbytes(pe->gate_type, pe->in_dim);
        size_t db = (size_t)llm_ggml_nbytes(pe->down_type, pe->ff_dim);
        if (pe->store && pe->store_get) {
            /* STREAMING/CACHE (preferito): l'mmap non è cache e viene sfrattato → SSD a ogni
             * token. La cache condivisa tiene l'esperto in RAM. Fetch per-esperto in forward. */
            m->gate[il] = (LlmDsv4GpuView){ NULL, pe->gate_type, gb };
            m->up[il]   = (LlmDsv4GpuView){ NULL, pe->up_type,   gb };
            m->down[il] = (LlmDsv4GpuView){ NULL, pe->down_type, db };
        } else if (pe->gate && pe->up && pe->down) {
            /* RESIDENTE: view zero-copy sull'mmap. */
            m->gate[il] = (LlmDsv4GpuView){ pe->gate, pe->gate_type, gb };
            m->up[il]   = (LlmDsv4GpuView){ pe->up,   pe->up_type,   gb };
            m->down[il] = (LlmDsv4GpuView){ pe->down, pe->down_type, db };
        } else {
            return false;   /* né cache né residente: la GPU non può (fallback C) */
        }
    }
    m->inited = 1;
    return true;
}

/* MoE completo GPU. `x` [embd] -> `out` [embd].false -> il chiamante usa il path C.
 * Il routing e lo shared expert restano nel path C (riuso); solo gli esperti instradati
 * girano sui kernel Metal. */
static inline bool llm_dsv4_gpu_moe(LlmDsv4MoeGpu* m, const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
                                    int il, const float* x, int token, float* out) {
    if (!llm_dsv4_gpu_enabled() || !m || !m->inited || il < 0 || il >= c->n_layer) return false;
    llm_dsv4_arena_reset();   /* .mm TU: l'arena e' per-TU; reset a ogni chiamata (bounded) */
    const int gt = m->gate[il].type, ut = m->up[il].type;
    if (!((gt == 16 && ut == 16) || ((gt == 10 || gt == 12) && (ut == 10 || ut == 12)))) return false;
    const int used = c->n_expert_used, ff = c->n_ff_exp, embd = c->n_embd;
    const int dt = m->down[il].type;
    const LlmDsv4PackedExperts* pe = &m->exps[il];
    const int streaming = (pe->store && pe->store_get) != 0;
    int selected[LLM_DSV4_MAX_EXPERT_USED];
    float weight[LLM_DSV4_MAX_EXPERT_USED];
    int pf_g = (g_dsv4_prof_global > 0) ? 1 : 0;   /* flag condiviso (registro opzioni per-TU) */
    double _pt = pf_g ? llm_dsv4_gm_now() : 0;
    llm_dsv4_moe_route(c, L, x, il, token, selected, weight);   /* riuso C */
    if (pf_g) { g_dsv4_pt_route_ms += llm_dsv4_gm_now() - _pt; _pt = llm_dsv4_gm_now(); }
    bool ok = true;
    int shexp_gpu = 0;

    if ((gt == 16 && ut == 16 && dt == 10) || (gt == 12 && ut == 12 && dt == 12)) {
        /* PATH ds4 (IQ2_XXS/Q2_K o Q4_K): pair_swiglu + sum6 in UNA command buffer (mid f32 su device, nessun
         * round-trip CPU). 37/43 layer di ds4flash (gate/up IQ2_XXS, down Q2_K). */
        const void* gp[LLM_DSV4_MAX_EXPERT_USED];
        const void* upp[LLM_DSV4_MAX_EXPERT_USED];
        const void* dp[LLM_DSV4_MAX_EXPERT_USED];
        float rw[LLM_DSV4_MAX_EXPERT_USED];
        for (int i = 0; ok && i < used; i++) {
            int e = selected[i];
            if (streaming) {
                const uint8_t *g = NULL, *u = NULL, *d = NULL;
                pe->store_get(pe->store, il, e, &g, &u, &d);
                if (!g || !u || !d) { ok = false; break; }
                gp[i] = g; upp[i] = u; dp[i] = d;
            } else {
                gp[i]  = (const uint8_t*)m->gate[il].w + (size_t)e * (size_t)ff * m->gate[il].row_bytes;
                upp[i] = (const uint8_t*)m->up[il].w   + (size_t)e * (size_t)ff * m->up[il].row_bytes;
                dp[i]  = (const uint8_t*)m->down[il].w + (size_t)e * (size_t)embd * m->down[il].row_bytes;
            }
            rw[i] = weight[i];
        }
        if (pf_g) { g_dsv4_pt_store_ms += llm_dsv4_gm_now() - _pt; _pt = llm_dsv4_gm_now(); }
        const void* sg = (pe->shexp_gate && pe->shexp_up && pe->shexp_down) ? pe->shexp_gate : NULL;
        if (ok) ok = (gt == 12 ? llm_dsv4_metal_moe_q4k_batch(m->be, 1, gp, upp, dp, x, rw, used,
                                                c->swiglu_clamp_exp, (uint32_t)embd,
                                                (uint32_t)m->gate[il].row_bytes, (uint32_t)ff,
                                                (uint32_t)m->down[il].row_bytes,
                                                sg, sg ? pe->shexp_up : NULL, sg ? pe->shexp_down : NULL,
                                                (uint32_t)llm_ggml_nbytes(pe->shexp_gate_type, embd),
                                                (uint32_t)llm_ggml_nbytes(pe->shexp_down_type, ff),
                                                out) : llm_dsv4_metal_moe_iq2_q2k(m->be, gp, upp, dp, x, rw, used, c->swiglu_clamp_exp, (uint32_t)embd,
                                                (uint32_t)m->gate[il].row_bytes, (uint32_t)ff, (uint32_t)m->down[il].row_bytes,
                                                (pe->shexp_gate && pe->shexp_up && pe->shexp_down) ? pe->shexp_gate : NULL,
                                                (pe->shexp_gate && pe->shexp_up && pe->shexp_down) ? pe->shexp_up : NULL,
                                                (pe->shexp_gate && pe->shexp_up && pe->shexp_down) ? pe->shexp_down : NULL,
                                                (uint32_t)llm_ggml_nbytes(pe->shexp_gate_type, embd), (uint32_t)llm_ggml_nbytes(pe->shexp_down_type, ff), out)) == 0;
        if (pf_g) { g_dsv4_pt_metal_ms += llm_dsv4_gm_now() - _pt; }
        if (ok && sg) shexp_gpu = 1;
    } else {
        float* midv  = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)ff * used);
        float* downv = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)embd * used);
        ok = midv && downv;
        for (int o = 0; ok && o < embd; o++) out[o] = 0.0f;
        /* batch 1: gate+up (+swiglu). */
        if (ok) {
            const void* gp[LLM_DSV4_MAX_EXPERT_USED];
            const void* upp[LLM_DSV4_MAX_EXPERT_USED];
            float* mp[LLM_DSV4_MAX_EXPERT_USED];
            float rw[LLM_DSV4_MAX_EXPERT_USED];
            for (int i = 0; ok && i < used; i++) { mp[i] = midv + (size_t)i * ff; rw[i] = weight[i]; }
            if (gt == 16) {
                for (int i = 0; ok && i < used; i++) {
                    int e = selected[i];
                    if (streaming) {
                        const uint8_t *g = NULL, *u = NULL, *d = NULL;
                        pe->store_get(pe->store, il, e, &g, &u, &d);
                        if (!g || !u) { ok = false; break; }
                        gp[i] = g; upp[i] = u;
                    } else {
                        gp[i]  = (const uint8_t*)m->gate[il].w + (size_t)e * (size_t)ff * m->gate[il].row_bytes;
                        upp[i] = (const uint8_t*)m->up[il].w   + (size_t)e * (size_t)ff * m->up[il].row_bytes;
                    }
                }
                if (ok && llm_dsv4_metal_pair_swiglu_batch(m->be, gp, upp, x, mp, rw, used,
                                                           c->swiglu_clamp_exp, (uint32_t)embd,
                                                           (uint32_t)m->gate[il].row_bytes, (uint32_t)ff) != 0) ok = false;
            } else {
                /* gate/up Q4_K/Q2_K: niente pair kernel -> gemv Q8_K (x) + swiglu*peso su CPU. */
                float* gbuf = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)ff * used);
                float* ubuf = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)ff * used);
                ok = gbuf && ubuf;
                LlmDsv4GpuJob gj[2 * LLM_DSV4_MAX_EXPERT_USED]; int ng = 0;
                for (int i = 0; ok && i < used; i++) {
                    int e = selected[i]; const uint8_t *g = NULL, *u = NULL, *d = NULL;
                    if (streaming) pe->store_get(pe->store, il, e, &g, &u, &d);
                    else { g  = (const uint8_t*)m->gate[il].w + (size_t)e * (size_t)ff * m->gate[il].row_bytes;
                           u  = (const uint8_t*)m->up[il].w   + (size_t)e * (size_t)ff * m->up[il].row_bytes; }
                    if (!g || !u) { ok = false; break; }
                    gj[ng++] = (LlmDsv4GpuJob){ g, 0, gt, m->gate[il].row_bytes, (uint32_t)embd, (uint32_t)ff, x, gbuf + (size_t)i * ff };
                    gj[ng++] = (LlmDsv4GpuJob){ u, 0, ut, m->up[il].row_bytes,   (uint32_t)embd, (uint32_t)ff, x, ubuf + (size_t)i * ff };
                }
                if (ok) ok = llm_dsv4_metal_gemv_batch(m->be, gj, ng) == 0;
                for (int i = 0; ok && i < used; i++) {
                    llm_dsv4_swiglu(mp[i], gbuf + (size_t)i * ff, ubuf + (size_t)i * ff, ff, c->swiglu_clamp_exp);
                    for (int j = 0; j < ff; j++) mp[i][j] *= rw[i];
                    for (int j = 0; j < ff; j++) if (!isfinite(mp[i][j])) mp[i][j] = 0.0f;
                }
                /* arena: nessun free */
            }
        }
        /* batch 2: down dei used esperti in UNA command buffer (stessa cache/streaming). */
        if (ok) {
            LlmDsv4GpuJob jobs[LLM_DSV4_MAX_EXPERT_USED];
            int nj = 0;
            for (int i = 0; i < used; i++) {
                int e = selected[i];
                const uint8_t *wd = m->down[il].w;
                size_t od = (size_t)e * (size_t)embd * m->down[il].row_bytes;
                if (streaming) {
                    const uint8_t *g = NULL, *u = NULL, *d = NULL;
                    pe->store_get(pe->store, il, e, &g, &u, &d);
                    if (!d) { ok = false; break; }
                    wd = d; od = 0;
                }
                jobs[nj++] = (LlmDsv4GpuJob){ wd, od, m->down[il].type, m->down[il].row_bytes, (uint32_t)ff, (uint32_t)embd, midv + (size_t)i * ff, downv + (size_t)i * embd };
            }
            if (ok) ok = llm_dsv4_metal_gemv_batch(m->be, jobs, nj) == 0;
        }
        for (int i = 0; ok && i < used; i++)
            for (int o = 0; o < embd; o++) if (isfinite(downv[(size_t)i * embd + o])) out[o] += downv[(size_t)i * embd + o];
        /* arena: nessun free */
    }

    /* shared expert: Q8_0 packed (dot intero) se disponibile, altrimenti f32. */
    if (ok && !shexp_gpu) {
        float* g = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)ff);
        float* u = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)ff);
        float* mid = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)ff);
        ok = g && u && mid;
        if (ok) {
            const LlmDsv4PackedExperts* pe = &m->exps[il];
            if (pe->shexp_gate && pe->shexp_up && pe->shexp_down) {
                /* Q8_0 packed: dot intero NEON (come il path C) — ~5x meno del f32 scalare. */
                int64_t nb_g = llm_ggml_nbytes(pe->shexp_gate_type, embd);
                int64_t nb_d = llm_ggml_nbytes(pe->shexp_down_type, ff);
                int8_t* xq8 = (int8_t*)llm_dsv4_arena_alloc((size_t)embd);
                float*  xs8 = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)((embd + 31) / 32));
                llm_dsv4_q8_0_quantize_row(x, embd, xq8, xs8);
                for (int o = 0; o < ff; o++) {
                    g[o] = llm_dsv4_dot_q8_0_prequant(pe->shexp_gate + (int64_t)o * nb_g, embd, xq8, xs8);
                    u[o] = llm_dsv4_dot_q8_0_prequant(pe->shexp_up   + (int64_t)o * nb_g, embd, xq8, xs8);
                }
                llm_dsv4_swiglu(mid, g, u, ff, c->swiglu_clamp_exp);
                int8_t* mq8 = (int8_t*)llm_dsv4_arena_alloc((size_t)ff);
                float*  ms8 = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)((ff + 31) / 32));
                llm_dsv4_q8_0_quantize_row(mid, ff, mq8, ms8);
                for (int o = 0; o < embd; o++) {
                    float v = llm_dsv4_dot_q8_0_prequant(pe->shexp_down + (int64_t)o * nb_d, ff, mq8, ms8);
                    if (isfinite(v)) out[o] += v;
                }
                /* arena: nessun free */
            } else {
                llm_dsv4_linear(g, L->ffn_gate_shexp, ff, embd, x);
                llm_dsv4_linear(u, L->ffn_up_shexp,   ff, embd, x);
                llm_dsv4_swiglu(mid, g, u, ff, c->swiglu_clamp_exp);
                for (int o = 0; o < embd; o++) {
                    const float* wr = L->ffn_down_shexp + (long)o * ff;
                    double acc = 0.0;
                    for (int j = 0; j < ff; j++) acc += (double)wr[j] * mid[j];
                    out[o] += (float)acc;
                }
            }
        }
        /* arena: nessun free */
    }
    return ok;
}

/* MoE GPU per un CHUNK di `nt` token (X [nt*embd] -> OUT [nt*embd]) in UNA command buffer. Solo per i layer con
 * gate/up IQ2_XXS + down Q2_K (path ds4); altrimenti false e il chiamante ricade sul per-token. Routing su CPU
 * per token; gli esperti dell'unione sono letti dallo store una volta (puntatori stabili: cap >= 6*nt). */
static inline bool llm_dsv4_gpu_moe_batch(LlmDsv4MoeGpu* m, const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
                                          int il, const float* X, const int* tokens, int nt, float* OUT) {
    if (!llm_dsv4_gpu_enabled() || !m || !m->inited || il < 0 || il >= c->n_layer || nt <= 0) return false;
    llm_dsv4_arena_reset();   /* .mm TU: reset a ogni chiamata (bounded) */
    const int used = c->n_expert_used, ff = c->n_ff_exp, embd = c->n_embd;
    const LlmDsv4PackedExperts* pe = &m->exps[il];
    const int streaming = (pe->store && pe->store_get) != 0;
    /* PUNTATORI STABILI: il batch raccoglie i puntatori di TUTTI gli esperti dell'unione PRIMA di
     * usarli (grandi gp/upp/dp), quindi lo store deve poterli tenere TUTTI senza riusare slot:
     * serve cap >= used*nt. Se cap e' piu' piccolo (chunk grande, o cap ridotto dalla pressione di
     * memoria -> bug "garbage a freddo", cfr. Bugs/(open) - DS4-Garbage-Cold-Prefill-Batch-Zeri),
     * processa il chunk in SUB-CHUNK di cap/used token (l'unione di ognuno entra nella cache). */
    if (streaming) {
        int capn = (int)((const LlmDsv4Store*)pe->store)->cap;
        int used = c->n_expert_used > 0 ? c->n_expert_used : 1;
        if (capn < nt * used) {
            int sub = capn / used;
            if (sub < 1) return false;   /* cap < used: puntatori non garantibili -> fallback per-token */
            for (int t0 = 0; t0 < nt; t0 += sub) {
                int m2 = nt - t0 < sub ? nt - t0 : sub;
                if (!llm_dsv4_gpu_moe_batch(m, c, L, il, X + (size_t)t0 * (size_t)c->n_embd, tokens + t0, m2,
                                            OUT + (size_t)t0 * (size_t)c->n_embd))
                    return false;
            }
            return true;
        }
    }
    const int is_iq2 = m->gate[il].type == 16 && m->up[il].type == 16 && m->down[il].type == 10;
    const int is_q4  = m->gate[il].type == 12 && m->up[il].type == 12 && m->down[il].type == 12;
    if (!(is_iq2 || is_q4) || used != 6) return false;
    const uint8_t *g = NULL, *u = NULL, *d = NULL;
    int f16_ok = 1; { int v; if (llm_ai_mlx_opt_get_bool("dsv4_prefill_f16", &v) == 0) f16_ok = v; }
    /* Layer Q4_K con ARENA (opzione dsv4_q4_arena, SPERIMENTALE) + attivazioni half: mul_mm_id sui tensori impilati. Altrimenti
     * il percorso normale sotto (kernel ds4 slots6 Q4_K, attivazioni f32) con gli esperti dello store. */
    if (is_q4 && f16_ok && pe->q4_stacks && pe->q4_ensure && pe->q4_stacks(pe->store, il, &g, &u, &d) && pe->shexp_gate && pe->shexp_up && pe->shexp_down) {
        int* sel = (int*)llm_dsv4_arena_alloc(sizeof(int) * (size_t)nt * used); float* wts = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)nt * used);
        if (!sel || !wts) return false;
        for (int t = 0; t < nt; t++) llm_dsv4_moe_route(c, L, X + (size_t)t * embd, il, tokens[t], sel + (size_t)t * used, wts + (size_t)t * used);
        bool okq = pe->q4_ensure(pe->store, il, sel, nt * used) == 0;
        if (okq) okq = llm_dsv4_metal_moe_mm_batch(m->be, nt, g, u, d, 12, 12, (uint32_t)c->n_expert, (uint32_t)embd, (uint32_t)ff,
                                                   (uint32_t)m->gate[il].row_bytes, (uint32_t)m->down[il].row_bytes, sel, wts, X,
                                                   c->swiglu_clamp_exp, pe->shexp_gate, pe->shexp_up, pe->shexp_down,
                                                   (uint32_t)llm_ggml_nbytes(pe->shexp_gate_type, embd),
                                                   (uint32_t)llm_ggml_nbytes(pe->shexp_down_type, ff), OUT) == 0;
        /* arena: nessun free */
        return okq;
    }
    const void** gp = (const void**)llm_dsv4_arena_alloc(sizeof(void*) * 3 * (size_t)nt * used);
    float* rw = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)nt * used);
    if (!gp || !rw) return false;
    const void** upp = gp + (size_t)nt * used; const void** dp = upp + (size_t)nt * used;
    bool ok = true;
    int* sel_all = (int*)llm_dsv4_arena_alloc(sizeof(int) * (size_t)nt * used);
    float* w_all = (float*)llm_dsv4_arena_alloc(sizeof(float) * (size_t)nt * used);
    if (!sel_all || !w_all) return false;
    { int gpu_rt = 1; { int v; if (llm_ai_mlx_opt_get_bool("dsv4_gpu_hc", &v) == 0) gpu_rt = v; }   /* stessa opzione dell'mHC GPU */
      if (!(gpu_rt && llm_dsv4_gpu_route_batch_try(c, L, il, X, tokens, nt, sel_all, w_all)))
          for (int t = 0; t < nt; t++) llm_dsv4_moe_route(c, L, X + (size_t)t * embd, il, tokens[t], sel_all + (size_t)t * used, w_all + (size_t)t * used); }
    int pf_m = 0; { int v = 0; if (llm_ai_mlx_opt_get_bool("dsv4_profile", &v) == 0) pf_m = v; }
    double _tm0 = 0; if (pf_m) _tm0 = llm_dsv4_gm_now();
    if (streaming && pe->store_prefetch) pe->store_prefetch(pe->store, il, sel_all, nt * used);   /* letture parallele dei mancanti */
    for (int t = 0; ok && t < nt; t++) {
        int* selected = sel_all + (size_t)t * used; float* weight = w_all + (size_t)t * used;
        for (int i = 0; ok && i < used; i++) {
            const int e = selected[i], k = t * used + i;
            if (streaming) {
                const uint8_t *g = NULL, *u = NULL, *d = NULL;
                pe->store_get(pe->store, il, e, &g, &u, &d);
                if (!g || !u || !d) { ok = false; break; }
                gp[k] = g; upp[k] = u; dp[k] = d;
            } else {
                gp[k]  = (const uint8_t*)m->gate[il].w + (size_t)e * (size_t)ff * m->gate[il].row_bytes;
                upp[k] = (const uint8_t*)m->up[il].w   + (size_t)e * (size_t)ff * m->up[il].row_bytes;
                dp[k]  = (const uint8_t*)m->down[il].w + (size_t)e * (size_t)embd * m->down[il].row_bytes;
            }
            rw[k] = weight[i];
        }
    }
    if (pf_m) { double _n = llm_dsv4_gm_now(); g_dsv4_moe_io_ms += _n - _tm0; _tm0 = _n; }
    /* Chunk grande (come ds4: tanti token per esperto): mul_mm_id con tabella di indirizzi degli esperti dello store, shared expert incluso.
     * Sotto la soglia i kernel per-token (slots6) sono piu' veloci. Attivazioni half (opzione dsv4_prefill_f16). */
    int done_mm = 0;
    int mm_on = 1; { int v; if (llm_ai_mlx_opt_get_bool("dsv4_prefill_mm", &v) == 0) mm_on = v; }   /* default ON (verificato E2E: prefill 2000 tok 13 -> 38 t/s) */
    if (ok && streaming && mm_on && f16_ok && nt >= LLM_DSV4_MM_MIN_TOKENS && c->n_expert > 0 && c->n_expert <= 1024 &&
        pe->shexp_gate && pe->shexp_up && pe->shexp_down) {
        const int ne = c->n_expert;
        const void** ga = (const void**)llm_dsv4_arena_alloc0((size_t)ne * sizeof(void*)); const void** ua = (const void**)llm_dsv4_arena_alloc0((size_t)ne * sizeof(void*));
        const void** da = (const void**)llm_dsv4_arena_alloc0((size_t)ne * sizeof(void*));
        if (ga && ua && da) {
            for (int k = 0; k < nt * used; k++) { const int ex = sel_all[k]; ga[ex] = gp[k]; ua[ex] = upp[k]; da[ex] = dp[k]; }
            done_mm = llm_dsv4_metal_moe_mm_addr_batch(m->be, nt, ga, ua, da, m->gate[il].type, m->down[il].type, (uint32_t)ne, (uint32_t)embd, (uint32_t)ff,
                                                       (uint32_t)m->gate[il].row_bytes, (uint32_t)m->down[il].row_bytes, sel_all, rw, X, c->swiglu_clamp_exp,
                                                       pe->shexp_gate, pe->shexp_up, pe->shexp_down, (uint32_t)llm_ggml_nbytes(pe->shexp_gate_type, embd),
                                                       (uint32_t)llm_ggml_nbytes(pe->shexp_down_type, ff), OUT) == 0;
        }
        /* arena: nessun free */
    }
    if (ok && !done_mm) {
        const void* sg = (pe->shexp_gate && pe->shexp_up && pe->shexp_down) ? pe->shexp_gate : NULL;
        ok = (is_q4 ? llm_dsv4_metal_moe_q4k_batch : llm_dsv4_metal_moe_iq2_q2k_batch)(m->be, nt, gp, upp, dp, X, rw, used, c->swiglu_clamp_exp, (uint32_t)embd,
                                              (uint32_t)m->gate[il].row_bytes, (uint32_t)ff, (uint32_t)m->down[il].row_bytes,
                                              sg, sg ? pe->shexp_up : NULL, sg ? pe->shexp_down : NULL,
                                              (uint32_t)llm_ggml_nbytes(pe->shexp_gate_type, embd),
                                              (uint32_t)llm_ggml_nbytes(pe->shexp_down_type, ff), OUT) == 0;
        /* senza shared expert su GPU il path per-token lo aggiunge su CPU: qui non e' coperto -> false */
        if (ok && !sg) ok = false;
    }
    if (pf_m) { double _n = llm_dsv4_gm_now(); g_dsv4_moe_mm_ms += _n - _tm0; }
    /* arena: nessun free */
    return ok;
}

#endif /* LLM_DSV4_GPU_MOE_H */
