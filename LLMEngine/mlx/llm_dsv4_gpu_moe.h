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
    const int gt = m->gate[il].type, ut = m->up[il].type;
    if (!((gt == 16 && ut == 16) || ((gt == 10 || gt == 12) && (ut == 10 || ut == 12)))) return false;
    const int used = c->n_expert_used, ff = c->n_ff_exp, embd = c->n_embd;
    int selected[LLM_DSV4_MAX_EXPERT_USED];
    float weight[LLM_DSV4_MAX_EXPERT_USED];
    llm_dsv4_moe_route(c, L, x, il, token, selected, weight);   /* riuso C */

    float* midv  = (float*)malloc(sizeof(float) * (size_t)ff * used);   /* swiglu*peso per esperto */
    float* downv = (float*)malloc(sizeof(float) * (size_t)embd * used);
    bool ok = midv && downv;
    for (int o = 0; ok && o < embd; o++) out[o] = 0.0f;

    /* MoE FUSO ds4 (kernel verificato bit-exact vs path C): gate+up in un pass + swiglu*peso
     * in-kernel -> midv[i]. Esperto dalla CACHE (store_get) se streaming, altrimenti da mmap. */
    {
        const LlmDsv4PackedExperts* pe = &m->exps[il];
        int streaming = (pe->store && pe->store_get);
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
            float* gbuf = (float*)malloc(sizeof(float) * (size_t)ff * used);
            float* ubuf = (float*)malloc(sizeof(float) * (size_t)ff * used);
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
            free(gbuf); free(ubuf);
        }
    }
    /* batch 2: down dei used esperti in UNA command buffer (stessa cache/streaming). */
    if (ok) {
        LlmDsv4GpuJob jobs[LLM_DSV4_MAX_EXPERT_USED];
        int nj = 0;
        const LlmDsv4PackedExperts* pe = &m->exps[il];
        int streaming = (pe->store && pe->store_get);
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

    /* shared expert: Q8_0 packed (dot intero) se disponibile, altrimenti f32. */
    if (ok) {
        float* g = (float*)malloc(sizeof(float) * (size_t)ff);
        float* u = (float*)malloc(sizeof(float) * (size_t)ff);
        float* mid = (float*)malloc(sizeof(float) * (size_t)ff);
        ok = g && u && mid;
        if (ok) {
            const LlmDsv4PackedExperts* pe = &m->exps[il];
            if (pe->shexp_gate && pe->shexp_up && pe->shexp_down) {
                /* Q8_0 packed: dot intero NEON (come il path C) — ~5x meno del f32 scalare. */
                int64_t nb_g = llm_ggml_nbytes(pe->shexp_gate_type, embd);
                int64_t nb_d = llm_ggml_nbytes(pe->shexp_down_type, ff);
                int8_t* xq8 = (int8_t*)malloc((size_t)embd);
                float*  xs8 = (float*)malloc(sizeof(float) * (size_t)((embd + 31) / 32));
                llm_dsv4_q8_0_quantize_row(x, embd, xq8, xs8);
                for (int o = 0; o < ff; o++) {
                    g[o] = llm_dsv4_dot_q8_0_prequant(pe->shexp_gate + (int64_t)o * nb_g, embd, xq8, xs8);
                    u[o] = llm_dsv4_dot_q8_0_prequant(pe->shexp_up   + (int64_t)o * nb_g, embd, xq8, xs8);
                }
                llm_dsv4_swiglu(mid, g, u, ff, c->swiglu_clamp_exp);
                int8_t* mq8 = (int8_t*)malloc((size_t)ff);
                float*  ms8 = (float*)malloc(sizeof(float) * (size_t)((ff + 31) / 32));
                llm_dsv4_q8_0_quantize_row(mid, ff, mq8, ms8);
                for (int o = 0; o < embd; o++) {
                    float v = llm_dsv4_dot_q8_0_prequant(pe->shexp_down + (int64_t)o * nb_d, ff, mq8, ms8);
                    if (isfinite(v)) out[o] += v;
                }
                free(ms8); free(mq8); free(xs8); free(xq8);
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
        free(g); free(u); free(mid);
    }
    free(downv); free(midv);
    return ok;
}

#endif /* LLM_DSV4_GPU_MOE_H */
