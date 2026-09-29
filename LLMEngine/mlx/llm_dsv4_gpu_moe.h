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
    if (m->gate[il].type != 16 || m->up[il].type != 16) return false;   /* pair kernel = solo IQ2_XXS */
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
            mp[i] = midv + (size_t)i * ff;
            rw[i] = weight[i];
        }
        if (ok && llm_dsv4_metal_pair_swiglu_batch(m->be, gp, upp, x, mp, rw, used,
                                                   c->swiglu_clamp_exp, (uint32_t)embd,
                                                   (uint32_t)m->gate[il].row_bytes, (uint32_t)ff) != 0) ok = false;
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

    /* shared expert: f32 (riuso C); scratch dedicato */
    if (ok) {
        float* g = (float*)malloc(sizeof(float) * (size_t)ff);
        float* u = (float*)malloc(sizeof(float) * (size_t)ff);
        float* mid = (float*)malloc(sizeof(float) * (size_t)ff);
        ok = g && u && mid;
        if (ok) {
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
        free(g); free(u); free(mid);
    }
    free(downv); free(midv);
    return ok;
}

#endif /* LLM_DSV4_GPU_MOE_H */
