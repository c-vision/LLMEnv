/* llm_dsv4_gpu_attn.h — attenzione DS4 (F2): proiezioni Q8_0 su GPU (kernel Metal),
 * RoPE/RMSNorm riusano il path C (§0bis). GPU attiva di default, fallback -> false.
 * Semantica = ramo f32 di llm_dsv4_attn_project/llm_dsv4_attn_finish (attivazioni f32). */
#ifndef LLM_DSV4_GPU_ATTN_H
#define LLM_DSV4_GPU_ATTN_H

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "llm_ggml.h"
#include "llm_dsv4.h"
#include "llm_dsv4_weights.h"
#include "llm_dsv4_attn_fwd.h"   /* rms/rope riusati */
#include "llm_dsv4_gpu.h"
#include "llm_dsv4_metal.h"

typedef struct {
    LlmDsv4Metal* be;
    LlmDsv4GpuView qa[LLM_DSV4_MAX_LAYER], qb[LLM_DSV4_MAX_LAYER], kv[LLM_DSV4_MAX_LAYER];
    LlmDsv4GpuView outa[LLM_DSV4_MAX_LAYER], outb[LLM_DSV4_MAX_LAYER];
    int inited;
} LlmDsv4AttnGpu;

static inline bool llm_dsv4_gpu_view(LlmDsv4Metal* be, const uint8_t* p, int type, int in_dim,
                                     int rows, LlmDsv4GpuView* v) {
    (void)be; (void)rows;
    if (!p) return false;
    v->w = p; v->type = type; v->row_bytes = (size_t)llm_ggml_nbytes(type, in_dim);
    return true;
}

static inline void llm_dsv4_gpu_attn_destroy(LlmDsv4AttnGpu* m) {
    if (!m) return;
    memset(m, 0, sizeof *m);   /* nessuna view persistente (transitorie) */
}

static inline bool llm_dsv4_gpu_attn_init(LlmDsv4AttnGpu* m, LlmDsv4Metal* be, const LlmDsv4Config* c,
                                          const LlmDsv4PackedExperts* pe) {
    memset(m, 0, sizeof *m);
    m->be = be;
    const int lq = c->n_lora_q, embd = c->n_embd, qd = c->n_head * c->n_head_dim, hd = c->n_head_dim;
    const int ng = c->n_out_group, rank = c->n_lora_o, gd = hd * (c->n_head / ng), ol = ng * rank;
    for (int il = 0; il < c->n_layer; il++) {
        const LlmDsv4PackedExperts* p = &pe[il];
        if (!p->q_a || !p->q_b || !p->kv || !p->out_a || !p->out_b) return false;
        if (!llm_dsv4_gpu_view(be, p->q_a,   p->q_a_type,   embd, lq, &m->qa[il]))   { llm_dsv4_gpu_attn_destroy(m); return false; }
        if (!llm_dsv4_gpu_view(be, p->q_b,   p->q_b_type,   lq,   qd, &m->qb[il]))   { llm_dsv4_gpu_attn_destroy(m); return false; }
        if (!llm_dsv4_gpu_view(be, p->kv,    p->kv_type,    embd, hd, &m->kv[il]))   { llm_dsv4_gpu_attn_destroy(m); return false; }
        if (!llm_dsv4_gpu_view(be, p->out_a, p->out_a_type, gd,   ol, &m->outa[il])) { llm_dsv4_gpu_attn_destroy(m); return false; }
        if (!llm_dsv4_gpu_view(be, p->out_b, p->out_b_type, ol,   embd, &m->outb[il])) { llm_dsv4_gpu_attn_destroy(m); return false; }
    }
    m->inited = 1;
    return true;
}

/* Proiezioni Q-LoRA + KV (input già normalizzato `xn`): q [n_head*hd], kv [hd], qrn [lq]. */
static inline bool llm_dsv4_gpu_attn_project(LlmDsv4AttnGpu* m, const LlmDsv4Config* c,
        const LlmDsv4LayerWeights* L, int il, const float* xn, int pos, float* q, float* kv, float* qrn) {
    if (!llm_dsv4_gpu_enabled() || !m || !m->inited) return false;
    const int lq = c->n_lora_q, embd = c->n_embd, qd = c->n_head * c->n_head_dim, hd = c->n_head_dim;
    float* qr = (float*)malloc(sizeof(float) * (size_t)lq);
    float* kv_raw = (float*)malloc(sizeof(float) * (size_t)hd);
    bool ok = qr && kv_raw;
    if (ok) {
        LlmDsv4GpuJob jobs[2] = {
            { m->qa[il].w, 0, m->qa[il].type, m->qa[il].row_bytes, (uint32_t)embd, (uint32_t)lq, xn, qr },
            { m->kv[il].w, 0, m->kv[il].type, m->kv[il].row_bytes, (uint32_t)embd, (uint32_t)hd, xn, kv_raw },
        };
        ok = llm_dsv4_metal_gemv_batch(m->be, jobs, 2) == 0;
    }
    if (ok) {
        llm_dsv4_rms_norm_weight(qrn, qr, L->attn_q_a_norm, lq, c->rms_eps);
        LlmDsv4GpuJob j = { m->qb[il].w, 0, m->qb[il].type, m->qb[il].row_bytes,
                            (uint32_t)lq, (uint32_t)qd, qrn, q };
        ok = llm_dsv4_metal_gemv_batch(m->be, &j, 1) == 0;
    }
    if (ok) {
        llm_dsv4_head_rms_norm_inplace(q, c->n_head, hd, c->rms_eps);
        llm_dsv4_rope_tail_layer_inplace(c, q, c->n_head, hd, c->n_rot, pos, il, 0);
        llm_dsv4_rms_norm_weight(kv, kv_raw, L->attn_kv_a_norm, hd, c->rms_eps);
        llm_dsv4_rope_tail_layer_inplace(c, kv, 1, hd, c->n_rot, pos, il, 0);
    }
    free(qr); free(kv_raw);
    return ok;
}

/* Output grouped-LoRA (dopo la RoPE inversa sulle head): heads [n_head*hd] -> out [embd]. */
static inline bool llm_dsv4_gpu_attn_finish(LlmDsv4AttnGpu* m, const LlmDsv4Config* c,
        const LlmDsv4LayerWeights* L, int il, float* heads, int pos, float* out) {
    if (!llm_dsv4_gpu_enabled() || !m || !m->inited) return false;
    const int ng = c->n_out_group, rank = c->n_lora_o, embd = c->n_embd;
    const int gd = c->n_head_dim * (c->n_head / ng), ol = ng * rank;
    llm_dsv4_rope_tail_layer_inplace(c, heads, c->n_head, c->n_head_dim, c->n_rot, pos, il, 1);
    float* low = (float*)malloc(sizeof(float) * (size_t)ol);
    bool ok = low != NULL;
    if (ok) {
        LlmDsv4GpuJob jobs[64];
        int nj = 0;
        for (int g = 0; g < ng && nj < 64; g++) {
            jobs[nj++] = (LlmDsv4GpuJob){ m->outa[il].w, (size_t)(g * rank) * m->outa[il].row_bytes,
                                          m->outa[il].type, m->outa[il].row_bytes,
                                          (uint32_t)gd, (uint32_t)rank, heads + (long)g * gd, low + g * rank };
        }
        ok = llm_dsv4_metal_gemv_batch(m->be, jobs, nj) == 0;
    }
    if (ok) {
        LlmDsv4GpuJob j = { m->outb[il].w, 0, m->outb[il].type,
                            m->outb[il].row_bytes, (uint32_t)ol, (uint32_t)embd, low, out };
        ok = llm_dsv4_metal_gemv_batch(m->be, &j, 1) == 0;
    }
    free(low);
    return ok;
}

#endif /* LLM_DSV4_GPU_ATTN_H */
