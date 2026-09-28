/* llm_mlx_adamw.h — §D R3.4: AdamW persistente (host) con weight decay, gradient clipping,
 * learning-rate scheduler (warmup lineare + cosine decay) e gradient accumulation.
 *
 * PURO (nessuna dipendenza MLX): opera su array float host. I gradienti arrivano da
 * `llm_ai_mlx_train_lora_loss_grad` (§D R3.3, autograd MLX). Stato persistente (m/v, step t)
 * cosi' un training loop puo' fare N micro-batch e applicare un update ogni `accum`.
 *
 * Update AdamW (Loshchilov & Hutter): m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2;
 * mh = m/(1-b1^t), vh = v/(1-b2^t); p -= lr (mh/(sqrt(vh)+eps) + wd p). */
#ifndef LLM_MLX_ADAMW_H
#define LLM_MLX_ADAMW_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LLM_ADAMW_PI 3.14159265358979323846f

typedef struct {
    int    n;               /* numero di parametri (flat) */
    float* m;               /* primo momento */
    float* v;               /* secondo momento */
    float* gacc;            /* accumulatore dei micro-gradienti */
    long   t;               /* step GIA' completati (per bias correction e scheduler) */
    int    accum;           /* micro-batch per update (>=1) */
    int    accum_count;     /* micro-batch accumulati finora */
    float  lr, beta1, beta2, eps, wd;
    float  warmup;          /* step di warmup lineare (0 = nessuno) */
    float  total;           /* step totali per il cosine decay (0 = lr costante) */
    float  clip_norm;       /* norma massima del gradiente (0 = off) */
    long   steps;           /* numero di update applicati (diagnostica) */
    float  last_lr;         /* ultimo lr usato */
    float  last_grad_norm;  /* norma (pre-clip) del gradiente medio */
    float  last_clip_scale; /* fattore di clipping applicato (1 = nessuno) */
} LlmAdamW;

static inline int llm_adamw_init(LlmAdamW* a, int n, float lr, float beta1, float beta2, float eps, float wd,
                                 int accum, float warmup, float total, float clip_norm) {
    if (!a || n <= 0) return -1;
    memset(a, 0, sizeof(*a));
    a->n = n;
    a->lr = lr; a->beta1 = beta1; a->beta2 = beta2; a->eps = eps; a->wd = wd;
    a->accum = accum > 0 ? accum : 1;
    a->warmup = warmup; a->total = total; a->clip_norm = clip_norm;
    a->m = (float*)calloc((size_t)n, sizeof(float));
    a->v = (float*)calloc((size_t)n, sizeof(float));
    a->gacc = (float*)calloc((size_t)n, sizeof(float));
    if (!a->m || !a->v || !a->gacc) { free(a->m); free(a->v); free(a->gacc); memset(a, 0, sizeof(*a)); return -1; }
    return 0;
}
static inline void llm_adamw_free(LlmAdamW* a) {
    if (!a) return;
    free(a->m); free(a->v); free(a->gacc);
    a->m = a->v = a->gacc = NULL;
}

/* LR per lo step `t` (1-based): warmup lineare poi cosine decay fino a 0 a `total`. */
static inline float llm_adamw_lr_at(const LlmAdamW* a, long t) {
    if (a->warmup > 0 && (float)t <= a->warmup) return a->lr * ((float)t / a->warmup);
    if (a->total > 0 && (float)t >= a->total) return 0.0f;
    if (a->total > a->warmup) {
        float p = ((float)t - a->warmup) / (a->total - a->warmup);
        return a->lr * 0.5f * (1.0f + cosf(LLM_ADAMW_PI * p));
    }
    return a->lr;   /* costante */
}

static inline void llm_adamw_zero_grad(LlmAdamW* a) { if (a && a->gacc) memset(a->gacc, 0, (size_t)a->n * sizeof(float)); }
static inline void llm_adamw_accumulate(LlmAdamW* a, const float* g) {
    if (!a || !g) return;
    for (int i = 0; i < a->n; i++) a->gacc[i] += g[i];
    a->accum_count++;
}
static inline int llm_adamw_ready(const LlmAdamW* a) { return a && a->accum_count >= a->accum; }

/* Applica l'update SE accum e' completo (media dei micro-grad + clip + AdamW), azzera l'accum.
 * Ritorna l'LR usato (>0) se ha aggiornato, 0 se non pronto. */
static inline float llm_adamw_step(LlmAdamW* a, float* params) {
    if (!a || !params || a->accum_count < a->accum) return 0.0f;
    const float inv = 1.0f / (float)a->accum_count;
    float scale = 1.0f;
    { double ns = 0.0; for (int i = 0; i < a->n; i++) { double g = (double)a->gacc[i] * inv; ns += g * g; }
      a->last_grad_norm = (float)sqrt(ns); }
    if (a->clip_norm > 0.0f && a->last_grad_norm > a->clip_norm) scale = a->clip_norm / a->last_grad_norm;
    a->last_clip_scale = scale;
    a->t++;
    const float lr = llm_adamw_lr_at(a, a->t);
    const float b1t = powf(a->beta1, (float)a->t), b2t = powf(a->beta2, (float)a->t);
    for (int i = 0; i < a->n; i++) {
        float g = a->gacc[i] * inv * scale;
        float m = a->beta1 * a->m[i] + (1.0f - a->beta1) * g;
        float v = a->beta2 * a->v[i] + (1.0f - a->beta2) * g * g;
        a->m[i] = m; a->v[i] = v;
        float mh = m / (1.0f - b1t), vh = v / (1.0f - b2t);
        params[i] -= lr * (mh / (sqrtf(vh) + a->eps) + a->wd * params[i]);
    }
    llm_adamw_zero_grad(a);
    a->accum_count = 0;
    a->steps++;
    a->last_lr = lr;
    return lr;
}

#endif /* LLM_MLX_ADAMW_H */
