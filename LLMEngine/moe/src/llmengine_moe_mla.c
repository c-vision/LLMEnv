/* llmengine_moe_mla.c — attenzione MLA (DeepSeek-V2/V3) NON assorbita. Vedi include/llmengine_moe_mla.h.
 * Riusa llmengine_moe_rmsnorm/llmengine_moe_matvec dal forward (stessa libreria). */
#include "llmengine_moe_mla.h"
#include "llmengine_moe_qwen3_forward.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double mla_corr_dim(int dims, double orig, double nrot, double base) {
    return (double)dims * log(orig / (nrot * 2.0 * 3.14159265358979323846)) / (2.0 * log(base));
}
static float mla_mscale(float factor, float mscale) {
    if (factor <= 1.0f) return 1.0f;
    return 0.1f * mscale * logf(factor) + 1.0f;
}

/* inv_freq[i] (i in [0, rope_dim/2)) — YaRN se factor>1, altrimenti base^(-2i/dim). */
static void mla_inv_freqs(const MlaCfg* c, float* inv, uint32_t n) {
    int dims = (int)c->qk_rope;
    for (uint32_t i = 0; i < n; ++i) {
        double extra = pow((double)c->rope_theta, (2.0 * (double)i) / (double)dims);
        double v;
        if (c->yarn_factor > 1.0f) {
            double factor = (double)c->yarn_factor;
            double low_d  = floor(mla_corr_dim(dims, c->yarn_orig_max, c->yarn_beta_fast, c->rope_theta));
            double high_d = ceil(mla_corr_dim(dims, c->yarn_orig_max, c->yarn_beta_slow, c->rope_theta));
            if (low_d < 0) low_d = 0;
            if (high_d > dims - 1) high_d = dims - 1;
            double denom = high_d - low_d; if (denom == 0.0) denom = 0.001;
            double ramp = ((double)i - low_d) / denom;
            if (ramp < 0.0) ramp = 0.0; if (ramp > 1.0) ramp = 1.0;
            double mask = 1.0 - ramp;
            double inter = factor * extra;
            double d = inter * mask + extra * (1.0 - mask);
            v = (d != 0.0) ? d / (inter * extra) : (1.0 / extra);   /* inv_freq = 1/yarn_freqs */
        } else {
            v = 1.0 / extra;
        }
        inv[i] = (float)v;
    }
}

/* RoPE su [n_heads*rope_dim] con mscale; variant 0 = half-split, 1 = interleaved. */
static void mla_rope(float* v, uint32_t n_heads, uint32_t rope_dim, uint32_t pos,
                     const float* inv, float mscale, uint32_t variant) {
    uint32_t half = rope_dim / 2;
    for (uint32_t h = 0; h < n_heads; ++h) {
        float* x = v + (size_t)h * rope_dim;
        for (uint32_t i = 0; i < half; ++i) {
            float ang = (float)pos * inv[i];
            float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
            if (variant == 1) {
                float x1 = x[2 * i], x2 = x[2 * i + 1];
                x[2 * i]     = x1 * c - x2 * s;
                x[2 * i + 1] = x2 * c + x1 * s;
            } else {
                float x1 = x[i], x2 = x[i + half];
                x[i]        = x1 * c - x2 * s;
                x[i + half] = x2 * c + x1 * s;
            }
        }
    }
}

LlmEngineMoeStatus llmengine_moe_mla_decode(const MlaCfg* cfg, const MlaLayer* ly, uint32_t layer, uint32_t pos,
                                const float* x, float* out, MlaKv* kv) {
    if (cfg == NULL || ly == NULL || x == NULL || out == NULL || kv == NULL) return LLMENGINE_MOE_ERR_INVALID;
    uint32_t H = cfg->n_heads, qn = cfg->qk_nope, qr = cfg->qk_rope, vhd = cfg->v_head;
    uint32_t kvl = cfg->kv_lora, hidden = cfg->hidden, qh = qn + qr;
    if (layer >= kv->n_layers || pos >= kv->ctx) return LLMENGINE_MOE_ERR_INVALID;

    float* xn  = (float*)malloc((size_t)hidden * sizeof(float));
    float* q   = (float*)malloc((size_t)H * qh * sizeof(float));
    float* ckv = (float*)malloc((size_t)(kvl + qr) * sizeof(float));
    float* ckn = (float*)malloc((size_t)kvl * sizeof(float));
    float* kvb = (float*)malloc((size_t)H * (qn + vhd) * sizeof(float));
    float* ctx = (float*)malloc((size_t)H * vhd * sizeof(float));
    float* scores = (float*)malloc((size_t)(pos + 1) * sizeof(float));
    float inv[256];
    if (!xn || !q || !ckv || !ckn || !kvb || !ctx || !scores || (qr/2) > 256) {
        free(xn); free(q); free(ckv); free(ckn); free(kvb); free(ctx); free(scores);
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    llmengine_moe_rmsnorm(x, ly->attn_norm, cfg->eps, hidden, xn);

    if (cfg->q_lora > 0 && ly->wq_a != NULL) {
        float* qa  = (float*)malloc((size_t)cfg->q_lora * sizeof(float));
        float* qan = (float*)malloc((size_t)cfg->q_lora * sizeof(float));
        if (!qa || !qan) { free(qa); free(qan); free(xn); free(q); free(ckv); free(ckn); free(kvb); free(ctx); free(scores); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        llmengine_moe_matvec(ly->wq_a, cfg->q_lora, hidden, xn, qa);
        llmengine_moe_rmsnorm(qa, ly->q_a_norm, cfg->eps, cfg->q_lora, qan);
        llmengine_moe_matvec(ly->wq_b, H * qh, cfg->q_lora, qan, q);
        free(qa); free(qan);
    } else {
        llmengine_moe_matvec(ly->wq, H * qh, hidden, xn, q);
    }
    llmengine_moe_matvec(ly->wkv_a_mqa, kvl + qr, hidden, xn, ckv);
    llmengine_moe_rmsnorm(ckv, ly->kv_a_norm, cfg->eps, kvl, ckn);
    llmengine_moe_matvec(ly->wkv_b, H * (qn + vhd), kvl, ckn, kvb);

    /* RoPE su q_pe (per head) e kpe (1 testa); mscale = mscale/mscale_all_dim (V2-Lite: 1). */
    uint32_t half = qr / 2;
    mla_inv_freqs(cfg, inv, half);
    float num = mla_mscale(cfg->yarn_factor, cfg->yarn_mscale);
    float den = mla_mscale(cfg->yarn_factor, cfg->yarn_mscale_all_dim);
    float msc = (den > 1e-9f) ? num / den : 1.0f;
    for (uint32_t h = 0; h < H; ++h) mla_rope(q + (size_t)h * qh + qn, 1, qr, pos, inv, msc, cfg->rope_variant);
    float* kpe = ckv + kvl;
    mla_rope(kpe, 1, qr, pos, inv, msc, cfg->rope_variant);

    /* scrive keys [H][qn+qr] e values [H][vhd] in cache, poi attention causale. */
    size_t kstride = (size_t)kv->ctx * qh, vstride = (size_t)kv->ctx * vhd;
    for (uint32_t h = 0; h < H; ++h) {
        float* krow = kv->k + ((size_t)layer * H + h) * kstride + (size_t)pos * qh;
        float* vrow = kv->v + ((size_t)layer * H + h) * vstride + (size_t)pos * vhd;
        memcpy(krow, kvb + (size_t)h * (qn + vhd), (size_t)qn * sizeof(float));
        memcpy(krow + qn, kpe, (size_t)qr * sizeof(float));
        memcpy(vrow, kvb + (size_t)h * (qn + vhd) + qn, (size_t)vhd * sizeof(float));
    }
    float inv_scale = 1.0f / sqrtf((float)qh);
    for (uint32_t h = 0; h < H; ++h) {
        const float* qh_vec = q + (size_t)h * qh;              /* queries = [q_nope | q_pe] già concat */
        const float* kbase = kv->k + ((size_t)layer * H + h) * kstride;
        const float* vbase = kv->v + ((size_t)layer * H + h) * vstride;
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* kt = kbase + (size_t)t * qh;
            double a = 0.0;
            for (uint32_t d = 0; d < qh; ++d) a += (double)qh_vec[d] * (double)kt[d];
            scores[t] = (float)a * inv_scale;
        }
        llmengine_moe_softmax(scores, pos + 1);
        float* ch = ctx + (size_t)h * vhd;
        for (uint32_t d = 0; d < vhd; ++d) {
            double a = 0.0;
            for (uint32_t t = 0; t <= pos; ++t) a += (double)scores[t] * (double)vbase[(size_t)t * vhd + d];
            ch[d] = (float)a;
        }
    }
    llmengine_moe_matvec(ly->wo, hidden, H * vhd, ctx, out);

    free(xn); free(q); free(ckv); free(ckn); free(kvb); free(ctx); free(scores);
    return LLMENGINE_MOE_OK;
}
