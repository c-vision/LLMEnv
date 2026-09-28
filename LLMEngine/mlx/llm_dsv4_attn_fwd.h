/* llm_dsv4_attn_fwd.h — forward attenzione di un layer DS4 (V4 Flash), port fedele del
 * riferimento CPU di ds4.c: Q-LoRA (attn_q_a -> norm -> attn_q_b), KV single-head
 * (attn_kv -> norm), head-RMSNorm, RoPE tail (base/scale per-layer, YaRN sui compressi),
 * sink attention su righe KV (raw+compressa, K=V), output grouped-LoRA (attn_output_a/b).
 * Header-only. Test: llm_dsv4_attn_fwd_test.c. */
#ifndef LLM_DSV4_ATTN_FWD_H
#define LLM_DSV4_ATTN_FWD_H

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "llm_dsv4.h"
#include "llm_dsv4_weights.h"
#include "llm_dsv4_actquant.h"

/* y[o] = sum_i W[o*in+i]*x[i]  (W row-major [out,in]) */
static inline void llm_dsv4_linear(float* y, const float* W, int out, int in, const float* x) {
    for (int o = 0; o < out; o++) {
        const float* wr = W + (long)o * in;
#if defined(__ARM_NEON)
        /* stesso ordine di ds4 dot_f16_row: 2 accumulatori, 8 per iterazione, FMA */
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        for (; i + 8 <= in; i += 8) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(wr + i), vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(wr + i + 4), vld1q_f32(x + i + 4));
        }
        float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < in; i++) acc += wr[i] * x[i];
        y[o] = acc;
#else
        float acc = 0.0f;
        for (int i = 0; i < in; i++) acc += wr[i] * x[i];
        y[o] = acc;
#endif
    }
}

static inline void llm_dsv4_rms_norm_weight(float* out, const float* x, const float* w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static inline void llm_dsv4_head_rms_norm_inplace(float* x, int n_head, int head_dim, float eps) {
    for (int h = 0; h < n_head; h++) {
        float* head = x + (long)h * head_dim;
        double ss = 0.0;
        for (int i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];
        float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (int i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static inline float llm_dsv4_rope_yarn_ramp(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}
static inline float llm_dsv4_rope_yarn_corr_dim(int n_dims, long n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}
static inline void llm_dsv4_rope_yarn_corr_dims(int n_dims, long n_ctx_orig, float freq_base,
                                                float beta_fast, float beta_slow, float dims[2]) {
    float start = floorf(llm_dsv4_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(llm_dsv4_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

static inline void llm_dsv4_rope_tail_ext_inplace(float* x, int n_head, int head_dim, int n_rot,
        int pos, long n_ctx_orig, float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, int inverse) {
    int n_nope = head_dim - n_rot;
    float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float sin_sign = inverse ? -1.0f : 1.0f;
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f)
        llm_dsv4_rope_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    for (int h = 0; h < n_head; h++) {
        float* tail = x + (long)h * head_dim + n_nope;
        float theta_extrap = (float)pos;
        for (int i = 0; i < n_rot; i += 2) {
            float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp, mscale = attn_factor;
            if (ext_factor != 0.0f) {
                float ramp_mix = llm_dsv4_rope_yarn_ramp(corr_dims[0], corr_dims[1], i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            float c = cosf(theta) * mscale, s = sin_sign * sinf(theta) * mscale;
            float x0 = tail[i], x1 = tail[i + 1];
            tail[i]     = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            theta_extrap *= theta_scale;
        }
    }
}

/* RoPE tail per layer: base/scale dipendono dal tipo di layer (compresso o no). */
static inline void llm_dsv4_rope_tail_layer_inplace(const LlmDsv4Config* c, float* x, int n_head,
        int head_dim, int n_rot, int pos, int il, int inverse) {
    int compressed = llm_dsv4_layer_compress_ratio(c, il) != 0;
    float freq_base = (compressed && c->compress_rope_freq_base > 0.0f)
                    ? c->compress_rope_freq_base : c->rope_freq_base;
    float freq_scale = (!compressed || c->rope_scale_factor <= 0.0f) ? 1.0f : 1.0f / c->rope_scale_factor;
    float ext_factor = (compressed && c->rope_scale_factor > 1.0f) ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f)
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    llm_dsv4_rope_tail_ext_inplace(x, n_head, head_dim, n_rot, pos,
        compressed ? c->rope_orig_ctx : 0, freq_base, freq_scale, ext_factor, attn_factor,
        c->rope_yarn_beta_fast, c->rope_yarn_beta_slow, inverse);
}

/* Sink attention su righe KV (single head K=V). out_heads [n_head*head_dim]. */
static inline void llm_dsv4_attention_rows(float* out_heads, const float* q, const float* kv_rows,
        int n_kv, const float* sinks, int n_head, int head_dim) {
    float kq_scale = 1.0f / sqrtf((float)head_dim);
    float* score = (float*)malloc(sizeof(float) * (size_t)n_kv);
    for (int h = 0; h < n_head; h++) {
        const float* qh = q + (long)h * head_dim;
        float max_score = sinks[h];
        for (int r = 0; r < n_kv; r++) {
            const float* kv = kv_rows + (long)r * head_dim;
            score[r] = llm_dsv4_dot_f32(qh, kv, head_dim) * kq_scale;
            if (score[r] > max_score) max_score = score[r];
        }
        float* oh = out_heads + (long)h * head_dim;
        memset(oh, 0, sizeof(float) * (size_t)head_dim);
        float denom = expf(sinks[h] - max_score);
        for (int r = 0; r < n_kv; r++) {
            float w = expf(score[r] - max_score);
            const float* kv = kv_rows + (long)r * head_dim;
            denom += w;
            for (int e = 0; e < head_dim; e++) oh[e] = fmaf(w, kv[e], oh[e]);   /* axpy ds4 (FMA) */
        }
        float inv = 1.0f / denom;
        for (int e = 0; e < head_dim; e++) oh[e] *= inv;
    }
    free(score);
}

/* Output grouped-LoRA: low[g*rank+r] = W_a[(g*rank+r), group_dim] @ heads[g*group_dim..]; out = W_b @ low. */
static inline void llm_dsv4_grouped_out(float* out, const float* heads, const float* Wa,
        const float* Wb, int n_groups, int group_dim, int rank, int n_embd) {
    int out_low = n_groups * rank;
    float* low = (float*)calloc((size_t)out_low, sizeof(float));
    for (int g = 0; g < n_groups; g++) {
        const float* hg = heads + (long)g * group_dim;
        for (int r = 0; r < rank; r++) {
            const float* wr = Wa + ((long)g * rank + r) * group_dim;
            float acc = 0.0f;
            for (int i = 0; i < group_dim; i++) acc += wr[i] * hg[i];
            low[g * rank + r] = acc;
        }
    }
    for (int o = 0; o < n_embd; o++) {
        const float* wr = Wb + (long)o * out_low;
        float acc = 0.0f;
        for (int i = 0; i < out_low; i++) acc += wr[i] * low[i];
        out[o] = acc;
    }
    free(low);
}

/* Proiezioni Q-LoRA + KV (input già normalizzato). Output: q [n_head*head_dim] e kv [head_dim]
 * già ropate, e qr_norm [n_lora_q] (serve all'indexer). */
static inline void llm_dsv4_attn_project(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, const float* xn, int pos, int il,
        float* q, float* kv, float* qrn) {
    int q_dim = c->n_head * c->n_head_dim, embd = c->n_embd, lq = c->n_lora_q;
    float* qr = (float*)malloc(sizeof(float) * (size_t)lq);
    float* kv_raw = (float*)malloc(sizeof(float) * (size_t)c->n_head_dim);
    if (pe && pe->q_a && pe->q_b && pe->kv) {
        /* path Q8_0 prequant (parità ds4) */
        int8_t* xq = (int8_t*)malloc((size_t)embd);
        float* xs = (float*)malloc(sizeof(float) * (size_t)((embd + 31) / 32));
        llm_dsv4_q8_0_quantize_row(xn, embd, xq, xs);
        int64_t nba = llm_ggml_nbytes(pe->q_a_type, embd), nbk = llm_ggml_nbytes(pe->kv_type, embd);
        for (int o = 0; o < lq; o++) qr[o] = llm_dsv4_dot_q8_0_prequant(pe->q_a + (int64_t)o * nba, embd, xq, xs);
        for (int o = 0; o < c->n_head_dim; o++) kv_raw[o] = llm_dsv4_dot_q8_0_prequant(pe->kv + (int64_t)o * nbk, embd, xq, xs);
        free(xs); free(xq);
        llm_dsv4_rms_norm_weight(qrn, qr, L->attn_q_a_norm, lq, c->rms_eps);
        int8_t* qq = (int8_t*)malloc((size_t)lq);
        float* qs = (float*)malloc(sizeof(float) * (size_t)((lq + 31) / 32));
        llm_dsv4_q8_0_quantize_row(qrn, lq, qq, qs);
        int64_t nbb = llm_ggml_nbytes(pe->q_b_type, lq);
        for (int o = 0; o < q_dim; o++) q[o] = llm_dsv4_dot_q8_0_prequant(pe->q_b + (int64_t)o * nbb, lq, qq, qs);
        free(qs); free(qq);
    } else {
        llm_dsv4_linear(qr, L->attn_q_a, lq, embd, xn);
        llm_dsv4_rms_norm_weight(qrn, qr, L->attn_q_a_norm, lq, c->rms_eps);
        llm_dsv4_linear(q, L->attn_q_b, q_dim, lq, qrn);
        llm_dsv4_linear(kv_raw, L->attn_kv, c->n_head_dim, embd, xn);
    }
    llm_dsv4_head_rms_norm_inplace(q, c->n_head, c->n_head_dim, c->rms_eps);
    llm_dsv4_rope_tail_layer_inplace(c, q, c->n_head, c->n_head_dim, c->n_rot, pos, il, 0);
    { const char* dq = getenv("LLM_DSV4_DUMP_Q");
      if (dq && dq[0] && il == 0) { FILE* f = fopen(dq, "wb"); if (f) { fwrite(q, sizeof(float), (size_t)q_dim, f); fclose(f); } } }
    llm_dsv4_rms_norm_weight(kv, kv_raw, L->attn_kv_a_norm, c->n_head_dim, c->rms_eps);
    llm_dsv4_rope_tail_layer_inplace(c, kv, 1, c->n_head_dim, c->n_rot, pos, il, 0);
    { const char* dk = getenv("LLM_DSV4_DUMP_KV");
      if (dk && dk[0] && il == 0) { FILE* f = fopen(dk, "wb"); if (f) { fwrite(kv, sizeof(float), (size_t)c->n_head_dim, f); fclose(f); } } }
    free(kv_raw); free(qr);
}

/* Applica la RoPE inversa alle head e la proiezione grouped (come ds4: inverse rope su heads). */
static inline void llm_dsv4_attn_finish(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, float* heads, int pos, int il, float* out) {
    int ng = c->n_out_group, rank = c->n_lora_o, embd = c->n_embd;
    int group_dim = c->n_head_dim * (c->n_head / ng), out_low = ng * rank;
    llm_dsv4_rope_tail_layer_inplace(c, heads, c->n_head, c->n_head_dim, c->n_rot, pos, il, 1);
    { const char* dh = getenv("LLM_DSV4_DUMP_HEADS");
      if (dh && dh[0] && il == 0) { FILE* f = fopen(dh, "wb"); if (f) { fwrite(heads, sizeof(float), (size_t)c->n_head * c->n_head_dim, f); fclose(f); } } }
    if (pe && pe->out_a && pe->out_b) {
        float* low = (float*)malloc(sizeof(float) * (size_t)out_low);
        int64_t nba = llm_ggml_nbytes(pe->out_a_type, group_dim);
        for (int g = 0; g < ng; g++) {
            int8_t* hq = (int8_t*)malloc((size_t)group_dim);
            float* hs = (float*)malloc(sizeof(float) * (size_t)((group_dim + 31) / 32));
            llm_dsv4_q8_0_quantize_row(heads + (long)g * group_dim, group_dim, hq, hs);
            for (int r = 0; r < rank; r++)
                low[g * rank + r] = llm_dsv4_dot_q8_0_prequant(pe->out_a + (int64_t)(g * rank + r) * nba, group_dim, hq, hs);
            free(hs); free(hq);
        }
        int8_t* lq = (int8_t*)malloc((size_t)out_low);
        float* ls = (float*)malloc(sizeof(float) * (size_t)((out_low + 31) / 32));
        llm_dsv4_q8_0_quantize_row(low, out_low, lq, ls);
        int64_t nbb = llm_ggml_nbytes(pe->out_b_type, out_low);
        for (int o = 0; o < embd; o++) out[o] = llm_dsv4_dot_q8_0_prequant(pe->out_b + (int64_t)o * nbb, out_low, lq, ls);
        free(ls); free(lq); free(low);
    } else {
        llm_dsv4_grouped_out(out, heads, L->attn_output_a, L->attn_output_b, ng, group_dim, rank, embd);
    }
}

/* Forward attenzione a partire dall'input GIÀ normalizzato (attn_norm applicato dal chiamante,
 * es. llm_dsv4_block_pre). `kv` = righe KV (raw+compresse, già ropate); se NULL usa la kv corrente. */
static inline void llm_dsv4_attn_from_normed(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* xn, int pos, int il, const float* kv, int n_kv, float* out) {
    int q_dim = c->n_head * c->n_head_dim;
    float* q = (float*)malloc(sizeof(float) * (size_t)q_dim);
    float* kvn = (float*)malloc(sizeof(float) * (size_t)c->n_head_dim);
    float* qrn = (float*)malloc(sizeof(float) * (size_t)c->n_lora_q);
    float* heads = (float*)malloc(sizeof(float) * (size_t)q_dim);

    llm_dsv4_attn_project(c, L, NULL, xn, pos, il, q, kvn, qrn);
    const float* rows = kv; int nr = n_kv;
    if (!rows) { rows = kvn; nr = 1; }
    llm_dsv4_attention_rows(heads, q, rows, nr, L->attn_sinks, c->n_head, c->n_head_dim);
    llm_dsv4_attn_finish(c, L, NULL, heads, pos, il, out);

    free(heads); free(qrn); free(kvn); free(q);
}

/* Convenienza: applica attn_norm a x e poi l'attenzione. */
static inline void llm_dsv4_attn_layer_forward(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* x, int pos, int il, const float* kv, int n_kv, float* out) {
    float* xn = (float*)malloc(sizeof(float) * (size_t)c->n_embd);
    llm_dsv4_rms_norm_weight(xn, x, L->attn_norm, c->n_embd, c->rms_eps);
    llm_dsv4_attn_from_normed(c, L, xn, pos, il, kv, n_kv, out);
    free(xn);
}

#endif /* LLM_DSV4_ATTN_FWD_H */
