/* llm_dsv4_block.h — blocco mHC DeepSeek V4 Flash (pure C), riusa la matematica di deepseek_v41
 * (mHC/Sinkhorn/RMSNorm sono identici). Header-only. Test: llm_dsv4_block_test.c. */
#ifndef LLM_DSV4_BLOCK_H
#define LLM_DSV4_BLOCK_H

#include "llm_dsv41_math.h"
#include "llm_dsv4_weights.h"
#include "llm_dsv4_attn_fwd.h"   /* llm_dsv4_linear (ordine dot_f16 di ds4) */

/* Input di un sublayer (attn o ffn): da residual_hc [n_hc*dim] calcola pre/post/comb (mHC) e
 * x = rmsnorm(hc_pre(residual_hc, pre), norm_w). */
/* RMS senza peso con accumulo double (come ds4 rms_norm_no_weight). */
static inline void llm_dsv4b_rms_no_weight(float* out, const float* x, int n, float eps) {
    double ss = 0.0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}
/* RMS con peso, accumulo double (come ds4 rms_norm_weight). */
static inline void llm_dsv4b_rms_weight(float* out, const float* x, const float* w, int n, float eps) {
    double ss = 0.0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* Port VERBATIM di ds4 hc_split_sinkhorn_one (anche 2/(1+expf(-z)), layout comb [src+dst*n]). */
static inline void llm_dsv4b_hc_split_sinkhorn(float* out, const float* mix, const float* scale,
        const float* base, int n_hc, int iters, float eps) {
    const float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
    for (int i = 0; i < n_hc; i++) { float z = mix[i]*pre_scale + base[i]; out[i] = 1.0f/(1.0f+expf(-z)) + eps; }
    for (int i = 0; i < n_hc; i++) { int off = n_hc+i; float z = mix[off]*post_scale + base[off]; out[off] = 2.0f/(1.0f+expf(-z)); }
    float c[16*16];
    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = -INFINITY;
        for (int src = 0; src < n_hc; src++) { int idx = src + dst*n_hc; int off = 2*n_hc+idx; float v = mix[off]*comb_scale + base[off]; c[idx]=v; if(v>row_max)row_max=v; }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) { int idx=src+dst*n_hc; float v=expf(c[idx]-row_max); c[idx]=v; row_sum+=v; }
        float inv = 1.0f/row_sum;
        for (int src = 0; src < n_hc; src++) { int idx=src+dst*n_hc; c[idx]=c[idx]*inv + eps; }
    }
    for (int src = 0; src < n_hc; src++) { float sum=0.0f; for (int dst=0;dst<n_hc;dst++) sum+=c[src+dst*n_hc]; float inv=1.0f/(sum+eps); for (int dst=0;dst<n_hc;dst++) c[src+dst*n_hc]*=inv; }
    for (int iter=1; iter<iters; iter++) {
        for (int dst=0;dst<n_hc;dst++){ float sum=0.0f; for(int src=0;src<n_hc;src++) sum+=c[src+dst*n_hc]; float inv=1.0f/(sum+eps); for(int src=0;src<n_hc;src++) c[src+dst*n_hc]*=inv; }
        for (int src=0;src<n_hc;src++){ float sum=0.0f; for(int dst=0;dst<n_hc;dst++) sum+=c[src+dst*n_hc]; float inv=1.0f/(sum+eps); for(int dst=0;dst<n_hc;dst++) c[src+dst*n_hc]*=inv; }
    }
    for (int i=0;i<n_hc*n_hc;i++) out[2*n_hc+i] = c[i];
}

/* Port ESATTO di ds4 hc_pre_from_state_one: rms_norm_no_weight (double) -> matvec_f16 ->
 * hc_split_sinkhorn -> hc_weighted_sum; poi rms_norm_weight con double. */
static inline void llm_dsv4_block_pre(const LlmDsv4Config* c, const float* fn, const float* scale,
                                      const float* base, const float* norm_w,
                                      const float* residual_hc, float* out_x,
                                      float* post_out, float* comb_out) {
    const int HC = c->n_hc, D = c->n_embd, SIZE = (2 + HC) * HC, HCD = HC * D;
    float* flat = (float*)malloc((size_t)HCD * sizeof(float));
    float* mixes = (float*)malloc((size_t)SIZE * sizeof(float));
    llm_dsv4b_rms_no_weight(flat, residual_hc, HCD, c->rms_eps);
    llm_dsv4_linear(mixes, fn, SIZE, HCD, flat);         /* ordine dot_f16 */
    float* pre = (float*)malloc((size_t)HC * sizeof(float));
    float* split = (float*)malloc((size_t)SIZE * sizeof(float));
    llm_dsv4b_hc_split_sinkhorn(split, mixes, scale, base, HC, c->n_hc_sinkhorn_iter, c->hc_eps);
    for (int i = 0; i < HC; i++) pre[i] = split[i];
    for (int i = 0; i < HC; i++) post_out[i] = split[HC + i];
    for (int i = 0; i < HC * HC; i++) comb_out[i] = split[2 * HC + i];
    free(split);
    float* hin = (float*)malloc((size_t)D * sizeof(float));
    for (int d = 0; d < D; d++) { float a = 0.0f; for (int h = 0; h < HC; h++) a += residual_hc[(long)h * D + d] * pre[h]; hin[d] = a; }
    llm_dsv4b_rms_weight(out_x, hin, norm_w, D, c->rms_eps);
    free(hin); free(pre); free(mixes); free(flat);
}

/* Output di un sublayer: nuovo stato mHC [n_hc*dim]. */
/* Port ESATTO di ds4 hc_post_one: acc = block_out[d]*post[dst] + sum_src comb[dst+src*HC]*res[src]. */
static inline void llm_dsv4_block_post(const LlmDsv4Config* c, const float* block_out,
                                       const float* residual_hc, const float* post, const float* comb,
                                       float* out_hc) {
    const int HC = c->n_hc, D = c->n_embd;
    for (int dst = 0; dst < HC; dst++) {
        for (int d = 0; d < D; d++) {
            float acc = block_out[d] * post[dst];
            for (int src = 0; src < HC; src++)
                acc += comb[dst + src * HC] * residual_hc[(long)src * D + d];
            out_hc[(long)dst * D + d] = acc;
        }
    }
}

/* Embedding -> stato mHC (tutte le stream = vettore embedding). */
static inline void llm_dsv4_hc_from_embedding(const LlmDsv4Config* c, const float* x, float* out_hc) {
    for (int h = 0; h < c->n_hc; h++) memcpy(out_hc + (size_t)h * c->n_embd, x, (size_t)c->n_embd * sizeof(float));
}

#endif /* LLM_DSV4_BLOCK_H */
