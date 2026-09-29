/* llm_dsv4_cache.h — cache KV DS4 (V4 Flash): raw window ring + righe compresse per-layer
 * (compressor ratio 4/128), attenzione mista raw+compresse, selezione indexer top-k.
 * Port fedele del riferimento CPU ds4.c. Header-only. Test: llm_dsv4_cache_test.c.
 * NOTA: quantizzazione fp8 del KV e QAT del path indexer non ancora applicate (identità). */
#ifndef LLM_DSV4_CACHE_H
#define LLM_DSV4_CACHE_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "llm_dsv4_attn_fwd.h"
#include "llm_dsv4_quant.h"
#include "llm_dsv4_gpu_ctx.h"   /* hook GPU opt-in (F4) */

typedef struct {
    float* raw_kv;        int n_raw, raw_cap;       /* [raw_cap, head_dim] */
    float* attn_comp;     int n_comp;               /* [comp_cap, head_dim] */
    float* index_comp;    int n_index_comp;         /* [comp_cap, indexer_head_dim] */
    float* attn_state_kv; float* attn_state_score;  /* [2*coff*head_dim*ratio] */
    float* index_state_kv; float* index_state_score;
    int compress_ratio, comp_cap, head_dim, indexer_head_dim;
} LlmDsv4LayerCache;

static inline void llm_dsv4_cache_init(LlmDsv4LayerCache* k, const LlmDsv4Config* c, int il, int raw_cap, int comp_cap) {
    memset(k, 0, sizeof *k);
    k->compress_ratio = llm_dsv4_layer_compress_ratio(c, il);
    k->comp_cap = comp_cap; k->head_dim = c->n_head_dim; k->indexer_head_dim = c->n_indexer_head_dim;
    int co = (k->compress_ratio == 4) ? 2 : 1;
    k->raw_cap = raw_cap;
    k->raw_kv = (float*)calloc((size_t)raw_cap * c->n_head_dim, sizeof(float));
    if (k->compress_ratio) {
        k->attn_comp = (float*)calloc((size_t)comp_cap * c->n_head_dim, sizeof(float));
        int ssz = co * k->compress_ratio * (co * c->n_head_dim);
        k->attn_state_kv = (float*)calloc((size_t)ssz, sizeof(float));
        k->attn_state_score = (float*)malloc((size_t)ssz * sizeof(float));
        for (int i = 0; i < ssz; i++) k->attn_state_score[i] = -1.0e30f;
        if (k->compress_ratio == 4) {
            int isz = co * k->compress_ratio * (co * c->n_indexer_head_dim);
            k->index_comp = (float*)calloc((size_t)comp_cap * c->n_indexer_head_dim, sizeof(float));
            k->index_state_kv = (float*)calloc((size_t)isz, sizeof(float));
            k->index_state_score = (float*)malloc((size_t)isz * sizeof(float));
            for (int i = 0; i < isz; i++) k->index_state_score[i] = -1.0e30f;
        }
    }
}
static inline void llm_dsv4_cache_free(LlmDsv4LayerCache* k) {
    free(k->raw_kv); free(k->attn_comp); free(k->index_comp);
    free(k->attn_state_kv); free(k->attn_state_score);
    free(k->index_state_kv); free(k->index_state_score);
    memset(k, 0, sizeof *k);
}

static inline void llm_dsv4_cache_push_raw(LlmDsv4LayerCache* k, const float* kv) {
    /* ds4 kv_cache_push_raw: la riga KV e' arrotondata a f16 quando entra in cache. */
    float* dst;
    if (k->n_raw < k->raw_cap) { dst = k->raw_kv + (long)k->n_raw * k->head_dim; k->n_raw++; }
    else { memmove(k->raw_kv, k->raw_kv + k->head_dim, sizeof(float) * (size_t)k->head_dim * (k->raw_cap - 1)); dst = k->raw_kv + (long)(k->raw_cap - 1) * k->head_dim; }
    for (int i = 0; i < k->head_dim; i++) dst[i] = llm_ggml_f16_round(kv[i]);
}
static inline void llm_dsv4_cache_push_comp(float* rows, int* n_rows, int cap, int dim, const float* kv) {
    if (*n_rows >= cap) return;
    /* ds4 kv_cache_push_comp: arrotondamento a f16. */
    float* dst = rows + (long)(*n_rows) * dim;
    for (int i = 0; i < dim; i++) dst[i] = llm_ggml_f16_round(kv[i]);
    (*n_rows)++;
}

/* Pool dello stato compressor (port di compressor_pool_decode_state). */
static inline void llm_dsv4_compressor_pool_state(float* out, const float* state_kv, const float* state_score,
                                                  int head_dim, int ratio) {
    int co = (ratio == 4) ? 2 : 1, width = co * head_dim;
    for (int j = 0; j < head_dim; j++) {
        float mx = -1.0e30f;
        if (ratio == 4) {
            for (int r = 0; r < ratio; r++) {
                float sp = state_score[(long)r * width + j];
                float sc = state_score[(long)(ratio + r) * width + head_dim + j];
                if (sp > mx) mx = sp; if (sc > mx) mx = sc;
            }
        } else for (int r = 0; r < ratio; r++) { float s = state_score[(long)r * width + j]; if (s > mx) mx = s; }
        if (mx <= -1.0e30f * 0.5f) { out[j] = 0.0f; continue; }
        float den = 0.0f, sum = 0.0f;
        if (ratio == 4) {
            for (int r = 0; r < ratio; r++) {
                float wp = expf(state_score[(long)r * width + j] - mx);
                float wc = expf(state_score[(long)(ratio + r) * width + head_dim + j] - mx);
                den += wp + wc;
                sum += wp * state_kv[(long)r * width + j];
                sum += wc * state_kv[(long)(ratio + r) * width + head_dim + j];
            }
        } else {
            for (int r = 0; r < ratio; r++) {
                float w = expf(state_score[(long)r * width + j] - mx);
                den += w; sum += w * state_kv[(long)r * width + j];
            }
        }
        out[j] = sum / den;
    }
}

/* Compressor decode: ritorna true se ha prodotto una riga compressa (out_comp). */
static inline int llm_dsv4_compressor_decode(const LlmDsv4Config* c, float* out_comp,
        const float* wkv, const float* wgate, const float* ape, const float* norm_w,
        const float* x, float* state_kv, float* state_score,
        int embd, int head_dim, int ratio, int il, int pos) {
    int co = (ratio == 4) ? 2 : 1, width = co * head_dim;
    int pos_mod = pos % ratio;
    int row = (ratio == 4) ? ratio + pos_mod : pos_mod;
    int should = ((pos + 1) % ratio) == 0;
    float* kv_cur = (float*)malloc(sizeof(float) * (size_t)width);
    float* sc_cur = (float*)malloc(sizeof(float) * (size_t)width);
    llm_dsv4_linear(kv_cur, wkv, width, embd, x);
    llm_dsv4_linear(sc_cur, wgate, width, embd, x);
    for (int j = 0; j < width; j++) sc_cur[j] += ape[(long)pos_mod * width + j];
    memcpy(state_kv + (long)row * width, kv_cur, sizeof(float) * (size_t)width);
    memcpy(state_score + (long)row * width, sc_cur, sizeof(float) * (size_t)width);
    free(sc_cur); free(kv_cur);
    if (!should) return 0;
    float* pooled = (float*)malloc(sizeof(float) * (size_t)head_dim);
    { const char* dx = getenv("LLM_DSV4_DUMP_CX"); if (dx && dx[0] && il == 2 && head_dim == c->n_head_dim) { FILE* f = fopen(dx, "ab"); if (f) { unsigned p2 = (unsigned)pos; fwrite(&p2, sizeof(unsigned), 1, f); fwrite(x, sizeof(float), (size_t)embd, f); fclose(f); } } }
    { const char* st = getenv("LLM_DSV4_DUMP_STATE"); if (st && st[0] && il == 2 && head_dim == c->n_head_dim) { int w = co * head_dim; FILE* f = fopen(st, "wb"); if (f) { fwrite(state_kv, sizeof(float), (size_t)co*ratio*w, f); fwrite(state_score, sizeof(float), (size_t)co*ratio*w, f); fclose(f); } } }
    llm_dsv4_compressor_pool_state(pooled, state_kv, state_score, head_dim, ratio);
    { const char* dp = getenv("LLM_DSV4_DUMP_POOL"); if (dp && dp[0] && il == 2 && head_dim == c->n_head_dim) { FILE* f = fopen(dp, "wb"); if (f) { fwrite(pooled, sizeof(float), (size_t)head_dim, f); fclose(f); } } }
    double ss = 0.0;
    for (int i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + c->rms_eps);
    for (int i = 0; i < head_dim; i++) out_comp[i] = pooled[i] * rms * norm_w[i];
    int comp_pos = pos + 1 - ratio;
    llm_dsv4_rope_tail_layer_inplace(c, out_comp, 1, head_dim, c->n_rot, comp_pos, il, 0);
    if (head_dim == c->n_head_dim) llm_dsv4_fp8_kv_quantize_row(out_comp, head_dim, c->n_rot);
    else llm_dsv4_indexer_qat_row(out_comp, head_dim);
    if (ratio == 4) {
        for (int r = 0; r < ratio; r++) {
            memcpy(state_kv + (long)r * width, state_kv + (long)(ratio + r) * width, sizeof(float) * (size_t)width);
            memcpy(state_score + (long)r * width, state_score + (long)(ratio + r) * width, sizeof(float) * (size_t)width);
        }
    }
    free(pooled);
    return 1;
}

/* Attenzione mista su raw + compresse (comp_allowed puo' essere NULL). */
static inline void llm_dsv4_attention_mixed(float* out_heads, const LlmDsv4Config* c, const float* q,
        const float* raw_kv, int n_raw, const float* comp_kv, int n_comp, const int* comp_allowed,
        const float* sinks, int n_head, int head_dim) {
    float kq = 1.0f / sqrtf((float)head_dim);
    int n_tot = n_raw + n_comp;
    float* score = (float*)malloc(sizeof(float) * (size_t)(n_tot > 0 ? n_tot : 1));
    for (int h = 0; h < n_head; h++) {
        const float* qh = q + (long)h * head_dim;
        float mx = sinks[h]; int idx = 0;
        for (int r = 0; r < n_raw; r++, idx++) {
            const float* kv = raw_kv + (long)r * head_dim;
            score[idx] = llm_dsv4_dot_f32(qh, kv, head_dim) * kq; if (score[idx] > mx) mx = score[idx];
        }
        for (int r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) { score[idx] = -1.0e30f; continue; }
            const float* kv = comp_kv + (long)r * head_dim;
            score[idx] = llm_dsv4_dot_f32(qh, kv, head_dim) * kq; if (score[idx] > mx) mx = score[idx];
        }
        float* oh = out_heads + (long)h * head_dim;
        memset(oh, 0, sizeof(float) * (size_t)head_dim);
        float den = expf(sinks[h] - mx); idx = 0;
        for (int r = 0; r < n_raw; r++, idx++) {
            float w = expf(score[idx] - mx); const float* kv = raw_kv + (long)r * head_dim;
            den += w; for (int e = 0; e < head_dim; e++) oh[e] = fmaf(w, kv[e], oh[e]);
        }
        for (int r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= -1.0e30f * 0.5f) continue;
            float w = expf(score[idx] - mx); const float* kv = comp_kv + (long)r * head_dim;
            den += w; for (int e = 0; e < head_dim; e++) oh[e] = fmaf(w, kv[e], oh[e]);
        }
        float inv = 1.0f / den;
        for (int e = 0; e < head_dim; e++) oh[e] *= inv;
    }
    free(score);
}

/* Indexer: allowed[c] per le righe compresse (top-k per score). Ritorna array [n_comp] (caller free). */
static inline int* llm_dsv4_indexer_allowed(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const float* cur, const float* qr_norm, const float* index_comp, int n_comp, int il, int pos) {
    if (n_comp == 0) return NULL;
    int* allowed = (int*)calloc((size_t)n_comp, sizeof(int));
    int top_k = c->n_indexer_top_k < n_comp ? c->n_indexer_top_k : n_comp;
    if (top_k == n_comp) { for (int i = 0; i < n_comp; i++) allowed[i] = 1; return allowed; }
    int hd = c->n_indexer_head_dim, nh = c->n_indexer_head;
    int qdim = hd * nh;
    float* q = (float*)malloc(sizeof(float) * (size_t)qdim);
    float* weights = (float*)malloc(sizeof(float) * (size_t)nh);
    float* scores = (float*)malloc(sizeof(float) * (size_t)n_comp);
    llm_dsv4_linear(q, L->idx_q_b, qdim, c->n_lora_q, qr_norm);
    llm_dsv4_rope_tail_layer_inplace(c, q, nh, hd, c->n_rot, pos, il, 0);
    llm_dsv4_indexer_qat_rows(q, nh, hd);
    llm_dsv4_linear(weights, L->idx_proj, nh, c->n_embd, cur);
    float scale = 1.0f / sqrtf((float)(hd * nh));
    for (int h = 0; h < nh; h++) weights[h] *= scale;
    for (int cc = 0; cc < n_comp; cc++) {
        const float* kv = index_comp + (long)cc * hd; float s = 0.0f;
        for (int h = 0; h < nh; h++) {
            const float* qh = q + (long)h * hd;
            float dot = llm_dsv4_dot_f32(qh, kv, hd); if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[cc] = s;
    }
    /* Top-k deterministico: score decrescente, a pari score indice crescente (ds4 #1060).
     * Il confronto `>` stretto con scansione crescente seleziona il PRIMO massimo -> index asc. */
    for (int k = 0; k < top_k; k++) {
        int best = 0; float bs = -1.0e30f;
        for (int cc = 0; cc < n_comp; cc++) if (!allowed[cc] && scores[cc] > bs) { best = cc; bs = scores[cc]; }
        allowed[best] = 1;
    }
    free(scores); free(weights); free(q);
    return allowed;
}

/* Attenzione di un layer con cache: proiezioni, push raw, compressor (attn+indexer), indexer
 * top-k, attenzione mista, RoPE inversa + grouped out. `xn` = input già normalizzato (attn_norm). */
/* Proiezioni + mid (compressor/indexer/attention) -> `heads` (n_head*hd), SENZA il finish. */
static inline int llm_dsv4_attn_cached_mid(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* k, const float* xn, int pos, int il, float* heads) {
    int hd = c->n_head_dim, lq = c->n_lora_q, q_dim = c->n_head * c->n_head_dim;
    (void)q_dim;
    float* q = (float*)malloc(sizeof(float) * (size_t)(c->n_head * hd));
    float* kv = (float*)malloc(sizeof(float) * (size_t)hd);
    float* qrn = (float*)malloc(sizeof(float) * (size_t)lq);
    if (!q || !kv || !qrn) { free(q); free(kv); free(qrn); return 0; }
    if (!llm_dsv4_gpu_attn_project_try(c, L, il, xn, pos, q, kv, qrn))
        llm_dsv4_attn_project(c, L, pe, xn, pos, il, q, kv, qrn);
    llm_dsv4_fp8_kv_quantize_row(kv, c->n_head_dim, c->n_rot);
    llm_dsv4_cache_push_raw(k, kv);
    int ratio = k->compress_ratio;
    int* allowed = NULL;
    if (ratio) {
        float* comp = (float*)malloc(sizeof(float) * (size_t)hd);
        if (llm_dsv4_compressor_decode(c, comp, L->comp_kv, L->comp_gate, L->comp_ape, L->comp_norm,
                xn, k->attn_state_kv, k->attn_state_score, c->n_embd, hd, ratio, il, pos))
            llm_dsv4_cache_push_comp(k->attn_comp, &k->n_comp, k->comp_cap, hd, comp);
        free(comp);
        if (ratio == 4) {
            int ihd = c->n_indexer_head_dim;
            float* ic = (float*)malloc(sizeof(float) * (size_t)ihd);
            if (llm_dsv4_compressor_decode(c, ic, L->idx_c_kv, L->idx_c_gate, L->idx_c_ape, L->idx_c_norm,
                    xn, k->index_state_kv, k->index_state_score, c->n_embd, ihd, ratio, il, pos))
                llm_dsv4_cache_push_comp(k->index_comp, &k->n_index_comp, k->comp_cap, ihd, ic);
            free(ic);
            allowed = llm_dsv4_indexer_allowed(c, L, xn, qrn, k->index_comp, k->n_index_comp, il, pos);
        }
        llm_dsv4_attention_mixed(heads, c, q, k->raw_kv, k->n_raw, k->attn_comp, k->n_comp, allowed,
                                 L->attn_sinks, c->n_head, hd);
    } else {
        llm_dsv4_attention_rows(heads, q, k->raw_kv, k->n_raw, L->attn_sinks, c->n_head, hd);
    }
    free(allowed); free(qrn); free(kv); free(q);
    return 1;
}

static inline void llm_dsv4_attn_cached(const LlmDsv4Config* c, const LlmDsv4LayerWeights* L,
        const LlmDsv4PackedExperts* pe, LlmDsv4LayerCache* k, const float* xn, int pos, int il, float* out) {
    int q_dim = c->n_head * c->n_head_dim;
    float* heads = (float*)malloc(sizeof(float) * (size_t)q_dim);
    if (!heads) return;
    if (llm_dsv4_attn_cached_mid(c, L, pe, k, xn, pos, il, heads)) {
        if (!llm_dsv4_gpu_attn_finish_try(c, L, il, heads, pos, out))
            llm_dsv4_attn_finish(c, L, pe, heads, pos, il, out);
    }
    free(heads);
}

#endif /* LLM_DSV4_CACHE_H */
