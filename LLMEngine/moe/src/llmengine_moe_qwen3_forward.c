/* llmengine_moe_qwen3_forward.c — primitive + decode a singolo token + prefill batched (Qwen3 MoE). */
#include "llmengine_moe_qwen3_forward.h"
#include "llmengine_moe_qwen3.h"   /* llmengine_moe_q3_expert_swiglu (usato dal prefill batched M17) */
#include "llmengine_moe_arch.h"    /* LLMENGINE_MOE_FLAG_* (varianti del router) */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Norm di blocco secondo la variante del descriptor (M7.2): RMSNorm (default) o LayerNorm. */
static void norm_block(const Q3Cfg* cfg, const float* x, const float* w, float* out) {
    if (cfg->norm_kind == LLMENGINE_MOE_NORM_LAYER) llmengine_moe_layernorm(x, w, cfg->eps, cfg->hidden, out);
    else llmengine_moe_rmsnorm(x, w, cfg->eps, cfg->hidden, out);
}

void llmengine_moe_rmsnorm(const float* x, const float* w, float eps, uint32_t n, float* out) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    float r = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

/* LayerNorm (M7.2): media 0, varianza 1, scale opzionale. Bias non gestito. */
void llmengine_moe_layernorm(const float* x, const float* w, float eps, uint32_t n, float* out) {
    double mean = 0.0;
    for (uint32_t i = 0; i < n; ++i) mean += (double)x[i];
    mean /= (double)n;
    double var = 0.0;
    for (uint32_t i = 0; i < n; ++i) { double d = (double)x[i] - mean; var += d * d; }
    var /= (double)n;
    double inv = 1.0 / sqrt(var + (double)eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = (float)(((double)x[i] - mean) * inv) * (w ? w[i] : 1.0f);
}

void llmengine_moe_softmax(float* x, uint32_t n) {
    if (n == 0) return;
    float m = x[0];
    for (uint32_t i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    double s = 0.0;
    for (uint32_t i = 0; i < n; ++i) { x[i] = expf(x[i] - m); s += x[i]; }
    if (s == 0.0) return;
    for (uint32_t i = 0; i < n; ++i) x[i] = (float)(x[i] / s);
}

void llmengine_moe_rope_ex(float* vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta, uint32_t variant) {
    for (uint32_t h = 0; h < n_heads; ++h) {
        float* v = vec + (size_t)h * head_dim;
        if (variant == 1) {
            /* interleaved (GPT-NeoX): coppie adiacenti (v[2i], v[2i+1]) */
            for (uint32_t i = 0; i < head_dim / 2; ++i) {
                float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
                float ang = (float)pos * freq, c = cosf(ang), s = sinf(ang);
                float x1 = v[2 * i], x2 = v[2 * i + 1];
                v[2 * i] = x1 * c - x2 * s;
                v[2 * i + 1] = x2 * c + x1 * s;
            }
        } else {
            /* half-split (rotate_half, Qwen/Llama) */
            uint32_t half = head_dim / 2;
            for (uint32_t i = 0; i < half; ++i) {
                float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
                float ang = (float)pos * freq, c = cosf(ang), s = sinf(ang);
                float x1 = v[i], x2 = v[i + half];
                v[i] = x1 * c - x2 * s;
                v[i + half] = x2 * c + x1 * s;
            }
        }
    }
}
void llmengine_moe_rope(float* vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta) {
    llmengine_moe_rope_ex(vec, n_heads, head_dim, pos, theta, 0);
}

void llmengine_moe_matvec(const float* w, uint32_t rows, uint32_t cols, const float* x, float* out) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float* wr = w + (size_t)r * cols;
        double a = 0.0;
        for (uint32_t c = 0; c < cols; ++c) a += (double)wr[c] * (double)x[c];
        out[r] = (float)a;
    }
}

/* MLP SwiGLU di UN esperto CONDIVISO (shared-expert, M7.1): acc[hidden] += down(silu(gate·xn)*(up·xn)).
 * Pesi f32 row-major [out,in] (come il resto del forward). No-op se l'esperto condiviso è assente. */
static void shared_moe_accum(const Q3Layer* ly, uint32_t hidden, const float* xn, float* acc) {
    if (ly->shared_gate == NULL || ly->shared_up == NULL || ly->shared_down == NULL || ly->shared_inter == 0) return;
    uint32_t inter = ly->shared_inter;
    float* g = (float*)malloc((size_t)inter * sizeof(float));
    float* d = (float*)malloc((size_t)hidden * sizeof(float));
    if (g == NULL || d == NULL) { free(g); free(d); return; }
    llmengine_moe_matvec(ly->shared_gate, inter, hidden, xn, g);          /* gate in g */
    {
        float* u = (float*)malloc((size_t)inter * sizeof(float));
        if (u == NULL) { free(g); free(d); return; }
        llmengine_moe_matvec(ly->shared_up, inter, hidden, xn, u);        /* up in u */
        for (uint32_t o = 0; o < inter; ++o) {
            float z = g[o];
            g[o] = (z / (1.0f + expf(-z))) * u[o];                  /* silu(gate)*up */
        }
        free(u);
    }
    llmengine_moe_matvec(ly->shared_down, hidden, inter, g, d);           /* down */
    for (uint32_t i = 0; i < hidden; ++i) acc[i] += d[i];
    free(g); free(d);
}

static LlmEngineMoeStatus q3_attention(const Q3Cfg* cfg, const Q3Layer* ly, uint32_t layer, uint32_t pos,
                                 const float* xn, Q3KvCache* kv, float* att_out) {
    uint32_t H = cfg->n_heads, KVH = cfg->n_kv_heads, D = cfg->head_dim, hidden = cfg->hidden;
    uint32_t group = H / KVH;
    float* q = (float*)malloc((size_t)H * D * sizeof(float));
    float* k = (float*)malloc((size_t)KVH * D * sizeof(float));
    float* v = (float*)malloc((size_t)KVH * D * sizeof(float));
    float* ctx = (float*)malloc((size_t)H * D * sizeof(float));
    if (!q || !k || !v || !ctx) { free(q); free(k); free(v); free(ctx); return LLMENGINE_MOE_ERR_NO_MEMORY; }

    llmengine_moe_matvec(ly->q_w, H * D, hidden, xn, q);
    llmengine_moe_matvec(ly->k_w, KVH * D, hidden, xn, k);
    llmengine_moe_matvec(ly->v_w, KVH * D, hidden, xn, v);
    /* per-head q_norm/k_norm + RoPE (oppure q/k-norm su n_embd intero: variante OLMoE) */
    if (cfg->flags & LLMENGINE_MOE_FLAG_QK_NORM_FULL) {
        if (ly->q_norm) llmengine_moe_rmsnorm(q, ly->q_norm, cfg->eps, H * D, q);
        if (ly->k_norm) llmengine_moe_rmsnorm(k, ly->k_norm, cfg->eps, KVH * D, k);
    } else {
        for (uint32_t h = 0; h < H; ++h) llmengine_moe_rmsnorm(q + (size_t)h * D, ly->q_norm, cfg->eps, D, q + (size_t)h * D);
        for (uint32_t h = 0; h < KVH; ++h) llmengine_moe_rmsnorm(k + (size_t)h * D, ly->k_norm, cfg->eps, D, k + (size_t)h * D);
    }
    llmengine_moe_rope_ex(q, H, D, pos, cfg->rope_theta, cfg->rope_variant);
    llmengine_moe_rope_ex(k, KVH, D, pos, cfg->rope_theta, cfg->rope_variant);

    /* scrivi KV cache */
    size_t kvstride_layer = (size_t)KVH * cfg->ctx * D;
    for (uint32_t h = 0; h < KVH; ++h) {
        float* kd = kv->k + (size_t)layer * kvstride_layer + (size_t)h * cfg->ctx * D + (size_t)pos * D;
        float* vd = kv->v + (size_t)layer * kvstride_layer + (size_t)h * cfg->ctx * D + (size_t)pos * D;
        memcpy(kd, k + (size_t)h * D, D * sizeof(float));
        memcpy(vd, v + (size_t)h * D, D * sizeof(float));
    }

    float inv = 1.0f / sqrtf((float)D);
    float* scores = (float*)malloc((size_t)(pos + 1) * sizeof(float));
    if (!scores) { free(q); free(k); free(v); free(ctx); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    for (uint32_t h = 0; h < H; ++h) {
        uint32_t kh = h / group;
        const float* qh = q + (size_t)h * D;
        const float* kbase = kv->k + (size_t)layer * kvstride_layer + (size_t)kh * cfg->ctx * D;
        const float* vbase = kv->v + (size_t)layer * kvstride_layer + (size_t)kh * cfg->ctx * D;
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* kt = kbase + (size_t)t * D;
            double a = 0.0;
            for (uint32_t d = 0; d < D; ++d) a += (double)qh[d] * (double)kt[d];
            scores[t] = (float)a * inv;
        }
        llmengine_moe_softmax(scores, pos + 1);
        float* oh = ctx + (size_t)h * D;
        for (uint32_t d = 0; d < D; ++d) {
            double a = 0.0;
            for (uint32_t t = 0; t <= pos; ++t) a += (double)scores[t] * (double)vbase[(size_t)t * D + d];
            oh[d] = (float)a;
        }
    }
    free(scores);
    llmengine_moe_matvec(ly->o_w, hidden, H * D, ctx, att_out);
    free(q); free(k); free(v); free(ctx);
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_qwen3_decode_layer(const Q3Cfg* cfg, const Q3Layer* ly,
                                        Q3ExpertFn expert_fn, void* ectx,
                                        Q3RouteFn route_fn, void* route_ctx,
                                        uint32_t layer, uint32_t pos,
                                        const float* x, float* out, Q3KvCache* kv) {
    if (!cfg || !ly || !x || !out || !kv) return LLMENGINE_MOE_ERR_INVALID;
    uint32_t hidden = cfg->hidden;
    float* xn = (float*)malloc((size_t)hidden * sizeof(float));
    float* att = (float*)malloc((size_t)hidden * sizeof(float));
    float* res = (float*)malloc((size_t)hidden * sizeof(float));
    float* logits = (float*)malloc((size_t)cfg->n_experts * sizeof(float));
    float* eout = (float*)malloc((size_t)hidden * sizeof(float));
    if (!xn || !att || !res || !logits || !eout) {
        free(xn); free(att); free(res); free(logits); free(eout);
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    /* attention + residual 1 */
    norm_block(cfg, x, ly->input_ln, xn);
    LlmEngineMoeStatus s = q3_attention(cfg, ly, layer, pos, xn, kv, att);
    if (s != LLMENGINE_MOE_OK) { free(xn); free(att); free(res); free(logits); free(eout); return s; }
    for (uint32_t i = 0; i < hidden; ++i) res[i] = x[i] + att[i];
    /* MoE + residual 2 */
    norm_block(cfg, res, ly->post_attn_ln, xn);
    llmengine_moe_matvec(ly->router, cfg->n_experts, hidden, xn, logits);
    /* Qwen3/Mixtral: softmax (o sigmoid) su TUTTI gli esperti, POI top-k, POI rinormalizza
     * (norm_topk_prob). NON è top-k poi softmax sui soli top-k. */
    if (cfg->router_kind == 1) {
        for (uint32_t e = 0; e < cfg->n_experts; ++e) logits[e] = 1.0f / (1.0f + expf(-logits[e]));
    } else {
        llmengine_moe_softmax(logits, cfg->n_experts);
    }
    /* top-k (selezione parziale semplice) */
    uint32_t k = cfg->topk;
    uint32_t* idx = (uint32_t*)malloc((size_t)k * sizeof(uint32_t));
    float* sel = (float*)malloc((size_t)k * sizeof(float));
    if (!idx || !sel) { free(xn); free(att); free(res); free(logits); free(eout); free(idx); free(sel); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    for (uint32_t t = 0; t < k; ++t) {
        int best = -1; float bv = -INFINITY;
        for (uint32_t e = 0; e < cfg->n_experts; ++e) {
            int used = 0;
            for (uint32_t u = 0; u < t; ++u) if (idx[u] == e) { used = 1; break; }
            if (!used && logits[e] > bv) { bv = logits[e]; best = (int)e; }
        }
        idx[t] = (uint32_t)best; sel[t] = bv;
    }
    {   /* rinormalizzazione dei top-k (norm_topk_prob) — salvo modelli che NON rinormalizzano */
        if (!(cfg->flags & LLMENGINE_MOE_FLAG_NO_RENORM)) {
            double z2 = 0.0;
            for (uint32_t t = 0; t < k; ++t) z2 += sel[t];
            if (z2 > 0.0) for (uint32_t t = 0; t < k; ++t) sel[t] = (float)(sel[t] / z2);
        }
    }
    if (route_fn != NULL) route_fn(route_ctx, layer, idx, sel, k);   /* M6 v2: feed router-based (+route_prob) */
    for (uint32_t i = 0; i < hidden; ++i) eout[i] = 0.0f;
    for (uint32_t t = 0; t < k; ++t) {
        float* ex = (float*)malloc((size_t)hidden * sizeof(float));
        if (!ex) { free(xn); free(att); free(res); free(logits); free(eout); free(idx); free(sel); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        s = expert_fn(ectx, layer, idx[t], xn, ex);
        if (s != LLMENGINE_MOE_OK) { free(ex); free(xn); free(att); free(res); free(logits); free(eout); free(idx); free(sel); return s; }
        for (uint32_t i = 0; i < hidden; ++i) eout[i] += sel[t] * ex[i];
        free(ex);
    }
    shared_moe_accum(ly, hidden, xn, eout);   /* M7.1: shared-expert sempre applicato */
    for (uint32_t i = 0; i < hidden; ++i) out[i] = res[i] + eout[i];

    free(xn); free(att); free(res); free(logits); free(eout); free(idx); free(sel);
    return LLMENGINE_MOE_OK;
}

/* M17 — prefill BATCHED: stessa matematica di decode_layer, ma per layer si prendono i top-k di
 * TUTTE le posizioni, si calcola l'UNIONE e ogni esperto e' caricato (batch_fn) UNA volta e
 * applicato a tutte le posizioni che lo hanno selezionato. Letture da O(seq*topk*layer) a
 * O(union*layer): con routing correlato tra posizioni (prompt reale) l'unione ~ topk.
 * M6.4: `batch_fn` e' sorgente-agnostico (GGUF dequant-on-read, container MLX). */
LlmEngineMoeStatus llmengine_moe_qwen3_prefill_batchfn(const Q3Cfg* cfg, const Q3Layer* ly, uint32_t n_layers,
                                           Q3ExpertBatchFn batch_fn, void* bctx,
                                           Q3RouteFn route_fn, void* route_ctx,
                                           uint32_t seq, uint32_t pos0,
                                           const float* x, float* out, Q3KvCache* kv) {
    if (!cfg || !ly || !batch_fn || !x || !out || !kv || seq == 0) return LLMENGINE_MOE_ERR_INVALID;
    uint32_t hidden = cfg->hidden, ne = cfg->n_experts, k = cfg->topk;
    if (k == 0 || ne == 0) return LLMENGINE_MOE_ERR_INVALID;
    float* xn = (float*)malloc((size_t)hidden * sizeof(float));
    float* att = (float*)malloc((size_t)hidden * sizeof(float));
    float* res = (float*)malloc((size_t)hidden * sizeof(float));
    float* logits = (float*)malloc((size_t)ne * sizeof(float));
    float* ex = (float*)malloc((size_t)hidden * sizeof(float));
    float* xn_all = (float*)malloc((size_t)seq * hidden * sizeof(float)); /* post-attn normed */
    uint32_t* idx = (uint32_t*)malloc((size_t)seq * k * sizeof(uint32_t));
    float* sel = (float*)malloc((size_t)seq * k * sizeof(float));
    uint8_t* used = (uint8_t*)malloc(ne);
    float* a = (float*)malloc((size_t)seq * hidden * sizeof(float));
    float* b = (float*)malloc((size_t)seq * hidden * sizeof(float));
    float* xg = (float*)malloc((size_t)seq * hidden * sizeof(float)); /* gather posizioni per esperto */
    float* og = (float*)malloc((size_t)seq * hidden * sizeof(float));
    uint32_t* gpos = (uint32_t*)malloc((size_t)seq * sizeof(uint32_t));
    float* gw = (float*)malloc((size_t)seq * sizeof(float));
    if (!xn || !att || !res || !logits || !ex || !xn_all || !idx || !sel || !used || !a || !b ||
        !xg || !og || !gpos || !gw) {
        free(xn); free(att); free(res); free(logits); free(ex); free(xn_all);
        free(idx); free(sel); free(used); free(a); free(b);
        free(xg); free(og); free(gpos); free(gw);
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    }
    memcpy(a, x, (size_t)seq * hidden * sizeof(float));
    const float* cur = a;
    float* dst = b;
    LlmEngineMoeStatus st = LLMENGINE_MOE_OK;
    for (uint32_t l = 0; l < n_layers; ++l) {
        const Q3Layer* L = &ly[l];
        for (uint32_t t = 0; t < seq; ++t) {
            const float* xt = cur + (size_t)t * hidden;
            float* xn_t = xn_all + (size_t)t * hidden;
            norm_block(cfg, xt, L->input_ln, xn);
            st = q3_attention(cfg, L, l, pos0 + t, xn, kv, att);
            if (st != LLMENGINE_MOE_OK) goto done;
            for (uint32_t i = 0; i < hidden; ++i) res[i] = xt[i] + att[i];
            norm_block(cfg, res, L->post_attn_ln, xn_t);
            llmengine_moe_matvec(L->router, ne, hidden, xn_t, logits);
            if (cfg->router_kind == 1) {
                for (uint32_t e = 0; e < ne; ++e) logits[e] = 1.0f / (1.0f + expf(-logits[e]));
            } else {
                llmengine_moe_softmax(logits, ne);
            }
            /* top-k (stessa logica di decode_layer: softmax su tutti -> top-k -> rinormalizza) */
            uint32_t* ti = idx + (size_t)t * k;
            float* ts = sel + (size_t)t * k;
            for (uint32_t j = 0; j < k; ++j) {
                int best = -1; float bv = -INFINITY;
                for (uint32_t e = 0; e < ne; ++e) {
                    int u = 0;
                    for (uint32_t q = 0; q < j; ++q) if (ti[q] == e) { u = 1; break; }
                    if (!u && logits[e] > bv) { bv = logits[e]; best = (int)e; }
                }
                ti[j] = (uint32_t)best; ts[j] = bv;
            }
            double z2 = 0.0;
            if (!(cfg->flags & LLMENGINE_MOE_FLAG_NO_RENORM)) {
                for (uint32_t j = 0; j < k; ++j) z2 += ts[j];
                if (z2 > 0.0) for (uint32_t j = 0; j < k; ++j) ts[j] = (float)(ts[j] / z2);
            }
            memcpy(dst + (size_t)t * hidden, res, (size_t)hidden * sizeof(float));
        }
        /* unione degli esperti del layer; ogni esperto caricato UNA volta (M17) */
        memset(used, 0, ne);
        for (uint32_t j = 0; j < seq * k; ++j) used[idx[j]] = 1;
        if (route_fn != NULL) {   /* M6 v2: notifica l'UNIONE del layer (feed router-based) */
            uint32_t ulist[512]; uint32_t un = 0;
            for (uint32_t e = 0; e < ne && un < 512u; ++e) if (used[e]) ulist[un++] = e;
            route_fn(route_ctx, l, ulist, NULL, un);   /* unione prefill: probs non per-esperto */
        }
        for (uint32_t e = 0; e < ne; ++e) {
            if (!used[e]) continue;
            /* raccogli le posizioni che hanno scelto e: il peso viene caricato UNA volta
             * dentro batch_fn e applicato a tutte (GEMM BATCHED, M17/M6.4). */
            uint32_t gn = 0;
            for (uint32_t t = 0; t < seq; ++t) {
                const uint32_t* ti = idx + (size_t)t * k;
                const float* ts = sel + (size_t)t * k;
                for (uint32_t j = 0; j < k; ++j) if (ti[j] == e) {
                    memcpy(xg + (size_t)gn * hidden, xn_all + (size_t)t * hidden, (size_t)hidden * sizeof(float));
                    gpos[gn] = t; gw[gn] = ts[j]; gn++;
                    break;
                }
            }
            if (gn == 0) continue;
            st = batch_fn(bctx, l, e, xg, gn, og);
            if (st != LLMENGINE_MOE_OK) goto done;
            for (uint32_t j = 0; j < gn; ++j) {
                float* dt = dst + (size_t)gpos[j] * hidden;
                const float* oj = og + (size_t)j * hidden;
                for (uint32_t i = 0; i < hidden; ++i) dt[i] += gw[j] * oj[i];
            }
        }
        /* M7.1: shared-expert sempre applicato, per ogni posizione del layer */
        if (L->shared_gate != NULL)
            for (uint32_t t = 0; t < seq; ++t)
                shared_moe_accum(L, hidden, xn_all + (size_t)t * hidden, dst + (size_t)t * hidden);
        { const float* tmp = cur; cur = dst; dst = (float*)tmp; }
    }
    memcpy(out, cur, (size_t)seq * hidden * sizeof(float));
done:
    free(xn); free(att); free(res); free(logits); free(ex); free(xn_all);
    free(idx); free(sel); free(used); free(a); free(b);
    free(xg); free(og); free(gpos); free(gw);
    return st;
}

/* ── wrapper retro-compatibile: blob MLX → callback batch (M6.4) ── */
typedef struct { Q3ExpertBlobFn blob_fn; void* bctx; } Q3BlobCtx;

static LlmEngineMoeStatus q3_blob_trampoline(void* c, uint32_t l, uint32_t e,
                                       const float* X, uint32_t n, float* out) {
    Q3BlobCtx* bc = (Q3BlobCtx*)c;
    const void* blob = NULL; size_t bn = 0;
    LlmEngineMoeStatus s = bc->blob_fn(bc->bctx, l, e, &blob, &bn);
    if (s != LLMENGINE_MOE_OK) return s;
    return llmengine_moe_q3_expert_swiglu_batch(blob, bn, X, n, out);
}

LlmEngineMoeStatus llmengine_moe_qwen3_prefill(const Q3Cfg* cfg, const Q3Layer* ly, uint32_t n_layers,
                                   Q3ExpertBlobFn blob_fn, void* bctx,
                                   Q3RouteFn route_fn, void* route_ctx,
                                   uint32_t seq, uint32_t pos0,
                                   const float* x, float* out, Q3KvCache* kv) {
    Q3BlobCtx bc = { blob_fn, bctx };
    return llmengine_moe_qwen3_prefill_batchfn(cfg, ly, n_layers, q3_blob_trampoline, &bc,
                                         route_fn, route_ctx, seq, pos0, x, out, kv);
}
