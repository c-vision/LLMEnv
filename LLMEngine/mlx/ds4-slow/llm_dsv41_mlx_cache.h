/* llm_dsv41_mlx_cache.h — DeepSeek V4.1: forward MLX con cache per-layer (prefill+decode),
 * fedele a `language.py` (`DeepseekV41Cache`, 7 slot). Header-only.
 *
 * La cache replica i 7 slot della reference: [0]=offset, [1]=window KV (fp8, valori
 * roundtrippati), [2]=KV compresso (fp4 gs16), [3]=index K (fp4 gs32), [4]/[5]=remainder
 * kv/gate del compressor, [6]=history Engram (non usata). Memorizzare i valori gia'
 * roundtrippati e' equivalente allo store packed della reference.
 *
 * LIMITI DICHIARATI (guardie a errore esplicito): batch di una riga (un solo stream). I layer
 * index_source-only (reuse di index_k/KV cross-layer), pure-reuse (ratio>0 senza source che
 * riusa KV compresso e top-k condivisi) e candidate_source_layer SONO supportati e verificati
 * (prefill+decode vs riferimento: llm_dsv41_mlx_share_test/_purereuse_test/_cand_test).
 */
#ifndef LLM_DSV41_MLX_CACHE_H
#define LLM_DSV41_MLX_CACHE_H

#include <stdlib.h>
#include <string.h>

#include "llm_dsv41_mlx.h"
#include "llm_dsv41_dspark_attn.h"

typedef struct { mlx_array slot[7]; bool present[7]; } LlmDsv41LayerCache;
typedef struct {
    LlmDsv41LayerCache layers[LLM_DSV41_MAX_LAYERS];
    int offset;
    int64_t* history;   /* history Engram (slot[6] condiviso), depth valori; NULL se nessun engram */
    int history_n;
    float* last_logits; /* logits (host) della posizione pending, per generate_constrained */
    int last_logits_n;
    LlmDsv41DSparkCache* dspark_ctx;  /* contesto DSpark/MTP per-stage; NULL se MTP off */
    int n_dspark_ctx;
} LlmDsv41Cache;

static inline void llm_dsv41_cache_init(LlmDsv41Cache* ck) {
    memset(ck, 0, sizeof *ck);
    for (int l = 0; l < LLM_DSV41_MAX_LAYERS; l++)
        for (int i = 0; i < 7; i++) { ck->layers[l].slot[i] = mlx_array_new(); ck->layers[l].present[i] = false; }
    ck->offset = 0;
}
static inline void llm_dsv41_cache_free(LlmDsv41Cache* ck) {
    for (int l = 0; l < LLM_DSV41_MAX_LAYERS; l++)
        for (int i = 0; i < 7; i++) mlx_array_free(ck->layers[l].slot[i]);
    if (ck->dspark_ctx) {
        for (int i = 0; i < ck->n_dspark_ctx; i++) mlx_array_free(ck->dspark_ctx[i].keys);
        free(ck->dspark_ctx);
        ck->dspark_ctx = NULL; ck->n_dspark_ctx = 0;
    }
    free(ck->history);
    free(ck->last_logits);
}

/* a[:hi] (rows) se presente, altrimenti vuoto. */
static inline mlx_array llm_dsv41_rows_to(mlx_array a, bool present, int hi, mlx_stream s, bool* ok) {
    if (!present) return mlx_array_new();
    int rows = (int)mlx_array_shape(a)[1];
    if (hi > rows) hi = rows;
    return llm_dsv41_mlx_slice_axis(a, 1, 0, hi, s, ok);
}

static inline mlx_array llm_dsv41_concat1(mlx_array a, bool ap, mlx_array b, bool bp,
                                          mlx_stream s, bool* ok) {
    if (!ap) return b;
    if (!bp) return a;
    mlx_array out = mlx_array_new();
    mlx_vector_array v = mlx_vector_array_new();
    mlx_vector_array_append_value(v, a);
    mlx_vector_array_append_value(v, b);
    if (mlx_concatenate_axis(&out, v, 1, s) != 0) *ok = false;
    mlx_vector_array_free(v);
    return out;
}

/* Compressor con cache: aggiorna slot[4]/[5] e ritorna il latent [1,m2,D] (o vuoto). */
static inline mlx_array llm_dsv41_mlx_compressor_cached(mlx_array x, const LlmDsv41LayerWeights* L,
                                                        LlmDsv41LayerCache* lc, int ratio, int start,
                                                        int S, const LlmDsv41Config* c,
                                                        mlx_stream s, bool* ok) {
    const int D = c->head_dim;
    mlx_array kv = llm_dsv41_mlx_linear(x, L->comp_wkv, s, ok);
    if (ratio <= 1) return llm_dsv41_mlx_rmsnorm(kv, L->comp_norm, c->norm_eps, s, ok);
    mlx_array gate = llm_dsv41_mlx_linear(x, L->comp_wgate, s, ok);
    int rem = start % ratio;
    if (rem > 0) {
        mlx_array okv = llm_dsv41_rows_to(lc->slot[4], lc->present[4], rem, s, ok);
        mlx_array ogt = llm_dsv41_rows_to(lc->slot[5], lc->present[5], rem, s, ok);
        kv = llm_dsv41_concat1(okv, lc->present[4], kv, true, s, ok);
        gate = llm_dsv41_concat1(ogt, lc->present[5], gate, true, s, ok);
    }
    int Ltot = (int)mlx_array_shape(kv)[1];
    int cutoff = (Ltot / ratio) * ratio;
    lc->slot[4] = llm_dsv41_mlx_slice_axis(kv, 1, cutoff, Ltot, s, ok); lc->present[4] = true;
    lc->slot[5] = llm_dsv41_mlx_slice_axis(gate, 1, cutoff, Ltot, s, ok); lc->present[5] = true;
    if (cutoff == 0) {
        mlx_array z = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_zeros, z, (const int[]){ 1, 0, D }, 3, MLX_FLOAT32);
        return z;
    }
    int m2 = cutoff / ratio;
    mlx_array kvc = llm_dsv41_mlx_slice_axis(kv, 1, 0, cutoff, s, ok);
    mlx_array gtc = llm_dsv41_mlx_slice_axis(gate, 1, 0, cutoff, s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, kvc, kvc, (const int[]){ 1, m2, ratio, D }, 4);
    LLM_DSV41_MLX_RET(mlx_reshape, gtc, gtc, (const int[]){ 1, m2, ratio, D }, 4);
    mlx_array probs = mlx_array_new(), prod = mlx_array_new(), pooled = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_softmax_axis, probs, gtc, 2, false);
    LLM_DSV41_MLX_RET(mlx_multiply, prod, kvc, probs);
    LLM_DSV41_MLX_RET(mlx_sum_axis, pooled, prod, 2, false);
    return llm_dsv41_mlx_rmsnorm(pooled, L->comp_norm, c->norm_eps, s, ok);
}

/* Attn CSA2 con cache (prefill start==0 o decode start>0). */
static inline mlx_array llm_dsv41_mlx_attention_cached(const LlmDsv41Weights* w, int l, mlx_array x,
                                                       LlmDsv41LayerCache* lc, int start, int S,
                                                       LlmDsv41Shared* sh,
                                                       mlx_stream s, bool* ok) {
    const LlmDsv41Config* c = &w->cfg;
    const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
    const int H = c->n_heads, D = c->head_dim, OG = c->o_groups, OL = c->o_lora_rank;
    const int GRP = H * D / OG;
    const int ratio = c->compress_ratios[l];
    const int comp = ratio > 0;

    mlx_array posa = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_arange, posa, (double)start, (double)(start + S), 1.0, MLX_INT32);

    mlx_array qr = llm_dsv41_mlx_linear(x, L->wq_a, s, ok);
    qr = llm_dsv41_mlx_rmsnorm(qr, L->q_norm, c->norm_eps, s, ok);
    mlx_array q = llm_dsv41_mlx_linear(qr, L->wq_b, s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, q, q, (const int[]){ 1, S, H, D }, 4);
    q = llm_dsv41_mlx_rope(q, posa, c, comp, 0, s, ok);

    /* window */
    mlx_array nkv = llm_dsv41_mlx_linear(x, L->wkv, s, ok);
    nkv = llm_dsv41_mlx_rmsnorm(nkv, L->kv_norm, c->norm_eps, s, ok);
    nkv = llm_dsv41_mlx_rope(nkv, posa, c, comp, 0, s, ok);
    nkv = llm_dsv41_mlx_fp8_roundtrip(nkv, D, s, ok);
    int old_rows = lc->present[1] ? (int)mlx_array_shape(lc->slot[1])[1] : 0;
    int old_len = start < c->window_size ? start : c->window_size;
    if (old_len > old_rows) old_len = old_rows;
    mlx_array old = llm_dsv41_rows_to(lc->slot[1], lc->present[1], old_len, s, ok);
    mlx_array kv = llm_dsv41_concat1(old, lc->present[1], nkv, true, s, ok);
    int Kb = (int)mlx_array_shape(kv)[1];
    if (Kb > c->window_size)
        { lc->slot[1] = llm_dsv41_mlx_slice_axis(kv, 1, Kb - c->window_size, Kb, s, ok); lc->present[1] = true; }
    else { lc->slot[1] = kv; lc->present[1] = true; }

    int W = (start == 0) ? (S < c->window_size ? S : c->window_size) : c->window_size;
    int* widx = (int*)malloc((size_t)S * (size_t)W * sizeof(int));
    if (!widx) { *ok = false; return mlx_array_new(); }
    long lo = start - old_len; if (lo < 0) lo = 0;
    for (int i = 0; i < S; i++) {
        long pos = start + i;
        long base = pos - c->window_size + 1;
        if (start == 0 && base < 0) base = 0;
        for (int k = 0; k < W; k++) {
            long loc = base + k;
            widx[i * W + k] = (loc >= lo && loc <= pos) ? (int)(loc - (start - old_len)) : -1;
        }
    }
    mlx_array idx = mlx_array_new_data(widx, (const int[]){ S, W }, 2, MLX_INT32);
    mlx_array idx_flat = mlx_array_new(), idx_cl = mlx_array_new(), zero_i = mlx_array_new_int(0);
    LLM_DSV41_MLX_RET(mlx_reshape, idx_flat, idx, (const int[]){ S * W }, 1);
    LLM_DSV41_MLX_RET(mlx_maximum, idx_cl, idx_flat, zero_i);
    mlx_array wsel = llm_dsv41_mlx_gather_rows(kv, idx_cl, Kb, s, ok);
    LLM_DSV41_MLX_RET(mlx_reshape, wsel, wsel, (const int[]){ 1, S, W, D }, 4);
    free(widx);
    mlx_array scores = mlx_array_new(), mask = mlx_array_new();
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, q);
        mlx_vector_array_append_value(ops, wsel);
        if (mlx_einsum(&scores, "bihd,bikd->bhik", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    mlx_array qscale = llm_dsv41_mlx_scalar(1.0f / sqrtf((float)D));
    LLM_DSV41_MLX_RET(mlx_multiply, scores, scores, qscale);
    LLM_DSV41_MLX_RET(mlx_greater_equal, mask, idx, zero_i);
    LLM_DSV41_MLX_RET(mlx_reshape, mask, mask, (const int[]){ 1, 1, S, W }, 4);
    int N = W;

    mlx_array csel = mlx_array_new();
    int C = 0;
    if (comp) {
        const int IH = c->index_head_dim, NH = c->index_n_heads;
        mlx_array ik_rows = mlx_array_new(), att_kv_rows = mlx_array_new();
        int Kc = 0;
        mlx_array ids = mlx_array_new();
        bool pure_reuse = (!c->in_kv_source[l] && !c->in_index_source[l]);
        if (pure_reuse) {
            /* ratio>0 ma né source: riusa KV compresso e top-k dal layer sorgente. */
            if (!sh->has_kv || !sh->has_ci) {
                fprintf(stderr, "llm_dsv41_cache: layer %d ratio>0 senza shared kv/ci\n", l);
                *ok = false; return mlx_array_new();
            }
            att_kv_rows = sh->kv; ids = sh->ci; Kc = (int)mlx_array_shape(att_kv_rows)[1];
        } else {
        if (!c->in_index_source[l]) {
            fprintf(stderr, "llm_dsv41_cache: layer %d kv_source senza index_source non supportato\n", l);
            *ok = false; return mlx_array_new();
        }
        if (c->in_kv_source[l]) {
        mlx_array latent = llm_dsv41_mlx_compressor_cached(x, L, lc, ratio, start, S, c, s, ok);
        int m2 = mlx_array_ndim(latent) == 0 ? 0 : (int)mlx_array_shape(latent)[1];
        int rows_before = start / ratio;
        mlx_array cpos = mlx_array_new();
        if (m2 > 0) {
            mlx_array ar = mlx_array_new(), off = mlx_array_new_int((start / ratio) * ratio);
            LLM_DSV41_MLX_RET(mlx_arange, ar, 0.0, (double)m2, 1.0, MLX_INT32);
            mlx_array rr = mlx_array_new_int(ratio);
            LLM_DSV41_MLX_RET(mlx_multiply, cpos, ar, rr);
            LLM_DSV41_MLX_RET(mlx_add, cpos, cpos, off);
        }
        mlx_array base2 = llm_dsv41_rows_to(lc->slot[2], lc->present[2], rows_before, s, ok);
        mlx_array newc = mlx_array_new();
        if (m2 > 0) {
            mlx_array cr = llm_dsv41_mlx_rope(latent, cpos, c, 1, 0, s, ok);
            newc = llm_dsv41_mlx_fp4_roundtrip(cr, D, 16, 1, s, ok);
        }
        lc->slot[2] = llm_dsv41_concat1(base2, lc->present[2], newc, m2 > 0, s, ok); lc->present[2] = true;
        mlx_array base3 = llm_dsv41_rows_to(lc->slot[3], lc->present[3], rows_before, s, ok);
        mlx_array newk = mlx_array_new();
        if (m2 > 0) {
            mlx_array ik = llm_dsv41_mlx_linear(latent, L->idx_wk, s, ok);
            ik = llm_dsv41_mlx_rmsnorm(ik, L->idx_k_norm, c->norm_eps, s, ok);
            ik = llm_dsv41_mlx_rope(ik, cpos, c, 1, 0, s, ok);
            newk = llm_dsv41_mlx_fp4_roundtrip(ik, IH, 32, 0, s, ok);
        }
        lc->slot[3] = llm_dsv41_concat1(base3, lc->present[3], newk, m2 > 0, s, ok); lc->present[3] = true;
        ik_rows = lc->slot[3]; att_kv_rows = lc->slot[2];
        Kc = (int)mlx_array_shape(ik_rows)[1];
        sh->kv = att_kv_rows; sh->index_k = ik_rows; sh->has_kv = true; sh->has_index_k = true;
        } else {
            if (!sh->has_kv || !sh->has_index_k) {
                fprintf(stderr, "llm_dsv41_cache: layer %d index_source-only senza shared kv/index_k\n", l);
                *ok = false; return mlx_array_new();
            }
            att_kv_rows = sh->kv; ik_rows = sh->index_k;
            Kc = (int)mlx_array_shape(ik_rows)[1];
        }
        mlx_array iq = llm_dsv41_mlx_linear(qr, L->idx_wq_b, s, ok);
        LLM_DSV41_MLX_RET(mlx_reshape, iq, iq, (const int[]){ 1, S, NH, IH }, 4);
        iq = llm_dsv41_mlx_rope(iq, posa, c, 1, 0, s, ok);
        iq = llm_dsv41_mlx_fp4_roundtrip(iq, IH, 32, 0, s, ok);
        mlx_array wp = llm_dsv41_mlx_linear(x, L->idx_weights_proj, s, ok);
        {
            mlx_array ws = llm_dsv41_mlx_scalar(1.0f / sqrtf((float)IH) / sqrtf((float)NH));
            LLM_DSV41_MLX_RET(mlx_multiply, wp, wp, ws);
        }
        mlx_array sidx = mlx_array_new();
        if (Kc > 0) {
            mlx_array dots = mlx_array_new(), rel = mlx_array_new(), wpk = mlx_array_new();
            mlx_vector_array ops = mlx_vector_array_new();
            mlx_vector_array_append_value(ops, iq);
            mlx_vector_array_append_value(ops, ik_rows);
            if (mlx_einsum(&dots, "bshd,bkd->bshk", ops, s) != 0) *ok = false;
            mlx_vector_array_free(ops);
            if (!*ok) return mlx_array_new();
            mlx_array zero = llm_dsv41_mlx_scalar(0.0f);
            LLM_DSV41_MLX_RET(mlx_maximum, rel, dots, zero);
            LLM_DSV41_MLX_RET(mlx_expand_dims, wpk, wp, -1);
            LLM_DSV41_MLX_RET(mlx_multiply, rel, rel, wpk);
            LLM_DSV41_MLX_RET(mlx_sum_axis, sidx, rel, 2, false);
        } else {
            LLM_DSV41_MLX_RET(mlx_zeros, sidx, (const int[]){ 1, S, 0 }, 3, MLX_FLOAT32);
        }
        int* vis = (int*)malloc((size_t)S * sizeof(int));
        if (!vis) { *ok = false; return mlx_array_new(); }
        for (int i = 0; i < S; i++) vis[i] = (start + i + 1) / ratio;
        mlx_array visv = mlx_array_new_data(vis, (const int[]){ S }, 1, MLX_INT32);
        free(vis);
        mlx_array visc = mlx_array_new(), jarr = mlx_array_new(), jjc = mlx_array_new(), cvis = mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_reshape, visc, visv, (const int[]){ S, 1 }, 2);
        LLM_DSV41_MLX_RET(mlx_arange, jarr, 0.0, (double)Kc, 1.0, MLX_INT32);
        LLM_DSV41_MLX_RET(mlx_reshape, jjc, jarr, (const int[]){ 1, Kc }, 2);
        LLM_DSV41_MLX_RET(mlx_less, cvis, jjc, visc);
        LLM_DSV41_MLX_RET(mlx_reshape, cvis, cvis, (const int[]){ 1, S, Kc }, 3);
        mlx_array neginf0 = mlx_array_new_float(-INFINITY), sidx2 = mlx_array_new();
        if (Kc > 0) { LLM_DSV41_MLX_RET(mlx_where, sidx2, cvis, sidx, neginf0); sidx = sidx2; }
        const int cand_src = c->candidate_source_layer;
        if (cand_src >= 0 && l > cand_src && sh->has_candidates) {
            int nb = (int)mlx_array_shape(sh->candidates)[2];
            int bs = c->candidate_block_size;
            int Ncand = nb * bs;
            mlx_array ble = mlx_array_new(), bss = mlx_array_new_int(bs), off = mlx_array_new(), ar = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_expand_dims, ble, sh->candidates, -1);
            LLM_DSV41_MLX_RET(mlx_multiply, ble, ble, bss);
            LLM_DSV41_MLX_RET(mlx_arange, ar, 0.0, (double)bs, 1.0, MLX_INT32);
            LLM_DSV41_MLX_RET(mlx_add, off, ble, ar);
            mlx_array z0 = mlx_array_new_int(0), bmask = mlx_array_new(), cand = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_greater_equal, bmask, ble, z0);
            { mlx_array mone = mlx_array_new_int(-1); LLM_DSV41_MLX_RET(mlx_where, cand, bmask, off, mone); }
            LLM_DSV41_MLX_RET(mlx_reshape, cand, cand, (const int[]){ 1, S, Ncand }, 3);
            mlx_array cf = mlx_array_new(), ccl = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_reshape, cf, cand, (const int[]){ S * Ncand }, 1);
            LLM_DSV41_MLX_RET(mlx_maximum, ccl, cf, z0);
            mlx_array ksel = llm_dsv41_mlx_gather_rows(ik_rows, ccl, Kc, s, ok);
            LLM_DSV41_MLX_RET(mlx_reshape, ksel, ksel, (const int[]){ 1, S, Ncand, IH }, 4);
            mlx_array dots2 = mlx_array_new(), rel2 = mlx_array_new(), wpk2 = mlx_array_new();
            {
                mlx_vector_array ops = mlx_vector_array_new();
                mlx_vector_array_append_value(ops, iq);
                mlx_vector_array_append_value(ops, ksel);
                if (mlx_einsum(&dots2, "bshd,bskd->bshk", ops, s) != 0) *ok = false;
                mlx_vector_array_free(ops);
            }
            if (!*ok) return mlx_array_new();
            { mlx_array zero = llm_dsv41_mlx_scalar(0.0f); LLM_DSV41_MLX_RET(mlx_maximum, rel2, dots2, zero); }
            LLM_DSV41_MLX_RET(mlx_expand_dims, wpk2, wp, -1);
            LLM_DSV41_MLX_RET(mlx_multiply, rel2, rel2, wpk2);
            mlx_array scnd = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_sum_axis, scnd, rel2, 2, false);
            mlx_array nonsc = mlx_array_new(), lt = mlx_array_new(), cnd2 = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_greater_equal, nonsc, cand, z0);
            LLM_DSV41_MLX_RET(mlx_less, lt, cand, visc);
            LLM_DSV41_MLX_RET(mlx_logical_and, cnd2, nonsc, lt);
            mlx_array scnd2 = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_where, scnd2, cnd2, scnd, neginf0);
            int ccnt = c->index_topk < Ncand ? c->index_topk : Ncand;
            ids = llm_dsv41_mlx_topk_vals(scnd2, cand, ccnt, s, ok);
        } else {
            int count = c->index_topk < Kc ? c->index_topk : Kc;
            ids = llm_dsv41_mlx_topk_ids(sidx, count, s, ok);
            if (cand_src >= 0 && l == cand_src && Kc > 0) {
                sh->candidates = llm_dsv41_mlx_block_select(sidx, Kc, c->candidate_block_size,
                                                             c->candidate_topk_blocks, start, ratio, s, ok);
                sh->has_candidates = true;
            }
        }
        sh->ci = ids; sh->has_ci = true;
        }
        if (!*ok) return mlx_array_new();
        C = (int)mlx_array_shape(ids)[2];
        /* Path Metal packed DECODE (gated): window dal buffer cache (wi=idx) + KV compresso cache
         * + top-k ids come `ci`. Solo decode (start>0). Altrimenti path semantico. */
        if (start > 0 && llm_dsv41_metal_enabled()) {
            mlx_array mo = mlx_array_new();
            if (llm_dsv41_mlx_attention_packed_dec(&mo, q, kv, Kb, att_kv_rows, Kc, idx, ids, C,
                                                   L->attn_sink, c, S, s))
                return llm_dsv41_mlx_attn_tail(w, l, mo, posa, S, s, ok);
        }
        if (C > 0) {
            mlx_array ids_flat = mlx_array_new(), ids_cl = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_reshape, ids_flat, ids, (const int[]){ S * C }, 1);
            LLM_DSV41_MLX_RET(mlx_maximum, ids_cl, ids_flat, zero_i);
            mlx_array sel = llm_dsv41_mlx_gather_rows(att_kv_rows, ids_cl, Kc, s, ok);
            LLM_DSV41_MLX_RET(mlx_reshape, csel, sel, (const int[]){ 1, S, C, D }, 4);
            mlx_array mask_c = mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_greater_equal, mask_c, ids, zero_i);
            LLM_DSV41_MLX_RET(mlx_reshape, mask_c, mask_c, (const int[]){ 1, 1, S, C }, 4);
            mlx_array scores_c = mlx_array_new();
            mlx_vector_array ops = mlx_vector_array_new();
            mlx_vector_array_append_value(ops, q);
            mlx_vector_array_append_value(ops, csel);
            if (mlx_einsum(&scores_c, "bihd,bikd->bhik", ops, s) != 0) *ok = false;
            mlx_vector_array_free(ops);
            if (!*ok) return mlx_array_new();
            LLM_DSV41_MLX_RET(mlx_multiply, scores_c, scores_c, qscale);
            mlx_array sc2 = mlx_array_new(), cond2 = mlx_array_new();
            mlx_vector_array v1 = mlx_vector_array_new_data((mlx_array[]){ scores, scores_c }, 2);
            mlx_vector_array v2 = mlx_vector_array_new_data((mlx_array[]){ mask, mask_c }, 2);
            if (mlx_concatenate_axis(&sc2, v1, 3, s) != 0) *ok = false;
            if (mlx_concatenate_axis(&cond2, v2, 3, s) != 0) *ok = false;
            mlx_vector_array_free(v1); mlx_vector_array_free(v2);
            if (!*ok) return mlx_array_new();
            scores = sc2; mask = cond2; N = W + C;
        }
    }

    /* Path Metal packed DECODE senza chiavi compresse (ratio==0, start>0): solo window + sink. */
    if (!comp && start > 0 && llm_dsv41_metal_enabled()) {
        mlx_array mo = mlx_array_new();
        if (llm_dsv41_mlx_attention_packed_dec(&mo, q, kv, Kb, mlx_array_new(), 0, idx,
                                               mlx_array_new(), 0, L->attn_sink, c, S, s))
            return llm_dsv41_mlx_attn_tail(w, l, mo, posa, S, s, ok);
    }

    mlx_array neginf = mlx_array_new_float(-INFINITY);
    mlx_array masked = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_where, masked, mask, scores, neginf);
    mlx_array sink = mlx_array_new(), sinkb = mlx_array_new(), cat = mlx_array_new(), probs = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, sink, L->attn_sink, (const int[]){ 1, H, 1, 1 }, 4);
    LLM_DSV41_MLX_RET(mlx_broadcast_to, sinkb, sink, (const int[]){ 1, H, S, 1 }, 4);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, masked);
        mlx_vector_array_append_value(ops, sinkb);
        if (mlx_concatenate_axis(&cat, ops, 3, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_softmax_axis, probs, cat, 3, false);
    probs = llm_dsv41_mlx_slice_last(probs, 0, N, s, ok);
    mlx_array out = mlx_array_new();
    mlx_array probs_w = llm_dsv41_mlx_slice_last(probs, 0, W, s, ok);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, probs_w);
        mlx_vector_array_append_value(ops, wsel);
        if (mlx_einsum(&out, "bhik,bikd->bihd", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    if (C > 0) {
        mlx_array probs_c = llm_dsv41_mlx_slice_last(probs, W, N, s, ok);
        mlx_array outc = mlx_array_new();
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, probs_c);
        mlx_vector_array_append_value(ops, csel);
        if (mlx_einsum(&outc, "bhik,bikd->bihd", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
        if (!*ok) return mlx_array_new();
        LLM_DSV41_MLX_RET(mlx_add, out, out, outc);
    }
    out = llm_dsv41_mlx_rope(out, posa, c, comp, 1, s, ok);
    mlx_array grp = mlx_array_new(), woa = mlx_array_new(), y = mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, grp, out, (const int[]){ 1, S, OG, GRP }, 4);
    LLM_DSV41_MLX_RET(mlx_reshape, woa, L->wo_a, (const int[]){ OG, OL, GRP }, 3);
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, grp);
        mlx_vector_array_append_value(ops, woa);
        if (mlx_einsum(&y, "bsgd,grd->bsgr", ops, s) != 0) *ok = false;
        mlx_vector_array_free(ops);
    }
    if (!*ok) return mlx_array_new();
    LLM_DSV41_MLX_RET(mlx_reshape, y, y, (const int[]){ 1, S, OG * OL }, 3);
    return llm_dsv41_mlx_linear(y, L->wo_b, s, ok);
}

/* Forward completo con cache: prefill (offset 0) o decode (offset>0). Aggiorna offset. */
static inline bool llm_dsv41_mlx_forward_cached(const LlmDsv41Weights* w, LlmDsv41Cache* ck,
                                                const int* ids, int S, mlx_array* out_logits,
                                                const int* cap_layers, int n_cap, mlx_array* out_capture,
                                                mlx_stream s) {
    bool ok = true;
    const LlmDsv41Config* c = &w->cfg;
    const int HC = c->hc_mult, DIM = c->dim;
    int start = ck->offset;

    mlx_array ids_a = mlx_array_new_data(ids, (const int[]){ S }, 1, MLX_INT32);
    mlx_array h = mlx_array_new();
    if (mlx_take_axis(&h, w->embed, ids_a, 0, s) != 0) return false;
    if (mlx_reshape(&h, h, (const int[]){ 1, S, 1, DIM }, 4, s) != 0) return false;
    if (mlx_repeat_axis(&h, h, HC, 2, s) != 0) return false;
    float pre_hc[64];
    for (int i = 0; i < HC; i++) pre_hc[i] = (i == 0) ? 1.0f : 0.0f;
    mlx_array pre = llm_dsv41_array_f32(pre_hc, 1, (const int[]){ HC });
    if (mlx_broadcast_to(&pre, pre, (const int[]){ 1, S, HC }, 3, s) != 0) return false;

    /* history Engram (condivisa): init -1, aggiornata a fine forward */
    int eng_depth = 0;
    for (int l = 0; l < c->n_layers; l++)
        if (llm_dsv41_mlx_L(w, l)->engram.present) { eng_depth = llm_dsv41_mlx_L(w, l)->engram.depth; break; }
    int64_t hout_buf[64];
    if (eng_depth > 0 && !ck->history) {
        ck->history = (int64_t*)malloc((size_t)eng_depth * sizeof(int64_t));
        ck->history_n = eng_depth;
        for (int j = 0; j < eng_depth; j++) ck->history[j] = -1;
    }

    LlmDsv41Shared shared; llm_dsv41_shared_init(&shared);
    bool do_cap = (n_cap > 0 && out_capture != NULL);
    mlx_vector_array caps = mlx_vector_array_new();

    for (int l = 0; l < c->n_layers; l++) {
        const LlmDsv41LayerWeights* L = llm_dsv41_mlx_L(w, l);
        LlmDsv41LayerCache* lc = &ck->layers[l];
        if (L->engram.present) {
            h = llm_dsv41_mlx_engram(w, l, h, ids, S, ck->history, hout_buf, s, &ok);
            if (!ok) return false;
        }
        if (do_cap) for (int ci = 0; ci < n_cap; ci++) if (cap_layers[ci] == l) {
            mlx_array cm = mlx_array_new();
            if (mlx_mean_axis(&cm, h, 2, false, s) != 0) { ok = false; break; }
            mlx_vector_array_append_value(caps, cm);
        }
        if (!ok) return false;
        mlx_array ap = mlx_array_new(), ao = mlx_array_new(), ac = mlx_array_new();
        llm_dsv41_mlx_hc_mixes(h, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, c,
                               &ap, &ao, &ac, s, &ok);
        if (!ok) return false;
        mlx_array x = llm_dsv41_mlx_hc_pre_norm(h, pre, L->attn_norm, c->norm_eps, s, &ok);
        mlx_array attn_out = llm_dsv41_mlx_attention_cached(w, l, x, lc, start, S,
                                                            &shared, s, &ok);
        h = llm_dsv41_mlx_hc_post(attn_out, h, ao, ac, s, &ok);
        mlx_array fp = mlx_array_new(), fo = mlx_array_new(), fc = mlx_array_new();
        llm_dsv41_mlx_hc_mixes(h, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, c,
                               &fp, &fo, &fc, s, &ok);
        if (!ok) return false;
        x = llm_dsv41_mlx_hc_pre_norm(h, ap, L->ffn_norm, c->norm_eps, s, &ok);
        mlx_array f = llm_dsv41_mlx_moe(w, l, x, S, s, &ok);
        h = llm_dsv41_mlx_hc_post(f, h, fo, fc, s, &ok);
        pre = fp;
        if (!ok) return false;
    }
    h = llm_dsv41_mlx_hc_pre_norm(h, pre, w->norm, c->norm_eps, s, &ok);
    *out_logits = llm_dsv41_mlx_linear(h, w->head, s, &ok);
    if (do_cap) {
        if ((int)mlx_vector_array_size(caps) != n_cap) ok = false;
        else if (mlx_concatenate_axis(out_capture, caps, 2, s) != 0) ok = false;
    }
    mlx_vector_array_free(caps);
    if (!ok) return false;
    if (eng_depth > 0) for (int j = 0; j < eng_depth; j++) ck->history[j] = hout_buf[j];
    ck->offset = start + S;
    return ok;
}

#endif /* LLM_DSV41_MLX_CACHE_H */
