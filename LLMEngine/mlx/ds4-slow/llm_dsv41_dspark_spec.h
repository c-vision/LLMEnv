/* llm_dsv41_dspark_spec.h — DeepSeek V4.1: DSpark stage/proposal_forward/forward_spec (MLX),
 * separato da llm_dsv41_dspark.h (matematica pura). Verifica: llm_dsv41_dspark_spec_test.c. */
#ifndef LLM_DSV41_DSPARK_SPEC_H
#define LLM_DSV41_DSPARK_SPEC_H
#include "llm_dsv41_dspark.h"
#include "llm_dsv41_dspark_attn.h"

/* errore in funzioni che ritornano bool (diverse da LLM_DSV41_MLX_RET). */
#define LLM_DSV41_DSB(op, res, ...) do { if ((op)(&(res), __VA_ARGS__, s) != 0) { ok = false; return false; } } while (0)

/* ── DSparkStage / proposal_forward / forward_spec ─────────────────────────────
 * Pesi di uno stage MTP: il blocco (Block con DSparkAttention + MoE draft) piu':
 *   - stage 0:   main_proj/main_norm (proiezione degli hidden target)
 *   - stage finale: out_norm, markov (embed+head), conf_proj
 * `draft_cfg` = config principale con n_routed/n_activated_experts DSpark ed engram off. */
typedef struct {
    LlmDsv41LayerWeights block;
    mlx_array main_proj, main_norm;                          /* stage 0 */
    mlx_array out_norm, markov_embed, markov_head, conf_proj;/* stage finale */
} LlmDsv41DSparkStage;

typedef struct {
    LlmDsv41Config cfg;        /* config principale (dim/hc/heads/dspark_*) */
    LlmDsv41Config draft_cfg;  /* expert counts draft, engram off */
    mlx_array embed, head;     /* embed [vocab,dim] · head [vocab,dim] */
    LlmDsv41DSparkStage* stages; int n_stages;
} LlmDsv41DSpark;

static inline void llm_dsv41_dspark_free(LlmDsv41DSpark* m) {
    if (!m) return;
    for (int st = 0; st < m->n_stages; st++) {
        LlmDsv41DSparkStage* s = &m->stages[st];
        mlx_array_free(s->block.hc_attn_fn); mlx_array_free(s->block.hc_attn_scale);
        mlx_array_free(s->block.hc_attn_base); mlx_array_free(s->block.hc_ffn_fn);
        mlx_array_free(s->block.hc_ffn_scale); mlx_array_free(s->block.hc_ffn_base);
        mlx_array_free(s->block.attn_norm); mlx_array_free(s->block.ffn_norm);
        mlx_array_free(s->block.wq_a); mlx_array_free(s->block.q_norm); mlx_array_free(s->block.wq_b);
        mlx_array_free(s->block.wkv); mlx_array_free(s->block.kv_norm);
        mlx_array_free(s->block.wo_a); mlx_array_free(s->block.wo_b); mlx_array_free(s->block.attn_sink);
        mlx_array_free(s->block.gate_w); mlx_array_free(s->block.gate_b);
        mlx_array_free(s->block.experts_w1); mlx_array_free(s->block.experts_w3);
        mlx_array_free(s->block.experts_w2); mlx_array_free(s->block.shared_w1);
        mlx_array_free(s->block.shared_w3); mlx_array_free(s->block.shared_w2);
        mlx_array_free(s->main_proj); mlx_array_free(s->main_norm);
        mlx_array_free(s->out_norm); mlx_array_free(s->markov_embed);
        mlx_array_free(s->markov_head); mlx_array_free(s->conf_proj);
    }
    free(m->stages);
    m->stages = NULL; m->n_stages = 0;
}

/* Blocco stage: h [1,S,hc,dim] · pre [1,S,hc] -> h' + fp (pre del layer successivo). */
static inline mlx_array llm_dsv41_dspark_stage_forward(const LlmDsv41DSpark* m, int st,
                                                       mlx_array h, mlx_array pre,
                                                       LlmDsv41DSparkCache* cache,
                                                       mlx_array* out_fp, mlx_stream s, bool* ok) {
    const LlmDsv41Config* c = &m->cfg;
    const LlmDsv41LayerWeights* L = &m->stages[st].block;
    mlx_array ap = mlx_array_new(), ao = mlx_array_new(), ac = mlx_array_new();
    llm_dsv41_mlx_hc_mixes(h, L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, c, &ap, &ao, &ac, s, ok);
    if (!*ok) return h;
    mlx_array x = llm_dsv41_mlx_hc_pre(h, pre, s, ok);
    x = llm_dsv41_mlx_rmsnorm(x, L->attn_norm, c->norm_eps, s, ok);
    mlx_array attn = llm_dsv41_dspark_attention(x, cache, c, L->wq_a, L->q_norm, L->wq_b,
                                                L->wkv, L->kv_norm, L->wo_a, L->wo_b, L->attn_sink, s, ok);
    h = llm_dsv41_mlx_hc_post(attn, h, ao, ac, s, ok);
    mlx_array fp = mlx_array_new(), fo = mlx_array_new(), fc = mlx_array_new();
    llm_dsv41_mlx_hc_mixes(h, L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, c, &fp, &fo, &fc, s, ok);
    x = llm_dsv41_mlx_hc_pre(h, ap, s, ok);
    x = llm_dsv41_mlx_rmsnorm(x, L->ffn_norm, c->norm_eps, s, ok);
    LlmDsv41Weights tmp; memset(&tmp, 0, sizeof tmp);
    tmp.cfg = m->draft_cfg; tmp.layers = (LlmDsv41LayerWeights*)L;
    mlx_array f = llm_dsv41_mlx_moe(&tmp, 0, x, (int)mlx_array_shape(x)[1], s, ok);
    h = llm_dsv41_mlx_hc_post(f, h, fo, fc, s, ok);
    *out_fp = fp;
    return h;
}

/* append_context: kv = quantize_activation(rope(kv_norm(wkv(main_x)))) -> cache. */
static inline void llm_dsv41_dspark_append_context(const LlmDsv41DSpark* m, int st, mlx_array main_x,
                                                   LlmDsv41DSparkCache* cache, mlx_stream s, bool* ok) {
    const LlmDsv41Config* c = &m->cfg;
    const LlmDsv41LayerWeights* L = &m->stages[st].block;
    int P = (int)mlx_array_shape(main_x)[1];
    mlx_array positions = mlx_array_new();
    if (mlx_arange(&positions, (double)cache->offset, (double)(cache->offset + P), 1.0, MLX_INT32, s) != 0) *ok = false;
    mlx_array kv = llm_dsv41_mlx_linear(main_x, L->wkv, s, ok);
    kv = llm_dsv41_mlx_rmsnorm(kv, L->kv_norm, c->norm_eps, s, ok);
    kv = llm_dsv41_mlx_rope(kv, positions, c, 0, 0, s, ok);
    kv = llm_dsv41_mlx_fp8_roundtrip(kv, c->head_dim, s, ok);
    mlx_array kv2 = mlx_array_new();
    LLM_DSV41_MLX_DO(mlx_expand_dims, kv2, kv, 1);
    llm_dspark_cache_append(cache, kv2, s, ok);
}

/* proposal_forward: logits draft paralleli [1,width,vocab] + h [1,width,dim]. */
static inline bool llm_dsv41_dspark_proposal_forward(const LlmDsv41DSpark* m, int anchor,
                                                     LlmDsv41DSparkCache* caches, mlx_array* out_logits,
                                                     mlx_array* out_h, mlx_stream s) {
    bool ok = true;
    const LlmDsv41Config* c = &m->cfg;
    int width = c->dspark_block_size < 1 ? 1 : c->dspark_block_size;
    int hc = c->hc_mult, dim = c->dim;
    int32_t* ids_h = (int32_t*)malloc((size_t)width * sizeof(int32_t));
    if (!ids_h) return false;
    ids_h[0] = anchor;
    for (int i = 1; i < width; i++) ids_h[i] = c->dspark_noise_token_id;
    mlx_array ids_a = mlx_array_new_data(ids_h, (const int[]){ width }, 1, MLX_INT32);
    free(ids_h);
    mlx_array h = mlx_array_new();
    if (mlx_take_axis(&h, m->embed, ids_a, 0, s) != 0) return false;
    if (mlx_reshape(&h, h, (const int[]){ 1, width, 1, dim }, 4, s) != 0) return false;
    if (mlx_repeat_axis(&h, h, hc, 2, s) != 0) return false;
    float* pre_h = (float*)malloc((size_t)hc * sizeof(float));
    if (!pre_h) return false;
    for (int i = 0; i < hc; i++) pre_h[i] = (i == 0) ? 1.0f : 0.0f;
    mlx_array pre = llm_dsv41_array_f32(pre_h, 1, (const int[]){ hc });
    free(pre_h);
    if (mlx_broadcast_to(&pre, pre, (const int[]){ 1, width, hc }, 3, s) != 0) return false;
    for (int st = 0; st < m->n_stages; st++) {
        mlx_array fp = mlx_array_new();
        h = llm_dsv41_dspark_stage_forward(m, st, h, pre, &caches[st], &fp, s, &ok);
        if (!ok) return false;
        pre = fp;
    }
    h = llm_dsv41_mlx_hc_pre(h, pre, s, &ok);
    mlx_array hn = llm_dsv41_mlx_rmsnorm(h, m->stages[m->n_stages - 1].out_norm, c->norm_eps, s, &ok);
    *out_logits = llm_dsv41_mlx_linear(hn, m->head, s, &ok);
    *out_h = h;
    return ok;
}

/* forward_spec: priming (offset 0) -> false/None; altrimenti propone un blocco draft.
 * Ritorna proposed [1,width+1], biases [1,width,vocab], confidence [1,width]. */
static inline bool llm_dsv41_dspark_forward_spec(const LlmDsv41DSpark* m, mlx_array input_ids,
                                                 mlx_array main_hidden, LlmDsv41DSparkCache* caches,
                                                 mlx_array* out_proposed, mlx_array* out_biases,
                                                 mlx_array* out_confidence, mlx_stream s) {
    bool ok = true;
    const LlmDsv41Config* c = &m->cfg;
    int initial = (caches[0].offset == 0);
    mlx_array main_x = llm_dsv41_mlx_linear(main_hidden, m->stages[0].main_proj, s, &ok);
    main_x = llm_dsv41_mlx_rmsnorm(main_x, m->stages[0].main_norm, c->norm_eps, s, &ok);
    for (int st = 0; st < m->n_stages; st++)
        llm_dsv41_dspark_append_context(m, st, main_x, &caches[st], s, &ok);
    if (!ok) return false;
    if (initial) return false;   /* priming: nessuna proposta */
    mlx_array logits = mlx_array_new(), h = mlx_array_new();
    int32_t* ip = (int32_t*)mlx_array_data_int32(input_ids);
    int anchor = ip[0];
    if (!llm_dsv41_dspark_proposal_forward(m, anchor, caches, &logits, &h, s)) return false;
    int width = c->dspark_block_size < 1 ? 1 : c->dspark_block_size;
    const LlmDsv41DSparkStage* fin = &m->stages[m->n_stages - 1];
    mlx_vector_array proposed = mlx_vector_array_new();
    mlx_vector_array biases = mlx_vector_array_new();
    mlx_vector_array embs = mlx_vector_array_new();
    mlx_array anchor_a = mlx_array_new_data(&anchor, (const int[]){ 1, 1 }, 2, MLX_INT32);
    mlx_vector_array_append_value(proposed, anchor_a);
    mlx_array prev_a = mlx_array_new_data(&anchor, (const int[]){ 1 }, 1, MLX_INT32);
    for (int i = 0; i < width; i++) {
        mlx_array pida = prev_a;
        mlx_array emb = mlx_array_new();
        if (mlx_take_axis(&emb, fin->markov_embed, pida, 0, s) != 0) return false;   /* [1,rank] */
        mlx_array bias = llm_dsv41_mlx_linear(emb, fin->markov_head, s, &ok);        /* [1,vocab] */
        mlx_array li = llm_dsv41_mlx_slice_axis(logits, 1, i, i + 1, s, &ok);        /* [1,1,vocab] */
        mlx_array li2 = mlx_array_new();
        LLM_DSV41_DSB(mlx_reshape, li2, li, (const int[]){ 1, (int)mlx_array_shape(logits)[2] }, 2);
        mlx_array step = mlx_array_new();
        LLM_DSV41_DSB(mlx_add, step, li2, bias);
        mlx_array tok = mlx_array_new();
        LLM_DSV41_DSB(mlx_argmax_axis, tok, step, 1, false);                     /* [1] uint32 */
        mlx_array toki = mlx_array_new();
        LLM_DSV41_DSB(mlx_astype, toki, tok, MLX_INT32);
        mlx_array tok2 = mlx_array_new(), emb2 = mlx_array_new(), step2 = mlx_array_new();
        LLM_DSV41_DSB(mlx_reshape, tok2, toki, (const int[]){ 1, 1 }, 2);
        LLM_DSV41_DSB(mlx_expand_dims, emb2, emb, 1);                           /* [1,1,rank] */
        LLM_DSV41_DSB(mlx_expand_dims, step2, step, 1);                         /* [1,1,vocab] */
        mlx_vector_array_append_value(proposed, tok2);
        mlx_vector_array_append_value(biases, step2);
        mlx_vector_array_append_value(embs, emb2);
        if (!ok) return false;
        LLM_DSV41_DSB(mlx_reshape, prev_a, toki, (const int[]){ 1 }, 1);
    }
    mlx_array proposed_c = mlx_array_new(), biases_c = mlx_array_new(), embs_c = mlx_array_new();
    if (mlx_concatenate_axis(&proposed_c, proposed, 1, s) != 0) ok = false;
    if (mlx_concatenate_axis(&biases_c, biases, 1, s) != 0) ok = false;
    if (mlx_concatenate_axis(&embs_c, embs, 1, s) != 0) ok = false;
    mlx_vector_array_free(proposed); mlx_vector_array_free(biases); mlx_vector_array_free(embs);
    if (!ok) return false;
    mlx_array joined = mlx_array_new();
    {
        mlx_vector_array ops = mlx_vector_array_new();
        mlx_vector_array_append_value(ops, h);
        mlx_vector_array_append_value(ops, embs_c);
        if (mlx_concatenate_axis(&joined, ops, 2, s) != 0) ok = false;
        mlx_vector_array_free(ops);
    }
    if (!ok) return false;
    mlx_array conf = llm_dsv41_mlx_linear(joined, fin->conf_proj, s, &ok);   /* [1,width,1] */
    mlx_array conf2 = mlx_array_new();
    if (mlx_squeeze_axis(&conf2, conf, 2, s) != 0) ok = false;
    *out_proposed = proposed_c; *out_biases = biases_c; *out_confidence = conf2;
    return ok;
}


#endif /* LLM_DSV41_DSPARK_SPEC_H */
