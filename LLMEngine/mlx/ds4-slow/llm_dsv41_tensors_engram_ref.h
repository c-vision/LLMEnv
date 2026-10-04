/* llm_dsv41_tensors_engram_ref.h — loader del pak tiny Engram (`llm_dsv41_engram_fwd_ref.h`)
 * in `LlmDsv41Weights` MLX, inclusi tabella/wkv/q_weight/k_weight Engram e i parametrici di hash
 * (token_map/primes/multipliers/offsets). Per la verifica dell'integrazione Engram nel forward. */
#ifndef LLM_DSV41_TENSORS_ENGRAM_REF_H
#define LLM_DSV41_TENSORS_ENGRAM_REF_H

#include "llm_dsv41_tensors.h"
#include "llm_dsv41_engram_fwd_ref.h"

static inline bool llm_dsv41_mlx_weights_load_ref_engram(LlmDsv41Weights* w) {
    LlmDsv41Config c = llm_dsv41_config_engram_ref();
    if (!llm_dsv41_weights_alloc(w, &c)) return false;

    const int DIM = c.dim, HC = c.hc_mult, H = c.n_heads, D = c.head_dim;
    const int QD = c.q_lora_rank, OG = c.o_groups, OL = c.o_lora_rank;
    const int E = c.n_routed_experts, DI = c.moe_inter_dim, V = c.vocab_size;
    const int SIZE = (2 + HC) * HC, HCD = HC * DIM, GRP = H * D / OG;
    const int sh_dim[] = { DIM }, sh_fn[] = { SIZE, HCD }, sh_scale[] = { 3 }, sh_base[] = { SIZE };
    const int sh_qd[] = { QD }, sh_wqb[] = { H * D, QD }, sh_wkv[] = { D, DIM }, sh_d[] = { D };
    const int sh_woa[] = { OG * OL, GRP }, sh_wob[] = { DIM, OG * OL }, sh_sink[] = { H };
    const int sh_gw[] = { E, DIM }, sh_gb[] = { E };
    const int sh_ew1[] = { E, DI, DIM }, sh_ew2[] = { E, DIM, DI };
    const int sh_sw1[] = { DI, DIM }, sh_sw2[] = { DIM, DI };

    w->embed = llm_dsv41_array_f32(REF_ENGF_EMBED_G, 2, (const int[]){ V, DIM });
    w->norm  = llm_dsv41_array_f32(REF_ENGF_FNORM, 1, sh_dim);
    w->head  = llm_dsv41_array_f32(REF_ENGF_HEAD, 2, (const int[]){ V, DIM });

#define ENGF_LAYER(i) do { \
        LlmDsv41LayerWeights* L = &w->layers[i]; \
        L->hc_attn_fn = llm_dsv41_array_f32(REF_ENGF_HCAFN##i, 2, sh_fn); \
        L->hc_attn_scale = llm_dsv41_array_f32(REF_ENGF_HCAS##i, 1, sh_scale); \
        L->hc_attn_base = llm_dsv41_array_f32(REF_ENGF_HCAB##i, 1, sh_base); \
        L->hc_ffn_fn = llm_dsv41_array_f32(REF_ENGF_HCFFN##i, 2, sh_fn); \
        L->hc_ffn_scale = llm_dsv41_array_f32(REF_ENGF_HCFFS##i, 1, sh_scale); \
        L->hc_ffn_base = llm_dsv41_array_f32(REF_ENGF_HCFFB##i, 1, sh_base); \
        L->attn_norm = llm_dsv41_array_f32(REF_ENGF_ANORM##i, 1, sh_dim); \
        L->ffn_norm = llm_dsv41_array_f32(REF_ENGF_FNORM##i, 1, sh_dim); \
        L->wq_a = llm_dsv41_array_f32(REF_ENGF_WQA##i, 2, (const int[]){ QD, DIM }); \
        L->q_norm = llm_dsv41_array_f32(REF_ENGF_QN##i, 1, sh_qd); \
        L->wq_b = llm_dsv41_array_f32(REF_ENGF_WQB##i, 2, sh_wqb); \
        L->wkv = llm_dsv41_array_f32(REF_ENGF_WKV##i, 2, sh_wkv); \
        L->kv_norm = llm_dsv41_array_f32(REF_ENGF_KVN##i, 1, sh_d); \
        L->wo_a = llm_dsv41_array_f32(REF_ENGF_WOA##i, 2, sh_woa); \
        L->wo_b = llm_dsv41_array_f32(REF_ENGF_WOB##i, 2, sh_wob); \
        L->attn_sink = llm_dsv41_array_f32(REF_ENGF_SINK##i, 1, sh_sink); \
        L->gate_w = llm_dsv41_array_f32(REF_ENGF_GW##i, 2, sh_gw); \
        L->gate_b = llm_dsv41_array_f32(REF_ENGF_GB##i, 1, sh_gb); \
        L->experts_w1 = llm_dsv41_array_f32(REF_ENGF_EW1##i, 3, sh_ew1); \
        L->experts_w3 = llm_dsv41_array_f32(REF_ENGF_EW3##i, 3, sh_ew1); \
        L->experts_w2 = llm_dsv41_array_f32(REF_ENGF_EW2##i, 3, sh_ew2); \
        L->shared_w1 = llm_dsv41_array_f32(REF_ENGF_SW1##i, 2, sh_sw1); \
        L->shared_w3 = llm_dsv41_array_f32(REF_ENGF_SW3##i, 2, sh_sw1); \
        L->shared_w2 = llm_dsv41_array_f32(REF_ENGF_SW2##i, 2, sh_sw2); \
    } while (0)
    ENGF_LAYER(0);
    ENGF_LAYER(1);
#undef ENGF_LAYER

    /* Engram layer 1 */
    {
        LlmDsv41Engram* G = &w->layers[1].engram;
        const int depth = c.engram_max_ngram_size - 1, nh = c.engram_n_heads;
        const int hd = c.engram_head_dim, ncols = depth * nh, rows = c.engram_num_embeddings[0];
        G->present = true;
        G->depth = depth; G->nheads = nh; G->hd = hd; G->ncols = ncols; G->rows = rows;
        G->pad_id = (int64_t)c.engram_pad_id;
        G->embed = llm_dsv41_array_f32(REF_ENGF_EMBED, 2, (const int[]){ rows, hd });
        G->wkv = llm_dsv41_array_f32(REF_ENGF_WKV, 2, (const int[]){ DIM * (HC + 1), ncols * hd });
        G->qw = llm_dsv41_array_f32(REF_ENGF_QW, 2, (const int[]){ HC, DIM });
        G->kw = llm_dsv41_array_f32(REF_ENGF_KW, 2, (const int[]){ HC, DIM });
        int tm_n = c.engram_compressed_vocab_size;
        G->token_map = (int64_t*)malloc((size_t)tm_n * sizeof(int64_t));
        G->token_map_n = tm_n;
        for (int i = 0; i < tm_n; i++) G->token_map[i] = (int64_t)REF_ENGF_TOKEN_MAP[i];
        G->primes = (int64_t*)malloc((size_t)ncols * sizeof(int64_t));
        G->offsets = (int64_t*)malloc((size_t)ncols * sizeof(int64_t));
        G->mult = (int64_t*)malloc((size_t)(depth + 1) * sizeof(int64_t));
        for (int i = 0; i < ncols; i++) { G->primes[i] = REF_ENGF_PRIMES[i]; G->offsets[i] = REF_ENGF_OFFSETS[i]; }
        for (int i = 0; i <= depth; i++) G->mult[i] = REF_ENGF_MULT[i];
    }
    return true;
}

#endif /* LLM_DSV41_TENSORS_ENGRAM_REF_H */
