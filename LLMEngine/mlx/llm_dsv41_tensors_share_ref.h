/* llm_dsv41_tensors_share_ref.h — loader del pak tiny con condivisione cross-layer
 * (`llm_dsv41_share_ref.h`): layer1 kv_source+index_source, layer2 index_source-only. */
#ifndef LLM_DSV41_TENSORS_SHARE_REF_H
#define LLM_DSV41_TENSORS_SHARE_REF_H

#include "llm_dsv41_tensors.h"
#include "llm_dsv41_share_ref.h"

static inline bool llm_dsv41_mlx_weights_load_ref_share(LlmDsv41Weights* w) {
    LlmDsv41Config c = llm_dsv41_config_share_ref();
    if (!llm_dsv41_weights_alloc(w, &c)) return false;

    const int DIM = c.dim, HC = c.hc_mult, H = c.n_heads, D = c.head_dim;
    const int QD = c.q_lora_rank, OG = c.o_groups, OL = c.o_lora_rank;
    const int E = c.n_routed_experts, DI = c.moe_inter_dim, V = c.vocab_size;
    const int SIZE = (2 + HC) * HC, HCD = HC * DIM, GRP = H * D / OG;
    const int IH = c.index_head_dim, NH = c.index_n_heads;
    const int sh_dim[] = { DIM }, sh_fn[] = { SIZE, HCD }, sh_scale[] = { 3 }, sh_base[] = { SIZE };
    const int sh_qd[] = { QD }, sh_wqb[] = { H * D, QD }, sh_wkv[] = { D, DIM }, sh_d[] = { D };
    const int sh_woa[] = { OG * OL, GRP }, sh_wob[] = { DIM, OG * OL }, sh_sink[] = { H };
    const int sh_gw[] = { E, DIM }, sh_gb[] = { E };
    const int sh_ew1[] = { E, DI, DIM }, sh_ew2[] = { E, DIM, DI };
    const int sh_sw1[] = { DI, DIM }, sh_sw2[] = { DIM, DI };

    w->embed = llm_dsv41_array_f32(REF_SH_EMBED, 2, (const int[]){ V, DIM });
    w->norm  = llm_dsv41_array_f32(REF_SH_FNORM, 1, sh_dim);
    w->head  = llm_dsv41_array_f32(REF_SH_HEAD, 2, (const int[]){ V, DIM });

#define SH_LAYER(i) do { \
        LlmDsv41LayerWeights* L = &w->layers[i]; \
        L->hc_attn_fn = llm_dsv41_array_f32(REF_SH_HCAFN##i, 2, sh_fn); \
        L->hc_attn_scale = llm_dsv41_array_f32(REF_SH_HCAS##i, 1, sh_scale); \
        L->hc_attn_base = llm_dsv41_array_f32(REF_SH_HCAB##i, 1, sh_base); \
        L->hc_ffn_fn = llm_dsv41_array_f32(REF_SH_HCFFN##i, 2, sh_fn); \
        L->hc_ffn_scale = llm_dsv41_array_f32(REF_SH_HCFFS##i, 1, sh_scale); \
        L->hc_ffn_base = llm_dsv41_array_f32(REF_SH_HCFFB##i, 1, sh_base); \
        L->attn_norm = llm_dsv41_array_f32(REF_SH_ANORM##i, 1, sh_dim); \
        L->ffn_norm = llm_dsv41_array_f32(REF_SH_FNORM##i, 1, sh_dim); \
        L->wq_a = llm_dsv41_array_f32(REF_SH_WQA##i, 2, (const int[]){ QD, DIM }); \
        L->q_norm = llm_dsv41_array_f32(REF_SH_QN##i, 1, sh_qd); \
        L->wq_b = llm_dsv41_array_f32(REF_SH_WQB##i, 2, sh_wqb); \
        L->wkv = llm_dsv41_array_f32(REF_SH_WKV##i, 2, sh_wkv); \
        L->kv_norm = llm_dsv41_array_f32(REF_SH_KVN##i, 1, sh_d); \
        L->wo_a = llm_dsv41_array_f32(REF_SH_WOA##i, 2, sh_woa); \
        L->wo_b = llm_dsv41_array_f32(REF_SH_WOB##i, 2, sh_wob); \
        L->attn_sink = llm_dsv41_array_f32(REF_SH_SINK##i, 1, sh_sink); \
        L->gate_w = llm_dsv41_array_f32(REF_SH_GW##i, 2, sh_gw); \
        L->gate_b = llm_dsv41_array_f32(REF_SH_GB##i, 1, sh_gb); \
        L->experts_w1 = llm_dsv41_array_f32(REF_SH_EW1##i, 3, sh_ew1); \
        L->experts_w3 = llm_dsv41_array_f32(REF_SH_EW3##i, 3, sh_ew1); \
        L->experts_w2 = llm_dsv41_array_f32(REF_SH_EW2##i, 3, sh_ew2); \
        L->shared_w1 = llm_dsv41_array_f32(REF_SH_SW1##i, 2, sh_sw1); \
        L->shared_w3 = llm_dsv41_array_f32(REF_SH_SW3##i, 2, sh_sw1); \
        L->shared_w2 = llm_dsv41_array_f32(REF_SH_SW2##i, 2, sh_sw2); \
    } while (0)
    SH_LAYER(0);
    SH_LAYER(1);
    SH_LAYER(2);
#undef SH_LAYER

    /* layer1: compressor + indexer (incl. wk/k_norm) */
    w->layers[1].comp_wkv = llm_dsv41_array_f32(REF_SH_CMPWKV1, 2, (const int[]){ D, DIM });
    w->layers[1].comp_norm = llm_dsv41_array_f32(REF_SH_CMPNORM1, 1, sh_d);
    w->layers[1].comp_wgate = llm_dsv41_array_f32(REF_SH_CMPWGATE1, 2, (const int[]){ D, DIM });
    w->layers[1].idx_wq_b = llm_dsv41_array_f32(REF_SH_IDXWQB1, 2, (const int[]){ NH * IH, QD });
    w->layers[1].idx_weights_proj = llm_dsv41_array_f32(REF_SH_IDXWP1, 2, (const int[]){ NH, DIM });
    w->layers[1].idx_wk = llm_dsv41_array_f32(REF_SH_IDXWK1, 2, (const int[]){ IH, D });
    w->layers[1].idx_k_norm = llm_dsv41_array_f32(REF_SH_IDXKN1, 1, (const int[]){ IH });
    /* layer2: index_source-only (niente compressor/wk: usa lo shared di layer1) */
    w->layers[2].idx_wq_b = llm_dsv41_array_f32(REF_SH_IDXWQB2, 2, (const int[]){ NH * IH, QD });
    w->layers[2].idx_weights_proj = llm_dsv41_array_f32(REF_SH_IDXWP2, 2, (const int[]){ NH, DIM });
    return true;
}

#endif /* LLM_DSV41_TENSORS_SHARE_REF_H */
