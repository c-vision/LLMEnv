/* llm_dsv41_tensors_csa_ref.h — loader del pak tiny CSA2 (`llm_dsv41_fwd_csa_ref.h`) in
 * `LlmDsv41Weights` MLX. Include il Compressor del layer kv_source (ratio>0). Per la verifica
 * del forward CSA2 senza modello reale. Vedi `llm_dsv41_tensors_ref.h` per l'analogo ratio=0. */
#ifndef LLM_DSV41_TENSORS_CSA_REF_H
#define LLM_DSV41_TENSORS_CSA_REF_H

#include "llm_dsv41_tensors.h"
#include "llm_dsv41_fwd_csa_ref.h"

#define LLM_DSV41_CSA_REF_LAYER(i) { \
    REF_CSA_HCAFN##i, REF_CSA_HCAS##i, REF_CSA_HCAB##i, REF_CSA_ANORM##i, \
    REF_CSA_WQA##i, REF_CSA_QN##i, REF_CSA_WQB##i, REF_CSA_WKV##i, REF_CSA_KVN##i, \
    REF_CSA_WOA##i, REF_CSA_WOB##i, REF_CSA_SINK##i, \
    REF_CSA_HCFFN##i, REF_CSA_HCFFS##i, REF_CSA_HCFFB##i, REF_CSA_FNORM##i, \
    REF_CSA_GW##i, REF_CSA_GB##i, \
    REF_CSA_EW1##i, REF_CSA_EW3##i, REF_CSA_EW2##i, REF_CSA_SW1##i, REF_CSA_SW3##i, REF_CSA_SW2##i \
}

typedef struct {
    const float *hcafn, *hcas, *hcab, *anorm, *wqa, *qn, *wqb, *wkv, *kvn, *woa, *wob, *sink;
    const float *hcffn, *hcfs, *hcfb, *fnorm, *gw, *gb;
    const float *ew1, *ew3, *ew2, *sw1, *sw3, *sw2;
} LlmDsv41CsaRefLayer;

static const LlmDsv41CsaRefLayer g_dsv41_csa_ref_layers[REF_CSA_L] = {
    LLM_DSV41_CSA_REF_LAYER(0),
    LLM_DSV41_CSA_REF_LAYER(1),
};

static inline bool llm_dsv41_mlx_weights_load_ref_csa(LlmDsv41Weights* w) {
    LlmDsv41Config c = llm_dsv41_config_csa_ref();
    if (c.n_layers != REF_CSA_L) return false;
    if (!llm_dsv41_weights_alloc(w, &c)) return false;

    const int DIM = c.dim, HC = c.hc_mult, H = c.n_heads, D = c.head_dim;
    const int QD = c.q_lora_rank, OG = c.o_groups, OL = c.o_lora_rank;
    const int E = c.n_routed_experts, DI = c.moe_inter_dim, V = c.vocab_size;
    const int SIZE = (2 + HC) * HC, HCD = HC * DIM, GRP = H * D / OG;

    const int sh_embed[] = { V, DIM }, sh_dim[] = { DIM };
    const int sh_fn[] = { SIZE, HCD }, sh_scale[] = { 3 }, sh_base[] = { SIZE };
    const int sh_qd[] = { QD }, sh_wqb[] = { H * D, QD }, sh_wkv[] = { D, DIM }, sh_d[] = { D };
    const int sh_woa[] = { OG * OL, GRP }, sh_wob[] = { DIM, OG * OL }, sh_sink[] = { H };
    const int sh_gw[] = { E, DIM }, sh_gb[] = { E };
    const int sh_ew1[] = { E, DI, DIM }, sh_ew2[] = { E, DIM, DI };
    const int sh_sw1[] = { DI, DIM }, sh_sw2[] = { DIM, DI };

    w->embed = llm_dsv41_array_f32(REF_CSA_EMBED, 2, sh_embed);
    w->norm  = llm_dsv41_array_f32(REF_CSA_FNORM, 1, sh_dim);
    w->head  = llm_dsv41_array_f32(REF_CSA_HEAD, 2, sh_embed);

    for (int i = 0; i < c.n_layers; i++) {
        const LlmDsv41CsaRefLayer* r = &g_dsv41_csa_ref_layers[i];
        LlmDsv41LayerWeights* L = &w->layers[i];
        L->hc_attn_fn    = llm_dsv41_array_f32(r->hcafn, 2, sh_fn);
        L->hc_attn_scale = llm_dsv41_array_f32(r->hcas,  1, sh_scale);
        L->hc_attn_base  = llm_dsv41_array_f32(r->hcab,  1, sh_base);
        L->hc_ffn_fn     = llm_dsv41_array_f32(r->hcffn, 2, sh_fn);
        L->hc_ffn_scale  = llm_dsv41_array_f32(r->hcfs,  1, sh_scale);
        L->hc_ffn_base   = llm_dsv41_array_f32(r->hcfb,  1, sh_base);
        L->attn_norm     = llm_dsv41_array_f32(r->anorm, 1, sh_dim);
        L->ffn_norm      = llm_dsv41_array_f32(r->fnorm, 1, sh_dim);
        L->wq_a          = llm_dsv41_array_f32(r->wqa,   2, (const int[]){ QD, DIM });
        L->q_norm        = llm_dsv41_array_f32(r->qn,    1, sh_qd);
        L->wq_b          = llm_dsv41_array_f32(r->wqb,   2, sh_wqb);
        L->wkv           = llm_dsv41_array_f32(r->wkv,   2, sh_wkv);
        L->kv_norm       = llm_dsv41_array_f32(r->kvn,   1, sh_d);
        L->wo_a          = llm_dsv41_array_f32(r->woa,   2, sh_woa);
        L->wo_b          = llm_dsv41_array_f32(r->wob,   2, sh_wob);
        L->attn_sink     = llm_dsv41_array_f32(r->sink,  1, sh_sink);
        L->gate_w        = llm_dsv41_array_f32(r->gw,    2, sh_gw);
        L->gate_b        = llm_dsv41_array_f32(r->gb,    1, sh_gb);
        L->experts_w1    = llm_dsv41_array_f32(r->ew1,   3, sh_ew1);
        L->experts_w3    = llm_dsv41_array_f32(r->ew3,   3, sh_ew1);
        L->experts_w2    = llm_dsv41_array_f32(r->ew2,   3, sh_ew2);
        L->shared_w1     = llm_dsv41_array_f32(r->sw1,   2, sh_sw1);
        L->shared_w3     = llm_dsv41_array_f32(r->sw3,   2, sh_sw1);
        L->shared_w2     = llm_dsv41_array_f32(r->sw2,   2, sh_sw2);
    }
    if (c.in_kv_source[1]) {
        LlmDsv41LayerWeights* L = &w->layers[1];
        L->comp_wkv   = llm_dsv41_array_f32(REF_CSA_CMPWKV1,   2, (const int[]){ D, DIM });
        L->comp_norm  = llm_dsv41_array_f32(REF_CSA_CMPNORM1,  1, sh_d);
        L->comp_wgate = llm_dsv41_array_f32(REF_CSA_CMPWGATE1, 2, (const int[]){ D, DIM });
    }
    if (c.in_index_source[1]) {
        LlmDsv41LayerWeights* L = &w->layers[1];
        const int IH = c.index_head_dim, NH = c.index_n_heads;
        L->idx_wq_b         = llm_dsv41_array_f32(REF_CSA_IDXWQB1, 2, (const int[]){ NH * IH, QD });
        L->idx_weights_proj = llm_dsv41_array_f32(REF_CSA_IDXWP1,  2, (const int[]){ NH, DIM });
        if (c.in_kv_source[1]) {
            L->idx_wk     = llm_dsv41_array_f32(REF_CSA_IDXWK1, 2, (const int[]){ IH, D });
            L->idx_k_norm = llm_dsv41_array_f32(REF_CSA_IDXKN1, 1, (const int[]){ IH });
        }
    }
    return true;
}

#endif /* LLM_DSV41_TENSORS_CSA_REF_H */
