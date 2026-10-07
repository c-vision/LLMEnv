/* llm_dsv4_hcroute.h — parametri del "tail leggero" del decode: mHC post(attn) + mHC pre(ffn) + route MoE,
 * encodati nella STESSA command buffer condivisa dell'attn (project+mid+finish). Dopo il wait unico l'host ha
 * gia' hc `after`, l'input normalizzato dell'FFN, lo split mHC (pre|post|comb) e gli esperti selezionati+pesi:
 * salta mhc_post_attn / mhc_pre_ffn / route CPU e va dritto a store_get + MoE. Solo layer NON hash. */
#ifndef LLM_DSV4_HCROUTE_H
#define LLM_DSV4_HCROUTE_H

typedef struct {
    int D, HC, SIZE, HCD;                       /* SIZE = (2+HC)*HC */
    int sinkhorn_iters; float hc_eps, rms_eps;
    int n_expert, used; float expert_scale;
    const float* res;                           /* hc PRIMA dell'attn [HC*D] (residuo del mHC post) */
    const float* post_attn;                     /* [HC]    dal mHC pre dell'attn */
    const float* comb_attn;                     /* [HC*HC] dal mHC pre dell'attn */
    const float* fn; const float* scale; const float* base; const float* norm_w;   /* mHC ffn */
    const float* router_w; const float* exp_probs_b;                               /* routing */
    float* after;                               /* out [HC*D]  hc dopo mHC post(attn) */
    float* normed;                              /* out [D]     input FFN normalizzato */
    float* split;                               /* out [SIZE]  pre|post|comb dell'mHC ffn */
    int* sel; float* rw;                        /* out [used]  esperti scelti e pesi */
} LlmDsv4HcRouteParams;

#ifdef __cplusplus
extern "C" {
#endif
struct LlmDsv4Metal;
/* Encoda mHC post/pre + route nel cb condiviso attivo (0 = ok). */
int llm_dsv4_metal_attn_hc_route(struct LlmDsv4Metal* b, const LlmDsv4HcRouteParams* q);
/* Imposta (o con sel==NULL azzera) la route pre-calcolata, consumata dalla prossima llm_dsv4_gpu_moe. */
void llm_dsv4_gpu_set_route(const int* sel, const float* rw, int used);
#ifdef __cplusplus
}
#endif

#endif /* LLM_DSV4_HCROUTE_H */
