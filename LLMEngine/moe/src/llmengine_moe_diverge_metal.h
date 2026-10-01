#ifndef LLMENGINE_MOE_DIVERGE_METAL_H
#define LLMENGINE_MOE_DIVERGE_METAL_H
#ifdef __cplusplus
extern "C" {
#endif
/* Divergence-aware MoE forward su Metal (GPU). start/sorted da llmengine_moe_diverge_plan.
 * 0 ok, <0 errore (no device/lib). */
int llmengine_moe_diverge_forward_metal(const float* x,int T,int H,int O,const float* W,int E,
    const int* aex,const float* aw,int K,const int* start,const int* sorted,float* out);
#ifdef __cplusplus
}
#endif
#endif
