/* llm_gemvq4_bridge.m — implementazione Objective-C del bridge C-ABI, vedi
 * llm_gemvq4_bridge.h per il contratto e il motivo di questa separazione. */
#import "llm_gemvq4_bridge.h"
#import "../../LLMMetal/gemv/llmmetal_gemv_q4.h"
#include <stdlib.h>

struct LlmGemvQ4Ctx {
    LlmMetalGemvQ4Ctx inner;
};

LlmGemvQ4Ctx* llm_gemvq4_create(void) {
    LlmGemvQ4Ctx* c = (LlmGemvQ4Ctx*)calloc(1, sizeof(LlmGemvQ4Ctx));
    if (!c) return NULL;
    if (llmmetal_gemv_q4_init(&c->inner) != 0) { free(c); return NULL; }
    return c;
}

void llm_gemvq4_destroy(LlmGemvQ4Ctx* ctx) {
    if (!ctx) return;
    llmmetal_gemv_q4_free(&ctx->inner);
    free(ctx);
}

int llm_gemvq4_load_weights(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                             const uint16_t* scales, const uint16_t* biases,
                             uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx) return -1;
    return llmmetal_gemv_q4_load_weights(&ctx->inner, Wq, scales, biases, H, M, group_size);
}

int llm_gemvq4_load_weights_nocopy(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                                   const uint16_t* scales, const uint16_t* biases,
                                   uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx) return -1;
    return llmmetal_gemv_q4_load_weights_nocopy(&ctx->inner, Wq, scales, biases, H, M, group_size);
}

int llm_gemvq4_load_weights_wonly_nocopy(LlmGemvQ4Ctx* ctx, const uint8_t* Wq,
                                        const uint16_t* scales, const uint16_t* biases,
                                        uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx) return -1;
    return llmmetal_gemv_q4_load_weights_wonly_nocopy(&ctx->inner, Wq, scales, biases, H, M, group_size);
}

int llm_gemvq4_run(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    if (!ctx) return -1;
    return llmmetal_gemv_q4_run(&ctx->inner, x, out);
}

void llm_gemvq4_set_scale_bf16(LlmGemvQ4Ctx* ctx, int bf16) {
    if (ctx) ctx->inner.scale_bf16 = bf16 ? 1u : 0u;
}

int llm_gemvq4_load_weights_3d_nocopy(LlmGemvQ4Ctx* ctx, const uint8_t* Wq, const uint16_t* scales, const uint16_t* biases,
                                      uint32_t H, uint32_t M, uint32_t group_size, uint32_t E, size_t w_stride, size_t s_stride) {
    if (!ctx) return -1;
    return llmmetal_gemv_q4_load_weights_3d_nocopy(&ctx->inner, Wq, scales, biases, H, M, group_size, E, w_stride, s_stride);
}

int llm_gemvq4_run_expert(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out, uint32_t expert) {
    if (!ctx) return -1;
    return llmmetal_gemv_q4_run_expert(&ctx->inner, x, out, expert);
}

int llm_gemvq4_run_moe_mid(LlmGemvQ4Ctx* cg, LlmGemvQ4Ctx* cu, LlmGemvQ4Ctx* sg, LlmGemvQ4Ctx* su,
                           const uint16_t* x, float* act, const int32_t* experts, int K) {
    if (!cg || !cu) return -1;
    return llmmetal_gemv_q4_run_moe_mid(&cg->inner, &cu->inner, sg ? &sg->inner : NULL, su ? &su->inner : NULL,
                                        x, act, experts, K);
}
int llm_gemvq4_run_moe(LlmGemvQ4Ctx* cg, LlmGemvQ4Ctx* cu, LlmGemvQ4Ctx* sg, LlmGemvQ4Ctx* su,
                       LlmGemvQ4Ctx* cd, LlmGemvQ4Ctx* sd, const uint16_t* x, float* act, float* out,
                       const int32_t* experts, int K, uint32_t x_stride) {
    if (!cg || !cu || !cd) return -1;
    return llmmetal_gemv_q4_run_moe(&cg->inner, &cu->inner, sg?&sg->inner:NULL, su?&su->inner:NULL,
                                    &cd->inner, sd?&sd->inner:NULL, x, act, out, experts, K, x_stride);
}
int llm_gemvq4_run_moe_down(LlmGemvQ4Ctx* cd, LlmGemvQ4Ctx* sd, const float* act,
                            float* out, const int32_t* experts, int K, uint32_t x_stride) {
    if (!cd) return -1;
    return llmmetal_gemv_q4_run_moe_down(&cd->inner, sd ? &sd->inner : NULL, act, out, experts, K, x_stride);
}
int llm_gemvq4_run_experts(LlmGemvQ4Ctx* ctx, const uint16_t* x, uint16_t* out_all, const int32_t* experts, int K, uint32_t x_stride) {
    if (!ctx || !experts || K <= 0) return -1;
    return llmmetal_gemv_q4_run_experts(&ctx->inner, x, out_all, experts, K, x_stride);
}

int llm_gemvq4_run_batch(const LlmGemvQ4Job* jobs, int n) {
    if (n <= 0) return 0;
    if (n > 64 || !jobs) return -1;
    LlmMetalGemvQ4Job mj[64];
    for (int i = 0; i < n; i++) {
        if (!jobs[i].ctx) return -1;
        mj[i].ctx = &jobs[i].ctx->inner; mj[i].expert = jobs[i].expert; mj[i].x = jobs[i].x; mj[i].out = jobs[i].out;
    }
    return llmmetal_gemv_q4_run_batch(mj, n);
}

int llm_gemvq4_router_upload(LlmGemvQ4Ctx* cg, const float* W, int rows, int H) {
    if (!cg) return -1;
    return llmmetal_gemv_q4_router_upload(&cg->inner, W, rows, H);
}
int llm_gemvq4_run_moe_gpu(LlmGemvQ4Ctx* cg, LlmGemvQ4Ctx* cu, LlmGemvQ4Ctx* sg, LlmGemvQ4Ctx* su,
                           LlmGemvQ4Ctx* cd, LlmGemvQ4Ctx* sd, const float* x, float* out, int K) {
    if (!cg || !cu || !cd || !sg || !su || !sd) return -1;
    return llmmetal_gemv_q4_run_moe_gpu(&cg->inner, &cu->inner, &sg->inner, &su->inner, &cd->inner, &sd->inner, x, out, K);
}
