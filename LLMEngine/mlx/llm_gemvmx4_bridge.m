/* llm_gemvmx4_bridge.m — implementazione Objective-C del bridge C-ABI, vedi
 * llm_gemvmx4_bridge.h per il contratto e il motivo della separazione. */
#import "llm_gemvmx4_bridge.h"
#import "../../LLMMetal/gemv/llmmetal_gemv_mxfp4.h"
#include <stdlib.h>

struct LlmGemvMx4Ctx {
    LlmMetalGemvMxfp4Ctx inner;
};

LlmGemvMx4Ctx* llm_gemvmx4_create(void) {
    LlmGemvMx4Ctx* c = (LlmGemvMx4Ctx*)calloc(1, sizeof(LlmGemvMx4Ctx));
    if (!c) return NULL;
    if (llmmetal_gemv_mxfp4_init(&c->inner) != 0) { free(c); return NULL; }
    return c;
}

void llm_gemvmx4_destroy(LlmGemvMx4Ctx* ctx) {
    if (!ctx) return;
    llmmetal_gemv_mxfp4_free(&ctx->inner);
    free(ctx);
}

int llm_gemvmx4_load_weights(LlmGemvMx4Ctx* ctx, const uint8_t* Wq,
                             const uint8_t* scales,
                             uint32_t H, uint32_t M, uint32_t group_size) {
    if (!ctx) return -1;
    return llmmetal_gemv_mxfp4_load_weights(&ctx->inner, Wq, scales, H, M, group_size);
}

int llm_gemvmx4_run(LlmGemvMx4Ctx* ctx, const uint16_t* x, uint16_t* out) {
    if (!ctx) return -1;
    return llmmetal_gemv_mxfp4_run(&ctx->inner, x, out);
}
