/* llm_kvmetal_bridge.m — implementazione Objective-C del bridge C-ABI, vedi
 * llm_kvmetal_bridge.h per il contratto e il motivo di questa separazione. */
#import "llm_kvmetal_bridge.h"
#import "../../LLMMetal/kv/llmmetal_kv.h"
#include <stdlib.h>
#include <string.h>

struct LlmKvMetalCtx {
    LlmMetalKvCtx inner;
};

LlmKvMetalCtx* llm_kvmetal_create(int n_pages, int page_tokens, int n_q_heads, int n_kv_heads, int head_dim) {
    LlmKvMetalCtx* c = (LlmKvMetalCtx*)calloc(1, sizeof(LlmKvMetalCtx));
    if (!c) return NULL;
    if (llmmetal_kv_init(&c->inner, n_pages, page_tokens, n_q_heads, n_kv_heads, head_dim) != 0) {
        free(c);
        return NULL;
    }
    return c;
}

void llm_kvmetal_destroy(LlmKvMetalCtx* ctx) {
    if (!ctx) return;
    llmmetal_kv_free(&ctx->inner);
    free(ctx);
}

int llm_kvmetal_write(LlmKvMetalCtx* ctx, int page_id, int offset,
                       const uint16_t* k_tok, const uint16_t* v_tok) {
    if (!ctx) return -1;
    return llmmetal_kv_write(&ctx->inner, page_id, offset, k_tok, v_tok);
}

int llm_kvmetal_decode_attention(LlmKvMetalCtx* ctx, const uint16_t* q,
                                  const int* page_table, int n_pages_used, int last_page_len,
                                  float scale, uint16_t* out) {
    if (!ctx) return -1;
    return llmmetal_kv_decode_attention(&ctx->inner, q, page_table, n_pages_used, last_page_len, scale, out);
}
