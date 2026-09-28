/* llm_engine_moe.c — adapter del runtime MoE all'ABI comune `llmengine`.
 * Il runtime MoE è one-shot (prompt -> token), non incrementale: la sessione accumula i token
 * (append_ids) e generate_ids li passa a llmengine_moe_generate. caps = IDS. */
#include "llmengine.h"
#include "llmengine_moe.h"
#include <stdlib.h>

typedef struct { LlmEngineMoeRuntime* rt; int32_t* ids; int n, cap; } MoeSession;

static LlmEngineLlmModel moe_load(const char* path) {
    LlmEngineMoeRuntime* rt = llmengine_moe_create();
    if (rt == NULL) return NULL;
    if (llmengine_moe_load_model(rt, path) != LLMENGINE_MOE_OK) { llmengine_moe_destroy(rt); return NULL; }
    return (LlmEngineLlmModel)rt;
}
static void moe_free(LlmEngineLlmModel m) { llmengine_moe_destroy((LlmEngineMoeRuntime*)m); }

static LlmEngineLlmSession moe_sess(LlmEngineLlmModel m) {
    MoeSession* s = (MoeSession*)calloc(1, sizeof *s);
    if (s) s->rt = (LlmEngineMoeRuntime*)m;
    return (LlmEngineLlmSession)s;
}
static void moe_dsess(LlmEngineLlmSession s) { MoeSession* x = (MoeSession*)s; if (x) { free(x->ids); free(x); } }

static int moe_append_ids(LlmEngineLlmSession s, const int32_t* ids, int n) {
    MoeSession* x = (MoeSession*)s;
    if (x == NULL || ids == NULL || n <= 0) return -1;
    if (x->n + n > x->cap) {
        int c = x->cap ? x->cap * 2 : 64; while (c < x->n + n) c *= 2;
        int32_t* p = (int32_t*)realloc(x->ids, (size_t)c * sizeof(int32_t));
        if (p == NULL) return -1;
        x->ids = p; x->cap = c;
    }
    for (int i = 0; i < n; i++) x->ids[x->n++] = ids[i];
    return 0;
}
static int moe_generate_ids(LlmEngineLlmSession s, int max_new, int32_t eos, int32_t* out) {
    (void)eos;   /* llmengine_moe_generate è greedy e non ha eos: parametro non supportato */
    MoeSession* x = (MoeSession*)s;
    if (x == NULL || max_new <= 0 || out == NULL) return -1;
    return llmengine_moe_generate(x->rt, x->ids, (uint32_t)x->n, out, (uint32_t)max_new);
}

static const LlmEngineLlmEngine LLMENGINE_MOE_ENGINE = {
    .name = "moe",
    .caps = LLMMODELIO_CAP_IDS,
    .load_model = moe_load, .free_model = moe_free,
    .create_session = moe_sess, .destroy_session = moe_dsess,
    .append_ids = moe_append_ids, .append_text = NULL,
    .generate_ids = moe_generate_ids, .generate_text = NULL,
    .set_sampling = NULL, .kv_clear = NULL, .kv_stats = NULL,
    .tokenize = NULL, .detokenize = NULL, .free_str = NULL,
};

const LlmEngineLlmEngine* llmengine_llm_engine_moe(void) { return &LLMENGINE_MOE_ENGINE; }
