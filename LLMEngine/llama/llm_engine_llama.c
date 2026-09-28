/* llm_engine_llama.c — adapter del motore llama.cpp all'ABI comune `llmengine`.
 * Delega 1:1 alle funzioni native `llm_ai_*` (nessuna logica duplicata). llama.cpp possiede
 * tokenizer/template e lavora a chat di testo: caps = TEXT. Gli handle nativi sono `int`:
 * li incapsuliamo in un void* (intptr_t), non-NULL per handle validi. */
#include "llmengine.h"
#include "llmengine_llama.h"
#include "llmengine_llama_ocr.h"
#include <stdint.h>

static LlmEngineLlmModel lc_load(const char* path) { return (LlmEngineLlmModel)(intptr_t)llm_ai_load_model(path, 0); }
static LlmEngineLlmModel lc_load_opts(const char* path, int n_gpu_layers) { return (LlmEngineLlmModel)(intptr_t)llm_ai_load_model(path, n_gpu_layers); }
static void        lc_free(LlmEngineLlmModel m)    { llm_ai_unload_model((int)(intptr_t)m); }
static LlmEngineLlmSession lc_sess(LlmEngineLlmModel m)  { return (LlmEngineLlmSession)(intptr_t)llm_ai_create_session((int)(intptr_t)m, 0, 0, 0); }
static void        lc_dsess(LlmEngineLlmSession s) { llm_ai_destroy_session((int)(intptr_t)s); }

static int lc_append_text(LlmEngineLlmSession s, const char* role, const char* text) {
    return llm_ai_session_append((int)(intptr_t)s, role, text);
}
static char* lc_generate_text(LlmEngineLlmSession s, int max_new) {
    return llm_ai_session_generate((int)(intptr_t)s, max_new);
}
static int lc_set_sampling(LlmEngineLlmSession s, const LlmEngineLlmSampling* sp) {
    if (sp == NULL) return -1;
    return llm_ai_session_set_sampling((int)(intptr_t)s, sp->temperature, sp->top_p, sp->top_k, sp->seed);
}
static void lc_free_str(char* p) { llm_ai_free_string(p); }

/* EMBED (CAP_EMBED) e VISION/OCR (CAP_VISION): capacità del wrapper llama.cpp/mtmd. */
static int lc_emb_dim(LlmEngineLlmModel m) { return llm_ai_emb_dim((int)(intptr_t)m); }
static int lc_embed_text(LlmEngineLlmModel m, const char* text, float* out, int cap) {
    return llm_ai_model_embed_text((int)(intptr_t)m, text, out, cap);
}
static LlmEngineLlmVision lc_vision_load(const char* model_path, const char* mmproj_path, int n_ctx, int n_gpu_layers) {
    return (LlmEngineLlmVision)llm_ai_ocr_load(model_path, mmproj_path, n_ctx, n_gpu_layers);
}
static char* lc_vision_read_image(LlmEngineLlmVision v, const char* image_path, const char* prompt, int max_tokens) {
    return llm_ai_ocr_read_image((llm_ai_ocr_handle)v, image_path, prompt, max_tokens);
}
static char* lc_vision_read_audio(LlmEngineLlmVision v, const char* audio_path, const char* prompt, int max_tokens) {
    return llm_ai_ocr_read_audio((llm_ai_ocr_handle)v, audio_path, prompt, max_tokens);
}
static void lc_vision_unload(LlmEngineLlmVision v) { llm_ai_ocr_unload((llm_ai_ocr_handle)v); }

static const LlmEngineLlmEngine LLMENGINE_LLAMA_LLAMACPP_ENGINE = {
    .name = "llamacpp",
    .caps = LLMMODELIO_CAP_TEXT | LLMMODELIO_CAP_EMBED | LLMMODELIO_CAP_VISION,
    .load_model = lc_load, .load_model_opts = lc_load_opts, .free_model = lc_free,
    .create_session = lc_sess, .destroy_session = lc_dsess,
    .append_ids = NULL, .append_text = lc_append_text,
    .generate_ids = NULL, .generate_text = lc_generate_text,
    .set_sampling = lc_set_sampling,
    .kv_clear = NULL, .kv_stats = NULL,
    .tokenize = NULL, .detokenize = NULL,
    .set_prefill_chunk = NULL, .append_ids_prefix = NULL, .save_prefix = NULL, .generate_batch = NULL,
    .emb_dim = lc_emb_dim, .embed_text = lc_embed_text,
    .vision_load = lc_vision_load, .vision_read_image = lc_vision_read_image,
    .vision_read_audio = lc_vision_read_audio, .vision_unload = lc_vision_unload,
    .lora_clear = NULL, .lora_add = NULL, .lora_set_enabled = NULL, .lora_save = NULL, .lora_load = NULL, .capture_hidden = NULL,
    .free_str = lc_free_str,
};

const LlmEngineLlmEngine* llmengine_llm_engine_llamacpp(void) { return &LLMENGINE_LLAMA_LLAMACPP_ENGINE; }
