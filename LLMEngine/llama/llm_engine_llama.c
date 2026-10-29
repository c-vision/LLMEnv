/* llm_engine_llama.c — adapter del motore llama.cpp all'ABI comune `llmengine`.
 * Delega 1:1 alle funzioni native `llm_ai_*` (nessuna logica duplicata). llama.cpp possiede
 * tokenizer/template e lavora a chat di testo: caps = TEXT. Gli handle nativi sono `int`:
 * li incapsuliamo in un void* (intptr_t), non-NULL per handle validi. */
#include "llmengine.h"
#include "llmengine_llama.h"
#include "llmengine_llama_ocr.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

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

/* ── Variante a ID di token (`llamacpp_ids`) ──────────────────────────────────────────────────────────────
 * Stessa libreria llama.cpp, ma esposta come motore a id (CAP_IDS): il chiamante porta tokenizer e chat template (FastLLM li
 * costruisce dai metadati del GGUF) e la sessione decodifica/genera id. Serve per eseguire qualunque GGUF che llama.cpp sa
 * caricare con lo stesso flusso (streaming, stop, ragionamento) degli altri motori. */
/* Gli handle nativi partono da 0 e l ABI comune tratta NULL come errore: qui si codificano come handle+1. */
static int g_ids_ctx = 0;   /* contesto richiesto con kv_set_max: 0 = automatico (contesto di addestramento, entro la memoria) */
void llm_ai_shutdown(void);   /* definita in llmengine_llama.c */
static LlmEngineLlmModel lci_load(const char* path) {
    int h = llm_ai_load_model(path, 99);   /* tutti gli strati su GPU */
    /* All uscita libera sessioni e modelli PRIMA che ggml distrugga il device Metal (altrimenti GGML_ASSERT sui residency set).
     * Va registrato DOPO il load: gli atexit girano in ordine inverso e il device Metal nasce (e registra il suo distruttore)
     * durante il load, quindi un atexit registrato prima verrebbe eseguito dopo. */
    { static int reg = 0; if (!reg) { reg = 1; atexit(llm_ai_shutdown); } }
    return h < 0 ? NULL : (LlmEngineLlmModel)(intptr_t)(h + 1);
}
static void lci_free(LlmEngineLlmModel m) { llm_ai_unload_model((int)(intptr_t)m - 1); }
/* Contesto della sessione. llama.cpp preassegna l intera KV-cache alla creazione, quindi la scelta dipende dalla memoria:
 *  - richiesta esplicita (> 0): rispettata, ridotta solo se la KV non entra nel budget (meta' della RAM meno i pesi);
 *  - automatico (0): il contesto di addestramento del modello, sempre entro il budget, minimo 2048.
 * Se llama.cpp rifiuta l allocazione si dimezza fino a 2048. Una riga su stderr dice cosa e' stato scelto e perche'. */
static LlmEngineLlmSession lci_sess(LlmEngineLlmModel m) {
    const int mh = (int)(intptr_t)m - 1;
    int n_train = 0; size_t kv_tok = 0, mbytes = 0;
    (void)llm_ai_model_ctx_info(mh, &n_train, &kv_tok, &mbytes);
    /* Budget per la KV: il 60% della memoria libera o recuperabile ORA (i pesi sono gia' caricati), al massimo un terzo della RAM. */
    uint64_t ram = 0; size_t rl = sizeof ram;
    if (sysctlbyname("hw.memsize", &ram, &rl, NULL, 0) != 0) ram = 0;
    uint64_t avail = 0;
    { vm_statistics64_data_t vs; mach_msg_type_number_t vc = HOST_VM_INFO64_COUNT;
      if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &vc) == KERN_SUCCESS)
          avail = ((uint64_t)vs.free_count + vs.inactive_count + vs.speculative_count) * (uint64_t)vm_page_size; }
    uint64_t budget = avail > 0 ? avail / 10 * 6 : ram / 4;
    if (ram > 0 && budget > ram / 3) budget = ram / 3;
    (void)mbytes;
    /* contesto massimo la cui KV entra nel budget (KV monotona in n_ctx: ricerca binaria); 0 = ignoto */
    long long fit = 0;
    if (llm_ai_model_kv_bytes(mh, 2048) > 0) {
        long long lo = 2048, hi = 16777216;
        if (llm_ai_model_kv_bytes(mh, hi) <= budget) lo = hi;
        else while (hi - lo > 1024) { long long mid = lo + (hi - lo) / 2; if (llm_ai_model_kv_bytes(mh, mid) <= budget) lo = mid; else hi = mid; }
        fit = lo;
    }
    long long want = g_ids_ctx > 0 ? g_ids_ctx : (n_train > 0 ? n_train : 8192);
    long long eff = want;
    if (fit > 0 && eff > fit) eff = fit;
    if (eff < 2048) eff = 2048;
    int s = -1;
    for (;;) {
        s = llm_ai_create_session(mh, (int)eff, 0, 0);
        if (s >= 0 || eff <= 2048) break;
        eff /= 2; if (eff < 2048) eff = 2048;
    }
    if (s < 0) return NULL;
    fprintf(stderr, "[llamacpp] contesto %lld token (%s%lld; addestramento %d; KV ~%.2f GiB)%s%s\n", eff,
            g_ids_ctx > 0 ? "richiesto " : "automatico, ", want, n_train,
            (double)llm_ai_model_kv_bytes(mh, eff) / 1073741824.0,
            eff < want ? " -- ridotto per la memoria disponibile" : "",
            (n_train > 0 && eff > n_train) ? " -- oltre il contesto di addestramento: la qualita' puo' degradare" : "");
    return (LlmEngineLlmSession)(intptr_t)(s + 1);
}
static void lci_dsess(LlmEngineLlmSession s) { llm_ai_destroy_session((int)(intptr_t)s - 1); }
static int lci_kv_set_max(LlmEngineLlmModel m, int n) {   /* contesto delle prossime sessioni; 0 = automatico */
    (void)m; g_ids_ctx = n < 0 ? 0 : n; return 0;
}
static int lci_set_sampling(LlmEngineLlmSession s, const LlmEngineLlmSampling* sp) {
    if (sp == NULL) return -1;
    return llm_ai_session_set_sampling((int)(intptr_t)s - 1, sp->temperature, sp->top_p, sp->top_k, sp->seed);
}
static int lci_append_ids(LlmEngineLlmSession s, const int32_t* ids, int n) {
    return llm_ai_session_append_ids((int)(intptr_t)s - 1, ids, n);
}
static int lci_generate_ids(LlmEngineLlmSession s, int max_new, int32_t eos, int32_t* out) {
    return llm_ai_session_generate_ids((int)(intptr_t)s - 1, max_new, eos, out);
}
static int lci_kv_clear(LlmEngineLlmSession s) { return llm_ai_session_cache_clear((int)(intptr_t)s - 1); }
static int lci_kv_stats(LlmEngineLlmSession s, int* used, int* cap) {
    int n_ctx = 0, n_tok = 0;
    if (llm_ai_session_cache_stats((int)(intptr_t)s - 1, &n_ctx, &n_tok, NULL, NULL, NULL) != 0) return -1;
    if (used) *used = n_tok;
    if (cap) *cap = n_ctx;
    return 0;
}

static const LlmEngineLlmEngine LLMENGINE_LLAMA_IDS_ENGINE = {
    .name = "llamacpp_ids",
    .caps = LLMMODELIO_CAP_IDS | LLMMODELIO_CAP_KV | LLMMODELIO_CAP_KV_MAX,
    .load_model = lci_load, .free_model = lci_free,
    .create_session = lci_sess, .destroy_session = lci_dsess,
    .append_ids = lci_append_ids, .generate_ids = lci_generate_ids,
    .set_sampling = lci_set_sampling,
    .kv_clear = lci_kv_clear, .kv_stats = lci_kv_stats, .kv_set_max = lci_kv_set_max,
    .free_str = lc_free_str,
};

const LlmEngineLlmEngine* llmengine_llm_engine_llamacpp_ids(void) { return &LLMENGINE_LLAMA_IDS_ENGINE; }
